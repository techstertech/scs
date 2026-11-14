"""Verify pinned model files before exposing a loadable local directory.

The manifest is the complete download allowlist. Completed files survive an
interrupted preparation, but a partial file never occupies an artifact path.
No library cache, credentials, downloaded Python, or model runtime is involved.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import hashlib
from http.client import HTTPMessage
from importlib.resources import files
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from tempfile import mkstemp
from threading import Event
from typing import IO, BinaryIO, Final, Protocol, cast, override
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

DOWNLOAD_CHUNK_BYTES: Final[int] = 1_048_576
DOWNLOAD_TIMEOUT_SECONDS: Final[float] = 30.0
MAX_MODEL_DOWNLOAD_BYTES: Final[int] = 8_000_000_000
HUB_ORIGIN: Final[str] = "https://huggingface.co"
HEX_DIGEST: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
COMMIT_REVISION: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}")
REPOSITORY_NAME: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")

Progress = Callable[[str, int, int], None]


class ModelArtifactError(RuntimeError):
    """A model directory could not be prepared without losing integrity."""


class ModelDownloadCancelled(ModelArtifactError):
    """Model preparation stopped at a cooperative cancellation boundary."""


class ModelRole(StrEnum):
    """The three model responsibilities owned by the bundled profile."""

    EMBEDDING = "embedding"
    RERANKING = "reranking"
    DECISION = "decision"


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """One relative, size-bounded artifact with an expected SHA-256 digest."""

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        relative = PurePosixPath(self.path)
        if (
            not self.path
            or not relative.parts
            or relative.is_absolute()
            or relative.as_posix() != self.path
            or ".." in relative.parts
            or "\\" in self.path
            or "\x00" in self.path
        ):
            raise ValueError("model artifact path must be a canonical relative path")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("model artifact size must be a nonnegative integer")
        if not HEX_DIGEST.fullmatch(self.sha256):
            raise ValueError(
                "model artifact checksum must be a lowercase SHA-256 digest"
            )


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Bind one model responsibility to immutable upstream files."""

    role: ModelRole
    repository: str
    revision: str
    artifacts: tuple[ModelArtifact, ...]

    def __post_init__(self) -> None:
        if not REPOSITORY_NAME.fullmatch(self.repository) or any(
            part in {".", ".."} for part in self.repository.split("/")
        ):
            raise ValueError("model repository must be a Hugging Face owner/name")
        if not COMMIT_REVISION.fullmatch(self.revision):
            raise ValueError("model revision must be a lowercase Git commit digest")
        paths = {artifact.path for artifact in self.artifacts}
        if not self.artifacts or len(paths) != len(self.artifacts):
            raise ValueError("model artifacts must be nonempty and have unique paths")

    @property
    def identity(self) -> str:
        """Return the checkpoint identity used by provider diagnostics."""

        return f"{self.repository}@{self.revision}"


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("model manifest entries must be objects")
    return cast(Mapping[str, object], value)


def _items(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("model manifest collections must be lists")
    return cast(list[object], value)


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("model manifest text fields must be strings")
    return value


def _size(value: object) -> int:
    if type(value) is not int:
        raise ValueError("model manifest sizes must be integers")
    return value


def load_manifest() -> dict[ModelRole, ModelSpec]:
    """Load the packaged allowlist without creating files or making requests."""

    contents = (
        files("scs.local_models").joinpath("manifest.json").read_text(encoding="utf-8")
    )
    manifest = _mapping(cast(object, json.loads(contents)))
    if _size(manifest.get("schema_version")) != 1:
        raise ValueError("unsupported bundled model manifest version")

    specifications: dict[ModelRole, ModelSpec] = {}
    for raw in _items(manifest.get("models")):
        entry = _mapping(raw)
        artifacts: list[ModelArtifact] = []
        for artifact_raw in _items(entry.get("artifacts")):
            artifact = _mapping(artifact_raw)
            artifacts.append(
                ModelArtifact(
                    path=_text(artifact.get("path")),
                    size_bytes=_size(artifact.get("size_bytes")),
                    sha256=_text(artifact.get("sha256")),
                )
            )
        spec = ModelSpec(
            role=ModelRole(_text(entry.get("role"))),
            repository=_text(entry.get("repository")),
            revision=_text(entry.get("revision")),
            artifacts=tuple(artifacts),
        )
        if spec.role in specifications:
            raise ValueError("bundled model manifest has a repeated role")
        specifications[spec.role] = spec

    if set(specifications) != set(ModelRole):
        raise ValueError("bundled model manifest must provide every model role")
    total = sum(
        artifact.size_bytes
        for spec in specifications.values()
        for artifact in spec.artifacts
    )
    if total >= MAX_MODEL_DOWNLOAD_BYTES:
        raise ValueError("bundled model manifest exceeds the model download budget")
    return specifications


class ArtifactOpener(Protocol):
    """Open an anonymous artifact stream; the cache owns and closes it."""

    def __call__(self, url: str, /, *, timeout: float) -> BinaryIO: ...


class _HTTPSRedirectHandler(HTTPRedirectHandler):
    """Allow HTTPS CDN redirects without permitting transport downgrades."""

    @override
    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        parsed = urlsplit(newurl)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ModelArtifactError(
                "model artifact redirects must use credential-free HTTPS"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_artifact(url: str, *, timeout: float) -> BinaryIO:
    opener = build_opener(_HTTPSRedirectHandler())
    return cast(BinaryIO, cast(object, opener.open(url, timeout=timeout)))


class ModelCache:
    """Prepare model directories within one private, caller-owned cache root.

    This synchronous owner is called by the bounded inference worker. The caller
    owns serialization; each download uses a unique partial file so cleanup can
    never remove another request's temporary data.
    """

    def __init__(self, root: Path, *, opener: ArtifactOpener | None = None) -> None:
        self.root: Path = root.expanduser().absolute()
        self._opener: ArtifactOpener = opener or _open_artifact

    def cached(self, spec: ModelSpec, *, cancelled: Event | None = None) -> Path | None:
        """Find a complete verified model without downloads, repairs, or writes.

        Missing, unreadable, and corrupt entries are cache misses. Cancellation
        and unsafe aliases retain their typed failures instead of becoming misses.
        """

        self._check_cancelled(cancelled)
        directory = self.root / spec.role.value / spec.revision
        try:
            self._assert_contained(directory)
            for artifact in spec.artifacts:
                self._check_cancelled(cancelled)
                target = directory / artifact.path
                self._assert_contained(target)
                if not stat.S_ISREG(target.lstat().st_mode) or not self._verified(
                    target, artifact, cancelled
                ):
                    self._check_cancelled(cancelled)
                    return None
        except OSError:
            self._check_cancelled(cancelled)
            return None
        self._check_cancelled(cancelled)
        return directory

    def ensure(
        self,
        spec: ModelSpec,
        *,
        progress: Progress | None = None,
        cancelled: Event | None = None,
    ) -> Path:
        """Return a model directory only after every artifact passes verification."""

        self._check_cancelled(cancelled)
        directory = self.root / spec.role.value / spec.revision
        total = sum(artifact.size_bytes for artifact in spec.artifacts)
        completed = 0

        def report(current: int) -> None:
            if progress is not None:
                progress(spec.identity, current, total)

        try:
            self._ensure_directory(directory)
            report(completed)
            for artifact in spec.artifacts:
                self._check_cancelled(cancelled)
                target = directory / artifact.path
                self._assert_contained(target)
                self._ensure_directory(target.parent)
                if not self._verified(target, artifact, cancelled):
                    self._download(
                        spec,
                        artifact,
                        target,
                        cancelled=cancelled,
                        progress=lambda amount: report(completed + amount),
                    )
                completed += artifact.size_bytes
                report(completed)
        except OSError as error:
            raise ModelArtifactError(
                f"model artifact preparation failed: {error}"
            ) from error
        self._check_cancelled(cancelled)
        return directory

    @staticmethod
    def _check_cancelled(cancelled: Event | None) -> None:
        if cancelled is not None and cancelled.is_set():
            raise ModelDownloadCancelled("model preparation was cancelled")

    def _assert_contained(self, path: Path) -> None:
        if self.root.is_symlink():
            raise ModelArtifactError("model cache root cannot be a symlink")
        try:
            relative = path.relative_to(self.root)
            path.resolve(strict=False).relative_to(self.root.resolve(strict=False))
        except ValueError as error:
            raise ModelArtifactError(
                "model artifact path escapes the cache root"
            ) from error
        current = self.root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ModelArtifactError("model artifact path cannot contain a symlink")

    def _ensure_directory(self, path: Path) -> None:
        self._assert_contained(path)
        current = self.root
        directories = [current]
        for part in path.relative_to(self.root).parts:
            current /= part
            directories.append(current)
        for directory in directories:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not stat.S_ISDIR(directory.lstat().st_mode):
                raise ModelArtifactError("model cache entry is not a real directory")
            self._assert_contained(directory)
            directory.chmod(0o700)

    def _verified(
        self, target: Path, artifact: ModelArtifact, cancelled: Event | None
    ) -> bool:
        self._assert_contained(target)
        try:
            info = target.lstat()
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(info.st_mode):
            raise ModelArtifactError("model artifact entry must be a regular file")
        if info.st_size != artifact.size_bytes:
            return False
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            while True:
                self._check_cancelled(cancelled)
                block = stream.read(DOWNLOAD_CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest() == artifact.sha256

    def _download(
        self,
        spec: ModelSpec,
        artifact: ModelArtifact,
        target: Path,
        *,
        cancelled: Event | None,
        progress: Callable[[int], None],
    ) -> None:
        url = f"{HUB_ORIGIN}/{spec.repository}/resolve/{spec.revision}/{quote(artifact.path, safe='/')}"
        descriptor, temporary_name = mkstemp(
            prefix=f".{target.name}.", suffix=".part", dir=target.parent
        )
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        written = 0

        try:
            with os.fdopen(descriptor, "wb") as output:
                with self._opener(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as stream:
                    while True:
                        self._check_cancelled(cancelled)
                        block = stream.read(
                            min(DOWNLOAD_CHUNK_BYTES, artifact.size_bytes - written + 1)
                        )
                        self._check_cancelled(cancelled)
                        if not block:
                            break
                        written += len(block)
                        if written > artifact.size_bytes:
                            raise ModelArtifactError(
                                f"model artifact size exceeds manifest: {artifact.path}"
                            )
                        output.write(block)
                        digest.update(block)
                        progress(written)
                if written != artifact.size_bytes:
                    raise ModelArtifactError(
                        f"model artifact size differs from manifest: {artifact.path}"
                    )
                if digest.hexdigest() != artifact.sha256:
                    raise ModelArtifactError(
                        f"model artifact checksum differs from manifest: {artifact.path}"
                    )
                output.flush()
                os.fsync(output.fileno())
            self._check_cancelled(cancelled)
            self._assert_contained(target)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
