"""Bundled models become loadable only after every artifact is verified."""

from __future__ import annotations

import hashlib
from http.client import HTTPMessage
from io import BytesIO
from pathlib import Path
from threading import Event
from typing import BinaryIO, Protocol, override
from urllib.parse import unquote, urlsplit
from urllib.request import BaseHandler, HTTPRedirectHandler, OpenerDirector, Request

import pytest

from scs.local_models.artifacts import (
    ModelArtifact,
    ModelArtifactError,
    ModelCache,
    ModelDownloadCancelled,
    ModelRole,
    ModelSpec,
    load_manifest,
)


def payload(seed: int, size: int = 41) -> bytes:
    return bytes((index * 73 + seed) % 256 for index in range(size))


def model(files: dict[str, bytes]) -> ModelSpec:
    return ModelSpec(
        role=ModelRole.EMBEDDING,
        repository="scs-fixtures/generated",
        revision="a" * 40,
        artifacts=tuple(
            ModelArtifact(name, len(contents), hashlib.sha256(contents).hexdigest())
            for name, contents in files.items()
        ),
    )


class MemoryOpener:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files: dict[str, bytes] = files
        self.calls: list[str] = []

    def __call__(self, url: str, *, timeout: float) -> BinaryIO:
        assert timeout > 0
        assert urlsplit(url).scheme == "https"
        assert urlsplit(url).netloc == "huggingface.co"
        self.calls.append(url)
        relative = unquote(urlsplit(url).path.split("/resolve/", 1)[1].split("/", 1)[1])
        return BytesIO(self.files[relative])


class RedirectDirector(Protocol):
    """Expose urllib's handler registry, which its standard stub omits."""

    handlers: list[BaseHandler]


def test_manifest_loads_complete_bounded_model_specs() -> None:
    specs = load_manifest()

    assert set(specs) == set(ModelRole)
    assert (
        0
        < sum(item.size_bytes for spec in specs.values() for item in spec.artifacts)
        < 8_000_000_000
    )
    assert all(
        spec.identity == f"{spec.repository}@{spec.revision}" for spec in specs.values()
    )


@pytest.mark.parametrize(
    "relative",
    [
        ".",
        "../weights",
        "/weights",
        "nested/../../weights",
        "foo\\bar",
        "./config.json",
        "bad//config.json",
    ],
)
def test_artifact_paths_cannot_escape_or_alias_cache_entries(relative: str) -> None:
    with pytest.raises(ValueError, match="path"):
        ModelArtifact(relative, 0, hashlib.sha256(b"").hexdigest())


def test_verified_cache_reloads_without_network_and_reports_progress(
    tmp_path: Path,
) -> None:
    files = {"nested/config.json": payload(1), "model.safetensors": payload(2, 103)}
    spec = model(files)
    opener = MemoryOpener(files)
    root = tmp_path / "models"
    cache = ModelCache(root, opener=opener)
    observed: list[tuple[str, int, int]] = []

    def report(identity: str, current: int, total: int) -> None:
        observed.append((identity, current, total))

    assert not root.exists()

    destination = cache.ensure(spec, progress=report)

    assert len(opener.calls) == len(files)
    assert observed[0] == (spec.identity, 0, sum(map(len, files.values())))
    assert observed[-1] == (
        spec.identity,
        sum(map(len, files.values())),
        sum(map(len, files.values())),
    )
    assert [item[1] for item in observed] == sorted(item[1] for item in observed)
    assert all(
        (destination / name).read_bytes() == contents
        for name, contents in files.items()
    )
    assert root.stat().st_mode & 0o777 == 0o700
    assert all((destination / name).stat().st_mode & 0o777 == 0o600 for name in files)

    def offline(_url: str, *, timeout: float) -> BinaryIO:
        del timeout
        raise OSError("network denied")

    assert ModelCache(root, opener=offline).ensure(spec) == destination


def cache_snapshot(root: Path) -> dict[str, tuple[int, int, bytes | None]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_mode & 0o777,
            path.stat().st_mtime_ns,
            path.read_bytes() if path.is_file() else None,
        )
        for path in [root, *root.rglob("*")]
    }


def test_readonly_lookup_does_not_create_a_missing_cache(tmp_path: Path) -> None:
    root = tmp_path / "models"
    opener = MemoryOpener({"model.safetensors": payload(17)})

    assert ModelCache(root, opener=opener).cached(model(opener.files)) is None
    assert not root.exists()
    assert not opener.calls


def test_readonly_lookup_verifies_cache_without_network_or_mode_changes(
    tmp_path: Path,
) -> None:
    files = {"nested/config.json": payload(18), "model.safetensors": payload(19)}
    opener = MemoryOpener(files)
    root = tmp_path / "models"
    cache = ModelCache(root, opener=opener)
    spec = model(files)
    directory = cache.ensure(spec)
    root.chmod(0o755)
    (directory / "model.safetensors").chmod(0o644)
    before = cache_snapshot(root)
    opener.calls.clear()

    assert cache.cached(spec) == directory
    assert cache_snapshot(root) == before
    assert not opener.calls


@pytest.mark.parametrize("damage", ["missing", "size", "checksum", "directory"])
def test_readonly_lookup_leaves_missing_or_corrupt_entries_unchanged(
    tmp_path: Path,
    damage: str,
) -> None:
    files = {"config.json": payload(20), "model.safetensors": payload(21)}
    opener = MemoryOpener(files)
    root = tmp_path / "models"
    cache = ModelCache(root, opener=opener)
    spec = model(files)
    directory = cache.ensure(spec)
    target = directory / "model.safetensors"
    if damage in {"missing", "directory"}:
        target.unlink()
        if damage == "directory":
            target.mkdir()
    else:
        contents = files["model.safetensors"][:-1] if damage == "size" else payload(22)
        target.write_bytes(contents)
    before = cache_snapshot(root)
    opener.calls.clear()

    assert cache.cached(spec) is None
    assert cache_snapshot(root) == before
    assert not opener.calls


def test_readonly_lookup_honors_cancellation_before_touching_cache(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models"
    opener = MemoryOpener({"model.safetensors": payload(23)})
    cancelled = Event()
    cancelled.set()

    with pytest.raises(ModelDownloadCancelled):
        ModelCache(root, opener=opener).cached(model(opener.files), cancelled=cancelled)

    assert not root.exists()
    assert not opener.calls


@pytest.mark.parametrize("damage", ["truncated", "extra", "wrong_hash"])
def test_invalid_download_is_not_promoted_or_left_as_a_partial(
    tmp_path: Path, damage: str
) -> None:
    contents = payload(3)
    corrupted = {
        "truncated": contents[:-1],
        "extra": contents + payload(4, 1),
        "wrong_hash": payload(5, len(contents)),
    }[damage]
    root = tmp_path / "models"
    cache = ModelCache(root, opener=MemoryOpener({"model.safetensors": corrupted}))

    with pytest.raises(ModelArtifactError, match="size|checksum"):
        cache.ensure(model({"model.safetensors": contents}))

    assert not list(root.rglob("model.safetensors"))
    assert not list(root.rglob("*.part"))


def test_failed_repair_cannot_make_a_corrupt_cached_file_available(
    tmp_path: Path,
) -> None:
    contents = payload(6)
    spec = model({"model.safetensors": contents})
    root = tmp_path / "models"
    destination = ModelCache(
        root, opener=MemoryOpener({"model.safetensors": contents})
    ).ensure(spec)
    invalid = payload(7, len(contents))
    (destination / "model.safetensors").write_bytes(invalid)

    def offline(_url: str, *, timeout: float) -> BinaryIO:
        del timeout
        raise OSError("network denied")

    with pytest.raises(ModelArtifactError, match="network denied"):
        ModelCache(root, opener=offline).ensure(spec)

    assert (destination / "model.safetensors").read_bytes() == invalid
    ModelCache(root, opener=MemoryOpener({"model.safetensors": contents})).ensure(spec)
    assert (destination / "model.safetensors").read_bytes() == contents


def test_retry_reuses_completed_artifacts_and_discards_failed_partial(
    tmp_path: Path,
) -> None:
    files = {"config.json": payload(8), "model.safetensors": payload(9)}
    spec = model(files)
    broken = MemoryOpener(
        {**files, "model.safetensors": files["model.safetensors"][:-1]}
    )
    root = tmp_path / "models"

    with pytest.raises(ModelArtifactError):
        ModelCache(root, opener=broken).ensure(spec)

    assert len(list(root.rglob("config.json"))) == 1
    assert not list(root.rglob("*.part"))
    retry = MemoryOpener(files)
    ModelCache(root, opener=retry).ensure(spec)
    assert len(retry.calls) == 1
    assert retry.calls[0].endswith("/model.safetensors")


def test_cancellation_stops_streaming_and_cleans_only_its_partial(
    tmp_path: Path,
) -> None:
    cancelled = Event()
    contents = payload(10)
    root = tmp_path / "models"
    unrelated = root / "unrelated.part"
    root.mkdir()
    unrelated.write_bytes(payload(11))

    class CancellingStream(BytesIO):
        @override
        def read(self, size: int | None = -1) -> bytes:
            result = super().read(size)
            cancelled.set()
            return result

    def opener(_url: str, *, timeout: float) -> BinaryIO:
        del timeout
        return CancellingStream(contents)

    with pytest.raises(ModelDownloadCancelled):
        ModelCache(root, opener=opener).ensure(
            model({"model.safetensors": contents}), cancelled=cancelled
        )

    assert unrelated.read_bytes() == payload(11)
    assert list(root.rglob("*.part")) == [unrelated]
    assert not list(root.rglob("model.safetensors"))


def test_cancellation_at_cached_completion_prevents_model_loading(
    tmp_path: Path,
) -> None:
    files = {"model.safetensors": payload(16)}
    cache = ModelCache(tmp_path / "models", opener=MemoryOpener(files))
    spec = model(files)
    destination = cache.ensure(spec)
    cancelled = Event()

    def report(_identity: str, completed: int, total: int) -> None:
        if completed == total:
            cancelled.set()

    with pytest.raises(ModelDownloadCancelled):
        cache.ensure(spec, progress=report, cancelled=cancelled)

    assert (destination / "model.safetensors").read_bytes() == files[
        "model.safetensors"
    ]


@pytest.mark.parametrize("alias", ["root", "role", "artifact_parent", "artifact"])
@pytest.mark.parametrize("operation", ["ensure", "cached"])
def test_symlink_aliases_cannot_escape_cache(
    tmp_path: Path,
    alias: str,
    operation: str,
) -> None:
    root = tmp_path / "models"
    outside = tmp_path / "outside"
    outside.mkdir()
    contents = payload(12)
    spec = model({"nested/model.safetensors": contents})
    destination = root / spec.role.value / spec.revision
    if alias == "root":
        root.symlink_to(outside, target_is_directory=True)
    elif alias == "role":
        root.mkdir()
        (root / spec.role.value).symlink_to(outside, target_is_directory=True)
    else:
        destination.mkdir(parents=True)
        if alias == "artifact_parent":
            (destination / "nested").symlink_to(outside, target_is_directory=True)
        else:
            (destination / "nested").mkdir()
            (destination / "nested/model.safetensors").symlink_to(outside / "stolen")
    opener = MemoryOpener({"nested/model.safetensors": contents})

    with pytest.raises(ModelArtifactError, match="symlink|escape"):
        cache = ModelCache(root, opener=opener)
        if operation == "cached":
            cache.cached(spec)
        else:
            cache.ensure(spec)

    assert not list(outside.iterdir())
    assert not opener.calls


def test_download_reads_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk_size = 16
    monkeypatch.setattr("scs.local_models.artifacts.DOWNLOAD_CHUNK_BYTES", chunk_size)
    contents = payload(13, chunk_size * 3 + 7)
    reads: list[int] = []

    class BoundedStream(BytesIO):
        @override
        def read(self, size: int | None = -1) -> bytes:
            assert size is not None and 0 < size <= chunk_size
            reads.append(size)
            return super().read(size)

    def opener(_url: str, *, timeout: float) -> BinaryIO:
        del timeout
        return BoundedStream(contents)

    ModelCache(tmp_path / "models", opener=opener).ensure(
        model({"model.safetensors": contents})
    )

    assert len(reads) >= 4


@pytest.mark.parametrize("scheme", ["https", "http", "ftp"])
def test_artifact_downloads_allow_only_https_redirects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scheme: str,
) -> None:
    contents = payload(15)

    def redirected_open(
        director: RedirectDirector,
        fullurl: str | Request,
        data: object = None,
        timeout: float | None = None,
    ) -> BinaryIO:
        del data, timeout
        handler = next(
            item for item in director.handlers if isinstance(item, HTTPRedirectHandler)
        )
        request = fullurl if isinstance(fullurl, Request) else Request(fullurl)
        redirected = handler.redirect_request(
            request,
            BytesIO(),
            302,
            "Found",
            HTTPMessage(),
            f"{scheme}://cdn.example.invalid/model.safetensors",
        )
        assert redirected is not None
        return BytesIO(contents)

    monkeypatch.setattr(OpenerDirector, "open", redirected_open)
    cache = ModelCache(tmp_path / "models")
    spec = model({"model.safetensors": contents})
    if scheme == "https":
        destination = cache.ensure(spec)
        assert (destination / "model.safetensors").read_bytes() == contents
    else:
        with pytest.raises(ModelArtifactError, match="HTTPS"):
            cache.ensure(spec)


@pytest.mark.parametrize(
    ("size", "digest"),
    [
        (-1, hashlib.sha256(b"").hexdigest()),
        (0, "not-a-digest"),
    ],
)
def test_artifact_metadata_rejects_invalid_bounds(
    size: int,
    digest: str,
) -> None:
    with pytest.raises(ValueError):
        ModelArtifact("config.json", size, digest)
