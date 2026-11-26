"""Upgrade the executing uv-installed SCS tool from verified GitHub Releases."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import http.client
import os
import platform
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol, cast, override

from scs import __version__
from scs.config import SCSSettings
from scs.daemon import DaemonController
from scs.service import ProcessLock

TOOL_NAME = "scs"
UV_RECEIPT = "uv-receipt.toml"
PYTHON_VERSION = "3.14"
RELEASE_HOST = "github.com"
RELEASE_REPOSITORY = "techstertech/scs"
LATEST_RELEASE_URL = f"https://{RELEASE_HOST}/{RELEASE_REPOSITORY}/releases/latest"
RELEASE_DOWNLOAD_BASE = (
    f"https://{RELEASE_HOST}/{RELEASE_REPOSITORY}/releases/download"
)
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_CHECKSUM_MANIFEST_BYTES = 65_536
DOWNLOAD_CHUNK_BYTES = 65_536

_VERSION_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+){2}(?:[.-][0-9A-Za-z]+)*$")
_TAG_PATH_PATTERN = re.compile(r"/releases/tag/v([^/?#]+)$")
_SUPPORTED_PLATFORMS: dict[tuple[str, str], str] = {
    ("Darwin", "arm64"): "cp314-cp314-macosx_11_0_arm64",
    ("Linux", "x86_64"): "cp314-cp314-manylinux_2_28_x86_64",
}


class UpgradeError(RuntimeError):
    """Upgrade could not complete; the message identifies its stage."""


@dataclass(frozen=True, slots=True)
class UpgradeResult:
    """Outcome of one `scs upgrade` invocation."""

    current_version: str
    target_version: str
    updated: bool
    checked_only: bool
    launcher: Path | None = None


@dataclass(frozen=True, slots=True)
class _Installation:
    uv: str
    root: Path


class _HttpResponse(Protocol):
    def __enter__(self) -> "_HttpResponse": ...
    def __exit__(self, *_args: object) -> None: ...
    def geturl(self) -> str: ...
    def read(self, size: int = -1) -> bytes: ...


class _UrlOpener(Protocol):
    def open(
        self, fullurl: urllib.request.Request, *, timeout: float
    ) -> _HttpResponse: ...


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse any redirect that downgrades from HTTPS."""

    @override
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or not parsed.netloc:
            raise UpgradeError(f"refused non-HTTPS redirect target: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _normalize_version(raw: str) -> str:
    """Validate an explicit or resolved release version without a leading `v`."""

    cleaned = raw.strip()
    if cleaned.startswith(("v", "V")):
        cleaned = cleaned[1:]
    if not cleaned or not _VERSION_PATTERN.fullmatch(cleaned):
        raise UpgradeError(f"invalid SCS release version: {raw!r}")
    return cleaned


def _platform_wheel_name(
    version: str,
    *,
    system: str | None = None,
    machine: str | None = None,
) -> str:
    """Return the exact release wheel filename for the active host platform."""

    os_name = system if system is not None else platform.system()
    architecture = machine if machine is not None else platform.machine()
    wheel_tag = _SUPPORTED_PLATFORMS.get((os_name, architecture))
    if wheel_tag is None:
        raise UpgradeError(f"unsupported platform: {os_name} {architecture}")
    return f"{TOOL_NAME}-{version}-{wheel_tag}.whl"


def _installation() -> _Installation:
    """Verify the executing process comes from a uv-managed SCS installation."""

    uv = shutil.which("uv")
    if uv is None:
        raise UpgradeError(
            "uv is required on PATH; install uv before upgrading SCS"
        )

    root = Path(sys.prefix).resolve()
    if (
        root.name != TOOL_NAME
        or not (root / UV_RECEIPT).is_file()
        or not Path(__file__).resolve().is_relative_to(root)
    ):
        raise UpgradeError(
            "run upgrade with the uv-installed SCS executable; "
            "source-tree, pip, and pipx environments are not supported"
        )
    return _Installation(uv=uv, root=root)


def _https_opener() -> _UrlOpener:
    context = ssl.create_default_context()
    return cast(
        _UrlOpener,
        urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=context),
            _HttpsOnlyRedirectHandler(),
        ),
    )


def _resolve_latest_version() -> str:
    """Resolve the latest stable release tag via GitHub's `/releases/latest` redirect."""

    request = urllib.request.Request(
        LATEST_RELEASE_URL,
        method="HEAD",
        headers={"User-Agent": f"scs/{__version__}"},
    )
    opener = _https_opener()
    try:
        with opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            final_url = str(response.geturl())
    except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
        raise UpgradeError(f"failed to resolve latest release: {error}") from error

    parsed = urllib.parse.urlsplit(final_url)
    if parsed.scheme != "https":
        raise UpgradeError(f"refused non-HTTPS release URL: {final_url}")
    match = _TAG_PATH_PATTERN.search(parsed.path)
    if match is None:
        raise UpgradeError(
            f"could not determine latest release version from {final_url}"
        )
    return _normalize_version(match.group(1))


def _download_file(url: str, destination: Path, *, max_bytes: int | None = None) -> None:
    """Download one HTTPS release asset to disk with optional byte ceiling."""

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise UpgradeError(f"refused non-HTTPS download URL: {url}")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": f"scs/{__version__}"},
    )
    opener = _https_opener()
    total = 0
    try:
        with opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            final_scheme = urllib.parse.urlsplit(str(response.geturl())).scheme
            if final_scheme != "https":
                raise UpgradeError(
                    f"refused non-HTTPS asset URL: {response.geturl()}"
                )
            with destination.open("wb") as handle:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    if max_bytes is not None and total > max_bytes:
                        raise UpgradeError(
                            f"release asset exceeded maximum size ({max_bytes} bytes): {destination.name}"
                        )
                    handle.write(chunk)
    except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
        raise UpgradeError(f"failed to download {destination.name}: {error}") from error


def _parse_sha256sums(manifest_text: str) -> dict[str, str]:
    """Parse a standard `SHA256SUMS` file into `{filename: digest}` entries."""

    digests: dict[str, str] = {}
    for line in manifest_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            continue
        digest, name = parts[0].lower(), parts[1].lstrip("*").strip()
        if len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest) and name:
            digests[name] = digest
    return digests


def _verify_sha256(path: Path, expected_digest: str) -> None:
    """Verify the SHA-256 checksum of a downloaded file."""

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    actual_digest = hasher.hexdigest()
    if not hmac.compare_digest(actual_digest, expected_digest.lower()):
        raise UpgradeError(
            f"checksum mismatch for {path.name}: expected {expected_digest}, got {actual_digest}"
        )


def _launcher_path() -> Path:
    bin_dir_raw = os.environ.get("UV_TOOL_BIN_DIR")
    bin_dir = (
        Path(bin_dir_raw).expanduser()
        if bin_dir_raw
        else Path.home() / ".local" / "bin"
    )
    return bin_dir / TOOL_NAME


def upgrade(
    settings: SCSSettings,
    *,
    version: str | None = None,
    check_only: bool = False,
    force: bool = False,
) -> UpgradeResult:
    """Upgrade or roll back the executing uv-installed SCS release safely.

    Asset downloads and SHA-256 verification complete before acquiring lifecycle
    locks or stopping the running daemon. The daemon stays stopped after tool
    replacement so the next CLI command or MCP bridge starts it lazily.
    """

    current_version = __version__
    target_version = (
        _normalize_version(version)
        if version is not None
        else _resolve_latest_version()
    )
    wheel_name = _platform_wheel_name(target_version)

    if check_only:
        return UpgradeResult(
            current_version=current_version,
            target_version=target_version,
            updated=False,
            checked_only=True,
        )

    installation = _installation()
    if target_version == current_version and not force:
        return UpgradeResult(
            current_version=current_version,
            target_version=target_version,
            updated=False,
            checked_only=False,
        )

    paths = settings.paths
    bootstrap_path = paths.runtime / ".bootstrap.lock"
    writer_path = paths.home / ".daemon.lock"
    for lock_path in (bootstrap_path, writer_path):
        if lock_path.is_symlink():
            raise UpgradeError(f"unsafe lifecycle lock symlink: {lock_path}")

    constraints_name = f"{TOOL_NAME}-{target_version}-constraints.txt"
    release_base = f"{RELEASE_DOWNLOAD_BASE}/v{target_version}"

    temp_dir = Path(tempfile.mkdtemp(prefix="scs-upgrade-"))
    try:
        temp_dir.chmod(0o700)
        checksums_path = temp_dir / "SHA256SUMS"
        wheel_path = temp_dir / wheel_name
        constraints_path = temp_dir / constraints_name

        _download_file(
            f"{release_base}/SHA256SUMS",
            checksums_path,
            max_bytes=MAX_CHECKSUM_MANIFEST_BYTES,
        )
        digests = _parse_sha256sums(checksums_path.read_text(encoding="utf-8"))
        for asset_name in (wheel_name, constraints_name):
            if asset_name not in digests:
                raise UpgradeError(
                    f"SHA256SUMS for v{target_version} is missing {asset_name}"
                )

        _download_file(f"{release_base}/{wheel_name}", wheel_path)
        _download_file(f"{release_base}/{constraints_name}", constraints_path)
        _verify_sha256(wheel_path, digests[wheel_name])
        _verify_sha256(constraints_path, digests[constraints_name])

        stage = "lifecycle shutdown"
        try:
            with ProcessLock(bootstrap_path):
                asyncio.run(DaemonController(settings).stop(cancel_active=True))
                with ProcessLock(writer_path):
                    stage = "package installation"
                    # Select the executing tool directory even when UV_TOOL_DIR
                    # changed after initial installation.
                    environment = os.environ | {
                        "UV_TOOL_DIR": str(installation.root.parent)
                    }
                    _ = subprocess.run(
                        [
                            installation.uv,
                            "tool",
                            "install",
                            "--python",
                            PYTHON_VERSION,
                            "--constraints",
                            str(constraints_path),
                            "--force",
                            str(wheel_path),
                        ],
                        env=environment,
                        check=True,
                    )

                    stage = "post-install verification"
                    launcher = _launcher_path()
                    if not launcher.is_file() or not os.access(launcher, os.X_OK):
                        raise UpgradeError(
                            f"installed SCS launcher not found at {launcher}"
                        )
                    completed = subprocess.run(
                        [str(launcher), "version"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    installed_version = completed.stdout.strip()
                    if installed_version != target_version:
                        raise UpgradeError(
                            f"installed version mismatch: expected {target_version}, found {installed_version}"
                        )
        except UpgradeError:
            raise
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            raise UpgradeError(f"{stage} failed: {error}") from error
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return UpgradeResult(
        current_version=current_version,
        target_version=target_version,
        updated=True,
        checked_only=False,
        launcher=launcher,
    )
