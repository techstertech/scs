"""Upgrading verifies checksums before shutdown and replaces only the uv tool."""

from __future__ import annotations

import hashlib
import http.client
import io
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest

import scs.upgrade as module
from scs.config import SCSSettings
from scs.service import ProcessLock


@dataclass
class UpgradeSandbox:
    settings: SCSSettings
    user_home: Path
    prefix: Path
    launcher: Path
    events: list[str]
    downloads: list[str]
    install_commands: list[list[str]]
    wheel_bytes: bytes
    constraints_bytes: bytes


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> UpgradeSandbox:
    user_home = tmp_path / "user"
    prefix = user_home / "tools" / "scs"
    prefix.mkdir(parents=True)
    (prefix / "uv-receipt.toml").write_text("[tool]\n", encoding="utf-8")
    package_file = prefix / "lib" / "scs" / "upgrade.py"
    package_file.parent.mkdir(parents=True)
    package_file.touch()

    bin_dir = user_home / "bin"
    bin_dir.mkdir(parents=True)
    launcher = bin_dir / "scs"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    launcher.chmod(0o755)

    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: user_home))
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(bin_dir))
    monkeypatch.setattr(module.sys, "prefix", str(prefix))
    monkeypatch.setattr(module, "__file__", str(package_file))
    monkeypatch.setattr(module.shutil, "which", lambda _command: "/fake/bin/uv")
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(module, "__version__", "0.7.2")

    settings = SCSSettings(
        home=user_home / "data",
        runtime_dir=user_home / "runtime",
        model_cache=user_home / "cache",
        log_dir=user_home / "logs",
    )
    settings.home.mkdir()
    settings.runtime_dir.mkdir()

    wheel_bytes = b"wheel-binary-payload"
    constraints_bytes = b"pydantic==2.13.4\n"
    events: list[str] = []
    downloads: list[str] = []
    install_commands: list[list[str]] = []

    def fake_resolve_latest() -> str:
        events.append("resolve_latest")
        return "0.8.0"

    def fake_download(
        url: str, destination: Path, *, max_bytes: int | None = None
    ) -> None:
        del max_bytes
        downloads.append(url)
        if destination.name == "SHA256SUMS":
            version = url.split("/v", 1)[1].split("/", 1)[0]
            wheel_name = f"scs-{version}-cp314-cp314-macosx_11_0_arm64.whl"
            constraints_name = f"scs-{version}-constraints.txt"
            wheel_sha = hashlib.sha256(wheel_bytes).hexdigest()
            constraints_sha = hashlib.sha256(constraints_bytes).hexdigest()
            destination.write_text(
                f"{wheel_sha}  {wheel_name}\n{constraints_sha} *{constraints_name}\n",
                encoding="utf-8",
            )
        elif destination.name.endswith(".whl"):
            destination.write_bytes(wheel_bytes)
        elif destination.name.endswith("-constraints.txt"):
            destination.write_bytes(constraints_bytes)
        else:
            raise AssertionError(f"unexpected download target: {destination}")

    class Controller:
        def __init__(self, actual: SCSSettings) -> None:
            assert actual is settings

        async def stop(self, *, cancel_active: bool = False) -> bool:
            assert cancel_active is True
            with pytest.raises(RuntimeError):
                with ProcessLock(settings.paths.runtime / ".bootstrap.lock"):
                    pytest.fail("bootstrap lock must protect shutdown")
            events.append("stop")
            return True

    def fake_run(
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        if command[0] == "/fake/bin/uv":
            assert env is not None
            assert env["UV_TOOL_DIR"] == str(prefix.parent)
            with pytest.raises(RuntimeError):
                with ProcessLock(settings.paths.home / ".daemon.lock"):
                    pytest.fail("writer lock must protect replacement")
            install_commands.append(command)
            events.append("install")
            return subprocess.CompletedProcess(command, 0, "", "")
        if command == [str(launcher), "version"]:
            assert capture_output is True
            assert text is True
            events.append("verify")
            installed = Path(install_commands[-1][-1]).name.split("-")[1]
            return subprocess.CompletedProcess(command, 0, f"{installed}\n", "")
        raise AssertionError(f"unexpected subprocess command: {command}")

    monkeypatch.setattr(module, "_resolve_latest_version", fake_resolve_latest)
    monkeypatch.setattr(module, "_download_file", fake_download)
    monkeypatch.setattr(module, "DaemonController", Controller)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    return UpgradeSandbox(
        settings=settings,
        user_home=user_home,
        prefix=prefix,
        launcher=launcher,
        events=events,
        downloads=downloads,
        install_commands=install_commands,
        wheel_bytes=wheel_bytes,
        constraints_bytes=constraints_bytes,
    )


def test_upgrade_resolves_latest_verifies_checksums_stops_daemon_and_installs(
    sandbox: UpgradeSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UV_TOOL_DIR", str(sandbox.user_home / "other-tools"))

    result = module.upgrade(sandbox.settings)

    assert result == module.UpgradeResult(
        current_version="0.7.2",
        target_version="0.8.0",
        updated=True,
        checked_only=False,
        launcher=sandbox.launcher,
    )
    assert sandbox.events == ["resolve_latest", "stop", "install", "verify"]
    assert sandbox.downloads == [
        "https://github.com/techstertech/scs/releases/download/v0.8.0/SHA256SUMS",
        "https://github.com/techstertech/scs/releases/download/v0.8.0/scs-0.8.0-cp314-cp314-macosx_11_0_arm64.whl",
        "https://github.com/techstertech/scs/releases/download/v0.8.0/scs-0.8.0-constraints.txt",
    ]
    assert sandbox.install_commands[0][:6] == [
        "/fake/bin/uv",
        "tool",
        "install",
        "--python",
        "3.14",
        "--constraints",
    ]
    assert sandbox.install_commands[0][7] == "--force"


def test_upgrade_is_noop_when_already_on_target_unless_forced(
    sandbox: UpgradeSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "__version__", "0.8.0")

    noop = module.upgrade(sandbox.settings)
    assert noop.updated is False
    assert noop.checked_only is False
    assert sandbox.downloads == []
    assert sandbox.events == ["resolve_latest"]

    forced = module.upgrade(sandbox.settings, force=True)
    assert forced.updated is True
    assert forced.target_version == "0.8.0"
    assert sandbox.events == ["resolve_latest", "resolve_latest", "stop", "install", "verify"]


def test_upgrade_check_only_never_downloads_or_stops_daemon(
    sandbox: UpgradeSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --check works even outside a uv-managed tool installation.
    monkeypatch.setattr(module.shutil, "which", lambda _cmd: None)

    result = module.upgrade(sandbox.settings, check_only=True)

    assert result == module.UpgradeResult(
        current_version="0.7.2",
        target_version="0.8.0",
        updated=False,
        checked_only=True,
        launcher=None,
    )
    assert sandbox.downloads == []
    assert sandbox.events == ["resolve_latest"]


def test_upgrade_supports_explicit_version_with_or_without_v_prefix(
    sandbox: UpgradeSandbox,
) -> None:
    result = module.upgrade(sandbox.settings, version="v0.7.1")

    assert result.updated is True
    assert result.target_version == "0.7.1"
    assert "resolve_latest" not in sandbox.events
    assert any("v0.7.1/scs-0.7.1-" in url for url in sandbox.downloads)


def test_upgrade_rejects_checksum_mismatch_before_stopping_daemon(
    sandbox: UpgradeSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_download = module._download_file

    def tampered_download(
        url: str, destination: Path, *, max_bytes: int | None = None
    ) -> None:
        original_download(url, destination, max_bytes=max_bytes)
        if destination.name.endswith(".whl"):
            destination.write_bytes(b"corrupted-wheel")

    monkeypatch.setattr(module, "_download_file", tampered_download)

    with pytest.raises(module.UpgradeError, match="checksum mismatch"):
        module.upgrade(sandbox.settings)

    assert "stop" not in sandbox.events
    assert "install" not in sandbox.events


def test_upgrade_rejects_missing_checksum_entry_before_stopping_daemon(
    sandbox: UpgradeSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    def incomplete_manifest(
        url: str, destination: Path, *, max_bytes: int | None = None
    ) -> None:
        del url, max_bytes
        destination.write_text("deadbeef  unrelated.txt\n", encoding="utf-8")

    monkeypatch.setattr(module, "_download_file", incomplete_manifest)

    with pytest.raises(module.UpgradeError, match="SHA256SUMS.*missing"):
        module.upgrade(sandbox.settings)

    assert "stop" not in sandbox.events


def test_upgrade_rejects_non_uv_environment_and_unsupported_platform(
    sandbox: UpgradeSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module.platform, "system", lambda: "Windows")
    with pytest.raises(module.UpgradeError, match="unsupported platform"):
        module.upgrade(sandbox.settings, version="0.8.0")

    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    (sandbox.prefix / "uv-receipt.toml").unlink()
    with pytest.raises(module.UpgradeError, match="uv-installed SCS executable"):
        module.upgrade(sandbox.settings, version="0.8.0")


def test_upgrade_rejects_invalid_version_and_non_https_redirect(
    sandbox: UpgradeSandbox,
) -> None:
    with pytest.raises(module.UpgradeError, match="invalid SCS release version"):
        module.upgrade(sandbox.settings, version="not-a-version")

    handler = module._HttpsOnlyRedirectHandler()
    request = urllib.request.Request("https://github.com/techstertech/scs/releases/latest")
    with pytest.raises(module.UpgradeError, match="refused non-HTTPS redirect"):
        handler.redirect_request(
            request,
            io.BytesIO(),
            302,
            "Found",
            http.client.HTTPMessage(),
            "http://example.invalid/downgrade",
        )


def test_upgrade_detects_post_install_version_mismatch(
    sandbox: UpgradeSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_run = module.subprocess.run

    def mismatched_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if command == [str(sandbox.launcher), "version"]:
            return subprocess.CompletedProcess(command, 0, "0.7.2\n", "")
        return original_run(command, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(module.subprocess, "run", mismatched_run)

    with pytest.raises(module.UpgradeError, match="installed version mismatch"):
        module.upgrade(sandbox.settings)


def test_resolve_latest_version_parses_https_tag_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://github.com/techstertech/scs/releases/tag/v0.8.0"

    class FakeOpener:
        def open(self, request: urllib.request.Request, timeout: float) -> FakeResponse:
            assert request.full_url == module.LATEST_RELEASE_URL
            assert request.get_method() == "HEAD"
            assert timeout == module.REQUEST_TIMEOUT_SECONDS
            return FakeResponse()

    monkeypatch.setattr(module, "_https_opener", lambda: FakeOpener())

    assert module._resolve_latest_version() == "0.8.0"


def test_platform_wheel_name_supports_macos_arm64_and_linux_x86_64() -> None:
    assert (
        module._platform_wheel_name("0.8.0", system="Darwin", machine="arm64")
        == "scs-0.8.0-cp314-cp314-macosx_11_0_arm64.whl"
    )
    assert (
        module._platform_wheel_name("0.8.0", system="Linux", machine="x86_64")
        == "scs-0.8.0-cp314-cp314-manylinux_2_28_x86_64.whl"
    )
    assert os.access(Path(__file__), os.R_OK)
