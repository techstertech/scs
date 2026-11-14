"""Local-model smoke checks own their disposable state and fail closed."""

from __future__ import annotations

import errno
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import NoReturn, Protocol, cast

import pytest


class State(Protocol):
    root: Path
    data: Path
    fixture: Path
    models: Path
    runtime: Path
    logs: Path


class Smoke(Protocol):
    def prepare_state(self, root: Path) -> State: ...
    def validate_output(self, root: Path, output: Path) -> Path: ...
    def child_environment(self, state: State) -> dict[str, str]: ...
    def network_control(self, offline: bool) -> dict[str, object]: ...
    def validate_query(
        self, result: dict[str, object], expected_model: str
    ) -> None: ...
    def footprint_lines(self, contents: str) -> list[str]: ...


def load_smoke() -> Smoke:
    path = Path(__file__).parents[2] / "scripts/smoke-local-models.py"
    spec = importlib.util.spec_from_file_location("local_model_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(Smoke, cast(object, module))


def test_unowned_nonempty_state_is_never_adopted_or_modified(tmp_path: Path) -> None:
    smoke = load_smoke()
    state_root = tmp_path / "state"
    state_root.mkdir()
    sentinel = state_root / "user-data"
    sentinel.write_bytes(b"preserve existing state")

    with pytest.raises(ValueError, match="ownership"):
        smoke.prepare_state(state_root)

    assert list(state_root.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"preserve existing state"


def test_fixture_survives_reuse_and_modified_source_is_not_replaced(
    tmp_path: Path,
) -> None:
    smoke = load_smoke()
    state_root = tmp_path / "state"
    state = smoke.prepare_state(state_root)
    before = {path.name: path.read_bytes() for path in state.fixture.iterdir()}

    reused = smoke.prepare_state(state_root)
    assert {path.name: path.read_bytes() for path in reused.fixture.iterdir()} == before
    source = next(iter(state.fixture.glob("*.py")))
    source.chmod(0o600)
    source.write_text("# preserve user edit\n")

    with pytest.raises(ValueError, match="fixture"):
        smoke.prepare_state(state_root)

    assert source.read_text() == "# preserve user edit\n"


@pytest.mark.parametrize(
    "protected",
    ["fixture/helpers.py", "data/models/weight.json", ".scs-local-model-smoke.json"],
)
def test_output_cannot_overwrite_fixture_cache_or_ownership(
    tmp_path: Path,
    protected: str,
) -> None:
    smoke = load_smoke()
    state = smoke.prepare_state(tmp_path / "state")

    with pytest.raises(ValueError, match="output"):
        smoke.validate_output(state.root, state.root / protected)

    alias = tmp_path / "alias.json"
    alias.symlink_to(state.root / protected)
    with pytest.raises(ValueError, match="output"):
        smoke.validate_output(state.root, alias)


def test_child_environment_replaces_model_settings_without_changing_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = load_smoke()
    state = smoke.prepare_state(tmp_path / "state")
    monkeypatch.setenv("HOME", str(tmp_path / "real-user"))
    monkeypatch.setenv("SCS_HOME", "/must-not-use")
    monkeypatch.setenv("scs_decision_model", "must-not-use")
    monkeypatch.setenv("SCS_UNRELATED_SETTING", "must-not-inherit")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret")
    monkeypatch.setenv("PYTHONPATH", "/must-not-import")

    environment = smoke.child_environment(state)

    assert environment["HOME"] == str(tmp_path / "real-user")
    assert environment["SCS_HOME"] == str(state.data)
    assert environment["SCS_MODEL_CACHE"] == str(state.models)
    assert environment["SCS_MODEL_PROFILE"] == "local"
    assert environment["SCS_EMBEDDING_PROVIDER"] == "local"
    assert environment["SCS_RERANKING_PROVIDER"] == "local"
    assert environment["SCS_DECISION_PROVIDER"] == "local"
    assert environment["SCS_DECISION_MODEL"].startswith("aac6fef/laya-mlx@")
    assert environment["SCS_EMBEDDING_MODEL"].startswith("Qwen/Qwen3-Embedding-0.6B@")
    assert "OPENAI_API_KEY" not in environment
    assert "PYTHONPATH" not in environment
    assert "SCS_UNRELATED_SETTING" not in environment
    assert "scs_decision_model" not in environment


def test_offline_control_requires_explicit_network_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = load_smoke()

    def denied(_address: object, *, timeout: float) -> NoReturn:
        del timeout
        raise PermissionError(errno.EPERM, "denied")

    monkeypatch.setattr("socket.create_connection", denied)
    assert smoke.network_control(True)["errno"] == errno.EPERM

    def unavailable(_address: object, *, timeout: float) -> NoReturn:
        del timeout
        raise OSError(errno.ENETUNREACH, "unreachable")

    monkeypatch.setattr("socket.create_connection", unavailable)
    with pytest.raises(ValueError, match="EPERM"):
        smoke.network_control(True)


def test_routing_fallback_is_a_failed_smoke_result() -> None:
    smoke = load_smoke()
    identity = "fixture/local-router@revision"
    result: dict[str, object] = {
        "routing": {"provider": "local", "model": identity, "fallback_applied": False},
        "evidence": {
            "symbols": [{"name": "preserve_input", "file_path": "helpers.py"}]
        },
    }
    smoke.validate_query(result, identity)

    result["routing"] = {
        "provider": "local",
        "model": identity,
        "fallback_applied": True,
    }
    with pytest.raises(ValueError, match="fallback"):
        smoke.validate_query(result, identity)


def test_residency_parser_keeps_only_memory_totals() -> None:
    smoke = load_smoke()
    values = [f"{amount / 10:.1f}G" for amount in range(30, 32)]
    contents = "\n".join(
        [
            "Process: sensitive-path-and-arguments",
            "Private mapping: do not retain",
            f" Physical footprint: {values[0]}",
            f" Physical footprint (peak): {values[1]}",
            "Physical footprint detail: not an allowed total",
        ]
    )

    assert smoke.footprint_lines(contents) == [
        f"Physical footprint: {values[0]}",
        f"Physical footprint (peak): {values[1]}",
    ]


def test_failed_child_invalidates_previous_success_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text(json.dumps({"status": "completed"}))
    script = Path(__file__).parents[2] / "scripts/smoke-local-models.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--command",
            "/usr/bin/false",
            "--state-root",
            str(tmp_path / "state"),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode != 0
    assert json.loads(output.read_text())["status"] == "failed"
