"""Validate bundled models through fresh MCP sessions and persistent private state.

Run once online, then run again under an OS network sandbox with ``--offline``.
The second run queries the existing index before any ingestion request. A routing
fallback is a failed check; the command never retries it into a later success.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import perf_counter
from typing import Final, Protocol, cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from scs.local_models.artifacts import ModelRole, load_manifest
from scs.mcp.inventory import MCP_TOOL_NAMES
from scs.wire.client import SCSClient

OWNERSHIP_FILE: Final[str] = ".scs-local-model-smoke.json"
FIXTURE_VERSION: Final[int] = 1
GOAL: Final[str] = "Find the function that returns the input integer unchanged."
EXPECTED_SYMBOL: Final[str] = "preserve_input"
MCP_TIMEOUT_SECONDS: Final[float] = 60.0
INDEX_TIMEOUT_SECONDS: Final[float] = 900.0
SHUTDOWN_TIMEOUT_SECONDS: Final[float] = 60.0


@dataclass(frozen=True, slots=True)
class StatePaths:
    root: Path
    data: Path
    fixture: Path
    models: Path
    runtime: Path
    logs: Path


def mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected an object in the smoke response")
    return cast(dict[str, object], value)


def objects(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("expected a list in the smoke response")
    return [mapping(item) for item in cast(list[object], value)]


def fixture_sources() -> dict[str, str]:
    definitions = (
        ("preserve_input", "Return the input integer unchanged.", "value"),
        ("increment_input", "Add one to the input integer.", "value + 1"),
        ("double_input", "Multiply the input integer by two.", "value * 2"),
    )
    helpers = "\n\n".join(
        f'def {name}(value: int) -> int:\n    """{description}"""\n    return {expression}\n'
        for name, description, expression in definitions
    )
    return {
        "helpers.py": helpers,
        "consumer.py": (
            "from helpers import preserve_input\n\n"
            "def preserve_measurement(value: int) -> int:\n"
            "    return preserve_input(value)\n"
        ),
    }


def _paths(root: Path) -> StatePaths:
    return StatePaths(
        root=root,
        data=root / "data",
        fixture=root / "fixture",
        models=root / "data/models",
        runtime=root / "runtime",
        logs=root / "logs",
    )


def _safe_directory(root: Path, path: Path) -> None:
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("smoke state contains an unsafe directory alias")
    if path.exists() and not path.is_dir():
        raise ValueError("smoke state requires real directories")


def prepare_state(root: Path) -> StatePaths:
    """Create only empty state, or reuse a root with this command's exact marker."""

    if root.is_symlink():
        raise ValueError("smoke ownership cannot use a symlink root")
    root = root.expanduser().resolve()
    marker = root / OWNERSHIP_FILE
    expected: dict[str, object] = {
        "schema_version": 1,
        "kind": "scs-local-model-smoke",
        "root": str(root),
        "fixture_version": FIXTURE_VERSION,
    }
    if root.exists() and not root.is_dir():
        raise ValueError("smoke ownership requires a directory")
    if root.exists() and any(root.iterdir()):
        if marker.is_symlink() or not marker.is_file():
            raise ValueError("nonempty state root lacks smoke ownership")
        observed = cast(object, json.loads(marker.read_text()))
        if observed != expected:
            raise ValueError(
                "smoke ownership marker does not match this root or fixture"
            )
    else:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.chmod(0o700)
        marker.write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n")
        marker.chmod(0o600)

    paths = _paths(root)
    for directory in (
        paths.data,
        paths.models,
        paths.runtime,
        paths.logs,
        paths.fixture,
    ):
        _safe_directory(root, directory)
    sources = fixture_sources()
    if paths.fixture.exists() and any(paths.fixture.iterdir()):
        if {path.name for path in paths.fixture.iterdir()} != set(sources):
            raise ValueError("existing smoke fixture has unexpected entries")
        for name, text in sources.items():
            path = paths.fixture / name
            if path.is_symlink() or not path.is_file() or path.read_text() != text:
                raise ValueError(
                    "existing smoke fixture differs from its versioned source"
                )
    else:
        paths.fixture.mkdir(mode=0o700, exist_ok=True)
        for name, text in sources.items():
            path = paths.fixture / name
            path.write_text(text)
            path.chmod(0o444)
    return paths


def validate_output(root: Path, output: Path) -> Path:
    """Preserve fixture, model cache, lifecycle files, and ownership metadata."""

    root = root.expanduser().resolve()
    output = output.expanduser().resolve()
    protected = (root / "fixture", root / "data", root / "runtime", root / "logs")
    if (
        output == root
        or output == root / OWNERSHIP_FILE
        or any(output.is_relative_to(path) for path in protected)
        or output.suffix != ".json"
    ):
        raise ValueError("output must be a JSON report outside protected smoke state")
    return output


def child_environment(state: StatePaths) -> dict[str, str]:
    """Override model ownership in the child without changing the user's HOME."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("SCS_")
        and key.upper() not in {"OPENAI_API_KEY", "PYTHONPATH"}
    }
    models = load_manifest()
    environment.update(
        {
            "SCS_HOME": str(state.data),
            "SCS_MODEL_CACHE": str(state.models),
            "SCS_RUNTIME_DIR": str(state.runtime),
            "SCS_LOG_DIR": str(state.logs),
            "SCS_MODEL_PROFILE": "local",
            "SCS_AUTO_REINDEX_ENABLED": "false",
            "SCS_EMBEDDING_PROVIDER": "local",
            "SCS_EMBEDDING_DIMENSION": "1024",
            "SCS_EMBEDDING_MODEL": models[ModelRole.EMBEDDING].identity,
            "SCS_RERANKING_PROVIDER": "local",
            "SCS_RERANKING_MODEL": models[ModelRole.RERANKING].identity,
            "SCS_DECISION_PROVIDER": "local",
            "SCS_DECISION_MODEL": models[ModelRole.DECISION].identity,
            "SCS_OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:9/v1",
            "SCS_OPENAI_COMPATIBLE_TRUSTED_HOSTS": "[]",
            "SCS_DECISION_BASE_URL": "http://127.0.0.1:9/v1",
            "SCS_DECISION_TRUSTED_HOSTS": "[]",
        }
    )
    return environment


def network_control(offline: bool) -> dict[str, object]:
    if not offline:
        return {"requested": False, "network_denial_verified": False}
    try:
        connection = socket.create_connection(("1.1.1.1", 443), timeout=2.0)
    except OSError as error:
        if error.errno == errno.EPERM:
            return {
                "requested": True,
                "network_denial_verified": True,
                "errno": error.errno,
            }
        raise ValueError(
            "offline control requires EPERM, not an unreachable network"
        ) from error
    connection.close()
    raise ValueError("offline control requires EPERM; outbound networking is available")


def source_hashes(state: StatePaths) -> dict[str, str]:
    return {
        name: hashlib.sha256((state.fixture / name).read_bytes()).hexdigest()
        for name in fixture_sources()
    }


def validate_query(result: dict[str, object], expected_model: str) -> None:
    routing = mapping(result["routing"])
    if routing.get("fallback_applied") is not False:
        raise ValueError("first local query used routing fallback")
    if routing.get("provider") != "local" or routing.get("model") != expected_model:
        raise ValueError("first local query used another decision implementation")
    symbols = objects(mapping(result["evidence"]).get("symbols", []))
    if not any(
        item.get("name") == EXPECTED_SYMBOL and item.get("file_path") == "helpers.py"
        for item in symbols
    ):
        raise ValueError("local query did not return the expected fixture function")


def _save(output: Path, report: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")


def _identity(state: StatePaths) -> dict[str, object] | None:
    path = state.runtime / "daemon-service.json"
    return (
        mapping(cast(object, json.loads(path.read_text()))) if path.exists() else None
    )


def _cache_summary(state: StatePaths) -> dict[str, object]:
    roles: dict[str, object] = {}
    observed_bytes = 0
    expected_bytes = 0
    complete = True
    for role, model in load_manifest().items():
        present = 0
        size = 0
        for artifact in model.artifacts:
            path = state.models / role.value / model.revision / artifact.path
            if path.is_file() and not path.is_symlink():
                size += path.stat().st_size
                present += int(path.stat().st_size == artifact.size_bytes)
        roles[role.value] = {
            "model": model.identity,
            "complete_files": present,
            "expected_files": len(model.artifacts),
            "observed_bytes": size,
        }
        observed_bytes += size
        expected_bytes += sum(artifact.size_bytes for artifact in model.artifacts)
        complete = complete and present == len(model.artifacts)
    return {
        "roles": roles,
        "observed_bytes": observed_bytes,
        "expected_bytes": expected_bytes,
        "complete": complete,
    }


def footprint_lines(contents: str) -> list[str]:
    """Retain only public memory totals from vmmap's broader process summary."""

    return [
        line.strip()
        for line in contents.splitlines()
        if line.strip().startswith(
            ("Physical footprint:", "Physical footprint (peak):")
        )
    ]


def _process_residency(pid: int) -> dict[str, object]:
    try:
        observation = subprocess.run(
            ["/usr/bin/vmmap", "-summary", str(pid)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "limitation": type(error).__name__}
    lines = footprint_lines(observation.stdout)
    if observation.returncode != 0 or not lines:
        return {
            "available": False,
            "limitation": "vmmap did not expose memory totals",
            "returncode": observation.returncode,
        }
    return {
        "available": True,
        "source": "vmmap -summary",
        "totals": lines,
        "scope": "Combined daemon after indexing, routing, and reranking; physical 16 GB hardware remains untested.",
    }


async def _wait_for_shutdown(state: StatePaths, generation: object) -> None:
    async with asyncio.timeout(SHUTDOWN_TIMEOUT_SECONDS):
        while observed := _identity(state):
            if observed.get("generation") != generation:
                raise ValueError(
                    "another daemon generation appeared during smoke shutdown"
                )
            await asyncio.sleep(0.05)


async def run(
    command: str, state: StatePaths, output: Path, report: dict[str, object]
) -> None:
    """Observe first query behavior without warming or retrying it in the driver."""

    report["source_before"] = source_hashes(state)
    previous = _identity(state)
    report["previous_daemon_identity"] = previous
    identity: dict[str, object] | None = None
    started = perf_counter()
    parameters = StdioServerParameters(
        command=command,
        args=["mcp"],
        env=child_environment(state),
        cwd=state.root,
    )
    try:
        async with stdio_client(parameters) as streams:
            async with ClientSession(
                *streams, read_timeout_seconds=MCP_TIMEOUT_SECONDS
            ) as session:
                await session.initialize()
                names = {tool.name for tool in (await session.list_tools()).tools}
                if names != set(MCP_TOOL_NAMES):
                    raise ValueError(
                        "fresh MCP session exposes an unexpected tool inventory"
                    )

                async def call(
                    name: str, arguments: dict[str, object]
                ) -> dict[str, object]:
                    result = await session.call_tool(name, arguments)
                    if result.is_error:
                        raise ValueError(
                            f"MCP {name} returned an error: {result.model_dump(mode='json')}"
                        )
                    return mapping(cast(object, result.structured_content))

                stats = await call("get_graph_stats", {"repo_path": str(state.fixture)})
                report["startup_seconds"] = perf_counter() - started
                identity = _identity(state)
                if identity is None or not isinstance(identity.get("pid"), int):
                    raise ValueError(
                        "fresh daemon did not publish its process identity"
                    )
                report["daemon_identity"] = identity
                if previous and previous.get("generation") == identity.get(
                    "generation"
                ):
                    raise ValueError(
                        "smoke reused an existing daemon instead of a fresh generation"
                    )
                report["initial_graph_stats"] = stats
                reused = stats.get("structural_search_ready") is True
                report["reused_index"] = reused
                if reused:
                    if stats.get("semantic_search_ready") is not True:
                        raise ValueError(
                            "cached fixture index lost semantic readiness after restart"
                        )
                else:
                    indexed_started = perf_counter()
                    acknowledgement = await call(
                        "ingest_project", {"repo_path": str(state.fixture)}
                    )
                    job_id = mapping(acknowledgement["job"])["id"]
                    report["ingestion_acknowledgement"] = acknowledgement
                    progress: list[dict[str, object]] = []
                    report["index_progress"] = progress
                    async with asyncio.timeout(INDEX_TIMEOUT_SECONDS):
                        while True:
                            stats = await call(
                                "get_graph_stats",
                                {
                                    "repo_path": str(state.fixture),
                                    "wait_job_id": job_id,
                                    "wait_timeout_seconds": 2,
                                },
                            )
                            job = mapping(mapping(stats["wait"])["job"])
                            progress.append(
                                {
                                    "status": job.get("status"),
                                    "phase": job.get("phase"),
                                    "current": job.get("current"),
                                    "total": job.get("total"),
                                    "message": job.get("message"),
                                    "elapsed_seconds": perf_counter() - indexed_started,
                                }
                            )
                            _save(output, report)
                            if job.get("status") == "completed":
                                break
                            if job.get("status") in {"failed", "cancelled"}:
                                raise ValueError(f"fixture ingestion terminated: {job}")
                    report["index_seconds"] = perf_counter() - indexed_started
                    stats = await call(
                        "get_graph_stats", {"repo_path": str(state.fixture)}
                    )
                report["ready_graph_stats"] = stats
                if stats.get("semantic_search_ready") is not True:
                    raise ValueError("fixture index is not semantically ready")
                _save(output, report)

                query_started = perf_counter()
                result = await call(
                    "query_code",
                    {
                        "repo_path": str(state.fixture),
                        "goal": GOAL,
                        "mode": "balanced",
                        "limit": 20,
                    },
                )
                report["first_query_seconds"] = perf_counter() - query_started
                report["first_query"] = result
                _save(output, report)
                validate_query(result, load_manifest()[ModelRole.DECISION].identity)

                client = SCSClient(
                    state.runtime / "scs.sock", timeout_seconds=MCP_TIMEOUT_SECONDS
                )
                health = await client.call("system.health")
                report["health"] = health
                if health.get("generation") != identity.get("generation"):
                    raise ValueError(
                        "MCP and SCSWire reached different daemon generations"
                    )
                search_started = perf_counter()
                search = await client.call(
                    "knowledge.search",
                    {
                        "repo_path": str(state.fixture),
                        "query": GOAL,
                        "search_mode": "balanced",
                        "limit": 10,
                        "result_detail": "compact",
                    },
                )
                report["search_seconds"] = perf_counter() - search_started
                report["search"] = search
                if (
                    search.get("semantic_available") is not True
                    or search.get("reranker_applied") is not True
                ):
                    raise ValueError(
                        "direct search did not apply local embeddings and reranking"
                    )
                if search.get("degraded_reason") or search.get("timed_out") is True:
                    raise ValueError("direct local search degraded or timed out")
                pid = identity["pid"]
                assert isinstance(pid, int)
                report["process_residency"] = await asyncio.to_thread(
                    _process_residency, pid
                )
                _save(output, report)
    finally:
        report["source_after"] = source_hashes(state)
        report["cache"] = _cache_summary(state)
        _save(output, report)
        if identity is not None:
            await _wait_for_shutdown(state, identity.get("generation"))
            report["daemon_shutdown_observed"] = True
        if report["source_before"] != report["source_after"]:
            raise ValueError("SCS changed the generated fixture source")
    if mapping(report["cache"]).get("complete") is not True:
        raise ValueError(
            "successful inference did not leave a complete pinned model cache"
        )


def _error_details(error: BaseException) -> list[str]:
    if isinstance(error, BaseExceptionGroup):
        group = cast(BaseExceptionGroup[BaseException], error)
        return [
            message for nested in group.exceptions for message in _error_details(nested)
        ]
    return [f"{type(error).__name__}: {str(error)[:1000]}"]


class Arguments(Protocol):
    command: str
    state_root: Path
    output: Path
    offline: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    arguments = cast(Arguments, cast(object, parser.parse_args(argv)))
    report: dict[str, object] = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "offline_requested": arguments.offline,
    }
    output: Path | None = None
    owned = False
    root = arguments.state_root.expanduser().resolve()
    try:
        output = validate_output(root, arguments.output)
        state = prepare_state(arguments.state_root)
        owned = True
        _save(output, report)
        report["network_control"] = network_control(arguments.offline)
        command = str(Path(arguments.command).expanduser().resolve())
        report["command"] = command
        report["state_root"] = str(state.root)
        asyncio.run(run(command, state, output, report))
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["errors"] = _error_details(error)
        if output is not None and (owned or not output.is_relative_to(root)):
            _save(output, report)
        print(
            json.dumps({"status": "failed", "errors": report["errors"]}),
            file=sys.stderr,
        )
        return 1
    assert output is not None
    _save(output, report)
    print(json.dumps({"status": "completed", "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
