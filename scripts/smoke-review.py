"""Manual release scenarios through fresh stdio MCP and the live SCS daemon."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def objects(value: object) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], value) if isinstance(value, list) else []


def mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict), value
    return cast(dict[str, object], value)


async def run(command: str, output: Path) -> None:
    checks: list[dict[str, object]] = []
    sources = {
        "core.py": "# [contract](specs/old.md)\ndef review_changed_value(value: int) -> int:\n    return value + 1\n\ndef review_unchanged_value(value: int) -> int:\n    return value - 1\n\ndef review_isolated_value(value: int) -> int:\n    return value * 2\n",
        "consumer.py": "from core import review_changed_value\n\ndef review_consume(value: int) -> int:\n    return review_changed_value(value)\n",
        "unrelated.py": "from core import review_unchanged_value\n",
        "tests/test_consumer.py": "import consumer\n\ndef test_review_consume():\n    assert consumer.review_consume(1) == 2\n",
        "duplicate_one.py": "def review_duplicate():\n    return 1\n",
        "duplicate_two.py": "def review_duplicate():\n    return 2\n",
        "specs/current.md": "---\nstatus: approved\n---\n# Preserve input\nThe input remains unchanged. [implementation](../core.py)\n",
        "specs/old.md": "---\nstatus: shipped\nsuperseded-by: specs/current.md\n---\n# Historical requirement\n",
    }
    with tempfile.TemporaryDirectory(prefix="scs-review-manual-") as temp:
        root = Path(temp)
        repo = root / "repo"
        repo.mkdir()
        for relative, text in sources.items():
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        (root / "outside.md").write_text("Manual containment sentinel.\n")
        (repo / "escape.md").symlink_to(root / "outside.md")
        unindexed = root / "unindexed"
        unindexed.mkdir()
        fingerprints = {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in sources
        }
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        parameters = StdioServerParameters(
            command=command, args=["mcp"], env=environment, cwd=root
        )
        async with stdio_client(parameters) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                names = {tool.name for tool in tools}
                assert names == {
                    "query_code",
                    "get_graph_stats",
                    "ingest_project",
                    "ingest_files",
                    "delete_repository",
                }
                schema = cast(
                    dict[str, object],
                    next(
                        tool.input_schema for tool in tools if tool.name == "query_code"
                    ),
                )
                assert {"changed_ranges", "contract_paths"} <= mapping(
                    schema["properties"]
                ).keys()
                checks.append(
                    {"check": "fresh MCP inventory and schema", "passed": True}
                )

                async def call(name: str, args: dict[str, object]) -> dict[str, object]:
                    result = await session.call_tool(name, args)
                    assert not result.is_error, result.model_dump(mode="json")
                    return mapping(cast(object, result.structured_content))

                async def wait(job: dict[str, object], path: Path) -> None:
                    for _ in range(120):
                        state = await call(
                            "get_graph_stats",
                            {
                                "repo_path": str(path),
                                "wait_job_id": job["id"],
                                "wait_timeout_seconds": 2,
                            },
                        )
                        observation = mapping(state.get("wait"))
                        current = mapping(observation.get("job"))
                        if current.get("status") == "completed":
                            return
                        assert current.get("status") not in {"failed", "cancelled"}, (
                            current
                        )
                    raise AssertionError("index job did not finish")

                indexed = False
                try:
                    ingestion = await call("ingest_project", {"repo_path": str(repo)})
                    indexed = True
                    await wait(mapping(ingestion["job"]), repo)
                    stats = await call("get_graph_stats", {"repo_path": str(repo)})
                    assert stats["structural_search_ready"] is True
                    checks.append(
                        {
                            "check": "native fixture ingestion",
                            "passed": True,
                            "nodes": stats["total_nodes"],
                        }
                    )

                    async def query(**kwargs: object) -> dict[str, object]:
                        return await call(
                            "query_code",
                            {
                                "goal": "Review the helper",
                                "repo_path": str(repo),
                                "mode": "balanced",
                                **kwargs,
                            },
                        )

                    changed = await query(
                        changed_ranges=[
                            {"file_path": "core.py", "start_line": 2, "end_line": 2}
                        ]
                    )
                    evidence = mapping(changed["evidence"])
                    targets = objects(evidence.get("test_targets"))
                    assert any(
                        item["file_path"] == "tests/test_consumer.py"
                        for item in targets
                    ), changed
                    assert not any(
                        item.get("file_path") == "unrelated.py"
                        for item in objects(evidence.get("symbols"))
                        + objects(evidence.get("files"))
                    ), changed
                    assert all(
                        item.get("evidence_kind") == "static_dependency"
                        for item in targets
                    ), changed
                    assert any(
                        len(objects(item.get("dependency_path"))) == 2
                        for target in targets
                        for item in objects(target.get("evidence"))
                    ), changed
                    contracts = objects(evidence.get("contracts"))
                    assert any(
                        item["file_path"] == "specs/current.md"
                        and item["active"] is True
                        for item in contracts
                    ), changed
                    assert any(
                        item["file_path"] == "specs/old.md" and item["active"] is False
                        for item in contracts
                    ), changed
                    assert (
                        mapping(mapping(changed["investigation"])["references"])[
                            "status"
                        ]
                        == "not_assessed"
                    )
                    assert changed["complete"] is True, changed
                    checks.append(
                        {
                            "check": "changed-symbol paths, tests, contract authority, coverage",
                            "passed": True,
                            "test_targets": [item["file_path"] for item in targets],
                        }
                    )

                    references = await query(
                        goal="Find references to review_changed_value"
                    )
                    routing = mapping(references["routing"])
                    assert routing["playbook"] == "REFERENCES", references
                    assert (
                        mapping(routing["anchor_resolution"])["status"] == "resolved"
                    ), references
                    assert objects(mapping(references["evidence"]).get("references")), (
                        references
                    )
                    checks.append(
                        {"check": "grounded reference lookup", "passed": True}
                    )

                    ambiguous = await query(goal="Find references to review_duplicate")
                    resolution = mapping(
                        mapping(ambiguous["routing"])["anchor_resolution"]
                    )
                    assert resolution["status"] == "ambiguous", ambiguous
                    assert len(objects(resolution["candidates"])) == 2, ambiguous
                    assert not any(
                        stage["stage"] == "references"
                        for stage in objects(ambiguous["trace"])
                    ), ambiguous
                    checks.append(
                        {
                            "check": "duplicate symbol ambiguity stops traversal",
                            "passed": True,
                        }
                    )

                    partial = await query(
                        changed_ranges=[
                            {"file_path": "core.py", "start_line": 2, "end_line": 2}
                        ],
                        limit=1,
                    )
                    assert partial["truncated"] is True, partial
                    assert (
                        mapping(mapping(partial["investigation"])["dependents"])[
                            "status"
                        ]
                        == "partial"
                    ), partial
                    checks.append(
                        {
                            "check": "bounded output reports partial coverage",
                            "passed": True,
                        }
                    )

                    empty = await query(
                        changed_ranges=[
                            {"file_path": "core.py", "start_line": 8, "end_line": 8}
                        ]
                    )
                    assert (
                        mapping(mapping(empty["investigation"])["tests"])["status"]
                        == "empty"
                    ), empty
                    unavailable = await call(
                        "query_code",
                        {
                            "goal": "Review missing index",
                            "repo_path": str(unindexed),
                            "changed_ranges": [
                                {
                                    "file_path": "missing.py",
                                    "start_line": 0,
                                    "end_line": 0,
                                }
                            ],
                        },
                    )
                    assert (
                        mapping(mapping(unavailable["investigation"])["tests"])[
                            "status"
                        ]
                        == "unavailable"
                    ), unavailable
                    checks.append(
                        {"check": "empty and unavailable are distinct", "passed": True}
                    )

                    missing = await query(
                        changed_ranges=[
                            {"file_path": "core.py", "start_line": 999, "end_line": 999}
                        ]
                    )
                    assert missing["complete"] is False, missing
                    assert any(
                        item.get("kind") == "unresolved_range"
                        for item in objects(mapping(missing["evidence"]).get("files"))
                    ), missing
                    for invalid in [
                        {
                            "changed_ranges": [
                                {"file_path": "core.py", "start_line": 2, "end_line": 1}
                            ]
                        },
                        {
                            "changed_ranges": [
                                {
                                    "file_path": "../outside.py",
                                    "start_line": 0,
                                    "end_line": 0,
                                }
                            ]
                        },
                        {"contract_paths": ["escape.md"]},
                    ]:
                        result = await session.call_tool(
                            "query_code",
                            {
                                "goal": "Review invalid input",
                                "repo_path": str(repo),
                                **invalid,
                            },
                        )
                        assert result.is_error, result
                    checks.append(
                        {"check": "range validation and containment", "passed": True}
                    )
                    assert fingerprints == {
                        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                        for name in sources
                    }
                    checks.append({"check": "source bytes preserved", "passed": True})
                finally:
                    if indexed:
                        deleted = await call(
                            "delete_repository", {"repo_path": str(repo)}
                        )
                        if deleted.get("job"):
                            await wait(mapping(deleted["job"]), repo)
                        checks.append(
                            {"check": "fixture index removed", "passed": True}
                        )
    output.write_text(json.dumps({"passed": True, "checks": checks}, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = cast(dict[str, object], vars(parser.parse_args()))
    command, output = arguments["command"], arguments["output"]
    assert isinstance(command, str) and isinstance(output, Path)
    command = str(Path(command).expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    try:
        asyncio.run(run(command, output))
    except Exception as error:
        output.write_text(
            json.dumps({"passed": False, "error": type(error).__name__}) + "\n"
        )
        raise
