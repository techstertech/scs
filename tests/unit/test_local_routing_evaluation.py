"""Local routing trials distinguish model decisions from graph-dependent policy."""

from __future__ import annotations

import json
import argparse
import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
import runpy
import subprocess
import sys
from typing import NoReturn, cast

import pytest

from scs.orchestration.decision import (
    DecisionProvider,
    DeterministicDecisionProvider,
    Playbook,
    RoutingDecision,
    RoutingRequest,
    eligible_playbooks,
)


SCRIPT = Path(__file__).parents[2] / "scripts" / "evaluate-local-routing.py"


def report_object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def read_report(path: Path) -> dict[str, object]:
    return report_object(cast(object, json.loads(path.read_text())))


def write_suite(path: Path) -> None:
    cases = [
        ("Find the parser implementation", "DISCOVER"),
        ("Explain how parsing works", "UNDERSTAND"),
        ("Find references to Parser without an explicit anchor", "REFERENCES"),
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "procedural-routing-cases",
                "cases": [
                    {
                        "goal": goal,
                        "expected_playbook": playbook,
                        "relevant": [{"identity": "parser.py", "grade": 3}],
                        "baseline": [{"method": "knowledge.search", "params": {}}],
                    }
                    for goal, playbook in cases
                ],
            }
        ),
        encoding="utf-8",
    )


def test_deterministic_trial_excludes_impossible_raw_labels(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    output = tmp_path / "report.json"
    write_suite(suite)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--backend",
            "deterministic",
            "--suite",
            str(suite),
            "--repeats",
            "2",
            "--warmups",
            "0",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    report = read_report(output)
    summary = report_object(report["summary"])
    assert report["status"] == "completed"
    assert summary["sample_count"] == 6
    assert summary["raw_scored_count"] == 4
    assert summary["raw_accuracy"] == 0.5
    assert summary["policy_preview_accuracy"] == 2 / 3
    assert summary["policy_requires_anchor_resolution_count"] == 2
    unresolved = cast(list[dict[str, object]], report["samples"])[-1]
    assert unresolved["raw_scored"] is False
    assert unresolved["policy_preview"] == "REFERENCES"
    assert unresolved["requires_anchor_resolution"] is True
    assert report_object(report["metadata"])["optional_model_modules_loaded"] == []


def test_failed_trial_replaces_an_earlier_success_report(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    output = tmp_path / "report.json"
    suite.write_text("{}", encoding="utf-8")
    output.write_text('{"status":"completed"}', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--suite", str(suite), "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode != 0
    assert read_report(output)["status"] == "failed"


def test_offline_external_trial_fails_before_contacting_a_service(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite.json"
    output = tmp_path / "report.json"
    write_suite(suite)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--backend",
            "external",
            "--offline",
            "--suite",
            str(suite),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode != 0
    report = read_report(output)
    assert report["status"] == "failed"
    assert "offline" in str(report["error"]).lower()


@pytest.mark.asyncio
async def test_malformed_laya_probabilities_cannot_be_scored_as_success() -> None:
    class InvalidAgent:
        def predict(
            self, state: dict[str, object], questions: dict[str, object]
        ) -> object:
            del state, questions
            request = RoutingRequest(goal="Find the parser")
            return {
                "answers": {
                    "playbook": {
                        "choice": Playbook.DISCOVER.value,
                        "confidence": 1.0,
                        "probabilities": {
                            item.value: 0.0 for item in eligible_playbooks(request)
                        },
                    }
                }
            }

    namespace = runpy.run_path(str(SCRIPT))
    provider_factory = cast(
        Callable[[object, str], DecisionProvider], namespace["LayaProvider"]
    )
    provider = provider_factory(InvalidAgent(), "fixture@revision")

    with pytest.raises(ValueError, match="sum to one"):
        await provider.classify(RoutingRequest(goal="Find the parser"))


def test_output_alias_cannot_overwrite_the_suite(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    alias = tmp_path / "report.json"
    write_suite(suite)
    original = suite.read_bytes()
    alias.symlink_to(suite)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--suite", str(suite), "--output", str(alias)],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode != 0
    assert suite.read_bytes() == original


def test_cancellation_invalidates_an_earlier_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "report.json"
    output.write_text('{"status":"completed"}', encoding="utf-8")
    namespace = runpy.run_path(str(SCRIPT))
    main = cast(Callable[[], int], namespace["main"])

    def interrupt(
        coroutine: Coroutine[object, object, dict[str, object]],
    ) -> NoReturn:
        coroutine.close()
        assert read_report(output)["status"] == "running"
        raise asyncio.CancelledError

    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(output)])
    monkeypatch.setattr(asyncio, "run", interrupt)

    with pytest.raises(asyncio.CancelledError):
        main()

    assert read_report(output)["status"] == "running"


@pytest.mark.asyncio
async def test_suite_changes_during_inference_reject_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = tmp_path / "suite.json"
    write_suite(suite)
    original_classify = DeterministicDecisionProvider.classify

    async def mutate_suite(
        provider: DeterministicDecisionProvider,
        request: RoutingRequest,
    ) -> RoutingDecision:
        suite.write_text(suite.read_text() + "\n", encoding="utf-8")
        return await original_classify(provider, request)

    namespace = runpy.run_path(str(SCRIPT))
    run = cast(
        Callable[[argparse.Namespace], Awaitable[dict[str, object]]], namespace["run"]
    )
    monkeypatch.setattr(DeterministicDecisionProvider, "classify", mutate_suite)
    arguments = argparse.Namespace(
        backend="deterministic",
        offline=False,
        suite=suite,
        repeats=1,
        warmups=0,
    )

    with pytest.raises(ValueError, match="suite changed"):
        await run(arguments)
