from __future__ import annotations

import importlib.util
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

import pytest


class Runner(Protocol):
    def run(self, arguments: argparse.Namespace) -> dict[str, object]: ...

    def validate_candidates(
        self,
        report: Mapping[str, object],
        documents: Mapping[str, object],
        *,
        corpus_sha256: str,
        suite_sha256: str,
        queries: Sequence[str],
        candidates: int,
    ) -> list[list[str]]: ...

    def rank_candidates(
        self,
        identities: Sequence[str],
        scores: Sequence[float],
    ) -> list[str]: ...


def load_runner() -> Runner:
    path = Path(__file__).parents[2] / "scripts/evaluate-local-reranking.py"
    spec = importlib.util.spec_from_file_location("local_reranking", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(Runner, cast(object, module))


def test_candidate_validation_rejects_mixed_snapshots_and_unknown_documents() -> None:
    runner = load_runner()
    documents = {str(index): {"id": str(index)} for index in range(4)}
    report = {
        "corpus_sha256": "corpus",
        "suite_sha256": "suite",
        "rankings": [{"query": "find code", "document_ids": list(documents)}],
    }

    assert runner.validate_candidates(
        report,
        documents,
        corpus_sha256="corpus",
        suite_sha256="suite",
        queries=["find code"],
        candidates=3,
    ) == [list(documents)[:3]]

    for key in ("corpus_sha256", "suite_sha256"):
        changed = dict(report, **{key: "different"})
        with pytest.raises(ValueError, match="snapshot"):
            runner.validate_candidates(
                changed,
                documents,
                corpus_sha256="corpus",
                suite_sha256="suite",
                queries=["find code"],
                candidates=3,
            )

    report["rankings"] = [{"query": "find code", "document_ids": ["missing"]}]
    with pytest.raises(ValueError, match="candidate"):
        runner.validate_candidates(
            report,
            documents,
            corpus_sha256="corpus",
            suite_sha256="suite",
            queries=["find code"],
            candidates=1,
        )


@pytest.mark.parametrize("scores", [[1.0], [math.nan, 0.0], [math.inf, 0.0]])
def test_ranking_rejects_incomplete_or_nonfinite_scores(scores: list[float]) -> None:
    runner = load_runner()

    with pytest.raises(ValueError, match="scores"):
        runner.rank_candidates(["first", "second"], scores)


def test_ranking_preserves_candidate_order_for_ties() -> None:
    runner = load_runner()

    assert runner.rank_candidates(["first", "second", "third"], [0.2, 0.8, 0.8]) == [
        "second",
        "third",
        "first",
    ]


def test_candidate_validation_rejects_duplicates_and_changed_query_order() -> None:
    runner = load_runner()
    documents = {str(index): {"id": str(index)} for index in range(2)}
    report = {
        "corpus_sha256": "corpus",
        "suite_sha256": "suite",
        "rankings": [{"query": "other", "document_ids": list(documents)}],
    }

    with pytest.raises(ValueError, match="query"):
        runner.validate_candidates(
            report,
            documents,
            corpus_sha256="corpus",
            suite_sha256="suite",
            queries=["find code"],
            candidates=2,
        )

    report["rankings"] = [{"query": "find code", "document_ids": ["0", "0"]}]
    with pytest.raises(ValueError, match="candidate"):
        runner.validate_candidates(
            report,
            documents,
            corpus_sha256="corpus",
            suite_sha256="suite",
            queries=["find code"],
            candidates=2,
        )


@pytest.mark.parametrize("change_suite", [False, True])
def test_reranking_metrics_share_cutoff_and_reject_suite_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change_suite: bool,
) -> None:
    """Keep an unjudged top result distinct from a judgment below the cutoff."""

    runner = load_runner()
    documents = [
        {
            "id": str(index),
            "qualified_name": f"code.item_{index}",
            "file_path": "code.py",
            "rerank_text": f"function item_{index}",
        }
        for index in range(4)
    ]
    corpus_path = tmp_path / "corpus.json"
    suite_path = tmp_path / "suite.json"
    candidate_path = tmp_path / "candidates.json"
    corpus_hash = hashlib.sha256(
        json.dumps(documents, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    corpus_path.write_text(
        json.dumps(
            {
                "documents": documents,
                "corpus_sha256": corpus_hash,
                "git_revision": "revision",
            }
        )
    )
    suite_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "generated",
                "cases": [
                    {
                        "query": "find code",
                        "relevant": [{"qualified_name": "code.item_3"}],
                    }
                ],
            }
        )
    )
    candidate_path.write_text(
        json.dumps(
            {
                "corpus_sha256": corpus_hash,
                "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
                "rankings": [
                    {"query": "find code", "document_ids": [str(i) for i in range(4)]}
                ],
            }
        )
    )
    model_dir = tmp_path / "model"
    weight = model_dir / "onnx/model_qint8_arm64.onnx"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"generated fake weights")

    class FakeBackend:
        truncated_pairs: int = 0

        def __init__(self, _profile: object, _path: str, _batch_size: int) -> None:
            pass

        def score(self, _query: str, inputs: list[str]) -> list[float]:
            if change_suite:
                suite_path.write_text(suite_path.read_text() + "\n")
            return [-float(i) for i in range(len(inputs))]

    from types import ModuleType

    hub = ModuleType("huggingface_hub")

    def download(_repository: str, **_options: object) -> str:
        return str(model_dir)

    monkeypatch.setattr(hub, "snapshot_download", download, raising=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(runner, "OnnxBackend", FakeBackend)
    arguments = argparse.Namespace(
        corpus=corpus_path,
        suite=suite_path,
        candidates=candidate_path,
        candidate_count=4,
        repeats=1,
        k=2,
        offline=False,
        profile="minilm-int8",
        cache_dir=tmp_path,
        batch_size=4,
    )
    if change_suite:
        with pytest.raises(ValueError, match="suite changed"):
            runner.run(arguments)
    else:
        report = runner.run(arguments)
        metrics = cast(dict[str, object], report["aggregate"])
        assert metrics["mean_recall_at_k"] == 0.0
        assert metrics["mean_reciprocal_rank"] == 0.0
