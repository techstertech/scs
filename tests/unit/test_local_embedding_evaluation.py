"""Keep local model comparisons reproducible without installing model runtimes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

import pytest

from scs.evaluation.search import EvaluationSuite


class _Profile(Protocol):
    revision: str


class _Corpus(Protocol):
    documents: Sequence[object]


class _DocumentFactory(Protocol):
    def model_validate(self, row: object) -> object: ...


class _Encoder(Protocol):
    def token_lengths(self, texts: list[str]) -> list[int]: ...
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class _Downloader(Protocol):
    def __call__(self, **arguments: object) -> str: ...


class Evaluator(Protocol):
    """The research script's test surface; optional ML libraries stay unloaded."""

    PROFILES: Mapping[str, _Profile]
    Document: _DocumentFactory

    def documents_hash(self, rows: object) -> str: ...
    def load_corpus(self, path: Path) -> _Corpus: ...
    def validate_judgments(
        self, documents: Sequence[object], suite: EvaluationSuite
    ) -> None: ...
    def format_text(
        self, profile: _Profile, text: str, *, query: bool, scs_prefixes: bool = False
    ) -> str: ...
    def cosine_rank(
        self, vectors: Sequence[Sequence[float]], query: Sequence[float]
    ) -> list[int]: ...
    def acquire_model(
        self,
        profile: _Profile,
        cache_dir: Path,
        *,
        offline: bool,
        download: _Downloader,
    ) -> Path: ...
    def main(self, argv: Sequence[str]) -> int: ...
    def evaluate_embeddings(
        self,
        encoder: _Encoder,
        documents: list[object],
        suite: EvaluationSuite,
        *,
        profile: _Profile,
        max_length: int,
        batch_size: int,
        repeats: int,
        k: int,
        scs_prefixes: bool,
    ) -> dict[str, object]: ...


def mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def sequence(value: object) -> list[object]:
    assert isinstance(value, list)
    return cast(list[object], value)


@pytest.fixture(scope="module")
def evaluator() -> Evaluator:
    path = Path(__file__).parents[2] / "scripts" / "evaluate-local-embeddings.py"
    specification = importlib.util.spec_from_file_location("local_embeddings", path)
    assert specification is not None and specification.loader is not None

    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return cast(Evaluator, cast(object, module))


def documents() -> list[dict[str, object]]:
    return [
        {
            "id": str(index),
            "qualified_name": f"package.function_{index}",
            "file_path": f"src/module_{index}.py",
            "kind": "function",
            "start_line": index + 1,
            "embedding_text": f"function {index}",
            "rerank_text": f"def function_{index}(): pass",
        }
        for index in range(3)
    ]


def test_corpus_hash_rejects_changed_text(evaluator: Evaluator, tmp_path: Path) -> None:
    rows = documents()
    payload = {
        "schema_version": 1,
        "git_revision": "a" * 40,
        "corpus_sha256": evaluator.documents_hash(rows),
        "documents": rows,
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload))

    corpus = evaluator.load_corpus(path)
    assert len(corpus.documents) == 3

    rows[0]["embedding_text"] = "changed after export"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        evaluator.load_corpus(path)


@pytest.mark.parametrize("include_path", [False, True])
def test_judgments_resolve_optional_source_paths(
    evaluator: Evaluator, include_path: bool
) -> None:
    from scs.evaluation.search import EvaluationCase, RelevanceJudgment

    rows = [evaluator.Document.model_validate(row) for row in documents()]
    judgment = RelevanceJudgment(
        qualified_name="package.function_0",
        file_path="src/module_0.py" if include_path else None,
    )
    suite = EvaluationSuite(
        schema_version=1,
        name="generated",
        cases=[EvaluationCase(query="find function", relevant=[judgment])],
    )

    evaluator.validate_judgments(rows, suite)


@pytest.mark.parametrize("mismatch", ["name", "path"])
def test_missing_judgments_fail_before_model_acquisition(
    evaluator: Evaluator,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    from scs.evaluation.search import EvaluationCase, RelevanceJudgment

    rows = documents()
    corpus = {
        "schema_version": 1,
        "git_revision": "a" * 40,
        "corpus_sha256": evaluator.documents_hash(rows),
        "documents": rows,
    }
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus))
    suite = EvaluationSuite(
        schema_version=1,
        name="stale",
        cases=[
            EvaluationCase(
                query="find function",
                relevant=[
                    RelevanceJudgment(
                        qualified_name="package.missing"
                        if mismatch == "name"
                        else "package.function_0",
                        file_path="src/renamed.py" if mismatch == "path" else None,
                    )
                ],
            )
        ],
    )
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(suite.model_dump_json())
    output = tmp_path / "report.json"

    def forbidden_download(*_arguments: object, **_keywords: object) -> Path:
        raise AssertionError("model acquisition preceded judgment validation")

    monkeypatch.setattr(evaluator, "acquire_model", forbidden_download)

    with pytest.raises(ValueError, match="suite judgments"):
        evaluator.main(
            [
                "--profile",
                "bge-small",
                "--corpus",
                str(corpus_path),
                "--suite",
                str(suite_path),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--output",
                str(output),
            ]
        )

    assert mapping(cast(object, json.loads(output.read_text())))["status"] == "failed"


def test_profiles_apply_native_and_current_scs_prompts(evaluator: Evaluator) -> None:
    qwen = evaluator.PROFILES["qwen3-0.6b"]
    nomic = evaluator.PROFILES["nomic-v1.5-int8"]
    jina = evaluator.PROFILES["jina-v2-code-int8"]

    assert evaluator.format_text(qwen, "find parser", query=True).startswith(
        "Instruct:"
    )
    assert evaluator.format_text(qwen, "source", query=False) == "source"
    assert evaluator.format_text(nomic, "goal", query=True) == "search_query: goal"
    assert (
        evaluator.format_text(nomic, "source", query=False) == "search_document: source"
    )
    assert evaluator.format_text(jina, "goal", query=True) == "goal"
    assert (
        evaluator.format_text(qwen, "goal", query=True, scs_prefixes=True)
        == "search_query: goal"
    )


def test_cosine_rank_uses_norms_and_stable_ties(evaluator: Evaluator) -> None:
    assert evaluator.cosine_rank([[4.0, 0.0], [0.0, 2.0], [2.0, 0.0]], [1.0, 0.0]) == [
        0,
        2,
        1,
    ]

    for vectors, query in [
        ([[0.0, 0.0]], [1.0, 0.0]),
        ([[1.0]], [1.0, 0.0]),
        ([[float("nan")]], [1.0]),
    ]:
        with pytest.raises(ValueError):
            evaluator.cosine_rank(vectors, query)


def test_download_is_pinned_and_offline_is_enforced(
    evaluator: Evaluator, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    def download(**arguments: object) -> str:
        observed.update(arguments)
        return str(tmp_path)

    profile = evaluator.PROFILES["nomic-v1.5-int8"]
    assert (
        evaluator.acquire_model(profile, tmp_path, offline=True, download=download)
        == tmp_path
    )
    assert observed["revision"] == profile.revision
    assert observed["local_files_only"] is True
    assert "onnx/model_int8.onnx" in cast(list[str], observed["allow_patterns"])


def test_failed_evaluation_invalidates_previous_success(
    evaluator: Evaluator, tmp_path: Path
) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"documents": [], "corpus_sha256": "wrong"}))
    output = tmp_path / "report.json"
    output.write_text(json.dumps({"status": "completed", "aggregate": {}}))

    with pytest.raises(ValueError, match="hash"):
        evaluator.main(
            [
                "--profile",
                "bge-small",
                "--corpus",
                str(corpus_path),
                "--suite",
                str(tmp_path / "unused-suite.json"),
                "--cache-dir",
                str(tmp_path / "empty-cache"),
                "--offline",
                "--output",
                str(output),
            ]
        )

    report = mapping(cast(object, json.loads(output.read_text())))
    assert report["status"] == "failed"
    assert "aggregate" not in report


def test_component_evaluation_reuses_search_metrics(evaluator: Evaluator) -> None:
    from scs.evaluation.search import EvaluationCase, EvaluationSuite, RelevanceJudgment

    class Encoder:
        def token_lengths(self, texts: list[str]) -> list[int]:
            return [len(text) for text in texts]

        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] if text.endswith("0") else [0.0, 1.0] for text in texts]

    suite = EvaluationSuite(
        schema_version=1,
        name="generated",
        cases=[
            EvaluationCase(
                query="0",
                relevant=[RelevanceJudgment(qualified_name="package.function_0")],
            )
        ],
    )
    rows = [evaluator.Document.model_validate(row) for row in documents()]
    profile = evaluator.PROFILES["jina-v2-code-int8"]

    report = evaluator.evaluate_embeddings(
        Encoder(),
        rows,
        suite,
        profile=profile,
        max_length=5,
        batch_size=2,
        repeats=3,
        k=1,
        scs_prefixes=False,
    )

    query = mapping(sequence(report["queries"])[0])
    assert mapping(report["aggregate"])["mean_recall_at_k"] == 1.0
    assert mapping(sequence(query["ranked_documents"])[0])["id"] == "0"
    assert len(sequence(query["latency_samples_ms"])) == 3
    assert mapping(report["documents"])["truncated_count"] == 3
    assert query["truncated"] is False


@pytest.mark.parametrize("target", ["corpus", "suite", "corpus-symlink"])
def test_report_cannot_overwrite_evaluation_inputs(
    evaluator: Evaluator, tmp_path: Path, target: str
) -> None:
    corpus_path = tmp_path / "corpus.json"
    suite_path = tmp_path / "suite.json"
    corpus_path.write_text("preserve corpus bytes")
    suite_path.write_text("preserve suite bytes")
    output = suite_path if target == "suite" else corpus_path
    if target == "corpus-symlink":
        output = tmp_path / "output.json"
        output.symlink_to(corpus_path)

    with pytest.raises(ValueError, match="output must differ"):
        evaluator.main(
            [
                "--profile",
                "bge-small",
                "--corpus",
                str(corpus_path),
                "--suite",
                str(suite_path),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--output",
                str(output),
            ]
        )

    assert corpus_path.read_text() == "preserve corpus bytes"
    assert suite_path.read_text() == "preserve suite bytes"
