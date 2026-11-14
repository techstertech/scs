#!/usr/bin/env python3
"""Compare local rerankers over one fixed embedding candidate set.

This measures a retrieval component, not the complete SCS hybrid search route.
Corpus and suite hashes prevent comparisons across different source snapshots.
Optional inference dependencies belong in an isolated research environment.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
from typing import Protocol, cast

from scs.evaluation.search import (
    QueryMetrics,
    aggregate_metrics,
    evaluate_case,
    load_evaluation_suite,
)


@dataclass(frozen=True)
class Profile:
    repository: str
    revision: str
    onnx_file: str | None = None


PROFILES = {
    "minilm": Profile(
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    ),
    "minilm-int8": Profile(
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        "onnx/model_qint8_arm64.onnx",
    ),
    "qwen": Profile(
        "Qwen/Qwen3-Reranker-0.6B",
        "e61197ed45024b0ed8a2d74b80b4d909f1255473",
    ),
}
CODE_INSTRUCTION = (
    "Given a code search query, retrieve relevant code definitions and documentation."
)


def mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("expected a JSON object")
    return cast(Mapping[str, object], value)


def objects(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise ValueError("expected a JSON list")
    return [mapping(item) for item in cast(list[object], value)]


def validate_candidates(
    report: Mapping[str, object],
    documents: Mapping[str, object],
    *,
    corpus_sha256: str,
    suite_sha256: str,
    queries: Sequence[str],
    candidates: int,
) -> list[list[str]]:
    """Reject stale or incomplete retrieval before loading a model."""

    if (
        report.get("corpus_sha256") != corpus_sha256
        or report.get("suite_sha256") != suite_sha256
    ):
        raise ValueError("candidate snapshot differs from the corpus or suite")
    rankings = objects(report.get("rankings"))
    if len(rankings) != len(queries):
        raise ValueError("candidate query count differs from the suite")

    selected: list[list[str]] = []
    for query, ranking in zip(queries, rankings, strict=True):
        if ranking.get("query") != query:
            raise ValueError("candidate query order differs from the suite")
        raw = ranking.get("document_ids")
        if not isinstance(raw, list):
            raise ValueError("candidate IDs must be a list")
        values = cast(list[object], raw)
        if (
            len(values) < candidates
            or not all(isinstance(item, str) and item in documents for item in values)
            or len(set(cast(list[str], values))) != len(values)
        ):
            raise ValueError("candidate IDs are missing, unknown, or duplicated")
        selected.append(cast(list[str], values)[:candidates])
    return selected


def rank_candidates(identities: Sequence[str], scores: Sequence[float]) -> list[str]:
    """Keep retrieval order for ties and reject unusable model output."""

    if len(scores) != len(identities) or not all(
        math.isfinite(item) for item in scores
    ):
        raise ValueError("reranking scores must be complete and finite")
    return [
        identity
        for _, identity in sorted(
            zip(scores, identities, strict=True), key=lambda item: -item[0]
        )
    ]


class Backend(Protocol):
    truncated_pairs: int

    def score(self, query: str, documents: list[str]) -> list[float]: ...


class ArrayResult(Protocol):
    def reshape(self, dimension: int) -> ArrayResult: ...
    def tolist(self) -> object: ...


class MetalBackend(Protocol):
    def synchronize(self) -> None: ...
    def driver_allocated_memory(self) -> int: ...


class TorchModule(Protocol):
    bfloat16: object
    mps: MetalBackend


class CrossEncoderModel(Protocol):
    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> ArrayResult: ...


class SentenceTransformersModule(Protocol):
    def CrossEncoder(
        self,
        path: str,
        *,
        device: str,
        trust_remote_code: bool,
        local_files_only: bool,
        max_length: int,
        model_kwargs: dict[str, object],
        prompts: dict[str, str] | None,
    ) -> CrossEncoderModel: ...


class Encoding(Protocol):
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


class PairTokenizer(Protocol):
    def enable_padding(self, *, pad_id: int, pad_token: str) -> None: ...
    def no_truncation(self) -> None: ...
    def enable_truncation(self, *, max_length: int) -> None: ...
    def encode_batch(self, pairs: list[tuple[str, str]]) -> list[Encoding]: ...


class TokenizerLoader(Protocol):
    def from_file(self, path: str) -> PairTokenizer: ...


class TokenizersModule(Protocol):
    Tokenizer: TokenizerLoader


class NumpyModule(Protocol):
    def array(self, values: list[list[int]], *, dtype: str) -> object: ...


class SessionOptions(Protocol):
    intra_op_num_threads: int
    inter_op_num_threads: int


class OnnxSession(Protocol):
    def run(
        self, output_names: None, inputs: dict[str, object]
    ) -> list[ArrayResult]: ...


class OnnxModule(Protocol):
    def SessionOptions(self) -> SessionOptions: ...
    def InferenceSession(
        self,
        path: str,
        *,
        sess_options: SessionOptions,
        providers: list[str],
    ) -> OnnxSession: ...


class HubModule(Protocol):
    def snapshot_download(
        self,
        repository: str,
        *,
        revision: str,
        cache_dir: str,
        local_files_only: bool,
        token: bool,
        allow_patterns: list[str],
    ) -> str: ...


def score_values(value: ArrayResult) -> list[float]:
    raw = value.reshape(-1).tolist()
    if not isinstance(raw, list):
        raise ValueError("reranking scores must be a numeric list")
    values = cast(list[object], raw)
    if not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in values
    ):
        raise ValueError("reranking scores must be numeric")
    return [float(cast(int | float, item)) for item in values]


class CrossEncoderBackend:
    """Use the upstream model modules, including Qwen's yes/no logit head."""

    def __init__(self, profile: Profile, path: str, batch_size: int) -> None:
        torch = cast(TorchModule, cast(object, import_module("torch")))
        sentence_transformers = cast(
            SentenceTransformersModule,
            cast(object, import_module("sentence_transformers")),
        )

        self.truncated_pairs: int = 0
        self.batch_size: int = batch_size
        self.torch: TorchModule = torch
        self.model: CrossEncoderModel = sentence_transformers.CrossEncoder(
            path,
            device="mps",
            trust_remote_code=False,
            local_files_only=True,
            max_length=512 if profile.repository.startswith("cross-encoder/") else 2048,
            model_kwargs={"dtype": torch.bfloat16}
            if profile.repository.startswith("Qwen/")
            else {},
            prompts={"query": CODE_INSTRUCTION}
            if profile.repository.startswith("Qwen/")
            else None,
        )

    def score(self, query: str, documents: list[str]) -> list[float]:
        values = self.model.predict(
            [(query, document) for document in documents],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self.torch.mps.synchronize()
        return score_values(values)


class OnnxBackend:
    """Run the publisher's ARM64 INT8 artifact without remote Python code."""

    def __init__(self, profile: Profile, path: str, batch_size: int) -> None:
        np = cast(NumpyModule, cast(object, import_module("numpy")))
        ort = cast(OnnxModule, cast(object, import_module("onnxruntime")))
        tokenizers = cast(TokenizersModule, cast(object, import_module("tokenizers")))

        self.truncated_pairs: int = 0
        self.batch_size: int = batch_size
        self.np: NumpyModule = np
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session: OnnxSession = ort.InferenceSession(
            str(Path(path) / cast(str, profile.onnx_file)),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer: PairTokenizer = tokenizers.Tokenizer.from_file(
            str(Path(path) / "tokenizer.json")
        )
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    def score(self, query: str, documents: list[str]) -> list[float]:
        scores: list[float] = []
        for offset in range(0, len(documents), self.batch_size):
            pairs = [
                (query, item) for item in documents[offset : offset + self.batch_size]
            ]
            self.tokenizer.no_truncation()
            unbounded = self.tokenizer.encode_batch(pairs)
            self.truncated_pairs += sum(
                sum(item.attention_mask) > 512 for item in unbounded
            )
            self.tokenizer.enable_truncation(max_length=512)
            encoded = self.tokenizer.encode_batch(pairs)
            arrays = {
                "input_ids": self.np.array(
                    [item.ids for item in encoded], dtype="int64"
                ),
                "attention_mask": self.np.array(
                    [item.attention_mask for item in encoded], dtype="int64"
                ),
                "token_type_ids": self.np.array(
                    [item.type_ids for item in encoded], dtype="int64"
                ),
            }
            values = self.session.run(None, arrays)[0]
            scores.extend(score_values(values))
        return scores


def run(arguments: argparse.Namespace) -> dict[str, object]:
    corpus_path = cast(Path, arguments.corpus)
    suite_path = cast(Path, arguments.suite)
    candidate_path = cast(Path, arguments.candidates)
    corpus = mapping(cast(object, json.loads(corpus_path.read_text())))
    rows = objects(corpus["documents"])
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    corpus_hash = hashlib.sha256(canonical.encode()).hexdigest()
    if corpus_hash != corpus["corpus_sha256"]:
        raise ValueError("corpus content differs from its snapshot hash")
    documents = {cast(str, item["id"]): item for item in rows}
    suite = load_evaluation_suite(suite_path)
    suite_hash = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    retrieval = mapping(cast(object, json.loads(candidate_path.read_text())))
    candidate_count = cast(int, arguments.candidate_count)
    repeats = cast(int, arguments.repeats)
    k = cast(int, arguments.k)
    if not 1 <= k <= candidate_count <= 200 or not 1 <= repeats <= 20:
        raise ValueError("require 1 <= k <= candidates <= 200 and 1 <= repeats <= 20")
    rankings = validate_candidates(
        retrieval,
        documents,
        corpus_sha256=corpus_hash,
        suite_sha256=suite_hash,
        queries=[case.query for case in suite.cases],
        candidates=candidate_count,
    )

    offline = cast(bool, arguments.offline)
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    hub = cast(HubModule, cast(object, import_module("huggingface_hub")))

    profile = PROFILES[cast(str, arguments.profile)]
    path = hub.snapshot_download(
        profile.repository,
        revision=profile.revision,
        cache_dir=str(cast(Path, arguments.cache_dir)),
        local_files_only=offline,
        token=False,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.jinja",
            profile.onnx_file or "model.safetensors",
        ],
    )
    started = perf_counter()
    backend: Backend = (
        OnnxBackend(profile, path, cast(int, arguments.batch_size))
        if profile.onnx_file
        else CrossEncoderBackend(profile, path, cast(int, arguments.batch_size))
    )
    load_seconds = perf_counter() - started
    backend.score(
        suite.cases[0].query, [cast(str, documents[rankings[0][0]]["rerank_text"])]
    )
    backend.truncated_pairs = 0
    metrics: list[QueryMetrics] = []
    results: list[dict[str, object]] = []
    for case, identities in zip(suite.cases, rankings, strict=True):
        inputs = [cast(str, documents[item]["rerank_text"]) for item in identities]
        samples: list[float] = []
        scores: list[float] = []
        for _ in range(repeats):
            started = perf_counter()
            scores = backend.score(case.query, inputs)
            samples.append((perf_counter() - started) * 1000)
        ordered = rank_candidates(identities, scores)
        metrics.append(
            evaluate_case(
                case,
                [documents[item] for item in ordered[:k]],
                k=k,
                latency_ms=sum(samples) / len(samples),
                response_bytes=0,
                retrieval_mode="component_semantic_reranked",
                latency_samples_ms=samples,
            )
        )
        results.append(
            {
                "query": case.query,
                "document_ids": ordered,
                "latency_samples_ms": samples,
            }
        )
        print(
            f"reranked {len(results)}/{len(suite.cases)}", flush=True, file=sys.stderr
        )

    if hashlib.sha256(suite_path.read_bytes()).hexdigest() != suite_hash:
        raise ValueError("suite changed during evaluation")

    weight = Path(path) / (profile.onnx_file or "model.safetensors")
    with weight.open("rb") as stream:
        weight_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    versions: dict[str, str] = {}
    for package in (
        "sentence-transformers",
        "torch",
        "transformers",
        "onnxruntime",
        "tokenizers",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    torch_module = cast(TorchModule | None, cast(object, sys.modules.get("torch")))
    metal_bytes: int | None = None
    if torch_module is not None and hasattr(torch_module, "mps"):
        metal_bytes = int(torch_module.mps.driver_allocated_memory())
    return {
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_sha256": corpus_hash,
        "suite_sha256": suite_hash,
        "git_revision": corpus["git_revision"],
        "suite": suite.name,
        "model": profile.repository,
        "revision": profile.revision,
        "profile": cast(str, arguments.profile),
        "offline_requested": offline,
        "candidate_source": str(candidate_path.name),
        "candidate_count": candidate_count,
        "k": k,
        "repeats": repeats,
        "load_seconds": load_seconds,
        "aggregate": aggregate_metrics(metrics).model_dump(mode="json"),
        "queries": [item.model_dump(mode="json") for item in metrics],
        "rankings": results,
        "metadata": {
            "python": platform.python_version(),
            "versions": versions,
            "weight_bytes": weight.stat().st_size,
            "weight_sha256": weight_hash,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "mps_driver_allocated_bytes_at_end": metal_bytes,
            "truncated_pair_evaluations": backend.truncated_pairs
            if profile.onnx_file
            else None,
            "device": "cpu" if profile.onnx_file else "mps",
        },
        "limitations": [
            "Component-level exact-vector candidates; production lexical fusion and HTTP transport are excluded.",
            "RSS and MPS driver allocation overlap and must not be added.",
            "Offline mode uses local-only loading; operating-system network isolation is not applied.",
            "CrossEncoder truncation count is unavailable; context is capped at 512 for MiniLM and 2048 for Qwen.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--suite", type=Path, default=Path("evals/scs-local-search-v1.json")
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = cast(Path, arguments.output).resolve()
    inputs = {
        cast(Path, getattr(arguments, key)).resolve()
        for key in ("corpus", "suite", "candidates")
    }
    if output in inputs:
        parser.error("output must differ from all input files")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "running"}) + "\n")
    result: dict[str, object]
    try:
        result = run(arguments)
    except Exception as error:
        result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
