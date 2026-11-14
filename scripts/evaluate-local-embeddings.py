#!/usr/bin/env python3
"""Compare embedding components on one immutable SCS corpus, without a daemon.

This research command measures exact cosine retrieval. It does not reproduce
SCS's lexical fusion, graph expansion, or public-query latency. Model runtimes
are optional and imported only after the corpus and judgments pass validation.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import resource
from statistics import mean
import sys
from time import perf_counter
from typing import ClassVar, Final, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from scs.evaluation.search import (
    EvaluationSuite,
    QueryMetrics,
    aggregate_metrics,
    evaluate_case,
    load_evaluation_suite,
)

CODE_INSTRUCTION: Final[str] = (
    "Given a question about a code repository, retrieve the relevant code and documentation."
)
CANDIDATE_COUNT: Final[int] = 40


@dataclass(frozen=True)
class Profile:
    """Pin weights, preprocessing, and an independently documented context limit."""

    repository: str
    revision: str
    backend: Literal["sentence-transformers", "onnx"]
    dimension: int
    context: int
    query_prefix: str
    document_prefix: str = ""
    pooling: str = "cls"
    weight_file: str = "model.safetensors"
    dtype: Literal["float32", "bfloat16", "int8"] = "float32"


PROFILES: Final[dict[str, Profile]] = {
    "bge-small": Profile(
        "BAAI/bge-small-en-v1.5",
        "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "sentence-transformers",
        384,
        512,
        "Represent this sentence for searching relevant passages: ",
    ),
    "qwen3-0.6b": Profile(
        "Qwen/Qwen3-Embedding-0.6B",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "sentence-transformers",
        1024,
        32768,
        f"Instruct: {CODE_INSTRUCTION}\nQuery:",
        pooling="last-token",
        dtype="bfloat16",
    ),
    "nomic-v1.5-int8": Profile(
        "nomic-ai/nomic-embed-text-v1.5",
        "e9b6763023c676ca8431644204f50c2b100d9aab",
        "onnx",
        768,
        8192,
        "search_query: ",
        "search_document: ",
        "masked-mean, layer-normalization",
        "onnx/model_int8.onnx",
        dtype="int8",
    ),
    "jina-v2-code-int8": Profile(
        "jinaai/jina-embeddings-v2-base-code",
        "516f4baf13dec4ddddda8631e019b5737c8bc250",
        "onnx",
        768,
        8192,
        "",
        pooling="masked-mean",
        weight_file="onnx/model_quantized.onnx",
        dtype="int8",
    ),
}


class Document(BaseModel):
    """Read the shared export while retaining identity for existing judgments."""

    model_config: ClassVar[ConfigDict] = ConfigDict(strict=True, frozen=True)

    id: str
    qualified_name: str
    file_path: str
    kind: str
    start_line: int = Field(ge=0)
    embedding_text: str
    rerank_text: str


class Corpus(BaseModel):
    """Bind a comparison to a committed source snapshot and canonical bytes."""

    model_config: ClassVar[ConfigDict] = ConfigDict(strict=True, frozen=True)

    schema_version: Literal[1]
    git_revision: str
    corpus_sha256: str
    documents: list[Document] = Field(min_length=1)


def documents_hash(documents: object) -> str:
    canonical = json.dumps(documents, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_corpus(path: Path) -> Corpus:
    raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    if documents_hash(raw.get("documents")) != raw.get("corpus_sha256"):
        raise ValueError("corpus hash differs from the exported documents")
    corpus = Corpus.model_validate(raw)
    if len({document.id for document in corpus.documents}) != len(corpus.documents):
        raise ValueError("corpus document identities must be unique")
    return corpus


def validate_judgments(documents: Sequence[Document], suite: EvaluationSuite) -> None:
    """Reject stale judgments before acquiring weights or loading a runtime."""

    identities = {
        (document.qualified_name, document.file_path) for document in documents
    }
    names = {name for name, _ in identities}
    missing: list[tuple[str, str | None]] = []
    for case in suite.cases:
        for judgment in case.relevant:
            matches = (
                judgment.qualified_name in names
                if judgment.file_path is None
                else (judgment.qualified_name, judgment.file_path) in identities
            )
            if not matches:
                missing.append((judgment.qualified_name, judgment.file_path))
    if missing:
        raise ValueError(f"suite judgments do not match corpus identities: {missing}")


def format_text(
    profile: Profile, text: str, *, query: bool, scs_prefixes: bool = False
) -> str:
    if scs_prefixes:
        return f"{'search_query' if query else 'search_document'}: {text}"
    return f"{profile.query_prefix if query else profile.document_prefix}{text}"


def _normalized(vector: Sequence[float]) -> list[float]:
    if not vector or not all(math.isfinite(component) for component in vector):
        raise ValueError("embedding must contain finite components")
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0:
        raise ValueError("embedding has zero norm")
    return [component / norm for component in vector]


def cosine_rank(
    vectors: Sequence[Sequence[float]], query: Sequence[float]
) -> list[int]:
    """Rank exact cosine scores; corpus order resolves identical scores."""

    normalized_query = _normalized(query)
    scores: list[float] = []
    for vector in vectors:
        if len(vector) != len(query):
            raise ValueError("query and document embedding dimensions differ")
        scores.append(
            sum(
                a * b
                for a, b in zip(_normalized(vector), normalized_query, strict=True)
            )
        )
    return sorted(range(len(scores)), key=lambda index: -scores[index])


class SnapshotDownloader(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        revision: str,
        cache_dir: str,
        local_files_only: bool,
        allow_patterns: list[str],
    ) -> str: ...


class _HubModule(Protocol):
    snapshot_download: SnapshotDownloader


def acquire_model(
    profile: Profile,
    cache_dir: Path,
    *,
    offline: bool,
    download: SnapshotDownloader | None = None,
) -> Path:
    """Restrict downloads to a pinned model, tokenizer, and pooling metadata."""

    if download is None:
        module = cast(
            _HubModule, cast(object, importlib.import_module("huggingface_hub"))
        )
        download = module.snapshot_download
    return Path(
        download(
            repo_id=profile.repository,
            revision=profile.revision,
            cache_dir=str(cache_dir),
            local_files_only=offline,
            allow_patterns=[
                profile.weight_file,
                "*.json",
                "*.txt",
                "*.model",
                "*.jinja",
            ],
        )
    )


class Encoder(Protocol):
    def token_lengths(self, texts: list[str]) -> list[int]: ...
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class _ArrayResult(Protocol):
    def tolist(self) -> object: ...


class _HFTokenizer(Protocol):
    def encode(self, text: str, *, truncation: bool) -> list[int]: ...


class _SentenceModel(Protocol):
    max_seq_length: int
    tokenizer: _HFTokenizer

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> _ArrayResult: ...


class _SentenceConstructor(Protocol):
    def __call__(
        self,
        path: str,
        *,
        device: str,
        trust_remote_code: bool,
        local_files_only: bool,
        model_kwargs: dict[str, object],
    ) -> _SentenceModel: ...


class _SentenceModule(Protocol):
    SentenceTransformer: _SentenceConstructor


class _MPS(Protocol):
    def is_available(self) -> bool: ...
    def current_allocated_memory(self) -> int: ...
    def driver_allocated_memory(self) -> int: ...


class _Backends(Protocol):
    mps: _MPS


class _TorchModule(Protocol):
    bfloat16: object
    float32: object
    mps: _MPS
    backends: _Backends


class _Encoding(Protocol):
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


class _Tokenizer(Protocol):
    def no_truncation(self) -> None: ...
    def no_padding(self) -> None: ...
    def enable_truncation(self, *, max_length: int) -> None: ...
    def enable_padding(self, *, pad_id: int, pad_token: str) -> None: ...
    def encode_batch(self, texts: list[str]) -> list[_Encoding]: ...


class _TokenizerFactory(Protocol):
    def from_file(self, path: str) -> _Tokenizer: ...


class _TokenizersModule(Protocol):
    Tokenizer: _TokenizerFactory


class _SessionOptions(Protocol):
    intra_op_num_threads: int


class _SessionInput(Protocol):
    name: str


class _ONNXSession(Protocol):
    def get_inputs(self) -> list[_SessionInput]: ...
    def run(self, output_names: None, inputs: dict[str, object]) -> list[object]: ...


class _SessionFactory(Protocol):
    def __call__(
        self, path: str, *, sess_options: _SessionOptions, providers: list[str]
    ) -> _ONNXSession: ...


class _ONNXModule(Protocol):
    SessionOptions: Callable[[], _SessionOptions]
    InferenceSession: _SessionFactory


class SentenceEncoder:
    """Load local-only Sentence Transformers without executable model code."""

    def __init__(self, path: Path, *, max_length: int, device: str, dtype: str) -> None:
        module = cast(
            _SentenceModule,
            cast(object, importlib.import_module("sentence_transformers")),
        )
        torch = cast(_TorchModule, cast(object, importlib.import_module("torch")))
        precision = torch.bfloat16 if dtype == "bfloat16" else torch.float32

        self.model: _SentenceModel = module.SentenceTransformer(
            str(path),
            device=device,
            trust_remote_code=False,
            local_files_only=True,
            model_kwargs={"torch_dtype": precision},
        )
        self.model.max_seq_length = max_length

    def token_lengths(self, texts: list[str]) -> list[int]:
        return [
            len(self.model.tokenizer.encode(text, truncation=False)) for text in texts
        ]

    def encode(self, texts: list[str]) -> list[list[float]]:
        result = self.model.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return cast(list[list[float]], result.tolist())


class ONNXEncoder:
    """Run official INT8 exports on CPU without importing remote Python code."""

    def __init__(self, path: Path, profile: Profile, *, max_length: int) -> None:
        ort = cast(_ONNXModule, cast(object, importlib.import_module("onnxruntime")))
        tokenizers = cast(
            _TokenizersModule, cast(object, importlib.import_module("tokenizers"))
        )

        self.profile: Profile = profile
        self.tokenizer: _Tokenizer = tokenizers.Tokenizer.from_file(
            str(path / "tokenizer.json")
        )
        self.counter: _Tokenizer = tokenizers.Tokenizer.from_file(
            str(path / "tokenizer.json")
        )
        self.counter.no_truncation()
        self.counter.no_padding()
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session: _ONNXSession = ort.InferenceSession(
            str(path / profile.weight_file),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def token_lengths(self, texts: list[str]) -> list[int]:
        return [len(encoding.ids) for encoding in self.counter.encode_batch(texts)]

    def encode(self, texts: list[str]) -> list[list[float]]:
        import numpy as np
        from numpy.typing import NDArray

        encoded = self.tokenizer.encode_batch(texts)
        arrays = {
            "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
            "attention_mask": np.asarray(
                [item.attention_mask for item in encoded], dtype=np.int64
            ),
            "token_type_ids": np.asarray(
                [item.type_ids for item in encoded], dtype=np.int64
            ),
        }
        inputs: dict[str, object] = {
            entry.name: arrays[entry.name] for entry in self.session.get_inputs()
        }
        outputs = self.session.run(None, inputs)
        states = cast("NDArray[np.float32]", outputs[0])
        mask = arrays["attention_mask"][..., None].astype(np.float32)
        pooled = (states * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1)
        if self.profile.repository.startswith("nomic-ai/"):
            pooled = (pooled - pooled.mean(axis=1, keepdims=True)) / np.sqrt(
                pooled.var(axis=1, keepdims=True) + 1e-5
            )
        pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
        return cast(list[list[float]], pooled.tolist())


def evaluate_embeddings(
    encoder: Encoder,
    documents: list[Document],
    suite: EvaluationSuite,
    *,
    profile: Profile,
    max_length: int,
    batch_size: int,
    repeats: int,
    k: int,
    scs_prefixes: bool,
) -> dict[str, object]:
    """Measure document indexing once and individual query inference repeatedly."""

    texts = [
        format_text(profile, row.embedding_text, query=False, scs_prefixes=scs_prefixes)
        for row in documents
    ]
    lengths = encoder.token_lengths(texts)
    vectors: list[list[float]] = []
    started = perf_counter()
    for offset in range(0, len(texts), batch_size):
        vectors.extend(encoder.encode(texts[offset : offset + batch_size]))
    document_seconds = perf_counter() - started
    if len(vectors) != len(documents):
        raise ValueError("encoder changed the document count")

    metrics: list[QueryMetrics] = []
    query_reports: list[dict[str, object]] = []
    rankings: list[dict[str, object]] = []
    for case in suite.cases:
        text = format_text(profile, case.query, query=True, scs_prefixes=scs_prefixes)
        token_count = encoder.token_lengths([text])[0]
        samples: list[float] = []
        query_vector: list[float] = []
        for _ in range(repeats):
            started = perf_counter()
            query_vector = encoder.encode([text])[0]
            samples.append((perf_counter() - started) * 1000)
        started = perf_counter()
        order = cosine_rank(vectors, query_vector)
        ranking_ms = (perf_counter() - started) * 1000
        results = [
            documents[index].model_dump(exclude={"embedding_text", "rerank_text"})
            for index in order[: max(k, CANDIDATE_COUNT)]
        ]
        metric = evaluate_case(
            case,
            results[:k],
            k=k,
            latency_ms=mean(samples),
            latency_samples_ms=samples,
            response_bytes=0,
            retrieval_mode="component_exact_cosine",
        )
        metrics.append(metric)
        query_reports.append(
            {
                **metric.model_dump(),
                "token_count": token_count,
                "truncated": token_count > max_length,
                "ranking_ms": ranking_ms,
                "ranked_documents": results,
            }
        )
        rankings.append(
            {
                "query": case.query,
                "document_ids": [
                    documents[index].id for index in order[: max(k, CANDIDATE_COUNT)]
                ],
            }
        )

    return {
        "documents": {
            "count": len(documents),
            "encoding_seconds": document_seconds,
            "documents_per_second": len(documents) / document_seconds,
            "truncated_count": sum(length > max_length for length in lengths),
            "max_tokens_before_truncation": max(lengths),
        },
        "queries": query_reports,
        "rankings": rankings,
        "aggregate": aggregate_metrics(metrics).model_dump(),
    }


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def memory_observations() -> dict[str, int | str]:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result: dict[str, int | str] = {
        "process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
        "scope": "process RSS; Metal current allocations are separate, not a peak",
    }
    if "torch" in sys.modules:
        torch = cast(_TorchModule, cast(object, sys.modules["torch"]))

        if torch.backends.mps.is_available():
            result["metal_current_allocated_bytes"] = (
                torch.mps.current_allocated_memory()
            )
            result["metal_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--scs-prefixes", action="store_true")
    return parser


class _Arguments(Protocol):
    profile: str
    corpus: Path
    suite: Path
    cache_dir: Path
    output: Path
    max_length: int | None
    batch_size: int
    repeats: int
    k: int
    device: str
    offline: bool
    scs_prefixes: bool


def main(argv: Sequence[str] | None = None) -> int:
    arguments = cast(_Arguments, cast(object, build_parser().parse_args(argv)))
    output = arguments.output
    if output.resolve() in {arguments.corpus.resolve(), arguments.suite.resolve()}:
        raise ValueError("output must differ from the corpus and suite input files")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "running"}) + "\n", encoding="utf-8")
    try:
        profile = PROFILES[arguments.profile]
        requested_length = (
            profile.context if arguments.max_length is None else arguments.max_length
        )
        max_length = min(requested_length, profile.context)
        if min(max_length, arguments.batch_size, arguments.repeats, arguments.k) < 1:
            raise ValueError("length, batch size, repeats, and cutoff must be positive")
        corpus = load_corpus(arguments.corpus)
        suite_hash = hashlib.sha256(arguments.suite.read_bytes()).hexdigest()
        suite = load_evaluation_suite(arguments.suite)
        validate_judgments(corpus.documents, suite)
        started = perf_counter()
        path = acquire_model(profile, arguments.cache_dir, offline=arguments.offline)
        acquisition_seconds = perf_counter() - started
        started = perf_counter()
        encoder: Encoder = (
            SentenceEncoder(
                path,
                max_length=max_length,
                device=arguments.device,
                dtype=profile.dtype,
            )
            if profile.backend == "sentence-transformers"
            else ONNXEncoder(path, profile, max_length=max_length)
        )
        load_seconds = perf_counter() - started
        observations = evaluate_embeddings(
            encoder,
            corpus.documents,
            suite,
            profile=profile,
            max_length=max_length,
            batch_size=arguments.batch_size,
            repeats=arguments.repeats,
            k=arguments.k,
            scs_prefixes=arguments.scs_prefixes,
        )
        if hashlib.sha256(arguments.suite.read_bytes()).hexdigest() != suite_hash:
            raise ValueError("suite changed during evaluation")
        artifacts = {profile.weight_file: file_identity(path / profile.weight_file)}
        packages: dict[str, str] = {}
        for package in (
            "sentence-transformers",
            "transformers",
            "tokenizers",
            "onnxruntime",
            "torch",
            "numpy",
            "huggingface-hub",
        ):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                continue
        report: dict[str, object] = {
            "schema_version": 1,
            "status": "completed",
            "generated_at": datetime.now(UTC).isoformat(),
            "scope": "embedding component only; exact cosine, no lexical fusion or graph expansion",
            "git_revision": corpus.git_revision,
            "corpus_sha256": corpus.corpus_sha256,
            "suite_name": suite.name,
            "suite_sha256": suite_hash,
            "profile": arguments.profile,
            "model": profile.repository,
            "revision": profile.revision,
            "backend": profile.backend,
            "dimension": profile.dimension,
            "dtype": profile.dtype,
            "device": arguments.device
            if profile.backend == "sentence-transformers"
            else "CPUExecutionProvider (4 threads)",
            "preprocessing": {
                "query_prefix": "search_query: "
                if arguments.scs_prefixes
                else profile.query_prefix,
                "document_prefix": "search_document: "
                if arguments.scs_prefixes
                else profile.document_prefix,
                "pooling": profile.pooling,
                "normalization": "L2",
                "max_length": max_length,
            },
            "batch_size": arguments.batch_size,
            "repeats": arguments.repeats,
            "k": arguments.k,
            "acquisition_seconds": acquisition_seconds,
            "load_seconds": load_seconds,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": packages,
            },
            "memory": memory_observations(),
            "artifacts": artifacts,
            **observations,
        }
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "aggregate": report["aggregate"],
                    "documents": report["documents"],
                }
            )
        )
        return 0
    except Exception as exc:
        output.write_text(
            json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
