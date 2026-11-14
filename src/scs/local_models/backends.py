"""Synchronous inference over verified local model directories.

The daemon's shared local runtime owns serialization and batching. This module
owns model-native formatting and validation, without downloads or service I/O.
Optional inference libraries are imported only when their model role is loaded.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import gc
import hashlib
import importlib
import math
from pathlib import Path
from typing import Final, Protocol, cast

from scs.local_models.artifacts import ModelRole, ModelSpec, load_manifest
from scs.orchestration.decision import (
    PLAYBOOK_DESCRIPTIONS,
    PLAYBOOK_SELECTION_INSTRUCTIONS,
    DecisionProviderKind,
    Playbook,
    RoutingDecision,
    RoutingRequest,
    eligible_playbooks,
)
from scs.providers.base import ProviderUnavailableError


EMBEDDING_DIMENSION: Final[int] = 1024
EMBEDDING_MAX_LENGTH: Final[int] = 1024
RERANKING_MAX_LENGTH: Final[int] = 2048
INFERENCE_BATCH_SIZE: Final[int] = 4

EMBEDDING_INSTRUCTION: Final[str] = (
    "Given a question about a code repository, retrieve the relevant code and documentation."
)
EMBEDDING_QUERY_PREFIX: Final[str] = f"Instruct: {EMBEDDING_INSTRUCTION}\nQuery:"
RERANKING_INSTRUCTION: Final[str] = (
    "Given a code search query, retrieve relevant code definitions and documentation."
)
# Vectors with another instruction or context cap must never share an index.
EMBEDDING_FORMAT_FINGERPRINT: Final[str] = hashlib.sha256(
    f"{EMBEDDING_QUERY_PREFIX}|bare-documents|last-token|L2|{EMBEDDING_MAX_LENGTH}".encode()
).hexdigest()[:16]


class _Array(Protocol):
    def tolist(self) -> object: ...
    def reshape(self, dimension: int) -> _Array: ...


class _SentenceModel(Protocol):
    max_seq_length: int

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> _Array: ...


class _CrossEncoder(Protocol):
    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> _Array: ...


class _SentenceTransformers(Protocol):
    def SentenceTransformer(
        self,
        path: str,
        *,
        device: str,
        trust_remote_code: bool,
        local_files_only: bool,
        model_kwargs: dict[str, object],
    ) -> _SentenceModel: ...

    def CrossEncoder(
        self,
        path: str,
        *,
        device: str,
        trust_remote_code: bool,
        local_files_only: bool,
        max_length: int,
        model_kwargs: dict[str, object],
        prompts: dict[str, str],
    ) -> _CrossEncoder: ...


class _MPS(Protocol):
    def synchronize(self) -> None: ...
    def empty_cache(self) -> None: ...


class _Torch(Protocol):
    bfloat16: object
    mps: _MPS


class _LayaAgent(Protocol):
    def predict(
        self, state: dict[str, object], questions: dict[str, object]
    ) -> object: ...


class _Laya(Protocol):
    def load(
        self, path: str, *, revision: str, local_files_only: bool
    ) -> _LayaAgent: ...


class _MLX(Protocol):
    def clear_cache(self) -> None: ...


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProviderUnavailableError("local model returned a non-object answer")
    return cast(Mapping[str, object], value)


def _numbers(value: object) -> list[float]:
    if not isinstance(value, list):
        raise ProviderUnavailableError("local model returned a non-list numeric result")
    result: list[float] = []
    for item in cast(list[object], value):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ProviderUnavailableError(
                "local model returned a non-numeric component"
            )
        number = float(item)
        if not math.isfinite(number):
            raise ProviderUnavailableError(
                "local model returned a non-finite component"
            )
        result.append(number)
    return result


def _vectors(value: object, count: int) -> list[list[float]]:
    if not isinstance(value, list) or len(cast(list[object], value)) != count:
        raise ProviderUnavailableError(
            "local embedding model returned the wrong row count"
        )
    vectors = [_numbers(item) for item in cast(list[object], value)]
    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSION:
            raise ProviderUnavailableError(
                "local embedding model returned the wrong dimension"
            )
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or norm == 0:
            raise ProviderUnavailableError(
                "local embedding model returned an invalid vector norm"
            )
    return vectors


def _require_complete_input(usage: Mapping[str, object]) -> None:
    """A truncated state cannot support a reliable workflow selection."""

    truncated = usage.get("truncated")
    dropped = usage.get("state_tokens_dropped", 0)
    questions = usage.get("truncated_questions", [])
    if (
        not isinstance(truncated, bool)
        or not isinstance(dropped, int)
        or isinstance(dropped, bool)
        or dropped < 0
        or not isinstance(questions, list)
    ):
        raise ProviderUnavailableError(
            "local decision model returned invalid input usage"
        )
    if truncated or dropped > 0 or questions:
        raise ProviderUnavailableError("local decision model truncated its input")


class InferenceBackend:
    """Own model objects accessed exclusively by the runtime's worker thread.

    ``load`` accepts only existing local directories. The artifact cache must
    verify those directories before this boundary. Model libraries cannot fetch
    replacements or executable model code. Instances close permanently.
    """

    def __init__(
        self,
        *,
        module_loader: Callable[[str], object] = importlib.import_module,
        specs: Mapping[ModelRole, ModelSpec] | None = None,
    ) -> None:
        self._module_loader: Callable[[str], object] = module_loader
        self._specs: Mapping[ModelRole, ModelSpec] | None = (
            dict(specs) if specs is not None else None
        )
        self._paths: dict[ModelRole, Path] = {}
        self._embedding: _SentenceModel | None = None
        self._reranking: _CrossEncoder | None = None
        self._decision: _LayaAgent | None = None
        self._torch: _Torch | None = None
        self._sentences: _SentenceTransformers | None = None
        self._mlx: _MLX | None = None
        self._closed: bool = False

    def _require_open(self) -> None:
        if self._closed:
            raise ProviderUnavailableError("local inference backend is closed")

    def _spec(self, role: ModelRole) -> ModelSpec:
        if self._specs is None:
            self._specs = load_manifest()
        return self._specs[role]

    def _qwen_modules(self) -> tuple[_Torch, _SentenceTransformers]:
        if self._torch is None:
            self._torch = cast(_Torch, self._module_loader("torch"))
        if self._sentences is None:
            self._sentences = cast(
                _SentenceTransformers, self._module_loader("sentence_transformers")
            )
        return self._torch, self._sentences

    def load(self, role: ModelRole, path: Path) -> None:
        """Load one pinned role once; report dependency or model errors explicitly."""

        self._require_open()
        try:
            directory = path.resolve(strict=True)
            if not directory.is_dir():
                raise ProviderUnavailableError("local model directory is unavailable")
        except OSError as error:
            raise ProviderUnavailableError(
                "local model directory is unavailable"
            ) from error
        if role in self._paths:
            if self._paths[role] != directory:
                raise ProviderUnavailableError(
                    "local model is already loaded from another directory"
                )
            return

        try:
            if role is ModelRole.EMBEDDING:
                torch, sentences = self._qwen_modules()
                model = sentences.SentenceTransformer(
                    str(directory),
                    device="mps",
                    trust_remote_code=False,
                    local_files_only=True,
                    model_kwargs={
                        "torch_dtype": torch.bfloat16,
                        "use_safetensors": True,
                    },
                )
                model.max_seq_length = EMBEDDING_MAX_LENGTH
                self._embedding = model
            elif role is ModelRole.RERANKING:
                torch, sentences = self._qwen_modules()
                self._reranking = sentences.CrossEncoder(
                    str(directory),
                    device="mps",
                    trust_remote_code=False,
                    local_files_only=True,
                    max_length=RERANKING_MAX_LENGTH,
                    model_kwargs={"dtype": torch.bfloat16, "use_safetensors": True},
                    prompts={"query": RERANKING_INSTRUCTION},
                )
            else:
                laya = cast(_Laya, self._module_loader("laya_mlx"))
                self._mlx = cast(_MLX, self._module_loader("mlx.core"))
                self._decision = laya.load(
                    str(directory),
                    revision=self._spec(role).revision,
                    local_files_only=True,
                )
        except Exception as error:
            raise ProviderUnavailableError(
                f"local {role.value} model could not load: {error}"
            ) from error
        self._paths[role] = directory

    def embed(self, texts: list[str], *, query: bool) -> list[list[float]]:
        """Encode bare documents or Qwen's native instructed queries in order."""

        self._require_open()
        if not texts:
            return []
        model, torch = self._embedding, self._torch
        if model is None or torch is None:
            raise ProviderUnavailableError("local embedding model is not loaded")
        inputs = [
            f"{EMBEDDING_QUERY_PREFIX}{text}" if query else text for text in texts
        ]
        try:
            result = model.encode(
                inputs,
                batch_size=INFERENCE_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            torch.mps.synchronize()
            return _vectors(result.tolist(), len(inputs))
        except ProviderUnavailableError:
            raise
        except Exception as error:
            raise ProviderUnavailableError(
                f"local embedding inference failed: {error}"
            ) from error

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Return upstream yes/no logit scores at their original document positions."""

        self._require_open()
        if not documents:
            return []
        model, torch = self._reranking, self._torch
        if model is None or torch is None:
            raise ProviderUnavailableError("local reranking model is not loaded")
        try:
            result = model.predict(
                [(query, document) for document in documents],
                batch_size=INFERENCE_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            torch.mps.synchronize()
            scores = _numbers(result.reshape(-1).tolist())
            if len(scores) != len(documents):
                raise ProviderUnavailableError(
                    "local reranker returned the wrong score count"
                )
            return scores
        except ProviderUnavailableError:
            raise
        except Exception as error:
            raise ProviderUnavailableError(
                f"local reranking inference failed: {error}"
            ) from error

    def classify(self, request: RoutingRequest) -> RoutingDecision:
        """Select an eligible playbook only when Laya consumed the complete input."""

        self._require_open()
        if self._decision is None:
            raise ProviderUnavailableError("local decision model is not loaded")
        available = eligible_playbooks(request)
        question: dict[str, object] = {
            "type": "choice",
            "instructions": PLAYBOOK_SELECTION_INSTRUCTIONS,
            "criteria": {
                item.value: PLAYBOOK_DESCRIPTIONS[item]
                for item in Playbook
                if item in available
            },
        }
        try:
            raw = _mapping(
                self._decision.predict(
                    request.model_dump(exclude_none=True),
                    {"playbook": question},
                )
            )
            _require_complete_input(_mapping(raw.get("usage")))
            answer = _mapping(_mapping(raw.get("answers")).get("playbook"))
            probabilities = _mapping(answer.get("probabilities"))
            if set(probabilities) != {item.value for item in available}:
                raise ProviderUnavailableError(
                    "local decision probabilities differ from eligible playbooks"
                )
            choice = answer.get("choice")
            if not isinstance(choice, str) or choice not in {
                item.value for item in available
            }:
                raise ProviderUnavailableError(
                    "local decision model selected an ineligible playbook"
                )
            return RoutingDecision.model_validate(
                {
                    "playbook": Playbook(choice),
                    "provider": DecisionProviderKind.LOCAL,
                    "model": self._spec(ModelRole.DECISION).identity,
                    "confidence": answer.get("confidence"),
                    "probabilities": {
                        item: probabilities.get(item.value, 0.0) for item in Playbook
                    },
                }
            )
        except ProviderUnavailableError:
            raise
        except Exception as error:
            raise ProviderUnavailableError(
                f"local decision inference failed: {error}"
            ) from error

    def close(self) -> None:
        """Release model references and unused allocations on the owning worker."""

        if self._closed:
            return
        self._closed = True
        self._embedding = None
        self._reranking = None
        self._decision = None
        self._paths.clear()
        gc.collect()
        torch, mlx = self._torch, self._mlx
        self._torch = None
        self._sentences = None
        self._mlx = None
        if torch is not None:
            torch.mps.synchronize()
            torch.mps.empty_cache()
        if mlx is not None:
            mlx.clear_cache()
