"""Exercise local model boundaries without downloading weights or importing ML."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast
import weakref

import pytest

from scs.local_models.artifacts import ModelRole, load_manifest
from scs.local_models.backends import (
    EMBEDDING_DIMENSION,
    EMBEDDING_FORMAT_FINGERPRINT,
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_QUERY_PREFIX,
    RERANKING_INSTRUCTION,
    RERANKING_MAX_LENGTH,
    InferenceBackend,
)
from scs.orchestration.decision import Playbook, RoutingRequest, eligible_playbooks
from scs.providers.base import ProviderUnavailableError


class Array:
    def __init__(self, values: object) -> None:
        self.values: object = values

    def tolist(self) -> object:
        return self.values

    def reshape(self, dimension: int) -> Array:
        assert dimension == -1
        if isinstance(self.values, list):
            values = cast(list[object], self.values)
            if all(isinstance(item, list) for item in values):
                return Array(
                    [
                        component
                        for item in values
                        for component in cast(list[object], item)
                    ]
                )
        return self


class MPS:
    def __init__(self) -> None:
        self.synchronizations: int = 0
        self.cache_clears: int = 0

    def synchronize(self) -> None:
        self.synchronizations += 1

    def empty_cache(self) -> None:
        self.cache_clears += 1


class Torch:
    def __init__(self) -> None:
        self.bfloat16: object = object()
        self.mps: MPS = MPS()


class MLX:
    def __init__(self) -> None:
        self.cache_clears: int = 0

    def clear_cache(self) -> None:
        self.cache_clears += 1


class SentenceModel:
    def __init__(self) -> None:
        self.max_seq_length: int = 0
        self.texts: list[str] = []
        self.arguments: dict[str, object] = {}
        self.output: object | None = None

    def encode(self, texts: list[str], **arguments: object) -> Array:
        self.texts = texts.copy()
        self.arguments = arguments
        vectors = [[1.0] + [0.0] * (EMBEDDING_DIMENSION - 1) for _ in texts]
        return Array(vectors if self.output is None else self.output)


class CrossModel:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []
        self.arguments: dict[str, object] = {}
        self.output: object | None = None

    def predict(self, pairs: list[tuple[str, str]], **arguments: object) -> Array:
        self.pairs = pairs.copy()
        self.arguments = arguments
        return Array(
            [float(index) for index in range(len(pairs))]
            if self.output is None
            else self.output
        )


class Sentences:
    def __init__(self) -> None:
        self.encoder: SentenceModel = SentenceModel()
        self.reranker: CrossModel = CrossModel()
        self.loads: list[tuple[str, str, dict[str, object]]] = []

    def SentenceTransformer(self, path: str, **arguments: object) -> SentenceModel:
        self.loads.append(("embedding", path, arguments))
        return self.encoder

    def CrossEncoder(self, path: str, **arguments: object) -> CrossModel:
        self.loads.append(("reranking", path, arguments))
        return self.reranker


class LayaAgent:
    def __init__(self) -> None:
        self.state: dict[str, object] = {}
        self.questions: dict[str, object] = {}
        self.output: object | None = None

    def predict(self, state: dict[str, object], questions: dict[str, object]) -> object:
        self.state = state
        self.questions = questions
        if self.output is not None:
            return self.output
        question = cast(Mapping[str, object], questions["playbook"])
        criteria = cast(Mapping[str, object], question["criteria"])
        return {
            "answers": {
                "playbook": {
                    "choice": "DISCOVER",
                    "confidence": 1.0,
                    "probabilities": {
                        key: float(key == "DISCOVER") for key in criteria
                    },
                }
            },
            "usage": {"truncated": False},
        }


class Laya:
    def __init__(self) -> None:
        self.agent: LayaAgent = LayaAgent()
        self.loads: list[tuple[str, dict[str, object]]] = []

    def load(self, path: str, **arguments: object) -> LayaAgent:
        self.loads.append((path, arguments))
        return self.agent


class Modules:
    def __init__(self) -> None:
        self.torch: Torch = Torch()
        self.sentences: Sentences = Sentences()
        self.laya: Laya = Laya()
        self.mlx: MLX = MLX()
        self.imports: list[str] = []

    def load(self, name: str) -> object:
        self.imports.append(name)
        return {
            "torch": self.torch,
            "sentence_transformers": self.sentences,
            "laya_mlx": self.laya,
            "mlx.core": self.mlx,
        }[name]


def test_construction_and_empty_work_do_not_import_model_libraries() -> None:
    modules = Modules()
    backend = InferenceBackend(module_loader=modules.load)

    assert backend.embed([], query=False) == []
    assert backend.rerank("query", []) == []
    backend.close()
    assert modules.imports == []
    assert len(EMBEDDING_FORMAT_FINGERPRINT) == 16


def test_embedding_load_is_local_only_and_uses_native_query_format(
    tmp_path: Path,
) -> None:
    modules = Modules()
    backend = InferenceBackend(module_loader=modules.load)

    backend.load(ModelRole.EMBEDDING, tmp_path)
    backend.load(ModelRole.EMBEDDING, tmp_path)
    assert len(modules.sentences.loads) == 1
    _, path, arguments = modules.sentences.loads[0]
    assert path == str(tmp_path.resolve())
    assert arguments == {
        "device": "mps",
        "trust_remote_code": False,
        "local_files_only": True,
        "model_kwargs": {
            "torch_dtype": modules.torch.bfloat16,
            "use_safetensors": True,
        },
    }
    assert modules.sentences.encoder.max_seq_length == EMBEDDING_MAX_LENGTH
    assert len(backend.embed(["function one", "function two"], query=False)) == 2
    assert modules.sentences.encoder.texts == ["function one", "function two"]
    assert modules.sentences.encoder.arguments["normalize_embeddings"] is True

    backend.embed(["find one"], query=True)
    assert modules.sentences.encoder.texts == [f"{EMBEDDING_QUERY_PREFIX}find one"]
    assert modules.torch.mps.synchronizations == 2
    assert "laya_mlx" not in modules.imports


@pytest.mark.parametrize(
    "output",
    [
        [],
        [[0.0] * 1024],
        [[True] * 1024],
        [[float("nan")] * 1024],
        [[float("inf")] * 1024],
        [[1.0] * 1023],
        [["invalid"] * 1024],
    ],
)
def test_embedding_rejects_unusable_vectors(tmp_path: Path, output: object) -> None:
    modules = Modules()
    backend = InferenceBackend(module_loader=modules.load)
    backend.load(ModelRole.EMBEDDING, tmp_path)
    modules.sentences.encoder.output = output

    with pytest.raises(ProviderUnavailableError):
        backend.embed(["document"], query=False)


def test_reranker_uses_upstream_cross_encoder_modules(tmp_path: Path) -> None:
    modules = Modules()
    backend = InferenceBackend(module_loader=modules.load)
    backend.load(ModelRole.RERANKING, tmp_path)

    _, path, arguments = modules.sentences.loads[0]
    assert path == str(tmp_path.resolve())
    assert arguments == {
        "device": "mps",
        "trust_remote_code": False,
        "local_files_only": True,
        "max_length": RERANKING_MAX_LENGTH,
        "model_kwargs": {"dtype": modules.torch.bfloat16, "use_safetensors": True},
        "prompts": {"query": RERANKING_INSTRUCTION},
    }
    modules.sentences.reranker.output = [[-1.0], [3.0]]
    assert backend.rerank("find implementation", ["first", "second"]) == [-1.0, 3.0]
    assert modules.sentences.reranker.pairs == [
        ("find implementation", "first"),
        ("find implementation", "second"),
    ]
    assert modules.sentences.reranker.arguments["batch_size"] == 4


@pytest.mark.parametrize("output", [[1.0], [True, 1.0], [float("inf"), 1.0]])
def test_reranker_rejects_invalid_scores(tmp_path: Path, output: object) -> None:
    modules = Modules()
    backend = InferenceBackend(module_loader=modules.load)
    backend.load(ModelRole.RERANKING, tmp_path)
    modules.sentences.reranker.output = output

    with pytest.raises(ProviderUnavailableError):
        backend.rerank("query", ["first", "second"])


def test_laya_uses_only_eligible_workflows_and_pinned_identity(tmp_path: Path) -> None:
    modules = Modules()
    backend = InferenceBackend(module_loader=modules.load)
    backend.load(ModelRole.DECISION, tmp_path)
    request = RoutingRequest(goal="find code")

    decision = backend.classify(request)
    assert decision.provider.value == "local"
    assert decision.model == load_manifest()[ModelRole.DECISION].identity
    assert decision.playbook is Playbook.DISCOVER
    assert set(decision.probabilities) == set(Playbook)
    assert decision.probabilities[Playbook.REFERENCES] == 0.0
    question = cast(Mapping[str, object], modules.laya.agent.questions["playbook"])
    assert set(cast(Mapping[str, object], question["criteria"])) == {
        item.value for item in eligible_playbooks(request)
    }
    assert modules.laya.loads == [
        (
            str(tmp_path.resolve()),
            {
                "revision": load_manifest()[ModelRole.DECISION].revision,
                "local_files_only": True,
            },
        )
    ]
    assert "torch" not in modules.imports


@pytest.mark.parametrize(
    "failure",
    [
        "truncated",
        "dropped_state",
        "truncated_question",
        "missing_probability",
        "ineligible_choice",
    ],
)
def test_laya_rejects_incomplete_or_truncated_decisions(
    tmp_path: Path,
    failure: str,
) -> None:
    modules = Modules()
    backend = InferenceBackend(module_loader=modules.load)
    backend.load(ModelRole.DECISION, tmp_path)
    request = RoutingRequest(goal="find code")
    probabilities = {
        item.value: float(item is Playbook.DISCOVER)
        for item in eligible_playbooks(request)
    }
    if failure == "missing_probability":
        probabilities.pop("UNDERSTAND")
    modules.laya.agent.output = {
        "answers": {
            "playbook": {
                "choice": "REFERENCES"
                if failure == "ineligible_choice"
                else "DISCOVER",
                "confidence": 1.0,
                "probabilities": probabilities,
            }
        },
        "usage": {
            "truncated": failure == "truncated",
            "state_tokens_dropped": int(failure == "dropped_state"),
            "truncated_questions": ["playbook"]
            if failure == "truncated_question"
            else [],
        },
    }

    with pytest.raises(ProviderUnavailableError):
        backend.classify(request)


def test_close_releases_loaded_models_and_rejects_further_work(tmp_path: Path) -> None:
    modules = Modules()
    backend = InferenceBackend(module_loader=modules.load)
    for role in ModelRole:
        backend.load(role, tmp_path)

    models = [
        weakref.ref(modules.sentences.encoder),
        weakref.ref(modules.sentences.reranker),
        weakref.ref(modules.laya.agent),
    ]
    del modules.sentences.encoder
    del modules.sentences.reranker
    del modules.laya.agent

    backend.close()
    backend.close()

    assert all(model() is None for model in models)
    assert modules.torch.mps.cache_clears == 1
    assert modules.mlx.cache_clears == 1
    with pytest.raises(ProviderUnavailableError, match="closed"):
        backend.embed(["document"], query=False)
    with pytest.raises(ProviderUnavailableError, match="closed"):
        backend.load(ModelRole.EMBEDDING, tmp_path)


def test_unloaded_role_cannot_execute_inference(tmp_path: Path) -> None:
    backend = InferenceBackend(module_loader=Modules().load)

    with pytest.raises(ProviderUnavailableError, match="loaded"):
        backend.embed(["document"], query=False)
    with pytest.raises(ProviderUnavailableError, match="directory"):
        backend.load(ModelRole.EMBEDDING, tmp_path / "absent")
