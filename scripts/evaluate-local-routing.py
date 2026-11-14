#!/usr/bin/env python3
"""Compare local routing models without changing SCS configuration or indexes.

Raw model accuracy uses only labels the model could select. Policy previews
apply SCS's explicit-intent precedence, but do not claim graph anchor resolution
or end-to-end query quality. Model inference runs sequentially in this process.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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

from scs.evaluation.query import QueryEvaluationCase, load_query_suite
from scs.orchestration.decision import (
    PLAYBOOK_DESCRIPTIONS,
    DecisionProvider,
    DeterministicDecisionProvider,
    Playbook,
    RoutingDecision,
    RoutingRequest,
    eligible_playbooks,
)
from scs.orchestration.external_decision_provider import (
    QUESTION,
    ChoiceResponse,
    ExternalDecisionProvider,
)
from scs.orchestration.query import MODE_BUDGETS, requested_playbook


LAYA_REVISIONS = {
    "aac6fef/laya-mlx": "20aed815fc6acde75733882e7ec0e3f28aeb9717",
    "aac6fef/laya-multilingual-mlx": "f2b4faf51023039425946074e2cf1361d2db11d5",
}
MAX_CASES = 200
MAX_REPEATS = 20
MAX_WARMUPS = 10


class LayaAgent(Protocol):
    model_dir: Path
    revision: str | None

    def predict(
        self, state: dict[str, object], questions: dict[str, object]
    ) -> object: ...


class LayaModule(Protocol):
    def load(
        self, model: str, *, revision: str, local_files_only: bool
    ) -> LayaAgent: ...


class MetalModule(Protocol):
    def get_peak_memory(self) -> int: ...
    def get_active_memory(self) -> int: ...


def mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("model response must contain an object")
    return cast(Mapping[str, object], value)


class LayaProvider:
    """Translate one direct Laya answer through SCS's production validators."""

    def __init__(self, agent: LayaAgent, identity: str) -> None:
        self.agent: LayaAgent = agent
        self.identity: str = identity
        self.last_usage: dict[str, object] = {}

    async def classify(self, request: RoutingRequest) -> RoutingDecision:
        available = eligible_playbooks(request)
        question: dict[str, object] = {
            "type": "choice",
            "instructions": QUESTION,
            "criteria": {
                item.value: PLAYBOOK_DESCRIPTIONS[item]
                for item in Playbook
                if item in available
            },
        }
        raw = mapping(
            await asyncio.to_thread(
                self.agent.predict,
                request.model_dump(exclude_none=True),
                {"playbook": question},
            )
        )
        answer = mapping(mapping(raw["answers"])["playbook"])
        self.last_usage = dict(mapping(raw.get("usage", {})))
        response = ChoiceResponse.model_validate(
            {
                "model": self.identity,
                "choice": answer["choice"],
                "confidence": answer["confidence"],
                "probabilities": answer["probabilities"],
            }
        )
        if set(response.probabilities) != {item.value for item in available}:
            raise ValueError("model probabilities differ from eligible playbooks")
        if response.choice not in {item.value for item in available}:
            raise ValueError("model selected an ineligible playbook")
        return RoutingDecision(
            playbook=Playbook(response.choice),
            model=self.identity,
            confidence=response.confidence,
            probabilities={
                item: response.probabilities.get(item.value, 0.0) for item in Playbook
            },
        )


@dataclass(frozen=True)
class Sample:
    case_index: int
    repeat: int
    goal: str
    expected: str
    choice: str
    raw_scored: bool
    raw_correct: bool | None
    policy_preview: str
    policy_preview_correct: bool
    requires_anchor_resolution: bool
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    usage: dict[str, object]


def sample(
    case: QueryEvaluationCase,
    decision: RoutingDecision,
    *,
    case_index: int,
    repeat: int,
    latency_ms: float,
    usage: dict[str, object],
) -> Sample:
    """Keep impossible model labels separate from post-classifier intent policy.

    This mirrors only the precedence immediately after classification. An intent
    without eligible anchors still needs a successful graph lookup at execution.
    """

    request = RoutingRequest.model_validate({"goal": case.goal, **case.anchors})
    available = eligible_playbooks(request)
    raw_scored = case.expected_playbook in available
    intent = requested_playbook(request.goal)
    preview = intent or decision.playbook
    if preview not in available and intent is None:
        preview = Playbook.DISCOVER
    return Sample(
        case_index=case_index,
        repeat=repeat,
        goal=case.goal,
        expected=case.expected_playbook.value,
        choice=decision.playbook.value,
        raw_scored=raw_scored,
        raw_correct=(decision.playbook == case.expected_playbook)
        if raw_scored
        else None,
        policy_preview=preview.value,
        policy_preview_correct=preview == case.expected_playbook,
        requires_anchor_resolution=preview not in available,
        confidence=decision.confidence,
        probabilities={
            item.value: probability
            for item, probability in decision.probabilities.items()
        },
        latency_ms=latency_ms,
        usage=usage,
    )


def summarize(samples: Sequence[Sample]) -> dict[str, object]:
    scored = [item for item in samples if item.raw_scored]
    ordered = sorted(item.latency_ms for item in samples)

    return {
        "sample_count": len(samples),
        "raw_scored_count": len(scored),
        "raw_accuracy": sum(bool(item.raw_correct) for item in scored) / len(scored)
        if scored
        else None,
        "raw_multiclass_brier": sum(
            sum(
                (probability - float(label == item.expected)) ** 2
                for label, probability in item.probabilities.items()
            )
            for item in scored
        )
        / len(scored)
        if scored
        else None,
        "policy_preview_accuracy": sum(item.policy_preview_correct for item in samples)
        / len(samples),
        "policy_requires_anchor_resolution_count": sum(
            item.requires_anchor_resolution for item in samples
        ),
        "p50_latency_ms": ordered[math.ceil(len(ordered) * 0.5) - 1],
        "p95_latency_ms": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "observed_deadline_miss_rates": {
            mode: sum(
                item.latency_ms > budget.classifier_seconds * 1_000 for item in samples
            )
            / len(samples)
            for mode, budget in MODE_BUDGETS.items()
        },
        "truncated_sample_count": sum(
            bool(item.usage.get("truncated")) for item in samples
        ),
    }


def model_metadata(provider: DecisionProvider) -> dict[str, object]:
    if not isinstance(provider, LayaProvider):
        return {}
    path = provider.agent.model_dir
    weight = path / "model.safetensors"
    with weight.open("rb") as contents:
        digest = hashlib.file_digest(contents, "sha256").hexdigest()
    return {
        "model_directory": str(path),
        "resolved_snapshot_revision": provider.agent.revision,
        "weight_bytes": weight.stat().st_size,
        "weight_sha256": digest,
    }


async def run(arguments: argparse.Namespace) -> dict[str, object]:
    backend = cast(str, arguments.backend)
    offline = cast(bool, arguments.offline)

    if offline and backend == "external":
        raise ValueError(
            "offline trials require direct inference, not an external service"
        )
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    suite_path = cast(Path, arguments.suite)
    suite_bytes = suite_path.read_bytes()
    suite_sha256 = hashlib.sha256(suite_bytes).hexdigest()
    suite = load_query_suite(suite_path)
    if suite_path.read_bytes() != suite_bytes:
        raise ValueError("evaluation suite changed while loading")
    repeats = cast(int, arguments.repeats)
    warmups = cast(int, arguments.warmups)
    if not 1 <= repeats <= MAX_REPEATS or not 0 <= warmups <= MAX_WARMUPS:
        raise ValueError("repeats or warmups exceed the bounded trial limits")
    if len(suite.cases) > MAX_CASES:
        raise ValueError("suite exceeds the bounded case limit")

    provider: DecisionProvider
    external: ExternalDecisionProvider | None = None
    identity = "deterministic-v1"
    setup_started = perf_counter()
    if backend == "external":
        identity = cast(str | None, arguments.model) or ""
        if not identity:
            raise ValueError("external trials require an exact --model identity")
        external = ExternalDecisionProvider(
            cast(str, arguments.base_url), model=identity
        )
        provider = external
    elif backend == "laya":
        model = cast(str | None, arguments.model) or next(iter(LAYA_REVISIONS))
        if model not in LAYA_REVISIONS:
            raise ValueError(
                "Laya trials support only the two pinned conversion repositories"
            )
        revision = LAYA_REVISIONS[model]
        identity = f"{model}@{revision}"
        module = cast(LayaModule, cast(object, import_module("laya_mlx")))
        agent = module.load(model, revision=revision, local_files_only=offline)
        if agent.revision != revision:
            raise ValueError("loaded snapshot differs from the pinned model revision")
        provider = LayaProvider(agent, identity)
    else:
        provider = DeterministicDecisionProvider()

    samples: list[Sample] = []
    warmup_latencies: list[float] = []
    try:
        if external is not None:
            await external.start()
        setup_seconds = perf_counter() - setup_started
        for _ in range(warmups):
            request = RoutingRequest.model_validate(
                {"goal": suite.cases[0].goal, **suite.cases[0].anchors}
            )
            started = perf_counter()
            await provider.classify(request)
            warmup_latencies.append((perf_counter() - started) * 1_000)
        for repeat in range(repeats):
            for index, case in enumerate(suite.cases):
                request = RoutingRequest.model_validate(
                    {"goal": case.goal, **case.anchors}
                )
                started = perf_counter()
                decision = RoutingDecision.model_validate(
                    await provider.classify(request)
                )
                elapsed = (perf_counter() - started) * 1_000
                samples.append(
                    sample(
                        case,
                        decision,
                        case_index=index,
                        repeat=repeat,
                        latency_ms=elapsed,
                        usage=provider.last_usage.copy()
                        if isinstance(provider, LayaProvider)
                        else {},
                    )
                )
    finally:
        if external is not None:
            await external.close()

    metal = cast(MetalModule | None, sys.modules.get("mlx.core"))
    versions: dict[str, str] = {}
    for package in ("pydantic", "laya-mlx", "mlx", "tokenizers", "huggingface-hub"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    checkpoint_metadata = model_metadata(provider)
    if suite_path.read_bytes() != suite_bytes:
        raise ValueError("evaluation suite changed during the trial")

    return {
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "suite": suite.name,
        "suite_sha256": suite_sha256,
        "backend": backend,
        "model": identity,
        "offline_requested": offline,
        "setup_seconds": setup_seconds,
        "warmup_latencies_ms": warmup_latencies,
        "summary": summarize(samples),
        "samples": [asdict(item) for item in samples],
        "metadata": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": versions,
            "optional_model_modules_loaded": [
                name
                for name in ("laya_mlx", "mlx.core", "torch")
                if name in sys.modules
            ],
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "metal_peak_allocated_bytes": metal.get_peak_memory() if metal else None,
            "metal_active_allocated_bytes": metal.get_active_memory()
            if metal
            else None,
            **checkpoint_metadata,
        },
        "limitations": [
            "Policy previews precede graph anchor resolution; no index or query execution is measured.",
            "Deadline miss rates observe inference duration; cancellation and production fallback are not executed.",
            "Deadline rates use mode ceilings without the configured decision_timeout_seconds cap.",
            "Confidence definitions differ across providers; Brier scoring uses the complete probability vector.",
            "RSS is a process lifetime peak; Metal allocations are separate and must not be added to RSS.",
            "External setup measures client readiness; external service loading and memory are excluded.",
            "Offline mode restricts the loader to local files; operating-system network isolation must be applied and recorded externally.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("deterministic", "external", "laya"),
        default="deterministic",
    )
    parser.add_argument("--model")
    parser.add_argument("--base-url", default="http://127.0.0.1:10000/v1")
    parser.add_argument("--suite", type=Path, default=Path("evals/scs-query-v4.json"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = cast(Path, arguments.output).resolve()
    if output == cast(Path, arguments.suite).resolve():
        parser.error("--output must differ from the evaluation suite")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"status":"running"}\n', encoding="utf-8")

    result: dict[str, object]
    try:
        result = asyncio.run(run(arguments))
    except Exception as error:
        result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
