"""One classifier decision followed by a closed, bounded code-query playbook."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
import re
from time import perf_counter
from typing import ClassVar, Literal, TypedDict, cast

from pydantic import ConfigDict, Field, model_validator

from scs.graph.models import SYMBOL_NODE_TYPES, NodeType, RelationshipType
from scs.orchestration.laya_provider import MODEL_REPOSITORY
from scs.orchestration.decision import (
    DecisionProvider,
    DeterministicDecisionProvider,
    Playbook,
    RoutingDecision,
    RoutingRequest,
    SourcePosition,
    eligible_playbooks,
)
from scs.source_paths import validated_source_path
from scs.services.impact import ChangedRange
from scs.services.contracts import collect_contracts
from scs.orchestration.coverage import AspectCoverage, investigation_coverage

QueryMode = Literal["fast", "balanced", "thorough"]
ServiceCall = Callable[[str, dict[str, object]], Awaitable[dict[str, object]]]


@dataclass(frozen=True, slots=True)
class ModeBudget:
    """Fixed per-mode ceilings that a classifier cannot change."""

    classifier_seconds: float
    search_seeds: int
    graph_depth: int
    hydrated_files: int
    total_seconds: float


MODE_BUDGETS: dict[QueryMode, ModeBudget] = {
    "fast": ModeBudget(0.15, 5, 1, 1, 1.0),
    "balanced": ModeBudget(0.5, 10, 2, 3, 7.0),
    "thorough": ModeBudget(1.0, 20, 3, 5, 35.0),
}

REFERENCE_RELATIONSHIPS = (
    RelationshipType.IMPORTS,
    RelationshipType.CALLS,
    RelationshipType.REFERENCES,
    RelationshipType.INHERITS,
    RelationshipType.IMPLEMENTS,
)


REFERENCE_GOAL = re.compile(r"\b(references to|callers of|call sites|usages of|incoming references)\b", re.I)
IMPACT_GOAL = re.compile(r"\b(affected (?:tests|dependents|consumers)|impact of|dependents of|tests affected)\b", re.I)


def requested_playbook(goal: str) -> Playbook | None:
    """Recognize explicit investigation phrases without interpreting arbitrary prose."""

    if IMPACT_GOAL.search(goal):
        return Playbook.IMPACT
    if REFERENCE_GOAL.search(goal):
        return Playbook.REFERENCES
    return None


class QueryRequest(RoutingRequest):
    """Validated public request, including repository-contained anchor paths."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    repo_path: str = Field(min_length=1, max_length=2048)
    mode: QueryMode = "balanced"
    limit: int = Field(default=10, ge=1, le=50)
    changed_ranges: list[ChangedRange] = Field(default_factory=list, max_length=100)
    contract_paths: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def _normalize_anchors(self) -> "QueryRequest":
        root = Path(self.repo_path).expanduser()
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("repo_path must be an existing absolute directory")
        root = root.resolve()
        self.repo_path = str(root)
        if self.node_type is not None:
            NodeType(self.node_type)
        self.node_ids: list[str] = list(dict.fromkeys(self.node_ids))
        if any(not identity or len(identity) > 256 for identity in self.node_ids):
            raise ValueError("node_ids must contain bounded non-empty identities")
        self.file_paths: list[str] = list(
            dict.fromkeys(self._relative_path(path, root) for path in self.file_paths)
        )
        self.contract_paths = list(dict.fromkeys(self._relative_path(path, root) for path in self.contract_paths))
        if any(not path.lower().endswith(".md") for path in self.contract_paths):
            raise ValueError("contract_paths must select Markdown documents")
        for changed in self.changed_ranges:
            changed.file_path = self._relative_path(changed.file_path, root)
            if changed.file_path not in self.file_paths:
                self.file_paths.append(changed.file_path)
        if len(self.file_paths) > 20:
            raise ValueError("at most 20 review files are allowed")
        if self.source_position is not None:
            self.source_position: SourcePosition | None = SourcePosition(
                file_path=self._relative_path(
                    self.source_position.file_path, root, require_file=True
                ),
                line=self.source_position.line,
            )
        return self

    @staticmethod
    def _relative_path(path: str, root: Path, *, require_file: bool = False) -> str:
        source = validated_source_path(path, str(root), require_file=require_file)
        return Path(source).relative_to(root).as_posix()

    def routing_request(self) -> RoutingRequest:
        """Expose only bounded caller data, without repository source or index data."""

        return RoutingRequest(
            goal=self.goal,
            node_type=self.node_type,
            symbol_name=self.symbol_name,
            node_ids=self.node_ids,
            file_paths=self.file_paths,
            source_position=self.source_position,
        )


class QueryCodeOutput(TypedDict):
    """Stable response envelope shared by all seven playbooks."""

    goal: str
    repo_path: str
    routing: dict[str, object]
    evidence: dict[str, list[dict[str, object]]]
    trace: list[dict[str, object]]
    complete: bool
    truncated: bool
    degraded_stages: list[str]
    timings: dict[str, float]
    investigation: dict[str, AspectCoverage]


def _node_evidence(
    node: dict[str, object], *, kind: str, stage: str
) -> dict[str, object]:
    metadata = node.get("metadata")
    attributes = cast(dict[str, object], metadata) if isinstance(metadata, dict) else {}
    return {
        "kind": kind,
        "id": str(node.get("id", "")),
        "name": str(node.get("name", "")),
        "node_type": str(node.get("type", "")),
        "file_path": node.get("file_path", attributes.get("file_path")),
        "start_line": node.get("start_line", attributes.get("start_line")),
        "end_line": node.get("end_line", attributes.get("end_line")),
        "stage": stage,
    }


def _objects(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [
        cast(dict[str, object], item)
        for item in cast(list[object], value)
        if isinstance(item, dict)
    ]


class QueryOrchestrator:
    """Run exactly one selected fixed playbook over finite service routes."""

    def __init__(
        self,
        *,
        provider: DecisionProvider,
        call: ServiceCall,
        classifier_timeout_seconds: float | None = None,
        wait_until_ready: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._provider: DecisionProvider = provider
        self._call: ServiceCall = call
        self._classifier_timeout: float | None = classifier_timeout_seconds
        self._wait_until_ready: Callable[[], Awaitable[None]] | None = wait_until_ready
        self._fallback: DeterministicDecisionProvider = DeterministicDecisionProvider()

    async def query(self, params: dict[str, object]) -> QueryCodeOutput:
        request = QueryRequest.model_validate(params)
        budget = MODE_BUDGETS[request.mode]
        started = perf_counter()
        if self._wait_until_ready is not None:
            await self._wait_until_ready()
        classification_started = perf_counter()
        degraded_reason: str | None = None
        try:
            decision = await asyncio.wait_for(
                self._provider.classify(request.routing_request()),
                timeout=min(
                    budget.classifier_seconds,
                    self._classifier_timeout or budget.classifier_seconds,
                ),
            )
            # A custom provider still has to cross the same strict boundary.
            decision = RoutingDecision.model_validate(decision)
        except Exception as error:
            decision = await self._fallback.classify(request.routing_request())
            degraded_reason = "classifier_timeout" if isinstance(error, TimeoutError) else "classifier_unavailable"
        classification_ms = (perf_counter() - classification_started) * 1_000
        intent = requested_playbook(request.goal)
        selected = Playbook.IMPACT if request.changed_ranges else (intent or decision.playbook)
        if selected not in eligible_playbooks(request) and intent is None:
            selected = Playbook.DISCOVER
            degraded_reason = "ineligible_playbook"

        evidence: dict[str, list[dict[str, object]]] = {
            key: [] for key in ("symbols", "files", "relationships", "references", "test_targets")
        }
        trace: list[dict[str, object]] = []
        degraded_stages: list[str] = []
        truncated = False
        anchor_resolution: dict[str, object] = {"status": "not_needed", "candidates": []}

        async def stage(
            name: str, method: str, arguments: dict[str, object]
        ) -> dict[str, object]:
            nonlocal truncated
            stage_started = perf_counter()
            try:
                response = await self._call(method, arguments)
                unavailable = response.get("available") is False
                reason = response.get("degraded_reason") or ("unavailable" if unavailable else None)
                timed_out = response.get("timed_out") is True
                was_truncated = any(
                    response.get(key) is True
                    for key in (
                        "nodes_truncated", "edges_truncated",
                        "dependents_truncated", "test_targets_truncated",
                    )
                )
                truncated = truncated or was_truncated or timed_out
                if reason or timed_out:
                    degraded_stages.append(name)
                trace.append(
                    {
                        "stage": name,
                        "status": "unavailable" if unavailable else "degraded" if reason or timed_out else "ok",
                        "counts": {
                            key: len(cast(list[object], value))
                            for key, value in response.items()
                            if isinstance(value, list)
                        },
                        "truncated": was_truncated or timed_out,
                        "elapsed_ms": (perf_counter() - stage_started) * 1_000,
                    }
                )
                return response
            except asyncio.CancelledError:
                trace.append({"stage": name, "status": "timeout", "counts": {},
                              "truncated": True, "elapsed_ms": (perf_counter() - stage_started) * 1_000})
                raise
            except Exception as error:
                degraded_stages.append(name)
                trace.append(
                    {
                        "stage": name,
                        "status": "timeout" if isinstance(error, TimeoutError) else "failed",
                        "counts": {},
                        "truncated": True,
                        "elapsed_ms": (perf_counter() - stage_started) * 1_000,
                    }
                )
                truncated = True
                return {}

        async def search() -> list[dict[str, object]]:
            response = await stage(
                "search",
                "knowledge.search",
                {
                    "query": request.goal,
                    "node_type": request.node_type,
                    "repo_path": request.repo_path,
                    "limit": min(request.limit, budget.search_seeds),
                    "search_mode": request.mode,
                    "result_detail": "compact",
                },
            )
            nodes = _objects(response.get("results"))
            self._append_nodes(evidence, nodes, "search")
            return nodes

        async def resolve(seeds: list[dict[str, object]], *, files: bool) -> bool:
            nonlocal anchor_resolution
            candidates: dict[str, dict[str, object]] = {}
            for seed in seeds:
                path = self._file_path(seed)
                if not isinstance(path, str):
                    continue
                try:
                    relative = Path(validated_source_path(path, request.repo_path, require_file=False)).relative_to(request.repo_path).as_posix()
                except ValueError:
                    continue
                if request.file_paths and relative not in request.file_paths:
                    continue
                identity = seed.get("id")
                if not files and (seed.get("type") not in SYMBOL_NODE_TYPES or not isinstance(identity, str) or not identity):
                    continue
                candidates[relative if files else str(identity)] = {**seed, "file_path": relative}
            exact = {
                key: node for key, node in candidates.items()
                if (name := node.get("file_path" if files else "name"))
                and re.search(r"(?<![\w/])" + re.escape(str(name)) + r"(?![\w/])", request.goal)
            }
            choices = exact or candidates
            # A full search page cannot establish uniqueness of a symbol candidate.
            discovery_available = not any(item["stage"] == "search" and (item["status"] != "ok" or item["truncated"]) for item in trace)
            unambiguous = discovery_available and len(choices) == 1 and (files or len(seeds) < min(request.limit, budget.search_seeds))
            anchor_resolution = {
                "status": ("unavailable" if not discovery_available else
                           "resolved" if unambiguous else ("ambiguous" if choices else "missing")),
                "scope": "returned_candidates",
                "candidates": [_node_evidence(node, kind="anchor", stage="resolve") for node in choices.values()],
            }
            trace.append({"stage": "resolve", "status": "ok" if unambiguous else "degraded",
                          "counts": {"candidates": len(choices)}, "truncated": False, "elapsed_ms": 0.0})
            if not unambiguous:
                degraded_stages.append("resolve")
                return False
            node = next(iter(choices.values()))
            if files:
                request.file_paths = [str(node["file_path"])]
            else:
                request.node_ids = [str(node["id"])]
            return True

        async def related(*, incoming: bool = False) -> None:
            identities = request.node_ids[:1]
            symbol = request.symbol_name if not identities else None
            if not identities and symbol is None:
                seeds = await search()
                if not await resolve(seeds, files=False):
                    return
                identities = request.node_ids[:1]
            if not identities and symbol is None:
                return
            edge_types: tuple[RelationshipType | None, ...] = (
                REFERENCE_RELATIONSHIPS if incoming else (None,)
            )
            for edge_type in edge_types:
                stage_name = "references" if incoming else "traversal"
                response = await stage(
                    stage_name,
                    "knowledge.related",
                    {
                        "node_id": identities[0] if identities else None,
                        "symbol_name": symbol,
                        "repo_path": request.repo_path,
                        "depth": 1 if incoming else budget.graph_depth,
                        "direction": "incoming" if incoming else "both",
                        "relationship": edge_type.value if edge_type else None,
                    },
                )
                matches = _objects(response.get("matches"))
                self._append_nodes(evidence, matches, stage_name)
                matched_seed_id = (
                    identities[0] if identities
                    else next((match.get("id") for match in matches if match.get("id")), None)
                )
                for item in _objects(response.get("related")):
                    # Traversal includes its seed. It is a definition, not a reference.
                    if incoming and item.get("depth") == 0:
                        continue
                    raw_node = item.get("node")
                    if isinstance(raw_node, dict):
                        node = cast(dict[str, object], raw_node)
                        self._append_nodes(evidence, [node], stage_name)
                    else:
                        node = item
                    relationship = {
                        "kind": "reference" if incoming else "relationship",
                        "id": str(item.get("id", node.get("id", ""))),
                        "seed_id": item.get("seed_id", matched_seed_id),
                        "target_id": node.get("id"),
                        "relationship": edge_type.value if edge_type else item.get("relationship"),
                        "direction": item.get("direction", "incoming" if incoming else "both"),
                        "file_path": self._file_path(node),
                        "start_line": node.get("start_line"),
                        "end_line": node.get("end_line"),
                        "stage": stage_name,
                    }
                    evidence["references" if incoming else "relationships"].append(relationship)

        try:
            async with asyncio.timeout(budget.total_seconds):
                if selected not in eligible_playbooks(request):
                    grounded = await resolve(await search(), files=selected is Playbook.IMPACT)
                    if not grounded:
                        selected = Playbook.DISCOVER
                        degraded_reason = "anchor_" + str(anchor_resolution["status"])
                match selected:
                    case Playbook.DISCOVER:
                        if anchor_resolution["status"] not in {"ambiguous", "missing", "unavailable"}:
                            await search()
                    case Playbook.UNDERSTAND:
                        await related()
                    case Playbook.RELATIONSHIPS:
                        await related()
                    case Playbook.REFERENCES:
                        if request.source_position is not None:
                            position = request.source_position
                            response = await stage(
                                "references", "lsp.references",
                                {
                                    "file_path": str(Path(request.repo_path) / position.file_path),
                                    "line": position.line,
                                },
                            )
                            raw_symbol = response.get("symbol")
                            if isinstance(raw_symbol, dict):
                                self._append_nodes(
                                    evidence, [cast(dict[str, object], raw_symbol)],
                                    "references",
                                )
                            self._append_nodes(
                                evidence, _objects(response.get("references")), "references",
                                category="references",
                            )
                        else:
                            await related(incoming=True)
                    case Playbook.INSPECT_FILES:
                        files = request.file_paths
                        if not files:
                            seeds = await search()
                            files = [
                                str(path) for seed in seeds
                                if (path := self._file_path(seed)) is not None
                            ]
                        unique_files = list(dict.fromkeys(files))
                        truncated = truncated or len(unique_files) > budget.hydrated_files
                        for file_path in unique_files[: budget.hydrated_files]:
                            response = await stage(
                                "inspect", "knowledge.inspect_file",
                                {
                                    "repo_path": request.repo_path,
                                    "file_path": file_path,
                                    "node_limit": min(request.limit, 50),
                                    "edge_limit": min(request.limit * 2, 100),
                                },
                            )
                            evidence["files"].append(
                                {"kind": "file", "file_path": file_path, "stage": "inspect"}
                            )
                            self._append_nodes(evidence, _objects(response.get("nodes")), "inspect")
                            raw_edges = response.get("edges")
                            edge_groups = (
                                cast(dict[str, object], raw_edges).values()
                                if isinstance(raw_edges, dict)
                                else ()
                            )
                            for edges in edge_groups:
                                for edge in _objects(edges):
                                    evidence["relationships"].append(
                                        {"kind": "relationship", "id": edge.get("id"),
                                         "source_id": edge.get("source_id"),
                                         "target_id": edge.get("target_id"),
                                         "relationship": edge.get("relationship"),
                                         "stage": "inspect"}
                                    )
                    case Playbook.IMPACT:
                        response = await stage(
                            "impact", "knowledge.composite.regression_risk",
                            {"repo_path": request.repo_path,
                             "file_paths": request.file_paths,
                             "dependent_limit": min(request.limit, 50),
                             "test_target_limit": min(request.limit, 50),
                             **({"changed_ranges": [item.model_dump() for item in request.changed_ranges]}
                                if request.changed_ranges else {})},
                        )
                        self._append_nodes(evidence, _objects(response.get("dependents")), "impact")
                        self._append_nodes(evidence, _objects(response.get("changed_symbols")), "impact")
                        for unresolved in _objects(response.get("unresolved_ranges")):
                            evidence["files"].append({"kind": "unresolved_range", **unresolved, "stage": "impact"})
                        for path in _objects(response.get("dependency_paths")):
                            evidence["relationships"].append({"kind": "dependency_path", **path, "stage": "impact"})
                        for target in _objects(response.get("test_targets")):
                            evidence["test_targets"].append(
                                {"kind": "test_target", "file_path": target.get("file_path"),
                                 "evidence": target.get("evidence", []),
                                 "evidence_kind": target.get("evidence_kind", "static_dependency"), "stage": "impact"}
                            )
                    case Playbook.INVENTORY:
                        response = await stage(
                            "inventory", "knowledge.nodes.list",
                            {"repo_path": request.repo_path,
                             "node_type": request.node_type or NodeType.FUNCTION.value,
                             "limit": request.limit, "offset": 0},
                        )
                        self._append_nodes(evidence, _objects(response.get("nodes")), "inventory")
                        raw_total = response.get("total")
                        total = raw_total if isinstance(raw_total, int) else 0
                        truncated = truncated or total > len(evidence["symbols"])
                if request.contract_paths or request.file_paths:
                    contract_started = perf_counter()
                    collection = await asyncio.to_thread(
                        collect_contracts, Path(request.repo_path), request.file_paths,
                        request.contract_paths, limit=min(request.limit, budget.hydrated_files),
                    )
                    evidence["contracts"] = collection["contracts"]
                    if collection["gaps"]:
                        evidence["contract_gaps"] = collection["gaps"]
                        degraded_stages.append("contracts")
                    truncated = truncated or collection["truncated"]
                    trace.append({"stage": "contracts", "status": "degraded" if collection["gaps"] else "ok",
                                  "truncated": collection["truncated"],
                                  "counts": {"contracts": len(collection["contracts"]), "gaps": len(collection["gaps"])},
                                  "elapsed_ms": (perf_counter() - contract_started) * 1_000})
        except TimeoutError:
            degraded_stages.append("execution")
            truncated = True
            trace.append({"stage": "execution", "status": "timeout", "counts": {},
                          "truncated": True, "elapsed_ms": (perf_counter() - started) * 1_000})

        for category, items in evidence.items():
            evidence[category] = list(
                {self._evidence_key(item): item for item in items}.values()
            )
        total_ms = (perf_counter() - started) * 1_000
        return {
            "goal": request.goal,
            "repo_path": request.repo_path,
            "investigation": investigation_coverage(evidence, trace),
            "routing": {
                "requested_mode": request.mode,
                "anchor_resolution": anchor_resolution,
                "playbook": selected.value,
                "provider": "laya" if decision.model.startswith(f"{MODEL_REPOSITORY}@") else "deterministic",
                "model": decision.model,
                "confidence": decision.confidence,
                "probabilities": {key.value: value for key, value in decision.probabilities.items()},
                "fallback_applied": degraded_reason is not None,
                "degraded_reason": degraded_reason,
            },
            "evidence": evidence,
            "trace": trace,
            "complete": not truncated and not degraded_stages and degraded_reason is None,
            "truncated": truncated,
            "degraded_stages": degraded_stages,
            "timings": {
                "classification_ms": classification_ms,
                "execution_ms": total_ms - classification_ms,
                "total_ms": total_ms,
            },
        }

    @staticmethod
    def _file_path(node: dict[str, object]) -> object | None:
        metadata = node.get("metadata")
        if node.get("file_path"):
            return node["file_path"]
        return cast(dict[str, object], metadata).get("file_path") if isinstance(metadata, dict) else None

    @staticmethod
    def _append_nodes(
        evidence: dict[str, list[dict[str, object]]],
        nodes: list[dict[str, object]],
        stage: str,
        *,
        category: str = "symbols",
    ) -> None:
        for node in nodes:
            if category == "symbols" and node.get("type") == NodeType.FILE.value:
                evidence["files"].append(
                    {"kind": "file", "id": node.get("id"),
                     "file_path": QueryOrchestrator._file_path(node), "stage": stage}
                )
                continue
            projected = _node_evidence(node, kind="reference" if category == "references" else "symbol", stage=stage)
            evidence[category].append(projected)

    @staticmethod
    def _evidence_key(item: dict[str, object]) -> tuple[str, ...]:
        return (
            str(item.get("kind", "")),
            str(item.get("id", item.get("dependent_node_id", item.get("file_path", "")))),
            str(item.get("target_id", item.get("file_path", ""))),
            str(item.get("relationship", "")),
            str(item.get("start_line", "")),
            str(item.get("end_line", "")),
            str(item.get("reason", "")),
        )
