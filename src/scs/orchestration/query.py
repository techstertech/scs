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
from scs.indexing.git_diff import collect_git_diff_ranges
from scs.orchestration.decision import (
    DecisionProvider,
    DecisionProviderKind,
    DeterministicDecisionProvider,
    Playbook,
    RoutingDecision,
    RoutingRequest,
    SourcePosition,
    eligible_playbooks,
)
from scs.orchestration.snippets import hydrate_evidence_text
from scs.source_paths import validated_source_path
from scs.services.impact import ChangedRange, is_test_path
from scs.services.contracts import collect_contracts
from scs.orchestration.coverage import AspectCoverage, investigation_coverage

QueryMode = Literal["fast", "balanced", "thorough"]
PlaybookName = Literal[
    "DISCOVER",
    "UNDERSTAND",
    "RELATIONSHIPS",
    "REFERENCES",
    "INSPECT_FILES",
    "IMPACT",
    "INVENTORY",
]
ServiceCall = Callable[[str, dict[str, object]], Awaitable[dict[str, object]]]

_LIMIT_BUCKETS = ("symbols", "files", "relationships", "references", "test_targets")
_REF_KIND_TIER: dict[str, int] = {
    "calls": 0,
    "renders": 0,
    "imports_named": 1,
    "imports_default": 1,
    "extends": 1,
    "implements": 1,
    "type_references": 1,
    "http_route": 1,
    "rpc_bridge": 1,
    "script_src": 2,
    "imports_asset": 2,
    "imports_namespace": 2,
    "reexports": 2,
    "imports_side_effect": 3,
}
_REL_TYPE_TIER: dict[str, int] = {
    RelationshipType.CALLS.value: 0,
    RelationshipType.REFERENCES.value: 1,
    RelationshipType.INHERITS.value: 1,
    RelationshipType.IMPLEMENTS.value: 1,
    RelationshipType.IMPORTS.value: 3,
}


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
    "balanced": ModeBudget(1.5, 10, 2, 3, 7.0),
    "thorough": ModeBudget(2.5, 20, 3, 5, 35.0),
}

REFERENCE_RELATIONSHIPS = (
    RelationshipType.IMPORTS,
    RelationshipType.CALLS,
    RelationshipType.REFERENCES,
    RelationshipType.INHERITS,
    RelationshipType.IMPLEMENTS,
)


REFERENCE_GOAL = re.compile(r"\b(references to|callers of|call sites|usages of|incoming references|which code references)\b", re.I)
REFERENCE_NAME = re.compile(
    r"\b(?:references to|callers of|usages of|which code references)\s+`?([A-Za-z_]\w*(?:\.\w+)*)", re.I,
)
REFERENCE_NAME_STOP_WORDS = frozenset({"a", "an", "the", "this", "that"})
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
    playbook: PlaybookName | None = None
    include_signatures: bool = False
    include_snippets: bool = False
    diff_impact: bool = False
    refresh_dirty: bool = False

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
    total_counts: dict[str, int]
    trace: list[dict[str, object]]
    complete: bool
    truncated: bool
    degraded_stages: list[str]
    timings: dict[str, float]
    investigation: dict[str, AspectCoverage]


def _specificity_tier(item: dict[str, object]) -> int:
    ref_kind = item.get("reference_kind")
    if isinstance(ref_kind, str) and ref_kind in _REF_KIND_TIER:
        return _REF_KIND_TIER[ref_kind]
    rel = item.get("relationship")
    if isinstance(rel, str) and rel in _REL_TYPE_TIER:
        return _REL_TYPE_TIER[rel]
    return 4


def _node_evidence(
    node: dict[str, object],
    *,
    kind: str,
    stage: str,
    include_signatures: bool = False,
) -> dict[str, object]:
    metadata = node.get("metadata")
    attributes = cast(dict[str, object], metadata) if isinstance(metadata, dict) else {}
    projected: dict[str, object] = {
        "kind": kind,
        "id": str(node.get("id", "")),
        "name": str(node.get("name", "")),
        "node_type": str(node.get("type", "")),
        "file_path": node.get("file_path", attributes.get("file_path")),
        "start_line": node.get("start_line", attributes.get("start_line")),
        "end_line": node.get("end_line", attributes.get("end_line")),
        "stage": stage,
    }
    if include_signatures:
        signature = node.get("signature", attributes.get("signature"))
        if isinstance(signature, str) and signature:
            projected["signature"] = signature
    return projected


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
        evidence: dict[str, list[dict[str, object]]] = {
            key: [] for key in ("symbols", "files", "relationships", "references", "test_targets")
        }
        trace: list[dict[str, object]] = []
        degraded_stages: list[str] = []
        truncated = False
        anchor_resolution: dict[str, object] = {"status": "not_needed", "candidates": []}
        depth_hints: dict[tuple[str, ...], int] = {}
        upstream_totals: dict[str, int] = {}

        if request.diff_impact:
            diff_started = perf_counter()
            diff_result = await asyncio.to_thread(
                collect_git_diff_ranges,
                Path(request.repo_path),
                timeout_seconds=min(budget.total_seconds / 2, 5.0),
                max_files=20,
                max_ranges=100,
            )
            diff_status = (
                "degraded" if diff_result.degraded_reason is not None else "ok"
            )
            if diff_result.degraded_reason is not None:
                degraded_stages.append("git_diff")
            truncated = truncated or diff_result.truncated
            trace.append(
                {
                    "stage": "git_diff",
                    "status": diff_status,
                    "counts": {
                        "ranges": len(diff_result.ranges),
                        "files": len(diff_result.file_paths),
                    },
                    "truncated": diff_result.truncated,
                    "elapsed_ms": (perf_counter() - diff_started) * 1_000,
                }
            )
            existing_range_keys = {
                (r.file_path, r.start_line, r.end_line) for r in request.changed_ranges
            }
            for changed in diff_result.ranges:
                if len(request.changed_ranges) >= 100:
                    truncated = True
                    break
                r_key = (changed.file_path, changed.start_line, changed.end_line)
                if r_key not in existing_range_keys:
                    existing_range_keys.add(r_key)
                    request.changed_ranges.append(
                        ChangedRange(
                            file_path=changed.file_path,
                            start_line=changed.start_line,
                            end_line=changed.end_line,
                        )
                    )
            for rel_path in diff_result.file_paths:
                if len(request.file_paths) >= 20:
                    truncated = True
                    break
                if rel_path not in request.file_paths:
                    request.file_paths.append(rel_path)

        degraded_reason: str | None = None
        classification_ms = 0.0
        if request.playbook is not None:
            explicit_playbook = Playbook(request.playbook)
            decision = RoutingDecision(
                playbook=explicit_playbook,
                provider=DecisionProviderKind.EXPLICIT,
                model="caller-override",
                confidence=1.0,
                probabilities={
                    playbook: float(playbook is explicit_playbook)
                    for playbook in Playbook
                },
            )
            selected = explicit_playbook
            if (
                selected is Playbook.INVENTORY
                and selected not in eligible_playbooks(request)
            ):
                selected = Playbook.DISCOVER
                degraded_reason = "ineligible_playbook"
        else:
            if self._wait_until_ready is not None:
                await self._wait_until_ready()
            classification_started = perf_counter()
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
            selected = (
                Playbook.IMPACT
                if (request.changed_ranges or request.diff_impact)
                else (intent or decision.playbook)
            )
            if (
                selected not in eligible_playbooks(request)
                and intent is None
                and not request.diff_impact
            ):
                selected = Playbook.DISCOVER
                degraded_reason = "ineligible_playbook"

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
                        "dependents_truncated", "test_targets_truncated", "matches_truncated",
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
            self._append_nodes(
                evidence,
                nodes,
                "search",
                include_signatures=request.include_signatures,
            )
            return nodes

        async def resolve(seeds: list[dict[str, object]], *, files: bool) -> bool:
            nonlocal anchor_resolution
            def collect(nodes: list[dict[str, object]]) -> dict[str, dict[str, object]]:
                candidates: dict[str, dict[str, object]] = {}
                for seed in nodes:
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
                return candidates

            candidates = collect(seeds)
            exact = {
                key: node for key, node in candidates.items()
                if (name := node.get("file_path" if files else "name"))
                and re.search(r"(?<![\w/])" + re.escape(str(name)) + r"(?![\w/])", request.goal)
            }
            requested_name = REFERENCE_NAME.search(request.goal) if not files else None
            name_hint = requested_name.group(1) if requested_name else None
            if name_hint and name_hint.casefold() not in REFERENCE_NAME_STOP_WORDS:
                exact = {key: node for key, node in candidates.items()
                         if str(node.get("name", "")).casefold() == name_hint.casefold()}
            else:
                name_hint = None
            choices = exact or candidates
            # A mixed search page can be full even when only one declaration matches.
            # Verify that exact name through the existing symbol owner, without traversal.
            discovery_available = not any(item["stage"] == "search" and (item["status"] != "ok" or item["truncated"]) for item in trace)
            full_page = len(seeds) >= min(request.limit, budget.search_seeds)
            scope = "returned_candidates"
            lookup_complete = False
            needs_lookup = (len(exact) == 1 and full_page) or (name_hint is not None and not exact)
            if discovery_available and not files and needs_lookup:
                name = name_hint or str(next(iter(exact.values()))["name"])
                lookup = await stage("resolve_lookup", "knowledge.related", {
                    "repo_path": request.repo_path, "symbol_name": name, "resolve_only": True,
                })
                scope = "exact_symbol_lookup"
                lookup_complete = trace[-1]["status"] == "ok" and not trace[-1]["truncated"]
                choices = {
                    key: node for key, node in collect(_objects(lookup.get("matches"))).items()
                    if str(node.get("name", "")).casefold() == name.casefold()
                }
                discovery_available = trace[-1]["status"] == "ok"
            unambiguous = discovery_available and len(choices) == 1 and (files or (lookup_complete if needs_lookup else not full_page))
            anchor_resolution = {
                "status": ("unavailable" if not discovery_available else
                           "resolved" if unambiguous else ("ambiguous" if choices else "missing")),
                "scope": scope,
                "candidates": [
                    _node_evidence(
                        node,
                        kind="anchor",
                        stage="resolve",
                        include_signatures=request.include_signatures,
                    )
                    for node in choices.values()
                ],
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
                # Broad exploration retains its ranked seed. It does not assert
                # that this seed uniquely identifies a symbol requested by name.
                identities = [str(seed["id"]) for seed in seeds[:1] if seed.get("id")]
            if not identities and symbol is None:
                return
            edge_types: tuple[RelationshipType | None, ...] = (
                REFERENCE_RELATIONSHIPS if incoming else (None,)
            )
            for edge_type in edge_types:
                stage_name = "references" if incoming else "traversal"
                payload: dict[str, object] = {
                    "node_id": identities[0] if identities else None,
                    "symbol_name": symbol,
                    "repo_path": request.repo_path,
                    "depth": 1 if incoming else budget.graph_depth,
                    "direction": "incoming" if incoming else "both",
                    "relationship": edge_type.value if edge_type else None,
                }
                if symbol is not None and request.file_paths:
                    payload["file_paths"] = request.file_paths
                response = await stage(
                    stage_name,
                    "knowledge.related",
                    payload,
                )
                matches = _objects(response.get("matches"))
                self._append_nodes(
                    evidence,
                    matches,
                    stage_name,
                    include_signatures=request.include_signatures,
                )
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
                        if incoming:
                            rel_file = self._file_path(node)
                            if node.get("type") == NodeType.FILE.value:
                                evidence["files"].append(
                                    {
                                        "kind": "file",
                                        "id": node.get("id"),
                                        "file_path": rel_file,
                                        "stage": stage_name,
                                    }
                                )
                            elif isinstance(rel_file, str) and rel_file:
                                evidence["files"].append(
                                    {
                                        "kind": "file",
                                        "file_path": rel_file,
                                        "stage": stage_name,
                                    }
                                )
                        else:
                            self._append_nodes(
                                evidence,
                                [node],
                                stage_name,
                                include_signatures=request.include_signatures,
                            )
                    else:
                        node = item
                    edge_start = item.get("start_line")
                    edge_end = item.get("end_line")
                    raw_node_meta = node.get("metadata")
                    node_meta = (
                        cast(dict[str, object], raw_node_meta)
                        if isinstance(raw_node_meta, dict)
                        else {}
                    )
                    is_non_file = node.get("type") != NodeType.FILE.value
                    fallback_start = (
                        node.get("start_line", node_meta.get("start_line"))
                        if is_non_file
                        else node.get("start_line")
                    )
                    fallback_end = (
                        node.get("end_line", node_meta.get("end_line"))
                        if is_non_file
                        else node.get("end_line")
                    )
                    rel_value = (
                        edge_type.value if edge_type else item.get("relationship")
                    )
                    relationship: dict[str, object] = {
                        "kind": "reference" if incoming else "relationship",
                        "id": str(item.get("id", node.get("id", ""))),
                        "seed_id": item.get("seed_id", matched_seed_id),
                        "target_id": node.get("id"),
                        "relationship": rel_value,
                        "direction": item.get("direction", "incoming" if incoming else "both"),
                        "file_path": self._file_path(node),
                        "start_line": edge_start if type(edge_start) is int else fallback_start,
                        "end_line": edge_end if type(edge_end) is int else fallback_end,
                        "stage": stage_name,
                    }
                    has_edge_meta = any(
                        item.get(key) is not None
                        for key in ("reference_kind", "enclosing_symbol", "sites", "via_barrel")
                    )
                    for key in ("reference_kind", "enclosing_symbol", "sites", "via_barrel"):
                        if item.get(key) is not None:
                            relationship[key] = item[key]
                    if has_edge_meta and node.get("type") != NodeType.FILE.value:
                        if isinstance(node.get("name"), str) and node["name"]:
                            relationship["name"] = node["name"]
                        if isinstance(node.get("type"), str) and node["type"]:
                            relationship["node_type"] = node["type"]
                    raw_depth = item.get("depth")
                    hop_depth = raw_depth if type(raw_depth) is int else 1
                    depth_hints[self._evidence_key(relationship)] = hop_depth
                    evidence["references" if incoming else "relationships"].append(relationship)

        try:
            async with asyncio.timeout(budget.total_seconds):
                if request.refresh_dirty:
                    refresh_paths = list(request.file_paths)
                    if (
                        request.source_position is not None
                        and request.source_position.file_path not in refresh_paths
                    ):
                        refresh_paths.append(request.source_position.file_path)
                    await stage(
                        "refresh",
                        "knowledge.refresh_dirty",
                        {
                            "repo_path": request.repo_path,
                            "file_paths": refresh_paths,
                            "timeout_seconds": min(budget.total_seconds / 2, 5.0),
                        },
                    )
                if selected is Playbook.IMPACT and request.diff_impact and not request.changed_ranges and not request.file_paths:
                    trace.append(
                        {
                            "stage": "impact",
                            "status": "ok",
                            "counts": {"dependents": 0, "test_targets": 0},
                            "truncated": False,
                            "elapsed_ms": 0.0,
                        }
                    )
                else:
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
                                        include_signatures=request.include_signatures,
                                    )
                                for ref_item in _objects(response.get("references")):
                                    projected_ref = _node_evidence(
                                        ref_item,
                                        kind="reference",
                                        stage="references",
                                        include_signatures=request.include_signatures,
                                    )
                                    for key in (
                                        "relationship",
                                        "reference_kind",
                                        "enclosing_symbol",
                                        "sites",
                                    ):
                                        if ref_item.get(key) is not None:
                                            projected_ref[key] = ref_item[key]
                                    evidence["references"].append(projected_ref)
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
                                self._append_nodes(
                                    evidence,
                                    _objects(response.get("nodes")),
                                    "inspect",
                                    include_signatures=request.include_signatures,
                                )
                                raw_edges = response.get("edges")
                                edge_groups = (
                                    cast(dict[str, object], raw_edges).values()
                                    if isinstance(raw_edges, dict)
                                    else ()
                                )
                                for edges in edge_groups:
                                    for edge in _objects(edges):
                                        rel_entry: dict[str, object] = {
                                            "kind": "relationship",
                                            "id": edge.get("id"),
                                            "source_id": edge.get("source_id"),
                                            "target_id": edge.get("target_id"),
                                            "relationship": edge.get("relationship"),
                                            "file_path": file_path,
                                            "stage": "inspect",
                                        }
                                        edge_meta = edge.get("metadata")
                                        if isinstance(edge_meta, dict):
                                            em = cast(dict[str, object], edge_meta)
                                            for mk in (
                                                "start_line",
                                                "end_line",
                                                "reference_kind",
                                                "enclosing_symbol",
                                                "sites",
                                            ):
                                                if em.get(mk) is not None:
                                                    rel_entry[mk] = em[mk]
                                        evidence["relationships"].append(rel_entry)
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
                            self._append_nodes(
                                evidence,
                                _objects(response.get("dependents")),
                                "impact",
                                include_signatures=request.include_signatures,
                            )
                            self._append_nodes(
                                evidence,
                                _objects(response.get("changed_symbols")),
                                "impact",
                                include_signatures=request.include_signatures,
                            )
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
                            raw_total_tests = response.get("total_test_targets")
                            if type(raw_total_tests) is int:
                                upstream_totals["test_targets"] = max(
                                    upstream_totals.get("test_targets", 0),
                                    raw_total_tests,
                                )
                            raw_total_deps = response.get("total_dependents")
                            if type(raw_total_deps) is int:
                                upstream_totals["relationships"] = max(
                                    upstream_totals.get("relationships", 0),
                                    raw_total_deps,
                                )
                        case Playbook.INVENTORY:
                            response = await stage(
                                "inventory", "knowledge.nodes.list",
                                {"repo_path": request.repo_path,
                                 "node_type": request.node_type or NodeType.FUNCTION.value,
                                 "limit": request.limit, "offset": 0},
                            )
                            self._append_nodes(
                                evidence,
                                _objects(response.get("nodes")),
                                "inventory",
                                include_signatures=request.include_signatures,
                            )
                            raw_total = response.get("total")
                            total = raw_total if isinstance(raw_total, int) else 0
                            if type(raw_total) is int:
                                upstream_totals["symbols"] = max(
                                    upstream_totals.get("symbols", 0), raw_total
                                )
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

        # Deduplicate evidence["files"] by file_path, merging repeated
        # unresolved_range hunks so one file does not crowd out the limit bucket.
        dedup_by_path: dict[str, dict[str, object]] = {}
        ranges_by_path: dict[str, list[tuple[int, int]]] = {}
        other_files: dict[tuple[str, str], dict[str, object]] = {}
        for item in evidence["files"]:
            fp = item.get("file_path")
            if not isinstance(fp, str) or not fp:
                other_files[(str(item.get("kind", "")), str(item.get("id", "")))] = item
                continue
            if item.get("kind") == "unresolved_range":
                s_line = item.get("start_line")
                e_line = item.get("end_line")
                if type(s_line) is int and type(e_line) is int:
                    ranges_by_path.setdefault(fp, []).append((s_line, e_line))
            prev = dedup_by_path.get(fp)
            if prev is None:
                dedup_by_path[fp] = dict(item)
            elif prev.get("kind") != "file" and item.get("kind") == "file":
                dedup_by_path[fp] = dict(item)
            elif prev.get("id") is None and item.get("id") is not None:
                dedup_by_path[fp] = dict(item)

        merged_files: list[dict[str, object]] = []
        for fp, entry in dedup_by_path.items():
            raw_ranges = ranges_by_path.get(fp)
            if raw_ranges:
                unique_ranges = sorted(set(raw_ranges))[:16]
                if entry.get("kind") == "unresolved_range":
                    entry["start_line"] = unique_ranges[0][0]
                    entry["end_line"] = unique_ranges[0][1]
                    if len(unique_ranges) > 1:
                        entry["ranges"] = [
                            {"start_line": s, "end_line": e} for s, e in unique_ranges
                        ]
                elif entry.get("kind") == "file":
                    entry["unresolved_ranges"] = [
                        {"start_line": s, "end_line": e} for s, e in unique_ranges
                    ]
            merged_files.append(entry)
        merged_files.extend(other_files.values())
        evidence["files"] = merged_files

        for category, items in evidence.items():
            if category == "files":
                continue
            evidence[category] = list(
                {self._evidence_key(item): item for item in items}.values()
            )

        # In REFERENCES, drop redundant coarse IMPORTS fallback when a symbol-level edge
        # already links the exact same (seed_id, file_path).
        if evidence["references"]:
            specific_pairs = {
                (str(item.get("seed_id", "")), str(item.get("file_path", "")))
                for item in evidence["references"]
                if item.get("seed_id")
                and item.get("file_path")
                and _specificity_tier(item) < 3
            }
            if specific_pairs:
                evidence["references"] = [
                    item
                    for item in evidence["references"]
                    if not (
                        item.get("relationship") == RelationshipType.IMPORTS.value
                        and (str(item.get("seed_id", "")), str(item.get("file_path", "")))
                        in specific_pairs
                    )
                ]

        # Rank references and relationships by specificity and hop distance.
        for edge_bucket in ("references", "relationships"):
            if evidence[edge_bucket]:
                evidence[edge_bucket].sort(
                    key=lambda item: (
                        _specificity_tier(item),
                        depth_hints.get(self._evidence_key(item), 1),
                        0 if type(item.get("start_line")) is int else 1,
                        str(item.get("file_path") or ""),
                        item["start_line"] if type(item.get("start_line")) is int else 0,
                        str(item.get("id") or ""),
                    )
                )

        # Populate test_targets for dependency playbooks using canonical is_test_path().
        if selected in {
            Playbook.IMPACT,
            Playbook.REFERENCES,
            Playbook.RELATIONSHIPS,
            Playbook.UNDERSTAND,
        } and selected is not Playbook.IMPACT:
            by_test_file: dict[str, list[dict[str, object]]] = {}
            for item in [*evidence["references"], *evidence["relationships"]]:
                fpath = item.get("file_path")
                if isinstance(fpath, str) and fpath and is_test_path(fpath):
                    by_test_file.setdefault(fpath, []).append(item)
            existing_paths = {
                str(t.get("file_path")) for t in evidence["test_targets"] if t.get("file_path")
            }
            for test_path, items in by_test_file.items():
                if test_path in existing_paths:
                    continue
                ev_list: list[dict[str, object]] = []
                for ev in items:
                    row: dict[str, object] = {
                        "dependent_node_id": ev.get("id"),
                        "affected_node_id": ev.get("seed_id"),
                        "relationship": ev.get("relationship"),
                    }
                    for mk in ("reference_kind", "start_line", "end_line", "enclosing_symbol"):
                        if ev.get(mk) is not None:
                            row[mk] = ev[mk]
                    ev_list.append(row)
                evidence["test_targets"].append(
                    {
                        "kind": "test_target",
                        "file_path": test_path,
                        "evidence": ev_list,
                        "evidence_kind": "static_dependency",
                        "stage": items[0].get("stage", "references"),
                    }
                )

        if evidence["test_targets"]:
            def _test_target_rank(target: dict[str, object]) -> tuple[int, str]:
                evs = _objects(target.get("evidence"))
                best = min((_specificity_tier(ev) for ev in evs), default=4)
                return (best, str(target.get("file_path") or ""))

            evidence["test_targets"].sort(key=_test_target_rank)

        total_counts: dict[str, int] = {
            category: max(len(items), upstream_totals.get(category, 0))
            for category, items in evidence.items()
        }
        for category in _LIMIT_BUCKETS:
            items = evidence[category]
            if len(items) > request.limit:
                evidence[category] = items[: request.limit]
                truncated = True

        if request.include_signatures or request.include_snippets:
            await asyncio.to_thread(
                hydrate_evidence_text,
                request.repo_path,
                evidence,
                include_signatures=request.include_signatures,
                include_snippets=request.include_snippets,
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
                "provider": decision.provider.value,
                "model": decision.model,
                "confidence": decision.confidence,
                "probabilities": {key.value: value for key, value in decision.probabilities.items()},
                "fallback_applied": degraded_reason is not None,
                "degraded_reason": degraded_reason,
            },
            "evidence": evidence,
            "total_counts": total_counts,
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
        include_signatures: bool = False,
    ) -> None:
        for node in nodes:
            if category == "symbols" and node.get("type") == NodeType.FILE.value:
                evidence["files"].append(
                    {"kind": "file", "id": node.get("id"),
                     "file_path": QueryOrchestrator._file_path(node), "stage": stage}
                )
                continue
            projected = _node_evidence(
                node,
                kind="reference" if category == "references" else "symbol",
                stage=stage,
                include_signatures=include_signatures,
            )
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
