"""Bounded changed-symbol analysis with inspectable static dependency paths."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scs.graph.models import SYMBOL_NODE_TYPES, Edge, Node, RelationshipType
from scs.graph.native import NativeGraph

MAX_REVIEW_NODES = 400
MAX_REVIEW_EDGES = 2_000
REVIEW_DEPTH = 3
DEPENDENCY_RELATIONSHIPS = frozenset({
    RelationshipType.CALLS, RelationshipType.IMPORTS, RelationshipType.REFERENCES,
    RelationshipType.INHERITS, RelationshipType.IMPLEMENTS,
})


class ChangedRange(BaseModel):
    """Inclusive zero-based lines in the indexed source revision."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    file_path: str = Field(min_length=1, max_length=2048)
    start_line: int = Field(ge=0)
    end_line: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> ChangedRange:
        if self.end_line < self.start_line:
            raise ValueError("end_line must not precede start_line")
        return self


def is_test_path(path: str) -> bool:
    """Recognize conventional test locations without matching parent substrings."""

    source = Path(path)
    return bool(
        {"test", "tests", "__tests__"}.intersection(source.parts)
        or source.name.startswith("test_")
        or any(
            marker in source.name
            for marker in ("_test.", ".test.", ".spec.", "Tests.swift", "Test.java", "Tests.java")
        )
    )


def _span(node: Node) -> tuple[int, int] | None:
    start, end = node.metadata.get("start_line"), node.metadata.get("end_line")
    if type(start) is int and type(end) is int:
        return start, end
    return None


def changed_symbol_impact(
    graph: NativeGraph, repo: str, ranges: list[ChangedRange], *, limit: int,
) -> dict[str, object]:
    """Return shortest supported dependency paths, bounded to two incoming hops.

    Containing classes are excluded when a more specific changed symbol exists.
    Output limits apply after traversal so callers cannot hide test candidates.
    The graph describes indexed source; no execution or coverage is inferred.
    """

    seeds: dict[str, Node] = {}
    unresolved: list[dict[str, object]] = []
    bounded = False
    for changed in ranges:
        ids = graph.get_node_ids_for_file_sync(repo, changed.file_path)
        bounded |= len(ids) > MAX_REVIEW_NODES
        nodes = graph.batch_get_nodes_sync(ids[:MAX_REVIEW_NODES])
        matches = [node for node in nodes if node.type in SYMBOL_NODE_TYPES
                   and (span := _span(node)) is not None
                   and span[0] <= changed.end_line and changed.start_line <= span[1]]
        specific = [node for node in matches if not any(
            other.id != node.id and (outer := _span(node)) is not None
            and (inner := _span(other)) is not None
            and outer != inner and outer[0] <= inner[0] and inner[1] <= outer[1]
            and inner[0] <= changed.start_line and changed.end_line <= inner[1]
            for other in matches
        )]
        if not specific:
            unresolved.append(changed.model_dump())
        for node in specific:
            if len(seeds) < MAX_REVIEW_NODES:
                seeds[node.id] = node
            else:
                bounded = True

    # One deterministic shortest path per node prevents cycles and duplicate work.
    paths: dict[str, list[dict[str, object]]] = {key: [] for key in sorted(seeds)}
    frontier = list(paths)
    edge_count = 0
    for hop_idx in range(REVIEW_DEPTH):
        batches = graph.batch_get_edges_sync(frontier, direction="incoming") if frontier else {}
        owner_map: dict[str, str] = {}
        owner_batches: dict[str, list[Edge]] = {}
        if hop_idx >= 1 and frontier:
            frontier_nodes = graph.batch_get_nodes_sync(frontier)
            non_file_ids = {
                node.id
                for node in frontier_nodes
                if node.type.value != "file"
            }
            for target in frontier:
                if target not in non_file_ids:
                    continue
                for edge in batches.get(target, []):
                    if (
                        edge.relationship is RelationshipType.CONTAINS
                        and edge.target_id == target
                        and edge.source_id != target
                    ):
                        owner_map[target] = edge.source_id
                        break
            if owner_map:
                owner_batches = graph.batch_get_edges_sync(
                    sorted(set(owner_map.values())), direction="incoming"
                )
        following: list[str] = []
        for target in frontier:
            candidate_edges = list(batches.get(target, []))
            owner_fid = owner_map.get(target)
            if owner_fid is not None:
                for owner_edge in owner_batches.get(owner_fid, []):
                    if owner_edge not in candidate_edges:
                        candidate_edges.append(owner_edge)
            for edge in sorted(candidate_edges, key=lambda item: (item.source_id, item.relationship)):
                expected_target = target if edge.target_id == target else owner_fid
                if edge.relationship not in DEPENDENCY_RELATIONSHIPS or edge.target_id != expected_target:
                    continue
                edge_count += 1
                if edge_count > MAX_REVIEW_EDGES or len(paths) >= MAX_REVIEW_NODES:
                    bounded = True
                    break
                if edge.source_id in paths or edge.source_id == expected_target:
                    continue
                step: dict[str, object] = {
                    "dependent_node_id": edge.source_id, "affected_node_id": expected_target,
                    "relationship": edge.relationship.value,
                }
                for meta_key in (
                    "start_line",
                    "end_line",
                    "reference_kind",
                    "enclosing_symbol",
                    "via_barrel",
                ):
                    if meta_key in edge.metadata and edge.metadata[meta_key] is not None:
                        step[meta_key] = edge.metadata[meta_key]
                paths[edge.source_id] = [*paths[target], step]
                following.append(edge.source_id)
        frontier = following

    def _path_rank(node_id: str) -> tuple[int, int, str]:
        chain = paths[node_id]
        has_imports_only = all(
            step.get("relationship") == RelationshipType.IMPORTS.value
            for step in chain
        )
        return (len(chain), 1 if has_imports_only else 0, node_id)

    dependent_ids = sorted(set(paths) - set(seeds), key=_path_rank)
    dependents = sorted(
        graph.batch_get_nodes_sync(dependent_ids[:MAX_REVIEW_NODES]),
        key=lambda node: _path_rank(node.id),
    )
    dep_files = {
        node.id: str(node.metadata.get("file_path", "")) for node in dependents
    }
    for chain in paths.values():
        for step in chain:
            dep_id = step.get("dependent_node_id")
            if isinstance(dep_id, str) and dep_id in dep_files and dep_files[dep_id]:
                step.setdefault("file_path", dep_files[dep_id])
    test_nodes = [node for node in dependents if is_test_path(str(node.metadata.get("file_path", "")))]
    test_paths = sorted({str(node.metadata["file_path"]) for node in test_nodes})
    targets = [{
        "file_path": path, "evidence_kind": "static_dependency",
        "evidence": [{"dependent_node_id": node.id, "dependency_path": paths[node.id]}
                     for node in test_nodes if node.metadata.get("file_path") == path],
    } for path in test_paths[:limit]]
    dependents_truncated = len(dependent_ids) > limit or bounded
    tests_truncated = len(test_paths) > limit or bounded
    return {
        "affected_node_ids": sorted(seeds),
        "changed_symbols": [seeds[key].model_dump(mode="json") for key in sorted(seeds)],
        "dependents": [node.model_dump(mode="json") for node in dependents[:limit]],
        "total_dependents": len(dependent_ids),
        "test_targets": targets,
        "total_test_targets": len(test_paths),
        "dependency_paths": [{"dependent_node_id": key, "path": paths[key]} for key in dependent_ids[:limit]],
        "unresolved_ranges": unresolved,
        "dependents_truncated": dependents_truncated,
        "test_targets_truncated": tests_truncated,
        "analysis_depth": REVIEW_DEPTH,
        "complete": not (dependents_truncated or tests_truncated or unresolved),
        "degraded_reason": "unresolved_ranges" if unresolved else None,
    }
