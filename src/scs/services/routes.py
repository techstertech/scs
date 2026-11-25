"""Transport-neutral implementations of SCS's public code-intelligence routes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
import hashlib
from heapq import nsmallest
from pathlib import Path
from time import perf_counter
from typing import cast

from scs.graph.models import SYMBOL_NODE_TYPES, Edge, Node, NodeType, RelationshipType
from scs.graph.native import NativeGraph
from scs.indexing.git_diff import collect_git_dirty_paths
from scs.indexing.jobs import (
    ACTIVE_JOB_STATUSES,
    IngestionJob,
    IngestionJobStore,
    job_to_dict,
)
from scs.indexing.repository_paths import canonicalize_repo_path
from scs.indexing.search import (
    MAX_SEARCH_QUERIES,
    CodeSearchMatch,
    CodeSearchResponse,
    CodeSearchService,
    SearchMode,
)
from scs.indexing.symbol_edges import STRUCTURAL_EDGE_SCHEMA_VERSION
from scs.providers.base import EmbeddingProvider, RerankingProvider
from scs.source_paths import validated_source_path
from scs.services.impact import (
    MAX_REVIEW_EDGES,
    MAX_REVIEW_NODES,
    REVIEW_DEPTH,
    ChangedRange,
    changed_symbol_impact,
    is_test_path,
)

GraphForRepository = Callable[[str], NativeGraph | None]
BindingForRepository = Callable[[str], tuple[str, str] | None]

DEPENDENCY_RELATIONSHIPS = frozenset(
    {
        RelationshipType.CALLS,
        RelationshipType.IMPORTS,
        RelationshipType.INHERITS,
        RelationshipType.IMPLEMENTS,
        RelationshipType.REFERENCES,
    }
)
DEFAULT_INSPECT_NODE_LIMIT = 50
DEFAULT_INSPECT_EDGE_LIMIT = 100
MAX_INSPECT_NODE_LIMIT = 200
MAX_INSPECT_EDGE_LIMIT = 500
INSPECT_READ_BATCH_SIZE = 128
UNKNOWN_SOURCE_LINE = 2**63 - 1
COMPACT_CONTENT_CHARACTERS = 1_024
MAX_JOB_WAIT_SECONDS = 10.0
JOB_WAIT_POLL_SECONDS = 0.1
ACTIVE_JOB_RETRY_AFTER_MS = 250
TRAVERSAL_DIRECTIONS = frozenset({"outgoing", "incoming", "both"})
TRAVERSAL_DIRECTION_ALIASES = {
    "dependencies": "outgoing",
    "dependents": "incoming",
}
DEFAULT_DEPENDENT_LIMIT = 200
DEFAULT_TEST_TARGET_LIMIT = 50
MAX_REGRESSION_RISK_LIMIT = 1_000
SYMBOL_RESOLUTION_LIMIT = 20


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1_000


def _search_diagnostics(response: CodeSearchResponse) -> dict[str, object]:
    """Serialize content-free search execution facts and query evidence."""

    timings = response.timings
    return {
        "queries": list(response.queries),
        "query_matches": {
            match.node.id: list(match.matched_query_indexes)
            for match in response.matches
        },
        "semantic_available": response.semantic_available,
        "reranker_applied": response.reranker_applied,
        "degraded_stage": response.degraded_stage,
        "timed_out": response.timed_out,
        "degraded_reason": response.degraded_reason,
        "timings": {
            "lexical_ms": timings.lexical_ms,
            "embedding_ms": timings.embedding_ms,
            "vector_ms": timings.vector_ms,
            "rerank_ms": timings.rerank_ms,
            "total_ms": timings.total_ms,
        },
    }


def _node_dict(node: Node) -> dict[str, object]:
    return node.model_dump(mode="json")


def _compact_match_dict(match: CodeSearchMatch) -> dict[str, object]:
    """Project one result onto the bounded fields code consumers need most."""

    node = match.node
    metadata = node.metadata
    return {
        "id": node.id,
        "type": node.type.value,
        "name": node.name,
        "qualified_name": metadata.get("qualified_name"),
        "file_path": metadata.get("file_path"),
        "start_line": metadata.get("start_line"),
        "end_line": metadata.get("end_line"),
        "signature": metadata.get("signature"),
        "content": node.content[:COMPACT_CONTENT_CHARACTERS],
        "distance": match.semantic_distance,
    }


def _edge_dict(edge: Edge) -> dict[str, object]:
    return edge.model_dump(mode="json")


def _node_type(value: object) -> NodeType | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("node_type must be a string")
    try:
        return NodeType(value)
    except ValueError as error:
        raise ValueError(f"unsupported node_type: {value}") from error


def _integer(
    params: dict[str, object], key: str, default: int, *, minimum: int = 0
) -> int:
    value = params.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return max(minimum, value)


def _bounded_seconds(params: dict[str, object], key: str) -> float:
    value = params.get(key, 0.0)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return min(float(value), MAX_JOB_WAIT_SECONDS)


def _job_summary(job: IngestionJob | None) -> dict[str, object] | None:
    """Project safe-to-share durable progress without stored job internals."""

    if job is None:
        return None
    return {
        "id": job.id,
        "mode": job.mode,
        "status": job.status,
        "phase": job.phase,
        "current": job.current,
        "total": job.total,
        "message": job.message,
        "attempts": job.attempts,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "finished_at": job.finished_at,
    }


def _string(
    params: dict[str, object], key: str, *, required: bool = False
) -> str | None:
    value = params.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _string_list(
    params: dict[str, object], key: str, *, default: bool = True
) -> list[str]:
    """Validate an untrusted route parameter as a concrete string list."""

    value = params.get(key)
    if default and value is None:
        value = []
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list of strings")
    items = cast(list[object], value)
    if not all(isinstance(item, str) for item in items):
        raise ValueError(f"{key} must be a list of strings")
    return [item for item in items if isinstance(item, str)]


def _dict_list(value: object, *, key: str) -> list[dict[str, object]]:
    """Narrow a service response field before downstream composition."""

    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list of objects")
    items = cast(list[object], value)
    if not all(isinstance(item, dict) for item in items):
        raise TypeError(f"{key} must be a list of objects")
    return [cast(dict[str, object], item) for item in items]


def _search_queries(params: dict[str, object], primary: str) -> list[str]:
    queries = _string_list(params, "queries")
    requested = [primary, *queries]
    if len(requested) > MAX_SEARCH_QUERIES:
        raise ValueError(f"search accepts at most {MAX_SEARCH_QUERIES} queries")
    if any(not query.strip() for query in requested):
        raise ValueError("search queries cannot be blank")
    return queries


def _search_mode(params: dict[str, object]) -> SearchMode:
    value = _string(params, "search_mode") or "thorough"
    if value not in {"fast", "balanced", "thorough"}:
        raise ValueError("search_mode must be fast, balanced, or thorough")
    return cast(SearchMode, value)


def _traversal_direction(
    params: dict[str, object],
    *,
    key: str = "direction",
    default: str,
) -> str:
    """Normalize dependency vocabulary onto graph traversal edges."""

    value = _string(params, key) or default
    value = TRAVERSAL_DIRECTION_ALIASES.get(value, value)
    if value not in TRAVERSAL_DIRECTIONS:
        raise ValueError(
            "direction must be outgoing, incoming, both, dependencies, or dependents"
        )
    return value


def _integer_metadata(value: object, *, key: str, default: int) -> int:
    """Convert graph metadata that originated outside Python's type boundary."""

    if value is None:
        return default
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        raise TypeError(f"{key} must be numeric")
    return int(value)


def _inspection_rank(node: Node) -> tuple[int, int, str, str]:
    """Prefer declarations, then source order, instead of opaque ID order."""

    if node.type in SYMBOL_NODE_TYPES:
        category = 0
    elif node.type in {NodeType.FILE, NodeType.MODULE}:
        category = 1
    else:
        category = 2

    line = node.metadata.get("start_line")
    source_line = line if isinstance(line, int) and line >= 0 else UNKNOWN_SOURCE_LINE

    return category, source_line, node.name.casefold(), node.id


def _inspection_nodes(graph: NativeGraph, node_ids: list[str], limit: int) -> list[Node]:
    """Rank all candidates while retaining only a batch and the capped result."""

    def candidates() -> Iterator[Node]:
        for offset in range(0, len(node_ids), INSPECT_READ_BATCH_SIZE):
            batch = node_ids[offset : offset + INSPECT_READ_BATCH_SIZE]

            yield from graph.batch_get_nodes_sync(batch)

    return nsmallest(limit, candidates(), key=_inspection_rank)


class SCSServiceRoutes:
    """Implement every finite method consumed by the SCS MCP gateway."""

    def __init__(
        self,
        *,
        graph: Callable[[], NativeGraph],
        jobs: Callable[[], IngestionJobStore],
        embeddings: Callable[[], EmbeddingProvider],
        reranker: Callable[[], RerankingProvider | None] | None = None,
        graph_for_repository: GraphForRepository | None = None,
        binding_for_repository: BindingForRepository | None = None,
        on_job_enqueued: Callable[[], None] | None = None,
    ) -> None:
        self._graph: Callable[[], NativeGraph] = graph
        self._jobs: Callable[[], IngestionJobStore] = jobs
        self._embeddings: Callable[[], EmbeddingProvider] = embeddings
        self._reranker: Callable[[], RerankingProvider | None] = (
            reranker if reranker is not None else lambda: None
        )
        self._graph_for_repository: GraphForRepository | None = graph_for_repository
        self._binding_for_repository: BindingForRepository | None = (
            binding_for_repository
        )
        self._on_job_enqueued: Callable[[], None] | None = on_job_enqueued

    def _read_graph(self, repo_path: object) -> NativeGraph | None:
        """Resolve an existing project graph without registering a repository."""

        if repo_path is None or self._graph_for_repository is None:
            try:
                return self._graph()
            except RuntimeError:
                return None
        if not isinstance(repo_path, str):
            raise ValueError("repo_path must be a string")
        return self._graph_for_repository(canonicalize_repo_path(repo_path))

    def _repo_id(self, repo_path: object) -> int | None:
        if repo_path is None:
            return None
        if not isinstance(repo_path, str):
            raise ValueError("repo_path must be a string")
        graph = self._read_graph(repo_path)
        if graph is None:
            return None
        return graph.resolve_repo_id_sync(canonicalize_repo_path(repo_path))

    async def search(self, params: dict[str, object]) -> dict[str, object]:
        query = _string(params, "query", required=True)
        assert query is not None
        limit = min(200, _integer(params, "limit", 10, minimum=1))
        result_detail = _string(params, "result_detail") or "full"
        if result_detail not in {"full", "compact"}:
            raise ValueError("result_detail must be full or compact")
        node_type = _node_type(params.get("node_type"))
        queries = _search_queries(params, query)
        search_mode = _search_mode(params)
        repo_path = params.get("repo_path")
        repo_id = self._repo_id(repo_path)
        if repo_path is not None and repo_id is None:
            return {
                "query": query,
                "results": [],
                "neighbors": [],
                "total": 0,
                "retrieval_mode": "none",
                **self._empty_search_diagnostics(query, queries),
            }
        graph = self._read_graph(repo_path)
        if graph is None:
            return {
                "query": query,
                "results": [],
                "neighbors": [],
                "total": 0,
                "retrieval_mode": "none",
                **self._empty_search_diagnostics(query, queries),
            }
        response = await CodeSearchService(
            graph,
            self._embeddings(),
            self._reranker(),
        ).search(
            query,
            node_type=node_type,
            limit=limit,
            repo_id=repo_id,
            queries=queries,
            search_mode=search_mode,
        )
        results = (
            [_compact_match_dict(match) for match in response.matches]
            if result_detail == "compact"
            else [
                {**_node_dict(match.node), "distance": match.semantic_distance}
                for match in response.matches
            ]
        )
        neighbors: list[dict[str, object]] = []
        if bool(params.get("include_neighbors")):
            seen = {str(item["id"]) for item in results}
            for item in results:
                adjacent = await asyncio.to_thread(
                    graph.get_neighbors_sync, str(item["id"]), limit=20
                )
                for node in adjacent:
                    if node.id not in seen:
                        seen.add(node.id)
                        neighbors.append(_node_dict(node))
        return {
            "query": query,
            "results": results,
            "neighbors": neighbors,
            "total": len(results),
            "retrieval_mode": response.retrieval_mode,
            **_search_diagnostics(response),
        }

    @staticmethod
    def _empty_search_diagnostics(
        query: str, queries: list[str]
    ) -> dict[str, object]:
        effective_queries = list(dict.fromkeys((query, *queries)))
        return {
            "queries": effective_queries,
            "query_matches": {},
            "semantic_available": False,
            "reranker_applied": False,
            "degraded_stage": "semantic",
            "timed_out": False,
            "degraded_reason": "repository is not indexed",
            "timings": {
                "lexical_ms": 0.0,
                "embedding_ms": 0.0,
                "vector_ms": 0.0,
                "rerank_ms": 0.0,
                "total_ms": 0.0,
            },
        }

    async def nodes_list(self, params: dict[str, object]) -> dict[str, object]:
        node_type = _node_type(params.get("node_type", NodeType.FUNCTION.value))
        if node_type not in SYMBOL_NODE_TYPES:
            raise ValueError("node_type must identify a code symbol")
        limit = min(200, _integer(params, "limit", 50, minimum=1))
        offset = _integer(params, "offset", 0)
        repo_path = params.get("repo_path")
        graph = self._read_graph(repo_path)
        if graph is None:
            return {"nodes": [], "total": 0, "limit": limit, "offset": offset}
        repo_id = self._repo_id(repo_path)
        if repo_path is not None and repo_id is None:
            return {"nodes": [], "total": 0, "limit": limit, "offset": offset}
        nodes, total = await asyncio.gather(
            asyncio.to_thread(
                graph.list_nodes_sync,
                node_type=node_type,
                limit=limit,
                offset=offset,
                repo_id=repo_id,
            ),
            asyncio.to_thread(
                graph.count_nodes_sync, node_type=node_type, repo_id=repo_id
            ),
        )
        return {
            "nodes": [_node_dict(node) for node in nodes],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    async def stats(self, params: dict[str, object]) -> dict[str, object]:
        repo_path = _string(params, "repo_path")
        wait_job_id = _string(params, "wait_job_id")
        wait_timeout = _bounded_seconds(params, "wait_timeout_seconds")
        if wait_job_id is not None and repo_path is None:
            raise ValueError("repo_path is required when wait_job_id is provided")

        canonical_repo = canonicalize_repo_path(repo_path) if repo_path else None
        job_state = (
            await asyncio.to_thread(
                self._jobs().repository_job_state,
                canonical_repo,
            )
            if canonical_repo is not None
            else None
        )
        wait = await self._wait_for_job(
            repo_path=canonical_repo,
            job_id=wait_job_id,
            timeout_seconds=wait_timeout,
        )
        job_fields = {
            "active_job": _job_summary(job_state.active) if job_state else None,
            "latest_job": _job_summary(job_state.latest) if job_state else None,
            "retry_after_ms": (
                ACTIVE_JOB_RETRY_AFTER_MS
                if job_state is not None and job_state.active is not None
                else None
            ),
            "wait": wait,
        }
        graph = self._read_graph(repo_path)
        if graph is None:
            return {
                "repo_path": canonical_repo,
                "status": "indexing"
                if job_state is not None and job_state.active is not None
                else "empty",
                "total_nodes": 0,
                "nodes_by_type": {},
                "embedding_count": 0,
                "vector_index_count": 0,
                "vector_index_scope": "project",
                "ingestion_stats": {},
                "database_size_bytes": 0,
                "vector_available": False,
                "vector_unavailable_reason": "repository is not indexed",
                "structural_search_ready": False,
                "semantic_search_ready": False,
                "semantic_search_unavailable_reason": "repository is not indexed",
                **job_fields,
            }
        repo_id = self._repo_id(repo_path)
        if repo_path is not None and repo_id is None:
            counts: dict[str, int] = {}
            without_embeddings = 0
            vector_index_count, all_ingestion = await asyncio.gather(
                asyncio.to_thread(graph.count_embeddings_sync),
                asyncio.to_thread(graph.get_ingestion_stats_sync),
            )
        else:
            (
                counts,
                without_embeddings,
                vector_index_count,
                all_ingestion,
            ) = await asyncio.gather(
                asyncio.to_thread(graph.count_nodes_by_type_sync, repo_id),
                asyncio.to_thread(
                    graph.count_nodes_without_embeddings_sync, repo_id=repo_id
                ),
                asyncio.to_thread(graph.count_embeddings_sync),
                asyncio.to_thread(graph.get_ingestion_stats_sync),
            )
        total_nodes = sum(counts.values())
        embedding_count = total_nodes - without_embeddings
        provider_metadata = self._embeddings().metadata
        semantic_search_ready = (
            provider_metadata.available
            and graph.vector_state.available
            and embedding_count > 0
        )
        if semantic_search_ready:
            semantic_search_unavailable_reason: str | None = None
        elif not provider_metadata.available:
            semantic_search_unavailable_reason = provider_metadata.reason
        elif not graph.vector_state.available:
            semantic_search_unavailable_reason = graph.vector_state.reason
        else:
            semantic_search_unavailable_reason = (
                "no indexed embeddings are available for this scope"
            )
        ingestion = (
            {canonical_repo: all_ingestion.get(canonical_repo, {})}
            if canonical_repo is not None
            else all_ingestion
        )
        return {
            "repo_path": canonical_repo,
            "status": "ready" if total_nodes else "empty",
            "total_nodes": total_nodes,
            "nodes_by_type": counts,
            "embedding_count": embedding_count,
            "vector_index_count": vector_index_count,
            # A NativeGraph is opened from exactly one catalog generation, so its
            # vector count cannot span repositories.  Retaining the former
            # "global" label would invite clients to merge unrelated stores.
            "vector_index_scope": "project",
            "ingestion_stats": ingestion,
            "database_size_bytes": graph.database_path.stat().st_size
            if graph.database_path.exists()
            else 0,
            "vector_available": graph.vector_state.available,
            "vector_unavailable_reason": graph.vector_state.reason,
            "structural_search_ready": total_nodes > 0,
            "semantic_search_ready": semantic_search_ready,
            "semantic_search_unavailable_reason": semantic_search_unavailable_reason,
            **job_fields,
        }

    async def _wait_for_job(
        self,
        *,
        repo_path: str | None,
        job_id: str | None,
        timeout_seconds: float,
    ) -> dict[str, object] | None:
        """Observe one scoped durable job without changing queue state."""

        if job_id is None:
            return None
        assert repo_path is not None
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            job = await asyncio.to_thread(self._jobs().get, job_id)
            if job is None or job.repo_path != repo_path:
                return {"outcome": "not_found", "job": None}
            if job.status not in ACTIVE_JOB_STATUSES:
                return {"outcome": "terminal", "job": _job_summary(job)}
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return {"outcome": "timeout", "job": _job_summary(job)}
            await asyncio.sleep(min(JOB_WAIT_POLL_SECONDS, remaining))

    @staticmethod
    def _containing_file_id_sync(graph: NativeGraph, node: Node) -> str | None:
        """Walk incoming CONTAINS edges from a symbol up to its enclosing file node."""

        if node.type is NodeType.FILE:
            return None
        current_id = node.id
        visited: set[str] = {current_id}
        for _ in range(6):
            incoming = graph.get_edges_sync(
                current_id,
                relationship=RelationshipType.CONTAINS.value,
                direction="incoming",
            )
            parent_id = next(
                (
                    edge.source_id
                    for edge in incoming
                    if edge.target_id == current_id and edge.source_id not in visited
                ),
                None,
            )
            if parent_id is None:
                return None
            visited.add(parent_id)
            parent = graph.get_node_sync(parent_id)
            if parent is None:
                return None
            if parent.type is NodeType.FILE:
                return parent.id
            current_id = parent.id
        return None

    @staticmethod
    def _related_item_node_id(item: dict[str, object]) -> str | None:
        raw_node = item.get("node")
        if isinstance(raw_node, dict):
            raw_id = cast(dict[str, object], raw_node).get("id")
            return str(raw_id) if isinstance(raw_id, str) and raw_id else None
        raw_id = item.get("id")
        return str(raw_id) if isinstance(raw_id, str) and raw_id else None

    async def related(self, params: dict[str, object]) -> dict[str, object]:
        resolve_only = params.get("resolve_only", False)
        if type(resolve_only) is not bool:
            raise ValueError("resolve_only must be a boolean")
        matches_truncated = False
        symbol = _string(params, "symbol_name")
        node_id = _string(params, "node_id")
        if (symbol is None) == (node_id is None):
            raise ValueError("exactly one of symbol_name or node_id is required")
        allowed_files = set(_string_list(params, "file_paths"))
        repo_path = params.get("repo_path")
        graph = self._read_graph(repo_path)
        if graph is None:
            return {"symbol_name": symbol, "node_id": node_id, "matches": [], "related": [], "available": False}
        repo_id = self._repo_id(repo_path)
        direction = _traversal_direction(params, default="outgoing")
        if repo_path is not None and repo_id is None:
            matches = []
        elif node_id is not None:
            node = await asyncio.to_thread(graph.get_node_sync, node_id)
            matches: list[Node] = [
                node
                for node in [node]
                if node is not None and (repo_path is None or node.repo_id == repo_id)
            ]
        else:
            assert symbol is not None
            matches_by_type = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        graph.search_by_name_sync,
                        symbol,
                        node_type=node_type,
                        limit=SYMBOL_RESOLUTION_LIMIT + int(resolve_only),
                        repo_id=repo_id,
                    )
                    for node_type in SYMBOL_NODE_TYPES
                )
            )
            matches = sorted(
                {
                    node.id: node
                    for nodes in matches_by_type
                    for node in nodes
                    if not allowed_files
                    or node.metadata.get("file_path") in allowed_files
                }.values(),
                key=lambda node: (node.name.casefold() != symbol.casefold(), node.name, node.id),
            )
            # An exact declaration is the deterministic answer to a symbol lookup.
            # Retaining substring matches only when no exact symbol exists preserves
            # discovery without letting a similarly named test/helper dilute lookup.
            exact_matches = [
                node for node in matches if node.name.casefold() == symbol.casefold()
            ]
            selected_matches = exact_matches or matches
            matches_truncated = len(selected_matches) > SYMBOL_RESOLUTION_LIMIT or (
                resolve_only and any(len(nodes) > SYMBOL_RESOLUTION_LIMIT for nodes in matches_by_type)
            )
            matches = selected_matches[:SYMBOL_RESOLUTION_LIMIT]
        if resolve_only:
            return {
                "symbol_name": symbol, "node_id": node_id,
                "matches": [_node_dict(node) for node in matches], "related": [],
                "matches_truncated": matches_truncated,
            }
        if not matches:
            return {
                "symbol_name": symbol,
                "node_id": node_id,
                "matches": [],
                "related": [],
            }
        depth = min(3, _integer(params, "depth", 2, minimum=1))
        relationship = _string(params, "relationship")
        related: list[dict[str, object]] = []
        directions = ("outgoing", "incoming") if direction == "both" else (direction,)

        seed_ids = [match.id for match in matches]
        incident_by_seed: dict[str, list[Edge]] = await asyncio.to_thread(
            graph.batch_get_edges_sync, seed_ids, direction=direction
        )

        owner_file_ids: dict[str, str] = {}
        owner_files: dict[str, Node | None] = {}
        owner_incoming_imports: dict[str, dict[str, Edge]] = {}
        if "incoming" in directions and relationship == RelationshipType.IMPORTS.value:
            for match in matches:
                if match.type is not NodeType.FILE:
                    owner_fid = await asyncio.to_thread(
                        self._containing_file_id_sync, graph, match
                    )
                    if owner_fid is not None:
                        owner_file_ids[match.id] = owner_fid
            if owner_file_ids:
                unique_owner_ids = list(dict.fromkeys(owner_file_ids.values()))
                owner_nodes = await asyncio.to_thread(
                    graph.batch_get_nodes_sync, unique_owner_ids
                )
                for owner_node in owner_nodes:
                    owner_files[owner_node.id] = owner_node
                owner_edges = await asyncio.to_thread(
                    graph.batch_get_edges_sync, unique_owner_ids, direction="incoming"
                )
                for owner_fid, edge_list in owner_edges.items():
                    by_source: dict[str, Edge] = {}
                    for edge in edge_list:
                        if (
                            edge.relationship == RelationshipType.IMPORTS.value
                            and edge.target_id == owner_fid
                        ):
                            by_source[edge.source_id] = edge
                    owner_incoming_imports[owner_fid] = by_source

        def _best_direct_edge(
            seed_id: str, target_node_id: str, active_dir: str
        ) -> Edge | None:
            candidates: list[Edge] = []
            for edge in incident_by_seed.get(seed_id, []):
                if relationship is not None and edge.relationship != relationship:
                    continue
                if active_dir == "incoming":
                    matches_end = edge.target_id == seed_id and edge.source_id == target_node_id
                else:
                    matches_end = edge.source_id == seed_id and edge.target_id == target_node_id
                if matches_end:
                    candidates.append(edge)
            if not candidates:
                return None
            priority = {
                RelationshipType.CALLS.value: 0,
                RelationshipType.REFERENCES.value: 1,
                RelationshipType.INHERITS.value: 2,
                RelationshipType.IMPLEMENTS.value: 3,
                RelationshipType.IMPORTS.value: 4,
            }
            return min(
                candidates,
                key=lambda e: (priority.get(e.relationship, 10), e.relationship, e.id),
            )

        def _attach_edge_metadata(
            item: dict[str, object], edge: Edge | None
        ) -> dict[str, object]:
            if edge is None:
                return item
            enriched = dict(item)
            enriched.setdefault("relationship", edge.relationship)
            for key in (
                "start_line",
                "end_line",
                "reference_kind",
                "enclosing_symbol",
                "sites",
                "via_barrel",
            ):
                val = edge.metadata.get(key)
                if val is not None:
                    enriched[key] = val
            return enriched

        for match in matches:
            for active_direction in directions:
                traversal = await asyncio.to_thread(
                    graph.traverse_sync,
                    match.id,
                    depth=depth,
                    relationship=relationship,
                    direction=active_direction,
                )
                seen_node_ids: set[str] = {
                    item_id
                    for item in traversal
                    if (item_id := self._related_item_node_id(item)) is not None
                }
                for item in traversal:
                    enriched_item = {
                        **item,
                        "direction": active_direction,
                        "seed_id": match.id,
                    }
                    item_id = self._related_item_node_id(item)
                    if item.get("depth") == 1 and item_id is not None:
                        direct_edge = _best_direct_edge(
                            match.id, item_id, active_direction
                        )
                        enriched_item = _attach_edge_metadata(
                            enriched_item, direct_edge
                        )
                    related.append(enriched_item)
                if (
                    active_direction == "incoming"
                    and relationship == RelationshipType.IMPORTS.value
                    and match.type is not NodeType.FILE
                ):
                    owner_file_id = owner_file_ids.get(match.id)
                    if owner_file_id is not None:
                        owner_node = owner_files.get(owner_file_id)
                        owner_default = (
                            owner_node.metadata.get("default_export")
                            if owner_node is not None
                            else None
                        )
                        imports_by_source = owner_incoming_imports.get(
                            owner_file_id, {}
                        )
                        file_traversal = await asyncio.to_thread(
                            graph.traverse_sync,
                            owner_file_id,
                            depth=depth,
                            relationship=relationship,
                            direction="incoming",
                        )
                        for item in file_traversal:
                            raw_depth = item.get("depth", 0)
                            if (
                                isinstance(raw_depth, bool)
                                or not isinstance(raw_depth, int)
                                or raw_depth <= 0
                            ):
                                continue
                            item_id = self._related_item_node_id(item)
                            if (
                                item_id is None
                                or item_id == match.id
                                or item_id in seen_node_ids
                            ):
                                continue
                            import_edge = (
                                imports_by_source.get(item_id) if raw_depth == 1 else None
                            )
                            if import_edge is not None:
                                raw_syms = import_edge.metadata.get("imported_symbols")
                                wildcard = bool(
                                    import_edge.metadata.get("wildcard_import", False)
                                )
                                if (
                                    isinstance(raw_syms, list)
                                    and raw_syms
                                    and not wildcard
                                ):
                                    imported_names = {
                                        str(sym) for sym in cast(list[object], raw_syms)
                                    }
                                    matches_default = (
                                        "default" in imported_names
                                        and owner_default == match.name
                                    )
                                    if (
                                        match.name not in imported_names
                                        and not matches_default
                                    ):
                                        continue
                            seen_node_ids.add(item_id)
                            enriched_fallback = {
                                **item,
                                "direction": active_direction,
                                "seed_id": match.id,
                            }
                            if import_edge is not None:
                                enriched_fallback = _attach_edge_metadata(
                                    enriched_fallback, import_edge
                                )
                            related.append(enriched_fallback)
        return {
            "symbol_name": symbol,
            "node_id": node_id,
            "matches": [_node_dict(node) for node in matches],
            "related": related,
        }

    async def graph_context(self, params: dict[str, object]) -> dict[str, object]:
        search_params = {
            **params,
            "limit": params.get("vector_limit", 5),
            "include_neighbors": False,
        }
        seeds = await self.search(search_params)
        direction = _traversal_direction(params, default="both")
        graph = self._read_graph(params.get("repo_path"))
        if graph is None:
            return {
                "query": seeds["query"],
                "direction": direction,
                "seeds": [],
                "context": [],
                "search": {
                    key: seeds[key]
                    for key in (
                        "queries",
                        "query_matches",
                        "semantic_available",
                        "reranker_applied",
                        "degraded_stage",
                        "timed_out",
                        "degraded_reason",
                        "timings",
                    )
                },
            }
        directions = ("outgoing", "incoming") if direction == "both" else (direction,)
        context: list[dict[str, object]] = []
        seen: set[str] = set()
        seed_results = _dict_list(seeds.get("results"), key="results")
        for seed in seed_results:
            for active_direction in directions:
                traversal = await asyncio.to_thread(
                    graph.traverse_sync,
                    str(seed["id"]),
                    depth=min(3, _integer(params, "hop_limit", 2, minimum=1)),
                    direction=active_direction,
                )
                for item in traversal:
                    raw_node = item.get("node", item)
                    node = (
                        cast(dict[str, object], raw_node)
                        if isinstance(raw_node, dict)
                        else None
                    )
                    key = str(node.get("id", "")) if node is not None else repr(item)
                    if key not in seen:
                        seen.add(key)
                        context.append({**item, "direction": active_direction})
        return {
            "query": seeds["query"],
            "direction": direction,
            "seeds": seed_results,
            "context": context,
            "search": {
                key: seeds[key]
                for key in (
                    "queries",
                    "query_matches",
                    "semantic_available",
                    "reranker_applied",
                    "degraded_stage",
                    "timed_out",
                    "degraded_reason",
                    "timings",
                )
            },
        }

    async def inspect_file(self, params: dict[str, object]) -> dict[str, object]:
        repo_path = _string(params, "repo_path", required=True)
        file_path = _string(params, "file_path", required=True)
        assert repo_path is not None and file_path is not None
        graph = self._read_graph(repo_path)
        if graph is None:
            return {
                "repo_path": canonicalize_repo_path(repo_path),
                "file_path": Path(file_path).as_posix(),
                "nodes": [],
                "edges": {},
                "nodes_truncated": False,
                "edges_truncated": False,
            }
        node_ids = await asyncio.to_thread(
            graph.get_node_ids_for_file_sync,
            canonicalize_repo_path(repo_path),
            Path(file_path).as_posix(),
        )
        node_limit = min(
            MAX_INSPECT_NODE_LIMIT,
            _integer(
                params,
                "node_limit",
                DEFAULT_INSPECT_NODE_LIMIT,
                minimum=1,
            ),
        )
        edge_limit = min(
            MAX_INSPECT_EDGE_LIMIT,
            _integer(
                params,
                "edge_limit",
                DEFAULT_INSPECT_EDGE_LIMIT,
                minimum=1,
            ),
        )
        nodes = await asyncio.to_thread(_inspection_nodes, graph, node_ids, node_limit)
        selected_node_ids = [node.id for node in nodes]
        all_edges = (
            await asyncio.to_thread(graph.batch_get_edges_sync, selected_node_ids)
            if selected_node_ids
            else {}
        )
        ordered_edges = [
            (node_id, edge)
            for node_id in sorted(all_edges)
            for edge in sorted(
                all_edges[node_id],
                key=lambda edge: (edge.source_id, edge.target_id, edge.relationship, edge.id),
            )
        ]
        limited_edges = ordered_edges[:edge_limit]
        edges: dict[str, list[Edge]] = {}
        for node_id, edge in limited_edges:
            edges.setdefault(node_id, []).append(edge)
        return {
            "repo_path": canonicalize_repo_path(repo_path),
            "file_path": Path(file_path).as_posix(),
            "nodes": [_node_dict(node) for node in nodes],
            "edges": {
                node_id: [_edge_dict(edge) for edge in values]
                for node_id, values in edges.items()
            },
            "nodes_truncated": len(node_ids) > len(selected_node_ids),
            "edges_truncated": len(ordered_edges) > len(limited_edges),
        }

    async def ingest_files(self, params: dict[str, object]) -> dict[str, object]:
        repo_path = _string(params, "repo_path", required=True)
        assert repo_path is not None
        canonical = canonicalize_repo_path(repo_path)
        raw_files = _string_list(params, "file_paths")
        raw_deleted = _string_list(params, "deleted_paths")
        files = [validated_source_path(raw_path, canonical) for raw_path in raw_files]
        deleted: list[str] = []
        for raw_path in raw_deleted:
            candidate = Path(raw_path)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(
                    f"deleted path must remain repository-relative: {raw_path}"
                )
            deleted.append(candidate.as_posix())
        if not files and not deleted:
            raise ValueError("at least one changed or deleted file is required")
        binding = (
            self._binding_for_repository(canonical)
            if self._binding_for_repository is not None
            else None
        )
        if self._binding_for_repository is not None and binding is None:
            raise ValueError("repository does not have an indexed project store")
        job = await asyncio.to_thread(
            self._jobs().enqueue,
            repo_path=canonical,
            store_id=binding[0] if binding is not None else None,
            store_generation=binding[1] if binding is not None else None,
            mode="files",
            reason="explicit_files",
            payload={"file_paths": files, "deleted_paths": deleted},
        )
        return {"accepted": True, "job": job_to_dict(job)}

    async def refresh_dirty_files(self, params: dict[str, object]) -> dict[str, object]:
        """Hash-check dirty or anchor files and wait for single-writer re-indexing if stale."""

        repo_path = _string(params, "repo_path", required=True)
        assert repo_path is not None
        canonical = canonicalize_repo_path(repo_path)
        root = Path(canonical)
        timeout_seconds = min(5.0, max(0.2, float(_bounded_seconds(params, "timeout_seconds") or 4.0)))
        requested_paths = _string_list(params, "file_paths")
        graph = self._read_graph(canonical)
        if graph is None:
            return {
                "available": False,
                "status": "unavailable",
                "refreshed": [],
                "deleted": [],
                "degraded_reason": "index_unavailable",
            }
        rel_candidates: list[str] = []
        if requested_paths:
            for raw in requested_paths[:20]:
                try:
                    safe = Path(validated_source_path(raw, canonical, require_file=False))
                    rel_candidates.append(safe.relative_to(root).as_posix())
                except ValueError:
                    continue
        else:
            rel_candidates = await asyncio.to_thread(
                collect_git_dirty_paths, root, max_files=20
            )
        if not rel_candidates:
            return {
                "available": True,
                "status": "fresh",
                "refreshed": [],
                "deleted": [],
            }

        hashes = await asyncio.to_thread(graph.get_all_ingested_files_sync, canonical)

        def _detect_changes() -> tuple[list[str], list[str]]:
            dirty: list[str] = []
            deleted: list[str] = []
            for rel in dict.fromkeys(rel_candidates):
                target = root / rel
                if target.is_file():
                    try:
                        if target.stat().st_size > 1_048_576:
                            continue
                        digest = hashlib.sha256(target.read_bytes()).hexdigest()
                    except OSError:
                        continue
                    if hashes.get(rel) != digest:
                        dirty.append(str(target))
                    else:
                        fids = graph.get_node_ids_for_file_sync(canonical, rel)
                        if fids:
                            fnodes = graph.batch_get_nodes_sync(fids[:50])
                            for fnode in fnodes:
                                if fnode.type is NodeType.FILE and (
                                    fnode.metadata.get("language") != "text"
                                    or rel.lower().endswith((".html", ".htm"))
                                ):
                                    if (
                                        fnode.metadata.get("parser_version")
                                        != STRUCTURAL_EDGE_SCHEMA_VERSION
                                    ):
                                        dirty.append(str(target))
                                    break
                elif rel in hashes:
                    deleted.append(rel)
            return dirty, deleted

        dirty_files, deleted_paths = await asyncio.to_thread(_detect_changes)
        if not dirty_files and not deleted_paths:
            return {
                "available": True,
                "status": "fresh",
                "refreshed": [],
                "deleted": [],
            }

        try:
            enqueued = await self.ingest_files(
                {
                    "repo_path": canonical,
                    "file_paths": dirty_files,
                    "deleted_paths": deleted_paths,
                }
            )
            if self._on_job_enqueued is not None:
                self._on_job_enqueued()
        except ValueError as exc:
            return {
                "available": False,
                "status": "unavailable",
                "refreshed": [],
                "deleted": [],
                "degraded_reason": str(exc),
            }
        raw_job = cast(dict[str, object], enqueued["job"])
        job_id = str(raw_job["id"])
        wait = await self._wait_for_job(
            repo_path=canonical,
            job_id=job_id,
            timeout_seconds=timeout_seconds,
        )
        if wait is not None and wait.get("outcome") == "terminal":
            job_state = cast(dict[str, object] | None, wait.get("job"))
            if job_state is not None and job_state.get("status") == "completed":
                return {
                    "available": True,
                    "status": "refreshed",
                    "refreshed": [
                        Path(p).relative_to(root).as_posix() for p in dirty_files
                    ],
                    "deleted": deleted_paths,
                }
            return {
                "available": True,
                "status": "degraded",
                "refreshed": [],
                "deleted": [],
                "degraded_reason": "refresh_failed",
            }
        return {
            "available": True,
            "status": "degraded",
            "refreshed": [],
            "deleted": [],
            "degraded_reason": "refresh_timeout",
        }

    async def composite_regression_risk(
        self, params: dict[str, object]
    ) -> dict[str, object]:
        raw_paths = _string_list(params, "file_paths", default=False)
        repo_path = _string(params, "repo_path", required=True)
        assert repo_path is not None
        root = Path(canonicalize_repo_path(repo_path))
        dependent_limit = min(
            MAX_REGRESSION_RISK_LIMIT,
            _integer(params, "dependent_limit", DEFAULT_DEPENDENT_LIMIT, minimum=1),
        )
        test_target_limit = min(
            MAX_REGRESSION_RISK_LIMIT,
            _integer(params, "test_target_limit", DEFAULT_TEST_TARGET_LIMIT, minimum=1),
        )
        ranges_raw = params.get("changed_ranges", [])
        if not isinstance(ranges_raw, list) or len(cast(list[object], ranges_raw)) > 100:
            raise ValueError("changed_ranges must contain at most 100 ranges")
        ranges = [ChangedRange.model_validate(item) for item in cast(list[object], ranges_raw)]
        for changed in ranges:
            changed.file_path = Path(validated_source_path(
                changed.file_path, str(root), require_file=False,
            )).relative_to(root).as_posix()
        started = perf_counter()
        graph = self._read_graph(repo_path)
        if graph is None:
            empty = self._empty_regression_risk(raw_paths, started)
            if ranges:
                empty.update(complete=False, degraded_reason="index_unavailable")
            return empty
        if ranges:
            return await asyncio.to_thread(
                changed_symbol_impact, graph, str(root), ranges, limit=min(dependent_limit, test_target_limit),
            )
        lookup_started = perf_counter()
        affected_ids: list[str] = []
        for raw_path in raw_paths:
            source = Path(validated_source_path(raw_path, str(root), require_file=False))
            rel_path = source.relative_to(root).as_posix()
            affected_ids.extend(
                await asyncio.to_thread(
                    graph.get_node_ids_for_file_sync, str(root), rel_path
                )
            )
        affected_ids = list(dict.fromkeys(affected_ids))
        file_lookup_ms = _elapsed_ms(lookup_started)
        affected_id_set = set(affected_ids)
        edge_started = perf_counter()
        paths: dict[str, list[dict[str, object]]] = {
            aid: [] for aid in sorted(affected_id_set)
        }
        frontier = list(paths)
        edge_count = 0
        bounded = False
        for hop_idx in range(REVIEW_DEPTH):
            batches = (
                await asyncio.to_thread(
                    graph.batch_get_edges_sync, frontier, direction="incoming"
                )
                if frontier
                else {}
            )
            owner_map: dict[str, str] = {}
            owner_batches: dict[str, list[Edge]] = {}
            if hop_idx >= 1 and frontier:
                frontier_nodes = await asyncio.to_thread(
                    graph.batch_get_nodes_sync, frontier
                )
                non_file_ids = {
                    node.id
                    for node in frontier_nodes
                    if node.type is not NodeType.FILE
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
                    owner_batches = await asyncio.to_thread(
                        graph.batch_get_edges_sync,
                        sorted(set(owner_map.values())),
                        direction="incoming",
                    )
            following: list[str] = []
            for target in frontier:
                candidate_edges = list(batches.get(target, []))
                owner_fid = owner_map.get(target)
                if owner_fid is not None:
                    for owner_edge in owner_batches.get(owner_fid, []):
                        if owner_edge not in candidate_edges:
                            candidate_edges.append(owner_edge)
                for edge in sorted(
                    candidate_edges,
                    key=lambda item: (item.source_id, item.relationship),
                ):
                    expected_target = target if edge.target_id == target else owner_fid
                    if (
                        edge.relationship not in DEPENDENCY_RELATIONSHIPS
                        or edge.target_id != expected_target
                    ):
                        continue
                    edge_count += 1
                    if edge_count > MAX_REVIEW_EDGES or len(paths) >= MAX_REVIEW_NODES:
                        bounded = True
                        break
                    if edge.source_id in paths or edge.source_id == expected_target:
                        continue
                    step: dict[str, object] = {
                        "dependent_node_id": edge.source_id,
                        "affected_node_id": expected_target,
                        "relationship": edge.relationship.value,
                    }
                    for meta_key in (
                        "start_line",
                        "end_line",
                        "reference_kind",
                        "enclosing_symbol",
                        "via_barrel",
                    ):
                        val = edge.metadata.get(meta_key)
                        if val is not None:
                            step[meta_key] = val
                    paths[edge.source_id] = [*paths[target], step]
                    following.append(edge.source_id)
            frontier = following
        edge_traversal_ms = _elapsed_ms(edge_started)

        dependent_ids = sorted(set(paths) - affected_id_set)
        capped_dependent_ids = dependent_ids[:MAX_REVIEW_NODES]
        bounded |= len(dependent_ids) > MAX_REVIEW_NODES
        hydration_started = perf_counter()
        all_hydrated = (
            await asyncio.to_thread(graph.batch_get_nodes_sync, capped_dependent_ids)
            if capped_dependent_ids
            else []
        )
        node_hydration_ms = _elapsed_ms(hydration_started)
        projection_started = perf_counter()
        dep_files = {
            node.id: str(node.metadata.get("file_path", "")) for node in all_hydrated
        }
        for chain in paths.values():
            for step in chain:
                dep_id = step.get("dependent_node_id")
                if (
                    isinstance(dep_id, str)
                    and dep_id in dep_files
                    and dep_files[dep_id]
                ):
                    step.setdefault("file_path", dep_files[dep_id])

        def _dependent_rank(node: Node) -> tuple[int, int, str, str]:
            chain = paths[node.id]
            only_imports = bool(chain) and all(
                step.get("relationship") == RelationshipType.IMPORTS.value
                for step in chain
            )
            return (
                len(chain),
                1 if only_imports else 0,
                str(node.metadata.get("file_path", "")),
                node.id,
            )

        ranked_dependents = sorted(all_hydrated, key=_dependent_rank)
        dependents = ranked_dependents[:dependent_limit]
        test_dependents = [
            node
            for node in ranked_dependents
            if is_test_path(str(node.metadata.get("file_path", "")))
        ]
        test_target_paths = sorted(
            {
                str(node.metadata.get("file_path"))
                for node in test_dependents
                if node.metadata.get("file_path")
            }
        )
        selected_test_paths = test_target_paths[:test_target_limit]
        test_targets = [
            {
                "file_path": path,
                "evidence": [
                    {
                        k: v
                        for k, v in paths[node.id][0].items()
                        if k != "file_path"
                    }
                    if len(paths[node.id]) == 1
                    else {
                        "dependent_node_id": node.id,
                        "dependency_path": paths[node.id],
                    }
                    for node in test_dependents
                    if str(node.metadata.get("file_path")) == path
                ],
            }
            for path in selected_test_paths
        ]
        projection_ms = _elapsed_ms(projection_started)
        dependents_truncated = len(dependent_ids) > len(dependents) or bounded
        test_targets_truncated = len(test_target_paths) > len(selected_test_paths) or bounded
        return {
            "file_paths": raw_paths,
            "affected_node_ids": affected_ids,
            "dependents": [_node_dict(node) for node in dependents],
            "test_dependents": [_node_dict(node) for node in test_dependents],
            "dependency_paths": [
                {"dependent_node_id": node.id, "path": paths[node.id]}
                for node in dependents
            ],
            "total_dependents": len(dependent_ids),
            "dependents_truncated": dependents_truncated,
            "test_targets": test_targets,
            "total_test_targets": len(test_target_paths),
            "test_targets_truncated": test_targets_truncated,
            "complete": not dependents_truncated and not test_targets_truncated,
            "timings": {
                "file_lookup_ms": file_lookup_ms,
                "edge_traversal_ms": edge_traversal_ms,
                "node_hydration_ms": node_hydration_ms,
                "projection_ms": projection_ms,
                "total_ms": _elapsed_ms(started),
            },
        }

    @staticmethod
    def _empty_regression_risk(
        file_paths: list[str], started: float
    ) -> dict[str, object]:
        return {
            "file_paths": file_paths,
            "available": False,
            "degraded_reason": "index_unavailable",
            "affected_node_ids": [],
            "dependents": [],
            "test_dependents": [],
            "total_dependents": 0,
            "dependents_truncated": False,
            "test_targets": [],
            "total_test_targets": 0,
            "test_targets_truncated": False,
            "complete": True,
            "timings": {
                "file_lookup_ms": 0.0,
                "edge_traversal_ms": 0.0,
                "node_hydration_ms": 0.0,
                "projection_ms": 0.0,
                "total_ms": _elapsed_ms(started),
            },
        }

    async def lsp_references(self, params: dict[str, object]) -> dict[str, object]:
        located = await self._node_at_position(params)
        if located is None:
            return self._lsp_unavailable(
                str(params.get("file_path", "")),
                "no indexed symbol exists at this position",
            )
        graph, node = located
        edges = await asyncio.to_thread(
            graph.get_edges_sync, node.id, direction="incoming"
        )
        references: list[dict[str, object]] = []
        for edge in edges:
            related = await asyncio.to_thread(graph.get_node_sync, edge.source_id)
            if related is None:
                continue
            item = _node_dict(related)
            item["relationship"] = edge.relationship
            for meta_key in (
                "start_line",
                "end_line",
                "reference_kind",
                "enclosing_symbol",
                "sites",
            ):
                val = edge.metadata.get(meta_key)
                if val is not None:
                    item[meta_key] = val
            references.append(item)
        return {
            "available": True,
            "source": "index",
            "symbol": _node_dict(node),
            "references": references,
        }

    async def _indexed_location(
        self, file_path: Path
    ) -> tuple[NativeGraph | None, str | None, str]:
        resolver = self._graph_for_repository
        if resolver is not None:
            # Start at the nearest ancestor so nested registered repositories
            # retain ownership of their source paths.
            for root in file_path.parents:
                repo_path = canonicalize_repo_path(str(root))
                graph = await asyncio.to_thread(resolver, repo_path)
                if graph is None:
                    continue
                try:
                    source = Path(validated_source_path(str(file_path), repo_path))
                except ValueError:
                    continue
                return graph, repo_path, source.relative_to(repo_path).as_posix()

        graph = self._read_graph(None)
        if graph is None:
            return None, None, file_path.name
        ingestion_stats = await asyncio.to_thread(graph.get_ingestion_stats_sync)
        for repo_path in ingestion_stats:
            root = Path(repo_path)
            try:
                source = Path(validated_source_path(str(file_path), str(root)))
            except ValueError:
                continue
            return graph, repo_path, source.relative_to(root).as_posix()
        return None, None, file_path.name

    async def _node_at_position(
        self, params: dict[str, object]
    ) -> tuple[NativeGraph, Node] | None:
        file_path = _string(params, "file_path", required=True)
        assert file_path is not None
        line = _integer(params, "line", 0)
        graph, repo_path, rel_path = await self._indexed_location(
            Path(validated_source_path(file_path))
        )
        if graph is None or repo_path is None:
            return None
        node_ids = await asyncio.to_thread(
            graph.get_node_ids_for_file_sync, repo_path, rel_path
        )
        candidates = [
            node
            for node_id in node_ids
            if (node := await asyncio.to_thread(graph.get_node_sync, node_id))
            is not None
        ]
        containing = [
            node
            for node in candidates
            if _integer_metadata(
                node.metadata.get("start_line"), key="start_line", default=-1
            )
            <= line
            <= _integer_metadata(
                node.metadata.get("end_line"), key="end_line", default=-1
            )
        ]
        node = min(
            containing,
            key=lambda node: (
                _integer_metadata(
                    node.metadata.get("end_line"), key="end_line", default=line
                )
                - _integer_metadata(
                    node.metadata.get("start_line"), key="start_line", default=line
                )
            ),
            default=None,
        )
        return (graph, node) if node is not None else None

    @staticmethod
    def _lsp_unavailable(file_path: str, reason: str) -> dict[str, object]:
        return {
            "available": False,
            "source": "index",
            "file_path": file_path,
            "reason": reason,
            "language_server_configured": False,
        }
