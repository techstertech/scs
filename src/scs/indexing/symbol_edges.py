"""Resolve symbol-level edges, barrel re-exports, and cross-boundary links."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import posixpath
import re
from typing import Protocol, cast

from scs.graph.models import Edge, Node, NodeType, RelationshipType, SYMBOL_NODE_TYPES
from scs.indexing.js_ts_imports import (
    JS_TS_LANGUAGES,
    TsConfigAliases,
    resolve_js_ts_import_target,
)
from scs.indexing.parser.base import ParsedEdge, ParsedEntity

STRUCTURAL_EDGE_SCHEMA_VERSION = 2
MAX_REEXPORT_DEPTH = 5
MAX_SITES_PER_EDGE = 16
MAX_FILE_DESCRIPTORS = 200
MAX_BRIDGE_FILES = 10_000

_HTML_SCRIPT_SRC_RE = re.compile(
    r"""<script\b[^>]*\bsrc\s*=\s*(?:"([^"]+)"|'([^']+)')""",
    re.IGNORECASE,
)
_EXTERNAL_URL_PREFIXES = (
    "http://",
    "https://",
    "//",
    "data:",
    "blob:",
    "javascript:",
)

DESCRIPTOR_REFERENCE_KINDS: frozenset[str] = frozenset(
    {
        "http_route_handler",
        "http_route_client",
        "bridge_provider",
        "bridge_caller",
        "export_default",
    }
)
CROSS_BOUNDARY_REFERENCE_KINDS: frozenset[str] = frozenset(
    {"http_route", "rpc_bridge"}
)
_REFERENCE_PRIORITY: dict[str, int] = {
    "calls": 0,
    "renders": 1,
    "imports_named": 2,
    "imports_default": 3,
    "http_route": 4,
    "rpc_bridge": 5,
    "script_src": 6,
    "imports_asset": 7,
    "extends": 8,
    "implements": 9,
    "type_references": 10,
    "reexports": 11,
    "imports_namespace": 12,
    "imports_side_effect": 13,
}


class ParsedEntryLike(Protocol):
    rel_path: str
    language: str


class ParsedFileLike(Protocol):
    @property
    def entry(self) -> ParsedEntryLike: ...
    @property
    def entities(self) -> list[ParsedEntity]: ...
    @property
    def edges(self) -> list[ParsedEdge]: ...


class SymbolEdgeGraph(Protocol):
    def resolve_node_id_by_qualified_name_sync(
        self, repo_path: str, qualified_name: str
    ) -> str | None: ...
    def get_file_node_map_sync(self, repo_id: int) -> dict[str, str]: ...
    def get_node_ids_for_file_sync(
        self, repo_path: str, rel_path: str
    ) -> list[str]: ...
    def list_nodes_sync(
        self,
        *,
        node_type: NodeType | None = None,
        limit: int = 100,
        offset: int = 0,
        repo_id: int | None = None,
    ) -> list[Node]: ...
    def get_edges_sync(
        self, node_id: str, *, direction: str = "both"
    ) -> list[Edge]: ...
    def batch_get_edges_sync(
        self, node_ids: list[str], *, direction: str = "both"
    ) -> dict[str, list[Edge]]: ...
    def batch_upsert_edges_sync(self, edges: list[dict[str, object]]) -> int: ...


def extract_html_script_edges(source: str, rel_path: str) -> list[ParsedEdge]:
    """Extract bounded local `<script src="...">` import edges from HTML files."""

    edges: list[ParsedEdge] = []
    seen: set[tuple[str, int]] = set()
    for match in _HTML_SCRIPT_SRC_RE.finditer(source):
        if len(edges) >= MAX_FILE_DESCRIPTORS:
            break
        raw_src = (match.group(1) or match.group(2) or "").strip()
        if not raw_src:
            continue
        lower_src = raw_src.lower()
        if lower_src.startswith(_EXTERNAL_URL_PREFIXES):
            continue
        line_no = source.count("\n", 0, match.start())
        key = (raw_src, line_no)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            ParsedEdge(
                source_qualified_name=rel_path,
                target_qualified_name=raw_src,
                relationship=RelationshipType.IMPORTS,
                metadata={
                    "reference_kind": "script_src",
                    "start_line": line_no,
                    "end_line": line_no,
                    "enclosing_symbol": rel_path,
                    "wildcard_import": True,
                },
            )
        )
    return edges


def extract_file_descriptors(
    edges: Sequence[ParsedEdge],
) -> tuple[list[ParsedEdge], dict[str, object]]:
    """Strip synthetic cross-boundary descriptors and summarize onto File metadata."""

    structural: list[ParsedEdge] = []
    http_handlers: list[dict[str, object]] = []
    http_clients: list[dict[str, object]] = []
    bridge_providers: list[dict[str, object]] = []
    bridge_callers: list[dict[str, object]] = []
    seen_descriptors: set[tuple[str, str]] = set()
    default_export: str | None = None

    for edge in edges:
        ref_kind = edge.metadata.get("reference_kind")
        if ref_kind not in DESCRIPTOR_REFERENCE_KINDS:
            structural.append(edge)
            continue
        meta = dict(edge.metadata)
        if ref_kind == "export_default":
            if default_export is None:
                local_name = meta.get("local_name")
                if isinstance(local_name, str) and local_name:
                    default_export = local_name
            continue
        dedup_key = (str(ref_kind), json.dumps(meta, sort_keys=True))
        if dedup_key in seen_descriptors:
            continue
        seen_descriptors.add(dedup_key)
        if ref_kind == "http_route_handler":
            if len(http_handlers) < MAX_FILE_DESCRIPTORS:
                http_handlers.append(meta)
        elif ref_kind == "http_route_client":
            if len(http_clients) < MAX_FILE_DESCRIPTORS:
                http_clients.append(meta)
        elif ref_kind == "bridge_provider":
            if len(bridge_providers) < MAX_FILE_DESCRIPTORS:
                bridge_providers.append(meta)
        elif ref_kind == "bridge_caller":
            if len(bridge_callers) < MAX_FILE_DESCRIPTORS:
                bridge_callers.append(meta)

    descriptors: dict[str, object] = {}
    if http_handlers:
        http_handlers.sort(
            key=lambda item: (
                str(item.get("norm_path", "")),
                str(item.get("http_method", "")),
                _int_or(item.get("start_line"), 0),
            )
        )
        descriptors["http_handlers"] = http_handlers
    if http_clients:
        http_clients.sort(
            key=lambda item: (
                str(item.get("norm_path", "")),
                str(item.get("http_method", "")),
                _int_or(item.get("start_line"), 0),
            )
        )
        descriptors["http_clients"] = http_clients
    if bridge_providers:
        bridge_providers.sort(
            key=lambda item: (
                str(item.get("bridge_namespace", "")),
                str(item.get("bridge_member", "")),
                _int_or(item.get("start_line"), 0),
            )
        )
        descriptors["bridge_providers"] = bridge_providers
    if bridge_callers:
        bridge_callers.sort(
            key=lambda item: (
                str(item.get("bridge_namespace", "")),
                str(item.get("bridge_member", "")),
                _int_or(item.get("start_line"), 0),
            )
        )
        descriptors["bridge_callers"] = bridge_callers
    if default_export is not None:
        descriptors["default_export"] = default_export

    return structural, descriptors


def _int_or(value: object, default: int) -> int:
    return value if type(value) is int else default


def _merge_sites(
    existing_sites: object,
    incoming_sites: object,
    *,
    fallback_meta: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    combined: dict[tuple[int, int, str, str], dict[str, object]] = {}

    def ingest(raw: object) -> None:
        if not isinstance(raw, list):
            return
        for item in cast(list[object], raw):
            if not isinstance(item, dict):
                continue
            site = cast(dict[str, object], item)
            start = _int_or(site.get("start_line"), -1)
            end = _int_or(site.get("end_line"), start)
            if start < 0:
                continue
            ref_kind = str(site.get("reference_kind", ""))
            qualifier = str(site.get("qualifier", ""))
            key = (start, end, ref_kind, qualifier)
            if key not in combined:
                entry: dict[str, object] = {"start_line": start, "end_line": end}
                if ref_kind:
                    entry["reference_kind"] = ref_kind
                if qualifier:
                    entry["qualifier"] = qualifier
                combined[key] = entry

    ingest(existing_sites)
    ingest(incoming_sites)
    if not combined and fallback_meta is not None:
        start = _int_or(fallback_meta.get("start_line"), -1)
        end = _int_or(fallback_meta.get("end_line"), start)
        if start >= 0:
            fallback_entry: dict[str, object] = {"start_line": start, "end_line": end}
            ref_kind = fallback_meta.get("reference_kind")
            if isinstance(ref_kind, str) and ref_kind:
                fallback_entry["reference_kind"] = ref_kind
            qualifier = fallback_meta.get("qualifier")
            if isinstance(qualifier, str) and qualifier:
                fallback_entry["qualifier"] = qualifier
            combined[(start, end, str(ref_kind or ""), str(qualifier or ""))] = (
                fallback_entry
            )

    ordered = [combined[k] for k in sorted(combined)]
    return ordered[:MAX_SITES_PER_EDGE]


def merge_edge_metadata(
    existing: dict[str, object], incoming: Mapping[str, object]
) -> dict[str, object]:
    """Merge metadata deterministically when two occurrences share the same edge ID."""

    if not existing and not incoming:
        return {}
    if not existing:
        return dict(incoming)
    if not incoming:
        return dict(existing)

    merged = dict(existing)

    # Merge explicit imported symbol lists cleanly.
    sym_candidates: tuple[object, object] = (
        merged.get("imported_symbols"),
        incoming.get("imported_symbols"),
    )
    symbols: set[str] = set()
    for raw_list in sym_candidates:
        if isinstance(raw_list, list):
            for item in cast(list[object], raw_list):
                if isinstance(item, str) and item:
                    symbols.add(item)
    if symbols:
        merged["imported_symbols"] = sorted(symbols)

    if merged.get("wildcard_import") is True or incoming.get("wildcard_import") is True:
        merged["wildcard_import"] = True

    if merged.get("is_reexport") is True or incoming.get("is_reexport") is True:
        merged["is_reexport"] = True

    re_candidates: tuple[object, object] = (
        merged.get("reexport_bindings"),
        incoming.get("reexport_bindings"),
    )
    re_list: list[dict[str, object]] = []
    seen_re: set[tuple[str, str, str]] = set()
    for raw_list in re_candidates:
        if isinstance(raw_list, list):
            for item in cast(list[object], raw_list):
                if not isinstance(item, dict):
                    continue
                b = cast(dict[str, object], item)
                key = (
                    str(b.get("kind", "")),
                    str(b.get("imported", "")),
                    str(b.get("exported", "")),
                )
                if key not in seen_re:
                    seen_re.add(key)
                    re_list.append(dict(b))
    if re_list:
        merged["reexport_bindings"] = re_list

    # Preserve best-specificity reference_kind.
    rk_a = existing.get("reference_kind")
    rk_b = incoming.get("reference_kind")
    if isinstance(rk_a, str) and isinstance(rk_b, str):
        merged["reference_kind"] = min(
            (rk_a, rk_b),
            key=lambda rk: (_REFERENCE_PRIORITY.get(rk, 50), rk),
        )
    elif isinstance(rk_b, str):
        merged["reference_kind"] = rk_b

    sites = _merge_sites(
        existing.get("sites")
        or ([{"start_line": existing["start_line"], "end_line": existing.get("end_line", existing["start_line"]), "reference_kind": existing.get("reference_kind", "")}] if type(existing.get("start_line")) is int else None),
        incoming.get("sites")
        or ([{"start_line": incoming["start_line"], "end_line": incoming.get("end_line", incoming["start_line"]), "reference_kind": incoming.get("reference_kind", "")}] if type(incoming.get("start_line")) is int else None),
    )
    if sites:
        merged["sites"] = sites
        merged["start_line"] = sites[0]["start_line"]
        merged["end_line"] = sites[0]["end_line"]
    elif type(merged.get("start_line")) is not int and type(incoming.get("start_line")) is int:
        merged["start_line"] = incoming["start_line"]
        merged["end_line"] = incoming.get("end_line", incoming["start_line"])

    if "enclosing_symbol" not in merged and "enclosing_symbol" in incoming:
        merged["enclosing_symbol"] = incoming["enclosing_symbol"]
    if "via_barrel" not in merged and "via_barrel" in incoming:
        merged["via_barrel"] = incoming["via_barrel"]

    return merged


def _fetch_nodes_for_ids(graph: SymbolEdgeGraph, node_ids: list[str]) -> list[Node]:
    if not node_ids:
        return []
    batch_get = getattr(graph, "batch_get_nodes_sync", None)
    if callable(batch_get):
        return cast(list[Node], batch_get(node_ids))
    wanted = set(node_ids)
    found: list[Node] = []
    offset = 0
    page_size = 500
    while len(found) < len(wanted):
        page = graph.list_nodes_sync(limit=page_size, offset=offset)
        if not page:
            break
        for node in page:
            if node.id in wanted:
                found.append(node)
        if len(page) < page_size:
            break
        offset += len(page)
    return found


def _resolve_python_module_node(
    rel_path: str,
    specifier: str,
    resolve: Callable[[str], str | None],
) -> str | None:
    if not specifier:
        return None
    clean = specifier.lstrip(".")
    candidates: list[str] = []
    if specifier.startswith("."):
        dir_parts = [p for p in posixpath.dirname(rel_path).split("/") if p and p != "."]
        dots = len(specifier) - len(clean)
        if dots > 1:
            up = dots - 1
            if up <= len(dir_parts):
                dir_parts = dir_parts[: len(dir_parts) - up]
        prefix = ".".join(dir_parts)
        if clean:
            dotted = f"{prefix}.{clean}" if prefix else clean
            candidates.extend((dotted, f"src.{dotted}"))
    else:
        candidates.extend((clean, f"src.{clean}"))
    for candidate in candidates:
        hit = resolve(candidate)
        if hit is not None:
            return hit
    return None


class _ExportResolver:
    """Resolve file symbols and barrel re-exports without recursive closure cycles."""

    def __init__(
        self,
        *,
        graph: SymbolEdgeGraph,
        repo_path: str,
        parsed_by_path: Mapping[str, ParsedFileLike],
        file_id_to_path: Mapping[str, str],
        file_qnames: Mapping[str, Mapping[str, str]],
        qualified_to_id: Mapping[str, str],
        file_descriptors_by_path: Mapping[str, Mapping[str, object]],
        planned_reexports_by_file_id: Mapping[str, list[tuple[str, dict[str, object]]]],
    ) -> None:
        self._graph: SymbolEdgeGraph = graph
        self._repo_path: str = repo_path
        self._parsed_by_path: Mapping[str, ParsedFileLike] = parsed_by_path
        self._file_id_to_path: Mapping[str, str] = file_id_to_path
        self._file_qnames: Mapping[str, Mapping[str, str]] = file_qnames
        self._qualified_to_id: Mapping[str, str] = qualified_to_id
        self._file_descriptors_by_path: Mapping[str, Mapping[str, object]] = (
            file_descriptors_by_path
        )
        self._planned_reexports: Mapping[str, list[tuple[str, dict[str, object]]]] = (
            planned_reexports_by_file_id
        )
        self._file_symbols_cache: dict[str, dict[str, str]] = {}
        self._file_default_export_cache: dict[str, str | None] = {}
        self._export_lookup_cache: dict[tuple[str, str], tuple[str | None, str | None]] = {}
        self._file_edges_cache: dict[str, list[Edge]] = {}

    def file_symbols(self, file_id: str) -> dict[str, str]:
        if file_id in self._file_symbols_cache:
            return self._file_symbols_cache[file_id]
        symbols: dict[str, str] = {}
        rel = self._file_id_to_path.get(file_id)
        if rel is not None and rel in self._parsed_by_path:
            item = self._parsed_by_path[rel]
            local_q = self._file_qnames.get(rel, {})
            file_qname = next(
                (e.qualified_name for e in item.entities if e.kind is NodeType.FILE),
                None,
            )
            for entity in item.entities:
                if entity.kind not in SYMBOL_NODE_TYPES:
                    continue
                nid = local_q.get(entity.qualified_name) or self._qualified_to_id.get(
                    entity.qualified_name
                )
                if not nid:
                    continue
                if entity.parent_qualified_name == file_qname or entity.name not in symbols:
                    symbols[entity.name] = nid
            desc = self._file_descriptors_by_path.get(rel, {})
            raw_default = desc.get("default_export")
            self._file_default_export_cache[file_id] = (
                raw_default if isinstance(raw_default, str) else None
            )
        elif rel is not None:
            node_ids = self._graph.get_node_ids_for_file_sync(self._repo_path, rel)
            nodes = _fetch_nodes_for_ids(self._graph, node_ids)
            file_qname = None
            for n in nodes:
                if n.id == file_id or n.type is NodeType.FILE:
                    raw_fq = n.metadata.get("qualified_name")
                    file_qname = raw_fq if isinstance(raw_fq, str) else None
                    raw_def = n.metadata.get("default_export")
                    self._file_default_export_cache[file_id] = (
                        raw_def if isinstance(raw_def, str) else None
                    )
                    break
            for n in nodes:
                if n.type not in SYMBOL_NODE_TYPES:
                    continue
                is_top = (
                    isinstance(file_qname, str)
                    and str(n.metadata.get("qualified_name", "")) == f"{file_qname}.{n.name}"
                )
                if is_top or n.name not in symbols:
                    symbols[n.name] = n.id
        self._file_symbols_cache[file_id] = symbols
        return symbols

    def _reexports_for_file(self, file_id: str) -> list[tuple[str, Mapping[str, object]]]:
        if file_id in self._planned_reexports:
            return list(self._planned_reexports[file_id])
        rel = self._file_id_to_path.get(file_id)
        if rel is not None and rel in self._parsed_by_path:
            return []
        if file_id not in self._file_edges_cache:
            self._file_edges_cache[file_id] = self._graph.get_edges_sync(
                file_id, direction="outgoing"
            )
        out: list[tuple[str, Mapping[str, object]]] = []
        for edge in self._file_edges_cache[file_id]:
            if (
                edge.relationship is RelationshipType.IMPORTS
                and edge.metadata.get("is_reexport") is True
                and edge.target_id != file_id
            ):
                out.append((edge.target_id, edge.metadata))
        return out

    def resolve_exported_symbol(
        self,
        target_file_id: str,
        symbol_name: str,
        visited: frozenset[str] = frozenset(),
        depth: int = 0,
    ) -> tuple[str | None, str | None]:
        if target_file_id in visited or depth > MAX_REEXPORT_DEPTH or not symbol_name:
            return None, None
        cache_key = (target_file_id, symbol_name)
        if depth == 0 and cache_key in self._export_lookup_cache:
            return self._export_lookup_cache[cache_key]

        symbols = self.file_symbols(target_file_id)
        hit: str | None = None
        if symbol_name == "default":
            default_local = self._file_default_export_cache.get(target_file_id)
            if default_local and default_local in symbols:
                hit = symbols[default_local]
            elif "default" in symbols:
                hit = symbols["default"]
            else:
                rel = self._file_id_to_path.get(target_file_id, "")
                stem = posixpath.splitext(posixpath.basename(rel))[0]
                if stem and stem in symbols:
                    hit = symbols[stem]
                elif len(symbols) == 1:
                    hit = next(iter(symbols.values()))
        elif symbol_name in symbols:
            hit = symbols[symbol_name]

        if hit is not None:
            result = (hit, None)
            if depth == 0:
                self._export_lookup_cache[cache_key] = result
            return result

        next_visited = visited | {target_file_id}
        barrel_path = self._file_id_to_path.get(target_file_id)
        if depth < MAX_REEXPORT_DEPTH:
            reexports = self._reexports_for_file(target_file_id)
            for downstream_id, meta in reexports:
                raw_bindings = meta.get("reexport_bindings")
                if not isinstance(raw_bindings, list):
                    continue
                for raw_b in cast(list[object], raw_bindings):
                    if not isinstance(raw_b, dict):
                        continue
                    b = cast(dict[str, object], raw_b)
                    if b.get("kind") == "named" and b.get("exported") == symbol_name:
                        orig = b.get("imported")
                        if isinstance(orig, str) and orig:
                            found_id, inner_via = self.resolve_exported_symbol(
                                downstream_id, orig, next_visited, depth + 1
                            )
                            if found_id is not None:
                                res = (found_id, barrel_path or inner_via)
                                if depth == 0:
                                    self._export_lookup_cache[cache_key] = res
                                return res

            for downstream_id, meta in reexports:
                raw_bindings = meta.get("reexport_bindings")
                if not isinstance(raw_bindings, list):
                    continue
                has_star = any(
                    isinstance(raw_b, dict)
                    and cast(dict[str, object], raw_b).get("kind") == "star"
                    for raw_b in cast(list[object], raw_bindings)
                )
                if has_star:
                    found_id, inner_via = self.resolve_exported_symbol(
                        downstream_id, symbol_name, next_visited, depth + 1
                    )
                    if found_id is not None:
                        res = (found_id, barrel_path or inner_via)
                        if depth == 0:
                            self._export_lookup_cache[cache_key] = res
                        return res

        if depth == 0:
            self._export_lookup_cache[cache_key] = (None, None)
        return None, None


def _record_merged_edge(
    merged_edges: dict[tuple[str, str, str], dict[str, object]],
    source_id: str,
    target_id: str,
    rel_type: str,
    weight: float,
    metadata: Mapping[str, object] | None,
) -> None:
    key = (source_id, target_id, rel_type)
    clean_meta = dict(metadata) if metadata else {}
    existing = merged_edges.get(key)
    if existing is None:
        entry: dict[str, object] = {
            "source_id": source_id,
            "target_id": target_id,
            "relationship": rel_type,
            "weight": weight,
        }
        if clean_meta:
            entry["metadata"] = clean_meta
        merged_edges[key] = entry
    elif clean_meta:
        prev_meta = cast(dict[str, object], existing.get("metadata") or {})
        existing["metadata"] = merge_edge_metadata(prev_meta, clean_meta)


def resolve_structural_edges(
    *,
    graph: SymbolEdgeGraph,
    repo_path: str,
    parsed: Sequence[ParsedFileLike],
    qualified_to_id: Mapping[str, str],
    resolve: Callable[[str], str | None],
    file_qnames: Mapping[str, Mapping[str, str]],
    file_descriptors_by_path: Mapping[str, Mapping[str, object]],
    file_nodes_by_path: Mapping[str, str] | None,
    lower_file_nodes_by_path: Mapping[str, str] | None,
    alias_configs: Sequence[TsConfigAliases],
    planned_file_nodes: Mapping[str, str] | None,
    resolve_js_ts_target: Callable[..., str | None] = resolve_js_ts_import_target,
) -> list[dict[str, object]]:
    """Resolve intra-file, cross-file, named-import, and barrel re-export edges."""

    parsed_by_path: dict[str, ParsedFileLike] = {
        item.entry.rel_path: item for item in parsed
    }
    file_id_to_path: dict[str, str] = {}
    if file_nodes_by_path:
        for path, fid in file_nodes_by_path.items():
            file_id_to_path[fid] = path
    if planned_file_nodes:
        for path, fid in planned_file_nodes.items():
            file_id_to_path[fid] = path

    planned_reexports_by_file_id: dict[str, list[tuple[str, dict[str, object]]]] = {}
    for item in parsed:
        src_fid = (planned_file_nodes or {}).get(item.entry.rel_path)
        if src_fid is None:
            continue
        for edge in item.edges:
            if (
                edge.relationship is RelationshipType.IMPORTS
                and edge.metadata.get("is_reexport") is True
            ):
                target_fid = resolve_js_ts_target(
                    item.entry.rel_path,
                    edge.target_qualified_name,
                    resolve,
                    file_nodes_by_path=file_nodes_by_path,
                    lower_file_nodes_by_path=lower_file_nodes_by_path,
                    alias_configs=alias_configs,
                )
                if target_fid is not None and target_fid != src_fid:
                    planned_reexports_by_file_id.setdefault(src_fid, []).append(
                        (target_fid, dict(edge.metadata))
                    )

    export_resolver = _ExportResolver(
        graph=graph,
        repo_path=repo_path,
        parsed_by_path=parsed_by_path,
        file_id_to_path=file_id_to_path,
        file_qnames=file_qnames,
        qualified_to_id=qualified_to_id,
        file_descriptors_by_path=file_descriptors_by_path,
        planned_reexports_by_file_id=planned_reexports_by_file_id,
    )
    merged_edges: dict[tuple[str, str, str], dict[str, object]] = {}

    for item in parsed:
        rel_path = item.entry.rel_path
        local_qnames = file_qnames.get(rel_path, {})
        file_qname = next(
            (e.qualified_name for e in item.entities if e.kind is NodeType.FILE),
            None,
        )
        source_file_id = (
            (planned_file_nodes or {}).get(rel_path)
            or (local_qnames.get(file_qname) if file_qname else None)
            or (file_nodes_by_path.get(rel_path) if file_nodes_by_path else None)
        )

        # Build local symbol map (same-file declarations + resolved imports).
        local_bindings: dict[str, str] = {}
        if source_file_id is not None:
            local_bindings.update(export_resolver.file_symbols(source_file_id))
        namespace_imports: dict[str, str] = {}

        deferred_edges: list[ParsedEdge] = []

        # Pass 1: resolve IMPORTS edges first so named/default/namespace bindings are
        # ready before resolving CALLS, renders, extends, implements, and type refs.
        for edge in item.edges:
            if edge.relationship is not RelationshipType.IMPORTS:
                deferred_edges.append(edge)
                continue

            source = local_qnames.get(edge.source_qualified_name) or resolve(
                edge.source_qualified_name
            )
            target = local_qnames.get(edge.target_qualified_name) or resolve(
                edge.target_qualified_name
            )
            if target is None and item.entry.language == "python":
                target = resolve(f"src.{edge.target_qualified_name}")
                if target is None:
                    target = _resolve_python_module_node(
                        rel_path, edge.target_qualified_name, resolve
                    )
            elif target is None and (
                item.entry.language in JS_TS_LANGUAGES
                or edge.metadata.get("reference_kind") == "script_src"
            ):
                target = resolve_js_ts_target(
                    rel_path,
                    edge.target_qualified_name,
                    resolve,
                    file_nodes_by_path=file_nodes_by_path,
                    lower_file_nodes_by_path=lower_file_nodes_by_path,
                    alias_configs=alias_configs,
                )
                if (
                    target is None
                    and edge.metadata.get("reference_kind") == "script_src"
                    and file_nodes_by_path is not None
                ):
                    cleaned = (
                        edge.target_qualified_name.split("?", 1)[0]
                        .split("#", 1)[0]
                        .strip()
                        .lstrip("/")
                        .lstrip("./")
                    )
                    if cleaned:
                        suffix = f"/{cleaned.lower()}"
                        exact_lower = cleaned.lower()
                        hits = [
                            fid
                            for path, fid in file_nodes_by_path.items()
                            if path.lower() == exact_lower or path.lower().endswith(suffix)
                        ]
                        if len(hits) == 1:
                            target = hits[0]

            if not source:
                continue

            if target is not None:
                _record_merged_edge(
                    merged_edges,
                    source,
                    target,
                    edge.relationship.value,
                    edge.weight,
                    edge.metadata,
                )

            # For Python `from mod import sym as alias`, track local binding to
            # target symbol directly (or via its module file) without adding extra IMPORTS.
            if item.entry.language == "python" and edge.metadata:
                raw_bindings = edge.metadata.get("import_bindings")
                py_target_file: str | None = (
                    target if target in file_id_to_path else None
                )
                if py_target_file is None:
                    from_mod = edge.metadata.get("from_module")
                    if isinstance(from_mod, str) and from_mod:
                        mod_node = _resolve_python_module_node(
                            rel_path, from_mod, resolve
                        )
                        if mod_node in file_id_to_path:
                            py_target_file = mod_node
                if isinstance(raw_bindings, list):
                    for raw_b in cast(list[object], raw_bindings):
                        if not isinstance(raw_b, dict):
                            continue
                        b = cast(dict[str, object], raw_b)
                        local_name = b.get("local")
                        imported_name = b.get("imported")
                        if not isinstance(local_name, str) or not local_name:
                            continue
                        if (
                            target is not None
                            and target not in file_id_to_path
                            and isinstance(imported_name, str)
                            and edge.target_qualified_name.endswith(f".{imported_name}")
                        ):
                            local_bindings.setdefault(local_name, target)
                        elif py_target_file is not None and isinstance(imported_name, str):
                            sym_id, _ = export_resolver.resolve_exported_symbol(
                                py_target_file, imported_name
                            )
                            if sym_id is not None:
                                local_bindings.setdefault(local_name, sym_id)

            if target is None:
                continue

            # Expand named/default/namespace bindings when target is a File node.
            if target in file_id_to_path and edge.metadata:
                raw_bindings = edge.metadata.get("import_bindings")
                if isinstance(raw_bindings, list):
                    for raw_b in cast(list[object], raw_bindings):
                        if not isinstance(raw_b, dict):
                            continue
                        b = cast(dict[str, object], raw_b)
                        local_name = b.get("local")
                        imported_name = b.get("imported")
                        b_kind = b.get("kind")
                        if not isinstance(local_name, str) or not local_name:
                            continue
                        if b_kind == "namespace":
                            namespace_imports[local_name] = target
                            continue
                        if isinstance(imported_name, str) and imported_name:
                            sym_id, via_barrel = export_resolver.resolve_exported_symbol(
                                target, imported_name
                            )
                            if sym_id is not None:
                                local_bindings.setdefault(local_name, sym_id)
                                if source_file_id is not None and edge.metadata.get("is_reexport") is not True:
                                    ref_meta: dict[str, object] = {
                                        "reference_kind": (
                                            "imports_default"
                                            if b_kind == "default"
                                            else "imports_named"
                                        ),
                                        "enclosing_symbol": edge.metadata.get(
                                            "enclosing_symbol", file_qname or rel_path
                                        ),
                                    }
                                    if type(edge.metadata.get("start_line")) is int:
                                        ref_meta["start_line"] = edge.metadata["start_line"]
                                    if type(edge.metadata.get("end_line")) is int:
                                        ref_meta["end_line"] = edge.metadata.get(
                                            "end_line", edge.metadata["start_line"]
                                        )
                                    if via_barrel:
                                        ref_meta["via_barrel"] = via_barrel
                                    _record_merged_edge(
                                        merged_edges,
                                        source_file_id,
                                        sym_id,
                                        RelationshipType.REFERENCES.value,
                                        1.0,
                                        ref_meta,
                                    )

        # Pass 2: resolve CONTAINS, CALLS, REFERENCES, INHERITS, IMPLEMENTS.
        for edge in deferred_edges:
            if edge.metadata.get("reference_kind") in DESCRIPTOR_REFERENCE_KINDS:
                continue

            source = local_qnames.get(edge.source_qualified_name) or resolve(
                edge.source_qualified_name
            )
            if not source:
                continue

            target_name = edge.target_qualified_name
            target = local_qnames.get(target_name) or resolve(target_name)

            if target is None and edge.relationship is not RelationshipType.CONTAINS:
                qualifier = edge.metadata.get("qualifier")
                if isinstance(qualifier, str) and qualifier in namespace_imports:
                    sym_id, _ = export_resolver.resolve_exported_symbol(
                        namespace_imports[qualifier], target_name
                    )
                    target = sym_id
                if (
                    target is None
                    and qualifier in {"this", "self"}
                    and "." in edge.source_qualified_name
                ):
                    class_scope = edge.source_qualified_name.rsplit(".", 1)[0]
                    target = local_qnames.get(f"{class_scope}.{target_name}") or resolve(
                        f"{class_scope}.{target_name}"
                    )
                if target is None:
                    target = local_bindings.get(target_name)

            if target:
                _record_merged_edge(
                    merged_edges,
                    source,
                    target,
                    edge.relationship.value,
                    edge.weight,
                    edge.metadata,
                )

    return list(merged_edges.values())


def _all_file_nodes_for_repo(graph: SymbolEdgeGraph, repo_id: int) -> list[Node]:
    nodes: list[Node] = []
    offset = 0
    page_size = 500
    while len(nodes) < MAX_BRIDGE_FILES:
        page = graph.list_nodes_sync(
            node_type=NodeType.FILE, limit=page_size, offset=offset, repo_id=repo_id
        )
        nodes.extend(page)
        if len(page) < page_size:
            break
        offset += len(page)
    return nodes


def _resolve_node_for_enclosing(
    graph: SymbolEdgeGraph,
    repo_path: str,
    enclosing_qname: object,
    fallback_file_id: str,
) -> str:
    if isinstance(enclosing_qname, str) and enclosing_qname:
        nid = graph.resolve_node_id_by_qualified_name_sync(repo_path, enclosing_qname)
        if nid is not None:
            return nid
    return fallback_file_id


def _method_compatible(client_method: str, handler_method: str) -> bool:
    return (
        client_method == "ANY"
        or handler_method == "ALL"
        or client_method == handler_method
    )


def _delete_stale_cross_boundary_edges(
    graph: SymbolEdgeGraph,
    file_nodes: Sequence[Node],
    desired_edges: Mapping[tuple[str, str, str], dict[str, object]],
) -> None:
    delete_edge = getattr(graph, "delete_edge_sync", None)
    if not callable(delete_edge):
        return
    file_ids = [node.id for node in file_nodes]
    for start in range(0, len(file_ids), 200):
        edges_by_id = graph.batch_get_edges_sync(
            file_ids[start : start + 200], direction="both"
        )
        for edge_list in edges_by_id.values():
            for edge in edge_list:
                if (
                    edge.relationship is RelationshipType.REFERENCES
                    and edge.metadata.get("reference_kind") in CROSS_BOUNDARY_REFERENCE_KINDS
                    and (edge.source_id, edge.target_id, edge.relationship.value)
                    not in desired_edges
                ):
                    delete_edge(edge.id)


def _http_segments(norm_path: str) -> tuple[str, ...]:
    if norm_path == "/":
        return ()
    return tuple(seg for seg in norm_path.split("/") if seg)


def _http_paths_match(client_norm: str, handler_norm: str) -> bool:
    if client_norm == handler_norm:
        return True
    c_segs = _http_segments(client_norm)
    h_segs = _http_segments(handler_norm)
    if len(c_segs) != len(h_segs) or not c_segs:
        return False
    has_static_match = False
    for c_seg, h_seg in zip(c_segs, h_segs, strict=True):
        if c_seg == h_seg:
            if c_seg != ":param":
                has_static_match = True
            continue
        if c_seg == ":param" or h_seg == ":param":
            continue
        return False
    return has_static_match


def reconcile_cross_boundary_edges(
    *,
    graph: SymbolEdgeGraph,
    repo_path: str,
    repo_id: int,
    file_nodes_by_path: Mapping[str, str],
    lower_file_nodes_by_path: Mapping[str, str],
    alias_configs: Sequence[TsConfigAliases],
    resolve: Callable[[str], str | None],
) -> int:
    """Link generic HTTP route handlers/clients and global/RPC bridges across files."""

    file_nodes = _all_file_nodes_for_repo(graph, repo_id)
    has_descriptors = any(
        any(
            key in node.metadata
            for key in (
                "http_handlers",
                "http_clients",
                "bridge_providers",
                "bridge_callers",
            )
        )
        for node in file_nodes
    )
    if not has_descriptors:
        return 0

    handlers_by_path: dict[str, list[tuple[Node, dict[str, object]]]] = {}
    handlers_by_seg_count: dict[int, list[tuple[str, Node, dict[str, object]]]] = {}
    clients: list[tuple[Node, dict[str, object]]] = []
    bridge_providers_exact: dict[tuple[str, str], list[tuple[Node, dict[str, object]]]] = {}
    bridge_providers_wild: dict[str, list[tuple[Node, dict[str, object]]]] = {}
    bridge_callers: list[tuple[Node, dict[str, object]]] = []

    for fnode in file_nodes:
        for raw in cast(list[object], fnode.metadata.get("http_handlers", []) or []):
            if isinstance(raw, dict):
                item = cast(dict[str, object], raw)
                norm = item.get("norm_path")
                if isinstance(norm, str) and norm:
                    handlers_by_path.setdefault(norm, []).append((fnode, item))
                    seg_count = len(_http_segments(norm))
                    handlers_by_seg_count.setdefault(seg_count, []).append(
                        (norm, fnode, item)
                    )
        for raw in cast(list[object], fnode.metadata.get("http_clients", []) or []):
            if isinstance(raw, dict):
                clients.append((fnode, cast(dict[str, object], raw)))
        for raw in cast(list[object], fnode.metadata.get("bridge_providers", []) or []):
            if isinstance(raw, dict):
                item = cast(dict[str, object], raw)
                ns = item.get("bridge_namespace")
                member = item.get("bridge_member")
                if isinstance(ns, str) and isinstance(member, str) and ns and member:
                    if member == "*":
                        bridge_providers_wild.setdefault(ns, []).append((fnode, item))
                    else:
                        bridge_providers_exact.setdefault((ns, member), []).append(
                            (fnode, item)
                        )
        for raw in cast(list[object], fnode.metadata.get("bridge_callers", []) or []):
            if isinstance(raw, dict):
                bridge_callers.append((fnode, cast(dict[str, object], raw)))

    desired: dict[tuple[str, str, str], dict[str, object]] = {}

    def record_cross_edge(
        source_id: str,
        target_id: str,
        metadata: dict[str, object],
    ) -> None:
        if not source_id or not target_id or source_id == target_id:
            return
        key = (source_id, target_id, RelationshipType.REFERENCES.value)
        existing = desired.get(key)
        if existing is None:
            desired[key] = {
                "source_id": source_id,
                "target_id": target_id,
                "relationship": RelationshipType.REFERENCES.value,
                "weight": 1.0,
                "metadata": metadata,
            }
        else:
            prev_meta = cast(dict[str, object], existing["metadata"])
            existing["metadata"] = merge_edge_metadata(prev_meta, metadata)

    # Cache symbol tables lazily only for files that participate in routes/bridges.
    lazy_file_symbols: dict[str, dict[str, str]] = {}

    def symbols_for_file(rel_path: str) -> dict[str, str]:
        if rel_path in lazy_file_symbols:
            return lazy_file_symbols[rel_path]
        node_ids = graph.get_node_ids_for_file_sync(repo_path, rel_path)
        sym_map: dict[str, str] = {}
        for node in _fetch_nodes_for_ids(graph, node_ids):
            if node.type in SYMBOL_NODE_TYPES:
                sym_map[node.name] = node.id
                fq = node.metadata.get("qualified_name")
                if isinstance(fq, str) and fq:
                    sym_map[fq] = node.id
        lazy_file_symbols[rel_path] = sym_map
        return sym_map

    def resolve_handler_node(file_node: Node, desc: dict[str, object]) -> str:
        rel_path = str(file_node.metadata.get("file_path", file_node.name))
        handler_sym = desc.get("handler_symbol")
        if isinstance(handler_sym, str) and handler_sym:
            simple = handler_sym.rsplit(".", 1)[-1]
            sym_map = symbols_for_file(rel_path)
            if handler_sym in sym_map:
                return sym_map[handler_sym]
            if simple in sym_map:
                return sym_map[simple]
            by_qname = graph.resolve_node_id_by_qualified_name_sync(
                repo_path, handler_sym
            )
            if by_qname is not None:
                return by_qname
            # Fallback: look for an imported named reference matching handler_sym
            for edge in graph.get_edges_sync(file_node.id, direction="outgoing"):
                if edge.relationship is RelationshipType.REFERENCES:
                    target_nodes = _fetch_nodes_for_ids(graph, [edge.target_id])
                    if target_nodes and target_nodes[0].name == simple:
                        return target_nodes[0].id
        return _resolve_node_for_enclosing(
            graph, repo_path, desc.get("enclosing_symbol"), file_node.id
        )

    # 1) Match HTTP route clients -> handlers
    for client_file, client_desc in clients:
        norm = client_desc.get("norm_path")
        client_method = str(client_desc.get("http_method", "GET"))
        if not isinstance(norm, str) or not norm:
            continue
        matched_handlers = list(handlers_by_path.get(norm, []))
        for h_norm, handler_file, handler_desc in handlers_by_seg_count.get(
            len(_http_segments(norm)), []
        ):
            if h_norm != norm and _http_paths_match(norm, h_norm):
                matched_handlers.append((handler_file, handler_desc))
        if not matched_handlers:
            continue
        source_id = _resolve_node_for_enclosing(
            graph, repo_path, client_desc.get("enclosing_symbol"), client_file.id
        )
        for handler_file, handler_desc in matched_handlers:
            handler_method = str(handler_desc.get("http_method", "GET"))
            if not _method_compatible(client_method, handler_method):
                continue
            target_id = resolve_handler_node(handler_file, handler_desc)
            start = _int_or(client_desc.get("start_line"), 0)
            end = _int_or(client_desc.get("end_line"), start)
            record_cross_edge(
                source_id,
                target_id,
                {
                    "reference_kind": "http_route",
                    "http_method": handler_method,
                    "http_path": str(client_desc.get("http_path", norm)),
                    "norm_path": norm,
                    "start_line": start,
                    "end_line": end,
                    "enclosing_symbol": client_desc.get(
                        "enclosing_symbol", client_file.metadata.get("qualified_name", client_file.name)
                    ),
                    "sites": [
                        {
                            "start_line": start,
                            "end_line": end,
                            "reference_kind": "http_route",
                        }
                    ],
                },
            )

    # Expand wildcard bridge providers (e.g. `commands: require('./browser_source_commands.cjs')`)
    expanded_wild: dict[str, list[tuple[Node, dict[str, object], dict[str, str]]]] = {}
    file_path_by_id = {fid: path for path, fid in file_nodes_by_path.items()}
    for ns, entries in bridge_providers_wild.items():
        for prov_file, prov_desc in entries:
            prov_rel = str(prov_file.metadata.get("file_path", prov_file.name))
            target_file_id: str | None = None
            target_spec = prov_desc.get("target_specifier")
            if isinstance(target_spec, str) and target_spec:
                target_file_id = resolve_js_ts_import_target(
                    prov_rel,
                    target_spec,
                    resolve,
                    file_nodes_by_path=file_nodes_by_path,
                    lower_file_nodes_by_path=lower_file_nodes_by_path,
                    alias_configs=alias_configs,
                )
            elif isinstance(prov_desc.get("target_local"), str):
                target_local = str(prov_desc["target_local"])
                for edge in graph.get_edges_sync(prov_file.id, direction="outgoing"):
                    if edge.relationship is RelationshipType.IMPORTS:
                        raw_bindings = edge.metadata.get("import_bindings")
                        if isinstance(raw_bindings, list) and any(
                            isinstance(rb, dict)
                            and cast(dict[str, object], rb).get("local") == target_local
                            for rb in cast(list[object], raw_bindings)
                        ):
                            target_file_id = edge.target_id
                            break
            if target_file_id and target_file_id in file_path_by_id:
                sym_map = symbols_for_file(file_path_by_id[target_file_id])
                expanded_wild.setdefault(ns, []).append(
                    (prov_file, prov_desc, sym_map)
                )

    # 2) Match global / RPC bridge callers -> providers
    caller_local_namespaces: dict[str, set[str]] = {}
    for caller_file, caller_desc in bridge_callers:
        ns = str(caller_desc.get("bridge_namespace", ""))
        member = str(caller_desc.get("bridge_member", ""))
        if not ns or not member:
            continue
        # If the caller file imported `ns` as a local ES/CJS namespace import,
        # normal CALLS resolution already handles it directly; don't confuse it with RPC.
        if ns == "commands":
            if caller_file.id not in caller_local_namespaces:
                locals_set: set[str] = set()
                for edge in graph.get_edges_sync(caller_file.id, direction="outgoing"):
                    if edge.relationship is RelationshipType.IMPORTS:
                        for raw_b in cast(
                            list[object], edge.metadata.get("import_bindings", []) or []
                        ):
                            if isinstance(raw_b, dict):
                                loc = cast(dict[str, object], raw_b).get("local")
                                if isinstance(loc, str):
                                    locals_set.add(loc)
                caller_local_namespaces[caller_file.id] = locals_set
            if "commands" in caller_local_namespaces[caller_file.id]:
                continue

        source_id = _resolve_node_for_enclosing(
            graph, repo_path, caller_desc.get("enclosing_symbol"), caller_file.id
        )
        start = _int_or(caller_desc.get("start_line"), 0)
        end = _int_or(caller_desc.get("end_line"), start)
        edge_meta: dict[str, object] = {
            "reference_kind": "rpc_bridge",
            "bridge_namespace": ns,
            "bridge_member": member,
            "start_line": start,
            "end_line": end,
            "enclosing_symbol": caller_desc.get(
                "enclosing_symbol",
                caller_file.metadata.get("qualified_name", caller_file.name),
            ),
            "sites": [
                {
                    "start_line": start,
                    "end_line": end,
                    "reference_kind": "rpc_bridge",
                }
            ],
        }

        matched_targets: list[str] = []
        for prov_file, prov_desc in bridge_providers_exact.get((ns, member), []):
            matched_targets.append(resolve_handler_node(prov_file, prov_desc))
        for _prov_file, _prov_desc, sym_map in expanded_wild.get(ns, []):
            if member in sym_map:
                matched_targets.append(sym_map[member])
        if not matched_targets and ns.startswith("window."):
            head_member = ns.removeprefix("window.")
            for prov_file, prov_desc in bridge_providers_exact.get(
                ("window", head_member), []
            ):
                matched_targets.append(resolve_handler_node(prov_file, prov_desc))

        for target_id in dict.fromkeys(matched_targets):
            record_cross_edge(source_id, target_id, dict(edge_meta))

    _delete_stale_cross_boundary_edges(graph, file_nodes, desired)

    if not desired:
        return 0
    return graph.batch_upsert_edges_sync(list(desired.values()))
