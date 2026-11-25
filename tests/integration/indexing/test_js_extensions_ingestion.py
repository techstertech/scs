"""End-to-end non-standard JS/TS extension ingestion and upgrade tests."""

from __future__ import annotations

from pathlib import Path

from scs.graph.models import NodeType, RelationshipType
from scs.graph.native import NativeGraph
from scs.indexing.parser.native import NativeParser
from scs.indexing.pipeline import IngestionPipeline
from scs.providers.base import ProviderMetadata


class _PreUpgradeParser(NativeParser):
    """Simulate a pre-upgrade parser without .cjs/.mjs/.cts/.mts support."""

    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".py", ".ts", ".tsx", ".js", ".jsx"})


def _graph(tmp_path: Path) -> NativeGraph:
    provider = ProviderMetadata("test", "disabled", 2, False, "offline")
    return NativeGraph(
        database_path=tmp_path / "graph.db",
        vector_path=tmp_path / "vectors.usearch",
        provider_metadata_path=tmp_path / "provider.json",
        provider=provider,
    )


def _target_file_path(graph: NativeGraph, node_id: str) -> object:
    node = graph.get_node_sync(node_id)
    assert node is not None
    return node.metadata["file_path"]


def test_js_ts_variant_extensions_ingest_symbols_and_resolve_relative_imports(
    repository: Path, tmp_path: Path
) -> None:
    (repository / "config").mkdir(parents=True)
    (repository / "utils").mkdir(parents=True)
    (repository / "src").mkdir(parents=True)

    (repository / "config" / "settings.json").write_text(
        '{"port": 3000}\n', encoding="utf-8"
    )
    (repository / "helper.cjs").write_text(
        "exports.computeValue = (x) => x * 2;\n", encoding="utf-8"
    )
    (repository / "utils" / "index.mjs").write_text(
        "export const formatValue = (v) => String(v);\n", encoding="utf-8"
    )
    (repository / "src" / "types.cts").write_text(
        "export const KIND = 'CJS';\n", encoding="utf-8"
    )
    (repository / "src" / "entry.mts").write_text(
        "import { KIND } from './types.cjs';\nexport const readKind = () => KIND;\n",
        encoding="utf-8",
    )
    (repository / "src" / "widget.mjsx").write_text(
        "export function Widget() { return <div />; }\n", encoding="utf-8"
    )
    (repository / "main.cjs").write_text(
        """const { computeValue } = require('./helper.cjs');
const settings = require('./config/settings.json');
const { formatValue } = require('./utils');

exports.run = (input) => formatValue(computeValue(input) + settings.port);
exports.API_VERSION = 'v1';
module.exports.fallback = function(val) {
    return val;
};
""",
        encoding="utf-8",
    )

    graph = _graph(tmp_path)
    result = IngestionPipeline(graph=graph, parser=NativeParser()).ingest(repository)

    assert result.files_discovered == 7
    assert result.files_failed == 0

    recorded_languages = graph.get_all_ingested_file_languages_sync(
        str(repository.resolve())
    )
    assert recorded_languages == {
        "config/settings.json": "text",
        "helper.cjs": "javascript",
        "main.cjs": "javascript",
        "src/entry.mts": "typescript",
        "src/types.cts": "typescript",
        "src/widget.mjsx": "jsx",
        "utils/index.mjs": "javascript",
    }

    for symbol_name, expected_qname in [
        ("run", "main.run"),
        ("fallback", "main.fallback"),
        ("API_VERSION", "main.API_VERSION"),
        ("computeValue", "helper.computeValue"),
        ("Widget", "src.widget.Widget"),
    ]:
        matches = graph.search_by_name_sync(symbol_name)
        assert any(
            node.metadata.get("qualified_name") == expected_qname for node in matches
        ), f"missing {expected_qname}"

    file_nodes = {
        node.metadata["file_path"]: node
        for node in graph.list_nodes_sync(node_type=NodeType.FILE)
    }

    main_outgoing = {
        (edge.relationship, _target_file_path(graph, edge.target_id))
        for edge in graph.get_edges_sync(file_nodes["main.cjs"].id, direction="outgoing")
    }
    assert (RelationshipType.IMPORTS.value, "helper.cjs") in main_outgoing
    assert (RelationshipType.IMPORTS.value, "config/settings.json") in main_outgoing
    assert (RelationshipType.IMPORTS.value, "utils/index.mjs") in main_outgoing

    mts_outgoing = {
        (edge.relationship, _target_file_path(graph, edge.target_id))
        for edge in graph.get_edges_sync(
            file_nodes["src/entry.mts"].id, direction="outgoing"
        )
    }
    assert (RelationshipType.IMPORTS.value, "src/types.cts") in mts_outgoing

    for target_path in ("config/settings.json", "helper.cjs"):
        incoming_files = {
            _target_file_path(graph, edge.source_id)
            for edge in graph.get_edges_sync(
                file_nodes[target_path].id, direction="incoming"
            )
            if edge.relationship == RelationshipType.IMPORTS.value
        }
        assert "main.cjs" in incoming_files


def test_previously_text_indexed_cjs_file_upgrades_incrementally_without_force(
    repository: Path, tmp_path: Path
) -> None:
    (repository / "helper.cjs").write_text(
        "exports.helper = () => 42;\n", encoding="utf-8"
    )
    (repository / "main.cjs").write_text(
        "const { helper } = require('./helper.cjs');\nexports.run = () => helper();\n",
        encoding="utf-8",
    )

    graph = _graph(tmp_path)
    repo_root = str(repository.resolve())

    initial = IngestionPipeline(graph=graph, parser=_PreUpgradeParser()).ingest(
        repository
    )
    assert initial.files_failed == 0
    assert graph.get_all_ingested_file_languages_sync(repo_root) == {
        "helper.cjs": "text",
        "main.cjs": "text",
    }
    assert graph.search_by_name_sync("run", node_type=NodeType.FUNCTION) == []

    upgraded = IngestionPipeline(graph=graph, parser=NativeParser()).ingest(
        repository, force=False
    )
    assert upgraded.files_changed == 2
    assert upgraded.files_failed == 0
    assert graph.get_all_ingested_file_languages_sync(repo_root) == {
        "helper.cjs": "javascript",
        "main.cjs": "javascript",
    }

    run_matches = graph.search_by_name_sync("run", node_type=NodeType.FUNCTION)
    assert [node.metadata.get("qualified_name") for node in run_matches] == [
        "main.run"
    ]

    file_nodes = {
        node.metadata["file_path"]: node
        for node in graph.list_nodes_sync(node_type=NodeType.FILE)
    }
    outgoing_targets = {
        _target_file_path(graph, edge.target_id)
        for edge in graph.get_edges_sync(file_nodes["main.cjs"].id, direction="outgoing")
        if edge.relationship == RelationshipType.IMPORTS.value
    }
    assert outgoing_targets == {"helper.cjs"}
