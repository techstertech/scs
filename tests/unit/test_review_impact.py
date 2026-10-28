"""Review impact follows indexed dependency evidence, never filename guesses."""

from pathlib import Path

import pytest

from scs.graph.native import NativeGraph
from scs.providers.base import ProviderMetadata
from scs.services.impact import ChangedRange, changed_symbol_impact


def indexed_graph(tmp_path: Path) -> tuple[NativeGraph, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    graph = NativeGraph(
        database_path=tmp_path / "graph.db",
        vector_path=tmp_path / "vectors.usearch",
        provider_metadata_path=tmp_path / "provider.json",
        provider=ProviderMetadata("test", "structural", 2, False, "not used"),
    )
    repo_id = graph.get_or_create_repo_sync(str(repo))
    definitions = [
        ("changed", "src.py", 0, 3), ("unchanged", "src.py", 5, 8),
        ("caller", "client.py", 0, 2), ("test", "tests/test_client.py", 0, 2),
        ("unrelated", "tests/test_other.py", 0, 2),
    ]
    graph.batch_upsert_nodes_sync([
        {"id": name, "type": "function", "name": name, "repo_id": repo_id,
         "metadata": {"file_path": path, "start_line": start, "end_line": end}}
        for name, path, start, end in definitions
    ])
    graph.batch_upsert_edges_sync([
        {"source_id": source, "target_id": target, "relationship": "calls"}
        for source, target in [("caller", "changed"), ("test", "caller"),
                               ("unrelated", "unchanged"), ("changed", "caller")]
    ])
    return graph, str(repo)


def test_range_selects_changed_symbol_and_explains_indirect_test(tmp_path: Path) -> None:
    graph, repo = indexed_graph(tmp_path)
    result = changed_symbol_impact(graph, repo, [ChangedRange(
        file_path="src.py", start_line=1, end_line=1,
    )], limit=10)

    assert result["affected_node_ids"] == ["changed"]
    assert {node["id"] for node in result["dependents"]} == {"caller", "test"}
    target, = result["test_targets"]
    assert target["file_path"] == "tests/test_client.py"
    assert target["evidence_kind"] == "static_dependency"
    assert [edge["dependent_node_id"] for edge in target["evidence"][0]["dependency_path"]] == ["caller", "test"]
    assert result["complete"] is True


def test_test_search_is_independent_of_dependent_output_limit(tmp_path: Path) -> None:
    graph, repo = indexed_graph(tmp_path)
    result = changed_symbol_impact(graph, repo, [ChangedRange(
        file_path="src.py", start_line=0, end_line=3,
    )], limit=1)

    assert result["dependents_truncated"] is True
    assert result["test_targets"][0]["file_path"] == "tests/test_client.py"
    assert result["complete"] is False


def test_unmatched_ranges_are_visible(tmp_path: Path) -> None:
    graph, repo = indexed_graph(tmp_path)
    result = changed_symbol_impact(graph, repo, [ChangedRange(
        file_path="missing.py", start_line=0, end_line=0,
    )], limit=10)

    assert result["affected_node_ids"] == []
    assert result["unresolved_ranges"]
    assert result["complete"] is False


@pytest.mark.parametrize("start,end", [(-1, 0), (3, 2)])
def test_invalid_ranges_are_rejected(start: int, end: int) -> None:
    with pytest.raises(ValueError):
        ChangedRange(file_path="src.py", start_line=start, end_line=end)
