"""A review crosses the public query and real indexed service boundaries."""

from pathlib import Path
from typing import cast

import pytest

from scs.graph.native import NativeGraph
from scs.orchestration.decision import DeterministicDecisionProvider
from scs.orchestration.query import QueryOrchestrator
from scs.providers.base import EmbeddingProvider, ProviderMetadata
from scs.services.routes import SCSServiceRoutes


@pytest.mark.asyncio
async def test_changed_range_contracts_and_coverage_share_one_investigation(tmp_path: Path) -> None:
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'spec.md').write_text('Status: approved\nFailed parsing must preserve input.\n')
    (repo / 'source.py').write_text('# [contract](spec.md)\ndef parse():\n    return 1\n')
    graph = NativeGraph(
        database_path=tmp_path / 'graph.db', vector_path=tmp_path / 'vectors.usearch',
        provider_metadata_path=tmp_path / 'provider.json',
        provider=ProviderMetadata('test', 'structural', 2, False, 'not used'),
    )
    repo_id = graph.get_or_create_repo_sync(str(repo))
    graph.batch_upsert_nodes_sync([
        {'id': name, 'type': 'function', 'name': name, 'repo_id': repo_id,
         'metadata': {'file_path': path, 'start_line': 1, 'end_line': 2}}
        for name, path in [('parse', 'source.py'), ('caller', 'client.py'), ('test', 'tests/test_client.py')]]
    )
    graph.batch_upsert_edges_sync([
        {'source_id': source, 'target_id': target, 'relationship': 'calls'}
        for source, target in [('caller', 'parse'), ('test', 'caller')]]
    )
    routes = SCSServiceRoutes(
        graph=lambda: graph, jobs=lambda: pytest.fail('read-only query opened jobs'),
        embeddings=lambda: cast(EmbeddingProvider, object()),
    )

    async def call(method: str, params: dict[str, object]) -> dict[str, object]:
        assert method == 'knowledge.composite.regression_risk'
        return await routes.composite_regression_risk(params)

    query = QueryOrchestrator(provider=DeterministicDecisionProvider(), call=call)
    result = await query.query({
        'goal': 'review parser', 'repo_path': str(repo),
        'changed_ranges': [{'file_path': 'source.py', 'start_line': 1, 'end_line': 2}],
    })

    assert result['complete'] is True
    assert len(result['evidence']['relationships']) == 2
    assert result['evidence']['test_targets'][0]['file_path'] == 'tests/test_client.py'
    assert result['evidence']['contracts'][0]['active'] is True
    assert result['investigation']['dependents']['evidence_count'] == 2
    assert result['investigation']['tests']['status'] == 'assessed'
    assert result['investigation']['references']['status'] == 'not_assessed'


@pytest.mark.asyncio
@pytest.mark.parametrize('declarations', [1, 21])
async def test_symbol_resolution_only_is_bounded_and_never_traverses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declarations: int,
) -> None:
    repo = tmp_path / 'repo'
    repo.mkdir()
    graph = NativeGraph(
        database_path=tmp_path / 'graph.db', vector_path=tmp_path / 'vectors.usearch',
        provider_metadata_path=tmp_path / 'provider.json',
        provider=ProviderMetadata('test', 'structural', 2, False, 'not used'),
    )
    repo_id = graph.get_or_create_repo_sync(str(repo))
    graph.batch_upsert_nodes_sync([
        {'id': f'parse-{index}', 'type': 'function', 'name': 'parse', 'repo_id': repo_id,
         'metadata': {'file_path': f'module_{index}.py'}}
        for index in range(declarations)
    ])
    monkeypatch.setattr(NativeGraph, 'traverse_sync', lambda *_args, **_kwargs: pytest.fail('resolution traversed dependencies'))
    routes = SCSServiceRoutes(
        graph=lambda: graph, jobs=lambda: pytest.fail('resolution opened jobs'),
        embeddings=lambda: cast(EmbeddingProvider, object()),
    )

    result = await routes.related({'repo_path': str(repo), 'symbol_name': 'parse', 'resolve_only': True})

    assert len(result['matches']) == min(declarations, 20)
    assert result['matches_truncated'] is (declarations > 20)
    assert result['related'] == []
