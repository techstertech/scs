"""Every model choice maps to one fixed, bounded service playbook."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from scs.orchestration.decision import Playbook, RoutingDecision, RoutingRequest
from scs.orchestration.query import QueryOrchestrator


class FixedProvider:
    def __init__(self, playbook: Playbook) -> None:
        self.playbook = playbook
        self.calls = 0

    async def classify(self, request: RoutingRequest) -> RoutingDecision:
        self.calls += 1
        assert "repository bytes" not in request.model_dump_json()
        return RoutingDecision(
            playbook=self.playbook,
            model="test-provider",
            confidence=1.0,
            probabilities={
                label: float(label is self.playbook) for label in Playbook
            },
        )


class FakeRoutes:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call(self, method: str, params: dict[str, object]) -> dict[str, object]:
        self.calls.append((method, params))
        node = {
            "id": "node-1", "type": "function", "name": "parse",
            "file_path": "parser.py", "start_line": 1, "end_line": 3,
        }
        match method:
            case "knowledge.search":
                return {"results": [node], "timed_out": False, "degraded_reason": None}
            case "knowledge.related":
                return {"matches": [node], "related": []}
            case "knowledge.inspect_file":
                return {"file_path": "parser.py", "nodes": [node], "edges": {},
                        "nodes_truncated": False, "edges_truncated": False}
            case "knowledge.nodes.list":
                return {"nodes": [node], "total": 1}
            case "knowledge.composite.regression_risk":
                return {"dependents": [node], "test_targets": [], "complete": True,
                        "dependents_truncated": False, "test_targets_truncated": False}
            case "lsp.references":
                return {"available": True, "symbol": node, "references": [node]}
            case _:
                raise AssertionError(f"unrecognized service route: {method}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("playbook", "anchors", "expected_route"),
    [
        (Playbook.DISCOVER, {}, "knowledge.search"),
        (Playbook.UNDERSTAND, {}, "knowledge.related"),
        (Playbook.RELATIONSHIPS, {"node_ids": ["node-1"]}, "knowledge.related"),
        (Playbook.REFERENCES, {"node_ids": ["node-1"]}, "knowledge.related"),
        (Playbook.INSPECT_FILES, {"file_paths": ["parser.py"]}, "knowledge.inspect_file"),
        (Playbook.IMPACT, {"file_paths": ["parser.py"]}, "knowledge.composite.regression_risk"),
        (Playbook.INVENTORY, {}, "knowledge.nodes.list"),
    ],
)
async def test_each_playbook_uses_only_its_fixed_routes(
    tmp_path: Path,
    playbook: Playbook,
    anchors: dict[str, object],
    expected_route: str,
) -> None:
    (tmp_path / "parser.py").write_text("def parse(): pass\n")
    provider = FixedProvider(playbook)
    routes = FakeRoutes()
    query = QueryOrchestrator(provider=provider, call=routes.call)

    result = await query.query({"goal": "find parser", "repo_path": str(tmp_path), **anchors})

    assert result["routing"]["playbook"] == playbook.value
    assert provider.calls == 1
    assert expected_route in [method for method, _params in routes.calls]
    assert result["trace"]
    assert set(result["evidence"]) >= {
        "symbols", "files", "relationships", "references", "test_targets"
    }


@pytest.mark.asyncio
async def test_ineligible_and_failed_classifier_fall_back_once(tmp_path: Path) -> None:
    provider = FixedProvider(Playbook.IMPACT)
    routes = FakeRoutes()
    query = QueryOrchestrator(provider=provider, call=routes.call)

    result = await query.query({"goal": "find parser", "repo_path": str(tmp_path)})

    assert result["routing"]["playbook"] == Playbook.DISCOVER.value
    assert result["routing"]["degraded_reason"] == "ineligible_playbook"
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_references_include_direct_dependency_edges_only(tmp_path: Path) -> None:
    class DependencyRoutes(FakeRoutes):
        async def call(self, method: str, params: dict[str, object]) -> dict[str, object]:
            self.calls.append((method, params))
            assert method == "knowledge.related"
            seed = {
                "id": "symbol", "type": "class", "name": "Parser",
                "metadata": {"file_path": "parser.py"},
            }
            source = {
                "id": "importer", "type": "file", "name": "test_parser.py",
                "metadata": {"file_path": "test_parser.py"},
            }
            related = [{"node": seed, "depth": 0}]
            if params["relationship"] == "imports":
                related.append({"node": source, "depth": 1})
            return {"matches": [seed], "related": related}

    routes = DependencyRoutes()
    query = QueryOrchestrator(provider=FixedProvider(Playbook.REFERENCES), call=routes.call)

    result = await query.query({
        "goal": "Find references to Parser", "repo_path": str(tmp_path),
        "symbol_name": "Parser",
    })

    assert {params["relationship"] for _method, params in routes.calls} == {
        "calls", "imports", "inherits", "implements", "references",
    }
    assert all(params["depth"] == 1 for _method, params in routes.calls)
    assert result["evidence"]["files"] == [{
        "kind": "file", "id": "importer", "file_path": "test_parser.py",
        "stage": "references",
    }]
    assert result["evidence"]["references"] == [{
        "kind": "reference", "id": "importer", "seed_id": "symbol",
        "target_id": "importer", "relationship": "imports",
        "direction": "incoming", "file_path": "test_parser.py",
        "start_line": None, "end_line": None, "stage": "references",
    }]


@pytest.mark.asyncio
async def test_recovery_wait_precedes_classifier_deadline(tmp_path: Path) -> None:
    ready = asyncio.Event()
    routes = FakeRoutes()

    async def wait_until_ready() -> None:
        await ready.wait()

    query = QueryOrchestrator(
        provider=FixedProvider(Playbook.REFERENCES),
        call=routes.call,
        classifier_timeout_seconds=0.01,
        wait_until_ready=wait_until_ready,
    )
    task = asyncio.create_task(
        query.query(
            {
                "goal": "Find references to Parser",
                "repo_path": str(tmp_path),
                "symbol_name": "Parser",
            }
        )
    )
    await asyncio.sleep(0.03)
    assert task.done() is False
    assert routes.calls == []

    ready.set()
    result = await asyncio.wait_for(task, timeout=1)
    assert result["routing"]["playbook"] == Playbook.REFERENCES.value
    assert result["routing"]["fallback_applied"] is False


@pytest.mark.asyncio
async def test_changed_ranges_select_impact_and_join_file_anchors(tmp_path: Path) -> None:
    routes = FakeRoutes()
    query = QueryOrchestrator(provider=FixedProvider(Playbook.DISCOVER), call=routes.call)
    changed = {"file_path": "parser.py", "start_line": 1, "end_line": 2}

    result = await query.query({
        "goal": "review parser", "repo_path": str(tmp_path), "changed_ranges": [changed],
    })

    method, params = routes.calls[0]
    assert method == "knowledge.composite.regression_risk"
    assert params["file_paths"] == ["parser.py"]
    assert params["changed_ranges"] == [changed]
    assert result["routing"]["playbook"] == "IMPACT"


@pytest.mark.asyncio
async def test_changed_ranges_cannot_escape_repository(tmp_path: Path) -> None:
    routes = FakeRoutes()
    query = QueryOrchestrator(provider=FixedProvider(Playbook.DISCOVER), call=routes.call)

    with pytest.raises(ValueError, match="escapes repository"):
        await query.query({
            "goal": "review", "repo_path": str(tmp_path),
            "changed_ranges": [{"file_path": "../outside.py", "start_line": 0, "end_line": 1}],
        })
    assert routes.calls == []


@pytest.mark.asyncio
async def test_contract_evidence_is_returned_with_source_status(tmp_path: Path) -> None:
    (tmp_path / 'contract.md').write_text('Status: approved\nRequired behavior.\n')
    query = QueryOrchestrator(provider=FixedProvider(Playbook.DISCOVER), call=FakeRoutes().call)

    result = await query.query({
        'goal': 'review parser', 'repo_path': str(tmp_path), 'contract_paths': ['contract.md'],
    })

    assert result['evidence']['contracts'][0]['active'] is True
    assert result['evidence']['contracts'][0]['content'].endswith('Required behavior.\n')
    assert result['trace'][-1]['stage'] == 'contracts'


@pytest.mark.asyncio
async def test_missing_reference_anchor_is_grounded_before_traversal(tmp_path: Path) -> None:
    routes = FakeRoutes()
    query = QueryOrchestrator(provider=FixedProvider(Playbook.DISCOVER), call=routes.call)

    result = await query.query({'goal': 'Find callers of parse', 'repo_path': str(tmp_path)})

    assert routes.calls[0][0] == 'knowledge.search'
    assert all(params['node_id'] == 'node-1' for method, params in routes.calls if method == 'knowledge.related')
    assert result['routing']['playbook'] == 'REFERENCES'
    assert result['routing']['anchor_resolution']['status'] == 'resolved'


@pytest.mark.asyncio
async def test_duplicate_symbol_candidates_do_not_trigger_traversal(tmp_path: Path) -> None:
    class AmbiguousRoutes(FakeRoutes):
        async def call(self, method: str, params: dict[str, object]) -> dict[str, object]:
            self.calls.append((method, params))
            assert method == 'knowledge.search'
            return {'results': [
                {'id': f'parse-{index}', 'type': 'function', 'name': 'parse', 'file_path': f'{index}.py'}
                for index in range(2)
            ]}

    routes = AmbiguousRoutes()
    query = QueryOrchestrator(provider=FixedProvider(Playbook.DISCOVER), call=routes.call)
    result = await query.query({'goal': 'Find references to parse', 'repo_path': str(tmp_path)})

    assert result['routing']['anchor_resolution']['status'] == 'ambiguous'
    assert len(result['routing']['anchor_resolution']['candidates']) == 2
    assert result['complete'] is False
    assert len(routes.calls) == 1


@pytest.mark.asyncio
async def test_missing_impact_file_is_resolved_without_second_classification(tmp_path: Path) -> None:
    routes = FakeRoutes()
    provider = FixedProvider(Playbook.DISCOVER)
    query = QueryOrchestrator(provider=provider, call=routes.call)
    result = await query.query({'goal': 'Find affected dependents of parse', 'repo_path': str(tmp_path)})

    assert provider.calls == 1
    assert routes.calls[1][0] == 'knowledge.composite.regression_risk'
    assert routes.calls[1][1]['file_paths'] == ['parser.py']
    assert result['routing']['playbook'] == 'IMPACT'


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['empty', 'unsafe', 'partial', 'full_page'])
async def test_unreliable_discovery_never_becomes_a_reference_anchor(tmp_path: Path, failure: str) -> None:
    class UnreliableRoutes(FakeRoutes):
        async def call(self, method: str, params: dict[str, object]) -> dict[str, object]:
            self.calls.append((method, params))
            assert method == 'knowledge.search'
            nodes = [{'id': 'node', 'name': 'parse', 'type': 'function',
                      'file_path': '../outside.py' if failure == 'unsafe' else 'parser.py'}]
            if failure == 'empty':
                nodes = []
            elif failure == 'full_page':
                nodes *= 10
            return {'results': nodes, 'timed_out': failure == 'partial'}

    routes = UnreliableRoutes()
    query = QueryOrchestrator(provider=FixedProvider(Playbook.DISCOVER), call=routes.call)
    result = await query.query({'goal': 'Find references to parse', 'repo_path': str(tmp_path)})
    assert result['complete'] is False
    assert result['routing']['anchor_resolution']['status'] != 'resolved'
    assert len(routes.calls) == 1


@pytest.mark.asyncio
async def test_discovery_coverage_does_not_claim_test_analysis(tmp_path: Path) -> None:
    query = QueryOrchestrator(provider=FixedProvider(Playbook.DISCOVER), call=FakeRoutes().call)
    result = await query.query({'goal': 'find parser', 'repo_path': str(tmp_path)})

    assert result['investigation']['implementation']['status'] == 'assessed'
    assert result['investigation']['tests']['status'] == 'not_assessed'
    assert result['complete'] is True


@pytest.mark.asyncio
async def test_lsp_unavailability_is_not_an_empty_reference_result(tmp_path: Path) -> None:
    (tmp_path / 'parser.py').write_text('def parse(): pass\n')

    class UnavailableRoutes(FakeRoutes):
        async def call(self, method: str, params: dict[str, object]) -> dict[str, object]:
            return {'available': False, 'references': []}

    query = QueryOrchestrator(provider=FixedProvider(Playbook.REFERENCES), call=UnavailableRoutes().call)
    result = await query.query({
        'goal': 'references', 'repo_path': str(tmp_path),
        'source_position': {'file_path': 'parser.py', 'line': 0},
    })
    assert result['investigation']['references']['status'] == 'unavailable'
    assert result['complete'] is False


@pytest.mark.asyncio
async def test_changed_symbols_do_not_count_as_dependents(tmp_path: Path) -> None:
    class IsolatedRoutes(FakeRoutes):
        async def call(self, method: str, params: dict[str, object]) -> dict[str, object]:
            return {'changed_symbols': [{'id': 'parse', 'type': 'function', 'file_path': 'parser.py'}],
                    'dependents': [], 'test_targets': []}

    query = QueryOrchestrator(provider=FixedProvider(Playbook.IMPACT), call=IsolatedRoutes().call)
    result = await query.query({
        'goal': 'review', 'repo_path': str(tmp_path), 'file_paths': ['parser.py'],
    })
    assert result['investigation']['dependents']['status'] == 'empty'


@pytest.mark.asyncio
async def test_execution_deadline_marks_interrupted_reference_stage_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scs.orchestration.query import MODE_BUDGETS, ModeBudget

    (tmp_path / 'parser.py').write_text('def parse(): pass\n')
    monkeypatch.setitem(MODE_BUDGETS, 'fast', ModeBudget(0.15, 5, 1, 1, 0.01))

    class WaitingRoutes(FakeRoutes):
        async def call(self, method: str, params: dict[str, object]) -> dict[str, object]:
            await asyncio.Event().wait()
            raise AssertionError('unreachable')

    query = QueryOrchestrator(provider=FixedProvider(Playbook.REFERENCES), call=WaitingRoutes().call)
    result = await query.query({
        'goal': 'references', 'repo_path': str(tmp_path), 'mode': 'fast',
        'source_position': {'file_path': 'parser.py', 'line': 0},
    })
    assert result['complete'] is False
    assert result['truncated'] is True
    assert result['investigation']['references']['status'] == 'unavailable'
    assert 'execution' in result['degraded_stages']
