"""Investigation coverage describes execution, not proof of correctness."""

import pytest

from scs.orchestration.coverage import investigation_coverage


def test_discovery_does_not_claim_tests_or_references_were_assessed() -> None:
    report = investigation_coverage(
        {'symbols': [{'id': 'parse'}]},
        [{'stage': 'search', 'status': 'ok', 'truncated': False}],
    )
    assert report['implementation']['status'] == 'assessed'
    assert report['tests']['status'] == 'not_assessed'
    assert report['references']['status'] == 'not_assessed'


@pytest.mark.parametrize('status,expected', [
    ('ok', 'empty'), ('failed', 'unavailable'), ('timeout', 'unavailable'),
])
def test_empty_impact_is_distinct_from_unavailable(status: str, expected: str) -> None:
    report = investigation_coverage({}, [{'stage': 'impact', 'status': status, 'truncated': False}])
    assert report['tests']['status'] == expected


def test_partial_impact_and_unavailable_reference_provider_are_visible() -> None:
    report = investigation_coverage(
        {'test_targets': [{'file_path': 'test.py'}]},
        [{'stage': 'impact', 'status': 'ok', 'truncated': True},
         {'stage': 'references', 'status': 'unavailable', 'truncated': False}],
    )
    assert report['tests']['status'] == 'partial'
    assert report['references']['status'] == 'unavailable'


def test_failed_stage_after_usable_evidence_is_partial() -> None:
    report = investigation_coverage(
        {'references': [{'id': 'caller'}]},
        [{'stage': 'references', 'status': 'ok'}, {'stage': 'references', 'status': 'failed'}],
    )
    assert report['references']['status'] == 'partial'
