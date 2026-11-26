"""Explicit document links preserve authority, lifecycle, and containment."""

from pathlib import Path

from scs.services.contracts import MAX_DOCUMENT_BYTES, collect_contracts


def test_linked_contracts_preserve_lifecycle_and_source_evidence(tmp_path: Path) -> None:
    (tmp_path / 'specs').mkdir()
    (tmp_path / 'source.py').write_text('# [contract](specs/current.md)\n')
    (tmp_path / 'specs/current.md').write_text('---\nstatus: approved\n---\n# Current contract\n')
    (tmp_path / 'specs/old.md').write_text('---\nstatus: shipped\nsuperseded-by:\n  - current.md\n---\n[source](../source.py)\n')

    result = collect_contracts(tmp_path, ['source.py'], [], limit=10)
    documents = {item['file_path']: item for item in result['contracts']}

    assert documents['specs/current.md']['active'] is True
    assert documents['specs/current.md']['status'] == 'approved'
    assert documents['specs/old.md']['active'] is False
    assert documents['specs/old.md']['superseded_by'] == ['specs/current.md']
    assert documents['specs/current.md']['links'][0]['file_path'] == 'source.py'
    assert documents['specs/current.md']['links'][0]['line'] == 0


def test_unknown_status_and_unlinked_documents_are_not_active(tmp_path: Path) -> None:
    (tmp_path / 'specs').mkdir()
    for name in ('selected', 'unlinked'):
        (tmp_path / f'specs/{name}.md').write_text(f'# {name}\n')
    result = collect_contracts(tmp_path, [], ['specs/selected.md'], limit=10)

    assert len(result['contracts']) == 1
    assert result['contracts'][0]['status'] == 'unknown'
    assert result['contracts'][0]['active'] is False


def test_escaping_symlink_and_missing_links_are_reported(tmp_path: Path) -> None:
    repo = tmp_path / 'repo'
    repo.mkdir()
    (tmp_path / 'outside.md').write_text('Status: active\nSECRET\n')
    (repo / 'escape.md').symlink_to(tmp_path / 'outside.md')
    (repo / 'source.py').write_text('# [escape](escape.md) [missing](missing.md)\n')
    result = collect_contracts(repo, ['source.py'], [], limit=10)

    assert result['contracts'] == []
    assert {item['reason'] for item in result['gaps']} == {'unsafe_path', 'unreadable'}
    assert 'SECRET' not in str(result)


def test_oversized_document_and_output_limit_remain_visible(tmp_path: Path) -> None:
    (tmp_path / 'large.md').write_text('x' * (MAX_DOCUMENT_BYTES + 1024))
    result = collect_contracts(tmp_path, [], ['large.md'], limit=1)
    assert result['gaps'][0]['reason'] == 'oversized'

    for index in range(2):
        (tmp_path / f'{index}.md').write_text('Status: active\n')
    result = collect_contracts(tmp_path, [], ['0.md', '1.md'], limit=1)
    assert len(result['contracts']) == 1
    assert result['truncated'] is True


def test_empty_successor_field_does_not_consume_following_metadata(tmp_path: Path) -> None:
    (tmp_path / 'contract.md').write_text(
        '---\nstatus: approved\nsuperseded-by:\nimplementation:\n  commits: []\n---\n'
    )
    result = collect_contracts(tmp_path, [], ['contract.md'], limit=1)
    assert result['contracts'][0]['active'] is True
    assert result['contracts'][0]['superseded_by'] == []


def test_external_successor_still_disqualifies_active_authority(tmp_path: Path) -> None:
    (tmp_path / 'old.md').write_text('---\nstatus: shipped\nsuperseded-by: https://example.invalid/new\n---\n')
    result = collect_contracts(tmp_path, [], ['old.md'], limit=1)
    assert result['contracts'][0]['active'] is False


def test_explicit_roots_keep_read_budget_and_follow_supersession(tmp_path: Path) -> None:
    (tmp_path / 'specs').mkdir()
    for index in range(130):
        (tmp_path / f'specs/unrelated-{index}.md').write_text('Status: proposed\n')
    (tmp_path / 'old.md').write_text('---\nstatus: shipped\nsuperseded-by: current.md\n---\n')
    (tmp_path / 'current.md').write_text('Status: approved\n')

    result = collect_contracts(tmp_path, [], ['old.md'], limit=1)

    assert result['contracts'][0]['file_path'] == 'current.md'
    assert result['contracts'][0]['active'] is True
    assert result['truncated'] is True


def test_large_code_sources_do_not_trigger_oversized_contract_gaps(tmp_path: Path) -> None:
    (tmp_path / 'asset-management-door.js').write_text('x' * 75_000, encoding='utf-8')
    (tmp_path / 'spec.md').write_text('y' * (MAX_DOCUMENT_BYTES + 1024), encoding='utf-8')

    source_result = collect_contracts(tmp_path, ['asset-management-door.js'], [], limit=10)
    assert source_result['gaps'] == []

    doc_result = collect_contracts(tmp_path, [], ['spec.md'], limit=10)
    assert doc_result['gaps'] == [{'file_path': 'spec.md', 'reason': 'oversized'}]


def test_large_spec_catalog_and_cross_repo_links_do_not_degrade_inspect(tmp_path: Path) -> None:
    repo = tmp_path / 'repo'
    (repo / 'specs').mkdir(parents=True)
    (repo / 'decisions').mkdir(parents=True)
    (repo / 'src').mkdir(parents=True)
    (repo / 'src/sessions-wire.ts').write_text('export function loadSessions() {}\n', encoding='utf-8')

    # Spec >64KB (within 256KB) and unrelated spec with cross-repo links must not degrade.
    (repo / 'specs/001-large-spec.md').write_text(
        '---\nstatus: approved\n---\n' + ('a' * 90_000) + '\n[wire](../src/sessions-wire.ts)\n',
        encoding='utf-8',
    )
    (repo / 'specs/002-cross-repo.md').write_text(
        '---\nstatus: approved\n---\n[other](../../../other-repo/spec.md) [abs](/tmp/outside.ts)\n',
        encoding='utf-8',
    )
    # Large catalog (>128 docs) still finds a decision backlink without premature truncation.
    for index in range(140):
        (repo / f'specs/unrelated-{index:03d}.md').write_text('Status: proposed\n', encoding='utf-8')
    (repo / 'decisions/200-wire-decision.md').write_text(
        '---\nstatus: accepted\n---\nSee [wire](../src/sessions-wire.ts).\n',
        encoding='utf-8',
    )

    result = collect_contracts(repo, ['src/sessions-wire.ts'], [], limit=10)

    assert result['gaps'] == []
    assert result['truncated'] is False
    paths = {item['file_path'] for item in result['contracts']}
    assert paths == {'specs/001-large-spec.md', 'decisions/200-wire-decision.md'}

    # Escaping symlink inside specs/ must still report unsafe_path.
    (tmp_path / 'secret.md').write_text('Status: active\nSECRET\n', encoding='utf-8')
    (repo / 'specs/999-escape.md').symlink_to(tmp_path / 'secret.md')
    escaped = collect_contracts(repo, ['src/sessions-wire.ts'], [], limit=10)
    assert any(
        gap['file_path'] == 'specs/999-escape.md' and gap['reason'] == 'unsafe_path'
        for gap in escaped['gaps']
    )
    assert 'SECRET' not in str(escaped)
