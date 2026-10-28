"""Explicit document links preserve authority, lifecycle, and containment."""

from pathlib import Path

from scs.services.contracts import collect_contracts


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
    (tmp_path / 'large.md').write_text('x' * 70_000)
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
