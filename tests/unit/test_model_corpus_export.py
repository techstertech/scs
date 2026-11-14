"""Model comparisons must use one immutable, source-backed corpus."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
import subprocess
import sys
from typing import Protocol, TypedDict, cast

import pytest

from scs.indexing.parser.native import NativeParser


class _Document(TypedDict):
    file_path: str
    kind: str
    qualified_name: str
    embedding_text: str
    rerank_text: str
    start_line: int


class _Corpus(TypedDict):
    git_revision: str
    file_count: int
    document_count: int
    documents: list[_Document]
    skipped_entries: dict[str, int]
    corpus_sha256: str


class Exporter(Protocol):
    """Expose the script's test surface without untyped module attributes."""

    NativeParser: type[NativeParser]

    def export_corpus(self, repo: Path, revision: str = "HEAD") -> _Corpus: ...
    def main(self, argv: Sequence[str] | None = None) -> int: ...


def exporter() -> Exporter:
    path = Path(__file__).parents[2] / "scripts" / "export-model-corpus.py"
    spec = importlib.util.spec_from_file_location("export_model_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(Exporter, cast(object, module))


def git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--quiet")
    git(repo, "config", "user.name", "Corpus Test")
    git(repo, "config", "user.email", "corpus@example.invalid")
    files = {
        "src/worker.py": 'def work(value: int) -> int:\n    """Return one more."""\n    return value + 1\n',
        "README.md": "# Example\n\nA generated benchmark repository.\n",
        ".gitignore": "tracked.txt\n",
        ".gitattributes": "README.md export-ignore\n",
        "tracked.txt": "This tracked file remains part of the snapshot.\n",
        "node_modules/dependency.py": "dependency = True\n",
    }
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (repo / "binary.dat").write_bytes(bytes(range(256)))
    (repo / "alias.py").symlink_to("src/worker.py")
    git(repo, "add", "--force", ".")
    git(repo, "commit", "--quiet", "-m", "test: create corpus fixture")
    return repo


def test_export_uses_commit_bytes_and_production_entity_text(tmp_path: Path) -> None:
    module = exporter()
    repo = repository(tmp_path)
    revision = git(repo, "rev-parse", "HEAD")

    first = module.export_corpus(repo, revision)
    (repo / "src/worker.py").write_text("replacement = True\n", encoding="utf-8")
    (repo / "untracked.py").write_text("untracked = True\n", encoding="utf-8")
    second = module.export_corpus(repo, revision)

    assert first == second
    assert first["git_revision"] == revision
    assert first["file_count"] == 5
    assert first["document_count"] == len(first["documents"])
    paths = {document["file_path"] for document in first["documents"]}
    assert paths == {
        ".gitattributes",
        ".gitignore",
        "README.md",
        "src/worker.py",
        "tracked.txt",
    }
    function = next(item for item in first["documents"] if item["kind"] == "function")
    assert function["qualified_name"] == "src.worker.work"
    assert "Return one more." in function["embedding_text"]
    assert "return value + 1" in function["rerank_text"]
    assert function["start_line"] == 0
    assert first["skipped_entries"]["symlink"] == 1
    assert (repo / "src/worker.py").read_text() == "replacement = True\n"
    assert git(repo, "diff", "--cached", "--stat") == ""

    serialized = json.dumps(first["documents"], sort_keys=True, separators=(",", ":"))
    assert first["corpus_sha256"] == hashlib.sha256(serialized.encode()).hexdigest()


def test_invalid_revision_cannot_leave_an_old_success_report(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    output = tmp_path / "corpus.json"
    output.write_text('{"schema_version": 1, "documents": []}\n', encoding="utf-8")
    script = Path(__file__).parents[2] / "scripts" / "export-model-corpus.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo",
            str(repo),
            "--revision",
            "invalid-revision",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode != 0
    assert not output.exists()
    assert "invalid-revision" in result.stderr


def test_parser_failure_does_not_return_a_partial_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = exporter()
    repo = repository(tmp_path)

    def fail(*_arguments: object) -> object:
        raise RuntimeError("parser failed")

    monkeypatch.setattr(module.NativeParser, "parse", fail)

    with pytest.raises(RuntimeError, match="parser failed"):
        module.export_corpus(repo)


def test_output_cannot_overwrite_repository_source(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    source = repo / "src/worker.py"
    original = source.read_bytes()
    module = exporter()

    with pytest.raises(ValueError, match="outside"):
        module.main(["--repo", str(repo), "--output", str(source)])

    assert source.read_bytes() == original
