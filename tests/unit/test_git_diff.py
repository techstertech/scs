from __future__ import annotations

from pathlib import Path
import subprocess

from scs.indexing.git_diff import (
    DiffRange,
    collect_git_diff_ranges,
    collect_git_dirty_paths,
)


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return proc.stdout


def _init_repo(repo: Path) -> None:
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")


def test_non_git_directory_degrades_gracefully(tmp_path: Path) -> None:
    non_git = tmp_path / "plain_dir"
    non_git.mkdir()
    (non_git / "app.ts").write_text("export const x = 1;\n", encoding="utf-8")

    result = collect_git_diff_ranges(non_git)

    assert result.ranges == []
    assert result.file_paths == []
    assert result.truncated is False
    assert result.degraded_reason == "git_diff_unavailable"
    assert collect_git_dirty_paths(non_git) == []


def test_clean_git_repository_has_empty_diff(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "mod.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
    _git(repo, "add", "mod.py")
    _git(repo, "commit", "-m", "init")

    result = collect_git_diff_ranges(repo)

    assert result.ranges == []
    assert result.file_paths == []
    assert result.truncated is False
    assert result.degraded_reason is None
    assert collect_git_dirty_paths(repo) == []


def test_unborn_head_handles_staged_and_untracked_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "staged.ts").write_text("export const a = 1;\nexport const b = 2;\n", encoding="utf-8")
    _git(repo, "add", "staged.ts")
    (repo / "untracked.ts").write_text("export const c = 3;\n", encoding="utf-8")

    result = collect_git_diff_ranges(repo)
    dirty = collect_git_dirty_paths(repo)

    assert result.degraded_reason is None
    assert result.truncated is False
    assert set(result.file_paths) == {"staged.ts", "untracked.ts"}
    assert DiffRange("staged.ts", 0, 1) in result.ranges
    assert DiffRange("untracked.ts", 0, 0) in result.ranges
    assert set(dirty) == {"staged.ts", "untracked.ts"}


def test_unstaged_staged_deletion_and_untracked_ranges(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "unstaged.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    (repo / "staged.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    (repo / "deleted_lines.py").write_text(
        "line0 = 0\nline1 = 1\nline2 = 2\nline3 = 3\n", encoding="utf-8"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")

    # Unstaged edit on line index 1 (1-based line 2)
    (repo / "unstaged.py").write_text("a = 1\nb = 20\nc = 3\n", encoding="utf-8")
    # Staged edit on line index 1
    (repo / "staged.py").write_text("x = 1\ny = 200\n", encoding="utf-8")
    _git(repo, "add", "staged.py")
    # Pure deletion hunk (count == 0 in new file header)
    (repo / "deleted_lines.py").write_text("line0 = 0\nline3 = 3\n", encoding="utf-8")
    # Untracked file
    (repo / "fresh.ts").write_text("export function f() {}\nexport const g = 1;\n", encoding="utf-8")

    result = collect_git_diff_ranges(repo)
    dirty = collect_git_dirty_paths(repo)

    assert result.degraded_reason is None
    assert result.truncated is False
    assert set(result.file_paths) == {
        "unstaged.py",
        "staged.py",
        "deleted_lines.py",
        "fresh.ts",
    }
    assert DiffRange("unstaged.py", 1, 1) in result.ranges
    assert DiffRange("staged.py", 1, 1) in result.ranges
    assert any(
        r.file_path == "deleted_lines.py" and 0 <= r.start_line <= r.end_line
        for r in result.ranges
    )
    assert DiffRange("fresh.ts", 0, 1) in result.ranges
    assert set(dirty) == {
        "unstaged.py",
        "staged.py",
        "deleted_lines.py",
        "fresh.ts",
    }


def test_top_of_file_deletion_clamps_line_and_limits_truncate(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "top_delete.py").write_text("first = 1\nsecond = 2\nthird = 3\n", encoding="utf-8")
    for idx in range(5):
        (repo / f"f_{idx}.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")

    # Delete every line -> git produces @@ -1,3 +0,0 @@
    (repo / "top_delete.py").write_text("", encoding="utf-8")
    top_result = collect_git_diff_ranges(repo)
    assert any(
        r.file_path == "top_delete.py" and r.start_line == 0 and r.end_line == 0
        for r in top_result.ranges
    )

    for idx in range(5):
        (repo / f"f_{idx}.py").write_text("x = 2\n", encoding="utf-8")
    bounded = collect_git_diff_ranges(repo, max_files=2, max_ranges=2)
    assert bounded.truncated is True
    assert len(bounded.file_paths) <= 2
    assert len(bounded.ranges) <= 2

