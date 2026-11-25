"""Read-only Git working-tree diff and dirty-path helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess

from scs.source_paths import validated_source_path

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_LINE_COUNT_LIMIT_BYTES = 1_048_576
_PORCELAIN_PATH_OFFSET = 3
_PORCELAIN_TWO_PATH_STATUSES = (ord("R"), ord("C"))


@dataclass(frozen=True, slots=True)
class DiffRange:
    """Zero-based inclusive changed line range extracted from Git diff."""

    file_path: str
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class GitDiffResult:
    """Bounded working-tree changed ranges extracted safely via Git."""

    ranges: list[DiffRange]
    file_paths: list[str]
    truncated: bool
    degraded_reason: str | None = None


def _run_git(
    repo_path: Path, args: list[str], *, timeout_seconds: float
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_seconds,
    )


def _safe_rel_path(repo_root: Path, candidate: str) -> str | None:
    if not candidate or candidate == "/dev/null":
        return None
    clean = candidate[2:] if candidate.startswith("b/") else candidate
    if not clean:
        return None
    try:
        resolved = validated_source_path(clean, str(repo_root), require_file=False)
        return Path(resolved).relative_to(repo_root).as_posix()
    except ValueError:
        return None


def _parse_unified_diff(
    diff_text: str,
    repo_root: Path,
    ranges: list[DiffRange],
    file_paths: list[str],
    seen_files: set[str],
    *,
    max_files: int,
    max_ranges: int,
) -> bool:
    truncated = False
    current_rel: str | None = None
    skip_current = False

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            raw_target = line[4:].strip()
            rel = _safe_rel_path(repo_root, raw_target)
            current_rel = rel
            skip_current = rel is None
            if rel is not None and rel not in seen_files:
                if len(file_paths) >= max_files:
                    truncated = True
                    skip_current = True
                else:
                    seen_files.add(rel)
                    file_paths.append(rel)
            continue
        if line.startswith("--- "):
            continue
        if skip_current or current_rel is None:
            continue
        match = _HUNK_RE.match(line)
        if not match:
            continue
        if len(ranges) >= max_ranges:
            truncated = True
            break
        start_one = int(match.group(1))
        count = int(match.group(2)) if match.group(2) is not None else 1
        start_zero = max(0, start_one - 1)
        end_zero = start_zero if count <= 1 else start_zero + count - 1
        ranges.append(
            DiffRange(
                file_path=current_rel,
                start_line=start_zero,
                end_line=end_zero,
            )
        )
    return truncated


def collect_git_dirty_paths(
    repo_path: Path,
    *,
    timeout_seconds: float = 3.0,
    max_files: int = 20,
) -> list[str]:
    """Return bounded repository-relative paths dirty in Git working tree."""

    root = repo_path.resolve()
    try:
        proc = _run_git(
            root,
            ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
            timeout_seconds=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []

    paths: list[str] = []
    seen: set[str] = set()
    records = iter(proc.stdout.split(b"\0"))
    for record in records:
        if not record:
            continue
        if len(record) <= _PORCELAIN_PATH_OFFSET or record[2:3] != b" ":
            continue
        raw_path = os.fsdecode(record[_PORCELAIN_PATH_OFFSET:])
        rel = _safe_rel_path(root, raw_path)
        if rel is not None and rel not in seen:
            seen.add(rel)
            paths.append(rel)
            if len(paths) >= max_files:
                break
        if any(code in _PORCELAIN_TWO_PATH_STATUSES for code in record[:2]):
            _ = next(records, None)
    return paths


def collect_git_diff_ranges(
    repo_path: Path,
    *,
    timeout_seconds: float = 5.0,
    max_files: int = 20,
    max_ranges: int = 100,
) -> GitDiffResult:
    """Compute bounded zero-based ChangedRange items from current Git working-tree changes."""

    root = repo_path.resolve()
    try:
        head = _run_git(root, ["rev-parse", "--verify", "HEAD"], timeout_seconds=timeout_seconds)
    except subprocess.TimeoutExpired:
        return GitDiffResult([], [], True, "git_diff_timeout")
    except (OSError, subprocess.SubprocessError):
        return GitDiffResult([], [], False, "git_diff_unavailable")

    ranges: list[DiffRange] = []
    file_paths: list[str] = []
    seen_files: set[str] = set()
    truncated = False

    try:
        if head.returncode == 0:
            diff_args = [["diff", "--unified=0", "--no-color", "--no-ext-diff", "HEAD"]]
        else:
            # Unborn HEAD repo: inspect staged and unstaged changes separately.
            diff_args = [
                ["diff", "--unified=0", "--no-color", "--no-ext-diff", "--cached"],
                ["diff", "--unified=0", "--no-color", "--no-ext-diff"],
            ]
        for args in diff_args:
            proc = _run_git(root, args, timeout_seconds=timeout_seconds)
            if proc.returncode != 0:
                return GitDiffResult([], [], False, "git_diff_unavailable")
            diff_text = proc.stdout.decode("utf-8", errors="replace")
            if _parse_unified_diff(
                diff_text,
                root,
                ranges,
                file_paths,
                seen_files,
                max_files=max_files,
                max_ranges=max_ranges,
            ):
                truncated = True
                break

        untracked = _run_git(
            root,
            ["ls-files", "--others", "--exclude-standard", "-z"],
            timeout_seconds=timeout_seconds,
        )
        if untracked.returncode == 0 and not truncated:
            for raw in untracked.stdout.split(b"\0"):
                if not raw:
                    continue
                rel = _safe_rel_path(root, os.fsdecode(raw))
                if rel is None:
                    continue
                if rel not in seen_files:
                    if len(file_paths) >= max_files:
                        truncated = True
                        break
                    seen_files.add(rel)
                    file_paths.append(rel)
                if len(ranges) >= max_ranges:
                    truncated = True
                    break
                full_path = root / rel
                if full_path.is_file():
                    try:
                        with full_path.open("rb") as handle:
                            data = handle.read(_LINE_COUNT_LIMIT_BYTES + 1)
                        line_count = data[:_LINE_COUNT_LIMIT_BYTES].count(b"\n") + (
                            1 if data and not data.endswith(b"\n") else 0
                        )
                        end_line = max(0, line_count - 1)
                    except OSError:
                        end_line = 0
                    ranges.append(
                        DiffRange(file_path=rel, start_line=0, end_line=end_line)
                    )
    except subprocess.TimeoutExpired:
        return GitDiffResult(ranges, file_paths, True, "git_diff_timeout")
    except (OSError, subprocess.SubprocessError):
        return GitDiffResult([], [], False, "git_diff_unavailable")

    return GitDiffResult(ranges=ranges, file_paths=file_paths, truncated=truncated)
