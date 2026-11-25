"""Bounded declaration signature and call-site excerpt hydration."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from scs.source_paths import validated_source_path

MAX_SNIPPET_FILES = 50
MAX_FILE_BYTES = 1_048_576
MAX_SIGNATURE_CHARS = 240
MAX_SNIPPET_CHARS = 240
MAX_SNIPPET_LINES = 2


def _compact_text(text: str, *, max_chars: int) -> str:
    collapsed = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


class _BoundedSourceReader:
    """Cache validated line slices within strict per-request file & byte caps."""

    def __init__(self, repo_path: str) -> None:
        self._repo_path: str = repo_path
        self._cache: dict[str, list[str] | None] = {}

    def lines(self, file_path: str) -> list[str] | None:
        if file_path in self._cache:
            return self._cache[file_path]
        if len(self._cache) >= MAX_SNIPPET_FILES:
            return None
        self._cache[file_path] = None
        try:
            safe = Path(
                validated_source_path(file_path, self._repo_path, require_file=True)
            )
            if safe.stat().st_size > MAX_FILE_BYTES:
                return None
            with safe.open("rb") as handle:
                raw = handle.read(MAX_FILE_BYTES + 1)
            if len(raw) > MAX_FILE_BYTES or b"\x00" in raw:
                return None
            lines = raw.decode("utf-8", errors="replace").splitlines()
            self._cache[file_path] = lines
            return lines
        except (OSError, ValueError):
            return None

    def excerpt(
        self, file_path: str, start_line: int, end_line: int | None
    ) -> str | None:
        lines = self.lines(file_path)
        if not lines or start_line < 0 or start_line >= len(lines):
            return None
        stop = start_line if end_line is None or end_line < start_line else end_line
        stop = min(stop, start_line + MAX_SNIPPET_LINES - 1, len(lines) - 1)
        joined = "\n".join(lines[start_line : stop + 1])
        compact = _compact_text(joined, max_chars=MAX_SNIPPET_CHARS)
        return compact or None

    def signature_header(self, file_path: str, start_line: int) -> str | None:
        lines = self.lines(file_path)
        if not lines or start_line < 0 or start_line >= len(lines):
            return None
        collected: list[str] = []
        for idx in range(start_line, min(start_line + 2, len(lines))):
            stripped = lines[idx].strip()
            if not stripped:
                break
            collected.append(stripped)
            if "{" in stripped or stripped.endswith(":") or "=>" in stripped:
                break
        if not collected:
            return None
        header = " ".join(collected)
        if "{" in header:
            header = header.split("{", 1)[0].strip()
        return _compact_text(header, max_chars=MAX_SIGNATURE_CHARS) or None


def hydrate_evidence_text(
    repo_path: str,
    evidence: dict[str, list[dict[str, object]]],
    *,
    include_signatures: bool,
    include_snippets: bool,
) -> None:
    """Attach optional compact signatures and 1–2 line excerpts after limit slicing."""

    if not include_signatures and not include_snippets:
        return

    reader = _BoundedSourceReader(repo_path)

    if include_signatures:
        for bucket in ("symbols", "references", "relationships"):
            for item in evidence.get(bucket, []):
                raw_sig = item.get("signature")
                if isinstance(raw_sig, str) and raw_sig.strip():
                    item["signature"] = _compact_text(
                        raw_sig, max_chars=MAX_SIGNATURE_CHARS
                    )
                elif bucket == "symbols":
                    file_path = item.get("file_path")
                    start_line = item.get("start_line")
                    if isinstance(file_path, str) and type(start_line) is int:
                        header = reader.signature_header(file_path, start_line)
                        if header:
                            item["signature"] = header

    if include_snippets:
        for bucket in ("references", "relationships", "symbols"):
            for item in evidence.get(bucket, []):
                file_path = item.get("file_path")
                start_line = item.get("start_line")
                end_line = item.get("end_line")
                if isinstance(file_path, str) and type(start_line) is int:
                    excerpt = reader.excerpt(
                        file_path,
                        start_line,
                        end_line if type(end_line) is int else None,
                    )
                    if excerpt:
                        item["snippet"] = excerpt
                # Also hydrate first site inside test_targets dependency evidence if present
                if bucket == "relationships":
                    raw_path = item.get("path")
                    if isinstance(raw_path, list):
                        for raw_step in cast(list[object], raw_path):
                            if not isinstance(raw_step, dict):
                                continue
                            step = cast(dict[str, object], raw_step)
                            s_file = step.get("file_path")
                            s_line = step.get("start_line")
                            s_end = step.get("end_line")
                            if isinstance(s_file, str) and type(s_line) is int:
                                excerpt = reader.excerpt(
                                    s_file,
                                    s_line,
                                    s_end if type(s_end) is int else None,
                                )
                                if excerpt:
                                    step["snippet"] = excerpt
