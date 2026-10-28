"""Read explicitly linked Markdown contracts without inferring compliance."""

from __future__ import annotations

import hashlib
from itertools import islice
from pathlib import Path
import re
from typing import TypedDict
from urllib.parse import unquote, urlsplit

from scs.source_paths import validated_source_path

MAX_DOCUMENT_BYTES = 65_536
MAX_SCAN_BYTES = 2_097_152
MAX_SCAN_FILES = 128
MAX_CONTENT_CHARACTERS = 2_048
ACTIVE_STATUSES = frozenset({"active", "accepted", "approved", "implementing", "validating", "shipped"})
LINK = re.compile(r"\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)")
STATUS = re.compile(r"^\s*(?:\*\*)?status:(?:\*\*)?\s*(.*?)\s*$", re.I | re.M)


class ContractCollection(TypedDict):
    contracts: list[dict[str, object]]
    gaps: list[dict[str, object]]
    truncated: bool


def _target(root: Path, owner: str, raw: str) -> str:
    """Resolve local link targets through the shared containment boundary."""

    url = urlsplit(raw)
    if url.scheme or url.netloc or not url.path:
        return ""
    path = unquote(url.path)
    candidate = root / path if path.startswith(("specs/", "decisions/", "docs/")) else root / Path(owner).parent / path
    return Path(validated_source_path(str(candidate), str(root), require_file=False)).relative_to(root).as_posix()


def collect_contracts(
    root: Path, sources: list[str], explicit: list[str], *, limit: int,
) -> ContractCollection:
    """Return linked contracts and truthful gaps within fixed read budgets.

    Source links and document backlinks establish relevance, not compliance.
    Lifecycle labels qualify authority; absent status never becomes active.
    No source bytes are sent to a routing model or persisted in a new store.
    """

    root = root.resolve()
    gaps: list[dict[str, object]] = []
    cache: dict[str, str | None] = {}
    consumed = 0
    truncated = False

    def read(path: str) -> str | None:
        nonlocal consumed, truncated
        if path in cache:
            return cache[path]
        if len(cache) >= MAX_SCAN_FILES or consumed >= MAX_SCAN_BYTES:
            truncated = True
            return None
        cache[path] = None
        try:
            safe = Path(validated_source_path(path, str(root), require_file=False))
            if not safe.is_file():
                raise OSError("not a regular file")
            with safe.open("rb") as stream:
                data = stream.read(MAX_DOCUMENT_BYTES + 1)
            consumed += len(data)
            if len(data) > MAX_DOCUMENT_BYTES:
                gaps.append({"file_path": path, "reason": "oversized"})
                return None
            cache[path] = data.decode("utf-8")
        except ValueError:
            gaps.append({"file_path": path, "reason": "unsafe_path"})
        except (OSError, UnicodeError):
            gaps.append({"file_path": path, "reason": "unreadable"})
        return cache[path]

    selected: dict[str, list[dict[str, object]]] = {
        path: [{"kind": "caller_selected", "file_path": path}] for path in explicit
    }

    def links(owner: str, text: str) -> list[tuple[str, int]]:
        found: list[tuple[str, int]] = []
        for match in LINK.finditer(text):
            try:
                target = _target(root, owner, match.group(1))
                if target:
                    found.append((target, text.count("\n", 0, match.start())))
            except ValueError:
                gaps.append({"file_path": owner, "reason": "unsafe_path"})
        return found

    for source in sources:
        text = read(source)
        if text is None:
            continue
        for target, line in links(source, text):
            if target.lower().endswith(".md"):
                selected.setdefault(target, []).append({"kind": "source_link", "file_path": source, "line": line})

    # Existing records can link to code without requiring new annotations.
    for directory in ("specs", "decisions"):
        folder = root / directory
        try:
            validated_source_path(str(folder), str(root), require_file=False)
        except ValueError:
            gaps.append({"file_path": directory, "reason": "unsafe_path"})
            continue
        for document in islice(folder.glob("*.md"), MAX_SCAN_FILES + 1):
            if len(cache) >= MAX_SCAN_FILES or consumed >= MAX_SCAN_BYTES:
                truncated = True
                break
            relative = document.relative_to(root).as_posix()
            text = read(relative)
            if text is None:
                continue
            for target, line in links(relative, text):
                if target in sources:
                    selected.setdefault(relative, []).append({"kind": "document_link", "file_path": relative, "target": target, "line": line})

    contracts: list[dict[str, object]] = []
    for path, relevance in sorted(selected.items()):
        if len(contracts) >= limit:
            truncated = True
            break
        text = read(path)
        if text is None:
            continue
        status_match = STATUS.search(text)
        status = status_match.group(1).strip("'\" ").lower() if status_match else "unknown"
        successors: list[str] = []
        successor_match = re.search(r"^superseded-by:[ \t]*([^\n]*)\n((?:[ \t]+[^\n]*\n)*)", text, re.M)
        if successor_match:
            raw_successors = [successor_match.group(1), *successor_match.group(2).splitlines()]
            for raw in raw_successors:
                raw = raw.strip().removeprefix("- ").split(" (", 1)[0].strip("'\" ")
                if not raw or raw in {"[]", "null"}:
                    continue
                try:
                    target = _target(root, path, raw)
                    if target:
                        successors.append(target)
                except ValueError:
                    gaps.append({"file_path": path, "reason": "unsafe_successor"})
                    successors.append("unresolved")
        contracts.append({
            "kind": "contract", "file_path": path, "status": status,
            "active": status in ACTIVE_STATUSES and not successors,
            "superseded_by": successors, "links": relevance,
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "content": text[:MAX_CONTENT_CHARACTERS],
            "content_truncated": len(text) > MAX_CONTENT_CHARACTERS,
        })
    return {"contracts": contracts, "gaps": gaps, "truncated": truncated}
