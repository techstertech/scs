#!/usr/bin/env python3
"""Export commit-bound model inputs without reading a live SCS index.

The temporary checkout contains regular Git blobs only. Production discovery,
parsing, embedding text, and reranking text owners define the corpus. Symlinks
and submodules are counted but excluded; their targets are not commit-bound.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Final, TypedDict, cast

from scs import __version__
from scs.graph.models import Node, NodeType
from scs.indexing.discovery import IngestionPolicy, discover
from scs.indexing.parser.base import ParsedEntity
from scs.indexing.parser.native import NativeParser
from scs.indexing.pipeline import RAW_TEXT_LIMIT
from scs.indexing.search import CodeSearchService

REGULAR_MODES: Final[frozenset[bytes]] = frozenset({b"100644", b"100755"})
GIT_TIMEOUT_SECONDS: Final[float] = 60.0


class Document(TypedDict):
    """One model input with enough identity to inspect relevance judgments."""

    id: str
    qualified_name: str
    file_path: str
    kind: str
    name: str
    start_line: int
    end_line: int
    signature: str
    content: str
    embedding_text: str
    rerank_text: str


class CorpusFile(TypedDict):
    """One successfully parsed or text-indexed committed source file."""

    file_path: str
    blob_oid: str
    content_sha256: str
    language: str
    entity_count: int


class Corpus(TypedDict):
    """Deterministic provenance and document payload for local model trials."""

    schema_version: int
    git_revision: str
    scs_version: str
    corpus_sha256: str
    source_sha256: str
    file_count: int
    document_count: int
    skipped_entries: dict[str, int]
    files: list[CorpusFile]
    documents: list[Document]


def _git(repo: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=GIT_TIMEOUT_SECONDS,
    ).stdout


def _digest(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _materialize(
    repo: Path, revision: str, snapshot: Path
) -> tuple[dict[str, str], dict[str, int]]:
    """Read raw blobs rather than archive rules or mutable worktree bytes."""

    records = _git(repo, "ls-tree", "-rz", "--long", "--full-tree", revision)
    blobs: list[tuple[str, str, int]] = []
    index_records: list[bytes] = []
    skipped = {"symlink": 0, "submodule": 0, "other": 0}

    for record in records.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        mode, kind, object_id, byte_size = metadata.split()
        if mode not in REGULAR_MODES or kind != b"blob":
            category = {b"120000": "symlink", b"160000": "submodule"}.get(mode, "other")
            skipped[category] += 1
            continue

        relative = encoded_path.decode("utf-8", errors="surrogateescape")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
            raise ValueError(f"unsafe committed path: {relative}")
        blobs.append((relative, object_id.decode("ascii"), int(byte_size)))
        index_records.append(mode + b" " + object_id + b"\t" + encoded_path + b"\0")

    # Size-only placeholders retain production directory-pruning inputs while
    # preventing an oversized committed artifact from entering model memory.
    policy = IngestionPolicy()
    bounded = [item for item in blobs if item[2] <= policy.max_file_bytes]
    requested = "".join(f"{object_id}\n" for _, object_id, _ in bounded).encode("ascii")
    response = BytesIO(_git(repo, "cat-file", "--batch", input_bytes=requested))
    for relative, object_id, byte_size in bounded:
        header = response.readline().split()
        if header != [
            object_id.encode("ascii"),
            b"blob",
            str(byte_size).encode("ascii"),
        ]:
            raise RuntimeError(f"invalid Git blob response for {relative}")
        contents = response.read(byte_size)
        if len(contents) != byte_size or response.read(1) != b"\n":
            raise RuntimeError(f"incomplete Git blob response for {relative}")
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)

    for relative, _, byte_size in blobs:
        if byte_size <= policy.max_file_bytes:
            continue
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            handle.truncate(byte_size)

    # An index is sufficient for tracked-file discovery. update-index avoids
    # clean filters and hooks; no source-repository metadata is changed.
    _git(snapshot, "init", "--quiet", "--template=")
    _git(
        snapshot,
        "update-index",
        "-z",
        "--index-info",
        input_bytes=b"".join(index_records),
    )
    return {relative: object_id for relative, object_id, _ in blobs}, skipped


def _document(entity: ParsedEntity, file_path: str, occurrence: int) -> Document:
    identity = f"{file_path}:{entity.kind.value}:{entity.qualified_name}:{occurrence}"
    node = Node(
        id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32],
        type=entity.kind,
        name=entity.name,
        content=entity.raw_text[:RAW_TEXT_LIMIT],
        metadata={
            "qualified_name": entity.qualified_name,
            "signature": entity.signature,
            "file_path": file_path,
        },
    )
    return Document(
        id=node.id,
        qualified_name=entity.qualified_name,
        file_path=file_path,
        kind=entity.kind.value,
        name=entity.name,
        start_line=entity.start_line,
        end_line=entity.end_line,
        signature=entity.signature,
        content=node.content,
        embedding_text=entity.embed_text(),
        # Reuse production formatting without publishing a research-only API.
        rerank_text=CodeSearchService._rerank_document(node),  # pyright: ignore[reportPrivateUsage]
    )


def export_corpus(repo: Path, revision: str = "HEAD") -> Corpus:
    """Export one immutable revision; parser failures abort the whole export."""

    commit = (
        _git(repo.resolve(), "rev-parse", "--verify", f"{revision}^{{commit}}")
        .decode()
        .strip()
    )
    parser = NativeParser()
    documents: list[Document] = []
    files: list[CorpusFile] = []

    with TemporaryDirectory(prefix="scs-model-corpus-") as directory:
        snapshot = Path(directory)
        object_ids, skipped = _materialize(repo.resolve(), commit, snapshot)
        entries = discover(snapshot, extensions=parser.supported_extensions())

        for entry in entries:
            source_bytes = entry.abs_path.read_bytes()
            source = source_bytes.decode("utf-8", errors="replace")
            if hashlib.sha256(source_bytes).hexdigest() != entry.content_hash:
                raise RuntimeError(
                    f"snapshot changed after discovery: {entry.rel_path}"
                )
            if entry.language == "text":
                entities = [
                    ParsedEntity(
                        kind=NodeType.FILE,
                        name=entry.rel_path,
                        qualified_name=entry.rel_path,
                        start_line=0,
                        end_line=source.count("\n"),
                        docstring=source,
                        raw_text=source,
                    )
                ]
            else:
                entities, _ = parser.parse(source, entry.rel_path)

            occurrences: dict[tuple[NodeType, str], int] = {}
            for entity in entities:
                identity = (entity.kind, entity.qualified_name)
                occurrence = occurrences.get(identity, 0)
                occurrences[identity] = occurrence + 1
                documents.append(_document(entity, entry.rel_path, occurrence))
            files.append(
                CorpusFile(
                    file_path=entry.rel_path,
                    blob_oid=object_ids[entry.rel_path],
                    content_sha256=entry.content_hash,
                    language=entry.language,
                    entity_count=len(entities),
                )
            )

    return Corpus(
        schema_version=1,
        git_revision=commit,
        scs_version=__version__,
        corpus_sha256=_digest(documents),
        source_sha256=_digest(files),
        file_count=len(files),
        document_count=len(documents),
        skipped_entries=skipped,
        files=files,
        documents=documents,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--repo", type=Path, default=Path.cwd())
    arguments.add_argument("--revision", default="HEAD")
    arguments.add_argument("--output", type=Path, required=True)
    parsed = arguments.parse_args(argv)
    repo = Path(
        _git(cast(Path, parsed.repo).resolve(), "rev-parse", "--show-toplevel")
        .decode()
        .strip()
    ).resolve()
    output = cast(Path, parsed.output).resolve()
    if output.is_relative_to(repo):
        raise ValueError("corpus output must be outside the source repository")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    corpus = export_corpus(repo, cast(str, parsed.revision))
    output.write_text(
        json.dumps(corpus, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Exported {corpus['document_count']} documents from {corpus['file_count']} files at {corpus['git_revision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
