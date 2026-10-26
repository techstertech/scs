---
status: implemented
project: scs
project-root: /Users/leonardo/Repositories/techstertech/scs
created: 2026-10-26
updated: 2026-10-26
owner: code-intelligence
---

# Independent nested Git repositories

## Outcome and approval

The user selected normal directory traversal with a stop at nested Git
repositories, which must be ingested only through separate explicit requests.
A `.git` directory or file marks a repository root, including submodules and
linked checkouts. No nested repository is automatically enrolled.

This replaces only the blanket nested-root denial described in
[the coverage contract](2026-08-05-expand-risk-based-test-coverage.md).
The home-directory rejection and normal subdirectory overlap rejection remain.

## Contracts and scope

Full discovery, filesystem fallback, and individual-file ingestion exclude
files beneath nested Git roots. Source aliases cannot bypass the lexical or
resolved repository boundary. Filesystem fallback prunes boundaries before
entering them and does not follow directory symlinks. Normal source folders
remain traversable. The selected root's own `.git` marker does not exclude
its source; Git metadata itself is never ingested.

Explicit ingestion of a nested Git root is permitted with its parent indexed,
regardless of ingestion order. Existing reconciliation removes parent-owned
files which become excluded; no automatic destructive cleanup or enrollment
is introduced. Existing indexes require reconciliation to apply the new rule.
No MCP schema, dependency, or persisted-data format changes occur.

## Acceptance criteria

1. Parent discovery returns normal-folder sources and excludes nested-root
   sources for Git listings, fallback traversal, and library traversal.
2. Fallback never enters a nested repository or follows a directory-symlink cycle.
3. Individual-file ingestion rejects nested sources and aliases across the
   boundary, but accepts those sources when the nested root is selected.
4. Parent and child roots index independently in either order. Normal overlapping
   subdirectories and the user home remain rejected.
5. Reconciliation removes previously parent-owned nested files without removing
   child-owned files.

## Verification plan

Generate directory/file Git markers and source fixtures in temporary folders.
Exercise candidate filtering, actual fallback traversal, aliases, enrollment,
and pipeline reconciliation. Run focused tests first, then `just verify`.

## Recovery and limitations

Revert the task commit to restore previous behavior; no data migration is needed.
The prior code may index nested sources under the parent again. Existing indexes
must reconcile before the exclusion is reflected in stored results.

## Verification results

All five acceptance criteria passed in generated unit/integration fixtures.

- Criteria 1 and 3: tests cover Git candidate lists, filesystem fallback,
  library traversal, `.git` directory/file markers, and aliases on either side
  of the boundary. Selecting the child root accepts its own source.
- Criterion 2: an instrumented real filesystem scan permits entry only into
  the parent and normal source directory, with a nested repository and symlink
  cycle present. No nested root or directory alias is entered.
- Criterion 4: pipeline tests ingest both roots in each order and verify
  disjoint stored file sets. Existing home and ordinary-folder overlap tests
  remain passing.
- Criterion 5: a pipeline test indexes an ordinary folder, adds a nested Git
  marker, ingests the child separately, and reconciles the parent. Exactly the
  parent-owned nested file is removed; the child-owned file remains.

The initial ten regression cases failed before implementation. Focused checks
passed 52 tests. `RUSTUP_TOOLCHAIN=1.98 just verify` then passed strict typing,
Ruff, all 453 Python tests, 108 Rust tests, and 87.44% branch-aware coverage.
`git diff --check` passed. No external services or production indexes were
modified. Publication and installed-package changes remain outside this unit.

Review shared source boundaries first, discovery and enrollment second, then
the generated tests and indexing guidance. No plan deviations or additional
runtime dependencies were introduced.
