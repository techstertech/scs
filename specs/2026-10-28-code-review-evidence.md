---
status: implementing
project: scs
project-root: /Users/leonardo/Repositories/scs
created: 2026-10-28
updated: 2026-10-28
owner: code-intelligence
supersedes:
  - specs/2026-09-22-unified-query-orchestration.md (additive review inputs and evidence only)
superseded-by:
implementation:
  commits: []
---

# Code review evidence

## Outcome

Agents can review supplied file paths and optional changed-line ranges, identify
supported test candidates, retrieve explicitly linked contracts, resolve missing
anchors, and distinguish assessed work from missing evidence.

The user approved optional ranges and additive extensions to `query_code`.
Implementation proceeds in four separately tested and committed units.

## Contracts

- Keep the five-tool inventory, existing request defaults, and existing result fields.
- Add `changed_ranges`: up to 100 repository-contained `{file_path, start_line,
  end_line}` records. Lines are zero-based and inclusive, matching indexed locations.
  Range paths join `file_paths`; no automatic Git inspection occurs.
- Explicit ranges select IMPACT deterministically. Existing file-only queries keep
  their selected playbook and legacy direct-impact behavior.
- Range impact follows incoming dependency edges for at most two hops, with
  bounded nodes and deterministic shortest-path explanations. Static candidates
  never claim runtime coverage. Unknown spans and missing indexed symbols are visible.
- Add optional `contract_paths` for explicit Markdown roots. Also inspect explicit
  Markdown links from selected source files and backlinks in `specs/` and
  `decisions/`. Read only repository-contained regular files, with bounded file
  counts and bytes. Status and supersession qualify authority; unknown status is
  never promoted to active. Do not infer compliance from matching words.
- Resolve missing reference/impact anchors through one bounded discovery stage.
  Proceed only with a unique exact name/path or one unambiguous symbol candidate.
  Report ambiguity and missing anchors rather than choosing the first match.
  Keep one classifier call and deterministic execution stages.
- Add `investigation` result facts for implementation, dependents, tests,
  references, and contracts. States distinguish assessed, empty, partial,
  unavailable, and not assessed. Preserve existing `complete` semantics.
- No new dependencies, persisted formats, external services, or source mutations.
  SCS does not execute tests or claim defect absence.

## Acceptance criteria and verification

1. Changing one of several symbols excludes unrelated callers. Direct and indirect
   test candidates carry real dependency paths; cycles and bounds terminate.
   Test generated graphs through the real NativeGraph adapter and MCP serialization.
2. Explicit contract links return status, source location, and linkage evidence.
   Superseded/unknown documents cannot masquerade as active. Escaping paths,
   symlinks, oversized files, and missing targets are reported safely.
   Test generated document trees and public query evidence.
3. Missing anchors resolve only when grounded and unambiguous. Duplicate names
   remain visible; absent anchors do not execute misleading traversal.
   Test route sequences with existing injected providers and route fakes.
4. Empty, failed, truncated, unavailable, and unexecuted stages remain distinct.
   Test public output contracts and failed/partial investigations.

Use targeted tests first, then type checks, lint, and repository verification.
Required checks that depend on unavailable infrastructure remain explicitly recorded.

## Risks and recovery

Indexed source can be stale. Line ranges refer to the indexed revision; results
must be checked against current source. Static paths cannot establish dynamic
behavior. Explicit Markdown links identify candidate authority, not compliance.

Each unit is reversible through its own Git commit. No index migration or service
replacement is required. Keep Cortex and active terminal sessions running.

## Execution

- [x] Changed-symbol impact and explainable tests.
- [ ] Active contract retrieval.
- [ ] Grounded anchor resolution.
- [ ] Investigation coverage.

## Verification results

Unit 1: 49 targeted tests passed, including real NativeGraph dependency paths,
range validation, MCP dispatch, source aliases, and service path containment.
`uv run --frozen basedpyright` and `uv run --frozen ruff check src tests` passed.
`just verify` reached native build; its maturin resolver cannot reach PyPI DNS.
The installed native adapter was used for targeted tests. Full verification
remains outstanding; later units will reuse targeted and broader checks.
