---
status: shipped
project: scs
project-root: /Users/leonardo/Repositories/scs
created: 2026-10-28
updated: 2026-10-28
owner: code-intelligence
supersedes:
  - specs/2026-09-22-unified-query-orchestration.md (additive review inputs and evidence only)
superseded-by:
implementation:
  commits: [0acfc16, ec4e558, ebab3e2]
  final-unit: investigation coverage (commit containing this sign-off)
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
- [x] Active contract retrieval.
- [x] Grounded anchor resolution.
- [x] Investigation coverage.

## Verification results

Unit 1: 49 targeted tests passed, including real NativeGraph dependency paths,
range validation, MCP dispatch, source aliases, and service path containment.
`uv run --frozen basedpyright` and `uv run --frozen ruff check src tests` passed.
`just verify` reached native build; its maturin resolver cannot reach PyPI DNS.
The installed native adapter was used for targeted tests. Full verification
remains outstanding; later units will reuse targeted and broader checks.

Unit 2: 29 targeted tests passed. Contract cases cover source links, backlinks,
supersession, unknown status, escaping symlinks, missing files, byte limits,
output limits, and public query evidence. Strict types and lint passed.

The full Python suite passed outside the sandbox: 447 tests. The sandbox-only
run failed on Unix socket binding and process inspection; the permitted rerun
resolved those failures.

Unit 3: targeted query, routing, and MCP tests passed. Explicit reference and
impact goals resolve candidates through one discovery stage. Ambiguous, unsafe,
partial, or absent candidates cannot silently choose a first symbol. Resolution
reports its returned-candidate scope, not global symbol uniqueness.

`just verify` passed outside the sandbox after unit 2: 447 Python tests,
87.27% coverage, 108 Rust tests, strict types, lint, and native build.

Unit 4: all five review aspects report assessed, empty, partial, unavailable, or
not assessed. Executed dependency counts exclude changed symbols. Provider
unavailability and interrupted stages cannot appear as empty success. The
combined integration test runs changed ranges, native graph traversal, contract
retrieval, and coverage through the real service boundary.

Final verification: `just verify` passed outside the sandbox. Its native build,
strict types, lint, full Python suite, coverage gate, and Rust workspace tests
all completed. Exact totals are recorded below. Documentation links and
`git diff --check` passed. Native build storage remained 1,975,420 KiB.

Acceptance sign-off:

| Criterion | Result | Evidence |
|---|---|---|
| Changed symbols and explained test candidates | passed | Real NativeGraph tests cover unrelated symbols, indirect tests, cycles, limits, and missing ranges. |
| Linked contracts and lifecycle authority | passed | Generated documents cover explicit links, backlinks, supersession, unknown status, external successors, containment, and read budgets. |
| Grounded anchors and explicit ambiguity | passed | Injected route tests cover unique, duplicate, missing, unsafe, full-page, and incomplete discovery with one classifier call. |
| Explicit investigation coverage | passed | Public MCP tests and service integration distinguish empty, failed, partial, unassessed, and interrupted stages. |

The implementation keeps the approved input and tool boundaries. The final
coverage unit also corrected contract-budget priority, bounded supersession
following, and deterministic dependency ordering found during integration review.
No new dependency or persisted-format change was needed.

Review order: `docs/mcp-tools.md`, `services/impact.py`, `services/contracts.py`,
`orchestration/query.py`, `orchestration/coverage.py`, then the combined service
test. Rollback can revert the four implementation commits in reverse order;
no data migration is required. The running SCS daemon was not restarted.
Live-model relevance evaluation and production activation were not performed.
Static evidence remains bounded to the indexed source and configured traversal.

Final Python result: 468 passed in 30.70s.
Coverage summary: `TOTAL                                               6192    573   1578    323  87.70%`.
Rust result: 108 tests passed (2 core, 96 parser, 10 store).
