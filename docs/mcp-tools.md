# MCP tools and query behavior

SCS exposes five model-facing operations: `query_code`, `ingest_project`,
`ingest_files`, `delete_repository`, and `get_graph_stats`. Query tools are
annotated read-only and closed-world. Ingestion tools are marked destructive
because reconciliation can remove stale SCS-owned index state.

## Code queries

An agent sends `query_code` a goal, repository path, and optional anchors:

```text
query_code(
    goal="Find tests affected by changes to the parser",
    repo_path="/repo",
    file_paths=["src/parser.py"],
    mode="balanced",
)
```

Optional anchors are `node_type`, `symbol_name`, `node_ids`, `file_paths`, and
`source_position`. The modes `fast`, `balanced`, and `thorough` set fixed time
and evidence budgets. SCS chooses one of DISCOVER, UNDERSTAND, RELATIONSHIPS,
REFERENCES, INSPECT_FILES, IMPACT, or INVENTORY, then runs only that bounded
playbook. Results include the selected route, compact evidence, stage traces,
timings, and `complete`, `truncated`, and `degraded_stages` flags.

`query_code` does not expose the former search tool's full node records, raw
query-angle controls, or pagination. Use the SCSWire service routes for
callers that need those lower-level responses. Run `just eval-query` to compare
the unified tool with versioned internal-route sequences on this repository.
The v4 suite preserves v3 relevance judgments and baseline steps; its unanchored
reference case now expects grounded REFERENCES execution. Historical suites remain
unchanged.

The `IMPACT` playbook uses bounded incoming dependency analysis and includes
test-file targets backed by direct dependency edges.

For a changed-symbol review, supply `changed_ranges`, for example
`[{"file_path": "src/parser.py", "start_line": 10, "end_line": 14}]`.
Lines are zero-based and inclusive in the indexed revision. Up to 100 ranges
across 20 files are accepted. Range paths join `file_paths` and select IMPACT.
Two incoming dependency hops identify consumers and test candidates. Results
include dependency paths and label tests as `static_dependency`, without claiming
runtime coverage. Unmatched ranges remain visible as `unresolved_range` file
evidence and degrade the investigation. File-only requests retain their behavior.

`contract_paths` optionally selects up to 20 repository-contained Markdown
records. Source links and explicit backlinks from `specs/` and `decisions/`
provide additional candidates for file-scoped queries. Contract evidence includes
status, superseding records, link locations, a content hash, and a bounded source
excerpt. Unknown or superseded records are never labeled active. Linkage establishes
relevance, not compliance. Explicit roots take priority over catalog scans; local
supersession links are followed within the read budget. Active candidates sort
before historical records. Unreadable targets and bounded scans remain visible.
No new index format or dependency is required.

Explicit goals such as "Find references to parse", "Find callers of parse", and
"Find affected dependents of parse" select the corresponding bounded investigation.
If its required anchor is absent, one discovery stage resolves a unique candidate.
Duplicate symbols and unavailable discovery cannot silently select a first symbol.
A full mixed search page uses a bounded exact-name lookup without graph traversal
before deciding uniqueness. Explicit reference identifiers survive search ranking.
Broad exploration retains its ranked context seed without claiming name uniqueness. `routing.anchor_resolution` exposes candidates, status,
and its scope. It describes returned candidates, not global name uniqueness.
Grounding retains one classifier call and the existing mode deadline.

Every response also includes `investigation` facts for implementation, dependents,
tests, references, and contracts. Each records `status`, `evidence_count`, executed
`stages`, and a reason. Status is `assessed`, `empty`, `partial`, `unavailable`, or
`not_assessed`. Finding a test filename during discovery does not mean dependency
analysis ran. `assessed` describes static evidence, not correctness or execution
coverage. Existing `complete` still describes the bounded playbook. Unavailable
reference providers and missing impact indexes cannot masquerade as empty success.

When configured, the external decision service receives only the goal, explicit anchors,
and SCS playbook choices. A configured daemon reports ready after the service confirms
the configured model is ready. It waits through short outages and shuts down if
recovery fails. An inference failure during a query returns deterministic
discovery evidence with a degradation reason. An unavailable or mismatched external model prevents SCS daemon startup.
`routing.provider` is `external` for service decisions and `deterministic` for
disabled routing or classifier fallback. `routing.model` retains the actual
requested model identity; the provider is never inferred from the model name.

The seven former read tools are now internal service routes. Existing MCP
callers must use `query_code`; the [migration guide](query-code-migration.md)
maps each retired tool to a goal and anchors.

## Index and deletion tools

`delete_repository(repo_path=...)` durably removes one repository's SCS-owned
index and catalog registration, stops its watcher, and supersedes pending
indexing work without reading, changing, or deleting repository source. The
source directory does not need to exist. The operation is annotated
non-read-only, destructive, closed-world, and idempotent; deleting an already
absent repository succeeds with `already_absent=true` and no queued job. All
SCS tools preserve repository source.

`get_graph_stats` separately reports structural and semantic readiness. With
a repository path it also includes redacted active/latest durable jobs and a
retry hint. `wait_job_id` can observe one job for up to 10 seconds without
running, retrying, or cancelling it.

Operational diagnostics remain available through the CLI and SCSWire instead
of occupying the model's tool catalog.
