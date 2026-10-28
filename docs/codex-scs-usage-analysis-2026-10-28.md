# Codex SCS effectiveness research

Date: 2026-10-28

Status: Research complete; feature proposals require outcome evaluation.

Source baseline: SCS commit `a4b90fb`, release 0.2.3.

This research identifies opportunities to improve agent effectiveness from
local Codex history. It prioritizes correct investigations, relevant tests,
preserved requirements, and verifiable completion. Speed and latency are not
the selection criteria.

The strongest candidates are grounded anchor resolution, explainable test
selection, and retrieval of active contracts. Framework behavior tracing
remains a lower-confidence experiment.

This follows the [September usage analysis](codex-scs-usage-analysis-2026-09-14.md).
That report describes an older tool inventory and separate implementation work.
Use the [current MCP reference](mcp-tools.md) for the active interface.

## Scope and method

The primary window covers 30 days ending at `2026-10-28T22:16:00Z`.
The recent comparison covers the final seven days. The fixed cutoff excludes
this research session and its live probes from historical counts.

| Population | 30 days | Seven days |
|---|---:|---:|
| SCS `query_code` calls | 998 | 181 |
| Threads containing those queries | 304 | 62 |
| Root user messages, excluding approval-review prompts | 1,021 | 270 |
| Threads containing those user messages | 145 | 32 |

The broader 30-day dataset contains 487 threads: 145 root threads, 265
subagent threads, and 77 approval-review threads. SCS counts include root and
subagent queries. User-message analysis excludes approval-review prompts.

The sample spans application, infrastructure, library, and development-tool
projects. Queries targeting SCS itself were a small minority. The sample is
specific to this workstation and its workflows.

Analysis began with the local `analyze-codex-history` skill's aggregate script.
Supplementary read-only queries used `thread_history_1.sqlite` and joined
`state_5.sqlite` for thread source and working-directory metadata. The bundled
script's older project-catalog lookup labeled all projects unknown.

Query-intent categories use overlapping keyword matches. They measure demand
signals, not incorrect results or confirmed defects. A bounded sample of 36
message selections, with some overlap, supported interpretation of broader
workflow patterns.

Current source, decisions, evaluation cases, and live query results were checked.
Three subagents examined capability gaps, broader workflows, and counterarguments.
Their discussion distinguished existing capabilities from proposed extensions.

The report preserves aggregates and paraphrased patterns. It contains no raw
chat messages, thread identifiers, credentials, or individual assessments.

## Observed demand

| Query-intent signal | 30-day queries | 30-day threads | Seven-day queries |
|---|---:|---:|---:|
| Tests or verification | 654 | 240 | 117 |
| Behavior flows, lifecycle, pipelines, or propagation | 189 | 104 | 31 |
| Contracts, specifications, invariants, decisions, or runbooks | 205 | 96 | 31 |
| References, callers, call sites, or usages | 86 | 60 | 11 |
| Change impact, affected code, dependents, or refactoring | 32 | 26 | 7 |

For test-related goals, 352 queries returned at least one apparent test path
over 30 days. The recent count was 82. Dedicated `test_targets` appeared only
rarely. Tests can appear in other result categories, so this is not a miss rate.

The corresponding goal filters were:

```text
tests:      \btest|verification|verify
behavior:   flow|lifecycle|trace|end.to.end|pipeline|propagat
contracts:  \bspec|contract|invariant|decision|runbook
references: reference|callers?|call.sites?|incoming|usage|uses of
impact:     impact|affected|blast radius|dependent|refactor
```

These filters intentionally favor recall. Manual relevance judgments would be
needed before using the matches as evaluation cases.

## What SCS delivered

| Selected playbook | 30 days | Seven days |
|---|---:|---:|
| `DISCOVER` | 620 | 164 |
| `INSPECT_FILES` | 223 | 13 |
| `INVENTORY` | 81 | 0 |
| `IMPACT` | 27 | 1 |
| `UNDERSTAND` | 6 | 0 |
| `RELATIONSHIPS` | 2 | 1 |
| `REFERENCES` | 0 | 0 |
| Failed calls without a structured result | 39 | 2 |

Discovery dominance does not establish that every query required another
playbook. Automatic fallback also selects discovery, which affects this mix.

Of 86 reference-related queries, 72 lacked the structured symbol or position
anchor required for `REFERENCES`. A name in goal prose does not satisfy that
requirement. See [`eligible_playbooks`](../src/scs/orchestration/decision.py).

Only one query repeated identical arguments within its thread during the
30-day window. Repeated searches therefore do not generally represent exact
retries.

### Paired live query probe

One read-only experiment asked for tests and dependents affected by MCP server
edits. Both calls used balanced mode and a result limit of ten.

| Input difference | Selected playbook | Returned evidence |
|---|---|---|
| No file anchor | `RELATIONSHIPS` | 30 traversal entries; no dedicated test targets |
| `file_paths=["src/scs/mcp/server.py"]` | `IMPACT` | Five dependents and three test targets with import evidence |

Both results reported `complete=true`. The
[query contract](../specs/2026-09-22-unified-query-orchestration.md) defines this
as playbook completion, not semantic completeness or task correctness.

The probe demonstrates sensitivity to grounded anchors. It does not establish
a population-wide failure rate or prove the value of framework tracing.

### Current implementation constraints

- `UNDERSTAND` and `RELATIONSHIPS` invoke the same traversal in
  [`QueryOrchestrator.query`](../src/scs/orchestration/query.py).
- `IMPACT` starts with every indexed symbol in the supplied files and follows
  incoming dependencies once. It identifies tests through path markers among
  capped direct dependents. See
  [`composite_regression_risk`](../src/scs/services/routes.py).
- [Indexing](../src/scs/indexing/pipeline.py) omits parsed edges whose endpoints
  cannot be resolved. Absence of an edge cannot establish absence of behavior.
- [Query evidence](../src/scs/orchestration/query.py) exposes locations and
  stages, but lacks source revision identity and relationship-resolution provenance.
- The [current decision](../decisions/2026-09-22-unify-agent-code-queries.md)
  deliberately selects one bounded playbook and rejects an open-ended
  model-controlled loop.

## Feature priorities

### Grounded anchor resolution and composed investigations

The missing anchors and paired probe provide the clearest support for a small
experiment using existing capabilities.

```text
Goal
  -> candidate files and symbols
  -> verified anchors or explicit ambiguity
  -> references and impact investigation
  -> evidence and remaining gaps
```

Start with a helper that carries discovered anchors into follow-up queries.
Improve caller guidance alongside it. If outcomes improve, consider a fixed
investigation sequence within the existing query surface.

Success means recovering the expected implementation owner, dependents, and
tests on reviewed tasks. Ambiguous names must remain visible. Wrong anchor
selection must not silently produce a confident investigation.

The mechanism has source and probe support. Improvement in completed engineering
tasks remains unmeasured. A general autonomous planner is not justified by this
evidence.

### Explainable test selection

The test-related query volume establishes repeated demand. Current direct
dependency analysis and path markers limit how SCS explains test relevance.

Combine existing test search with dependency evidence. Return the reason for
each candidate and the path connecting it to changed code. Distinguish static
candidates, observed coverage, and work that was not assessed.

Start with existing operations. Add changed-symbol ranges or bounded indirect
dependencies only where reviewed regression cases demonstrate a miss.

Success means selecting tests that detect known regressions, while limiting
unsupported recommendations. A larger list of test files is not sufficient.
Coverage evidence must identify its revision and become stale when relevant
source changes.

Demand is well supported. The historical data does not establish how many test
recommendations were incorrect.

### Retrieval of active contracts

Contract-related queries recur alongside user requests to preserve behavior,
reuse established mechanisms, and respect persistence expectations.

```text
Active requirement -> implementation location -> related verification
        |
        +-> authority, lifecycle status, and superseding record
```

Start with explicit links in existing specs, decisions, repository instructions,
code comments, and tests. Preserve lifecycle status. Semantic matching can find
candidates, but cannot establish authority or compliance.

Success means retrieving the applicable requirement before a proposed change.
Evaluation cases must include superseded records and contradictory evidence.
The result must distinguish intended, implemented, and tested behavior.

This is a demand-backed proposal. The chief risks are obsolete authority,
false compliance claims, and maintenance of links after source changes.

### Explicit investigation coverage

The live probe shows why playbook completion cannot establish that every
requested aspect was investigated.

Expose assessed and unassessed aspects using existing trace facts where
possible. For example, show that references were not assessed because a symbol
anchor was unavailable. Distinguish an empty result from an unexecuted step.

Start with clearer caller guidance and result presentation. Change the public
response contract only if those measures cannot express the required evidence.

Success means agents recognize missing evidence and select a valid follow-up.
The result must not imply that a completed investigation proves correctness.

### Evidence provenance

Stronger investigations need source identity, extraction provenance, and
explicit unresolved relationships. Attach these facts where they affect a
claim. Invalidate findings when their supporting source changes.

Start with the evidence used by anchor and test-selection experiments. Avoid
a general knowledge-management subsystem before concrete reuse exists.

Success means stale or unresolved evidence remains identifiable. An agent must
not interpret unsupported relationships as verified facts.

## Broader workflow opportunities

These overlapping categories describe user demand. They are not failure rates.

| Demand | 30-day messages / threads | Seven-day messages / threads |
|---|---:|---:|
| Deployment, release, restart, or production | 71 / 45 | 27 / 16 |
| Visual design and interaction | 78 / 51 | 15 / 13 |
| Existing contracts, reuse, defaults, or persistence | 72 / 44 | 13 / 9 |
| Testing and verification | 30 / 26 | 10 / 8 |
| Environment isolation and cleanup | 26 / 12 | 12 / 5 |

The bounded message sample included device, theme, zoom, interaction-state,
preserved-behavior, test-isolation, and delivery concerns. These observations
support connecting accepted requirements to relevant verification.

### Verification recipes with evidence capture

Existing global instructions require verification against acceptance criteria.
Existing browser, deployment, LaunchAgent, and iOS workflows already define
execution or verification steps. More general policy text would duplicate them.

Extend project recipes to connect each criterion with the checked revision,
environment, executed check, observed result, and evidence location.

For interaction changes, recheck the original device and interaction state.
For delivery tasks, include the relevant product transaction and readiness
state. An exit code alone cannot establish product behavior.

Success means each completion claim points to a relevant executed check.
Skipped checks remain explicit. Evidence capture must exclude credentials and
unrelated personal data.

### Test resource ownership and cleanup

Extend existing project recipes with environment preconditions, ownership
identifiers, and cleanup checks. Use these facts to distinguish disposable
resources from pre-existing services.

Success means a representative run uses the intended isolated resources and
removes only resources created by that run. Preserve production services and
active sessions.

## Lower confidence ideas

### Framework behavior tracing

Framework tracing would connect code through route registration, client
requests, subscriptions, and state ownership. A possible trace is:

```text
UI action -> client request -> server route -> handler -> stored state
    ^                                                       |
    +------------ rendered state <- response ----------------+
```

Behavior-flow requests provide indirect support. Visual and interaction requests
also include tasks that may gain little from framework tracing.
A narrower defect-word filter matched 12 messages across nine threads.

The research did not establish that missing framework links caused incorrect
diagnoses or repeated fixes. The profile-saving example discussed during review
was illustrative, not a verified historical failure.

Before building an adapter, compare ordinary SCS results with manually assembled
traces for historical bugs with known causes. Measure diagnosis accuracy,
selection of discriminating tests, preserved behavior, and unsupported claims.

Only automate connections that improve those outcomes. Static relationships
must remain separate from observed runtime behavior.

### Persistent investigation memory

A compact record could preserve accepted choices, constraints, rejected
hypotheses, unresolved questions, and verification evidence across handoffs.
The agent harness should own conversational intent. SCS could link reviewed
records to source without importing raw chats.

The dataset contains 350 context compactions: 246 in root threads and 104 in
subagent threads. These counts do not establish memory failures or causation.
Generic memory infrastructure therefore has weaker support than the priorities
above.

## Evaluation and suggested order

The [current query suite](../evals/scs-query-v3.json) contains 17 cases about SCS
itself. Its [scoring implementation](../src/scs/evaluation/query.py) primarily
measures retrieval identities and routing. Those measures cannot establish
correct diagnosis or preservation of product behavior.

Use a small private set of reviewed historical task shapes across several
repositories. Hold some repositories or tasks out of development. Include
known regressions, ambiguous anchors, superseded contracts, and cases requiring
an insufficient-evidence conclusion.

Compare current behavior with each proposed intervention. Score:

- Correct implementation owner and diagnosis.
- Relevant affected consumers and regression tests.
- Preserved requirements and behavior.
- Unsupported claims and incorrectly resolved ambiguity.
- Verification that addresses the original acceptance criterion.

Combine automated checks with human review. This follows
[OpenAI evaluation guidance](https://developers.openai.com/api/docs/guides/evaluation-best-practices#how-to-read-evals)
on representative tasks and calibrated judgment. Passing tests alone cannot
establish that every requirement was preserved.

Suggested order:

1. Establish the small outcome baseline alongside the first experiment.
2. Test grounded anchor handoff and composed existing operations.
3. Improve explainable test selection where reviewed cases show gaps.
4. Evaluate active contract retrieval and explicit investigation coverage.
5. Expand evidence provenance with those concrete consumers.
6. Consider framework tracing or persistent memory only after a demonstrated gain.

Observed usage and implementation constraints support this order. The proposed
capability gains remain hypotheses. Public API, persisted-format, security,
dependency, or architectural changes require their own scoped design and approval.
