---
status: shipped
project: scs
project-root: /Users/leonardo/Repositories/scs
created: 2026-10-28
updated: 2026-10-28
owner: code-intelligence
decision: decisions/2026-09-25-use-external-laya-inference.md
supersedes: specs/2026-09-25-consume-external-laya-inference.md
superseded-by:
implementation:
  commits: [1fadf58]
  pull-request:
---

# Configured external decision API

## Outcome

SCS selects query playbooks through a model-neutral external decision API.
Configuration supplies the model identity; SCS neither installs nor runs a
decision model. Provider metadata describes the selected implementation rather
than inferring it from a model repository name.

## Problem and scope

The current HTTP client still pins a historical model identity and names its
provider after that model. MES now serves a different model behind that
compatibility identity. This makes SCS diagnostics misleading and prevents
operators from selecting another compatible model.

This work covers the SCS client, settings, query metadata, tests, documentation,
MES model-qualified readiness, and the local decision-model configuration.
The decision runtime was already removed from SCS. Existing embedding providers,
query playbooks, budgets, storage, and MCP inventory retain their contracts.
Historical decisions and evaluation reports retain their original evidence.

## Contracts

- `decision_model` contains the external service's model identity, bounded to
  256 nonblank characters. Omission disables external decisions. The existing
  `disabled` marker also disables them for compatibility.
- `decision_base_url`, trusted hosts, concurrency, and timeout controls retain
  their current meaning. SCS has no built-in decision checkpoint or model alias.
- Readiness uses `GET {decision_base_url}/decisions/ready?model=<encoded ID>`.
  HTTP 200 must return exactly `{"status":"ok","model":"<configured ID>"}`.
  A different identity cannot make SCS ready.
- MES accepts the optional `model` query parameter, rejects unsupported IDs
  with HTTP 400, and reports an unavailable runtime with HTTP 503. Its unqualified
  readiness response stays compatible with older clients. A readiness request
  never performs inference.
- Choice requests and responses retain their existing shapes. SCS supplies the
  configured model and checks the response identity, eligible choice, confidence,
  and complete finite probabilities before execution.
- `routing.provider` is `external` for external decisions and `deterministic`
  for disabled routing or classifier fallback. `routing.model` reports the
  decision's model identity without using it to infer the provider.
- Startup gating, health monitoring, bounded recovery, orderly shutdown, and
  in-flight deterministic degradation retain their behavior.

```text
model ID + endpoint configuration
                 |
                 v
SCS HTTP client -- model-qualified readiness --> external service
                 -- bounded choice request --> external model
                 <-- validated choice --------
                 |
                 v
fixed SCS playbook + external/deterministic metadata
```

## Security and data

Existing credential-free HTTP and exact-host trust controls remain in force.
Only bounded goals, caller anchors, and playbook descriptions reach the decision
service. No retrieved source, persisted format, production dependency, or
credential changes are required.

## Acceptance criteria and verification

1. Different configured model identities work without SCS source changes.
   Provider tests cover model-qualified readiness and choice requests.
2. Unsupported, mismatched, or unavailable models cannot make SCS ready.
   SCS provider and daemon tests plus MES readiness tests cover these cases.
3. Invalid answers cannot execute an arbitrary route; classifier failure retains
   deterministic fallback. Existing routing tests and focused model-mismatch
   checks verify this behavior and provider metadata.
4. Disabled decisions require no model service. Existing standalone and isolation
   tests verify this behavior.
5. Active SCS code and public guidance contain no model-specific provider names,
   pinned decision model, installer, or execution path. Source review verifies
   this boundary; historical evidence remains available.
6. The local configuration selects the current external model by its actual ID.
   A model-qualified readiness probe and a fresh SCS query verify the live path.

Run targeted tests first, then `just verify` in SCS and the relevant MES suite.
Do not add tests for the retired model runner or the local TOML edit.

## Risks and recovery

An older service that ignores the model query parameter cannot satisfy readiness
for a different configured identity. Activate the compatible MES endpoint before
changing local SCS configuration. Preserve the previous configuration privately
and replace only the decision-model value. Revert the changes or set
`decision_model = "disabled"` to restore deterministic operation. Index migration
is unnecessary. Preserve Cortex, its terminal sessions, and unrelated workloads.

## Verification results

User approved SCS, MES, and local configuration changes with configured-model
readiness before implementation.

- **Passed:** the focused provider, routing-contract, orchestration, and daemon
  tests completed with 52 passing tests. The new HTTP tests first failed before
  the generic client existed. Provider metadata assertions reproduced the old
  model-name inference defect before the fix.
- **Passed:** `just verify` completed with 500 Python tests, 88.04% branch
  coverage, strict Basedpyright and Ruff checks, and 108 Rust tests.
- **Passed:** MES's focused decision/retrieval/process/server suite completed
  with 44 passing tests, including model-qualified readiness without inference.
- **Passed:** source review found no model-specific decision names or pinned
  decision model in active SCS code, public documentation, tests, or scripts.
  Existing historical specs, decisions, and evaluation reports retain evidence.
- **Passed:** MES activation through its existing per-user LaunchAgent retained
  its program, working directory, and model settings. Main MES health returned
  `READY`. Model-qualified readiness returned the configured model identity;
  an unsupported model returned HTTP 400. Unqualified readiness retained its
  exact legacy response.
- **Passed:** a locally built wheel contains the generic client and excludes the
  retired provider module. The installed SCS tool uses the tested locked runtime
  dependencies. `scs doctor` reported the installed daemon ready.
- **Passed:** local configuration now selects the external service's actual
  model identity. All other values and mode `0600` were preserved. A private
  external-volume backup passed checksum and configuration-restoration checks.
- **Passed:** explicit indexing used repository-relative changed/deleted paths.
  A fresh installed stdio MCP query returned `provider = external`, the configured
  model, complete evidence, no degraded stages, and no fallback. A balanced
  request with `limit = 50` completed in 251 ms, including 215 ms classification.
  The default evidence limit can still produce ordinary bounded truncation.
- **Passed:** Cortex's web engine and terminal runtime retained their process
  identities throughout activation. No unrelated workload was restarted.

Acceptance criteria 1–6 passed. MES's implementation is commit `060fce2` in the
sibling repository. The MES focused suite reported existing UV prerelease-mode,
SWIG, and Starlette/httpx warnings. No audio generation, paid API call, or broad
MES suite was required or run for this decision-only change.

No production dependency, index format, or core-service change was required.
