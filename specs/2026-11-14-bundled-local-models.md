---
status: validating
project: scs
project-root: /Users/leonardo/Repositories/scs
created: 2026-11-14
updated: 2026-11-14
owner: code-intelligence
decision: decisions/2026-11-14-bundle-local-default-models.md
supersedes:
  - specs/2026-10-28-external-decision-api.md (external-only runtime ownership and unconfigured Apple Silicon selection only)
superseded-by:
implementation:
  commits: []
  pull-request:
---

# Bundled local model profile

## Outcome and authorization

A fresh Apple Silicon installation can index and investigate code without an
API key or a separately installed model service. The first model operation
downloads pinned artifacts. Later processes can work offline.

The user authorized implementation of the measured Qwen embedding/reranking
and English Laya profile. Existing custom external services must retain their
behavior. The target is a 16 GB Mac and an automatic model download below 8 GB.
The [comparison](../docs/local-model-comparison-2026-11-14.md) records measured
components and the remaining hardware limitation.

## Scope

- Add local adapters behind the existing embedding, reranking, and decision ports.
- Bundle their runtime dependencies on supported Apple Silicon macOS systems.
- Retain OpenAI, OpenAI-compatible, legacy MLX, and external decision behavior.
- Reuse durable indexing for first-use model work and existing query fallback.
- Update provider selection, lifecycle ownership, installation guidance, and tests.

No installed service is upgraded or restarted by this task. No external service
implementation, repository index format, MCP inventory, or repository source
mutation capability changes. Linux retains the existing external defaults.

## Selection and compatibility

`model_profile` accepts `auto`, `local`, and `external`; its default is `auto`.
Automatic local selection requires Apple Silicon and macOS 14 or later.

In automatic mode, any explicitly supplied existing model setting selects the
legacy behavior. This includes providers, models, dimensions, endpoints, trust
lists, credentials, decision controls, and disabled decision values. Omitted
reranking and decision models stay disabled in those existing configurations.
Explicit constructor values, environment, and TOML retain their precedence.

`local` explicitly selects the bundled defaults. Per-role provider selectors
permit mixed configurations. The existing `embedding_provider` gains `local`.
`reranking_provider` and `decision_provider` accept `local`, `external`, or `disabled`.
The external selector retains model-presence activation and existing disable
semantics. A local selector accepts only its shipped pinned model identity.
Explicit model settings are never silently sent to a different implementation.

The bundled profile uses Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, and English
Laya. Only roles selected locally receive a local runtime or artifacts. External
profiles never instantiate that runtime or import inference libraries.

External request bodies, model identities, credentials, prefixes, endpoint trust,
readiness gates, outage recovery, and fallback remain unchanged. In particular,
model-native Qwen formatting applies to the new local adapter only.

## Ownership and flow

```text
settings -> explicit provider selection
               |                     |
               v                     v
      existing external ports   one local runtime
                                    |
                            verified artifact cache
                                    |
                           one inference worker
                                    |
                         Qwen / Qwen / Laya
```

The daemon owns one local runtime when a role needs it. Construction and daemon
startup perform no model downloads. Cold startup performs no heavy imports.
A complete verified cache is warmed before publishing readiness. This prevents
the first cached query from spending its routing deadline on model loading.
Missing or invalid caches retain the first-use preparation path.
The first local operation prepares the selected local roles. Indexing performs
this work inside its existing
durable background job. The daemon socket and health requests remain responsive.
Indexing receives typed model-preparation progress through its existing owner.
Optional preparation failures do not disable an available embedding model.

The runtime serializes native work on one dedicated worker. A cancelled caller
does not release native capacity before the underlying work completes. It does
not enqueue an unbounded executor backlog. Its admission gate supports both the
indexing worker's event loop and the daemon's query event loop.

Native embedding and reranking batches contain at most four inputs. Embeddings
use a 1,024-token cap and model-native query instructions, bare documents, and L2
normalization. Reranking uses the upstream yes/no scoring module with a 2,048-token
cap. Decision questions reuse SCS instructions, descriptions, and eligible choices.
Truncated Laya requests fail into deterministic routing instead of using incomplete
input. Existing mode deadlines and reranking fallback stay in force.

Shutdown cancels outstanding artifact transfer cooperatively, drains admitted
native work, drops model references, and stops the worker. External service health
and lifecycle ownership remain separate.

## Artifacts, security, and data

The shipped manifest fixes every artifact's model, revision, relative path, byte
length, and SHA-256. Its three model snapshots total 3.26 GB. Downloads use fixed,
anonymous HTTPS model URLs. Query text and source text never enter those requests.

Bundled artifacts live under `SCS_HOME/models`. Readers verify bytes before model
loading. Writers stream to private temporary files, enforce the expected length,
verify checksums, and atomically promote complete artifacts. Interrupted work
retains complete files and removes its own partials. Symlink escapes are rejected.
No model loader can download extra files or execute remote Python code.

Provider metadata includes the pinned local model and embedding-format identity.
The existing vector compatibility mechanism handles an explicitly selected
provider change. Existing externally configured stores retain their identities.
No stored schema changes are required.

Routing metadata gains `provider = local` for local decisions. Existing external
and deterministic values retain their meanings. This additive diagnostic value
does not alter the MCP request inventory or investigation playbooks.

## Acceptance criteria

1. Fresh supported Macs select local models without credentials or manual services.
2. Every existing external configuration retains provider choices and actual HTTP
   payloads. Omitted optional roles remain disabled and no local models load.
3. First indexing accepts a background job promptly, reports model preparation,
   and indexes a generated repository using verified local embeddings.
4. Cached models support index reads and queries after a fresh process restart
   with OS network access denied.
5. Corrupt, oversized, missing, or interrupted artifacts never load as verified
   models and never record a successful embedding ingestion hash.
6. Concurrent indexing, queries, cancellation, and shutdown preserve one bounded
   native worker across the distinct event loops.
7. Model outputs are finite and complete; model-native formatting matches the
   measured profile. Truncated or unavailable local decisions report fallback.
8. Existing external readiness, trust, and credential controls pass unchanged.
9. Distribution installs required runtimes on supported Macs and retains a usable
   external-only installation on Linux and older unsupported Mac systems.

## Verification strategy

Write generated-fixture tests before implementation. Test artifact failures,
provider outputs, and cancellation at their owners. Add generated HTTP peers for
external composition and omitted-role compatibility. Use real models in a
disposable SCS root for first indexing, query execution, restart, and network-denied
verification. Keep all live services and user configuration intact.

Run targeted tests, strict checking, `just verify`, documentation links, and wheel
inspection before committing. Actual 16 GB hardware is unavailable; record that
limit and measure combined residency on the available host without claiming an
equivalent hardware test.

## Risks and recovery

The initial download is substantial and can fail offline. Cached complete files
remain reusable. Optional decision and reranking failures preserve existing query
fallback. Embedding failure leaves the job visibly unsuccessful and repairable.

CPU/GPU contention can exceed classifier deadlines. Native work stays bounded;
timeouts do not create additional inference jobs. The physical 16 GB test remains
a release qualification gap rather than an inferred pass.

Set `model_profile = external` to restore legacy selection. Existing external
settings already retain that behavior in automatic mode. Explicit provider changes
use existing vector quarantine and rebuild rules. No deployed state is changed
by this source implementation task.

## Verification results

Implementation is complete in source. The spec remains `validating` because
physical 16 GB qualification and execution on unsupported platforms remain open.
This task does not publish or install a release.

| Criterion | State | Evidence |
| --- | --- | --- |
| 1. Fresh supported selection | Passed | Generated fresh-Mac settings and real local daemon composition; no credentials or separate model service. |
| 2. External compatibility | Passed | Five daemon/HTTP integration cases preserve custom identities, endpoints, prefixes, missing optional roles, credentials, restart, and mixed providers. Existing transport implementations retain their behavior. |
| 3. Background first use | Passed | A fresh MCP session downloaded and verified all three models, indexed a generated fixture, and passed all ten review smoke checks. A concurrent probe observed a ready daemon during `phase = models`. |
| 4. Offline restart | Passed | A fresh process from the installed wheel reused the index, classified its first query locally, embedded a query, and applied reranking under OS-enforced IP denial. No first-query retry was permitted. |
| 5. Artifact failures | Passed | Generated corrupt, oversized, missing, cancelled, unsafe-path, and incomplete cache cases fail closed. A failed required model job never acknowledges successful embedding ingestion. |
| 6. Bounded ownership | Passed | Runtime and daemon tests cover concurrent event loops, cancelled native waiters, transfer cancellation before wire drain, shutdown, and cleanup failure. |
| 7. Output and formatting | Passed | Backend tests verify dimensions, finiteness, native Qwen formatting, eligibility, complete probabilities, and truncation fallback. Real fixture queries return the expected symbol. This is an integration check, not a general routing-accuracy claim. |
| 8. External controls | Passed | Existing readiness, trust, authentication, outage recovery, and decision tests pass. The actual user configuration still resolves to external providers without creating a local runtime; that probe never starts a daemon. |
| 9. Distribution | Passed on supported host; inferred elsewhere | The built wheel includes the manifest and native module. A clean disposable environment installs locked dependencies and runs the real offline smoke. Marker checks exclude inference dependencies on macOS 13 and Linux. Linux CPython 3.14 dependency resolution passes; no Linux or older-Mac runtime was executed. |

Executed checks:

- `just verify`: strict typing and Ruff pass; 639 Python tests pass with 88.79%
  coverage, including branch checks. All 108 Rust tests and existing performance
  ceilings pass.
- `just eval-local-check`: strict typing, scoped Ruff, and 42 evaluation/smoke
  tests pass after adding the reusable smoke driver.
- `uv lock --check`, `scripts/verify-doc-links.sh`, and `git diff --check` pass.
- `uv build --wheel --out-dir <private-root>/dist` builds the immutable wheel.
  Wheel inspection verifies packaged model pins, conditional dependency markers,
  and source-byte equality. Disposable installation uses exported locked runtime
  constraints and imports only from the installed wheel.
- `just smoke-local-models <private-root> <wheel-command> <online-report>` and
  `just smoke-local-models-offline <private-root> <wheel-command> <offline-report>`
  exercise the reusable commands. The offline command verifies `EPERM` for IP
  networking while preserving SCSWire's Unix socket.

[Retained evidence](../evals/reports/2026-11-14-bundled-local-models/README.md)
contains first-use, failure, installed-wheel, and restart reports. Combined
daemon residency is observational evidence on the available larger Mac. It does
not establish performance on a physical 16 GB Mac.

### Verification findings and scope

The first real cached restart exposed a classifier timeout while models loaded.
Startup now warms a complete verified cache before publishing readiness. It
still performs no downloads. A regression also proves that a transient cached
embedding load failure recovers during indexing without stale vector readiness.
These corrections preserve the selected providers and existing request deadlines.

The final review added the two smoke recipes and extended the existing tool
check recipe. It also documented state ownership, network denial, and cleanup.
The existing global review guidance covers the remaining process lessons; no
additional global rule or framework was needed.

Recommended review order is configuration selection, daemon composition, runtime
admission, artifact integrity, model adapters, and compatibility/integration tests.
Set `model_profile = external` to return to legacy selection. An explicit provider
change can require the existing vector rebuild. No production rollback is needed
because no installed service, user configuration, or live index was changed.
