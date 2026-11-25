# Architecture

SCS owns source parsing, indexing, query orchestration, its local daemon, and
its CLI and MCP contracts. Its durable graph and embedding engine, TSG, lives
in-tree at `crates/tsg`.

A new SCS installation starts with an empty index. SCS does not migrate,
inspect, or recreate data from other products.

## Storage

The Rust `scs-store` crate maps SCS repository scopes, typed code nodes and
relationships, ingestion checkpoints, metadata filters, traversal, and semantic
search onto generic `tsg` primitives. The Python, SCSWire, MCP, and CLI
contracts remain SCS-owned without coupling `crates/tsg` to code intelligence.

Because `crates/tsg` is part of the SCS Cargo workspace, building SCS requires
no external TSG repository checkout. TSG keeps canonical graph, catalog, and
embedding state transactionally in SQLite and treats its vector index as a
rebuildable accelerator.

An index created by the former SCS storage engine is not migrated in place.
When that incompatible database is first opened, SCS moves the database, WAL,
SHM, and legacy `.usearch` sidecar (when present) to unique
`*.pre-tsg.backup` names, then creates an empty TSG index. Run
`scs reindex <repo>` to rebuild derived state from repository source. For
rollback, stop SCS, retain the new TSG files separately, restore the backed-up
legacy filenames, and run the previous SCS binary. Backup removal is always an
explicit operator action.

Semantic embeddings are generated from parser-owned entity text. SCS does not
send repository files to a summarization service; an embedding provider receives
only the entity text used to build the semantic index. See
[Embedding configuration](configuration.md) for provider choices.

## Runtime ownership

MCP uses stdio between each harness and its bridge, then SCSWire over one
owner-only Unix socket between bridges and the daemon. No TCP port or platform
service manager is required. A bootstrap lock serializes simultaneous first
clients; the daemon independently holds the storage writer lock. Each bridge
connection is its lease, so abrupt termination cannot leave an orphan lease.

Runtime artifacts live under `~/Library/Application Support/SCS/` on macOS and
`$XDG_RUNTIME_DIR/scs` on Linux, falling back to
`~/.local/state/scs/runtime`. Persistent indexes live only under `SCS_HOME`.

## Model providers

Unconfigured Apple Silicon Macs with macOS 14 or later select the bundled local
profile. It supplies Qwen embeddings, Qwen reranking, and English Laya routing.
Automatic selection preserves existing explicit model configuration. OpenAI,
OpenAI-compatible services, legacy MLX embeddings, and custom decision services
retain their provider contracts. Explicit selectors can combine local and
external roles. See [provider configuration](configuration.md).

```text
provider configuration -> existing external adapters -> configured services
                       |
                       +-> shared local runtime
                                |
                         verified artifact cache
                                |
                         one native worker
                                |
                        Qwen / Qwen / Laya
```

Local runtime construction loads no models. Startup checks the cache without
downloading. If every selected model is complete and verified, startup attempts
to load those models before publishing its ready socket. This keeps checkpoint
loading outside the first query's classifier deadline.

With an incomplete cache, startup avoids heavy imports and first use prepares
the selected local roles. During explicit indexing, this work stays inside the
durable background job and reports `models` progress. The socket and health
requests remain responsive. A cold daemon can report ready while model preparation
is pending. Repository statistics report semantic readiness separately.

Bundled artifacts default to `SCS_HOME/models`; local cache overrides must remain
inside `SCS_HOME`. The shipped manifest pins every model revision, path, length,
and checksum. Downloads use anonymous HTTPS requests for those fixed artifacts.
Source and query text never enter download requests. Completed files are verified
before loading. Partial transfers cannot replace a verified artifact. Local
loaders cannot fetch replacements or execute remote Python code.

One bounded worker serializes local native operations across indexing and query
event loops. A cancelled caller does not free native capacity before its operation
finishes. Shutdown signals transfer cancellation before draining active service
requests, then releases model references and the worker.

Model-native Qwen formatting belongs to the local adapter. External adapters keep
their existing request formatting. Local embedding metadata includes the checkpoint
and formatting identity, so incompatible vectors use the existing quarantine and
rebuild mechanism. Explicit external configurations keep their existing identities.

Local routing and reranking failures use the existing query fallbacks. Required
embedding failures leave indexing unsuccessful without acknowledging source hashes.
Configured external decisions retain their model-qualified readiness gate, health
monitoring, and bounded outage recovery. Local models do not require that HTTP
service. The [bundled profile decision](../decisions/2026-11-14-bundle-local-default-models.md)
and [implementation spec](../specs/2026-11-14-bundled-local-models.md) define ownership
and acceptance criteria.

## Verification

`just coverage` reports uncovered Python lines and enforces the committed
risk-based floor. The pre-commit hook runs the whole-source type gate.
Isolation gates cover exact stdio MCP inventory, multi-bridge daemon
convergence, bounded frames, generation-safe cleanup, stale/live socket
ownership, empty-startup behavior, runtime isolation, repository source
fingerprints, and committed RSS/index/query budgets.
