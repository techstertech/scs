# SCS — Semantic Code System

SCS is a headless code-intelligence service for coding agents. It indexes source
repositories and exposes a local Model Context Protocol (MCP) interface. Agents
use `query_code` to find implementations, inspect dependencies, identify affected
tests, and retrieve linked specifications and decisions.

Each query returns bounded evidence, the selected investigation workflow
(playbook), and coverage details. SCS starts with an empty index. It indexes a
repository only after an explicit request and never changes repository source.

[Quick start](#quick-start) · [Code queries](#how-code-queries-work) ·
[Decision models](#optional-external-decision-routing) ·
[Operations](#operations) · [Development](#development)

## Requirements

- Stable releases support Apple Silicon macOS and x86-64 Linux with CPython 3.14.
- Indexing requires an embedding provider. The default uses the OpenAI API;
  local providers are also available. See [provider configuration](docs/configuration.md).
- Decision models are optional and run in an external service. The
  [KEV-4B setup](#optional-external-decision-routing) below enables model-based
  routing. Without a configured model, SCS uses deterministic routing.

## Quick start

Download the installer and checksum manifest from the same
[GitHub Release](https://github.com/techstertech/scs/releases), verify the
installer, and install SCS:

```bash
VERSION=0.4.0
curl -fsSLO "https://github.com/techstertech/scs/releases/download/v${VERSION}/scs-installer-${VERSION}.sh"
curl -fsSLO "https://github.com/techstertech/scs/releases/download/v${VERSION}/SHA256SUMS"
shasum -a 256 -c SHA256SUMS --ignore-missing
sh "scs-installer-${VERSION}.sh"
scs version
```

On Linux, use `sha256sum -c SHA256SUMS --ignore-missing`. The installer
verifies its wheel and constraints, installs without `sudo`, and uses a pinned,
checksum-verified `uv` binary when necessary. Current macOS releases are not
Apple-signed or notarized. See
[distribution and upgrade details](docs/github-releases-distribution.md).

Configure an embedding provider before indexing. For the default OpenAI
provider, create the configuration directory:

```bash
mkdir -p ~/.scs
```

Add these settings to `~/.scs/config.toml`:

```toml
embedding_provider = "openai"
embedding_model = "text-embedding-3-large"
embedding_dimension = 3072
openai_api_key = "replace-with-your-key"
```

Restrict access to the file before starting SCS:

```bash
chmod 600 ~/.scs/config.toml
```

The OpenAI provider sends source-derived entity text to its API. For local
embeddings and reranking options, see [provider configuration](docs/configuration.md).

Register the installed stdio bridge with Codex:

```bash
codex mcp add scs -- "$HOME/.local/bin/scs" mcp
codex mcp get scs
```

If an existing `scs` MCP entry points elsewhere, remove it first with
`codex mcp remove scs`. Restart open Codex clients after changing MCP
configuration. The command above assumes the installer's default `~/.local/bin`
location. Other MCP clients can run the same executable with the `mcp` argument.

From the repository you want to index, run:

```bash
scs index "$PWD"
scs status
```

Indexing runs as a durable background job. Ask your agent to check
`get_graph_stats(repo_path="/absolute/path/to/repo")` for structural and semantic
readiness before querying. An accepted indexing request does not mean indexing
has finished.

## How code queries work

```text
agent goal + repository + optional anchors
                 |
                 v
        validate request and paths
                 |
                 v
     select one of seven playbooks  <--- optional external decision API
                 |
                 v
     bounded index search and graph reads
                 |
                 v
   evidence + routing + trace + investigation coverage
```

For example, an agent can ask:

```text
query_code(
    goal="Find tests affected by changes to the parser",
    repo_path="/repo",
    file_paths=["src/parser.py"],
    mode="balanced",
)
```

The `fast`, `balanced`, and `thorough` modes set fixed time and evidence
budgets. Use an absolute repository path and repository-relative file anchors.
Check these result fields before drawing conclusions:

- `routing` identifies the playbook, decision provider, model, and any routing fallback.
- `complete`, `truncated`, and `degraded_stages` describe the bounded execution.
- `investigation` reports coverage for implementation, dependents, tests,
  references, and contracts. A completed playbook does not establish full repository coverage.

Test targets are static dependency candidates. SCS does not execute tests.
See the [MCP tool reference](docs/mcp-tools.md) for anchors and review examples,
and the [`query_code` migration guide](docs/query-code-migration.md) for retired tools.

## Embeddings and indexing

SCS indexes supported source files structurally. Other regular UTF-8 text
files can be indexed at file level for lexical and semantic search. Git ignore
rules and size limits apply. Once a repository is enrolled, SCS watches
Git-visible changes and updates its index in the background. See
[indexing and project management](docs/indexing.md) for coverage, limits,
reindexing, and deletion.

The default OpenAI embedding provider sends source-derived entity text to
the configured API. SCS does not send whole repository files to a
summarization service. You can instead configure a local OpenAI-compatible
server or an in-process MLX provider. Provider details and trust controls
are in [embedding configuration](docs/configuration.md).

## Optional external decision routing

Use [KEV-4B](https://huggingface.co/jaredpalmer/kev-4b) to select query playbooks
through an external service. Laya and other decision models can also work
through a service that implements the [SCS decision API](docs/decision-api.md).
SCS accepts the service's exact model ID and defines no model aliases.

A decision model chooses an investigation workflow. SCS performs the index
search and graph reads. The service receives the goal, explicit anchors, and
eligible playbook choices. SCS sends no retrieved source, graph records, or
embeddings to the decision service.

### Configure KEV-4B

Start a service that loads KEV-4B and implements both SCS endpoints:

- `GET /v1/decisions/ready?model=<model ID>` confirms that the requested model is warm.
- `POST /v1/decisions` returns a choice and probabilities for the supplied playbooks.

The [upstream KEV server](https://github.com/jaredpalmer/kev) exposes the System
One API. An SCS-compatible service or adapter must provide the endpoints above.
The service owns model installation, runtime, warmup, and memory use.

If upgrading an older model-specific configuration, follow the
[migration order](docs/decision-api.md#migration) before changing the model ID.

Add these settings to `~/.scs/config.toml`:

```toml
decision_model = "jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
decision_base_url = "http://127.0.0.1:10000/v1"
```

This example uses a specific KEV revision. Use the exact identity supported by
your service. For Laya or another model, change `decision_model` to that service's
supported identity. Changing the decision model requires no index rebuild.

For a trusted network service, replace the loopback URL and allow its exact hostname:

```toml
decision_base_url = "http://decision-host.example:10000/v1"
decision_trusted_hosts = ["decision-host.example"]
```

The decision API uses HTTP and sends no credentials. Use only a trusted host
and network for query goals and anchors. See [provider configuration](docs/configuration.md#optional-external-decisions)
for environment variables, timeouts, and concurrency limits.

### Verify routing

Probe readiness for the configured model. Adjust the URL and model ID to match your service:

```bash
curl --fail --silent --show-error --get \
  'http://127.0.0.1:10000/v1/decisions/ready' \
  --data-urlencode 'model=jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52'
```

The response must contain exactly:

```json
{
  "status": "ok",
  "model": "jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
}
```

Restart SCS to apply the settings, then check its health:

```bash
scs daemon restart
scs doctor
```

Ask your agent to run `query_code` as usual. Confirm `routing.provider` is
`"external"`, `routing.model` matches the configured ID, and `routing.fallback_applied`
is `false`. A readiness response verifies availability; a query verifies decision inference.

### Failure and recovery

An unavailable service or mismatched model ID prevents configured startup.
During a later outage, SCS becomes unready and waits for bounded recovery.
If recovery fails, SCS shuts down. A failed or slow inference request uses
deterministic routing and reports `routing.degraded_reason`.

To restore operation without a decision service, set:

```toml
decision_model = "disabled"
```

If `SCS_DECISION_MODEL` is set, update or remove that environment override too.
Then run `scs daemon restart`. Removing `decision_model` also disables external
routing. Indexes remain available.

## Operations

| Command | Purpose |
| --- | --- |
| `scs list` | List enrolled projects and stable numeric IDs. |
| `scs reingest ID\|PATH` | Rebuild one project's index. |
| `scs delete ID\|PATH` | Remove one project's SCS-owned index state. |
| `scs status` | Show daemon availability, readiness, and version. |
| `scs doctor` | Check storage and daemon health. |
| `scs metrics --days 7 --json` | Report aggregate operation metrics. |

Metrics exclude query text, source text, file paths, job payloads, and results.
See [indexing and project management](docs/indexing.md) for lifecycle details.

Each MCP client runs a small stdio bridge. Bridges share one lazily started
daemon, which shuts down after the last bridge disconnects. Persistent state
lives under `SCS_HOME`. See [architecture](docs/architecture.md) for storage,
runtime ownership, and legacy-index migration.

## Development

Source builds require CPython 3.14, Rust 1.98 or later, `uv`, and `just`.
From a source checkout, run:

```bash
just setup
just verify
just eval-search
```

If your default Rust is older, prefix these commands with `RUSTUP_TOOLCHAIN=1.98`
to select that installed toolchain.

`just setup` syncs dependencies, builds the private native extension, and
installs the repository's pre-commit hook. `just verify` runs strict
Basedpyright checks, Ruff, Python tests with branch coverage, and the Rust
workspace tests. Search and query evaluation guidance lives in
[evals/README.md](evals/README.md).

After documentation edits, run `scripts/verify-doc-links.sh` to check local link
targets. Review section anchors and external URLs separately.

## Uninstall

Close connected MCP clients, then choose one command:

```bash
scs uninstall          # Preserve indexes, configuration, cache, and logs.
scs uninstall --purge  # Also delete SCS state; retain coordination lock files.
```

Both commands require `uv` on `PATH` and leave MCP registrations in place.
Remove the SCS entry from your clients manually (`codex mcp remove scs` for Codex).
See [uninstall details](docs/github-releases-distribution.md#upgrade-rollback-and-uninstall)
for custom paths, safety checks, and recovery.
