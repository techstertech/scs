# SCS

SCS is a headless code-intelligence service. It indexes source repositories and
lets coding agents investigate them through a local MCP tool. An agent asks
`query_code` a question; SCS returns bounded evidence from its structural and
semantic index.

SCS starts with an empty index. It enrolls a repository only after an explicit
CLI, MCP, or client request, and never changes repository source.

## Requirements

- Stable releases support Apple Silicon macOS and x86-64 Linux with CPython 3.14.
- Indexing needs an embedding provider. The default uses the OpenAI embeddings
  API and sends source-derived entity text to it. Local providers are available.
- An optional external decision API can choose query playbooks.
  SCS works without it on both supported platforms and never runs a decision
  model. See [external decision routing](#optional-external-decision-routing).

## Quick start

Download the installer and checksum manifest from the same
[GitHub Release](https://github.com/leonardoventurini/scs/releases), verify the
installer, and install SCS:

```bash
VERSION=0.4.0
curl -fsSLO "https://github.com/leonardoventurini/scs/releases/download/v${VERSION}/scs-installer-${VERSION}.sh"
curl -fsSLO "https://github.com/leonardoventurini/scs/releases/download/v${VERSION}/SHA256SUMS"
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
provider, put this in `~/.scs/config.toml`:

```toml
embedding_provider = "openai"
embedding_model = "text-embedding-3-large"
embedding_dimension = 3072
openai_api_key = "replace-with-your-key"
```

Keep the file owner-readable only (`chmod 600 ~/.scs/config.toml`). See
[embedding configuration](docs/configuration.md) for local provider and
reranking options.

Register the installed stdio bridge with Codex, then index the repository
containing your current directory:

```bash
codex mcp add scs -- "$HOME/.local/bin/scs" mcp
codex mcp get scs
scs index "$PWD"
scs status
```

If an existing `scs` MCP entry points elsewhere, remove it first with
`codex mcp remove scs`. Restart open Codex clients after changing MCP
configuration. Indexing runs as a durable background job; use `scs status` or
`get_graph_stats` to check when it is ready. The path above assumes the
installer's default `~/.local/bin` location.

## Uninstall

Close connected MCP clients, then run:

```bash
scs uninstall          # Preserve indexes, configuration, cache, and logs.
scs uninstall --purge  # Also delete SCS state; retain coordination lock files.
```

Choose one command. Both require uv on PATH and leave MCP registrations in place;
remove the SCS entry from your clients manually (`codex mcp remove scs` for
Codex). See [uninstall details](docs/github-releases-distribution.md#upgrade-rollback-and-uninstall)
for custom paths, safety checks, and recovery.

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
   evidence + routing + trace + completeness
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
budgets. Results show which playbook ran and whether evidence was complete,
truncated, or degraded. See the [MCP tool reference](docs/mcp-tools.md) for
anchors, tool contracts, and the
[`query_code` migration guide](docs/query-code-migration.md) for retired tools.

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

A compatible external service can choose one of SCS's bounded query playbooks.
SCS sends the goal, explicit anchors, and playbook choices to its configured
HTTP API. It sends no repository source, embeddings, or retrieved evidence.
SCS performs the search and graph reads. Without a configured decision model,
routing follows deterministic rules.

After starting a compatible service, set its endpoint and exact model identity
in `~/.scs/config.toml`:

```toml
decision_model = "your-provider/decision-model@revision"
decision_base_url = "http://127.0.0.1:10000/v1"
```

The model identity belongs to the service. SCS contains no built-in decision
checkpoint or model aliases. Omit `decision_model` to disable external routing;
`decision_model = "disabled"` remains supported. Older model-specific
configurations must replace their selector with the service's exact model ID.

Then restart SCS with `scs daemon restart`. SCS requests readiness for the
configured model and requires an exact identity match before publishing its
socket. An unavailable or incompatible service prevents startup. An inference
failure during a query reports degradation and uses deterministic routing.

The [decision API contract](docs/decision-api.md) defines readiness and choice
validation. `routing.provider` reports `external` or `deterministic`, while
`routing.model` reports the selected model identity. The external service owns
model installation, warmup, runtime, and memory use.

## Operations and development

`scs list` shows enrolled projects and their stable numeric IDs.
`scs reingest ID|PATH` forces a full rebuild, and `scs delete ID|PATH`
removes only SCS-owned derived state. `scs doctor` checks daemon health;
`scs metrics --days 7 --json` reports aggregate operations without query
text, source text, file paths, job payloads, or results. See
[indexing and project management](docs/indexing.md) for lifecycle details.

Each MCP client runs a small stdio bridge. Bridges share one lazily started
daemon, which shuts down after the last bridge disconnects. Persistent state
lives under `SCS_HOME`. See [architecture](docs/architecture.md) for storage,
runtime ownership, and legacy-index migration.

For a source checkout:

```bash
just setup
just verify
just eval-search
```

`just setup` syncs dependencies, builds the private native extension, and
installs the repository's pre-commit hook. `just verify` runs strict
Basedpyright checks, Ruff, Python tests with branch coverage, and the Rust
workspace tests. Search and query evaluation guidance lives in
[evals/README.md](evals/README.md).
