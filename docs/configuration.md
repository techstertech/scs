# Provider configuration

Persistent configuration lives at `~/.scs/config.toml`. Explicit Python
settings take precedence over environment variables, environment variables
take precedence over TOML, and TOML takes precedence over defaults. The
standard `OPENAI_API_KEY` environment variable overrides `openai_api_key` in
the file. Storing the key in the owner-only configuration file makes it
available to lazily spawned daemon processes without placing it in MCP config.

## OpenAI embeddings

The default provider settings are below. Supply your own API key:

```toml
embedding_provider = "openai"
embedding_model = "text-embedding-3-large"
embedding_dimension = 3072
openai_base_url = "https://api.openai.com/v1"
openai_api_key = "replace-with-your-key"
```

The OpenAI provider sends source-derived entity text to the configured API to
generate the semantic index. Keep the config file owner-readable only:
`chmod 600 ~/.scs/config.toml`.

## Local OpenAI-compatible embeddings

To use a local OpenAI-compatible server without an API key:

```toml
embedding_provider = "openai_compatible"
embedding_model = "Qwen3-Embedding-8B-4bit-DWQ"
embedding_dimension = 4096
openai_compatible_base_url = "http://127.0.0.1:10001/v1"
```

The compatible provider defaults to loopback HTTP URLs. To use a server on an
explicitly trusted network host, configure both its URL and exact hostname:

```toml
openai_compatible_base_url = "http://embedding-host.example:10001/v1"
openai_compatible_trusted_hosts = ["embedding-host.example"]
```

The trust list is empty by default and matches hostnames case-insensitively;
it does not match subdomains or wildcards. Only opt in to a host and network
that you trust with source-derived embedding text. SCS continues to run locally.
The environment equivalent is
`SCS_OPENAI_COMPATIBLE_TRUSTED_HOSTS='["embedding-host.example"]'`.
In compatible-provider mode, SCS ignores OpenAI credentials and sends no
authorization header. Changing the provider, model, or dimension quarantines
incompatible vectors; the next indexing pass regenerates embeddings while
preserving the structural graph.

## In-process MLX embeddings

On Apple Silicon, SCS can load an embedding model in its own process when the
`mlx_embedding_models` package and the selected model are installed separately:

```toml
embedding_provider = "mlx"
embedding_model = "Qwen3-Embedding-8B-4bit-DWQ"
embedding_dimension = 4096
```

This provider is separate from the optional external decision API. Neither
local provider requires an OpenAI API key.

## Optional external decisions

External decisions are disabled by default. The external service owns its
model and runtime. Set the service's exact model identity and endpoint:

```toml
decision_model = "your-provider/decision-model@revision"
decision_base_url = "http://127.0.0.1:10000/v1"
```

An absent `decision_model` disables external decisions. The existing `disabled`
value also disables them. Other values select a model at the external service;
SCS does not translate model names into checkpoints or run decision inference.
The model identity must contain 1–256 nonblank characters.

For an explicitly trusted remote host, set `decision_trusted_hosts` to a
list containing its exact hostname. The decision API uses credential-free HTTP
and sends goals and anchors to that host. SCS verifies the configured model's
readiness before its daemon becomes ready and validates the model identity in
every choice response. See [the API contract](decision-api.md).

`decision_timeout_seconds` defaults to 0.5 seconds and can lower each mode's
classifier deadline. `decision_max_concurrency` defaults to 1 and accepts 1–4.
The environment equivalents use the `SCS_` prefix, including
`SCS_DECISION_MODEL` and `SCS_DECISION_BASE_URL`.

When migrating from the former model-specific client, replace the old selector
with the external service's model identity. The service must implement
model-qualified readiness; an older endpoint that returns a different identity
cannot satisfy startup checks. Restart SCS after changing configuration.

## Reranking

Search always fuses bounded semantic and lexical candidates. To rerank that
candidate set through the same local reranking endpoint, opt in explicitly:

```toml
reranking_model = "your-installed-reranker"
```

An absent `reranking_model` disables reranking. A configured model uses the
same validated `openai_compatible_base_url` and exact-host trust list
(`openai_compatible_trusted_hosts`) as local embeddings and sends no
credentials. An unavailable or malformed reranker degrades to deterministic
fused retrieval without making search unavailable.
