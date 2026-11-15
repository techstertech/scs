# Provider configuration

Persistent configuration lives at `~/.scs/config.toml`. Explicit Python
settings take precedence over environment variables, environment variables
take precedence over TOML, and TOML takes precedence over defaults. The
standard `OPENAI_API_KEY` environment variable overrides `openai_api_key` in
the file. Storing the key in the owner-only configuration file makes it
available to lazily spawned daemon processes without placing it in MCP config.

## Model profile selection

`model_profile` controls defaults. Explicit role settings retain their precedence.

| Profile | Default behavior |
| --- | --- |
| `auto` | Use local models on Apple Silicon with macOS 14+ when no model settings exist. Preserve legacy behavior when model settings are present. |
| `local` | Supply bundled defaults for roles without explicit settings. |
| `external` | Supply the previous external defaults. |

Automatic selection treats existing providers, models, endpoints, trust lists,
credentials, model-cache paths, and decision controls as explicit model settings.
An existing `OPENAI_API_KEY` also preserves OpenAI selection. Omitted optional
roles remain disabled for those existing configurations. SCS does not rewrite TOML.

## Bundled local models

Version 0.5.0 and later include inference dependencies on supported Macs. First
model use downloads the selected roles. Later operations use verified files
from `SCS_HOME/models`.

For a fresh configuration, no settings are required. To choose bundled defaults
explicitly, use:

```toml
model_profile = "local"
```

When switching an existing role, remove its previous model identity and provider
settings first. A local role accepts only its shipped model identity and dimension.
Do not retain an external `model_cache` path outside `SCS_HOME` for a local role.

The profile ships Qwen3-Embedding-0.6B with 1,024 dimensions,
Qwen3-Reranker-0.6B, and English Laya. Downloads total 3.26 GB. Only selected local
roles download or load. External profiles never create the inference worker.

The first indexing job reports `phase = "models"` during preparation. Its `current`
and `total` values represent artifact bytes during that phase. Other indexing
phases retain their existing units. Cached files are checked before loading.
Invalid or interrupted files never become successful embeddings.

With a complete cache, startup verifies and loads models before publishing
readiness. An incomplete cache leaves startup fast and prepares models on first
use. Startup never downloads models. A transient load failure can recover during
a later operation without restarting the daemon.

Use explicit disable controls when changing a local profile:

```toml
model_profile = "local"
reranking_provider = "disabled"
decision_provider = "disabled"
```

The existing `decision_model = "disabled"` marker also remains supported.
For local embeddings alone, use `embedding_provider = "local"`; optional roles
then retain their legacy disabled defaults.

Mix providers with explicit role settings. For example, keep a custom embedding
and decision service while using the bundled reranker:

```toml
model_profile = "local"
embedding_provider = "openai_compatible"
embedding_model = "my-installed-embedding"
embedding_dimension = 4096
openai_compatible_base_url = "http://127.0.0.1:10001/v1"
decision_provider = "external"
decision_model = "my-installed-decision-model"
decision_base_url = "http://127.0.0.1:10000/v1"
```

Local adapters apply their own model-native instructions. Custom external
embedding services retain the existing `search_query:` and `search_document:`
prefixes. No credentials accompany compatible-provider requests or model downloads.

Local inference runs on one native worker shared by indexing and queries.
Native batches contain at most four inputs. Embeddings use a 1,024-token cap;
reranking uses 2,048 tokens. English Laya has a 512-token total context. Truncated
decision requests use deterministic fallback. Existing mode deadlines still apply.

Routing diagnostics report `provider = "local"` and the pinned Laya identity.
Local embedding metadata includes the revision and formatting fingerprint.
An explicit provider change uses existing vector quarantine and reindexing rules.
The [comparison](local-model-comparison-2026-11-14.md) describes performance limits.

## OpenAI embeddings

External OpenAI settings are below. Supply your own API key:

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

External decisions are disabled in the legacy external profile by default. The external service owns its
model and runtime. For KEV-4B, set the service's exact model identity and endpoint:

```toml
decision_model = "jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
decision_base_url = "http://127.0.0.1:10000/v1"
```

This is an example revision, not an SCS model pin. Laya or another model can use
the same API when the service supports its exact identity. See the
[README setup](../README.md#configure-kev-4b) for service requirements and verification.

An absent `decision_model` disables external decisions. The existing `disabled`
value also disables them. Other values select a model at the external service;
The external provider does not translate model names into checkpoints or run
decision inference.
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

When migrating from the former model-specific client, follow the
[migration order](decision-api.md#migration) before writing the new model ID.
The service must implement model-qualified readiness; a different identity
cannot satisfy startup checks. Restart SCS after changing configuration.
Changing the decision model requires no index rebuild.

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
