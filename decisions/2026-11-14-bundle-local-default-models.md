---
status: accepted
project: scs
project-root: /Users/leonardo/Repositories/scs
created: 2026-11-14
updated: 2026-11-14
supersedes:
  - decisions/2026-09-03-default-to-configurable-openai-embeddings.md (unconfigured supported Apple Silicon defaults only)
  - decisions/2026-09-25-use-external-laya-inference.md (external-only inference ownership only)
superseded-by:
related-specs:
  - specs/2026-11-14-bundled-local-models.md
implementation:
  commits: []
  pull-request:
---

# Bundle local default models while preserving external providers

## Context

A fresh SCS installation currently requires embedding credentials or a separate
model service. The user requested automatic first-use downloads and later offline
operation on a 16 GB Apple Silicon Mac, with a model budget below 8 GB.

Measured Qwen embeddings, Qwen reranking, and English Laya fit in 3.26 GB of
artifacts. Their component tests passed with network access denied. Custom
external services remain an explicit compatibility requirement.

## Decision

SCS owns a lazy local runtime for the bundled profile. Existing provider ports
remain the integration boundary. A fresh supported Mac selects this profile;
explicit existing model settings retain legacy external behavior. Explicit role
selectors support local and external combinations.

The daemon owns the verified artifact cache and one bounded inference worker.
First-use work runs within existing background indexing or bounded query work.
Inference libraries load only when needed. Fixed artifacts and model-native
formatting define local model identity.

Configured external services retain their transport, trust, identity, readiness,
recovery, and fallback contracts. Linux and unsupported Macs retain external
defaults. No external service implementation changes are required.

## Alternatives

Requiring a separately installed server preserves the former ownership rule but
does not provide the requested installation experience. A new local HTTP service
adds transport and service-manager ownership without serving another consumer.
An optional runtime extra alone leaves fresh users with a manual dependency step.

Changing defaults for all existing users could redirect traffic or rebuild
vectors unexpectedly. Automatic selection therefore preserves existing explicit
model settings. External adapters keep their existing formatting.

## Consequences

Supported Mac installations gain inference dependencies and a first-use artifact
download. SCS now owns local model memory, serialization, and cleanup. The former
external-only decision ownership rule is replaced; the external API itself stays
authoritative for custom services.

Model-specific cache validation and bounded work become product invariants.
The measured host has more than 16 GB RAM. Physical target validation remains
explicitly unverified until suitable hardware is available.
