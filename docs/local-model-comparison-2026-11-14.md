# Local model comparison for SCS

Date: 2026-11-14. Status: completed research; runtime integration remains a proposal.

## Recommendation

Prototype this default for 16 GB Apple Silicon Macs:

- **Embeddings:** Qwen3-Embedding-0.6B, BF16, with its native query instruction.
- **Reranking:** Qwen3-Reranker-0.6B, BF16, for balanced and thorough searches.
- **Routing:** English Laya through MLX, with the existing deterministic fallback.

These weights total **3,225,783,921 bytes, or 3.23 GB**. The downloaded snapshots,
including tokenizers and configuration, total **3,261,138,146 bytes, or 3.26 GB**.
The selected artifact manifest records exact sizes and checksums.
This leaves substantial space within the agreed 8 GB model
download budget. Runtime packages are additional installation space.

The smaller alternative uses Nomic INT8 embeddings, MiniLM INT8 reranking, and
English Laya. Its weights total **1,003,106,233 bytes, or 1.00 GB**. It trades
ranking quality for smaller downloads and CPU inference for retrieval components.

The quality recommendation follows this SCS sample. It is not a claim that these
models lead every code-search benchmark. The combined stack has not run on an
actual 16 GB Mac.

## Scope and current implementation

The requested target is Apple Silicon with 16 GB RAM. The first run can download
models. Later operation must work offline. The default model download must stay
below 8 GB.

SCS currently requires embeddings for semantic indexing. Reranking and external
decision routing are optional. Its default embedding provider requires an OpenAI
API key. The local MLX provider requires a separately installed runtime and model.
See [provider configuration](configuration.md).

```text
first run -> pinned artifact manifest -> download -> verify -> local cache
                                                               |
query -> local routing -> SCS playbook -> embeddings -> reranking
```

This is a proposed installation flow. This task changed research tools and
documentation. It did not change SCS providers, public APIs, persisted formats,
production dependencies, configuration, or service ownership.

The [external runtime ownership decision](../decisions/2026-09-25-use-external-laya-inference.md)
remains authoritative. Bundling decision inference requires an approved successor
contract before implementation. A separate process does not remove that ownership change.

## Method

Measurements used an M3 Ultra with 256 GB RAM, CPython 3.14.4, PyTorch 2.15.0,
Sentence Transformers 5.7.0, ONNX Runtime 1.32.0, MLX 0.32.4, and laya-mlx 0.6.0.
They do not establish M1/M2/M3 laptop latency or combined memory requirements.
Existing services stayed running. Host load was not controlled.

The immutable corpus uses SCS commit
`ec223664e309a14c979d8d27879fd27d7879dab4`: **305 files and 3,730 documents**.
Its document SHA-256 is
`8eceb8e2e65ac70741f815f6e56870fe61ebb35f434a6f6c19037a097b8edc73`.
The exporter reuses production discovery, parsing, embedding text, and reranking
text. It reads committed blobs and excludes symlinks and submodules explicitly.

The [new search suite](../evals/scs-local-search-v1.json) contains 20 queries and
28 source-checked judgments. The older search suite references a retired reranker
module. Its labels were unsuitable for this comparison.

Embedding runs used batch size four and three individual query repeats. The
context cap was 1,024 tokens, except BGE's native 512-token limit. No embedding
inputs were truncated. Exact cosine retrieval isolated embedding quality from
lexical fusion, graph expansion, HTTP transport, and daemon startup.

Rerankers received the same top 40 candidates from each embedding run. Both
metrics use a cutoff of 10. MiniLM used its 512-token limit; Qwen used a 2,048-token
cap and upstream yes/no scoring modules. Quantized and BF16 configurations are
deployment candidates, not a controlled precision-ablation study.

Recall@10 measures the fraction of judged targets returned. nDCG@10 measures
graded ranking quality; higher is better. The judgments are incomplete and cover
one repository. Repeats provide latency samples, not additional independent goals.

## Embeddings

| Candidate | Tested weights | Recall@10 | nDCG@10 | Query p95 | Corpus encoding |
| --- | ---: | ---: | ---: | ---: | ---: |
| [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), FP32/MPS | 133 MB | 52.5% | 0.421 | 9.6 ms | 10.6 s |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), BF16/MPS | 1,192 MB | 65.0% | 0.420 | 30.7 ms | 41.4 s |
| [Nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), INT8/CPU | 137 MB | 60.0% | 0.434 | 6.5 ms | 44.6 s |
| [Jina-v2-base-code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code), INT8/CPU | 162 MB | 50.0% | 0.276 | 5.6 ms | 42.1 s |

CPU ONNX runs used four threads. MPS runs used the GPU. These timings include
different runtimes and precisions. They do not isolate architecture efficiency.

Nomic gives the strongest compact embedding result here. Qwen retrieves more
judged targets and supplies a better candidate set for reranking. BGE builds the
corpus fastest. Jina's code-specific training did not improve these short,
metadata-focused SCS queries.

**Formatting is a material integration requirement.** SCS currently adds
`search_query:` and `search_document:` to every compatible-provider request.
Those are Nomic task prefixes. With those prefixes, Qwen recall fell from
65.0% to **35.0%**, and nDCG fell from 0.420 to **0.194**. Changing the model name
alone will not reproduce the native-format results.

## Reranking

| Candidate source | Reranker | Recall@10 | nDCG@10 | Added p95 latency |
| --- | --- | ---: | ---: | ---: |
| Qwen embeddings | None | 65.0% | 0.420 | — |
| Qwen embeddings | [MiniLM-L6-v2 INT8](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | 70.0% | 0.524 | 683 ms |
| Qwen embeddings | [Qwen3-Reranker-0.6B BF16](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | **75.0%** | **0.610** | 1,323 ms |
| Nomic embeddings | None | 60.0% | 0.434 | — |
| Nomic embeddings | MiniLM-L6-v2 INT8 | 67.5% | 0.474 | 799 ms |
| Nomic embeddings | Qwen3-Reranker-0.6B BF16 | 70.0% | 0.576 | 2,123 ms |

MiniLM adds only 23 MB of weights. Qwen adds 1,192 MB and improves quality more.
Both measured configurations stayed within SCS's five-second balanced reranking
ceiling on this host. Fast searches already skip reranking.

MiniLM truncated 246 of 2,400 pair evaluations with Qwen candidates. Its smaller
context is part of the deployment tradeoff. The Qwen wrapper did not expose a
truncation count; its configured cap is recorded explicitly.

## Decision routing

The existing v4 suite contains 17 goals. One expected REFERENCES label is
ineligible for the raw classifier because its anchors are absent. Raw accuracy
therefore uses **16 distinct goals**. SCS's explicit-intent rules run after model
classification. Policy previews measure those rules before graph anchor resolution.

| Candidate | Weight download | Raw accuracy | Policy preview accuracy | Initial p95 |
| --- | ---: | ---: | ---: | ---: |
| Production deterministic baseline | 0 | 25.0% | 41.2% | 0.007 ms |
| Existing [KEV-4B](https://huggingface.co/jaredpalmer/kev-4b) service | >9.45 GB unquantized bundle | 87.5% | 100% | 149.7 ms |
| [English Laya MLX](https://huggingface.co/aac6fef/laya-mlx) | 843 MB | **100%** | **100%** | 18.8 ms |
| [Multilingual Laya MLX](https://huggingface.co/aac6fef/laya-multilingual-mlx) | 644 MB | 87.5% | 88.2% | 8.9 ms |

The live service's implementation was inspected: it loads KEV-4B even when a
legacy Laya identity is requested. Laya trials therefore loaded actual Laya
weights directly. The external baseline excludes server loading and memory.
Provider confidence values have different definitions; raw probabilities and
Brier scores are preserved in the reports.

English Laya's later offline p95 values reached **206 ms** without the OS sandbox
and **271 ms** with it. Its initial low latency is not a guaranteed fast-mode
budget. All measured calls stayed below the default 500 ms decision timeout.
The tiny, hand-authored suite does not establish general routing accuracy.

Laya emitted a configuration warning for an unused 11-or-more-choice temperature
bucket. SCS supplied five to seven choices. No routing input was truncated.

## Offline operation and memory

Fresh processes used cached, pinned artifacts and `--offline`. The recommended
components also ran with network access denied:

```sh
/usr/bin/sandbox-exec -p '(version 1) (allow default) (deny network*)' ... --offline
```

A socket control failed with `EPERM` under the same profile. English Laya's 51
choices and full probability vectors matched the initial run exactly. Qwen's
embedding rankings also matched. Qwen reranking preserved every ranking and all
aggregate quality metrics. The evidence summary records the executed wrapper.
This tests components, not a complete
offline SCS installer or MCP session.

| Component | Process peak RSS | Separate Metal observation |
| --- | ---: | ---: |
| Qwen embeddings | 0.96 GB | 1.53 GB driver allocation at end |
| Qwen reranker, Qwen candidates | 0.81 GB | 2.50 GB driver allocation at end |
| English Laya | 1.09 GB | 1.17 GB peak allocation |
| Nomic INT8 embeddings | 0.70 GB | CPU only |
| MiniLM INT8 reranker, Qwen candidates | 0.47 GB | CPU only |

RSS and Metal measurements overlap. Do not add them. End-of-run driver allocation
is not a Metal peak. Combined residency, SCS graph memory, OS headroom, and actual
16 GB hardware remain unmeasured. Size and separate component observations make
the proposed stack plausible, not verified, for that target.

## Packaging implications and untested alternatives

The selected weights are ungated and Apache-2.0 licensed. BGE uses MIT. The tested
Python 3.14 ARM64 packages installed in an isolated environment. A user would not
need an API key or a separately managed model server once a local runtime is bundled.

Before integration, define model-specific formatting, bounded inference, pinned
artifact verification, complete-cache readiness, and local runtime ownership.
One early offline attempt correctly failed because tokenizer/configuration files
were missing despite cached weights. Readiness must cover the complete artifact set.

The tested runtimes are PyTorch/Transformers for Qwen, ONNX Runtime for compact
encoders/rerankers, and MLX for Laya. A [llama.cpp](https://github.com/ggml-org/llama.cpp)
deployment could use official Q8 Qwen embedding and reranker artifacts, about
639 MB each. Those quantized artifacts and that runtime were **not benchmarked**.
Do not transfer the BF16 quality or latency results to them.

[KEV-0.8B](https://huggingface.co/jaredpalmer/kev-0.8b), approximately 1.79 GB before
tokenizer files, is a feasible later routing challenger. It was researched but
not run. [Papr code embeddings](https://huggingface.co/papr-ai/papr-embed-code-1-0.6b)
are another untested Apache-2.0 candidate. No additional model was needed to
establish a workable profile within the current budget.

[EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m) adds an official
access gate and Gemma terms. Newer
[Jina code embeddings](https://huggingface.co/jinaai/jina-code-embeddings-0.5b) use
CC-BY-NC-4.0; this differs from the Apache-2.0 Jina-v2 model tested here.
The [Blaizzy MLX embeddings runtime](https://github.com/Blaizzy/mlx-embeddings)
declares GPLv3 and differs from SCS's older `mlx_embedding_models` package.
These are installation and distribution considerations, not measured quality failures.

## Verification and review

Research artifacts, exact revisions, checksums, per-query results, routing vectors,
offline replays, and environment requirements live in
[the report directory](../evals/reports/2026-11-14-local-models).
[Evaluation usage](../evals/README.md#local-model-research) explains the commands.

| Acceptance criterion | Result |
| --- | --- |
| Default model artifacts remain below 8 GB | Passed: 3.26 GB, with per-file checksums. |
| First use downloads the required artifacts | Passed for the component commands; an SCS installer was not built. |
| Subsequent inference works without network access | Passed for all three recommended components in fresh sandboxed processes. |
| Different model profiles receive measured comparisons | Passed: four embeddings, two rerankers, three learned routers, and a deterministic baseline. |
| The profile works on a physical 16 GB Mac | Not executed; suitability remains inferred from separate measurements on the larger host. |
| SCS can install and manage the complete profile | Deferred: runtime integration and ownership changes require a separate approved implementation. |

Executed repository checks:

- `just eval-local-check`: strict checking and Ruff passed for all eight research
  scripts/test files; all 32 focused tests passed.
- `just verify`: strict application checking, Ruff, all 535 Python tests, and Rust
  workspace tests passed. Python branch coverage was 88.13%.
- `scripts/verify-doc-links.sh`: passed. Scoped Ruff formatting and
  `git diff --check` also passed.
- Live SCS remained structurally and semantically ready after the trials.

The post-work review added reusable corpus export, model comparisons, and the
`eval-local-check` command. Guards reject stale judgments, mixed corpus/suite
snapshots, input overwrite, malformed scores, and stale success reports.
The existing global model-provenance guidance already covered the service alias
finding; no duplicate global rule was added. Next steps are physical 16 GB
validation, broader repository judgments, and approved runtime integration.

Review this report first, then the raw reports, then the evaluation scripts and
their tests. No deployment occurred. Reverting the research commit removes its
tools and documentation without changing installed SCS behavior.
