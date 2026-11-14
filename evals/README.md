# SCS search evaluation suites

Search suites use strict, versioned JSON:

~~~json
{
  "schema_version": 1,
  "name": "repository-name",
  "cases": [
    {
      "query": "natural-language code question",
      "relevant": [
        {
          "qualified_name": "package.module.Symbol.method",
          "file_path": "src/package/module.py",
          "relevance": 3
        }
      ]
    }
  ]
}
~~~

The qualified name is required. The file path is optional but should be
included when the same qualified name occurs in more than one indexed source
path. Relevance is a positive integer; larger values indicate more useful
results.

Run the bundled SCS suite against the current checkout:

~~~bash
just eval-search
~~~

Override the repository, suite, cutoff, or number of repeated requests:

~~~bash
just eval-search evals/scs-search-v1.json /path/to/scs 10 3
~~~

The JSON report includes per-query Recall@k, reciprocal rank, nDCG@k, response
bytes, mean request latency, retrieval mode, graph statistics, and reranker
identity. Model-backed timings are intentionally observational rather than CI
gates. Commit representative suites for each evaluated repository and compare
reports produced with the same suite version and cutoff. The runner waits for
active indexing jobs to finish before collecting measurements.

## Unified query judgments

`scs-query-v4.json` is the current 17-case, seven-playbook suite. V2 expanded
file-level relevance judgments after auditing pooled candidates against source.
Earlier suites and reports remain historical. Compare reports only when their
suite versions match.

Grades are source-backed: 3 is the direct implementation or primary test, 2 is
an entry point, direct reference, or focused supporting test, and 1 is useful
context or a valid Python function file. The IMPACT case includes all three
tests that directly import `build_mcp` from the changed file. The REFERENCES
case includes files that instantiate or import `SCSDaemon`; unrelated
`DaemonController` matches are unjudged. The INVENTORY case judges Python
function files from the pooled top results; Rust function files remain
unjudged because the goal specifies Python.

The suite contains positive judgments, not an exhaustive list of every useful
file. `unsupported_rate` therefore means absent from this judgment set, not
proven incorrect. The file-level metric does not measure symbol-level relevance
or whether a returned function has the requested language. Keep those limits in
mind when reviewing the retirement gate.

## Local model research

See the [November comparison](../docs/local-model-comparison-2026-11-14.md) for
measured results, offline proof, artifact sizes, and target hardware limitations.

The research commands compare model components with fixed inputs. They leave
live SCS configuration and indexes unchanged. Run `just setup` first to install
development dependencies and build the native parser.

Check the four research scripts and their tests with:

~~~bash
just eval-local-check
~~~

This command runs strict Basedpyright, scoped Ruff, and focused tests. It needs
no inference runtime or model download. Normal `just typecheck` and `just lint`
do not cover these scripts.

Export one committed source snapshot before comparing models:

~~~bash
uv run --frozen python scripts/export-model-corpus.py \
  --revision ec22366 --output /tmp/scs-model-corpus.json
~~~

The exporter uses production discovery, parsing, embedding text, and reranking
text. Output must be outside the source repository. The export records its
commit, hashes, file counts, and document counts. It counts and skips symlinks
and submodules. Committed input bytes remain stable across worktree edits.
The current SCS parser and text formatters determine the exported documents.

`scs-local-search-v1.json` contains 20 queries and 28 source-backed judgments,
validated against the frozen corpus. This comparison uses updated symbols
instead of the older search suite's stale judgments. Keep the corpus, suite
bytes, cutoff, and candidate count identical between compared runs.

Optional runtime packages use disposable `uv run --with` environments. The
following versions match the November 2026 comparison. They are research
dependencies, not SCS production dependencies.

~~~bash
uv run --frozen --with sentence-transformers==5.7.0 --with torch==2.15.0 \
  --with onnxruntime==1.32.0 python scripts/evaluate-local-embeddings.py \
  --profile qwen3-0.6b --corpus /tmp/scs-model-corpus.json \
  --suite evals/scs-local-search-v1.json --max-length 1024 \
  --cache-dir /tmp/scs-model-cache --output /tmp/scs-embedding-qwen.json
~~~

The four pinned profiles are `bge-small`, `qwen3-0.6b`, `nomic-v1.5-int8`, and
`jina-v2-code-int8`. Each profile fixes the model revision and preprocessing.
First use downloads the selected official model and tokenizer. Reuse the cache
and add `--offline` for a local-only reload. Use `--scs-prefixes` for a separate
comparison with SCS's current Nomic-style prefixes.

Compare rerankers against the same saved top-40 embedding candidates:

~~~bash
uv run --frozen --with sentence-transformers==5.7.0 --with torch==2.15.0 \
  --with onnxruntime==1.32.0 python scripts/evaluate-local-reranking.py \
  --profile minilm-int8 --corpus /tmp/scs-model-corpus.json \
  --candidates /tmp/scs-embedding-qwen.json --candidate-count 40 \
  --cache-dir /tmp/scs-model-cache --output /tmp/scs-reranker-minilm.json
~~~

Replace `minilm-int8` with `qwen` for the Qwen reranker. Keep the same candidate
file. The scripts check corpus and suite hashes before loading a reranker.

Compare decision routing on the separate 17-case query suite:

~~~bash
uv run --frozen python scripts/evaluate-local-routing.py \
  --backend deterministic --output /tmp/scs-routing-deterministic.json
uv run --frozen --with laya-mlx==0.6.0 python scripts/evaluate-local-routing.py \
  --backend laya --output /tmp/scs-routing-laya.json
~~~

Laya runs directly on Apple Silicon with a pinned model revision. Its loader
also supports `--offline` after the initial download. The `external` backend
requires a running service, its exact model ID, and its decision API URL.

Loader flags alone do not prove that networking is blocked. Apply an operating
system network sandbox and record a separate denial check for offline proof.

Embedding and reranking reports measure exact-vector retrieval components.
They exclude lexical fusion, graph expansion, HTTP transport, and public query
latency. Routing reports separate raw model choices from policy previews;
previews do not prove successful graph anchor resolution. Positive judgments
are not exhaustive relevance labels. Record the hardware for every run.
Measurements on a larger Mac do not establish performance on a 16 GB Mac.
