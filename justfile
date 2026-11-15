set shell := ["sh", "-cu"]

repository := "leonardoventurini/scs"
local_eval_scripts := "scripts/export-model-corpus.py scripts/evaluate-local-embeddings.py scripts/evaluate-local-reranking.py scripts/evaluate-local-routing.py scripts/smoke-local-models.py scripts/smoke-review.py"
local_eval_tests := "tests/unit/test_model_corpus_export.py tests/unit/test_local_embedding_evaluation.py tests/unit/test_local_reranking_evaluation.py tests/unit/test_local_routing_evaluation.py tests/unit/test_local_model_smoke.py tests/unit/test_review_smoke.py"

setup:
    uv sync --all-groups
    ./scripts/build-native.sh
    uv run --all-groups pre-commit install

test:
    uv run --all-groups pytest -v

coverage: native-build
    uv run --all-groups coverage erase
    uv run --all-groups coverage run -m pytest -q
    uv run --all-groups coverage report

typecheck:
    uv run --all-groups basedpyright

lint:
    uv run ruff check src tests

native-test:
    cargo test --workspace

native-build:
    ./scripts/build-native.sh

eval-search suite="evals/scs-search-v1.json" repo="." k="10" repeats="1":
    uv run python scripts/evaluate-search.py --suite "{{suite}}" --repo "{{repo}}" --k "{{k}}" --repeats "{{repeats}}" --result-detail compact

eval-query suite="evals/scs-query-v4.json" repo="." mode="balanced" k="10" repeats="1":
    uv run python scripts/evaluate-query.py --suite "{{suite}}" --repo "{{repo}}" --mode "{{mode}}" --k "{{k}}" --repeats "{{repeats}}"

# Check model evaluation and smoke tools without loading models.
eval-local-check:
    uv run --frozen --all-groups basedpyright {{local_eval_scripts}} {{local_eval_tests}}
    uv run --frozen --all-groups ruff check {{local_eval_scripts}} {{local_eval_tests}}
    uv run --frozen --all-groups pytest -q {{local_eval_tests}}

verify: typecheck lint coverage native-test

# Install or upgrade the local SCS tool from a GitHub release (default: latest).
install version="":
    #!/usr/bin/env bash
    set -euo pipefail
    version="{{version}}"
    if [ -z "$version" ]; then
        tag="$(gh release view --repo {{repository}} --json tagName --jq .tagName)"
        version="${tag#v}"
    fi
    scripts/install.sh --version "$version"

alias upgrade := install

# Exercise review features through fresh stdio MCP; enroll and remove one fixture.
smoke-review command=".venv/bin/scs" output="/tmp/scs-review-smoke.json":
    uv run python scripts/smoke-review.py --command "{{command}}" --output "{{output}}"

# Retain private fixture state and verified models for a fresh-process rerun.
smoke-local-models state_root command=".venv/bin/scs" output="/tmp/scs-local-model-smoke.json":
    uv run --frozen python scripts/smoke-local-models.py --command "{{command}}" --state-root "{{state_root}}" --output "{{output}}"

# On macOS, deny IP traffic while retaining the daemon's Unix socket transport.
smoke-local-models-offline state_root command=".venv/bin/scs" output="/tmp/scs-local-model-offline.json":
    /usr/bin/sandbox-exec -p '(version 1) (allow default) (deny network-outbound (remote ip)) (deny network-inbound (local ip))' .venv/bin/python scripts/smoke-local-models.py --command "{{command}}" --state-root "{{state_root}}" --output "{{output}}" --offline
