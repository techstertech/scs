# Bundled model integration evidence

Date: 2026-11-14. This is source implementation evidence, not a published release.
The [spec](../../../specs/2026-11-14-bundled-local-models.md) records acceptance
states. The [component comparison](../../../docs/local-model-comparison-2026-11-14.md)
retains separate retrieval and routing measurements.

The host was an Apple M3 Ultra with 256 GiB RAM and CPython 3.14.4. All existing
services remained running. Validation used one private SCS root and generated
repository sources. It did not change user configuration or installed services.
Temporary paths and process identities in the reports identify only that run.

## First use and restart

| Evidence | Observation |
| --- | --- |
| [First-use MCP](first-use-mcp.json) | All ten review checks pass after automatic downloads into an initially empty model cache. Native indexing creates 18 nodes. Fixture source remains unchanged. |
| [Concurrent progress probe](first-use-progress.json) | Health stays ready while the durable job reports `models`, at 2,404,927,377 of 3,261,138,146 bytes. |
| [Initial persistent fixture](initial-online.json) | The dedicated smoke driver creates its retained fixture index using the verified cache. Indexing takes 6.23 seconds; local first-query routing succeeds without fallback. |
| [Restart before correction](restart-timeout-before-fix.json) | A cached restart exposes a first-query classifier timeout during lazy loading. The failed report remains failed. |
| [First isolated wheel restart](installed-wheel-first-offline.json) | Offline startup takes 24.99 seconds on the first use of the disposable Python installation. The first query takes 37 ms and direct semantic search with reranking takes 294 ms. |
| [Reusable online recipe](installed-wheel-online.json) | A fresh daemon from the same wheel starts in 5.04 seconds, then queries the existing index before any ingestion. |
| [Reusable offline recipe](installed-wheel-offline.json) | OS IP denial returns `EPERM`. A fresh daemon starts in 5.05 seconds; its first query takes 39 ms and direct semantic search with reranking takes 236 ms. |

The corrected runtime verifies and loads a complete cache before publishing
readiness. A missing cache still prepares models in the background indexing job.
Startup never downloads models. Regression tests also cover recovery from a
transient cached embedding load failure without restarting the daemon.

Each successful installed-wheel run verifies the pinned Laya identity without
fallback, the expected `preserve_input` symbol, semantic embeddings, reranking,
unchanged fixture hashes, complete cached artifact bytes, and daemon shutdown.
The model selected `INVENTORY` for this broad function query. The fixture passes
because the relevant function is returned; it does not establish optimal workflow
selection or general routing accuracy.

## Distribution and limits

[Wheel inspection](wheel-inspection.json) records SHA-256
`c8bbe94438a7d93aaa86f5dbc2a6d16f85a138b793832f34ebac419d3b6d1320`.
All 74 packaged Python, typing, and manifest files match the source bytes tested.
The wheel includes the native module and model manifest. Its metadata selects
the three inference dependencies only on supported Macs. Locked Linux CPython
3.14 dependency resolution succeeds without inference packages. Marker checks
also exclude them on macOS 13. No Linux or older-Mac execution was performed.

The disposable installation used the exported `uv.lock` runtime constraints.
Import checks resolved SCS, its daemon, and its native module inside that
installation. The driver also ran outside the checkout for the first isolated
wheel proof. Both reusable recipes then exercised the same immutable wheel.

`vmmap -summary` reports a combined daemon physical footprint and peak of **4.3G**
after indexing, routing, and reranking. Only those totals are retained. This is a
small-fixture observation on a larger Mac. It does not prove memory or latency
on physical 16 GB hardware, under long inputs, or under controlled host load.
That qualification remains open before a release claims the 16 GB target.

No installed service was replaced. Raw constraints, wheel environments, model
downloads, and temporary fixture state are disposable; reports remain in Git.
