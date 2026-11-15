# SCS 0.5.0 release evidence

[SCS 0.5.0](https://github.com/techstertech/scs/releases/tag/v0.5.0) was published
from signed source commit `67903f4ea23afa724db26d67fef6b27a75a107c3` and installed
locally over 0.4.0. The [release spec](../../../specs/2026-11-14-release-0.5.0.md)
records acceptance and recovery. No external model service was restarted.

## Publication and installation

- [Main CI](main-ci.json) and [release CI](release-ci.json) passed, including both
  supported platforms and the supply-chain gate.
- [Asset inspection](assets.json) verifies all six payload checksums. Both wheels
  contain 74 source, typing, and manifest files equal to the tagged source.
  The source archive contains the same manifest; the installer matches its
  version-bound source.
- [Provenance](provenance.json) records successful verification of all seven
  assets. The policy requires the maintained release workflow, exact tag and
  source commit, SLSA provenance, and GitHub-hosted runners.
- [Disposable installation](disposable-install.json) used the public installer
  with the global SCS launcher hidden from its PATH. All 77 applicable locked
  dependencies match, and imports resolve inside that private installation.
- [Installed package](installed-package.json) verifies all 75 application and
  native files against the public wheel. The global installer replaced 0.4.0
  only after the public checks passed.

The release gates ran 639 Python tests, 108 Rust tests, and 42 tool tests.
Strict typing, Ruff, native builds, coverage at 88.79%, and performance ceilings
passed. The source-only review improvement below then passed 641 Python tests,
108 Rust tests, and 44 tool tests without changing the published application.

## Actual model and external-service checks

The [online model run](public-models-online.json) indexed an initially empty
generated fixture using the public wheel. Pinned model files were prefetched
and verified with SCS's `ModelCache` while hosted builds ran. This checks the
public wheel's indexing, rather than repeating the earlier source-only download
proof. Its first local query succeeds without routing fallback, and direct
search applies embeddings and reranking.

The [offline restart](public-models-offline.json) starts a fresh public-wheel
daemon and queries the retained index before any ingestion. macOS IP denial
returns `EPERM`; Unix sockets remain available. Local classification, semantic
search, reranking, source preservation, and daemon shutdown all pass.

Both runs report a combined physical footprint and peak of 4.3G on an M3 Ultra
with 256 GiB RAM. Physical 16 GB and older-macOS runtime qualification remain
unverified. Host load and long-input behavior were not controlled by this fixture.

The [initial external review](initial-external-review-failed.json) remains a
failure. Its first classifier call exceeds the unchanged 0.5-second deadline.
The original helper recorded only `ExceptionGroup`; the
[diagnostic extraction](external-diagnostic.json) preserves the observed
`classifier_timeout` and 501.2 ms classification time from stderr.

Separate [old-version](old-external-review.json),
[public-wheel warm](public-external-review-warm.json), and
[installed-version](installed-external-review.json) runs each pass all ten review
checks. The actual installed old and public new decision classes have identical
ASTs, instructions, and query-orchestrator bytes. This supports a transient
latency explanation; it does not prove identical cold-service latency. No
deadline, provider, or model setting was changed to obtain the later passes.

## Local state and lifecycle

[Upgrade verification](local-upgrade.json) confirms configuration bytes,
permissions, resolved settings, all 23 original project identities, and active
store generations are preserved. All retained graphs report structural and
semantic readiness. Fresh MCP inventory, one existing-source MCP read, and
`scs doctor` pass against the installed 0.5.0 runtime.

Four retained catalog roots have unavailable source folders. MCP rejects those
source paths as before. Storage checks use the same SCSWire boundary before and
after the upgrade; their healthy indexes were preserved. No project was deleted
or relocated to satisfy validation.

The final read-only job check finds 42 completed jobs created after the upgrade
and no outstanding jobs. A connection-refused embedding job predates the upgrade
while 0.4.0 was running; later full reconciliations of that repository succeed.
The old failed record remains intact.

Existing MCP sessions require reconnection after replacement. Fresh sessions
work. SCS remains a lazy daemon and can stop after the final bridge disconnects.
Live project paths, configuration contents, credentials, and daemon generation
identifiers are excluded from the retained upgrade summary.

## Review improvement

Commit `09e0365` improves the repository's smoke helper after the immutable
release. Failed reports retain their existing `passed` and `error` fields and
add bounded leaf messages in `errors`. Generated subprocess regressions cover a
missing executable and an MCP process that closes during initialization. Both
failed before the correction and pass afterward.

`just eval-local-check` now checks both MCP smoke drivers. Maintainer guidance
also distinguishes source-folder availability from retained-index readiness.
No new runtime layer or global skill rule was needed. The
[cleanup proof](cleanup.json) confirms that temporary model downloads,
installations, raw attestations, and private state proofs were removed after
all 14 retained report checksums verified. The reused native build cache remains
available.
