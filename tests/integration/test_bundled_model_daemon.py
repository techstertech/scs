"""Exercise bundled composition through the real daemon, worker, and storage.

Only native inference and model transfers are generated fakes. SCSWire,
durable indexing, native parsing, vector storage, and runtime admission remain
real so startup, progress, failure, and shutdown observations cover their owners.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
import threading
from typing import NoReturn, cast, override

import pytest

from scs.config import SCSSettings
from scs.local_models.artifacts import (
    ModelArtifactError,
    ModelDownloadCancelled,
    ModelRole,
    ModelSpec,
    load_manifest,
)
from scs.local_models.backends import EMBEDDING_DIMENSION
from scs.local_models.runtime import LocalModelRuntime, Progress
import scs.main as scs_main
from scs.main import SCSDaemon
from scs.orchestration.decision import (
    DecisionProviderKind,
    Playbook,
    RoutingDecision,
    RoutingRequest,
)
from scs.orchestration.external_decision_provider import ExternalDecisionProvider
from scs.wire.client import SCSClient

TEST_TIMEOUT_SECONDS = 10.0
POLL_SECONDS = 0.02


def _object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _objects(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    return [_object(item) for item in cast(list[object], value)]


@dataclass(slots=True)
class GeneratedCache:
    root: Path
    block: bool = False
    fail: bool = False
    cached_roles: frozenset[ModelRole] = frozenset()
    forbid_ensure: bool = False
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    calls: list[ModelRole] = field(default_factory=list)
    cached_calls: list[ModelRole] = field(default_factory=list)

    def cached(
        self,
        spec: ModelSpec,
        *,
        cancelled: threading.Event | None = None,
    ) -> Path | None:
        if cancelled is not None and cancelled.is_set():
            raise ModelDownloadCancelled("generated cache verification cancelled")
        self.cached_calls.append(spec.role)
        if spec.role not in self.cached_roles:
            return None
        directory = self.root / spec.role.value
        assert directory.is_dir()
        return directory

    def ensure(
        self,
        spec: ModelSpec,
        *,
        progress: Progress | None = None,
        cancelled: threading.Event | None = None,
    ) -> Path:
        if self.forbid_ensure:
            raise AssertionError("cached startup attempted model provisioning")
        self.calls.append(spec.role)
        total = sum(artifact.size_bytes for artifact in spec.artifacts)
        if progress is not None:
            progress(spec.identity, 0, total)
        self.entered.set()
        if self.block:
            while not self.release.wait(POLL_SECONDS):
                if cancelled is not None and cancelled.is_set():
                    self.cancelled.set()
                    raise ModelDownloadCancelled("generated transfer cancelled")
        if self.fail:
            raise ModelArtifactError("generated artifact checksum mismatch")
        directory = self.root / spec.role.value
        directory.mkdir(parents=True, exist_ok=True)
        if progress is not None:
            progress(spec.identity, total, total)
        return directory


@dataclass(slots=True)
class GeneratedBackend:
    loads: list[ModelRole] = field(default_factory=list)
    document_batches: int = 0
    query_batches: int = 0
    rerank_batches: int = 0
    decision_requests: list[RoutingRequest] = field(default_factory=list)
    closed: bool = False
    load_entered: threading.Event | None = None
    load_release: threading.Event | None = None
    embedding_load_failures: int = 0

    def load(self, role: ModelRole, path: Path) -> None:
        assert path.is_dir()
        if self.load_entered is not None:
            self.load_entered.set()
        if self.load_release is not None:
            assert self.load_release.wait(TEST_TIMEOUT_SECONDS)
        if role is ModelRole.EMBEDDING and self.embedding_load_failures:
            self.embedding_load_failures -= 1
            raise RuntimeError("generated transient embedding load failure")
        self.loads.append(role)

    def embed(self, texts: list[str], *, query: bool) -> list[list[float]]:
        if query:
            self.query_batches += 1
        else:
            self.document_batches += 1
        return [[1.0] + [0.0] * (EMBEDDING_DIMENSION - 1) for _ in texts]

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        assert query
        self.rerank_batches += 1
        return [float(len(documents) - index) for index in range(len(documents))]

    def classify(self, request: RoutingRequest) -> RoutingDecision:
        self.decision_requests.append(request)
        return RoutingDecision(
            playbook=Playbook.DISCOVER,
            provider=DecisionProviderKind.LOCAL,
            model=load_manifest()[ModelRole.DECISION].identity,
            confidence=1.0,
            probabilities={item: float(item is Playbook.DISCOVER) for item in Playbook},
        )

    def close(self) -> None:
        self.closed = True


class ObservedRuntime(LocalModelRuntime):
    """Expose query admission without replacing the production worker."""

    def __init__(
        self,
        cache_root: Path,
        *,
        roles: Sequence[ModelRole],
        harness: Harness,
    ) -> None:
        self._query_entered: asyncio.Event = harness.query_entered
        super().__init__(
            cache_root,
            roles=roles,
            cache=harness.cache,
            backend_factory=harness.backend,
        )

    @override
    async def embed_query(self, text: str) -> list[float]:
        self._query_entered.set()
        return await super().embed_query(text)


@dataclass(slots=True)
class Harness:
    cache: GeneratedCache
    runtimes: list[LocalModelRuntime] = field(default_factory=list)
    backends: list[GeneratedBackend] = field(default_factory=list)
    query_entered: asyncio.Event = field(default_factory=asyncio.Event)
    block_load: bool = False
    embedding_load_failures: int = 0
    load_entered: threading.Event = field(default_factory=threading.Event)
    load_release: threading.Event = field(default_factory=threading.Event)

    def backend(self) -> GeneratedBackend:
        backend = GeneratedBackend(
            load_entered=self.load_entered if self.block_load else None,
            load_release=self.load_release if self.block_load else None,
            embedding_load_failures=self.embedding_load_failures,
        )
        self.backends.append(backend)
        return backend

    def runtime(
        self, cache_root: Path, *, roles: Sequence[ModelRole]
    ) -> LocalModelRuntime:
        runtime = ObservedRuntime(cache_root, roles=roles, harness=self)
        self.runtimes.append(runtime)
        return runtime


@pytest.fixture
def runtime_directory() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="scs-bundled-", dir="/tmp") as directory:
        yield Path(directory)


def _settings(
    tmp_path: Path,
    runtime: Path,
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> SCSSettings:
    user_home = tmp_path / "user"

    def home(_cls: type[Path]) -> Path:
        return user_home

    monkeypatch.setattr(Path, "home", classmethod(home))
    monkeypatch.setattr("scs.config.supports_local_models", lambda: True)
    for name in tuple(os.environ):
        if name.startswith("SCS_") or name == "OPENAI_API_KEY":
            monkeypatch.delenv(name, raising=False)

    def forbid_external(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("bundled composition constructed an external provider")

    class ForbiddenDecision(ExternalDecisionProvider):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            forbid_external()

    monkeypatch.setattr(scs_main, "LocalModelRuntime", harness.runtime)
    monkeypatch.setattr(scs_main, "OpenAICompatibleEmbeddingProvider", forbid_external)
    monkeypatch.setattr(scs_main, "OpenAICompatibleRerankingProvider", forbid_external)
    monkeypatch.setattr(scs_main, "MLXEmbeddingProvider", forbid_external)
    monkeypatch.setattr(scs_main, "ExternalDecisionProvider", ForbiddenDecision)
    home_path = tmp_path / "scs"
    return SCSSettings(
        model_profile="local",
        home=home_path,
        runtime_dir=runtime,
        log_dir=home_path / "logs",
        model_cache=home_path / "models",
        auto_reindex_enabled=False,
    )


def _repository(path: Path) -> bytes:
    path.mkdir()
    source = "\n\n".join(
        f'def {name}(value: int) -> int:\n    """Compute {name}."""\n    return value + {index}\n'
        for index, name in enumerate(("alpha", "beta"))
    ).encode()
    (path / "example.py").write_bytes(source)
    return source


async def _index(client: SCSClient, repository: Path) -> str:
    accepted = await client.call("repository.index", {"repo_path": str(repository)})
    assert accepted["accepted"] is True
    identity = _object(accepted["job"])["id"]
    assert isinstance(identity, str)
    return identity


async def _terminal_job(
    client: SCSClient,
    repository: Path,
    identity: str,
) -> tuple[dict[str, object], dict[str, object]]:
    async with asyncio.timeout(TEST_TIMEOUT_SECONDS):
        while True:
            stats = await client.call("knowledge.stats", {"repo_path": str(repository)})
            job = _object(stats["latest_job"])
            assert job["id"] == identity
            if job["status"] in {"completed", "failed", "cancelled"}:
                return job, stats
            await asyncio.sleep(POLL_SECONDS)


@pytest.mark.asyncio
async def test_complete_cache_warms_before_daemon_publishes_readiness(
    tmp_path: Path,
    runtime_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = GeneratedCache(
        tmp_path / "scs/models",
        cached_roles=frozenset(ModelRole),
        forbid_ensure=True,
    )
    for role in ModelRole:
        (cache.root / role.value).mkdir(parents=True)
    harness = Harness(cache, block_load=True)
    settings = _settings(tmp_path, runtime_directory, harness, monkeypatch)
    daemon = SCSDaemon(settings)
    repository = tmp_path / "repository"
    _repository(repository)
    started = asyncio.create_task(daemon.start())
    try:
        async with asyncio.timeout(TEST_TIMEOUT_SECONDS):
            while not harness.load_entered.is_set():
                if started.done():
                    await started
                    pytest.fail("daemon became ready before loading its cached models")
                await asyncio.sleep(POLL_SECONDS)
        assert not started.done()
        assert not (runtime_directory / "scs.sock").exists()
        assert cache.calls == []

        harness.load_release.set()
        await asyncio.wait_for(started, timeout=TEST_TIMEOUT_SECONDS)
        client = SCSClient(runtime_directory / "scs.sock")
        assert (await client.call("system.health"))["ready"] is True
        assert (await client.call("knowledge.stats"))["total_nodes"] == 0
        assert set(cache.cached_calls) == set(ModelRole)
        assert set(harness.backends[0].loads) == set(ModelRole)
        result = await client.call(
            "knowledge.query",
            {
                "repo_path": str(repository),
                "goal": "find the alpha implementation",
                "mode": "balanced",
            },
        )
        routing = _object(result["routing"])
        assert routing["provider"] == "local"
        assert routing["model"] == load_manifest()[ModelRole.DECISION].identity
        assert routing["fallback_applied"] is False
        assert len(harness.backends[0].decision_requests) == 1
        assert cache.calls == []
    finally:
        harness.load_release.set()
        await asyncio.gather(started, return_exceptions=True)
        await daemon.stop()


@pytest.mark.asyncio
async def test_cached_embedding_load_failure_recovers_without_daemon_restart(
    tmp_path: Path,
    runtime_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = GeneratedCache(tmp_path / "scs/models", cached_roles=frozenset(ModelRole))
    for role in ModelRole:
        (cache.root / role.value).mkdir(parents=True)
    harness = Harness(cache, embedding_load_failures=1)
    settings = _settings(tmp_path, runtime_directory, harness, monkeypatch)
    daemon = SCSDaemon(settings)
    client = SCSClient(
        runtime_directory / "scs.sock", timeout_seconds=TEST_TIMEOUT_SECONDS
    )
    repository = tmp_path / "repository"
    _repository(repository)
    try:
        await daemon.start()
        health = await client.call("system.health")
        assert health["ready"] is True
        assert harness.backends[0].embedding_load_failures == 0
        assert ModelRole.EMBEDDING not in harness.backends[0].loads
        assert harness.runtimes[0].unavailable_reason(ModelRole.EMBEDDING) is not None

        identity = await _index(client, repository)
        job, stats = await _terminal_job(client, repository, identity)
        assert job["status"] == "completed", job
        assert harness.runtimes[0].unavailable_reason(ModelRole.EMBEDDING) is None
        embedding_count = stats["embedding_count"]
        assert isinstance(embedding_count, int) and embedding_count > 0
        assert stats["semantic_search_ready"] is True
        assert (await client.call("system.health"))["generation"] == health[
            "generation"
        ]
        assert len(harness.runtimes) == 1 and len(harness.backends) == 1
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_local_preparation_keeps_daemon_responsive_and_finishes_query(
    tmp_path: Path,
    runtime_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(GeneratedCache(tmp_path / "scs/models", block=True))
    settings = _settings(tmp_path, runtime_directory, harness, monkeypatch)
    daemon = SCSDaemon(settings)
    client = SCSClient(
        runtime_directory / "scs.sock", timeout_seconds=TEST_TIMEOUT_SECONDS
    )
    repository = tmp_path / "repository"
    original = _repository(repository)
    try:
        await daemon.start()
        assert (await client.call("system.health"))["ready"] is True
        assert (await client.call("knowledge.stats"))["total_nodes"] == 0
        assert (await client.call("jobs.recent"))["jobs"] == []
        assert harness.cache.calls == [] and harness.backends == []

        await daemon.stop()
        daemon = SCSDaemon(settings)
        await daemon.start()
        assert (await client.call("knowledge.stats"))["total_nodes"] == 0
        assert (await client.call("jobs.recent"))["jobs"] == []
        assert harness.cache.calls == [] and harness.backends == []

        identity = await _index(client, repository)
        assert await asyncio.to_thread(harness.cache.entered.wait, TEST_TIMEOUT_SECONDS)
        assert not harness.cache.release.is_set()
        assert (await client.call("system.health"))["ready"] is True
        preparing = await client.call("knowledge.stats", {"repo_path": str(repository)})
        job = _object(preparing["latest_job"])
        assert job["id"] == identity
        assert job["status"] == "running" and job["phase"] == "models"
        assert isinstance(job["total"], int) and job["total"] > 0
        assert preparing["semantic_search_ready"] is False
        assert harness.backends[0].loads == []

        harness.cache.release.set()
        job, stats = await _terminal_job(client, repository, identity)
        assert job["status"] == "completed", job
        assert stats["semantic_search_ready"] is True
        assert set(harness.cache.calls) == set(ModelRole)
        backend = harness.backends[0]
        assert set(backend.loads) == set(ModelRole)
        result = await client.call(
            "knowledge.query",
            {
                "repo_path": str(repository),
                "goal": "find the alpha implementation",
                "mode": "balanced",
                "limit": 10,
            },
        )
        routing = _object(result["routing"])
        assert routing["provider"] == "local"
        assert routing["model"] == load_manifest()[ModelRole.DECISION].identity
        assert routing["fallback_applied"] is False
        symbols = _objects(_object(result["evidence"])["symbols"])
        assert any(item.get("name") == "alpha" for item in symbols), symbols
        assert backend.document_batches > 0 and backend.query_batches > 0
        assert backend.rerank_batches > 0 and len(backend.decision_requests) == 1
    finally:
        harness.cache.release.set()
        await daemon.stop()

    assert harness.backends[0].closed is True
    assert (repository / "example.py").read_bytes() == original


@pytest.mark.asyncio
async def test_required_model_failure_never_acknowledges_source_hash(
    tmp_path: Path,
    runtime_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(GeneratedCache(tmp_path / "scs/models", fail=True))
    settings = _settings(tmp_path, runtime_directory, harness, monkeypatch)
    daemon = SCSDaemon(settings)
    client = SCSClient(
        runtime_directory / "scs.sock", timeout_seconds=TEST_TIMEOUT_SECONDS
    )
    repository = tmp_path / "repository"
    original = _repository(repository)
    try:
        await daemon.start()
        identity = await _index(client, repository)
        job, stats = await _terminal_job(client, repository, identity)
        assert job["status"] == "failed", job
        assert "checksum mismatch" in str(job["error"])
        assert stats["semantic_search_ready"] is False
        assert stats["embedding_count"] == 0
        # These statistics read native ingestion acknowledgements. Structure
        # alone cannot create a successful source-hash record.
        ingestion = _object(stats["ingestion_stats"])
        assert set(ingestion) == {str(repository.resolve())}
        acknowledged = _object(ingestion[str(repository.resolve())])
        assert acknowledged["file_count"] == 0
        assert acknowledged["last_indexed"] == ""
        assert (await client.call("system.health"))["ready"] is True
        assert harness.backends[0].loads == []
        assert set(harness.cache.calls) == {ModelRole.EMBEDDING}
    finally:
        await daemon.stop()

    assert (repository / "example.py").read_bytes() == original


@pytest.mark.asyncio
async def test_shutdown_interrupts_transfer_before_draining_active_search(
    tmp_path: Path,
    runtime_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(GeneratedCache(tmp_path / "scs/models", block=True))
    settings = _settings(tmp_path, runtime_directory, harness, monkeypatch)
    daemon = SCSDaemon(settings)
    # This direct search intentionally outlives the shutdown guard. Query-code
    # classification deadlines would otherwise conceal the server-drain cycle.
    client = SCSClient(
        runtime_directory / "scs.sock", timeout_seconds=TEST_TIMEOUT_SECONDS * 2
    )
    repository = tmp_path / "repository"
    _repository(repository)
    search: asyncio.Task[dict[str, object]] | None = None
    try:
        await daemon.start()
        await _index(client, repository)
        assert await asyncio.to_thread(harness.cache.entered.wait, TEST_TIMEOUT_SECONDS)
        search = asyncio.create_task(
            client.call(
                "knowledge.search",
                {
                    "repo_path": str(repository),
                    "query": "alpha",
                    "search_mode": "fast",
                },
            )
        )
        await asyncio.wait_for(
            harness.query_entered.wait(), timeout=TEST_TIMEOUT_SECONDS
        )
        assert not search.done()

        await asyncio.wait_for(daemon.stop(), timeout=TEST_TIMEOUT_SECONDS)
        result = await search
        assert result["semantic_available"] is False
        assert harness.cache.cancelled.is_set()
        assert harness.backends[0].closed is True
        assert not (runtime_directory / "scs.sock").exists()
    finally:
        harness.cache.release.set()
        if search is not None and not search.done():
            search.cancel()
            await asyncio.gather(search, return_exceptions=True)
        await daemon.stop()
