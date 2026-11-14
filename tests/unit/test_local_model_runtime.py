from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
import threading
from typing import override

import pytest

from scs.local_models.artifacts import (
    ModelDownloadCancelled,
    ModelRole,
    ModelSpec,
    load_manifest,
)
from scs.local_models.runtime import LocalModelRuntime
from scs.orchestration.decision import (
    DecisionProviderKind,
    Playbook,
    RoutingDecision,
    RoutingRequest,
)
from scs.providers.base import ProviderUnavailableError


class Cache:
    def __init__(self, root: Path) -> None:
        self.root: Path = root
        self.roles: list[ModelRole] = []

    def cached(
        self,
        spec: ModelSpec,
        *,
        cancelled: threading.Event | None = None,
    ) -> Path | None:
        del spec, cancelled
        return None

    def ensure(
        self,
        spec: ModelSpec,
        *,
        progress: Callable[[str, int, int], None] | None = None,
        cancelled: threading.Event | None = None,
    ) -> Path:
        if cancelled is not None and cancelled.is_set():
            raise ModelDownloadCancelled("stopped")
        self.roles.append(spec.role)
        total = sum(item.size_bytes for item in spec.artifacts)
        if progress is not None:
            progress(spec.identity, total, total)
        return self.root / spec.role.value


class Backend:
    def __init__(self) -> None:
        self.loaded: list[ModelRole] = []
        self.batches: list[list[str]] = []
        self.threads: set[int] = set()
        self.closed: bool = False

    def load(self, role: ModelRole, path: Path) -> None:
        del path
        self.loaded.append(role)
        self.threads.add(threading.get_ident())

    def embed(self, texts: list[str], *, query: bool) -> list[list[float]]:
        del query
        self.threads.add(threading.get_ident())
        self.batches.append(texts)
        return [[1.0] + [0.0] * 1023 for _ in texts]

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        del query
        self.threads.add(threading.get_ident())
        self.batches.append(documents)
        return [float(item.removeprefix("document-")) for item in documents]

    def classify(self, request: RoutingRequest) -> RoutingDecision:
        del request
        self.threads.add(threading.get_ident())
        return RoutingDecision(
            provider=DecisionProviderKind.LOCAL,
            model="generated-model",
            playbook=Playbook.DISCOVER,
            confidence=1.0,
            probabilities={item: float(item is Playbook.DISCOVER) for item in Playbook},
        )

    def close(self) -> None:
        self.threads.add(threading.get_ident())
        self.closed = True


@pytest.mark.asyncio
async def test_cached_warmup_never_downloads_or_loads_a_partial_profile(
    tmp_path: Path,
) -> None:
    backend = Backend()
    cache = Cache(tmp_path)
    runtime = LocalModelRuntime(
        tmp_path,
        roles=(ModelRole.EMBEDDING, ModelRole.DECISION),
        cache=cache,
        backend_factory=lambda: backend,
    )
    try:
        assert await runtime.warm_cached() is False
        assert cache.roles == []
        assert backend.loaded == []
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_complete_cache_warms_before_first_query_without_transfer(
    tmp_path: Path,
) -> None:
    class CompleteCache(Cache):
        @override
        def cached(
            self,
            spec: ModelSpec,
            *,
            cancelled: threading.Event | None = None,
        ) -> Path | None:
            del cancelled
            return self.root / spec.role.value

    backend = Backend()
    cache = CompleteCache(tmp_path)
    roles = (ModelRole.EMBEDDING, ModelRole.RERANKING, ModelRole.DECISION)
    runtime = LocalModelRuntime(
        tmp_path, roles=roles, cache=cache, backend_factory=lambda: backend
    )
    try:
        assert await runtime.warm_cached() is True
        assert backend.loaded == list(roles)
        assert (
            await runtime.classify(RoutingRequest(goal="find code"))
        ).provider is DecisionProviderKind.LOCAL
        assert cache.roles == []
        assert len(backend.threads) == 1
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_local_construction_is_lazy_and_batches_remain_bounded(
    tmp_path: Path,
) -> None:
    cache = Cache(tmp_path)
    backend = Backend()
    runtime = LocalModelRuntime(
        tmp_path,
        roles=(ModelRole.EMBEDDING, ModelRole.DECISION),
        cache=cache,
        backend_factory=lambda: backend,
    )

    assert cache.roles == []
    assert backend.loaded == []
    assert not tmp_path.joinpath("models").exists()
    progress: list[tuple[str, int, int]] = []
    try:
        vectors = await runtime.embed_documents(
            [f"document-{i}" for i in range(11)],
            progress=lambda model, done, total: progress.append((model, done, total)),
        )
        assert len(vectors) == 11
        assert cache.roles == [ModelRole.EMBEDDING, ModelRole.DECISION]
        assert backend.loaded == cache.roles
        assert [len(batch) for batch in backend.batches] == [4, 4, 3]
        assert progress[-1][1] == progress[-1][2]
        await runtime.embed_query("another")
        assert cache.roles == [ModelRole.EMBEDDING, ModelRole.DECISION]
    finally:
        await runtime.close()
    assert backend.closed
    assert len(backend.threads) == 1


@pytest.mark.asyncio
async def test_rerank_positions_survive_multiple_native_batches(tmp_path: Path) -> None:
    backend = Backend()
    runtime = LocalModelRuntime(
        tmp_path,
        roles=(ModelRole.RERANKING,),
        cache=Cache(tmp_path),
        backend_factory=lambda: backend,
    )
    try:
        ranked = await runtime.rerank(
            "query",
            [f"document-{i}" for i in range(11)],
            limit=3,
        )
        assert [item.index for item in ranked] == [10, 9, 8]
        assert [len(batch) for batch in backend.batches] == [4, 4, 3]
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_empty_rerank_requests_never_prepare_models(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    backend = Backend()
    runtime = LocalModelRuntime(
        tmp_path,
        roles=(ModelRole.RERANKING,),
        cache=cache,
        backend_factory=lambda: backend,
    )
    try:
        assert await runtime.rerank("query", ["document-1"], limit=0) == []
        assert await runtime.rerank("query", [], limit=1) == []
        assert cache.roles == []
        assert backend.loaded == []
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_cancellation_does_not_admit_a_second_native_operation(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingBackend(Backend):
        @override
        def embed(self, texts: list[str], *, query: bool) -> list[list[float]]:
            if texts == ["first"]:
                entered.set()
                assert release.wait(timeout=3)
            return super().embed(texts, query=query)

    backend = BlockingBackend()
    runtime = LocalModelRuntime(
        tmp_path,
        roles=(ModelRole.EMBEDDING,),
        cache=Cache(tmp_path),
        backend_factory=lambda: backend,
    )
    first = asyncio.create_task(runtime.embed_query("first"))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(runtime.embed_query("second"))
        await asyncio.sleep(0.02)
        assert backend.batches == []
        assert not second.done()
        release.set()
        assert await asyncio.wait_for(second, 3)
        assert backend.batches == [["first"], ["second"]]
        assert len(backend.threads) == 1
    finally:
        release.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_indexing_and_query_event_loops_share_one_native_worker(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingBackend(Backend):
        @override
        def embed(self, texts: list[str], *, query: bool) -> list[list[float]]:
            entered.set()
            assert release.wait(timeout=3)
            return super().embed(texts, query=query)

    backend = BlockingBackend()
    runtime = LocalModelRuntime(
        tmp_path,
        roles=(ModelRole.EMBEDDING, ModelRole.DECISION),
        cache=Cache(tmp_path),
        backend_factory=lambda: backend,
    )
    indexing = asyncio.create_task(
        asyncio.to_thread(
            lambda: asyncio.run(runtime.embed_documents(["indexed document"]))
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        query = asyncio.create_task(runtime.classify(RoutingRequest(goal="find code")))
        release.set()
        assert await indexing
        assert (await query).provider is DecisionProviderKind.LOCAL
        assert len(backend.threads) == 1
    finally:
        release.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_shutdown_cancels_transfer_and_rejects_late_work(tmp_path: Path) -> None:
    entered = threading.Event()

    class SlowCache(Cache):
        @override
        def ensure(
            self,
            spec: ModelSpec,
            *,
            progress: Callable[[str, int, int], None] | None = None,
            cancelled: threading.Event | None = None,
        ) -> Path:
            del spec, progress
            entered.set()
            assert cancelled is not None
            assert cancelled.wait(timeout=3)
            raise ModelDownloadCancelled("stopped")

    backend = Backend()
    runtime = LocalModelRuntime(
        tmp_path,
        roles=(ModelRole.EMBEDDING,),
        cache=SlowCache(tmp_path),
        backend_factory=lambda: backend,
    )
    operation = asyncio.create_task(runtime.embed_query("pending"))
    assert await asyncio.to_thread(entered.wait, 3)
    await asyncio.wait_for(runtime.close(), 3)
    with pytest.raises(ProviderUnavailableError):
        await operation
    with pytest.raises(ProviderUnavailableError, match="closed"):
        await runtime.embed_query("late")
    await runtime.close()


@pytest.mark.asyncio
async def test_optional_partial_failure_keeps_embedding_and_truthful_progress(
    tmp_path: Path,
) -> None:
    class PartialCache(Cache):
        @override
        def ensure(
            self,
            spec: ModelSpec,
            *,
            progress: Callable[[str, int, int], None] | None = None,
            cancelled: threading.Event | None = None,
        ) -> Path:
            if spec.role is ModelRole.RERANKING:
                self.roles.append(spec.role)
                if progress is not None:
                    progress(
                        spec.identity,
                        1,
                        sum(item.size_bytes for item in spec.artifacts),
                    )
                raise ProviderUnavailableError("partial optional model unavailable")
            return super().ensure(spec, progress=progress, cancelled=cancelled)

    backend = Backend()
    runtime = LocalModelRuntime(
        tmp_path,
        roles=tuple(ModelRole),
        cache=PartialCache(tmp_path),
        backend_factory=lambda: backend,
    )
    observed: list[tuple[str, int, int]] = []
    specs = load_manifest()
    total = sum(item.size_bytes for spec in specs.values() for item in spec.artifacts)
    missing = sum(item.size_bytes for item in specs[ModelRole.RERANKING].artifacts) - 1

    def report(identity: str, completed: int, size: int) -> None:
        observed.append((identity, completed, size))

    try:
        assert await runtime.embed_documents(["working document"], progress=report)
        assert backend.loaded == [ModelRole.EMBEDDING, ModelRole.DECISION]
        assert (
            runtime.unavailable_reason(ModelRole.RERANKING)
            == "partial optional model unavailable"
        )
        assert runtime.unavailable_reason(ModelRole.EMBEDDING) is None
        assert observed[-1][1:] == (total - missing, total)
        assert [item[1] for item in observed] == sorted(item[1] for item in observed)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_cleanup_failure_still_terminates_the_native_worker(
    tmp_path: Path,
) -> None:
    workers: set[threading.Thread] = set()

    class FailingCloseBackend(Backend):
        @override
        def close(self) -> None:
            workers.add(threading.current_thread())
            super().close()
            raise RuntimeError("generated GPU cleanup failure")

    backend = FailingCloseBackend()
    runtime = LocalModelRuntime(
        tmp_path,
        roles=(ModelRole.EMBEDDING,),
        cache=Cache(tmp_path),
        backend_factory=lambda: backend,
    )
    try:
        assert await runtime.embed_query("load model")
        with pytest.raises(RuntimeError, match="generated GPU cleanup failure"):
            await runtime.close()
        assert backend.closed
        assert workers and all(not worker.is_alive() for worker in workers)
        with pytest.raises(ProviderUnavailableError, match="closed"):
            await runtime.embed_query("late request")
    finally:
        # A second close must also be harmless to worker ownership. Its original
        # cleanup exception can still be reported to callers.
        with pytest.raises(RuntimeError, match="generated GPU cleanup failure"):
            await runtime.close()
