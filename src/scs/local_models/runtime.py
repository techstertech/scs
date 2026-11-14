"""One lazy native worker shared by indexing and query event loops.

Cancellation ends a caller's wait. It cannot stop an admitted native operation.
Capacity therefore belongs to the concurrent future, not to its async waiter.
The thread-safe admission gate also covers indexing's short-lived event loops.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
import logging
from pathlib import Path
import threading
from typing import Final, Protocol, TypeVar

from scs.local_models.artifacts import ModelCache, ModelRole, ModelSpec, load_manifest
from scs.local_models.backends import (
    EMBEDDING_FORMAT_FINGERPRINT,
    INFERENCE_BATCH_SIZE,
    InferenceBackend,
)
from scs.orchestration.decision import RoutingDecision, RoutingRequest
from scs.providers.base import ProviderUnavailableError, RankedDocument

logger = logging.getLogger(__name__)
NATIVE_BATCH_SIZE: Final[int] = INFERENCE_BATCH_SIZE
ADMISSION_POLL_SECONDS: Final[float] = 0.005
Progress = Callable[[str, int, int], None]
ResultT = TypeVar("ResultT")


class ArtifactCache(Protocol):
    def cached(
        self,
        spec: ModelSpec,
        *,
        cancelled: threading.Event | None = None,
    ) -> Path | None: ...

    def ensure(
        self,
        spec: ModelSpec,
        *,
        progress: Progress | None = None,
        cancelled: threading.Event | None = None,
    ) -> Path: ...


class ModelBackend(Protocol):
    def load(self, role: ModelRole, path: Path) -> None: ...
    def embed(self, texts: list[str], *, query: bool) -> list[list[float]]: ...
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...
    def classify(self, request: RoutingRequest) -> RoutingDecision: ...
    def close(self) -> None: ...


class LocalModelRuntime:
    """Own selected artifacts, model residency, and bounded native execution."""

    def __init__(
        self,
        cache_root: Path,
        *,
        roles: Sequence[ModelRole],
        cache: ArtifactCache | None = None,
        backend_factory: Callable[[], ModelBackend] = InferenceBackend,
    ) -> None:
        selected = set(roles)
        if not selected:
            raise ValueError("local runtime requires at least one selected role")
        self._roles: tuple[ModelRole, ...] = tuple(
            role for role in ModelRole if role in selected
        )
        self._specs: dict[ModelRole, ModelSpec] = load_manifest()
        self._cache: ArtifactCache = (
            cache if cache is not None else ModelCache(cache_root)
        )
        self._factory: Callable[[], ModelBackend] = backend_factory
        self._backend: ModelBackend | None = None
        self._loaded: set[ModelRole] = set()
        self._errors: dict[ModelRole, str] = {}
        self._state: threading.Lock = threading.Lock()
        self._capacity: threading.BoundedSemaphore = threading.BoundedSemaphore(1)
        self._stopping: threading.Event = threading.Event()
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="scs-models"
        )
        self._closing: Future[None] | None = None

    def identity(self, role: ModelRole) -> str:
        """Include formatting in durable embedding identity, not remote identities."""

        identity = self._specs[role].identity
        if role is ModelRole.EMBEDDING:
            return f"{identity}#{EMBEDDING_FORMAT_FINGERPRINT}"
        return identity

    def unavailable_reason(self, role: ModelRole) -> str | None:
        with self._state:
            return self._errors.get(role)

    def _check_open(self) -> None:
        if self._stopping.is_set():
            raise ProviderUnavailableError("local model runtime is closed")

    def _prepare(self, required: ModelRole, progress: Progress | None) -> ModelBackend:
        self._check_open()
        if self._backend is None:
            self._backend = self._factory()
        backend = self._backend
        if required in self._loaded:
            return backend

        total = sum(
            artifact.size_bytes
            for role in self._roles
            for artifact in self._specs[role].artifacts
        )
        completed = 0
        # Warm other selected roles during first indexing, but keep optional
        # download or load failures from disabling a working embedding model.
        order = (required, *(role for role in self._roles if role is not required))
        for role in order:
            spec = self._specs[role]
            size = sum(item.size_bytes for item in spec.artifacts)
            if role in self._loaded:
                completed += size
                continue
            role_completed = 0

            def report(identity: str, done: int, _total: int) -> None:
                nonlocal role_completed
                role_completed = done
                if progress is not None:
                    progress(identity, completed + done, total)

            try:
                self._check_open()
                path = self._cache.ensure(
                    spec,
                    progress=report,
                    cancelled=self._stopping,
                )
                # Returned cache paths have every byte verified. A later model
                # load failure does not undo that completed transfer.
                role_completed = size
                backend.load(role, path)
                self._loaded.add(role)
                with self._state:
                    self._errors.pop(role, None)
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                with self._state:
                    self._errors[role] = str(error)
                self._check_open()
                if role is required:
                    raise ProviderUnavailableError(
                        f"local {role.value} model is unavailable: {error}"
                    ) from error
                logger.warning(
                    "Optional local %s model is unavailable: %s", role.value, error
                )
            completed += role_completed
        return backend

    def _execute(
        self,
        role: ModelRole,
        operation: Callable[[ModelBackend], ResultT],
        progress: Progress | None,
    ) -> ResultT:
        try:
            result = operation(self._prepare(role, progress))
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            with self._state:
                self._errors[role] = str(error)
            if isinstance(error, ProviderUnavailableError):
                raise
            raise ProviderUnavailableError(
                f"local {role.value} inference failed: {error}"
            ) from error
        with self._state:
            self._errors.pop(role, None)
        return result

    async def _admit(self, operation: Callable[[], ResultT]) -> ResultT:
        self._check_open()
        while not self._capacity.acquire(blocking=False):
            self._check_open()
            await asyncio.sleep(ADMISSION_POLL_SECONDS)
        try:
            with self._state:
                self._check_open()
                future = self._executor.submit(operation)
        except BaseException:
            self._capacity.release()
            raise
        future.add_done_callback(lambda _future: self._capacity.release())
        wrapped = asyncio.wrap_future(future)
        # A timeout can leave the shielded future without a waiter. Observe a
        # late failure while preserving its exception for any active waiter.
        wrapped.add_done_callback(_observe_completion)
        return await asyncio.shield(wrapped)

    async def _invoke(
        self,
        role: ModelRole,
        operation: Callable[[ModelBackend], ResultT],
        progress: Progress | None = None,
    ) -> ResultT:
        if role not in self._roles:
            raise ProviderUnavailableError(f"local {role.value} role is not selected")
        return await self._admit(lambda: self._execute(role, operation, progress))

    def _warm_cached(self) -> bool:
        """Keep verification and model loading outside the first query's deadline."""

        verified: dict[ModelRole, Path] = {}
        for role in self._roles:
            self._check_open()
            try:
                path = self._cache.cached(self._specs[role], cancelled=self._stopping)
            except (OSError, RuntimeError, ValueError) as error:
                self._check_open()
                logger.warning(
                    "Cached local %s model is unavailable: %s", role.value, error
                )
                return False
            if path is None:
                return False
            verified[role] = path

        if self._backend is None:
            self._backend = self._factory()
        for role, path in verified.items():
            if role in self._loaded:
                continue
            try:
                self._check_open()
                self._backend.load(role, path)
                self._loaded.add(role)
                with self._state:
                    self._errors.pop(role, None)
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                self._check_open()
                with self._state:
                    self._errors[role] = str(error)
                logger.warning(
                    "Cached local %s model could not load: %s", role.value, error
                )
        return all(role in self._loaded for role in self._roles)

    async def warm_cached(self) -> bool:
        """Warm a complete verified profile without downloads or cache mutation.

        Cold startup stays cheap. Cached startup pays local loading costs
        before readiness, instead of timing out the first classified query.
        """

        return await self._admit(self._warm_cached)

    async def embed_documents(
        self,
        texts: Sequence[str],
        *,
        progress: Progress | None = None,
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), NATIVE_BATCH_SIZE):
            batch = list(texts[offset : offset + NATIVE_BATCH_SIZE])
            vectors.extend(
                await self._invoke(
                    ModelRole.EMBEDDING,
                    lambda backend: backend.embed(batch, query=False),
                    progress,
                )
            )
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        rows = await self._invoke(
            ModelRole.EMBEDDING,
            lambda backend: backend.embed([text], query=True),
        )
        return rows[0]

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        limit: int,
    ) -> list[RankedDocument]:
        if not documents or limit < 1:
            return []
        scores: list[float] = []
        for offset in range(0, len(documents), NATIVE_BATCH_SIZE):
            batch = list(documents[offset : offset + NATIVE_BATCH_SIZE])
            scores.extend(
                await self._invoke(
                    ModelRole.RERANKING,
                    lambda backend: backend.rerank(query, batch),
                )
            )
        order = sorted(range(len(scores)), key=lambda index: -scores[index])
        return [
            RankedDocument(index=index, score=scores[index]) for index in order[:limit]
        ]

    async def classify(self, request: RoutingRequest) -> RoutingDecision:
        return await self._invoke(
            ModelRole.DECISION, lambda backend: backend.classify(request)
        )

    def request_stop(self) -> None:
        """Interrupt transfer before waiting for the durable indexing worker."""

        self._stopping.set()

    def _dispose(self) -> None:
        try:
            if self._backend is not None:
                self._backend.close()
        finally:
            self._backend = None
            self._loaded.clear()

    async def close(self) -> None:
        """Drain admitted work and release models on their owning thread."""

        with self._state:
            self._stopping.set()
            if self._closing is None:
                self._closing = self._executor.submit(self._dispose)
            closing = self._closing
        wrapped = asyncio.wrap_future(closing)
        wrapped.add_done_callback(_observe_completion)
        try:
            await asyncio.shield(wrapped)
        finally:
            await asyncio.to_thread(self._executor.shutdown, wait=True)


def _observe_completion(future: asyncio.Future[ResultT]) -> None:
    if not future.cancelled():
        future.exception()
