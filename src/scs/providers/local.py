"""Local inference adapters over SCS's existing semantic provider ports."""

from __future__ import annotations

from collections.abc import Sequence

from scs.local_models.artifacts import ModelRole
from scs.local_models.backends import EMBEDDING_DIMENSION
from scs.local_models.runtime import LocalModelRuntime, Progress
from scs.providers.base import ProviderMetadata, RankedDocument, RerankerMetadata


class LocalEmbeddingProvider:
    """Bind local embedding identity and optional indexing progress to one runtime."""

    def __init__(
        self, runtime: LocalModelRuntime, *, progress: Progress | None = None
    ) -> None:
        self._runtime: LocalModelRuntime = runtime
        self._progress: Progress | None = progress

    @property
    def metadata(self) -> ProviderMetadata:
        reason = self._runtime.unavailable_reason(ModelRole.EMBEDDING)
        return ProviderMetadata(
            "local",
            self._runtime.identity(ModelRole.EMBEDDING),
            EMBEDDING_DIMENSION,
            available=reason is None,
            reason=reason,
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._runtime.embed_documents(texts, progress=self._progress)

    async def embed_query(self, text: str) -> list[float]:
        return await self._runtime.embed_query(text)


class LocalRerankingProvider:
    """Keep local score ordering behind the existing reranking interface."""

    def __init__(self, runtime: LocalModelRuntime) -> None:
        self._runtime: LocalModelRuntime = runtime

    @property
    def metadata(self) -> RerankerMetadata:
        reason = self._runtime.unavailable_reason(ModelRole.RERANKING)
        return RerankerMetadata(
            "local",
            self._runtime.identity(ModelRole.RERANKING),
            available=reason is None,
            reason=reason,
        )

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        limit: int,
    ) -> list[RankedDocument]:
        return await self._runtime.rerank(query, documents, limit=limit)
