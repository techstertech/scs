"""Adapt bundled routing to the same decision port as custom services."""

from __future__ import annotations

from scs.local_models.runtime import LocalModelRuntime
from scs.orchestration.decision import RoutingDecision, RoutingRequest


class LocalDecisionProvider:
    def __init__(self, runtime: LocalModelRuntime) -> None:
        self._runtime: LocalModelRuntime = runtime

    async def classify(self, request: RoutingRequest) -> RoutingDecision:
        return await self._runtime.classify(request)
