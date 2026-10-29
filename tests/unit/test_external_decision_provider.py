"""SCS sends only policy choices and validates external choice responses."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from urllib.parse import parse_qs, urlsplit

import pytest

from scs.orchestration.decision import Playbook, RoutingRequest, eligible_playbooks
from scs.orchestration.external_decision_provider import ExternalDecisionProvider


MODEL_ID = "fixture/decision-model@revision"
READY_ROUTE = "/decisions/ready?model=fixture%2Fdecision-model%40revision"


def decision() -> dict[str, object]:
    available = eligible_playbooks(RoutingRequest(goal="find parser"))

    return {
        "model": MODEL_ID,
        "choice": "UNDERSTAND",
        "confidence": 0.8,
        "probabilities": {
            playbook.value: 0.8 if playbook is Playbook.UNDERSTAND else 0.2 / (len(available) - 1)
            for playbook in available
        },
    }


@pytest.fixture
def decision_service() -> Iterator[tuple[str, list[tuple[str, object]]]]:
    calls: list[tuple[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            query = parse_qs(urlsplit(self.path).query)
            calls.append(("GET", query))
            self.respond({"status": "ok", "model": query.get("model", [""])[0]})

        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(("POST", payload))
            self.respond({**decision(), "model": payload["model"]})

        def respond(self, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [MODEL_ID, "another/model@revision+build"])
async def test_configured_model_crosses_readiness_and_choice_http_boundaries(
    decision_service: tuple[str, list[tuple[str, object]]], model: str
) -> None:
    base_url, calls = decision_service
    provider = ExternalDecisionProvider(base_url, model=model)

    try:
        await provider.start()
        answer = await provider.classify(RoutingRequest(goal="find parser"))

        assert answer.model == model
        assert calls[0] == ("GET", {"model": [model]})
        method, payload = calls[-1]
        assert method == "POST"
        assert isinstance(payload, dict)
        assert payload["model"] == model
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_provider_sends_bounded_request_and_accepts_complete_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def request(
        _self: ExternalDecisionProvider,
        method: str,
        route: str,
        payload: dict[str, object] | None = None,
    ) -> object:
        calls.append((method, route, payload))
        if route == READY_ROUTE:
            return {"status": "ok", "model": MODEL_ID}
        return decision()

    monkeypatch.setattr(ExternalDecisionProvider, "_request", request)
    provider = ExternalDecisionProvider("http://127.0.0.1:10000/v1", model=MODEL_ID)
    try:
        await provider.start()
        answer = await provider.classify(RoutingRequest(goal="find parser"))
        assert answer.playbook is Playbook.UNDERSTAND
        method, route, payload = calls[-1]
        assert (method, route) == ("POST", "/decisions")
        assert payload is not None
        assert payload["state"] == {"goal": "find parser", "node_ids": [], "file_paths": []}
        assert payload["model"] == MODEL_ID
        question = payload["question"]
        assert isinstance(question, dict)
        assert set(question["criteria"]) == {
            playbook.value for playbook in eligible_playbooks(RoutingRequest(goal="find parser"))
        }
        assert set(answer.probabilities) == set(Playbook)
        assert answer.probabilities[Playbook.IMPACT] == 0
        assert answer.probabilities[Playbook.REFERENCES] == 0
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_provider_rejects_incompatible_service_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ExternalDecisionProvider,
        "_request",
        lambda *_args, **_kwargs: {"status": "ok", "model": "wrong"},
    )
    provider = ExternalDecisionProvider("http://127.0.0.1:10000/v1", model=MODEL_ID)
    with pytest.raises(RuntimeError, match="unavailable or incompatible"):
        await provider.start()
    await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_choice", ["ARBITRARY_TOOL", Playbook.REFERENCES.value])
async def test_provider_rejects_invalid_choice_response(
    monkeypatch: pytest.MonkeyPatch,
    invalid_choice: str,
) -> None:
    def request(
        _self: ExternalDecisionProvider,
        _method: str,
        route: str,
        _payload: dict[str, object] | None = None,
    ) -> object:
        if route == READY_ROUTE:
            return {"status": "ok", "model": MODEL_ID}
        answer = decision()
        answer["choice"] = invalid_choice
        return answer

    monkeypatch.setattr(ExternalDecisionProvider, "_request", request)
    provider = ExternalDecisionProvider("http://127.0.0.1:10000/v1", model=MODEL_ID)
    try:
        await provider.start()
        with pytest.raises(ValueError):
            await provider.classify(RoutingRequest(goal="find parser"))
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_provider_rejects_response_from_a_different_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def request(
        _self: ExternalDecisionProvider,
        _method: str,
        route: str,
        _payload: dict[str, object] | None = None,
    ) -> object:
        if route == READY_ROUTE:
            return {"status": "ok", "model": MODEL_ID}
        return {**decision(), "model": "another/model@revision"}

    monkeypatch.setattr(ExternalDecisionProvider, "_request", request)
    provider = ExternalDecisionProvider("http://127.0.0.1:10000/v1", model=MODEL_ID)

    try:
        await provider.start()
        with pytest.raises(ValueError, match="model identity"):
            await provider.classify(RoutingRequest(goal="find parser"))
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_provider_pauses_during_service_outage_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthy = True

    def request(
        _self: ExternalDecisionProvider,
        _method: str,
        route: str,
        _payload: dict[str, object] | None = None,
    ) -> object:
        if route == READY_ROUTE:
            return {"status": "ok", "model": MODEL_ID} if healthy else None
        return decision()

    monkeypatch.setattr(ExternalDecisionProvider, "_request", request)
    monkeypatch.setattr("scs.orchestration.external_decision_provider.HEALTH_INTERVAL_SECONDS", 0.01)
    provider = ExternalDecisionProvider("http://127.0.0.1:10000/v1", model=MODEL_ID)
    try:
        await provider.start()
        healthy = False
        async with asyncio.timeout(1):
            while provider.is_ready:
                await asyncio.sleep(0.01)
        waiting = asyncio.create_task(provider.wait_ready())
        await asyncio.sleep(0.02)
        assert not waiting.done()
        healthy = True
        await asyncio.wait_for(waiting, timeout=1)
        assert provider.is_ready
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_provider_shuts_down_after_bounded_failed_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthy = True
    unavailable = asyncio.Event()

    def request(
        _self: ExternalDecisionProvider,
        _method: str,
        _route: str,
        _payload: dict[str, object] | None = None,
    ) -> object:
        return {"status": "ok", "model": MODEL_ID} if healthy else None

    monkeypatch.setattr(ExternalDecisionProvider, "_request", request)
    monkeypatch.setattr("scs.orchestration.external_decision_provider.HEALTH_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr("scs.orchestration.external_decision_provider.RECOVERY_TIMEOUT_SECONDS", 0.02)
    provider = ExternalDecisionProvider(
        "http://127.0.0.1:10000/v1", model=MODEL_ID, on_unavailable=unavailable.set
    )
    try:
        await provider.start()
        healthy = False
        await asyncio.wait_for(unavailable.wait(), timeout=1)
        assert not provider.is_ready
        with pytest.raises(RuntimeError, match="recovery failed"):
            await provider.wait_ready()
    finally:
        await provider.close()
