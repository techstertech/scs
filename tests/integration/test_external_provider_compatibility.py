"""Existing service settings retain their transport and composition contracts.

These tests exercise a daemon, native index, and generated HTTP peer together.
The peer records actual requests, so selecting a provider by name is not enough
to pass. Bundled model construction is forbidden for each external-only case.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from threading import Thread
from typing import Literal, NoReturn, cast, override
from urllib.parse import parse_qs, urlsplit

import pytest

from scs.config import SCSSettings
from scs.local_models.artifacts import ModelRole
import scs.main as scs_main
from scs.main import SCSDaemon
from scs.orchestration.decision import (
    DecisionProviderKind,
    Playbook,
    RoutingDecision,
    RoutingRequest,
)
from scs.wire.client import SCSClient

EMBEDDING_ID = "vendor/custom-embedding@revision+build"
RERANKING_ID = "vendor/custom-reranking@revision+build"
DECISION_ID = "vendor/custom-routing@revision+build"
UNRELATED_API_KEY = "fixture-openai-key-must-not-reach-compatible-services"
GOAL = "find the alpha implementation"
INDEX_TIMEOUT_SECONDS = 10.0


def _object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


@dataclass(frozen=True, slots=True)
class Request:
    method: str
    path: str
    authorization: str | None
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ServicePeer:
    """Serve each configured role while retaining wire-level evidence."""

    base_url: str = ""
    malformed_decision: bool = False
    wrong_ready_model: bool = False
    requests: list[Request] = field(default_factory=list)

    def posts(self, route: str) -> list[Request]:
        return [
            request
            for request in self.requests
            if request.method == "POST" and request.path == route
        ]


@pytest.fixture
def external_peer() -> Iterator[ServicePeer]:
    peer = ServicePeer()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            peer.requests.append(
                Request("GET", self.path, self.headers.get("Authorization"))
            )
            requested_model = query.get("model", [""])[0]
            self.respond(
                {
                    "status": "ok",
                    "model": "another-model"
                    if peer.wrong_ready_model
                    else requested_model,
                }
            )

        def do_POST(self) -> None:
            contents = self.rfile.read(int(self.headers["Content-Length"]))
            payload = _object(cast(object, json.loads(contents)))
            peer.requests.append(
                Request("POST", self.path, self.headers.get("Authorization"), payload)
            )
            if self.path == "/v1/embeddings":
                inputs = payload["input"]
                assert isinstance(inputs, list)
                self.respond(
                    {
                        "data": [
                            {"index": index, "embedding": [1.0, 0.5]}
                            for index in range(len(cast(list[object], inputs)))
                        ]
                    }
                )
            elif self.path == "/v1/rerank":
                documents = payload["documents"]
                assert isinstance(documents, list)
                count = len(cast(list[object], documents))
                top_n = payload["top_n"]
                assert isinstance(top_n, int)
                self.respond(
                    {
                        "results": [
                            {"index": index, "relevance_score": 1.0 - rank / count}
                            for rank, index in enumerate(
                                list(reversed(range(count)))[:top_n]
                            )
                        ]
                    }
                )
            elif self.path == "/v1/decisions":
                question = _object(payload["question"])
                criteria = _object(question["criteria"])
                self.respond(
                    {
                        "model": payload["model"],
                        "choice": "NOT_A_PLAYBOOK"
                        if peer.malformed_decision
                        else "DISCOVER",
                        "confidence": 1.0,
                        "probabilities": {
                            label: float(label == "DISCOVER") for label in criteria
                        },
                    }
                )
            else:
                self.send_error(404)

        def respond(self, payload: Mapping[str, object]) -> None:
            contents = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(contents)))
            self.end_headers()
            self.wfile.write(contents)

        @override
        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    peer.base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield peer
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.fixture
def short_runtime() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="scs-external-", dir="/tmp") as directory:
        yield Path(directory)


def _settings(
    tmp_path: Path,
    runtime: Path,
    peer: ServicePeer,
    monkeypatch: pytest.MonkeyPatch,
    *,
    roles: Literal["all", "embeddings"],
    with_api_key: bool = False,
) -> SCSSettings:
    user_home = tmp_path / "user"
    config_path = user_home / ".scs/config.toml"
    config_path.parent.mkdir(parents=True)
    lines = [
        'embedding_provider = "openai_compatible"',
        f'embedding_model = "{EMBEDDING_ID}"',
        "embedding_dimension = 2",
        f'openai_compatible_base_url = "{peer.base_url}"',
        'openai_compatible_trusted_hosts = ["127.0.0.1"]',
    ]
    if roles == "all":
        lines.extend(
            [
                f'reranking_model = "{RERANKING_ID}"',
                f'decision_model = "{DECISION_ID}"',
                f'decision_base_url = "{peer.base_url}"',
            ]
        )
    config_path.write_text("\n".join(lines) + "\n")

    def home(_cls: type[Path]) -> Path:
        return user_home

    monkeypatch.setattr(Path, "home", classmethod(home))
    if with_api_key:
        monkeypatch.setenv("OPENAI_API_KEY", UNRELATED_API_KEY)
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for name in (
        "SCS_MODEL_PROFILE",
        "SCS_EMBEDDING_PROVIDER",
        "SCS_EMBEDDING_MODEL",
        "SCS_EMBEDDING_DIMENSION",
        "SCS_OPENAI_COMPATIBLE_BASE_URL",
        "SCS_RERANKING_PROVIDER",
        "SCS_RERANKING_MODEL",
        "SCS_DECISION_PROVIDER",
        "SCS_DECISION_MODEL",
        "SCS_DECISION_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    def forbid_local_runtime(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("an external profile attempted bundled model construction")

    monkeypatch.setattr(
        scs_main, "LocalModelRuntime", forbid_local_runtime, raising=False
    )
    return SCSSettings(
        home=tmp_path / "scs",
        runtime_dir=runtime,
        log_dir=tmp_path / "logs",
        model_cache=tmp_path / "models",
        auto_reindex_enabled=False,
        decision_timeout_seconds=1.0,
    )


async def _index(client: SCSClient, repository: Path) -> None:
    repository.mkdir()
    (repository / "example.py").write_text(
        'def alpha(value: int) -> int:\n    """Return the given value."""\n    return value\n'
    )
    accepted = await client.call("repository.index", {"repo_path": str(repository)})
    assert accepted["accepted"] is True
    job_id = _object(accepted["job"])["id"]

    async with asyncio.timeout(INDEX_TIMEOUT_SECONDS):
        while True:
            stats = await client.call("knowledge.stats", {"repo_path": str(repository)})
            job = _object(stats["latest_job"])
            assert job["id"] == job_id
            assert job["status"] not in {"failed", "cancelled"}, job
            if job["status"] == "completed":
                assert stats["semantic_search_ready"] is True
                break
            await asyncio.sleep(0.05)


def _assert_external_vector_identity(home: Path) -> None:
    metadata_paths = list((home / "projects").glob("*/generations/*/provider.json"))
    assert len(metadata_paths) == 1
    metadata = _object(cast(object, json.loads(metadata_paths[0].read_text())))

    assert metadata["provider"] == "openai-compatible"
    assert metadata["model"] == EMBEDDING_ID
    assert metadata["dimension"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_decision", [False, True])
async def test_existing_service_profile_keeps_ids_payloads_and_failure_fallback(
    tmp_path: Path,
    short_runtime: Path,
    external_peer: ServicePeer,
    monkeypatch: pytest.MonkeyPatch,
    malformed_decision: bool,
) -> None:
    external_peer.malformed_decision = malformed_decision
    settings = _settings(
        tmp_path,
        short_runtime,
        external_peer,
        monkeypatch,
        roles="all",
        with_api_key=malformed_decision,
    )
    daemon = SCSDaemon(settings)
    try:
        await daemon.start()
        client = SCSClient(short_runtime / "scs.sock")
        repository = tmp_path / "repository"
        await _index(client, repository)
        _assert_external_vector_identity(settings.home)
        result = await client.call(
            "knowledge.query",
            {"repo_path": str(repository), "goal": GOAL, "mode": "balanced"},
        )

        routing = _object(result["routing"])
        assert routing["provider"] == (
            "deterministic" if malformed_decision else "external"
        )
        assert routing["fallback_applied"] is malformed_decision
        assert routing["model"] == (
            "deterministic-v1" if malformed_decision else DECISION_ID
        )
        assert (await client.call("system.health"))["ready"] is True
    finally:
        await daemon.stop()

    readiness = [
        request for request in external_peer.requests if request.method == "GET"
    ]
    assert readiness
    for request in readiness:
        parsed = urlsplit(request.path)
        assert parsed.path == "/v1/decisions/ready"
        assert parse_qs(parsed.query) == {"model": [DECISION_ID]}

    embeddings = external_peer.posts("/v1/embeddings")
    assert embeddings
    sent_text = [
        text
        for request in embeddings
        for text in cast(list[str], request.payload["input"])
    ]
    assert any(text.startswith("search_document: ") for text in sent_text)
    assert f"search_query: {GOAL}" in sent_text
    assert all(request.payload["model"] == EMBEDDING_ID for request in embeddings)

    reranking = external_peer.posts("/v1/rerank")
    assert reranking
    assert all(request.payload["model"] == RERANKING_ID for request in reranking)
    assert all(request.payload["return_documents"] is False for request in reranking)
    decisions = external_peer.posts("/v1/decisions")
    assert decisions
    assert all(request.payload["model"] == DECISION_ID for request in decisions)
    assert all(request.authorization is None for request in external_peer.requests)


@pytest.mark.asyncio
async def test_existing_embeddings_only_profile_keeps_optional_roles_disabled(
    tmp_path: Path,
    short_runtime: Path,
    external_peer: ServicePeer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        tmp_path, short_runtime, external_peer, monkeypatch, roles="embeddings"
    )
    assert settings.decision_model is None
    assert settings.reranking_model is None
    daemon = SCSDaemon(settings)
    try:
        await daemon.start()
        client = SCSClient(short_runtime / "scs.sock")
        repository = tmp_path / "repository"
        await _index(client, repository)
        _assert_external_vector_identity(settings.home)
        await daemon.stop()
        daemon = SCSDaemon(settings)
        await daemon.start()
        client = SCSClient(short_runtime / "scs.sock")
        stats = await client.call("knowledge.stats", {"repo_path": str(repository)})
        assert stats["semantic_search_ready"] is True
        _assert_external_vector_identity(settings.home)
        result = await client.call(
            "knowledge.query",
            {"repo_path": str(repository), "goal": GOAL, "mode": "balanced"},
        )
        assert _object(result["routing"])["provider"] == "deterministic"
    finally:
        await daemon.stop()

    assert external_peer.posts("/v1/embeddings")
    assert {request.path for request in external_peer.requests} == {"/v1/embeddings"}
    assert all(request.authorization is None for request in external_peer.requests)


@pytest.mark.asyncio
async def test_existing_decision_identity_mismatch_still_prevents_daemon_readiness(
    tmp_path: Path,
    short_runtime: Path,
    external_peer: ServicePeer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_peer.wrong_ready_model = True
    settings = _settings(
        tmp_path, short_runtime, external_peer, monkeypatch, roles="all"
    )
    daemon = SCSDaemon(settings)

    with pytest.raises(RuntimeError, match="unavailable or incompatible"):
        await daemon.start()

    assert not (short_runtime / "scs.sock").exists()
    assert external_peer.requests
    assert {request.method for request in external_peer.requests} == {"GET"}


@pytest.mark.asyncio
async def test_mixed_profile_runs_only_decisions_locally_and_preserves_external_retrieval(
    tmp_path: Path,
    short_runtime: Path,
    external_peer: ServicePeer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local routing must not acquire externally configured model responsibilities."""

    _settings(tmp_path, short_runtime, external_peer, monkeypatch, roles="embeddings")
    monkeypatch.setattr("scs.config.supports_local_models", lambda: True)
    settings = SCSSettings(
        model_profile="local",
        home=tmp_path / "scs",
        runtime_dir=short_runtime,
        log_dir=tmp_path / "logs",
        model_cache=tmp_path / "scs/models",
        auto_reindex_enabled=False,
        reranking_model=RERANKING_ID,
        decision_provider="local",
    )
    assert settings.embedding_provider == "openai_compatible"
    assert settings.reranking_provider == "external"
    local_identity = settings.decision_model
    assert isinstance(local_identity, str)
    instances: list[FakeLocalRuntime] = []

    class FakeLocalRuntime:
        def __init__(self, cache_root: Path, *, roles: Sequence[ModelRole]) -> None:
            self.cache_root: Path = cache_root
            self.roles: tuple[ModelRole, ...] = tuple(roles)
            self.requests: list[RoutingRequest] = []
            self.closed: bool = False
            instances.append(self)

        async def warm_cached(self) -> bool:
            return False

        async def classify(self, request: RoutingRequest) -> RoutingDecision:
            self.requests.append(request)
            return RoutingDecision(
                playbook=Playbook.DISCOVER,
                provider=DecisionProviderKind.LOCAL,
                model=local_identity,
                confidence=1.0,
                probabilities={
                    item: float(item is Playbook.DISCOVER) for item in Playbook
                },
            )

        def request_stop(self) -> None:
            pass

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(scs_main, "LocalModelRuntime", FakeLocalRuntime)
    daemon = SCSDaemon(settings)
    assert len(instances) == 1
    local_runtime = instances[0]
    assert local_runtime.roles == (ModelRole.DECISION,)
    assert local_runtime.cache_root == settings.paths.model_cache
    try:
        await daemon.start()
        client = SCSClient(short_runtime / "scs.sock")
        assert local_runtime.requests == []
        repository = tmp_path / "repository"
        await _index(client, repository)
        _assert_external_vector_identity(settings.home)
        assert local_runtime.requests == []
        result = await client.call(
            "knowledge.query",
            {"repo_path": str(repository), "goal": GOAL, "mode": "balanced"},
        )
        routing = _object(result["routing"])
        assert routing["provider"] == "local"
        assert routing["model"] == local_identity
        assert routing["fallback_applied"] is False
    finally:
        await daemon.stop()

    assert [request.goal for request in local_runtime.requests] == [GOAL]
    assert local_runtime.closed is True
    assert {request.path for request in external_peer.requests} == {
        "/v1/embeddings",
        "/v1/rerank",
    }
    assert external_peer.posts("/v1/embeddings")
    assert external_peer.posts("/v1/rerank")
    assert all(
        request.payload["model"] == EMBEDDING_ID
        for request in external_peer.posts("/v1/embeddings")
    )
    assert all(
        request.payload["model"] == RERANKING_ID
        for request in external_peer.posts("/v1/rerank")
    )
    assert all(request.authorization is None for request in external_peer.requests)
