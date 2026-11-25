from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import cast

import pytest

from scs.config import SCSSettings
from scs.main import SCSDaemon
from scs.providers.base import ProviderMetadata
from scs.wire.client import SCSClient


class _ImmediateEmbeddings:
    """Keep integration tests independent from external embedding providers."""

    @property
    def metadata(self) -> ProviderMetadata:
        return ProviderMetadata("test", "immediate", 2)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        del text
        return [0.0, 1.0]


@asynccontextmanager
async def _running_indexed_daemon(
    tmp_path: Path,
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[SCSDaemon, SCSClient]]:
    monkeypatch.setattr(
        "scs.main.OpenAICompatibleEmbeddingProvider",
        lambda **_kwargs: _ImmediateEmbeddings(),
    )
    runtime_dir = Path(tempfile.mkdtemp(dir="/tmp", prefix="scs-up-rt-"))
    settings = SCSSettings(
        decision_model="disabled",
        embedding_provider="openai_compatible",
        home=tmp_path / "home",
        model_cache=tmp_path / "models",
        runtime_dir=runtime_dir,
        log_dir=tmp_path / "logs",
        embedding_dimension=2,
        auto_reindex_enabled=False,
    )
    daemon = SCSDaemon(settings)
    await daemon.start()
    client = SCSClient(settings.paths.runtime / "scs.sock")
    try:
        idx = await client.call("repository.index", {"repo_path": str(repo)})
        job = cast(dict[str, object], idx["job"])
        stats = await client.call(
            "knowledge.stats",
            {
                "repo_path": str(repo),
                "wait_job_id": str(job["id"]),
                "wait_timeout_seconds": 4.0,
            },
        )
        wait = cast(dict[str, object], stats["wait"])
        assert wait["outcome"] == "terminal"
        wait_job = cast(dict[str, object], wait["job"])
        assert wait_job["status"] == "completed"
        yield daemon, client
    finally:
        await daemon.stop()
        shutil.rmtree(runtime_dir, ignore_errors=True)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


@pytest.mark.asyncio
async def test_symbol_call_sites_jsx_heritage_and_false_positive_elimination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / "ui").mkdir(parents=True)
    (repo / "browser_source_commands.cjs").write_text(
        "function readAdapterBundle() {\n"
        "  return { adapter: true };\n"
        "}\n"
        "\n"
        "function readAppFunctions() {\n"
        "  return ['fn1'];\n"
        "}\n"
        "\n"
        "module.exports = { readAdapterBundle, readAppFunctions };\n",
        encoding="utf-8",
    )
    (repo / "source_functions.test.mjs").write_text(
        'import { readAppFunctions } from "./browser_source_commands.cjs";\n'
        "\n"
        "export function checkAppFunctions() {\n"
        "  return readAppFunctions();\n"
        "}\n",
        encoding="utf-8",
    )
    (repo / "adapter_runner.mjs").write_text(
        'import { readAdapterBundle } from "./browser_source_commands.cjs";\n'
        "\n"
        "export function runAdapter() {\n"
        "  return readAdapterBundle();\n"
        "}\n",
        encoding="utf-8",
    )
    (repo / "ui" / "Badge.tsx").write_text(
        "export function Badge() {\n  return null;\n}\n",
        encoding="utf-8",
    )
    (repo / "ui" / "Widget.tsx").write_text(
        'import { Badge } from "./Badge";\n'
        "\n"
        "export class BasePanel {}\n"
        "export interface Renderable { render(): void; }\n"
        "export type WidgetConfig = { panel: BasePanel };\n"
        "\n"
        "export class CustomWidget extends BasePanel implements Renderable {\n"
        "  render() {\n"
        "    return <Badge />;\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    async with _running_indexed_daemon(tmp_path, repo, monkeypatch) as (_, client):
        bundle_res = await client.call(
            "knowledge.query",
            {
                "goal": "References to readAdapterBundle",
                "repo_path": str(repo),
                "symbol_name": "readAdapterBundle",
                "file_paths": ["browser_source_commands.cjs"],
                "playbook": "REFERENCES",
            },
        )
        bundle_ev = cast(dict[str, list[dict[str, object]]], bundle_res["evidence"])
        bundle_ref_files = {r.get("file_path") for r in bundle_ev["references"]}
        assert "adapter_runner.mjs" in bundle_ref_files
        assert "source_functions.test.mjs" not in bundle_ref_files
        assert bundle_ev["test_targets"] == []
        adapter_refs = [
            r
            for r in bundle_ev["references"]
            if r.get("file_path") == "adapter_runner.mjs"
            and r.get("reference_kind") == "calls"
        ]
        assert adapter_refs
        assert adapter_refs[0]["start_line"] == 3
        assert adapter_refs[0].get("enclosing_symbol") == "adapter_runner.runAdapter"

        fn_res = await client.call(
            "knowledge.query",
            {
                "goal": "References to readAppFunctions",
                "repo_path": str(repo),
                "symbol_name": "readAppFunctions",
                "file_paths": ["browser_source_commands.cjs"],
                "playbook": "REFERENCES",
            },
        )
        fn_ev = cast(dict[str, list[dict[str, object]]], fn_res["evidence"])
        fn_test_files = {t.get("file_path") for t in fn_ev["test_targets"]}
        assert "source_functions.test.mjs" in fn_test_files

        badge_res = await client.call(
            "knowledge.query",
            {
                "goal": "References to Badge",
                "repo_path": str(repo),
                "symbol_name": "Badge",
                "file_paths": ["ui/Badge.tsx"],
                "playbook": "REFERENCES",
            },
        )
        badge_refs = cast(dict[str, list[dict[str, object]]], badge_res["evidence"])[
            "references"
        ]
        renders = [r for r in badge_refs if r.get("reference_kind") == "renders"]
        assert renders
        assert renders[0]["file_path"] == "ui/Widget.tsx"
        assert renders[0]["start_line"] == 8

        panel_res = await client.call(
            "knowledge.query",
            {
                "goal": "References to BasePanel",
                "repo_path": str(repo),
                "symbol_name": "BasePanel",
                "file_paths": ["ui/Widget.tsx"],
                "playbook": "REFERENCES",
            },
        )
        panel_refs = cast(dict[str, list[dict[str, object]]], panel_res["evidence"])[
            "references"
        ]
        ref_kinds = {r.get("reference_kind") for r in panel_refs}
        assert "extends" in ref_kinds
        assert "type_references" in ref_kinds

        iface_res = await client.call(
            "knowledge.query",
            {
                "goal": "References to Renderable",
                "repo_path": str(repo),
                "symbol_name": "Renderable",
                "file_paths": ["ui/Widget.tsx"],
                "playbook": "REFERENCES",
            },
        )
        iface_refs = cast(dict[str, list[dict[str, object]]], iface_res["evidence"])[
            "references"
        ]
        assert any(r.get("reference_kind") == "implements" for r in iface_refs)


@pytest.mark.asyncio
async def test_high_fan_in_limit_test_targets_signatures_snippets_and_playbook_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests" / "__tests__").mkdir(parents=True)

    (repo / "src" / "doc.ts").write_text(
        "export function createAppDocument(name: string, strict: boolean): string {\n"
        "  return `${name}:${strict}`;\n"
        "}\n",
        encoding="utf-8",
    )
    for idx in range(22):
        (repo / "src" / f"caller_{idx:02d}.ts").write_text(
            'import { createAppDocument } from "./doc";\n'
            "\n"
            f"export function buildDoc{idx:02d}() {{\n"
            f'  return createAppDocument("doc_{idx:02d}", true);\n'
            "}\n",
            encoding="utf-8",
        )
    for test_rel in (
        "tests/app.browser.spec.mjs",
        "tests/doc.test.mjs",
        "tests/__tests__/helper.spec.ts",
        "tests/widget.test.tsx",
    ):
        import_target = (
            "../../src/doc" if "/__tests__/" in test_rel else "../src/doc"
        )
        (repo / test_rel).write_text(
            f'import {{ createAppDocument }} from "{import_target}";\n'
            "\n"
            "export function verifyDoc() {\n"
            '  return createAppDocument("spec", false);\n'
            "}\n",
            encoding="utf-8",
        )

    async with _running_indexed_daemon(tmp_path, repo, monkeypatch) as (_, client):
        bounded = await client.call(
            "knowledge.query",
            {
                "goal": "References to createAppDocument",
                "repo_path": str(repo),
                "symbol_name": "createAppDocument",
                "file_paths": ["src/doc.ts"],
                "playbook": "REFERENCES",
                "limit": 5,
            },
        )
        assert bounded["routing"]["provider"] == "explicit"
        assert bounded["timings"]["classification_ms"] == 0.0
        assert bounded["truncated"] is True
        ev = cast(dict[str, list[dict[str, object]]], bounded["evidence"])
        totals = cast(dict[str, int], bounded["total_counts"])
        assert len(ev["references"]) == 5
        assert len(ev["test_targets"]) == 4
        assert totals["references"] >= 26
        assert totals["test_targets"] == 4
        assert all("signature" not in s for s in ev["symbols"])
        assert all("snippet" not in r for r in ev["references"])

        hydrated = await client.call(
            "knowledge.query",
            {
                "goal": "References to createAppDocument with snippets",
                "repo_path": str(repo),
                "symbol_name": "createAppDocument",
                "file_paths": ["src/doc.ts"],
                "playbook": "REFERENCES",
                "limit": 5,
                "include_signatures": True,
                "include_snippets": True,
            },
        )
        hev = cast(dict[str, list[dict[str, object]]], hydrated["evidence"])
        assert any(
            s.get("name") == "createAppDocument"
            and s.get("signature") == "(name: string, strict: boolean): string"
            for s in hev["symbols"]
        )
        assert all(
            "createAppDocument(" in str(r.get("snippet", ""))
            for r in hev["references"]
            if r.get("reference_kind") == "calls"
        )

        impact = await client.call(
            "knowledge.query",
            {
                "goal": "Impact of createAppDocument change",
                "repo_path": str(repo),
                "changed_ranges": [
                    {"file_path": "src/doc.ts", "start_line": 0, "end_line": 2}
                ],
                "playbook": "IMPACT",
                "limit": 3,
            },
        )
        iev = cast(dict[str, list[dict[str, object]]], impact["evidence"])
        itotals = cast(dict[str, int], impact["total_counts"])
        assert len(iev["test_targets"]) == 3
        assert itotals["test_targets"] == 4
        assert impact["truncated"] is True

        discover = await client.call(
            "knowledge.query",
            {
                "goal": "Discover createAppDocument",
                "repo_path": str(repo),
                "symbol_name": "createAppDocument",
                "playbook": "DISCOVER",
                "limit": 5,
            },
        )
        investigation = cast(dict[str, dict[str, object]], discover["investigation"])
        assert investigation["tests"]["status"] == "not_assessed"


@pytest.mark.asyncio
async def test_barrels_cycles_http_contracts_and_rpc_bridges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / "components" / "nested").mkdir(parents=True)
    (repo / "cycle").mkdir(parents=True)
    (repo / "pages").mkdir(parents=True)
    (repo / "server").mkdir(parents=True)
    (repo / "api").mkdir(parents=True)
    (repo / "client").mkdir(parents=True)
    (repo / "bridge").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)

    (repo / "components" / "Button.tsx").write_text(
        "export function Button() {\n  return null;\n}\n",
        encoding="utf-8",
    )
    (repo / "components" / "nested" / "index.ts").write_text(
        'export { Button } from "../Button";\n',
        encoding="utf-8",
    )
    (repo / "components" / "index.ts").write_text(
        'export * from "./nested";\n',
        encoding="utf-8",
    )
    (repo / "pages" / "Home.tsx").write_text(
        'import { Button } from "../components";\n'
        "\n"
        "export function Home() {\n"
        "  return <Button />;\n"
        "}\n",
        encoding="utf-8",
    )

    # Re-export cycle must terminate without error
    (repo / "cycle" / "a.ts").write_text(
        'export * from "./b";\nexport const alpha = 1;\n',
        encoding="utf-8",
    )
    (repo / "cycle" / "b.ts").write_text(
        'export * from "./a";\nexport const beta = 2;\n',
        encoding="utf-8",
    )

    # Express route + fetch client
    (repo / "server" / "orders.ts").write_text(
        "export function createOrder(req: any, res: any) {\n  return res.json({});\n}\n"
        'router.post("/api/orders/:id", createOrder);\n',
        encoding="utf-8",
    )
    (repo / "client" / "ordersApi.ts").write_text(
        "export async function submitOrder(id: string) {\n"
        '  return fetch(`/api/orders/${id}`, { method: "POST" });\n'
        "}\n",
        encoding="utf-8",
    )

    # FastAPI route + fetch client
    (repo / "api" / "items.py").write_text(
        '@router.get("/api/items/{item_id}")\n'
        "def read_item(item_id: str):\n"
        '    return {"id": item_id}\n',
        encoding="utf-8",
    )
    (repo / "client" / "itemsApi.ts").write_text(
        "export async function loadItem(itemId: string) {\n"
        "  return fetch(`/api/items/${itemId}`);\n"
        "}\n",
        encoding="utf-8",
    )

    # window.__live global bridge
    (repo / "bridge" / "live.ts").write_text(
        "export function syncNow() {}\n"
        "(window as any).__live = { syncNow };\n",
        encoding="utf-8",
    )
    (repo / "bridge" / "caller.ts").write_text(
        "export function triggerSync() {\n  return window.__live.syncNow();\n}\n",
        encoding="utf-8",
    )

    # Vitest commands RPC bridge
    (repo / "tests" / "commands.cjs").write_text(
        "function readAdapterBundle() {\n  return {};\n}\n"
        "module.exports = { readAdapterBundle };\n",
        encoding="utf-8",
    )
    (repo / "tests" / "vitest.config.ts").write_text(
        'export default { browser: { commands: require("./commands.cjs") } };\n',
        encoding="utf-8",
    )
    (repo / "tests" / "widget.browser.spec.mjs").write_text(
        "export async function checkWidget() {\n"
        "  return commands.readAdapterBundle();\n"
        "}\n",
        encoding="utf-8",
    )

    async with _running_indexed_daemon(tmp_path, repo, monkeypatch) as (_, client):
        button_res = await client.call(
            "knowledge.query",
            {
                "goal": "References to Button",
                "repo_path": str(repo),
                "symbol_name": "Button",
                "file_paths": ["components/Button.tsx"],
                "playbook": "REFERENCES",
            },
        )
        button_refs = cast(dict[str, list[dict[str, object]]], button_res["evidence"])[
            "references"
        ]
        assert any(r.get("file_path") == "pages/Home.tsx" for r in button_refs)

        order_res = await client.call(
            "knowledge.query",
            {
                "goal": "References to createOrder",
                "repo_path": str(repo),
                "symbol_name": "createOrder",
                "file_paths": ["server/orders.ts"],
                "playbook": "REFERENCES",
            },
        )
        order_refs = cast(dict[str, list[dict[str, object]]], order_res["evidence"])[
            "references"
        ]
        assert any(r.get("file_path") == "client/ordersApi.ts" for r in order_refs)

        item_res = await client.call(
            "knowledge.query",
            {
                "goal": "References to read_item",
                "repo_path": str(repo),
                "symbol_name": "read_item",
                "file_paths": ["api/items.py"],
                "playbook": "REFERENCES",
            },
        )
        item_refs = cast(dict[str, list[dict[str, object]]], item_res["evidence"])[
            "references"
        ]
        assert any(r.get("file_path") == "client/itemsApi.ts" for r in item_refs)

        live_res = await client.call(
            "knowledge.query",
            {
                "goal": "References to syncNow",
                "repo_path": str(repo),
                "symbol_name": "syncNow",
                "file_paths": ["bridge/live.ts"],
                "playbook": "REFERENCES",
            },
        )
        live_refs = cast(dict[str, list[dict[str, object]]], live_res["evidence"])[
            "references"
        ]
        assert any(r.get("file_path") == "bridge/caller.ts" for r in live_refs)

        rpc_res = await client.call(
            "knowledge.query",
            {
                "goal": "References to readAdapterBundle",
                "repo_path": str(repo),
                "symbol_name": "readAdapterBundle",
                "file_paths": ["tests/commands.cjs"],
                "playbook": "REFERENCES",
            },
        )
        rpc_ev = cast(dict[str, list[dict[str, object]]], rpc_res["evidence"])
        assert any(
            r.get("file_path") == "tests/widget.browser.spec.mjs"
            for r in rpc_ev["references"]
        )
        assert any(
            t.get("file_path") == "tests/widget.browser.spec.mjs"
            for t in rpc_ev["test_targets"]
        )


@pytest.mark.asyncio
async def test_diff_impact_and_refresh_dirty_live_daemon_auto_freshness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ["init"],
        ["config", "user.name", "Test User"],
        ["config", "user.email", "test@example.com"],
    ):
        _git(repo, *args)

    (repo / "calc.ts").write_text(
        "export function baseRate(): number {\n"
        "  return 0.1;\n"
        "}\n"
        "\n"
        "export function computeTax(amount: number): number {\n"
        "  return amount * baseRate();\n"
        "}\n",
        encoding="utf-8",
    )
    (repo / "calc.test.ts").write_text(
        'import { computeTax } from "./calc";\n'
        "\n"
        "export function testTax() {\n"
        "  return computeTax(100);\n"
        "}\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")

    async with _running_indexed_daemon(tmp_path, repo, monkeypatch) as (_, client):
        # Edit files on disk WITHOUT calling repository.ingest_files
        (repo / "calc.ts").write_text(
            "export function baseRate(): number {\n"
            "  const factor = 2;\n"
            "  const adjusted = 0.1 * factor;\n"
            "  return adjusted;\n"
            "}\n"
            "\n"
            "export function computeTax(amount: number): number {\n"
            "  return amount * baseRate();\n"
            "}\n",
            encoding="utf-8",
        )
        (repo / "calc.test.ts").write_text(
            'import { computeTax } from "./calc";\n'
            "\n"
            "export function testTaxUpdated() {\n"
            "  return computeTax(250);\n"
            "}\n",
            encoding="utf-8",
        )

        res = await client.call(
            "knowledge.query",
            {
                "goal": "Assess current git diff impact",
                "repo_path": str(repo),
                "diff_impact": True,
                "refresh_dirty": True,
                "include_signatures": True,
                "include_snippets": True,
                "playbook": "IMPACT",
            },
        )
        trace_by_stage = {
            str(step["stage"]): step
            for step in cast(list[dict[str, object]], res["trace"])
        }
        assert trace_by_stage["git_diff"]["status"] == "ok"
        assert trace_by_stage["refresh"]["status"] == "ok"
        assert trace_by_stage["impact"]["status"] == "ok"

        ev = cast(dict[str, list[dict[str, object]]], res["evidence"])
        assert any(t.get("file_path") == "calc.test.ts" for t in ev["test_targets"])
        assert any(
            s.get("name") == "computeTax" and s.get("start_line") == 6
            for s in ev["symbols"]
        )
        assert any(
            "testTaxUpdated" in str(dep)
            for dep in ev["relationships"] + ev["symbols"] + ev["test_targets"]
        )
        refs_after = await client.call(
            "knowledge.query",
            {
                "goal": "References to computeTax after live refresh",
                "repo_path": str(repo),
                "symbol_name": "computeTax",
                "file_paths": ["calc.ts"],
                "playbook": "REFERENCES",
            },
        )
        ref_list = cast(dict[str, list[dict[str, object]]], refs_after["evidence"])[
            "references"
        ]
        assert any(
            r.get("enclosing_symbol") == "calc.test.testTaxUpdated" for r in ref_list
        )
