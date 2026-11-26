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


@pytest.mark.asyncio
async def test_iife_legacy_js_root_raw_html_script_globals_and_no_oversized_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / "apps" / "web" / "public" / "assets").mkdir(parents=True)
    (repo / "apps" / "web" / "src").mkdir(parents=True)
    (repo / "apps" / "web" / "tests").mkdir(parents=True)

    (repo / "apps" / "web" / "public" / "assets" / "vendor-invoices-panel.js").write_text(
        "(function (global) {\n"
        "  function renderInvoiceRows(rows, currency) {\n"
        "    return rows.length + ':' + currency;\n"
        "  }\n"
        "  const formatTotal = (amount) => String(amount);\n"
        "  global.__psInvoices = { renderInvoiceRows, formatTotal };\n"
        "})(window);\n",
        encoding="utf-8",
    )

    # Classic >70 KB IIFE JS module (previously flagged 'oversized' in contract_gaps)
    padding = "// legacy padding line\n" * 3_100
    (repo / "apps" / "web" / "public" / "assets" / "asset-management-door.js").write_text(
        "(function () {\n"
        "  function initDoor(rootId) {\n"
        "    return rootId;\n"
        "  }\n"
        "  window.__psDoor = initDoor;\n"
        "  window.__psDoorApi = { initDoor };\n"
        f"{padding}"
        "})();\n",
        encoding="utf-8",
    )
    (repo / "apps" / "web" / "public" / "door.html").write_text(
        "<!doctype html>\n"
        "<html>\n"
        "  <body>\n"
        '    <script src="/assets/asset-management-door.js"></script>\n'
        "  </body>\n"
        "</html>\n",
        encoding="utf-8",
    )
    (repo / "apps" / "web" / "tests" / "door.browser.spec.mjs").write_text(
        'import doorSource from "/assets/asset-management-door.js?raw";\n'
        "\n"
        "export function testRawDoor() {\n"
        "  return doorSource.length;\n"
        "}\n",
        encoding="utf-8",
    )
    (repo / "apps" / "web" / "tests" / "door_url.test.mjs").write_text(
        "export function loadLegacyUrl() {\n"
        '  return new URL("../public/assets/asset-management-door.js", import.meta.url);\n'
        "}\n",
        encoding="utf-8",
    )
    (repo / "apps" / "web" / "src" / "DoorConsumer.tsx").write_text(
        "export function DoorConsumer() {\n"
        "  const fn = window.__psDoor;\n"
        "  return fn('main') + window.__psDoorApi.initDoor('sub');\n"
        "}\n",
        encoding="utf-8",
    )

    async with _running_indexed_daemon(tmp_path, repo, monkeypatch) as (_, client):
        inspect_res = await client.call(
            "knowledge.query",
            {
                "goal": "Inspect vendor-invoices-panel.js",
                "repo_path": str(repo),
                "file_paths": ["apps/web/public/assets/vendor-invoices-panel.js"],
                "playbook": "INSPECT_FILES",
                "include_signatures": True,
            },
        )
        sym_names = {
            str(s.get("name"))
            for s in cast(list[dict[str, object]], inspect_res["evidence"]["symbols"])
        }
        assert "renderInvoiceRows" in sym_names
        assert "formatTotal" in sym_names

        impact_res = await client.call(
            "knowledge.query",
            {
                "goal": "Impact of asset-management-door.js",
                "repo_path": str(repo),
                "file_paths": ["apps/web/public/assets/asset-management-door.js"],
                "playbook": "IMPACT",
                "limit": 20,
            },
        )
        ev = cast(dict[str, object], impact_res["evidence"])
        assert ev.get("contract_gaps", []) == []
        rel_files = {
            str(r.get("path", [{}])[-1].get("file_path"))
            for r in cast(list[dict[str, object]], ev["relationships"])
            if r.get("path")
        }
        dep_files = {
            str(f.get("file_path"))
            for f in cast(list[dict[str, object]], ev["files"])
        }
        all_dep_paths = rel_files | dep_files
        assert "apps/web/public/door.html" in all_dep_paths
        assert "apps/web/tests/door.browser.spec.mjs" in all_dep_paths
        assert "apps/web/tests/door_url.test.mjs" in all_dep_paths
        assert "apps/web/src/DoorConsumer.tsx" in all_dep_paths
        test_paths = {
            str(t.get("file_path"))
            for t in cast(list[dict[str, object]], ev["test_targets"])
        }
        assert "apps/web/tests/door.browser.spec.mjs" in test_paths
        assert "apps/web/tests/door_url.test.mjs" in test_paths


@pytest.mark.asyncio
async def test_transitive_impact_paths_and_test_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)

    (repo / "src" / "CaptureFields.tsx").write_text(
        "export function CaptureFields() {\n"
        "  return <div>capture</div>;\n"
        "}\n",
        encoding="utf-8",
    )
    (repo / "src" / "TaxCaptureFields.tsx").write_text(
        'import { CaptureFields } from "./CaptureFields";\n'
        "\n"
        "export function TaxCaptureFields() {\n"
        "  return <CaptureFields />;\n"
        "}\n",
        encoding="utf-8",
    )
    (repo / "tests" / "tax_capture_foundation.browser.spec.mjs").write_text(
        'import { TaxCaptureFields } from "../src/TaxCaptureFields";\n'
        "\n"
        "export function testTaxCapture() {\n"
        "  return TaxCaptureFields();\n"
        "}\n",
        encoding="utf-8",
    )

    async with _running_indexed_daemon(tmp_path, repo, monkeypatch) as (_, client):
        file_impact = await client.call(
            "knowledge.query",
            {
                "goal": "Impact of CaptureFields.tsx",
                "repo_path": str(repo),
                "file_paths": ["src/CaptureFields.tsx"],
                "playbook": "IMPACT",
                "limit": 10,
            },
        )
        fev = cast(dict[str, list[dict[str, object]]], file_impact["evidence"])
        ftotals = cast(dict[str, int], file_impact["total_counts"])
        assert fev["relationships"]
        assert ftotals["relationships"] >= len(fev["relationships"])
        assert any(
            t.get("file_path") == "tests/tax_capture_foundation.browser.spec.mjs"
            for t in fev["test_targets"]
        )

        range_impact = await client.call(
            "knowledge.query",
            {
                "goal": "Range impact of CaptureFields",
                "repo_path": str(repo),
                "changed_ranges": [
                    {"file_path": "src/CaptureFields.tsx", "start_line": 0, "end_line": 2}
                ],
                "playbook": "IMPACT",
                "limit": 10,
            },
        )
        rev = cast(dict[str, list[dict[str, object]]], range_impact["evidence"])
        assert any(
            t.get("file_path") == "tests/tax_capture_foundation.browser.spec.mjs"
            for t in rev["test_targets"]
        )


@pytest.mark.asyncio
async def test_legacy_parser_version_upgrades_automatically_via_ingest_and_refresh_dirty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scs.graph.models import NodeType
    from scs.graph.native import NativeGraph
    from scs.indexing.parser.native import NativeParser
    from scs.indexing.pipeline import IngestionPipeline
    from scs.providers.base import ProviderMetadata

    repo = tmp_path / "pipeline_repo"
    repo.mkdir()
    (repo / "lib.ts").write_text(
        "export function helperFn(x: number): number {\n  return x + 1;\n}\n",
        encoding="utf-8",
    )
    (repo / "consumer.ts").write_text(
        'import { helperFn } from "./lib";\n'
        "\n"
        "export function callHelper() {\n"
        "  return helperFn(42);\n"
        "}\n",
        encoding="utf-8",
    )

    graph = NativeGraph(
        database_path=tmp_path / "graph.db",
        vector_path=tmp_path / "vectors.usearch",
        provider_metadata_path=tmp_path / "provider.json",
        provider=ProviderMetadata("test", "disabled", 2, False, "offline"),
    )
    pipeline = IngestionPipeline(graph=graph, parser=NativeParser())
    pipeline.ingest(repo)

    # Replace FILE nodes without parser_version to simulate a pre-v0.7.1 index
    # (TSG upsert merges attributes in place unless the node is replaced).
    repo_id = graph.get_or_create_repo_sync(str(repo.resolve()))
    legacy_nodes = []
    legacy_ids = []
    for fnode in graph.list_nodes_sync(node_type=NodeType.FILE, repo_id=repo_id):
        meta = dict(fnode.metadata)
        meta.pop("parser_version", None)
        legacy_ids.append(fnode.id)
        legacy_nodes.append(
            {
                "id": fnode.id,
                "type": fnode.type.value,
                "name": fnode.name,
                "content": fnode.content,
                "metadata": meta,
                "repo_id": repo_id,
            }
        )
    graph.delete_nodes_sync(legacy_ids)
    graph.batch_upsert_nodes_sync(legacy_nodes)

    upgraded = pipeline.ingest(repo, force=False)
    assert upgraded.files_changed == 2
    noop = pipeline.ingest(repo, force=False)
    assert noop.files_changed == 0

    # Also verify refresh_dirty=True upgrades a stale FILE node in live daemon
    repo2 = tmp_path / "daemon_repo"
    repo2.mkdir()
    (repo2 / "mod.ts").write_text(
        "(function () {\n  function insideIife(v: number) { return v; }\n})();\n",
        encoding="utf-8",
    )
    async with _running_indexed_daemon(tmp_path / "d", repo2, monkeypatch) as (daemon, client):
        _rec, dgraph = daemon._require_stores().ensure_graph(repo2)
        drepo_id = dgraph.get_or_create_repo_sync(str(repo2.resolve()))
        for fnode in dgraph.list_nodes_sync(node_type=NodeType.FILE, repo_id=drepo_id):
            dgraph.batch_upsert_nodes_sync(
                [
                    {
                        "id": fnode.id,
                        "type": fnode.type.value,
                        "name": fnode.name,
                        "content": fnode.content,
                        "metadata": {**dict(fnode.metadata), "parser_version": 1},
                        "repo_id": drepo_id,
                    }
                ]
            )
        refreshed = await client.call(
            "knowledge.query",
            {
                "goal": "Inspect mod.ts after schema upgrade",
                "repo_path": str(repo2),
                "file_paths": ["mod.ts"],
                "playbook": "INSPECT_FILES",
                "refresh_dirty": True,
                "include_signatures": True,
            },
        )
        stages = {
            str(s["stage"]): s
            for s in cast(list[dict[str, object]], refreshed["trace"])
        }
        assert stages["refresh"]["status"] == "ok"
        assert stages["refresh"]["counts"]["refreshed"] == 1
        assert any(
            s.get("name") == "insideIife"
            for s in cast(list[dict[str, object]], refreshed["evidence"]["symbols"])
        )


@pytest.mark.asyncio
async def test_monorepo_python_and_ts_receiver_types_and_multi_rel_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    py_root = repo / "apps" / "cortex" / "cortex-engine"
    (py_root / "auth").mkdir(parents=True)
    (py_root / "tests").mkdir(parents=True)
    ts_root = repo / "apps" / "cortex" / "webview" / "src"
    (ts_root / "lib").mkdir(parents=True)
    (ts_root / "backend").mkdir(parents=True)

    (py_root / "auth" / "service.py").write_text(
        "class AdminAuthService:\n"
        "    def set_session_observer(self, observer: object) -> None:\n"
        "        self._observer = observer\n\n"
        "    async def list_active_sessions(self) -> list[dict[str, str]]:\n"
        "        return []\n",
        encoding="utf-8",
    )
    (py_root / "auth" / "__init__.py").write_text(
        "from .service import AdminAuthService\n",
        encoding="utf-8",
    )
    (py_root / "session_inventory.py").write_text(
        "from auth.service import AdminAuthService\n\n"
        "class SessionInventory:\n"
        "    def page(self, rows: list[dict[str, str]]) -> dict[str, object]:\n"
        "        return {'sessions': rows}\n\n"
        "def register_session_inventory_handlers(\n"
        "    router: object, auth: AdminAuthService, inventory: SessionInventory\n"
        ") -> None:\n"
        "    async def sessions_list() -> dict[str, object]:\n"
        "        logins = await auth.list_active_sessions()\n"
        "        return inventory.page(logins)\n",
        encoding="utf-8",
    )
    (py_root / "web_app.py").write_text(
        "from auth import AdminAuthService\n"
        "from session_inventory import SessionInventory\n\n"
        "def create_web_app(auth: AdminAuthService) -> None:\n"
        "    inventory = SessionInventory()\n"
        "    auth.set_session_observer(inventory)\n",
        encoding="utf-8",
    )
    (py_root / "tests" / "test_admin_auth.py").write_text(
        "from auth.service import AdminAuthService\n"
        "from session_inventory import SessionInventory\n\n"
        "def _auth_service() -> AdminAuthService:\n"
        "    return AdminAuthService()\n\n"
        "async def test_login_observation() -> None:\n"
        "    service = _auth_service()\n"
        "    inventory = SessionInventory()\n"
        "    service.set_session_observer(inventory)\n"
        "    inventory.page(await service.list_active_sessions())\n",
        encoding="utf-8",
    )

    # TypeScript multi-relationship graph (`calls` + `references` from same caller)
    # plus typed receiver method calls.
    (ts_root / "backend" / "typeferry-client.ts").write_text(
        "export class CortexApiError extends Error {\n"
        "  readonly code: string;\n"
        "  constructor(code: string) { super(code); this.code = code; }\n"
        "}\n\n"
        "export class BridgeCallClient {\n"
        "  async call(method: string): Promise<string> { return method; }\n"
        "}\n\n"
        "export function responseError(code: string): CortexApiError {\n"
        "  return new CortexApiError(code);\n"
        "}\n",
        encoding="utf-8",
    )
    (ts_root / "lib" / "sessions-wire.ts").write_text(
        "import { BridgeCallClient, CortexApiError } from '../backend/typeferry-client';\n\n"
        "async function loadOnce(client: BridgeCallClient): Promise<string> {\n"
        "  return client.call('sessions.list');\n"
        "}\n\n"
        "export async function loadSessions(client: BridgeCallClient): Promise<string> {\n"
        "  try {\n"
        "    return await loadOnce(client);\n"
        "  } catch (err) {\n"
        "    if (err instanceof CortexApiError) return loadOnce(client);\n"
        "    throw err;\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    (ts_root / "lib" / "sessions-wire.test.ts").write_text(
        "import { BridgeCallClient } from '../backend/typeferry-client';\n"
        "import { loadSessions } from './sessions-wire';\n\n"
        "export async function runTest() {\n"
        "  const client = new BridgeCallClient();\n"
        "  return loadSessions(client);\n"
        "}\n"
        "loadSessions(new BridgeCallClient());\n",
        encoding="utf-8",
    )

    async with _running_indexed_daemon(tmp_path / "d", repo, monkeypatch) as (_daemon, client):
        # 1. Python REFERENCES for list_active_sessions and set_session_observer
        list_refs = await client.call(
            "knowledge.query",
            {
                "goal": "References to list_active_sessions",
                "repo_path": str(repo),
                "symbol_name": "list_active_sessions",
                "playbook": "REFERENCES",
                "limit": 20,
            },
        )
        list_ref_files = {
            str(r.get("file_path"))
            for r in cast(list[dict[str, object]], list_refs["evidence"]["references"])
        }
        assert "apps/cortex/cortex-engine/session_inventory.py" in list_ref_files
        assert "apps/cortex/cortex-engine/tests/test_admin_auth.py" in list_ref_files
        assert any(
            t.get("file_path") == "apps/cortex/cortex-engine/tests/test_admin_auth.py"
            for t in cast(list[dict[str, object]], list_refs["evidence"]["test_targets"])
        )

        obs_refs = await client.call(
            "knowledge.query",
            {
                "goal": "References to set_session_observer",
                "repo_path": str(repo),
                "symbol_name": "set_session_observer",
                "playbook": "REFERENCES",
                "limit": 20,
            },
        )
        obs_ref_files = {
            str(r.get("file_path"))
            for r in cast(list[dict[str, object]], obs_refs["evidence"]["references"])
        }
        assert "apps/cortex/cortex-engine/web_app.py" in obs_ref_files
        assert "apps/cortex/cortex-engine/tests/test_admin_auth.py" in obs_ref_files

        # 2. TypeScript typed receiver call resolution (`client: BridgeCallClient` -> `client.call`)
        ts_call_refs = await client.call(
            "knowledge.query",
            {
                "goal": "References to BridgeCallClient.call",
                "repo_path": str(repo),
                "symbol_name": "call",
                "file_paths": ["apps/cortex/webview/src/backend/typeferry-client.ts"],
                "playbook": "REFERENCES",
                "limit": 20,
            },
        )
        ts_call_files = {
            str(r.get("file_path"))
            for r in cast(list[dict[str, object]], ts_call_refs["evidence"]["references"])
        }
        assert "apps/cortex/webview/src/lib/sessions-wire.ts" in ts_call_files

        # 3. UNDERSTAND and RELATIONSHIPS succeed even when callers have BOTH calls + references
        for sym in ("loadSessions", "loadOnce", "CortexApiError"):
            for playbook in ("UNDERSTAND", "RELATIONSHIPS"):
                res = await client.call(
                    "knowledge.query",
                    {
                        "goal": f"{playbook} {sym}",
                        "repo_path": str(repo),
                        "symbol_name": sym,
                        "playbook": playbook,
                        "limit": 20,
                    },
                )
                assert res["degraded_stages"] == [], f"{playbook}({sym}) degraded: {res['trace']}"
                assert res["evidence"]["relationships"], f"{playbook}({sym}) had no relationships"
