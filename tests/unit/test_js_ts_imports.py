"""Unit tests for JS/TS relative, extensionless, and tsconfig alias resolution."""

from __future__ import annotations

from pathlib import Path

from scs.indexing.js_ts_imports import (
    load_js_ts_alias_configs,
    resolve_js_ts_import_target,
)


def _no_qname(_qname: str) -> str | None:
    return None


def test_extensionless_and_index_imports_prefer_code_over_sibling_styles() -> None:
    file_nodes = {
        "src/components/Button.css": "file-button-css",
        "src/components/Button.tsx": "file-button-tsx",
        "src/components/index.tsx": "file-components-index",
        "src/utils/date.mts": "file-date-mts",
        "src/features/Body.tsx": "file-body-tsx",
    }
    lower_nodes = {k.lower(): v for k, v in file_nodes.items()}

    assert (
        resolve_js_ts_import_target(
            "src/features/Body.tsx",
            "../components/Button",
            _no_qname,
            file_nodes_by_path=file_nodes,
            lower_file_nodes_by_path=lower_nodes,
        )
        == "file-button-tsx"
    )
    assert (
        resolve_js_ts_import_target(
            "src/features/Body.tsx",
            "../components",
            _no_qname,
            file_nodes_by_path=file_nodes,
            lower_file_nodes_by_path=lower_nodes,
        )
        == "file-components-index"
    )
    assert (
        resolve_js_ts_import_target(
            "src/features/Body.tsx",
            "../utils/date.mjs?raw#hash",
            _no_qname,
            file_nodes_by_path=file_nodes,
            lower_file_nodes_by_path=lower_nodes,
        )
        == "file-date-mts"
    )
    assert (
        resolve_js_ts_import_target(
            "src/features/Body.tsx",
            "../components/Button.css",
            _no_qname,
            file_nodes_by_path=file_nodes,
            lower_file_nodes_by_path=lower_nodes,
        )
        == "file-button-css"
    )
    assert (
        resolve_js_ts_import_target(
            "src/features/Body.tsx",
            "../../../outside/secret",
            _no_qname,
            file_nodes_by_path=file_nodes,
            lower_file_nodes_by_path=lower_nodes,
        )
        is None
    )


def test_tsconfig_jsonc_extends_baseurl_and_paths_aliases(tmp_path: Path) -> None:
    (tmp_path / "tsconfig.base.json").write_text(
        """{
  // Base configuration with URL literal inside string
  "compilerOptions": {
    "baseUrl": "./src",
    "paths": {
      "@base/*": ["base/*"],
    },
  },
}
""",
        encoding="utf-8",
    )
    (tmp_path / "tsconfig.json").write_text(
        """{
  /* App-level override */
  "extends": "./tsconfig.base.json",
  "compilerOptions": {
    "paths": {
      "@/*": ["./*"],
      "@button": ["components/Button.tsx"],
    },
  },
}
""",
        encoding="utf-8",
    )
    (tmp_path / "apps" / "web").mkdir(parents=True)
    (tmp_path / "apps" / "web" / "jsconfig.json").write_text(
        """{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "~/*": ["./lib/*"]
    }
  }
}
""",
        encoding="utf-8",
    )

    file_nodes = {
        "tsconfig.base.json": "cfg-base",
        "tsconfig.json": "cfg-root",
        "apps/web/jsconfig.json": "cfg-web",
        "src/components/Button.tsx": "file-button-tsx",
        "src/base/theme.ts": "file-theme-ts",
        "apps/web/lib/helper.tsx": "file-web-helper",
    }
    configs = load_js_ts_alias_configs(tmp_path, file_nodes.keys())

    assert (
        resolve_js_ts_import_target(
            "src/pages/Home.tsx",
            "@/components/Button",
            _no_qname,
            file_nodes_by_path=file_nodes,
            alias_configs=configs,
        )
        == "file-button-tsx"
    )
    assert (
        resolve_js_ts_import_target(
            "src/pages/Home.tsx",
            "@button",
            _no_qname,
            file_nodes_by_path=file_nodes,
            alias_configs=configs,
        )
        == "file-button-tsx"
    )
    assert (
        resolve_js_ts_import_target(
            "src/pages/Home.tsx",
            "@base/theme",
            _no_qname,
            file_nodes_by_path=file_nodes,
            alias_configs=configs,
        )
        == "file-theme-ts"
    )
    assert (
        resolve_js_ts_import_target(
            "src/pages/Home.tsx",
            "components/Button",
            _no_qname,
            file_nodes_by_path=file_nodes,
            alias_configs=configs,
        )
        == "file-button-tsx"
    )
    assert (
        resolve_js_ts_import_target(
            "apps/web/src/Page.tsx",
            "~/helper",
            _no_qname,
            file_nodes_by_path=file_nodes,
            alias_configs=configs,
        )
        == "file-web-helper"
    )


def test_tsconfig_rejects_symlink_escaping_repository(tmp_path: Path) -> None:
    outside = tmp_path.parent / "escaped.json"
    outside.write_text('{"compilerOptions":{"paths":{"@/*":["*"]}}}', encoding="utf-8")
    try:
        (tmp_path / "tsconfig.json").symlink_to(outside)
        configs = load_js_ts_alias_configs(tmp_path, ("tsconfig.json",))
        assert configs == ()
    finally:
        outside.unlink(missing_ok=True)
