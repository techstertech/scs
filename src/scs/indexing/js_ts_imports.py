"""Resolve relative, extensionless, and configured alias JS/TS imports."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import posixpath
from typing import cast

JS_TS_LANGUAGES: frozenset[str] = frozenset(
    {"javascript", "typescript", "jsx", "tsx"}
)
JS_TS_SUFFIXES: tuple[str, ...] = (
    ".ts",
    ".tsx",
    ".mts",
    ".cts",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".mjsx",
    ".es6",
    ".jsm",
)
_CONFIG_FILENAMES: frozenset[str] = frozenset({"tsconfig.json", "jsconfig.json"})
_MAX_CONFIG_BYTES = 262_144
_MAX_EXTENDS_DEPTH = 4


@dataclass(frozen=True, slots=True)
class TsConfigAliases:
    """Path-mapping and baseUrl rules scoped to one tsconfig or jsconfig directory."""

    config_dir: str
    is_tsconfig: bool
    base_url: str | None
    exact_paths: dict[str, tuple[tuple[str, str], ...]]
    wildcard_paths: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...]


def _strip_jsonc(raw: str) -> str:
    """Strip // comments, /* */ comments, and trailing commas outside JSON strings."""

    without_comments: list[str] = []
    in_string = False
    escaped = False
    index = 0
    length = len(raw)

    while index < length:
        ch = raw[index]
        if in_string:
            without_comments.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            index += 1
            continue

        if ch == '"':
            in_string = True
            without_comments.append(ch)
            index += 1
            continue

        if ch == "/" and index + 1 < length:
            nxt = raw[index + 1]
            if nxt == "/":
                index += 2
                while index < length and raw[index] != "\n":
                    index += 1
                continue
            if nxt == "*":
                index += 2
                while index + 1 < length and not (
                    raw[index] == "*" and raw[index + 1] == "/"
                ):
                    if raw[index] == "\n":
                        without_comments.append("\n")
                    index += 1
                index = min(length, index + 2)
                continue

        without_comments.append(ch)
        index += 1

    cleaned: list[str] = []
    in_string = False
    escaped = False
    comment_free = "".join(without_comments)
    cf_len = len(comment_free)

    for pos, ch in enumerate(comment_free):
        if in_string:
            cleaned.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            cleaned.append(ch)
            continue

        if ch == ",":
            lookahead = pos + 1
            while lookahead < cf_len and comment_free[lookahead] in " \t\r\n":
                lookahead += 1
            if lookahead < cf_len and comment_free[lookahead] in "}]":
                continue

        cleaned.append(ch)

    return "".join(cleaned)


def _normalize_repo_path(base_dir: str, rel_target: str) -> str | None:
    """Normalize a repository-relative target, rejecting escapes outside root."""

    cleaned = rel_target.replace("\\", "/")
    if cleaned.startswith("/"):
        return None
    joined = posixpath.join(base_dir, cleaned) if base_dir else cleaned
    normalized = posixpath.normpath(joined)
    if normalized == ".." or normalized.startswith("../") or normalized.startswith("/"):
        return None
    return normalized


def _parse_jsonc_file(path: Path, repo_root: Path) -> Mapping[str, object] | None:
    try:
        resolved = path.resolve()
        if not resolved.is_relative_to(repo_root) or not resolved.is_file():
            return None
        if resolved.stat().st_size > _MAX_CONFIG_BYTES:
            return None
        raw = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    try:
        parsed = cast(object, json.loads(_strip_jsonc(raw)))
    except ValueError:
        return None
    return cast(Mapping[str, object], parsed) if isinstance(parsed, Mapping) else None


def _resolve_extends_path(
    config_file: Path, extends_spec: str, repo_root: Path
) -> Path | None:
    cleaned = extends_spec.replace("\\", "/").strip()
    if not cleaned or not (
        cleaned.startswith("./") or cleaned.startswith("../") or cleaned in {".", ".."}
    ):
        return None
    parent_dir = config_file.parent
    for suffix in ("", ".json"):
        candidate = (parent_dir / f"{cleaned}{suffix}").resolve()
        if candidate.is_relative_to(repo_root) and candidate.is_file():
            return candidate
    return None


def _read_layer_chain(
    config_file: Path,
    repo_root: Path,
    visited: set[Path],
    depth: int = 0,
) -> tuple[
    str | None,
    dict[str, tuple[tuple[str, str], ...]],
]:
    if depth > _MAX_EXTENDS_DEPTH:
        return None, {}
    resolved = config_file.resolve()
    if resolved in visited:
        return None, {}
    data = _parse_jsonc_file(resolved, repo_root)
    if data is None:
        return None, {}
    visited = visited | {resolved}

    try:
        rel_dir = resolved.parent.relative_to(repo_root).as_posix()
    except ValueError:
        return None, {}
    config_dir = "" if rel_dir == "." else rel_dir

    inherited_base: str | None = None
    inherited_paths: dict[str, tuple[tuple[str, str], ...]] = {}
    extends_value = data.get("extends")
    extends_list: list[str] = []
    if isinstance(extends_value, str):
        extends_list = [extends_value]
    elif isinstance(extends_value, list):
        extends_list = [
            item for item in cast(list[object], extends_value) if isinstance(item, str)
        ]

    for extends_spec in extends_list:
        parent_file = _resolve_extends_path(resolved, extends_spec, repo_root)
        if parent_file is None:
            continue
        p_base, p_paths = _read_layer_chain(
            parent_file, repo_root, visited, depth=depth + 1
        )
        if p_base is not None:
            inherited_base = p_base
        inherited_paths.update(p_paths)

    compiler_options = data.get("compilerOptions")
    if not isinstance(compiler_options, Mapping):
        return inherited_base, inherited_paths

    options = cast(Mapping[str, object], compiler_options)
    base_url = inherited_base
    raw_base_url = options.get("baseUrl")
    if isinstance(raw_base_url, str) and raw_base_url.strip():
        normalized_base = _normalize_repo_path(config_dir, raw_base_url.strip())
        if normalized_base is not None:
            base_url = "" if normalized_base == "." else normalized_base

    raw_paths = options.get("paths")
    if isinstance(raw_paths, Mapping):
        # Paths targets resolve relative to baseUrl when specified, else config dir.
        paths_base = base_url if base_url is not None else config_dir
        merged = dict(inherited_paths)
        for pattern, raw_targets in cast(Mapping[object, object], raw_paths).items():
            if not isinstance(pattern, str) or not isinstance(raw_targets, list):
                continue
            targets = tuple(
                (paths_base, item.strip())
                for item in cast(list[object], raw_targets)
                if isinstance(item, str) and item.strip()
            )
            if targets:
                merged[pattern] = targets
        inherited_paths = merged

    return base_url, inherited_paths


def load_js_ts_alias_configs(
    repo_root: Path, rel_paths: Iterable[str]
) -> tuple[TsConfigAliases, ...]:
    """Load tsconfig.json/jsconfig.json alias rules discovered within repo_root."""

    resolved_root = repo_root.resolve()
    candidates: set[Path] = {
        resolved_root / "tsconfig.json",
        resolved_root / "jsconfig.json",
    }
    for rel in rel_paths:
        name = posixpath.basename(rel).lower()
        if name in _CONFIG_FILENAMES:
            candidates.add(resolved_root / rel)

    configs: list[TsConfigAliases] = []
    seen_files: set[Path] = set()
    for candidate in sorted(candidates):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if (
            resolved in seen_files
            or not resolved.is_relative_to(resolved_root)
            or not resolved.is_file()
        ):
            continue
        seen_files.add(resolved)

        base_url, path_entries = _read_layer_chain(resolved, resolved_root, set())
        if base_url is None and not path_entries:
            continue

        rel_dir = resolved.parent.relative_to(resolved_root).as_posix()
        config_dir = "" if rel_dir == "." else rel_dir
        exact: dict[str, tuple[tuple[str, str], ...]] = {}
        wildcards: list[tuple[str, str, tuple[tuple[str, str], ...]]] = []
        for pattern, targets in path_entries.items():
            if "*" in pattern:
                prefix, suffix = pattern.split("*", 1)
                wildcards.append((prefix, suffix, targets))
            else:
                exact[pattern] = targets

        wildcards.sort(
            key=lambda item: (-(len(item[0]) + len(item[1])), -len(item[0]), item[0])
        )
        configs.append(
            TsConfigAliases(
                config_dir=config_dir,
                is_tsconfig=resolved.name.lower() == "tsconfig.json",
                base_url=base_url,
                exact_paths=exact,
                wildcard_paths=tuple(wildcards),
            )
        )

    configs.sort(
        key=lambda item: (
            -len(item.config_dir),
            0 if item.is_tsconfig else 1,
            item.config_dir,
        )
    )
    return tuple(configs)


def _matching_alias_configs(
    rel_path: str, alias_configs: Sequence[TsConfigAliases]
) -> list[TsConfigAliases]:
    return [
        cfg
        for cfg in alias_configs
        if cfg.config_dir == ""
        or rel_path == cfg.config_dir
        or rel_path.startswith(f"{cfg.config_dir}/")
    ]


def _candidate_targets_for_specifier(
    rel_path: str,
    specifier: str,
    alias_configs: Sequence[TsConfigAliases],
) -> list[str]:
    """Translate a module specifier into candidate repository-relative POSIX paths."""

    cleaned = specifier.split("?", 1)[0].split("#", 1)[0]
    if not cleaned:
        return []

    is_relative = (
        cleaned in {".", ".."}
        or cleaned.startswith("./")
        or cleaned.startswith("../")
    )
    if is_relative:
        normalized = _normalize_repo_path(posixpath.dirname(rel_path), cleaned)
        return [normalized] if normalized is not None else []

    if cleaned.startswith("/"):
        rel = cleaned.lstrip("/")
        if not rel:
            return []
        root_candidates: list[str] = []
        cur = posixpath.dirname(rel_path)
        while True:
            for base in (
                cur,
                f"{cur}/public" if cur else "public",
                f"{cur}/static" if cur else "static",
            ):
                normalized = _normalize_repo_path(base, rel)
                if normalized is not None:
                    root_candidates.append(normalized)
            if not cur or cur == ".":
                break
            parent = posixpath.dirname(cur)
            if parent == cur:
                break
            cur = parent
        return list(dict.fromkeys(root_candidates))

    candidates: list[str] = []
    for cfg in _matching_alias_configs(rel_path, alias_configs):
        exact = cfg.exact_paths.get(cleaned)
        if exact:
            for base_dir, target in exact:
                normalized = _normalize_repo_path(base_dir, target)
                if normalized is not None:
                    candidates.append(normalized)

        for prefix, suffix, targets in cfg.wildcard_paths:
            if (
                cleaned.startswith(prefix)
                and cleaned.endswith(suffix)
                and len(cleaned) >= len(prefix) + len(suffix)
            ):
                end = len(cleaned) - len(suffix) if suffix else None
                capture = cleaned[len(prefix) : end]
                for base_dir, target in targets:
                    substituted = target.replace("*", capture)
                    normalized = _normalize_repo_path(base_dir, substituted)
                    if normalized is not None:
                        candidates.append(normalized)

        if cfg.base_url is not None:
            normalized = _normalize_repo_path(cfg.base_url, cleaned)
            if normalized is not None:
                candidates.append(normalized)

    return list(dict.fromkeys(candidates))


def _resolve_normalized_path(
    normalized: str,
    resolve_qualified_name: Callable[[str], str | None],
    *,
    file_nodes_by_path: Mapping[str, str] | None = None,
    lower_file_nodes_by_path: Mapping[str, str] | None = None,
) -> str | None:
    lower_norm = normalized.lower()
    matched_suffix = next(
        (suffix for suffix in JS_TS_SUFFIXES if lower_norm.endswith(suffix)),
        None,
    )

    if file_nodes_by_path is not None:
        direct = file_nodes_by_path.get(normalized)
        if direct is not None:
            return direct
        if lower_file_nodes_by_path is not None:
            lower_direct = lower_file_nodes_by_path.get(lower_norm)
            if lower_direct is not None:
                return lower_direct

        path_candidates: list[str] = []
        if matched_suffix is not None:
            stem = normalized[: -len(matched_suffix)]
            path_candidates.extend(f"{stem}{ext}" for ext in JS_TS_SUFFIXES)
        elif normalized == ".":
            path_candidates.extend(f"index{ext}" for ext in JS_TS_SUFFIXES)
        else:
            path_candidates.extend(f"{normalized}{ext}" for ext in JS_TS_SUFFIXES)
            path_candidates.extend(f"{normalized}/index{ext}" for ext in JS_TS_SUFFIXES)

        for candidate in path_candidates:
            hit = file_nodes_by_path.get(candidate)
            if hit is not None:
                return hit
        if lower_file_nodes_by_path is not None:
            for candidate in path_candidates:
                hit = lower_file_nodes_by_path.get(candidate.lower())
                if hit is not None:
                    return hit

    qname_candidates: list[str]
    if matched_suffix is not None:
        stem = normalized[: -len(matched_suffix)]
        qname_candidates = [stem.replace("/", "."), normalized]
    elif normalized == ".":
        qname_candidates = ["index"]
    else:
        dotted = normalized.replace("/", ".")
        qname_candidates = [normalized, dotted, f"{dotted}.index"]

    for candidate in dict.fromkeys(qname_candidates):
        hit = resolve_qualified_name(candidate)
        if hit is not None:
            return hit
    return None


def resolve_js_ts_import_target(
    rel_path: str,
    specifier: str,
    resolve_qualified_name: Callable[[str], str | None],
    *,
    file_nodes_by_path: Mapping[str, str] | None = None,
    lower_file_nodes_by_path: Mapping[str, str] | None = None,
    alias_configs: Sequence[TsConfigAliases] = (),
) -> str | None:
    """Resolve a JS/TS import specifier to an indexed file node ID."""

    for normalized in _candidate_targets_for_specifier(rel_path, specifier, alias_configs):
        hit = _resolve_normalized_path(
            normalized,
            resolve_qualified_name,
            file_nodes_by_path=file_nodes_by_path,
            lower_file_nodes_by_path=lower_file_nodes_by_path,
        )
        if hit is not None:
            return hit
    return None
