//! TypeScript/JavaScript AST extraction via tree-sitter.
//!
//! Extracts classes, functions, interfaces, type aliases, imports,
//! and constants from TypeScript, TSX, JavaScript, and JSX files.
//! Uses a scope stack for qualified name construction.

use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};
use tree_sitter::{Node, Parser};

use scs_core::node_types::NodeType;

use super::{
    build_call_edges, count_complexity, extract_call_sites, normalize_http_path, CallSite,
    LanguageParser, ParsedEdge, ParsedEntity, MAX_SITES_PER_EDGE,
};

/// File extensions handled by [`TypeScriptParser`].
pub const JS_TS_EXTENSIONS: &[&str] = &[
    ".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs", ".mjsx", ".es6", ".jsm",
];

/// Built-in names to exclude from call-graph edges (TypeScript/JavaScript).
const TS_BUILTIN_CALLS: &[&str] = &[
    "console",
    "Math",
    "JSON",
    "Object",
    "Array",
    "String",
    "Number",
    "Boolean",
    "Promise",
    "setTimeout",
    "setInterval",
    "clearTimeout",
    "clearInterval",
    "parseInt",
    "parseFloat",
    "require",
    "fetch",
    "alert",
    "confirm",
    "prompt",
    "Error",
    "TypeError",
    "RangeError",
    "SyntaxError",
    "ReferenceError",
    "Map",
    "Set",
    "WeakMap",
    "WeakSet",
    "Symbol",
    "Proxy",
    "Reflect",
    "RegExp",
    "Date",
    "Intl",
    "URL",
    "URLSearchParams",
    "TextEncoder",
    "TextDecoder",
    "atob",
    "btoa",
    "structuredClone",
    "queueMicrotask",
    "requestAnimationFrame",
    "cancelAnimationFrame",
    "addEventListener",
    "removeEventListener",
    "log",
    "warn",
    "error",
    "info",
    "debug",
    "trace",
    "assert",
];

/// Built-in TS/DOM/React utility types excluded from type_references edges.
const TS_BUILTIN_TYPES: &[&str] = &[
    "Array",
    "ReadonlyArray",
    "Promise",
    "Record",
    "Partial",
    "Required",
    "Readonly",
    "Pick",
    "Omit",
    "Exclude",
    "Extract",
    "NonNullable",
    "Parameters",
    "ConstructorParameters",
    "ReturnType",
    "InstanceType",
    "ThisParameterType",
    "OmitThisParameter",
    "Awaited",
    "Map",
    "Set",
    "WeakMap",
    "WeakSet",
    "Date",
    "Error",
    "RegExp",
    "Function",
    "Object",
    "String",
    "Number",
    "Boolean",
    "Symbol",
    "BigInt",
    "HTMLElement",
    "HTMLInputElement",
    "HTMLButtonElement",
    "HTMLDivElement",
    "HTMLFormElement",
    "Element",
    "Document",
    "Window",
    "Node",
    "Event",
    "MouseEvent",
    "KeyboardEvent",
    "ChangeEvent",
    "ReactNode",
    "ReactElement",
    "FC",
    "JSX",
];

/// Pattern to detect UPPER_CASE constants.
fn is_constant_name(name: &str) -> bool {
    !name.is_empty()
        && name.chars().next().unwrap().is_ascii_uppercase()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

fn get_text<'a>(node: &Node, source: &'a [u8]) -> &'a str {
    std::str::from_utf8(&source[node.start_byte()..node.end_byte()]).unwrap_or("")
}

fn find_child_by_kind<'a>(node: &Node<'a>, kind: &str) -> Option<Node<'a>> {
    let mut cursor = node.walk();
    let result = node.children(&mut cursor).find(|c| c.kind() == kind);
    result
}

/// Extract a JSDoc comment (`/** ... */`) preceding a declaration node.
///
/// Tree-sitter for TypeScript uses `comment` for all comment types.
/// Only `/** ... */` JSDoc-style comments are extracted — regular
/// `//` and `/* */` comments are ignored.
fn get_doc_comment(node: &Node, source: &[u8]) -> String {
    let mut sibling = node.prev_named_sibling();

    // Also check prev_sibling (unnamed) — comments may not be named nodes.
    if sibling.is_none() {
        let mut unnamed = node.prev_sibling();
        while let Some(s) = unnamed {
            if s.kind() == "comment" {
                sibling = Some(s);
                break;
            }
            // Skip decorators and other non-comment siblings.
            if !s.kind().contains("decorator") {
                break;
            }
            unnamed = s.prev_sibling();
        }
    }

    if let Some(ref s) = sibling {
        if s.kind() == "comment" {
            let text = get_text(s, source).trim().to_string();
            if text.starts_with("/**") && text.ends_with("*/") {
                let stripped = &text[3..text.len() - 2];
                let lines: Vec<&str> = stripped
                    .lines()
                    .map(|l| l.trim().trim_start_matches('*').trim())
                    .filter(|l| !l.is_empty())
                    .collect();
                return lines.join("\n");
            }
        }
    }

    String::new()
}

/// Extract a JSDoc comment (`/** ... */`) preceding a declaration node,
/// falling back to an enclosing `export_statement` or `expression_statement`.
fn get_doc_comment_for_declaration(node: &Node, source: &[u8]) -> String {
    let direct = get_doc_comment(node, source);
    if !direct.is_empty() {
        return direct;
    }
    if let Some(parent) = node.parent() {
        if matches!(parent.kind(), "export_statement" | "expression_statement") {
            return get_doc_comment(&parent, source);
        }
    }
    String::new()
}

fn extract_string_literal(node: &Node, source: &[u8]) -> Option<String> {
    if node.kind() != "string" {
        return None;
    }
    let raw = get_text(node, source).trim();
    let unwrapped = raw
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .or_else(|| raw.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
        .unwrap_or(raw);
    if unwrapped.is_empty() {
        None
    } else {
        Some(unwrapped.to_string())
    }
}

fn extract_module_import_call(node: &Node, source: &[u8]) -> Option<String> {
    if node.kind() != "call_expression" {
        return None;
    }
    let func = node.child_by_field_name("function")?;
    let is_require = func.kind() == "identifier" && get_text(&func, source) == "require";
    let is_dynamic_import = func.kind() == "import";
    if !is_require && !is_dynamic_import {
        return None;
    }
    let args = node.child_by_field_name("arguments")?;
    let first_arg = args.named_child(0)?;
    extract_string_literal(&first_arg, source)
}

fn extract_string_or_template_path(node: &Node, source: &[u8]) -> Option<String> {
    if let Some(raw) = extract_string_literal(node, source) {
        return Some(raw);
    }
    if node.kind() != "template_string" {
        return None;
    }
    let mut assembled = String::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "string_fragment" => assembled.push_str(get_text(&child, source)),
            "template_substitution" => assembled.push_str(":param"),
            _ => {}
        }
    }
    (!assembled.is_empty()).then_some(assembled)
}

#[derive(Debug, Clone)]
struct SiteOccurrence {
    target: String,
    qualifier: Option<String>,
    start_line: usize,
    end_line: usize,
}

fn emit_grouped_references(
    enclosing_qname: &str,
    reference_kind: &str,
    sites: Vec<SiteOccurrence>,
    edges: &mut Vec<ParsedEdge>,
) {
    let mut grouped: HashMap<String, Vec<SiteOccurrence>> = HashMap::new();
    for site in sites {
        if !site.target.is_empty() {
            grouped.entry(site.target.clone()).or_default().push(site);
        }
    }
    let mut targets: Vec<String> = grouped.keys().cloned().collect();
    targets.sort();
    for target in targets {
        let mut items = grouped.remove(&target).unwrap_or_default();
        items.sort_by(|a, b| {
            (a.start_line, a.end_line, a.qualifier.as_deref().unwrap_or(""))
                .cmp(&(b.start_line, b.end_line, b.qualifier.as_deref().unwrap_or("")))
        });
        items.dedup_by(|a, b| {
            a.start_line == b.start_line
                && a.end_line == b.end_line
                && a.qualifier == b.qualifier
        });
        let mut qualifiers: Vec<String> = Vec::new();
        for item in &items {
            if let Some(q) = &item.qualifier {
                if !qualifiers.contains(q) {
                    qualifiers.push(q.clone());
                }
            }
        }
        let mut meta = HashMap::new();
        meta.insert("reference_kind".to_string(), json!(reference_kind));
        meta.insert("enclosing_symbol".to_string(), json!(enclosing_qname));
        if let Some(first) = items.first() {
            meta.insert("start_line".to_string(), json!(first.start_line));
            meta.insert("end_line".to_string(), json!(first.end_line));
        }
        if let Some(first_q) = qualifiers.first() {
            meta.insert("qualifier".to_string(), json!(first_q));
        }
        let site_entries: Vec<Value> = items
            .into_iter()
            .take(MAX_SITES_PER_EDGE)
            .map(|item| {
                let mut entry = serde_json::Map::new();
                entry.insert("start_line".to_string(), json!(item.start_line));
                entry.insert("end_line".to_string(), json!(item.end_line));
                entry.insert("reference_kind".to_string(), json!(reference_kind));
                if let Some(q) = item.qualifier {
                    entry.insert("qualifier".to_string(), json!(q));
                }
                Value::Object(entry)
            })
            .collect();
        meta.insert("sites".to_string(), Value::Array(site_entries));
        edges.push(ParsedEdge::with_metadata(
            enclosing_qname.to_string(),
            target,
            "references",
            meta,
        ));
    }
}

fn dotted_member_name(node: &Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "identifier" | "property_identifier" | "type_identifier" => {
            let raw = get_text(node, source).trim();
            (!raw.is_empty()).then(|| raw.to_string())
        }
        "member_expression" | "nested_type_identifier" => {
            let obj = node
                .child_by_field_name("object")
                .or_else(|| node.named_child(0))?;
            let prop = node
                .child_by_field_name("property")
                .or_else(|| node.named_child(1))?;
            let prefix = dotted_member_name(&obj, source)?;
            let suffix = get_text(&prop, source).trim();
            (!suffix.is_empty()).then(|| format!("{prefix}.{suffix}"))
        }
        _ => None,
    }
}

fn collect_object_pattern_bindings(
    pattern: Node,
    source: &[u8],
) -> (Vec<Value>, Vec<Value>, bool) {
    let mut bindings = Vec::new();
    let mut imported_symbols = Vec::new();
    let mut has_wildcard = false;
    let mut cursor = pattern.walk();
    for child in pattern.named_children(&mut cursor) {
        match child.kind() {
            "shorthand_property_identifier_pattern" => {
                let name = get_text(&child, source).trim().to_string();
                if !name.is_empty() {
                    imported_symbols.push(json!(name));
                    bindings.push(json!({
                        "local": name,
                        "imported": name,
                        "kind": "named",
                    }));
                }
            }
            "pair_pattern" => {
                let imported = child
                    .child_by_field_name("key")
                    .map(|k| get_text(&k, source).trim().to_string())
                    .unwrap_or_default();
                let local = child
                    .child_by_field_name("value")
                    .and_then(|v| {
                        (v.kind() == "identifier").then(|| get_text(&v, source).trim().to_string())
                    })
                    .unwrap_or_else(|| imported.clone());
                if !imported.is_empty() && !local.is_empty() {
                    imported_symbols.push(json!(imported));
                    bindings.push(json!({
                        "local": local,
                        "imported": imported,
                        "kind": "named",
                    }));
                }
            }
            "object_assignment_pattern" => {
                if let Some(left) = child.child_by_field_name("left") {
                    if matches!(
                        left.kind(),
                        "shorthand_property_identifier_pattern" | "identifier"
                    ) {
                        let name = get_text(&left, source).trim().to_string();
                        if !name.is_empty() {
                            imported_symbols.push(json!(name));
                            bindings.push(json!({
                                "local": name,
                                "imported": name,
                                "kind": "named",
                            }));
                        }
                    }
                }
            }
            "rest_pattern" => {
                has_wildcard = true;
            }
            _ => {}
        }
    }
    (bindings, imported_symbols, has_wildcard)
}

fn extract_url_import_meta(node: &Node, source: &[u8]) -> Option<String> {
    if node.kind() != "new_expression" {
        return None;
    }
    let ctor = node.child_by_field_name("constructor")?;
    if get_text(&ctor, source).trim() != "URL" {
        return None;
    }
    let args = node.child_by_field_name("arguments")?;
    let first = args.named_child(0)?;
    let second = args.named_child(1)?;
    if !get_text(&second, source).contains("import.meta.url") {
        return None;
    }
    extract_string_literal(&first, source)
}

fn require_member_access(node: Node, source: &[u8]) -> Option<(String, String)> {
    if node.kind() != "member_expression" {
        return None;
    }
    let obj = node.child_by_field_name("object")?;
    let prop = node.child_by_field_name("property")?;
    let spec = extract_module_import_call(&obj, source)?;
    let prop_name = get_text(&prop, source).trim().to_string();
    (!prop_name.is_empty()).then_some((spec, prop_name))
}

const STANDARD_GLOBAL_PROPS: &[&str] = &[
    "location", "document", "console", "history", "navigator", "screen", "performance",
    "localStorage", "sessionStorage", "crypto", "fetch", "alert", "confirm", "prompt",
    "open", "close", "focus", "blur", "print", "scroll", "scrollTo", "scrollBy",
    "addEventListener", "removeEventListener", "dispatchEvent", "setTimeout", "clearTimeout",
    "setInterval", "clearInterval", "requestAnimationFrame", "cancelAnimationFrame",
    "queueMicrotask", "postMessage", "matchMedia", "getComputedStyle", "atob", "btoa",
    "structuredClone", "onload", "onerror", "onresize", "onscroll", "onbeforeunload",
    "onunload", "onhashchange", "onpopstate", "onmessage", "name", "status", "title",
    "innerWidth", "innerHeight", "outerWidth", "outerHeight", "scrollX", "scrollY",
    "pageXOffset", "pageYOffset", "devicePixelRatio", "parent", "top", "opener", "frames",
    "self", "window", "globalThis", "undefined", "NaN", "Infinity", "Object", "Array",
    "String", "Number", "Boolean", "Symbol", "BigInt", "Math", "Date", "RegExp", "Error",
    "Promise", "Map", "Set", "WeakMap", "WeakSet", "JSON", "Intl", "URL", "URLSearchParams",
    "Headers", "Request", "Response", "Event", "CustomEvent", "AbortController", "FormData",
    "Blob", "File", "FileReader", "WebSocket", "Worker", "MutationObserver",
    "ResizeObserver", "IntersectionObserver", "HTMLElement", "Node", "Element", "Document",
    "Window", "CSS", "DOMParser", "XMLHttpRequest", "Image", "Audio", "Option",
];

fn is_custom_global_prop(prop: &str) -> bool {
    !prop.is_empty() && !STANDARD_GLOBAL_PROPS.contains(&prop)
}

fn contains_global_word(raw: &str) -> bool {
    raw.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '$'))
        .any(|tok| matches!(tok, "window" | "globalThis" | "self"))
}

fn param_binding_name(param: Node, source: &[u8]) -> Option<String> {
    match param.kind() {
        "identifier" => {
            let s = get_text(&param, source).trim();
            (!s.is_empty()).then(|| s.to_string())
        }
        "required_parameter" | "optional_parameter" => param
            .child_by_field_name("pattern")
            .filter(|n| n.kind() == "identifier")
            .map(|n| get_text(&n, source).trim().to_string())
            .filter(|s| !s.is_empty()),
        _ => None,
    }
}

fn extract_top_level_iife<'a>(node: Node<'a>, source: &[u8]) -> Option<(Node<'a>, HashSet<String>)> {
    let mut target = unwrap_ts_expr(node);
    if target.kind() == "unary_expression" {
        let arg = target
            .child_by_field_name("argument")
            .or_else(|| target.named_child(0))?;
        target = unwrap_ts_expr(arg);
    }
    if target.kind() != "call_expression" {
        return None;
    }
    let callee = target.child_by_field_name("function")?;
    let fn_node = unwrap_ts_expr(callee);
    if !matches!(
        fn_node.kind(),
        "function_expression" | "generator_function" | "arrow_function"
    ) {
        return None;
    }
    let body = fn_node.child_by_field_name("body")?;
    if body.kind() != "statement_block" {
        return None;
    }

    let mut param_names: Vec<String> = Vec::new();
    if let Some(params) = fn_node.child_by_field_name("parameters") {
        let mut pc = params.walk();
        for p in params.named_children(&mut pc) {
            param_names.push(param_binding_name(p, source).unwrap_or_default());
        }
    } else if let Some(single) = fn_node.child_by_field_name("parameter") {
        param_names.push(param_binding_name(single, source).unwrap_or_default());
    }

    let mut aliases = HashSet::new();
    if let Some(args_node) = target.child_by_field_name("arguments") {
        let mut ac = args_node.walk();
        for (idx, arg) in args_node.named_children(&mut ac).enumerate() {
            if let Some(param_name) = param_names.get(idx).filter(|s| !s.is_empty()) {
                if contains_global_word(get_text(&arg, source)) {
                    aliases.insert(param_name.clone());
                }
            }
        }
    }

    Some((body, aliases))
}

fn normalize_global_namespace(raw: &str) -> Option<String> {
    let rest = raw
        .strip_prefix("window.")
        .or_else(|| raw.strip_prefix("globalThis."))
        .or_else(|| raw.strip_prefix("self."))?;
    let first_seg = rest.split('.').next().unwrap_or("");
    (is_custom_global_prop(first_seg) && !rest.is_empty()).then(|| format!("window.{rest}"))
}

fn unwrap_ts_expr(mut node: Node) -> Node {
    loop {
        let next = match node.kind() {
            "parenthesized_expression"
            | "as_expression"
            | "satisfies_expression"
            | "non_null_expression" => node
                .child_by_field_name("expression")
                .or_else(|| node.named_child(0)),
            "type_assertion" => node
                .child_by_field_name("expression")
                .or_else(|| node.named_child(1)),
            _ => None,
        };
        match next {
            Some(inner) => node = inner,
            None => return node,
        }
    }
}

fn is_global_root_node(node: Node, source: &[u8], global_aliases: &HashSet<String>) -> bool {
    let unwrapped = unwrap_ts_expr(node);
    let raw = get_text(&unwrapped, source).trim();
    matches!(raw, "window" | "globalThis" | "self") || global_aliases.contains(raw)
}

fn extract_global_namespace_node(
    node: Node,
    source: &[u8],
    global_aliases: &HashSet<String>,
) -> Option<String> {
    let unwrapped = unwrap_ts_expr(node);
    if unwrapped.kind() == "member_expression" {
        if let (Some(obj), Some(prop)) = (
            unwrapped.child_by_field_name("object"),
            unwrapped.child_by_field_name("property"),
        ) {
            if is_global_root_node(obj, source, global_aliases) {
                let prop_name = get_text(&prop, source).trim();
                if is_custom_global_prop(prop_name) {
                    return Some(format!("window.{prop_name}"));
                }
            }
        }
    }
    normalize_global_namespace(get_text(&unwrapped, source).trim())
}

fn extract_object_option_string(obj: Node, source: &[u8], target_key: &str) -> Option<String> {
    if obj.kind() != "object" {
        return None;
    }
    let mut cursor = obj.walk();
    for child in obj.named_children(&mut cursor) {
        if child.kind() == "pair" {
            let key = child
                .child_by_field_name("key")
                .map(|k| get_text(&k, source).trim().trim_matches(['"', '\'']).to_string());
            if key.as_deref() == Some(target_key) {
                if let Some(val) = child.child_by_field_name("value") {
                    return extract_string_or_template_path(&val, source);
                }
            }
        }
    }
    None
}

fn emit_bridge_caller_edge(
    enclosing_qname: &str,
    ns: &str,
    member: &str,
    node: &Node,
    edges: &mut Vec<ParsedEdge>,
) {
    let mut meta = HashMap::new();
    meta.insert("reference_kind".to_string(), json!("bridge_caller"));
    meta.insert("bridge_namespace".to_string(), json!(ns));
    meta.insert("bridge_member".to_string(), json!(member));
    meta.insert("start_line".to_string(), json!(node.start_position().row));
    meta.insert("end_line".to_string(), json!(node.end_position().row));
    meta.insert("enclosing_symbol".to_string(), json!(enclosing_qname));
    edges.push(ParsedEdge::with_metadata(
        enclosing_qname.to_string(),
        format!("__scs_bridge_caller__:{ns}:{member}"),
        "references",
        meta,
    ));
}

fn emit_global_object_providers(
    ns: &str,
    obj: Node,
    source: &[u8],
    enclosing_qname: &str,
    edges: &mut Vec<ParsedEdge>,
    filter_custom: bool,
) {
    let mut oc = obj.walk();
    for item in obj.named_children(&mut oc) {
        match item.kind() {
            "shorthand_property_identifier" => {
                let m = get_text(&item, source).trim();
                if !m.is_empty() && (!filter_custom || is_custom_global_prop(m)) {
                    let mut meta = HashMap::new();
                    meta.insert("reference_kind".to_string(), json!("bridge_provider"));
                    meta.insert("bridge_namespace".to_string(), json!(ns));
                    meta.insert("bridge_member".to_string(), json!(m));
                    meta.insert("handler_symbol".to_string(), json!(m));
                    meta.insert("start_line".to_string(), json!(item.start_position().row));
                    meta.insert("end_line".to_string(), json!(item.end_position().row));
                    meta.insert("enclosing_symbol".to_string(), json!(enclosing_qname));
                    edges.push(ParsedEdge::with_metadata(
                        enclosing_qname.to_string(),
                        format!("__scs_bridge_provider__:{ns}:{m}"),
                        "references",
                        meta,
                    ));
                }
            }
            "pair" => {
                if let Some(k) = item.child_by_field_name("key") {
                    let m = get_text(&k, source)
                        .trim()
                        .trim_matches(['"', '\''])
                        .trim();
                    if !m.is_empty() && (!filter_custom || is_custom_global_prop(m)) {
                        let mut meta = HashMap::new();
                        meta.insert("reference_kind".to_string(), json!("bridge_provider"));
                        meta.insert("bridge_namespace".to_string(), json!(ns));
                        meta.insert("bridge_member".to_string(), json!(m));
                        if let Some(v) = item.child_by_field_name("value") {
                            if v.kind() == "identifier" {
                                meta.insert(
                                    "handler_symbol".to_string(),
                                    json!(get_text(&v, source).trim()),
                                );
                            }
                        }
                        meta.insert("start_line".to_string(), json!(item.start_position().row));
                        meta.insert("end_line".to_string(), json!(item.end_position().row));
                        meta.insert("enclosing_symbol".to_string(), json!(enclosing_qname));
                        edges.push(ParsedEdge::with_metadata(
                            enclosing_qname.to_string(),
                            format!("__scs_bridge_provider__:{ns}:{m}"),
                            "references",
                            meta,
                        ));
                    }
                }
            }
            _ => {}
        }
    }
}

fn extract_ts_http_and_bridges(
    node: &Node,
    source: &[u8],
    enclosing_qname: &str,
    edges: &mut Vec<ParsedEdge>,
    global_aliases: &HashSet<String>,
) {
    match node.kind() {
        "call_expression" => {
            if let Some(func) = node.child_by_field_name("function") {
                if let Some(args_node) = node.child_by_field_name("arguments") {
                    let mut ac = args_node.walk();
                    let args: Vec<Node> = args_node.named_children(&mut ac).collect();

                    // 1) fetch(path, opts?)
                    if func.kind() == "identifier" && get_text(&func, source) == "fetch" {
                        if let Some(first) = args.first() {
                            if let Some(raw_path) = extract_string_or_template_path(first, source) {
                                if let Some(norm_path) = normalize_http_path(&raw_path) {
                                    let method = args
                                        .get(1)
                                        .map(|opts| {
                                            if opts.kind() == "object" {
                                                extract_object_option_string(*opts, source, "method")
                                                    .map(|m| m.to_ascii_uppercase())
                                                    .unwrap_or_else(|| "GET".to_string())
                                            } else {
                                                "ANY".to_string()
                                            }
                                        })
                                        .unwrap_or_else(|| "GET".to_string());
                                    let mut meta = HashMap::new();
                                    meta.insert(
                                        "reference_kind".to_string(),
                                        json!("http_route_client"),
                                    );
                                    meta.insert("http_method".to_string(), json!(method));
                                    meta.insert("http_path".to_string(), json!(raw_path));
                                    meta.insert("norm_path".to_string(), json!(norm_path));
                                    meta.insert(
                                        "start_line".to_string(),
                                        json!(node.start_position().row),
                                    );
                                    meta.insert(
                                        "end_line".to_string(),
                                        json!(node.end_position().row),
                                    );
                                    meta.insert(
                                        "enclosing_symbol".to_string(),
                                        json!(enclosing_qname),
                                    );
                                    edges.push(ParsedEdge::with_metadata(
                                        enclosing_qname.to_string(),
                                        format!("__scs_http_client__:{method}:{norm_path}"),
                                        "references",
                                        meta,
                                    ));
                                }
                            }
                        }
                    }

                    // 2) axios({ url: "/api/...", method: "..." }) or route({ method, url, handler })
                    if func.kind() == "identifier"
                        && get_text(&func, source) == "axios"
                        && args.first().is_some_and(|a| a.kind() == "object")
                    {
                        let opts = args[0];
                        if let Some(raw_path) = extract_object_option_string(opts, source, "url") {
                            if let Some(norm_path) = normalize_http_path(&raw_path) {
                                let method = extract_object_option_string(opts, source, "method")
                                    .map(|m| m.to_ascii_uppercase())
                                    .unwrap_or_else(|| "GET".to_string());
                                let mut meta = HashMap::new();
                                meta.insert(
                                    "reference_kind".to_string(),
                                    json!("http_route_client"),
                                );
                                meta.insert("http_method".to_string(), json!(method));
                                meta.insert("http_path".to_string(), json!(raw_path));
                                meta.insert("norm_path".to_string(), json!(norm_path));
                                meta.insert(
                                    "start_line".to_string(),
                                    json!(node.start_position().row),
                                );
                                meta.insert(
                                    "end_line".to_string(),
                                    json!(node.end_position().row),
                                );
                                meta.insert(
                                    "enclosing_symbol".to_string(),
                                    json!(enclosing_qname),
                                );
                                edges.push(ParsedEdge::with_metadata(
                                    enclosing_qname.to_string(),
                                    format!("__scs_http_client__:{method}:{norm_path}"),
                                    "references",
                                    meta,
                                ));
                            }
                        }
                    }

                    // 3) <obj>.<method>(...) -> HTTP handler / client OR global/commands bridge caller
                    if func.kind() == "member_expression" {
                        if let (Some(obj), Some(prop)) = (
                            func.child_by_field_name("object"),
                            func.child_by_field_name("property"),
                        ) {
                            let prop_name = get_text(&prop, source).trim();
                            let prop_lower = prop_name.to_ascii_lowercase();
                            let obj_raw = get_text(&obj, source).trim();
                            let obj_lower = obj_raw.to_ascii_lowercase();

                            // 3a) Object.assign(window, { ... })
                            if get_text(&func, source).trim() == "Object.assign"
                                && args.len() >= 2
                                && is_global_root_node(args[0], source, global_aliases)
                            {
                                for arg in &args[1..] {
                                    if arg.kind() == "object" {
                                        emit_global_object_providers(
                                            "window",
                                            *arg,
                                            source,
                                            enclosing_qname,
                                            edges,
                                            true,
                                        );
                                    }
                                }
                            }

                            // 3b) window.<ns>.<member>(...) / window.<member>(...) / globalThis / aliases
                            if let Some(ns) = extract_global_namespace_node(obj, source, global_aliases) {
                                if !prop_name.is_empty() {
                                    emit_bridge_caller_edge(
                                        enclosing_qname,
                                        &ns,
                                        prop_name,
                                        node,
                                        edges,
                                    );
                                }
                            } else if is_global_root_node(obj, source, global_aliases)
                                && is_custom_global_prop(prop_name)
                            {
                                emit_bridge_caller_edge(
                                    enclosing_qname,
                                    "window",
                                    prop_name,
                                    node,
                                    edges,
                                );
                            } else if obj_raw == "commands" && !prop_name.is_empty() {
                                // 3b) commands.<member>(...)
                                let mut meta = HashMap::new();
                                meta.insert("reference_kind".to_string(), json!("bridge_caller"));
                                meta.insert("bridge_namespace".to_string(), json!("commands"));
                                meta.insert("bridge_member".to_string(), json!(prop_name));
                                meta.insert(
                                    "start_line".to_string(),
                                    json!(node.start_position().row),
                                );
                                meta.insert(
                                    "end_line".to_string(),
                                    json!(node.end_position().row),
                                );
                                meta.insert(
                                    "enclosing_symbol".to_string(),
                                    json!(enclosing_qname),
                                );
                                edges.push(ParsedEdge::with_metadata(
                                    enclosing_qname.to_string(),
                                    format!("__scs_bridge_caller__:commands:{prop_name}"),
                                    "references",
                                    meta,
                                ));
                            }

                            // 3c) HTTP verbs on router/app or axios/api/http/client
                            if matches!(
                                prop_lower.as_str(),
                                "get" | "post" | "put" | "patch" | "delete" | "head" | "options" | "all"
                            ) {
                                if let Some(first) = args.first() {
                                    if let Some(raw_path) =
                                        extract_string_or_template_path(first, source)
                                    {
                                        if let Some(norm_path) = normalize_http_path(&raw_path) {
                                            let has_fn_arg = args.iter().skip(1).any(|a| {
                                                matches!(
                                                    a.kind(),
                                                    "arrow_function"
                                                        | "function_expression"
                                                        | "generator_function"
                                                )
                                            });
                                            let is_client_obj = obj_lower == "axios"
                                                || obj_lower == "http"
                                                || obj_lower == "api"
                                                || obj_lower.ends_with("client")
                                                || obj_lower.ends_with("api")
                                                || obj_lower.ends_with("fetcher")
                                                || (obj.kind() == "call_expression"
                                                    && (obj_lower.contains("request")
                                                        || obj_lower.contains("supertest")
                                                        || obj_lower.contains("fetch")));
                                            let last_non_fn_handler = args.iter().skip(1).rev().find_map(
                                                |a| {
                                                    matches!(
                                                        a.kind(),
                                                        "identifier" | "member_expression"
                                                    )
                                                    .then(|| get_text(a, source).trim().to_string())
                                                },
                                            );
                                            let is_router_obj = obj_lower == "router"
                                                || obj_lower == "app"
                                                || obj_lower == "server"
                                                || obj_lower.ends_with("router")
                                                || obj_lower.ends_with("routes")
                                                || obj_lower.ends_with("app")
                                                || obj_lower.ends_with("server");
                                            let looks_like_handler = has_fn_arg
                                                || (is_router_obj && !is_client_obj && args.len() >= 2);

                                            let http_method = prop_lower.to_ascii_uppercase();
                                            if looks_like_handler {
                                                let mut meta = HashMap::new();
                                                meta.insert(
                                                    "reference_kind".to_string(),
                                                    json!("http_route_handler"),
                                                );
                                                meta.insert(
                                                    "http_method".to_string(),
                                                    json!(http_method),
                                                );
                                                meta.insert(
                                                    "http_path".to_string(),
                                                    json!(raw_path),
                                                );
                                                meta.insert(
                                                    "norm_path".to_string(),
                                                    json!(norm_path),
                                                );
                                                meta.insert(
                                                    "start_line".to_string(),
                                                    json!(node.start_position().row),
                                                );
                                                meta.insert(
                                                    "end_line".to_string(),
                                                    json!(node.end_position().row),
                                                );
                                                meta.insert(
                                                    "enclosing_symbol".to_string(),
                                                    json!(enclosing_qname),
                                                );
                                                if let Some(h) = last_non_fn_handler {
                                                    meta.insert(
                                                        "handler_symbol".to_string(),
                                                        json!(h),
                                                    );
                                                }
                                                edges.push(ParsedEdge::with_metadata(
                                                    enclosing_qname.to_string(),
                                                    format!(
                                                        "__scs_http_handler__:{http_method}:{norm_path}"
                                                    ),
                                                    "references",
                                                    meta,
                                                ));
                                            } else if is_client_obj
                                                || (args.len() == 1 && !is_router_obj)
                                            {
                                                let mut meta = HashMap::new();
                                                meta.insert(
                                                    "reference_kind".to_string(),
                                                    json!("http_route_client"),
                                                );
                                                meta.insert(
                                                    "http_method".to_string(),
                                                    json!(http_method),
                                                );
                                                meta.insert(
                                                    "http_path".to_string(),
                                                    json!(raw_path),
                                                );
                                                meta.insert(
                                                    "norm_path".to_string(),
                                                    json!(norm_path),
                                                );
                                                meta.insert(
                                                    "start_line".to_string(),
                                                    json!(node.start_position().row),
                                                );
                                                meta.insert(
                                                    "end_line".to_string(),
                                                    json!(node.end_position().row),
                                                );
                                                meta.insert(
                                                    "enclosing_symbol".to_string(),
                                                    json!(enclosing_qname),
                                                );
                                                edges.push(ParsedEdge::with_metadata(
                                                    enclosing_qname.to_string(),
                                                    format!(
                                                        "__scs_http_client__:{http_method}:{norm_path}"
                                                    ),
                                                    "references",
                                                    meta,
                                                ));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        "assignment_expression" => {
            if let (Some(left), Some(right)) = (
                node.child_by_field_name("left"),
                node.child_by_field_name("right"),
            ) {
                if left.kind() == "member_expression" {
                    if let (Some(obj), Some(prop)) = (
                        left.child_by_field_name("object"),
                        left.child_by_field_name("property"),
                    ) {
                        let prop_name = get_text(&prop, source).trim();
                        if let Some(ns) = extract_global_namespace_node(obj, source, global_aliases) {
                            if !prop_name.is_empty() {
                                let mut meta = HashMap::new();
                                meta.insert("reference_kind".to_string(), json!("bridge_provider"));
                                meta.insert("bridge_namespace".to_string(), json!(ns));
                                meta.insert("bridge_member".to_string(), json!(prop_name));
                                meta.insert(
                                    "start_line".to_string(),
                                    json!(node.start_position().row),
                                );
                                meta.insert(
                                    "end_line".to_string(),
                                    json!(node.end_position().row),
                                );
                                meta.insert(
                                    "enclosing_symbol".to_string(),
                                    json!(enclosing_qname),
                                );
                                if right.kind() == "identifier" {
                                    meta.insert(
                                        "handler_symbol".to_string(),
                                        json!(get_text(&right, source).trim()),
                                    );
                                }
                                edges.push(ParsedEdge::with_metadata(
                                    enclosing_qname.to_string(),
                                    format!("__scs_bridge_provider__:{ns}:{prop_name}"),
                                    "references",
                                    meta,
                                ));
                            }
                        } else if is_global_root_node(obj, source, global_aliases)
                            && is_custom_global_prop(prop_name)
                        {
                            let mut meta = HashMap::new();
                            meta.insert("reference_kind".to_string(), json!("bridge_provider"));
                            meta.insert("bridge_namespace".to_string(), json!("window"));
                            meta.insert("bridge_member".to_string(), json!(prop_name));
                            meta.insert(
                                "start_line".to_string(),
                                json!(node.start_position().row),
                            );
                            meta.insert(
                                "end_line".to_string(),
                                json!(node.end_position().row),
                            );
                            meta.insert(
                                "enclosing_symbol".to_string(),
                                json!(enclosing_qname),
                            );
                            if right.kind() == "identifier" {
                                meta.insert(
                                    "handler_symbol".to_string(),
                                    json!(get_text(&right, source).trim()),
                                );
                            } else if matches!(
                                right.kind(),
                                "arrow_function" | "function_expression" | "generator_function"
                            ) {
                                meta.insert("handler_symbol".to_string(), json!(prop_name));
                            }
                            edges.push(ParsedEdge::with_metadata(
                                enclosing_qname.to_string(),
                                format!("__scs_bridge_provider__:window:{prop_name}"),
                                "references",
                                meta,
                            ));
                            if right.kind() == "object" {
                                let ns = format!("window.{prop_name}");
                                emit_global_object_providers(
                                    &ns,
                                    right,
                                    source,
                                    enclosing_qname,
                                    edges,
                                    false,
                                );
                            }
                        }
                    }
                }
            }
        }
        "pair" => {
            // Generic RPC / test command map: `commands: { a, b }` or `commands: require("./mod")` or `commands: sym`
            if let (Some(key), Some(val)) = (
                node.child_by_field_name("key"),
                node.child_by_field_name("value"),
            ) {
                if get_text(&key, source).trim() == "commands" {
                    if val.kind() == "object" {
                        let mut oc = val.walk();
                        for item in val.named_children(&mut oc) {
                            match item.kind() {
                                "shorthand_property_identifier" => {
                                    let m = get_text(&item, source).trim();
                                    if !m.is_empty() {
                                        let mut meta = HashMap::new();
                                        meta.insert(
                                            "reference_kind".to_string(),
                                            json!("bridge_provider"),
                                        );
                                        meta.insert(
                                            "bridge_namespace".to_string(),
                                            json!("commands"),
                                        );
                                        meta.insert("bridge_member".to_string(), json!(m));
                                        meta.insert("handler_symbol".to_string(), json!(m));
                                        meta.insert(
                                            "start_line".to_string(),
                                            json!(item.start_position().row),
                                        );
                                        meta.insert(
                                            "end_line".to_string(),
                                            json!(item.end_position().row),
                                        );
                                        meta.insert(
                                            "enclosing_symbol".to_string(),
                                            json!(enclosing_qname),
                                        );
                                        edges.push(ParsedEdge::with_metadata(
                                            enclosing_qname.to_string(),
                                            format!("__scs_bridge_provider__:commands:{m}"),
                                            "references",
                                            meta,
                                        ));
                                    }
                                }
                                "pair" => {
                                    if let Some(k) = item.child_by_field_name("key") {
                                        let m = get_text(&k, source).trim();
                                        if !m.is_empty() {
                                            let mut meta = HashMap::new();
                                            meta.insert(
                                                "reference_kind".to_string(),
                                                json!("bridge_provider"),
                                            );
                                            meta.insert(
                                                "bridge_namespace".to_string(),
                                                json!("commands"),
                                            );
                                            meta.insert("bridge_member".to_string(), json!(m));
                                            if let Some(v) = item.child_by_field_name("value") {
                                                if v.kind() == "identifier" {
                                                    meta.insert(
                                                        "handler_symbol".to_string(),
                                                        json!(get_text(&v, source).trim()),
                                                    );
                                                }
                                            }
                                            meta.insert(
                                                "start_line".to_string(),
                                                json!(item.start_position().row),
                                            );
                                            meta.insert(
                                                "end_line".to_string(),
                                                json!(item.end_position().row),
                                            );
                                            meta.insert(
                                                "enclosing_symbol".to_string(),
                                                json!(enclosing_qname),
                                            );
                                            edges.push(ParsedEdge::with_metadata(
                                                enclosing_qname.to_string(),
                                                format!("__scs_bridge_provider__:commands:{m}"),
                                                "references",
                                                meta,
                                            ));
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    } else if let Some(spec) = extract_module_import_call(&val, source) {
                        let mut meta = HashMap::new();
                        meta.insert("reference_kind".to_string(), json!("bridge_provider"));
                        meta.insert("bridge_namespace".to_string(), json!("commands"));
                        meta.insert("bridge_member".to_string(), json!("*"));
                        meta.insert("target_specifier".to_string(), json!(spec));
                        meta.insert("start_line".to_string(), json!(node.start_position().row));
                        meta.insert("end_line".to_string(), json!(node.end_position().row));
                        meta.insert("enclosing_symbol".to_string(), json!(enclosing_qname));
                        edges.push(ParsedEdge::with_metadata(
                            enclosing_qname.to_string(),
                            "__scs_bridge_provider__:commands:*".to_string(),
                            "references",
                            meta,
                        ));
                    } else if val.kind() == "identifier" {
                        let local = get_text(&val, source).trim();
                        if !local.is_empty() {
                            let mut meta = HashMap::new();
                            meta.insert("reference_kind".to_string(), json!("bridge_provider"));
                            meta.insert("bridge_namespace".to_string(), json!("commands"));
                            meta.insert("bridge_member".to_string(), json!("*"));
                            meta.insert("target_local".to_string(), json!(local));
                            meta.insert("start_line".to_string(), json!(node.start_position().row));
                            meta.insert("end_line".to_string(), json!(node.end_position().row));
                            meta.insert("enclosing_symbol".to_string(), json!(enclosing_qname));
                            edges.push(ParsedEdge::with_metadata(
                                enclosing_qname.to_string(),
                                "__scs_bridge_provider__:commands:*".to_string(),
                                "references",
                                meta,
                            ));
                        }
                    }
                }
            }
        }
        "member_expression" => {
            let parent = node.parent();
            let is_assignment_lhs = parent.is_some_and(|p| {
                p.kind() == "assignment_expression"
                    && p.child_by_field_name("left") == Some(*node)
            });
            let is_call_callee = parent.is_some_and(|p| {
                p.kind() == "call_expression"
                    && p.child_by_field_name("function")
                        .is_some_and(|f| unwrap_ts_expr(f) == *node)
            });
            let is_outer_member_obj = parent.is_some_and(|p| {
                p.kind() == "member_expression"
                    && p.child_by_field_name("object")
                        .is_some_and(|o| unwrap_ts_expr(o) == *node)
            });
            if !is_assignment_lhs && !is_call_callee && !is_outer_member_obj {
                if let (Some(obj), Some(prop)) = (
                    node.child_by_field_name("object"),
                    node.child_by_field_name("property"),
                ) {
                    let prop_name = get_text(&prop, source).trim();
                    if let Some(ns) = extract_global_namespace_node(obj, source, global_aliases) {
                        if !prop_name.is_empty() {
                            emit_bridge_caller_edge(enclosing_qname, &ns, prop_name, node, edges);
                        }
                    } else if is_global_root_node(obj, source, global_aliases)
                        && is_custom_global_prop(prop_name)
                    {
                        emit_bridge_caller_edge(
                            enclosing_qname,
                            "window",
                            prop_name,
                            node,
                            edges,
                        );
                    }
                }
            }
        }
        _ => {}
    }
}

fn collect_type_references_in_type_node(
    type_root: Node,
    source: &[u8],
    excluded_decl_name: Option<&str>,
    out: &mut Vec<SiteOccurrence>,
) {
    let builtins: HashSet<&str> = TS_BUILTIN_TYPES.iter().copied().collect();
    let mut cursor = type_root.walk();
    let mut reached_root = false;
    loop {
        let n = cursor.node();
        if n.kind() == "type_identifier" {
            let parent_kind = n.parent().map(|p| p.kind()).unwrap_or("");
            if !matches!(
                parent_kind,
                "type_parameter"
                    | "type_alias_declaration"
                    | "interface_declaration"
                    | "class_declaration"
                    | "extends_clause"
                    | "implements_clause"
                    | "extends_type_clause"
            ) {
                let (target, qualifier) = if parent_kind == "nested_type_identifier" {
                    let parent = n.parent().unwrap();
                    let full = dotted_member_name(&parent, source).unwrap_or_default();
                    if let Some((q, tail)) = full.rsplit_once('.') {
                        (tail.to_string(), Some(q.to_string()))
                    } else {
                        (get_text(&n, source).trim().to_string(), None)
                    }
                } else {
                    (get_text(&n, source).trim().to_string(), None)
                };
                if target.len() > 1
                    && !builtins.contains(target.as_str())
                    && Some(target.as_str()) != excluded_decl_name
                {
                    out.push(SiteOccurrence {
                        target,
                        qualifier,
                        start_line: n.start_position().row,
                        end_line: n.end_position().row,
                    });
                }
            }
        }
        if cursor.goto_first_child() {
            continue;
        }
        if cursor.goto_next_sibling() {
            continue;
        }
        loop {
            if !cursor.goto_parent() || cursor.node() == type_root {
                reached_root = true;
                break;
            }
            if cursor.goto_next_sibling() {
                break;
            }
        }
        if reached_root {
            break;
        }
    }
}

const TS_IGNORED_TYPES: &[&str] = &[
    "string",
    "number",
    "boolean",
    "bigint",
    "symbol",
    "void",
    "any",
    "unknown",
    "never",
    "object",
    "null",
    "undefined",
    "Record",
    "Array",
    "ReadonlyArray",
    "Map",
    "Set",
    "WeakMap",
    "WeakSet",
    "Date",
    "RegExp",
    "Error",
    "Promise",
    "Function",
];

fn split_ts_top_level(text: &str, sep: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0_usize;
    let mut start = 0_usize;
    for (idx, ch) in text.char_indices() {
        match ch {
            '<' | '(' | '[' | '{' => depth += 1,
            '>' | ')' | ']' | '}' => depth = depth.saturating_sub(1),
            c if c == sep && depth == 0 => {
                parts.push(text[start..idx].trim());
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(text[start..].trim());
    parts
}

fn unwrap_ts_generic<'a>(text: &'a str, wrapper: &str) -> Option<&'a str> {
    let rest = text.strip_prefix(wrapper)?.trim_start();
    let inner = rest.strip_prefix('<')?.strip_suffix('>')?;
    Some(inner.trim())
}

fn normalize_ts_type_name(raw: &str) -> Option<String> {
    let mut current = raw
        .trim()
        .strip_prefix(':')
        .unwrap_or(raw)
        .trim()
        .strip_prefix("readonly ")
        .unwrap_or_else(|| raw.trim().strip_prefix(':').unwrap_or(raw).trim())
        .trim();
    if current.is_empty() {
        return None;
    }

    for _ in 0..4 {
        if let Some(inner) = unwrap_ts_generic(current, "Promise")
            .or_else(|| unwrap_ts_generic(current, "Readonly"))
            .or_else(|| unwrap_ts_generic(current, "Partial"))
            .or_else(|| unwrap_ts_generic(current, "Required"))
            .or_else(|| unwrap_ts_generic(current, "NonNullable"))
        {
            current = inner;
            continue;
        }
        let union_parts = split_ts_top_level(current, '|');
        if union_parts.len() > 1 {
            let non_null: Vec<&str> = union_parts
                .into_iter()
                .filter(|p| !p.is_empty() && !matches!(*p, "null" | "undefined" | "void"))
                .collect();
            if non_null.len() == 1 {
                current = non_null[0];
                continue;
            }
            return None;
        }
        break;
    }

    if let Some(angle) = current.find('<') {
        current = current[..angle].trim();
    }
    if current.is_empty() || current.len() > 128 {
        return None;
    }
    if !current
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$' || c == '.')
    {
        return None;
    }
    let leaf = current.rsplit('.').next().unwrap_or(current);
    if leaf.is_empty() || TS_IGNORED_TYPES.contains(&leaf) {
        return None;
    }
    Some(current.to_string())
}

fn collect_ts_param_types(fn_node: Node, source: &[u8], out: &mut HashMap<String, String>) {
    let params = fn_node
        .child_by_field_name("parameters")
        .or_else(|| find_child_by_kind(&fn_node, "formal_parameters"));
    let Some(params) = params else {
        return;
    };
    let mut cursor = params.walk();
    for param in params.named_children(&mut cursor) {
        if !matches!(param.kind(), "required_parameter" | "optional_parameter") {
            continue;
        }
        let ty = param
            .child_by_field_name("type")
            .or_else(|| find_child_by_kind(&param, "type_annotation"))
            .and_then(|t| normalize_ts_type_name(get_text(&t, source)));
        let Some(ty_name) = ty else {
            continue;
        };
        let pattern = param.child_by_field_name("pattern").or_else(|| {
            let mut pc = param.walk();
            let found = param
                .named_children(&mut pc)
                .find(|c| matches!(c.kind(), "identifier" | "accessibility_modifier"));
            found
        });
        let mut is_prop_param = false;
        let mut ident_name: Option<String> = None;
        let mut pc = param.walk();
        for child in param.named_children(&mut pc) {
            if matches!(child.kind(), "accessibility_modifier" | "readonly") {
                is_prop_param = true;
            } else if ident_name.is_none() && child.kind() == "identifier" {
                ident_name = Some(get_text(&child, source).trim().to_string());
            }
        }
        if ident_name.is_none() {
            if let Some(p) = pattern {
                if p.kind() == "identifier" {
                    ident_name = Some(get_text(&p, source).trim().to_string());
                }
            }
        }
        if let Some(name) = ident_name {
            if !name.is_empty() {
                if is_prop_param {
                    out.insert(format!("this.{name}"), ty_name.clone());
                }
                out.insert(name, ty_name);
            }
        }
    }
}

fn collect_ts_file_return_types(program: Node, source: &[u8]) -> HashMap<String, String> {
    let mut returns = HashMap::new();
    let mut cursor = program.walk();
    for child in program.named_children(&mut cursor) {
        let decl = if child.kind() == "export_statement" {
            child.child_by_field_name("declaration").unwrap_or(child)
        } else {
            child
        };
        match decl.kind() {
            "function_declaration" | "generator_function_declaration" => {
                if let (Some(name_node), Some(ret_node)) = (
                    decl.child_by_field_name("name"),
                    decl.child_by_field_name("return_type"),
                ) {
                    if let Some(ty) = normalize_ts_type_name(get_text(&ret_node, source)) {
                        returns.insert(get_text(&name_node, source).trim().to_string(), ty);
                    }
                }
            }
            "lexical_declaration" | "variable_declaration" => {
                let mut vc = decl.walk();
                for sub in decl.named_children(&mut vc) {
                    if sub.kind() != "variable_declarator" {
                        continue;
                    }
                    let Some(name_node) = sub.child_by_field_name("name") else {
                        continue;
                    };
                    if name_node.kind() != "identifier" {
                        continue;
                    }
                    if let Some(val) = sub.child_by_field_name("value") {
                        if matches!(val.kind(), "arrow_function" | "function_expression") {
                            if let Some(ret_node) = val.child_by_field_name("return_type") {
                                if let Some(ty) =
                                    normalize_ts_type_name(get_text(&ret_node, source))
                                {
                                    returns.insert(
                                        get_text(&name_node, source).trim().to_string(),
                                        ty,
                                    );
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    returns
}

fn infer_ts_rhs_type(
    rhs: Node,
    source: &[u8],
    known_types: &HashMap<String, String>,
    file_return_types: &HashMap<String, String>,
) -> Option<String> {
    let expr = if rhs.kind() == "await_expression" {
        rhs.named_child(0)?
    } else {
        rhs
    };
    match expr.kind() {
        "identifier" => known_types.get(get_text(&expr, source).trim()).cloned(),
        "member_expression" => known_types.get(get_text(&expr, source).trim()).cloned(),
        "new_expression" => {
            let ctor = expr
                .child_by_field_name("constructor")
                .or_else(|| expr.named_child(0))?;
            normalize_ts_type_name(get_text(&ctor, source))
        }
        "call_expression" => {
            let func = expr.child_by_field_name("function")?;
            if func.kind() == "identifier" {
                return file_return_types
                    .get(get_text(&func, source).trim())
                    .cloned();
            }
            None
        }
        "as_expression" | "satisfies_expression" => expr
            .named_child(1)
            .and_then(|ty| normalize_ts_type_name(get_text(&ty, source))),
        _ => None,
    }
}

fn collect_ts_class_field_types(
    class_body: Node,
    source: &[u8],
    file_return_types: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut fields = HashMap::new();
    let mut cursor = class_body.walk();
    for child in class_body.named_children(&mut cursor) {
        match child.kind() {
            "public_field_definition" | "field_definition" => {
                let name = child
                    .child_by_field_name("name")
                    .map(|n| get_text(&n, source).trim().to_string())
                    .unwrap_or_default();
                if name.is_empty() {
                    continue;
                }
                let explicit = child
                    .child_by_field_name("type")
                    .or_else(|| find_child_by_kind(&child, "type_annotation"))
                    .and_then(|t| normalize_ts_type_name(get_text(&t, source)));
                let inferred = explicit.or_else(|| {
                    child
                        .child_by_field_name("value")
                        .and_then(|v| infer_ts_rhs_type(v, source, &fields, file_return_types))
                });
                if let Some(ty) = inferred {
                    fields.insert(format!("this.{name}"), ty);
                }
            }
            "method_definition" => {
                let is_ctor = child
                    .child_by_field_name("name")
                    .is_some_and(|n| get_text(&n, source).trim() == "constructor");
                if is_ctor {
                    let mut ctor_types = fields.clone();
                    collect_ts_param_types(child, source, &mut ctor_types);
                    if let Some(body) = child.child_by_field_name("body") {
                        collect_ts_body_var_types(body, source, &mut ctor_types, file_return_types);
                    }
                    for (k, v) in ctor_types {
                        if k.starts_with("this.") {
                            fields.insert(k, v);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    fields
}

fn collect_ts_body_var_types(
    body: Node,
    source: &[u8],
    local_types: &mut HashMap<String, String>,
    file_return_types: &HashMap<String, String>,
) {
    let mut cursor = body.walk();
    let mut reached_root = false;
    loop {
        let n = cursor.node();
        match n.kind() {
            "function_declaration" | "function_expression" | "arrow_function" | "method_definition" => {
                collect_ts_param_types(n, source, local_types);
            }
            "variable_declarator" => {
                if let Some(name_node) = n.child_by_field_name("name") {
                    if name_node.kind() == "identifier" {
                        let key = get_text(&name_node, source).trim().to_string();
                        let explicit = n
                            .child_by_field_name("type")
                            .or_else(|| find_child_by_kind(&n, "type_annotation"))
                            .and_then(|t| normalize_ts_type_name(get_text(&t, source)));
                        let inferred = explicit.or_else(|| {
                            n.child_by_field_name("value").and_then(|v| {
                                infer_ts_rhs_type(v, source, local_types, file_return_types)
                            })
                        });
                        if let Some(ty) = inferred {
                            local_types.insert(key, ty);
                        }
                    }
                }
            }
            "assignment_expression" => {
                if let (Some(lhs), Some(rhs)) = (
                    n.child_by_field_name("left"),
                    n.child_by_field_name("right"),
                ) {
                    let lhs_text = get_text(&lhs, source).trim();
                    let lhs_key = if lhs.kind() == "identifier"
                        || (lhs.kind() == "member_expression" && lhs_text.starts_with("this."))
                    {
                        Some(lhs_text.to_string())
                    } else {
                        None
                    };
                    if let Some(key) = lhs_key {
                        if let Some(ty) =
                            infer_ts_rhs_type(rhs, source, local_types, file_return_types)
                        {
                            local_types.insert(key, ty);
                        }
                    }
                }
            }
            _ => {}
        }
        if cursor.goto_first_child() {
            continue;
        }
        if cursor.goto_next_sibling() {
            continue;
        }
        loop {
            if !cursor.goto_parent() || cursor.node() == body {
                reached_root = true;
                break;
            }
            if cursor.goto_next_sibling() {
                break;
            }
        }
        if reached_root {
            break;
        }
    }
}

fn collect_ts_receiver_types_for_root(root: Node, source: &[u8]) -> HashMap<String, String> {
    let mut ancestors = Vec::new();
    let mut cur = root.parent();
    while let Some(a) = cur {
        ancestors.push(a);
        cur = a.parent();
    }
    let program = ancestors.last().copied().unwrap_or(root);
    let file_return_types = collect_ts_file_return_types(program, source);

    let mut local_types = HashMap::new();
    // Apply outer scopes before inner scopes so inner bindings win cleanly.
    for ancestor in ancestors.into_iter().rev() {
        match ancestor.kind() {
            "class_body" => {
                local_types.extend(collect_ts_class_field_types(
                    ancestor,
                    source,
                    &file_return_types,
                ));
            }
            "function_declaration"
            | "generator_function_declaration"
            | "function_expression"
            | "arrow_function"
            | "method_definition" => {
                collect_ts_param_types(ancestor, source, &mut local_types);
            }
            _ => {}
        }
    }
    collect_ts_body_var_types(root, source, &mut local_types, &file_return_types);
    local_types
}

fn attach_ts_receiver_types(sites: &mut [CallSite], local_types: &HashMap<String, String>) {
    for site in sites {
        if site.receiver_type.is_none() {
            if let Some(qualifier) = &site.qualifier {
                if let Some(ty) = local_types.get(qualifier) {
                    site.receiver_type = Some(ty.clone());
                }
            }
        }
    }
}

fn extract_body_references(
    root: Node,
    source: &[u8],
    language: &str,
    enclosing_qname: &str,
    edges: &mut Vec<ParsedEdge>,
    skip_named_decls: bool,
    global_aliases: &HashSet<String>,
) {
    let local_types = collect_ts_receiver_types_for_root(root, source);
    let blocklist: HashSet<&str> = TS_BUILTIN_CALLS.iter().copied().collect();
    if !skip_named_decls {
        let mut sites = extract_call_sites(&root, source, language, &blocklist);
        attach_ts_receiver_types(&mut sites, &local_types);
        edges.extend(build_call_edges(enclosing_qname, sites));
    } else {
        // For module-scope blocks/callbacks, collect call sites while skipping named declarations.
        let mut sites = Vec::new();
        let mut cursor = root.walk();
        let mut reached_root = false;
        loop {
            let n = cursor.node();
            let is_nested_named = n != root
                && matches!(
                    n.kind(),
                    "class_declaration"
                        | "function_declaration"
                        | "generator_function_declaration"
                        | "method_definition"
                );
            if is_nested_named {
                // Skip named declarations handled separately with their own scope.
            } else {
                if matches!(n.kind(), "call_expression" | "new_expression") {
                    for s in extract_call_sites(&n, source, language, &blocklist) {
                        if s.start_line == n.start_position().row {
                            sites.push(s);
                            break;
                        }
                    }
                }
                if cursor.goto_first_child() {
                    continue;
                }
            }
            if cursor.goto_next_sibling() {
                continue;
            }
            loop {
                if !cursor.goto_parent() || cursor.node() == root {
                    reached_root = true;
                    break;
                }
                if cursor.goto_next_sibling() {
                    break;
                }
            }
            if reached_root {
                break;
            }
        }
        attach_ts_receiver_types(&mut sites, &local_types);
        edges.extend(build_call_edges(enclosing_qname, sites));
    }

    let is_jsx = language == "tsx" || language == "jsx";
    let mut jsx_sites: Vec<SiteOccurrence> = Vec::new();
    let mut type_sites: Vec<SiteOccurrence> = Vec::new();

    let mut cursor = root.walk();
    let mut reached_root = false;
    loop {
        let n = cursor.node();
        let skip = skip_named_decls
            && n != root
            && matches!(
                n.kind(),
                "class_declaration"
                    | "function_declaration"
                    | "generator_function_declaration"
                    | "method_definition"
            );
        if !skip {
            if is_jsx
                && matches!(n.kind(), "jsx_opening_element" | "jsx_self_closing_element")
            {
                if let Some(tag) = n.child_by_field_name("name") {
                    match tag.kind() {
                        "identifier" => {
                            let name = get_text(&tag, source).trim();
                            if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
                                jsx_sites.push(SiteOccurrence {
                                    target: name.to_string(),
                                    qualifier: None,
                                    start_line: n.start_position().row,
                                    end_line: n.end_position().row,
                                });
                            }
                        }
                        "member_expression" => {
                            if let Some(full) = dotted_member_name(&tag, source) {
                                if let Some((q, tail)) = full.rsplit_once('.') {
                                    jsx_sites.push(SiteOccurrence {
                                        target: tail.to_string(),
                                        qualifier: Some(q.to_string()),
                                        start_line: n.start_position().row,
                                        end_line: n.end_position().row,
                                    });
                                }
                            }
                        }
                        _ => {}
                    }
                }
            } else if matches!(
                n.kind(),
                "type_annotation" | "type_arguments" | "as_expression" | "satisfies_expression"
            ) {
                collect_type_references_in_type_node(n, source, None, &mut type_sites);
            }
            extract_ts_http_and_bridges(&n, source, enclosing_qname, edges, global_aliases);

            if cursor.goto_first_child() {
                continue;
            }
        }
        if cursor.goto_next_sibling() {
            continue;
        }
        loop {
            if !cursor.goto_parent() || cursor.node() == root {
                reached_root = true;
                break;
            }
            if cursor.goto_next_sibling() {
                break;
            }
        }
        if reached_root {
            break;
        }
    }

    if !jsx_sites.is_empty() {
        emit_grouped_references(enclosing_qname, "renders", jsx_sites, edges);
    }
    if !type_sites.is_empty() {
        emit_grouped_references(enclosing_qname, "type_references", type_sites, edges);
    }
}

fn fn_signature(fn_node: &Node, params_field: &str, alt_field: Option<&str>, source: &[u8]) -> String {
    let params = fn_node
        .child_by_field_name(params_field)
        .or_else(|| alt_field.and_then(|f| fn_node.child_by_field_name(f)))
        .or_else(|| find_child_by_kind(fn_node, "formal_parameters"))
        .map(|p| get_text(&p, source).trim().to_string())
        .unwrap_or_default();
    let ret = fn_node
        .child_by_field_name("return_type")
        .map(|r| get_text(&r, source).trim().to_string())
        .unwrap_or_default();
    if !params.is_empty() && !ret.is_empty() {
        format!("{params}{ret}")
    } else {
        params
    }
}

fn emit_signature_type_references(
    fn_node: &Node,
    source: &[u8],
    enclosing_qname: &str,
    edges: &mut Vec<ParsedEdge>,
) {
    let mut type_sites = Vec::new();
    for field in ["parameters", "parameter", "return_type"] {
        if let Some(child) = fn_node.child_by_field_name(field) {
            collect_type_references_in_type_node(child, source, None, &mut type_sites);
        }
    }
    if let Some(formal) = find_child_by_kind(fn_node, "formal_parameters") {
        collect_type_references_in_type_node(formal, source, None, &mut type_sites);
    }
    if !type_sites.is_empty() {
        emit_grouped_references(enclosing_qname, "type_references", type_sites, edges);
    }
}

fn commonjs_export_target(left: Node, source: &[u8]) -> Option<Option<String>> {
    if left.kind() != "member_expression" {
        return None;
    }
    let obj = left.child_by_field_name("object")?;
    let prop = left.child_by_field_name("property")?;
    let prop_name = get_text(&prop, source);
    if obj.kind() == "identifier" {
        let obj_name = get_text(&obj, source);
        if obj_name == "module" && prop_name == "exports" {
            return Some(None);
        }
        if obj_name == "exports" {
            return Some(Some(prop_name.to_string()));
        }
    } else if obj.kind() == "member_expression" {
        let root = obj.child_by_field_name("object")?;
        let mid = obj.child_by_field_name("property")?;
        if root.kind() == "identifier"
            && get_text(&root, source) == "module"
            && get_text(&mid, source) == "exports"
        {
            return Some(Some(prop_name.to_string()));
        }
    }
    None
}

/// Parse TypeScript/JavaScript files using tree-sitter.
///
/// Handles classes, functions, arrow functions, interfaces, type
/// aliases, import/export statements, CommonJS `require`/`exports`,
/// and const/let/var declarations.
/// TSX/JSX are parsed using the TSX grammar.
pub struct TypeScriptParser {
    language_name: String,
}

impl TypeScriptParser {
    /// Create a parser for the given language variant.
    ///
    /// # Arguments
    /// * `language` — One of "typescript", "tsx", "javascript", "jsx"
    pub fn new(language: &str) -> Self {
        Self {
            language_name: language.to_string(),
        }
    }

    /// Create a fresh tree-sitter parser configured for the right language.
    fn create_parser(&self) -> Parser {
        let mut parser = Parser::new();
        let lang = if self.language_name == "tsx" || self.language_name == "jsx" {
            tree_sitter_typescript::LANGUAGE_TSX
        } else {
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT
        };
        parser.set_language(&lang.into()).unwrap();
        parser
    }

    fn emit_import(
        &self,
        node: Node,
        import_path: &str,
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
    ) {
        self.emit_import_with_metadata(
            node,
            import_path,
            module_name,
            entities,
            edges,
            HashMap::new(),
        );
    }

    fn emit_import_with_metadata(
        &self,
        node: Node,
        import_path: &str,
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        mut edge_meta: HashMap<String, Value>,
    ) {
        let qualified = format!("{module_name}.import.{import_path}");
        entities.push(ParsedEntity {
            kind: NodeType::Import,
            name: import_path.to_string(),
            qualified_name: qualified.clone(),
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            imports: vec![import_path.to_string()],
            signature: String::new(),
            docstring: String::new(),
            raw_text: String::new(),
            parent_qualified_name: Some(module_name.to_string()),
            bases: Vec::new(),
            cyclomatic_complexity: None,
        });
        // CONTAINS edge from file → import node so it isn't an orphan.
        edges.push(ParsedEdge::new(
            module_name.to_string(),
            qualified,
            "contains",
        ));
        edge_meta
            .entry("start_line".to_string())
            .or_insert_with(|| json!(node.start_position().row));
        edge_meta
            .entry("end_line".to_string())
            .or_insert_with(|| json!(node.end_position().row));
        edge_meta
            .entry("enclosing_symbol".to_string())
            .or_insert_with(|| json!(module_name));
        edges.push(ParsedEdge::with_metadata(
            module_name.to_string(),
            import_path.to_string(),
            "imports",
            edge_meta,
        ));
    }

    fn reexport_metadata(&self, node: Node, source: &[u8]) -> HashMap<String, Value> {
        let mut meta = HashMap::new();
        meta.insert("reference_kind".to_string(), json!("reexports"));
        meta.insert("is_reexport".to_string(), json!(true));
        let mut bindings: Vec<Value> = Vec::new();
        let mut imported_symbols: Vec<Value> = Vec::new();
        let mut wildcard = false;

        if let Some(clause) = find_child_by_kind(&node, "export_clause") {
            let mut cursor = clause.walk();
            for spec in clause.named_children(&mut cursor) {
                if spec.kind() == "export_specifier" {
                    let imported = spec
                        .child_by_field_name("name")
                        .map(|n| get_text(&n, source).trim().to_string())
                        .unwrap_or_default();
                    let exported = spec
                        .child_by_field_name("alias")
                        .map(|a| get_text(&a, source).trim().to_string())
                        .unwrap_or_else(|| imported.clone());
                    if !imported.is_empty() && !exported.is_empty() {
                        imported_symbols.push(json!(imported));
                        bindings.push(json!({
                            "imported": imported,
                            "exported": exported,
                            "kind": "named",
                        }));
                    }
                }
            }
        } else if let Some(ns) = find_child_by_kind(&node, "namespace_export") {
            let exported = find_child_by_kind(&ns, "identifier")
                .map(|id| get_text(&id, source).trim().to_string())
                .unwrap_or_default();
            wildcard = true;
            if !exported.is_empty() {
                bindings.push(json!({
                    "imported": "*",
                    "exported": exported,
                    "kind": "namespace",
                }));
            }
        } else {
            wildcard = true;
            bindings.push(json!({
                "imported": "*",
                "exported": "*",
                "kind": "star",
            }));
        }

        if !imported_symbols.is_empty() {
            meta.insert("imported_symbols".to_string(), Value::Array(imported_symbols));
        }
        if !bindings.is_empty() {
            meta.insert("reexport_bindings".to_string(), Value::Array(bindings));
        }
        meta.insert("wildcard_import".to_string(), json!(wildcard));
        meta
    }

    fn maybe_emit_default_export(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        edges: &mut Vec<ParsedEdge>,
    ) {
        let mut is_default = false;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if !child.is_named() && get_text(&child, source) == "default" {
                is_default = true;
                break;
            }
        }
        if !is_default {
            return;
        }
        let val = node.child_by_field_name("value").or_else(|| node.named_child(0));
        let Some(target_node) = val else {
            return;
        };
        let local_name = match target_node.kind() {
            "identifier" => Some(get_text(&target_node, source).trim().to_string()),
            "function_declaration" | "generator_function_declaration" => {
                find_child_by_kind(&target_node, "identifier")
                    .map(|id| get_text(&id, source).trim().to_string())
            }
            "class_declaration" => find_child_by_kind(&target_node, "type_identifier")
                .or_else(|| find_child_by_kind(&target_node, "identifier"))
                .map(|id| get_text(&id, source).trim().to_string()),
            _ => None,
        };
        if let Some(name) = local_name.filter(|s| !s.is_empty()) {
            let mut meta = HashMap::new();
            meta.insert("reference_kind".to_string(), json!("export_default"));
            meta.insert("local_name".to_string(), json!(name));
            meta.insert("start_line".to_string(), json!(node.start_position().row));
            meta.insert("end_line".to_string(), json!(node.end_position().row));
            meta.insert("enclosing_symbol".to_string(), json!(module_name));
            edges.push(ParsedEdge::with_metadata(
                module_name.to_string(),
                format!("__scs_export_default__:{name}"),
                "references",
                meta,
            ));
        }
    }

    fn extract_module_imports_in_subtree(
        &self,
        root: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
    ) {
        let mut cursor = root.walk();
        let mut reached_root = false;
        loop {
            let n = cursor.node();
            if let Some(import_path) = extract_module_import_call(&n, source) {
                self.emit_import(n, &import_path, module_name, entities, edges);
            } else if let Some(asset_path) = extract_url_import_meta(&n, source) {
                let mut meta = HashMap::new();
                meta.insert("reference_kind".to_string(), json!("imports_asset"));
                meta.insert("wildcard_import".to_string(), json!(true));
                self.emit_import_with_metadata(n, &asset_path, module_name, entities, edges, meta);
            }

            if cursor.node() == root {
                if !cursor.goto_first_child() {
                    break;
                }
                continue;
            }
            if cursor.goto_first_child() {
                continue;
            }
            if cursor.goto_next_sibling() {
                continue;
            }
            loop {
                if !cursor.goto_parent() || cursor.node() == root {
                    reached_root = true;
                    break;
                }
                if cursor.goto_next_sibling() {
                    break;
                }
            }
            if reached_root {
                break;
            }
        }
    }

    fn walk(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
        global_aliases: &HashSet<String>,
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.walk_node(
                child,
                source,
                module_name,
                entities,
                edges,
                scope_stack,
                global_aliases,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_iife_body(
        &self,
        body: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
        outer_aliases: &HashSet<String>,
        iife_aliases: &HashSet<String>,
    ) {
        let mut merged = outer_aliases.clone();
        merged.extend(iife_aliases.iter().cloned());
        let mut cursor = body.walk();
        for stmt in body.children(&mut cursor) {
            match stmt.kind() {
                "function_declaration"
                | "generator_function_declaration"
                | "class_declaration"
                | "lexical_declaration"
                | "variable_declaration"
                | "assignment_expression"
                | "expression_statement" => {
                    self.walk_node(
                        stmt,
                        source,
                        module_name,
                        entities,
                        edges,
                        scope_stack,
                        &merged,
                    );
                }
                "if_statement" | "try_statement" | "for_statement" | "for_in_statement"
                | "while_statement" | "do_statement" | "switch_statement" => {
                    extract_body_references(
                        stmt,
                        source,
                        &self.language_name,
                        module_name,
                        edges,
                        true,
                        &merged,
                    );
                    self.extract_module_imports_in_subtree(
                        stmt,
                        source,
                        module_name,
                        entities,
                        edges,
                    );
                    self.walk_iife_control_blocks(
                        stmt,
                        source,
                        module_name,
                        entities,
                        edges,
                        scope_stack,
                        &merged,
                    );
                }
                _ => {}
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_iife_control_blocks(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
        global_aliases: &HashSet<String>,
    ) {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "function_declaration" | "generator_function_declaration" => {
                    self.handle_function(
                        child,
                        source,
                        module_name,
                        entities,
                        edges,
                        scope_stack,
                        global_aliases,
                    );
                }
                "class_declaration" => {
                    self.handle_class(
                        child,
                        source,
                        module_name,
                        entities,
                        edges,
                        scope_stack,
                        global_aliases,
                    );
                }
                "lexical_declaration" | "variable_declaration" => {
                    self.handle_variable(
                        child,
                        source,
                        module_name,
                        entities,
                        edges,
                        scope_stack,
                        global_aliases,
                    );
                }
                "expression_statement" => {
                    if let Some(expr) = child.named_child(0) {
                        if expr.kind() == "assignment_expression"
                            && expr.child_by_field_name("right").is_some_and(|r| {
                                matches!(
                                    r.kind(),
                                    "arrow_function" | "function_expression" | "generator_function"
                                )
                            })
                        {
                            self.handle_assignment(
                                expr,
                                source,
                                module_name,
                                entities,
                                edges,
                                scope_stack,
                                global_aliases,
                            );
                        }
                    }
                }
                "statement_block" | "else_clause" | "catch_clause" | "finally_clause"
                | "switch_body" | "switch_case" | "switch_default" | "if_statement"
                | "try_statement" => {
                    self.walk_iife_control_blocks(
                        child,
                        source,
                        module_name,
                        entities,
                        edges,
                        scope_stack,
                        global_aliases,
                    );
                }
                _ => {}
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_node(
        &self,
        child: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
        global_aliases: &HashSet<String>,
    ) {
        if scope_stack.len() == 1
            && matches!(
                child.kind(),
                "call_expression" | "unary_expression" | "parenthesized_expression"
            )
        {
            if let Some((iife_body, iife_aliases)) = extract_top_level_iife(child, source) {
                self.walk_iife_body(
                    iife_body,
                    source,
                    module_name,
                    entities,
                    edges,
                    scope_stack,
                    global_aliases,
                    &iife_aliases,
                );
                return;
            }
        }

        match child.kind() {
            "class_declaration" => {
                self.handle_class(
                    child,
                    source,
                    module_name,
                    entities,
                    edges,
                    scope_stack,
                    global_aliases,
                );
            }
            "function_declaration" | "generator_function_declaration" => {
                self.handle_function(
                    child,
                    source,
                    module_name,
                    entities,
                    edges,
                    scope_stack,
                    global_aliases,
                );
            }
            "interface_declaration" | "type_alias_declaration" => {
                self.handle_type_declaration(
                    child,
                    source,
                    module_name,
                    entities,
                    edges,
                    scope_stack,
                );
            }
            "import_statement" | "import_declaration" => {
                self.handle_import(child, source, module_name, entities, edges);
            }
            "lexical_declaration" | "variable_declaration" => {
                self.handle_variable(
                    child,
                    source,
                    module_name,
                    entities,
                    edges,
                    scope_stack,
                    global_aliases,
                );
            }
            "assignment_expression" if scope_stack.len() == 1 => {
                self.handle_assignment(
                    child,
                    source,
                    module_name,
                    entities,
                    edges,
                    scope_stack,
                    global_aliases,
                );
            }
            "new_expression" => {
                if let Some(asset_path) = extract_url_import_meta(&child, source) {
                    let mut meta = HashMap::new();
                    meta.insert("reference_kind".to_string(), json!("imports_asset"));
                    meta.insert("wildcard_import".to_string(), json!(true));
                    self.emit_import_with_metadata(
                        child,
                        &asset_path,
                        module_name,
                        entities,
                        edges,
                        meta,
                    );
                } else {
                    extract_body_references(
                        child,
                        source,
                        &self.language_name,
                        module_name,
                        edges,
                        true,
                        global_aliases,
                    );
                    self.walk(
                        child,
                        source,
                        module_name,
                        entities,
                        edges,
                        scope_stack,
                        global_aliases,
                    );
                }
            }
            "call_expression" => {
                if let Some(import_path) = extract_module_import_call(&child, source) {
                    let mut meta = HashMap::new();
                    meta.insert("reference_kind".to_string(), json!("imports_namespace"));
                    meta.insert("wildcard_import".to_string(), json!(true));
                    self.emit_import_with_metadata(
                        child,
                        &import_path,
                        module_name,
                        entities,
                        edges,
                        meta,
                    );
                } else {
                    extract_body_references(
                        child,
                        source,
                        &self.language_name,
                        module_name,
                        edges,
                        true,
                        global_aliases,
                    );
                    if let Some(func) = child.child_by_field_name("function") {
                        self.walk_node(
                            func,
                            source,
                            module_name,
                            entities,
                            edges,
                            scope_stack,
                            global_aliases,
                        );
                    }
                }
                if let Some(args) = child.child_by_field_name("arguments") {
                    self.walk(
                        args,
                        source,
                        module_name,
                        entities,
                        edges,
                        scope_stack,
                        global_aliases,
                    );
                }
            }
            "arrow_function" | "function_expression" | "generator_function" => {
                if let Some(body) = child.child_by_field_name("body") {
                    self.extract_module_imports_in_subtree(
                        body,
                        source,
                        module_name,
                        entities,
                        edges,
                    );
                    extract_body_references(
                        child,
                        source,
                        &self.language_name,
                        module_name,
                        edges,
                        true,
                        global_aliases,
                    );
                }
            }
            "export_statement" => {
                if let Some(src) = child.child_by_field_name("source") {
                    if let Some(import_path) = extract_string_literal(&src, source) {
                        let meta = self.reexport_metadata(child, source);
                        self.emit_import_with_metadata(
                            child,
                            &import_path,
                            module_name,
                            entities,
                            edges,
                            meta,
                        );
                    }
                } else {
                    self.maybe_emit_default_export(child, source, module_name, edges);
                }
                // Unwrap export and process the inner declaration.
                self.walk(
                    child,
                    source,
                    module_name,
                    entities,
                    edges,
                    scope_stack,
                    global_aliases,
                );
            }
            "method_definition" | "public_field_definition" => {
                self.handle_method(
                    child,
                    source,
                    module_name,
                    entities,
                    edges,
                    scope_stack,
                    global_aliases,
                );
            }
            "pair" | "member_expression" => {
                extract_ts_http_and_bridges(
                    &child,
                    source,
                    module_name,
                    edges,
                    global_aliases,
                );
                self.walk(
                    child,
                    source,
                    module_name,
                    entities,
                    edges,
                    scope_stack,
                    global_aliases,
                );
            }
            _ => {
                // Don't recurse into function bodies.
                if child.child_count() > 0
                    && child.kind() != "statement_block"
                    && child.kind() != "function_body"
                {
                    self.walk(
                        child,
                        source,
                        module_name,
                        entities,
                        edges,
                        scope_stack,
                        global_aliases,
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_class(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
        global_aliases: &HashSet<String>,
    ) {
        let mut name = String::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "type_identifier" || child.kind() == "identifier" {
                name = get_text(&child, source).to_string();
                break;
            }
        }
        if name.is_empty() {
            return;
        }

        let qualified = format!("{}.{name}", scope_stack.join("."));

        // Extract heritage (extends/implements) with exact relationship kinds & lines.
        let mut bases = Vec::new();
        let mut inherit_targets = Vec::new();
        let mut implement_targets = Vec::new();
        let mut cursor2 = node.walk();
        for child in node.children(&mut cursor2) {
            match child.kind() {
                "class_heritage" => {
                    let mut hc = child.walk();
                    for clause in child.children(&mut hc) {
                        match clause.kind() {
                            "extends_clause" => {
                                let mut sc = clause.walk();
                                for sub in clause.children(&mut sc) {
                                    if matches!(sub.kind(), "type_identifier" | "identifier") {
                                        let target = get_text(&sub, source).to_string();
                                        bases.push(target.clone());
                                        inherit_targets.push((
                                            target,
                                            sub.start_position().row,
                                            sub.end_position().row,
                                        ));
                                    }
                                }
                            }
                            "implements_clause" => {
                                let mut sc = clause.walk();
                                for sub in clause.children(&mut sc) {
                                    if matches!(sub.kind(), "type_identifier" | "identifier") {
                                        let target = get_text(&sub, source).to_string();
                                        bases.push(target.clone());
                                        implement_targets.push((
                                            target,
                                            sub.start_position().row,
                                            sub.end_position().row,
                                        ));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "extends_clause" => {
                    let mut sc = child.walk();
                    for sub in child.children(&mut sc) {
                        if matches!(sub.kind(), "type_identifier" | "identifier") {
                            let target = get_text(&sub, source).to_string();
                            bases.push(target.clone());
                            inherit_targets.push((
                                target,
                                sub.start_position().row,
                                sub.end_position().row,
                            ));
                        }
                    }
                }
                "implements_clause" => {
                    let mut sc = child.walk();
                    for sub in child.children(&mut sc) {
                        if matches!(sub.kind(), "type_identifier" | "identifier") {
                            let target = get_text(&sub, source).to_string();
                            bases.push(target.clone());
                            implement_targets.push((
                                target,
                                sub.start_position().row,
                                sub.end_position().row,
                            ));
                        }
                    }
                }
                _ => {}
            }
        }

        let raw = get_text(&node, source);
        entities.push(ParsedEntity {
            kind: NodeType::Class,
            name: name.clone(),
            qualified_name: qualified.clone(),
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            raw_text: super::truncate_str(raw, super::RAW_TEXT_LIMIT).to_string(),
            parent_qualified_name: Some(scope_stack.join(".")),
            bases: bases.clone(),
            signature: String::new(),
            docstring: get_doc_comment_for_declaration(&node, source),
            imports: Vec::new(),
            cyclomatic_complexity: None,
        });

        edges.push(ParsedEdge::new(
            scope_stack.join("."),
            qualified.clone(),
            "contains",
        ));

        for (base, start_line, end_line) in inherit_targets {
            let mut meta = HashMap::new();
            meta.insert("reference_kind".to_string(), json!("extends"));
            meta.insert("start_line".to_string(), json!(start_line));
            meta.insert("end_line".to_string(), json!(end_line));
            meta.insert("enclosing_symbol".to_string(), json!(qualified));
            edges.push(ParsedEdge::with_metadata(
                qualified.clone(),
                base,
                "inherits",
                meta,
            ));
        }
        for (iface, start_line, end_line) in implement_targets {
            let mut meta = HashMap::new();
            meta.insert("reference_kind".to_string(), json!("implements"));
            meta.insert("start_line".to_string(), json!(start_line));
            meta.insert("end_line".to_string(), json!(end_line));
            meta.insert("enclosing_symbol".to_string(), json!(qualified));
            edges.push(ParsedEdge::with_metadata(
                qualified.clone(),
                iface,
                "implements",
                meta,
            ));
        }

        // Recurse into class body.
        if let Some(body) = find_child_by_kind(&node, "class_body") {
            let mut new_scope = scope_stack.to_vec();
            new_scope.push(name);
            self.walk(
                body,
                source,
                module_name,
                entities,
                edges,
                &new_scope,
                global_aliases,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_function_value(
        &self,
        name: String,
        fn_node: Node,
        doc_node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
        global_aliases: &HashSet<String>,
    ) {
        if name.is_empty() {
            return;
        }
        let qualified = format!("{}.{name}", scope_stack.join("."));
        let kind = if scope_stack.len() > 1 {
            NodeType::Method
        } else {
            NodeType::Function
        };
        let signature = fn_signature(&fn_node, "parameters", Some("parameter"), source);
        let raw = get_text(&fn_node, source);
        let complexity = count_complexity(&fn_node, source, &self.language_name);

        entities.push(ParsedEntity {
            kind,
            name,
            qualified_name: qualified.clone(),
            start_line: fn_node.start_position().row,
            end_line: fn_node.end_position().row,
            signature,
            raw_text: super::truncate_str(raw, super::RAW_TEXT_LIMIT).to_string(),
            parent_qualified_name: Some(scope_stack.join(".")),
            docstring: get_doc_comment_for_declaration(&doc_node, source),
            bases: Vec::new(),
            imports: Vec::new(),
            cyclomatic_complexity: Some(complexity),
        });

        edges.push(ParsedEdge::new(
            scope_stack.join("."),
            qualified.clone(),
            "contains",
        ));

        emit_signature_type_references(&fn_node, source, &qualified, edges);
        if let Some(body) = fn_node.child_by_field_name("body") {
            extract_body_references(
                body,
                source,
                &self.language_name,
                &qualified,
                edges,
                false,
                global_aliases,
            );
            self.extract_module_imports_in_subtree(body, source, module_name, entities, edges);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_function(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
        global_aliases: &HashSet<String>,
    ) {
        let mut name = String::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "identifier" {
                name = get_text(&child, source).to_string();
                break;
            }
        }
        if name.is_empty() {
            return;
        }

        let qualified = format!("{}.{name}", scope_stack.join("."));
        let is_method = scope_stack.len() > 1;
        let kind = if is_method {
            NodeType::Method
        } else {
            NodeType::Function
        };

        let signature = fn_signature(&node, "parameters", None, source);

        let raw = get_text(&node, source);
        let complexity = count_complexity(&node, source, &self.language_name);
        entities.push(ParsedEntity {
            kind,
            name,
            qualified_name: qualified.clone(),
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            signature,
            raw_text: super::truncate_str(raw, super::RAW_TEXT_LIMIT).to_string(),
            parent_qualified_name: Some(scope_stack.join(".")),
            docstring: get_doc_comment_for_declaration(&node, source),
            bases: Vec::new(),
            imports: Vec::new(),
            cyclomatic_complexity: Some(complexity),
        });

        edges.push(ParsedEdge::new(
            scope_stack.join("."),
            qualified.clone(),
            "contains",
        ));

        emit_signature_type_references(&node, source, &qualified, edges);
        // CALLS, JSX renders, type references, HTTP & bridge descriptors, and lazy imports.
        if let Some(body) = find_child_by_kind(&node, "statement_block") {
            extract_body_references(
                body,
                source,
                &self.language_name,
                &qualified,
                edges,
                false,
                global_aliases,
            );
            self.extract_module_imports_in_subtree(body, source, module_name, entities, edges);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_method(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
        global_aliases: &HashSet<String>,
    ) {
        let mut name = String::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "property_identifier" || child.kind() == "identifier" {
                name = get_text(&child, source).to_string();
                break;
            }
        }
        if name.is_empty() {
            return;
        }

        if node.kind() == "public_field_definition" {
            if let Some(value) = node.child_by_field_name("value") {
                if matches!(
                    value.kind(),
                    "arrow_function" | "function_expression" | "generator_function"
                ) {
                    self.handle_function_value(
                        name,
                        value,
                        node,
                        source,
                        module_name,
                        entities,
                        edges,
                        scope_stack,
                        global_aliases,
                    );
                    return;
                }
                self.extract_module_imports_in_subtree(
                    value,
                    source,
                    module_name,
                    entities,
                    edges,
                );
            }
        }

        let qualified = format!("{}.{name}", scope_stack.join("."));
        let signature = fn_signature(&node, "parameters", None, source);

        let raw = get_text(&node, source);
        let complexity = count_complexity(&node, source, &self.language_name);
        entities.push(ParsedEntity {
            kind: NodeType::Method,
            name,
            qualified_name: qualified.clone(),
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            signature,
            raw_text: super::truncate_str(raw, super::RAW_TEXT_LIMIT).to_string(),
            parent_qualified_name: Some(scope_stack.join(".")),
            docstring: get_doc_comment_for_declaration(&node, source),
            bases: Vec::new(),
            imports: Vec::new(),
            cyclomatic_complexity: Some(complexity),
        });

        edges.push(ParsedEdge::new(
            scope_stack.join("."),
            qualified.clone(),
            "contains",
        ));

        emit_signature_type_references(&node, source, &qualified, edges);
        // CALLS, JSX renders, type references, HTTP & bridge descriptors, and lazy imports.
        if let Some(body) = find_child_by_kind(&node, "statement_block") {
            extract_body_references(
                body,
                source,
                &self.language_name,
                &qualified,
                edges,
                false,
                global_aliases,
            );
            self.extract_module_imports_in_subtree(body, source, module_name, entities, edges);
        }
    }

    fn handle_type_declaration(
        &self,
        node: Node,
        source: &[u8],
        _module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
    ) {
        let mut name = String::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "type_identifier" || child.kind() == "identifier" {
                name = get_text(&child, source).to_string();
                break;
            }
        }
        if name.is_empty() {
            return;
        }

        let qualified = format!("{}.{name}", scope_stack.join("."));
        let raw = get_text(&node, source);

        // Interfaces are class-like; type aliases are TYPE_ALIAS.
        let kind = if node.kind() == "interface_declaration" {
            NodeType::Class
        } else {
            NodeType::TypeAlias
        };

        let mut bases = Vec::new();
        if let Some(ext) = find_child_by_kind(&node, "extends_type_clause") {
            let mut ec = ext.walk();
            for sub in ext.named_children(&mut ec) {
                if matches!(sub.kind(), "type_identifier" | "identifier") {
                    let base = get_text(&sub, source).trim().to_string();
                    if !base.is_empty() {
                        bases.push(base.clone());
                        let mut meta = HashMap::new();
                        meta.insert("reference_kind".to_string(), json!("extends"));
                        meta.insert("start_line".to_string(), json!(sub.start_position().row));
                        meta.insert("end_line".to_string(), json!(sub.end_position().row));
                        meta.insert("enclosing_symbol".to_string(), json!(qualified));
                        edges.push(ParsedEdge::with_metadata(
                            qualified.clone(),
                            base,
                            "inherits",
                            meta,
                        ));
                    }
                }
            }
        }

        entities.push(ParsedEntity {
            kind,
            name: name.clone(),
            qualified_name: qualified.clone(),
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            raw_text: super::truncate_str(raw, super::RAW_TEXT_SMALL_LIMIT).to_string(),
            parent_qualified_name: Some(scope_stack.join(".")),
            signature: String::new(),
            docstring: get_doc_comment_for_declaration(&node, source),
            bases,
            imports: Vec::new(),
            cyclomatic_complexity: None,
        });

        edges.push(ParsedEdge::new(
            scope_stack.join("."),
            qualified.clone(),
            "contains",
        ));

        let mut type_sites = Vec::new();
        collect_type_references_in_type_node(node, source, Some(&name), &mut type_sites);
        if !type_sites.is_empty() {
            emit_grouped_references(&qualified, "type_references", type_sites, edges);
        }
    }

    fn handle_import(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
    ) {
        let mut bindings: Vec<Value> = Vec::new();
        let mut imported_symbols: Vec<Value> = Vec::new();
        let mut wildcard = false;
        let mut has_named = false;
        let mut has_default = false;
        let mut has_namespace = false;

        if let Some(clause) = find_child_by_kind(&node, "import_clause") {
            let mut cc = clause.walk();
            for part in clause.named_children(&mut cc) {
                match part.kind() {
                    "identifier" => {
                        let local = get_text(&part, source).trim().to_string();
                        if !local.is_empty() {
                            has_default = true;
                            imported_symbols.push(json!("default"));
                            bindings.push(json!({
                                "local": local,
                                "imported": "default",
                                "kind": "default",
                            }));
                        }
                    }
                    "namespace_import" => {
                        if let Some(id) = find_child_by_kind(&part, "identifier") {
                            let local = get_text(&id, source).trim().to_string();
                            if !local.is_empty() {
                                has_namespace = true;
                                wildcard = true;
                                bindings.push(json!({
                                    "local": local,
                                    "imported": "*",
                                    "kind": "namespace",
                                }));
                            }
                        }
                    }
                    "named_imports" => {
                        let mut nc = part.walk();
                        for spec in part.named_children(&mut nc) {
                            if spec.kind() == "import_specifier" {
                                let orig = spec
                                    .child_by_field_name("name")
                                    .map(|n| get_text(&n, source).trim().to_string())
                                    .unwrap_or_default();
                                let local = spec
                                    .child_by_field_name("alias")
                                    .map(|a| get_text(&a, source).trim().to_string())
                                    .unwrap_or_else(|| orig.clone());
                                if !orig.is_empty() && !local.is_empty() {
                                    has_named = true;
                                    imported_symbols.push(json!(orig));
                                    bindings.push(json!({
                                        "local": local,
                                        "imported": orig,
                                        "kind": "named",
                                    }));
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        } else if let Some(req_clause) = find_child_by_kind(&node, "import_require_clause") {
            wildcard = true;
            has_namespace = true;
            if let Some(id) = find_child_by_kind(&req_clause, "identifier") {
                let local = get_text(&id, source).trim().to_string();
                if !local.is_empty() {
                    bindings.push(json!({
                        "local": local,
                        "imported": "*",
                        "kind": "namespace",
                    }));
                }
            }
        } else {
            // Side-effect import (`import './polyfill'`)
            wildcard = true;
        }

        let reference_kind = if has_named {
            "imports_named"
        } else if has_default {
            "imports_default"
        } else if has_namespace {
            "imports_namespace"
        } else {
            "imports_side_effect"
        };

        let mut meta = HashMap::new();
        meta.insert("reference_kind".to_string(), json!(reference_kind));
        meta.insert("wildcard_import".to_string(), json!(wildcard));
        if !imported_symbols.is_empty() {
            meta.insert("imported_symbols".to_string(), Value::Array(imported_symbols));
        }
        if !bindings.is_empty() {
            meta.insert("import_bindings".to_string(), Value::Array(bindings));
        }

        // Extract module specifier from ESM `source` or TS `import x = require(...)`.
        let source_node = node.child_by_field_name("source").or_else(|| {
            find_child_by_kind(&node, "import_require_clause")
                .and_then(|clause| clause.child_by_field_name("source"))
        });
        if let Some(src) = source_node {
            if let Some(import_path) = extract_string_literal(&src, source) {
                self.emit_import_with_metadata(
                    node,
                    &import_path,
                    module_name,
                    entities,
                    edges,
                    meta,
                );
                return;
            }
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if let Some(import_path) = extract_string_literal(&child, source) {
                self.emit_import_with_metadata(
                    node,
                    &import_path,
                    module_name,
                    entities,
                    edges,
                    meta,
                );
                break;
            }
        }
    }

    #[allow(clippy::ptr_arg, clippy::too_many_arguments)]
    fn handle_variable(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
        global_aliases: &HashSet<String>,
    ) {
        // Only extract top-level declarations.
        if scope_stack.len() > 1 {
            return;
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() != "variable_declarator" {
                continue;
            }

            let name_node = child.child_by_field_name("name");
            let value_node = child.child_by_field_name("value");
            let ident_name = name_node
                .filter(|n| n.kind() == "identifier")
                .map(|n| get_text(&n, source).to_string())
                .unwrap_or_default();

            if let Some(ann) = find_child_by_kind(&child, "type_annotation") {
                let mut type_sites = Vec::new();
                collect_type_references_in_type_node(ann, source, None, &mut type_sites);
                if !type_sites.is_empty() {
                    emit_grouped_references(module_name, "type_references", type_sites, edges);
                }
            }

            let mut handled_iife: Option<(Node, HashSet<String>)> = None;
            if let Some(val) = value_node {
                if matches!(
                    val.kind(),
                    "arrow_function" | "function_expression" | "generator_function"
                ) {
                    if !ident_name.is_empty() {
                        self.handle_function_value(
                            ident_name,
                            val,
                            node,
                            source,
                            module_name,
                            entities,
                            edges,
                            scope_stack,
                            global_aliases,
                        );
                    } else {
                        extract_body_references(
                            val,
                            source,
                            &self.language_name,
                            module_name,
                            edges,
                            true,
                            global_aliases,
                        );
                        self.extract_module_imports_in_subtree(
                            val,
                            source,
                            module_name,
                            entities,
                            edges,
                        );
                    }
                    continue;
                }

                if let Some((iife_body, iife_aliases)) = extract_top_level_iife(val, source) {
                    handled_iife = Some((iife_body, iife_aliases));
                } else if let Some(spec) = extract_module_import_call(&val, source) {
                    let mut meta = HashMap::new();
                    if let Some(pat) = name_node.filter(|n| n.kind() == "object_pattern") {
                        let (bindings, imported_symbols, wildcard) =
                            collect_object_pattern_bindings(pat, source);
                        meta.insert("reference_kind".to_string(), json!("imports_named"));
                        meta.insert("wildcard_import".to_string(), json!(wildcard));
                        if !imported_symbols.is_empty() {
                            meta.insert(
                                "imported_symbols".to_string(),
                                Value::Array(imported_symbols),
                            );
                        }
                        if !bindings.is_empty() {
                            meta.insert("import_bindings".to_string(), Value::Array(bindings));
                        }
                    } else if !ident_name.is_empty() {
                        meta.insert("reference_kind".to_string(), json!("imports_namespace"));
                        meta.insert("wildcard_import".to_string(), json!(true));
                        meta.insert(
                            "import_bindings".to_string(),
                            json!([{
                                "local": ident_name,
                                "imported": "*",
                                "kind": "namespace",
                            }]),
                        );
                    }
                    self.emit_import_with_metadata(val, &spec, module_name, entities, edges, meta);
                } else if let Some((spec, prop_name)) = require_member_access(val, source) {
                    let mut meta = HashMap::new();
                    meta.insert("reference_kind".to_string(), json!("imports_named"));
                    meta.insert("wildcard_import".to_string(), json!(false));
                    meta.insert("imported_symbols".to_string(), json!([prop_name]));
                    if !ident_name.is_empty() {
                        meta.insert(
                            "import_bindings".to_string(),
                            json!([{
                                "local": ident_name,
                                "imported": prop_name,
                                "kind": "named",
                            }]),
                        );
                    }
                    self.emit_import_with_metadata(val, &spec, module_name, entities, edges, meta);
                } else {
                    extract_body_references(
                        val,
                        source,
                        &self.language_name,
                        module_name,
                        edges,
                        true,
                        global_aliases,
                    );
                    self.extract_module_imports_in_subtree(
                        val,
                        source,
                        module_name,
                        entities,
                        edges,
                    );
                }
            }

            if ident_name.len() >= 2 {
                let qualified = format!("{}.{ident_name}", scope_stack.join("."));
                let raw = get_text(&child, source);
                let kind = if is_constant_name(&ident_name) {
                    NodeType::Constant
                } else {
                    NodeType::Variable
                };

                entities.push(ParsedEntity {
                    kind,
                    name: ident_name,
                    qualified_name: qualified.clone(),
                    start_line: child.start_position().row,
                    end_line: child.end_position().row,
                    signature: super::truncate_str(raw, super::RAW_TEXT_SMALL_LIMIT).to_string(),
                    raw_text: super::truncate_str(raw, super::RAW_TEXT_SMALL_LIMIT).to_string(),
                    parent_qualified_name: Some(scope_stack.join(".")),
                    docstring: String::new(),
                    bases: Vec::new(),
                    imports: Vec::new(),
                    cyclomatic_complexity: None,
                });

                // CONTAINS edge from parent scope — connects the
                // variable/constant to its module so it appears in
                // graph traversals. Without this, these nodes are
                // orphans with zero edges (audit issue #5).
                edges.push(ParsedEdge::new(
                    scope_stack.join("."),
                    qualified,
                    "contains",
                ));
            }

            if let Some((iife_body, iife_aliases)) = handled_iife {
                self.walk_iife_body(
                    iife_body,
                    source,
                    module_name,
                    entities,
                    edges,
                    scope_stack,
                    global_aliases,
                    &iife_aliases,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_assignment(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
        global_aliases: &HashSet<String>,
    ) {
        let Some(left) = node.child_by_field_name("left") else {
            return;
        };
        let Some(right) = node.child_by_field_name("right") else {
            return;
        };

        match commonjs_export_target(left, source) {
            Some(Some(prop_name)) => {
                if matches!(
                    right.kind(),
                    "arrow_function" | "function_expression" | "generator_function"
                ) {
                    let doc_node = node.parent().unwrap_or(node);
                    self.handle_function_value(
                        prop_name,
                        right,
                        doc_node,
                        source,
                        module_name,
                        entities,
                        edges,
                        scope_stack,
                        global_aliases,
                    );
                    return;
                }

                if prop_name.len() >= 2 {
                    let qualified = format!("{}.{prop_name}", scope_stack.join("."));
                    let raw = get_text(&node, source);
                    let kind = if is_constant_name(&prop_name) {
                        NodeType::Constant
                    } else {
                        NodeType::Variable
                    };
                    entities.push(ParsedEntity {
                        kind,
                        name: prop_name,
                        qualified_name: qualified.clone(),
                        start_line: node.start_position().row,
                        end_line: node.end_position().row,
                        signature: super::truncate_str(raw, super::RAW_TEXT_SMALL_LIMIT)
                            .to_string(),
                        raw_text: super::truncate_str(raw, super::RAW_TEXT_SMALL_LIMIT)
                            .to_string(),
                        parent_qualified_name: Some(scope_stack.join(".")),
                        docstring: String::new(),
                        bases: Vec::new(),
                        imports: Vec::new(),
                        cyclomatic_complexity: None,
                    });
                    edges.push(ParsedEdge::new(
                        scope_stack.join("."),
                        qualified,
                        "contains",
                    ));
                }

                self.walk_node(
                    right,
                    source,
                    module_name,
                    entities,
                    edges,
                    scope_stack,
                    global_aliases,
                );
            }
            Some(None) => {
                if right.kind() == "object" {
                    let mut cursor = right.walk();
                    for item in right.children(&mut cursor) {
                        if item.kind() == "pair" {
                            let key_name = item
                                .child_by_field_name("key")
                                .filter(|k| {
                                    k.kind() == "property_identifier" || k.kind() == "identifier"
                                })
                                .map(|k| get_text(&k, source).to_string())
                                .unwrap_or_default();
                            if let Some(value) = item.child_by_field_name("value") {
                                if !key_name.is_empty()
                                    && matches!(
                                        value.kind(),
                                        "arrow_function"
                                            | "function_expression"
                                            | "generator_function"
                                    )
                                {
                                    self.handle_function_value(
                                        key_name,
                                        value,
                                        item,
                                        source,
                                        module_name,
                                        entities,
                                        edges,
                                        scope_stack,
                                        global_aliases,
                                    );
                                    continue;
                                }
                            }
                        }
                        self.walk_node(
                            item,
                            source,
                            module_name,
                            entities,
                            edges,
                            scope_stack,
                            global_aliases,
                        );
                    }
                } else if matches!(right.kind(), "function_expression" | "generator_function") {
                    let fn_name = right
                        .child_by_field_name("name")
                        .filter(|n| n.kind() == "identifier")
                        .map(|n| get_text(&n, source).to_string())
                        .unwrap_or_default();
                    if !fn_name.is_empty() {
                        let doc_node = node.parent().unwrap_or(node);
                        self.handle_function_value(
                            fn_name,
                            right,
                            doc_node,
                            source,
                            module_name,
                            entities,
                            edges,
                            scope_stack,
                            global_aliases,
                        );
                    } else if let Some(body) = right.child_by_field_name("body") {
                        self.extract_module_imports_in_subtree(
                            body,
                            source,
                            module_name,
                            entities,
                            edges,
                        );
                    }
                } else {
                    self.walk_node(
                        right,
                        source,
                        module_name,
                        entities,
                        edges,
                        scope_stack,
                        global_aliases,
                    );
                }
            }
            None => {
                extract_ts_http_and_bridges(
                    &node,
                    source,
                    module_name,
                    edges,
                    global_aliases,
                );
                if left.kind() == "member_expression"
                    && matches!(
                        right.kind(),
                        "arrow_function" | "function_expression" | "generator_function"
                    )
                {
                    if let (Some(obj), Some(prop)) = (
                        left.child_by_field_name("object"),
                        left.child_by_field_name("property"),
                    ) {
                        let prop_name = get_text(&prop, source).trim();
                        if is_global_root_node(obj, source, global_aliases)
                            && is_custom_global_prop(prop_name)
                        {
                            let doc_node = node.parent().unwrap_or(node);
                            self.handle_function_value(
                                prop_name.to_string(),
                                right,
                                doc_node,
                                source,
                                module_name,
                                entities,
                                edges,
                                scope_stack,
                                global_aliases,
                            );
                            return;
                        }
                    }
                }
                self.walk_node(
                    right,
                    source,
                    module_name,
                    entities,
                    edges,
                    scope_stack,
                    global_aliases,
                );
            }
        }
    }
}

impl LanguageParser for TypeScriptParser {
    fn parse(&self, source: &str, file_path: &str) -> (Vec<ParsedEntity>, Vec<ParsedEdge>) {
        let source_bytes = source.as_bytes();
        let mut parser = self.create_parser();

        let tree = match parser.parse(source_bytes, None) {
            Some(t) => t,
            None => return (Vec::new(), Vec::new()),
        };

        let mut entities = Vec::new();
        let mut edges = Vec::new();

        // Module name: strip extension (case-insensitive) and replace slashes with dots.
        let mut module_name = file_path.replace('/', ".");
        let lower_name = module_name.to_ascii_lowercase();
        for suffix in JS_TS_EXTENSIONS {
            if lower_name.ends_with(suffix) {
                module_name.truncate(module_name.len() - suffix.len());
                break;
            }
        }

        entities.push(ParsedEntity {
            kind: NodeType::File,
            name: file_path.to_string(),
            qualified_name: module_name.clone(),
            start_line: 0,
            end_line: source.lines().count(),
            raw_text: format!("file: {file_path}"),
            signature: String::new(),
            docstring: String::new(),
            parent_qualified_name: None,
            bases: Vec::new(),
            imports: Vec::new(),
            cyclomatic_complexity: None,
        });

        let scope_stack = vec![module_name.clone()];
        let global_aliases = HashSet::new();
        self.walk(
            tree.root_node(),
            source_bytes,
            &module_name,
            &mut entities,
            &mut edges,
            &scope_stack,
            &global_aliases,
        );

        (entities, edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_typescript_class() {
        let parser = TypeScriptParser::new("typescript");
        let source = r#"
class UserService {
    private db: Database;

    async getUser(id: string): Promise<User> {
        return this.db.find(id);
    }
}
"#;
        let (entities, edges) = parser.parse(source, "services/user.ts");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"UserService"));
        assert!(!edges.is_empty());
    }

    #[test]
    fn parses_interface() {
        let parser = TypeScriptParser::new("typescript");
        let source = "interface Config {\n  port: number;\n  host: string;\n}\n";
        let (entities, _) = parser.parse(source, "config.ts");

        let iface = entities.iter().find(|e| e.name == "Config").unwrap();
        // Interfaces are mapped to Class type.
        assert_eq!(iface.kind, NodeType::Class);
    }

    #[test]
    fn cyclomatic_complexity() {
        let parser = TypeScriptParser::new("typescript");

        // Simple function — no branches → complexity 1.
        let source_simple =
            "function greet(name: string): string {\n  return `Hello, ${name}`;\n}\n";
        let (entities, _) = parser.parse(source_simple, "greet.ts");
        let func = entities.iter().find(|e| e.name == "greet").unwrap();
        assert_eq!(func.kind, NodeType::Function);
        assert_eq!(func.cyclomatic_complexity, Some(1));

        // Branching function — if + for → complexity 3.
        let source_branch = r#"
function process(items: number[]): void {
    for (const item of items) {
        if (item > 0) {
            console.log(item);
        }
    }
}
"#;
        let (entities2, _) = parser.parse(source_branch, "process.ts");
        let func2 = entities2.iter().find(|e| e.name == "process").unwrap();
        assert_eq!(func2.kind, NodeType::Function);
        // 1 base + for_in_statement + if_statement = 3
        assert_eq!(func2.cyclomatic_complexity, Some(3));

        // Interface (class-like) should not have cyclomatic complexity.
        let source_iface = "interface Config {\n  port: number;\n}\n";
        let (entities3, _) = parser.parse(source_iface, "config.ts");
        let iface = entities3.iter().find(|e| e.name == "Config").unwrap();
        assert_eq!(iface.kind, NodeType::Class);
        assert_eq!(iface.cyclomatic_complexity, None);
    }

    #[test]
    fn parses_imports() {
        let parser = TypeScriptParser::new("typescript");
        let source = "import { Router } from 'express';\n";
        let (entities, _) = parser.parse(source, "app.ts");

        let imports: Vec<&ParsedEntity> = entities
            .iter()
            .filter(|e| e.kind == NodeType::Import)
            .collect();
        assert!(!imports.is_empty());
    }

    #[test]
    fn jsdoc_comments_extracted() {
        let parser = TypeScriptParser::new("typescript");
        let source = "/** Creates a thing. */\nfunction create() {}\n";
        let (entities, _) = parser.parse(source, "factory.ts");
        let func = entities.iter().find(|e| e.name == "create").unwrap();
        assert!(
            func.docstring.contains("Creates a thing"),
            "Expected docstring to contain 'Creates a thing', got: {:?}",
            func.docstring
        );
    }

    #[test]
    fn jsdoc_on_class_extracted() {
        let parser = TypeScriptParser::new("typescript");
        let source = "/** User service. */\nclass UserService {}\n";
        let (entities, _) = parser.parse(source, "service.ts");
        let cls = entities.iter().find(|e| e.name == "UserService").unwrap();
        assert!(
            cls.docstring.contains("User service"),
            "Expected docstring to contain 'User service', got: {:?}",
            cls.docstring
        );
    }

    #[test]
    fn calls_edges_simple_function() {
        let parser = TypeScriptParser::new("typescript");
        let source = r#"
function handler() {
    validate();
    const result = transform(data);
}
"#;
        let (_, edges) = parser.parse(source, "handler.ts");
        let calls: Vec<&ParsedEdge> = edges.iter().filter(|e| e.relationship == "calls").collect();
        assert!(calls.iter().any(|e| e.target_qualified_name == "validate"));
        assert!(calls.iter().any(|e| e.target_qualified_name == "transform"));
    }

    #[test]
    fn calls_edges_method_call() {
        let parser = TypeScriptParser::new("typescript");
        let source = r#"
class Service {
    process() {
        this.helper();
        externalFn();
    }
}
"#;
        let (_, edges) = parser.parse(source, "service.ts");
        let calls: Vec<&ParsedEdge> = edges.iter().filter(|e| e.relationship == "calls").collect();
        assert!(calls.iter().any(|e| e.target_qualified_name == "helper"));
        assert!(calls
            .iter()
            .any(|e| e.target_qualified_name == "externalFn"));
    }

    #[test]
    fn calls_edges_skip_builtins() {
        let parser = TypeScriptParser::new("typescript");
        let source = "function f() { parseInt('42'); }\n";
        let (_, edges) = parser.parse(source, "num.ts");
        let calls: Vec<&ParsedEdge> = edges.iter().filter(|e| e.relationship == "calls").collect();
        assert!(
            calls.is_empty(),
            "Built-in calls should be filtered, got: {:?}",
            calls
                .iter()
                .map(|e| &e.target_qualified_name)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn calls_edges_deduplicated() {
        let parser = TypeScriptParser::new("typescript");
        let source = "function f() { foo(); foo(); foo(); }\n";
        let (_, edges) = parser.parse(source, "dedup.ts");
        let foo_calls: Vec<&ParsedEdge> = edges
            .iter()
            .filter(|e| e.relationship == "calls" && e.target_qualified_name == "foo")
            .collect();
        assert_eq!(foo_calls.len(), 1, "Duplicate calls should be deduplicated");
    }

    #[test]
    fn module_name_strips_all_js_ts_variant_extensions_case_insensitively() {
        let parser = TypeScriptParser::new("typescript");
        let tsx_parser = TypeScriptParser::new("tsx");
        let cases = [
            ("src/config.cjs", "src.config", &parser),
            ("src/entry.mjs", "src.entry", &parser),
            ("src/types.cts", "src.types", &parser),
            ("src/mod.mts", "src.mod", &parser),
            ("src/view.mjsx", "src.view", &tsx_parser),
            ("src/legacy.es6", "src.legacy", &parser),
            ("src/module.jsm", "src.module", &parser),
            ("src/UPPER.CJS", "src.UPPER", &parser),
            ("types/global.d.cts", "types.global.d", &parser),
        ];

        for (file_path, expected_module, active_parser) in cases {
            let (entities, _) = active_parser.parse("const VALUE = 1;\n", file_path);
            let file_entity = entities
                .iter()
                .find(|e| e.kind == NodeType::File)
                .expect("file entity");
            assert_eq!(
                file_entity.qualified_name, expected_module,
                "for {file_path}"
            );
        }
    }

    #[test]
    fn parses_commonjs_requires_dynamic_imports_reexports_and_ts_require() {
        let parser = TypeScriptParser::new("typescript");
        let source = r#"#!/usr/bin/env node
import type { Config } from './types.mts';
import legacy = require('./legacy.cts');
const direct = require('./direct.cjs');
const { alpha, beta } = require('./destructured.cjs');
const prop = require('./with-prop.cjs').prop;
require('./polyfill.cjs');
module.exports = require('./base.cjs');
const cfg = defineConfig({ plugins: [require('./plugin.cjs')] });
export { helper } from './reexport.mjs';
export * from './star.mjs';
function loadLazy() {
    const lazy = require('./lazy.cjs');
    return import('./dynamic.mjs');
}
"#;
        let (entities, edges) = parser.parse(source, "scripts/build.cjs");
        let imports: Vec<&ParsedEntity> = entities
            .iter()
            .filter(|e| e.kind == NodeType::Import)
            .collect();

        let expected = [
            "./types.mts",
            "./legacy.cts",
            "./direct.cjs",
            "./destructured.cjs",
            "./with-prop.cjs",
            "./polyfill.cjs",
            "./base.cjs",
            "./plugin.cjs",
            "./reexport.mjs",
            "./star.mjs",
            "./lazy.cjs",
            "./dynamic.mjs",
        ];

        for target in expected {
            let count = imports.iter().filter(|e| e.name == target).count();
            assert_eq!(
                count, 1,
                "expected exactly 1 Import entity for {target}, got {count}"
            );
            assert!(
                edges
                    .iter()
                    .any(|e| e.relationship == "imports"
                        && e.source_qualified_name == "scripts.build"
                        && e.target_qualified_name == target),
                "missing imports edge for {target}"
            );
        }
    }

    #[test]
    fn parses_commonjs_exports_arrow_functions_and_jsdoc() {
        let parser = TypeScriptParser::new("typescript");
        let source = r#"
/** Arrow helper doc. */
const arrowFn = (x) => runArrow(x);
const exprFn = function namedExpr(a, b) {
    return combineExpr(a, b);
};
/** Exported CJS handler. */
exports.handle = (req) => dispatch(req);
module.exports.check = function(val) {
    return verifyCheck(val);
};
exports.DEFAULT_PORT = 8080;
module.exports = {
    /** Object arrow method. */
    build: (cfg) => assemble(cfg),
    init() {
        bootstrap();
    },
    plugins: [require('./extra.cjs')],
};
"#;
        let (entities, edges) = parser.parse(source, "config/app.cjs");

        let arrow = entities.iter().find(|e| e.name == "arrowFn").unwrap();
        assert_eq!(arrow.kind, NodeType::Function);
        assert!(arrow.docstring.contains("Arrow helper doc"));
        assert!(edges.iter().any(|e| e.relationship == "calls"
            && e.source_qualified_name == "config.app.arrowFn"
            && e.target_qualified_name == "runArrow"));

        let expr = entities.iter().find(|e| e.name == "exprFn").unwrap();
        assert_eq!(expr.kind, NodeType::Function);
        assert!(edges.iter().any(|e| e.relationship == "calls"
            && e.source_qualified_name == "config.app.exprFn"
            && e.target_qualified_name == "combineExpr"));

        let handle = entities.iter().find(|e| e.name == "handle").unwrap();
        assert_eq!(handle.kind, NodeType::Function);
        assert!(handle.docstring.contains("Exported CJS handler"));
        assert!(edges.iter().any(|e| e.relationship == "calls"
            && e.source_qualified_name == "config.app.handle"
            && e.target_qualified_name == "dispatch"));

        let check = entities.iter().find(|e| e.name == "check").unwrap();
        assert_eq!(check.kind, NodeType::Function);
        assert!(edges.iter().any(|e| e.relationship == "calls"
            && e.source_qualified_name == "config.app.check"
            && e.target_qualified_name == "verifyCheck"));

        let port = entities.iter().find(|e| e.name == "DEFAULT_PORT").unwrap();
        assert_eq!(port.kind, NodeType::Constant);

        let build = entities.iter().find(|e| e.name == "build").unwrap();
        assert_eq!(build.kind, NodeType::Function);
        assert!(build.docstring.contains("Object arrow method"));
        assert!(edges.iter().any(|e| e.relationship == "calls"
            && e.source_qualified_name == "config.app.build"
            && e.target_qualified_name == "assemble"));

        let init = entities.iter().find(|e| e.name == "init").unwrap();
        assert_eq!(init.kind, NodeType::Method);
        assert!(edges.iter().any(|e| e.relationship == "calls"
            && e.source_qualified_name == "config.app.init"
            && e.target_qualified_name == "bootstrap"));

        assert!(entities
            .iter()
            .any(|e| e.kind == NodeType::Import && e.name == "./extra.cjs"));
    }

    #[test]
    fn extracts_call_sites_imports_jsx_heritage_types_signatures_and_bridges() {
        let tsx = TypeScriptParser::new("tsx");
        let source = r#"
import React, { useState, Button as UIButton } from './ui/Button';
import * as Icons from './icons';
const { readAppFunctions } = require('./browser_source_commands.cjs');
export { Panel, Helper as PublicHelper } from './panel';
export * from './star';

interface BaseProps {}
interface CustomProps extends BaseProps {
    item: DomainItem;
}

class CustomButton extends BaseControl implements Renderable {
    render(): JSXNode {
        return <UIButton />;
    }
}

export function Page(props: CustomProps): ScreenResult {
    readAppFunctions();
    Icons.draw();
    readAppFunctions();
    window.__psLive.sync();
    commands.readAdapterBundle();
    fetch('/api/items/42?full=1', { method: 'POST' });
    return <CustomButton />;
}

test('top level spec block', () => {
    Page({ item: {} as DomainItem });
});
"#;
        let (entities, edges) = tsx.parse(source, "src/Page.tsx");

        // Rich signature includes return type
        let page = entities.iter().find(|e| e.name == "Page").unwrap();
        assert_eq!(page.signature, "(props: CustomProps): ScreenResult");

        // Named import metadata
        let ui_import = edges
            .iter()
            .find(|e| e.relationship == "imports" && e.target_qualified_name == "./ui/Button")
            .unwrap();
        assert_eq!(
            ui_import.metadata.get("reference_kind"),
            Some(&json!("imports_named"))
        );
        assert_eq!(
            ui_import.metadata.get("imported_symbols"),
            Some(&json!(["default", "useState", "Button"]))
        );

        // CJS destructured require metadata
        let cjs_import = edges
            .iter()
            .find(|e| {
                e.relationship == "imports"
                    && e.target_qualified_name == "./browser_source_commands.cjs"
            })
            .unwrap();
        assert_eq!(
            cjs_import.metadata.get("imported_symbols"),
            Some(&json!(["readAppFunctions"]))
        );
        assert_eq!(cjs_import.metadata.get("wildcard_import"), Some(&json!(false)));

        // Re-export metadata
        let panel_reexport = edges
            .iter()
            .find(|e| e.relationship == "imports" && e.target_qualified_name == "./panel")
            .unwrap();
        assert_eq!(panel_reexport.metadata.get("is_reexport"), Some(&json!(true)));
        let star_reexport = edges
            .iter()
            .find(|e| e.relationship == "imports" && e.target_qualified_name == "./star")
            .unwrap();
        assert_eq!(star_reexport.metadata.get("wildcard_import"), Some(&json!(true)));

        // Heritage extends vs implements
        assert!(edges.iter().any(|e| e.relationship == "inherits"
            && e.target_qualified_name == "BaseControl"
            && e.metadata.get("reference_kind") == Some(&json!("extends"))));
        assert!(edges.iter().any(|e| e.relationship == "implements"
            && e.target_qualified_name == "Renderable"
            && e.metadata.get("reference_kind") == Some(&json!("implements"))));

        // Multiple call sites with line numbers
        let multi_call = edges
            .iter()
            .find(|e| {
                e.relationship == "calls"
                    && e.source_qualified_name == "src.Page.Page"
                    && e.target_qualified_name == "readAppFunctions"
            })
            .unwrap();
        assert_eq!(multi_call.metadata.get("start_line"), Some(&json!(19)));
        assert_eq!(
            multi_call
                .metadata
                .get("sites")
                .and_then(|v| v.as_array())
                .map(|a| a.len()),
            Some(2)
        );

        // Namespace-qualified call retains qualifier
        let ns_call = edges
            .iter()
            .find(|e| e.relationship == "calls" && e.target_qualified_name == "draw")
            .unwrap();
        assert_eq!(ns_call.metadata.get("qualifier"), Some(&json!("Icons")));

        // JSX renders
        assert!(edges.iter().any(|e| e.relationship == "references"
            && e.source_qualified_name == "src.Page.Page"
            && e.target_qualified_name == "CustomButton"
            && e.metadata.get("reference_kind") == Some(&json!("renders"))));

        // TypeScript type references
        assert!(edges.iter().any(|e| e.relationship == "references"
            && e.source_qualified_name == "src.Page.Page"
            && e.target_qualified_name == "CustomProps"
            && e.metadata.get("reference_kind") == Some(&json!("type_references"))));
        assert!(edges.iter().any(|e| e.relationship == "references"
            && e.source_qualified_name == "src.Page.CustomProps"
            && e.target_qualified_name == "DomainItem"
            && e.metadata.get("reference_kind") == Some(&json!("type_references"))));

        // Global bridge, commands bridge, and HTTP client descriptors
        assert!(edges.iter().any(|e| e.relationship == "references"
            && e.metadata.get("reference_kind") == Some(&json!("bridge_caller"))
            && e.metadata.get("bridge_namespace") == Some(&json!("window.__psLive"))
            && e.metadata.get("bridge_member") == Some(&json!("sync"))));
        assert!(edges.iter().any(|e| e.relationship == "references"
            && e.metadata.get("reference_kind") == Some(&json!("bridge_caller"))
            && e.metadata.get("bridge_namespace") == Some(&json!("commands"))
            && e.metadata.get("bridge_member") == Some(&json!("readAdapterBundle"))));
        assert!(edges.iter().any(|e| e.relationship == "references"
            && e.metadata.get("reference_kind") == Some(&json!("http_route_client"))
            && e.metadata.get("http_method") == Some(&json!("POST"))
            && e.metadata.get("norm_path") == Some(&json!("/api/items/42"))));

        // Top-level callback (test/describe) captures calls
        assert!(edges.iter().any(|e| e.relationship == "calls"
            && e.source_qualified_name == "src.Page"
            && e.target_qualified_name == "Page"));
    }

    #[test]
    fn extracts_express_routes_and_bridge_providers() {
        let parser = TypeScriptParser::new("typescript");
        let source = r#"
function createUser(req, res) {}
function syncNow() {}
router.post('/api/users/:id', createUser);
router.get('/api/users', (req, res) => { return null; });
window.__psLive.sync = createUser;
(window as any).__live = { syncNow };
(window as any).__live.syncNow();
export const config = {
    commands: require('./browser_source_commands.cjs'),
};
export default {
    browser: {
        commands: require('./commands.cjs'),
    },
};
"#;
        let (_, edges) = parser.parse(source, "server/routes.ts");

        let post_route = edges
            .iter()
            .find(|e| {
                e.metadata.get("reference_kind") == Some(&json!("http_route_handler"))
                    && e.metadata.get("http_method") == Some(&json!("POST"))
            })
            .unwrap();
        assert_eq!(
            post_route.metadata.get("norm_path"),
            Some(&json!("/api/users/:param"))
        );
        assert_eq!(
            post_route.metadata.get("handler_symbol"),
            Some(&json!("createUser"))
        );

        assert!(edges.iter().any(|e| {
            e.metadata.get("reference_kind") == Some(&json!("bridge_provider"))
                && e.metadata.get("bridge_namespace") == Some(&json!("window.__psLive"))
                && e.metadata.get("bridge_member") == Some(&json!("sync"))
        }));
        assert!(edges.iter().any(|e| {
            e.metadata.get("reference_kind") == Some(&json!("bridge_provider"))
                && e.metadata.get("bridge_namespace") == Some(&json!("window.__live"))
                && e.metadata.get("bridge_member") == Some(&json!("syncNow"))
        }));
        assert!(edges.iter().any(|e| {
            e.metadata.get("reference_kind") == Some(&json!("bridge_caller"))
                && e.metadata.get("bridge_namespace") == Some(&json!("window.__live"))
                && e.metadata.get("bridge_member") == Some(&json!("syncNow"))
        }));
        assert!(edges.iter().any(|e| {
            e.metadata.get("reference_kind") == Some(&json!("bridge_provider"))
                && e.metadata.get("bridge_namespace") == Some(&json!("commands"))
                && e.metadata.get("bridge_member") == Some(&json!("*"))
                && e.metadata.get("target_specifier") == Some(&json!("./browser_source_commands.cjs"))
        }));
        assert!(edges.iter().any(|e| {
            e.metadata.get("reference_kind") == Some(&json!("bridge_provider"))
                && e.metadata.get("bridge_namespace") == Some(&json!("commands"))
                && e.metadata.get("bridge_member") == Some(&json!("*"))
                && e.metadata.get("target_specifier") == Some(&json!("./commands.cjs"))
        }));
    }

    #[test]
    fn extracts_iife_symbols_two_segment_globals_and_url_asset_imports() {
        let parser = TypeScriptParser::new("javascript");
        let source = r#"
(function (global) {
    function initPanel(id) {
        return id;
    }
    const renderRow = (row) => row;
    global.__psVendor = { initPanel, renderRow };
    window.__psDoor = initPanel;
    window.__psOpen = function (flag) { return flag; };
    Object.assign(window, { __psApi: initPanel });
})(window);

!function () {
    function helper() {}
}();

const PanelMod = (function () {
    function mountPanel() {}
    return { mountPanel };
})();

const doorRef = window.__psDoor;
window.__psOpen(true);
const legacyUrl = new URL('./legacy.js', import.meta.url);
window.location.reload();
"#;
        let (entities, edges) = parser.parse(source, "panels/vendor-invoices-panel.js");
        let fn_names: Vec<&str> = entities
            .iter()
            .filter(|e| e.kind == NodeType::Function)
            .map(|e| e.name.as_str())
            .collect();
        assert!(fn_names.contains(&"initPanel"), "missing initPanel: {fn_names:?}");
        assert!(fn_names.contains(&"renderRow"), "missing renderRow: {fn_names:?}");
        assert!(fn_names.contains(&"__psOpen"), "missing __psOpen: {fn_names:?}");
        assert!(fn_names.contains(&"helper"), "missing helper: {fn_names:?}");
        assert!(fn_names.contains(&"mountPanel"), "missing mountPanel: {fn_names:?}");
        assert!(entities.iter().any(|e| e.name == "PanelMod"));

        // 2-segment & 3-segment providers (global alias + Object.assign + window function)
        assert!(edges.iter().any(|e| {
            e.metadata.get("reference_kind") == Some(&json!("bridge_provider"))
                && e.metadata.get("bridge_namespace") == Some(&json!("window"))
                && e.metadata.get("bridge_member") == Some(&json!("__psVendor"))
        }));
        assert!(edges.iter().any(|e| {
            e.metadata.get("reference_kind") == Some(&json!("bridge_provider"))
                && e.metadata.get("bridge_namespace") == Some(&json!("window.__psVendor"))
                && e.metadata.get("bridge_member") == Some(&json!("initPanel"))
                && e.metadata.get("handler_symbol") == Some(&json!("initPanel"))
        }));
        assert!(edges.iter().any(|e| {
            e.metadata.get("reference_kind") == Some(&json!("bridge_provider"))
                && e.metadata.get("bridge_namespace") == Some(&json!("window"))
                && e.metadata.get("bridge_member") == Some(&json!("__psDoor"))
                && e.metadata.get("handler_symbol") == Some(&json!("initPanel"))
        }));
        assert!(edges.iter().any(|e| {
            e.metadata.get("reference_kind") == Some(&json!("bridge_provider"))
                && e.metadata.get("bridge_namespace") == Some(&json!("window"))
                && e.metadata.get("bridge_member") == Some(&json!("__psOpen"))
                && e.metadata.get("handler_symbol") == Some(&json!("__psOpen"))
        }));
        assert!(edges.iter().any(|e| {
            e.metadata.get("reference_kind") == Some(&json!("bridge_provider"))
                && e.metadata.get("bridge_namespace") == Some(&json!("window"))
                && e.metadata.get("bridge_member") == Some(&json!("__psApi"))
        }));

        // 2-segment property read and direct call
        assert!(edges.iter().any(|e| {
            e.metadata.get("reference_kind") == Some(&json!("bridge_caller"))
                && e.metadata.get("bridge_namespace") == Some(&json!("window"))
                && e.metadata.get("bridge_member") == Some(&json!("__psDoor"))
        }));
        assert!(edges.iter().any(|e| {
            e.metadata.get("reference_kind") == Some(&json!("bridge_caller"))
                && e.metadata.get("bridge_namespace") == Some(&json!("window"))
                && e.metadata.get("bridge_member") == Some(&json!("__psOpen"))
        }));

        // Built-in window.location is filtered out
        assert!(!edges.iter().any(|e| {
            e.metadata.get("bridge_namespace") == Some(&json!("window.location"))
                || e.metadata.get("bridge_member") == Some(&json!("location"))
        }));

        // new URL('./legacy.js', import.meta.url) asset import
        assert!(edges.iter().any(|e| {
            e.relationship == "imports"
                && e.target_qualified_name == "./legacy.js"
                && e.metadata.get("reference_kind") == Some(&json!("imports_asset"))
        }));
    }

    #[test]
    fn extracts_ts_receiver_types_from_params_constructors_factories_and_class_fields() {
        let parser = TypeScriptParser::new("typescript");
        let source = r#"
import { ApiClient } from "./api";

function makeClient(): ApiClient {
  return new ApiClient();
}

export async function loadOnce(client: ApiClient): Promise<void> {
  await client.fetchSessions();
}

export class SessionController {
  constructor(private readonly client: ApiClient) {}

  async refresh(): Promise<void> {
    await this.client.fetchSessions();
    const local = makeClient();
    await local.close();
    const direct = new ApiClient();
    await direct.ping();
  }
}
"#;
        let (_, edges) = parser.parse(source, "src/sessions.ts");

        assert!(edges.iter().any(|e| {
            e.relationship == "calls"
                && e.source_qualified_name.ends_with(".loadOnce")
                && e.target_qualified_name == "fetchSessions"
                && e.metadata.get("qualifier") == Some(&json!("client"))
                && e.metadata.get("receiver_type") == Some(&json!("ApiClient"))
        }));
        assert!(edges.iter().any(|e| {
            e.relationship == "calls"
                && e.source_qualified_name.ends_with(".SessionController.refresh")
                && e.target_qualified_name == "fetchSessions"
                && e.metadata.get("qualifier") == Some(&json!("this.client"))
                && e.metadata.get("receiver_type") == Some(&json!("ApiClient"))
        }));
        assert!(edges.iter().any(|e| {
            e.relationship == "calls"
                && e.source_qualified_name.ends_with(".SessionController.refresh")
                && e.target_qualified_name == "close"
                && e.metadata.get("qualifier") == Some(&json!("local"))
                && e.metadata.get("receiver_type") == Some(&json!("ApiClient"))
        }));
        assert!(edges.iter().any(|e| {
            e.relationship == "calls"
                && e.source_qualified_name.ends_with(".SessionController.refresh")
                && e.target_qualified_name == "ping"
                && e.metadata.get("qualifier") == Some(&json!("direct"))
                && e.metadata.get("receiver_type") == Some(&json!("ApiClient"))
        }));
    }
}
