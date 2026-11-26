//! Python AST extraction via tree-sitter.
//!
//! Extracts classes, functions, methods, constants, imports, and type
//! aliases from Python source files. Uses a scope stack to build
//! fully qualified names (e.g. "MyModule.MyClass.my_method").

use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};
use tree_sitter::{Node, Parser};

use scs_core::node_types::NodeType;

use super::{
    build_call_edges, count_complexity, extract_call_sites, normalize_http_path, CallSite,
    LanguageParser, ParsedEdge, ParsedEntity,
};

/// Built-in function names to exclude from call-graph edges.
/// These are stdlib/builtin calls that would never match a project entity.
const PYTHON_BUILTIN_CALLS: &[&str] = &[
    "print",
    "len",
    "range",
    "str",
    "int",
    "float",
    "bool",
    "list",
    "dict",
    "set",
    "tuple",
    "type",
    "isinstance",
    "issubclass",
    "super",
    "getattr",
    "setattr",
    "hasattr",
    "delattr",
    "enumerate",
    "zip",
    "map",
    "filter",
    "sorted",
    "reversed",
    "any",
    "all",
    "min",
    "max",
    "sum",
    "abs",
    "round",
    "open",
    "format",
    "repr",
    "id",
    "hash",
    "iter",
    "next",
    "input",
    "vars",
    "dir",
    "callable",
    "classmethod",
    "staticmethod",
    "property",
    "object",
    "bytes",
    "bytearray",
    "memoryview",
    "frozenset",
    "complex",
    "hex",
    "oct",
    "bin",
    "ord",
    "chr",
    "ascii",
    "exec",
    "eval",
    "compile",
    "globals",
    "locals",
    "breakpoint",
    "exit",
    "quit",
    "help",
    "__import__",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "RuntimeError",
    "Exception",
    "NotImplementedError",
    "StopIteration",
    "OSError",
    "IOError",
];

/// Pattern to detect UPPER_CASE constants (Python convention).
fn is_constant_name(name: &str) -> bool {
    !name.is_empty()
        && name.chars().next().unwrap().is_ascii_uppercase()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// Extract the text content of a tree-sitter node.
fn get_text<'a>(node: &Node, source: &'a [u8]) -> &'a str {
    let bytes = &source[node.start_byte()..node.end_byte()];
    std::str::from_utf8(bytes).unwrap_or("")
}

/// Extract a docstring from the first statement in a function/class body.
///
/// Python docstrings are string literals appearing as the first statement
/// in a class or function body. Tree-sitter represents these as
/// `expression_statement` → `string` nodes.
fn get_docstring(body_node: Option<Node>, source: &[u8]) -> String {
    let body = match body_node {
        Some(b) => b,
        None => return String::new(),
    };

    let mut cursor = body.walk();
    for child in body.children(&mut cursor) {
        if child.kind() == "expression_statement" {
            let mut sub_cursor = child.walk();
            for sub in child.children(&mut sub_cursor) {
                if sub.kind() == "string" {
                    let text = get_text(&sub, source);
                    // Strip triple-quote delimiters.
                    for delim in ["\"\"\"", "'''"] {
                        if text.starts_with(delim)
                            && text.ends_with(delim)
                            && text.len() >= delim.len() * 2
                        {
                            return text[delim.len()..text.len() - delim.len()]
                                .trim()
                                .to_string();
                        }
                    }
                    return text
                        .trim_matches(|c| c == '"' || c == '\'')
                        .trim()
                        .to_string();
                }
            }
            break;
        }
        // Skip comments and pass statements.
        if child.kind() != "comment" && child.kind() != "pass_statement" {
            break;
        }
    }
    String::new()
}

/// Extract the parameter list from a function definition.
fn get_parameters(node: &Node, source: &[u8]) -> String {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "parameters" {
            return get_text(&child, source).to_string();
        }
    }
    "()".to_string()
}

/// Extract the return type annotation from a function definition.
fn get_return_type(node: &Node, source: &[u8]) -> String {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "type" {
            return get_text(&child, source).to_string();
        }
    }
    String::new()
}

fn function_name_of(node: &Node, source: &[u8]) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" {
            let name = get_text(&child, source);
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    None
}

fn python_string_literal(node: &Node, source: &[u8]) -> Option<String> {
    if node.kind() != "string" {
        return None;
    }
    let raw = get_text(node, source).trim();
    if raw.starts_with("f\"") || raw.starts_with("f'") {
        return None;
    }
    for delim in ["\"\"\"", "'''", "\"", "'"] {
        if raw.starts_with(delim) && raw.ends_with(delim) && raw.len() >= delim.len() * 2 {
            let inner = &raw[delim.len()..raw.len() - delim.len()];
            if !inner.is_empty() {
                return Some(inner.to_string());
            }
            return None;
        }
    }
    None
}

fn is_http_method_name(method: &str) -> bool {
    matches!(
        method,
        "get" | "post" | "put" | "patch" | "delete" | "head" | "options" | "route" | "api_route"
    )
}

fn extract_python_route_decorators(
    decorators: &[Node],
    source: &[u8],
    fn_qname: &str,
    edges: &mut Vec<ParsedEdge>,
) {
    for dec in decorators {
        let Some(call) = (0..dec.named_child_count())
            .filter_map(|i| dec.named_child(i))
            .find(|c| c.kind() == "call")
        else {
            continue;
        };
        let Some(func) = call.child_by_field_name("function") else {
            continue;
        };
        if func.kind() != "attribute" {
            continue;
        }
        let Some(attr) = func.child_by_field_name("attribute") else {
            continue;
        };
        let method_lower = get_text(&attr, source).to_ascii_lowercase();
        if !is_http_method_name(&method_lower) {
            continue;
        }
        let Some(args) = call.child_by_field_name("arguments") else {
            continue;
        };
        let mut path_raw: Option<String> = None;
        let mut explicit_methods: Vec<String> = Vec::new();
        let mut arg_cursor = args.walk();
        for arg in args.named_children(&mut arg_cursor) {
            if path_raw.is_none() && arg.kind() == "string" {
                path_raw = python_string_literal(&arg, source);
            } else if arg.kind() == "keyword_argument" {
                let key = arg
                    .child_by_field_name("name")
                    .map(|n| get_text(&n, source));
                let val = arg.child_by_field_name("value");
                if key == Some("path") || key == Some("rule") {
                    if let Some(v) = val {
                        path_raw = python_string_literal(&v, source);
                    }
                } else if key == Some("methods") {
                    if let Some(v) = val {
                        let mut mcursor = v.walk();
                        for elem in v.named_children(&mut mcursor) {
                            if let Some(lit) = python_string_literal(&elem, source) {
                                explicit_methods.push(lit.to_ascii_uppercase());
                            }
                        }
                    }
                }
            }
        }
        let Some(raw) = path_raw else {
            continue;
        };
        let Some(norm_path) = normalize_http_path(&raw) else {
            continue;
        };
        let methods = if !explicit_methods.is_empty() {
            explicit_methods
        } else if matches!(method_lower.as_str(), "route" | "api_route") {
            vec!["GET".to_string()]
        } else {
            vec![method_lower.to_ascii_uppercase()]
        };
        for http_method in methods {
            let mut meta = HashMap::new();
            meta.insert("reference_kind".to_string(), json!("http_route_handler"));
            meta.insert("http_method".to_string(), json!(http_method));
            meta.insert("http_path".to_string(), json!(raw));
            meta.insert("norm_path".to_string(), json!(norm_path));
            meta.insert("start_line".to_string(), json!(dec.start_position().row));
            meta.insert("end_line".to_string(), json!(dec.end_position().row));
            meta.insert("enclosing_symbol".to_string(), json!(fn_qname));
            meta.insert("handler_symbol".to_string(), json!(fn_qname));
            edges.push(ParsedEdge::with_metadata(
                fn_qname.to_string(),
                format!("__scs_http_handler__:{http_method}:{norm_path}"),
                "references",
                meta,
            ));
        }
    }
}

fn extract_python_http_clients(
    body: Node,
    source: &[u8],
    enclosing_qname: &str,
    edges: &mut Vec<ParsedEdge>,
) {
    let mut cursor = body.walk();
    let mut reached_root = false;
    loop {
        let n = cursor.node();
        if n.kind() == "call" {
            if let Some(func) = n.child_by_field_name("function") {
                if func.kind() == "attribute" {
                    let obj_name = func
                        .child_by_field_name("object")
                        .map(|o| get_text(&o, source).to_ascii_lowercase())
                        .unwrap_or_default();
                    let attr_name = func
                        .child_by_field_name("attribute")
                        .map(|a| get_text(&a, source).to_ascii_lowercase())
                        .unwrap_or_default();
                    let client_like = obj_name == "requests"
                        || obj_name == "httpx"
                        || obj_name.ends_with("client")
                        || obj_name.ends_with("session")
                        || obj_name == "api";
                    if client_like
                        && matches!(
                            attr_name.as_str(),
                            "get" | "post" | "put" | "patch" | "delete" | "head" | "options" | "request"
                        )
                    {
                        if let Some(args) = n.child_by_field_name("arguments") {
                            let mut http_method = if attr_name == "request" {
                                "ANY".to_string()
                            } else {
                                attr_name.to_ascii_uppercase()
                            };
                            let mut raw_path: Option<String> = None;
                            let mut ac = args.walk();
                            let named: Vec<Node> = args.named_children(&mut ac).collect();
                            if attr_name == "request" && named.len() >= 2 {
                                if let Some(m) = python_string_literal(&named[0], source) {
                                    http_method = m.to_ascii_uppercase();
                                }
                                raw_path = python_string_literal(&named[1], source);
                            } else if let Some(first) = named.first() {
                                raw_path = python_string_literal(first, source);
                            }
                            if raw_path.is_none() {
                                for arg in &named {
                                    if arg.kind() == "keyword_argument" {
                                        let key = arg
                                            .child_by_field_name("name")
                                            .map(|k| get_text(&k, source));
                                        if key == Some("url") {
                                            if let Some(val) = arg.child_by_field_name("value") {
                                                raw_path = python_string_literal(&val, source);
                                            }
                                        }
                                    }
                                }
                            }
                            if let Some(raw) = raw_path {
                                if let Some(norm_path) = normalize_http_path(&raw) {
                                    let mut meta = HashMap::new();
                                    meta.insert(
                                        "reference_kind".to_string(),
                                        json!("http_route_client"),
                                    );
                                    meta.insert("http_method".to_string(), json!(http_method));
                                    meta.insert("http_path".to_string(), json!(raw));
                                    meta.insert("norm_path".to_string(), json!(norm_path));
                                    meta.insert(
                                        "start_line".to_string(),
                                        json!(n.start_position().row),
                                    );
                                    meta.insert(
                                        "end_line".to_string(),
                                        json!(n.end_position().row),
                                    );
                                    meta.insert(
                                        "enclosing_symbol".to_string(),
                                        json!(enclosing_qname),
                                    );
                                    edges.push(ParsedEdge::with_metadata(
                                        enclosing_qname.to_string(),
                                        format!("__scs_http_client__:{http_method}:{norm_path}"),
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
        if cursor.goto_first_child() {
            continue;
        }
        if cursor.goto_next_sibling() {
            continue;
        }
        loop {
            if !cursor.goto_parent() {
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

/// Primitive / stdlib container names excluded from receiver-type inference.
const PYTHON_IGNORED_TYPES: &[&str] = &[
    "str",
    "int",
    "float",
    "bool",
    "bytes",
    "bytearray",
    "memoryview",
    "dict",
    "list",
    "set",
    "frozenset",
    "tuple",
    "object",
    "Any",
    "None",
    "NoReturn",
    "Never",
    "Self",
    "Literal",
    "Callable",
    "Awaitable",
    "Coroutine",
    "Sequence",
    "MutableSequence",
    "Mapping",
    "MutableMapping",
    "Iterable",
    "Iterator",
    "AsyncIterable",
    "AsyncIterator",
    "Collection",
    "AbstractSet",
    "Path",
    "PurePath",
    "Exception",
    "BaseException",
    "ValueError",
    "TypeError",
    "KeyError",
    "RuntimeError",
    "OSError",
];

fn unwrap_python_wrapper<'a>(text: &'a str, wrapper: &str) -> Option<&'a str> {
    let rest = text.strip_prefix(wrapper)?.trim_start();
    let inner = rest.strip_prefix('[')?.strip_suffix(']')?;
    Some(inner.trim())
}

fn split_top_level(text: &str, sep: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0_usize;
    let mut start = 0_usize;
    for (idx, ch) in text.char_indices() {
        match ch {
            '[' | '(' | '{' => depth += 1,
            ']' | ')' | '}' => depth = depth.saturating_sub(1),
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

/// Normalize a Python type annotation into a candidate class/interface name.
fn normalize_python_type_name(raw: &str) -> Option<String> {
    let mut current = raw
        .trim()
        .trim_matches(|c| c == '"' || c == '\'')
        .trim();
    if current.is_empty() {
        return None;
    }

    for _ in 0..4 {
        if let Some(inner) = unwrap_python_wrapper(current, "Optional")
            .or_else(|| unwrap_python_wrapper(current, "Final"))
            .or_else(|| unwrap_python_wrapper(current, "ClassVar"))
            .or_else(|| unwrap_python_wrapper(current, "Awaitable"))
            .or_else(|| unwrap_python_wrapper(current, "typing.Optional"))
            .or_else(|| unwrap_python_wrapper(current, "typing.Final"))
            .or_else(|| unwrap_python_wrapper(current, "typing.ClassVar"))
        {
            current = inner.trim_matches(|c| c == '"' || c == '\'').trim();
            continue;
        }
        if let Some(inner) = unwrap_python_wrapper(current, "Annotated")
            .or_else(|| unwrap_python_wrapper(current, "typing.Annotated"))
        {
            if let Some(first) = split_top_level(inner, ',').into_iter().next() {
                current = first.trim_matches(|c| c == '"' || c == '\'').trim();
                continue;
            }
        }
        if let Some(inner) = unwrap_python_wrapper(current, "Union")
            .or_else(|| unwrap_python_wrapper(current, "typing.Union"))
        {
            let non_none: Vec<&str> = split_top_level(inner, ',')
                .into_iter()
                .filter(|p| !p.is_empty() && *p != "None")
                .collect();
            if non_none.len() == 1 {
                current = non_none[0].trim_matches(|c| c == '"' || c == '\'').trim();
                continue;
            }
            return None;
        }
        let union_parts = split_top_level(current, '|');
        if union_parts.len() > 1 {
            let non_none: Vec<&str> = union_parts
                .into_iter()
                .filter(|p| !p.is_empty() && *p != "None")
                .collect();
            if non_none.len() == 1 {
                current = non_none[0].trim_matches(|c| c == '"' || c == '\'').trim();
                continue;
            }
            return None;
        }
        break;
    }

    if let Some(bracket) = current.find('[') {
        current = current[..bracket].trim();
    }
    if current.is_empty() || current.len() > 128 {
        return None;
    }
    if !current
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
    {
        return None;
    }
    let leaf = current.rsplit('.').next().unwrap_or(current);
    if leaf.is_empty() || PYTHON_IGNORED_TYPES.contains(&leaf) {
        return None;
    }
    Some(current.to_string())
}

fn collect_python_param_types(fn_node: &Node, source: &[u8], out: &mut HashMap<String, String>) {
    let Some(params) = fn_node.child_by_field_name("parameters") else {
        return;
    };
    let mut cursor = params.walk();
    for param in params.named_children(&mut cursor) {
        match param.kind() {
            "typed_parameter" | "typed_default_parameter" => {
                let mut name: Option<&str> = None;
                let mut type_node: Option<Node> = param.child_by_field_name("type");
                let mut pc = param.walk();
                for child in param.named_children(&mut pc) {
                    if name.is_none() && child.kind() == "identifier" {
                        name = Some(get_text(&child, source));
                    } else if type_node.is_none() && child.kind() == "type" {
                        type_node = Some(child);
                    }
                }
                if let (Some(var_name), Some(ty)) = (name, type_node) {
                    if let Some(norm) = normalize_python_type_name(get_text(&ty, source)) {
                        out.insert(var_name.to_string(), norm);
                    }
                }
            }
            _ => {}
        }
    }
}

fn infer_python_rhs_type(
    rhs: Node,
    source: &[u8],
    known_types: &HashMap<String, String>,
    file_return_types: &HashMap<String, String>,
) -> Option<String> {
    let expr = if rhs.kind() == "await" {
        rhs.named_child(0)?
    } else {
        rhs
    };
    match expr.kind() {
        "identifier" => known_types.get(get_text(&expr, source)).cloned(),
        "attribute" => known_types.get(get_text(&expr, source).trim()).cloned(),
        "call" => {
            let func = expr.child_by_field_name("function")?;
            match func.kind() {
                "identifier" => {
                    let callee = get_text(&func, source).trim();
                    if let Some(ret) = file_return_types.get(callee) {
                        return Some(ret.clone());
                    }
                    if callee
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_uppercase())
                        && !PYTHON_IGNORED_TYPES.contains(&callee)
                    {
                        return Some(callee.to_string());
                    }
                    None
                }
                "attribute" => {
                    let full = get_text(&func, source).trim();
                    let attr = func
                        .child_by_field_name("attribute")
                        .map(|a| get_text(&a, source).trim())
                        .unwrap_or("");
                    if attr.chars().next().is_some_and(|c| c.is_ascii_uppercase())
                        && !PYTHON_IGNORED_TYPES.contains(&attr)
                        && full
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
                    {
                        return Some(full.to_string());
                    }
                    None
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn collect_python_file_return_types(root: Node, source: &[u8]) -> HashMap<String, String> {
    let mut returns = HashMap::new();
    let mut cursor = root.walk();
    for child in root.named_children(&mut cursor) {
        let fn_node = match child.kind() {
            "function_definition" => Some(child),
            "decorated_definition" => {
                let mut sc = child.walk();
                let found = child
                    .named_children(&mut sc)
                    .find(|n| n.kind() == "function_definition");
                found
            }
            _ => None,
        };
        if let Some(f) = fn_node {
            if let Some(name) = function_name_of(&f, source) {
                let ret = get_return_type(&f, source);
                if let Some(norm) = normalize_python_type_name(&ret) {
                    returns.insert(name, norm);
                }
            }
        }
    }
    returns
}

fn collect_python_class_attr_types(
    class_body: Node,
    source: &[u8],
    file_return_types: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut attrs = HashMap::new();
    let mut cursor = class_body.walk();
    for child in class_body.named_children(&mut cursor) {
        match child.kind() {
            "expression_statement" => {
                if let Some(assign) = child.named_child(0) {
                    if assign.kind() == "assignment" {
                        if let (Some(lhs), Some(ty)) = (
                            assign.child_by_field_name("left"),
                            assign.child_by_field_name("type"),
                        ) {
                            if lhs.kind() == "identifier" {
                                if let Some(norm) = normalize_python_type_name(get_text(&ty, source))
                                {
                                    attrs.insert(format!("self.{}", get_text(&lhs, source)), norm);
                                }
                            }
                        }
                    }
                }
            }
            "function_definition" => {
                if function_name_of(&child, source).as_deref() == Some("__init__") {
                    let mut init_types = attrs.clone();
                    collect_python_param_types(&child, source, &mut init_types);
                    if let Some(body) = child.child_by_field_name("body") {
                        collect_python_body_var_types(
                            body,
                            source,
                            &mut init_types,
                            file_return_types,
                        );
                    }
                    for (k, v) in init_types {
                        if k.starts_with("self.") {
                            attrs.insert(k, v);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    attrs
}

fn collect_python_body_var_types(
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
            "function_definition" => {
                collect_python_param_types(&n, source, local_types);
            }
            "assignment" => {
                if let Some(lhs) = n.child_by_field_name("left") {
                    let lhs_key = match lhs.kind() {
                        "identifier" => Some(get_text(&lhs, source).trim().to_string()),
                        "attribute" => {
                            let text = get_text(&lhs, source).trim();
                            text.starts_with("self.").then(|| text.to_string())
                        }
                        _ => None,
                    };
                    if let Some(key) = lhs_key {
                        let explicit = n
                            .child_by_field_name("type")
                            .and_then(|ty| normalize_python_type_name(get_text(&ty, source)));
                        let inferred = explicit.or_else(|| {
                            n.child_by_field_name("right").and_then(|rhs| {
                                infer_python_rhs_type(rhs, source, local_types, file_return_types)
                            })
                        });
                        if let Some(ty_name) = inferred {
                            local_types.insert(key, ty_name);
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

fn attach_receiver_types(sites: &mut [CallSite], local_types: &HashMap<String, String>) {
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

/// Extract base classes from a class definition's argument_list.
fn get_bases(node: &Node, source: &[u8]) -> Vec<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "argument_list" {
            let mut bases = Vec::new();
            let mut arg_cursor = child.walk();
            for arg in child.children(&mut arg_cursor) {
                if arg.kind() == "identifier" || arg.kind() == "attribute" {
                    bases.push(get_text(&arg, source).to_string());
                }
            }
            return bases;
        }
    }
    Vec::new()
}

/// Parse Python source files using tree-sitter.
///
/// Walks the concrete syntax tree to extract structural entities and
/// their relationships. Handles nested classes and functions via a
/// scope stack for qualified name construction.
#[derive(Default)]
pub struct PythonParser;

impl PythonParser {
    /// Create a new Python parser (grammar is loaded on each parse call).
    pub fn new() -> Self {
        Self
    }

    /// Recursively walk the tree-sitter CST, extracting entities.
    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
        file_return_types: &HashMap<String, String>,
        enclosing_types: &HashMap<String, String>,
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "class_definition" => {
                    self.handle_class(
                        child,
                        source,
                        module_name,
                        entities,
                        edges,
                        scope_stack,
                        file_return_types,
                    );
                }
                "function_definition" => {
                    self.handle_function(
                        child,
                        source,
                        module_name,
                        entities,
                        edges,
                        scope_stack,
                        file_return_types,
                        enclosing_types,
                    );
                }
                "decorated_definition" => {
                    let mut decorators = Vec::new();
                    let mut sub_cursor = child.walk();
                    for sub in child.children(&mut sub_cursor) {
                        if sub.kind() == "decorator" {
                            decorators.push(sub);
                        }
                    }
                    let mut sub_cursor2 = child.walk();
                    for sub in child.children(&mut sub_cursor2) {
                        match sub.kind() {
                            "class_definition" => self.handle_class(
                                sub,
                                source,
                                module_name,
                                entities,
                                edges,
                                scope_stack,
                                file_return_types,
                            ),
                            "function_definition" => {
                                self.handle_function(
                                    sub,
                                    source,
                                    module_name,
                                    entities,
                                    edges,
                                    scope_stack,
                                    file_return_types,
                                    enclosing_types,
                                );
                                if !decorators.is_empty() {
                                    if let Some(fn_name) = function_name_of(&sub, source) {
                                        let fn_qname =
                                            format!("{}.{fn_name}", scope_stack.join("."));
                                        extract_python_route_decorators(
                                            &decorators,
                                            source,
                                            &fn_qname,
                                            edges,
                                        );
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "import_statement" => {
                    self.handle_import(child, source, module_name, entities, edges);
                }
                "import_from_statement" => {
                    self.handle_import_from(child, source, module_name, entities, edges);
                }
                "expression_statement" => {
                    self.handle_assignment(
                        child,
                        source,
                        module_name,
                        entities,
                        edges,
                        scope_stack,
                    );
                }
                "type_alias_statement" => {
                    self.handle_type_alias(
                        child,
                        source,
                        module_name,
                        entities,
                        edges,
                        scope_stack,
                    );
                }
                _ => {}
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
        file_return_types: &HashMap<String, String>,
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
        let bases = get_bases(&node, source);

        // Find the block/body node for docstring extraction and recursion.
        let mut body_node = None;
        let mut cursor2 = node.walk();
        for child in node.children(&mut cursor2) {
            if child.kind() == "block" {
                body_node = Some(child);
                break;
            }
        }

        let docstring = get_docstring(body_node, source);
        let raw = get_text(&node, source);
        let raw_truncated = super::truncate_str(raw, super::RAW_TEXT_LIMIT);

        entities.push(ParsedEntity {
            kind: NodeType::Class,
            name: name.clone(),
            qualified_name: qualified.clone(),
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            docstring,
            raw_text: raw_truncated.to_string(),
            parent_qualified_name: Some(scope_stack.join(".")),
            bases: bases.clone(),
            signature: String::new(),
            imports: Vec::new(),
            cyclomatic_complexity: None,
        });

        // CONTAINS edge from parent scope.
        edges.push(ParsedEdge::new(
            scope_stack.join("."),
            qualified.clone(),
            "contains",
        ));

        // INHERITS edges for base classes.
        for base in &bases {
            let mut meta = HashMap::new();
            meta.insert("reference_kind".to_string(), json!("extends"));
            meta.insert("start_line".to_string(), json!(node.start_position().row));
            meta.insert("end_line".to_string(), json!(node.end_position().row));
            meta.insert("enclosing_symbol".to_string(), json!(qualified));
            edges.push(ParsedEdge::with_metadata(
                qualified.clone(),
                base.clone(),
                "inherits",
                meta,
            ));
        }

        // Recurse into class body for methods and nested classes.
        if let Some(body) = body_node {
            let class_attrs = collect_python_class_attr_types(body, source, file_return_types);
            let mut new_scope = scope_stack.to_vec();
            new_scope.push(name);
            self.walk(
                body,
                source,
                module_name,
                entities,
                edges,
                &new_scope,
                file_return_types,
                &class_attrs,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_function(
        &self,
        node: Node,
        source: &[u8],
        _module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
        file_return_types: &HashMap<String, String>,
        enclosing_types: &HashMap<String, String>,
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

        // Determine if this is a method (inside a class) or a top-level function.
        let is_method = scope_stack.len() > 1; // module.Class.method = depth > 1
        let kind = if is_method {
            NodeType::Method
        } else {
            NodeType::Function
        };

        let params = get_parameters(&node, source);
        let return_type = get_return_type(&node, source);
        let mut signature = params;
        if !return_type.is_empty() {
            signature = format!("{signature} -> {return_type}");
        }

        // Find body for docstring.
        let mut body_node = None;
        let mut cursor2 = node.walk();
        for child in node.children(&mut cursor2) {
            if child.kind() == "block" {
                body_node = Some(child);
                break;
            }
        }
        let docstring = get_docstring(body_node, source);

        let raw = get_text(&node, source);
        let raw_truncated = super::truncate_str(raw, super::RAW_TEXT_LIMIT);
        let complexity = count_complexity(&node, source, "python");

        entities.push(ParsedEntity {
            kind,
            name,
            qualified_name: qualified.clone(),
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            signature,
            docstring,
            raw_text: raw_truncated.to_string(),
            parent_qualified_name: Some(scope_stack.join(".")),
            bases: Vec::new(),
            imports: Vec::new(),
            cyclomatic_complexity: Some(complexity),
        });

        // CONTAINS edge from parent scope.
        edges.push(ParsedEdge::new(
            scope_stack.join("."),
            qualified.clone(),
            "contains",
        ));

        // CALLS edges from this function/method to callees in its body.
        if let Some(body) = body_node {
            let mut local_types = enclosing_types.clone();
            collect_python_param_types(&node, source, &mut local_types);
            collect_python_body_var_types(body, source, &mut local_types, file_return_types);

            let blocklist: HashSet<&str> = PYTHON_BUILTIN_CALLS.iter().copied().collect();
            let mut sites = extract_call_sites(&body, source, "python", &blocklist);
            attach_receiver_types(&mut sites, &local_types);
            edges.extend(build_call_edges(&qualified, sites));
            extract_python_http_clients(body, source, &qualified, edges);
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
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "dotted_name" {
                let name = get_text(&child, source).to_string();
                let qualified = format!("{module_name}.import.{name}");
                entities.push(ParsedEntity {
                    kind: NodeType::Import,
                    name: name.clone(),
                    qualified_name: qualified.clone(),
                    start_line: node.start_position().row,
                    end_line: node.end_position().row,
                    imports: vec![name.clone()],
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
                let mut meta = HashMap::new();
                meta.insert("reference_kind".to_string(), json!("imports_namespace"));
                meta.insert("start_line".to_string(), json!(node.start_position().row));
                meta.insert("end_line".to_string(), json!(node.end_position().row));
                meta.insert("enclosing_symbol".to_string(), json!(module_name));
                meta.insert("wildcard_import".to_string(), json!(true));
                edges.push(ParsedEdge::with_metadata(
                    module_name.to_string(),
                    name,
                    "imports",
                    meta,
                ));
            }
        }
    }

    fn handle_import_from(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
    ) {
        let mut from_module = String::new();
        let mut imported_pairs: Vec<(String, String)> = Vec::new();
        let mut has_wildcard = false;

        if let Some(mod_node) = node.child_by_field_name("module_name") {
            from_module = get_text(&mod_node, source).trim().to_string();
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if Some(child) == node.child_by_field_name("module_name") {
                continue;
            }
            match child.kind() {
                "dotted_name" => {
                    if from_module.is_empty() {
                        from_module = get_text(&child, source).to_string();
                    } else {
                        let mut sub_cursor = child.walk();
                        for sub in child.children(&mut sub_cursor) {
                            if sub.kind() == "identifier" {
                                let name = get_text(&sub, source).to_string();
                                imported_pairs.push((name.clone(), name));
                            }
                        }
                    }
                }
                "aliased_import" => {
                    let orig = child
                        .child_by_field_name("name")
                        .map(|n| get_text(&n, source).trim().to_string())
                        .unwrap_or_default();
                    let alias = child
                        .child_by_field_name("alias")
                        .map(|n| get_text(&n, source).trim().to_string())
                        .unwrap_or_else(|| orig.clone());
                    if !orig.is_empty() {
                        imported_pairs.push((orig, alias));
                    }
                }
                "import_from_specifier" => {
                    let mut sub_cursor = child.walk();
                    for sub in child.children(&mut sub_cursor) {
                        if sub.kind() == "identifier" {
                            let name = get_text(&sub, source).to_string();
                            imported_pairs.push((name.clone(), name));
                        }
                    }
                }
                "wildcard_import" => {
                    has_wildcard = true;
                }
                "identifier" => {
                    let name = get_text(&child, source).to_string();
                    imported_pairs.push((name.clone(), name));
                }
                _ => {}
            }
        }

        let imported_symbols: Vec<Value> = imported_pairs
            .iter()
            .map(|(orig, _)| json!(orig))
            .collect();
        let bindings: Vec<Value> = imported_pairs
            .iter()
            .map(|(orig, local)| {
                json!({
                    "imported": orig,
                    "local": local,
                    "kind": "named",
                })
            })
            .collect();

        let is_init_module = module_name == "__init__" || module_name.ends_with(".__init__");
        for (orig_name, local_name) in &imported_pairs {
            let full_import = if !from_module.is_empty() {
                format!("{from_module}.{orig_name}")
            } else {
                orig_name.clone()
            };
            let clean_mod = from_module.trim_start_matches('.');
            let qualified = format!("{module_name}.import.{full_import}");
            entities.push(ParsedEntity {
                kind: NodeType::Import,
                name: orig_name.clone(),
                qualified_name: qualified.clone(),
                start_line: node.start_position().row,
                end_line: node.end_position().row,
                imports: vec![full_import.clone()],
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
            let mut meta = HashMap::new();
            meta.insert("reference_kind".to_string(), json!("imports_named"));
            meta.insert("start_line".to_string(), json!(node.start_position().row));
            meta.insert("end_line".to_string(), json!(node.end_position().row));
            meta.insert("enclosing_symbol".to_string(), json!(module_name));
            meta.insert("imported_symbols".to_string(), json!([orig_name]));
            meta.insert("wildcard_import".to_string(), json!(false));
            if !clean_mod.is_empty() {
                meta.insert("from_module".to_string(), json!(clean_mod));
            }
            if !bindings.is_empty() {
                meta.insert("import_bindings".to_string(), Value::Array(bindings.clone()));
            }
            if is_init_module && !from_module.is_empty() {
                meta.insert("is_reexport".to_string(), json!(true));
                meta.insert("reexport_from_module".to_string(), json!(from_module));
                meta.insert(
                    "reexport_bindings".to_string(),
                    json!([{"kind": "named", "imported": orig_name, "exported": local_name}]),
                );
            }
            edges.push(ParsedEdge::with_metadata(
                module_name.to_string(),
                full_import,
                "imports",
                meta,
            ));
        }

        if has_wildcard && imported_pairs.is_empty() && !from_module.is_empty() {
            let clean_mod = from_module.trim_start_matches('.').to_string();
            if !clean_mod.is_empty() {
                let mut meta = HashMap::new();
                meta.insert("reference_kind".to_string(), json!("imports_namespace"));
                meta.insert("start_line".to_string(), json!(node.start_position().row));
                meta.insert("end_line".to_string(), json!(node.end_position().row));
                meta.insert("enclosing_symbol".to_string(), json!(module_name));
                if !imported_symbols.is_empty() {
                    meta.insert("imported_symbols".to_string(), Value::Array(imported_symbols));
                }
                if !bindings.is_empty() {
                    meta.insert("import_bindings".to_string(), Value::Array(bindings));
                }
                meta.insert("wildcard_import".to_string(), json!(true));
                if is_init_module {
                    meta.insert("is_reexport".to_string(), json!(true));
                    meta.insert("reexport_from_module".to_string(), json!(from_module));
                    meta.insert("reexport_bindings".to_string(), json!([{"kind": "star"}]));
                }
                edges.push(ParsedEdge::with_metadata(
                    module_name.to_string(),
                    clean_mod,
                    "imports",
                    meta,
                ));
            }
        }
    }

    #[allow(clippy::ptr_arg)] // Signature mirrors walk() callback pattern.
    fn handle_assignment(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
    ) {
        // Only extract module-level assignments (depth 1 in scope stack).
        if scope_stack.len() > 1 {
            return;
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "assignment" {
                let mut sub_cursor = child.walk();
                for sub in child.children(&mut sub_cursor) {
                    if sub.kind() == "identifier" {
                        let name = get_text(&sub, source);
                        if name.len() < 2 {
                            continue;
                        }

                        let kind = if is_constant_name(name) {
                            NodeType::Constant
                        } else {
                            NodeType::Variable
                        };
                        let qualified = format!("{module_name}.{name}");
                        let raw = get_text(&child, source);
                        let signature = super::truncate_str(raw, super::RAW_TEXT_SMALL_LIMIT);

                        entities.push(ParsedEntity {
                            kind,
                            name: name.to_string(),
                            qualified_name: qualified.clone(),
                            start_line: node.start_position().row,
                            end_line: node.end_position().row,
                            signature: signature.to_string(),
                            raw_text: signature.to_string(),
                            parent_qualified_name: Some(module_name.to_string()),
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

                        break; // Only first identifier (left side of assignment).
                    }
                }
            }
        }
    }

    #[allow(clippy::ptr_arg)] // Signature mirrors walk() callback pattern.
    fn handle_type_alias(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        _edges: &mut Vec<ParsedEdge>,
        _scope_stack: &[String],
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "identifier" {
                let name = get_text(&child, source);
                if !name.is_empty() {
                    let qualified = format!("{module_name}.{name}");
                    let raw = get_text(&node, source);
                    let raw_truncated = super::truncate_str(raw, super::RAW_TEXT_SMALL_LIMIT);

                    entities.push(ParsedEntity {
                        kind: NodeType::TypeAlias,
                        name: name.to_string(),
                        qualified_name: qualified,
                        start_line: node.start_position().row,
                        end_line: node.end_position().row,
                        raw_text: raw_truncated.to_string(),
                        parent_qualified_name: Some(module_name.to_string()),
                        signature: String::new(),
                        docstring: String::new(),
                        bases: Vec::new(),
                        imports: Vec::new(),
                        cyclomatic_complexity: None,
                    });
                    break;
                }
            }
        }
    }
}

impl LanguageParser for PythonParser {
    fn parse(&self, source: &str, file_path: &str) -> (Vec<ParsedEntity>, Vec<ParsedEdge>) {
        let source_bytes = source.as_bytes();

        // We need a mutable reference to parse, but the trait takes &self.
        // tree-sitter's Parser requires &mut self for parse(), so we use
        // an unsafe workaround: create a fresh parser per call.
        // This is safe because PythonParser only holds the parser for config.
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser.set_language(&language.into()).unwrap();

        let tree = match parser.parse(source_bytes, None) {
            Some(t) => t,
            None => return (Vec::new(), Vec::new()),
        };

        let mut entities = Vec::new();
        let mut edges = Vec::new();

        // Module-level file entity.
        let module_name = file_path
            .replace('/', ".")
            .strip_suffix(".py")
            .unwrap_or(file_path)
            .to_string();

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

        // Walk the tree using a scope stack.
        let file_return_types = collect_python_file_return_types(tree.root_node(), source_bytes);
        let empty_types = HashMap::new();
        let scope_stack = vec![module_name.clone()];
        self.walk(
            tree.root_node(),
            source_bytes,
            &module_name,
            &mut entities,
            &mut edges,
            &scope_stack,
            &file_return_types,
            &empty_types,
        );

        (entities, edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_python_class() {
        let parser = PythonParser::new();
        let source = r#"
class MyClass:
    """A test class."""

    def my_method(self, x: int) -> str:
        return str(x)
"#;
        let (entities, edges) = parser.parse(source, "test_module.py");

        // Should find: file, class, method
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"test_module.py"));
        assert!(names.contains(&"MyClass"));
        assert!(names.contains(&"my_method"));

        // Method should have correct kind.
        let method = entities.iter().find(|e| e.name == "my_method").unwrap();
        assert_eq!(method.kind, NodeType::Method);

        // Should have CONTAINS edges.
        assert!(edges.iter().any(|e| e.relationship == "contains"));
    }

    #[test]
    fn parses_imports() {
        let parser = PythonParser::new();
        let source = "from pathlib import Path\nimport os\n";
        let (entities, _edges) = parser.parse(source, "mod.py");

        let imports: Vec<&ParsedEntity> = entities
            .iter()
            .filter(|e| e.kind == NodeType::Import)
            .collect();
        assert!(!imports.is_empty());
    }

    #[test]
    fn detects_constants() {
        let parser = PythonParser::new();
        let source = "MAX_RETRIES = 3\nmy_var = 42\n";
        let (entities, _) = parser.parse(source, "config.py");

        let constant = entities.iter().find(|e| e.name == "MAX_RETRIES");
        assert!(constant.is_some());
        assert_eq!(constant.unwrap().kind, NodeType::Constant);

        let variable = entities.iter().find(|e| e.name == "my_var");
        assert!(variable.is_some());
        assert_eq!(variable.unwrap().kind, NodeType::Variable);
    }

    #[test]
    fn cyclomatic_complexity_simple_function() {
        let parser = PythonParser::new();
        let source = "def greet(name: str) -> str:\n    return f'Hello, {name}'\n";
        let (entities, _) = parser.parse(source, "greet.py");

        let func = entities.iter().find(|e| e.name == "greet").unwrap();
        assert_eq!(func.kind, NodeType::Function);
        assert_eq!(func.cyclomatic_complexity, Some(1));
    }

    #[test]
    fn cyclomatic_complexity_branching() {
        let parser = PythonParser::new();
        let source = r#"
def process(items):
    for item in items:
        if item > 0:
            print(item)
"#;
        let (entities, _) = parser.parse(source, "process.py");

        let func = entities.iter().find(|e| e.name == "process").unwrap();
        assert_eq!(func.kind, NodeType::Function);
        // 1 base + for_statement + if_statement = 3
        assert_eq!(func.cyclomatic_complexity, Some(3));

        // Class entities should not have cyclomatic complexity.
        let parser2 = PythonParser::new();
        let source2 = "class Empty:\n    pass\n";
        let (entities2, _) = parser2.parse(source2, "empty.py");
        let cls = entities2.iter().find(|e| e.name == "Empty").unwrap();
        assert_eq!(cls.kind, NodeType::Class);
        assert_eq!(cls.cyclomatic_complexity, None);
    }

    #[test]
    fn parses_inheritance() {
        let parser = PythonParser::new();
        let source = "class Child(Base, Mixin):\n    pass\n";
        let (entities, edges) = parser.parse(source, "hierarchy.py");

        let child = entities.iter().find(|e| e.name == "Child").unwrap();
        assert_eq!(child.bases, vec!["Base", "Mixin"]);

        let inherits: Vec<&ParsedEdge> = edges
            .iter()
            .filter(|e| e.relationship == "inherits")
            .collect();
        assert_eq!(inherits.len(), 2);
    }

    #[test]
    fn calls_edges_simple() {
        let parser = PythonParser::new();
        let source = r#"
def caller():
    helper()
    result = process(data)
"#;
        let (_, edges) = parser.parse(source, "mod.py");
        let calls: Vec<&ParsedEdge> = edges.iter().filter(|e| e.relationship == "calls").collect();
        assert!(calls.iter().any(|e| e.target_qualified_name == "helper"));
        assert!(calls.iter().any(|e| e.target_qualified_name == "process"));
        assert!(calls
            .iter()
            .all(|e| e.source_qualified_name.contains("caller")));
    }

    #[test]
    fn calls_edges_method_call() {
        let parser = PythonParser::new();
        let source = r#"
class MyClass:
    def method(self):
        self.other_method()
        external_func()
"#;
        let (_, edges) = parser.parse(source, "cls.py");
        let calls: Vec<&ParsedEdge> = edges.iter().filter(|e| e.relationship == "calls").collect();
        assert!(calls
            .iter()
            .any(|e| e.target_qualified_name == "other_method"));
        assert!(calls
            .iter()
            .any(|e| e.target_qualified_name == "external_func"));
    }

    #[test]
    fn calls_edges_deduplicated() {
        let parser = PythonParser::new();
        let source = "def f():\n    foo()\n    foo()\n    foo()\n";
        let (_, edges) = parser.parse(source, "dedup.py");
        let foo_calls: Vec<&ParsedEdge> = edges
            .iter()
            .filter(|e| e.relationship == "calls" && e.target_qualified_name == "foo")
            .collect();
        assert_eq!(foo_calls.len(), 1, "Duplicate calls should be deduplicated");
    }

    #[test]
    fn calls_edges_skip_builtins() {
        let parser = PythonParser::new();
        let source = "def f():\n    print('hello')\n    len(data)\n    range(10)\n";
        let (_, edges) = parser.parse(source, "builtins.py");
        let calls: Vec<&ParsedEdge> = edges.iter().filter(|e| e.relationship == "calls").collect();
        assert!(
            calls.is_empty(),
            "Built-in calls should be filtered out, got: {:?}",
            calls
                .iter()
                .map(|e| &e.target_qualified_name)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn calls_edges_chained_method() {
        let parser = PythonParser::new();
        let source = "def f():\n    a.b.deep_call()\n";
        let (_, edges) = parser.parse(source, "chain.py");
        let calls: Vec<&ParsedEdge> = edges.iter().filter(|e| e.relationship == "calls").collect();
        // Should extract the rightmost identifier from the chain.
        assert!(calls.iter().any(|e| e.target_qualified_name == "deep_call"));
    }

    #[test]
    fn extracts_call_site_lines_aliased_imports_and_fastapi_routes() {
        let parser = PythonParser::new();
        let source = r#"
from app.services import create_item as make_item

@router.post("/api/items/{item_id}")
def post_item():
    return make_item()

def call_backend():
    return requests.post("/api/items/123")
"#;
        let (_, edges) = parser.parse(source, "app/routes.py");

        assert!(edges.iter().any(|e| {
            e.relationship == "calls"
                && e.target_qualified_name == "make_item"
                && e.metadata.get("start_line") == Some(&json!(5))
                && e.metadata.get("enclosing_symbol") == Some(&json!("app.routes.post_item"))
        }));
        assert!(edges.iter().any(|e| {
            e.relationship == "imports"
                && e.target_qualified_name == "app.services.create_item"
                && e.metadata.get("imported_symbols") == Some(&json!(["create_item"]))
                && e.metadata.get("from_module") == Some(&json!("app.services"))
        }));
        assert!(edges.iter().any(|e| {
            e.metadata.get("reference_kind") == Some(&json!("http_route_handler"))
                && e.metadata.get("http_method") == Some(&json!("POST"))
                && e.metadata.get("norm_path") == Some(&json!("/api/items/:param"))
                && e.metadata.get("handler_symbol") == Some(&json!("app.routes.post_item"))
        }));
        assert!(edges.iter().any(|e| {
            e.metadata.get("reference_kind") == Some(&json!("http_route_client"))
                && e.metadata.get("http_method") == Some(&json!("POST"))
                && e.metadata.get("norm_path") == Some(&json!("/api/items/123"))
        }));
    }

    #[test]
    fn extracts_receiver_types_from_params_factories_constructors_and_class_attrs() {
        let parser = PythonParser::new();
        let source = r#"
from auth.service import AdminAuthService
from session_inventory import SessionInventory

def _auth_service() -> AdminAuthService:
    return AdminAuthService()

def register_handlers(auth: AdminAuthService | None) -> None:
    async def sessions_list() -> None:
        await auth.list_active_sessions()

class Controller:
    def __init__(self, auth: AdminAuthService) -> None:
        self._auth = auth

    def wire(self) -> None:
        self._auth.set_session_observer(SessionInventory())

async def test_flow() -> None:
    service = _auth_service()
    inventory = SessionInventory()
    service.set_session_observer(inventory)
    inventory.page(await service.list_active_sessions())
"#;
        let (_, edges) = parser.parse(source, "apps/cortex/cortex-engine/web_app.py");

        assert!(edges.iter().any(|e| {
            e.relationship == "calls"
                && e.source_qualified_name.ends_with(".register_handlers")
                && e.target_qualified_name == "list_active_sessions"
                && e.metadata.get("qualifier") == Some(&json!("auth"))
                && e.metadata.get("receiver_type") == Some(&json!("AdminAuthService"))
        }));
        assert!(edges.iter().any(|e| {
            e.relationship == "calls"
                && e.source_qualified_name.ends_with(".Controller.wire")
                && e.target_qualified_name == "set_session_observer"
                && e.metadata.get("qualifier") == Some(&json!("self._auth"))
                && e.metadata.get("receiver_type") == Some(&json!("AdminAuthService"))
        }));
        assert!(edges.iter().any(|e| {
            e.relationship == "calls"
                && e.source_qualified_name.ends_with(".test_flow")
                && e.target_qualified_name == "set_session_observer"
                && e.metadata.get("qualifier") == Some(&json!("service"))
                && e.metadata.get("receiver_type") == Some(&json!("AdminAuthService"))
        }));
        assert!(edges.iter().any(|e| {
            e.relationship == "calls"
                && e.source_qualified_name.ends_with(".test_flow")
                && e.target_qualified_name == "list_active_sessions"
                && e.metadata.get("qualifier") == Some(&json!("service"))
                && e.metadata.get("receiver_type") == Some(&json!("AdminAuthService"))
        }));
        assert!(edges.iter().any(|e| {
            e.relationship == "calls"
                && e.source_qualified_name.ends_with(".test_flow")
                && e.target_qualified_name == "page"
                && e.metadata.get("qualifier") == Some(&json!("inventory"))
                && e.metadata.get("receiver_type") == Some(&json!("SessionInventory"))
        }));
    }
}
