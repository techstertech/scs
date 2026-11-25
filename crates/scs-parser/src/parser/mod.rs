//! Language-specific AST parsers for code ingestion.
//!
//! All parsers produce the same output types ([`ParsedEntity`], [`ParsedEdge`])
//! regardless of the source language. This allows the ingestion pipeline
//! to treat all languages uniformly.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tree_sitter::Node;

use scs_core::node_types::NodeType;

pub mod bash;
pub mod css;
pub mod elixir;
pub mod go;
pub mod java;
pub mod python;
pub mod registry;
pub mod rust;
pub mod swift;
pub mod typescript;

/// Max bytes for raw_text on major entities (classes, functions, methods).
pub const RAW_TEXT_LIMIT: usize = 2048;
/// Max bytes for raw_text on minor entities (variables, constants, signatures).
pub const RAW_TEXT_SMALL_LIMIT: usize = 512;
/// Max bytes for signatures in embed_text output.
pub const SIGNATURE_LIMIT: usize = RAW_TEXT_SMALL_LIMIT;
/// Max bytes for docstrings in embed_text output.
pub const DOCSTRING_LIMIT: usize = 1024;
/// Maximum number of per-site line records retained in a single edge's metadata.
pub const MAX_SITES_PER_EDGE: usize = 16;

/// Normalize an HTTP route or client request path for cross-boundary matching.
///
/// Only accepts local paths starting with `/` (rejecting protocol-relative `//`),
/// strips query strings and fragments, normalizes `:param`, `{param}`, and
/// `${...}` segments to `:param`, and strips trailing slashes except for `/`.
pub fn normalize_http_path(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    let path_only = if trimmed.starts_with('/') && !trimmed.starts_with("//") {
        trimmed
    } else if let Some(rest) = trimmed
        .strip_prefix("http://localhost")
        .or_else(|| trimmed.strip_prefix("https://localhost"))
        .or_else(|| trimmed.strip_prefix("http://127.0.0.1"))
        .or_else(|| trimmed.strip_prefix("https://127.0.0.1"))
    {
        if let Some(idx) = rest.find('/') {
            &rest[idx..]
        } else {
            return None;
        }
    } else {
        return None;
    };

    let without_query = path_only
        .split('?')
        .next()
        .unwrap_or("")
        .split('#')
        .next()
        .unwrap_or("")
        .trim();
    if !without_query.starts_with('/') || without_query.starts_with("//") {
        return None;
    }

    let mut segments: Vec<&str> = Vec::new();
    for seg in without_query.split('/') {
        if seg.is_empty() {
            continue;
        }
        if seg.starts_with(':')
            || (seg.starts_with('{') && seg.ends_with('}'))
            || (seg.starts_with("${") && seg.ends_with('}'))
        {
            segments.push(":param");
        } else {
            segments.push(seg);
        }
    }

    if segments.is_empty() {
        Some("/".to_string())
    } else {
        Some(format!("/{}", segments.join("/")))
    }
}

/// Truncate a string to at most `max_bytes` without splitting a multi-byte
/// UTF-8 character. Always returns a valid `&str`.
pub fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    // Walk backwards from the limit to find a char boundary.
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Count McCabe cyclomatic complexity for a function/method body.
///
/// Walks all descendants of the given node, counting language-specific
/// decision-point nodes. The baseline is 1 (a straight-line function),
/// plus one for each branch, loop, exception handler, or short-circuit
/// boolean operator.
pub fn count_complexity(node: &Node, source: &[u8], language: &str) -> u32 {
    let decision_nodes: HashSet<&str> = match language {
        "python" => [
            "if_statement",
            "elif_clause",
            "for_statement",
            "while_statement",
            "except_clause",
            "with_statement",
            "assert_statement",
            "conditional_expression",
            "list_comprehension",
            "set_comprehension",
            "dictionary_comprehension",
            "generator_expression",
        ]
        .into_iter()
        .collect(),

        "typescript" | "javascript" | "tsx" | "jsx" => [
            "if_statement",
            "for_statement",
            "for_in_statement",
            "while_statement",
            "do_statement",
            "catch_clause",
            "ternary_expression",
        ]
        .into_iter()
        .collect(),

        "rust" => [
            "if_expression",
            "if_let_expression",
            "for_expression",
            "while_expression",
            "while_let_expression",
        ]
        .into_iter()
        .collect(),

        "go" => [
            "if_statement",
            "for_statement",
            "expression_case",
            "type_case",
            "communication_case",
        ]
        .into_iter()
        .collect(),

        "java" => [
            "if_statement",
            "for_statement",
            "enhanced_for_statement",
            "while_statement",
            "do_statement",
            "catch_clause",
            "ternary_expression",
        ]
        .into_iter()
        .collect(),

        "swift" => [
            "if_statement",
            "guard_statement",
            "for_statement",
            "while_statement",
            "repeat_while_statement",
            "catch_clause",
            "ternary_expression",
        ]
        .into_iter()
        .collect(),

        "bash" => [
            "if_statement",
            "elif_clause",
            "for_statement",
            "while_statement",
            "c_style_for_statement",
            "case_item",
        ]
        .into_iter()
        .collect(),

        "elixir" => HashSet::new(), // Elixir uses call-based dispatch below.

        _ => HashSet::new(),
    };

    let mut count: u32 = 0;
    let mut cursor = node.walk();
    // Walk all descendants via tree-sitter cursor.
    let mut reached_root = false;
    loop {
        let n = cursor.node();

        if decision_nodes.contains(n.kind()) {
            count += 1;
        }

        // Language-specific compound checks.
        match language {
            "python"
                // `and` / `or` boolean operators.
                if n.kind() == "boolean_operator" => {
                    count += 1;
                }
            "typescript" | "javascript" | "tsx" | "jsx" => {
                // `switch_case` but not default.
                if n.kind() == "switch_case" {
                    // A switch_case with a value (not `default:`) is a decision.
                    let text =
                        std::str::from_utf8(&source[n.start_byte()..n.end_byte()]).unwrap_or("");
                    if !text.starts_with("default") {
                        count += 1;
                    }
                }
                // `&&`, `||`, `??` in binary expressions.
                if n.kind() == "binary_expression" {
                    let mut child_cursor = n.walk();
                    for child in n.children(&mut child_cursor) {
                        let text =
                            std::str::from_utf8(&source[child.start_byte()..child.end_byte()])
                                .unwrap_or("");
                        if text == "&&" || text == "||" || text == "??" {
                            count += 1;
                        }
                    }
                }
            }
            "rust" => {
                // match_arm (non-wildcard).
                if n.kind() == "match_arm" {
                    count += 1;
                }
                // `&&` / `||` in binary expressions.
                if n.kind() == "binary_expression" {
                    let mut child_cursor = n.walk();
                    for child in n.children(&mut child_cursor) {
                        let text =
                            std::str::from_utf8(&source[child.start_byte()..child.end_byte()])
                                .unwrap_or("");
                        if text == "&&" || text == "||" {
                            count += 1;
                        }
                    }
                }
            }
            "go" if n.kind() == "binary_expression" => {
                let mut child_cursor = n.walk();
                for child in n.children(&mut child_cursor) {
                    let text =
                        std::str::from_utf8(&source[child.start_byte()..child.end_byte()])
                            .unwrap_or("");
                    if text == "&&" || text == "||" {
                        count += 1;
                    }
                }
            }
            "java" => {
                if n.kind() == "switch_label" {
                    let text =
                        std::str::from_utf8(&source[n.start_byte()..n.end_byte()]).unwrap_or("");
                    if !text.trim_start().starts_with("default") {
                        count += 1;
                    }
                }
                if n.kind() == "binary_expression" {
                    if let Some(op) = n.child_by_field_name("operator") {
                        let text =
                            std::str::from_utf8(&source[op.start_byte()..op.end_byte()])
                                .unwrap_or("");
                        if text == "&&" || text == "||" {
                            count += 1;
                        }
                    }
                }
            }
            "swift"
                // switch_case.
                if n.kind() == "switch_case" => {
                    count += 1;
                }
            "bash"
                // `&&` / `||` operators in pipelines or list constructs.
                if (n.kind() == "pipeline" || n.kind() == "list") => {
                    let text =
                        std::str::from_utf8(&source[n.start_byte()..n.end_byte()]).unwrap_or("");
                    // Count && and || operators (they join commands).
                    count += text.matches("&&").count() as u32;
                    count += text.matches("||").count() as u32;
                }
            "elixir" => {
                // Control flow is via `call` nodes: if/unless/cond/case/with/try.
                if n.kind() == "call" {
                    let mut child_cursor = n.walk();
                    for child in n.children(&mut child_cursor) {
                        if child.kind() == "identifier" {
                            let text =
                                std::str::from_utf8(&source[child.start_byte()..child.end_byte()])
                                    .unwrap_or("");
                            if matches!(text, "if" | "unless" | "cond" | "case" | "with" | "try") {
                                count += 1;
                            }
                            break;
                        }
                    }
                }
                // fn clause arms.
                if n.kind() == "stab_clause" {
                    count += 1;
                }
                // `and`/`or`/`&&`/`||` binary operators.
                if n.kind() == "binary_operator" {
                    let mut child_cursor = n.walk();
                    for child in n.children(&mut child_cursor) {
                        let text =
                            std::str::from_utf8(&source[child.start_byte()..child.end_byte()])
                                .unwrap_or("");
                        if matches!(text, "and" | "or" | "&&" | "||") {
                            count += 1;
                        }
                    }
                }
            }
            _ => {}
        }

        // Depth-first traversal via tree-sitter cursor.
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

    1 + count
}

/// A single call expression occurrence extracted from an AST body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallSite {
    pub name: String,
    pub qualifier: Option<String>,
    pub start_line: usize,
    pub end_line: usize,
}

fn dotted_receiver_qualifier(node: &Node, source: &[u8], language: &str) -> Option<String> {
    let text_of = |n: &Node| -> &str {
        std::str::from_utf8(&source[n.start_byte()..n.end_byte()]).unwrap_or("")
    };
    let valid_text = |n: &Node| -> Option<String> {
        let raw = text_of(n).trim();
        if raw.is_empty()
            || raw.len() > 128
            || raw.contains('(')
            || raw.contains('[')
            || raw.contains('\n')
            || raw.chars().any(char::is_whitespace)
        {
            return None;
        }
        Some(raw.to_string())
    };
    match language {
        "typescript" | "javascript" | "tsx" | "jsx" => match node.kind() {
            "identifier" | "this" | "super" => valid_text(node),
            "member_expression" => {
                let obj = node.child_by_field_name("object")?;
                let prop = node.child_by_field_name("property")?;
                if !matches!(
                    prop.kind(),
                    "property_identifier" | "identifier" | "private_property_identifier"
                ) {
                    return None;
                }
                let prefix = dotted_receiver_qualifier(&obj, source, language)?;
                let prop_text = valid_text(&prop)?;
                let combined = format!("{prefix}.{prop_text}");
                (combined.len() <= 128).then_some(combined)
            }
            _ => None,
        },
        "python" => match node.kind() {
            "identifier" => valid_text(node),
            "attribute" => {
                let obj = node.child_by_field_name("object")?;
                let attr = node.child_by_field_name("attribute")?;
                let prefix = dotted_receiver_qualifier(&obj, source, language)?;
                let attr_text = valid_text(&attr)?;
                let combined = format!("{prefix}.{attr_text}");
                (combined.len() <= 128).then_some(combined)
            }
            _ => None,
        },
        _ => None,
    }
}

/// Extract the simple callee name and optional receiver qualifier from a call expression node.
fn extract_callee_info(node: &Node, source: &[u8], language: &str) -> Option<(String, Option<String>)> {
    let text_of = |n: &Node| -> &str {
        std::str::from_utf8(&source[n.start_byte()..n.end_byte()]).unwrap_or("")
    };

    match language {
        "python" => {
            if let Some(func) = node.child_by_field_name("function") {
                match func.kind() {
                    "identifier" => return Some((text_of(&func).to_string(), None)),
                    "attribute" => {
                        let method = func
                            .child_by_field_name("attribute")
                            .map(|attr| text_of(&attr).to_string());
                        let qualifier = func
                            .child_by_field_name("object")
                            .and_then(|obj| dotted_receiver_qualifier(&obj, source, language));
                        if let Some(name) = method {
                            return Some((name, qualifier));
                        }
                    }
                    _ => {}
                }
            }
            // Fallback walk for alternate node child layout.
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                match child.kind() {
                    "identifier" if child == node.child(0)? => {
                        return Some((text_of(&child).to_string(), None));
                    }
                    "attribute" => {
                        let mut attr_cursor = child.walk();
                        let mut last_ident = None;
                        for sub in child.children(&mut attr_cursor) {
                            if sub.kind() == "identifier" {
                                last_ident = Some(text_of(&sub).to_string());
                            }
                        }
                        let qualifier = child
                            .child_by_field_name("object")
                            .and_then(|obj| dotted_receiver_qualifier(&obj, source, language));
                        return last_ident.map(|name| (name, qualifier));
                    }
                    _ => {}
                }
            }
            None
        }
        "typescript" | "javascript" | "tsx" | "jsx" => {
            if node.kind() == "new_expression" {
                let ctor = node
                    .child_by_field_name("constructor")
                    .or_else(|| node.named_child(0))?;
                return match ctor.kind() {
                    "identifier" | "type_identifier" => Some((text_of(&ctor).to_string(), None)),
                    "member_expression" => {
                        let prop = ctor.child_by_field_name("property")?;
                        let qualifier = ctor
                            .child_by_field_name("object")
                            .and_then(|obj| dotted_receiver_qualifier(&obj, source, language));
                        Some((text_of(&prop).to_string(), qualifier))
                    }
                    _ => None,
                };
            }
            let callee = node
                .child_by_field_name("function")
                .or_else(|| node.child(0))?;
            match callee.kind() {
                "identifier" => Some((text_of(&callee).to_string(), None)),
                "member_expression" => {
                    let qualifier = callee
                        .child_by_field_name("object")
                        .and_then(|obj| dotted_receiver_qualifier(&obj, source, language));
                    if let Some(prop) = callee.child_by_field_name("property") {
                        return Some((text_of(&prop).to_string(), qualifier));
                    }
                    let mut cursor = callee.walk();
                    for child in callee.children(&mut cursor) {
                        if child.kind() == "property_identifier" {
                            return Some((text_of(&child).to_string(), qualifier));
                        }
                    }
                    None
                }
                "new_expression" => {
                    let mut cursor = callee.walk();
                    for child in callee.children(&mut cursor) {
                        if child.kind() == "identifier" {
                            return Some((text_of(&child).to_string(), None));
                        }
                    }
                    None
                }
                _ => None,
            }
        }
        "rust" => {
            let callee = node.child(0)?;
            match callee.kind() {
                "identifier" => Some((text_of(&callee).to_string(), None)),
                "field_expression" => {
                    let mut cursor = callee.walk();
                    for child in callee.children(&mut cursor) {
                        if child.kind() == "field_identifier" {
                            return Some((text_of(&child).to_string(), None));
                        }
                    }
                    None
                }
                "scoped_identifier" => {
                    let mut cursor = callee.walk();
                    let mut last_ident = None;
                    for child in callee.children(&mut cursor) {
                        if child.kind() == "identifier" {
                            last_ident = Some(text_of(&child).to_string());
                        }
                    }
                    last_ident.map(|name| (name, None))
                }
                _ => None,
            }
        }
        "go" => {
            let callee = node.child_by_field_name("function")?;
            match callee.kind() {
                "identifier" => Some((text_of(&callee).to_string(), None)),
                "selector_expression" => callee
                    .child_by_field_name("field")
                    .map(|field| (text_of(&field).to_string(), None)),
                "index_expression" | "generic_type" => callee
                    .child_by_field_name("operand")
                    .or_else(|| callee.child_by_field_name("function"))
                    .or_else(|| callee.named_child(0))
                    .and_then(|inner| match inner.kind() {
                        "identifier" => Some((text_of(&inner).to_string(), None)),
                        "selector_expression" => inner
                            .child_by_field_name("field")
                            .map(|field| (text_of(&field).to_string(), None)),
                        _ => None,
                    }),
                _ => None,
            }
        }
        "java" => match node.kind() {
            "method_invocation" => node
                .child_by_field_name("name")
                .map(|name| (text_of(&name).to_string(), None)),
            "object_creation_expression" => node
                .child_by_field_name("type")
                .and_then(|ty| java::normalize_java_type_name(ty, source))
                .map(|ty| (ty.rsplit('.').next().unwrap_or(&ty).to_string(), None)),
            _ => None,
        },
        "swift" => {
            let callee = node.child(0)?;
            match callee.kind() {
                "simple_identifier" => Some((text_of(&callee).to_string(), None)),
                "navigation_expression" => {
                    let mut cursor = callee.walk();
                    let mut last_ident = None;
                    for child in callee.children(&mut cursor) {
                        if child.kind() == "simple_identifier" {
                            last_ident = Some(text_of(&child).to_string());
                        } else if child.kind() == "navigation_suffix" {
                            let mut sub_cursor = child.walk();
                            for sub in child.children(&mut sub_cursor) {
                                if sub.kind() == "simple_identifier" {
                                    last_ident = Some(text_of(&sub).to_string());
                                }
                            }
                        }
                    }
                    last_ident.map(|name| (name, None))
                }
                _ => None,
            }
        }
        "elixir" => {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "identifier" {
                    return Some((text_of(&child).to_string(), None));
                }
                if child.kind() == "dot" {
                    let mut dot_cursor = child.walk();
                    let mut last_ident = None;
                    for sub in child.children(&mut dot_cursor) {
                        if sub.kind() == "identifier" {
                            last_ident = Some(text_of(&sub).to_string());
                        }
                    }
                    return last_ident.map(|name| (name, None));
                }
            }
            None
        }
        _ => None,
    }
}

/// Extract call expression sites with 0-based line spans from a function/method body.
pub fn extract_call_sites(
    node: &Node,
    source: &[u8],
    language: &str,
    blocklist: &HashSet<&str>,
) -> Vec<CallSite> {
    let call_kinds: &[&str] = match language {
        "python" => &["call"],
        "typescript" | "javascript" | "tsx" | "jsx" => &["call_expression", "new_expression"],
        "rust" => &["call_expression"],
        "go" => &["call_expression"],
        "java" => &["method_invocation", "object_creation_expression"],
        "swift" => &["call_expression"],
        "elixir" => &["call"],
        _ => return Vec::new(),
    };

    let mut sites = Vec::new();
    let mut cursor = node.walk();
    let mut reached_root = false;

    loop {
        let n = cursor.node();

        if call_kinds.contains(&n.kind()) {
            if let Some((name, qualifier)) = extract_callee_info(&n, source, language) {
                if !name.is_empty() && !blocklist.contains(name.as_str()) {
                    sites.push(CallSite {
                        name,
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

    sites
}

/// Group extracted call sites by callee name into deduplicated `"calls"` edges with line metadata.
pub fn build_call_edges(enclosing_qname: &str, sites: Vec<CallSite>) -> Vec<ParsedEdge> {
    let mut grouped: HashMap<String, Vec<CallSite>> = HashMap::new();
    for site in sites {
        grouped.entry(site.name.clone()).or_default().push(site);
    }

    let mut names: Vec<String> = grouped.keys().cloned().collect();
    names.sort();

    names
        .into_iter()
        .map(|target| {
            let mut callee_sites = grouped.remove(&target).unwrap_or_default();
            callee_sites.sort_by(|a, b| {
                (a.start_line, a.end_line, a.qualifier.as_deref().unwrap_or(""))
                    .cmp(&(b.start_line, b.end_line, b.qualifier.as_deref().unwrap_or("")))
            });
            callee_sites.dedup_by(|a, b| {
                a.start_line == b.start_line
                    && a.end_line == b.end_line
                    && a.qualifier == b.qualifier
            });

            let mut qualifiers: Vec<String> = Vec::new();
            for site in &callee_sites {
                if let Some(q) = &site.qualifier {
                    if !qualifiers.contains(q) {
                        qualifiers.push(q.clone());
                    }
                }
            }

            let mut meta = HashMap::new();
            meta.insert("reference_kind".to_string(), json!("calls"));
            meta.insert(
                "enclosing_symbol".to_string(),
                json!(enclosing_qname),
            );
            if let Some(first) = callee_sites.first() {
                meta.insert("start_line".to_string(), json!(first.start_line));
                meta.insert("end_line".to_string(), json!(first.end_line));
            }
            if let Some(first_q) = qualifiers.first() {
                meta.insert("qualifier".to_string(), json!(first_q));
            }
            if qualifiers.len() > 1 {
                meta.insert("qualifiers".to_string(), json!(qualifiers));
            }

            let site_entries: Vec<Value> = callee_sites
                .into_iter()
                .take(MAX_SITES_PER_EDGE)
                .map(|site| {
                    let mut entry = serde_json::Map::new();
                    entry.insert("start_line".to_string(), json!(site.start_line));
                    entry.insert("end_line".to_string(), json!(site.end_line));
                    entry.insert("reference_kind".to_string(), json!("calls"));
                    if let Some(q) = site.qualifier {
                        entry.insert("qualifier".to_string(), json!(q));
                    }
                    Value::Object(entry)
                })
                .collect();
            meta.insert("sites".to_string(), Value::Array(site_entries));

            ParsedEdge::with_metadata(enclosing_qname.to_string(), target, "calls", meta)
        })
        .collect()
}

/// Extract unique callee names from a function/method body.
pub fn extract_calls(
    node: &Node,
    source: &[u8],
    language: &str,
    blocklist: &HashSet<&str>,
) -> HashSet<String> {
    extract_call_sites(node, source, language, blocklist)
        .into_iter()
        .map(|site| site.name)
        .collect()
}

/// A code entity extracted from a source file via tree-sitter.
///
/// Represents a discrete unit of code (class, function, method, import, etc.)
/// with enough structural metadata to build a navigable knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedEntity {
    /// The [`NodeType`] classification (Class, Function, Method, etc.).
    pub kind: NodeType,
    /// Simple name of the entity (e.g. "MyClass", "process").
    pub name: String,
    /// Dot-separated scope path (e.g. "MyModule.MyClass.process").
    pub qualified_name: String,
    /// 0-based first line of the entity in the source file.
    pub start_line: usize,
    /// 0-based last line of the entity in the source file.
    pub end_line: usize,
    /// Type signature or parameter list for functions/methods.
    pub signature: String,
    /// Documentation string if present.
    pub docstring: String,
    /// The raw source text of the entity (capped for storage).
    pub raw_text: String,
    /// Qualified name of the containing entity, if any.
    pub parent_qualified_name: Option<String>,
    /// Base classes/protocols for class-like entities.
    pub bases: Vec<String>,
    /// Import targets for import statements.
    pub imports: Vec<String>,
    /// McCabe cyclomatic complexity for functions/methods (1 + decision points).
    /// `None` for non-function entities (classes, imports, variables, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cyclomatic_complexity: Option<u32>,
}

impl ParsedEntity {
    /// Generate the text to be embedded for semantic search.
    ///
    /// Different entity kinds emphasize different parts of their
    /// definition to produce the most useful embedding for retrieval.
    pub fn embed_text(&self) -> String {
        match self.kind {
            NodeType::File => format!("file: {}", self.qualified_name),
            NodeType::Class => {
                let doc = if !self.docstring.is_empty() {
                    let truncated = truncate_str(&self.docstring, DOCSTRING_LIMIT);
                    format!(": {truncated}")
                } else {
                    String::new()
                };
                format!("class {}{doc}", self.name)
            }
            NodeType::Function | NodeType::Method => {
                let signature = truncate_str(&self.signature, SIGNATURE_LIMIT);
                let doc = if !self.docstring.is_empty() {
                    let truncated = truncate_str(&self.docstring, DOCSTRING_LIMIT);
                    format!(": {truncated}")
                } else {
                    String::new()
                };
                format!("{} {} {signature}{doc}", self.kind, self.qualified_name)
            }
            NodeType::Variable | NodeType::Constant => {
                let signature = truncate_str(&self.signature, SIGNATURE_LIMIT);
                format!("{} {}: {signature}", self.kind, self.qualified_name)
            }
            NodeType::Import => format!("import {}", self.name),
            NodeType::TypeAlias => {
                let truncated = truncate_str(&self.raw_text, RAW_TEXT_SMALL_LIMIT);
                format!("type {}: {truncated}", self.name)
            }
            _ => format!("{} {}", self.kind, self.name),
        }
    }
}

/// A relationship between two parsed entities.
///
/// Source and target are qualified names that will be resolved to
/// node IDs during the ingestion pipeline's edge resolution pass.
/// The relationship type is a free-form string — well-known types like
/// `"contains"` and `"inherits"` are defined as constants in
/// [`scs_core::node_types::RelationshipType`], but any string is accepted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedEdge {
    pub source_qualified_name: String,
    pub target_qualified_name: String,
    pub relationship: String,
    pub weight: f64,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, Value>,
}

impl ParsedEdge {
    /// Create a new edge with default weight of 1.0.
    pub fn new(source: String, target: String, relationship: &str) -> Self {
        Self {
            source_qualified_name: source,
            target_qualified_name: target,
            relationship: relationship.to_string(),
            weight: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Create a new edge with explicit metadata attributes.
    pub fn with_metadata(
        source: String,
        target: String,
        relationship: &str,
        metadata: HashMap<String, Value>,
    ) -> Self {
        Self {
            source_qualified_name: source,
            target_qualified_name: target,
            relationship: relationship.to_string(),
            weight: 1.0,
            metadata,
        }
    }
}

/// Trait for language-specific AST parsers.
///
/// Each implementation uses tree-sitter to parse source code into
/// a concrete syntax tree, then walks the tree to extract entities
/// and relationships.
pub trait LanguageParser: Send + Sync {
    /// Parse source code and extract entities and edges.
    ///
    /// # Arguments
    /// * `source` — The raw source code text.
    /// * `file_path` — Relative path of the file (used for qualified names).
    fn parse(&self, source: &str, file_path: &str) -> (Vec<ParsedEntity>, Vec<ParsedEdge>);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity_with_signature(kind: NodeType, signature: String) -> ParsedEntity {
        ParsedEntity {
            kind,
            name: "item".to_string(),
            qualified_name: "module.item".to_string(),
            start_line: 0,
            end_line: 0,
            signature,
            docstring: String::new(),
            raw_text: String::new(),
            parent_qualified_name: None,
            bases: Vec::new(),
            imports: Vec::new(),
            cyclomatic_complexity: None,
        }
    }

    #[test]
    fn embed_text_bounds_function_signature_without_splitting_unicode() {
        let entity = entity_with_signature(NodeType::Function, "型".repeat(300));
        let expected_signature = "型".repeat(SIGNATURE_LIMIT / "型".len());

        assert_eq!(
            entity.embed_text(),
            format!("function module.item {expected_signature}")
        );
    }

    #[test]
    fn embed_text_bounds_variable_signature() {
        let entity = entity_with_signature(NodeType::Variable, "x".repeat(1_000));

        assert_eq!(
            entity.embed_text(),
            format!("variable module.item: {}", "x".repeat(SIGNATURE_LIMIT))
        );
    }

    #[test]
    fn truncate_str_ascii() {
        assert_eq!(truncate_str("hello", 3), "hel");
        assert_eq!(truncate_str("hello", 10), "hello");
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    #[test]
    fn truncate_str_multibyte_boundary() {
        // '─' is 3 bytes (E2 94 80). Truncating at byte 1 or 2 into
        // the character must not panic — it should back up.
        let s = "ab─cd";
        assert_eq!(truncate_str(s, 3), "ab");
        assert_eq!(truncate_str(s, 4), "ab");
        assert_eq!(truncate_str(s, 5), "ab─");
        assert_eq!(truncate_str(s, 6), "ab─c");
    }

    #[test]
    fn truncate_str_emoji() {
        // '🔥' is 4 bytes. Cutting at byte 1–3 must back up.
        let s = "x🔥y";
        assert_eq!(truncate_str(s, 1), "x");
        assert_eq!(truncate_str(s, 2), "x");
        assert_eq!(truncate_str(s, 4), "x");
        assert_eq!(truncate_str(s, 5), "x🔥");
        assert_eq!(truncate_str(s, 6), "x🔥y");
    }

    #[test]
    fn python_parser_handles_multibyte_raw_text() {
        // Regression: the old `&raw[..raw.len().min(1024)]` panicked when
        // byte 1024 landed inside a multi-byte character like '─'.
        let parser = python::PythonParser::new();
        // Build a class whose docstring contains multi-byte chars near the 1024 boundary.
        let filler = "─".repeat(500); // 500 × 3 bytes = 1500 bytes
        let source = format!("class Foo:\n    \"\"\"{filler}\"\"\"\n    pass\n");
        // Should not panic.
        let (entities, _) = parser.parse(&source, "test.py");
        let cls = entities.iter().find(|e| e.name == "Foo").unwrap();
        assert!(cls.raw_text.len() <= RAW_TEXT_LIMIT);
        assert!(cls.raw_text.is_char_boundary(cls.raw_text.len()));
    }
}
