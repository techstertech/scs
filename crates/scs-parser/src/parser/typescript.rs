//! TypeScript/JavaScript AST extraction via tree-sitter.
//!
//! Extracts classes, functions, interfaces, type aliases, imports,
//! and constants from TypeScript, TSX, JavaScript, and JSX files.
//! Uses a scope stack for qualified name construction.

use std::collections::HashSet;

use tree_sitter::{Node, Parser};

use scs_core::node_types::NodeType;

use super::{count_complexity, extract_calls, LanguageParser, ParsedEdge, ParsedEntity};

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
        edges.push(ParsedEdge::new(
            module_name.to_string(),
            import_path.to_string(),
            "imports",
        ));
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
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.walk_node(child, source, module_name, entities, edges, scope_stack);
        }
    }

    fn walk_node(
        &self,
        child: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
    ) {
        match child.kind() {
            "class_declaration" => {
                self.handle_class(child, source, module_name, entities, edges, scope_stack);
            }
            "function_declaration" | "generator_function_declaration" => {
                self.handle_function(child, source, module_name, entities, edges, scope_stack);
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
                self.handle_variable(child, source, module_name, entities, edges, scope_stack);
            }
            "assignment_expression" if scope_stack.len() == 1 => {
                self.handle_assignment(child, source, module_name, entities, edges, scope_stack);
            }
            "call_expression" => {
                if let Some(import_path) = extract_module_import_call(&child, source) {
                    self.emit_import(child, &import_path, module_name, entities, edges);
                } else if let Some(func) = child.child_by_field_name("function") {
                    self.walk_node(func, source, module_name, entities, edges, scope_stack);
                }
                if let Some(args) = child.child_by_field_name("arguments") {
                    self.walk(args, source, module_name, entities, edges, scope_stack);
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
                }
            }
            "export_statement" => {
                if let Some(src) = child.child_by_field_name("source") {
                    if let Some(import_path) = extract_string_literal(&src, source) {
                        self.emit_import(child, &import_path, module_name, entities, edges);
                    }
                }
                // Unwrap export and process the inner declaration.
                self.walk(child, source, module_name, entities, edges, scope_stack);
            }
            "method_definition" | "public_field_definition" => {
                self.handle_method(child, source, module_name, entities, edges, scope_stack);
            }
            _ => {
                // Don't recurse into function bodies.
                if child.child_count() > 0
                    && child.kind() != "statement_block"
                    && child.kind() != "function_body"
                {
                    self.walk(child, source, module_name, entities, edges, scope_stack);
                }
            }
        }
    }

    fn handle_class(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
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

        // Extract heritage (extends/implements).
        let mut bases = Vec::new();
        let mut cursor2 = node.walk();
        for child in node.children(&mut cursor2) {
            if matches!(
                child.kind(),
                "class_heritage" | "extends_clause" | "implements_clause"
            ) {
                let mut sub_cursor = child.walk();
                for sub in child.children(&mut sub_cursor) {
                    if sub.kind() == "type_identifier" || sub.kind() == "identifier" {
                        bases.push(get_text(&sub, source).to_string());
                    }
                }
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

        for base in &bases {
            edges.push(ParsedEdge::new(qualified.clone(), base.clone(), "inherits"));
        }

        // Recurse into class body.
        if let Some(body) = find_child_by_kind(&node, "class_body") {
            let mut new_scope = scope_stack.to_vec();
            new_scope.push(name);
            self.walk(body, source, module_name, entities, edges, &new_scope);
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
        let params = fn_node
            .child_by_field_name("parameters")
            .or_else(|| fn_node.child_by_field_name("parameter"))
            .map(|p| get_text(&p, source).to_string())
            .unwrap_or_default();
        let raw = get_text(&fn_node, source);
        let complexity = count_complexity(&fn_node, source, &self.language_name);

        entities.push(ParsedEntity {
            kind,
            name,
            qualified_name: qualified.clone(),
            start_line: fn_node.start_position().row,
            end_line: fn_node.end_position().row,
            signature: params,
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

        if let Some(body) = fn_node.child_by_field_name("body") {
            let blocklist: HashSet<&str> = TS_BUILTIN_CALLS.iter().copied().collect();
            let callees = extract_calls(&body, source, &self.language_name, &blocklist);
            for callee in callees {
                edges.push(ParsedEdge::new(qualified.clone(), callee, "calls"));
            }
            self.extract_module_imports_in_subtree(body, source, module_name, entities, edges);
        }
    }

    fn handle_function(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
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

        let params = find_child_by_kind(&node, "formal_parameters")
            .map(|p| get_text(&p, source).to_string())
            .unwrap_or_default();

        let raw = get_text(&node, source);
        let complexity = count_complexity(&node, source, &self.language_name);
        entities.push(ParsedEntity {
            kind,
            name,
            qualified_name: qualified.clone(),
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            signature: params,
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

        // CALLS edges from this function to callees in its body, plus lazy imports.
        if let Some(body) = find_child_by_kind(&node, "statement_block") {
            let blocklist: HashSet<&str> = TS_BUILTIN_CALLS.iter().copied().collect();
            let callees = extract_calls(&body, source, &self.language_name, &blocklist);
            for callee in callees {
                edges.push(ParsedEdge::new(qualified.clone(), callee, "calls"));
            }
            self.extract_module_imports_in_subtree(body, source, module_name, entities, edges);
        }
    }

    fn handle_method(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
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
        let params = find_child_by_kind(&node, "formal_parameters")
            .map(|p| get_text(&p, source).to_string())
            .unwrap_or_default();

        let raw = get_text(&node, source);
        let complexity = count_complexity(&node, source, &self.language_name);
        entities.push(ParsedEntity {
            kind: NodeType::Method,
            name,
            qualified_name: qualified.clone(),
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            signature: params,
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

        // CALLS edges from this method to callees in its body, plus lazy imports.
        if let Some(body) = find_child_by_kind(&node, "statement_block") {
            let blocklist: HashSet<&str> = TS_BUILTIN_CALLS.iter().copied().collect();
            let callees = extract_calls(&body, source, &self.language_name, &blocklist);
            for callee in callees {
                edges.push(ParsedEdge::new(qualified.clone(), callee, "calls"));
            }
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

        entities.push(ParsedEntity {
            kind,
            name,
            qualified_name: qualified.clone(),
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            raw_text: super::truncate_str(raw, super::RAW_TEXT_SMALL_LIMIT).to_string(),
            parent_qualified_name: Some(scope_stack.join(".")),
            signature: String::new(),
            docstring: get_doc_comment_for_declaration(&node, source),
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

    fn handle_import(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
    ) {
        // Extract module specifier from ESM `source` or TS `import x = require(...)`.
        let source_node = node.child_by_field_name("source").or_else(|| {
            find_child_by_kind(&node, "import_require_clause")
                .and_then(|clause| clause.child_by_field_name("source"))
        });
        if let Some(src) = source_node {
            if let Some(import_path) = extract_string_literal(&src, source) {
                self.emit_import(node, &import_path, module_name, entities, edges);
                return;
            }
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if let Some(import_path) = extract_string_literal(&child, source) {
                self.emit_import(node, &import_path, module_name, entities, edges);
                break;
            }
        }
    }

    #[allow(clippy::ptr_arg)] // Signature mirrors walk() callback pattern.
    fn handle_variable(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
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
                        );
                    } else {
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

                self.extract_module_imports_in_subtree(val, source, module_name, entities, edges);
            }

            if ident_name.len() < 2 {
                continue;
            }

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
    }

    fn handle_assignment(
        &self,
        node: Node,
        source: &[u8],
        module_name: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        scope_stack: &[String],
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

                self.walk_node(right, source, module_name, entities, edges, scope_stack);
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
                                    );
                                    continue;
                                }
                            }
                        }
                        self.walk_node(item, source, module_name, entities, edges, scope_stack);
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
                    self.walk_node(right, source, module_name, entities, edges, scope_stack);
                }
            }
            None => {
                self.walk_node(right, source, module_name, entities, edges, scope_stack);
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
        self.walk(
            tree.root_node(),
            source_bytes,
            &module_name,
            &mut entities,
            &mut edges,
            &scope_stack,
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
}
