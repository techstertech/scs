//! Java AST extraction via tree-sitter.
//!
//! Extracts classes, interfaces, enums, records, annotation types,
//! methods, constructors, fields, constants, imports, package/module
//! descriptors, inheritance/implementation edges, call edges, Javadoc,
//! and cyclomatic complexity from `.java` source files.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use tree_sitter::{Node, Parser};

use scs_core::node_types::NodeType;

use super::{
    count_complexity, truncate_str, LanguageParser, ParsedEdge, ParsedEntity, RAW_TEXT_LIMIT,
    RAW_TEXT_SMALL_LIMIT,
};

const JAVA_BUILTIN_CALLS: &[&str] = &[
    "println",
    "print",
    "printf",
    "toString",
    "equals",
    "hashCode",
    "getClass",
    "notify",
    "notifyAll",
    "wait",
    "clone",
    "finalize",
];

fn is_java_lang_builtin_type(name: &str) -> bool {
    matches!(
        name,
        "Object"
            | "String"
            | "Throwable"
            | "Exception"
            | "RuntimeException"
            | "Error"
            | "Record"
            | "Enum"
            | "Comparable"
            | "Iterable"
            | "AutoCloseable"
            | "Cloneable"
            | "Runnable"
            | "Serializable"
            | "System"
            | "Math"
            | "Integer"
            | "Long"
            | "Double"
            | "Float"
            | "Boolean"
            | "Character"
            | "Byte"
            | "Short"
            | "Void"
            | "Class"
            | "Thread"
            | "Runtime"
    )
}

#[derive(Default)]
pub struct JavaParser;

impl JavaParser {
    pub fn new() -> Self {
        Self
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_import(
        &self,
        node: Node,
        source: &[u8],
        file_qualified: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
        imported_types: &mut HashMap<String, String>,
        static_imports: &mut HashMap<String, String>,
    ) {
        let mut is_static = false;
        let mut is_wildcard = false;
        let mut path_text = String::new();

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "static" => is_static = true,
                "asterisk" => is_wildcard = true,
                "scoped_identifier" | "identifier" => {
                    path_text = text(child, source).trim().to_string();
                }
                _ => {}
            }
        }

        if path_text.is_empty() {
            return;
        }

        let import_path = if is_wildcard {
            format!("{path_text}.*")
        } else {
            path_text.clone()
        };
        let name = if is_wildcard {
            import_path.clone()
        } else {
            path_text
                .rsplit('.')
                .next()
                .unwrap_or(&path_text)
                .to_string()
        };

        if !is_wildcard {
            if is_static {
                static_imports.insert(name.clone(), path_text.clone());
            } else {
                imported_types.insert(name.clone(), path_text.clone());
            }
        }

        let qualified = format!("{file_qualified}.import.{import_path}");
        let signature = truncate_str(text(node, source).trim(), RAW_TEXT_SMALL_LIMIT).to_string();

        entities.push(entity(
            NodeType::Import,
            name,
            qualified.clone(),
            node,
            source,
            Some(file_qualified.to_string()),
            signature,
            String::new(),
            vec![],
            vec![import_path],
            None,
        ));
        edges.push(ParsedEdge::new(
            file_qualified.to_string(),
            qualified,
            "contains",
        ));
        edges.push(ParsedEdge::new(
            file_qualified.to_string(),
            path_text,
            "imports",
        ));
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_type_declaration(
        &self,
        node: Node,
        source: &[u8],
        owner_edge_source: &str,
        type_namespace: &str,
        package_name: &str,
        imported_types: &HashMap<String, String>,
        static_imports: &HashMap<String, String>,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
    ) {
        let Some(name_node) = node.child_by_field_name("name") else {
            return;
        };
        let name = text(name_node, source).trim().to_string();
        if name.is_empty() {
            return;
        }

        let qualified = if type_namespace.is_empty() {
            name.clone()
        } else {
            format!("{type_namespace}.{name}")
        };

        let mut bases = Vec::new();
        let mut inherit_targets = Vec::new();
        let mut implement_targets = Vec::new();

        match node.kind() {
            "class_declaration" => {
                if let Some(super_node) = node.child_by_field_name("superclass") {
                    if let Some(raw_base) = extract_first_type_in_container(super_node, source) {
                        bases.push(simple_type_name(&raw_base));
                        inherit_targets.push(resolve_type_name(
                            &raw_base,
                            package_name,
                            imported_types,
                        ));
                    }
                }
                if let Some(interfaces_node) = node.child_by_field_name("interfaces") {
                    for raw_iface in extract_type_list(interfaces_node, source) {
                        bases.push(simple_type_name(&raw_iface));
                        implement_targets.push(resolve_type_name(
                            &raw_iface,
                            package_name,
                            imported_types,
                        ));
                    }
                }
            }
            "interface_declaration" => {
                let mut cursor = node.walk();
                for child in node.named_children(&mut cursor) {
                    if child.kind() == "extends_interfaces" {
                        for raw_iface in extract_type_list(child, source) {
                            bases.push(simple_type_name(&raw_iface));
                            inherit_targets.push(resolve_type_name(
                                &raw_iface,
                                package_name,
                                imported_types,
                            ));
                        }
                    }
                }
            }
            "enum_declaration" | "record_declaration" => {
                if let Some(interfaces_node) = node.child_by_field_name("interfaces") {
                    for raw_iface in extract_type_list(interfaces_node, source) {
                        bases.push(simple_type_name(&raw_iface));
                        implement_targets.push(resolve_type_name(
                            &raw_iface,
                            package_name,
                            imported_types,
                        ));
                    }
                }
            }
            _ => {}
        }

        let signature =
            truncate_str(&signature_before_body(node, source), RAW_TEXT_SMALL_LIMIT).to_string();

        entities.push(entity(
            NodeType::Class,
            name,
            qualified.clone(),
            node,
            source,
            Some(owner_edge_source.to_string()),
            signature,
            javadoc(node, source),
            bases,
            vec![],
            None,
        ));
        edges.push(ParsedEdge::new(
            owner_edge_source.to_string(),
            qualified.clone(),
            "contains",
        ));
        for target in inherit_targets {
            edges.push(ParsedEdge::new(qualified.clone(), target, "inherits"));
        }
        for target in implement_targets {
            edges.push(ParsedEdge::new(qualified.clone(), target, "implements"));
        }

        if node.kind() == "record_declaration" {
            if let Some(params_node) = node.child_by_field_name("parameters") {
                self.handle_record_components(params_node, source, &qualified, entities, edges);
            }
        }

        if let Some(body) = node.child_by_field_name("body") {
            match body.kind() {
                "enum_body" => self.walk_enum_body(
                    body,
                    source,
                    &qualified,
                    package_name,
                    imported_types,
                    static_imports,
                    entities,
                    edges,
                ),
                _ => self.walk_type_members(
                    body,
                    source,
                    &qualified,
                    package_name,
                    imported_types,
                    static_imports,
                    entities,
                    edges,
                ),
            }
        }
    }

    fn handle_record_components(
        &self,
        params_node: Node,
        source: &[u8],
        owner_type: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
    ) {
        let mut cursor = params_node.walk();
        for child in params_node.named_children(&mut cursor) {
            if child.kind() != "formal_parameter" {
                continue;
            }
            let Some(name_node) = child.child_by_field_name("name") else {
                continue;
            };
            let name = text(name_node, source).trim().to_string();
            if name.is_empty() {
                continue;
            }
            let qualified = format!("{owner_type}.{name}");
            let signature =
                truncate_str(text(child, source).trim(), RAW_TEXT_SMALL_LIMIT).to_string();
            entities.push(entity(
                NodeType::Variable,
                name,
                qualified.clone(),
                child,
                source,
                Some(owner_type.to_string()),
                signature,
                String::new(),
                vec![],
                vec![],
                None,
            ));
            edges.push(ParsedEdge::new(
                owner_type.to_string(),
                qualified,
                "contains",
            ));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_enum_body(
        &self,
        body: Node,
        source: &[u8],
        owner_type: &str,
        package_name: &str,
        imported_types: &HashMap<String, String>,
        static_imports: &HashMap<String, String>,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
    ) {
        let mut cursor = body.walk();
        for child in body.named_children(&mut cursor) {
            match child.kind() {
                "enum_constant" => {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        let name = text(name_node, source).trim().to_string();
                        if !name.is_empty() {
                            let qualified = format!("{owner_type}.{name}");
                            let signature =
                                truncate_str(text(child, source).trim(), RAW_TEXT_SMALL_LIMIT)
                                    .to_string();
                            entities.push(entity(
                                NodeType::Constant,
                                name,
                                qualified.clone(),
                                child,
                                source,
                                Some(owner_type.to_string()),
                                signature,
                                javadoc(child, source),
                                vec![],
                                vec![],
                                None,
                            ));
                            edges.push(ParsedEdge::new(
                                owner_type.to_string(),
                                qualified,
                                "contains",
                            ));
                        }
                    }
                }
                "enum_body_declarations" => {
                    self.walk_type_members(
                        child,
                        source,
                        owner_type,
                        package_name,
                        imported_types,
                        static_imports,
                        entities,
                        edges,
                    );
                }
                _ => {}
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_type_members(
        &self,
        body: Node,
        source: &[u8],
        owner_type: &str,
        package_name: &str,
        imported_types: &HashMap<String, String>,
        static_imports: &HashMap<String, String>,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
    ) {
        let mut cursor = body.walk();
        for child in body.named_children(&mut cursor) {
            match child.kind() {
                "class_declaration"
                | "interface_declaration"
                | "enum_declaration"
                | "record_declaration"
                | "annotation_type_declaration" => {
                    self.handle_type_declaration(
                        child,
                        source,
                        owner_type,
                        owner_type,
                        package_name,
                        imported_types,
                        static_imports,
                        entities,
                        edges,
                    );
                }
                "method_declaration"
                | "constructor_declaration"
                | "compact_constructor_declaration"
                | "annotation_type_element_declaration" => {
                    self.handle_callable(
                        child,
                        source,
                        owner_type,
                        package_name,
                        imported_types,
                        static_imports,
                        entities,
                        edges,
                    );
                }
                "field_declaration" | "constant_declaration" => {
                    self.handle_field(child, source, owner_type, entities, edges);
                }
                _ => {}
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_callable(
        &self,
        node: Node,
        source: &[u8],
        owner_type: &str,
        package_name: &str,
        imported_types: &HashMap<String, String>,
        static_imports: &HashMap<String, String>,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
    ) {
        let Some(name_node) = node.child_by_field_name("name") else {
            return;
        };
        let name = text(name_node, source).trim().to_string();
        if name.is_empty() {
            return;
        }

        let qualified = format!("{owner_type}.{name}");
        let body = node.child_by_field_name("body");
        let complexity = body.map(|b| count_complexity(&b, source, "java")).or(Some(1));
        let raw_signature = signature_before_body(node, source);
        let signature = truncate_str(
            raw_signature.trim_end_matches(';').trim(),
            RAW_TEXT_SMALL_LIMIT,
        )
        .to_string();

        entities.push(entity(
            NodeType::Method,
            name,
            qualified.clone(),
            node,
            source,
            Some(owner_type.to_string()),
            signature,
            javadoc(node, source),
            vec![],
            vec![],
            complexity,
        ));
        edges.push(ParsedEdge::new(
            owner_type.to_string(),
            qualified.clone(),
            "contains",
        ));

        if let Some(body_node) = body {
            let mut callees: Vec<String> = java_calls(
                body_node,
                source,
                owner_type,
                package_name,
                imported_types,
                static_imports,
            )
            .into_iter()
            .collect();
            callees.sort();
            for target in callees {
                edges.push(ParsedEdge::new(qualified.clone(), target, "calls"));
            }
        }
    }

    fn handle_field(
        &self,
        node: Node,
        source: &[u8],
        owner_type: &str,
        entities: &mut Vec<ParsedEntity>,
        edges: &mut Vec<ParsedEdge>,
    ) {
        let is_constant = if node.kind() == "constant_declaration" {
            true
        } else {
            has_static_final_modifiers(node, source)
        };
        let kind = if is_constant {
            NodeType::Constant
        } else {
            NodeType::Variable
        };

        let signature = truncate_str(
            text(node, source).trim().trim_end_matches(';').trim(),
            RAW_TEXT_SMALL_LIMIT,
        )
        .to_string();
        let doc = javadoc(node, source);

        for declarator in children_by_field(node, "declarator") {
            let Some(name_node) = declarator.child_by_field_name("name") else {
                continue;
            };
            let name = text(name_node, source).trim().to_string();
            if name.is_empty() {
                continue;
            }
            let qualified = format!("{owner_type}.{name}");
            entities.push(entity(
                kind,
                name,
                qualified.clone(),
                node,
                source,
                Some(owner_type.to_string()),
                signature.clone(),
                doc.clone(),
                vec![],
                vec![],
                None,
            ));
            edges.push(ParsedEdge::new(
                owner_type.to_string(),
                qualified,
                "contains",
            ));
        }
    }
}

impl LanguageParser for JavaParser {
    fn parse(&self, source: &str, file_path: &str) -> (Vec<ParsedEntity>, Vec<ParsedEdge>) {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_java::LANGUAGE.into())
            .expect("failed to load Java grammar");

        let Some(tree) = parser.parse(source, None) else {
            return (Vec::new(), Vec::new());
        };

        let source_bytes = source.as_bytes();
        let root = tree.root_node();
        let file_qualified = file_path.to_string();
        let is_package_info =
            Path::new(file_path).file_name().and_then(|n| n.to_str()) == Some("package-info.java");

        let mut entities = vec![entity(
            NodeType::File,
            file_path.to_string(),
            file_qualified.clone(),
            root,
            source_bytes,
            None,
            String::new(),
            String::new(),
            vec![],
            vec![],
            None,
        )];
        // Override File raw_text to match the cross-language file summary convention.
        entities[0].raw_text = format!("file: {file_path}");
        entities[0].end_line = source.lines().count();

        let mut edges = Vec::new();
        let mut package_name = String::new();
        let mut imported_types = HashMap::new();
        let mut static_imports = HashMap::new();

        // Pass 1: package/module declarations and all imports (so type & static maps are complete).
        let mut cursor = root.walk();
        for child in root.named_children(&mut cursor) {
            match child.kind() {
                "package_declaration" => {
                    if let Some(pkg) = extract_package_name(child, source_bytes) {
                        package_name = pkg.clone();
                        if is_package_info {
                            let signature =
                                truncate_str(text(child, source_bytes).trim(), RAW_TEXT_SMALL_LIMIT)
                                    .to_string();
                            entities.push(entity(
                                NodeType::Module,
                                pkg.clone(),
                                pkg.clone(),
                                child,
                                source_bytes,
                                Some(file_qualified.clone()),
                                signature,
                                javadoc(child, source_bytes),
                                vec![],
                                vec![],
                                None,
                            ));
                            edges.push(ParsedEdge::new(file_qualified.clone(), pkg, "contains"));
                        }
                    }
                }
                "module_declaration" => {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        let module_name = text(name_node, source_bytes).trim().to_string();
                        if !module_name.is_empty() {
                            let signature = truncate_str(
                                &signature_before_body(child, source_bytes),
                                RAW_TEXT_SMALL_LIMIT,
                            )
                            .to_string();
                            entities.push(entity(
                                NodeType::Module,
                                module_name.clone(),
                                module_name.clone(),
                                child,
                                source_bytes,
                                Some(file_qualified.clone()),
                                signature,
                                javadoc(child, source_bytes),
                                vec![],
                                vec![],
                                None,
                            ));
                            edges.push(ParsedEdge::new(
                                file_qualified.clone(),
                                module_name,
                                "contains",
                            ));
                        }
                    }
                }
                "import_declaration" => {
                    self.handle_import(
                        child,
                        source_bytes,
                        &file_qualified,
                        &mut entities,
                        &mut edges,
                        &mut imported_types,
                        &mut static_imports,
                    );
                }
                _ => {}
            }
        }

        // Pass 2: top-level type declarations.
        let mut cursor2 = root.walk();
        for child in root.named_children(&mut cursor2) {
            if matches!(
                child.kind(),
                "class_declaration"
                    | "interface_declaration"
                    | "enum_declaration"
                    | "record_declaration"
                    | "annotation_type_declaration"
            ) {
                self.handle_type_declaration(
                    child,
                    source_bytes,
                    &file_qualified,
                    &package_name,
                    &package_name,
                    &imported_types,
                    &static_imports,
                    &mut entities,
                    &mut edges,
                );
            }
        }

        (entities, edges)
    }
}

#[allow(clippy::too_many_arguments)]
fn entity(
    kind: NodeType,
    name: String,
    qualified_name: String,
    node: Node,
    source: &[u8],
    parent_qualified_name: Option<String>,
    signature: String,
    docstring: String,
    bases: Vec<String>,
    imports: Vec<String>,
    cyclomatic_complexity: Option<u32>,
) -> ParsedEntity {
    let limit = match kind {
        NodeType::Class | NodeType::Function | NodeType::Method | NodeType::File => RAW_TEXT_LIMIT,
        _ => RAW_TEXT_SMALL_LIMIT,
    };
    ParsedEntity {
        kind,
        name,
        qualified_name,
        start_line: node.start_position().row,
        end_line: node.end_position().row,
        signature,
        docstring,
        raw_text: truncate_str(text(node, source), limit).to_string(),
        parent_qualified_name,
        bases,
        imports,
        cyclomatic_complexity,
    }
}

fn text<'a>(node: Node, source: &'a [u8]) -> &'a str {
    std::str::from_utf8(&source[node.byte_range()]).unwrap_or("")
}

fn children_by_field<'tree>(node: Node<'tree>, field: &str) -> Vec<Node<'tree>> {
    let Some(field_id) = node.language().field_id_for_name(field) else {
        return vec![];
    };
    let mut cursor = node.walk();
    node.children_by_field_id(field_id, &mut cursor).collect()
}

fn signature_before_body(node: Node, source: &[u8]) -> String {
    let end = node
        .child_by_field_name("body")
        .map(|body| body.start_byte())
        .unwrap_or(node.end_byte());
    std::str::from_utf8(&source[node.start_byte()..end])
        .unwrap_or("")
        .trim()
        .to_string()
}

fn extract_package_name(node: Node, source: &[u8]) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if matches!(child.kind(), "scoped_identifier" | "identifier") {
            let pkg = text(child, source).trim();
            if !pkg.is_empty() {
                return Some(pkg.to_string());
            }
        }
    }
    None
}

fn has_static_final_modifiers(node: Node, source: &[u8]) -> bool {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "modifiers" {
            let tokens: Vec<&str> = text(child, source).split_whitespace().collect();
            return tokens.contains(&"static") && tokens.contains(&"final");
        }
    }
    false
}

fn javadoc(node: Node, source: &[u8]) -> String {
    let Some(comment) = node.prev_named_sibling() else {
        return String::new();
    };
    if comment.kind() != "block_comment" {
        return String::new();
    }
    let raw = text(comment, source).trim();
    if !(raw.starts_with("/**") && raw.ends_with("*/") && raw.len() >= 5) {
        return String::new();
    }
    let inner = &raw[3..raw.len() - 2];
    inner
        .lines()
        .map(|line| line.trim().trim_start_matches('*').trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn normalize_java_type_name(node: Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "type_identifier" => {
            let name = text(node, source).trim();
            (!name.is_empty()).then(|| name.to_string())
        }
        "generic_type" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if matches!(child.kind(), "scoped_type_identifier" | "type_identifier") {
                    return normalize_java_type_name(child, source);
                }
            }
            None
        }
        "scoped_type_identifier" => {
            let mut parts = Vec::new();
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if matches!(
                    child.kind(),
                    "scoped_type_identifier" | "generic_type" | "type_identifier"
                ) {
                    if let Some(part) = normalize_java_type_name(child, source) {
                        parts.push(part);
                    }
                }
            }
            (!parts.is_empty()).then(|| parts.join("."))
        }
        _ => None,
    }
}

fn extract_first_type_in_container(node: Node, source: &[u8]) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if let Some(name) = normalize_java_type_name(child, source) {
            return Some(name);
        }
    }
    None
}

fn extract_type_list(container: Node, source: &[u8]) -> Vec<String> {
    let mut types = Vec::new();
    let mut cursor = container.walk();
    for child in container.named_children(&mut cursor) {
        if child.kind() == "type_list" {
            let mut list_cursor = child.walk();
            for item in child.named_children(&mut list_cursor) {
                if let Some(name) = normalize_java_type_name(item, source) {
                    types.push(name);
                }
            }
        } else if let Some(name) = normalize_java_type_name(child, source) {
            types.push(name);
        }
    }
    types
}

fn simple_type_name(raw: &str) -> String {
    raw.rsplit('.').next().unwrap_or(raw).to_string()
}

fn resolve_type_name(
    raw: &str,
    package_name: &str,
    imported_types: &HashMap<String, String>,
) -> String {
    let first = raw.split('.').next().unwrap_or("");
    if let Some(imported) = imported_types.get(first) {
        if let Some(rest) = raw.strip_prefix(first).filter(|r| r.starts_with('.')) {
            return format!("{imported}{rest}");
        }
        return imported.clone();
    }
    if !raw.contains('.') && !package_name.is_empty() && !is_java_lang_builtin_type(raw) {
        return format!("{package_name}.{raw}");
    }
    raw.to_string()
}

fn java_calls(
    body: Node,
    source: &[u8],
    owner_type: &str,
    package_name: &str,
    imported_types: &HashMap<String, String>,
    static_imports: &HashMap<String, String>,
) -> HashSet<String> {
    #[allow(clippy::too_many_arguments)]
    fn walk(
        node: Node,
        source: &[u8],
        owner_type: &str,
        package_name: &str,
        imported_types: &HashMap<String, String>,
        static_imports: &HashMap<String, String>,
        calls: &mut HashSet<String>,
    ) {
        match node.kind() {
            "method_invocation" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    let method_name = text(name_node, source).trim();
                    if !method_name.is_empty() && !JAVA_BUILTIN_CALLS.contains(&method_name) {
                        let target = match node.child_by_field_name("object") {
                            None => {
                                let mut c = node.walk();
                                let has_super =
                                    node.children(&mut c).any(|child| child.kind() == "super");
                                if has_super {
                                    method_name.to_string()
                                } else if let Some(fqcn) = static_imports.get(method_name) {
                                    fqcn.clone()
                                } else {
                                    format!("{owner_type}.{method_name}")
                                }
                            }
                            Some(obj) => match obj.kind() {
                                "this" => format!("{owner_type}.{method_name}"),
                                "identifier" => {
                                    let obj_text = text(obj, source).trim();
                                    if let Some(fqcn) = imported_types.get(obj_text) {
                                        format!("{fqcn}.{method_name}")
                                    } else if obj_text
                                        .chars()
                                        .next()
                                        .is_some_and(|ch| ch.is_ascii_uppercase())
                                        && !package_name.is_empty()
                                        && !is_java_lang_builtin_type(obj_text)
                                    {
                                        format!("{package_name}.{obj_text}.{method_name}")
                                    } else {
                                        method_name.to_string()
                                    }
                                }
                                "scoped_identifier" => {
                                    let scope = text(obj, source).trim();
                                    format!("{scope}.{method_name}")
                                }
                                _ => method_name.to_string(),
                            },
                        };
                        calls.insert(target);
                    }
                }
            }
            "object_creation_expression" => {
                if let Some(type_node) = node.child_by_field_name("type") {
                    if let Some(raw_type) = normalize_java_type_name(type_node, source) {
                        let resolved = resolve_type_name(&raw_type, package_name, imported_types);
                        if !is_java_lang_builtin_type(&resolved) {
                            calls.insert(resolved);
                        }
                    }
                }
            }
            _ => {}
        }

        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            walk(
                child,
                source,
                owner_type,
                package_name,
                imported_types,
                static_imports,
                calls,
            );
        }
    }

    let mut calls = HashSet::new();
    walk(
        body,
        source,
        owner_type,
        package_name,
        imported_types,
        static_imports,
        &mut calls,
    );
    calls
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPRESENTATIVE_JAVA: &str = r#"
package com.acme.service;

import com.acme.model.User;
import com.acme.model.*;
import static com.acme.util.Checks.verify;

/**
 * Service handling user workflows.
 */
public class UserService<T> extends BaseService<T> implements Identifiable, Auditable {
    public static final String VERSION = "1.0";
    private final UserRepository repo;
    private int count, retries;

    /** Construct a service. */
    public UserService(UserRepository repo) {
        this.repo = repo;
    }

    // Regular line comment should not attach as Javadoc.
    public User findUser(String id, boolean strict) {
        verify(id);
        if (strict && count > 0 || retries > 1) {
            for (int i = 0; i < count; i++) {
                helper();
            }
        } else {
            while (retries > 0) {
                retries--;
            }
        }
        try {
            System.out.println(id);
        } catch (RuntimeException ex) {
            return count > 0 ? new User(id) : null;
        }
        switch (count) {
            case 1:
                break;
            case 2:
                break;
            default:
                break;
        }
        return Checks.lookup(id);
    }

    private void helper() {
        for (String item : new String[] { "a" }) {
            repo.save(item);
        }
    }

    public static class NestedConfig {
        public void reset() {}
    }
}

interface Repository extends BaseRepository, Closeable {
    String DEFAULT_NAME = "default";
    User load(String id);
}

enum Status implements Labelled {
    ACTIVE("active"),
    INACTIVE("inactive");

    private final String label;

    Status(String label) {
        this.label = label;
    }

    public String getLabel() {
        return label;
    }
}

record UserRecord(String id, String name) implements Serializable {
    public UserRecord {
        verify(id);
    }
}

@interface Marker {
    String value() default "";
}
"#;

    #[test]
    fn extracts_full_java_surface() {
        let (entities, edges) = JavaParser::new().parse(
            REPRESENTATIVE_JAVA,
            "src/main/java/com/acme/service/UserService.java",
        );

        assert!(entities.iter().any(|e| {
            e.kind == NodeType::File
                && e.qualified_name == "src/main/java/com/acme/service/UserService.java"
        }));
        assert!(!entities.iter().any(|e| e.kind == NodeType::Module));

        assert_eq!(
            entities
                .iter()
                .filter(|e| e.kind == NodeType::Import)
                .count(),
            3
        );
        let import_targets: HashSet<_> = edges
            .iter()
            .filter(|e| e.relationship == "imports")
            .map(|e| e.target_qualified_name.as_str())
            .collect();
        assert_eq!(
            import_targets,
            HashSet::from([
                "com.acme.model.User",
                "com.acme.model",
                "com.acme.util.Checks.verify",
            ])
        );

        let user_service = entities
            .iter()
            .find(|e| e.kind == NodeType::Class && e.name == "UserService")
            .expect("UserService class");
        assert_eq!(user_service.qualified_name, "com.acme.service.UserService");
        assert!(user_service
            .docstring
            .contains("Service handling user workflows."));
        assert_eq!(
            user_service.bases,
            vec!["BaseService", "Identifiable", "Auditable"]
        );

        assert!(edges.iter().any(|e| {
            e.relationship == "inherits"
                && e.source_qualified_name == "com.acme.service.UserService"
                && e.target_qualified_name == "com.acme.service.BaseService"
        }));
        assert!(edges.iter().any(|e| {
            e.relationship == "implements"
                && e.source_qualified_name == "com.acme.service.UserService"
                && e.target_qualified_name == "com.acme.service.Identifiable"
        }));
        assert!(entities.iter().any(|e| {
            e.kind == NodeType::Class
                && e.qualified_name == "com.acme.service.UserService.NestedConfig"
        }));
        assert!(entities.iter().any(|e| {
            e.kind == NodeType::Method
                && e.qualified_name == "com.acme.service.UserService.NestedConfig.reset"
        }));

        assert!(entities.iter().any(|e| {
            e.kind == NodeType::Constant && e.qualified_name == "com.acme.service.UserService.VERSION"
        }));
        assert!(entities.iter().any(|e| {
            e.kind == NodeType::Variable && e.qualified_name == "com.acme.service.UserService.repo"
        }));
        assert!(entities.iter().any(|e| {
            e.kind == NodeType::Variable && e.qualified_name == "com.acme.service.UserService.count"
        }));
        assert!(entities.iter().any(|e| {
            e.kind == NodeType::Variable
                && e.qualified_name == "com.acme.service.UserService.retries"
        }));

        let ctor = entities
            .iter()
            .find(|e| {
                e.kind == NodeType::Method
                    && e.qualified_name == "com.acme.service.UserService.UserService"
            })
            .expect("constructor");
        assert!(ctor.docstring.contains("Construct a service."));

        let find_user = entities
            .iter()
            .find(|e| {
                e.kind == NodeType::Method
                    && e.qualified_name == "com.acme.service.UserService.findUser"
            })
            .expect("findUser");
        assert_eq!(find_user.docstring, "");
        // 1 + if + && + || + for + while + catch + ternary + 2 switch_case = 10
        assert_eq!(find_user.cyclomatic_complexity, Some(10));

        assert!(edges.iter().any(|e| {
            e.relationship == "calls"
                && e.source_qualified_name == "com.acme.service.UserService.findUser"
                && e.target_qualified_name == "com.acme.util.Checks.verify"
        }));
        assert!(edges.iter().any(|e| {
            e.relationship == "calls"
                && e.source_qualified_name == "com.acme.service.UserService.findUser"
                && e.target_qualified_name == "com.acme.service.UserService.helper"
        }));
        assert!(edges.iter().any(|e| {
            e.relationship == "calls"
                && e.source_qualified_name == "com.acme.service.UserService.findUser"
                && e.target_qualified_name == "com.acme.model.User"
        }));
        assert!(edges.iter().any(|e| {
            e.relationship == "calls"
                && e.source_qualified_name == "com.acme.service.UserService.findUser"
                && e.target_qualified_name == "com.acme.service.Checks.lookup"
        }));
        assert!(!edges.iter().any(|e| {
            e.relationship == "calls" && e.target_qualified_name.ends_with("println")
        }));

        assert!(edges.iter().any(|e| {
            e.relationship == "inherits"
                && e.source_qualified_name == "com.acme.service.Repository"
                && e.target_qualified_name == "com.acme.service.BaseRepository"
        }));
        assert!(entities.iter().any(|e| {
            e.kind == NodeType::Constant
                && e.qualified_name == "com.acme.service.Repository.DEFAULT_NAME"
        }));
        assert!(entities.iter().any(|e| {
            e.kind == NodeType::Method && e.qualified_name == "com.acme.service.Repository.load"
        }));

        assert!(entities.iter().any(|e| {
            e.kind == NodeType::Constant && e.qualified_name == "com.acme.service.Status.ACTIVE"
        }));
        assert!(entities.iter().any(|e| {
            e.kind == NodeType::Method
                && e.qualified_name == "com.acme.service.Status.getLabel"
        }));

        assert!(entities.iter().any(|e| {
            e.kind == NodeType::Class && e.qualified_name == "com.acme.service.UserRecord"
        }));
        assert!(entities.iter().any(|e| {
            e.kind == NodeType::Variable && e.qualified_name == "com.acme.service.UserRecord.id"
        }));
        assert!(entities.iter().any(|e| {
            e.kind == NodeType::Method
                && e.qualified_name == "com.acme.service.UserRecord.UserRecord"
        }));

        assert!(entities.iter().any(|e| {
            e.kind == NodeType::Class && e.qualified_name == "com.acme.service.Marker"
        }));
        assert!(entities.iter().any(|e| {
            e.kind == NodeType::Method && e.qualified_name == "com.acme.service.Marker.value"
        }));
    }

    #[test]
    fn parses_package_info_and_module_info_as_modules() {
        let pkg_source = "/** Domain model package. */\npackage com.acme.model;\n";
        let (pkg_entities, pkg_edges) =
            JavaParser::new().parse(pkg_source, "src/main/java/com/acme/model/package-info.java");
        let pkg_mod = pkg_entities
            .iter()
            .find(|e| e.kind == NodeType::Module)
            .expect("package-info module");
        assert_eq!(pkg_mod.qualified_name, "com.acme.model");
        assert!(pkg_mod.docstring.contains("Domain model package."));
        assert!(pkg_edges.iter().any(|e| {
            e.relationship == "contains"
                && e.source_qualified_name == "src/main/java/com/acme/model/package-info.java"
                && e.target_qualified_name == "com.acme.model"
        }));

        let mod_source = "/** Core module. */\nmodule com.acme.core {\n    requires java.sql;\n}\n";
        let (mod_entities, _) =
            JavaParser::new().parse(mod_source, "src/main/java/module-info.java");
        let mod_entity = mod_entities
            .iter()
            .find(|e| e.kind == NodeType::Module)
            .expect("module-info module");
        assert_eq!(mod_entity.qualified_name, "com.acme.core");
        assert!(mod_entity.docstring.contains("Core module."));
    }

    #[test]
    fn parses_default_package_and_switch_rule_complexity() {
        let source = r#"
class Main {
    int score(int kind) {
        return switch (kind) {
            case 1 -> 10;
            case 2 -> 20;
            default -> 0;
        };
    }
}
"#;
        let (entities, _) = JavaParser::new().parse(source, "Main.java");
        let file = entities.iter().find(|e| e.kind == NodeType::File).unwrap();
        let main = entities.iter().find(|e| e.kind == NodeType::Class).unwrap();
        assert_eq!(file.qualified_name, "Main.java");
        assert_eq!(main.qualified_name, "Main");
        assert_ne!(file.qualified_name, main.qualified_name);

        let score = entities
            .iter()
            .find(|e| e.kind == NodeType::Method && e.name == "score")
            .unwrap();
        // 1 + 2 non-default switch rules = 3
        assert_eq!(score.cyclomatic_complexity, Some(3));
    }

    #[test]
    fn malformed_and_unicode_java_source_is_safe() {
        let source = format!("package demo;\n// {}\nclass Broken {{ void run(", "🔥".repeat(600));
        let (entities, _) = JavaParser::new().parse(&source, "Broken.java");
        assert!(!entities.is_empty());
        assert!(entities
            .iter()
            .all(|e| e.raw_text.is_char_boundary(e.raw_text.len())));
        assert!(entities.iter().all(|e| e.raw_text.len() <= RAW_TEXT_LIMIT));
    }
}
