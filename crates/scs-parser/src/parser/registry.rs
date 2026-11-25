//! File extension to parser mapping.
//!
//! Maps source file extensions to their corresponding tree-sitter parser
//! implementations. Parsers are instantiated lazily on first use to avoid
//! loading tree-sitter grammars until they're actually needed.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::bash::BashParser;
use super::css::CssParser;
use super::elixir::ElixirParser;
use super::go::GoParser;
use super::java::JavaParser;
use super::python::PythonParser;
use super::rust::RustParser;
use super::swift::SwiftParser;
use super::typescript::TypeScriptParser;
use super::LanguageParser;

/// Canonical extension specs: `(extension, language_name, parser_key)`.
///
/// Single source of truth for discovery, metadata, and parser selection.
const EXTENSION_SPECS: &[(&str, &str, &str)] = &[
    (".py", "python", "python"),
    (".ts", "typescript", "typescript"),
    (".tsx", "tsx", "tsx"),
    (".mts", "typescript", "typescript"),
    (".cts", "typescript", "typescript"),
    (".js", "javascript", "javascript"),
    (".jsx", "jsx", "tsx"), // JSX uses the TSX grammar.
    (".mjs", "javascript", "javascript"),
    (".cjs", "javascript", "javascript"),
    (".mjsx", "jsx", "tsx"),
    (".es6", "javascript", "javascript"),
    (".jsm", "javascript", "javascript"),
    (".rs", "rust", "rust"),
    (".swift", "swift", "swift"),
    (".ex", "elixir", "elixir"),
    (".exs", "elixir", "elixir"),
    (".sh", "bash", "bash"),
    (".bash", "bash", "bash"),
    (".css", "css", "css"),
    (".go", "go", "go"),
    (".java", "java", "java"),
];

fn extension_spec(extension: &str) -> Option<&'static (&'static str, &'static str, &'static str)> {
    static MAP: OnceLock<HashMap<&'static str, &'static (&'static str, &'static str, &'static str)>> =
        OnceLock::new();
    let map = MAP.get_or_init(|| {
        EXTENSION_SPECS
            .iter()
            .map(|spec| (spec.0, spec))
            .collect()
    });
    if let Some(spec) = map.get(extension) {
        return Some(*spec);
    }
    let normalized = extension.to_ascii_lowercase();
    map.get(normalized.as_str()).copied()
}

/// Lazily instantiated parser cache — one parser per language.
static PARSER_CACHE: OnceLock<Mutex<HashMap<String, Arc<dyn LanguageParser>>>> = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<String, Arc<dyn LanguageParser>>> {
    PARSER_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get a parser for the given file extension.
///
/// Returns `None` if the extension is not supported. Parsers are cached
/// after first instantiation to avoid repeated grammar loading.
pub fn get_parser(extension: &str) -> Option<Arc<dyn LanguageParser>> {
    let &(_, _, parser_key) = extension_spec(extension)?;
    let mut parsers = cache().lock().unwrap();

    if let Some(parser) = parsers.get(parser_key) {
        return Some(Arc::clone(parser));
    }

    let parser: Arc<dyn LanguageParser> = match parser_key {
        "python" => Arc::new(PythonParser::new()),
        "typescript" => Arc::new(TypeScriptParser::new("typescript")),
        "tsx" => Arc::new(TypeScriptParser::new("tsx")),
        "javascript" => Arc::new(TypeScriptParser::new("typescript")), // TS superset of JS
        "rust" => Arc::new(RustParser::new()),
        "swift" => Arc::new(SwiftParser::new()),
        "elixir" => Arc::new(ElixirParser::new()),
        "bash" => Arc::new(BashParser::new()),
        "css" => Arc::new(CssParser::new()),
        "go" => Arc::new(GoParser::new()),
        "java" => Arc::new(JavaParser::new()),
        _ => return None,
    };

    parsers.insert(parser_key.to_string(), Arc::clone(&parser));
    Some(parser)
}

/// Get the set of file extensions we can parse.
pub fn supported_extensions() -> Vec<&'static str> {
    EXTENSION_SPECS.iter().map(|spec| spec.0).collect()
}

/// Extension → language name for FileEntry.
pub fn extension_to_language(ext: &str) -> &'static str {
    extension_spec(ext).map_or("unknown", |spec| spec.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn go_is_a_supported_native_language() {
        assert!(supported_extensions().contains(&".go"));
        assert_eq!(extension_to_language(".go"), "go");
        assert!(get_parser(".go").is_some());
    }

    #[test]
    fn go_parser_is_cached() {
        let first = get_parser(".go").unwrap();
        let second = get_parser(".go").unwrap();

        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn js_ts_variant_extensions_are_supported_and_case_insensitive() {
        let supported = supported_extensions();
        let expected = [
            (".ts", "typescript"),
            (".tsx", "tsx"),
            (".mts", "typescript"),
            (".cts", "typescript"),
            (".js", "javascript"),
            (".jsx", "jsx"),
            (".mjs", "javascript"),
            (".cjs", "javascript"),
            (".mjsx", "jsx"),
            (".es6", "javascript"),
            (".jsm", "javascript"),
        ];

        for (ext, language) in expected {
            assert!(supported.contains(&ext), "missing {ext}");
            assert_eq!(extension_to_language(ext), language);
            assert!(get_parser(ext).is_some(), "missing parser for {ext}");
            let upper = ext.to_ascii_uppercase();
            assert_eq!(extension_to_language(&upper), language);
            assert!(get_parser(&upper).is_some(), "missing parser for {upper}");
        }

        let js_ts_set: HashSet<&str> = super::super::typescript::JS_TS_EXTENSIONS
            .iter()
            .copied()
            .collect();
        let registry_js_ts: HashSet<&str> = expected.iter().map(|(ext, _)| *ext).collect();
        assert_eq!(js_ts_set, registry_js_ts);
        assert!(Arc::ptr_eq(
            &get_parser(".cjs").unwrap(),
            &get_parser(".CJS").unwrap()
        ));
    }

    #[test]
    fn java_is_a_supported_native_language() {
        assert!(supported_extensions().contains(&".java"));
        assert_eq!(extension_to_language(".java"), "java");
        assert_eq!(extension_to_language(".JAVA"), "java");
        assert!(get_parser(".java").is_some());
    }

    #[test]
    fn java_parser_is_cached() {
        let first = get_parser(".java").unwrap();
        let second = get_parser(".JAVA").unwrap();

        assert!(Arc::ptr_eq(&first, &second));
    }
}
