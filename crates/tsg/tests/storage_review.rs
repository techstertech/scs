use tempfile::TempDir;
use tsg::{
    CatalogRecord, Direction, Edge, Embedding, Error, Node, NodeFilter, SearchBackend,
    SearchFilter, Store, WriteBatch,
};

const DIMENSIONS: usize = 2;
const ORIGINAL_VECTOR: [f32; DIMENSIONS] = [1.0, 0.0];

fn node(id: &str, scope_id: Option<i64>) -> Node {
    Node {
        id: id.to_string(),
        scope_id,
        kind: "test".to_string(),
        name: id.to_string(),
        content: String::new(),
        attributes: serde_json::json!({}),
    }
}

fn connected_nodes(store: &mut Store, scope_id: Option<i64>) -> Vec<Node> {
    let nodes: Vec<Node> = (0..2)
        .map(|index| node(&format!("node-{index}"), scope_id))
        .collect();
    store
        .apply_batch(&WriteBatch {
            nodes: nodes.clone(),
            edges: vec![Edge {
                id: "edge".to_string(),
                source_id: nodes[0].id.clone(),
                target_id: nodes[1].id.clone(),
                relationship: "connects".to_string(),
                weight: 1.0,
                attributes: serde_json::json!({}),
            }],
            embeddings: nodes
                .iter()
                .map(|node| Embedding {
                    node_id: node.id.clone(),
                    vector: ORIGINAL_VECTOR.to_vec(),
                })
                .collect(),
            ..WriteBatch::default()
        })
        .unwrap();
    nodes
}

#[test]
fn moving_one_endpoint_rejects_and_rolls_back_the_entire_batch() {
    for scoped in [false, true] {
        for endpoint in 0..2 {
            let directory = TempDir::new().unwrap();
            let mut store = Store::open(directory.path().join("graph.db"), DIMENSIONS, 10).unwrap();
            let original_scope = scoped.then(|| store.get_or_create_scope("original").unwrap().id);
            let destination = store.get_or_create_scope("destination").unwrap();
            let nodes = connected_nodes(&mut store, original_scope);
            let generation = store.generation().unwrap();
            let mut moved = nodes[endpoint].clone();
            moved.scope_id = Some(destination.id);
            moved.content = "must roll back".to_string();

            let result = store.apply_batch(&WriteBatch {
                nodes: vec![moved.clone()],
                embeddings: vec![Embedding {
                    node_id: moved.id.clone(),
                    vector: vec![0.0, 1.0],
                }],
                catalog_records: vec![CatalogRecord {
                    namespace: "review".to_string(),
                    key: "must-roll-back".to_string(),
                    value: serde_json::json!({"applied": true}),
                }],
                ..WriteBatch::default()
            });

            assert!(matches!(result, Err(Error::InvalidInput(_))));
            assert_eq!(store.generation().unwrap(), generation);
            for original in &nodes {
                assert_eq!(
                    store.get_node(&original.id).unwrap().as_ref(),
                    Some(original)
                );
            }
            assert!(
                store
                    .catalog_get("review", "must-roll-back")
                    .unwrap()
                    .is_none()
            );
            assert_eq!(
                store
                    .get_edges(&moved.id, Direction::Both, None)
                    .unwrap()
                    .len(),
                1
            );
            let results = store
                .search(
                    &ORIGINAL_VECTOR,
                    2,
                    SearchFilter::default(),
                    SearchBackend::Exact,
                )
                .unwrap();
            assert_eq!(results.hits.len(), 2);
            assert!(results.hits.iter().all(|hit| hit.distance == 0.0));
        }
    }
}

#[test]
fn connected_endpoints_can_move_together_in_one_batch() {
    let directory = TempDir::new().unwrap();
    let mut store = Store::open(directory.path().join("graph.db"), DIMENSIONS, 10).unwrap();
    let destination = store.get_or_create_scope("destination").unwrap();
    let mut nodes = connected_nodes(&mut store, None);
    for node in &mut nodes {
        node.scope_id = Some(destination.id);
    }
    store
        .apply_batch(&WriteBatch {
            nodes: nodes.clone(),
            ..WriteBatch::default()
        })
        .unwrap();

    let reached = store
        .traverse(&nodes[0].id, Direction::Outgoing, None, 1, 10)
        .unwrap();
    assert_eq!(reached, vec![nodes[1].clone()]);
}

#[test]
fn name_search_treats_backslashes_and_wildcards_literally() {
    let directory = TempDir::new().unwrap();
    let mut store = Store::open(directory.path().join("graph.db"), DIMENSIONS, 10).unwrap();
    for query in [r"\", r"\path", r"\%", r"\_", "%", "_"] {
        store.truncate().unwrap();
        let mut matching = node("matching", None);
        matching.name = format!("prefix{query}suffix");
        let unrelated = node("unrelated", None);
        store
            .apply_batch(&WriteBatch {
                nodes: vec![matching.clone(), unrelated],
                ..WriteBatch::default()
            })
            .unwrap();

        assert_eq!(
            store
                .find_nodes_by_name(query, NodeFilter::default(), 10, 0)
                .unwrap(),
            vec![matching],
            "literal query: {query:?}"
        );
    }
}

#[test]
fn usearch_search_honors_single_scope_multi_scope_and_kind_filters() {
    let directory = TempDir::new().unwrap();
    let mut store = Store::open(directory.path().join("graph.db"), DIMENSIONS, 0).unwrap();
    let scope_a = store.get_or_create_scope("repo-a").unwrap();
    let scope_b = store.get_or_create_scope("repo-b").unwrap();

    let mut nodes = Vec::new();
    let mut embeddings = Vec::new();
    for (id, scope_id, kind, vector) in [
        ("a-fn", Some(scope_a.id), "function", vec![1.0_f32, 0.0]),
        ("a-cls", Some(scope_a.id), "class", vec![0.99_f32, 0.1]),
        ("a-far", Some(scope_a.id), "function", vec![0.0_f32, 1.0]),
    ] {
        nodes.push(Node {
            id: id.to_string(),
            scope_id,
            kind: kind.to_string(),
            name: id.to_string(),
            content: String::new(),
            attributes: serde_json::json!({}),
        });
        embeddings.push(Embedding {
            node_id: id.to_string(),
            vector,
        });
    }
    store
        .apply_batch(&WriteBatch {
            nodes,
            embeddings,
            ..WriteBatch::default()
        })
        .unwrap();

    // Single-scope fast path (no nodes outside scope_a, kind is None).
    let single_scope = store
        .search(
            &[1.0, 0.0],
            2,
            SearchFilter {
                scope_id: Some(scope_a.id),
                kind: None,
            },
            SearchBackend::Usearch,
        )
        .unwrap();
    assert_eq!(
        single_scope
            .hits
            .iter()
            .map(|h| h.node.id.as_str())
            .collect::<Vec<_>>(),
        vec!["a-fn", "a-cls"]
    );

    // Kind-filtered search uses key-only filtered_search.
    let kind_filtered = store
        .search(
            &[1.0, 0.0],
            2,
            SearchFilter {
                scope_id: Some(scope_a.id),
                kind: Some("function"),
            },
            SearchBackend::Usearch,
        )
        .unwrap();
    assert_eq!(
        kind_filtered
            .hits
            .iter()
            .map(|h| h.node.id.as_str())
            .collect::<Vec<_>>(),
        vec!["a-fn", "a-far"]
    );

    // Add a second scope with an identical query vector to verify multi-scope isolation.
    store
        .apply_batch(&WriteBatch {
            nodes: vec![Node {
                id: "b-fn".to_string(),
                scope_id: Some(scope_b.id),
                kind: "function".to_string(),
                name: "b-fn".to_string(),
                content: String::new(),
                attributes: serde_json::json!({}),
            }],
            embeddings: vec![Embedding {
                node_id: "b-fn".to_string(),
                vector: vec![1.0, 0.0],
            }],
            ..WriteBatch::default()
        })
        .unwrap();

    let scope_b_hits = store
        .search(
            &[1.0, 0.0],
            2,
            SearchFilter {
                scope_id: Some(scope_b.id),
                kind: None,
            },
            SearchBackend::Usearch,
        )
        .unwrap();
    assert_eq!(
        scope_b_hits
            .hits
            .iter()
            .map(|h| h.node.id.as_str())
            .collect::<Vec<_>>(),
        vec!["b-fn"]
    );
}
