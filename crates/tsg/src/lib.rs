#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

mod error;
mod store;
mod types;
mod vector;

pub use error::{Error, Result};
pub use store::{Store, StoreBuilder};
pub use types::{
    AttributeFilter, CatalogKey, CatalogRecord, CommitReceipt, DeleteReceipt, Direction,
    Durability, Edge, Embedding, IntegrityReport, Node, NodeFilter, Scope, SearchBackend,
    SearchFilter, SearchHit, SearchResults, StoreStats, WriteBatch,
};

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    const DIMENSIONS: usize = 8;

    fn node(id: &str, _scope_id: i64, kind: &str) -> Node {
        Node {
            id: id.to_string(),
            scope_id: None,
            kind: kind.to_string(),
            name: format!("name_{id}"),
            content: format!("content for {id}"),
            attributes: serde_json::json!({}),
        }
    }

    fn unit_vector(axis: usize) -> Vec<f32> {
        let mut vector = vec![0.0; DIMENSIONS];
        vector[axis] = 1.0;
        vector
    }

    fn edge(source_id: &str, target_id: &str, relationship: &str) -> Edge {
        Edge {
            id: format!("{source_id}:{relationship}:{target_id}"),
            source_id: source_id.to_string(),
            target_id: target_id.to_string(),
            relationship: relationship.to_string(),
            weight: 1.0,
            attributes: serde_json::json!({}),
        }
    }

    fn store(directory: &TempDir, threshold: usize) -> Store {
        Store::open(directory.path().join("graph.db"), DIMENSIONS, threshold).unwrap()
    }

    fn seeded_batch(count: usize) -> WriteBatch {
        let nodes = (0..count)
            .map(|index| node(&format!("node-{index}"), 7, "function"))
            .collect();
        let embeddings = (0..count)
            .map(|index| Embedding {
                node_id: format!("node-{index}"),
                vector: unit_vector(index % DIMENSIONS),
            })
            .collect();
        WriteBatch {
            nodes,
            embeddings,
            ..WriteBatch::default()
        }
    }

    #[test]
    fn commits_graph_and_embeddings_in_one_generation() {
        let directory = TempDir::new().unwrap();
        let mut store = store(&directory, 4);
        let mut batch = seeded_batch(2);
        batch.edges.push(edge("node-0", "node-1", "calls"));

        let receipt = store.apply_batch(&batch).unwrap();

        assert_eq!(receipt.generation, 1);
        assert!(receipt.accelerator_ready);
        assert_eq!(store.node_count().unwrap(), 2);
        assert_eq!(store.embedding_count().unwrap(), 2);
        let related = store
            .traverse("node-0", Direction::Outgoing, Some("calls"), 1, 10)
            .unwrap();
        assert_eq!(related[0].id, "node-1");
    }

    #[test]
    fn invalid_embedding_rolls_back_the_complete_batch() {
        let directory = TempDir::new().unwrap();
        let mut store = store(&directory, 4);
        let batch = WriteBatch {
            nodes: vec![node("invalid", 1, "function")],
            embeddings: vec![Embedding {
                node_id: "invalid".to_string(),
                vector: vec![1.0; DIMENSIONS - 1],
            }],
            ..WriteBatch::default()
        };

        assert!(matches!(
            store.apply_batch(&batch),
            Err(Error::InvalidInput(_))
        ));
        assert_eq!(store.node_count().unwrap(), 0);
        assert_eq!(store.embedding_count().unwrap(), 0);
        assert_eq!(store.generation().unwrap(), 0);
    }

    #[test]
    fn cross_repository_edge_rolls_back_the_complete_batch() {
        let directory = TempDir::new().unwrap();
        let mut store = store(&directory, 4);
        let left_scope = store.get_or_create_scope("left").unwrap();
        let right_scope = store.get_or_create_scope("right").unwrap();
        let mut left = node("left", 1, "function");
        left.scope_id = Some(left_scope.id);
        let mut right = node("right", 2, "function");
        right.scope_id = Some(right_scope.id);
        let batch = WriteBatch {
            nodes: vec![left, right],
            edges: vec![edge("left", "right", "calls")],
            ..WriteBatch::default()
        };

        assert!(matches!(
            store.apply_batch(&batch),
            Err(Error::InvalidInput(_))
        ));
        assert_eq!(store.node_count().unwrap(), 0);
        assert_eq!(store.generation().unwrap(), 0);
    }

    #[test]
    fn traversal_is_directional_bounded_and_cycle_safe() {
        let directory = TempDir::new().unwrap();
        let mut store = store(&directory, 4);
        let mut batch = seeded_batch(3);
        batch.edges = vec![
            edge("node-0", "node-1", "calls"),
            edge("node-1", "node-2", "calls"),
            edge("node-2", "node-0", "calls"),
        ];
        store.apply_batch(&batch).unwrap();

        let one_hop = store
            .traverse("node-0", Direction::Outgoing, Some("calls"), 1, 10)
            .unwrap();
        let incoming = store
            .traverse("node-0", Direction::Incoming, Some("calls"), 2, 10)
            .unwrap();

        assert_eq!(
            one_hop.iter().map(|node| &node.id).collect::<Vec<_>>(),
            ["node-1"]
        );
        assert_eq!(incoming.len(), 2);
        assert!(!incoming.iter().any(|node| node.id == "node-0"));
    }

    #[test]
    fn exact_and_usearch_agree_for_separated_vectors() {
        let directory = TempDir::new().unwrap();
        let mut store = store(&directory, 4);
        store.apply_batch(&seeded_batch(DIMENSIONS)).unwrap();
        let query = unit_vector(3);

        let exact = store
            .search(&query, 1, SearchFilter::default(), SearchBackend::Exact)
            .unwrap();
        let accelerated = store
            .search(&query, 1, SearchFilter::default(), SearchBackend::Usearch)
            .unwrap();

        assert_eq!(exact.hits[0].node.id, "node-3");
        assert_eq!(accelerated.hits[0].node.id, exact.hits[0].node.id);
        assert!((accelerated.hits[0].distance - exact.hits[0].distance).abs() < 1e-6);
    }

    #[test]
    fn adaptive_search_uses_candidate_count_threshold() {
        let directory = TempDir::new().unwrap();
        let mut store = store(&directory, 2);
        store.apply_batch(&seeded_batch(3)).unwrap();

        let broad = store
            .search(
                &unit_vector(0),
                1,
                SearchFilter::default(),
                SearchBackend::Adaptive,
            )
            .unwrap();
        let narrow = store
            .search(
                &unit_vector(0),
                1,
                SearchFilter {
                    scope_id: Some(999),
                    kind: None,
                },
                SearchBackend::Adaptive,
            )
            .unwrap();

        assert_eq!(broad.backend, SearchBackend::Usearch);
        assert_eq!(narrow.backend, SearchBackend::Exact);
    }

    #[test]
    fn reopen_rebuilds_a_missing_accelerator_from_sqlite() {
        let directory = TempDir::new().unwrap();
        let database_path = directory.path().join("graph.db");
        {
            let mut store = Store::open(&database_path, DIMENSIONS, 4).unwrap();
            store.apply_batch(&seeded_batch(3)).unwrap();
        }
        std::fs::remove_file(database_path.with_extension("usearch")).unwrap();

        let reopened = Store::open(&database_path, DIMENSIONS, 4).unwrap();
        let results = reopened
            .search(
                &unit_vector(1),
                1,
                SearchFilter::default(),
                SearchBackend::Usearch,
            )
            .unwrap();

        assert_eq!(results.hits[0].node.id, "node-1");
        assert!(database_path.with_extension("usearch").exists());
    }

    #[test]
    fn writable_open_excludes_a_second_writer() {
        let directory = TempDir::new().unwrap();
        let database_path = directory.path().join("graph.db");
        let _owner = Store::open(&database_path, DIMENSIONS, 4).unwrap();

        let Err(error) = Store::open(&database_path, DIMENSIONS, 4) else {
            panic!("second writer unexpectedly acquired the store");
        };

        assert!(matches!(error, Error::WriterLocked(_)));
    }

    #[test]
    fn read_only_handle_coexists_and_rejects_mutation() {
        let directory = TempDir::new().unwrap();
        let database_path = directory.path().join("graph.db");
        let mut writer = Store::open(&database_path, DIMENSIONS, 4).unwrap();
        writer.apply_batch(&seeded_batch(2)).unwrap();
        let mut reader = Store::builder(&database_path, DIMENSIONS)
            .read_only(true)
            .build()
            .unwrap();

        assert!(reader.is_read_only());
        assert_eq!(reader.node_count().unwrap(), 2);
        assert!(matches!(
            reader.apply_batch(&seeded_batch(1)),
            Err(Error::ReadOnly)
        ));
    }

    #[test]
    fn catalog_and_graph_commit_atomically_and_survive_reopen() {
        let directory = TempDir::new().unwrap();
        let database_path = directory.path().join("graph.db");
        {
            let mut store = Store::open(&database_path, DIMENSIONS, 4).unwrap();
            let mut batch = seeded_batch(1);
            batch.catalog_records.push(CatalogRecord {
                namespace: "sources".to_string(),
                key: "source-1".to_string(),
                value: serde_json::json!({"digest": "abc", "ready": true}),
            });

            let receipt = store.apply_batch(&batch).unwrap();

            assert_eq!(receipt.generation, 1);
            assert_eq!(
                store
                    .catalog_get("sources", "source-1")
                    .unwrap()
                    .unwrap()
                    .value,
                serde_json::json!({"digest": "abc", "ready": true})
            );
        }

        let reopened = Store::open(&database_path, DIMENSIONS, 4).unwrap();
        assert!(
            reopened
                .catalog_get("sources", "source-1")
                .unwrap()
                .is_some()
        );
        assert_eq!(reopened.node_count().unwrap(), 1);
    }

    #[test]
    fn invalid_catalog_batch_rolls_back_graph_changes() {
        let directory = TempDir::new().unwrap();
        let mut store = store(&directory, 4);
        let mut batch = seeded_batch(1);
        batch.catalog_records.push(CatalogRecord {
            namespace: String::new(),
            key: "source-1".to_string(),
            value: serde_json::json!({}),
        });

        assert!(matches!(
            store.apply_batch(&batch),
            Err(Error::InvalidInput(_))
        ));
        assert_eq!(store.node_count().unwrap(), 0);
        assert_eq!(store.generation().unwrap(), 0);
    }

    #[test]
    fn catalog_namespaces_paginate_and_delete_independently() {
        let directory = TempDir::new().unwrap();
        let mut store = store(&directory, 4);
        let mut batch = WriteBatch::default();
        for (namespace, key) in [("alpha", "b"), ("alpha", "a"), ("beta", "a")] {
            batch.catalog_records.push(CatalogRecord {
                namespace: namespace.to_string(),
                key: key.to_string(),
                value: serde_json::json!({"key": key}),
            });
        }
        store.apply_batch(&batch).unwrap();

        let page = store.catalog_list("alpha", 1, 1).unwrap();
        assert_eq!(page.len(), 1);
        assert_eq!(page[0].key, "b");
        assert!(store.catalog_get("beta", "a").unwrap().is_some());

        store
            .apply_batch(&WriteBatch {
                catalog_deletes: vec![CatalogKey {
                    namespace: "alpha".to_string(),
                    key: "a".to_string(),
                }],
                ..WriteBatch::default()
            })
            .unwrap();

        assert!(store.catalog_get("alpha", "a").unwrap().is_none());
        assert!(store.catalog_get("beta", "a").unwrap().is_some());
    }

    #[test]
    fn catalog_rejects_conflicting_operations_before_mutation() {
        let directory = TempDir::new().unwrap();
        let mut store = store(&directory, 4);
        let identity = CatalogKey {
            namespace: "settings".to_string(),
            key: "default".to_string(),
        };
        let batch = WriteBatch {
            catalog_records: vec![CatalogRecord {
                namespace: identity.namespace.clone(),
                key: identity.key.clone(),
                value: serde_json::json!(true),
            }],
            catalog_deletes: vec![identity],
            ..WriteBatch::default()
        };

        assert!(matches!(
            store.apply_batch(&batch),
            Err(Error::InvalidInput(_))
        ));
        assert_eq!(store.generation().unwrap(), 0);
    }

    #[test]
    fn scopes_and_structured_attributes_survive_reopen() {
        let directory = TempDir::new().unwrap();
        let database_path = directory.path().join("graph.db");
        let scope_id;
        {
            let mut store = Store::open(&database_path, DIMENSIONS, 4).unwrap();
            let scope = store.get_or_create_scope("tenant-a").unwrap();
            scope_id = scope.id;
            let mut scoped = node("scoped", 0, "record");
            scoped.scope_id = Some(scope.id);
            scoped.attributes = serde_json::json!({"external_name": "alpha"});
            store
                .apply_batch(&WriteBatch {
                    nodes: vec![scoped],
                    ..WriteBatch::default()
                })
                .unwrap();
        }

        let reopened = Store::open(&database_path, DIMENSIONS, 4).unwrap();
        assert_eq!(
            reopened.scope_by_key("tenant-a").unwrap().unwrap().id,
            scope_id
        );
        let stored = reopened.get_node("scoped").unwrap().unwrap();
        assert_eq!(stored.attributes["external_name"], "alpha");
    }

    #[test]
    fn malformed_attributes_and_non_finite_weights_are_rejected() {
        let directory = TempDir::new().unwrap();
        let mut store = store(&directory, 4);
        let mut malformed = node("malformed", 0, "record");
        malformed.attributes = serde_json::json!([]);
        assert!(matches!(
            store.apply_batch(&WriteBatch {
                nodes: vec![malformed],
                ..WriteBatch::default()
            }),
            Err(Error::InvalidInput(_))
        ));

        let invalid_edge = Edge {
            weight: f64::NAN,
            ..edge("missing-a", "missing-b", "related")
        };
        assert!(matches!(
            store.apply_batch(&WriteBatch {
                edges: vec![invalid_edge],
                ..WriteBatch::default()
            }),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn bounded_reads_filter_attributes_missing_vectors_and_edges() {
        let directory = TempDir::new().unwrap();
        let mut store = store(&directory, 4);
        let mut batch = seeded_batch(2);
        batch.embeddings.pop();
        batch.nodes[0].attributes = serde_json::json!({"external": {"name": "alpha"}});
        batch.edges.push(edge("node-0", "node-1", "related"));
        store.apply_batch(&batch).unwrap();

        let page = store.list_nodes(NodeFilter::default(), 1, 1).unwrap();
        assert_eq!(page[0].id, "node-1");
        let matched = store
            .find_nodes_by_attribute(
                None,
                AttributeFilter {
                    path: "$.external.name",
                    value: &serde_json::json!("alpha"),
                },
                10,
                0,
            )
            .unwrap();
        assert_eq!(matched[0].id, "node-0");
        assert_eq!(
            store
                .list_nodes_without_embeddings(NodeFilter::default(), 10, 0)
                .unwrap()[0]
                .id,
            "node-1"
        );
        let edges = store
            .get_edges("node-0", Direction::Outgoing, Some("related"))
            .unwrap();
        assert!((edges[0].weight - 1.0).abs() < f64::EPSILON);
        assert!(store.delete_edge(&edges[0].id).unwrap());
        assert!(
            store
                .get_edges("node-0", Direction::Both, None)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn attribute_reads_reject_unbounded_or_unsafe_inputs() {
        let directory = TempDir::new().unwrap();
        let store = store(&directory, 4);
        assert!(matches!(
            store.list_nodes(NodeFilter::default(), 0, 0),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            store.find_nodes_by_attribute(
                None,
                AttributeFilter {
                    path: "$['unsafe']",
                    value: &serde_json::Value::Null
                },
                1,
                0,
            ),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn filtered_counts_name_reads_and_resets_are_consistent() {
        let directory = TempDir::new().unwrap();
        let mut store = store(&directory, 4);
        let mut batch = seeded_batch(2);
        batch.embeddings.pop();
        store.apply_batch(&batch).unwrap();

        assert_eq!(store.count_nodes(NodeFilter::default()).unwrap(), 2);
        assert_eq!(
            store
                .count_nodes_without_embeddings(NodeFilter::default())
                .unwrap(),
            1
        );
        assert_eq!(
            store
                .find_nodes_by_name("NAME_NODE-1", NodeFilter::default(), 10, 0)
                .unwrap()[0]
                .id,
            "node-1"
        );
        assert_eq!(
            store
                .get_nodes(&["node-1".to_string(), "unknown".to_string()])
                .unwrap()
                .len(),
            1
        );
        assert_eq!(store.clear_embeddings().unwrap(), 1);
        assert_eq!(store.embedding_count().unwrap(), 0);
        assert_eq!(store.truncate().unwrap(), 2);
        assert_eq!(store.node_count().unwrap(), 0);
        store.vacuum().unwrap();
    }
}
