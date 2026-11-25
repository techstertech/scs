use tempfile::TempDir;
use tsg::{Embedding, Node, SearchBackend, SearchFilter, Store, WriteBatch};

fn batch() -> WriteBatch {
    WriteBatch {
        nodes: (0..3)
            .map(|index| Node {
                id: format!("node-{index}"),
                scope_id: None,
                kind: "record".to_string(),
                name: format!("node-{index}"),
                content: String::new(),
                attributes: serde_json::json!({}),
            })
            .collect(),
        embeddings: (0..3)
            .map(|index| Embedding {
                node_id: format!("node-{index}"),
                vector: (0..3).map(|axis| f32::from(axis == index)).collect(),
            })
            .collect(),
        ..WriteBatch::default()
    }
}

#[test]
fn writable_adaptive_search_survives_sidecar_persistence_failure() {
    let directory = TempDir::new().unwrap();
    let database = directory.path().join("graph.db");
    // A directory at the sidecar destination deterministically rejects rename.
    std::fs::create_dir(database.with_extension("usearch")).unwrap();
    let mut store = Store::open(&database, 3, 0).unwrap();
    let receipt = store.apply_batch(&batch()).unwrap();
    assert!(!receipt.accelerator_ready);

    let results = store
        .search(
            &[0.0, 1.0, 0.0],
            1,
            SearchFilter::default(),
            SearchBackend::Adaptive,
        )
        .unwrap();
    assert_eq!(results.backend, SearchBackend::Exact);
    assert_eq!(results.hits[0].node.id, "node-1");
    assert!(
        store
            .search(
                &[0.0, 1.0, 0.0],
                1,
                SearchFilter::default(),
                SearchBackend::Usearch
            )
            .is_err()
    );

    std::fs::remove_dir(database.with_extension("usearch")).unwrap();
    store.repair_accelerator().unwrap();
    assert_eq!(
        store
            .search(
                &[0.0, 1.0, 0.0],
                1,
                SearchFilter::default(),
                SearchBackend::Adaptive
            )
            .unwrap()
            .backend,
        SearchBackend::Usearch
    );
}
