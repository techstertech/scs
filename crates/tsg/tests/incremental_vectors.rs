//! Synthetic vectors and records; no application data or private source fixtures.
use std::time::Instant;
use tempfile::TempDir;
use tsg::{Embedding, Node, SearchBackend, SearchFilter, Store, WriteBatch};

fn generated_batch(start: usize, count: usize, dimensions: usize) -> WriteBatch {
    WriteBatch {
        nodes: (start..start + count)
            .map(|index| Node {
                id: format!("node-{index}"),
                scope_id: None,
                kind: "record".into(),
                name: format!("Generated {index}"),
                content: String::new(),
                attributes: serde_json::json!({}),
            })
            .collect(),
        embeddings: (start..start + count)
            .map(|index| Embedding {
                node_id: format!("node-{index}"),
                vector: (0..dimensions)
                    .map(|axis| f32::from(axis == index % dimensions))
                    .collect(),
            })
            .collect(),
        ..WriteBatch::default()
    }
}

fn nearest(store: &Store, axis: usize, backend: SearchBackend) -> String {
    let query: Vec<_> = (0..64).map(|index| f32::from(index == axis)).collect();
    store
        .search(&query, 1, SearchFilter::default(), backend)
        .unwrap()
        .hits[0]
        .node
        .id
        .clone()
}

#[test]
fn additions_replacements_and_failed_transactions_survive_reopen() {
    let directory = TempDir::new().unwrap();
    let database = directory.path().join("graph.db");
    let mut store = Store::open(&database, 64, 0).unwrap();
    for start in [0, 16, 32] {
        assert!(
            store
                .apply_batch(&generated_batch(start, 16, 64))
                .unwrap()
                .accelerator_ready
        );
    }
    let mut replacement = generated_batch(0, 1, 64);
    replacement.embeddings[0].vector.rotate_right(63);
    assert!(store.apply_batch(&replacement).unwrap().accelerator_ready);
    assert_eq!(nearest(&store, 63, SearchBackend::Usearch), "node-0");
    assert_eq!(nearest(&store, 31, SearchBackend::Usearch), "node-31");
    let hits = store
        .search(
            &vec![1.0; 64],
            64,
            SearchFilter::default(),
            SearchBackend::Usearch,
        )
        .unwrap()
        .hits;
    let unique: std::collections::HashSet<_> = hits.iter().map(|hit| &hit.node.id).collect();
    assert_eq!(hits.len(), 48);
    assert_eq!(
        unique.len(),
        48,
        "replacement must not leave duplicate keys"
    );
    let old_query: Vec<_> = (0..64).map(|axis| f32::from(axis == 0)).collect();
    let replaced = store
        .search(
            &old_query,
            64,
            SearchFilter::default(),
            SearchBackend::Usearch,
        )
        .unwrap()
        .hits
        .into_iter()
        .find(|hit| hit.node.id == "node-0")
        .unwrap();
    assert!(
        (replaced.distance - 1.0).abs() < 0.0001,
        "old vector must be removed"
    );
    let mut metadata = generated_batch(0, 1, 64);
    metadata.embeddings.clear();
    metadata.nodes[0].name = "Updated record".into();
    assert!(store.apply_batch(&metadata).unwrap().accelerator_ready);
    let new_query: Vec<_> = (0..64).map(|axis| f32::from(axis == 63)).collect();
    let updated = store
        .search(
            &new_query,
            1,
            SearchFilter::default(),
            SearchBackend::Usearch,
        )
        .unwrap();
    assert_eq!(updated.hits[0].node.name, "Updated record");
    assert!(updated.hits[0].distance.abs() < 0.0001);
    let generation = store.generation().unwrap();
    let mut invalid = generated_batch(48, 1, 64);
    invalid.embeddings.push(Embedding {
        node_id: "missing".into(),
        vector: vec![1.0; 64],
    });
    assert!(store.apply_batch(&invalid).is_err());
    assert_eq!(store.generation().unwrap(), generation);
    assert_eq!(nearest(&store, 63, SearchBackend::Usearch), "node-0");
    drop(store);
    // Read-only reopen cannot repair the index, proving the persisted sidecar is current.
    let reopened = Store::builder(&database, 64)
        .read_only(true)
        .build()
        .unwrap();
    assert_eq!(nearest(&reopened, 63, SearchBackend::Usearch), "node-0");
    assert_eq!(nearest(&reopened, 31, SearchBackend::Usearch), "node-31");
}

#[test]
fn failed_incremental_persistence_discards_partial_index_and_reopen_repairs() {
    let directory = TempDir::new().unwrap();
    let database = directory.path().join("graph.db");
    let sidecar = database.with_extension("usearch");
    let mut store = Store::open(&database, 64, 0).unwrap();
    assert!(
        store
            .apply_batch(&generated_batch(0, 32, 64))
            .unwrap()
            .accelerator_ready
    );
    std::fs::remove_file(&sidecar).unwrap();
    std::fs::create_dir(&sidecar).unwrap();
    let mut replacement = generated_batch(0, 1, 64);
    replacement.embeddings[0].vector.rotate_right(63);
    let receipt = store.apply_batch(&replacement).unwrap();
    assert!(!receipt.accelerator_ready);
    assert_eq!(nearest(&store, 63, SearchBackend::Adaptive), "node-0");
    assert!(
        store
            .search(
                &vec![1.0; 64],
                1,
                SearchFilter::default(),
                SearchBackend::Usearch
            )
            .is_err()
    );
    drop(store);
    std::fs::remove_dir(&sidecar).unwrap();
    let reopened = Store::open(&database, 64, 0).unwrap();
    assert_eq!(reopened.generation().unwrap(), receipt.generation);
    assert_eq!(nearest(&reopened, 63, SearchBackend::Usearch), "node-0");
}

#[test]
#[ignore = "opt-in generated throughput diagnostic; no machine-dependent CI threshold"]
fn generated_4096_dimension_batch_throughput() {
    let directory = TempDir::new().unwrap();
    let mut store = Store::open(directory.path().join("benchmark.db"), 4096, 0).unwrap();
    // Deterministic dense vectors model the dimension and batch size of embedding
    // ingestion. A moderate corpus bounds local CPU cost; timings are diagnostic.
    let mut batch = generated_batch(0, 512, 4096);
    for (index, embedding) in batch.embeddings.iter_mut().enumerate() {
        for (axis, value) in embedding.vector.iter_mut().enumerate() {
            let integer = u16::try_from((index * 137 + axis * 73 + index * axis) % 1021).unwrap();
            *value = f32::from(integer) / 1021.0 - 0.5;
        }
    }
    let started = Instant::now();
    for (nodes, embeddings) in batch.nodes.chunks(32).zip(batch.embeddings.chunks(32)) {
        assert!(
            store
                .apply_batch(&WriteBatch {
                    nodes: nodes.to_vec(),
                    embeddings: embeddings.to_vec(),
                    ..WriteBatch::default()
                })
                .unwrap()
                .accelerator_ready
        );
    }
    eprintln!(
        "512 generated vectors, 4096 dimensions, 16 batches: {:?}",
        started.elapsed()
    );
}
