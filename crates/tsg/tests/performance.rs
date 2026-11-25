use std::time::{Duration, Instant};

use tempfile::TempDir;
use tsg::{Durability, Embedding, Node, SearchBackend, SearchFilter, Store, WriteBatch};

const DIMENSIONS: usize = 32;
const CI_CORPUS_SIZE: usize = 5_000;
const INGEST_BUDGET: Duration = Duration::from_secs(10);
const SEARCH_BUDGET: Duration = Duration::from_millis(100);

fn generated_batch(count: usize, dimensions: usize) -> WriteBatch {
    let mut nodes = Vec::with_capacity(count);
    let mut embeddings = Vec::with_capacity(count);
    for index in 0..count {
        let id = format!("node-{index}");
        let mut state = u64::try_from(index).unwrap().wrapping_add(1);
        let vector = (0..dimensions)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let sample = u8::try_from(state & 0xff).unwrap();
                f32::from(sample) / 255.0 - 0.5
            })
            .collect();
        nodes.push(Node {
            id: id.clone(),
            scope_id: None,
            kind: "function".to_string(),
            name: id.clone(),
            content: String::new(),
            attributes: serde_json::json!({}),
        });
        embeddings.push(Embedding {
            node_id: id,
            vector,
        });
    }
    WriteBatch {
        nodes,
        embeddings,
        ..WriteBatch::default()
    }
}

#[test]
fn deterministic_ingest_and_search_budgets() {
    let directory = TempDir::new().unwrap();
    let mut store = Store::builder(directory.path().join("graph.db"), DIMENSIONS)
        .durability(Durability::Full)
        .exact_search_threshold(100)
        .build()
        .unwrap();
    let batch = generated_batch(CI_CORPUS_SIZE, DIMENSIONS);

    let ingest_started = Instant::now();
    store.apply_batch(&batch).unwrap();
    let ingest_elapsed = ingest_started.elapsed();

    let query = &batch.embeddings[731].vector;
    let search_started = Instant::now();
    let results = store
        .search(query, 10, SearchFilter::default(), SearchBackend::Usearch)
        .unwrap();
    let search_elapsed = search_started.elapsed();

    assert_eq!(results.hits[0].node.id, "node-731");
    assert!(ingest_elapsed <= INGEST_BUDGET, "{ingest_elapsed:?}");
    assert!(search_elapsed <= SEARCH_BUDGET, "{search_elapsed:?}");
}

#[test]
#[ignore = "opt-in one-million-vector design-envelope validation"]
fn one_million_vector_scale_harness() {
    const SCALE: usize = 1_000_000;
    let directory = TempDir::new().unwrap();
    let mut store = Store::builder(directory.path().join("graph.db"), DIMENSIONS)
        .durability(Durability::Normal)
        .exact_search_threshold(10_000)
        .build()
        .unwrap();
    let batch = generated_batch(SCALE, DIMENSIONS);

    store.apply_batch(&batch).unwrap();

    assert_eq!(store.node_count().unwrap(), SCALE);
    assert_eq!(store.embedding_count().unwrap(), SCALE);
    assert!(store.verify_integrity().unwrap().is_healthy());
}
