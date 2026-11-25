use tempfile::TempDir;
use tsg::{
    Direction, Edge, Embedding, Error, Node, SearchBackend, SearchFilter, Store, WriteBatch,
};

const DIMENSIONS: usize = 8;
const LOCK_TEST_DATABASE: &str = "TSG_LOCK_TEST_DATABASE";
const LOCK_TEST_READY: &str = "TSG_LOCK_TEST_READY";
const CRASH_TEST_DATABASE: &str = "TSG_CRASH_TEST_DATABASE";
const CRASH_TEST_READY: &str = "TSG_CRASH_TEST_READY";

fn vector(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; DIMENSIONS];
    vector[axis] = 1.0;
    vector
}

fn batch() -> WriteBatch {
    WriteBatch {
        nodes: (0..3)
            .map(|index| Node {
                id: format!("node-{index}"),
                scope_id: None,
                kind: "function".to_string(),
                name: format!("function_{index}"),
                content: String::new(),
                attributes: serde_json::json!({}),
            })
            .collect(),
        edges: vec![
            Edge {
                id: "edge-0-1".to_string(),
                source_id: "node-0".to_string(),
                target_id: "node-1".to_string(),
                relationship: "calls".to_string(),
                weight: 1.0,
                attributes: serde_json::json!({}),
            },
            Edge {
                id: "edge-1-2".to_string(),
                source_id: "node-1".to_string(),
                target_id: "node-2".to_string(),
                relationship: "calls".to_string(),
                weight: 1.0,
                attributes: serde_json::json!({}),
            },
        ],
        embeddings: (0..3)
            .map(|index| Embedding {
                node_id: format!("node-{index}"),
                vector: vector(index),
            })
            .collect(),
        ..WriteBatch::default()
    }
}

#[test]
fn delete_cascades_and_remains_correct_after_reopen() {
    let directory = TempDir::new().unwrap();
    let database_path = directory.path().join("graph.db");
    {
        let mut store = Store::open(&database_path, DIMENSIONS, 1).unwrap();
        store.apply_batch(&batch()).unwrap();

        let receipt = store.delete_nodes(&["node-1".to_string()]).unwrap();

        assert_eq!(receipt.generation, 2);
        assert_eq!(receipt.nodes_deleted, 1);
        assert!(receipt.accelerator_ready);
        assert_eq!(store.stats().unwrap().edge_count, 0);
        assert_eq!(store.embedding_count().unwrap(), 2);
        assert!(
            store
                .traverse("node-0", Direction::Outgoing, None, 2, 10)
                .unwrap()
                .is_empty()
        );
    }

    let reopened = Store::open(&database_path, DIMENSIONS, 1).unwrap();
    let hits = reopened
        .search(&vector(1), 3, SearchFilter::default(), SearchBackend::Exact)
        .unwrap();
    assert!(!hits.hits.iter().any(|hit| hit.node.id == "node-1"));
    assert!(reopened.verify_integrity().unwrap().is_healthy());
}

#[test]
fn corrupt_sidecar_is_rebuilt_from_canonical_vectors() {
    let directory = TempDir::new().unwrap();
    let database_path = directory.path().join("graph.db");
    {
        let mut store = Store::open(&database_path, DIMENSIONS, 1).unwrap();
        store.apply_batch(&batch()).unwrap();
    }
    std::fs::write(database_path.with_extension("usearch"), b"corrupt").unwrap();

    let reopened = Store::open(&database_path, DIMENSIONS, 1).unwrap();

    assert!(reopened.verify_integrity().unwrap().is_healthy());
    let hits = reopened
        .search(
            &vector(2),
            1,
            SearchFilter::default(),
            SearchBackend::Usearch,
        )
        .unwrap();
    assert_eq!(hits.hits[0].node.id, "node-2");
}

#[test]
fn writer_lock_is_immediately_reusable_after_drop() {
    let directory = TempDir::new().unwrap();
    let database_path = directory.path().join("graph.db");

    for _ in 0..10 {
        drop(Store::open(&database_path, DIMENSIONS, 1).unwrap());
    }
}

#[test]
fn readonly_adaptive_search_falls_back_when_sidecar_is_missing() {
    let directory = TempDir::new().unwrap();
    let database_path = directory.path().join("graph.db");
    {
        let mut store = Store::open(&database_path, DIMENSIONS, 0).unwrap();
        store.apply_batch(&batch()).unwrap();
    }
    std::fs::remove_file(database_path.with_extension("usearch")).unwrap();
    let reader = Store::builder(&database_path, DIMENSIONS)
        .exact_search_threshold(0)
        .read_only(true)
        .build()
        .unwrap();

    let adaptive = reader
        .search(
            &vector(0),
            1,
            SearchFilter::default(),
            SearchBackend::Adaptive,
        )
        .unwrap();
    let accelerated = reader.search(
        &vector(0),
        1,
        SearchFilter::default(),
        SearchBackend::Usearch,
    );

    assert_eq!(adaptive.backend, SearchBackend::Exact);
    assert_eq!(adaptive.hits[0].node.id, "node-0");
    assert!(matches!(accelerated, Err(Error::AcceleratorUnavailable(_))));
}

#[test]
fn lock_holder_process() {
    let (Ok(database_path), Ok(ready_path)) = (
        std::env::var(LOCK_TEST_DATABASE),
        std::env::var(LOCK_TEST_READY),
    ) else {
        return;
    };
    let _store = Store::open(database_path, DIMENSIONS, 10).unwrap();
    std::fs::write(ready_path, b"ready").unwrap();
    std::thread::sleep(std::time::Duration::from_secs(5));
}

#[test]
fn writer_lock_excludes_another_process() {
    let directory = TempDir::new().unwrap();
    let database_path = directory.path().join("graph.db");
    let ready_path = directory.path().join("ready");
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "lock_holder_process", "--nocapture"])
        .env(LOCK_TEST_DATABASE, &database_path)
        .env(LOCK_TEST_READY, &ready_path)
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    while !ready_path.exists() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(ready_path.exists(), "child did not acquire writer lock");

    let result = Store::open(&database_path, DIMENSIONS, 10);

    child.kill().unwrap();
    child.wait().unwrap();
    assert!(matches!(result, Err(Error::WriterLocked(_))));
}

#[test]
fn crash_writer_process() {
    let (Ok(database_path), Ok(ready_path)) = (
        std::env::var(CRASH_TEST_DATABASE),
        std::env::var(CRASH_TEST_READY),
    ) else {
        return;
    };
    let mut store = Store::open(database_path, DIMENSIONS, 0).unwrap();
    let mut nodes = Vec::with_capacity(10_000);
    let mut embeddings = Vec::with_capacity(10_000);
    for index in 0..10_000 {
        let id = format!("crash-{index}");
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
            vector: vector(index % DIMENSIONS),
        });
    }
    std::fs::write(ready_path, b"ready").unwrap();
    let _receipt = store.apply_batch(&WriteBatch {
        nodes,
        embeddings,
        ..WriteBatch::default()
    });
}

#[test]
fn killed_writer_recovers_to_a_complete_generation() {
    let directory = TempDir::new().unwrap();
    let database_path = directory.path().join("graph.db");
    let ready_path = directory.path().join("ready");
    {
        let mut store = Store::open(&database_path, DIMENSIONS, 0).unwrap();
        let mut initial = batch();
        initial.nodes.truncate(1);
        initial.embeddings.truncate(1);
        initial.edges.clear();
        store.apply_batch(&initial).unwrap();
    }
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "crash_writer_process", "--nocapture"])
        .env(CRASH_TEST_DATABASE, &database_path)
        .env(CRASH_TEST_READY, &ready_path)
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    while !ready_path.exists() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(ready_path.exists(), "child did not begin its write");
    std::thread::sleep(std::time::Duration::from_millis(20));
    child.kill().unwrap();
    child.wait().unwrap();

    let recovered = Store::open(&database_path, DIMENSIONS, 0).unwrap();
    let stats = recovered.stats().unwrap();

    assert!(matches!(stats.generation, 1 | 2));
    assert!(matches!(stats.node_count, 1 | 10_001));
    assert!(recovered.verify_integrity().unwrap().is_healthy());
}
