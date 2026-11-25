use proptest::prelude::*;
use tempfile::TempDir;
use tsg::{Embedding, Error, Node, SearchBackend, SearchFilter, Store, WriteBatch};

const DIMENSIONS: usize = 8;

fn node_and_embedding(index: usize) -> (Node, Embedding) {
    let id = format!("node-{index}");
    let mut vector = vec![0.0; DIMENSIONS];
    vector[index % DIMENSIONS] = 1.0;
    (
        Node {
            id: id.clone(),
            scope_id: None,
            kind: "function".to_string(),
            name: id.clone(),
            content: id.clone(),
            attributes: serde_json::json!({}),
        },
        Embedding {
            node_id: id,
            vector,
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn exact_and_usearch_select_the_same_orthogonal_neighbor(
        corpus_size in 1_usize..=DIMENSIONS,
        raw_axis in 0_usize..DIMENSIONS,
    ) {
        let axis = raw_axis % corpus_size;
        let directory = TempDir::new().unwrap();
        let mut store = Store::open(directory.path().join("graph.db"), DIMENSIONS, 0).unwrap();
        let pairs: Vec<_> = (0..corpus_size).map(node_and_embedding).collect();
        let batch = WriteBatch {
            nodes: pairs.iter().map(|(node, _)| node.clone()).collect(),
            embeddings: pairs.into_iter().map(|(_, embedding)| embedding).collect(),
            ..WriteBatch::default()
        };
        store.apply_batch(&batch).unwrap();
        let query = node_and_embedding(axis).1.vector;

        let exact = store.search(&query, 1, SearchFilter::default(), SearchBackend::Exact).unwrap();
        let accelerated = store.search(
            &query,
            1,
            SearchFilter::default(),
            SearchBackend::Usearch,
        ).unwrap();

        prop_assert_eq!(&exact.hits[0].node.id, &accelerated.hits[0].node.id);
        prop_assert!((exact.hits[0].distance - accelerated.hits[0].distance).abs() < 1e-6);
    }

    #[test]
    fn every_non_finite_coordinate_is_rejected(index in 0_usize..DIMENSIONS, nan in any::<bool>()) {
        let directory = TempDir::new().unwrap();
        let mut store = Store::open(directory.path().join("graph.db"), DIMENSIONS, 10).unwrap();
        let (node, mut embedding) = node_and_embedding(0);
        embedding.vector[index] = if nan { f32::NAN } else { f32::INFINITY };
        let batch = WriteBatch {
            nodes: vec![node],
            embeddings: vec![embedding],
            ..WriteBatch::default()
        };

        prop_assert!(matches!(store.apply_batch(&batch), Err(Error::InvalidInput(_))));
        prop_assert_eq!(store.node_count().unwrap(), 0);
        prop_assert_eq!(store.generation().unwrap(), 0);
    }
}
