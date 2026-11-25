use tsg::{Embedding, Node, SearchBackend, SearchFilter, Store, WriteBatch};

fn main() -> tsg::Result<()> {
    let mut store = Store::open("graph.db", 8, 10_000)?;
    let mut vector = vec![0.0; 8];
    vector[0] = 1.0;
    store.apply_batch(&WriteBatch {
        nodes: vec![Node {
            id: "example".into(),
            scope_id: None,
            kind: "function".into(),
            name: "example".into(),
            content: "fn example() {}".into(),
            attributes: serde_json::json!({}),
        }],
        embeddings: vec![Embedding {
            node_id: "example".into(),
            vector: vector.clone(),
        }],
        ..WriteBatch::default()
    })?;
    let results = store.search(
        &vector,
        10,
        SearchFilter::default(),
        SearchBackend::Adaptive,
    )?;
    println!("{}", results.hits[0].node.id);
    Ok(())
}
