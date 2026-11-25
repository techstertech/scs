//! Generated fixtures for indexed attribute lookup semantics and path safety.

use tsg::{AttributeFilter, Node, Store, WriteBatch};

#[test]
fn attribute_hits_misses_scope_and_pagination_preserve_semantics() {
    let directory = tempfile::tempdir().unwrap();
    let mut store = Store::builder(directory.path().join("graph.db"), 8)
        .node_attribute_indexes(["$.symbol.name"])
        .build()
        .unwrap();
    let left = store.get_or_create_scope("left").unwrap().id;
    let right = store.get_or_create_scope("right").unwrap().id;
    let nodes = (0..2048)
        .map(|index| Node {
            id: format!("node-{index:04}"),
            scope_id: Some(if index % 2 == 0 { left } else { right }),
            kind: "symbol".into(),
            name: format!("symbol-{index}"),
            content: String::new(),
            attributes: serde_json::json!({"symbol": {"name": format!("group-{}", index / 8)}}),
        })
        .collect();
    store
        .apply_batch(&WriteBatch {
            nodes,
            ..WriteBatch::default()
        })
        .unwrap();
    let value = serde_json::json!("group-100");
    let lookup = |scope, limit, offset| {
        store
            .find_nodes_by_attribute(
                scope,
                AttributeFilter {
                    path: "$.symbol.name",
                    value: &value,
                },
                limit,
                offset,
            )
            .unwrap()
            .into_iter()
            .map(|node| node.id)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        lookup(Some(left), 10, 0),
        (800..808)
            .step_by(2)
            .map(|index| format!("node-{index:04}"))
            .collect::<Vec<_>>()
    );
    assert_eq!(lookup(Some(right), 2, 1), vec!["node-0803", "node-0805"]);
    assert_eq!(
        lookup(None, 3, 2),
        vec!["node-0802", "node-0803", "node-0804"]
    );
    for scope in [Some(left), Some(right), None] {
        for missing in [
            serde_json::json!("absent"),
            serde_json::json!("' OR 1=1 --"),
        ] {
            assert!(
                store
                    .find_nodes_by_attribute(
                        scope,
                        AttributeFilter {
                            path: "$.symbol.name",
                            value: &missing
                        },
                        1,
                        0
                    )
                    .unwrap()
                    .is_empty()
            );
        }
    }
    for path in [
        "$.symbol.name') OR 1=1 --",
        "$['symbol']",
        "$.symbol;DROP TABLE nodes",
    ] {
        assert!(
            store
                .find_nodes_by_attribute(
                    Some(left),
                    AttributeFilter {
                        path,
                        value: &value
                    },
                    1,
                    0
                )
                .is_err()
        );
    }
    assert_eq!(store.node_count().unwrap(), 2048);
}
