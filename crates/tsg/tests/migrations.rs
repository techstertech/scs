use rusqlite::Connection;
use tempfile::TempDir;
use tsg::{Error, Store};

const DIMENSIONS: usize = 8;

fn create_version_zero_store(path: &std::path::Path) {
    let connection = Connection::open(path).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE store_metadata (
                singleton INTEGER PRIMARY KEY,
                dimensions INTEGER NOT NULL,
                generation INTEGER NOT NULL
            );
            INSERT INTO store_metadata VALUES (1, 8, 0);
            CREATE TABLE nodes (
                key INTEGER PRIMARY KEY,
                id TEXT NOT NULL UNIQUE,
                repository_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                content TEXT NOT NULL
            );
            INSERT INTO nodes VALUES (1, 'legacy', 1, 'function', 'legacy', 'content');",
        )
        .unwrap();
}

#[test]
fn legacy_schema_requires_an_explicit_reindex() {
    let directory = TempDir::new().unwrap();
    let database_path = directory.path().join("graph.db");
    create_version_zero_store(&database_path);

    let result = Store::open(&database_path, DIMENSIONS, 10);

    assert!(matches!(
        result,
        Err(Error::ReindexRequired {
            found: 0,
            required: 2
        })
    ));
    let connection = Connection::open(&database_path).unwrap();
    let node_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM nodes", [], |row| row.get(0))
        .unwrap();
    assert_eq!(node_count, 1);
}

#[test]
fn future_schema_version_fails_closed() {
    let directory = TempDir::new().unwrap();
    let database_path = directory.path().join("future.db");
    let connection = Connection::open(&database_path).unwrap();
    connection.pragma_update(None, "user_version", 99).unwrap();
    drop(connection);

    let Err(error) = Store::open(&database_path, DIMENSIONS, 10) else {
        panic!("future schema unexpectedly opened");
    };

    assert!(matches!(
        error,
        Error::UnsupportedSchema {
            found: 99,
            supported: 2
        }
    ));
}

#[test]
fn read_only_open_does_not_migrate_old_schema() {
    let directory = TempDir::new().unwrap();
    let database_path = directory.path().join("legacy.db");
    create_version_zero_store(&database_path);

    let result = Store::builder(&database_path, DIMENSIONS)
        .read_only(true)
        .build();

    assert!(matches!(
        result,
        Err(Error::UnsupportedSchema {
            found: 0,
            supported: 2
        })
    ));
    let version: u32 = Connection::open(&database_path)
        .unwrap()
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(version, 0);
}

#[test]
fn version_one_requires_an_explicit_reindex() {
    let directory = TempDir::new().unwrap();
    let database_path = directory.path().join("version-one.db");
    create_version_zero_store(&database_path);
    Connection::open(&database_path)
        .unwrap()
        .pragma_update(None, "user_version", 1)
        .unwrap();

    let result = Store::open(&database_path, DIMENSIONS, 10);

    assert!(matches!(
        result,
        Err(Error::ReindexRequired {
            found: 1,
            required: 2
        })
    ));
}
