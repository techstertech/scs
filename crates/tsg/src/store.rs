use std::collections::{HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use fs2::FileExt;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};

use crate::error::{Error, Result};
use crate::types::{
    AttributeFilter, CatalogRecord, CommitReceipt, DeleteReceipt, Direction, Durability, Edge,
    IntegrityReport, Node, NodeFilter, Scope, SearchBackend, SearchFilter, SearchResults,
    StoreStats, WriteBatch,
};
use crate::vector::{VectorAccelerator, encode_vector, exact_search};

const CURRENT_SCHEMA_VERSION: u32 = 2;
const SCHEMA: &str = r"
CREATE TABLE IF NOT EXISTS store_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    dimensions INTEGER NOT NULL,
    generation INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    key INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    scope_id INTEGER REFERENCES scopes(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    content TEXT NOT NULL,
    attributes TEXT NOT NULL CHECK(json_valid(attributes))
);

CREATE TABLE IF NOT EXISTS scopes (
    id INTEGER PRIMARY KEY,
    key TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS nodes_scope_kind
ON nodes(scope_id, kind, key);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT NOT NULL UNIQUE,
    source_key INTEGER NOT NULL REFERENCES nodes(key) ON DELETE CASCADE,
    target_key INTEGER NOT NULL REFERENCES nodes(key) ON DELETE CASCADE,
    relationship TEXT NOT NULL,
    weight REAL NOT NULL,
    attributes TEXT NOT NULL CHECK(json_valid(attributes)),
    PRIMARY KEY (source_key, relationship, target_key)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS edges_target
ON edges(target_key, relationship, source_key);

CREATE TABLE IF NOT EXISTS embeddings (
    node_key INTEGER PRIMARY KEY REFERENCES nodes(key) ON DELETE CASCADE,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS catalog (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL CHECK(json_valid(value)),
    PRIMARY KEY(namespace, key)
) WITHOUT ROWID;
";

/// Configures and opens a [`Store`].
pub struct StoreBuilder {
    database_path: PathBuf,
    dimensions: usize,
    exact_search_threshold: usize,
    durability: Durability,
    read_only: bool,
    node_attribute_indexes: Vec<String>,
}

impl StoreBuilder {
    /// Creates a builder with full durability and a 10,000-candidate exact threshold.
    #[must_use]
    pub fn new(database_path: impl Into<PathBuf>, dimensions: usize) -> Self {
        Self {
            database_path: database_path.into(),
            dimensions,
            exact_search_threshold: 10_000,
            durability: Durability::Full,
            read_only: false,
            node_attribute_indexes: Vec::new(),
        }
    }

    /// Sets the maximum candidate count served by exact scan in adaptive mode.
    #[must_use]
    pub fn exact_search_threshold(mut self, threshold: usize) -> Self {
        self.exact_search_threshold = threshold;
        self
    }

    /// Sets the power-loss durability policy.
    #[must_use]
    pub fn durability(mut self, durability: Durability) -> Self {
        self.durability = durability;
        self
    }

    /// Selects a non-mutating read-only handle.
    #[must_use]
    pub fn read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Declares validated node-attribute JSON paths that require equality indexes.
    #[must_use]
    pub fn node_attribute_indexes<I, S>(mut self, paths: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.node_attribute_indexes = paths.into_iter().map(Into::into).collect();
        self
    }

    /// Opens the configured store.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid configuration, lock contention, schema
    /// incompatibility, or storage initialization failure.
    pub fn build(self) -> Result<Store> {
        Store::open_builder(&self)
    }
}

/// Transactional semantic graph handle.
pub struct Store {
    connection: Connection,
    dimensions: usize,
    exact_search_threshold: usize,
    accelerator: Mutex<Option<VectorAccelerator>>,
    vector_path: PathBuf,
    durability: Durability,
    read_only: bool,
    writer_lock: Option<File>,
}

impl Drop for Store {
    fn drop(&mut self) {
        if let Some(writer_lock) = &self.writer_lock {
            // Explicit release avoids relying on platform-specific close timing.
            let _ = FileExt::unlock(writer_lock);
        }
    }
}

impl Store {
    /// Opens or creates a TSG store and reconstructs its vector accelerator when needed.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions, incompatible existing stores, or
    /// filesystem and `SQLite` failures. Accelerator failures degrade to exact search.
    pub fn open(
        database_path: impl AsRef<Path>,
        dimensions: usize,
        exact_search_threshold: usize,
    ) -> Result<Self> {
        StoreBuilder::new(database_path.as_ref().to_path_buf(), dimensions)
            .exact_search_threshold(exact_search_threshold)
            .build()
    }

    fn open_builder(builder: &StoreBuilder) -> Result<Self> {
        if builder.dimensions == 0 {
            return Err(Error::InvalidInput(
                "embedding dimensions must be positive".to_string(),
            ));
        }
        let database_path = &builder.database_path;
        if !builder.read_only
            && let Some(parent) = database_path.parent()
        {
            std::fs::create_dir_all(parent)?;
        }
        if builder.read_only && !database_path.exists() {
            return Err(Error::InvalidInput(format!(
                "read-only database does not exist: {}",
                database_path.display()
            )));
        }

        let writer_lock = if builder.read_only {
            None
        } else {
            Some(acquire_writer_lock(database_path)?)
        };
        let flags = if builder.read_only {
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX
        } else {
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX
        };
        let mut connection = Connection::open_with_flags(database_path, flags)?;
        connection.execute_batch("PRAGMA foreign_keys = ON;")?;
        if builder.read_only {
            validate_schema(&connection)?;
        } else {
            connection.execute_batch("PRAGMA journal_mode = WAL;")?;
            match builder.durability {
                Durability::Full => connection.execute_batch("PRAGMA synchronous = FULL;")?,
                Durability::Normal => connection.execute_batch("PRAGMA synchronous = NORMAL;")?,
            }
            initialize_schema(
                &mut connection,
                database_path,
                builder.dimensions,
                builder.durability,
            )?;
            create_node_attribute_indexes(&connection, &builder.node_attribute_indexes)?;
        }

        let (stored_dimensions, generation): (i64, i64) = connection.query_row(
            "SELECT dimensions, generation FROM store_metadata WHERE singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let stored_dimensions = sql_usize(stored_dimensions, "embedding dimensions")?;
        let generation = sql_u64(generation, "store generation")?;
        if stored_dimensions != builder.dimensions {
            return Err(Error::InvalidInput(format!(
                "embedding dimension mismatch: stored {stored_dimensions}, requested {}",
                builder.dimensions
            )));
        }

        let vector_path = database_path.with_extension("usearch");
        let accelerator = if builder.read_only {
            VectorAccelerator::open_existing(vector_path.clone(), builder.dimensions, generation)
                .ok()
        } else {
            VectorAccelerator::open_or_rebuild(
                &connection,
                vector_path.clone(),
                builder.dimensions,
                generation,
                builder.durability,
            )
            .ok()
        };

        Ok(Self {
            connection,
            dimensions: builder.dimensions,
            exact_search_threshold: builder.exact_search_threshold,
            accelerator: Mutex::new(accelerator),
            vector_path,
            durability: builder.durability,
            read_only: builder.read_only,
            writer_lock,
        })
    }

    /// Creates a configurable store builder.
    #[must_use]
    pub fn builder(database_path: impl Into<PathBuf>, dimensions: usize) -> StoreBuilder {
        StoreBuilder::new(database_path, dimensions)
    }

    /// Returns whether this handle rejects mutation and repair.
    #[must_use]
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Creates an application scope if absent and returns its stable identity.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty key, a read-only handle, or a storage failure.
    pub fn get_or_create_scope(&mut self, key: &str) -> Result<Scope> {
        self.require_writable()?;
        if key.trim().is_empty() {
            return Err(Error::InvalidInput(
                "scope key must not be empty".to_string(),
            ));
        }
        self.connection
            .execute("INSERT OR IGNORE INTO scopes(key) VALUES (?1)", [key])?;
        self.connection
            .query_row("SELECT id, key FROM scopes WHERE key = ?1", [key], |row| {
                Ok(Scope {
                    id: row.get(0)?,
                    key: row.get(1)?,
                })
            })
            .map_err(Into::into)
    }

    /// Resolves a scope by its caller-owned key.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty key or a storage failure.
    pub fn scope_by_key(&self, key: &str) -> Result<Option<Scope>> {
        if key.trim().is_empty() {
            return Err(Error::InvalidInput(
                "scope key must not be empty".to_string(),
            ));
        }
        self.connection
            .query_row("SELECT id, key FROM scopes WHERE key = ?1", [key], |row| {
                Ok(Scope {
                    id: row.get(0)?,
                    key: row.get(1)?,
                })
            })
            .optional()
            .map_err(Into::into)
    }

    /// Resolves a scope by its TSG-assigned integer identifier.
    ///
    /// # Errors
    ///
    /// Returns an error when the storage query fails.
    pub fn scope_by_id(&self, id: i64) -> Result<Option<Scope>> {
        self.connection
            .query_row("SELECT id, key FROM scopes WHERE id = ?1", [id], |row| {
                Ok(Scope {
                    id: row.get(0)?,
                    key: row.get(1)?,
                })
            })
            .optional()
            .map_err(Into::into)
    }

    /// Returns one graph node by its caller-owned identifier.
    ///
    /// # Errors
    ///
    /// Returns an error when durable data is malformed or the query fails.
    pub fn get_node(&self, id: &str) -> Result<Option<Node>> {
        let key: Option<i64> = self
            .connection
            .query_row("SELECT key FROM nodes WHERE id = ?1", [id], |row| {
                row.get(0)
            })
            .optional()?;

        key.map(|key| load_node(&self.connection, key)).transpose()
    }

    /// Lists nodes in stable identifier order with bounded pagination.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid pagination or a storage failure.
    pub fn list_nodes(
        &self,
        filter: NodeFilter<'_>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Node>> {
        let (limit, offset) = pagination(limit, offset)?;
        let mut statement = self.connection.prepare(
            "SELECT key FROM nodes
             WHERE (?1 IS NULL OR scope_id = ?1) AND (?2 IS NULL OR kind = ?2)
             ORDER BY id LIMIT ?3 OFFSET ?4",
        )?;
        let keys = statement
            .query_map((filter.scope_id, filter.kind, limit, offset), |row| {
                row.get(0)
            })?
            .collect::<std::result::Result<Vec<i64>, _>>()?;

        keys.into_iter()
            .map(|key| load_node(&self.connection, key))
            .collect()
    }

    /// Lists nodes whose JSON attribute at `path` equals the supplied value.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsafe JSON path, invalid pagination, malformed
    /// JSON, or a storage failure.
    pub fn find_nodes_by_attribute(
        &self,
        scope_id: Option<i64>,
        filter: AttributeFilter<'_>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Node>> {
        let sql = node_attribute_query(filter.path, scope_id.is_some())?;
        let (limit, offset) = pagination(limit, offset)?;
        let value = serde_json::to_string(filter.value)
            .map_err(|error| Error::InvalidInput(format!("serialize filter JSON: {error}")))?;
        let mut statement = self.connection.prepare(&sql)?;
        let keys = statement
            .query_map((scope_id, value, limit, offset), |row| row.get(0))?
            .collect::<std::result::Result<Vec<i64>, _>>()?;

        keys.into_iter()
            .map(|key| load_node(&self.connection, key))
            .collect()
    }

    /// Lists nodes that do not yet have canonical embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid pagination or a storage failure.
    pub fn list_nodes_without_embeddings(
        &self,
        filter: NodeFilter<'_>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Node>> {
        let (limit, offset) = pagination(limit, offset)?;
        let mut statement = self.connection.prepare(
            "SELECT n.key FROM nodes AS n LEFT JOIN embeddings AS e ON e.node_key = n.key
             WHERE e.node_key IS NULL
               AND (?1 IS NULL OR n.scope_id = ?1) AND (?2 IS NULL OR n.kind = ?2)
             ORDER BY n.id LIMIT ?3 OFFSET ?4",
        )?;
        let keys = statement
            .query_map((filter.scope_id, filter.kind, limit, offset), |row| {
                row.get(0)
            })?
            .collect::<std::result::Result<Vec<i64>, _>>()?;

        keys.into_iter()
            .map(|key| load_node(&self.connection, key))
            .collect()
    }

    /// Counts nodes matching an optional scope and kind filter.
    ///
    /// # Errors
    ///
    /// Returns an error when the storage query fails.
    pub fn count_nodes(&self, filter: NodeFilter<'_>) -> Result<usize> {
        let count = self.connection.query_row(
            "SELECT COUNT(*) FROM nodes
             WHERE (?1 IS NULL OR scope_id = ?1) AND (?2 IS NULL OR kind = ?2)",
            (filter.scope_id, filter.kind),
            |row| row.get(0),
        )?;
        sql_usize(count, "filtered node count")
    }

    /// Counts matching nodes without canonical embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error when the storage query fails.
    pub fn count_nodes_without_embeddings(&self, filter: NodeFilter<'_>) -> Result<usize> {
        let count = self.connection.query_row(
            "SELECT COUNT(*) FROM nodes AS n LEFT JOIN embeddings AS e ON e.node_key = n.key
             WHERE e.node_key IS NULL
               AND (?1 IS NULL OR n.scope_id = ?1) AND (?2 IS NULL OR n.kind = ?2)",
            (filter.scope_id, filter.kind),
            |row| row.get(0),
        )?;
        sql_usize(count, "missing embedding count")
    }

    /// Fetches existing nodes for a bounded set of caller-owned identifiers.
    ///
    /// # Errors
    ///
    /// Returns an error for duplicate identifiers or malformed durable data.
    pub fn get_nodes(&self, ids: &[String]) -> Result<Vec<Node>> {
        let mut unique = HashSet::new();
        if ids.iter().any(|id| !unique.insert(id.as_str())) {
            return Err(Error::InvalidInput("node IDs must be unique".to_string()));
        }
        ids.iter()
            .filter_map(|id| self.get_node(id).transpose())
            .collect()
    }

    /// Performs bounded case-insensitive substring matching over node names.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty query, invalid pagination, or storage failure.
    pub fn find_nodes_by_name(
        &self,
        query: &str,
        filter: NodeFilter<'_>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Node>> {
        if query.trim().is_empty() {
            return Err(Error::InvalidInput(
                "name query must not be empty".to_string(),
            ));
        }
        let (limit, offset) = pagination(limit, offset)?;
        let pattern = format!(
            "%{}%",
            query
                .replace('\\', "\\\\")
                .replace('%', "\\%")
                .replace('_', "\\_")
        );
        let mut statement = self.connection.prepare(
            "SELECT key FROM nodes WHERE name LIKE ?1 ESCAPE '\\' COLLATE NOCASE
             AND (?2 IS NULL OR scope_id = ?2) AND (?3 IS NULL OR kind = ?3)
             ORDER BY name, id LIMIT ?4 OFFSET ?5",
        )?;
        let keys = statement
            .query_map(
                (pattern, filter.scope_id, filter.kind, limit, offset),
                |row| row.get(0),
            )?
            .collect::<std::result::Result<Vec<i64>, _>>()?;
        keys.into_iter()
            .map(|key| load_node(&self.connection, key))
            .collect()
    }

    /// Atomically applies nodes, edges, embeddings, and a new durable generation.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is invalid, references an unknown node, or
    /// cannot be committed to `SQLite`. A post-commit accelerator failure is reported
    /// through [`CommitReceipt::accelerator_ready`] instead of as a commit failure.
    #[allow(clippy::too_many_lines)] // Keeping every mutation visibly inside one transaction guards atomicity.
    pub fn apply_batch(&mut self, batch: &WriteBatch) -> Result<CommitReceipt> {
        self.require_writable()?;
        self.validate_batch(batch)?;
        if batch.nodes.is_empty()
            && batch.edges.is_empty()
            && batch.embeddings.is_empty()
            && batch.catalog_records.is_empty()
            && batch.catalog_deletes.is_empty()
        {
            return Err(Error::InvalidInput(
                "write batch must not be empty".to_string(),
            ));
        }

        let transaction = self.connection.transaction()?;
        let previous_generation: i64 = transaction.query_row(
            "SELECT generation FROM store_metadata WHERE singleton = 1",
            [],
            |row| row.get(0),
        )?;
        let previous_generation = sql_u64(previous_generation, "store generation")?;
        let mut changed_vectors = Vec::with_capacity(batch.embeddings.len());
        for node in &batch.nodes {
            let attributes = serde_json::to_string(&node.attributes)
                .map_err(|error| Error::InvalidInput(format!("serialize node JSON: {error}")))?;
            transaction.execute(
                "INSERT INTO nodes(id, scope_id, kind, name, content, attributes)
                 VALUES (?1, ?2, ?3, ?4, ?5, json(?6))
                 ON CONFLICT(id) DO UPDATE SET
                    scope_id = COALESCE(excluded.scope_id, nodes.scope_id),
                    kind = excluded.kind,
                    name = excluded.name,
                    content = excluded.content,
                    attributes = json_patch(nodes.attributes, excluded.attributes)",
                params![
                    node.id,
                    node.scope_id,
                    node.kind,
                    node.name,
                    node.content,
                    attributes
                ],
            )?;
        }
        // Check the final node scopes, so connected endpoints can move together
        // while a node-only upsert cannot invalidate an existing edge.
        for node in &batch.nodes {
            let key = node_key(&transaction, &node.id)?;
            let crossing: Option<(String, String)> = transaction
                .query_row(
                    "SELECT source.id, target.id FROM edges AS edge
                     JOIN nodes AS source ON source.key = edge.source_key
                     JOIN nodes AS target ON target.key = edge.target_key
                     WHERE (edge.source_key = ?1 OR edge.target_key = ?1)
                       AND source.scope_id IS NOT target.scope_id
                     LIMIT 1",
                    [key],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            if let Some((source, target)) = crossing {
                return Err(Error::InvalidInput(format!(
                    "edge crosses scope boundary: {source} -> {target}"
                )));
            }
        }
        for embedding in &batch.embeddings {
            let key = node_key(&transaction, &embedding.node_id)?;
            transaction.execute(
                "INSERT INTO embeddings(node_key, vector) VALUES (?1, ?2)
                 ON CONFLICT(node_key) DO UPDATE SET vector = excluded.vector",
                params![key, encode_vector(&embedding.vector)],
            )?;
            changed_vectors.push((sql_u64(key, "node key")?, embedding.vector.as_slice()));
        }
        for edge in &batch.edges {
            let source_key = node_key(&transaction, &edge.source_id)?;
            let target_key = node_key(&transaction, &edge.target_id)?;
            let source_scope = node_scope(&transaction, source_key)?;
            let target_scope = node_scope(&transaction, target_key)?;
            if source_scope != target_scope {
                return Err(Error::InvalidInput(format!(
                    "edge crosses scope boundary: {} -> {}",
                    edge.source_id, edge.target_id
                )));
            }
            transaction.execute(
                "INSERT INTO edges(id, source_key, target_key, relationship, weight, attributes)
                 VALUES (?1, ?2, ?3, ?4, ?5, json(?6))
                 ON CONFLICT(source_key, relationship, target_key) DO UPDATE SET
                    id = excluded.id,
                    weight = excluded.weight,
                    attributes = json_patch(edges.attributes, excluded.attributes)",
                params![
                    edge.id,
                    source_key,
                    target_key,
                    edge.relationship,
                    edge.weight,
                    serde_json::to_string(&edge.attributes).map_err(
                        |error| Error::InvalidInput(format!("serialize edge JSON: {error}"))
                    )?
                ],
            )?;
        }
        for record in &batch.catalog_records {
            let value = serde_json::to_string(&record.value)
                .map_err(|error| Error::InvalidInput(format!("serialize catalog JSON: {error}")))?;
            transaction.execute(
                "INSERT INTO catalog(namespace, key, value) VALUES (?1, ?2, json(?3))
                 ON CONFLICT(namespace, key) DO UPDATE SET value = excluded.value",
                params![record.namespace, record.key, value],
            )?;
        }
        for record in &batch.catalog_deletes {
            transaction.execute(
                "DELETE FROM catalog WHERE namespace = ?1 AND key = ?2",
                params![record.namespace, record.key],
            )?;
        }
        let generation: i64 = transaction.query_row(
            "UPDATE store_metadata SET generation = generation + 1
             WHERE singleton = 1 RETURNING generation",
            [],
            |row| row.get(0),
        )?;
        let generation = sql_u64(generation, "store generation")?;
        transaction.commit()?;

        let mut accelerator = self
            .accelerator
            .lock()
            .map_err(|_| Error::Storage("vector accelerator lock is poisoned".to_string()))?;
        let accelerator_ready = if let Some(current) = accelerator
            .as_mut()
            .filter(|current| current.is_current(previous_generation))
        {
            // SQL has committed. Any partial remove/add or persistence failure
            // invalidates this accelerator; readers must use authoritative SQL.
            if current
                .upsert(&changed_vectors, generation, self.durability)
                .is_ok()
            {
                true
            } else {
                *accelerator = None;
                false
            }
        } else {
            *accelerator = VectorAccelerator::rebuild(
                &self.connection,
                self.vector_path.clone(),
                self.dimensions,
                generation,
                self.durability,
            )
            .ok();
            accelerator.is_some()
        };

        Ok(CommitReceipt {
            generation,
            accelerator_ready,
        })
    }

    /// Deletes nodes and their edges and embeddings in one transaction.
    ///
    /// Unknown IDs are ignored. The durable generation advances only when at
    /// least one node is deleted.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty or duplicate ID, a read-only handle, or a
    /// failed `SQLite` transaction.
    pub fn delete_nodes(&mut self, node_ids: &[String]) -> Result<DeleteReceipt> {
        self.require_writable()?;
        if node_ids.is_empty() || node_ids.iter().any(|node_id| node_id.trim().is_empty()) {
            return Err(Error::InvalidInput(
                "delete requires at least one non-empty node ID".to_string(),
            ));
        }
        let unique: HashSet<&str> = node_ids.iter().map(String::as_str).collect();
        if unique.len() != node_ids.len() {
            return Err(Error::InvalidInput(
                "delete node IDs must be unique".to_string(),
            ));
        }

        let previous_generation = self.generation()?;

        let transaction = self.connection.transaction()?;
        let mut nodes_deleted = 0_usize;
        let mut removed_keys = Vec::new();
        for node_id in node_ids {
            let key = transaction
                .query_row("SELECT key FROM nodes WHERE id = ?1", [node_id], |row| {
                    row.get::<_, i64>(0)
                })
                .optional()?;
            let deleted = transaction.execute("DELETE FROM nodes WHERE id = ?1", [node_id])?;
            nodes_deleted += deleted;
            if deleted > 0 {
                let key = key.ok_or_else(|| {
                    Error::Storage("deleted node is missing its storage key".to_string())
                })?;
                removed_keys.push(
                    u64::try_from(key)
                        .map_err(|_| Error::Storage("stored node key is negative".to_string()))?,
                );
            }
        }
        let generation = if nodes_deleted == 0 {
            transaction.rollback()?;
            self.generation()?
        } else {
            let generation: i64 = transaction.query_row(
                "UPDATE store_metadata SET generation = generation + 1
                 WHERE singleton = 1 RETURNING generation",
                [],
                |row| row.get(0),
            )?;
            let generation = sql_u64(generation, "store generation")?;
            transaction.commit()?;
            generation
        };

        let accelerator_ready = if nodes_deleted == 0 {
            self.accelerator_ready(generation)?
        } else {
            let mut accelerator = self
                .accelerator
                .lock()
                .map_err(|_| Error::Storage("vector accelerator lock is poisoned".to_string()))?;
            if let Some(current) = accelerator
                .as_mut()
                .filter(|current| current.is_current(previous_generation))
            {
                // SQL has committed. A failed incremental persistence drops the
                // stale accelerator so authoritative SQL remains available.
                if current
                    .remove(&removed_keys, generation, self.durability)
                    .is_ok()
                {
                    true
                } else {
                    *accelerator = None;
                    false
                }
            } else {
                drop(accelerator);
                self.rebuild_accelerator(generation)
            }
        };
        Ok(DeleteReceipt {
            generation,
            nodes_deleted,
            accelerator_ready,
        })
    }

    /// Removes every canonical embedding while retaining graph and catalog data.
    ///
    /// # Errors
    ///
    /// Returns an error for a read-only handle or failed durable transaction.
    pub fn clear_embeddings(&mut self) -> Result<usize> {
        self.require_writable()?;
        let transaction = self.connection.transaction()?;
        let removed = transaction.execute("DELETE FROM embeddings", [])?;
        if removed == 0 {
            transaction.rollback()?;
            return Ok(0);
        }
        let generation: i64 = transaction.query_row(
            "UPDATE store_metadata SET generation = generation + 1
             WHERE singleton = 1 RETURNING generation",
            [],
            |row| row.get(0),
        )?;
        transaction.commit()?;
        self.rebuild_accelerator(sql_u64(generation, "store generation")?);
        Ok(removed)
    }

    /// Removes all graph, embedding, scope, and catalog records.
    ///
    /// # Errors
    ///
    /// Returns an error for a read-only handle or failed durable transaction.
    pub fn truncate(&mut self) -> Result<usize> {
        self.require_writable()?;
        let transaction = self.connection.transaction()?;
        let removed = transaction.execute("DELETE FROM nodes", [])?;
        transaction.execute("DELETE FROM scopes", [])?;
        transaction.execute("DELETE FROM catalog", [])?;
        let generation: i64 = transaction.query_row(
            "UPDATE store_metadata SET generation = generation + 1
             WHERE singleton = 1 RETURNING generation",
            [],
            |row| row.get(0),
        )?;
        transaction.commit()?;
        self.rebuild_accelerator(sql_u64(generation, "store generation")?);
        Ok(removed)
    }

    /// Reclaims unused pages in the authoritative SQLite database.
    ///
    /// # Errors
    ///
    /// Returns an error for a read-only handle or storage failure.
    pub fn vacuum(&mut self) -> Result<()> {
        self.require_writable()?;
        self.connection.execute_batch("VACUUM")?;
        Ok(())
    }

    /// Searches canonical embeddings using the requested retrieval strategy.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid query, unavailable requested accelerator,
    /// poisoned internal synchronization, or storage failure.
    pub fn search(
        &self,
        query: &[f32],
        limit: usize,
        filter: SearchFilter<'_>,
        requested_backend: SearchBackend,
    ) -> Result<SearchResults> {
        self.validate_vector(query, "query")?;
        if limit == 0 {
            return Err(Error::InvalidInput(
                "search limit must be positive".to_string(),
            ));
        }

        let candidate_count = self.candidate_count(filter)?;
        // A failed sidecar write never invalidates canonical embeddings. Both
        // writable and read-only handles must retain adaptive exact fallback.
        let accelerator_unavailable = !self.accelerator_ready(self.generation()?)?;
        let backend = match requested_backend {
            SearchBackend::Adaptive
                if candidate_count <= self.exact_search_threshold || accelerator_unavailable =>
            {
                SearchBackend::Exact
            }
            SearchBackend::Adaptive => SearchBackend::Usearch,
            explicit => explicit,
        };
        let hits = match backend {
            SearchBackend::Exact => exact_search(&self.connection, query, limit, filter)?,
            SearchBackend::Usearch => {
                let generation = self.generation()?;
                let mut accelerator = self.accelerator.lock().map_err(|_| {
                    Error::Storage("vector accelerator lock is poisoned".to_string())
                })?;
                let needs_rebuild = accelerator.as_ref().is_none_or(|index| {
                    !index.is_current(generation)
                        || index.dimensions() != self.dimensions
                        || index.path() != self.vector_path
                });
                if needs_rebuild {
                    if self.read_only {
                        return Err(Error::AcceleratorUnavailable(
                            "read-only stores cannot rebuild a missing or stale sidecar"
                                .to_string(),
                        ));
                    }
                    *accelerator = Some(VectorAccelerator::rebuild(
                        &self.connection,
                        self.vector_path.clone(),
                        self.dimensions,
                        generation,
                        self.durability,
                    )?);
                }
                accelerator
                    .as_ref()
                    .ok_or_else(|| {
                        Error::Storage("vector accelerator failed to initialize".to_string())
                    })?
                    .search(&self.connection, query, limit, filter)?
            }
            SearchBackend::Adaptive => unreachable!("adaptive backend resolves before execution"),
        };

        Ok(SearchResults { backend, hits })
    }

    /// Traverses a bounded, scope-local graph neighborhood.
    ///
    /// # Errors
    ///
    /// Returns an error when the start node does not exist or a storage query fails.
    pub fn traverse(
        &self,
        start_id: &str,
        direction: Direction,
        relationship: Option<&str>,
        max_hops: usize,
        max_results: usize,
    ) -> Result<Vec<Node>> {
        if max_hops == 0 || max_results == 0 {
            return Ok(Vec::new());
        }
        let start_key = node_key(&self.connection, start_id)?;
        let scope_id: Option<i64> = self.connection.query_row(
            "SELECT scope_id FROM nodes WHERE key = ?1",
            [start_key],
            |row| row.get(0),
        )?;
        let mut visited = HashSet::from([start_key]);
        let mut frontier = VecDeque::from([(start_key, 0_usize)]);
        let mut nodes = Vec::new();

        while let Some((key, depth)) = frontier.pop_front() {
            if depth == max_hops {
                continue;
            }
            let mut neighbors = self.neighbor_keys(key, direction, relationship)?;
            neighbors.sort_unstable();
            for neighbor_key in neighbors {
                if !visited.insert(neighbor_key) {
                    continue;
                }
                let node = load_node(&self.connection, neighbor_key)?;
                if node.scope_id != scope_id {
                    continue;
                }
                frontier.push_back((neighbor_key, depth + 1));
                nodes.push(node);
                if nodes.len() == max_results {
                    return Ok(nodes);
                }
            }
        }

        Ok(nodes)
    }

    /// Returns incident edges in stable identifier order.
    ///
    /// # Errors
    ///
    /// Returns an error when the node is unknown or durable data is malformed.
    pub fn get_edges(
        &self,
        node_id: &str,
        direction: Direction,
        relationship: Option<&str>,
    ) -> Result<Vec<Edge>> {
        let key = node_key(&self.connection, node_id)?;
        let mut edge_ids = Vec::new();
        if matches!(direction, Direction::Outgoing | Direction::Both) {
            edge_ids.extend(self.edge_ids("source_key", key, relationship)?);
        }
        if matches!(direction, Direction::Incoming | Direction::Both) {
            edge_ids.extend(self.edge_ids("target_key", key, relationship)?);
        }
        edge_ids.sort_unstable();
        edge_ids.dedup();
        let mut edges = edge_ids
            .into_iter()
            .map(|edge_id| load_edge(&self.connection, &edge_id))
            .collect::<Result<Vec<_>>>()?;
        edges.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(edges)
    }

    /// Deletes an edge by its caller-owned identifier.
    ///
    /// # Errors
    ///
    /// Returns an error for a read-only handle or a storage failure.
    pub fn delete_edge(&mut self, edge_id: &str) -> Result<bool> {
        self.require_writable()?;
        Ok(self
            .connection
            .execute("DELETE FROM edges WHERE id = ?1", [edge_id])?
            > 0)
    }

    /// Returns the current durable store generation.
    ///
    /// # Errors
    ///
    /// Returns an error when metadata cannot be read.
    pub fn generation(&self) -> Result<u64> {
        let generation = self.connection.query_row(
            "SELECT generation FROM store_metadata WHERE singleton = 1",
            [],
            |row| row.get(0),
        )?;

        sql_u64(generation, "store generation")
    }

    /// Returns the number of durable nodes.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` cannot execute the count.
    pub fn node_count(&self) -> Result<usize> {
        let count = self
            .connection
            .query_row("SELECT COUNT(*) FROM nodes", [], |row| row.get(0))?;

        sql_usize(count, "node count")
    }

    /// Returns the number of durable canonical embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` cannot execute the count.
    pub fn embedding_count(&self) -> Result<usize> {
        let count = self
            .connection
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))?;

        sql_usize(count, "embedding count")
    }

    /// Returns one application catalog record by namespace and key.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid identifiers, malformed durable JSON, or a
    /// storage query failure.
    pub fn catalog_get(&self, namespace: &str, key: &str) -> Result<Option<CatalogRecord>> {
        validate_catalog_identity(namespace, key)?;
        let value: Option<String> = self
            .connection
            .query_row(
                "SELECT value FROM catalog WHERE namespace = ?1 AND key = ?2",
                (namespace, key),
                |row| row.get(0),
            )
            .optional()?;

        value
            .map(|value| {
                Ok(CatalogRecord {
                    namespace: namespace.to_string(),
                    key: key.to_string(),
                    value: serde_json::from_str(&value).map_err(|error| {
                        Error::Storage(format!("stored catalog JSON is malformed: {error}"))
                    })?,
                })
            })
            .transpose()
    }

    /// Lists one namespace in stable key order with bounded pagination.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid namespace, a zero limit, malformed
    /// durable JSON, or a storage query failure.
    pub fn catalog_list(
        &self,
        namespace: &str,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<CatalogRecord>> {
        validate_catalog_namespace(namespace)?;
        if limit == 0 {
            return Err(Error::InvalidInput(
                "catalog list limit must be positive".to_string(),
            ));
        }
        let limit = i64::try_from(limit)
            .map_err(|_| Error::InvalidInput("catalog list limit is too large".to_string()))?;
        let offset = i64::try_from(offset)
            .map_err(|_| Error::InvalidInput("catalog list offset is too large".to_string()))?;
        let mut statement = self.connection.prepare(
            "SELECT key, value FROM catalog
             WHERE namespace = ?1 ORDER BY key LIMIT ?2 OFFSET ?3",
        )?;
        let rows = statement.query_map((namespace, limit, offset), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut records = Vec::new();
        for row in rows {
            let (key, value) = row?;
            records.push(CatalogRecord {
                namespace: namespace.to_string(),
                key,
                value: serde_json::from_str(&value).map_err(|error| {
                    Error::Storage(format!("stored catalog JSON is malformed: {error}"))
                })?,
            });
        }
        Ok(records)
    }

    /// Returns operational counts and accelerator readiness.
    ///
    /// # Errors
    ///
    /// Returns an error when durable metadata or counts cannot be read.
    pub fn stats(&self) -> Result<StoreStats> {
        let generation = self.generation()?;
        let edge_count = self
            .connection
            .query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))?;
        let edge_count = sql_usize(edge_count, "edge count")?;
        let accelerator_ready = self.accelerator_ready(generation)?;

        Ok(StoreStats {
            generation,
            node_count: self.node_count()?,
            edge_count,
            embedding_count: self.embedding_count()?,
            accelerator_ready,
            read_only: self.read_only,
        })
    }

    /// Checks durable relational integrity, vector payload shape, and sidecar currency.
    ///
    /// This operation never repairs or mutates the store.
    ///
    /// # Errors
    ///
    /// Returns an error only when the diagnostic queries themselves cannot run.
    pub fn verify_integrity(&self) -> Result<IntegrityReport> {
        let generation = self.generation()?;
        let integrity: String = self
            .connection
            .query_row("PRAGMA integrity_check", [], |row| row.get(0))?;
        let sqlite_ok = integrity == "ok";
        let mut issues = Vec::new();
        if !sqlite_ok {
            issues.push(format!("SQLite integrity check failed: {integrity}"));
        }
        let expected_bytes = self
            .dimensions
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Error::Storage("embedding byte width overflow".to_string()))?;
        let malformed_vectors: i64 = self.connection.query_row(
            "SELECT COUNT(*) FROM embeddings WHERE length(vector) != ?1",
            [i64::try_from(expected_bytes).map_err(|_| {
                Error::Storage("embedding byte width exceeds SQLite integer range".to_string())
            })?],
            |row| row.get(0),
        )?;
        let malformed_vectors = sql_usize(malformed_vectors, "malformed vector count")?;
        if malformed_vectors > 0 {
            issues.push(format!(
                "{malformed_vectors} canonical vector payload(s) have an invalid byte length"
            ));
        }
        let foreign_key_issue: Option<String> = self
            .connection
            .query_row("PRAGMA foreign_key_check", [], |row| row.get(0))
            .optional()?;
        if foreign_key_issue.is_some() {
            issues.push("foreign-key violations detected".to_string());
        }

        Ok(IntegrityReport {
            generation,
            sqlite_ok,
            accelerator_ready: self.accelerator_ready(generation)?,
            issues,
        })
    }

    /// Rebuilds the `USearch` accelerator from canonical `SQLite` embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error for read-only handles, poisoned synchronization, or
    /// sidecar construction and persistence failure.
    pub fn repair_accelerator(&self) -> Result<()> {
        self.require_writable()?;
        let generation = self.generation()?;
        let rebuilt = VectorAccelerator::rebuild(
            &self.connection,
            self.vector_path.clone(),
            self.dimensions,
            generation,
            self.durability,
        )?;
        *self
            .accelerator
            .lock()
            .map_err(|_| Error::Storage("vector accelerator lock is poisoned".to_string()))? =
            Some(rebuilt);
        Ok(())
    }

    fn require_writable(&self) -> Result<()> {
        if self.read_only {
            Err(Error::ReadOnly)
        } else {
            Ok(())
        }
    }

    fn accelerator_ready(&self, generation: u64) -> Result<bool> {
        Ok(self
            .accelerator
            .lock()
            .map_err(|_| Error::Storage("vector accelerator lock is poisoned".to_string()))?
            .as_ref()
            .is_some_and(|accelerator| accelerator.is_current(generation)))
    }

    fn rebuild_accelerator(&self, generation: u64) -> bool {
        let rebuilt = VectorAccelerator::rebuild(
            &self.connection,
            self.vector_path.clone(),
            self.dimensions,
            generation,
            self.durability,
        )
        .ok();
        let ready = rebuilt.is_some();
        if let Ok(mut accelerator) = self.accelerator.lock() {
            *accelerator = rebuilt;
        }
        ready
    }

    fn validate_batch(&self, batch: &WriteBatch) -> Result<()> {
        let mut node_ids = HashSet::new();
        for node in &batch.nodes {
            if node.id.trim().is_empty() || !node_ids.insert(node.id.as_str()) {
                return Err(Error::InvalidInput(
                    "node IDs must be non-empty and unique within a batch".to_string(),
                ));
            }
            validate_attributes(&node.attributes, "node")?;
        }
        let mut edge_ids = HashSet::new();
        for edge in &batch.edges {
            if edge.id.trim().is_empty() || !edge_ids.insert(edge.id.as_str()) {
                return Err(Error::InvalidInput(
                    "edge IDs must be non-empty and unique within a batch".to_string(),
                ));
            }
            if !edge.weight.is_finite() {
                return Err(Error::InvalidInput(
                    "edge weight must be finite".to_string(),
                ));
            }
            validate_attributes(&edge.attributes, "edge")?;
        }
        let mut embedding_ids = HashSet::new();
        for embedding in &batch.embeddings {
            if !embedding_ids.insert(embedding.node_id.as_str()) {
                return Err(Error::InvalidInput(
                    "embedding node IDs must be unique within a batch".to_string(),
                ));
            }
            self.validate_vector(&embedding.vector, "embedding")?;
        }
        let mut catalog_upserts = HashSet::new();
        for record in &batch.catalog_records {
            validate_catalog_identity(&record.namespace, &record.key)?;
            if !catalog_upserts.insert((record.namespace.as_str(), record.key.as_str())) {
                return Err(Error::InvalidInput(
                    "catalog identities must be unique within batch upserts".to_string(),
                ));
            }
        }
        let mut catalog_deletes = HashSet::new();
        for record in &batch.catalog_deletes {
            validate_catalog_identity(&record.namespace, &record.key)?;
            let identity = (record.namespace.as_str(), record.key.as_str());
            if !catalog_deletes.insert(identity) {
                return Err(Error::InvalidInput(
                    "catalog identities must be unique within batch deletes".to_string(),
                ));
            }
            if catalog_upserts.contains(&identity) {
                return Err(Error::InvalidInput(
                    "catalog identity cannot be upserted and deleted in one batch".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn validate_vector(&self, vector: &[f32], label: &str) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(Error::InvalidInput(format!(
                "{label} dimension mismatch: expected {}, received {}",
                self.dimensions,
                vector.len()
            )));
        }
        if vector.iter().any(|coordinate| !coordinate.is_finite()) {
            return Err(Error::InvalidInput(format!(
                "{label} contains a non-finite coordinate"
            )));
        }
        Ok(())
    }

    fn candidate_count(&self, filter: SearchFilter<'_>) -> Result<usize> {
        let count = self.connection.query_row(
            "SELECT COUNT(*) FROM nodes AS n
             JOIN embeddings AS e ON e.node_key = n.key
             WHERE (?1 IS NULL OR n.scope_id = ?1)
               AND (?2 IS NULL OR n.kind = ?2)",
            (filter.scope_id, filter.kind),
            |row| row.get(0),
        )?;

        sql_usize(count, "search candidate count")
    }

    fn neighbor_keys(
        &self,
        key: i64,
        direction: Direction,
        relationship: Option<&str>,
    ) -> Result<Vec<i64>> {
        let mut keys = Vec::new();
        if matches!(direction, Direction::Outgoing | Direction::Both) {
            let mut statement = self.connection.prepare(
                "SELECT target_key FROM edges
                 WHERE source_key = ?1 AND (?2 IS NULL OR relationship = ?2)",
            )?;
            keys.extend(
                statement
                    .query_map((key, relationship), |row| row.get::<_, i64>(0))?
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            );
        }
        if matches!(direction, Direction::Incoming | Direction::Both) {
            let mut statement = self.connection.prepare(
                "SELECT source_key FROM edges
                 WHERE target_key = ?1 AND (?2 IS NULL OR relationship = ?2)",
            )?;
            keys.extend(
                statement
                    .query_map((key, relationship), |row| row.get::<_, i64>(0))?
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            );
        }
        Ok(keys)
    }

    fn edge_ids(&self, column: &str, key: i64, relationship: Option<&str>) -> Result<Vec<String>> {
        let sql = match column {
            "source_key" => {
                "SELECT id FROM edges WHERE source_key = ?1 AND (?2 IS NULL OR relationship = ?2)"
            }
            "target_key" => {
                "SELECT id FROM edges WHERE target_key = ?1 AND (?2 IS NULL OR relationship = ?2)"
            }
            _ => unreachable!("edge column is selected internally"),
        };
        self.connection
            .prepare(sql)?
            .query_map((key, relationship), |row| row.get(0))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }
}

fn acquire_writer_lock(database_path: &Path) -> Result<File> {
    let lock_path = database_path.with_extension("tsg.lock");
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)?;
    lock.try_lock_exclusive()
        .map_err(|_| Error::WriterLocked(lock_path.display().to_string()))?;
    Ok(lock)
}

fn initialize_schema(
    connection: &mut Connection,
    _database_path: &Path,
    dimensions: usize,
    _durability: Durability,
) -> Result<Option<PathBuf>> {
    let version: u32 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if version > CURRENT_SCHEMA_VERSION {
        return Err(Error::UnsupportedSchema {
            found: version,
            supported: CURRENT_SCHEMA_VERSION,
        });
    }
    let has_existing_schema: bool = connection.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'store_metadata'
        )",
        [],
        |row| row.get(0),
    )?;
    if version < CURRENT_SCHEMA_VERSION && has_existing_schema {
        return Err(Error::ReindexRequired {
            found: version,
            required: CURRENT_SCHEMA_VERSION,
        });
    }
    let backup = None;

    let transaction = connection.transaction()?;
    transaction.execute_batch(SCHEMA)?;
    let dimensions_i64 = i64::try_from(dimensions).map_err(|_| {
        Error::InvalidInput("embedding dimensions exceed SQLite integer range".to_string())
    })?;
    transaction.execute(
        "INSERT OR IGNORE INTO store_metadata(singleton, dimensions, generation)
         VALUES (1, ?1, 0)",
        [dimensions_i64],
    )?;
    transaction.pragma_update(None, "user_version", CURRENT_SCHEMA_VERSION)?;
    transaction.commit()?;
    Ok(backup)
}

fn validate_catalog_namespace(namespace: &str) -> Result<()> {
    if namespace.trim().is_empty() {
        return Err(Error::InvalidInput(
            "catalog namespace must not be empty".to_string(),
        ));
    }
    Ok(())
}

fn validate_attributes(attributes: &serde_json::Value, owner: &str) -> Result<()> {
    if !attributes.is_object() {
        return Err(Error::InvalidInput(format!(
            "{owner} attributes must be a JSON object"
        )));
    }
    Ok(())
}

fn validate_catalog_identity(namespace: &str, key: &str) -> Result<()> {
    validate_catalog_namespace(namespace)?;
    if key.trim().is_empty() {
        return Err(Error::InvalidInput(
            "catalog key must not be empty".to_string(),
        ));
    }
    Ok(())
}

fn validate_schema(connection: &Connection) -> Result<()> {
    let version: u32 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if version != CURRENT_SCHEMA_VERSION {
        return Err(Error::UnsupportedSchema {
            found: version,
            supported: CURRENT_SCHEMA_VERSION,
        });
    }
    Ok(())
}

/**
 * Converts a non-negative `SQLite` integer into an in-memory count without
 * relying on platform-width SQL decoding.
 */
fn sql_usize(value: i64, label: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::Storage(format!("stored {label} is negative or too large")))
}

/**
 * Converts a non-negative `SQLite` integer into a public generation value.
 */
fn sql_u64(value: i64, label: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::Storage(format!("stored {label} is negative")))
}

fn node_key(connection: &Connection, node_id: &str) -> Result<i64> {
    connection
        .query_row("SELECT key FROM nodes WHERE id = ?1", [node_id], |row| {
            row.get(0)
        })
        .optional()?
        .ok_or_else(|| Error::InvalidInput(format!("unknown node ID: {node_id}")))
}

fn load_node(connection: &Connection, key: i64) -> Result<Node> {
    Ok(connection.query_row(
        "SELECT id, scope_id, kind, name, content, attributes FROM nodes WHERE key = ?1",
        [key],
        |row| {
            Ok(Node {
                id: row.get(0)?,
                scope_id: row.get(1)?,
                kind: row.get(2)?,
                name: row.get(3)?,
                content: row.get(4)?,
                attributes: serde_json::from_str(&row.get::<_, String>(5)?).map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(
                        5,
                        rusqlite::types::Type::Text,
                        Box::new(error),
                    )
                })?,
            })
        },
    )?)
}

fn node_scope(connection: &Connection, key: i64) -> Result<Option<i64>> {
    Ok(
        connection.query_row("SELECT scope_id FROM nodes WHERE key = ?1", [key], |row| {
            row.get(0)
        })?,
    )
}

fn load_edge(connection: &Connection, edge_id: &str) -> Result<Edge> {
    Ok(connection.query_row(
        "SELECT e.id, source.id, target.id, e.relationship, e.weight, e.attributes
         FROM edges AS e JOIN nodes AS source ON source.key = e.source_key
         JOIN nodes AS target ON target.key = e.target_key WHERE e.id = ?1",
        [edge_id],
        |row| {
            Ok(Edge {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                relationship: row.get(3)?,
                weight: row.get(4)?,
                attributes: serde_json::from_str(&row.get::<_, String>(5)?).map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(
                        5,
                        rusqlite::types::Type::Text,
                        Box::new(error),
                    )
                })?,
            })
        },
    )?)
}

fn pagination(limit: usize, offset: usize) -> Result<(i64, i64)> {
    if limit == 0 {
        return Err(Error::InvalidInput(
            "pagination limit must be positive".to_string(),
        ));
    }
    Ok((
        i64::try_from(limit)
            .map_err(|_| Error::InvalidInput("pagination limit is too large".to_string()))?,
        i64::try_from(offset)
            .map_err(|_| Error::InvalidInput("pagination offset is too large".to_string()))?,
    ))
}

fn validate_json_path(path: &str) -> Result<()> {
    let valid = path
        .strip_prefix("$.")
        .is_some_and(|tail| !tail.is_empty() && tail.split('.').all(valid_json_path_segment));
    if !valid {
        return Err(Error::InvalidInput(format!(
            "invalid JSON attribute path: {path}"
        )));
    }
    Ok(())
}

fn create_node_attribute_indexes(connection: &Connection, paths: &[String]) -> Result<()> {
    let mut unique = HashSet::new();
    for path in paths {
        validate_json_path(path)?;
        if !unique.insert(path) {
            return Err(Error::InvalidInput(
                "node attribute index paths must be unique".to_string(),
            ));
        }
        let identifier = stable_path_hash(path);
        connection.execute_batch(&format!(
            "CREATE INDEX IF NOT EXISTS tsg_node_attr_{identifier:016x} \
             ON nodes(scope_id, json_extract(attributes, '{path}'));"
        ))?;
    }
    Ok(())
}

fn stable_path_hash(path: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    path.as_bytes().iter().fold(FNV_OFFSET, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

fn valid_json_path_segment(segment: &str) -> bool {
    segment
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || character == '_')
}

fn node_attribute_query(path: &str, scoped: bool) -> Result<String> {
    validate_json_path(path)?;
    // SQLite matches expression indexes syntactically: a bound JSON path cannot
    // use the registered literal-path index. Only validated path segments enter
    // SQL; values remain parameters. Separate scope predicates allow index seeks.
    let scope = if scoped {
        "scope_id = ?1"
    } else {
        "?1 IS NULL"
    };
    Ok(format!(
        "SELECT key FROM nodes WHERE {scope}
         AND json_extract(attributes, '{path}') = json_extract(?2, '$')
         ORDER BY id LIMIT ?3 OFFSET ?4"
    ))
}

#[cfg(test)]
mod attribute_query_tests {
    use super::*;

    #[test]
    fn scoped_attribute_lookup_uses_registered_expression_index() {
        let directory = tempfile::tempdir().unwrap();
        let store = Store::builder(directory.path().join("graph.db"), 8)
            .node_attribute_indexes(["$.qualified_name"])
            .build()
            .unwrap();
        let sql = format!(
            "EXPLAIN QUERY PLAN {}",
            node_attribute_query("$.qualified_name", true).unwrap()
        );
        let mut statement = store.connection.prepare(&sql).unwrap();
        let details = statement
            .query_map((1, "\"missing\"", 1, 0), |row| row.get::<_, String>(3))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            details.iter().any(
                |detail| detail.contains("SEARCH nodes USING INDEX tsg_node_attr_")
                    && detail.contains("scope_id=?")
                    && detail.contains("<expr>=?")
            ),
            "{details:?}"
        );
    }
}

#[cfg(test)]
mod incremental_accelerator_tests {
    use super::*;
    use crate::types::Embedding;

    #[test]
    fn committed_batches_reuse_accelerator_and_stale_state_rebuilds() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = Store::open(directory.path().join("graph.db"), 64, 0).unwrap();
        let identity = |store: &Store| {
            store
                .accelerator
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .instance_id
        };
        let initial = identity(&store);
        for batch_index in 0..4 {
            let batch = WriteBatch {
                nodes: (0..32)
                    .map(|offset| Node {
                        id: format!("node-{}", batch_index * 32 + offset),
                        scope_id: None,
                        kind: "record".into(),
                        name: String::new(),
                        content: String::new(),
                        attributes: serde_json::json!({}),
                    })
                    .collect(),
                embeddings: (0..32)
                    .map(|offset| Embedding {
                        node_id: format!("node-{}", batch_index * 32 + offset),
                        vector: (0..64).map(|axis| f32::from(axis == offset)).collect(),
                    })
                    .collect(),
                ..WriteBatch::default()
            };
            assert!(store.apply_batch(&batch).unwrap().accelerator_ready);
            assert_eq!(
                identity(&store),
                initial,
                "embedding batch reconstructed index"
            );
            assert!(store.apply_batch(&batch).unwrap().accelerator_ready);
            assert_eq!(
                identity(&store),
                initial,
                "replacement batch reconstructed index"
            );
        }
        let metadata = WriteBatch {
            catalog_records: vec![CatalogRecord {
                namespace: "test".into(),
                key: "progress".into(),
                value: serde_json::json!(4),
            }],
            ..WriteBatch::default()
        };
        assert!(store.apply_batch(&metadata).unwrap().accelerator_ready);
        assert_eq!(identity(&store), initial, "metadata reconstructed index");

        // An older in-memory generation must never be promoted by applying only
        // the newest batch: simulate a missed generation and require full repair.
        store
            .connection
            .execute("UPDATE store_metadata SET generation = generation + 1", [])
            .unwrap();
        assert!(store.apply_batch(&metadata).unwrap().accelerator_ready);
        let repaired = identity(&store);
        assert_ne!(repaired, initial);
        *store.accelerator.lock().unwrap() = None;
        assert!(store.apply_batch(&metadata).unwrap().accelerator_ready);
        assert_ne!(identity(&store), repaired);
    }

    #[test]
    fn node_deletes_remove_vectors_without_rebuilding_current_index() {
        let directory = tempfile::tempdir().unwrap();
        let database = directory.path().join("graph.db");
        let mut store = Store::open(&database, 64, 0).unwrap();
        let batch = WriteBatch {
            nodes: (0..2)
                .map(|offset| Node {
                    id: format!("node-{offset}"),
                    scope_id: None,
                    kind: "record".into(),
                    name: String::new(),
                    content: String::new(),
                    attributes: serde_json::json!({}),
                })
                .collect(),
            embeddings: (0..2)
                .map(|offset| Embedding {
                    node_id: format!("node-{offset}"),
                    vector: (0..64).map(|axis| f32::from(axis == offset)).collect(),
                })
                .collect(),
            ..WriteBatch::default()
        };
        assert!(store.apply_batch(&batch).unwrap().accelerator_ready);

        let identity = |store: &Store| {
            store
                .accelerator
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .instance_id
        };
        let initial = identity(&store);
        let receipt = store.delete_nodes(&["node-0".to_string()]).unwrap();

        assert_eq!(receipt.nodes_deleted, 1);
        assert!(receipt.accelerator_ready);
        assert_eq!(identity(&store), initial, "node delete reconstructed index");

        let query: Vec<_> = (0..64).map(|axis| f32::from(axis == 0)).collect();
        let hit = &store
            .search(&query, 1, SearchFilter::default(), SearchBackend::Usearch)
            .unwrap()
            .hits[0];
        assert_eq!(hit.node.id, "node-1");
        drop(store);

        let reopened = Store::builder(&database, 64)
            .read_only(true)
            .build()
            .unwrap();
        let hit = &reopened
            .search(&query, 1, SearchFilter::default(), SearchBackend::Usearch)
            .unwrap()
            .hits[0];
        assert_eq!(hit.node.id, "node-1");
    }
}
