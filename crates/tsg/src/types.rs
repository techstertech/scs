/// Durable graph node supplied to and returned by TSG.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Node {
    /// Stable caller-owned identifier.
    pub id: String,
    /// Optional application scope identifier.
    pub scope_id: Option<i64>,
    /// Caller-defined node category.
    pub kind: String,
    /// Human-readable node name.
    pub name: String,
    /// Searchable or inspectable source content.
    pub content: String,
    /// Application-defined structured attributes.
    pub attributes: serde_json::Value,
}

/// Directed typed relationship between two nodes in the same application scope.
#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    /// Stable caller-owned identifier.
    pub id: String,
    /// Source node public identifier.
    pub source_id: String,
    /// Target node public identifier.
    pub target_id: String,
    /// Caller-defined relationship category.
    pub relationship: String,
    /// Finite caller-defined strength or confidence.
    pub weight: f64,
    /// Application-defined structured attributes.
    pub attributes: serde_json::Value,
}

/// Stable application scope used to partition graph entities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Scope {
    /// Durable TSG-assigned integer identifier.
    pub id: i64,
    /// Stable caller-owned scope key.
    pub key: String,
}

/// Canonical vector associated one-to-one with a node.
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding {
    /// Public identifier of the owning node.
    pub node_id: String,
    /// Finite fixed-dimensional `f32` coordinates.
    pub vector: Vec<f32>,
}

/// Application-defined JSON document stored outside the graph model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CatalogRecord {
    /// Caller-defined namespace isolating one record family.
    pub namespace: String,
    /// Stable key unique within the namespace.
    pub key: String,
    /// Structured value serialized canonically by TSG.
    pub value: serde_json::Value,
}

/// Identity of an application catalog record to delete.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CatalogKey {
    /// Caller-defined namespace isolating one record family.
    pub namespace: String,
    /// Stable key unique within the namespace.
    pub key: String,
}

/// Atomic collection of graph and embedding upserts.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WriteBatch {
    /// Nodes inserted or replaced by public identifier.
    pub nodes: Vec<Node>,
    /// Edges inserted after their endpoints resolve.
    pub edges: Vec<Edge>,
    /// Embeddings inserted or replaced for their nodes.
    pub embeddings: Vec<Embedding>,
    /// Application records inserted or replaced by namespace and key.
    pub catalog_records: Vec<CatalogRecord>,
    /// Application records deleted by namespace and key.
    pub catalog_deletes: Vec<CatalogKey>,
}

/// Direction used by bounded graph traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction {
    /// Follow source-to-target edges.
    Outgoing,
    /// Follow target-to-source edges.
    Incoming,
    /// Follow edges in either direction.
    Both,
}

/// Vector retrieval implementation requested or reported by search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchBackend {
    /// Exhaustively score canonical `SQLite` vectors.
    Exact,
    /// Search the generation-matched `USearch` HNSW accelerator.
    Usearch,
    /// Select exact or `USearch` based on candidate count and availability.
    Adaptive,
}

/// Optional scope and node-kind constraints for vector search.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SearchFilter<'a> {
    /// Restrict results to one application scope.
    pub scope_id: Option<i64>,
    /// Restrict results to one node kind.
    pub kind: Option<&'a str>,
}

/// Optional scope and kind constraints for stable node reads.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NodeFilter<'a> {
    /// Restrict results to one application scope.
    pub scope_id: Option<i64>,
    /// Restrict results to one caller-defined node kind.
    pub kind: Option<&'a str>,
}

/// Equality predicate over one validated node-attribute JSON path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttributeFilter<'a> {
    /// JSON path such as `$.qualified_name`.
    pub path: &'a str,
    /// JSON value compared for equality.
    pub value: &'a serde_json::Value,
}

/// One vector search match.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    /// Hydrated durable graph node.
    pub node: Node,
    /// Cosine distance in ascending-result order.
    pub distance: f32,
}

/// Search response with evidence of the backend actually used.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResults {
    /// Resolved backend after adaptive planning.
    pub backend: SearchBackend,
    /// Ordered top-k matches.
    pub hits: Vec<SearchHit>,
}

/// Outcome of an atomic upsert batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommitReceipt {
    /// Newly committed durable generation.
    pub generation: u64,
    /// Whether the derived accelerator reached the same generation.
    pub accelerator_ready: bool,
}

/// Outcome of an atomic node-deletion batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeleteReceipt {
    /// Current generation, advanced when at least one node was deleted.
    pub generation: u64,
    /// Number of nodes that existed and were deleted.
    pub nodes_deleted: usize,
    /// Whether the derived accelerator reached the same generation.
    pub accelerator_ready: bool,
}

/// `SQLite` and sidecar power-loss durability policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Durability {
    /// Synchronize acknowledged `SQLite` and sidecar writes to durable storage.
    #[default]
    Full,
    /// Use `SQLite` `NORMAL` synchronization and skip explicit sidecar syncs.
    Normal,
}

/// Operational counts and state for one open store.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoreStats {
    /// Current durable generation.
    pub generation: u64,
    /// Number of nodes.
    pub node_count: usize,
    /// Number of directed edges.
    pub edge_count: usize,
    /// Number of canonical embeddings.
    pub embedding_count: usize,
    /// Whether the `USearch` sidecar matches the durable generation.
    pub accelerator_ready: bool,
    /// Whether this handle rejects mutation.
    pub read_only: bool,
}

/// Read-only consistency assessment of `SQLite` and its vector accelerator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntegrityReport {
    /// Durable generation assessed by this report.
    pub generation: u64,
    /// Whether `SQLite`'s own integrity check succeeded.
    pub sqlite_ok: bool,
    /// Whether the `USearch` sidecar matches the durable generation.
    pub accelerator_ready: bool,
    /// Human-readable invariant violations.
    pub issues: Vec<String>,
}

impl IntegrityReport {
    /// Returns whether every checked durable and derived invariant passed.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.sqlite_ok && self.accelerator_ready && self.issues.is_empty()
    }
}
