use thiserror::Error;

/// Result type returned by TSG operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Typed failures produced by configuration, lifecycle, and storage operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Caller input violates a documented contract.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Durable or derived storage could not complete an operation.
    #[error("storage error: {0}")]
    Storage(String),

    /// Another process currently owns the writable store lease.
    #[error("store is locked by another writer: {0}")]
    WriterLocked(String),

    /// A mutating operation was requested through a read-only handle.
    #[error("operation requires a writable store")]
    ReadOnly,

    /// The database was created by a newer, unsupported schema version.
    #[error("unsupported schema version {found}; maximum supported version is {supported}")]
    UnsupportedSchema {
        /// Version found in the database.
        found: u32,
        /// Maximum version understood by this crate.
        supported: u32,
    },

    /// The database uses an intentionally non-migratable pre-1.0 schema.
    #[error("schema version {found} requires rebuilding the store for version {required}")]
    ReindexRequired {
        /// Version found in the database.
        found: u32,
        /// Version required by this crate.
        required: u32,
    },

    /// Accelerated retrieval was explicitly requested but cannot be served.
    #[error("vector accelerator is unavailable: {0}")]
    AcceleratorUnavailable(String),

    /// `SQLite` returned an error.
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),

    /// A filesystem operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
