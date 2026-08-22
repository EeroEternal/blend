use thiserror::Error;

pub type Result<T> = std::result::Result<T, FtError>;

#[derive(Debug, Error)]
pub enum FtError {
    #[error("out of memory: {0}")]
    Oom(String),
    #[error("invalid request: {0}")]
    Invalid(String),
    #[error("stale handle (gen {handle}, current {current})")]
    StaleHandle { handle: u32, current: u32 },
    #[error("kernel backend: {0}")]
    Kernel(String),
    #[error("model not supported: {0}")]
    UnsupportedModel(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}
