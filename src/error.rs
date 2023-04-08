/// Main library error type.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Incorrect number of elements.
    #[error("wrong element count {element_count} for dims {dims:?}")]
    WrongElementCount { dims: Vec<usize>, element_count: usize },

    /// Error from the xla C++ library.
    #[error("xla error {0}")]
    XlaError(String),

    #[error("unexpected element type {0}")]
    UnexpectedElementType(i32),

    #[error("element type mismatch, on-device: {on_device:?}, on-host: {on_host:?}")]
    ElementTypeMismatch { on_device: crate::PrimitiveType, on_host: crate::PrimitiveType },

    #[error(
        "target buffer is too large, offset {offset}, shape {shape:?}, buffer_len: {buffer_len}"
    )]
    TargetBufferIsTooLarge { offset: usize, shape: crate::Shape, buffer_len: usize },
}

pub type Result<T> = std::result::Result<T, Error>;