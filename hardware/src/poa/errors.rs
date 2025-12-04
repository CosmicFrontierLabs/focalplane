use thiserror::Error;

#[derive(Error, Debug)]
pub enum OrinDevError {
    #[error("Camera error: {0}")]
    Camera(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image processing error: {0}")]
    Image(String),

    #[error("Configuration error: {0}")]
    Config(String),
}
