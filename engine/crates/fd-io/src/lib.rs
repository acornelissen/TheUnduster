//! Image I/O: native-depth pixel buffers, decode/encode, metadata passthrough.

mod buf;
mod png_jpeg;

pub use buf::{ImageBuf, PixelData};

use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("cannot read image {path}: {reason}")]
    Decode { path: String, reason: String },
    #[error("cannot write image {path}: {reason}")]
    Encode { path: String, reason: String },
    #[error("unsupported format: {0}")]
    Unsupported(String),
}

fn ext_of(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

pub fn decode(path: &Path) -> Result<ImageBuf, IoError> {
    match ext_of(path).as_str() {
        "png" | "jpg" | "jpeg" => png_jpeg::decode(path),
        other => Err(IoError::Unsupported(other.to_string())),
    }
}

pub fn encode(path: &Path, img: &ImageBuf) -> Result<(), IoError> {
    match ext_of(path).as_str() {
        "png" | "jpg" | "jpeg" => png_jpeg::encode(path, img),
        other => Err(IoError::Unsupported(other.to_string())),
    }
}

pub(crate) fn decode_err(path: &Path, reason: impl ToString) -> IoError {
    IoError::Decode {
        path: path.display().to_string(),
        reason: reason.to_string(),
    }
}

pub(crate) fn encode_err(path: &Path, reason: impl ToString) -> IoError {
    IoError::Encode {
        path: path.display().to_string(),
        reason: reason.to_string(),
    }
}
