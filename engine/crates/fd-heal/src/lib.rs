//! Tiered defect healing with a bit-exactness guarantee outside masks.

mod classical;
mod components;
mod dilate;
mod grain;
mod group;
mod heal;
mod inpaint;

pub use classical::classical_fill;
pub use components::{components, Bbox, Defect};
pub use dilate::dilate;
pub use grain::add_grain;
pub use group::{group_defects, Group};
pub use heal::{heal, heal_with_progress, HealReport, TINY_MAX_DIM};
pub use inpaint::Inpainter;

#[derive(Debug, thiserror::Error)]
pub enum HealError {
    #[error("inpaint model error: {0}")]
    Model(String),
    #[error("mask size {got} does not match image size {want}")]
    MaskSize { got: usize, want: usize },
    /// The progress callback returned `Break`: the caller asked for a
    /// cooperative abort. The image was not modified (write-back never ran).
    #[error("heal cancelled")]
    Cancelled,
}
