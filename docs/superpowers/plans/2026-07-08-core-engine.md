# Core Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Rust engine workspace: image I/O with metadata passthrough, tile pyramid + cache, ONNX tiled detection matching the Python reference bit-for-near-bit, and tiered healing with a bit-exactness guarantee outside masks.

**Architecture:** A cargo workspace under `engine/` with four crates. `fd-io` owns the `ImageBuf` type (native-depth pixels, u8 or u16 — never lossy-converted) and decode/encode with ICC/EXIF passthrough. `fd-tiles` builds display pyramids and caches tiles under a byte budget. `fd-infer` wraps ONNX Runtime and ports the exact tiling arithmetic from `training/src/unduster_training/detectors.py`, proven by committed cross-language parity fixtures. `fd-heal` labels defects, heals them in tiers (classical fill for tiny, ONNX inpaint + grain re-synthesis for larger), and composes results so pixels outside masks stay integer-identical to the source.

**Tech Stack:** Rust (stable, via mise), cargo workspace; crates: image, tiff, img-parts, thiserror, lru, ndarray, ort (ONNX Runtime), serde_json (fixtures). Python (existing training env) generates test fixtures.

## Global Constraints

- Everything lives under `engine/` in `/Users/albert/Development/TheUnduster`, except the fixture generator script (`training/scripts/make_engine_fixtures.py`) and committed fixtures (`engine/fixtures/`).
- Trunk-based development: commit directly to `main`, one atomic commit per task, tests green before every commit. No feature branches.
- Pixel data is native-depth end to end: `u8` or `u16` per channel, 1 (grey) or 3 (RGB) channels. f32 views are derived, normalized to [0, 1] (255 or 65535 -> 1.0), and never written back except inside heal masks.
- Detection tiling: TILE = 512, OVERLAP = 64, stride 448, edge-replicate padding, probability averaging in overlaps — must match `training/src/unduster_training/detectors.py` exactly; the parity fixture test is the proof.
- ONNX model contract (from the training pipeline): detector input `"image"` NCHW f32, output `"logits"`, N/H/W dynamic, C static.
- Originals are opened read-only; no code path writes to a source file's path.
- Tests run with the CPU execution provider; CoreML is selected at runtime via `Ep::CoreML` and never required by tests or CI.
- Rust toolchain pinned via `engine/mise.toml`; format with rustfmt, lint with clippy, both clean (`-D warnings`).
- No emoji anywhere. Commit messages plain English, no Co-Authored-By lines.
- If the `ort` crate's rc API differs from the code shown at compile time, keep the semantics identical (same EP selection, same input/output names) and consult docs.rs/ort — do not change tiling arithmetic to fit the API.

---

### Task 1: Workspace scaffold and fd-io ImageBuf with PNG round-trip

**Files:**
- Create: `engine/Cargo.toml`, `engine/mise.toml`, `engine/rustfmt.toml`, `engine/.gitignore`
- Create: `engine/crates/fd-io/Cargo.toml`
- Create: `engine/crates/fd-io/src/lib.rs`
- Create: `engine/crates/fd-io/src/buf.rs`
- Create: `engine/crates/fd-io/src/png_jpeg.rs`
- Test: `engine/crates/fd-io/tests/png_roundtrip.rs`

**Interfaces:**
- Produces: `PixelData::{U8(Vec<u8>), U16(Vec<u16>)}`; `ImageBuf { width: u32, height: u32, channels: u8, data: PixelData, icc: Option<Vec<u8>>, exif: Option<Vec<u8>> }` with `to_f32(&self) -> Vec<f32>` (interleaved, [0,1]) and `pixel_count(&self) -> usize`; `IoError`; `decode(path: &Path) -> Result<ImageBuf, IoError>`; `encode(path: &Path, img: &ImageBuf) -> Result<(), IoError>`. Every later crate consumes `ImageBuf`.

- [ ] **Step 1: Scaffold the workspace**

`engine/mise.toml`:

```toml
[tools]
rust = "stable"
```

`engine/Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = ["crates/fd-io", "crates/fd-tiles", "crates/fd-infer", "crates/fd-heal"]

[workspace.package]
edition = "2021"
version = "0.1.0"

[workspace.dependencies]
thiserror = "2"
```

Create empty stub crates for the other three members so the workspace builds from day one: for each of `fd-tiles`, `fd-infer`, `fd-heal` create `engine/crates/<name>/Cargo.toml`:

```toml
[package]
name = "<name>"
edition.workspace = true
version.workspace = true

[dependencies]
```

and `engine/crates/<name>/src/lib.rs` containing only a doc comment line: `//! Placeholder — implemented in a later task.`

`engine/rustfmt.toml`: empty file (defaults). `engine/.gitignore`:

```
target/
```

`engine/crates/fd-io/Cargo.toml`:

```toml
[package]
name = "fd-io"
edition.workspace = true
version.workspace = true

[dependencies]
thiserror.workspace = true
image = { version = "0.25", default-features = false, features = ["png", "jpeg"] }
tiff = "0.9"
img-parts = "0.3"
```

Run: `cd /Users/albert/Development/TheUnduster/engine && mise install && cargo build`
Expected: workspace compiles (fd-io still empty lib).

- [ ] **Step 2: Write the failing test**

`engine/crates/fd-io/tests/png_roundtrip.rs`:

```rust
use fd_io::{decode, encode, ImageBuf, PixelData};

fn gradient(width: u32, height: u32, channels: u8, sixteen: bool) -> ImageBuf {
    let n = (width * height) as usize * channels as usize;
    let data = if sixteen {
        PixelData::U16((0..n).map(|i| ((i * 65535) / n) as u16).collect())
    } else {
        PixelData::U8((0..n).map(|i| ((i * 255) / n) as u8).collect())
    };
    ImageBuf { width, height, channels, data, icc: None, exif: None }
}

#[test]
fn png_16bit_rgb_roundtrip_is_lossless() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.png");
    let img = gradient(64, 48, 3, true);
    encode(&path, &img).unwrap();
    let back = decode(&path).unwrap();
    assert_eq!((back.width, back.height, back.channels), (64, 48, 3));
    match (&img.data, &back.data) {
        (PixelData::U16(a), PixelData::U16(b)) => assert_eq!(a, b),
        _ => panic!("expected 16-bit data back"),
    }
}

#[test]
fn png_8bit_gray_roundtrip_is_lossless() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.png");
    let img = gradient(32, 32, 1, false);
    encode(&path, &img).unwrap();
    let back = decode(&path).unwrap();
    match (&img.data, &back.data) {
        (PixelData::U8(a), PixelData::U8(b)) => assert_eq!(a, b),
        _ => panic!("expected 8-bit data back"),
    }
}

#[test]
fn to_f32_normalizes_both_depths() {
    let img8 = gradient(4, 4, 1, false);
    let img16 = gradient(4, 4, 1, true);
    let f8 = img8.to_f32();
    let f16 = img16.to_f32();
    assert!(f8.iter().all(|&v| (0.0..=1.0).contains(&v)));
    assert!(f16.iter().all(|&v| (0.0..=1.0).contains(&v)));
    assert_eq!(f8.len(), 16);
}

#[test]
fn decode_missing_file_names_the_path() {
    let err = decode(std::path::Path::new("/nonexistent/y.png")).unwrap_err();
    assert!(err.to_string().contains("y.png"));
}
```

Add to `engine/crates/fd-io/Cargo.toml`:

```toml
[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p fd-io`
Expected: compile error — `decode`, `encode`, `ImageBuf` not found.

- [ ] **Step 4: Implement**

`engine/crates/fd-io/src/buf.rs`:

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum PixelData {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

/// Pixels in native depth. f32 views are derived; native data is the
/// source of truth for the bit-exactness guarantee.
#[derive(Debug, Clone)]
pub struct ImageBuf {
    pub width: u32,
    pub height: u32,
    pub channels: u8, // 1 = grey, 3 = RGB
    pub data: PixelData,
    pub icc: Option<Vec<u8>>,
    pub exif: Option<Vec<u8>>,
}

impl ImageBuf {
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize * self.channels as usize
    }

    /// Interleaved f32 in [0, 1]. 255 or 65535 maps to 1.0.
    pub fn to_f32(&self) -> Vec<f32> {
        match &self.data {
            PixelData::U8(v) => v.iter().map(|&p| p as f32 / 255.0).collect(),
            PixelData::U16(v) => v.iter().map(|&p| p as f32 / 65535.0).collect(),
        }
    }
}
```

`engine/crates/fd-io/src/lib.rs`:

```rust
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
    IoError::Decode { path: path.display().to_string(), reason: reason.to_string() }
}

pub(crate) fn encode_err(path: &Path, reason: impl ToString) -> IoError {
    IoError::Encode { path: path.display().to_string(), reason: reason.to_string() }
}
```

`engine/crates/fd-io/src/png_jpeg.rs`:

```rust
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use image::{ColorType, DynamicImage, ImageDecoder, ImageEncoder};

use crate::{decode_err, encode_err, ImageBuf, IoError, PixelData};

pub fn decode(path: &Path) -> Result<ImageBuf, IoError> {
    let file = File::open(path).map_err(|e| decode_err(path, e))?;
    let reader = image::ImageReader::new(BufReader::new(file))
        .with_guessed_format()
        .map_err(|e| decode_err(path, e))?;
    let mut decoder = reader.into_decoder().map_err(|e| decode_err(path, e))?;
    let icc = decoder.icc_profile().ok().flatten();
    let exif = decoder.exif_metadata().ok().flatten();
    let dyn_img = DynamicImage::from_decoder(decoder).map_err(|e| decode_err(path, e))?;
    let (width, height) = (dyn_img.width(), dyn_img.height());
    let (channels, data) = match dyn_img {
        DynamicImage::ImageLuma8(b) => (1, PixelData::U8(b.into_raw())),
        DynamicImage::ImageRgb8(b) => (3, PixelData::U8(b.into_raw())),
        DynamicImage::ImageLuma16(b) => (1, PixelData::U16(b.into_raw())),
        DynamicImage::ImageRgb16(b) => (3, PixelData::U16(b.into_raw())),
        // Alpha and exotic layouts: normalize to RGB at matching depth.
        other => match other.color().bytes_per_channel() {
            2 => (3, PixelData::U16(other.into_rgb16().into_raw())),
            _ => (3, PixelData::U8(other.into_rgb8().into_raw())),
        },
    };
    Ok(ImageBuf { width, height, channels, data, icc, exif })
}

pub fn encode(path: &Path, img: &ImageBuf) -> Result<(), IoError> {
    let file = File::create(path).map_err(|e| encode_err(path, e))?;
    let w = std::io::BufWriter::new(file);
    let ext = crate::ext_of(path);
    let color = match (img.channels, &img.data) {
        (1, PixelData::U8(_)) => ColorType::L8,
        (3, PixelData::U8(_)) => ColorType::Rgb8,
        (1, PixelData::U16(_)) => ColorType::L16,
        (3, PixelData::U16(_)) => ColorType::Rgb16,
        _ => return Err(crate::IoError::Unsupported(format!("{} channels", img.channels))),
    };
    match (ext.as_str(), &img.data) {
        ("png", PixelData::U8(v)) => {
            image::codecs::png::PngEncoder::new(w)
                .write_image(v, img.width, img.height, color.into())
                .map_err(|e| encode_err(path, e))
        }
        ("png", PixelData::U16(v)) => {
            let bytes: Vec<u8> = v.iter().flat_map(|p| p.to_be_bytes()).collect();
            image::codecs::png::PngEncoder::new(w)
                .write_image(&bytes, img.width, img.height, color.into())
                .map_err(|e| encode_err(path, e))
        }
        ("jpg" | "jpeg", PixelData::U8(v)) => {
            image::codecs::jpeg::JpegEncoder::new_with_quality(w, 95)
                .write_image(v, img.width, img.height, color.into())
                .map_err(|e| encode_err(path, e))
        }
        ("jpg" | "jpeg", PixelData::U16(_)) => {
            Err(IoError::Unsupported("16-bit JPEG".to_string()))
        }
        _ => Err(IoError::Unsupported(ext)),
    }
}
```

Make `ext_of` `pub(crate)` in lib.rs (it is referenced from png_jpeg.rs).

Note: `write_image` takes `ExtendedColorType` — the `color.into()` handles it. PNG 16-bit expects big-endian bytes, hence `to_be_bytes`. If the image crate version in the registry has renamed `ImageReader` (older: `io::Reader`), use the current name — semantics identical.

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p fd-io`
Expected: 4 tests pass.

- [ ] **Step 6: Format, lint, commit**

Run: `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test -p fd-io`

```bash
git add engine
git commit -m "Scaffold engine workspace and add fd-io PNG/JPEG round-trip"
```

---

### Task 2: fd-io TIFF support (8/16-bit, grey/RGB)

**Files:**
- Create: `engine/crates/fd-io/src/tiff_io.rs`
- Modify: `engine/crates/fd-io/src/lib.rs` (route "tif"/"tiff")
- Test: `engine/crates/fd-io/tests/tiff_roundtrip.rs`

**Interfaces:**
- Consumes: `ImageBuf`, `PixelData`, error helpers from Task 1.
- Produces: TIFF handling inside the same `decode`/`encode` entry points. 16-bit TIFF is the primary scan format for the whole product.

- [ ] **Step 1: Write the failing test**

`engine/crates/fd-io/tests/tiff_roundtrip.rs`:

```rust
use fd_io::{decode, encode, ImageBuf, PixelData};

fn make(width: u32, height: u32, channels: u8, sixteen: bool) -> ImageBuf {
    let n = (width * height) as usize * channels as usize;
    let data = if sixteen {
        PixelData::U16((0..n).map(|i| ((i * 65535) / n) as u16).collect())
    } else {
        PixelData::U8((0..n).map(|i| ((i * 255) / n) as u8).collect())
    };
    ImageBuf { width, height, channels, data, icc: None, exif: None }
}

#[test]
fn tiff_16bit_rgb_roundtrip_is_lossless() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scan.tif");
    let img = make(70, 50, 3, true);
    encode(&path, &img).unwrap();
    let back = decode(&path).unwrap();
    assert_eq!((back.width, back.height, back.channels), (70, 50, 3));
    match (&img.data, &back.data) {
        (PixelData::U16(a), PixelData::U16(b)) => assert_eq!(a, b),
        _ => panic!("expected 16-bit"),
    }
}

#[test]
fn tiff_8bit_gray_roundtrip_is_lossless() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.tiff");
    let img = make(33, 21, 1, false);
    encode(&path, &img).unwrap();
    let back = decode(&path).unwrap();
    match (&img.data, &back.data) {
        (PixelData::U8(a), PixelData::U8(b)) => assert_eq!(a, b),
        _ => panic!("expected 8-bit"),
    }
}

#[test]
fn tiff_rgba_input_drops_alpha() {
    // hand-write a tiny RGBA8 TIFF via the tiff crate, expect RGB back
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tif");
    {
        let file = std::fs::File::create(&path).unwrap();
        let mut t = tiff::encoder::TiffEncoder::new(file).unwrap();
        let rgba: Vec<u8> = (0..4 * 4 * 4).map(|i| i as u8).collect();
        t.write_image::<tiff::encoder::colortype::RGBA8>(4, 4, &rgba).unwrap();
    }
    let back = decode(&path).unwrap();
    assert_eq!(back.channels, 3);
    match &back.data {
        PixelData::U8(v) => assert_eq!(v.len(), 4 * 4 * 3),
        _ => panic!("expected 8-bit"),
    }
}
```

Add `tiff = "0.9"` to `[dev-dependencies]` too (used directly by the test).

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fd-io --test tiff_roundtrip`
Expected: FAIL — `Unsupported("tif")`.

- [ ] **Step 3: Implement**

`engine/crates/fd-io/src/tiff_io.rs`:

```rust
use std::fs::File;
use std::path::Path;

use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::{colortype, TiffEncoder};
use tiff::ColorType;

use crate::{decode_err, encode_err, ImageBuf, IoError, PixelData};

fn drop_alpha<T: Copy>(data: &[T], n_px: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(n_px * 3);
    for px in data.chunks_exact(4) {
        out.extend_from_slice(&px[..3]);
    }
    out
}

pub fn decode(path: &Path) -> Result<ImageBuf, IoError> {
    let file = File::open(path).map_err(|e| decode_err(path, e))?;
    let mut d = Decoder::new(file).map_err(|e| decode_err(path, e))?;
    let (width, height) = d.dimensions().map_err(|e| decode_err(path, e))?;
    let color = d.colortype().map_err(|e| decode_err(path, e))?;
    let icc = d
        .get_tag_u8_vec(tiff::tags::Tag::Unknown(34675))
        .ok();
    let result = d.read_image().map_err(|e| decode_err(path, e))?;
    let n_px = (width * height) as usize;
    let (channels, data) = match (color, result) {
        (ColorType::Gray(8), DecodingResult::U8(v)) => (1, PixelData::U8(v)),
        (ColorType::Gray(16), DecodingResult::U16(v)) => (1, PixelData::U16(v)),
        (ColorType::RGB(8), DecodingResult::U8(v)) => (3, PixelData::U8(v)),
        (ColorType::RGB(16), DecodingResult::U16(v)) => (3, PixelData::U16(v)),
        (ColorType::RGBA(8), DecodingResult::U8(v)) => (3, PixelData::U8(drop_alpha(&v, n_px))),
        (ColorType::RGBA(16), DecodingResult::U16(v)) => (3, PixelData::U16(drop_alpha(&v, n_px))),
        (c, _) => {
            return Err(decode_err(path, format!("unsupported TIFF layout: {c:?}")));
        }
    };
    Ok(ImageBuf { width, height, channels, data, icc, exif: None })
}

pub fn encode(path: &Path, img: &ImageBuf) -> Result<(), IoError> {
    let file = File::create(path).map_err(|e| encode_err(path, e))?;
    let mut t = TiffEncoder::new(file).map_err(|e| encode_err(path, e))?;
    let (w, h) = (img.width, img.height);
    match (img.channels, &img.data) {
        (1, PixelData::U8(v)) => t
            .write_image::<colortype::Gray8>(w, h, v)
            .map_err(|e| encode_err(path, e)),
        (1, PixelData::U16(v)) => t
            .write_image::<colortype::Gray16>(w, h, v)
            .map_err(|e| encode_err(path, e)),
        (3, PixelData::U8(v)) => t
            .write_image::<colortype::RGB8>(w, h, v)
            .map_err(|e| encode_err(path, e)),
        (3, PixelData::U16(v)) => t
            .write_image::<colortype::RGB16>(w, h, v)
            .map_err(|e| encode_err(path, e)),
        _ => Err(IoError::Unsupported(format!("{} channels", img.channels))),
    }
}
```

In `lib.rs`, add `mod tiff_io;` and route `"tif" | "tiff"` in both `decode` and `encode` to `tiff_io::{decode, encode}`.

Note: ICC writing for TIFF comes in Task 3; this task only reads tag 34675 if present. If the tiff crate's tag API differs (`get_tag_u8_vec` name), adapt the call, keep the tag number 34675.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p fd-io`
Expected: all fd-io tests pass.

- [ ] **Step 5: Format, lint, commit**

Run: `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test -p fd-io`

```bash
git add engine
git commit -m "Add 8/16-bit TIFF decode and encode to fd-io"
```

---

### Task 3: fd-io metadata passthrough (ICC and EXIF)

**Files:**
- Create: `engine/crates/fd-io/src/metadata.rs`
- Modify: `engine/crates/fd-io/src/png_jpeg.rs` (attach on encode)
- Modify: `engine/crates/fd-io/src/tiff_io.rs` (write ICC tag)
- Test: `engine/crates/fd-io/tests/metadata.rs`

**Interfaces:**
- Consumes: Task 1/2 modules.
- Produces: ICC round-trips for PNG, JPEG, TIFF; EXIF round-trips for JPEG. PNG EXIF and TIFF EXIF are documented non-goals for v1 of the engine (rare in scan output; the app preserves originals anyway). This behavior table is what the app layer will rely on.

- [ ] **Step 1: Write the failing test**

`engine/crates/fd-io/tests/metadata.rs`:

```rust
use fd_io::{decode, encode, ImageBuf, PixelData};

fn tiny_with_meta(icc: Option<Vec<u8>>, exif: Option<Vec<u8>>) -> ImageBuf {
    ImageBuf {
        width: 8,
        height: 8,
        channels: 3,
        data: PixelData::U8(vec![128; 8 * 8 * 3]),
        icc,
        exif,
    }
}

// A plausible-looking little ICC blob; content is opaque to us.
fn fake_icc() -> Vec<u8> {
    let mut b = vec![0u8; 128];
    b[36..40].copy_from_slice(b"acsp"); // ICC signature at offset 36
    b
}

#[test]
fn jpeg_icc_and_exif_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.jpg");
    let exif = b"Exif\0\0MM\0*\0\0\0\x08".to_vec();
    let img = tiny_with_meta(Some(fake_icc()), Some(exif.clone()));
    encode(&path, &img).unwrap();
    let back = decode(&path).unwrap();
    assert_eq!(back.icc.as_deref(), Some(fake_icc().as_slice()));
    assert!(back.exif.is_some());
}

#[test]
fn png_icc_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.png");
    let img = tiny_with_meta(Some(fake_icc()), None);
    encode(&path, &img).unwrap();
    let back = decode(&path).unwrap();
    assert_eq!(back.icc.as_deref(), Some(fake_icc().as_slice()));
}

#[test]
fn tiff_icc_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.tif");
    let img = tiny_with_meta(Some(fake_icc()), None);
    encode(&path, &img).unwrap();
    let back = decode(&path).unwrap();
    assert_eq!(back.icc.as_deref(), Some(fake_icc().as_slice()));
}

#[test]
fn no_metadata_stays_none() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("n.png");
    encode(&path, &tiny_with_meta(None, None)).unwrap();
    let back = decode(&path).unwrap();
    assert!(back.icc.is_none());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fd-io --test metadata`
Expected: FAIL — icc comes back `None` after encode (we never attach it).

- [ ] **Step 3: Implement**

`engine/crates/fd-io/src/metadata.rs`:

```rust
//! Post-encode metadata attachment for PNG/JPEG via img-parts.
//! Strategy: encode pixels to bytes first, then splice metadata chunks in.

use img_parts::{Bytes, ImageEXIF, ImageICC};

pub fn attach_jpeg(bytes: Vec<u8>, icc: Option<&[u8]>, exif: Option<&[u8]>) -> Vec<u8> {
    let mut jpeg = match img_parts::jpeg::Jpeg::from_bytes(Bytes::from(bytes.clone())) {
        Ok(j) => j,
        Err(_) => return bytes,
    };
    if let Some(p) = icc {
        jpeg.set_icc_profile(Some(Bytes::copy_from_slice(p)));
    }
    if let Some(e) = exif {
        jpeg.set_exif(Some(Bytes::copy_from_slice(e)));
    }
    let mut out = Vec::new();
    if jpeg.encoder().write_to(&mut out).is_ok() {
        out
    } else {
        bytes
    }
}

pub fn attach_png(bytes: Vec<u8>, icc: Option<&[u8]>) -> Vec<u8> {
    let mut png = match img_parts::png::Png::from_bytes(Bytes::from(bytes.clone())) {
        Ok(p) => p,
        Err(_) => return bytes,
    };
    if let Some(p) = icc {
        png.set_icc_profile(Some(Bytes::copy_from_slice(p)));
    }
    let mut out = Vec::new();
    if png.encoder().write_to(&mut out).is_ok() {
        out
    } else {
        bytes
    }
}
```

Modify `png_jpeg.rs::encode`: instead of writing straight to the file, encode into a `Vec<u8>` cursor, then run it through `metadata::attach_jpeg`/`attach_png` when `img.icc`/`img.exif` are present, then write the final bytes with `std::fs::write`. Concretely, replace the body:

```rust
pub fn encode(path: &Path, img: &ImageBuf) -> Result<(), IoError> {
    let ext = crate::ext_of(path);
    let mut raw: Vec<u8> = Vec::new();
    {
        let w = std::io::Cursor::new(&mut raw);
        let color = match (img.channels, &img.data) {
            (1, PixelData::U8(_)) => ColorType::L8,
            (3, PixelData::U8(_)) => ColorType::Rgb8,
            (1, PixelData::U16(_)) => ColorType::L16,
            (3, PixelData::U16(_)) => ColorType::Rgb16,
            _ => return Err(IoError::Unsupported(format!("{} channels", img.channels))),
        };
        match (ext.as_str(), &img.data) {
            ("png", PixelData::U8(v)) => image::codecs::png::PngEncoder::new(w)
                .write_image(v, img.width, img.height, color.into())
                .map_err(|e| encode_err(path, e))?,
            ("png", PixelData::U16(v)) => {
                let bytes: Vec<u8> = v.iter().flat_map(|p| p.to_be_bytes()).collect();
                image::codecs::png::PngEncoder::new(w)
                    .write_image(&bytes, img.width, img.height, color.into())
                    .map_err(|e| encode_err(path, e))?
            }
            ("jpg" | "jpeg", PixelData::U8(v)) => {
                image::codecs::jpeg::JpegEncoder::new_with_quality(w, 95)
                    .write_image(v, img.width, img.height, color.into())
                    .map_err(|e| encode_err(path, e))?
            }
            ("jpg" | "jpeg", PixelData::U16(_)) => {
                return Err(IoError::Unsupported("16-bit JPEG".to_string()))
            }
            _ => return Err(IoError::Unsupported(ext)),
        }
    }
    let final_bytes = match ext.as_str() {
        "jpg" | "jpeg" if img.icc.is_some() || img.exif.is_some() => {
            crate::metadata::attach_jpeg(raw, img.icc.as_deref(), img.exif.as_deref())
        }
        "png" if img.icc.is_some() => crate::metadata::attach_png(raw, img.icc.as_deref()),
        _ => raw,
    };
    std::fs::write(path, final_bytes).map_err(|e| encode_err(path, e))
}
```

Add `mod metadata;` to lib.rs.

TIFF ICC write in `tiff_io.rs::encode`: the tiff crate writes tags through the image writer. Change each `write_image` arm to use `new_image::<...>` so tags can be added; pattern for one arm (repeat for all four):

```rust
(3, PixelData::U8(v)) => {
    let mut image = t
        .new_image::<colortype::RGB8>(w, h)
        .map_err(|e| encode_err(path, e))?;
    if let Some(icc) = &img.icc {
        image
            .encoder()
            .write_tag(tiff::tags::Tag::Unknown(34675), icc.as_slice())
            .map_err(|e| encode_err(path, e))?;
    }
    image.write_data(v).map_err(|e| encode_err(path, e))
}
```

Notes: img-parts' PNG ICC handling compresses into an iCCP chunk and image's PNG decoder decompresses on read — the test asserts the round-trip, which is what matters. EXIF blob format: image crate returns raw EXIF (without the "Exif\0\0" prefix in some versions) while img-parts expects the raw TIFF-structured payload; if the JPEG EXIF assertion fails on prefix mismatch, normalize by stripping a leading `Exif\0\0` in decode before storing — the test only asserts `is_some()` for EXIF to allow this tolerance.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p fd-io`
Expected: all pass.

- [ ] **Step 5: Format, lint, commit**

Run: `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test -p fd-io`

```bash
git add engine
git commit -m "Pass ICC and EXIF through fd-io encode and decode"
```

---

### Task 4: fd-tiles pyramid builder

**Files:**
- Modify: `engine/crates/fd-tiles/Cargo.toml`
- Create: `engine/crates/fd-tiles/src/lib.rs` (replacing stub)
- Create: `engine/crates/fd-tiles/src/pyramid.rs`
- Test: `engine/crates/fd-tiles/tests/pyramid.rs`

**Interfaces:**
- Consumes: `fd_io::{ImageBuf, PixelData}`.
- Produces: `pub const TILE_SIZE: u32 = 512`; `Level { width: u32, height: u32, rgba: Vec<u8> }`; `Pyramid { levels: Vec<Level> }` with `Pyramid::build(&ImageBuf) -> Pyramid`, `Pyramid::tile(&self, level: u8, tx: u32, ty: u32) -> Option<Tile>` where `Tile { width: u32, height: u32, rgba: Vec<u8> }` (edge tiles smaller than 512); `Pyramid::tiles_at(&self, level: u8) -> (u32, u32)` (tile grid dims). Task 5's cache and the future app viewer consume these.

- [ ] **Step 1: Write the failing test**

`engine/crates/fd-tiles/tests/pyramid.rs`:

```rust
use fd_io::{ImageBuf, PixelData};
use fd_tiles::{Pyramid, TILE_SIZE};

fn gray_image(width: u32, height: u32) -> ImageBuf {
    let n = (width * height) as usize;
    ImageBuf {
        width,
        height,
        channels: 1,
        data: PixelData::U16((0..n).map(|i| ((i * 65535) / n) as u16).collect()),
        icc: None,
        exif: None,
    }
}

#[test]
fn levels_halve_until_one_tile() {
    let p = Pyramid::build(&gray_image(2000, 1200));
    // 2000x1200 -> 1000x600 -> 500x300 (fits in one 512 tile)
    assert_eq!(p.levels.len(), 3);
    assert_eq!((p.levels[0].width, p.levels[0].height), (2000, 1200));
    assert_eq!((p.levels[1].width, p.levels[1].height), (1000, 600));
    assert_eq!((p.levels[2].width, p.levels[2].height), (500, 300));
}

#[test]
fn small_image_is_single_level() {
    let p = Pyramid::build(&gray_image(300, 200));
    assert_eq!(p.levels.len(), 1);
}

#[test]
fn tile_grid_and_edge_tiles() {
    let p = Pyramid::build(&gray_image(1100, 600));
    let (tx, ty) = p.tiles_at(0);
    assert_eq!((tx, ty), (3, 2)); // ceil(1100/512)=3, ceil(600/512)=2
    let full = p.tile(0, 0, 0).unwrap();
    assert_eq!((full.width, full.height), (TILE_SIZE, TILE_SIZE));
    let edge = p.tile(0, 2, 1).unwrap();
    assert_eq!((edge.width, edge.height), (1100 - 1024, 600 - 512));
    assert_eq!(edge.rgba.len(), (edge.width * edge.height * 4) as usize);
    assert!(p.tile(0, 3, 0).is_none());
    assert!(p.tile(9, 0, 0).is_none());
}

#[test]
fn rgba_is_opaque_and_downsample_averages() {
    // 2x2 image with values 0,0,65535,65535 -> level1 single pixel ~ mid gray
    let img = ImageBuf {
        width: 2,
        height: 2,
        channels: 1,
        data: PixelData::U16(vec![0, 0, 65535, 65535]),
        icc: None,
        exif: None,
    };
    let p = Pyramid::build(&img);
    let l0 = &p.levels[0];
    assert_eq!(l0.rgba[3], 255); // alpha opaque
    // level 0 exists only (2x2 fits one tile), so test averaging directly:
    let avg = fd_tiles::downsample_2x(&l0.rgba, 2, 2);
    // one output pixel, gray channels averaged: (0+0+255+255)/4 = 127 or 128
    assert!((avg.0[0] as i32 - 127).abs() <= 1);
    assert_eq!((avg.1, avg.2), (1, 1));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fd-tiles`
Expected: compile error — `Pyramid` not found.

- [ ] **Step 3: Implement**

`engine/crates/fd-tiles/Cargo.toml` dependencies:

```toml
[dependencies]
fd-io = { path = "../fd-io" }
```

`engine/crates/fd-tiles/src/lib.rs`:

```rust
//! Display tile pyramids and the byte-bounded tile cache.

mod pyramid;

pub use pyramid::{downsample_2x, Level, Pyramid, Tile, TILE_SIZE};
```

`engine/crates/fd-tiles/src/pyramid.rs`:

```rust
use fd_io::{ImageBuf, PixelData};

pub const TILE_SIZE: u32 = 512;

pub struct Level {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

pub struct Tile {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

pub struct Pyramid {
    pub levels: Vec<Level>,
}

fn base_rgba(img: &ImageBuf) -> Vec<u8> {
    let n = (img.width * img.height) as usize;
    let mut out = vec![255u8; n * 4];
    let ch = img.channels as usize;
    let write = |out: &mut [u8], i: usize, r: u8, g: u8, b: u8| {
        out[i * 4] = r;
        out[i * 4 + 1] = g;
        out[i * 4 + 2] = b;
    };
    match &img.data {
        PixelData::U8(v) => {
            for i in 0..n {
                if ch == 1 {
                    write(&mut out, i, v[i], v[i], v[i]);
                } else {
                    write(&mut out, i, v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
                }
            }
        }
        PixelData::U16(v) => {
            for i in 0..n {
                let g = |x: u16| (x >> 8) as u8;
                if ch == 1 {
                    write(&mut out, i, g(v[i]), g(v[i]), g(v[i]));
                } else {
                    write(&mut out, i, g(v[i * 3]), g(v[i * 3 + 1]), g(v[i * 3 + 2]));
                }
            }
        }
    }
    out
}

/// 2x2 box-average downsample of an RGBA buffer. Returns (rgba, w, h).
pub fn downsample_2x(rgba: &[u8], width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let (nw, nh) = (width.div_ceil(2).max(1), height.div_ceil(2).max(1));
    let mut out = vec![255u8; (nw * nh * 4) as usize];
    for oy in 0..nh {
        for ox in 0..nw {
            for c in 0..3usize {
                let mut sum = 0u32;
                let mut cnt = 0u32;
                for dy in 0..2u32 {
                    for dx in 0..2u32 {
                        let (sx, sy) = (ox * 2 + dx, oy * 2 + dy);
                        if sx < width && sy < height {
                            sum += rgba[((sy * width + sx) * 4) as usize + c] as u32;
                            cnt += 1;
                        }
                    }
                }
                out[((oy * nw + ox) * 4) as usize + c] = (sum / cnt) as u8;
            }
        }
    }
    (out, nw, nh)
}

impl Pyramid {
    pub fn build(img: &ImageBuf) -> Pyramid {
        let mut levels = vec![Level { width: img.width, height: img.height, rgba: base_rgba(img) }];
        while levels.last().unwrap().width.max(levels.last().unwrap().height) > TILE_SIZE {
            let last = levels.last().unwrap();
            let (rgba, w, h) = downsample_2x(&last.rgba, last.width, last.height);
            levels.push(Level { width: w, height: h, rgba });
        }
        levels.into()
    }

    pub fn tiles_at(&self, level: u8) -> (u32, u32) {
        let l = &self.levels[level as usize];
        (l.width.div_ceil(TILE_SIZE), l.height.div_ceil(TILE_SIZE))
    }

    pub fn tile(&self, level: u8, tx: u32, ty: u32) -> Option<Tile> {
        let l = self.levels.get(level as usize)?;
        let (gx, gy) = (l.width.div_ceil(TILE_SIZE), l.height.div_ceil(TILE_SIZE));
        if tx >= gx || ty >= gy {
            return None;
        }
        let x0 = tx * TILE_SIZE;
        let y0 = ty * TILE_SIZE;
        let w = (l.width - x0).min(TILE_SIZE);
        let h = (l.height - y0).min(TILE_SIZE);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for row in 0..h {
            let src = (((y0 + row) * l.width + x0) * 4) as usize;
            let dst = (row * w * 4) as usize;
            rgba[dst..dst + (w * 4) as usize].copy_from_slice(&l.rgba[src..src + (w * 4) as usize]);
        }
        Some(Tile { width: w, height: h, rgba })
    }
}

impl From<Vec<Level>> for Pyramid {
    fn from(levels: Vec<Level>) -> Self {
        Pyramid { levels }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p fd-tiles`
Expected: 4 tests pass.

- [ ] **Step 5: Format, lint, commit**

Run: `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`

```bash
git add engine
git commit -m "Build display tile pyramids in fd-tiles"
```

---

### Task 5: fd-tiles byte-bounded LRU tile cache

**Files:**
- Create: `engine/crates/fd-tiles/src/cache.rs`
- Modify: `engine/crates/fd-tiles/src/lib.rs` (export)
- Modify: `engine/crates/fd-tiles/Cargo.toml` (add `lru = "0.12"`)
- Test: `engine/crates/fd-tiles/tests/cache.rs`

**Interfaces:**
- Consumes: `Pyramid`, `Tile` from Task 4.
- Produces: `TileKey { image_id: u64, level: u8, tx: u32, ty: u32 }`; `TileCache::new(byte_budget: usize) -> TileCache`; `get_or_insert(&mut self, key: TileKey, build: impl FnOnce() -> Option<Tile>) -> Option<Arc<Tile>>`; `bytes_used(&self) -> usize`; `evict_image(&mut self, image_id: u64)`. The app shell will own one cache across all open frames — this is the "degrade to slower, never to crashed" memory bound from the spec.

- [ ] **Step 1: Write the failing test**

`engine/crates/fd-tiles/tests/cache.rs`:

```rust
use fd_tiles::{Tile, TileCache, TileKey, TILE_SIZE};

fn tile_of_bytes(n: u32) -> Tile {
    // width n/4 x height 1 rgba => exactly n bytes of pixel data
    Tile { width: n / 4, height: 1, rgba: vec![0; n as usize] }
}

fn key(image: u64, tx: u32) -> TileKey {
    TileKey { image_id: image, level: 0, tx, ty: 0 }
}

#[test]
fn caches_and_returns_same_tile() {
    let mut c = TileCache::new(10_000);
    let mut builds = 0;
    for _ in 0..3 {
        let t = c
            .get_or_insert(key(1, 0), || {
                builds += 1;
                Some(tile_of_bytes(1000))
            })
            .unwrap();
        assert_eq!(t.rgba.len(), 1000);
    }
    assert_eq!(builds, 1);
    assert_eq!(c.bytes_used(), 1000);
}

#[test]
fn evicts_least_recently_used_beyond_budget() {
    let mut c = TileCache::new(2500);
    c.get_or_insert(key(1, 0), || Some(tile_of_bytes(1000)));
    c.get_or_insert(key(1, 1), || Some(tile_of_bytes(1000)));
    // touch tile 0 so tile 1 is LRU
    c.get_or_insert(key(1, 0), || panic!("should be cached"));
    c.get_or_insert(key(1, 2), || Some(tile_of_bytes(1000)));
    assert!(c.bytes_used() <= 2500);
    let mut rebuilt = false;
    c.get_or_insert(key(1, 1), || {
        rebuilt = true;
        Some(tile_of_bytes(1000))
    });
    assert!(rebuilt, "LRU tile 1 should have been evicted");
}

#[test]
fn build_failure_is_not_cached() {
    let mut c = TileCache::new(1000);
    assert!(c.get_or_insert(key(1, 9), || None).is_none());
    assert_eq!(c.bytes_used(), 0);
}

#[test]
fn evict_image_drops_only_that_image() {
    let mut c = TileCache::new(100_000);
    c.get_or_insert(key(1, 0), || Some(tile_of_bytes(1000)));
    c.get_or_insert(key(2, 0), || Some(tile_of_bytes(1000)));
    c.evict_image(1);
    assert_eq!(c.bytes_used(), 1000);
    let _ = TILE_SIZE; // keep the import honest
}

#[test]
fn oversized_tile_still_served_but_not_retained_forever() {
    let mut c = TileCache::new(100);
    let t = c.get_or_insert(key(1, 0), || Some(tile_of_bytes(1000)));
    assert!(t.is_some());
    assert!(c.bytes_used() <= 1000); // inserted, then budget enforcement may evict it next insert
    c.get_or_insert(key(1, 1), || Some(tile_of_bytes(40)));
    assert!(c.bytes_used() <= 140);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fd-tiles --test cache`
Expected: compile error — `TileCache` not found.

- [ ] **Step 3: Implement**

`engine/crates/fd-tiles/src/cache.rs`:

```rust
use std::sync::Arc;

use lru::LruCache;

use crate::Tile;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TileKey {
    pub image_id: u64,
    pub level: u8,
    pub tx: u32,
    pub ty: u32,
}

/// LRU tile cache bounded by total pixel bytes, not entry count.
pub struct TileCache {
    inner: LruCache<TileKey, Arc<Tile>>,
    byte_budget: usize,
    bytes_used: usize,
}

fn tile_bytes(t: &Tile) -> usize {
    t.rgba.len()
}

impl TileCache {
    pub fn new(byte_budget: usize) -> TileCache {
        TileCache {
            inner: LruCache::unbounded(),
            byte_budget: byte_budget.max(1),
            bytes_used: 0,
        }
    }

    pub fn bytes_used(&self) -> usize {
        self.bytes_used
    }

    pub fn get_or_insert(
        &mut self,
        key: TileKey,
        build: impl FnOnce() -> Option<Tile>,
    ) -> Option<Arc<Tile>> {
        if let Some(hit) = self.inner.get(&key) {
            return Some(hit.clone());
        }
        let tile = Arc::new(build()?);
        self.bytes_used += tile_bytes(&tile);
        self.inner.put(key, tile.clone());
        // Evict LRU entries until under budget, but never the one just inserted.
        while self.bytes_used > self.byte_budget && self.inner.len() > 1 {
            if let Some((_, evicted)) = self.inner.pop_lru() {
                self.bytes_used -= tile_bytes(&evicted);
            } else {
                break;
            }
        }
        Some(tile)
    }

    pub fn evict_image(&mut self, image_id: u64) {
        let keys: Vec<TileKey> = self
            .inner
            .iter()
            .map(|(k, _)| *k)
            .filter(|k| k.image_id == image_id)
            .collect();
        for k in keys {
            if let Some(t) = self.inner.pop(&k) {
                self.bytes_used -= tile_bytes(&t);
            }
        }
    }
}
```

Add to lib.rs: `mod cache;` and `pub use cache::{TileCache, TileKey};`. Add `lru = "0.12"` to dependencies. `LruCache::unbounded()` needs no capacity argument — the byte budget is enforced manually, which is the point.

Careful with `pop_lru` when the just-inserted tile is the only entry: the `self.inner.len() > 1` guard keeps it resident even when oversized (serve it; it becomes evictable on the next insert). This matches the last test.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p fd-tiles`
Expected: all pass.

- [ ] **Step 5: Format, lint, commit**

Run: `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`

```bash
git add engine
git commit -m "Add byte-bounded LRU tile cache to fd-tiles"
```

---

### Task 6: Cross-language test fixtures (Python side)

**Files:**
- Create: `training/scripts/make_engine_fixtures.py`
- Create (generated, committed): `engine/fixtures/tiny-detector.onnx`, `engine/fixtures/tiny-inpaint.onnx`, `engine/fixtures/parity-input.bin`, `engine/fixtures/parity-expected.bin`, `engine/fixtures/parity-meta.json`

**Interfaces:**
- Consumes: the training env (`uv run` from `training/`), `unduster_training.detectors.OnnxDetector`.
- Produces: the fixtures fd-infer (Task 7) and fd-heal (Task 8/9) load. Formats: `parity-input.bin` = little-endian u16, row-major, grey, 600x540 (both dims > 512 forces a 2x2 tile grid with overlap averaging); `parity-expected.bin` = little-endian u16 quantized probabilities (`round(p * 65535)`), same shape; `parity-meta.json` = `{"width": 600, "height": 540, "tolerance": 0.002}`. tiny-detector: input "image" 1x1xHxW dynamic H/W, output "logits". tiny-inpaint: inputs "image" (1x3xHxW) and "mask" (1x1xHxW), output "output" = image*(1-mask) + per-channel-mean*mask.

- [ ] **Step 1: Write the generator**

`training/scripts/make_engine_fixtures.py`:

```python
"""Generate committed fixtures the Rust engine tests use.

Run from training/:  uv run python scripts/make_engine_fixtures.py
Deterministic: fixed seeds, no timestamps. Overwrites engine/fixtures/.
"""

import json
from pathlib import Path

import numpy as np
import torch

from unduster_training.detectors import OnnxDetector

OUT = Path(__file__).resolve().parents[2] / "engine" / "fixtures"
WIDTH, HEIGHT = 600, 540  # both > 512: forces 2x2 tile grid with overlaps


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.c1 = torch.nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = torch.nn.Conv2d(8, 1, 3, padding=1)

    def forward(self, x):
        return self.c2(torch.relu(self.c1(x)))


class TinyInpaint(torch.nn.Module):
    def forward(self, image, mask):
        mean = image.mean(dim=(2, 3), keepdim=True)
        return image * (1.0 - mask) + mean * mask


def export_detector() -> Path:
    path = OUT / "tiny-detector.onnx"
    model = TinyDetector().eval()
    torch.onnx.export(
        model,
        (torch.zeros(1, 1, 512, 512),),
        str(path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes={"x": {2: "h", 3: "w"}},
        opset_version=17,
        dynamo=True,
    )
    return path


def export_inpaint() -> None:
    model = TinyInpaint().eval()
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, 64, 64), torch.zeros(1, 1, 64, 64)),
        str(OUT / "tiny-inpaint.onnx"),
        input_names=["image", "mask"],
        output_names=["output"],
        dynamic_shapes={"image": {2: "h", 3: "w"}, "mask": {2: "h", 3: "w"}},
        opset_version=17,
        dynamo=True,
    )


def make_parity() -> None:
    rng = np.random.default_rng(11)
    img_u16 = (rng.random((HEIGHT, WIDTH)) * 65535.0).astype("<u2")
    (OUT / "parity-input.bin").write_bytes(img_u16.tobytes())
    img_f32 = img_u16.astype(np.float32) / 65535.0
    det = OnnxDetector(OUT / "tiny-detector.onnx")
    probs = det.probabilities(img_f32)
    probs_u16 = np.round(probs * 65535.0).astype("<u2")
    (OUT / "parity-expected.bin").write_bytes(probs_u16.tobytes())
    (OUT / "parity-meta.json").write_text(
        json.dumps({"width": WIDTH, "height": HEIGHT, "tolerance": 0.002}, indent=2)
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    export_detector()
    export_inpaint()
    make_parity()
    for f in sorted(OUT.iterdir()):
        print(f"{f.name}: {f.stat().st_size} bytes")


if __name__ == "__main__":
    main()
```

Note on `dynamic_shapes` keys: for `TinyDetector` the forward arg is `x`, for `TinyInpaint` the args are `image`/`mask` — keys follow FORWARD ARG NAMES, not ONNX input names (this is the dynamo exporter's rule; the training repo's export.py works the same way). If the inpaint export complains about dict keys, use the tuple form `dynamic_shapes=({2: "h", 3: "w"}, {2: "h", 3: "w"})`.

- [ ] **Step 2: Run the generator and sanity-check**

Run: `cd /Users/albert/Development/TheUnduster/training && uv run python scripts/make_engine_fixtures.py`
Expected: prints five files; parity bins are 648,000 bytes each (600*540*2); onnx files a few KB each.

Sanity check the expected map is nontrivial:

```bash
uv run python -c "
import numpy as np
p = np.frombuffer(open('../engine/fixtures/parity-expected.bin','rb').read(), dtype='<u2')
print('mean prob', p.mean()/65535, 'std', p.std()/65535)
assert 0.001 < p.std()/65535, 'expected map is flat; fixture is useless'
"
```
Expected: nonzero std.

- [ ] **Step 3: Commit**

```bash
cd /Users/albert/Development/TheUnduster
git add training/scripts/make_engine_fixtures.py engine/fixtures
git commit -m "Add cross-language parity fixtures for the Rust engine

The Rust fd-infer tiling must reproduce the Python OnnxDetector
reference output; these committed fixtures make that a test."
```

---

### Task 7: fd-infer tiled detection with parity test

**Files:**
- Modify: `engine/crates/fd-infer/Cargo.toml`
- Create: `engine/crates/fd-infer/src/lib.rs` (replacing stub)
- Test: `engine/crates/fd-infer/tests/parity.rs`

**Interfaces:**
- Consumes: `fd_io::ImageBuf`; fixtures from Task 6.
- Produces: `pub const TILE: usize = 512; pub const OVERLAP: usize = 64;`; `Ep::{Cpu, CoreML}`; `InferError`; `Detector::load(path: &Path, ep: Ep) -> Result<Detector, InferError>`; `Detector::probabilities(&self, img: &ImageBuf) -> Result<Vec<f32>, InferError>` (HxW row-major, [0,1]); `Detector::mask(&self, img: &ImageBuf, threshold: f32) -> Result<Vec<bool>, InferError>`. fd-heal and the app consume these.

- [ ] **Step 1: Write the failing test**

`engine/crates/fd-infer/tests/parity.rs`:

```rust
use std::path::PathBuf;

use fd_infer::{Detector, Ep};
use fd_io::{ImageBuf, PixelData};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn load_parity_input() -> (ImageBuf, u32, u32, f32) {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("parity-meta.json")).unwrap())
            .unwrap();
    let (w, h) = (meta["width"].as_u64().unwrap() as u32, meta["height"].as_u64().unwrap() as u32);
    let tol = meta["tolerance"].as_f64().unwrap() as f32;
    let bytes = std::fs::read(fixtures().join("parity-input.bin")).unwrap();
    let pixels: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    assert_eq!(pixels.len(), (w * h) as usize);
    let img = ImageBuf {
        width: w,
        height: h,
        channels: 1,
        data: PixelData::U16(pixels),
        icc: None,
        exif: None,
    };
    (img, w, h, tol)
}

#[test]
fn probabilities_match_python_reference() {
    let (img, w, h, tol) = load_parity_input();
    let det = Detector::load(&fixtures().join("tiny-detector.onnx"), Ep::Cpu).unwrap();
    let probs = det.probabilities(&img).unwrap();
    assert_eq!(probs.len(), (w * h) as usize);

    let expected_bytes = std::fs::read(fixtures().join("parity-expected.bin")).unwrap();
    let expected: Vec<f32> = expected_bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]) as f32 / 65535.0)
        .collect();

    let mut max_diff = 0f32;
    for (a, b) in probs.iter().zip(expected.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(
        max_diff < tol,
        "max deviation from Python reference: {max_diff} (tolerance {tol})"
    );
}

#[test]
fn mask_thresholds_probabilities() {
    let (img, ..) = load_parity_input();
    let det = Detector::load(&fixtures().join("tiny-detector.onnx"), Ep::Cpu).unwrap();
    let probs = det.probabilities(&img).unwrap();
    let mask = det.mask(&img, 0.5).unwrap();
    for (p, m) in probs.iter().zip(mask.iter()) {
        assert_eq!(*m, *p > 0.5);
    }
}

#[test]
fn rgb_input_to_gray_model_is_adapted() {
    // 3-channel image against the 1-channel tiny detector must not error
    let img = ImageBuf {
        width: 100,
        height: 80,
        channels: 3,
        data: PixelData::U8(vec![100; 100 * 80 * 3]),
        icc: None,
        exif: None,
    };
    let det = Detector::load(&fixtures().join("tiny-detector.onnx"), Ep::Cpu).unwrap();
    let probs = det.probabilities(&img).unwrap();
    assert_eq!(probs.len(), 100 * 80);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fd-infer`
Expected: compile error — `Detector` not found.

- [ ] **Step 3: Implement**

`engine/crates/fd-infer/Cargo.toml`:

```toml
[package]
name = "fd-infer"
edition.workspace = true
version.workspace = true

[dependencies]
fd-io = { path = "../fd-io" }
thiserror.workspace = true
ort = { version = "2.0.0-rc.10", features = ["coreml"] }
ndarray = "0.16"

[dev-dependencies]
serde_json = "1"
```

`engine/crates/fd-infer/src/lib.rs`:

```rust
//! Tiled ONNX detection. The tiling arithmetic mirrors
//! training/src/unduster_training/detectors.py (the reference):
//! 512px tiles, 64px overlap (stride 448), edge-replicate padding,
//! probability averaging in overlaps.

use std::path::Path;

use fd_io::{ImageBuf, PixelData};
use ndarray::Array4;
use ort::session::Session;

pub const TILE: usize = 512;
pub const OVERLAP: usize = 64;

#[derive(Clone, Copy, Debug)]
pub enum Ep {
    Cpu,
    CoreML,
}

#[derive(Debug, thiserror::Error)]
pub enum InferError {
    #[error("cannot load model {path}: {reason}")]
    Load { path: String, reason: String },
    #[error("inference failed: {0}")]
    Run(String),
    #[error("model has unsupported input channels: {0}")]
    Channels(i64),
}

pub struct Detector {
    session: Session,
    input_name: String,
    in_ch: usize,
}

/// Rec.709 grey, matching unduster_training.io.to_gray.
fn to_gray_f32(img: &ImageBuf) -> Vec<f32> {
    let f = img.to_f32();
    if img.channels == 1 {
        return f;
    }
    f.chunks_exact(3)
        .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
        .collect()
}

/// Channel-first planes, adapting channel count to the model.
fn planes_for(img: &ImageBuf, in_ch: usize) -> Vec<Vec<f32>> {
    if in_ch == 1 {
        vec![to_gray_f32(img)]
    } else if img.channels == 1 {
        let g = img.to_f32();
        vec![g.clone(), g.clone(), g]
    } else {
        let f = img.to_f32();
        let n = (img.width * img.height) as usize;
        let mut planes = vec![vec![0f32; n]; 3];
        for i in 0..n {
            for c in 0..3 {
                planes[c][i] = f[i * 3 + c];
            }
        }
        planes
    }
}

impl Detector {
    pub fn load(path: &Path, ep: Ep) -> Result<Detector, InferError> {
        let mk_err = |e: ort::Error| InferError::Load {
            path: path.display().to_string(),
            reason: e.to_string(),
        };
        let mut builder = Session::builder().map_err(mk_err)?;
        if let Ep::CoreML = ep {
            builder = builder
                .with_execution_providers([
                    ort::execution_providers::CoreMLExecutionProvider::default().build()
                ])
                .map_err(mk_err)?;
        }
        let session = builder.commit_from_file(path).map_err(mk_err)?;
        let input = &session.inputs[0];
        let input_name = input.name.clone();
        let in_ch = match input.input_type.tensor_dimensions() {
            Some(dims) if dims.len() == 4 => match dims[1] {
                1 => 1usize,
                3 => 3usize,
                other => return Err(InferError::Channels(other)),
            },
            _ => 1, // dynamic or unusual: assume grey, the safer default for our models
        };
        Ok(Detector { session, input_name, in_ch })
    }

    pub fn probabilities(&self, img: &ImageBuf) -> Result<Vec<f32>, InferError> {
        let planes = planes_for(img, self.in_ch);
        let (w, h) = (img.width as usize, img.height as usize);
        let stride = TILE - OVERLAP;
        let mut acc = vec![0f32; w * h];
        let mut weight = vec![0f32; w * h];

        let mut y0 = 0usize;
        loop {
            let mut x0 = 0usize;
            loop {
                let y1 = (y0 + TILE).min(h);
                let x1 = (x0 + TILE).min(w);
                // Build an edge-replicate padded TILE x TILE tensor.
                let mut tile = Array4::<f32>::zeros((1, self.in_ch, TILE, TILE));
                for c in 0..self.in_ch {
                    for ty in 0..TILE {
                        let sy = (y0 + ty).min(h - 1).min(y1 - 1).max(y0);
                        for tx in 0..TILE {
                            let sx = (x0 + tx).min(w - 1).min(x1 - 1).max(x0);
                            tile[[0, c, ty, tx]] = planes[c][sy * w + sx];
                        }
                    }
                }
                let outputs = self
                    .session
                    .run(ort::inputs![self.input_name.as_str() => tile.view()]
                        .map_err(|e| InferError::Run(e.to_string()))?)
                    .map_err(|e| InferError::Run(e.to_string()))?;
                let logits = outputs[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| InferError::Run(e.to_string()))?;
                let view = logits.view();
                for ty in 0..(y1 - y0) {
                    for tx in 0..(x1 - x0) {
                        let l = view[[0, 0, ty, tx]];
                        let p = 1.0 / (1.0 + (-l).exp());
                        let idx = (y0 + ty) * w + (x0 + tx);
                        acc[idx] += p;
                        weight[idx] += 1.0;
                    }
                }
                if x0 + stride >= w.saturating_sub(OVERLAP).max(1) {
                    break;
                }
                x0 += stride;
            }
            if y0 + stride >= h.saturating_sub(OVERLAP).max(1) {
                break;
            }
            y0 += stride;
        }
        for i in 0..acc.len() {
            acc[i] /= weight[i];
        }
        Ok(acc)
    }

    pub fn mask(&self, img: &ImageBuf, threshold: f32) -> Result<Vec<bool>, InferError> {
        Ok(self.probabilities(img)?.iter().map(|&p| p > threshold).collect())
    }
}
```

Two precision points the implementer must preserve:
1. The loop bounds replicate Python's `range(0, max(h - OVERLAP, 1), stride)`: tile origins are 0, 448, 896, ... while origin < max(h - 64, 1). The `loop`/`break` construction above encodes exactly that — verify against the reference before committing if in doubt.
2. Padding fills beyond (y1, x1) with the last in-tile row/column (edge replicate of the *tile region*, matching numpy `np.pad(tile, mode="edge")` applied to the cropped tile — NOT replicate of the whole image). The `sy`/`sx` clamping expressions do this: they clamp into `[y0, y1-1]`, the cropped tile's own extent.

If the ort rc API differs (`Session::builder`, `commit_from_file`, `ort::inputs!`, `try_extract_tensor` names), adapt to the installed version per docs.rs — semantics fixed.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p fd-infer`
Expected: 3 tests pass. The parity test failing on max_diff means the tiling arithmetic diverges from the reference — fix the Rust side, never regenerate the fixture to match.

- [ ] **Step 5: Format, lint, commit**

Run: `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`

```bash
git add engine
git commit -m "Add tiled ONNX detection to fd-infer, parity-tested against Python"
```

---

### Task 8: fd-heal defect components and classical tiny fill

**Files:**
- Modify: `engine/crates/fd-heal/Cargo.toml`
- Create: `engine/crates/fd-heal/src/lib.rs` (replacing stub)
- Create: `engine/crates/fd-heal/src/components.rs`
- Create: `engine/crates/fd-heal/src/classical.rs`
- Test: `engine/crates/fd-heal/tests/components.rs`
- Test: `engine/crates/fd-heal/tests/classical.rs`

**Interfaces:**
- Consumes: `fd_io::{ImageBuf, PixelData}`.
- Produces: `Defect { pixels: Vec<(u32, u32)>, bbox: Bbox }` where `Bbox { x0: u32, y0: u32, x1: u32, y1: u32 }` (exclusive upper); `components(mask: &[bool], width: u32, height: u32) -> Vec<Defect>` (4-connectivity); `Defect::max_dim(&self) -> u32`; `classical_fill(planes: &mut [Vec<f32>], width: u32, height: u32, defect: &Defect, mask: &[bool])` — fills each defect pixel with the median of unmasked pixels in an expanding window. Task 9 composes these.

- [ ] **Step 1: Write the failing tests**

`engine/crates/fd-heal/tests/components.rs`:

```rust
use fd_heal::components;

fn mask_from(rows: &[&str]) -> (Vec<bool>, u32, u32) {
    let h = rows.len() as u32;
    let w = rows[0].len() as u32;
    let m = rows
        .iter()
        .flat_map(|r| r.chars().map(|c| c == '#'))
        .collect();
    (m, w, h)
}

#[test]
fn finds_separate_components_with_4_connectivity() {
    let (m, w, h) = mask_from(&[
        "##....",
        "##....",
        ".....#", // diagonal-only touch from the blob below => separate
        "....#.",
    ]);
    let comps = components(&m, w, h);
    assert_eq!(comps.len(), 3);
    let sizes: Vec<usize> = {
        let mut s: Vec<usize> = comps.iter().map(|c| c.pixels.len()).collect();
        s.sort();
        s
    };
    assert_eq!(sizes, vec![1, 1, 4]);
}

#[test]
fn bbox_is_tight_and_exclusive() {
    let (m, w, h) = mask_from(&["....", ".##.", "....", "...."]);
    let comps = components(&m, w, h);
    assert_eq!(comps.len(), 1);
    let b = &comps[0].bbox;
    assert_eq!((b.x0, b.y0, b.x1, b.y1), (1, 1, 3, 2));
    assert_eq!(comps[0].max_dim(), 2);
}

#[test]
fn empty_mask_no_components() {
    let (m, w, h) = mask_from(&["....", "...."]);
    assert!(components(&m, w, h).is_empty());
}
```

`engine/crates/fd-heal/tests/classical.rs`:

```rust
use fd_heal::{classical_fill, components};

#[test]
fn fill_replaces_speck_with_surround_median() {
    let (w, h) = (16u32, 16u32);
    let mut plane = vec![0.5f32; (w * h) as usize];
    let mut mask = vec![false; (w * h) as usize];
    // 2x2 dark speck at (7,7)
    for y in 7..9u32 {
        for x in 7..9u32 {
            plane[(y * w + x) as usize] = 0.05;
            mask[(y * w + x) as usize] = true;
        }
    }
    let defects = components(&mask, w, h);
    assert_eq!(defects.len(), 1);
    let mut planes = vec![plane];
    classical_fill(&mut planes, w, h, &defects[0], &mask);
    for y in 7..9u32 {
        for x in 7..9u32 {
            let v = planes[0][(y * w + x) as usize];
            assert!((v - 0.5).abs() < 1e-6, "pixel ({x},{y}) = {v}, want 0.5");
        }
    }
}

#[test]
fn fill_ignores_other_masked_pixels_when_sampling() {
    // speck adjacent to another masked region: the median must come from
    // clean pixels only, never from other defects
    let (w, h) = (16u32, 16u32);
    let mut plane = vec![0.8f32; (w * h) as usize];
    let mut mask = vec![false; (w * h) as usize];
    plane[(8 * w + 8) as usize] = 0.0;
    mask[(8 * w + 8) as usize] = true;
    // neighboring "other defect" pixels with extreme values
    for x in 9..12u32 {
        plane[(8 * w + x) as usize] = 0.0;
        mask[(8 * w + x) as usize] = true;
    }
    let defects = components(&mask, w, h);
    let target = defects.iter().find(|d| d.pixels.len() == 4).unwrap();
    let mut planes = vec![plane];
    classical_fill(&mut planes, w, h, target, &mask);
    for (x, y) in &target.pixels {
        let v = planes[0][(y * w + x) as usize];
        assert!((v - 0.8).abs() < 1e-6);
    }
}
```

(Note: in the second test all four masked pixels are 4-connected, forming ONE component of size 4 — the test fills it and asserts clean-median fill; adjust the find accordingly.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p fd-heal`
Expected: compile error.

- [ ] **Step 3: Implement**

`engine/crates/fd-heal/Cargo.toml` dependencies:

```toml
[dependencies]
fd-io = { path = "../fd-io" }
fd-infer = { path = "../fd-infer" }
thiserror.workspace = true
ort = { version = "2.0.0-rc.10", features = ["coreml"] }
ndarray = "0.16"
```

`engine/crates/fd-heal/src/components.rs`:

```rust
#[derive(Debug, Clone, Copy)]
pub struct Bbox {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32, // exclusive
    pub y1: u32, // exclusive
}

#[derive(Debug, Clone)]
pub struct Defect {
    pub pixels: Vec<(u32, u32)>,
    pub bbox: Bbox,
}

impl Defect {
    pub fn max_dim(&self) -> u32 {
        (self.bbox.x1 - self.bbox.x0).max(self.bbox.y1 - self.bbox.y0)
    }
}

/// Connected components over a boolean mask, 4-connectivity (matches
/// scipy.ndimage.label's default used by the training metrics).
pub fn components(mask: &[bool], width: u32, height: u32) -> Vec<Defect> {
    let (w, h) = (width as usize, height as usize);
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    for start in 0..w * h {
        if !mask[start] || seen[start] {
            continue;
        }
        let mut pixels = Vec::new();
        let mut stack = vec![start];
        seen[start] = true;
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        while let Some(i) = stack.pop() {
            let (x, y) = ((i % w) as u32, (i / w) as u32);
            pixels.push((x, y));
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
            let neighbors = [
                (x > 0).then(|| i - 1),
                (x + 1 < width).then(|| i + 1),
                (y > 0).then(|| i - w),
                (y + 1 < height).then(|| i + w),
            ];
            for n in neighbors.into_iter().flatten() {
                if mask[n] && !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        out.push(Defect { pixels, bbox: Bbox { x0, y0, x1, y1 } });
    }
    out
}
```

`engine/crates/fd-heal/src/classical.rs`:

```rust
use crate::Defect;

/// Fill each defect pixel with the median of clean (unmasked) pixels in an
/// expanding square window. Grain-aware in the sense that the median of a
/// grainy neighborhood carries local intensity statistics.
pub fn classical_fill(
    planes: &mut [Vec<f32>],
    width: u32,
    height: u32,
    defect: &Defect,
    mask: &[bool],
) {
    let w = width as usize;
    for &(px, py) in &defect.pixels {
        for radius in 2..=16i64 {
            let mut samples: Vec<Vec<f32>> = vec![Vec::new(); planes.len()];
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let (sx, sy) = (px as i64 + dx, py as i64 + dy);
                    if sx < 0 || sy < 0 || sx >= width as i64 || sy >= height as i64 {
                        continue;
                    }
                    let idx = sy as usize * w + sx as usize;
                    if mask[idx] {
                        continue;
                    }
                    for (c, plane) in planes.iter().enumerate() {
                        samples[c].push(plane[idx]);
                    }
                }
            }
            if samples[0].len() >= 5 {
                for (c, s) in samples.iter_mut().enumerate() {
                    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    planes[c][py as usize * w + px as usize] = s[s.len() / 2];
                }
                break;
            }
        }
    }
}
```

`engine/crates/fd-heal/src/lib.rs`:

```rust
//! Tiered defect healing with a bit-exactness guarantee outside masks.

mod classical;
mod components;

pub use classical::classical_fill;
pub use components::{components, Bbox, Defect};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p fd-heal`
Expected: 5 tests pass.

- [ ] **Step 5: Format, lint, commit**

Run: `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`

```bash
git add engine
git commit -m "Label defect components and add classical median fill to fd-heal"
```

---

### Task 9: fd-heal inpaint tier, grain re-synthesis, and bit-exact compose

**Files:**
- Create: `engine/crates/fd-heal/src/inpaint.rs`
- Create: `engine/crates/fd-heal/src/grain.rs`
- Create: `engine/crates/fd-heal/src/heal.rs`
- Modify: `engine/crates/fd-heal/src/lib.rs` (exports)
- Test: `engine/crates/fd-heal/tests/heal.rs`

**Interfaces:**
- Consumes: Tasks 6-8 (`tiny-inpaint.onnx` fixture, `components`, `classical_fill`, `fd_io::ImageBuf`).
- Produces: `HealError`; `Inpainter::load(path: &Path, ep: fd_infer::Ep) -> Result<Inpainter, HealError>` running the inpaint model on RGB crops (`image` 1x3xHxW + `mask` 1x1xHxW -> `output`); `pub const TINY_MAX_DIM: u32 = 5;`; `heal(img: &mut ImageBuf, mask: &[bool], inpainter: Option<&Inpainter>) -> Result<HealReport, HealError>` with `HealReport { defects: usize, tiny: usize, inpainted: usize }`. Guarantee: after `heal`, every pixel where `mask` is false is bit-identical (native integer compare) to before. The desktop app calls exactly this function.

- [ ] **Step 1: Write the failing test**

`engine/crates/fd-heal/tests/heal.rs`:

```rust
use std::path::PathBuf;

use fd_heal::{heal, Inpainter, TINY_MAX_DIM};
use fd_io::{ImageBuf, PixelData};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// Deterministic pseudo-random 16-bit RGB image.
fn noisy_image(width: u32, height: u32) -> ImageBuf {
    let n = (width * height * 3) as usize;
    let mut state = 0x2545F4914F6CDD1Du64;
    let data: Vec<u16> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 48) as u16
        })
        .collect();
    ImageBuf { width, height, channels: 3, data: PixelData::U16(data), icc: None, exif: None }
}

fn blob_mask(width: u32, height: u32, cx: u32, cy: u32, r: u32) -> Vec<bool> {
    (0..width * height)
        .map(|i| {
            let (x, y) = (i % width, i / width);
            (x as i64 - cx as i64).pow(2) + (y as i64 - cy as i64).pow(2) <= (r as i64).pow(2)
        })
        .collect()
}

#[test]
fn unmasked_pixels_are_bit_identical_after_heal() {
    let mut img = noisy_image(200, 160);
    let before = img.clone();
    let mask = blob_mask(200, 160, 100, 80, 12); // big blob -> inpaint tier
    let inp = Inpainter::load(&fixtures().join("tiny-inpaint.onnx"), fd_infer::Ep::Cpu).unwrap();
    let report = heal(&mut img, &mask, Some(&inp)).unwrap();
    assert_eq!(report.defects, 1);
    assert_eq!(report.inpainted, 1);
    let (PixelData::U16(a), PixelData::U16(b)) = (&before.data, &img.data) else {
        panic!("expected u16")
    };
    let mut changed_inside = 0;
    for i in 0..(200 * 160) as usize {
        for c in 0..3 {
            if mask[i] {
                if a[i * 3 + c] != b[i * 3 + c] {
                    changed_inside += 1;
                }
            } else {
                assert_eq!(a[i * 3 + c], b[i * 3 + c], "unmasked pixel {i} changed");
            }
        }
    }
    assert!(changed_inside > 0, "healing did nothing inside the mask");
}

#[test]
fn tiny_defects_use_classical_tier_without_model() {
    let mut img = noisy_image(64, 64);
    let mask = blob_mask(64, 64, 32, 32, 2); // 5px diameter -> tiny tier
    let report = heal(&mut img, &mask, None).unwrap();
    assert_eq!(report.tiny, 1);
    assert_eq!(report.inpainted, 0);
    assert!(TINY_MAX_DIM >= 5);
}

#[test]
fn large_defect_without_inpainter_falls_back_to_classical() {
    let mut img = noisy_image(128, 128);
    let before = img.clone();
    let mask = blob_mask(128, 128, 64, 64, 10);
    let report = heal(&mut img, &mask, None).unwrap();
    assert_eq!(report.defects, 1);
    assert_eq!(report.tiny + report.inpainted, 1);
    // guarantee still holds on the fallback path
    let (PixelData::U16(a), PixelData::U16(b)) = (&before.data, &img.data) else {
        panic!()
    };
    for i in 0..(128 * 128) as usize {
        if !mask[i] {
            for c in 0..3 {
                assert_eq!(a[i * 3 + c], b[i * 3 + c]);
            }
        }
    }
}

#[test]
fn healed_region_carries_grain() {
    // inpaint fixture returns a flat mean fill; grain re-synthesis must add
    // texture so the filled region is not flat
    let mut img = noisy_image(200, 160);
    let mask = blob_mask(200, 160, 100, 80, 12);
    let inp = Inpainter::load(&fixtures().join("tiny-inpaint.onnx"), fd_infer::Ep::Cpu).unwrap();
    heal(&mut img, &mask, Some(&inp)).unwrap();
    let PixelData::U16(v) = &img.data else { panic!() };
    let inside: Vec<f64> = (0..(200 * 160) as usize)
        .filter(|&i| mask[i])
        .map(|i| v[i * 3] as f64)
        .collect();
    let mean = inside.iter().sum::<f64>() / inside.len() as f64;
    let var = inside.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / inside.len() as f64;
    assert!(var.sqrt() > 100.0, "filled region is flat: std {}", var.sqrt());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fd-heal --test heal`
Expected: compile error — `heal`, `Inpainter` not found.

- [ ] **Step 3: Implement**

`engine/crates/fd-heal/src/inpaint.rs`:

```rust
use std::path::Path;

use ndarray::Array4;
use ort::session::Session;

use crate::HealError;

pub struct Inpainter {
    session: Session,
}

impl Inpainter {
    pub fn load(path: &Path, _ep: fd_infer::Ep) -> Result<Inpainter, HealError> {
        // CoreML wiring mirrors fd_infer::Detector::load; CPU is fine for the
        // fixture-scale tests and CI.
        let session = Session::builder()
            .and_then(|b| b.commit_from_file(path))
            .map_err(|e| HealError::Model(e.to_string()))?;
        Ok(Inpainter { session })
    }

    /// image: 3 planes HxW in [0,1]; mask: HxW (true = fill). Returns 3 planes.
    pub fn inpaint(
        &self,
        planes: &[Vec<f32>; 3],
        mask: &[bool],
        width: usize,
        height: usize,
    ) -> Result<[Vec<f32>; 3], HealError> {
        let mut image = Array4::<f32>::zeros((1, 3, height, width));
        let mut m = Array4::<f32>::zeros((1, 1, height, width));
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    image[[0, c, y, x]] = planes[c][y * width + x];
                }
                m[[0, 0, y, x]] = if mask[y * width + x] { 1.0 } else { 0.0 };
            }
        }
        let outputs = self
            .session
            .run(
                ort::inputs!["image" => image.view(), "mask" => m.view()]
                    .map_err(|e| HealError::Model(e.to_string()))?,
            )
            .map_err(|e| HealError::Model(e.to_string()))?;
        let out = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| HealError::Model(e.to_string()))?;
        let view = out.view();
        let mut result = [
            vec![0f32; width * height],
            vec![0f32; width * height],
            vec![0f32; width * height],
        ];
        for c in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    result[c][y * width + x] = view[[0, c, y, x]];
                }
            }
        }
        Ok(result)
    }
}
```

`engine/crates/fd-heal/src/grain.rs`:

```rust
//! Grain re-synthesis: measure high-pass residual statistics in the clean
//! ring around a fill, then re-apply matched Gaussian noise inside it.
//! Deterministic: the RNG is seeded from the defect's bbox, so healing the
//! same defect twice gives the same pixels.

use crate::{Bbox, Defect};

struct XorShift(u64);

impl XorShift {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 // [0,1)
    }

    /// Box-Muller standard normal.
    fn next_gauss(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

fn ring_sigma(plane: &[f32], width: usize, height: usize, bbox: &Bbox, mask: &[bool]) -> f32 {
    // residual = pixel - 3x3 mean, over unmasked pixels in a ring 8px around bbox
    let x0 = bbox.x0.saturating_sub(8) as usize;
    let y0 = bbox.y0.saturating_sub(8) as usize;
    let x1 = ((bbox.x1 + 8) as usize).min(width);
    let y1 = ((bbox.y1 + 8) as usize).min(height);
    let mut sum = 0f64;
    let mut sum2 = 0f64;
    let mut n = 0f64;
    for y in y0.max(1)..y1.min(height - 1) {
        for x in x0.max(1)..x1.min(width - 1) {
            if mask[y * width + x] {
                continue;
            }
            let mut local = 0f32;
            for dy in 0..3usize {
                for dx in 0..3usize {
                    local += plane[(y + dy - 1) * width + (x + dx - 1)];
                }
            }
            let r = (plane[y * width + x] - local / 9.0) as f64;
            sum += r;
            sum2 += r * r;
            n += 1.0;
        }
    }
    if n < 16.0 {
        return 0.0;
    }
    let mean = sum / n;
    ((sum2 / n - mean * mean).max(0.0) as f32).sqrt()
}

pub fn add_grain(
    planes: &mut [Vec<f32>],
    width: usize,
    height: usize,
    defect: &Defect,
    mask: &[bool],
) {
    let mut rng = XorShift(
        0x9E3779B97F4A7C15u64
            ^ ((defect.bbox.x0 as u64) << 32)
            ^ ((defect.bbox.y0 as u64) << 16)
            ^ defect.pixels.len() as u64,
    );
    for c in 0..planes.len() {
        let sigma = ring_sigma(&planes[c], width, height, &defect.bbox, mask);
        if sigma <= 0.0 {
            continue;
        }
        for &(x, y) in &defect.pixels {
            let idx = y as usize * width + x as usize;
            planes[c][idx] = (planes[c][idx] + rng.next_gauss() * sigma).clamp(0.0, 1.0);
        }
    }
}
```

`engine/crates/fd-heal/src/heal.rs`:

```rust
use fd_io::{ImageBuf, PixelData};

use crate::{add_grain, classical_fill, components, Defect, HealError, Inpainter};

pub const TINY_MAX_DIM: u32 = 5;

#[derive(Debug, Default)]
pub struct HealReport {
    pub defects: usize,
    pub tiny: usize,
    pub inpainted: usize,
}

fn to_planes(img: &ImageBuf) -> Vec<Vec<f32>> {
    let f = img.to_f32();
    let n = (img.width * img.height) as usize;
    let ch = img.channels as usize;
    let mut planes = vec![vec![0f32; n]; ch];
    for i in 0..n {
        for (c, plane) in planes.iter_mut().enumerate() {
            plane[i] = f[i * ch + c];
        }
    }
    planes
}

/// Write healed values back into native depth, ONLY at masked pixels.
/// Everything else in img.data is untouched, which makes the bit-exactness
/// guarantee structural rather than aspirational.
fn write_back(img: &mut ImageBuf, planes: &[Vec<f32>], mask: &[bool]) {
    let ch = img.channels as usize;
    match &mut img.data {
        PixelData::U8(v) => {
            for (i, &m) in mask.iter().enumerate() {
                if m {
                    for c in 0..ch {
                        v[i * ch + c] = (planes[c][i] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
        PixelData::U16(v) => {
            for (i, &m) in mask.iter().enumerate() {
                if m {
                    for c in 0..ch {
                        v[i * ch + c] =
                            (planes[c][i] * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
                    }
                }
            }
        }
    }
}

fn inpaint_defect(
    planes: &mut [Vec<f32>],
    width: usize,
    height: usize,
    defect: &Defect,
    mask: &[bool],
    inpainter: &Inpainter,
) -> Result<(), HealError> {
    // Crop: bbox padded by 2x its max dim, min 64px, clamped, multiple of 8.
    let pad = (defect.max_dim() * 2).max(24) as usize;
    let cx0 = (defect.bbox.x0 as usize).saturating_sub(pad);
    let cy0 = (defect.bbox.y0 as usize).saturating_sub(pad);
    let mut cx1 = (defect.bbox.x1 as usize + pad).min(width);
    let mut cy1 = (defect.bbox.y1 as usize + pad).min(height);
    // round crop dims down to a multiple of 8 (>= 8), extending toward origin if needed
    let round8 = |a: usize, b: usize, limit: usize| -> (usize, usize) {
        let mut lo = a;
        let mut hi = b;
        let dim = ((hi - lo).max(8) / 8) * 8;
        if lo + dim <= limit {
            hi = lo + dim;
        } else {
            hi = limit;
            lo = hi.saturating_sub(dim);
        }
        (lo, hi)
    };
    let (cx0, ncx1) = round8(cx0, cx1, width);
    cx1 = ncx1;
    let (cy0, ncy1) = round8(cy0, cy1, height);
    cy1 = ncy1;
    let (cw, chh) = (cx1 - cx0, cy1 - cy0);

    // Assemble RGB crop planes (grey images: replicate the single plane).
    let get_plane = |c: usize| -> Vec<f32> {
        let src = if planes.len() == 1 { &planes[0] } else { &planes[c] };
        let mut out = vec![0f32; cw * chh];
        for y in 0..chh {
            for x in 0..cw {
                out[y * cw + x] = src[(cy0 + y) * width + (cx0 + x)];
            }
        }
        out
    };
    let crop = [get_plane(0), get_plane(1), get_plane(2)];
    let mut crop_mask = vec![false; cw * chh];
    for y in 0..chh {
        for x in 0..cw {
            crop_mask[y * cw + x] = mask[(cy0 + y) * width + (cx0 + x)];
        }
    }
    let filled = inpainter.inpaint(&crop, &crop_mask, cw, chh)?;
    // Write filled values back into working planes at THIS defect's pixels only.
    for &(px, py) in &defect.pixels {
        let (lx, ly) = (px as usize - cx0, py as usize - cy0);
        for (c, plane) in planes.iter_mut().enumerate() {
            let src_c = if c < 3 { c } else { 0 };
            plane[py as usize * width + px as usize] = filled[src_c][ly * cw + lx];
        }
    }
    Ok(())
}

pub fn heal(
    img: &mut ImageBuf,
    mask: &[bool],
    inpainter: Option<&Inpainter>,
) -> Result<HealReport, HealError> {
    let (width, height) = (img.width as usize, img.height as usize);
    if mask.len() != width * height {
        return Err(HealError::MaskSize { got: mask.len(), want: width * height });
    }
    let defects = components(mask, img.width, img.height);
    let mut planes = to_planes(img);
    let mut report = HealReport { defects: defects.len(), ..Default::default() };
    for d in &defects {
        if d.max_dim() <= TINY_MAX_DIM || inpainter.is_none() {
            classical_fill(&mut planes, img.width, img.height, d, mask);
            report.tiny += 1;
        } else {
            inpaint_defect(&mut planes, width, height, d, mask, inpainter.unwrap())?;
            add_grain(&mut planes, width, height, d, mask);
            report.inpainted += 1;
        }
    }
    write_back(img, &planes, mask);
    Ok(report)
}
```

Update `lib.rs`:

```rust
//! Tiered defect healing with a bit-exactness guarantee outside masks.

mod classical;
mod components;
mod grain;
mod heal;
mod inpaint;

pub use classical::classical_fill;
pub use components::{components, Bbox, Defect};
pub use grain::add_grain;
pub use heal::{heal, HealReport, TINY_MAX_DIM};
pub use inpaint::Inpainter;

#[derive(Debug, thiserror::Error)]
pub enum HealError {
    #[error("inpaint model error: {0}")]
    Model(String),
    #[error("mask size {got} does not match image size {want}")]
    MaskSize { got: usize, want: usize },
}
```

Bookkeeping note for the test `tiny_defects_use_classical_tier_without_model`: with `inpainter: None`, large defects also route classical and increment `tiny` — that is the fallback behavior test 3 asserts (`tiny + inpainted == 1`). If you prefer a separate `classical_fallback` counter, do NOT: the report shape is part of the interface.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p fd-heal`
Expected: all pass (components, classical, heal).

- [ ] **Step 5: Run the whole workspace**

Run: `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`
Expected: everything green.

- [ ] **Step 6: Commit**

```bash
git add engine
git commit -m "Add inpaint tier, grain re-synthesis, and bit-exact heal compose"
```

---

### Task 10: Engine CI and README

**Files:**
- Create: `.github/workflows/engine.yml`
- Create: `engine/README.md`

**Interfaces:**
- Consumes: the whole workspace.
- Produces: CI on every push/PR touching `engine/**`; a README documenting crate responsibilities, the parity-fixture regeneration procedure, and the two contracts (tiling, bit-exactness).

- [ ] **Step 1: Write the workflow and README**

`.github/workflows/engine.yml`:

```yaml
name: engine

on:
  push:
    paths:
      - "engine/**"
      - ".github/workflows/engine.yml"
  pull_request:
    paths:
      - "engine/**"
      - ".github/workflows/engine.yml"

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: engine
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - uses: Swatinem/rust-cache@v2
        with:
          workspaces: engine
      - run: cargo fmt --check
      - run: cargo clippy --all-targets -- -D warnings
      - run: cargo test
```

`engine/README.md`:

```markdown
# TheUnduster engine

Rust workspace with the four core crates the desktop app is built on.

- fd-io: decode/encode TIFF, PNG, JPEG at 8/16 bit into native-depth
  pixel buffers (u8/u16, never lossy-converted). ICC passes through for
  all three formats; EXIF passes through for JPEG. Originals are opened
  read-only.
- fd-tiles: display pyramids (2x box downsample to a single 512 tile)
  and a byte-bounded LRU tile cache.
- fd-infer: tiled ONNX detection. 512px tiles, 64px overlap, edge
  replicate padding, probability averaging. This must match
  training/src/unduster_training/detectors.py exactly; the parity test
  against committed fixtures is the proof.
- fd-heal: defect components, tiered healing (classical median fill for
  defects up to 5px, ONNX inpaint plus grain re-synthesis above that),
  and the bit-exactness guarantee: pixels outside the mask are
  integer-identical after healing. Healing is deterministic: grain noise
  is seeded from the defect's position.

## Build and test

    mise install
    cargo test

Tests use the CPU execution provider. The app selects CoreML at runtime
via fd_infer::Ep::CoreML.

## Fixtures

engine/fixtures/ holds committed cross-language fixtures (tiny ONNX
models plus a parity input/expected pair generated by the Python
reference implementation). Regenerate only when the reference tiling
changes, from training/:

    uv run python scripts/make_engine_fixtures.py

If the Rust parity test fails, fix the Rust side. Never regenerate the
fixtures to make Rust pass; they encode the Python reference behavior.
```

- [ ] **Step 2: Verify workspace is green one more time**

Run: `cd /Users/albert/Development/TheUnduster/engine && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/engine.yml engine/README.md
git commit -m "Add engine CI workflow and README"
```

---

## Definition of done for this sub-project

- `cargo test` green across the workspace; fmt and clippy clean.
- Parity test proves fd-infer matches the Python reference on committed fixtures.
- Bit-exactness property test proves heal never touches unmasked pixels.
- CI runs on pushes touching `engine/**`.
- Not in scope (deliberately, per spec build order): Tauri app shell, sidecar project files, export-with-checksum flow, CoreML benchmarking — those are sub-project 3, which consumes these crates.
