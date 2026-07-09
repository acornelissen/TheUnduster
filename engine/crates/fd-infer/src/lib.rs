//! Tiled ONNX detection. The tiling arithmetic mirrors
//! training/src/unduster_training/detectors.py (the reference):
//! 512px tiles, 64px overlap (stride 448), edge-replicate padding,
//! probability averaging in overlaps.

use std::path::Path;

use fd_io::ImageBuf;
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
///
/// Streams straight from the native pixels: going through to_f32 first
/// materializes an interleaved f32 copy of the whole image (2 GB for a
/// 168MP RGB scan) only to immediately reduce it, and that transient spike
/// was enough to push the app past the OS memory watchdog on real color
/// rolls. Normalization and operation order are kept identical to the
/// to_f32-based reduction so the output stays bit-for-bit the same (pinned
/// by gray_matches_to_f32_reference_u8_and_u16).
fn to_gray_f32(img: &ImageBuf) -> Vec<f32> {
    if img.channels == 1 {
        return img.to_f32();
    }
    match &img.data {
        fd_io::PixelData::U8(v) => v
            .chunks_exact(3)
            .map(|p| {
                0.2126 * (p[0] as f32 / 255.0)
                    + 0.7152 * (p[1] as f32 / 255.0)
                    + 0.0722 * (p[2] as f32 / 255.0)
            })
            .collect(),
        fd_io::PixelData::U16(v) => v
            .chunks_exact(3)
            .map(|p| {
                0.2126 * (p[0] as f32 / 65535.0)
                    + 0.7152 * (p[1] as f32 / 65535.0)
                    + 0.0722 * (p[2] as f32 / 65535.0)
            })
            .collect(),
    }
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
            for (c, plane) in planes.iter_mut().enumerate() {
                plane[i] = f[i * 3 + c];
            }
        }
        planes
    }
}

impl Detector {
    pub fn load(path: &Path, ep: Ep) -> Result<Detector, InferError> {
        let mk_err = |reason: String| InferError::Load {
            path: path.display().to_string(),
            reason,
        };
        let mut builder = Session::builder().map_err(|e| mk_err(e.to_string()))?;
        if let Ep::CoreML = ep {
            builder = builder
                .with_execution_providers([
                    ort::execution_providers::CoreMLExecutionProvider::default().build(),
                ])
                .map_err(|e| mk_err(e.to_string()))?;
        }
        let session = builder
            .commit_from_file(path)
            .map_err(|e| mk_err(e.to_string()))?;
        let input = &session.inputs()[0];
        let input_name = input.name().to_string();
        let in_ch = match input.dtype().tensor_shape() {
            Some(dims) if dims.len() == 4 => match dims[1] {
                1 => 1usize,
                3 => 3usize,
                other => return Err(InferError::Channels(other)),
            },
            _ => 1, // dynamic or unusual: assume grey, the safer default for our models
        };
        Ok(Detector {
            session,
            input_name,
            in_ch,
        })
    }

    pub fn probabilities(&mut self, img: &ImageBuf) -> Result<Vec<f32>, InferError> {
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
                // Edge-replicate padded TILE x TILE tensor; replication clamps
                // into the cropped tile's own extent, matching numpy's
                // np.pad(tile, mode="edge") on the crop.
                let mut tile = Array4::<f32>::zeros((1, self.in_ch, TILE, TILE));
                for (c, plane) in planes.iter().enumerate() {
                    for ty in 0..TILE {
                        let sy = (y0 + ty).min(y1 - 1);
                        for tx in 0..TILE {
                            let sx = (x0 + tx).min(x1 - 1);
                            tile[[0, c, ty, tx]] = plane[sy * w + sx];
                        }
                    }
                }
                let tensor = ort::value::TensorRef::from_array_view(tile.view())
                    .map_err(|e| InferError::Run(e.to_string()))?;
                let outputs = self
                    .session
                    .run(ort::inputs![self.input_name.as_str() => tensor])
                    .map_err(|e| InferError::Run(e.to_string()))?;
                let logits = outputs[0]
                    .try_extract_array::<f32>()
                    .map_err(|e| InferError::Run(e.to_string()))?;
                for ty in 0..(y1 - y0) {
                    for tx in 0..(x1 - x0) {
                        let l = logits[[0, 0, ty, tx]];
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

    pub fn mask(&mut self, img: &ImageBuf, threshold: f32) -> Result<Vec<bool>, InferError> {
        Ok(self
            .probabilities(img)?
            .iter()
            .map(|&p| p > threshold)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fd_io::PixelData;

    fn rgb_image(data: PixelData, w: u32, h: u32) -> ImageBuf {
        ImageBuf {
            width: w,
            height: h,
            channels: 3,
            data,
            icc: None,
            exif: None,
        }
    }

    fn pseudo_random_bytes(n: usize) -> Vec<u8> {
        let mut s = 7u32;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8
            })
            .collect()
    }

    /// The streaming gray path must be bit-identical to the reference
    /// reduction over to_f32 (same normalization, same operation order) --
    /// the detector's output feeds threshold comparisons, so even 1-ulp
    /// drift would move defect boundaries between releases.
    #[test]
    fn gray_matches_to_f32_reference_u8_and_u16() {
        let (w, h) = (37u32, 23u32);
        let n = (w * h) as usize;

        let bytes = pseudo_random_bytes(n * 3);
        let img8 = rgb_image(PixelData::U8(bytes.clone()), w, h);
        let reference8: Vec<f32> = img8
            .to_f32()
            .chunks_exact(3)
            .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
            .collect();
        assert_eq!(to_gray_f32(&img8), reference8);

        let words: Vec<u16> = pseudo_random_bytes(n * 3)
            .into_iter()
            .map(|b| (b as u16) << 8 | 0x2f)
            .collect();
        let img16 = rgb_image(PixelData::U16(words), w, h);
        let reference16: Vec<f32> = img16
            .to_f32()
            .chunks_exact(3)
            .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
            .collect();
        assert_eq!(to_gray_f32(&img16), reference16);
    }

    #[test]
    fn gray_passes_single_channel_through() {
        let img = ImageBuf {
            width: 4,
            height: 2,
            channels: 1,
            data: PixelData::U8(vec![0, 51, 102, 153, 204, 255, 7, 91]),
            icc: None,
            exif: None,
        };
        assert_eq!(to_gray_f32(&img), img.to_f32());
    }
}
