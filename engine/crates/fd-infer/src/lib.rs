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
