use std::path::Path;

use ndarray::Array4;
use ort::session::Session;

use crate::HealError;

/// The two ONNX inpainting contracts in the wild:
/// - Dynamic: our dev fixture -- any HxW, output in [0,1].
/// - Fixed(n): LaMa-style exports -- static nxn dims (Fourier layers cannot
///   export dynamically), output scaled 0-255. Callers window their crops
///   to exactly nxn (see heal.rs) and the adapter unscales the output.
enum Contract {
    Dynamic,
    Fixed(usize),
}

pub struct Inpainter {
    session: Session,
    contract: Contract,
}

impl Inpainter {
    pub fn load(path: &Path, _ep: fd_infer::Ep) -> Result<Inpainter, HealError> {
        // CoreML wiring mirrors fd_infer::Detector::load; CPU is fine for the
        // fixture-scale tests and CI.
        let session = Session::builder()
            .map_err(|e| HealError::Model(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| HealError::Model(e.to_string()))?;

        // Detect the contract by inspecting the input shape.
        let contract = if let Some(dims) = session.inputs()[0].dtype().tensor_shape() {
            if dims.len() == 4 && dims[2] > 0 && dims[3] > 0 && dims[2] == dims[3] {
                Contract::Fixed(dims[2] as usize)
            } else {
                Contract::Dynamic
            }
        } else {
            Contract::Dynamic
        };

        Ok(Inpainter { session, contract })
    }

    /// Returns the fixed window size if the model uses a fixed-size contract,
    /// or None if it accepts arbitrary dimensions.
    pub fn window_size(&self) -> Option<usize> {
        match self.contract {
            Contract::Fixed(n) => Some(n),
            Contract::Dynamic => None,
        }
    }

    /// image: 3 planes HxW in [0,1]; mask: HxW (true = fill). Returns 3 planes.
    pub fn inpaint(
        &mut self,
        planes: &[Vec<f32>; 3],
        mask: &[bool],
        width: usize,
        height: usize,
    ) -> Result<[Vec<f32>; 3], HealError> {
        // Validate contract for fixed-size models.
        if let Contract::Fixed(n) = self.contract {
            if width != n || height != n {
                return Err(HealError::Model(format!(
                    "model expects {n}x{n} crops, got {width}x{height}"
                )));
            }
        }

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
        let image_t = ort::value::TensorRef::from_array_view(image.view())
            .map_err(|e| HealError::Model(e.to_string()))?;
        let mask_t = ort::value::TensorRef::from_array_view(m.view())
            .map_err(|e| HealError::Model(e.to_string()))?;
        let outputs = self
            .session
            .run(ort::inputs!["image" => image_t, "mask" => mask_t])
            .map_err(|e| HealError::Model(e.to_string()))?;
        let out = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| HealError::Model(e.to_string()))?;
        let mut result = [
            vec![0f32; width * height],
            vec![0f32; width * height],
            vec![0f32; width * height],
        ];

        // Extract and unscale output if using fixed-size contract.
        let divisor = match self.contract {
            Contract::Fixed(_) => 255.0,
            Contract::Dynamic => 1.0,
        };

        for (c, plane) in result.iter_mut().enumerate() {
            for y in 0..height {
                for x in 0..width {
                    plane[y * width + x] = out[[0, c, y, x]] / divisor;
                }
            }
        }
        Ok(result)
    }
}
