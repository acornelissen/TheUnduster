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
            .map_err(|e| HealError::Model(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| HealError::Model(e.to_string()))?;
        Ok(Inpainter { session })
    }

    /// image: 3 planes HxW in [0,1]; mask: HxW (true = fill). Returns 3 planes.
    pub fn inpaint(
        &mut self,
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
        for (c, plane) in result.iter_mut().enumerate() {
            for y in 0..height {
                for x in 0..width {
                    plane[y * width + x] = out[[0, c, y, x]];
                }
            }
        }
        Ok(result)
    }
}
