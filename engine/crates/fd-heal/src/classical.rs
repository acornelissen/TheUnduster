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
