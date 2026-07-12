use std::collections::HashSet;

use crate::Defect;

/// Fill each defect pixel with the median of nearby usable pixels, peeling
/// the defect inward layer by layer: once a pixel is filled it becomes a
/// sample source for pixels deeper in, so arbitrarily large defects fill to
/// the core. The old single-pass fill (expanding window capped at 16px,
/// needing 5 clean samples) left a 64px-radius disk ~half unfilled -- the
/// exact case a release build with no inpainting model hits when exporting
/// brush strokes, which shipped the dust intact in the center.
///
/// Grain-aware in the sense that the median of a grainy neighborhood
/// carries local intensity statistics; deep cores inherit medians of
/// medians, which flattens grain, but a flat core always beats shipping the
/// defect itself. Pixels of OTHER defects (masked but not in `defect`) are
/// never sampled, exactly as before. If no usable pixel is in reach of any
/// remaining pixel (a defect covering the whole image), the loop stops
/// instead of spinning.
pub fn classical_fill(
    planes: &mut [Vec<f32>],
    width: u32,
    height: u32,
    defect: &Defect,
    mask: &[bool],
) {
    let w = width as usize;
    // Two membership views drive the sampling rule below: a masked pixel is
    // usable only when it belongs to THIS defect (`mine`) and has already
    // been filled (left `unfilled`).
    let mine: HashSet<(u32, u32)> = defect.pixels.iter().copied().collect();
    let mut unfilled = mine.clone();

    while !unfilled.is_empty() {
        let mut filled_this_pass = 0usize;
        // Stored pixel order (row-major from the component walk) keeps the
        // fill deterministic; fills land in `planes` immediately, so later
        // pixels in the same pass already sample from them and one sweep
        // typically penetrates many layers.
        for &(px, py) in &defect.pixels {
            if !unfilled.contains(&(px, py)) {
                continue;
            }
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
                            let key = (sx as u32, sy as u32);
                            // Another defect's pixel, or one of ours still
                            // holding defect values: not a source.
                            if !mine.contains(&key) || unfilled.contains(&key) {
                                continue;
                            }
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
                    unfilled.remove(&(px, py));
                    filled_this_pass += 1;
                    break;
                }
            }
        }
        if filled_this_pass == 0 {
            // No usable pixel reachable from anything left (defect covers
            // the image): leave the rest rather than loop forever.
            break;
        }
    }
}
