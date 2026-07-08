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
    for plane in planes.iter_mut() {
        let sigma = ring_sigma(plane, width, height, &defect.bbox, mask);
        if sigma <= 0.0 {
            continue;
        }
        for &(x, y) in &defect.pixels {
            let idx = y as usize * width + x as usize;
            plane[idx] = (plane[idx] + rng.next_gauss() * sigma).clamp(0.0, 1.0);
        }
    }
}
