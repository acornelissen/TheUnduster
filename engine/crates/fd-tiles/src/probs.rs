//! Probability pyramids for the detection overlay: quantized to u8 at the
//! base and max-pooled down the display pyramid's level dims. Lives in this
//! crate (not the app) so the 168-megapixel base pass compiles optimized in
//! dev builds -- at the app crate's opt-level 0 it cost ~5s per build and
//! sat on the frame-activation path (field report).

pub struct ProbLevel {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

pub struct ProbPyramid {
    pub levels: Vec<ProbLevel>,
}

/// Quantize native-res probabilities and max-pool down the given level dims
/// (which must match the display pyramid). Max, not mean: a 3px dust speck
/// must stay visible when zoomed out.
pub fn build_prob_pyramid(probs: &[f32], level_dims: &[(u32, u32)]) -> ProbPyramid {
    let (w0, h0) = level_dims[0];
    let mut levels = Vec::with_capacity(level_dims.len());
    let base: Vec<u8> = probs
        .iter()
        .map(|p| (p.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect();
    levels.push(ProbLevel {
        width: w0,
        height: h0,
        data: base,
    });
    for &(w, h) in &level_dims[1..] {
        let prev = levels.last().unwrap();
        let mut data = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let mut m = 0u8;
                for dy in 0..2u32 {
                    for dx in 0..2u32 {
                        let (sx, sy) = (x * 2 + dx, y * 2 + dy);
                        if sx < prev.width && sy < prev.height {
                            m = m.max(prev.data[(sy * prev.width + sx) as usize]);
                        }
                    }
                }
                data[(y * w + x) as usize] = m;
            }
        }
        levels.push(ProbLevel {
            width: w,
            height: h,
            data,
        });
    }
    ProbPyramid { levels }
}
