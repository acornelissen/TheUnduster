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

/// The one quantization rule for detection probabilities, shared by every
/// caller that turns an f32 probability into the u8 the registry, the disk
/// cache, and the display pyramid all store: clamp to `[0, 1]` (guards any
/// out-of-range detector output -- NaN passes through `f32::clamp` unchanged,
/// since `clamp` propagates NaN rather than treating it as a bound; the
/// saturating `as u8` cast below is what actually turns it into 0), scale to
/// `[0, 255]`, round to the nearest integer, then cast to `u8`. Identical to
/// the codebase's older `+ 0.5` truncation rule for the cache's on-disk
/// bytes; the two rules differ at exactly one pathological f32 value
/// (p ~= 0.0019607842), and even there the shader's `step(0.004, p)`
/// suppresses any visible effect.
pub fn quantize_prob(p: f32) -> u8 {
    (p.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Quantize native-res probabilities and max-pool down the given level dims
/// (which must match the display pyramid). Max, not mean: a 3px dust speck
/// must stay visible when zoomed out.
pub fn build_prob_pyramid(probs: &[f32], level_dims: &[(u32, u32)]) -> ProbPyramid {
    let quantized: Vec<u8> = probs.iter().map(|&p| quantize_prob(p)).collect();
    build_prob_pyramid_u8(&quantized, level_dims)
}

/// Same as [`build_prob_pyramid`] but for probabilities already quantized to
/// u8 -- the registry restore path (disk cache is already u8) and any fresh
/// detection that quantized once at its own call boundary both use this to
/// avoid a second, redundant quantize pass.
pub fn build_prob_pyramid_u8(probs: &[u8], level_dims: &[(u32, u32)]) -> ProbPyramid {
    let (w0, h0) = level_dims[0];
    let mut levels = Vec::with_capacity(level_dims.len());
    levels.push(ProbLevel {
        width: w0,
        height: h0,
        data: probs.to_vec(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_prob_matches_hand_computed_values() {
        assert_eq!(quantize_prob(0.0), 0);
        assert_eq!(quantize_prob(1.0), 255);
        assert_eq!(quantize_prob(0.5), 128); // 127.5 rounds away from zero
        assert_eq!(quantize_prob(0.9), 230); // 229.5 rounds away from zero

        // Out-of-range inputs clamp instead of wrapping or panicking.
        assert_eq!(quantize_prob(-1.0), 0);
        assert_eq!(quantize_prob(2.0), 255);
        // f32::clamp propagates NaN rather than clamping it to a bound; the
        // saturating `as u8` cast is what turns the resulting NaN into 0.
        assert_eq!(quantize_prob(f32::NAN), 0);
    }

    #[test]
    fn quantize_prob_boundary_quanta_around_a_threshold() {
        // Threshold t = 0.5 quantizes to qt = 128. Probabilities exactly at,
        // just above, and just below the threshold quantum's f32 value
        // (128/255) must land on the expected side of `q > qt`.
        let qt = quantize_prob(0.5);
        assert_eq!(qt, 128);

        let at = 128.0 / 255.0;
        let just_above = at + 0.001;
        let just_below = at - 0.001;

        assert_eq!(quantize_prob(at), 128);
        assert!(quantize_prob(just_above) >= qt); // may still quantize to 128
        assert!(quantize_prob(just_below) <= qt);
        // A value a full quantum above must strictly exceed qt.
        assert!(quantize_prob(at + 1.0 / 255.0) > qt);
        // A value a full quantum below must strictly fall below qt.
        assert!(quantize_prob(at - 1.0 / 255.0) < qt);
    }

    #[test]
    fn build_prob_pyramid_u8_skips_requantizing() {
        // Feeding already-quantized u8 through the u8 entry point must not
        // alter the values -- this is the no-second-quantize-pass guarantee
        // the registry restore path and fresh-detect boundary both rely on.
        let probs: Vec<u8> = vec![0, 1, 254, 255];
        let p = build_prob_pyramid_u8(&probs, &[(4, 1)]);
        assert_eq!(p.levels[0].data, probs);
    }

    #[test]
    fn build_prob_pyramid_matches_build_prob_pyramid_u8_after_quantizing() {
        let mut probs = vec![0.0f32; 16];
        probs[5] = 0.9;
        let from_f32 = build_prob_pyramid(&probs, &[(4, 4), (2, 2)]);
        let quantized: Vec<u8> = probs.iter().map(|&p| quantize_prob(p)).collect();
        let from_u8 = build_prob_pyramid_u8(&quantized, &[(4, 4), (2, 2)]);
        assert_eq!(from_f32.levels[0].data, from_u8.levels[0].data);
        assert_eq!(from_f32.levels[1].data, from_u8.levels[1].data);
    }
}
