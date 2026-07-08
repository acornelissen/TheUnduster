//! Mask dilation. Detector masks cover a defect's confident core, not its
//! full visible extent; healing an under-covering mask leaves a visible rim
//! around the fill. Dilating by a couple of pixels before healing swallows
//! the rim (see the 3b-1 field notes).

/// Chebyshev (square-window) dilation: an output pixel is set when any set
/// input pixel lies within `radius` in both axes. Separable two-pass
/// (rows then columns), so cost is O(pixels * radius), not radius squared.
pub fn dilate(mask: &[bool], width: u32, height: u32, radius: u32) -> Vec<bool> {
    if radius == 0 {
        return mask.to_vec();
    }
    let (w, h) = (width as usize, height as usize);
    let r = radius as usize;
    // Pass 1: horizontal max over [x-r, x+r].
    let mut horiz = vec![false; w * h];
    for y in 0..h {
        let row = &mask[y * w..(y + 1) * w];
        let out = &mut horiz[y * w..(y + 1) * w];
        for (x, out_x) in out.iter_mut().enumerate() {
            let lo = x.saturating_sub(r);
            let hi = (x + r + 1).min(w);
            *out_x = row[lo..hi].iter().any(|&b| b);
        }
    }
    // Pass 2: vertical max over [y-r, y+r] of the horizontal pass.
    let mut out = vec![false; w * h];
    for y in 0..h {
        let lo = y.saturating_sub(r);
        let hi = (y + r + 1).min(h);
        for x in 0..w {
            out[y * w + x] = (lo..hi).any(|yy| horiz[yy * w + x]);
        }
    }
    out
}
