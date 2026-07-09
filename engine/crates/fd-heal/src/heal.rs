use fd_io::{ImageBuf, PixelData};

use crate::{add_grain, classical_fill, components, Defect, HealError, Inpainter};

pub const TINY_MAX_DIM: u32 = 5;

#[derive(Debug, Default)]
pub struct HealReport {
    pub defects: usize,
    pub tiny: usize,
    pub inpainted: usize,
}

fn to_planes(img: &ImageBuf) -> Vec<Vec<f32>> {
    let f = img.to_f32();
    let n = (img.width * img.height) as usize;
    let ch = img.channels as usize;
    let mut planes = vec![vec![0f32; n]; ch];
    for i in 0..n {
        for (c, plane) in planes.iter_mut().enumerate() {
            plane[i] = f[i * ch + c];
        }
    }
    planes
}

/// Write healed values back into native depth, ONLY at masked pixels.
/// Everything else in img.data is untouched, which makes the bit-exactness
/// guarantee structural rather than aspirational.
fn write_back(img: &mut ImageBuf, planes: &[Vec<f32>], mask: &[bool]) {
    let ch = img.channels as usize;
    match &mut img.data {
        PixelData::U8(v) => {
            for (i, &m) in mask.iter().enumerate() {
                if m {
                    for c in 0..ch {
                        v[i * ch + c] = (planes[c][i] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
        PixelData::U16(v) => {
            for (i, &m) in mask.iter().enumerate() {
                if m {
                    for c in 0..ch {
                        v[i * ch + c] = (planes[c][i] * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
                    }
                }
            }
        }
    }
}

fn inpaint_defect(
    planes: &mut [Vec<f32>],
    width: usize,
    height: usize,
    defect: &Defect,
    mask: &[bool],
    inpainter: &mut Inpainter,
) -> Result<(), HealError> {
    // Crop: bbox padded by 2x its max dim (min 24), clamped, multiple of 8.
    let pad = (defect.max_dim() * 2).max(24) as usize;
    let cx0 = (defect.bbox.x0 as usize).saturating_sub(pad);
    let cy0 = (defect.bbox.y0 as usize).saturating_sub(pad);
    let cx1 = (defect.bbox.x1 as usize + pad).min(width);
    let cy1 = (defect.bbox.y1 as usize + pad).min(height);
    // Round the crop span UP to a multiple of 8 so the crop contains the
    // padded bbox whenever the image allows it (rounding DOWN shipped a
    // field panic: an edge-hugging defect stuck out past the shrunken crop
    // and the write-back below indexed beyond the crop buffer). Capped at
    // the largest multiple of 8 that fits the image; when even that cannot
    // cover the defect (a full-span defect on a non-multiple-of-8 image),
    // the write-back's bounds guard degrades gracefully instead.
    let round8 = |a: usize, b: usize, limit: usize| -> (usize, usize) {
        let mut lo = a;
        let mut hi = b;
        let wanted = (hi - lo).max(8).div_ceil(8) * 8;
        let dim = wanted.min(((limit / 8) * 8).max(8.min(limit)));
        if lo + dim <= limit {
            hi = lo + dim;
        } else {
            hi = limit;
            lo = hi.saturating_sub(dim);
        }
        (lo, hi)
    };
    let (cx0, cx1) = round8(cx0, cx1, width);
    let (cy0, cy1) = round8(cy0, cy1, height);
    let (cw, chh) = (cx1 - cx0, cy1 - cy0);

    // Assemble RGB crop planes (grey images: replicate the single plane).
    let get_plane = |c: usize| -> Vec<f32> {
        let src = if planes.len() == 1 {
            &planes[0]
        } else {
            &planes[c]
        };
        let mut out = vec![0f32; cw * chh];
        for y in 0..chh {
            for x in 0..cw {
                out[y * cw + x] = src[(cy0 + y) * width + (cx0 + x)];
            }
        }
        out
    };
    let crop = [get_plane(0), get_plane(1), get_plane(2)];
    let mut crop_mask = vec![false; cw * chh];
    for y in 0..chh {
        for x in 0..cw {
            crop_mask[y * cw + x] = mask[(cy0 + y) * width + (cx0 + x)];
        }
    }
    let filled = inpainter.inpaint(&crop, &crop_mask, cw, chh)?;
    // Write filled values back into working planes at THIS defect's pixels
    // only. Pixels the crop could not cover (a full-span defect on an image
    // whose dimension is not a multiple of 8) stay untouched: an unhealed
    // sliver at the frame edge beats a panic or a wrapped read from the
    // wrong crop row.
    for &(px, py) in &defect.pixels {
        let (px, py) = (px as usize, py as usize);
        if px < cx0 || px >= cx1 || py < cy0 || py >= cy1 {
            continue;
        }
        let (lx, ly) = (px - cx0, py - cy0);
        for (c, plane) in planes.iter_mut().enumerate() {
            let src_c = if c < 3 { c } else { 0 };
            plane[py * width + px] = filled[src_c][ly * cw + lx];
        }
    }
    Ok(())
}

/// Fixed-size windowed inpainting: window interiors tile the defect bbox,
/// margins supply context, and each window writes back only its interior
/// defect pixels -- every masked pixel is filled exactly once, with no
/// resampling, so grain scale and sharpness survive. Windows shift (never
/// shrink) to stay inside the image; when the image itself is smaller than
/// the window, edge-replicate padding fills the remainder, mirroring the
/// detector's tile padding.
fn inpaint_defect_windowed(
    planes: &mut [Vec<f32>],
    width: usize,
    height: usize,
    defect: &Defect,
    mask: &[bool],
    inpainter: &mut Inpainter,
    n: usize,
) -> Result<(), HealError> {
    let margin = n / 8;
    let interior = n - 2 * margin;
    let (bx0, by0) = (defect.bbox.x0 as usize, defect.bbox.y0 as usize);
    let (bx1, by1) = (defect.bbox.x1 as usize, defect.bbox.y1 as usize); // exclusive

    let mut iy = by0;
    while iy < by1 {
        let iy1 = (iy + interior).min(by1);
        let mut ix = bx0;
        while ix < bx1 {
            let ix1 = (ix + interior).min(bx1);
            // Window start: interior minus margin, shifted into the image
            // when possible (image >= n), otherwise anchored at 0 with
            // edge-replicate covering the overhang.
            let wx0 = window_start(ix, margin, n, width);
            let wy0 = window_start(iy, margin, n, height);

            // Assemble the nxn crop with edge-replicate for out-of-image.
            let clamp_src = |v: isize, limit: usize| v.clamp(0, limit as isize - 1) as usize;
            let mut crop = [vec![0f32; n * n], vec![0f32; n * n], vec![0f32; n * n]];
            let mut crop_mask = vec![false; n * n];
            for y in 0..n {
                let sy = clamp_src(wy0 as isize + y as isize, height);
                for x in 0..n {
                    let sx = clamp_src(wx0 as isize + x as isize, width);
                    for (c, plane) in crop.iter_mut().enumerate() {
                        let src = if planes.len() == 1 {
                            &planes[0]
                        } else {
                            &planes[c]
                        };
                        plane[y * n + x] = src[sy * width + sx];
                    }
                    crop_mask[y * n + x] = mask[sy * width + sx];
                }
            }
            let filled = inpainter.inpaint(&crop, &crop_mask, n, n)?;

            // Write back: defect pixels inside THIS window's interior only.
            for &(px, py) in &defect.pixels {
                let (px, py) = (px as usize, py as usize);
                if px < ix || px >= ix1 || py < iy || py >= iy1 {
                    continue;
                }
                let (lx, ly) = (px - wx0, py - wy0);
                if lx >= n || ly >= n {
                    continue; // interior clamped past the window; unreachable when image >= n
                }
                for (c, plane) in planes.iter_mut().enumerate() {
                    let src_c = if c < 3 { c } else { 0 };
                    plane[py * width + px] = filled[src_c][ly * n + lx];
                }
            }
            ix = ix1;
        }
        iy = iy1;
    }
    Ok(())
}

/// Ideal window start is interior_start - margin; shift left/up to keep the
/// whole window inside the image when it fits, clamp to 0 when it does not
/// (edge-replicate covers the rest).
fn window_start(interior_start: usize, margin: usize, n: usize, limit: usize) -> usize {
    let ideal = interior_start.saturating_sub(margin);
    if ideal + n <= limit {
        ideal
    } else {
        limit.saturating_sub(n)
    }
}

pub fn heal(
    img: &mut ImageBuf,
    mask: &[bool],
    mut inpainter: Option<&mut Inpainter>,
) -> Result<HealReport, HealError> {
    let (width, height) = (img.width as usize, img.height as usize);
    if mask.len() != width * height {
        return Err(HealError::MaskSize {
            got: mask.len(),
            want: width * height,
        });
    }
    let defects = components(mask, img.width, img.height);
    let mut planes = to_planes(img);
    let mut report = HealReport {
        defects: defects.len(),
        ..Default::default()
    };
    for d in &defects {
        match inpainter.as_deref_mut() {
            Some(inp) if d.max_dim() > TINY_MAX_DIM => {
                match inp.window_size() {
                    Some(n) => {
                        inpaint_defect_windowed(&mut planes, width, height, d, mask, inp, n)?
                    }
                    None => inpaint_defect(&mut planes, width, height, d, mask, inp)?,
                }
                add_grain(&mut planes, width, height, d, mask);
                report.inpainted += 1;
            }
            _ => {
                classical_fill(&mut planes, img.width, img.height, d, mask);
                report.tiny += 1;
            }
        }
    }
    write_back(img, &planes, mask);
    Ok(report)
}
