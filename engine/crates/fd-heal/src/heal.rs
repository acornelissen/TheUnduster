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
    // round crop dims down to a multiple of 8 (>= 8), extending toward origin if needed
    let round8 = |a: usize, b: usize, limit: usize| -> (usize, usize) {
        let mut lo = a;
        let mut hi = b;
        let dim = ((hi - lo).max(8) / 8) * 8;
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
    // Write filled values back into working planes at THIS defect's pixels only.
    for &(px, py) in &defect.pixels {
        let (lx, ly) = (px as usize - cx0, py as usize - cy0);
        for (c, plane) in planes.iter_mut().enumerate() {
            let src_c = if c < 3 { c } else { 0 };
            plane[py as usize * width + px as usize] = filled[src_c][ly * cw + lx];
        }
    }
    Ok(())
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
                inpaint_defect(&mut planes, width, height, d, mask, inp)?;
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
