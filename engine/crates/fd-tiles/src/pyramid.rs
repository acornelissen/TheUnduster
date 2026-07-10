use fd_io::{ImageBuf, PixelData};

pub const TILE_SIZE: u32 = 512;

#[derive(Clone)]
pub struct Level {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

pub struct Tile {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Cloning this is a real memcpy of every level's RGBA buffer (~336MB for a
/// full pyramid) -- only done for the display-pyramid disk cache's
/// fire-and-forget background write, where the alternative (encoding on the
/// activation path, or blocking activation on the write) is worse. See
/// `decode_and_insert` in the app crate.
#[derive(Clone)]
pub struct Pyramid {
    pub levels: Vec<Level>,
}

fn base_rgba(img: &ImageBuf) -> Vec<u8> {
    let n = (img.width * img.height) as usize;
    let mut out = vec![255u8; n * 4];
    let ch = img.channels as usize;
    let write = |out: &mut [u8], i: usize, r: u8, g: u8, b: u8| {
        out[i * 4] = r;
        out[i * 4 + 1] = g;
        out[i * 4 + 2] = b;
    };
    match &img.data {
        PixelData::U8(v) => {
            for i in 0..n {
                if ch == 1 {
                    write(&mut out, i, v[i], v[i], v[i]);
                } else {
                    write(&mut out, i, v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
                }
            }
        }
        PixelData::U16(v) => {
            let g = |x: u16| (x >> 8) as u8;
            for i in 0..n {
                if ch == 1 {
                    write(&mut out, i, g(v[i]), g(v[i]), g(v[i]));
                } else {
                    write(&mut out, i, g(v[i * 3]), g(v[i * 3 + 1]), g(v[i * 3 + 2]));
                }
            }
        }
    }
    out
}

/// Output dims of one 2x downsample step. THE halving rule: `Pyramid::build`
/// levels follow exactly this, and `downsample_2x` derives its output size
/// from it, so any dims-only consumer (e.g. the disk cache's header-level
/// shape validation, which must not touch pixels) shares the single source
/// of truth instead of reimplementing the rounding.
pub fn downsample_dims(width: u32, height: u32) -> (u32, u32) {
    (width.div_ceil(2).max(1), height.div_ceil(2).max(1))
}

/// 2x2 box-average downsample of an RGBA buffer. Returns (rgba, w, h).
pub fn downsample_2x(rgba: &[u8], width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let (nw, nh) = downsample_dims(width, height);
    let mut out = vec![255u8; (nw * nh * 4) as usize];
    for oy in 0..nh {
        for ox in 0..nw {
            for c in 0..3usize {
                let mut sum = 0u32;
                let mut cnt = 0u32;
                for dy in 0..2u32 {
                    for dx in 0..2u32 {
                        let (sx, sy) = (ox * 2 + dx, oy * 2 + dy);
                        if sx < width && sy < height {
                            sum += rgba[((sy * width + sx) * 4) as usize + c] as u32;
                            cnt += 1;
                        }
                    }
                }
                out[((oy * nw + ox) * 4) as usize + c] = (sum / cnt) as u8;
            }
        }
    }
    (out, nw, nh)
}

impl Pyramid {
    pub fn build(img: &ImageBuf) -> Pyramid {
        let mut levels = vec![Level {
            width: img.width,
            height: img.height,
            rgba: base_rgba(img),
        }];
        while levels
            .last()
            .unwrap()
            .width
            .max(levels.last().unwrap().height)
            > TILE_SIZE
        {
            let last = levels.last().unwrap();
            let (rgba, w, h) = downsample_2x(&last.rgba, last.width, last.height);
            levels.push(Level {
                width: w,
                height: h,
                rgba,
            });
        }
        Pyramid { levels }
    }

    pub fn tiles_at(&self, level: u8) -> (u32, u32) {
        let l = &self.levels[level as usize];
        (l.width.div_ceil(TILE_SIZE), l.height.div_ceil(TILE_SIZE))
    }

    pub fn tile(&self, level: u8, tx: u32, ty: u32) -> Option<Tile> {
        let l = self.levels.get(level as usize)?;
        let (gx, gy) = (l.width.div_ceil(TILE_SIZE), l.height.div_ceil(TILE_SIZE));
        if tx >= gx || ty >= gy {
            return None;
        }
        let x0 = tx * TILE_SIZE;
        let y0 = ty * TILE_SIZE;
        let w = (l.width - x0).min(TILE_SIZE);
        let h = (l.height - y0).min(TILE_SIZE);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for row in 0..h {
            let src = (((y0 + row) * l.width + x0) * 4) as usize;
            let dst = (row * w * 4) as usize;
            rgba[dst..dst + (w * 4) as usize].copy_from_slice(&l.rgba[src..src + (w * 4) as usize]);
        }
        Some(Tile {
            width: w,
            height: h,
            rgba,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downsample_dims_matches_downsample_2x_output_dims() {
        // Pin the pure dims rule against the real downsample's returned
        // dims across odd/even/edge sizes, so a change to either that
        // breaks their agreement fails here instead of silently letting
        // dims-only consumers (the disk cache's shape validation) drift.
        let sizes = [1u32, 2, 3, 4, 5, 7, 8, 511, 512, 513, 1000, 1001];
        for &w in &sizes {
            for &h in &sizes {
                let rgba = vec![0u8; (w * h * 4) as usize];
                let (_, real_w, real_h) = downsample_2x(&rgba, w, h);
                assert_eq!(
                    downsample_dims(w, h),
                    (real_w, real_h),
                    "dims rule disagrees with downsample_2x at {w}x{h}"
                );
            }
        }
    }
}
