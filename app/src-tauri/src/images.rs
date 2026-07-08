use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use fd_tiles::{Pyramid, Tile, TileCache, TileKey, TILE_SIZE};
use serde::Serialize;

const CACHE_BUDGET_BYTES: usize = 512 * 1024 * 1024;

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

/// Extract a single-channel tile from a level's data, mirroring
/// `Pyramid::tile`'s grid/edge-size logic.
fn level_tile(
    width: u32,
    height: u32,
    data: &[u8],
    tx: u32,
    ty: u32,
) -> Option<(u32, u32, Vec<u8>)> {
    let (gx, gy) = (width.div_ceil(TILE_SIZE), height.div_ceil(TILE_SIZE));
    if tx >= gx || ty >= gy {
        return None;
    }
    let x0 = tx * TILE_SIZE;
    let y0 = ty * TILE_SIZE;
    let w = (width - x0).min(TILE_SIZE);
    let h = (height - y0).min(TILE_SIZE);
    let mut out = vec![0u8; (w * h) as usize];
    for row in 0..h {
        let src = ((y0 + row) * width + x0) as usize;
        let dst = (row * w) as usize;
        out[dst..dst + w as usize].copy_from_slice(&data[src..src + w as usize]);
    }
    Some((w, h, out))
}

#[derive(Serialize, Clone, Copy, Debug)]
pub struct LevelInfo {
    pub width: u32,
    pub height: u32,
}

#[derive(Serialize, Clone, Debug)]
pub struct ImageInfo {
    pub id: u64,
    pub width: u32,
    pub height: u32,
    pub levels: Vec<LevelInfo>,
}

struct Entry {
    image: Arc<fd_io::ImageBuf>,
    pyramid: Pyramid,
    /// Native-res f32 probabilities plus their max-pooled display pyramid.
    /// None until `detect` has run for this image.
    probs: Option<(Vec<f32>, ProbPyramid)>,
}

/// Heavy half of open: decoded image plus its built pyramid, no registry access yet.
pub struct Prepared {
    pub(crate) image: Arc<fd_io::ImageBuf>,
    pub(crate) pyramid: Pyramid,
}

pub struct Images {
    next_id: u64,
    entries: HashMap<u64, Entry>,
    cache: TileCache,
}

impl Default for Images {
    fn default() -> Self {
        Images {
            next_id: 1,
            entries: HashMap::new(),
            cache: TileCache::new(CACHE_BUDGET_BYTES),
        }
    }
}

impl Images {
    /// First heavy half of open: decode only. Blocking.
    pub fn decode_stage(path: &Path) -> Result<Arc<fd_io::ImageBuf>, String> {
        Ok(Arc::new(fd_io::decode(path).map_err(|e| e.to_string())?))
    }

    /// Second heavy half of open: pyramid build. Blocking.
    pub fn pyramid_stage(image: &Arc<fd_io::ImageBuf>) -> Pyramid {
        Pyramid::build(image)
    }

    /// Heavy half of open: decode + pyramid, no registry access. Blocking.
    pub fn prepare(path: &Path) -> Result<Prepared, String> {
        let image = Self::decode_stage(path)?;
        let pyramid = Self::pyramid_stage(&image);
        Ok(Prepared { image, pyramid })
    }

    /// Cheap half: registry insert under the caller's lock.
    pub fn insert(&mut self, prepared: Prepared) -> ImageInfo {
        let id = self.next_id;
        self.next_id += 1;
        let levels = prepared
            .pyramid
            .levels
            .iter()
            .map(|l| LevelInfo {
                width: l.width,
                height: l.height,
            })
            .collect();
        let info = ImageInfo {
            id,
            width: prepared.image.width,
            height: prepared.image.height,
            levels,
        };
        self.entries.insert(
            id,
            Entry {
                image: prepared.image,
                pyramid: prepared.pyramid,
                probs: None,
            },
        );
        info
    }

    // Exercised by the 3a test suite; the command path now goes through
    // `prepare` + `insert` directly.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn open(&mut self, path: &Path) -> Result<ImageInfo, String> {
        let prepared = Self::prepare(path)?;
        Ok(self.insert(prepared))
    }

    pub fn image(&self, id: u64) -> Option<Arc<fd_io::ImageBuf>> {
        self.entries.get(&id).map(|e| e.image.clone())
    }

    pub fn tile(&mut self, id: u64, level: u8, tx: u32, ty: u32) -> Option<Arc<Tile>> {
        let key = TileKey {
            image_id: id,
            level,
            tx,
            ty,
        };
        let entry = self.entries.get(&id)?;
        let pyramid = &entry.pyramid;
        self.cache
            .get_or_insert(key, || pyramid.tile(level, tx, ty))
    }

    /// Store native-res probabilities from detection, building the
    /// max-pooled display pyramid to match this entry's tile levels.
    /// Returns false if `id` is unknown (e.g. the image was closed while
    /// inference ran).
    pub fn set_probs(&mut self, id: u64, probs: Vec<f32>) -> bool {
        let Some(entry) = self.entries.get_mut(&id) else {
            return false;
        };
        if probs.len() != (entry.image.width * entry.image.height) as usize {
            return false;
        }
        let level_dims: Vec<(u32, u32)> = entry
            .pyramid
            .levels
            .iter()
            .map(|l| (l.width, l.height))
            .collect();
        let pyramid = build_prob_pyramid(&probs, &level_dims);
        entry.probs = Some((probs, pyramid));
        true
    }

    /// Single-channel probability tile, edge-sized like RGBA tiles.
    pub fn prob_tile(
        &mut self,
        id: u64,
        level: u8,
        tx: u32,
        ty: u32,
    ) -> Option<(u32, u32, Vec<u8>)> {
        let entry = self.entries.get(&id)?;
        let (_, pyramid) = entry.probs.as_ref()?;
        let l = pyramid.levels.get(level as usize)?;
        level_tile(l.width, l.height, &l.data, tx, ty)
    }

    /// Connected-component bounding boxes from the native-res thresholded
    /// probability map.
    pub fn components(&self, id: u64, threshold: f32) -> Option<Vec<[u32; 4]>> {
        let entry = self.entries.get(&id)?;
        let (probs, _) = entry.probs.as_ref()?;
        let mask: Vec<bool> = probs.iter().map(|&p| p > threshold).collect();
        let (w, h) = (entry.image.width, entry.image.height);
        Some(
            fd_heal::components(&mask, w, h)
                .into_iter()
                .map(|d| [d.bbox.x0, d.bbox.y0, d.bbox.x1, d.bbox.y1])
                .collect(),
        )
    }

    pub fn close(&mut self, id: u64) {
        self.entries.remove(&id);
        self.cache.evict_image(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fd_io::{encode, ImageBuf, PixelData};

    fn temp_png(dir: &tempfile::TempDir, w: u32, h: u32) -> std::path::PathBuf {
        let path = dir.path().join("t.png");
        let n = (w * h) as usize;
        let img = ImageBuf {
            width: w,
            height: h,
            channels: 1,
            data: PixelData::U8((0..n).map(|i| (i % 251) as u8).collect()),
            icc: None,
            exif: None,
        };
        encode(&path, &img).unwrap();
        path
    }

    #[test]
    fn open_reports_dims_and_levels() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 1100, 600);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        assert_eq!((info.width, info.height), (1100, 600));
        assert_eq!(info.levels.len(), 3); // 1100x600 -> 550x300 -> 275x150
        assert_eq!(info.id, 1);
    }

    #[test]
    fn tile_is_cached_and_edge_sized() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 1100, 600);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        let t = images.tile(info.id, 0, 2, 1).unwrap();
        assert_eq!((t.width, t.height), (1100 - 1024, 600 - 512));
        assert!(images.tile(info.id, 0, 9, 0).is_none());
        assert!(images.tile(999, 0, 0, 0).is_none());
    }

    #[test]
    fn open_missing_file_is_an_error_naming_it() {
        let mut images = Images::default();
        let err = images.open(Path::new("/nonexistent/x.png")).unwrap_err();
        assert!(err.contains("x.png"));
    }

    #[test]
    fn image_pixels_are_retained_for_inference() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 600, 400);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        let img = images.image(info.id).expect("pixels retained");
        assert_eq!((img.width, img.height), (600, 400));
        images.close(info.id);
        assert!(images.image(info.id).is_none());
    }

    #[test]
    fn close_evicts() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 600, 400);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        assert!(images.tile(info.id, 0, 0, 0).is_some());
        images.close(info.id);
        assert!(images.tile(info.id, 0, 0, 0).is_none());
    }

    #[test]
    fn prob_pyramid_max_pools_and_matches_level_dims() {
        // 4x4 probs with one hot pixel; two levels (4x4, 2x2)
        let mut probs = vec![0.0f32; 16];
        probs[5] = 0.9; // (1,1)
        let p = build_prob_pyramid(&probs, &[(4, 4), (2, 2)]);
        assert_eq!(p.levels.len(), 2);
        assert_eq!(p.levels[0].data[5], (0.9f32 * 255.0 + 0.5) as u8);
        // max-pool keeps the speck at (0,0) of the 2x2 level
        assert_eq!(p.levels[1].data[0], p.levels[0].data[5]);
        assert_eq!(p.levels[1].data[3], 0);
    }

    #[test]
    fn prob_tiles_and_components_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 600, 400);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        let mut probs = vec![0.0f32; 600 * 400];
        for y in 100..104 {
            for x in 200..205 {
                probs[y * 600 + x] = 0.8;
            }
        }
        images.set_probs(info.id, probs.clone());
        let (w, h, bytes) = images.prob_tile(info.id, 0, 0, 0).unwrap();
        assert_eq!((w, h), (512, 400));
        assert_eq!(bytes.len(), (512 * 400) as usize);
        assert!(bytes[100 * 512 + 200] > 200);
        let comps = images.components(info.id, 0.5).unwrap();
        assert_eq!(comps.len(), 1);
        let b = comps[0];
        assert_eq!((b[0], b[1], b[2], b[3]), (200, 100, 205, 104));
        assert!(images.prob_tile(999, 0, 0, 0).is_none());
        assert!(images.components(info.id, 0.9).unwrap().is_empty());
    }

    #[test]
    fn set_probs_rejects_wrong_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 600, 400);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        assert!(!images.set_probs(info.id, vec![0.0; 10]));
        assert!(images.components(info.id, 0.5).is_none()); // nothing stored
    }
}
