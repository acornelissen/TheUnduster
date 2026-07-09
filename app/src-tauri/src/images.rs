use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use fd_tiles::{Pyramid, Tile, TileCache, TileKey, TILE_SIZE};
use serde::Serialize;

const CACHE_BUDGET_BYTES: usize = 512 * 1024 * 1024;

/// Upper bound on defect bboxes returned to the UI; see [`Images::components`].
pub const MAX_COMPONENTS: usize = 2000;

pub use fd_tiles::{build_prob_pyramid, ProbPyramid};

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
    pub healed: bool,
}

/// A frame's healed pixels: a deep copy of the original with defects filled,
/// its own display pyramid, and the mask that was healed (needed again at
/// export time for the outside-mask verification).
pub struct HealedData {
    pub(crate) image: Arc<fd_io::ImageBuf>,
    pub(crate) pyramid: Pyramid,
    pub(crate) mask: Arc<Vec<bool>>,
}

/// (original, healed, mask) returned by [`Images::healed_parts`].
type HealedParts = (Arc<fd_io::ImageBuf>, Arc<fd_io::ImageBuf>, Arc<Vec<bool>>);

struct Entry {
    image: Arc<fd_io::ImageBuf>,
    pyramid: Pyramid,
    /// Native-res f32 probabilities plus their max-pooled display pyramid.
    /// None until `detect` has run for this image.
    probs: Option<(Vec<f32>, ProbPyramid)>,
    healed: Option<HealedData>,
}

/// Heavy half of open: decoded image plus its built pyramid, no registry access yet.
pub struct Prepared {
    pub(crate) image: Arc<fd_io::ImageBuf>,
    pub(crate) pyramid: Pyramid,
}

/// Connected-component bounding boxes from a thresholded probability map,
/// capped at [`MAX_COMPONENTS`]: a pathological mask (bad model or
/// threshold) can otherwise produce hundreds of thousands of boxes that are
/// useless for navigation and expensive to serialize. Free function so the
/// roll background queue (which never inserts its frame into the `Images`
/// registry -- see `scan_roll`) can compute bboxes without a registry entry.
pub fn components_from_probs(
    probs: &[f32],
    width: u32,
    height: u32,
    threshold: f32,
) -> Vec<[u32; 4]> {
    let mask: Vec<bool> = probs.iter().map(|&p| p > threshold).collect();
    fd_heal::components(&mask, width, height)
        .into_iter()
        .take(MAX_COMPONENTS)
        .map(|d| [d.bbox.x0, d.bbox.y0, d.bbox.x1, d.bbox.y1])
        .collect()
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
            healed: false,
        };
        self.entries.insert(
            id,
            Entry {
                image: prepared.image,
                pyramid: prepared.pyramid,
                probs: None,
                healed: None,
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

    /// Ids currently in the registry; dev-build tile-404 diagnostics only.
    pub fn known_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.entries.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    pub fn image(&self, id: u64) -> Option<Arc<fd_io::ImageBuf>> {
        self.entries.get(&id).map(|e| e.image.clone())
    }

    /// Snapshot of this entry's display-pyramid level dims, for building a
    /// matching prob pyramid outside the lock (see `set_probs_built`).
    pub fn level_dims(&self, id: u64) -> Option<Vec<(u32, u32)>> {
        let entry = self.entries.get(&id)?;
        Some(
            entry
                .pyramid
                .levels
                .iter()
                .map(|l| (l.width, l.height))
                .collect(),
        )
    }

    /// Approximate resident pixel bytes for an activated image: native
    /// pixels + display pyramid RGBA + (after detect) f32 probs and their u8
    /// pyramid. Drives byte-budget eviction in `activate_frame`.
    pub fn retained_bytes(&self, id: u64) -> Option<usize> {
        let entry = self.entries.get(&id)?;
        let px = (entry.image.width as usize) * (entry.image.height as usize);
        let depth = match entry.image.data {
            fd_io::PixelData::U8(_) => 1,
            fd_io::PixelData::U16(_) => 2,
        };
        let mut total = px * entry.image.channels as usize * depth;
        for l in &entry.pyramid.levels {
            total += (l.width as usize) * (l.height as usize) * 4;
        }
        if let Some((probs, pyr)) = &entry.probs {
            total += probs.len() * 4;
            for l in &pyr.levels {
                total += l.data.len();
            }
        }
        if let Some(healed) = &entry.healed {
            let hpx = (healed.image.width as usize) * (healed.image.height as usize);
            total += hpx * healed.image.channels as usize * depth;
            for l in &healed.pyramid.levels {
                total += (l.width as usize) * (l.height as usize) * 4;
            }
            total += healed.mask.len();
        }
        Some(total)
    }

    pub fn tile(&mut self, id: u64, level: u8, tx: u32, ty: u32) -> Option<Arc<Tile>> {
        let key = TileKey {
            image_id: id,
            layer: 0,
            level,
            tx,
            ty,
        };
        let entry = self.entries.get(&id)?;
        let pyramid = &entry.pyramid;
        self.cache
            .get_or_insert(key, || pyramid.tile(level, tx, ty))
    }

    /// Stores a healed copy. Returns false when the id is unknown, the
    /// healed dims/channels/bit-depth do not match the original, or the mask
    /// length is wrong -- mirrors set_probs_built's validate-at-the-boundary
    /// posture.
    pub fn set_healed(
        &mut self,
        id: u64,
        image: Arc<fd_io::ImageBuf>,
        pyramid: Pyramid,
        mask: Arc<Vec<bool>>,
    ) -> bool {
        let Some(entry) = self.entries.get_mut(&id) else {
            return false;
        };
        let same_depth =
            std::mem::discriminant(&image.data) == std::mem::discriminant(&entry.image.data);
        if image.width != entry.image.width
            || image.height != entry.image.height
            || image.channels != entry.image.channels
            || !same_depth
            || mask.len() != (image.width as usize) * (image.height as usize)
        {
            return false;
        }
        entry.healed = Some(HealedData {
            image,
            pyramid,
            mask,
        });
        true
    }

    /// Boolean mask of probs > threshold at native resolution; None until a
    /// detection has stored probabilities for this image.
    pub fn threshold_mask(&self, id: u64, threshold: f32) -> Option<Vec<bool>> {
        let entry = self.entries.get(&id)?;
        let (probs, _) = entry.probs.as_ref()?;
        Some(probs.iter().map(|&p| p > threshold).collect())
    }

    pub fn has_healed(&self, id: u64) -> bool {
        self.entries.get(&id).is_some_and(|e| e.healed.is_some())
    }

    /// True once a detection has stored probabilities for this image. The
    /// job worker's detect-skip check: answering "already detected?" through
    /// `threshold_mask` would clone a native-resolution Vec<bool> just to
    /// discard it.
    pub fn has_probs(&self, id: u64) -> bool {
        self.entries.get(&id).is_some_and(|e| e.probs.is_some())
    }

    /// (original, healed, mask) for export verification and encoding.
    // Exercised by tests; the export path that consumes this wires up in a
    // later task.
    pub fn healed_parts(&self, id: u64) -> Option<HealedParts> {
        let entry = self.entries.get(&id)?;
        let healed = entry.healed.as_ref()?;
        Some((
            entry.image.clone(),
            healed.image.clone(),
            healed.mask.clone(),
        ))
    }

    // Served by the tiles:// protocol for the /healed/{id}/{level}/{tx}/{ty} layer.
    pub fn healed_tile(&mut self, id: u64, level: u8, tx: u32, ty: u32) -> Option<Arc<Tile>> {
        let key = TileKey {
            image_id: id,
            layer: 1,
            level,
            tx,
            ty,
        };
        let entry = self.entries.get(&id)?;
        let pyramid = &entry.healed.as_ref()?.pyramid;
        self.cache
            .get_or_insert(key, || pyramid.tile(level, tx, ty))
    }

    /// Store native-res probabilities from detection, building the
    /// max-pooled display pyramid to match this entry's tile levels.
    /// Returns false if `id` is unknown (e.g. the image was closed while
    /// inference ran).
    ///
    /// This is the slow path: it builds the (already computed elsewhere,
    /// ideally) pyramid under the lock. Kept only so its existing tests stay
    /// green; the `detect` command uses `level_dims` + `set_probs_built`
    /// instead, which builds the pyramid outside the lock.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn set_probs(&mut self, id: u64, probs: Vec<f32>) -> bool {
        let Some(level_dims) = self.level_dims(id) else {
            return false;
        };
        let Some(entry) = self.entries.get(&id) else {
            return false;
        };
        if probs.len() != (entry.image.width * entry.image.height) as usize {
            return false;
        }
        let pyramid = build_prob_pyramid(&probs, &level_dims);
        self.set_probs_built(id, probs, pyramid)
    }

    /// Store already-built native-res probabilities and their display
    /// pyramid (typically built off-lock alongside inference). Validates
    /// `probs.len()` against the entry's native dims and each pyramid level's
    /// dims against the entry's display-pyramid levels; rejects on any
    /// mismatch or unknown id, storing nothing.
    pub fn set_probs_built(&mut self, id: u64, probs: Vec<f32>, pyramid: ProbPyramid) -> bool {
        let Some(entry) = self.entries.get_mut(&id) else {
            return false;
        };
        if probs.len() != (entry.image.width * entry.image.height) as usize {
            return false;
        }
        if pyramid.levels.len() != entry.pyramid.levels.len() {
            return false;
        }
        for (built, expected) in pyramid.levels.iter().zip(entry.pyramid.levels.iter()) {
            if built.width != expected.width || built.height != expected.height {
                return false;
            }
        }
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
    /// probability map, capped at [`MAX_COMPONENTS`]: a pathological mask
    /// (bad model or threshold) can otherwise produce hundreds of thousands
    /// of boxes that are useless for navigation and expensive to serialize.
    pub fn components(&self, id: u64, threshold: f32) -> Option<Vec<[u32; 4]>> {
        let entry = self.entries.get(&id)?;
        let (probs, _) = entry.probs.as_ref()?;
        let (w, h) = (entry.image.width, entry.image.height);
        Some(components_from_probs(probs, w, h, threshold))
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
    fn components_list_is_capped() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 600, 400);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        // 2500 isolated single-pixel defects on an 8px grid
        let mut probs = vec![0.0f32; 600 * 400];
        let mut painted = 0;
        'outer: for gy in 0..50 {
            for gx in 0..50 {
                let (x, y) = (gx * 8 + 4, gy * 8 + 4);
                if x < 600 && y < 400 {
                    probs[y * 600 + x] = 0.9;
                    painted += 1;
                    if painted == 2500 {
                        break 'outer;
                    }
                }
            }
        }
        assert!(painted > MAX_COMPONENTS);
        assert!(images.set_probs(info.id, probs));
        let comps = images.components(info.id, 0.5).unwrap();
        assert_eq!(comps.len(), MAX_COMPONENTS);
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

    #[test]
    fn components_from_probs_matches_the_method_it_replaces() {
        let mut probs = vec![0.0f32; 600 * 400];
        for y in 100..104 {
            for x in 200..205 {
                probs[y * 600 + x] = 0.8;
            }
        }
        let direct = components_from_probs(&probs, 600, 400, 0.5);
        assert_eq!(direct.len(), 1);
        assert_eq!(direct[0], [200, 100, 205, 104]);
        assert!(components_from_probs(&probs, 600, 400, 0.9).is_empty());
    }

    #[test]
    fn healed_storage_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 600, 400);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        assert!(!info.healed);
        assert!(!images.has_healed(info.id));
        assert!(images.healed_tile(info.id, 0, 0, 0).is_none());

        let original = images.image(info.id).unwrap();
        let mut healed_buf = (*original).clone();
        // change one pixel so healed tiles are distinguishable
        if let fd_io::PixelData::U8(v) = &mut healed_buf.data {
            v[0] = 255;
        }
        let healed = std::sync::Arc::new(healed_buf);
        let pyramid = fd_tiles::Pyramid::build(&healed);
        let mask = std::sync::Arc::new(vec![false; 600 * 400]);
        let base_bytes = images.retained_bytes(info.id).unwrap();
        assert!(images.set_healed(info.id, healed.clone(), pyramid, mask.clone()));

        assert!(images.has_healed(info.id));
        let t = images.healed_tile(info.id, 0, 0, 0).unwrap();
        assert_eq!(t.rgba[0], 255); // healed pixels, not original
        let (orig, heal, m) = images.healed_parts(info.id).unwrap();
        assert_eq!(orig.width, heal.width);
        assert_eq!(m.len(), 600 * 400);

        // healed image + pyramid must count against the eviction budget
        assert!(images.retained_bytes(info.id).unwrap() > base_bytes + 600 * 400);

        images.close(info.id);
        assert!(images.healed_tile(info.id, 0, 0, 0).is_none());
        assert!(!images.has_healed(info.id));
    }

    #[test]
    fn set_healed_rejects_mismatched_dims_and_unknown_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 600, 400);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        let wrong = std::sync::Arc::new(fd_io::ImageBuf {
            width: 100,
            height: 100,
            channels: 1,
            data: fd_io::PixelData::U8(vec![0; 100 * 100]),
            icc: None,
            exif: None,
        });
        let pyr = fd_tiles::Pyramid::build(&wrong);
        let mask = std::sync::Arc::new(vec![false; 100 * 100]);
        assert!(!images.set_healed(info.id, wrong.clone(), pyr, mask.clone()));
        let pyr2 = fd_tiles::Pyramid::build(&wrong);
        assert!(!images.set_healed(999, wrong, pyr2, mask));
    }

    #[test]
    fn set_healed_rejects_bit_depth_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 600, 400);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        // Same dims/channels as the original (U8), but U16 pixel data.
        let wrong_depth = std::sync::Arc::new(fd_io::ImageBuf {
            width: 600,
            height: 400,
            channels: 1,
            data: fd_io::PixelData::U16(vec![0; 600 * 400]),
            icc: None,
            exif: None,
        });
        let pyr = fd_tiles::Pyramid::build(&wrong_depth);
        let mask = std::sync::Arc::new(vec![false; 600 * 400]);
        assert!(!images.set_healed(info.id, wrong_depth, pyr, mask));
    }

    #[test]
    fn threshold_mask_matches_probs_above_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 600, 400);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        assert!(images.threshold_mask(info.id, 0.5).is_none());

        let mut probs = vec![0.0f32; 600 * 400];
        for y in 100..104 {
            for x in 200..205 {
                probs[y * 600 + x] = 0.8;
            }
        }
        images.set_probs(info.id, probs);

        let mask = images.threshold_mask(info.id, 0.5).unwrap();
        assert_eq!(mask.iter().filter(|&&b| b).count(), 4 * 5);
        let none = images.threshold_mask(info.id, 0.9).unwrap();
        assert!(none.iter().all(|&b| !b));
    }

    #[test]
    fn set_probs_built_rejects_mismatched_pyramid_level_dims() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 600, 400);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        let level_dims = images.level_dims(info.id).unwrap();
        assert_eq!(level_dims, vec![(600, 400), (300, 200)]);
        let probs = vec![0.0f32; 600 * 400];

        // Pyramid built against the right level count but a corrupted
        // second-level width: disagrees with the entry's real pyramid.
        let mut mismatched = build_prob_pyramid(&probs, &level_dims);
        mismatched.levels[1].width = 299;
        assert!(!images.set_probs_built(info.id, probs.clone(), mismatched));
        assert!(images.components(info.id, 0.5).is_none()); // nothing stored

        // Sanity: a correctly-shaped pyramid is accepted.
        let good = build_prob_pyramid(&probs, &level_dims);
        assert!(images.set_probs_built(info.id, probs.clone(), good));
        assert!(images.components(info.id, 0.5).is_some());

        // Unknown id is rejected too.
        let another = build_prob_pyramid(&probs, &level_dims);
        assert!(!images.set_probs_built(999, probs, another));
    }
}
