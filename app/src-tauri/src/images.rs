use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use fd_tiles::{Pyramid, Tile, TileCache, TileKey};
use serde::Serialize;

const CACHE_BUDGET_BYTES: usize = 512 * 1024 * 1024;

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
    pyramid: Pyramid,
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
    pub fn open(&mut self, path: &Path) -> Result<ImageInfo, String> {
        let img = fd_io::decode(path).map_err(|e| e.to_string())?;
        let pyramid = Pyramid::build(&img);
        let id = self.next_id;
        self.next_id += 1;
        let levels = pyramid
            .levels
            .iter()
            .map(|l| LevelInfo {
                width: l.width,
                height: l.height,
            })
            .collect();
        let info = ImageInfo {
            id,
            width: img.width,
            height: img.height,
            levels,
        };
        self.entries.insert(id, Entry { pyramid });
        Ok(info)
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
    fn close_evicts() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 600, 400);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        assert!(images.tile(info.id, 0, 0, 0).is_some());
        images.close(info.id);
        assert!(images.tile(info.id, 0, 0, 0).is_none());
    }
}
