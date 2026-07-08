use std::sync::Arc;

use lru::LruCache;

use crate::Tile;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TileKey {
    pub image_id: u64,
    /// Which pixel source this tile came from: 0 = original rgba pyramid,
    /// 1 = healed pyramid. Same image id, different pixels.
    pub layer: u8,
    pub level: u8,
    pub tx: u32,
    pub ty: u32,
}

/// LRU tile cache bounded by total pixel bytes, not entry count.
pub struct TileCache {
    inner: LruCache<TileKey, Arc<Tile>>,
    byte_budget: usize,
    bytes_used: usize,
}

fn tile_bytes(t: &Tile) -> usize {
    t.rgba.len()
}

impl TileCache {
    pub fn new(byte_budget: usize) -> TileCache {
        TileCache {
            inner: LruCache::unbounded(),
            byte_budget: byte_budget.max(1),
            bytes_used: 0,
        }
    }

    pub fn bytes_used(&self) -> usize {
        self.bytes_used
    }

    pub fn get_or_insert(
        &mut self,
        key: TileKey,
        build: impl FnOnce() -> Option<Tile>,
    ) -> Option<Arc<Tile>> {
        if let Some(hit) = self.inner.get(&key) {
            return Some(hit.clone());
        }
        let tile = Arc::new(build()?);
        self.bytes_used += tile_bytes(&tile);
        self.inner.put(key, tile.clone());
        // Evict LRU entries until under budget, but never the one just inserted.
        while self.bytes_used > self.byte_budget && self.inner.len() > 1 {
            if let Some((_, evicted)) = self.inner.pop_lru() {
                self.bytes_used -= tile_bytes(&evicted);
            } else {
                break;
            }
        }
        Some(tile)
    }

    pub fn evict_image(&mut self, image_id: u64) {
        let keys: Vec<TileKey> = self
            .inner
            .iter()
            .map(|(k, _)| *k)
            .filter(|k| k.image_id == image_id)
            .collect();
        for k in keys {
            if let Some(t) = self.inner.pop(&k) {
                self.bytes_used -= tile_bytes(&t);
            }
        }
    }
}
