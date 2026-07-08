use fd_tiles::{Tile, TileCache, TileKey, TILE_SIZE};

fn tile_of_bytes(n: u32) -> Tile {
    // width n/4 x height 1 rgba => exactly n bytes of pixel data
    Tile {
        width: n / 4,
        height: 1,
        rgba: vec![0; n as usize],
    }
}

fn key(image: u64, tx: u32) -> TileKey {
    TileKey {
        image_id: image,
        level: 0,
        tx,
        ty: 0,
    }
}

#[test]
fn caches_and_returns_same_tile() {
    let mut c = TileCache::new(10_000);
    let mut builds = 0;
    for _ in 0..3 {
        let t = c
            .get_or_insert(key(1, 0), || {
                builds += 1;
                Some(tile_of_bytes(1000))
            })
            .unwrap();
        assert_eq!(t.rgba.len(), 1000);
    }
    assert_eq!(builds, 1);
    assert_eq!(c.bytes_used(), 1000);
}

#[test]
fn evicts_least_recently_used_beyond_budget() {
    let mut c = TileCache::new(2500);
    c.get_or_insert(key(1, 0), || Some(tile_of_bytes(1000)));
    c.get_or_insert(key(1, 1), || Some(tile_of_bytes(1000)));
    // touch tile 0 so tile 1 is LRU
    c.get_or_insert(key(1, 0), || panic!("should be cached"));
    c.get_or_insert(key(1, 2), || Some(tile_of_bytes(1000)));
    assert!(c.bytes_used() <= 2500);
    let mut rebuilt = false;
    c.get_or_insert(key(1, 1), || {
        rebuilt = true;
        Some(tile_of_bytes(1000))
    });
    assert!(rebuilt, "LRU tile 1 should have been evicted");
}

#[test]
fn build_failure_is_not_cached() {
    let mut c = TileCache::new(1000);
    assert!(c.get_or_insert(key(1, 9), || None).is_none());
    assert_eq!(c.bytes_used(), 0);
}

#[test]
fn evict_image_drops_only_that_image() {
    let mut c = TileCache::new(100_000);
    c.get_or_insert(key(1, 0), || Some(tile_of_bytes(1000)));
    c.get_or_insert(key(2, 0), || Some(tile_of_bytes(1000)));
    c.evict_image(1);
    assert_eq!(c.bytes_used(), 1000);
    let _ = TILE_SIZE; // keep the import honest
}

#[test]
fn oversized_tile_still_served_but_not_retained_forever() {
    let mut c = TileCache::new(100);
    let t = c.get_or_insert(key(1, 0), || Some(tile_of_bytes(1000)));
    assert!(t.is_some());
    assert!(c.bytes_used() <= 1000); // inserted, then budget enforcement may evict it next insert
    c.get_or_insert(key(1, 1), || Some(tile_of_bytes(40)));
    assert!(c.bytes_used() <= 140);
}
