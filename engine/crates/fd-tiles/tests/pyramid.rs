use fd_io::{ImageBuf, PixelData};
use fd_tiles::{Pyramid, TILE_SIZE};

fn gray_image(width: u32, height: u32) -> ImageBuf {
    let n = (width * height) as usize;
    ImageBuf {
        width,
        height,
        channels: 1,
        data: PixelData::U16((0..n).map(|i| ((i * 65535) / n) as u16).collect()),
        icc: None,
        exif: None,
    }
}

#[test]
fn levels_halve_until_one_tile() {
    let p = Pyramid::build(&gray_image(2000, 1200));
    // 2000x1200 -> 1000x600 -> 500x300 (fits in one 512 tile)
    assert_eq!(p.levels.len(), 3);
    assert_eq!((p.levels[0].width, p.levels[0].height), (2000, 1200));
    assert_eq!((p.levels[1].width, p.levels[1].height), (1000, 600));
    assert_eq!((p.levels[2].width, p.levels[2].height), (500, 300));
}

#[test]
fn small_image_is_single_level() {
    let p = Pyramid::build(&gray_image(300, 200));
    assert_eq!(p.levels.len(), 1);
}

#[test]
fn tile_grid_and_edge_tiles() {
    let p = Pyramid::build(&gray_image(1100, 600));
    let (tx, ty) = p.tiles_at(0);
    assert_eq!((tx, ty), (3, 2)); // ceil(1100/512)=3, ceil(600/512)=2
    let full = p.tile(0, 0, 0).unwrap();
    assert_eq!((full.width, full.height), (TILE_SIZE, TILE_SIZE));
    let edge = p.tile(0, 2, 1).unwrap();
    assert_eq!((edge.width, edge.height), (1100 - 1024, 600 - 512));
    assert_eq!(edge.rgba.len(), (edge.width * edge.height * 4) as usize);
    assert!(p.tile(0, 3, 0).is_none());
    assert!(p.tile(9, 0, 0).is_none());
}

#[test]
fn rgba_is_opaque_and_downsample_averages() {
    // 2x2 image with values 0,0,65535,65535 -> level1 single pixel ~ mid gray
    let img = ImageBuf {
        width: 2,
        height: 2,
        channels: 1,
        data: PixelData::U16(vec![0, 0, 65535, 65535]),
        icc: None,
        exif: None,
    };
    let p = Pyramid::build(&img);
    let l0 = &p.levels[0];
    assert_eq!(l0.rgba[3], 255); // alpha opaque
                                 // level 0 exists only (2x2 fits one tile), so test averaging directly:
    let avg = fd_tiles::downsample_2x(&l0.rgba, 2, 2);
    // one output pixel, gray channels averaged: (0+0+255+255)/4 = 127 or 128
    assert!((avg.0[0] as i32 - 127).abs() <= 1);
    assert_eq!((avg.1, avg.2), (1, 1));
}
