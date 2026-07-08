use fd_io::{decode, encode, ImageBuf, PixelData};

fn make(width: u32, height: u32, channels: u8, sixteen: bool) -> ImageBuf {
    let n = (width * height) as usize * channels as usize;
    let data = if sixteen {
        PixelData::U16((0..n).map(|i| ((i * 65535) / n) as u16).collect())
    } else {
        PixelData::U8((0..n).map(|i| ((i * 255) / n) as u8).collect())
    };
    ImageBuf {
        width,
        height,
        channels,
        data,
        icc: None,
        exif: None,
    }
}

#[test]
fn tiff_16bit_rgb_roundtrip_is_lossless() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scan.tif");
    let img = make(70, 50, 3, true);
    encode(&path, &img).unwrap();
    let back = decode(&path).unwrap();
    assert_eq!((back.width, back.height, back.channels), (70, 50, 3));
    match (&img.data, &back.data) {
        (PixelData::U16(a), PixelData::U16(b)) => assert_eq!(a, b),
        _ => panic!("expected 16-bit"),
    }
}

#[test]
fn tiff_8bit_gray_roundtrip_is_lossless() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.tiff");
    let img = make(33, 21, 1, false);
    encode(&path, &img).unwrap();
    let back = decode(&path).unwrap();
    match (&img.data, &back.data) {
        (PixelData::U8(a), PixelData::U8(b)) => assert_eq!(a, b),
        _ => panic!("expected 8-bit"),
    }
}

#[test]
fn tiff_rgba_input_drops_alpha() {
    // hand-write a tiny RGBA8 TIFF via the tiff crate, expect RGB back
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tif");
    {
        let file = std::fs::File::create(&path).unwrap();
        let mut t = tiff::encoder::TiffEncoder::new(file).unwrap();
        let rgba: Vec<u8> = (0..4 * 4 * 4).map(|i| i as u8).collect();
        t.write_image::<tiff::encoder::colortype::RGBA8>(4, 4, &rgba)
            .unwrap();
    }
    let back = decode(&path).unwrap();
    assert_eq!(back.channels, 3);
    match &back.data {
        PixelData::U8(v) => assert_eq!(v.len(), 4 * 4 * 3),
        _ => panic!("expected 8-bit"),
    }
}

#[test]
fn tiff_single_strip_100mp_scan_decodes() {
    // Scanner software (and tifffile) often writes the whole image as one
    // strip; a 96MP grey u16 strip is 192MB, past the tiff crate's default
    // decode limits. The spec requires ~100MP scans, so decode raises them.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.tif");
    let (w, h) = (12000u32, 8000u32);
    {
        let file = std::fs::File::create(&path).unwrap();
        let mut t = tiff::encoder::TiffEncoder::new(file).unwrap();
        let mut image = t
            .new_image::<tiff::encoder::colortype::Gray16>(w, h)
            .unwrap();
        image.rows_per_strip(h).unwrap(); // everything in one strip
        let data: Vec<u16> = (0..(w * h) as usize).map(|i| (i % 65521) as u16).collect();
        image.write_data(&data).unwrap();
    }
    let back = decode(&path).unwrap();
    assert_eq!((back.width, back.height), (w, h));
}
