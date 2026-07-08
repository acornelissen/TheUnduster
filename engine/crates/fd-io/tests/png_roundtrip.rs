use fd_io::{decode, encode, ImageBuf, PixelData};

fn gradient(width: u32, height: u32, channels: u8, sixteen: bool) -> ImageBuf {
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
fn png_16bit_rgb_roundtrip_is_lossless() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.png");
    let img = gradient(64, 48, 3, true);
    encode(&path, &img).unwrap();
    let back = decode(&path).unwrap();
    assert_eq!((back.width, back.height, back.channels), (64, 48, 3));
    match (&img.data, &back.data) {
        (PixelData::U16(a), PixelData::U16(b)) => assert_eq!(a, b),
        _ => panic!("expected 16-bit data back"),
    }
}

#[test]
fn png_8bit_gray_roundtrip_is_lossless() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.png");
    let img = gradient(32, 32, 1, false);
    encode(&path, &img).unwrap();
    let back = decode(&path).unwrap();
    match (&img.data, &back.data) {
        (PixelData::U8(a), PixelData::U8(b)) => assert_eq!(a, b),
        _ => panic!("expected 8-bit data back"),
    }
}

#[test]
fn to_f32_normalizes_both_depths() {
    let img8 = gradient(4, 4, 1, false);
    let img16 = gradient(4, 4, 1, true);
    let f8 = img8.to_f32();
    let f16 = img16.to_f32();
    assert!(f8.iter().all(|&v| (0.0..=1.0).contains(&v)));
    assert!(f16.iter().all(|&v| (0.0..=1.0).contains(&v)));
    assert_eq!(f8.len(), 16);
}

#[test]
fn decode_missing_file_names_the_path() {
    let err = decode(std::path::Path::new("/nonexistent/y.png")).unwrap_err();
    assert!(err.to_string().contains("y.png"));
}
