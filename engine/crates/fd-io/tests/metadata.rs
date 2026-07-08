use fd_io::{decode, encode, ImageBuf, PixelData};

fn tiny_with_meta(icc: Option<Vec<u8>>, exif: Option<Vec<u8>>) -> ImageBuf {
    ImageBuf {
        width: 8,
        height: 8,
        channels: 3,
        data: PixelData::U8(vec![128; 8 * 8 * 3]),
        icc,
        exif,
    }
}

// A plausible-looking little ICC blob; content is opaque to us.
fn fake_icc() -> Vec<u8> {
    let mut b = vec![0u8; 128];
    b[36..40].copy_from_slice(b"acsp"); // ICC signature at offset 36
    b
}

#[test]
fn jpeg_icc_and_exif_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.jpg");
    let exif = b"Exif\0\0MM\0*\0\0\0\x08".to_vec();
    let img = tiny_with_meta(Some(fake_icc()), Some(exif.clone()));
    encode(&path, &img).unwrap();
    let back = decode(&path).unwrap();
    assert_eq!(back.icc.as_deref(), Some(fake_icc().as_slice()));
    assert!(back.exif.is_some());
}

#[test]
fn png_icc_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.png");
    let img = tiny_with_meta(Some(fake_icc()), None);
    encode(&path, &img).unwrap();
    let back = decode(&path).unwrap();
    assert_eq!(back.icc.as_deref(), Some(fake_icc().as_slice()));
}

#[test]
fn tiff_icc_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.tif");
    let img = tiny_with_meta(Some(fake_icc()), None);
    encode(&path, &img).unwrap();
    let back = decode(&path).unwrap();
    assert_eq!(back.icc.as_deref(), Some(fake_icc().as_slice()));
}

#[test]
fn no_metadata_stays_none() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("n.png");
    encode(&path, &tiny_with_meta(None, None)).unwrap();
    let back = decode(&path).unwrap();
    assert!(back.icc.is_none());
}
