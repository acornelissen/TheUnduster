//! Post-encode metadata attachment for PNG/JPEG via img-parts.
//! Strategy: encode pixels to bytes first, then splice metadata chunks in.

use img_parts::{Bytes, ImageEXIF, ImageICC};

pub fn attach_jpeg(bytes: Vec<u8>, icc: Option<&[u8]>, exif: Option<&[u8]>) -> Vec<u8> {
    let mut jpeg = match img_parts::jpeg::Jpeg::from_bytes(Bytes::from(bytes.clone())) {
        Ok(j) => j,
        Err(_) => return bytes,
    };
    if let Some(p) = icc {
        jpeg.set_icc_profile(Some(Bytes::copy_from_slice(p)));
    }
    if let Some(e) = exif {
        jpeg.set_exif(Some(Bytes::copy_from_slice(e)));
    }
    let mut out = Vec::new();
    if jpeg.encoder().write_to(&mut out).is_ok() {
        out
    } else {
        bytes
    }
}

pub fn attach_png(bytes: Vec<u8>, icc: Option<&[u8]>) -> Vec<u8> {
    let mut png = match img_parts::png::Png::from_bytes(Bytes::from(bytes.clone())) {
        Ok(p) => p,
        Err(_) => return bytes,
    };
    if let Some(p) = icc {
        png.set_icc_profile(Some(Bytes::copy_from_slice(p)));
    }
    let mut out = Vec::new();
    if png.encoder().write_to(&mut out).is_ok() {
        out
    } else {
        bytes
    }
}
