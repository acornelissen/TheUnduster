use std::fs::File;
use std::path::Path;

use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::{colortype, TiffEncoder};
use tiff::ColorType;

use crate::{decode_err, encode_err, ImageBuf, IoError, PixelData};

fn drop_alpha<T: Copy>(data: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(data.len() / 4 * 3);
    for px in data.chunks_exact(4) {
        out.extend_from_slice(&px[..3]);
    }
    out
}

pub fn decode(path: &Path) -> Result<ImageBuf, IoError> {
    let file = File::open(path).map_err(|e| decode_err(path, e))?;
    let mut d = Decoder::new(file).map_err(|e| decode_err(path, e))?;
    let (width, height) = d.dimensions().map_err(|e| decode_err(path, e))?;
    let color = d.colortype().map_err(|e| decode_err(path, e))?;
    let icc = d.get_tag_u8_vec(tiff::tags::Tag::Unknown(34675)).ok();
    let result = d.read_image().map_err(|e| decode_err(path, e))?;
    let (channels, data) = match (color, result) {
        (ColorType::Gray(8), DecodingResult::U8(v)) => (1, PixelData::U8(v)),
        (ColorType::Gray(16), DecodingResult::U16(v)) => (1, PixelData::U16(v)),
        (ColorType::RGB(8), DecodingResult::U8(v)) => (3, PixelData::U8(v)),
        (ColorType::RGB(16), DecodingResult::U16(v)) => (3, PixelData::U16(v)),
        (ColorType::RGBA(8), DecodingResult::U8(v)) => (3, PixelData::U8(drop_alpha(&v))),
        (ColorType::RGBA(16), DecodingResult::U16(v)) => (3, PixelData::U16(drop_alpha(&v))),
        (c, _) => {
            return Err(decode_err(path, format!("unsupported TIFF layout: {c:?}")));
        }
    };
    Ok(ImageBuf {
        width,
        height,
        channels,
        data,
        icc,
        exif: None,
    })
}

pub fn encode(path: &Path, img: &ImageBuf) -> Result<(), IoError> {
    let file = File::create(path).map_err(|e| encode_err(path, e))?;
    let mut t = TiffEncoder::new(file).map_err(|e| encode_err(path, e))?;
    let (w, h) = (img.width, img.height);
    match (img.channels, &img.data) {
        (1, PixelData::U8(v)) => t
            .write_image::<colortype::Gray8>(w, h, v)
            .map_err(|e| encode_err(path, e)),
        (1, PixelData::U16(v)) => t
            .write_image::<colortype::Gray16>(w, h, v)
            .map_err(|e| encode_err(path, e)),
        (3, PixelData::U8(v)) => t
            .write_image::<colortype::RGB8>(w, h, v)
            .map_err(|e| encode_err(path, e)),
        (3, PixelData::U16(v)) => t
            .write_image::<colortype::RGB16>(w, h, v)
            .map_err(|e| encode_err(path, e)),
        _ => Err(IoError::Unsupported(format!("{} channels", img.channels))),
    }
}
