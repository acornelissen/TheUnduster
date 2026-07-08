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
    // Scanner software often writes the whole image as ONE strip; a 100MP
    // 16-bit scan then needs a ~600MB decode buffer, far past the tiff
    // crate's defaults. Bounded (not unlimited) so a hostile file cannot
    // demand arbitrary allocations: 2GB covers 250MP 16-bit RGB with room.
    let mut limits = tiff::decoder::Limits::default();
    limits.decoding_buffer_size = 2 * 1024 * 1024 * 1024;
    limits.intermediate_buffer_size = 2 * 1024 * 1024 * 1024;
    limits.ifd_value_size = 8 * 1024 * 1024;
    let mut d = Decoder::new(file)
        .map_err(|e| decode_err(path, e))?
        .with_limits(limits);
    let (width, height) = d.dimensions().map_err(|e| decode_err(path, e))?;
    let color = d.colortype().map_err(|e| decode_err(path, e))?;
    // The tiff crate may hand the ICC tag back as bytes or as a list of
    // unsigned values depending on how it was written; accept both.
    let icc = d
        .get_tag_u8_vec(tiff::tags::Tag::Unknown(34675))
        .ok()
        .or_else(|| {
            let value = d.get_tag(tiff::tags::Tag::Unknown(34675)).ok()?;
            match value {
                tiff::decoder::ifd::Value::List(vals) => vals
                    .into_iter()
                    .map(|v| v.into_u32().ok().map(|u| u as u8))
                    .collect::<Option<Vec<u8>>>(),
                other => other.into_u32().ok().map(|u| vec![u as u8]),
            }
        });
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

fn write_with_icc<C: colortype::ColorType>(
    t: &mut TiffEncoder<File>,
    path: &Path,
    w: u32,
    h: u32,
    data: &[C::Inner],
    icc: Option<&[u8]>,
) -> Result<(), IoError>
where
    [C::Inner]: tiff::encoder::TiffValue,
{
    let mut image = t.new_image::<C>(w, h).map_err(|e| encode_err(path, e))?;
    if let Some(icc) = icc {
        image
            .encoder()
            .write_tag(tiff::tags::Tag::Unknown(34675), icc)
            .map_err(|e| encode_err(path, e))?;
    }
    image.write_data(data).map_err(|e| encode_err(path, e))
}

pub fn encode(path: &Path, img: &ImageBuf) -> Result<(), IoError> {
    let file = File::create(path).map_err(|e| encode_err(path, e))?;
    let mut t = TiffEncoder::new(file).map_err(|e| encode_err(path, e))?;
    let (w, h) = (img.width, img.height);
    let icc = img.icc.as_deref();
    match (img.channels, &img.data) {
        (1, PixelData::U8(v)) => write_with_icc::<colortype::Gray8>(&mut t, path, w, h, v, icc),
        (1, PixelData::U16(v)) => write_with_icc::<colortype::Gray16>(&mut t, path, w, h, v, icc),
        (3, PixelData::U8(v)) => write_with_icc::<colortype::RGB8>(&mut t, path, w, h, v, icc),
        (3, PixelData::U16(v)) => write_with_icc::<colortype::RGB16>(&mut t, path, w, h, v, icc),
        _ => Err(IoError::Unsupported(format!("{} channels", img.channels))),
    }
}
