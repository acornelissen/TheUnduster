use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use image::{ColorType, DynamicImage, ImageDecoder, ImageEncoder};

use crate::{decode_err, encode_err, ImageBuf, IoError, PixelData};

pub fn decode(path: &Path) -> Result<ImageBuf, IoError> {
    let file = File::open(path).map_err(|e| decode_err(path, e))?;
    let reader = image::ImageReader::new(BufReader::new(file))
        .with_guessed_format()
        .map_err(|e| decode_err(path, e))?;
    let mut decoder = reader.into_decoder().map_err(|e| decode_err(path, e))?;
    let icc = decoder.icc_profile().ok().flatten();
    let exif = decoder.exif_metadata().ok().flatten();
    let dyn_img = DynamicImage::from_decoder(decoder).map_err(|e| decode_err(path, e))?;
    let (width, height) = (dyn_img.width(), dyn_img.height());
    let (channels, data) = match dyn_img {
        DynamicImage::ImageLuma8(b) => (1, PixelData::U8(b.into_raw())),
        DynamicImage::ImageRgb8(b) => (3, PixelData::U8(b.into_raw())),
        DynamicImage::ImageLuma16(b) => (1, PixelData::U16(b.into_raw())),
        DynamicImage::ImageRgb16(b) => (3, PixelData::U16(b.into_raw())),
        // Alpha and exotic layouts: normalize to RGB at matching depth.
        other => {
            let c = other.color();
            let bytes_per_channel = c.bytes_per_pixel() / c.channel_count();
            match bytes_per_channel {
                2 => (3, PixelData::U16(other.into_rgb16().into_raw())),
                _ => (3, PixelData::U8(other.into_rgb8().into_raw())),
            }
        }
    };
    Ok(ImageBuf {
        width,
        height,
        channels,
        data,
        icc,
        exif,
    })
}

pub fn encode(path: &Path, img: &ImageBuf) -> Result<(), IoError> {
    let ext = crate::ext_of(path);
    let mut raw: Vec<u8> = Vec::new();
    {
        let w = std::io::Cursor::new(&mut raw);
        let color = match (img.channels, &img.data) {
            (1, PixelData::U8(_)) => ColorType::L8,
            (3, PixelData::U8(_)) => ColorType::Rgb8,
            (1, PixelData::U16(_)) => ColorType::L16,
            (3, PixelData::U16(_)) => ColorType::Rgb16,
            _ => return Err(IoError::Unsupported(format!("{} channels", img.channels))),
        };
        match (ext.as_str(), &img.data) {
            ("png", PixelData::U8(v)) => image::codecs::png::PngEncoder::new(w)
                .write_image(v, img.width, img.height, color.into())
                .map_err(|e| encode_err(path, e))?,
            ("png", PixelData::U16(v)) => {
                // image's encoders take 16-bit samples as native-endian bytes
                let bytes: Vec<u8> = v.iter().flat_map(|p| p.to_ne_bytes()).collect();
                image::codecs::png::PngEncoder::new(w)
                    .write_image(&bytes, img.width, img.height, color.into())
                    .map_err(|e| encode_err(path, e))?
            }
            ("jpg" | "jpeg", PixelData::U8(v)) => {
                image::codecs::jpeg::JpegEncoder::new_with_quality(w, 95)
                    .write_image(v, img.width, img.height, color.into())
                    .map_err(|e| encode_err(path, e))?
            }
            ("jpg" | "jpeg", PixelData::U16(_)) => {
                return Err(IoError::Unsupported("16-bit JPEG".to_string()))
            }
            _ => return Err(IoError::Unsupported(ext)),
        }
    }
    let final_bytes = match ext.as_str() {
        "jpg" | "jpeg" if img.icc.is_some() || img.exif.is_some() => {
            crate::metadata::attach_jpeg(raw, img.icc.as_deref(), img.exif.as_deref())
        }
        "png" if img.icc.is_some() => crate::metadata::attach_png(raw, img.icc.as_deref()),
        _ => raw,
    };
    std::fs::write(path, final_bytes).map_err(|e| encode_err(path, e))
}
