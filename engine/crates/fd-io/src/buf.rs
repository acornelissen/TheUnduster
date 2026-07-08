#[derive(Debug, Clone, PartialEq)]
pub enum PixelData {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

/// Pixels in native depth. f32 views are derived; native data is the
/// source of truth for the bit-exactness guarantee.
#[derive(Debug, Clone)]
pub struct ImageBuf {
    pub width: u32,
    pub height: u32,
    pub channels: u8, // 1 = grey, 3 = RGB
    pub data: PixelData,
    pub icc: Option<Vec<u8>>,
    pub exif: Option<Vec<u8>>,
}

impl ImageBuf {
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize * self.channels as usize
    }

    /// Interleaved f32 in [0, 1]. 255 or 65535 maps to 1.0.
    pub fn to_f32(&self) -> Vec<f32> {
        match &self.data {
            PixelData::U8(v) => v.iter().map(|&p| p as f32 / 255.0).collect(),
            PixelData::U16(v) => v.iter().map(|&p| p as f32 / 65535.0).collect(),
        }
    }
}
