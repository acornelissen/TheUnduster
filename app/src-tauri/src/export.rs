//! Healed-file export: the spec's untouched-pixel guarantee is enforced
//! here, at write time, not merely trusted from the heal step.

use std::path::Path;

use fd_io::{ImageBuf, PixelData};

#[derive(Debug)]
pub struct ExportReport {
    pub changed_pixels: usize,
}

/// Verifies the healed copy differs from the original ONLY inside the mask,
/// then writes it atomically to `dest` (temp sibling + rename). Any
/// outside-mask difference aborts before anything is written.
pub fn export_healed(
    original: &ImageBuf,
    healed: &ImageBuf,
    mask: &[bool],
    dest: &Path,
) -> Result<ExportReport, String> {
    if original.width != healed.width
        || original.height != healed.height
        || original.channels != healed.channels
    {
        return Err("healed dimensions do not match the original".to_string());
    }
    let px = (original.width as usize) * (original.height as usize);
    if mask.len() != px {
        return Err("mask length does not match the image".to_string());
    }
    let ch = original.channels as usize;
    let mut changed = 0usize;
    let mut check = |i: usize, differs: bool| -> Result<(), String> {
        if differs {
            if mask[i] {
                changed += 1;
            } else {
                let (x, y) = (i % original.width as usize, i / original.width as usize);
                return Err(format!(
                    "untouched-pixel guarantee violated at ({x}, {y}); export aborted"
                ));
            }
        }
        Ok(())
    };
    match (&original.data, &healed.data) {
        (PixelData::U8(a), PixelData::U8(b)) => {
            for i in 0..px {
                check(i, a[i * ch..(i + 1) * ch] != b[i * ch..(i + 1) * ch])?;
            }
        }
        (PixelData::U16(a), PixelData::U16(b)) => {
            for i in 0..px {
                check(i, a[i * ch..(i + 1) * ch] != b[i * ch..(i + 1) * ch])?;
            }
        }
        _ => return Err("healed bit depth does not match the original".to_string()),
    }

    let file_name = dest
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("bad destination: {}", dest.display()))?;
    let tmp = dest.with_file_name(format!(".unduster-tmp-{file_name}"));
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    fd_io::encode(&tmp, healed).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, dest).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", dest.display())
    })?;
    Ok(ExportReport {
        changed_pixels: changed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32, fill: u8) -> ImageBuf {
        ImageBuf {
            width: w,
            height: h,
            channels: 1,
            data: PixelData::U8(vec![fill; (w * h) as usize]),
            icc: Some(vec![1, 2, 3]),
            exif: None,
        }
    }

    #[test]
    fn exports_a_valid_heal_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let original = img(16, 16, 100);
        let mut healed = original.clone();
        let mut mask = vec![false; 256];
        mask[17] = true;
        if let PixelData::U8(v) = &mut healed.data {
            v[17] = 200;
        }
        let dest = dir.path().join("out.png");
        let report = export_healed(&original, &healed, &mask, &dest).unwrap();
        assert_eq!(report.changed_pixels, 1);
        assert!(dest.exists());
        assert!(!dir.path().join(".unduster-tmp-out.png").exists());
        let back = fd_io::decode(&dest).unwrap();
        assert_eq!(back.icc.as_deref(), Some([1u8, 2, 3].as_slice())); // metadata rode along
    }

    #[test]
    fn refuses_an_outside_mask_difference() {
        let dir = tempfile::tempdir().unwrap();
        let original = img(16, 16, 100);
        let mut healed = original.clone();
        if let PixelData::U8(v) = &mut healed.data {
            v[5] = 42; // tampered OUTSIDE the (empty) mask
        }
        let mask = vec![false; 256];
        let dest = dir.path().join("out.png");
        let err = export_healed(&original, &healed, &mask, &dest).unwrap_err();
        assert!(err.contains("untouched-pixel"));
        assert!(err.contains("(5, 0)"));
        assert!(!dest.exists()); // nothing written
    }

    #[test]
    fn end_to_end_with_a_real_heal() {
        // heal with the classical tier, then export: the engine guarantee
        // and the export verification must agree.
        let dir = tempfile::tempdir().unwrap();
        let mut noisy = img(64, 64, 0);
        if let PixelData::U8(v) = &mut noisy.data {
            for (i, p) in v.iter_mut().enumerate() {
                *p = ((i * 37) % 251) as u8;
            }
        }
        let original = noisy.clone();
        let mut mask = vec![false; 64 * 64];
        for y in 30..33 {
            for x in 30..33 {
                mask[y * 64 + x] = true;
            }
        }
        fd_heal::heal(&mut noisy, &mask, None).unwrap();
        let dest = dir.path().join("healed.tif");
        let report = export_healed(&original, &noisy, &mask, &dest).unwrap();
        assert!(report.changed_pixels > 0);
        assert!(dest.exists());
    }
}
