use std::path::PathBuf;

use fd_heal::{heal, Inpainter, TINY_MAX_DIM};
use fd_io::{ImageBuf, PixelData};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// Deterministic pseudo-random 16-bit RGB image.
fn noisy_image(width: u32, height: u32) -> ImageBuf {
    let n = (width * height * 3) as usize;
    let mut state = 0x2545F4914F6CDD1Du64;
    let data: Vec<u16> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 48) as u16
        })
        .collect();
    ImageBuf {
        width,
        height,
        channels: 3,
        data: PixelData::U16(data),
        icc: None,
        exif: None,
    }
}

fn blob_mask(width: u32, height: u32, cx: u32, cy: u32, r: u32) -> Vec<bool> {
    (0..width * height)
        .map(|i| {
            let (x, y) = (i % width, i / width);
            (x as i64 - cx as i64).pow(2) + (y as i64 - cy as i64).pow(2) <= (r as i64).pow(2)
        })
        .collect()
}

#[test]
fn unmasked_pixels_are_bit_identical_after_heal() {
    let mut img = noisy_image(200, 160);
    let before = img.clone();
    let mask = blob_mask(200, 160, 100, 80, 12); // big blob -> inpaint tier
    let mut inp =
        Inpainter::load(&fixtures().join("tiny-inpaint.onnx"), fd_infer::Ep::Cpu).unwrap();
    let report = heal(&mut img, &mask, Some(&mut inp)).unwrap();
    assert_eq!(report.defects, 1);
    assert_eq!(report.inpainted, 1);
    let (PixelData::U16(a), PixelData::U16(b)) = (&before.data, &img.data) else {
        panic!("expected u16")
    };
    let mut changed_inside = 0;
    for i in 0..(200 * 160) as usize {
        for c in 0..3 {
            if mask[i] {
                if a[i * 3 + c] != b[i * 3 + c] {
                    changed_inside += 1;
                }
            } else {
                assert_eq!(a[i * 3 + c], b[i * 3 + c], "unmasked pixel {i} changed");
            }
        }
    }
    assert!(changed_inside > 0, "healing did nothing inside the mask");
}

#[test]
fn tiny_defects_use_classical_tier_without_model() {
    let mut img = noisy_image(64, 64);
    // 5px diameter blob: exactly at the TINY_MAX_DIM boundary -> tiny tier
    let mask = blob_mask(64, 64, 32, 32, TINY_MAX_DIM / 2);
    let report = heal(&mut img, &mask, None).unwrap();
    assert_eq!(report.tiny, 1);
    assert_eq!(report.inpainted, 0);
}

#[test]
fn large_defect_without_inpainter_falls_back_to_classical() {
    let mut img = noisy_image(128, 128);
    let before = img.clone();
    let mask = blob_mask(128, 128, 64, 64, 10);
    let report = heal(&mut img, &mask, None).unwrap();
    assert_eq!(report.defects, 1);
    assert_eq!(report.tiny + report.inpainted, 1);
    // guarantee still holds on the fallback path
    let (PixelData::U16(a), PixelData::U16(b)) = (&before.data, &img.data) else {
        panic!()
    };
    for i in 0..(128 * 128) as usize {
        if !mask[i] {
            for c in 0..3 {
                assert_eq!(a[i * 3 + c], b[i * 3 + c]);
            }
        }
    }
}

#[test]
fn healed_region_carries_grain() {
    // inpaint fixture returns a flat mean fill; grain re-synthesis must add
    // texture so the filled region is not flat
    let mut img = noisy_image(200, 160);
    let mask = blob_mask(200, 160, 100, 80, 12);
    let mut inp =
        Inpainter::load(&fixtures().join("tiny-inpaint.onnx"), fd_infer::Ep::Cpu).unwrap();
    heal(&mut img, &mask, Some(&mut inp)).unwrap();
    let PixelData::U16(v) = &img.data else {
        panic!()
    };
    let inside: Vec<f64> = (0..(200 * 160) as usize)
        .filter(|&i| mask[i])
        .map(|i| v[i * 3] as f64)
        .collect();
    let mean = inside.iter().sum::<f64>() / inside.len() as f64;
    let var = inside.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / inside.len() as f64;
    assert!(
        var.sqrt() > 100.0,
        "filled region is flat: std {}",
        var.sqrt()
    );
}
