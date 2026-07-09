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

/// Regression: a large (inpaint-tier) defect hugging the image edge, on an
/// image whose dimensions are not multiples of 8. The inpaint crop is
/// rounded to a multiple of 8; if that rounding shrinks the crop below the
/// defect's extent, the write-back indexed past the crop buffer and
/// panicked (field report: brush stroke painted to the frame edge).
#[test]
fn edge_hugging_inpaint_defect_does_not_panic() {
    let mut img = noisy_image(100, 50);
    let original = img.clone();
    let mut mask = vec![false; 100 * 50];
    // 40x30 block hugging the bottom-right corner: bbox max_dim 40 >
    // TINY_MAX_DIM, and the multiple-of-8 crop rounding cuts both the right
    // and bottom defect extents, so unguarded write-back indexes past the
    // crop buffer (not merely wrapping within it).
    for y in 20..50 {
        for x in 60..100 {
            mask[y * 100 + x] = true;
        }
    }
    let mut inpainter = Inpainter::load(&fixtures().join("tiny-inpaint.onnx"), fd_infer::Ep::Cpu)
        .expect("fixture inpainter loads");
    let report = heal(&mut img, &mask, Some(&mut inpainter)).expect("heal succeeds");
    assert_eq!(report.inpainted, 1);
    // bit-exactness outside the mask still holds
    let (PixelData::U16(a), PixelData::U16(b)) = (&original.data, &img.data) else {
        panic!("depth changed");
    };
    for i in 0..100 * 50 {
        if !mask[i] {
            for c in 0..3 {
                assert_eq!(a[i * 3 + c], b[i * 3 + c], "pixel {i} changed outside mask");
            }
        }
    }
}

/// Harder variant: the defect spans the full image width, so no multiple-of-8
/// crop can contain it on a 100-wide image. The heal must degrade gracefully
/// (edge slivers may stay unhealed) rather than panic.
#[test]
fn full_width_defect_does_not_panic() {
    let mut img = noisy_image(100, 50);
    let mut mask = vec![false; 100 * 50];
    for y in 40..50 {
        for x in 0..100 {
            mask[y * 100 + x] = true;
        }
    }
    let mut inpainter = Inpainter::load(&fixtures().join("tiny-inpaint.onnx"), fd_infer::Ep::Cpu)
        .expect("fixture inpainter loads");
    heal(&mut img, &mask, Some(&mut inpainter)).expect("heal succeeds");
}

/// Progress callback fires once per defect with a stable total, ending at
/// (total, total) -- the app surfaces this during multi-minute LaMa heals.
#[test]
fn heal_reports_per_defect_progress() {
    let mut img = noisy_image(64, 64);
    let mut mask = vec![false; 64 * 64];
    for (cx, cy) in [(10u32, 10u32), (30, 30), (50, 50)] {
        for y in cy - 2..=cy + 2 {
            for x in cx - 2..=cx + 2 {
                mask[(y * 64 + x) as usize] = true;
            }
        }
    }
    let mut calls: Vec<(usize, usize)> = Vec::new();
    let report = fd_heal::heal_with_progress(&mut img, &mask, None, &mut |done, total| {
        calls.push((done, total));
    })
    .unwrap();
    assert_eq!(report.defects, 3);
    assert_eq!(calls, vec![(1, 3), (2, 3), (3, 3)]);
}
