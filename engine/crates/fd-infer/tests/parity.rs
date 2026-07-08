use std::path::PathBuf;

use fd_infer::{Detector, Ep};
use fd_io::{ImageBuf, PixelData};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn load_parity_input() -> (ImageBuf, u32, u32, f32) {
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("parity-meta.json")).unwrap(),
    )
    .unwrap();
    let (w, h) = (
        meta["width"].as_u64().unwrap() as u32,
        meta["height"].as_u64().unwrap() as u32,
    );
    let tol = meta["tolerance"].as_f64().unwrap() as f32;
    let bytes = std::fs::read(fixtures().join("parity-input.bin")).unwrap();
    let pixels: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    assert_eq!(pixels.len(), (w * h) as usize);
    let img = ImageBuf {
        width: w,
        height: h,
        channels: 1,
        data: PixelData::U16(pixels),
        icc: None,
        exif: None,
    };
    (img, w, h, tol)
}

#[test]
fn probabilities_match_python_reference() {
    let (img, w, h, tol) = load_parity_input();
    let mut det = Detector::load(&fixtures().join("tiny-detector.onnx"), Ep::Cpu).unwrap();
    let probs = det.probabilities(&img).unwrap();
    assert_eq!(probs.len(), (w * h) as usize);

    let expected_bytes = std::fs::read(fixtures().join("parity-expected.bin")).unwrap();
    let expected: Vec<f32> = expected_bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]) as f32 / 65535.0)
        .collect();

    let mut max_diff = 0f32;
    for (a, b) in probs.iter().zip(expected.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(
        max_diff < tol,
        "max deviation from Python reference: {max_diff} (tolerance {tol})"
    );
}

#[test]
fn mask_thresholds_probabilities() {
    let (img, ..) = load_parity_input();
    let mut det = Detector::load(&fixtures().join("tiny-detector.onnx"), Ep::Cpu).unwrap();
    let probs = det.probabilities(&img).unwrap();
    let mask = det.mask(&img, 0.5).unwrap();
    for (p, m) in probs.iter().zip(mask.iter()) {
        assert_eq!(*m, *p > 0.5);
    }
}

#[test]
fn rgb_input_to_gray_model_is_adapted() {
    // 3-channel image against the 1-channel tiny detector must not error
    let img = ImageBuf {
        width: 100,
        height: 80,
        channels: 3,
        data: PixelData::U8(vec![100; 100 * 80 * 3]),
        icc: None,
        exif: None,
    };
    let mut det = Detector::load(&fixtures().join("tiny-detector.onnx"), Ep::Cpu).unwrap();
    let probs = det.probabilities(&img).unwrap();
    assert_eq!(probs.len(), 100 * 80);
}
