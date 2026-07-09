use std::path::PathBuf;

use fd_heal::Inpainter;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

#[test]
fn fixed_contract_is_detected_and_output_unscaled() {
    let mut inp = Inpainter::load(
        &fixtures().join("tiny-inpaint-fixed.onnx"),
        fd_infer::Ep::Cpu,
    )
    .expect("fixed fixture loads");
    assert_eq!(inp.window_size(), Some(64));

    let n = 64usize;
    let planes = [
        vec![0.25f32; n * n],
        vec![0.5f32; n * n],
        vec![0.75f32; n * n],
    ];
    let mut mask = vec![false; n * n];
    for y in 20..40 {
        for x in 20..40 {
            mask[y * n + x] = true;
        }
    }
    let out = inp.inpaint(&planes, &mask, n, n).expect("inpaint runs");
    // unmasked pixels come back in [0,1], not 0-255: the adapter unscales
    assert!((out[0][0] - 0.25).abs() < 1e-3, "got {}", out[0][0]);
    assert!((out[2][0] - 0.75).abs() < 1e-3);
    // masked pixels are the channel mean (mean-fill fixture semantics)
    assert!((out[1][30 * n + 30] - 0.5).abs() < 1e-2);
}

#[test]
fn fixed_contract_rejects_wrong_crop_size() {
    let mut inp = Inpainter::load(
        &fixtures().join("tiny-inpaint-fixed.onnx"),
        fd_infer::Ep::Cpu,
    )
    .expect("fixed fixture loads");
    let planes = [
        vec![0f32; 32 * 32],
        vec![0f32; 32 * 32],
        vec![0f32; 32 * 32],
    ];
    let mask = vec![false; 32 * 32];
    assert!(inp.inpaint(&planes, &mask, 32, 32).is_err());
}

#[test]
fn dynamic_contract_unchanged() {
    let mut inp = Inpainter::load(&fixtures().join("tiny-inpaint.onnx"), fd_infer::Ep::Cpu)
        .expect("dynamic fixture loads");
    assert_eq!(inp.window_size(), None);
    let planes = [
        vec![0.5f32; 24 * 16],
        vec![0.5f32; 24 * 16],
        vec![0.5f32; 24 * 16],
    ];
    let mask = vec![false; 24 * 16];
    let out = inp
        .inpaint(&planes, &mask, 24, 16)
        .expect("dynamic path still works");
    assert!((out[0][0] - 0.5).abs() < 1e-3);
}
