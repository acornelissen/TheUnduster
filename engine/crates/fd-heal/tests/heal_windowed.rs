use std::path::PathBuf;

use fd_heal::{heal, Inpainter};
use fd_io::{ImageBuf, PixelData};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

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

fn fixed_inpainter() -> Inpainter {
    Inpainter::load(
        &fixtures().join("tiny-inpaint-fixed.onnx"),
        fd_infer::Ep::Cpu,
    )
    .unwrap()
}

/// A defect wider than one 64px window (interior 48) forces multi-window
/// tiling; every masked pixel must change (mean fill differs from noise
/// with overwhelming probability) and every unmasked pixel must not.
#[test]
fn defect_wider_than_a_window_is_fully_filled() {
    let mut img = noisy_image(200, 90);
    let original = img.clone();
    let mut mask = vec![false; 200 * 90];
    for y in 30..60 {
        for x in 20..180 {
            mask[y * 200 + x] = true; // 160x30: spans 4 interiors horizontally
        }
    }
    let report = heal(&mut img, &mask, Some(&mut fixed_inpainter())).unwrap();
    assert_eq!(report.inpainted, 1);
    let (PixelData::U16(a), PixelData::U16(b)) = (&original.data, &img.data) else {
        panic!("depth changed");
    };
    let mut unchanged_masked = 0usize;
    for i in 0..200 * 90 {
        if mask[i] {
            if (0..3).all(|c| a[i * 3 + c] == b[i * 3 + c]) {
                unchanged_masked += 1;
            }
        } else {
            for c in 0..3 {
                assert_eq!(a[i * 3 + c], b[i * 3 + c], "pixel {i} changed outside mask");
            }
        }
    }
    // grain re-synthesis could coincidentally reproduce a value; allow a
    // whisper of slack but any systematic gap (a missed window interior
    // would leave 48x30 = 1440+ pixels untouched) must fail loudly
    assert!(
        unchanged_masked < 50,
        "{unchanged_masked} masked pixels untouched -- a window was skipped"
    );
}

/// Image smaller than the window in one dimension: edge-replicate padding
/// must kick in rather than erroring or panicking.
#[test]
fn image_smaller_than_window_pads() {
    let mut img = noisy_image(50, 40); // both dims < 64
    let mut mask = vec![false; 50 * 40];
    for y in 10..30 {
        for x in 10..40 {
            mask[y * 50 + x] = true;
        }
    }
    let report = heal(&mut img, &mask, Some(&mut fixed_inpainter())).unwrap();
    assert_eq!(report.inpainted, 1);
}

/// Corner-hugging defect (the 0cfe94b regression class) through the
/// windowed path.
#[test]
fn corner_defect_windows_safely() {
    let mut img = noisy_image(100, 70);
    let mut mask = vec![false; 100 * 70];
    for y in 40..70 {
        for x in 60..100 {
            mask[y * 100 + x] = true;
        }
    }
    heal(&mut img, &mask, Some(&mut fixed_inpainter())).unwrap();
}

/// Paints five 8px specks into `mask`, clustered so every pairwise gap is
/// well under the window margin (n/8 = 8 for the 64px fixture, so the merge
/// threshold is a 16px gap) and the union bbox (32x32) fits inside one
/// window's 48px interior. Returns the specks' pixel count (should equal
/// 5 * 8 * 8 = 320) for reference.
fn five_clustered_specks(mask: &mut [bool], width: u32, x_off: u32, y_off: u32) {
    let speck = |mask: &mut [bool], x0: u32, y0: u32| {
        for y in y0..y0 + 8 {
            for x in x0..x0 + 8 {
                mask[(y * width + x) as usize] = true;
            }
        }
    };
    speck(mask, x_off + 10, y_off + 10);
    speck(mask, x_off + 22, y_off + 10); // 4px gap from the first, in x
    speck(mask, x_off + 10, y_off + 22); // 4px gap from the first, in y
    speck(mask, x_off + 34, y_off + 10); // 4px gap from the second, in x
    speck(mask, x_off + 10, y_off + 34); // 4px gap from the third, in y
}

/// Five clustered specks, all well within one 64px window's 48px interior,
/// must share ONE inpainter call -- not one call per defect. This is the
/// batching behaviour Task 2 adds; on unbatched code this test is red (5
/// calls, one per defect).
#[test]
fn clustered_specks_share_one_inpainter_call() {
    let mut img = noisy_image(64, 64);
    let original = img.clone();
    let mut mask = vec![false; 64 * 64];
    five_clustered_specks(&mut mask, 64, 0, 0);

    let mut inp = fixed_inpainter();
    let report = heal(&mut img, &mask, Some(&mut inp)).unwrap();

    assert_eq!(inp.calls(), 1, "clustered specks should share one window");
    assert_eq!(report.inpainted, 5);

    let (PixelData::U16(a), PixelData::U16(b)) = (&original.data, &img.data) else {
        panic!("depth changed");
    };
    let mut unchanged_masked = 0usize;
    for i in 0..64 * 64 {
        if mask[i] {
            if (0..3).all(|c| a[i * 3 + c] == b[i * 3 + c]) {
                unchanged_masked += 1;
            }
        } else {
            for c in 0..3 {
                assert_eq!(a[i * 3 + c], b[i * 3 + c], "pixel {i} changed outside mask");
            }
        }
    }
    assert!(
        unchanged_masked < 10,
        "{unchanged_masked} masked pixels untouched -- a speck was missed"
    );
}

/// Progress advances monotonically by member count and ends at (total,
/// total) even when a group heals more than one defect per callback.
#[test]
fn clustered_specks_progress_ends_at_total() {
    let mut img = noisy_image(64, 64);
    let mut mask = vec![false; 64 * 64];
    five_clustered_specks(&mut mask, 64, 0, 0);

    let mut inp = fixed_inpainter();
    let mut calls: Vec<(usize, usize)> = Vec::new();
    let report =
        fd_heal::heal_with_progress(&mut img, &mask, Some(&mut inp), &mut |done, total| {
            calls.push((done, total));
        })
        .unwrap();

    assert_eq!(report.inpainted, 5);
    assert_eq!(calls, vec![(5, 5)]);
}

/// Two clusters of five specks each, placed far enough apart that their
/// margin-expanded bboxes never touch, must NOT share a window: two
/// clusters, two inpainter calls.
#[test]
fn far_apart_clusters_get_separate_calls() {
    let mut img = noisy_image(300, 64);
    let original = img.clone();
    let mut mask = vec![false; 300 * 64];
    five_clustered_specks(&mut mask, 300, 0, 0);
    five_clustered_specks(&mut mask, 300, 200, 0); // >150px from the first cluster

    let mut inp = fixed_inpainter();
    let report = heal(&mut img, &mask, Some(&mut inp)).unwrap();

    assert_eq!(inp.calls(), 2, "far-apart clusters must not share a window");
    assert_eq!(report.inpainted, 10);

    let (PixelData::U16(a), PixelData::U16(b)) = (&original.data, &img.data) else {
        panic!("depth changed");
    };
    for i in 0..300 * 64 {
        if !mask[i] {
            for c in 0..3 {
                assert_eq!(a[i * 3 + c], b[i * 3 + c], "pixel {i} changed outside mask");
            }
        }
    }
}

/// Pins the Task 2 batching win as a regression test: 8 well-separated
/// clusters of 5 specks each (40 defects total) must heal through roughly
/// one inpainter call per cluster, not one call per defect.
///
/// Per-defect healing (grouping disabled) would cost >= 40 calls: each of
/// the 40 defects sits comfortably inside a single 48px window interior on
/// its own, so an ungrouped heal makes exactly one call per defect (the
/// same shape `far_apart_clusters_get_separate_calls` above shows for two
/// isolated clusters, scaled up). This was confirmed locally by setting
/// the grouping gap to 0 (`n as u32 / 8` -> `0` in heal.rs, so `touches`'s
/// `span < 2*gap` is never true and every defect is its own singleton
/// group): `inp.calls()` came back 40 and the `<= 12` assertion below
/// failed loudly. Reverted after confirming; see the Task 3 report for the
/// full before/after output.
///
/// Batched healing collapses each cluster (union bbox 32x32, comfortably
/// inside the 48px interior) to a single shared window, so the expected
/// shape is 8 clusters x 1-2 windows each -- asserted here as a <= 12
/// bound with headroom, not an exact count, so the test does not become
/// brittle to incidental window-boundary shifts.
///
/// Geometry: a 4x2 grid of clusters on a 200px pitch in both axes. Each
/// cluster's own footprint is 32x32 (`five_clustered_specks`'s doc
/// comment), so neighbouring clusters sit 200 - 32 = 168px apart --
/// ten times the 16px merge threshold (2 * margin, margin = 64 / 8 = 8),
/// so no two clusters can merge even diagonally. The spec's starting
/// point was a 2000x2000 image; the call-count ratio this test pins does
/// not depend on image size once clusters clear the merge threshold, so
/// the image is sized down to 742x342 (just enough to fit the grid plus
/// slack) to keep the test fast -- disclosed here rather than silently
/// shrunk.
#[test]
fn eight_clusters_share_windows_not_per_defect_calls() {
    let (width, height) = (742u32, 342u32);
    let mut img = noisy_image(width, height);
    let original = img.clone();
    let mut mask = vec![false; (width * height) as usize];
    for row in 0..2u32 {
        for col in 0..4u32 {
            five_clustered_specks(&mut mask, width, col * 200, row * 200);
        }
    }

    let mut inp = fixed_inpainter();
    let report = heal(&mut img, &mask, Some(&mut inp)).unwrap();

    assert_eq!(report.inpainted, 40);
    assert!(
        inp.calls() <= 12,
        "expected <= 12 batched calls for 8 clusters of 5 specks each, got {} \
         (per-defect healing of the same 40 defects would cost >= 40)",
        inp.calls()
    );

    let (PixelData::U16(a), PixelData::U16(b)) = (&original.data, &img.data) else {
        panic!("depth changed");
    };
    for i in 0..(width * height) as usize {
        if !mask[i] {
            for c in 0..3 {
                assert_eq!(a[i * 3 + c], b[i * 3 + c], "pixel {i} changed outside mask");
            }
        }
    }
}
