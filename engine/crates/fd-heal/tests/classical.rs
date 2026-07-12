use fd_heal::{classical_fill, components};

#[test]
fn fill_replaces_speck_with_surround_median() {
    let (w, h) = (16u32, 16u32);
    let mut plane = vec![0.5f32; (w * h) as usize];
    let mut mask = vec![false; (w * h) as usize];
    // 2x2 dark speck at (7,7)
    for y in 7..9u32 {
        for x in 7..9u32 {
            plane[(y * w + x) as usize] = 0.05;
            mask[(y * w + x) as usize] = true;
        }
    }
    let defects = components(&mask, w, h);
    assert_eq!(defects.len(), 1);
    let mut planes = vec![plane];
    classical_fill(&mut planes, w, h, &defects[0], &mask);
    for y in 7..9u32 {
        for x in 7..9u32 {
            let v = planes[0][(y * w + x) as usize];
            assert!((v - 0.5).abs() < 1e-6, "pixel ({x},{y}) = {v}, want 0.5");
        }
    }
}

#[test]
fn fill_samples_only_clean_pixels() {
    // a 4-connected run of masked pixels with extreme values: the fill for
    // every pixel must come from clean 0.8 pixels, never other masked ones
    let (w, h) = (16u32, 16u32);
    let mut plane = vec![0.8f32; (w * h) as usize];
    let mut mask = vec![false; (w * h) as usize];
    for x in 8..12u32 {
        plane[(8 * w + x) as usize] = 0.0;
        mask[(8 * w + x) as usize] = true;
    }
    let defects = components(&mask, w, h);
    assert_eq!(defects.len(), 1);
    let target = &defects[0];
    assert_eq!(target.pixels.len(), 4);
    let mut planes = vec![plane];
    classical_fill(&mut planes, w, h, target, &mask);
    for (x, y) in &target.pixels {
        let v = planes[0][(y * w + x) as usize];
        assert!((v - 0.8).abs() < 1e-6);
    }
}

#[test]
fn fill_reaches_the_core_of_a_large_defect() {
    // A 40px-radius disk is far deeper than any fixed sample window: the old
    // fill (expanding window capped at radius 16, needing 5 clean samples)
    // left ~half of a disk this size untouched -- a release build with no
    // model would export brush strokes with the dust intact in the center.
    // Onion-peel filling must reach every pixel: filled pixels become sample
    // sources for the layer inside them.
    let (w, h) = (128u32, 128u32);
    let core = 0.05f32;
    let mut plane = vec![0.8f32; (w * h) as usize];
    let mut mask = vec![false; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as i64 - 64, y as i64 - 64);
            if dx * dx + dy * dy <= 40 * 40 {
                let i = (y * w + x) as usize;
                plane[i] = core;
                mask[i] = true;
            }
        }
    }
    let defects = components(&mask, w, h);
    assert_eq!(defects.len(), 1);
    let mut planes = vec![plane];
    classical_fill(&mut planes, w, h, &defects[0], &mask);

    // Every masked pixel must have been pulled away from the defect value
    // toward the clean surround -- no untouched core.
    let mut untouched = 0usize;
    for &(x, y) in &defects[0].pixels {
        if (planes[0][(y * w + x) as usize] - core).abs() < 1e-6 {
            untouched += 1;
        }
    }
    assert_eq!(untouched, 0, "{untouched} core pixels never filled");
    // And the fill must come from the clean surround, not smeared defect
    // values: on a uniform 0.8 background every filled pixel is exactly 0.8.
    for &(x, y) in &defects[0].pixels {
        let v = planes[0][(y * w + x) as usize];
        assert!((v - 0.8).abs() < 1e-6, "pixel ({x},{y}) = {v}, want 0.8");
    }
}
