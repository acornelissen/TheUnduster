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
