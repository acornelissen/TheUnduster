use fd_heal::dilate;

fn mask_from(rows: &[&str]) -> (Vec<bool>, u32, u32) {
    let h = rows.len() as u32;
    let w = rows[0].len() as u32;
    let m = rows
        .iter()
        .flat_map(|r| r.chars().map(|c| c == '#'))
        .collect();
    (m, w, h)
}

fn render(mask: &[bool], w: u32) -> Vec<String> {
    mask.chunks(w as usize)
        .map(|row| row.iter().map(|&b| if b { '#' } else { '.' }).collect())
        .collect()
}

#[test]
fn radius_zero_is_identity() {
    let (m, w, h) = mask_from(&["..#..", ".....", "#...#"]);
    assert_eq!(dilate(&m, w, h, 0), m);
}

#[test]
fn single_pixel_dilates_to_a_clamped_square() {
    let (m, w, h) = mask_from(&[".....", "..#..", ".....", ".....", "....."]);
    let out = dilate(&m, w, h, 1);
    assert_eq!(
        render(&out, w),
        vec![".###.", ".###.", ".###.", ".....", "....."]
    );
}

#[test]
fn dilation_clamps_at_borders() {
    let (m, w, h) = mask_from(&["#....", ".....", ".....", ".....", "....#"]);
    let out = dilate(&m, w, h, 2);
    let rows = render(&out, w);
    assert_eq!(rows[0], "###..");
    // row 2 is reached by BOTH corners: (0,0) covers cols 0-2, (4,4) covers 2-4
    assert_eq!(rows[2], "#####");
    assert_eq!(rows[4], "..###");
}

#[test]
fn radius_two_covers_a_defect_rim() {
    // The product case: a 2px-under-covering mask grows to swallow the rim.
    let (m, w, h) = mask_from(&[
        ".......", //
        "...#...", //
        "..###..", //
        "...#...", //
        ".......", //
    ]);
    let out = dilate(&m, w, h, 2);
    // every pixel within Chebyshev distance 2 of a set pixel is set
    assert!(out.iter().filter(|&&b| b).count() > m.iter().filter(|&&b| b).count());
    assert!(out[0]); // (0,0) is within 2 of (2,2)
}
