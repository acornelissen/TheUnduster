use fd_heal::components;

fn mask_from(rows: &[&str]) -> (Vec<bool>, u32, u32) {
    let h = rows.len() as u32;
    let w = rows[0].len() as u32;
    let m = rows
        .iter()
        .flat_map(|r| r.chars().map(|c| c == '#'))
        .collect();
    (m, w, h)
}

#[test]
fn finds_separate_components_with_4_connectivity() {
    let (m, w, h) = mask_from(&[
        "##....", //
        "##....", //
        ".....#", // diagonal-only touch from the blob below => separate
        "....#.", //
    ]);
    let comps = components(&m, w, h);
    assert_eq!(comps.len(), 3);
    let sizes: Vec<usize> = {
        let mut s: Vec<usize> = comps.iter().map(|c| c.pixels.len()).collect();
        s.sort();
        s
    };
    assert_eq!(sizes, vec![1, 1, 4]);
}

#[test]
fn bbox_is_tight_and_exclusive() {
    let (m, w, h) = mask_from(&["....", ".##.", "....", "...."]);
    let comps = components(&m, w, h);
    assert_eq!(comps.len(), 1);
    let b = &comps[0].bbox;
    assert_eq!((b.x0, b.y0, b.x1, b.y1), (1, 1, 3, 2));
    assert_eq!(comps[0].max_dim(), 2);
}

#[test]
fn empty_mask_no_components() {
    let (m, w, h) = mask_from(&["....", "...."]);
    assert!(components(&m, w, h).is_empty());
}
