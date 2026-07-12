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

#[test]
fn components_up_to_stops_at_the_limit_with_the_same_scan_order_prefix() {
    // Five separate specks; a limit of 3 must return exactly the first
    // three the full walk would have found (row-major scan order), so
    // capping the WALK is behavior-identical to capping the returned list
    // -- it just stops paying for a pathological mask's tail.
    let (m, w, h) = mask_from(&["#.#.#", ".....", "#.#.."]);
    let full = fd_heal::components(&m, w, h);
    assert_eq!(full.len(), 5);
    let capped = fd_heal::components_up_to(&m, w, h, 3);
    assert_eq!(capped.len(), 3);
    for (a, b) in capped.iter().zip(full.iter()) {
        assert_eq!(a.pixels, b.pixels);
    }
    // Zero cap: nothing, immediately. Limit above the count: everything.
    assert!(fd_heal::components_up_to(&m, w, h, 0).is_empty());
    assert_eq!(fd_heal::components_up_to(&m, w, h, 99).len(), 5);
}
