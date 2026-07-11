use std::collections::BTreeMap;

use crate::components::{Bbox, Defect};

/// A batch of defect indices healed through shared windows. `bbox` is the
/// union of the members' bboxes (exclusive upper bounds, like Bbox).
#[derive(Debug, Clone)]
pub struct Group {
    pub members: Vec<usize>, // indices into the caller's defect slice
    pub bbox: Bbox,
}

/// Groups defects whose margin-expanded bboxes touch or overlap, by
/// union-find with transitive merging. Two defects group when expanding
/// each bbox by `gap` in every direction (saturating at 0) makes them
/// intersect -- i.e. clustered specks whose per-defect windows would
/// largely overlap. `gap` is the window margin (n/8), so grouped members
/// sit close enough that a shared window's context still surrounds them.
/// Singleton results are the degenerate case and carry one member each.
/// Deterministic: members ascend within a group; groups order by their
/// first member.
pub fn group_defects(defects: &[Defect], gap: u32) -> Vec<Group> {
    let n = defects.len();
    let mut parent: Vec<usize> = (0..n).collect();

    // O(k^2) pairwise pass: k is the defect count per frame, dozens at
    // most, so the quadratic scan is negligible next to the inpaint cost
    // it exists to reduce.
    for i in 0..n {
        for j in (i + 1)..n {
            if touches(defects[i].bbox, defects[j].bbox, gap) {
                union(&mut parent, i, j);
            }
        }
    }

    // `union` always attaches the higher root to the lower one, so the
    // canonical root of every component is its smallest member index.
    // Folding into a BTreeMap keyed by root therefore yields both
    // properties the doc comment promises for free: members are pushed in
    // ascending order (i walks 0..n), and iterating the map's keys visits
    // groups in ascending order of their first (= smallest = root) member.
    let mut by_root: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        by_root.entry(root).or_default().push(i);
    }

    by_root
        .into_values()
        .map(|members| {
            let bbox = union_bbox(defects, &members);
            Group { members, bbox }
        })
        .collect()
}

fn union_bbox(defects: &[Defect], members: &[usize]) -> Bbox {
    let mut acc = defects[members[0]].bbox;
    for &idx in &members[1..] {
        let b = defects[idx].bbox;
        acc = Bbox {
            x0: acc.x0.min(b.x0),
            y0: acc.y0.min(b.y0),
            x1: acc.x1.max(b.x1),
            y1: acc.y1.max(b.y1),
        };
    }
    acc
}

/// Expand `b` by `gap` in every direction, saturating at 0 on the low
/// sides (the high sides saturate at `u32::MAX` rather than overflow).
fn expand(b: Bbox, gap: u32) -> Bbox {
    Bbox {
        x0: b.x0.saturating_sub(gap),
        y0: b.y0.saturating_sub(gap),
        x1: b.x1.saturating_add(gap),
        y1: b.y1.saturating_add(gap),
    }
}

/// Exclusive-bound rectangle intersection: [x0, x1) meets [x0, x1) iff each
/// start is strictly less than the other's end, and likewise for y.
fn intersects(a: Bbox, b: Bbox) -> bool {
    a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1
}

/// True when expanding `a` by `gap` (saturating at 0) makes it intersect
/// `b`. Expanding only one side of the pair is enough: for the x axis,
/// `intersects` reduces to `expand(a).x0 < b.x1 && b.x0 < expand(a).x1`,
/// i.e. `a.x0.saturating_sub(gap) < b.x1  &&  b.x0 < a.x1 + gap`. The first
/// clause is trivially true whenever the subtraction actually saturates
/// (`a.x0 < gap` implies `b.x1 >= 1` for any non-empty bbox, hence `0 <
/// b.x1`), and reduces to `a.x0 < b.x1 + gap` otherwise -- the same
/// inequality expanding `b` instead of `a` would produce. So expanding
/// either defect's bbox and testing against the other's unexpanded bbox
/// gives an identical, symmetric result; this is exactly the effect of
/// expanding "each" bbox by `gap` that the doc comment on `group_defects`
/// describes.
fn touches(a: Bbox, b: Bbox, gap: u32) -> bool {
    intersects(expand(a, gap), b)
}

fn find(parent: &mut [usize], x: usize) -> usize {
    let mut x = x;
    while parent[x] != x {
        parent[x] = parent[parent[x]]; // path halving
        x = parent[x];
    }
    x
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra == rb {
        return;
    }
    // Attach the higher root to the lower one so every component's
    // canonical root is its smallest member index -- see the comment in
    // `group_defects` on why that gives deterministic ordering for free.
    if ra < rb {
        parent[rb] = ra;
    } else {
        parent[ra] = rb;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defect(x0: u32, y0: u32, x1: u32, y1: u32) -> Defect {
        Defect {
            pixels: vec![(x0, y0)],
            bbox: Bbox { x0, y0, x1, y1 },
        }
    }

    #[test]
    fn merges_specks_3px_apart_when_gap_is_8() {
        let defects = vec![defect(0, 0, 1, 1), defect(4, 0, 5, 1)];
        let groups = group_defects(&defects, 8);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].members, vec![0, 1]);
        let b = groups[0].bbox;
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (0, 0, 5, 1));
    }

    #[test]
    fn keeps_specks_3px_apart_separate_when_gap_is_1() {
        let defects = vec![defect(0, 0, 1, 1), defect(4, 0, 5, 1)];
        let groups = group_defects(&defects, 1);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].members, vec![0]);
        assert_eq!(groups[1].members, vec![1]);
        assert_eq!(
            (
                groups[0].bbox.x0,
                groups[0].bbox.y0,
                groups[0].bbox.x1,
                groups[0].bbox.y1
            ),
            (0, 0, 1, 1)
        );
        assert_eq!(
            (
                groups[1].bbox.x0,
                groups[1].bbox.y0,
                groups[1].bbox.x1,
                groups[1].bbox.y1
            ),
            (4, 0, 5, 1)
        );
    }

    #[test]
    fn transitive_chain_merges_into_one_group() {
        // A-B separation 4px, B-C separation 4px, A-C separation 9px.
        // gap=5 touches A-B and B-C directly but not A-C -- only the
        // union-find's transitivity via B pulls A and C into one group.
        let a = defect(0, 0, 1, 1);
        let b = defect(5, 0, 6, 1);
        let c = defect(10, 0, 11, 1);
        let defects = vec![a, b, c];

        assert!(
            !touches(defects[0].bbox, defects[2].bbox, 5),
            "A and C must not touch directly"
        );

        let groups = group_defects(&defects, 5);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].members, vec![0, 1, 2]);
        let bb = groups[0].bbox;
        assert_eq!((bb.x0, bb.y0, bb.x1, bb.y1), (0, 0, 11, 1));
    }

    #[test]
    fn lone_defect_is_a_singleton_with_its_own_bbox() {
        let defects = vec![defect(7, 9, 12, 20)];
        let groups = group_defects(&defects, 8);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].members, vec![0]);
        let b = groups[0].bbox;
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (7, 9, 12, 20));
    }

    #[test]
    fn grouping_is_invariant_under_input_order() {
        // Two clusters ({0,1} and {3,4}) plus a lone defect (2), gap=3.
        let defects = vec![
            defect(0, 0, 1, 1),   // 0 -- clusters with 1 (1px apart)
            defect(2, 0, 3, 1),   // 1
            defect(20, 0, 21, 1), // 2 -- isolated
            defect(30, 0, 31, 1), // 3 -- clusters with 4 (2px apart)
            defect(33, 0, 34, 1), // 4
        ];
        let gap = 3;

        let baseline = group_defects(&defects, gap);
        let mut baseline_sets: Vec<Vec<usize>> = baseline
            .iter()
            .map(|g| {
                let mut m = g.members.clone();
                m.sort_unstable();
                m
            })
            .collect();
        baseline_sets.sort();

        // Fixed permutation (no RNG): shuffled[k] = defects[perm[k]].
        let perm = [3usize, 0, 4, 2, 1];
        let shuffled: Vec<Defect> = perm.iter().map(|&i| defects[i].clone()).collect();
        let shuffled_groups = group_defects(&shuffled, gap);

        let mut mapped_sets: Vec<Vec<usize>> = shuffled_groups
            .iter()
            .map(|g| {
                let mut m: Vec<usize> = g.members.iter().map(|&pos| perm[pos]).collect();
                m.sort_unstable();
                m
            })
            .collect();
        mapped_sets.sort();

        assert_eq!(baseline_sets, mapped_sets);
    }

    #[test]
    fn expansion_at_the_origin_with_a_huge_gap_does_not_underflow() {
        let defects = vec![defect(0, 0, 2, 2), defect(9, 0, 11, 2)];
        // Must not panic (saturating_sub of 0 - u32::MAX would underflow
        // if implemented with plain subtraction).
        let groups = group_defects(&defects, u32::MAX);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].members, vec![0, 1]);
        let b = groups[0].bbox;
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (0, 0, 11, 2));
    }
}
