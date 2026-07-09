//! Manual heal-brush strokes: vector data (image-pixel coordinates), only
//! rasterized onto a mask at heal or export time. Strokes stay resolution-
//! independent, sidecar-friendly, and trivially undoable; the operator's
//! painted pixels are exactly what heals (no dilation on manual strokes).

use serde::{Deserialize, Serialize};

pub const MAX_STROKES: usize = 512;
pub const MAX_POINTS_PER_STROKE: usize = 4096;
pub const MIN_RADIUS: f32 = 1.0;
pub const MAX_RADIUS: f32 = 512.0;
const MAX_COORD: f32 = 1e7;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stroke {
    pub erase: bool,
    pub radius: f32,
    pub points: Vec<[f32; 2]>,
}

/// Boundary validation for stroke lists arriving over IPC or loaded from a
/// sidecar. Coordinates may exceed the image (drags can exit the canvas);
/// the rasterizer clamps. Non-finite values and absurd magnitudes are not
/// drawing, they are malformed input.
pub fn validate_strokes(strokes: &[Stroke]) -> Result<(), String> {
    if strokes.len() > MAX_STROKES {
        return Err(format!(
            "too many strokes ({} > {MAX_STROKES})",
            strokes.len()
        ));
    }
    for s in strokes {
        if !s.radius.is_finite() || !(MIN_RADIUS..=MAX_RADIUS).contains(&s.radius) {
            return Err(format!("stroke radius {} out of range", s.radius));
        }
        if s.points.is_empty() || s.points.len() > MAX_POINTS_PER_STROKE {
            return Err(format!(
                "stroke point count {} out of range",
                s.points.len()
            ));
        }
        for p in &s.points {
            if !p[0].is_finite()
                || !p[1].is_finite()
                || p[0].abs() > MAX_COORD
                || p[1].abs() > MAX_COORD
            {
                return Err("stroke coordinate out of range".to_string());
            }
        }
    }
    Ok(())
}

/// The one place mask composition order is defined: detector probs
/// thresholded, dilated (model masks under-cover; see fd_heal::dilate),
/// then manual strokes in chronological order, undilated.
pub fn compose_heal_mask(
    probs_mask: Vec<bool>,
    width: u32,
    height: u32,
    dilate_radius: u32,
    strokes: &[Stroke],
) -> Vec<bool> {
    let mut mask = fd_heal::dilate(&probs_mask, width, height, dilate_radius);
    apply_strokes(&mut mask, width, height, strokes);
    mask
}

/// Rasterizes strokes onto `mask` in order. Paint sets pixels; erase clears
/// them -- including pixels the detector set, which is the point of `e`.
/// Mismatched inputs (zero dimensions or mask length != width * height) are a
/// no-op, treating degenerate sidecar data as invalid rather than panicking.
pub fn apply_strokes(mask: &mut [bool], width: u32, height: u32, strokes: &[Stroke]) {
    // Early exit on zero dimensions to avoid clamp panics.
    if width == 0 || height == 0 {
        return;
    }

    // Validate mask length matches dimensions before rasterizing.
    let expected_len = width as usize * height as usize;
    if mask.len() != expected_len {
        return;
    }

    for s in strokes {
        let value = !s.erase;
        if s.points.len() == 1 {
            stamp_capsule(
                mask,
                width,
                height,
                s.points[0],
                s.points[0],
                s.radius,
                value,
            );
        } else {
            for pair in s.points.windows(2) {
                stamp_capsule(mask, width, height, pair[0], pair[1], s.radius, value);
            }
        }
    }
}

/// Fills every pixel within `radius` of segment ab (Euclidean round brush).
fn stamp_capsule(
    mask: &mut [bool],
    width: u32,
    height: u32,
    a: [f32; 2],
    b: [f32; 2],
    radius: f32,
    value: bool,
) {
    let (w, h) = (width as f32, height as f32);
    let x0 = (a[0].min(b[0]) - radius).floor().clamp(0.0, w - 1.0) as usize;
    let x1 = (a[0].max(b[0]) + radius).ceil().clamp(0.0, w - 1.0) as usize;
    let y0 = (a[1].min(b[1]) - radius).floor().clamp(0.0, h - 1.0) as usize;
    let y1 = (a[1].max(b[1]) + radius).ceil().clamp(0.0, h - 1.0) as usize;
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ab_len_sq = ab[0] * ab[0] + ab[1] * ab[1];
    let r_sq = radius * radius;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let p = [x as f32 + 0.5, y as f32 + 0.5];
            let ap = [p[0] - a[0], p[1] - a[1]];
            let t = if ab_len_sq > 0.0 {
                ((ap[0] * ab[0] + ap[1] * ab[1]) / ab_len_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let dx = ap[0] - t * ab[0];
            let dy = ap[1] - t * ab[1];
            if dx * dx + dy * dy <= r_sq {
                mask[y * width as usize + x] = value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dab(x: f32, y: f32, radius: f32, erase: bool) -> Stroke {
        Stroke {
            erase,
            radius,
            points: vec![[x, y]],
        }
    }

    #[test]
    fn compose_paints_beyond_detection_and_erase_protects() {
        let (w, h) = (32u32, 32u32);
        let mut probs_mask = vec![false; (w * h) as usize];
        probs_mask[(10 * w + 10) as usize] = true; // one detected pixel
        let strokes = vec![
            Stroke {
                erase: false,
                radius: 2.0,
                points: vec![[25.0, 25.0]],
            },
            Stroke {
                erase: true,
                radius: 4.0,
                points: vec![[10.0, 10.0]],
            },
        ];
        let mask = compose_heal_mask(probs_mask, w, h, 2, &strokes);
        assert!(mask[(25 * w + 25) as usize]); // painted region heals
        assert!(!mask[(10 * w + 10) as usize]); // erased detection protected
                                                // the erase covered the dilated ring too (radius 4 > dilate 2)
        assert!(!mask[(10 * w + 12) as usize]);
    }

    #[test]
    fn dab_fills_a_disc() {
        let mut mask = vec![false; 32 * 32];
        apply_strokes(&mut mask, 32, 32, &[dab(16.0, 16.0, 5.0, false)]);
        let count = mask.iter().filter(|&&b| b).count();
        // area of a radius-5 disc is ~78.5; rasterized count is close
        assert!((70..=90).contains(&count), "disc had {count} px");
        assert!(mask[16 * 32 + 16]); // center set
        assert!(!mask[0]); // far corner untouched
    }

    #[test]
    fn segment_fills_a_capsule_not_two_discs() {
        let mut mask = vec![false; 64 * 16];
        let stroke = Stroke {
            erase: false,
            radius: 3.0,
            points: vec![[10.0, 8.0], [50.0, 8.0]],
        };
        apply_strokes(&mut mask, 64, 16, &[stroke]);
        // midpoint between the two points must be covered
        assert!(mask[8 * 64 + 30]);
    }

    #[test]
    fn erase_wins_when_later() {
        let mut mask = vec![false; 32 * 32];
        let paint = dab(16.0, 16.0, 6.0, false);
        let erase = Stroke {
            erase: true,
            radius: 3.0,
            points: vec![[16.0, 16.0]],
        };
        apply_strokes(&mut mask, 32, 32, &[paint.clone(), erase.clone()]);
        assert!(!mask[16 * 32 + 16]); // center erased
        assert!(mask[16 * 32 + 21]); // ring outside erase radius still painted

        // order flipped: paint over erase wins
        let mut mask2 = vec![false; 32 * 32];
        apply_strokes(&mut mask2, 32, 32, &[erase, paint]);
        assert!(mask2[16 * 32 + 16]);
    }

    #[test]
    fn erase_removes_preexisting_mask() {
        // the detector-mask case: erase must clear pixels it never painted
        let mut mask = vec![true; 16 * 16];
        apply_strokes(&mut mask, 16, 16, &[dab(8.0, 8.0, 2.0, true)]);
        assert!(!mask[8 * 16 + 8]);
        assert!(mask[0]);
    }

    #[test]
    fn out_of_bounds_points_clamp_instead_of_panicking() {
        let mut mask = vec![false; 8 * 8];
        let stroke = Stroke {
            erase: false,
            radius: 4.0,
            points: vec![[-3.0, -3.0], [12.0, 12.0]],
        };
        apply_strokes(&mut mask, 8, 8, &[stroke]);
        assert!(mask[0]); // corner covered by the capsule passing through
    }

    #[test]
    fn validation_rejects_bad_strokes() {
        assert!(validate_strokes(&[dab(1.0, 1.0, 5.0, false)]).is_ok());
        assert!(validate_strokes(&[dab(1.0, 1.0, 0.2, false)]).is_err()); // radius too small
        assert!(validate_strokes(&[dab(1.0, 1.0, 9999.0, false)]).is_err()); // too large
        assert!(validate_strokes(&[dab(f32::NAN, 1.0, 5.0, false)]).is_err());
        assert!(validate_strokes(&[Stroke {
            erase: false,
            radius: 5.0,
            points: vec![],
        }])
        .is_err()); // empty stroke
        let too_many: Vec<Stroke> = (0..MAX_STROKES + 1)
            .map(|_| dab(1.0, 1.0, 5.0, false))
            .collect();
        assert!(validate_strokes(&too_many).is_err());
    }

    #[test]
    fn zero_dimensions_are_a_no_op() {
        // width == 0 case: empty mask
        let mut mask = vec![];
        apply_strokes(&mut mask, 0, 0, &[dab(1.0, 1.0, 4.0, false)]);
        assert!(mask.is_empty());

        // 0 x 8 case: empty mask
        let mut mask = vec![];
        apply_strokes(&mut mask, 0, 8, &[dab(1.0, 1.0, 4.0, false)]);
        assert!(mask.is_empty());
    }

    #[test]
    fn mismatched_mask_length_is_a_no_op() {
        let mut mask = vec![false; 10];
        apply_strokes(&mut mask, 8, 8, &[dab(4.0, 4.0, 2.0, false)]);
        // mask should be unchanged, all false
        assert!(mask.iter().all(|&b| !b));
    }
}
