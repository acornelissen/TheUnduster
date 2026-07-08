use std::sync::Mutex;

use crate::images::Images;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Layer {
    Rgba,
    Probs,
    Thumb,
}

/// Parses:
/// - "/{id}/{level}/{tx}/{ty}" (Rgba)
/// - "/probs/{id}/{level}/{tx}/{ty}" (Probs)
/// - "/thumb/{index}" (Thumb; level/tx/ty are unused and returned as 0)
///
/// Leading slash optional on all three forms.
pub fn parse_tile_path(path: &str) -> Option<(Layer, u64, u8, u32, u32)> {
    let trimmed = path.trim_start_matches('/');
    if let Some(rest) = trimmed.strip_prefix("thumb/") {
        if rest.is_empty() {
            return None;
        }
        let mut parts = rest.split('/');
        let index = parts.next()?.parse().ok()?;
        if parts.next().is_some() {
            return None;
        }
        return Some((Layer::Thumb, index, 0, 0, 0));
    }
    let (layer, rest) = match trimmed.strip_prefix("probs/") {
        Some(rest) => (Layer::Probs, rest),
        None => (Layer::Rgba, trimmed),
    };
    let mut parts = rest.split('/');
    let id = parts.next()?.parse().ok()?;
    let level = parts.next()?.parse().ok()?;
    let tx = parts.next()?.parse().ok()?;
    let ty = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((layer, id, level, tx, ty))
}

pub fn tile_response(
    images: &Mutex<Images>,
    roll: &Mutex<Option<crate::roll::Roll>>,
    path: &str,
) -> tauri::http::Response<Vec<u8>> {
    let respond = |status: u16, body: Vec<u8>, w: u32, h: u32| {
        let mut builder = tauri::http::Response::builder()
            .status(status)
            .header("Content-Type", "application/octet-stream")
            .header("Access-Control-Allow-Origin", "*")
            // Without this, cross-origin JS sees the custom headers as null
            // and uploads 0x0 textures: a blank canvas with no errors.
            .header(
                "Access-Control-Expose-Headers",
                "x-tile-width, x-tile-height",
            );
        if status == 200 {
            builder = builder
                .header("x-tile-width", w.to_string())
                .header("x-tile-height", h.to_string());
        }
        builder
            .body(body)
            .expect("static response headers are valid")
    };
    let respond_png = |status: u16, body: Vec<u8>| {
        tauri::http::Response::builder()
            .status(status)
            .header("Content-Type", "image/png")
            .header("Access-Control-Allow-Origin", "*")
            .body(body)
            .expect("static response headers are valid")
    };
    let Some((layer, id, level, tx, ty)) = parse_tile_path(path) else {
        #[cfg(debug_assertions)]
        eprintln!("[tiles] 400 malformed path: {path}");
        return respond(400, Vec::new(), 0, 0);
    };
    match layer {
        Layer::Thumb => {
            let index = id as usize;
            let Ok(roll_guard) = roll.lock() else {
                return respond_png(500, Vec::new());
            };
            let Some(roll) = roll_guard.as_ref() else {
                #[cfg(debug_assertions)]
                eprintln!("[tiles] 404 thumb {path}: no roll open");
                return respond_png(404, Vec::new());
            };
            if index >= roll.frames.len() {
                #[cfg(debug_assertions)]
                eprintln!(
                    "[tiles] 404 thumb {path}: index out of range ({} frames)",
                    roll.frames.len()
                );
                return respond_png(404, Vec::new());
            }
            let thumb_path = crate::roll::thumb_path(&roll.dir, index);
            match std::fs::read(&thumb_path) {
                Ok(bytes) => respond_png(200, bytes),
                Err(_) => {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "[tiles] 404 thumb {path}: {} not yet written",
                        thumb_path.display()
                    );
                    respond_png(404, Vec::new())
                }
            }
        }
        Layer::Rgba | Layer::Probs => {
            let Ok(mut images) = images.lock() else {
                return respond(500, Vec::new(), 0, 0);
            };
            match layer {
                Layer::Rgba => match images.tile(id, level, tx, ty) {
                    Some(tile) => respond(200, tile.rgba.clone(), tile.width, tile.height),
                    None => {
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "[tiles] 404 rgba {path}: known image ids {:?}",
                            images.known_ids()
                        );
                        respond(404, Vec::new(), 0, 0)
                    }
                },
                Layer::Probs => match images.prob_tile(id, level, tx, ty) {
                    Some((w, h, bytes)) => respond(200, bytes, w, h),
                    None => {
                        // Expected pre-detection; still logged in dev so blank-canvas
                        // sessions can tell harmless probs 404s from real rgba ones.
                        #[cfg(debug_assertions)]
                        eprintln!("[tiles] 404 probs {path} (no detection yet is normal)");
                        respond(404, Vec::new(), 0, 0)
                    }
                },
                Layer::Thumb => unreachable!(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_paths() {
        assert_eq!(parse_tile_path("/3/1/7/2"), Some((Layer::Rgba, 3, 1, 7, 2)));
        assert_eq!(parse_tile_path("3/1/7/2"), Some((Layer::Rgba, 3, 1, 7, 2)));
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(parse_tile_path("/3/1/7"), None);
        assert_eq!(parse_tile_path("/3/1/7/2/9"), None);
        assert_eq!(parse_tile_path("/a/b/c/d"), None);
        assert_eq!(parse_tile_path(""), None);
        assert_eq!(parse_tile_path("/-1/0/0/0"), None);
    }

    #[test]
    fn parses_probs_layer() {
        assert_eq!(
            parse_tile_path("/probs/3/1/7/2"),
            Some((Layer::Probs, 3, 1, 7, 2))
        );
        assert_eq!(parse_tile_path("/3/1/7/2"), Some((Layer::Rgba, 3, 1, 7, 2)));
        assert_eq!(parse_tile_path("/probs/3/1/7"), None);
        assert_eq!(parse_tile_path("/unknown/3/1/7/2"), None);
    }

    #[test]
    fn parses_thumb_layer() {
        assert_eq!(
            parse_tile_path("/thumb/7"),
            Some((Layer::Thumb, 7, 0, 0, 0))
        );
        assert_eq!(parse_tile_path("thumb/0"), Some((Layer::Thumb, 0, 0, 0, 0)));
        assert_eq!(parse_tile_path("/thumb/"), None);
        assert_eq!(parse_tile_path("/thumb/x"), None);
        assert_eq!(parse_tile_path("/thumb/7/8"), None);
    }

    #[test]
    fn missing_tile_is_404_and_malformed_is_400() {
        let images = Mutex::new(Images::default());
        let roll = Mutex::new(None);
        assert_eq!(tile_response(&images, &roll, "/1/0/0/0").status(), 404);
        assert_eq!(tile_response(&images, &roll, "/nope").status(), 400);
    }

    #[test]
    fn probs_tile_404_before_detection() {
        let images = Mutex::new(Images::default());
        let roll = Mutex::new(None);
        assert_eq!(
            tile_response(&images, &roll, "/probs/1/0/0/0").status(),
            404
        );
    }

    #[test]
    fn thumb_404_when_no_roll_open() {
        let images = Mutex::new(Images::default());
        let roll = Mutex::new(None);
        assert_eq!(tile_response(&images, &roll, "/thumb/0").status(), 404);
    }

    #[test]
    fn thumb_404_when_file_not_yet_written() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.tif"), b"x").unwrap();
        let images = Mutex::new(Images::default());
        let opened = crate::roll::Roll::open(dir.path()).unwrap();
        let roll = Mutex::new(Some(opened));
        // frame 0 exists but its thumbnail was never written by the queue
        assert_eq!(tile_response(&images, &roll, "/thumb/0").status(), 404);
        // out-of-range index also 404s, not a panic
        assert_eq!(tile_response(&images, &roll, "/thumb/9").status(), 404);
    }

    #[test]
    fn thumb_200_serves_png_bytes_with_the_right_content_type() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.tif"), b"x").unwrap();
        let opened = crate::roll::Roll::open(dir.path()).unwrap();
        let thumb_path = crate::roll::thumb_path(&opened.dir, 0);
        let rgba = vec![10u8, 20, 30, 255];
        crate::roll::write_thumbnail(&rgba, 1, 1, &thumb_path).unwrap();
        let images = Mutex::new(Images::default());
        let roll = Mutex::new(Some(opened));
        let resp = tile_response(&images, &roll, "/thumb/0");
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.headers().get("Content-Type").unwrap(), "image/png");
        assert!(!resp.body().is_empty());
    }
}
