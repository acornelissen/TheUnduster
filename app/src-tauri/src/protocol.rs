use std::sync::Mutex;

use crate::images::Images;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Layer {
    Rgba,
    Probs,
}

/// Parse "/{id}/{level}/{tx}/{ty}" or "/probs/{id}/{level}/{tx}/{ty}" (leading slash optional).
/// Returns (Layer, id, level, tx, ty).
pub fn parse_tile_path(path: &str) -> Option<(Layer, u64, u8, u32, u32)> {
    let trimmed = path.trim_start_matches('/');
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

pub fn tile_response(images: &Mutex<Images>, path: &str) -> tauri::http::Response<Vec<u8>> {
    let respond = |status: u16, body: Vec<u8>, w: u32, h: u32| {
        let mut builder = tauri::http::Response::builder()
            .status(status)
            .header("Content-Type", "application/octet-stream")
            .header("Access-Control-Allow-Origin", "*");
        if status == 200 {
            builder = builder
                .header("x-tile-width", w.to_string())
                .header("x-tile-height", h.to_string());
        }
        builder
            .body(body)
            .expect("static response headers are valid")
    };
    let Some((layer, id, level, tx, ty)) = parse_tile_path(path) else {
        #[cfg(debug_assertions)]
        eprintln!("[tiles] 400 malformed path: {path}");
        return respond(400, Vec::new(), 0, 0);
    };
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
    fn missing_tile_is_404_and_malformed_is_400() {
        let images = Mutex::new(Images::default());
        assert_eq!(tile_response(&images, "/1/0/0/0").status(), 404);
        assert_eq!(tile_response(&images, "/nope").status(), 400);
    }

    #[test]
    fn probs_tile_404_before_detection() {
        let images = Mutex::new(Images::default());
        assert_eq!(tile_response(&images, "/probs/1/0/0/0").status(), 404);
    }
}
