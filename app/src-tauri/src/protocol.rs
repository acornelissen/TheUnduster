use std::sync::Mutex;

use crate::images::Images;

/// Parse "/{id}/{level}/{tx}/{ty}" (leading slash optional).
pub fn parse_tile_path(path: &str) -> Option<(u64, u8, u32, u32)> {
    let mut parts = path.trim_start_matches('/').split('/');
    let id = parts.next()?.parse().ok()?;
    let level = parts.next()?.parse().ok()?;
    let tx = parts.next()?.parse().ok()?;
    let ty = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((id, level, tx, ty))
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
    let Some((id, level, tx, ty)) = parse_tile_path(path) else {
        return respond(400, Vec::new(), 0, 0);
    };
    let Ok(mut images) = images.lock() else {
        return respond(500, Vec::new(), 0, 0);
    };
    match images.tile(id, level, tx, ty) {
        Some(tile) => respond(200, tile.rgba.clone(), tile.width, tile.height),
        None => respond(404, Vec::new(), 0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_paths() {
        assert_eq!(parse_tile_path("/3/1/7/2"), Some((3, 1, 7, 2)));
        assert_eq!(parse_tile_path("3/1/7/2"), Some((3, 1, 7, 2)));
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
    fn missing_tile_is_404_and_malformed_is_400() {
        let images = Mutex::new(Images::default());
        assert_eq!(tile_response(&images, "/1/0/0/0").status(), 404);
        assert_eq!(tile_response(&images, "/nope").status(), 400);
    }
}
