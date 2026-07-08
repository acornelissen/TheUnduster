//! TheUnduster desktop shell: thin Tauri layer over the engine crates.

mod images;
mod protocol;

use std::sync::Mutex;

use images::{ImageInfo, Images};
use tauri::{Manager, State};

#[tauri::command]
fn open_image(state: State<'_, Mutex<Images>>, path: String) -> Result<ImageInfo, String> {
    let mut images = state.lock().map_err(|e| e.to_string())?;
    images.open(std::path::Path::new(&path))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(Mutex::new(Images::default()))
        .invoke_handler(tauri::generate_handler![open_image])
        .register_uri_scheme_protocol("tiles", |ctx, request| {
            let images = ctx.app_handle().state::<Mutex<Images>>();
            protocol::tile_response(&images, request.uri().path())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
