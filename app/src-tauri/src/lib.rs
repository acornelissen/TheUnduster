//! TheUnduster desktop shell: thin Tauri layer over the engine crates.

mod images;
mod protocol;

use std::sync::Mutex;

use images::{ImageInfo, Images};
use tauri::{Emitter, Manager, State};

#[derive(serde::Serialize, Clone)]
struct Progress {
    id: u64,
    stage: &'static str,
}

#[tauri::command]
async fn open_image(
    app: tauri::AppHandle,
    state: State<'_, Mutex<Images>>,
    path: String,
) -> Result<ImageInfo, String> {
    let _ = app.emit(
        "app-progress",
        Progress {
            id: 0,
            stage: "decoding",
        },
    );
    let prepared =
        tauri::async_runtime::spawn_blocking(move || Images::prepare(std::path::Path::new(&path)))
            .await
            .map_err(|e| e.to_string())??;
    let _ = app.emit(
        "app-progress",
        Progress {
            id: 0,
            stage: "building-pyramid",
        },
    );
    let info = {
        let mut images = state.lock().map_err(|e| e.to_string())?;
        images.insert(prepared)
    };
    let _ = app.emit(
        "app-progress",
        Progress {
            id: info.id,
            stage: "ready",
        },
    );
    Ok(info)
}

#[tauri::command]
fn close_image(state: State<'_, Mutex<Images>>, id: u64) -> Result<(), String> {
    let mut images = state.lock().map_err(|e| e.to_string())?;
    images.close(id);
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Mutex::new(Images::default()))
        .invoke_handler(tauri::generate_handler![open_image, close_image])
        .register_uri_scheme_protocol("tiles", |ctx, request| {
            let images = ctx.app_handle().state::<Mutex<Images>>();
            protocol::tile_response(&images, request.uri().path())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
