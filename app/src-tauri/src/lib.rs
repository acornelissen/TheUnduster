//! TheUnduster desktop shell: thin Tauri layer over the engine crates.

mod detect;
mod images;
mod protocol;

use std::sync::Mutex;

use images::{ImageInfo, Images, Prepared};
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
    let image = tauri::async_runtime::spawn_blocking(move || {
        Images::decode_stage(std::path::Path::new(&path))
    })
    .await
    .map_err(|e| e.to_string())??;
    let _ = app.emit(
        "app-progress",
        Progress {
            id: 0,
            stage: "building-pyramid",
        },
    );
    let pyramid = {
        let image = image.clone();
        tauri::async_runtime::spawn_blocking(move || Images::pyramid_stage(&image))
            .await
            .map_err(|e| e.to_string())?
    };
    let prepared = Prepared { image, pyramid };
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

#[tauri::command]
fn load_detector(
    state: tauri::State<'_, detect::DetectorState>,
    path: String,
) -> Result<(), String> {
    state.load(std::path::Path::new(&path))
}

#[derive(serde::Serialize, Clone)]
struct DetectReport {
    id: u64,
    components_at_half: usize,
}

#[tauri::command]
async fn detect(
    app: tauri::AppHandle,
    images: State<'_, Mutex<Images>>,
    detector: State<'_, detect::DetectorState>,
    id: u64,
) -> Result<DetectReport, String> {
    let img = {
        let images = images.lock().map_err(|e| e.to_string())?;
        images.image(id).ok_or_else(|| format!("no image {id}"))?
    }; // lock released; inference runs on the Arc clone
    let _ = app.emit(
        "app-progress",
        Progress {
            id,
            stage: "detecting",
        },
    );
    let detector = detector.inner().clone(); // DetectorState is Clone over an Arc
    let probs = tauri::async_runtime::spawn_blocking(move || detector.detect(&img))
        .await
        .map_err(|e| e.to_string())??;
    let report = {
        let mut images = images.lock().map_err(|e| e.to_string())?;
        if !images.set_probs(id, probs) {
            return Err(format!(
                "image {id} closed during detection or detector output size mismatch"
            ));
        }
        DetectReport {
            id,
            components_at_half: images.components(id, 0.5).unwrap_or_default().len(),
        }
    };
    let _ = app.emit("app-progress", Progress { id, stage: "ready" });
    Ok(report)
}

#[tauri::command]
fn components(
    images: State<'_, Mutex<Images>>,
    id: u64,
    threshold: f32,
) -> Result<Vec<[u32; 4]>, String> {
    let images = images.lock().map_err(|e| e.to_string())?;
    images
        .components(id, threshold)
        .ok_or_else(|| "no detection for image".to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Mutex::new(Images::default()))
        .manage(detect::DetectorState::default())
        .invoke_handler(tauri::generate_handler![
            open_image,
            close_image,
            load_detector,
            detect,
            components
        ])
        .register_uri_scheme_protocol("tiles", |ctx, request| {
            let images = ctx.app_handle().state::<Mutex<Images>>();
            protocol::tile_response(&images, request.uri().path())
        })
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                use tauri::Manager;
                let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../engine/fixtures/tiny-detector.onnx");
                let _ = app.state::<detect::DetectorState>().load(&fixture);
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
