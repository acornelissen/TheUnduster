//! TheUnduster desktop shell: thin Tauri layer over the engine crates.

mod detect;
mod images;
mod protocol;
mod roll;

use std::sync::Mutex;

use images::{build_prob_pyramid, ImageInfo, Images, Prepared};
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

/// Runs the fallible body of `detect`. Split out so `detect` can guarantee a
/// terminal "ready" emit after this resolves, on every exit path (success or
/// error alike) — the frontend gates its loading state on that emit and
/// would otherwise hang behind the loader on any failure.
async fn run_detect(
    app: &tauri::AppHandle,
    images: &State<'_, Mutex<Images>>,
    detector: &State<'_, detect::DetectorState>,
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
    let level_dims = {
        let images = images.lock().map_err(|e| e.to_string())?;
        images
            .level_dims(id)
            .ok_or_else(|| format!("no image {id}"))?
    };
    let detector = detector.inner().clone(); // DetectorState is Clone over an Arc
    let (probs, pyramid) = tauri::async_runtime::spawn_blocking(move || {
        let probs = detector.detect(&img)?;
        let pyramid = build_prob_pyramid(&probs, &level_dims);
        Ok::<_, String>((probs, pyramid))
    })
    .await
    .map_err(|e| e.to_string())??;
    let mut images = images.lock().map_err(|e| e.to_string())?;
    if !images.set_probs_built(id, probs, pyramid) {
        return Err(format!(
            "image {id} closed during detection or detector output size mismatch"
        ));
    }
    Ok(DetectReport {
        id,
        components_at_half: images.components(id, 0.5).unwrap_or_default().len(),
    })
}

#[tauri::command]
async fn detect(
    app: tauri::AppHandle,
    images: State<'_, Mutex<Images>>,
    detector: State<'_, detect::DetectorState>,
    id: u64,
) -> Result<DetectReport, String> {
    let result = run_detect(&app, &images, &detector, id).await;
    let _ = app.emit("app-progress", Progress { id, stage: "ready" });
    result
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
        .ok_or_else(|| format!("no detection for image {id}"))
}

#[tauri::command]
fn open_roll(state: State<'_, roll::RollState>, dir: String) -> Result<roll::RollInfo, String> {
    state.open(std::path::Path::new(&dir))
}

#[tauri::command]
async fn activate_frame(
    app: tauri::AppHandle,
    images: State<'_, Mutex<Images>>,
    roll: State<'_, roll::RollState>,
    index: usize,
) -> Result<ImageInfo, String> {
    // Reuse path: already activated and the registry still has it.
    if let Some(id) = roll.image_id(index)? {
        let known = {
            let images = images.lock().map_err(|e| e.to_string())?;
            images.image(id)
        };
        if let Some(image) = known {
            let levels = {
                let images = images.lock().map_err(|e| e.to_string())?;
                images
                    .level_dims(id)
                    .ok_or_else(|| format!("no image {id}"))?
            };
            return Ok(ImageInfo {
                id,
                width: image.width,
                height: image.height,
                levels: levels
                    .into_iter()
                    .map(|(width, height)| images::LevelInfo { width, height })
                    .collect(),
            });
        }
    }

    let path = roll.frame_path(index)?;
    let _ = app.emit(
        "app-progress",
        Progress {
            id: 0,
            stage: "decoding",
        },
    );
    let image = tauri::async_runtime::spawn_blocking(move || Images::decode_stage(&path))
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
        let mut images = images.lock().map_err(|e| e.to_string())?;
        images.insert(prepared)
    };
    roll.set_image_id(index, info.id)?;

    for (evict_index, evict_id) in roll.ids_to_evict(index)? {
        let mut images = images.lock().map_err(|e| e.to_string())?;
        images.close(evict_id);
        drop(images);
        roll.clear_image_id(evict_index)?;
    }

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
fn set_frame_threshold(
    state: State<'_, roll::RollState>,
    index: usize,
    threshold: f32,
) -> Result<(), String> {
    state.set_threshold(index, threshold)
}

#[tauri::command]
fn approve_frame(
    state: State<'_, roll::RollState>,
    index: usize,
    approved: bool,
) -> Result<(), String> {
    state.set_approved(index, approved)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Mutex::new(Images::default()))
        .manage(detect::DetectorState::default())
        .manage(roll::RollState::default())
        .invoke_handler(tauri::generate_handler![
            open_image,
            close_image,
            load_detector,
            detect,
            components,
            open_roll,
            activate_frame,
            set_frame_threshold,
            approve_frame
        ])
        .register_uri_scheme_protocol("tiles", |ctx, request| {
            let images = ctx.app_handle().state::<Mutex<Images>>();
            protocol::tile_response(&images, request.uri().path())
        })
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                use tauri::Manager;
                // Prefer the trained demo model; the random-weight tiny
                // detector exists only for protocol tests and fires on
                // everything when pointed at real scans.
                let fixtures =
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../engine/fixtures");
                let state = app.state::<detect::DetectorState>();
                if state.load(&fixtures.join("demo-detector.onnx")).is_err() {
                    let _ = state.load(&fixtures.join("tiny-detector.onnx"));
                }
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
