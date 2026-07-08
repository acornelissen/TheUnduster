//! TheUnduster desktop shell: thin Tauri layer over the engine crates.

mod detect;
mod images;
mod protocol;
mod roll;

use std::sync::atomic::Ordering;
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

/// Fixed threshold for the background queue's stored bboxes/count. The
/// operator's per-frame sensitivity slider (Task 2's `set_frame_threshold`)
/// only affects live overlay/z-navigation on the activated frame; the queue
/// runs once per roll at a stable threshold so counts are comparable across
/// frames regardless of what the operator was looking at when the queue
/// reached them.
const SCAN_THRESHOLD: f32 = 0.5;

#[derive(serde::Serialize, Clone)]
struct RollProgress {
    index: usize,
    count: Option<usize>,
}

#[derive(serde::Serialize, Clone)]
struct RollFrameError {
    index: usize,
    message: String,
}

#[tauri::command]
fn scan_roll(
    app: tauri::AppHandle,
    roll: State<'_, roll::RollState>,
    detector: State<'_, detect::DetectorState>,
) -> Result<(), String> {
    if roll
        .scanning
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Ok(()); // already running; idempotent from the caller's view
    }
    let roll_dir = roll.dir()?;
    let indices = roll.frames_to_scan()?;
    // RollState is managed (lives for the app's lifetime), so re-fetching the
    // `State` via `app.state()` inside the task avoids needing RollState to
    // be Clone or the borrowed `State` to outlive this function.
    let detector = detector.inner().clone();
    let app_for_task = app.clone();
    tauri::async_runtime::spawn(async move {
        for index in indices {
            let roll_state = app_for_task.state::<roll::RollState>();
            let path = match roll_state.frame_path(index) {
                Ok(p) => p,
                Err(e) => {
                    let _ = roll_state.record_scan_result(index, None, None);
                    let _ =
                        app_for_task.emit("roll-frame-error", RollFrameError { index, message: e });
                    continue;
                }
            };
            let detector = detector.clone();
            let outcome = tauri::async_runtime::spawn_blocking(move || {
                let prepared = images::Images::prepare(&path)?;
                let probs = detector.detect(&prepared.image)?;
                let counted = images::components_from_probs(
                    &probs,
                    prepared.image.width,
                    prepared.image.height,
                    SCAN_THRESHOLD,
                );
                let coarsest = prepared
                    .pyramid
                    .levels
                    .last()
                    .expect("pyramid always has at least one level");
                Ok::<_, String>((
                    counted,
                    coarsest.rgba.clone(),
                    coarsest.width,
                    coarsest.height,
                ))
                // `prepared` (and its full-res pixels) drops here, at the end
                // of the blocking closure, before the task moves to the next
                // frame -- this is the "at most 1 queue frame" memory bound.
            })
            .await
            .map_err(|e| e.to_string())
            .and_then(|r| r);

            match outcome {
                Ok((bboxes, thumb_rgba, tw, th)) => {
                    let thumb_path = roll::thumb_path(&roll_dir, index);
                    if let Err(e) = roll::write_thumbnail(&thumb_rgba, tw, th, &thumb_path) {
                        let _ = app_for_task.emit(
                            "roll-frame-error",
                            RollFrameError {
                                index,
                                message: format!("thumbnail: {e}"),
                            },
                        );
                    }
                    let count = bboxes.len();
                    let _ = roll_state.record_scan_result(index, Some(count), Some(bboxes));
                    let _ = app_for_task.emit(
                        "roll-progress",
                        RollProgress {
                            index,
                            count: Some(count),
                        },
                    );
                }
                Err(message) => {
                    let _ = roll_state.record_scan_result(index, None, None);
                    let _ =
                        app_for_task.emit("roll-frame-error", RollFrameError { index, message });
                }
            }
        }
        app_for_task
            .state::<roll::RollState>()
            .scanning
            .store(false, Ordering::SeqCst);
        let _ = app_for_task.emit("roll-done", ());
    });
    Ok(())
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
            approve_frame,
            scan_roll
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

#[cfg(test)]
mod roll_queue_tests {
    use super::*;

    fn fixture_detector() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../engine/fixtures/tiny-detector.onnx")
    }

    #[test]
    fn frames_to_scan_lists_only_uncounted_frames() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        std::fs::write(dir.path().join("b.png"), b"x").unwrap();
        let state = roll::RollState::default();
        state.open(dir.path()).unwrap();
        state.record_scan_result(0, Some(2), Some(vec![])).unwrap();
        assert_eq!(state.frames_to_scan().unwrap(), vec![1]);
    }

    #[test]
    fn detector_state_clone_shares_the_loaded_model() {
        // Guards the scan_roll assumption that `detector.inner().clone()`
        // yields a handle that still resolves after the original `State`
        // borrow is gone (used across an async task boundary).
        let state = detect::DetectorState::default();
        state.load(&fixture_detector()).unwrap();
        let cloned = state.clone();
        let img = fd_io::ImageBuf {
            width: 8,
            height: 8,
            channels: 1,
            data: fd_io::PixelData::U8(vec![0; 64]),
            icc: None,
            exif: None,
        };
        assert!(cloned.detect(&img).is_ok());
    }
}
