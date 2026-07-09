//! TheUnduster desktop shell: thin Tauri layer over the engine crates.

mod cache;
mod detect;
mod export;
mod images;
mod masks;
mod models;
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

/// Webview-side errors are invisible in the dev terminal; the frontend
/// forwards uncaught errors and rejections here so they surface in the log.
#[tauri::command]
fn log_js_error(message: String) {
    eprintln!("[webview] {message}");
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
    roll: &State<'_, roll::RollState>,
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
    // Resolved before spawn_blocking so the closure can write the probs
    // cache from the borrowed slice, before handing the Vec's ownership to
    // set_probs_built -- avoids cloning a 671MB vector for a 168MP frame.
    let cache_path = roll
        .frame_for_image(id)?
        .map(|(roll_dir, file_name)| roll::probs_cache_path(&roll_dir, &file_name));
    let detector_hash = detector.hash();
    let detector = detector.inner().clone(); // DetectorState is Clone over an Arc
    let (probs, pyramid) = tauri::async_runtime::spawn_blocking(move || {
        let probs = detector.detect(&img)?;
        if let (Some(path), Some(hash)) = (&cache_path, &detector_hash) {
            if let Err(_e) = cache::write_probs(path, &probs, img.width, img.height, hash) {
                #[cfg(debug_assertions)]
                eprintln!("[cache] probs write failed for image {id}: {_e}");
            }
        }
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
    roll: State<'_, roll::RollState>,
    detector: State<'_, detect::DetectorState>,
    id: u64,
) -> Result<DetectReport, String> {
    let result = run_detect(&app, &images, &roll, &detector, id).await;
    let _ = app.emit("app-progress", Progress { id, stage: "ready" });
    result
}

/// Dilation applied to the thresholded mask before healing: model masks
/// cover a defect's confident core, and healing an under-covering mask
/// leaves a visible rim (3b-1 field notes / issue TheUnduster-0s4).
const HEAL_DILATE_RADIUS: u32 = 2;

#[derive(serde::Serialize, Clone)]
struct HealProgress {
    id: u64,
    done: usize,
    total: usize,
}

#[derive(serde::Serialize, Clone)]
struct HealSummary {
    id: u64,
    defects: usize,
    tiny: usize,
    inpainted: usize,
}

/// Runs the fallible body of `heal_frame`. Split out so `heal_frame` can
/// guarantee a terminal "ready" emit on every exit path, mirroring
/// `run_detect`/`detect`.
async fn run_heal(
    app: &tauri::AppHandle,
    images: &State<'_, Mutex<Images>>,
    inpainter: &State<'_, detect::InpainterState>,
    id: u64,
    threshold: f32,
    strokes: Vec<masks::Stroke>,
) -> Result<HealSummary, String> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(format!("threshold {threshold} out of range"));
    }
    masks::validate_strokes(&strokes)?;
    let (image, mask) = {
        let images = images.lock().map_err(|e| e.to_string())?;
        let image = images.image(id).ok_or_else(|| format!("no image {id}"))?;
        let mask = images
            .threshold_mask(id, threshold)
            .ok_or_else(|| format!("no detection for image {id}"))?;
        (image, mask)
    };
    let inpainter = inpainter.inner().clone();
    let app_for_progress = app.clone();
    let (healed, pyramid, mask, report) = tauri::async_runtime::spawn_blocking(move || {
        let mask = masks::compose_heal_mask(
            mask,
            image.width,
            image.height,
            HEAL_DILATE_RADIUS,
            &strokes,
        );
        let mut copy = (*image).clone(); // the original Arc stays pristine
                                         // A real inpainting model costs seconds per defect window; per-defect
                                         // progress keeps a long heal visibly alive in the status line.
        let report = inpainter
            .with_inpainter(|inp| {
                fd_heal::heal_with_progress(&mut copy, &mask, inp, &mut |done, total| {
                    let _ =
                        app_for_progress.emit("heal-progress", HealProgress { id, done, total });
                })
            })?
            .map_err(|e| e.to_string())?;
        let healed = std::sync::Arc::new(copy);
        let pyramid = fd_tiles::Pyramid::build(&healed);
        Ok::<_, String>((healed, pyramid, std::sync::Arc::new(mask), report))
    })
    .await
    .map_err(|e| e.to_string())??;
    let mut images = images.lock().map_err(|e| e.to_string())?;
    if !images.set_healed(id, healed, pyramid, mask) {
        return Err(format!("image {id} closed during healing"));
    }
    Ok(HealSummary {
        id,
        defects: report.defects,
        tiny: report.tiny,
        inpainted: report.inpainted,
    })
}

#[tauri::command]
async fn heal_frame(
    app: tauri::AppHandle,
    images: State<'_, Mutex<Images>>,
    inpainter: State<'_, detect::InpainterState>,
    id: u64,
    threshold: f32,
    strokes: Vec<masks::Stroke>,
) -> Result<HealSummary, String> {
    let _ = app.emit(
        "app-progress",
        Progress {
            id,
            stage: "healing",
        },
    );
    let result = run_heal(&app, &images, &inpainter, id, threshold, strokes).await;
    let _ = app.emit("app-progress", Progress { id, stage: "ready" });
    result
}

#[tauri::command]
fn load_inpainter(state: State<'_, detect::InpainterState>, path: String) -> Result<(), String> {
    state.load(std::path::Path::new(&path))
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
fn open_roll(
    roll: State<'_, roll::RollState>,
    images: State<'_, Mutex<Images>>,
    dir: String,
) -> Result<roll::RollInfo, String> {
    let (info, stale_ids) = roll.open(std::path::Path::new(&dir))?;
    if !stale_ids.is_empty() {
        let mut images = images.lock().map_err(|e| e.to_string())?;
        for id in stale_ids {
            images.close(id);
        }
    }
    Ok(info)
}

/// Closes whatever roll is currently open (if any), releasing its live
/// activated frames from `Images`. Used when the operator opens a single
/// scan while a roll was open: `App.svelte`'s `openScan` previously just
/// nulled the client-side `roll` reference, leaking any activated frame ids
/// server-side.
#[tauri::command]
fn close_roll(
    roll: State<'_, roll::RollState>,
    images: State<'_, Mutex<Images>>,
) -> Result<(), String> {
    let stale_ids = roll.close()?;
    if !stale_ids.is_empty() {
        let mut images = images.lock().map_err(|e| e.to_string())?;
        for id in stale_ids {
            images.close(id);
        }
    }
    Ok(())
}

/// Retained-pixel budget for activated roll frames. Frames stay decoded
/// until this is exceeded, then the least-recently-viewed release first
/// (never the current frame or its immediate neighbors). Big-memory
/// machines keep whole small rolls warm; an 8GB machine degrades to
/// re-decoding old frames rather than swapping or crashing. Override with
/// UNDUSTER_PIXEL_BUDGET_GB for unusual setups.
fn pixel_budget_bytes() -> usize {
    const DEFAULT_GB: usize = 6;
    std::env::var("UNDUSTER_PIXEL_BUDGET_GB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&gb| gb > 0)
        .unwrap_or(DEFAULT_GB)
        * 1024
        * 1024
        * 1024
}

/// Closes least-recently-viewed activated frames until retained pixel bytes
/// fit the budget. The keep window around `current` is never evicted, so
/// the window's worst case (three very large frames) may exceed the budget;
/// that degrades to OS paging, never to a crash.
fn evict_over_budget(
    images: &State<'_, Mutex<Images>>,
    roll: &State<'_, roll::RollState>,
    current: usize,
) -> Result<(), String> {
    let candidates = roll.eviction_candidates(current)?; // LRU first
    let mut sized: Vec<(usize, u64, usize)> = Vec::new();
    let mut total: usize = 0;
    {
        let images = images.lock().map_err(|e| e.to_string())?;
        // Total includes protected frames: the budget bounds overall
        // retained memory, not just the evictable share.
        for id in images.known_ids() {
            total += images.retained_bytes(id).unwrap_or(0);
        }
        for (idx, id) in candidates {
            sized.push((idx, id, images.retained_bytes(id).unwrap_or(0)));
        }
    }
    let budget = pixel_budget_bytes();
    for (idx, id, bytes) in sized {
        if total <= budget {
            break;
        }
        let mut images = images.lock().map_err(|e| e.to_string())?;
        images.close(id);
        drop(images);
        roll.clear_image_id(idx)?;
        total = total.saturating_sub(bytes);
        #[cfg(debug_assertions)]
        eprintln!(
            "[evict] frame {idx} (id {id}, {}MB) released",
            bytes / (1024 * 1024)
        );
    }
    Ok(())
}

#[tauri::command]
async fn activate_frame(
    app: tauri::AppHandle,
    images: State<'_, Mutex<Images>>,
    roll: State<'_, roll::RollState>,
    detector: State<'_, detect::DetectorState>,
    index: usize,
) -> Result<ImageInfo, String> {
    #[cfg(debug_assertions)]
    eprintln!("[activate] frame {index} requested");
    // Reuse path: already activated and the registry still has it.
    if let Some(id) = roll.image_id(index)? {
        let known = {
            let images = images.lock().map_err(|e| e.to_string())?;
            images.image(id)
        };
        if let Some(image) = known {
            let (levels, healed) = {
                let images = images.lock().map_err(|e| e.to_string())?;
                let levels = images
                    .level_dims(id)
                    .ok_or_else(|| format!("no image {id}"))?;
                (levels, images.has_healed(id))
            };
            let info = ImageInfo {
                id,
                width: image.width,
                height: image.height,
                levels: levels
                    .into_iter()
                    .map(|(width, height)| images::LevelInfo { width, height })
                    .collect(),
                healed,
            };
            // Reuse path never emits "decoding"/"building-pyramid", but the
            // frontend's loading state is only cleared on "ready" -- without
            // this emit, reactivating a cached frame wedges the loader
            // forever since no terminal event ever arrives.
            let _ = app.emit("app-progress", Progress { id, stage: "ready" });
            #[cfg(debug_assertions)]
            eprintln!("[activate] frame {index} reused id {id}");
            roll.touch(index)?; // recency drives byte-budget eviction
            return Ok(info);
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
    // Concurrent activations of the same frame each decode independently;
    // whichever lands later replaces the frame's id, and the superseded
    // image must be closed or it is orphaned in the registry (about a
    // gigabyte of leaked pixels per rapid re-click on a 168MP scan).
    if let Some(superseded) = roll.set_image_id(index, info.id)? {
        let mut images = images.lock().map_err(|e| e.to_string())?;
        images.close(superseded);
    }

    // Decode-path frames start with no probs; check for a cache hit so a
    // rescanned frame arrives detection-ready across relaunches, without
    // re-running the (seconds-long) detector. File IO + zstd decompress of
    // tens of MB, so this stays off the lock like the decode above.
    if let (Some((roll_dir, file_name)), Some(hash)) =
        (roll.frame_for_image(info.id)?, detector.hash())
    {
        let cache_path = roll::probs_cache_path(&roll_dir, &file_name);
        let (width, height) = (info.width, info.height);
        let level_dims: Vec<(u32, u32)> = info.levels.iter().map(|l| (l.width, l.height)).collect();
        let hit = tauri::async_runtime::spawn_blocking(move || {
            cache::read_probs(&cache_path, width, height, &hash)
                .map(|probs| (build_prob_pyramid(&probs, &level_dims), probs))
        })
        .await
        .map_err(|e| e.to_string())?;
        if let Some((pyramid, probs)) = hit {
            let mut images = images.lock().map_err(|e| e.to_string())?;
            images.set_probs_built(info.id, probs, pyramid);
        }
    }

    roll.touch(index)?;
    evict_over_budget(&images, &roll, index)?;

    let _ = app.emit(
        "app-progress",
        Progress {
            id: info.id,
            stage: "ready",
        },
    );
    #[cfg(debug_assertions)]
    eprintln!("[activate] frame {index} decoded as id {}", info.id);
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

#[tauri::command]
fn set_frame_strokes(
    roll: State<'_, roll::RollState>,
    index: usize,
    strokes: Vec<masks::Stroke>,
    redo_strokes: Vec<masks::Stroke>,
) -> Result<(), String> {
    masks::validate_strokes(&strokes)?;
    masks::validate_strokes(&redo_strokes)?;
    roll.set_strokes(index, strokes, redo_strokes)
}

#[tauri::command]
async fn export_frame(
    images: State<'_, Mutex<Images>>,
    id: u64,
    dest: String,
) -> Result<usize, String> {
    let (original, healed, mask) = {
        let images = images.lock().map_err(|e| e.to_string())?;
        images
            .healed_parts(id)
            .ok_or_else(|| format!("image {id} has no healed data to export"))?
    };
    let report = tauri::async_runtime::spawn_blocking(move || {
        export::export_healed(&original, &healed, &mask, std::path::Path::new(&dest))
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(report.changed_pixels)
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
    /// The scan's defect boxes ride along so the viewer can draw ring
    /// markers for the active frame immediately; without them the frontend
    /// only learns bboxes when a roll is (re)opened from its sidecar.
    bboxes: Option<Vec<[u32; 4]>>,
}

#[derive(serde::Serialize, Clone)]
struct RollFrameError {
    index: usize,
    message: String,
}

/// Emitted as soon as a frame's thumbnail exists on disk, well before its
/// (slow) detection finishes, so the filmstrip can show previews early.
#[derive(serde::Serialize, Clone)]
struct RollThumb {
    index: usize,
}

/// Clears the roll-scan flag when dropped, including on unwind, so a panic
/// anywhere in the `scan_roll` queue task (outer async body, not just the
/// `spawn_blocking` closures) can never wedge scanning permanently.
struct ScanFlagGuard(tauri::AppHandle);

impl Drop for ScanFlagGuard {
    fn drop(&mut self) {
        self.0.state::<roll::RollState>().clear_scanning();
    }
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
    // The flag is set; if either lookup fails (no roll open) it must be
    // cleared here, since the task carrying the drop guard never spawns.
    let setup = roll.dir().and_then(|dir| Ok((dir, roll.frames_to_scan()?)));
    let (roll_dir, indices) = match setup {
        Ok(v) => v,
        Err(e) => {
            roll.clear_scanning();
            return Err(e);
        }
    };
    // RollState is managed (lives for the app's lifetime), so re-fetching the
    // `State` via `app.state()` inside the task avoids needing RollState to
    // be Clone or the borrowed `State` to outlive this function.
    let detector = detector.inner().clone();
    let app_for_task = app.clone();
    // Snapshot the generation this scan is running against. `RollState::open`
    // and `close` each bump it, so if the roll is replaced or torn down
    // mid-scan the loop below detects the mismatch and bails before touching
    // the (now wrong) roll_dir/sidecar -- see the generation check just
    // inside the loop for the actual enforcement point.
    let generation = roll.generation();
    tauri::async_runtime::spawn(async move {
        let _scan_flag_guard = ScanFlagGuard(app_for_task.clone());
        for index in indices {
            let roll_state = app_for_task.state::<roll::RollState>();
            // Enforcement point: a roll swap/close bumped the generation
            // counter since this task spawned, so `roll_dir`/`indices`
            // (captured once, above) no longer describe the roll that's
            // open now. Stop rather than writing this frame's thumbnail
            // into the old roll's directory or its result into the new
            // roll's sidecar. `ScanFlagGuard` clears the scanning flag on
            // the way out, so a fresh scan of the new roll can start.
            if roll_state.generation() != generation {
                break;
            }
            let path = match roll_state.frame_path(index) {
                Ok(p) => p,
                Err(e) => {
                    let _ = roll_state.record_scan_result(generation, index, None, None);
                    let _ =
                        app_for_task.emit("roll-frame-error", RollFrameError { index, message: e });
                    continue;
                }
            };
            // Thumbnails are keyed by file name, not index (see
            // `roll::thumb_path`), so indices that shift across sessions
            // never pair a frame with another image's stale thumbnail.
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            // Stage 1: decode + thumbnail only, so the filmstrip gets its
            // preview within seconds; the (much slower) detection follows in
            // stage 2. `prepared` crosses the await into stage 2 and drops at
            // its end, so the "at most 1 queue frame in memory" bound holds.
            let thumb_path = roll::thumb_path(&roll_dir, &file_name);
            #[cfg(debug_assertions)]
            eprintln!("[queue] frame {index} decode starting");
            let staged = tauri::async_runtime::spawn_blocking(move || {
                let prepared = images::Images::prepare(&path)?;
                let coarsest = prepared
                    .pyramid
                    .levels
                    .last()
                    .expect("pyramid always has at least one level");
                let thumb = roll::write_thumbnail(
                    &coarsest.rgba,
                    coarsest.width,
                    coarsest.height,
                    &thumb_path,
                );
                Ok::<_, String>((prepared, thumb))
            })
            .await
            .map_err(|e| e.to_string())
            .and_then(|r| r);

            let prepared = match staged {
                Ok((prepared, thumb)) => {
                    match thumb {
                        Ok(()) => {
                            let _ = app_for_task.emit("roll-thumb", RollThumb { index });
                        }
                        Err(e) => {
                            let _ = app_for_task.emit(
                                "roll-frame-error",
                                RollFrameError {
                                    index,
                                    message: format!("thumbnail: {e}"),
                                },
                            );
                        }
                    }
                    prepared
                }
                Err(message) => {
                    let _ = roll_state.record_scan_result(generation, index, None, None);
                    let _ =
                        app_for_task.emit("roll-frame-error", RollFrameError { index, message });
                    continue;
                }
            };

            #[cfg(debug_assertions)]
            eprintln!("[queue] frame {index} detect starting");
            // Stage 2: detection on the already-decoded frame. Only the
            // decoded pixels cross into the closure: the display pyramid
            // built for the thumbnail is over a gigabyte on a 168MP scan and
            // detection never reads it, while inference itself is the app's
            // peak-memory window -- keeping the pyramid alive through it
            // helped push real color rolls into the OS memory killer.
            let image = prepared.image;
            drop(prepared.pyramid);
            let detector = detector.clone();
            let detector_hash = detector.hash();
            let cache_path = roll::probs_cache_path(&roll_dir, &file_name);
            let outcome = tauri::async_runtime::spawn_blocking(move || {
                let probs = detector.detect(&image)?;
                let bboxes = images::components_from_probs(
                    &probs,
                    image.width,
                    image.height,
                    SCAN_THRESHOLD,
                );
                // Cache write is sequential with detection here (milliseconds
                // against a ~9s detect); failures eprintln in debug only,
                // never fail the scan.
                if let Some(hash) = detector_hash {
                    if let Err(_e) =
                        cache::write_probs(&cache_path, &probs, image.width, image.height, &hash)
                    {
                        #[cfg(debug_assertions)]
                        eprintln!("[queue] frame {index} probs cache write failed: {_e}");
                    }
                }
                Ok::<_, String>(bboxes)
                // `image` (the full-res pixels) drops here, before the task
                // moves to the next frame.
            })
            .await
            .map_err(|e| e.to_string())
            .and_then(|r| r);

            match outcome {
                Ok(bboxes) => {
                    let count = bboxes.len();
                    let _ = roll_state.record_scan_result(
                        generation,
                        index,
                        Some(count),
                        Some(bboxes.clone()),
                    );
                    let _ = app_for_task.emit(
                        "roll-progress",
                        RollProgress {
                            index,
                            count: Some(count),
                            bboxes: Some(bboxes),
                        },
                    );
                }
                Err(message) => {
                    let _ = roll_state.record_scan_result(generation, index, None, None);
                    let _ =
                        app_for_task.emit("roll-frame-error", RollFrameError { index, message });
                }
            }
        }
        let _ = app_for_task.emit("roll-done", ());
    });
    Ok(())
}

#[derive(serde::Serialize, Clone)]
struct ExportProgress {
    index: usize,
}

#[derive(serde::Serialize, Clone)]
struct ExportFrameError {
    index: usize,
    message: String,
}

#[derive(serde::Serialize, Clone)]
struct ExportFrameStage {
    index: usize,
    stage: &'static str,
}

#[derive(serde::Serialize, Clone)]
struct ExportHealProgress {
    index: usize,
    done: usize,
    total: usize,
}

/// Clears the roll-export flag when dropped, including on unwind, so a panic
/// anywhere in the `export_approved` queue task can never wedge exporting
/// permanently. Mirrors `ScanFlagGuard`.
struct ExportFlagGuard(tauri::AppHandle);

impl Drop for ExportFlagGuard {
    fn drop(&mut self) {
        self.0.state::<roll::RollState>().clear_exporting();
    }
}

#[tauri::command]
fn export_approved(
    app: tauri::AppHandle,
    roll: State<'_, roll::RollState>,
    detector: State<'_, detect::DetectorState>,
    inpainter: State<'_, detect::InpainterState>,
    dest_dir: String,
) -> Result<(), String> {
    if roll
        .exporting
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Ok(()); // already running; idempotent from the caller's view
    }
    // The flag is set; if the lookup fails (no roll open) it must be cleared
    // here, since the task carrying the drop guard never spawns.
    let indices = match roll.frames_to_export() {
        Ok(v) => v,
        Err(e) => {
            roll.clear_exporting();
            return Err(e);
        }
    };
    let dest_dir = std::path::PathBuf::from(dest_dir);
    let detector = detector.inner().clone();
    let inpainter = inpainter.inner().clone();
    let app_for_task = app.clone();
    // Snapshot the generation this export is running against; see
    // `scan_roll`'s identical comment for why this matters.
    let generation = roll.generation();
    tauri::async_runtime::spawn(async move {
        let _export_flag_guard = ExportFlagGuard(app_for_task.clone());
        for index in indices {
            let roll_state = app_for_task.state::<roll::RollState>();
            // Enforcement point: see scan_roll's identical check.
            if roll_state.generation() != generation {
                break;
            }
            let (path, file_name, frame_threshold, frame_strokes) =
                match roll_state.export_frame_meta(index) {
                    Ok(meta) => meta,
                    Err(e) => {
                        let _ = app_for_task
                            .emit("export-frame-error", ExportFrameError { index, message: e });
                        continue;
                    }
                };

            // Prefer already-healed registry data (the operator reviewed it).
            let registry_export = {
                let roll_state = app_for_task.state::<roll::RollState>();
                let images_state = app_for_task.state::<Mutex<Images>>();
                match roll_state.image_id(index) {
                    Ok(Some(id)) => {
                        let images = match images_state.lock() {
                            Ok(g) => g,
                            Err(_) => {
                                let _ = app_for_task.emit(
                                    "export-frame-error",
                                    ExportFrameError {
                                        index,
                                        message: "image registry lock poisoned".to_string(),
                                    },
                                );
                                continue;
                            }
                        };
                        images.healed_parts(id)
                    }
                    _ => None,
                }
            };
            let dest = dest_dir.join(&file_name);
            let outcome = if let Some((original, healed, mask)) = registry_export {
                tauri::async_runtime::spawn_blocking(move || {
                    export::export_healed(&original, &healed, &mask, &dest).map(|_| ())
                })
                .await
                .map_err(|e| e.to_string())
                .and_then(|r| r)
            } else {
                // Transient pipeline: decode, detect, heal, export -- one
                // frame's pixels at a time, dropped at the closure's end.
                // With a real inpainting model this path costs minutes per
                // frame, so it narrates its stages; without the events the
                // roll counter sits frozen and reads as a hang (field
                // report: "hangs at image 2").
                let detector = detector.clone();
                let inpainter = inpainter.clone();
                let threshold = frame_threshold;
                let app_for_stages = app_for_task.clone();
                tauri::async_runtime::spawn_blocking(move || {
                    let stage = |s: &'static str| {
                        let _ = app_for_stages
                            .emit("export-frame-stage", ExportFrameStage { index, stage: s });
                    };
                    stage("detecting");
                    let image = images::Images::decode_stage(&path)?;
                    let probs = detector.detect(&image)?;
                    masks::validate_strokes(&frame_strokes)?;
                    let raw: Vec<bool> = probs.iter().map(|&p| p > threshold).collect();
                    let mask = masks::compose_heal_mask(
                        raw,
                        image.width,
                        image.height,
                        HEAL_DILATE_RADIUS,
                        &frame_strokes,
                    );
                    stage("healing");
                    let mut copy = (*image).clone();
                    inpainter
                        .with_inpainter(|inp| {
                            fd_heal::heal_with_progress(
                                &mut copy,
                                &mask,
                                inp,
                                &mut |done, total| {
                                    let _ = app_for_stages.emit(
                                        "export-heal-progress",
                                        ExportHealProgress { index, done, total },
                                    );
                                },
                            )
                        })?
                        .map_err(|e| e.to_string())?;
                    stage("writing");
                    export::export_healed(&image, &copy, &mask, &dest).map(|_| ())
                })
                .await
                .map_err(|e| e.to_string())
                .and_then(|r| r)
            };
            match outcome {
                Ok(()) => {
                    let _ = app_for_task
                        .state::<roll::RollState>()
                        .set_exported(generation, index);
                    let _ = app_for_task.emit("export-progress", ExportProgress { index });
                }
                Err(message) => {
                    let _ = app_for_task
                        .emit("export-frame-error", ExportFrameError { index, message });
                }
            }
        }
        let _ = app_for_task.emit("export-done", ());
    });
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Mutex::new(Images::default()))
        .manage(detect::DetectorState::default())
        .manage(detect::InpainterState::default())
        .manage(roll::RollState::default())
        .manage(models::ModelDownloadState::default())
        .invoke_handler(tauri::generate_handler![
            log_js_error,
            open_image,
            close_image,
            load_detector,
            detect,
            load_inpainter,
            heal_frame,
            components,
            open_roll,
            close_roll,
            activate_frame,
            set_frame_threshold,
            approve_frame,
            set_frame_strokes,
            export_frame,
            scan_roll,
            export_approved,
            models::inpainter_status,
            models::download_inpaint_model
        ])
        .register_uri_scheme_protocol("tiles", |ctx, request| {
            let images = ctx.app_handle().state::<Mutex<Images>>();
            let roll = ctx.app_handle().state::<roll::RollState>();
            protocol::tile_response(&images, &roll.roll, request.uri().path())
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
            let inpainter = app.state::<detect::InpainterState>();
            let mut loaded = false;
            if let Ok(lama) = models::lama_path(app.handle()) {
                // A crash mid-download can orphan the ~200MB temp file; it is
                // never loaded, only wasted disk. Clear it on startup.
                let _ = std::fs::remove_file(lama.with_file_name("lama.onnx.tmp-unduster"));
                if lama.exists() {
                    match inpainter.load(&lama) {
                        Ok(()) => loaded = true,
                        Err(e) => eprintln!("[models] lama load failed, falling back: {e}"),
                    }
                }
            }
            // The fixture autoload makes dev builds report a loaded inpainter,
            // which hides the model-download UI entirely; the env var lets a
            // dev session exercise the missing/download states.
            #[cfg(debug_assertions)]
            if !loaded && std::env::var("UNDUSTER_NO_FIXTURE_INPAINT").is_err() {
                let fixtures =
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../engine/fixtures");
                let _ = inpainter.load(&fixtures.join("tiny-inpaint.onnx"));
            }
            #[cfg(not(debug_assertions))]
            let _ = loaded;
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
        state
            .record_scan_result(state.generation(), 0, Some(2), Some(vec![]))
            .unwrap();
        assert_eq!(state.frames_to_scan().unwrap(), vec![1]);
    }

    #[test]
    fn scan_result_from_a_replaced_roll_is_discarded() {
        // A frame decoded against roll A must never land in roll B's sidecar
        // when the operator swaps rolls mid-decode.
        let dir_a = tempfile::tempdir().unwrap();
        std::fs::write(dir_a.path().join("a.png"), b"x").unwrap();
        let dir_b = tempfile::tempdir().unwrap();
        std::fs::write(dir_b.path().join("b.png"), b"x").unwrap();
        let state = roll::RollState::default();
        state.open(dir_a.path()).unwrap();
        let stale_generation = state.generation();
        state.open(dir_b.path()).unwrap();
        let err = state
            .record_scan_result(stale_generation, 0, Some(9), Some(vec![]))
            .unwrap_err();
        assert!(err.contains("roll changed"));
        assert_eq!(state.frames_to_scan().unwrap(), vec![0]); // B untouched
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
