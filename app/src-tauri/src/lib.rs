//! TheUnduster desktop shell: thin Tauri layer over the engine crates.

mod cache;
mod detect;
mod export;
mod images;
mod jobs;
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

/// Stats `path` for a cache stamp, or `None` on any stat failure. Every
/// cache read/write site goes through this: a stat failure (permission
/// error, race with a delete, etc.) skips the cache interaction entirely --
/// it must never fail the surrounding operation, only cost it a cache hit or
/// a cache write.
fn stamp_or_skip(path: &std::path::Path) -> Option<cache::SourceStamp> {
    match cache::source_stamp(path) {
        Ok(s) => Some(s),
        Err(_e) => {
            #[cfg(debug_assertions)]
            eprintln!("[cache] stamp failed for {}: {_e}", path.display());
            None
        }
    }
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
    // The cache path and the source path (dir/file_name) both come from the
    // same frame mapping: the source path is what the write is stamped
    // against.
    let cache_source = roll.frame_for_image(id)?.map(|(roll_dir, file_name)| {
        let source_path = roll_dir.join(&file_name);
        let cache_path = roll::probs_cache_path(&roll_dir, &file_name);
        (cache_path, source_path)
    });
    let detector = detector.inner().clone(); // DetectorState is Clone over an Arc
    let (probs, pyramid) = tauri::async_runtime::spawn_blocking(move || {
        // detect_hashed pairs the output with the hash of the model that
        // produced it under one lock -- see its doc comment -- so the cache
        // write below can never record a different model's hash.
        let (probs, hash) = detector.detect_hashed(&img)?;
        if let Some((path, source_path)) = &cache_source {
            if let Some(stamp) = stamp_or_skip(source_path) {
                if let Err(_e) =
                    cache::write_probs(path, &probs, img.width, img.height, &hash, &stamp)
                {
                    #[cfg(debug_assertions)]
                    eprintln!("[cache] probs write failed for image {id}: {_e}");
                }
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
    /// True when this heal was restored from the on-disk delta cache
    /// (provenance-matched) instead of freshly computed. Informational only
    /// -- the frontend is free to ignore it.
    restored: bool,
}

/// Runs the fallible body of `heal_frame`. Split out so `heal_frame` can
/// guarantee a terminal "ready" emit on every exit path, mirroring
/// `run_detect`/`detect`.
// Params mirror the Tauri states/args heal_frame receives, plus the roll and
// detector handles the cache read/write needs; splitting them into a struct
// would obscure rather than clarify this thin a function.
#[allow(clippy::too_many_arguments)]
async fn run_heal(
    app: &tauri::AppHandle,
    images: &State<'_, Mutex<Images>>,
    roll: &State<'_, roll::RollState>,
    detector: &State<'_, detect::DetectorState>,
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
    // Resolved before spawn_blocking (owned values into the closure), mirroring
    // run_detect's cache_path resolution. `None` when the frame doesn't map to
    // a roll (single-image mode): the heal cache is roll-only. A stat failure
    // on the source also collapses to `None` here -- skip the cache
    // interaction entirely rather than fail the heal.
    let cache_path = roll.frame_for_image(id)?.and_then(|(roll_dir, file_name)| {
        let source_path = roll_dir.join(&file_name);
        let stamp = stamp_or_skip(&source_path)?;
        Some((roll::heal_cache_path(&roll_dir, &file_name), stamp))
    });
    // The CURRENT detector's hash: the thresholded probs in the registry came
    // from it. Zeros when none loaded -- that race (detector swapped between
    // detect and heal) is bead-tracked separately as unreachable in practice.
    // Read path only (feeds provenance, not a cache write): a race here is a
    // benign cache miss.
    let detector_hash = detector.hash().unwrap_or([0u8; 32]);
    let inpainter = inpainter.inner().clone();
    let app_for_progress = app.clone();
    let (healed, pyramid, mask, report, restored) =
        tauri::async_runtime::spawn_blocking(move || {
            // Everything provenance-dependent -- including the inpainter's
            // hash -- happens inside this closure, under with_inpainter_hashed's
            // single lock: the model that actually heals (on a miss) and the
            // hash recorded in provenance must be the same observation, or a
            // model download completing mid-flight could heal with model B
            // while provenance says model A.
            inpainter.with_inpainter_hashed(|pair| {
                let inpainter_hash = pair.as_ref().map(|(_, h)| *h).unwrap_or([0u8; 32]);
                // `path_and_provenance` is `None` whenever the frame isn't
                // roll-backed or the source stat failed above -- in either
                // case the cache interaction (read AND write) is skipped.
                let path_and_provenance = cache_path.as_ref().map(|(path, stamp)| {
                    let provenance = cache::heal_provenance(
                        threshold,
                        HEAL_DILATE_RADIUS,
                        &strokes,
                        &detector_hash,
                        &inpainter_hash,
                        stamp,
                    );
                    (path, provenance)
                });

                if let Some((path, provenance)) = &path_and_provenance {
                    if let Some((healed, mask)) = cache::read_heal(path, &image, provenance) {
                        let healed = std::sync::Arc::new(healed);
                        let pyramid = fd_tiles::Pyramid::build(&healed);
                        return Ok::<_, String>((
                            healed,
                            pyramid,
                            std::sync::Arc::new(mask),
                            fd_heal::HealReport::default(),
                            true,
                        ));
                    }
                }

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
                let inp = pair.map(|(inp, _)| inp);
                let report =
                    fd_heal::heal_with_progress(&mut copy, &mask, inp, &mut |done, total| {
                        let _ = app_for_progress
                            .emit("heal-progress", HealProgress { id, done, total });
                    })
                    .map_err(|e| e.to_string())?;
                if let Some((path, provenance)) = &path_and_provenance {
                    if let Err(_e) = cache::write_heal(path, &image, &copy, &mask, provenance) {
                        #[cfg(debug_assertions)]
                        eprintln!("[cache] heal write failed for image {id}: {_e}");
                    }
                }
                let healed = std::sync::Arc::new(copy);
                let pyramid = fd_tiles::Pyramid::build(&healed);
                Ok((healed, pyramid, std::sync::Arc::new(mask), report, false))
            })?
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
        restored,
    })
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
async fn heal_frame(
    app: tauri::AppHandle,
    images: State<'_, Mutex<Images>>,
    roll: State<'_, roll::RollState>,
    detector: State<'_, detect::DetectorState>,
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
    let result = run_heal(
        &app, &images, &roll, &detector, &inpainter, id, threshold, strokes,
    )
    .await;
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
    queue: State<'_, jobs::JobQueue>,
    dir: String,
) -> Result<roll::RollInfo, String> {
    let (info, stale_ids) = roll.open(std::path::Path::new(&dir))?;
    // Queued jobs referenced the replaced roll's indices; drop them. An
    // in-flight worker (if any) exits on its own via the generation check.
    queue.clear()?;
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
    queue: State<'_, jobs::JobQueue>,
) -> Result<(), String> {
    let stale_ids = roll.close()?;
    // Queued jobs referenced the closed roll's indices; drop them. An
    // in-flight worker (if any) exits on its own via the generation check.
    queue.clear()?;
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
/// that degrades to OS paging, never to a crash. The pinned frame (the one
/// job worker's active job, if any) is a fourth exempt frame beyond
/// current+/-1, but it is similarly bounded -- one worker, one pin -- so it
/// degrades the same way as the window overhang rather than opening an
/// unbounded exemption.
fn evict_over_budget(
    images: &State<'_, Mutex<Images>>,
    roll: &State<'_, roll::RollState>,
    queue: &State<'_, jobs::JobQueue>,
    current: usize,
) -> Result<(), String> {
    let candidates = roll.eviction_candidates(current)?; // LRU first
                                                         // The frame an active background job is operating on must not be pulled
                                                         // out from under it, wherever it sits in the LRU order.
    let pinned = queue.pinned()?;
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
        if Some(id) == pinned {
            continue;
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

    // Decode-path frames start with no probs; restore a cache hit so a
    // scanned frame becomes detection-ready across relaunches without
    // re-running the (seconds-long) detector. Fire-and-forget: the restore
    // costs file IO plus a full-image dequantize and pyramid build, and
    // awaiting it here made every first frame visit seconds slower (field
    // report) -- activation must return at decode speed. A stale or closed
    // id is harmless: set_probs_built validates and drops the result.
    // Read path only (cache lookup, not a write): a race against a
    // concurrent model swap is a benign cache miss.
    if let (Some((roll_dir, file_name)), Some(hash)) =
        (roll.frame_for_image(info.id)?, detector.hash())
    {
        let cache_path = roll::probs_cache_path(&roll_dir, &file_name);
        let source_path = roll_dir.join(&file_name);
        let (width, height) = (info.width, info.height);
        let level_dims: Vec<(u32, u32)> = info.levels.iter().map(|l| (l.width, l.height)).collect();
        let app_for_restore = app.clone();
        let id = info.id;
        tauri::async_runtime::spawn(async move {
            let hit = tauri::async_runtime::spawn_blocking(move || {
                let stamp = stamp_or_skip(&source_path)?;
                cache::read_probs(&cache_path, width, height, &hash, &stamp)
                    .map(|probs| (build_prob_pyramid(&probs, &level_dims), probs))
            })
            .await
            .ok()
            .flatten();
            if let Some((pyramid, probs)) = hit {
                if let Ok(mut guard) = app_for_restore.state::<Mutex<Images>>().lock() {
                    guard.set_probs_built(id, probs, pyramid);
                }
            }
        });
    }

    roll.touch(index)?;
    let queue = app.state::<jobs::JobQueue>();
    evict_over_budget(&images, &roll, &queue, index)?;

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
        // Bounded re-arm loop, closing the lost-wakeup window between the
        // final frame of a drain and the scanning flag actually clearing: if
        // `open_roll` swaps in roll B while roll A's scan is still draining,
        // B's own `scan_roll` call sees `scanning == true`, returns its
        // idempotent Ok, and never scans -- unless this task notices the new
        // roll's frames after it finishes A's and re-arms itself. Capped at
        // 3 re-arms against a pathological caller that keeps swapping rolls
        // faster than a single re-arm can drain; each iteration still
        // re-resolves generation, indices, AND roll_dir fresh, since a
        // re-arm can hand this task an entirely different roll.
        const MAX_REARMS: u32 = 3;
        let mut generation = generation;
        let mut roll_dir = roll_dir;
        let mut indices = indices;
        let mut rearms = 0u32;
        'rearm: loop {
            // Two passes over the same index snapshot: thumbnails first (cheap,
            // seconds each) so the filmstrip fills in immediately, then
            // detections (slow, ~9s each) trickle in behind them. A frame whose
            // decode fails in pass 1 has no pixels to detect on, so its index is
            // recorded here and pass 2 skips it outright.
            let mut failed_indices: std::collections::HashSet<usize> =
                std::collections::HashSet::new();

            // Pass 1: thumbnails.
            for &index in &indices {
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
                        let _ = app_for_task
                            .emit("roll-frame-error", RollFrameError { index, message: e });
                        failed_indices.insert(index);
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
                let thumb_path = roll::thumb_path(&roll_dir, &file_name);
                if thumb_path.exists() {
                    // Backfill run: this frame already has a thumbnail from an
                    // earlier scan, so there's nothing to decode here.
                    continue;
                }
                #[cfg(debug_assertions)]
                eprintln!("[queue] frame {index} thumbnail");
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
                    Ok::<_, String>(thumb)
                    // `prepared` (pyramid + decoded pixels) drops here; pass 2
                    // decodes its own copy so at most one frame's pixels are
                    // resident at a time across the whole scan.
                })
                .await
                .map_err(|e| e.to_string())
                .and_then(|r| r);

                match staged {
                    Ok(Ok(())) => {
                        let _ = app_for_task.emit("roll-thumb", RollThumb { index });
                    }
                    Ok(Err(e)) => {
                        // Decode succeeded but the thumbnail write failed: the
                        // frame still has pixels, so it still detects in pass 2.
                        let _ = app_for_task.emit(
                            "roll-frame-error",
                            RollFrameError {
                                index,
                                message: format!("thumbnail: {e}"),
                            },
                        );
                    }
                    Err(message) => {
                        // Decode itself failed: no pixels, so pass 2 can't
                        // detect on this frame either.
                        let _ = roll_state.record_scan_result(generation, index, None, None);
                        let _ = app_for_task
                            .emit("roll-frame-error", RollFrameError { index, message });
                        failed_indices.insert(index);
                    }
                }
            }

            // Pass 2: detections.
            for &index in &indices {
                if failed_indices.contains(&index) {
                    continue;
                }
                let roll_state = app_for_task.state::<roll::RollState>();
                if roll_state.generation() != generation {
                    break;
                }
                let path = match roll_state.frame_path(index) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = roll_state.record_scan_result(generation, index, None, None);
                        let _ = app_for_task
                            .emit("roll-frame-error", RollFrameError { index, message: e });
                        continue;
                    }
                };
                let file_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default()
                    .to_string();
                #[cfg(debug_assertions)]
                eprintln!("[queue] frame {index} detect starting");
                // No pyramid here (ff90566's export-queue shape): this pass only
                // ever reads the decoded pixels for detection, so building the
                // multi-resolution display pyramid would just waste memory/CPU.
                let detector = detector.clone();
                let cache_path = roll::probs_cache_path(&roll_dir, &file_name);
                let outcome = tauri::async_runtime::spawn_blocking(move || {
                    // Stamp BEFORE decoding (fail-safe direction: a source
                    // changed after the stat mismatches on the next read
                    // instead of pairing a fresh stamp with stale pixels).
                    let stamp = stamp_or_skip(&path);
                    let image = images::Images::decode_stage(&path)?;
                    // detect_hashed pairs the output with the hash of the model
                    // that produced it under one lock -- see its doc comment --
                    // so the cache write below can never record a different
                    // model's hash.
                    let (probs, hash) = detector.detect_hashed(&image)?;
                    let bboxes = images::components_from_probs(
                        &probs,
                        image.width,
                        image.height,
                        SCAN_THRESHOLD,
                    );
                    // Cache write is sequential with detection here (milliseconds
                    // against a ~9s detect); failures eprintln in debug only,
                    // never fail the scan. A source stat failure skipped the
                    // stamp above, which skips this write outright.
                    if let Some(stamp) = stamp {
                        if let Err(_e) = cache::write_probs(
                            &cache_path,
                            &probs,
                            image.width,
                            image.height,
                            &hash,
                            &stamp,
                        ) {
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
                        let _ = app_for_task
                            .emit("roll-frame-error", RollFrameError { index, message });
                    }
                }
            }
            // roll-done fires once per drain iteration, not once per task. That
            // matches the existing frontend contract (it sets scanDone = true on
            // receipt) and is harmless to repeat on a re-arm: the second receipt
            // is an idempotent no-op set, not a second "scan started" signal.
            let _ = app_for_task.emit("roll-done", ());

            // Clear-then-recheck handshake, mirroring the job worker's. Clear
            // the flag before re-resolving frames_to_scan, so any open_roll that
            // lands after this clear is guaranteed to observe scanning == false
            // rather than finding the flag held and returning its idempotent Ok
            // without scanning.
            let roll_state = app_for_task.state::<roll::RollState>();
            roll_state.clear_scanning();

            if rearms >= MAX_REARMS {
                #[cfg(debug_assertions)]
                eprintln!("[scan] re-arm cap ({MAX_REARMS}) reached; not re-arming further");
                break 'rearm;
            }

            // Re-resolve under the CURRENT generation -- not the one this
            // iteration started with. If open_roll swapped in a different roll
            // while the drain above was running, frames_to_scan() and dir() now
            // describe that new roll, which is exactly the case this loop
            // exists to catch: without it, the new roll's own scan_roll call
            // would have seen scanning == true, returned Ok, and never scanned.
            let next_generation = roll_state.generation();
            let setup = roll_state
                .dir()
                .and_then(|dir| Ok((dir, roll_state.frames_to_scan()?)));
            let (next_roll_dir, next_indices) = match setup {
                Ok(v) => v,
                Err(_) => break 'rearm, // no roll open (or lookup failed): nothing to re-arm for
            };
            if next_indices.is_empty() {
                break 'rearm;
            }

            // Only one of this task's re-arm and a racing scan_roll's own
            // compare_exchange may win the flag back. On loss, the racing call
            // already spawned (or is about to spawn) its own task that will
            // scan next_indices itself, so this task must not also proceed.
            if roll_state
                .scanning
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                break 'rearm;
            }

            generation = next_generation;
            roll_dir = next_roll_dir;
            indices = next_indices;
            rearms += 1;
        }
    });
    Ok(())
}

/// One background job's identity; the payload for the `job-queued`,
/// `job-started`, and `job-done` events. `generation` is the roll generation
/// the job was enqueued against (`job.generation`), so a listener can drop
/// events belonging to a roll that has since been swapped out -- closing the
/// same-index-across-roll-swap race at the event layer, not just the
/// registry-write layer.
#[derive(serde::Serialize, Clone)]
struct JobEvent {
    index: usize,
    kind: jobs::JobKind,
    generation: u64,
}

#[derive(serde::Serialize, Clone)]
struct JobError {
    index: usize,
    kind: jobs::JobKind,
    message: String,
    generation: u64,
}

#[derive(serde::Serialize, Clone)]
struct QueueIdlePayload {
    generation: u64,
}

/// A coarse stage marker for one frame's export, emitted from inside the
/// export job's `spawn_blocking` closure so a slow transient heal (cache
/// miss) shows the operator something more specific than "running".
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

#[derive(serde::Serialize, Clone)]
struct ExportProgress {
    index: usize,
}

/// Clears the job-worker running flag when dropped, including on unwind, so
/// a panic anywhere in the drain task can never wedge the queue permanently.
/// Mirrors `ScanFlagGuard`.
struct JobFlagGuard(tauri::AppHandle);

impl Drop for JobFlagGuard {
    fn drop(&mut self) {
        self.0.state::<jobs::JobQueue>().clear_running();
    }
}

/// Clears the job queue's eviction pin when dropped, so an early `?` in a
/// job body can never leave a stale pin shielding a long-gone frame from
/// eviction.
struct PinGuard(tauri::AppHandle);

impl Drop for PinGuard {
    fn drop(&mut self) {
        let _ = self.0.state::<jobs::JobQueue>().pin(None);
    }
}

/// Runs the fallible body of one background job. Split out so the worker
/// loop's done/error emission is a single match, mirroring the export
/// queue's per-frame outcome shape.
async fn run_job(app: &tauri::AppHandle, generation: u64, job: jobs::Job) -> Result<(), String> {
    let index = job.index;
    let (path, file_name, threshold, strokes) =
        app.state::<roll::RollState>().export_frame_meta(index)?;
    let image_id = app.state::<roll::RollState>().image_id(index)?;
    let _ = app.emit(
        "job-started",
        JobEvent {
            index,
            kind: job.kind,
            generation,
        },
    );
    // Cache paths are keyed by the frame's own roll directory, mirroring
    // run_heal's transient path.
    let roll_dir = path
        .parent()
        .unwrap_or_else(|| std::path::Path::new(""))
        .to_path_buf();

    match job.kind {
        jobs::JobKind::Detect => {
            let images_state = app.state::<Mutex<Images>>();
            // Pin before reading registry Arcs so eviction cannot pull the
            // entry out from under the job; the guard unpins on every exit.
            let mut _pin_guard = None;
            let resident = match image_id {
                Some(id) => {
                    app.state::<jobs::JobQueue>().pin(Some(id))?;
                    _pin_guard = Some(PinGuard(app.clone()));
                    let images = images_state.lock().map_err(|e| e.to_string())?;
                    match images.image(id) {
                        Some(img) => {
                            if images.has_probs(id) {
                                // Resident and already detected: nothing to do.
                                return Ok(());
                            }
                            let level_dims = images
                                .level_dims(id)
                                .ok_or_else(|| format!("no image {id}"))?;
                            Some((id, img, level_dims))
                        }
                        None => None, // stale id: fall through to a fresh decode
                    }
                }
                None => None,
            };
            // The early resident id only gated the pin (already applied above
            // via `image_id`) and the has-probs short-circuit; the write
            // target below is resolved fresh after compute (see I3's comment
            // at the write site), so it is not carried out of this match.
            let (image, level_dims) = match resident {
                Some((_id, img, dims)) => (img, Some(dims)),
                None => {
                    let path = path.clone();
                    let img =
                        tauri::async_runtime::spawn_blocking(move || Images::decode_stage(&path))
                            .await
                            .map_err(|e| e.to_string())??;
                    (img, None)
                }
            };
            let detector_state = app.state::<detect::DetectorState>();
            let detector = detector_state.inner().clone();
            // Read-path lookup only: a race against a concurrent model swap
            // means at worst a benign cache miss (falls through to a fresh,
            // correctly-paired detect below via detect_hashed).
            let detector_hash = detector_state.hash();
            let cache_path = roll::probs_cache_path(&roll_dir, &file_name);
            let source_path = path.clone();
            let (probs, pyramid, bboxes) = tauri::async_runtime::spawn_blocking(move || {
                // A stat failure skips the cache interaction entirely (both
                // the lookup below and the write on a miss).
                let stamp = stamp_or_skip(&source_path);
                // A valid cache entry for the current detector replaces the
                // (seconds-long) model run: a detect job means detection is
                // wanted, not necessarily recomputed.
                let cached = match (&detector_hash, &stamp) {
                    (Some(hash), Some(stamp)) => {
                        cache::read_probs(&cache_path, image.width, image.height, hash, stamp)
                    }
                    _ => None,
                };
                let probs = match cached {
                    Some(probs) => probs,
                    None => {
                        // detect_hashed pairs the output with the hash of the
                        // model that produced it under one lock -- see its
                        // doc comment -- so the cache write below can never
                        // record a different model's hash.
                        let (probs, hash) = detector.detect_hashed(&image)?;
                        if let Some(stamp) = &stamp {
                            if let Err(_e) = cache::write_probs(
                                &cache_path,
                                &probs,
                                image.width,
                                image.height,
                                &hash,
                                stamp,
                            ) {
                                #[cfg(debug_assertions)]
                                eprintln!("[jobs] frame {index} probs cache write failed: {_e}");
                            }
                        }
                        probs
                    }
                };
                let bboxes = images::components_from_probs(
                    &probs,
                    image.width,
                    image.height,
                    SCAN_THRESHOLD,
                );
                let pyramid = level_dims.map(|dims| build_prob_pyramid(&probs, &dims));
                Ok::<_, String>((probs, pyramid, bboxes))
            })
            .await
            .map_err(|e| e.to_string())??;
            // Re-resolve the resident id AFTER compute, not before: the frame
            // may have gone from non-resident to resident (the operator
            // activated it mid-job) since `image_id` was read at job start.
            // The file is unchanged either way, so a pyramid built now still
            // validates against whichever id is current; `set_probs_built`
            // rejects it otherwise.
            let current_id = app.state::<roll::RollState>().image_id(index)?;
            // A pyramid built off the early `level_dims` covers the common
            // case (already resident at job start); a late-resident frame
            // has no early `level_dims`, so fetch them now off the current id
            // instead of leaving the registry write unpopulated.
            let pyramid = match pyramid {
                Some(pyramid) => Some(pyramid),
                None => {
                    let dims = current_id.and_then(|id| {
                        images_state
                            .lock()
                            .ok()
                            .and_then(|images| images.level_dims(id))
                    });
                    dims.map(|dims| build_prob_pyramid(&probs, &dims))
                }
            };
            // Land in the registry only when this job's generation is still
            // the live one: a roll swap can free `index` and a fresh roll's
            // frame can activate the very same index before this job
            // finishes, giving `current_id` a real (but WRONG-roll) image to
            // write into. Same index, same dims, different roll -- nothing
            // else here catches that. The sidecar write below stays
            // unconditional: it is roll-dir-scoped and content-validated
            // (`record_scan_result` re-checks the generation itself), so a
            // stale-generation result simply lands in the (still-correct)
            // directory or is discarded there, never in the wrong roll.
            if job.generation == app.state::<roll::RollState>().generation() {
                if let (Some(id), Some(pyramid)) = (current_id, pyramid) {
                    let mut images = images_state.lock().map_err(|e| e.to_string())?;
                    // A frame closed mid-job is benign for a background detect:
                    // the sidecar result below still lands.
                    images.set_probs_built(id, probs, pyramid);
                }
            }
            let count = bboxes.len();
            app.state::<roll::RollState>().record_scan_result(
                generation,
                index,
                Some(count),
                Some(bboxes.clone()),
            )?;
            // The existing scan event shape, so the filmstrip count/rings
            // update exactly like a roll scan.
            let _ = app.emit(
                "roll-progress",
                RollProgress {
                    index,
                    count: Some(count),
                    bboxes: Some(bboxes),
                },
            );
            Ok(())
        }
        jobs::JobKind::Heal => {
            if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                return Err(format!("threshold {threshold} out of range"));
            }
            masks::validate_strokes(&strokes)?;
            let images_state = app.state::<Mutex<Images>>();
            // Pin before reading registry Arcs; see the detect arm.
            let mut _pin_guard = None;
            let (resident_id, registry_image, registry_mask) = match image_id {
                Some(id) => {
                    app.state::<jobs::JobQueue>().pin(Some(id))?;
                    _pin_guard = Some(PinGuard(app.clone()));
                    let images = images_state.lock().map_err(|e| e.to_string())?;
                    match images.image(id) {
                        Some(img) => {
                            // None until a detection is resident; the closure
                            // below then falls back to the probs cache or a
                            // fresh model run.
                            let mask = images.threshold_mask(id, threshold);
                            (Some(id), Some(img), mask)
                        }
                        None => (None, None, None), // stale id: fresh decode
                    }
                }
                None => (None, None, None),
            };
            let image = match registry_image {
                Some(img) => img,
                None => {
                    let path = path.clone();
                    tauri::async_runtime::spawn_blocking(move || Images::decode_stage(&path))
                        .await
                        .map_err(|e| e.to_string())??
                }
            };
            let detector_state = app.state::<detect::DetectorState>();
            let detector = detector_state.inner().clone();
            // Read path only: this hash feeds provenance and the probs-cache
            // lookup below, not a cache write, so a race against a concurrent
            // model swap means at worst a benign cache/provenance miss. The
            // Option distinguishes "no detector loaded" (no cache IO) from
            // the zeros sentinel run_heal folds into provenance.
            let detector_hash_opt = detector_state.hash();
            let detector_hash = detector_hash_opt.unwrap_or([0u8; 32]);
            let inpainter = app.state::<detect::InpainterState>().inner().clone();
            let heal_cache_path = roll::heal_cache_path(&roll_dir, &file_name);
            let probs_cache_path = roll::probs_cache_path(&roll_dir, &file_name);
            let source_path = path.clone();
            let app_for_progress = app.clone();
            let (healed, pyramid, mask) = tauri::async_runtime::spawn_blocking(move || {
                // A stat failure skips every cache interaction below (heal
                // read/write and probs read/write alike) -- never fails the
                // job.
                let stamp = stamp_or_skip(&source_path);

                // Everything provenance-dependent happens inside the one
                // with_inpainter_hashed lock, exactly as in run_heal and the
                // export queue's transient path.
                inpainter.with_inpainter_hashed(|pair| {
                    let inpainter_hash = pair.as_ref().map(|(_, h)| *h).unwrap_or([0u8; 32]);
                    let provenance = stamp.as_ref().map(|stamp| {
                        cache::heal_provenance(
                            threshold,
                            HEAL_DILATE_RADIUS,
                            &strokes,
                            &detector_hash,
                            &inpainter_hash,
                            stamp,
                        )
                    });

                    if let Some(provenance) = &provenance {
                        if let Some((healed, mask)) =
                            cache::read_heal(&heal_cache_path, &image, provenance)
                        {
                            let healed = std::sync::Arc::new(healed);
                            // Always build: the write-target id is resolved AFTER
                            // this closure returns (the frame may have gone
                            // non-resident-to-resident mid-job), so building only
                            // when `resident_id` was already Some would silently
                            // drop the registry write for a late-resident frame.
                            let pyramid = fd_tiles::Pyramid::build(&healed);
                            return Ok::<_, String>((
                                healed,
                                Some(pyramid),
                                std::sync::Arc::new(mask),
                            ));
                        }
                    }

                    // Probs mask: the registry's live detection first, then
                    // the probs cache, then a fresh model run (cached for
                    // next time) -- only after the heal cache misses, so a
                    // cached heal never pays for a detect.
                    let raw: Vec<bool> = match registry_mask {
                        Some(m) => m,
                        None => {
                            let cached = match (&detector_hash_opt, &stamp) {
                                (Some(hash), Some(stamp)) => cache::read_probs(
                                    &probs_cache_path,
                                    image.width,
                                    image.height,
                                    hash,
                                    stamp,
                                ),
                                _ => None,
                            };
                            let probs = match cached {
                                Some(p) => p,
                                None => {
                                    // detect_hashed pairs the output with the
                                    // hash of the model that produced it
                                    // under one lock -- see its doc comment
                                    // -- so the cache write below can never
                                    // record a different model's hash.
                                    let (probs, hash) = detector.detect_hashed(&image)?;
                                    if let Some(stamp) = &stamp {
                                        if let Err(_e) = cache::write_probs(
                                            &probs_cache_path,
                                            &probs,
                                            image.width,
                                            image.height,
                                            &hash,
                                            stamp,
                                        ) {
                                            #[cfg(debug_assertions)]
                                            eprintln!(
                                                "[jobs] frame {index} probs cache write failed: {_e}"
                                            );
                                        }
                                    }
                                    probs
                                }
                            };
                            probs.iter().map(|&p| p > threshold).collect()
                        }
                    };

                    let mask = masks::compose_heal_mask(
                        raw,
                        image.width,
                        image.height,
                        HEAL_DILATE_RADIUS,
                        &strokes,
                    );
                    let mut copy = (*image).clone(); // the original Arc stays pristine
                    let inp = pair.map(|(inp, _)| inp);
                    fd_heal::heal_with_progress(&mut copy, &mask, inp, &mut |done, total| {
                        // Per-defect progress only for a resident frame: the
                        // display contract keys heal-progress off a live
                        // image id (the current-frame status line).
                        if let Some(id) = resident_id {
                            let _ = app_for_progress
                                .emit("heal-progress", HealProgress { id, done, total });
                        }
                    })
                    .map_err(|e| e.to_string())?;
                    if let Some(provenance) = &provenance {
                        if let Err(_e) =
                            cache::write_heal(&heal_cache_path, &image, &copy, &mask, provenance)
                        {
                            #[cfg(debug_assertions)]
                            eprintln!(
                                "[cache] heal write failed for frame {index} ({file_name}): {_e}"
                            );
                        }
                    }
                    let healed = std::sync::Arc::new(copy);
                    // Always build; see the cache-hit branch above for why.
                    let pyramid = fd_tiles::Pyramid::build(&healed);
                    Ok((healed, Some(pyramid), std::sync::Arc::new(mask)))
                })?
            })
            .await
            .map_err(|e| e.to_string())??;
            // Re-resolve the resident id AFTER compute, not before: the frame
            // may have gone from non-resident to resident (the operator
            // activated it mid-job) since `image_id` was read at job start.
            // The file is the same either way, so the healed image's dims
            // still validate against whichever id is current;
            // `set_healed` rejects it otherwise. The early `resident_id` was
            // only needed for the pin and to reuse a resident decode/mask as
            // compute input -- both already happened above.
            let current_id = app.state::<roll::RollState>().image_id(index)?;
            // Land in the registry only when this job's generation is still
            // the live one -- see the Detect arm's identical check. The heal
            // cache write above stays unconditional: it is roll-dir-scoped
            // and content-validated, so it is safe regardless of generation.
            if job.generation == app.state::<roll::RollState>().generation() {
                if let (Some(id), Some(pyramid)) = (current_id, pyramid) {
                    let mut images = images_state.lock().map_err(|e| e.to_string())?;
                    // A frame closed mid-heal is benign here: the heal cache
                    // already holds the result for the next activation or export.
                    images.set_healed(id, healed, pyramid, mask);
                }
            }
            Ok(())
        }
        jobs::JobKind::Export => {
            let dest_dir = app
                .state::<roll::RollState>()
                .export_dest()?
                .ok_or("no export destination set")?;
            let dest = dest_dir.join(&file_name);

            // Prefer already-healed registry data (the operator reviewed it).
            // Pin before reading the Arcs so eviction cannot pull the entry
            // out from under the export; the guard unpins on every exit.
            let mut _pin_guard = None;
            let registry_export = match image_id {
                Some(id) => {
                    app.state::<jobs::JobQueue>().pin(Some(id))?;
                    _pin_guard = Some(PinGuard(app.clone()));
                    let images_state = app.state::<Mutex<Images>>();
                    let images = images_state.lock().map_err(|e| e.to_string())?;
                    images.healed_parts(id)
                }
                None => None,
            };

            if let Some((original, healed, mask)) = registry_export {
                let dest_for_write = dest.clone();
                tauri::async_runtime::spawn_blocking(move || {
                    export::export_healed(&original, &healed, &mask, &dest_for_write).map(|_| ())
                })
                .await
                .map_err(|e| e.to_string())
                .and_then(|r| r)?;
            } else {
                // No registry entry: heal cache, then the transient detect/heal
                // pipeline. Identical to run_heal's discipline: detector hash
                // resolved before the closure (read path, benign miss on race);
                // everything provenance-dependent inside one with_inpainter_hashed.
                let detector = app.state::<detect::DetectorState>().inner().clone();
                let inpainter = app.state::<detect::InpainterState>().inner().clone();
                let detector_hash = detector.hash().unwrap_or([0u8; 32]);
                let cache_path = roll::heal_cache_path(&roll_dir, &file_name);
                let app_for_stages = app.clone();
                let path_for_task = path.clone();
                let file_name_for_log = file_name.clone();
                tauri::async_runtime::spawn_blocking(move || {
                    let stage = |s: &'static str| {
                        let _ = app_for_stages
                            .emit("export-frame-stage", ExportFrameStage { index, stage: s });
                    };
                    // Stamp BEFORE decoding -- see run_heal's identical comment.
                    // A stat failure skips the heal-cache interaction (read AND
                    // write); it must never fail the export.
                    let stamp = stamp_or_skip(&path_for_task);
                    let image = images::Images::decode_stage(&path_for_task)?;
                    masks::validate_strokes(&strokes)?;

                    inpainter.with_inpainter_hashed(|pair| {
                        let inpainter_hash = pair.as_ref().map(|(_, h)| *h).unwrap_or([0u8; 32]);
                        let provenance = stamp.as_ref().map(|stamp| {
                            cache::heal_provenance(
                                threshold,
                                HEAL_DILATE_RADIUS,
                                &strokes,
                                &detector_hash,
                                &inpainter_hash,
                                stamp,
                            )
                        });

                        if let Some(provenance) = &provenance {
                            if let Some((healed, mask)) =
                                cache::read_heal(&cache_path, &image, provenance)
                            {
                                stage("writing");
                                return export::export_healed(&image, &healed, &mask, &dest)
                                    .map(|_| ());
                            }
                        }

                        stage("detecting");
                        let probs = detector.detect(&image)?;
                        let raw: Vec<bool> = probs.iter().map(|&p| p > threshold).collect();
                        let mask = masks::compose_heal_mask(
                            raw,
                            image.width,
                            image.height,
                            HEAL_DILATE_RADIUS,
                            &strokes,
                        );
                        stage("healing");
                        let mut copy = (*image).clone();
                        let inp = pair.map(|(inp, _)| inp);
                        fd_heal::heal_with_progress(&mut copy, &mask, inp, &mut |done, total| {
                            let _ = app_for_stages.emit(
                                "export-heal-progress",
                                ExportHealProgress { index, done, total },
                            );
                        })
                        .map_err(|e| e.to_string())?;
                        // Cache the fresh heal so the next export or heal of this
                        // frame is instant.
                        if let Some(provenance) = &provenance {
                            if let Err(_e) =
                                cache::write_heal(&cache_path, &image, &copy, &mask, provenance)
                            {
                                #[cfg(debug_assertions)]
                                eprintln!(
                                    "[cache] heal write failed for frame {index} ({file_name_for_log}): {_e}"
                                );
                            }
                        }
                        stage("writing");
                        export::export_healed(&image, &copy, &mask, &dest).map(|_| ())
                    })?
                })
                .await
                .map_err(|e| e.to_string())
                .and_then(|r| r)?;
            }

            // set_exported re-checks the generation under its own lock; the
            // worker's top-of-loop check plus this makes a roll-swap-mid-export
            // land nowhere.
            let _ = app
                .state::<roll::RollState>()
                .set_exported(generation, index);
            let _ = app.emit("export-progress", ExportProgress { index });
            Ok(())
        }
    }
}

#[tauri::command]
fn enqueue_job(
    app: tauri::AppHandle,
    roll: State<'_, roll::RollState>,
    queue: State<'_, jobs::JobQueue>,
    kind: jobs::JobKind,
    index: usize,
    front: bool,
) -> Result<(), String> {
    // Bounds/roll validation before anything lands in the queue: errors when
    // no roll is open or the index is out of range.
    roll.image_id(index)?;
    let generation = roll.generation();
    queue.enqueue(
        jobs::Job {
            kind,
            index,
            generation,
        },
        front,
    )?;
    let _ = app.emit(
        "job-queued",
        JobEvent {
            index,
            kind,
            generation,
        },
    );
    spawn_worker_if_idle(&app, generation);
    Ok(())
}

#[tauri::command]
fn enqueue_exports(
    app: tauri::AppHandle,
    roll: State<'_, roll::RollState>,
    queue: State<'_, jobs::JobQueue>,
    dest_dir: String,
) -> Result<(), String> {
    // Validation before anything lands in the queue: errors when no roll
    // is open. frames_to_export already includes previously exported
    // frames -- re-export is deliberate, predictable overwrite.
    let indices = roll.frames_to_export()?;
    roll.set_export_dest(std::path::PathBuf::from(dest_dir))?;
    let generation = roll.generation();
    for index in indices {
        let newly_queued = queue.enqueue(
            jobs::Job {
                kind: jobs::JobKind::Export,
                index,
                generation,
            },
            false, // back of queue: never jumps ahead of queued heals
        )?;
        if newly_queued {
            let _ = app.emit(
                "job-queued",
                JobEvent {
                    index,
                    kind: jobs::JobKind::Export,
                    generation,
                },
            );
        }
    }
    spawn_worker_if_idle(&app, generation);
    Ok(())
}

/// Claims the worker flag and spawns the drain loop if no worker is
/// running. `spawn_generation` is the roll generation at the caller's
/// enqueue time; see the comment inside for why the terminal queue-idle
/// must carry it rather than a generation read at emit time.
fn spawn_worker_if_idle(app: &tauri::AppHandle, spawn_generation: u64) {
    let queue = app.state::<jobs::JobQueue>();
    if !queue.try_start() {
        return; // a worker is already draining; it will reach the new jobs
    }
    // The flag is set; any fallible setup from here must clear it before
    // returning (the scan_roll discipline).
    let app_for_task = app.clone();
    tauri::async_runtime::spawn(async move {
        let _job_flag_guard = JobFlagGuard(app_for_task.clone());
        // Labeled so the exit handshake below can `continue 'drain` to adopt
        // a straggler without re-entering the whole spawn.
        'drain: loop {
            // Drains until empty; a poisoned queue lock also ends the drain
            // (falls through to the handshake below with a None-shaped pop,
            // same as a genuinely empty queue).
            while let Ok(Some(job)) = app_for_task.state::<jobs::JobQueue>().pop() {
                // Per-job check, not a snapshot-at-drain-start break: the
                // drain loop pops before it can know whether the job belongs
                // to the roll open when the worker started or a roll swapped
                // in since. A stale-roll job is silently discarded (its
                // siblings were already cleared from the queue by the roll
                // swap; this only catches the one straggler popped
                // mid-swap) while a job tagged with the CURRENT generation
                // always runs, regardless of when this worker began
                // draining.
                if job.generation != app_for_task.state::<roll::RollState>().generation() {
                    continue;
                }
                let generation = job.generation;
                match run_job(&app_for_task, generation, job).await {
                    Ok(()) => {
                        let _ = app_for_task.emit(
                            "job-done",
                            JobEvent {
                                index: job.index,
                                kind: job.kind,
                                generation,
                            },
                        );
                    }
                    Err(message) => {
                        let _ = app_for_task.emit(
                            "job-error",
                            JobError {
                                index: job.index,
                                kind: job.kind,
                                message,
                                generation,
                            },
                        );
                    }
                }
            }

            // Clear-then-recheck handshake, closing the lost-wakeup window
            // between `pop()` returning None and the flag actually clearing
            // (a caller's `enqueue_job` could observe `running == true` in
            // that gap, coalesce into the queue, and see the flag already
            // taken -- so it would return without spawning a drain, and
            // this worker was about to exit without seeing the new job).
            let queue = app_for_task.state::<jobs::JobQueue>();

            // Step 1: pop() just returned None. Clear the flag now, before
            // re-checking the queue, so any enqueue that lands from this
            // point on is guaranteed to observe `running == false` if it
            // arrives after this clear (the guard's later duplicate clear
            // on drop is harmless -- clearing an already-clear flag is a
            // no-op).
            queue.clear_running();

            // Step 2: re-check the queue under the now-cleared flag. If
            // it's still empty (or the lock is poisoned), no job could have
            // landed and lost its wakeup: nothing enqueued between step 1
            // and here would have found `running == true`, so nothing was
            // silently dropped. This worker is the one that gets to emit
            // the terminal idle event.
            let still_empty = queue.is_empty().unwrap_or(true);
            if still_empty {
                let _ = app_for_task.emit(
                    "queue-idle",
                    QueueIdlePayload {
                        generation: spawn_generation,
                    },
                );
                break 'drain;
            }

            // Step 3: the queue is non-empty, which means either (a) this
            // worker's own clear in step 1 raced an enqueue that landed a
            // job right after, or (b) that same enqueue also re-armed the
            // flag itself and spawned its own drain task. Only one of this
            // worker and that racing enqueue may resume draining, so settle
            // it with the same compare_exchange primitive both sides use.
            if queue.try_start() {
                // Won: the racing enqueue's `job-queued` event already
                // fired and its own try_start lost (or hasn't run yet), so
                // this worker adopts the straggler and keeps draining under
                // the same flag acquisition. No idle event yet -- the queue
                // is not idle.
                continue 'drain;
            }
            // Lost: the racing enqueue's try_start won first and spawned
            // its own worker, which will run this exact handshake and emit
            // its own queue-idle when it eventually drains empty. Emitting
            // here too would be a duplicate terminal event for a worker
            // that no longer owns the flag, so this one exits silently.
            break 'drain;
        }
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Mutex::new(Images::default()))
        .manage(detect::DetectorState::default())
        .manage(detect::InpainterState::default())
        .manage(roll::RollState::default())
        .manage(jobs::JobQueue::default())
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
            enqueue_job,
            enqueue_exports,
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
    fn frames_to_scan_lists_uncounted_and_uncached_frames() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.png"), b"x").unwrap();
        std::fs::write(dir.path().join("b.png"), b"x").unwrap();
        let state = roll::RollState::default();
        state.open(dir.path()).unwrap();
        state
            .record_scan_result(state.generation(), 0, Some(2), Some(vec![]))
            .unwrap();
        // Frame 0 is counted but has no probs cache file: it backfills, so
        // both frames are queued (rolls scanned before the cache existed
        // must not silently never cache).
        assert_eq!(state.frames_to_scan().unwrap(), vec![0, 1]);
        // Once its probs cache exists, a counted frame leaves the queue.
        let cache_path = roll::probs_cache_path(dir.path(), "a.png");
        let stamp = crate::cache::source_stamp(&dir.path().join("a.png")).unwrap();
        crate::cache::write_probs(&cache_path, &[0.5], 1, 1, &[7u8; 32], &stamp).unwrap();
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
