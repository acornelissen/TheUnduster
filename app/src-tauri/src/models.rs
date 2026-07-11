//! Application Support models directory, checksum verification, and
//! single-flight streaming download of the LaMa inpainting model.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use sha2::{Digest, Sha256};
use tauri::{Emitter, Manager, State};

use crate::detect::InpainterState;

/// Pinned to the current revision commit of huggingface.co/Carve/LaMa-ONNX
/// (resolved 2026-07-09 via the HF models API) rather than `main`, so the
/// URL can never silently start serving a different file.
pub const LAMA_URL: &str = "https://huggingface.co/Carve/LaMa-ONNX/resolve/c3c0c9e468934d62e79c329e35d82dd09ff8c444/lama_fp32.onnx";

/// SHA-256 of `lama_fp32.onnx` at revision c3c0c9e468934d62e79c329e35d82dd09ff8c444,
/// computed locally with `shasum -a 256` after downloading LAMA_URL.
pub const LAMA_SHA256: &str = "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6";

/// Generous ceiling above the model's actual size (~207MB); guards against a
/// misbehaving/compromised host streaming an unbounded response.
pub const LAMA_MAX_BYTES: u64 = 300_000_000;

/// Single-flight guard plus cooperative cancel flag for
/// `download_inpaint_model`, mirroring the job queue's running/cancel pair:
/// begin clears any stale cancel, cancel only lands while a download runs,
/// and the download loop polls the flag between chunks.
#[derive(Default)]
pub struct ModelDownloadState {
    running: AtomicBool,
    cancel_requested: AtomicBool,
}

impl ModelDownloadState {
    /// Claims the single-flight flag (false -> true) and clears any cancel
    /// left over from a previous download. Returns false when a download is
    /// already running.
    pub fn try_begin(&self) -> bool {
        let won = self
            .running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok();
        if won {
            self.cancel_requested.store(false, Ordering::SeqCst);
        }
        won
    }

    /// Clears both flags; called by the drop guard when the download task
    /// completes or unwinds.
    pub fn finish(&self) {
        self.cancel_requested.store(false, Ordering::SeqCst);
        self.running.store(false, Ordering::SeqCst);
    }

    /// Requests a cooperative abort of the running download, if any.
    /// Returns true when the request landed on a live download.
    pub fn request_cancel(&self) -> bool {
        if self.running.load(Ordering::SeqCst) {
            self.cancel_requested.store(true, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    /// True when the running download has been asked to stop; polled by the
    /// download loop between chunks.
    pub fn cancel_requested(&self) -> bool {
        self.cancel_requested.load(Ordering::SeqCst)
    }
}

/// Clears the download-in-progress flag when dropped, including on unwind,
/// so a panic anywhere in the download task can never wedge the single-flight
/// gate permanently. Mirrors `lib.rs`'s `ScanFlagGuard`.
struct DownloadFlagGuard(tauri::AppHandle);

impl Drop for DownloadFlagGuard {
    fn drop(&mut self) {
        self.0.state::<ModelDownloadState>().finish();
    }
}

#[derive(serde::Serialize, Clone)]
pub struct ModelProgress {
    pub received: u64,
    pub total: Option<u64>,
}

#[derive(serde::Serialize, Clone)]
pub struct ModelError {
    pub message: String,
}

/// `<app data dir>/models`, created on demand.
pub fn models_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("models");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// `<models_dir>/lama.onnx`.
pub fn lama_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(models_dir(app)?.join("lama.onnx"))
}

/// File name for the in-progress download before it is verified and renamed
/// into place. The `tmp-unduster` marker keeps it visually distinct from a
/// real model file (and is what `sweep_stale_temps` filters on); the pid
/// suffix keeps two app instances from clobbering each other's in-flight
/// download.
fn tmp_file_name(pid: u32) -> String {
    format!("lama.onnx.tmp-unduster-{pid}")
}

fn lama_tmp_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(models_dir(app)?.join(tmp_file_name(std::process::id())))
}

/// Removes `lama.onnx.tmp-unduster*` files in `dir` whose mtime is older
/// than `max_age`, returning how many went. Called once at startup so an
/// interrupted download (app quit or crash mid-stream) does not leave a
/// 100MB+ orphan sitting in Application Support forever. The age check is
/// what makes this safe to run while another instance downloads: a live
/// download rewrites its temp's mtime with every chunk, so anything an hour
/// old is genuinely dead.
pub fn sweep_stale_temps(dir: &Path, max_age: std::time::Duration) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("lama.onnx.tmp-unduster") {
            continue;
        }
        let stale = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|mtime| mtime.elapsed().ok())
            .is_some_and(|age| age >= max_age);
        if stale && std::fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// Streams `path` through SHA-256 in 1MB chunks, returning the raw digest.
/// Streaming (rather than reading the whole file into memory) matters here:
/// the model is ~200MB.
pub fn file_sha256(path: &Path) -> Result<[u8; 32], String> {
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let mut result = [0u8; 32];
    result.copy_from_slice(&digest);
    Ok(result)
}

/// Streams `path` through SHA-256 in 1MB chunks and compares against
/// `expected_hex`. Streaming (rather than reading the whole file into
/// memory) matters here: the model is ~200MB.
pub fn verify_sha256(path: &Path, expected_hex: &str) -> Result<(), String> {
    let digest = file_sha256(path)?;
    let actual_hex = hex_encode(&digest);
    if actual_hex.eq_ignore_ascii_case(expected_hex) {
        Ok(())
    } else {
        Err(format!(
            "checksum mismatch: expected {expected_hex}, got {actual_hex}"
        ))
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Pure decision table behind `inpainter_status`, split out so the four
/// outcomes are directly testable without a `tauri::AppHandle` (which needs
/// a running app context to construct). `loaded`/`fixture` come from one
/// `InpainterState` observation each; `available` is the on-disk check.
///
/// `fixture` takes priority over `available`/`missing` whenever `loaded` is
/// true: the debug autoload always tries real LaMa first and only falls
/// back to the fixture when it is missing or failed to load, so "loaded as
/// fixture" already implies "not really available" from the operator's
/// point of view, regardless of what happens to be sitting on disk.
fn status_str(loaded: bool, fixture: bool, available: bool) -> &'static str {
    if loaded {
        if fixture {
            "fixture"
        } else {
            "loaded"
        }
    } else if available {
        "available"
    } else {
        "missing"
    }
}

#[tauri::command]
pub fn inpainter_status(
    app: tauri::AppHandle,
    inpainter: State<'_, InpainterState>,
) -> Result<String, String> {
    let loaded = inpainter.with_inpainter(|i| i.is_some())?;
    let fixture = inpainter.is_fixture();
    let available = lama_path(&app)?.is_file();
    Ok(status_str(loaded, fixture, available).to_string())
}

/// The most recent real-LaMa load failure recorded against `inpainter`, if
/// any -- e.g. a `lama.onnx` present on disk but corrupt or otherwise
/// unloadable. Polled by the frontend alongside `inpainter_status` at mount
/// (and whenever that status is re-fetched) so a startup load failure that
/// used to be an eprintln-only dead end now reaches the operator. `None`
/// covers both "no failure" and "never attempted"; a subsequent successful
/// load clears it (see `InpainterState::load`).
#[tauri::command]
pub fn inpainter_load_error(
    inpainter: State<'_, InpainterState>,
) -> Result<Option<String>, String> {
    Ok(inpainter.load_error())
}

/// No chunk for this long means the connection stalled: fail the download
/// with a clear message instead of hanging forever (a stalled `reqwest::get`
/// used to wedge the single-flight flag until app restart). Also used as the
/// connect timeout -- reqwest's whole-request `timeout()` would be wrong for
/// a 207MB body, so stall detection is per chunk instead.
const DOWNLOAD_STALL_SECS: u64 = 30;

/// Runs the fallible body of the download: streams `LAMA_URL` to a temp
/// file, verifies its checksum, renames it atomically into place, and loads
/// it into the inpainter. Split out so `download_inpaint_model` can guarantee
/// a terminal event (`model-done`, `model-cancelled`, or `model-error`) on
/// every exit path. A cancel request or a stall surfaces as `Err` here; the
/// caller reads the cancel flag to tell the two apart.
async fn run_download(app: &tauri::AppHandle, inpainter: &InpainterState) -> Result<(), String> {
    let tmp = lama_tmp_path(app)?;
    let final_path = lama_path(app)?;

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(DOWNLOAD_STALL_SECS))
        .build()
        .map_err(|e| e.to_string())?;
    let response = client
        .get(LAMA_URL)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("download failed: HTTP {}", response.status()));
    }
    let total = response.content_length();
    let mut received: u64 = 0;
    let mut file = tokio::fs::File::create(&tmp)
        .await
        .map_err(|e| e.to_string())?;
    let mut stream = response.bytes_stream();
    let mut last_emit = std::time::Instant::now();
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;
    loop {
        // Stall detection per chunk: a healthy download delivers chunks many
        // times a second, so 30s of silence is a dead connection, not a slow
        // one.
        let next = tokio::time::timeout(
            std::time::Duration::from_secs(DOWNLOAD_STALL_SECS),
            stream.next(),
        )
        .await
        .map_err(|_| format!("download stalled: no data for {DOWNLOAD_STALL_SECS}s"))?;
        let Some(chunk) = next else { break };
        let chunk = chunk.map_err(|e| e.to_string())?;
        if app.state::<ModelDownloadState>().cancel_requested() {
            return Err("download cancelled".to_string());
        }
        received += chunk.len() as u64;
        if received > LAMA_MAX_BYTES {
            return Err("download exceeded the expected size".to_string());
        }
        file.write_all(&chunk).await.map_err(|e| e.to_string())?;
        if last_emit.elapsed().as_millis() > 250 {
            last_emit = std::time::Instant::now();
            let _ = app.emit("model-progress", ModelProgress { received, total });
        }
    }
    file.flush().await.map_err(|e| e.to_string())?;
    drop(file);

    let verify_path = tmp.clone();
    tauri::async_runtime::spawn_blocking(move || verify_sha256(&verify_path, LAMA_SHA256))
        .await
        .map_err(|e| e.to_string())??;

    std::fs::rename(&tmp, &final_path).map_err(|e| e.to_string())?;

    let inpainter = inpainter.clone();
    let load_path = final_path.clone();
    tauri::async_runtime::spawn_blocking(move || inpainter.load(&load_path))
        .await
        .map_err(|e| e.to_string())??;

    Ok(())
}

#[tauri::command]
pub fn download_inpaint_model(
    app: tauri::AppHandle,
    inpainter: State<'_, InpainterState>,
) -> Result<(), String> {
    if !app.state::<ModelDownloadState>().try_begin() {
        return Ok(()); // already running; idempotent from the caller's view
    }
    let inpainter = inpainter.inner().clone();
    let app_for_task = app.clone();
    tauri::async_runtime::spawn(async move {
        let _flag_guard = DownloadFlagGuard(app_for_task.clone());
        let result = run_download(&app_for_task, &inpainter).await;
        // Read the cancel flag BEFORE the guard drops and clears it: an Err
        // after a cancel request is the operator's own stop, not a failure
        // to toast about. Same decided-on-the-flag pattern as the job
        // worker's job-cancelled emit.
        let was_cancelled = app_for_task
            .state::<ModelDownloadState>()
            .cancel_requested();
        match result {
            Ok(()) => {
                let _ = app_for_task.emit("model-done", ());
            }
            Err(message) => {
                if let Ok(tmp) = lama_tmp_path(&app_for_task) {
                    let _ = std::fs::remove_file(&tmp);
                }
                if was_cancelled {
                    let _ = app_for_task.emit("model-cancelled", ());
                } else {
                    let _ = app_for_task.emit("model-error", ModelError { message });
                }
            }
        }
    });
    Ok(())
}

/// Requests a cooperative abort of the running model download; the download
/// loop notices between chunks (or its stall timeout fires first) and the
/// terminal `model-cancelled` event follows. A no-op when nothing is
/// downloading -- the outcome the operator wanted already holds.
#[tauri::command]
pub fn cancel_model_download(app: tauri::AppHandle) -> Result<(), String> {
    app.state::<ModelDownloadState>().request_cancel();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_verifies_and_rejects() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("blob.bin");
        std::fs::write(&p, b"the quick brown fox").unwrap();
        // shasum -a 256 of "the quick brown fox"
        let good = "9ecb36561341d18eb65484e833efea61edc74b84cf5e6ae1b81c63533e25fc8f";
        assert!(verify_sha256(&p, good).is_ok());
        let err = verify_sha256(&p, &good.replace('9', "a")).unwrap_err();
        assert!(err.contains("checksum"));
    }

    #[test]
    fn download_state_single_flights_and_cancels_only_while_running() {
        let s = ModelDownloadState::default();
        // Nothing running: a cancel request lands nowhere.
        assert!(!s.request_cancel());
        assert!(!s.cancel_requested());

        assert!(s.try_begin()); // wins the flag
        assert!(!s.try_begin()); // second caller loses
        assert!(s.request_cancel());
        assert!(s.cancel_requested());
        s.finish();
        assert!(!s.cancel_requested());

        // A stale cancel from a previous run must not abort the next one.
        assert!(s.try_begin());
        assert!(!s.cancel_requested());
        s.finish();
    }

    #[test]
    fn tmp_file_name_is_per_process_and_keeps_the_marker() {
        let a = tmp_file_name(123);
        let b = tmp_file_name(456);
        assert_ne!(a, b);
        for name in [&a, &b] {
            assert!(name.starts_with("lama.onnx.tmp-unduster"));
        }
        assert!(a.contains("123"));
    }

    #[test]
    fn sweep_removes_only_stale_temp_files() {
        let dir = tempfile::tempdir().unwrap();
        let stale_tmp = dir.path().join("lama.onnx.tmp-unduster-999");
        let real_model = dir.path().join("lama.onnx");
        let unrelated = dir.path().join("other.bin");
        for p in [&stale_tmp, &real_model, &unrelated] {
            std::fs::write(p, b"x").unwrap();
        }

        // max_age zero: everything just written already counts as stale, so
        // only the filename filter decides what goes.
        let removed = sweep_stale_temps(dir.path(), std::time::Duration::ZERO);
        assert_eq!(removed, 1);
        assert!(!stale_tmp.exists());
        assert!(real_model.exists());
        assert!(unrelated.exists());

        // A temp younger than max_age survives: it may belong to a live
        // download in another instance.
        let live_tmp = dir.path().join("lama.onnx.tmp-unduster-1000");
        std::fs::write(&live_tmp, b"x").unwrap();
        let removed = sweep_stale_temps(dir.path(), std::time::Duration::from_secs(3600));
        assert_eq!(removed, 0);
        assert!(live_tmp.exists());
    }

    #[test]
    fn status_str_reports_loaded_when_a_real_model_is_loaded() {
        assert_eq!(status_str(true, false, false), "loaded");
        assert_eq!(status_str(true, false, true), "loaded");
    }

    #[test]
    fn status_str_reports_fixture_when_the_loaded_model_is_the_dev_stub() {
        // Regardless of what happens to exist on disk: a fixture load means
        // the debug autoload already tried and failed (or found nothing) to
        // load real LaMa, so "available" would be misleading here.
        assert_eq!(status_str(true, true, false), "fixture");
        assert_eq!(status_str(true, true, true), "fixture");
    }

    #[test]
    fn status_str_reports_available_when_unloaded_but_the_file_exists() {
        assert_eq!(status_str(false, false, true), "available");
    }

    #[test]
    fn status_str_reports_missing_when_nothing_is_loaded_or_on_disk() {
        assert_eq!(status_str(false, false, false), "missing");
    }
}
