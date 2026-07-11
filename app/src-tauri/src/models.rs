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

/// Single-flight guard for `download_inpaint_model`, mirroring `RollState`'s
/// `scanning`/`exporting` flags.
pub struct ModelDownloadState(pub AtomicBool);

impl Default for ModelDownloadState {
    fn default() -> Self {
        ModelDownloadState(AtomicBool::new(false))
    }
}

/// Clears the download-in-progress flag when dropped, including on unwind,
/// so a panic anywhere in the download task can never wedge the single-flight
/// gate permanently. Mirrors `lib.rs`'s `ScanFlagGuard`.
struct DownloadFlagGuard(tauri::AppHandle);

impl Drop for DownloadFlagGuard {
    fn drop(&mut self) {
        self.0
            .state::<ModelDownloadState>()
            .0
            .store(false, Ordering::SeqCst);
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

/// Path for the in-progress download before it is verified and renamed into
/// place. The `tmp-unduster` suffix keeps it visually distinct from a real
/// model file and easy to filter out of any directory listing.
fn lama_tmp_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(models_dir(app)?.join("lama.onnx.tmp-unduster"))
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

/// Runs the fallible body of the download: streams `LAMA_URL` to a temp
/// file, verifies its checksum, renames it atomically into place, and loads
/// it into the inpainter. Split out so `download_inpaint_model` can guarantee
/// a terminal event (`model-done` or `model-error`) on every exit path.
async fn run_download(app: &tauri::AppHandle, inpainter: &InpainterState) -> Result<(), String> {
    let tmp = lama_tmp_path(app)?;
    let final_path = lama_path(app)?;

    let response = reqwest::get(LAMA_URL).await.map_err(|e| e.to_string())?;
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
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| e.to_string())?;
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
    let flag = app.state::<ModelDownloadState>();
    if flag
        .0
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Ok(()); // already running; idempotent from the caller's view
    }
    let inpainter = inpainter.inner().clone();
    let app_for_task = app.clone();
    tauri::async_runtime::spawn(async move {
        let _flag_guard = DownloadFlagGuard(app_for_task.clone());
        let result = run_download(&app_for_task, &inpainter).await;
        match result {
            Ok(()) => {
                let _ = app_for_task.emit("model-done", ());
            }
            Err(message) => {
                if let Ok(tmp) = lama_tmp_path(&app_for_task) {
                    let _ = std::fs::remove_file(&tmp);
                }
                let _ = app_for_task.emit("model-error", ModelError { message });
            }
        }
    });
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
