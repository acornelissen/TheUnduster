# TheUnduster

TheUnduster is a macOS desktop app that finds and removes dust and scratches from scanned film. It's a Tauri 2 + Svelte 5 frontend over a Rust engine. Defect detection runs a neural network through ONNX Runtime, using CoreML on Apple Silicon with a CPU fallback. Healing uses LaMa inpainting, downloaded on first run (207 MB). The app targets Apple Silicon Macs first.

See [docs/user-manual.md](docs/user-manual.md) for how to use the app.

## Repository layout

- `app/` — the desktop app. `app/src/` is the Svelte 5 frontend (Viewer, Filmstrip, StatusBar, queue and log panels). `app/src-tauri/` is the Rust backend: roll and sidecar state, the job queue, model download/verification, and the Tauri commands the frontend calls.
- `engine/` — a Rust workspace of the crates the app is built on:
  - `fd-io` — decode/encode TIFF, PNG, JPEG at 8/16 bit into native-depth pixel buffers.
  - `fd-tiles` — display pyramids and a byte-bounded LRU tile cache.
  - `fd-infer` — tiled ONNX defect detection (512px tiles, 64px overlap, probability averaging).
  - `fd-heal` — tiered healing: classical median fill for small defects, ONNX inpainting plus grain re-synthesis for larger ones, with a bit-exactness guarantee outside the healed mask.
- `training/` — a separate Python/uv pipeline that harvests real defects, trains the detector, exports it to ONNX, and benchmarks it against a labelled roll. Nothing here ships in the app except the exported `.onnx` files.
- `docs/superpowers/` — design docs and implementation plans.

## Toolchain

Mise manages runtimes. Each part of the repo pins its own tools:

- `app/mise.toml` — Node 22, Rust stable
- `engine/mise.toml` — Rust stable
- `training/mise.toml` — Python 3.12, uv

Rust builds run through `cargo`, the frontend through `npm`, training through `uv`.

## Build and run

```
cd app && npm install && mise exec -- npm run tauri dev
```

Release build:

```
cd app && npm run tauri build
```

## Tests and gates

Rust workspace (from the repo root):

```
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

Frontend (from `app/`):

```
npm run test
npm run check
```

Training (from `training/`):

```
uv run pytest
```

## Models

The healing model (LaMa, ONNX) is not bundled. The app downloads it on first use from a pinned Hugging Face revision and verifies it against a pinned SHA-256 before it's used (`app/src-tauri/src/models.rs`). Until it's downloaded, healing falls back to a classical fill with no neural inpainting.

The defect detector currently ships as a fixture model for development, not a trained model — see `training/DATA.md` for the data collection plan and `training/README.md` for the training pipeline. Detection quality is expected to improve once a model is trained on real data and passes the benchmark gate described there.

## Working state per roll

When you open a roll (a folder of scans), the app keeps its working state next to your files, in a `.unduster/` directory inside that folder (`app/src-tauri/src/roll.rs`). This holds:

- `roll.json` — per-frame state: sensitivity threshold, approval/export flags, brush strokes, detected defect boxes.
- `thumbs/` — filmstrip thumbnails.
- `cache/` — cached detection probabilities (`.probs`) and healed-pixel deltas (`.heal`), each keyed to the source file's content and the model that produced them, so a changed file or a changed model invalidates the cache automatically (`app/src-tauri/src/cache.rs`).

Nothing here touches your original scan files.
