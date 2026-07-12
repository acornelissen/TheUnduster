# TheUnduster — Handover

Snapshot for a fresh Claude Code session. Updated 2026-07-12 (cancellation + backlog sweep landed).

## Where things stand

- Branch `main`, working tree clean apart from `.beads/interactions.jsonl` (a passive bd export, leave it). The 2026-07-11/12 sessions landed a large sweep — highlights:
  - **Job/export cancellation (1jc)**: queue-panel row cancel + cancel-all; cooperative abort via `ControlFlow` progress callbacks (per defect / per tile / between export stages); `job-cancelled` event; no partial files, nothing committed on abort.
  - **Model download UX (np6, 36w)**: progress bar + MB counter, stall timeout, cancel button, per-pid temp + startup sweep. The dev fixture is now called the "placeholder model" everywhere, and the status bar always names the healing engine (LaMa / placeholder model / classical only).
  - **Export skip (jhk)**: export provenance recorded in the sidecar; "Export approved" skips unchanged frames, shift-click forces all. Per-frame engine receipts in the log (80i).
  - **Correctness**: activation-time stamps travel with registry pixels to every cache write (02y); classical_fill onion-peels to the core of large defects (cm2); one failed frame no longer re-drains the scan (sol).
  - **Perf/hygiene**: sidecar writes moved off the roll mutex with an ordered-write guard (vd5); last sync CCL off the Images lock (u98); capped component walk (csb); stale cache-temp sweeps (ckv).
  - **UI/a11y pass (dm2)**: register, separators, designed disabled state, chip contrast, always-mounted ARIA panels, filmstrip accessible names, distinct export icon.
- The app is a macOS Tauri 2 + Svelte 5 desktop tool over a Rust engine that removes dust/scratches from scanned film: scan a roll -> review -> detect (neural) -> adjust sensitivity -> brush corrections -> heal (LaMa inpainting) -> approve -> export.
- Two halves of the product have very different maturity:
  - **Healing/brushing/export: real and solid.** LaMa inpainting, bit-exact outside the mask, grain re-synthesis. Tonality-agnostic (works on negatives and positives alike).
  - **Detection: a development FIXTURE, not a trained model.** In debug builds it fires on noise; release builds currently load no detector at all (bead 4wj). Everything about detection quality is blocked on the owner (Albert) capturing training data — see `training/DATA.md`.

## Suggested first task

No P1 open. Everything doable without Albert's input or the trained model has been swept. Albert's decisions from 2026-07-12 are recorded on the beads: 4wj ships download-on-first-run with auto colour/bw variant selection (blocked on the trained model); vem shipped (Reset/Delete roll data menu items); fuj shipped (wipe compare); obr closed (superseded by CoreML + 1jc cancellation). What remains:

- **Blocked on the trained model**: 4wj (release detector: reuse the model-download infra for a second model, auto-variant from the scan), 3uz (CoreML-vs-CPU benchmark), jb2 (IR channel).
- **Deferred pending a decision to invest**: rcb — its PURE state-machine decisions are now extracted and tested (lib/jobstate.ts, lib/heal.ts, healingEngineFor in status.ts); the remaining async-sequencing items (activation guard, loader clearing, mode-switch, ring gating) need a component harness (happy-dom + @testing-library/svelte + mocked @tauri-apps/api). Deferred 2026-07-12 — see the bead note.
- **Needs the running app / hardware**: jn3 (profile the tile protocol under pan/zoom), fag (Windows port).

## How work is done here (match this)

- **Trunk-based**: commit directly to `main`, atomic commits, tests green each commit. Plain-English, why-focused commit messages. **No `Co-Authored-By`, no emoji** anywhere (code, commits, docs, UI).
- **Gates, every commit** (all must pass):
  - Rust (repo root): `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --all --check`
  - Frontend (from `app/`): `npm run test` (vitest) and `npm run check` (svelte-check: 0 errors; baseline is exactly 4 warnings — Viewer.svelte x3, Filmstrip.svelte x1 — never grow it)
  - Training (from `training/`): `uv run pytest`
- **beads (`bd`) is the issue tracker** — not TODO files, not TodoWrite. `bd ready` / `bd show <id>` / `bd close <id>`. Note: `bd update -d` REPLACES the whole description; append with `--notes` or resend old text + addendum.
- **Runtimes via mise** (`mise.toml` per subtree): Node 22 + Rust for `app/`, Rust for `engine/`, Python 3.12 + uv for `training/`. Don't suggest nvm/pyenv/etc.
- **Development rhythm for non-trivial work** (this repo was built with the superpowers subagent-driven-development skill): write a plan under `docs/superpowers/plans/YYYY-MM-DD-*.md` with complete code and TDD steps, then implement task-by-task with a reviewer pass after each. Even solo, the norms hold: TDD (failing test first), review diffs against the plan, verify fixer claims rather than trusting them. Session artifacts live in `.superpowers/sdd/` (gitignored) — plans, briefs, per-task reports.

## Architecture orientation

- `app/src/` — Svelte 5 (runes) frontend. `App.svelte` (~1750 lines) holds the whole workflow state machine: roll/frame state, the background job queue mirror (`jobStates`), all the Tauri event listeners, activation sequencing. `lib/Viewer.svelte` is the WebGL viewer (tiles, rings, brush). `lib/status.ts`, `lib/queue.ts`, `lib/toasts.ts`, `lib/drop.ts`, `lib/detections.ts` are the pure, tested helpers.
- `app/src-tauri/src/` — Rust backend. `lib.rs` (commands + the single-worker job queue: detect/heal/export/prefetch job kinds). `images.rs` (the in-memory registry: native image + display pyramid + u8 probs + healed parts, LRU-evicted against a 6GB pixel budget). `cache.rs` (three on-disk codecs — probs, heal, pyramid — all header/stamp/zstd with corrupt-delete vs mismatch-keep discipline). `roll.rs` (per-roll sidecar `.unduster/`, generation-guarded state). `detect.rs` (DetectorState CoreML-first, InpainterState CPU-only). `jobs.rs` (the queue).
- `engine/crates/` — `fd-io` (image decode/encode), `fd-tiles` (display pyramids + tile cache + `quantize_prob`/`threshold_mask_u8`), `fd-infer` (ONNX detection, tiled, per-tile progress), `fd-heal` (tiered healing: classical fill for tiny defects, LaMa windows for larger, grain re-synthesis; `group.rs` batches clustered defects into shared windows).
- `training/` — Python/uv pipeline (harvest defects, train the U-Net detector, export ONNX, benchmark). Nothing ships except the exported `.onnx`. `DATA.md` is the capture checklist (now covers both film polarities + optional IR channel).

## Load-bearing invariants — do not weaken without understanding them

- **Generation guards.** Every roll swap bumps `RollState.generation` (inside the roll lock, since commit `ffb582a`). Job events, cache writes, and sidecar setters (`set_exported`, `set_threshold_and_components`, `set_image_id`/`_if_absent`) carry and re-check a generation so a stale operation from a swapped-out roll lands nowhere. The generation travels with the work from where it was scheduled (the `record_scan_result` pattern), not re-derived at execution.
- **Bit-exactness outside the mask** is structural: `fd_heal::write_back` writes native-depth pixels only at mask-true positions.
- **Component walks are memoized and off the main thread** (bead 89m fix): `components` and `set_frame_threshold` are async + `spawn_blocking`, probs stored as `Arc<Vec<u8>>` so the CCL runs lock-free, result memoized per quantized threshold in the registry, cleared by every probs writer. The mask loop lives in `fd-tiles` for opt-3 in dev. Don't reintroduce a sync walk under the `Images` lock (the one remaining is `run_detect`'s post-detect prime — bead u98).
- **CoreML is detector-only.** The LaMa inpainter is CPU on purpose — measured 2026-07-10: CoreML fragments it into 621 partitions, runs ~3x slower, diverges up to ~44/255. The comment at the `Ep::Cpu` site records this; do not "upgrade" it without re-measuring.
- **Source stamps travel with the pixels they describe** (02y fix): every decode captures its `SourceStamp` immediately BEFORE reading pixels, stores it on the registry entry, and every cache write from those pixels reuses that stamp. Never stat at cache-write time for registry-backed data — an overwritten source would pair a fresh stamp with stale pixels and persist stale results across relaunch. A failed stat means "skip the cache interaction", never "fail the operation".
- **Sidecar writes happen outside the roll mutex** (vd5 fix): setters serialize + take a seq under the lock (`snapshot_sidecar`), then write after release (`commit_sidecar`, serial via its own lock, superseded snapshots skipped). Don't reintroduce `roll.save()` inside a roll-lock critical section.
- **Cancellation is cooperative and flag-decided** (1jc): engine progress callbacks return `ControlFlow`; the worker records the running job under the same lock cancel requests take, clears the flag before each job, and decides job-cancelled vs job-error by the flag, not by matching error strings. File writes are never interrupted mid-write.
- **The dev fixture inpainter is a mean-fill stub, not LaMa.** In debug builds, if `lama.onnx` is absent or fails to load, `lib.rs` setup autoloads `engine/fixtures/tiny-inpaint.onnx` via `InpainterState::load_fixture` — a mean-fill stub that produces flat grey fills on large brush strokes. As of the 58s fix this is now reported honestly: `inpainter_status` returns `"fixture"`, the UI shows the download card + a "healing: development stub" hint + the Heal button title warns, and a real-LaMa load failure toasts (release too). But the underlying behaviour is unchanged — if you heal in a dev build without real LaMa, results are stub-quality by design. Download real LaMa to heal for real. Caches made by the stub correctly miss when LaMa arrives (provenance binds the model hash — do not special-case this).

## Landed but pending Albert's manual gate (likely fine, just unconfirmed)

- **1cy** u8 probs retention (~650MB/frame saved) and **2vf** detect per-tile progress — both implemented, reviewed clean, awaiting a gate before formal close.
- Heal window batching (gmn), the pyramid disk cache (3e5), nav-lag fix (89m), active ring + delete (04k), healed indicator (696) — all landed and reviewed; the last app session confirmed the earlier UI passes but the newest batch (healed indicator especially — green pips on filmstrip thumbs for previously-healed frames) wants a look.

## Open backlog, prioritized

- **P1**: none open.
- **P2**: 4wj (release detector — decisions made, blocked on the model), 3uz (CoreML-vs-CPU benchmark — blocked on the model).
- **P3**: rcb (deferred — pure decisions tested, harness pending; see above), jn3 (tile-protocol profiling), fag (Windows port), jb2 (IR channel).
- Closed in the 07-11/12 sweep: 1jc, np6, 36w, jhk, ckv, 80i, cm2, u98, sol, 02y, dm2, csb, vd5, vem, fuj, obr, 1cy, 2vf.

## The critical path

The single highest-value thing is not code: Albert capturing the training data (blank-film scans, clean set, labelled benchmark roll per `training/DATA.md`) so the real detector can be trained. The plumbing is ready — caches are model-hash-keyed and self-invalidate, the benchmark harness scores against a labelled roll and Retouch4me. When the model lands, 4wj (getting it into a release build + variant selection + benchmark-derived thresholds) and 3uz (CoreML parity) become live.
