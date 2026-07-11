# TheUnduster — Handover

Snapshot for a fresh Claude Code session. Written 2026-07-11.

## Where things stand

- Branch `main`, HEAD `0d61a34`. Working tree clean, everything pushed (`.beads/interactions.jsonl` may show as modified — it is a passive bd export, leave it).
- The app is a macOS Tauri 2 + Svelte 5 desktop tool over a Rust engine that removes dust/scratches from scanned film: scan a roll -> review -> detect (neural) -> adjust sensitivity -> brush corrections -> heal (LaMa inpainting) -> approve -> export.
- Two halves of the product have very different maturity:
  - **Healing/brushing/export: real and solid.** LaMa inpainting, bit-exact outside the mask, grain re-synthesis. Tonality-agnostic (works on negatives and positives alike).
  - **Detection: a development FIXTURE, not a trained model.** In debug builds it fires on noise; release builds currently load no detector at all (bead 4wj). Everything about detection quality is blocked on the owner (Albert) capturing training data — see `training/DATA.md`.

## URGENT: pick this up first

### TheUnduster-58s (P1) — export heals painted areas without LaMa

**Diagnosed, fix NOT yet applied.** Root cause (verified with a scratch repro, numbers in `bd show 58s`):

In debug builds, `app/src-tauri/src/lib.rs` (~2509-2513) silently autoloads `engine/fixtures/tiny-inpaint.onnx` into `InpainterState` when `lama.onnx` is missing OR fails to load. That fixture is a **mean-fill stub** (`training/scripts/make_engine_fixtures.py`: `image*(1-mask) + mean*mask`) — it paints every masked pixel one flat colour, producing flat grey disks on large brush strokes while small detections mean-fill plausibly. The load failure is `eprintln`-only (~2500-2503), and `models.rs` `inpainter_status` (~127-134) reports the fixture as `"loaded"`, so nothing in the UI reveals the swap. When the model hash changes, all heal-cache entries miss (provenance binds the inpainter hash), so exports re-heal through the stub.

Ruled out with evidence: the mask composition, the heal-window-batching change (byte-identical checksums pre/post), and the u8-probs change. Not a real healing bug — the real LaMa was just not loaded.

**Immediate operator workaround** (already told Albert): run the app with real LaMa loaded and re-export; caches miss and re-heal correctly.

**Fix direction** (three parts, was about to be dispatched):
1. `inpainter_status` gains a distinct `"fixture"` state (alongside loaded/available/missing/downloading). Debug autoload keeps working but reports honestly; frontend `modelStatus` union widens; the Model toolbar group + download button show for `"fixture"`, plus an unmissable "development stub" hint when healing. Detect fixture-ness at the honest seam (record it at autoload time, not a hash compare at status time).
2. Surface the lama-load-failed path to the frontend (toast via the `pushError` funnel or an error-carrying status variant) instead of `eprintln`. Mind the startup race with webview readiness — status polling may be more reliable than an emit at setup.
3. Do NOT special-case provenance — the inpainter-hash miss that invalidates stub-made caches is correct and wanted.

Two P3 follow-ups already filed from the same diagnosis: **80i** (export summary flags which inpainter healed each frame) and **cm2** (`classical_fill` leaves cores of defects larger than ~22px unfilled — a real release-build gap when no model is present).

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

## Landed but pending Albert's manual gate (likely fine, just unconfirmed)

- **1cy** u8 probs retention (~650MB/frame saved) and **2vf** detect per-tile progress — both implemented, reviewed clean, awaiting a gate before formal close.
- Heal window batching (gmn), the pyramid disk cache (3e5), nav-lag fix (89m), active ring + delete (04k), healed indicator (696) — all landed and reviewed; the last app session confirmed the earlier UI passes but the newest batch (healed indicator especially — green pips on filmstrip thumbs for previously-healed frames) wants a look.

## Open backlog, prioritized

- **P1**: 58s (above).
- **P2**: 1jc (cancel queued/running jobs + exports — the biggest UX gap: a wrong "Export approved" commits minutes with no abort), 4wj (release detector story — needs Albert's decisions: bundle vs download, colour/bw variant selection), jhk (re-export only what changed), 36w (model-download timeout/cancel), 3uz (benchmark CoreML vs CPU detection on the real model).
- **P3**: 80i, cm2, u98 (nav-lag residual), ckv (sweep orphaned cache temps), dm2 (UI register/contrast nits), 1cy/2vf close-out, jb2 (IR channel — also near-free training labels), rcb (frontend component tests — real debt: App.svelte carries the whole state machine untested), plus perf/Windows/compare-mode items (jn3, vd5, obr, fag, fuj, csb, sol, 02y).

## The critical path

The single highest-value thing is not code: Albert capturing the training data (blank-film scans, clean set, labelled benchmark roll per `training/DATA.md`) so the real detector can be trained. The plumbing is ready — caches are model-hash-keyed and self-invalidate, the benchmark harness scores against a labelled roll and Retouch4me. When the model lands, 4wj (getting it into a release build + variant selection + benchmark-derived thresholds) and 3uz (CoreML parity) become live.
