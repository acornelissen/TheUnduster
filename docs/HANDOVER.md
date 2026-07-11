# TheUnduster — Handover

Snapshot for a fresh Claude Code session. Updated 2026-07-11 (58s fixed).

## Where things stand

- Branch `main`, HEAD is the tip of `main` after the 58s fix (`git log --oneline -5` to see it: fixture-status commits fe46be6 / a82ab92 / 02c147c). Working tree clean, everything pushed (`.beads/interactions.jsonl` may show as modified — it is a passive bd export, leave it).
- The app is a macOS Tauri 2 + Svelte 5 desktop tool over a Rust engine that removes dust/scratches from scanned film: scan a roll -> review -> detect (neural) -> adjust sensitivity -> brush corrections -> heal (LaMa inpainting) -> approve -> export.
- Two halves of the product have very different maturity:
  - **Healing/brushing/export: real and solid.** LaMa inpainting, bit-exact outside the mask, grain re-synthesis. Tonality-agnostic (works on negatives and positives alike).
  - **Detection: a development FIXTURE, not a trained model.** In debug builds it fires on noise; release builds currently load no detector at all (bead 4wj). Everything about detection quality is blocked on the owner (Albert) capturing training data — see `training/DATA.md`.

## Suggested first task

No P1 open. The highest-value next piece of work is **TheUnduster-1jc (P2) — cancel queued/running jobs and exports**: the biggest remaining UX gap, since a wrong "Export approved" currently commits the operator to minutes of healing with no abort. `bd show 1jc` has the shape (queue-row removal for queued jobs; cooperative abort for the running job via the existing per-defect/per-tile progress callbacks; a job-cancelled event the frontend clears state from). Alternatively **4wj (release detector story)** if Albert is ready to decide bundle-vs-download and colour/bw variant selection — that one needs his input before planning. See the full backlog below.

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
- **The dev fixture inpainter is a mean-fill stub, not LaMa.** In debug builds, if `lama.onnx` is absent or fails to load, `lib.rs` setup autoloads `engine/fixtures/tiny-inpaint.onnx` via `InpainterState::load_fixture` — a mean-fill stub that produces flat grey fills on large brush strokes. As of the 58s fix this is now reported honestly: `inpainter_status` returns `"fixture"`, the UI shows the download card + a "healing: development stub" hint + the Heal button title warns, and a real-LaMa load failure toasts (release too). But the underlying behaviour is unchanged — if you heal in a dev build without real LaMa, results are stub-quality by design. Download real LaMa to heal for real. Caches made by the stub correctly miss when LaMa arrives (provenance binds the model hash — do not special-case this).

## Landed but pending Albert's manual gate (likely fine, just unconfirmed)

- **1cy** u8 probs retention (~650MB/frame saved) and **2vf** detect per-tile progress — both implemented, reviewed clean, awaiting a gate before formal close.
- Heal window batching (gmn), the pyramid disk cache (3e5), nav-lag fix (89m), active ring + delete (04k), healed indicator (696) — all landed and reviewed; the last app session confirmed the earlier UI passes but the newest batch (healed indicator especially — green pips on filmstrip thumbs for previously-healed frames) wants a look.

## Open backlog, prioritized

- **P1**: none open.
- **P2**: 1jc (cancel queued/running jobs + exports — the biggest UX gap: a wrong "Export approved" commits minutes with no abort), 4wj (release detector story — needs Albert's decisions: bundle vs download, colour/bw variant selection), jhk (re-export only what changed), 36w (model-download timeout/cancel), 3uz (benchmark CoreML vs CPU detection on the real model).
- **P3**: 80i, cm2, u98 (nav-lag residual), ckv (sweep orphaned cache temps), dm2 (UI register/contrast nits), 1cy/2vf close-out, jb2 (IR channel — also near-free training labels), rcb (frontend component tests — real debt: App.svelte carries the whole state machine untested), plus perf/Windows/compare-mode items (jn3, vd5, obr, fag, fuj, csb, sol, 02y).

## The critical path

The single highest-value thing is not code: Albert capturing the training data (blank-film scans, clean set, labelled benchmark roll per `training/DATA.md`) so the real detector can be trained. The plumbing is ready — caches are model-hash-keyed and self-invalidate, the benchmark harness scores against a labelled roll and Retouch4me. When the model lands, 4wj (getting it into a release build + variant selection + benchmark-derived thresholds) and 3uz (CoreML parity) become live.
