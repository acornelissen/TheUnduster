# TheUnduster — Design

Date: 2026-07-07
Status: Approved pending final review

## What

A single-purpose macOS desktop app that removes dust and scratches from scanned
film photos, black and white or colour. Software only. Image in, healed image
out, everything else untouched.

## Who

Home-scanning hobbyists first: shoots film, scans with an Epson, Plustek, or
DSLR rig, edits a roll of 24-36 frames at a time. Wants speed with control.
Labs, archives, and prosumer plugin workflows are later audiences, not v1.

## Why we win

Measured against Retouch4me Dust and SRDx on a fixed benchmark roll:

1. **Detection you can trust.** Fewer false positives (stars, birds, grain)
   and fewer misses than the competition, and the mask is always shown before
   healing. Competitors hide theirs.
2. **Roll-at-once workflow.** Review-first batch: detect all frames in the
   background, review and tweak masks keyboard-first, heal all. A 36-frame
   roll reviewed in under 5 minutes. Nobody does this well.
3. **Healing quality.** Grain-matched, structure-aware inpainting that
   survives inspection at 200%.
4. **B&W excellence.** Silver halide film defeats infrared hardware cleaning
   everywhere. A B&W-tuned model makes us the only good answer for B&W dust.

## Scope

**In:** TIFF/JPEG/PNG input, 8 and 16 bit, up to ~100MP scans. ICC profile and
EXIF passthrough. Automatic detection with editable masks. Tiered healing.
Manual heal brush. Roll-based review UI. Non-destructive sidecar projects.
Export to new files.

**Out (explicit non-goals for v1):** negative inversion, colour correction,
grain tools, film border tools, asset management, Lightroom plugin, camera RAW
input, Windows build (planned v1.x, stack chosen to keep the port cheap),
Intel Mac support, cloud anything. Originals are never modified.

**Hardware floor:** Apple Silicon (M1 base or better), macOS.

## Architecture

Three sub-projects, each with its own implementation plan:

### 1. Training pipeline (Python/PyTorch, offline, never ships)

- Harvests real defect overlays (dust, scratches, hairs) from scans of blank
  and leader film.
- Composites overlays onto clean images with film response curves and grain
  simulation to build synthetic training data.
- Bootstraps from published film-restoration weights as baseline and
  benchmark; our trained models must beat them before shipping.
- Two detector variants: colour and B&W-tuned (heavier grain augmentation so
  it learns grain is not dust). Auto-selected by image stats, user override.
- Exports ONNX. Includes the benchmark harness: a hand-labelled test roll
  (colour and B&W) scored for precision/recall per defect type, with
  Retouch4me output on the same roll as the bar. Runs in CI; regressions
  don't ship.

### 2. Core engine (Rust workspace, no UI dependency)

- `fd-io` — TIFF/JPEG/PNG decode/encode, 8/16-bit, ICC and EXIF passthrough
  untouched. Originals opened read-only.
- `fd-tiles` — image pyramid builder plus bounded LRU tile cache, feeding
  both the viewer and inference.
- `fd-infer` — ONNX Runtime session management, CoreML execution provider,
  tiled detection on 512px tiles at native resolution with 64px overlap and
  seam blending. Output is a per-pixel defect probability map, not a binary
  mask.
- `fd-heal` — tiered healing:
  - Tiny defects (about 5px and under): classical grain-aware patch fill.
  - Larger: LaMa-class inpainting on a crop around the defect (ONNX).
  - Grain matching: sample noise statistics from the ring around each fill
    and re-synthesize residual grain over it.
  - The manual heal brush enters this same path: a brush stroke is a
    hand-drawn mask region. Mask in, heal out, no separate machinery.
- **Bit-exactness guarantee:** healing writes only inside mask regions.
  Pixels outside are byte-identical to the source and verified by checksum
  at export.

### 3. Desktop app (Tauri 2 + Svelte)

Thin shell over the engine. Every heavy operation is an async engine call
emitting progress events. No pixel work in JS.

**Data flow:** import folder → engine builds pyramids and queues per-frame
detection in parallel → filmstrip populates with defect counts as results
land → user reviews and edits masks → heal (per frame or all) → export.

**State:** one sidecar project file per roll holding masks, brush edits, undo
history, and per-frame status. Versioned schema, incremental saves, last-good
backup. A corrupt sidecar costs a review session, never a scan.

## UI and the smoothness contract

Perceived smoothness is a requirement with budgets, enforced in CI:

| Interaction | Budget |
|---|---|
| Pan/zoom at any resolution | 120fps on ProMotion, never below 60 |
| Sensitivity slider to mask update | under 16ms (one frame) |
| Brush stroke visual feedback | under 16ms, healing async behind it |
| Frame-to-frame in filmstrip | under 100ms to sharp image |
| Before/after toggle | instant, both buffers resident |

**Viewer.** The webview never receives pixels over IPC. Tiles stream through
a Tauri custom protocol (zero-copy, HTTP-like so browser cache semantics
apply) into a WebGL2/WebGPU-composited canvas. Image tiles and the defect
probability map are separate GPU textures; the mask tint is applied in the
shader. The sensitivity slider is a shader uniform thresholding the resident
probability texture — one frame, no engine round-trip. Pyramid levels
prefetch around the viewport. Adjacent filmstrip frames pre-decode so
next-frame is a texture swap. While sharp tiles stream in, blurred parent
level content renders underneath — no white flashes.

**Review flow, keyboard-first:**
left/right frames · space before/after · A approve and advance ·
S sensitivity drag · B brush · E erase-mask · [ ] brush size ·
Z zoom to next detection · cmd-Z undo everywhere.

Detections are also a navigable list; Z cycles through them at 100%, so
review means looking where the model looked. Filmstrip shows per-frame
status (queued/detected/approved/healed) and defect count. An approved roll
converges to one Heal All button.

**Perceived-performance rules:** UI thread never blocks; every engine op is
async with progress; mask edits apply optimistically and the engine confirms.

**Accessibility (WCAG 2.2 AA floor):** full keyboard operability including
brush placement by arrow-key nudge, visible focus states, contrast-checked
dark UI with neutral grey around images, respects reduced-motion.

## Error handling

Philosophy: a bad frame never takes down a roll; no failure path can touch
original files.

- **Corrupt/unsupported input:** frame marked failed in the filmstrip with a
  specific reason; the rest of the roll proceeds. Unsupported TIFF variants
  are detected and refused clearly, never half-decoded.
- **Model load or inference failure:** frame falls back to classical
  detection with a visible reduced-quality badge. Never silent.
- **Memory pressure:** bounded tile cache evicts under pressure; inference
  concurrency backs off. A 100MP roll on an 8GB M1 degrades to slower,
  never to crashed.
- **Export:** write to temp, verify (including the untouched-pixel
  checksum), atomic rename. Disk-full leaves no partial files.

## Testing

TDD throughout.

- **Engine:** unit tests per crate. Golden-image round-trip tests over a
  curated corpus of real scanner output (Epson Scan, VueScan, SilverFast,
  Noritsu; 8/16-bit; assorted ICC profiles). Property test for the
  bit-exactness guarantee: heal with mask M, assert all pixels outside M
  identical.
- **Model:** the benchmark harness is CI. Precision/recall per defect type
  on the labelled test roll; regressions block release.
- **Performance:** automated pan/zoom frame-time capture on the largest
  corpus image; budget violations fail CI.
- **App:** Tauri command integration tests; end-to-end keyboard-flow test
  (import → review → heal → export) on a mini-roll.

## Build order

1. Training pipeline far enough to produce a first ONNX detector and the
   benchmark harness (the moat, and the biggest risk — start it first).
2. Core engine: io and tiles, then infer, then heal.
3. App shell and viewer, then review workflow, then export.

Sub-project plans will sequence this in detail.
