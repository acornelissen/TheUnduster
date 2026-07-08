# App 3a: Shell and Tiled Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Tauri 2 desktop app that opens scans through the engine crates and pans/zooms 100MP images smoothly via a custom tile protocol and a WebGL2 tiled viewer.

**Architecture:** `app/src-tauri` is a thin Rust shell over the engine workspace: an `Images` registry (decode via fd-io, pyramids and cache via fd-tiles) exposed through one `open_image` command and a `tiles://` custom protocol that streams raw RGBA tile bytes — pixels never cross the IPC JSON boundary. The webview side splits pure viewport math (`viewport.ts`, unit-tested) from a thin WebGL2 renderer; parent-level tiles render beneath sharp tiles while they stream in, so zooming never white-flashes.

**Tech Stack:** Tauri 2, Rust (engine workspace crates via path), Svelte 5 + TypeScript + Vite, vitest, WebGL2.

## Global Constraints

- App lives under `app/` (frontend) and `app/src-tauri/` (shell crate, part of the `engine/` cargo workspace via path dependencies but its own package). Node pinned via `app/mise.toml`.
- Trunk-based development: commit directly to `main`, one atomic commit per task, tests green before every commit. No emoji; no Co-Authored-By.
- Pixels cross to the webview ONLY through the `tiles://` protocol as raw RGBA bytes; Tauri commands carry metadata JSON only.
- Tile size is `fd_tiles::TILE_SIZE` (512). Tile cache budget 512 MB, one cache for the whole app.
- Smoothness budgets from the spec bind this plan: pan/zoom 120fps target / 60 floor, frame-to-sharp under 100ms. Automated frame-time capture lands in plan 3c; this plan ends with a documented manual verification against a generated 100MP image.
- Accessibility floor: the viewer canvas is keyboard-operable (arrows pan, +/- zoom, 0 fit, 1 100%), focusable with a visible focus ring, and all controls have accessible names.
- Originals read-only (engine guarantees this; the app must never pass a source path to any write API).
- Rust: fmt + clippy `-D warnings` clean. TS: `svelte-check`/`tsc` clean, vitest green.

---

### Task 1: Scaffold the Tauri app

**Files:**
- Create: `app/mise.toml`, `app/package.json`, `app/vite.config.ts`, `app/tsconfig.json`, `app/index.html`, `app/src/main.ts`, `app/src/App.svelte`, `app/svelte.config.js`
- Create: `app/src-tauri/Cargo.toml`, `app/src-tauri/tauri.conf.json`, `app/src-tauri/build.rs`, `app/src-tauri/src/main.rs`, `app/src-tauri/src/lib.rs`
- Modify: `engine/Cargo.toml` (add `"../app/src-tauri"` to workspace members)
- Create: `app/.gitignore`

**Interfaces:**
- Produces: a launchable empty window titled "TheUnduster"; `npm run tauri dev` works; `cargo test -p unduster-app` runs (no tests yet); the crate compiles inside the engine workspace so later tasks can use fd-io/fd-tiles by path.

- [ ] **Step 1: Write the frontend scaffold**

`app/mise.toml`:

```toml
[tools]
node = "22"
```

`app/package.json`:

```json
{
  "name": "unduster-app",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "check": "svelte-check --tsconfig ./tsconfig.json",
    "test": "vitest run",
    "tauri": "tauri"
  },
  "dependencies": {
    "@tauri-apps/api": "^2.0.0"
  },
  "devDependencies": {
    "@sveltejs/vite-plugin-svelte": "^4.0.0",
    "@tauri-apps/cli": "^2.0.0",
    "svelte": "^5.0.0",
    "svelte-check": "^4.0.0",
    "typescript": "^5.5.0",
    "vite": "^5.4.0",
    "vitest": "^2.0.0"
  }
}
```

`app/vite.config.ts`:

```ts
import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  server: { port: 1420, strictPort: true },
});
```

`app/svelte.config.js`:

```js
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";
export default { preprocess: vitePreprocess() };
```

`app/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "types": ["svelte", "vite/client"],
    "skipLibCheck": true,
    "noEmit": true
  },
  "include": ["src/**/*.ts", "src/**/*.svelte"]
}
```

`app/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>TheUnduster</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

`app/src/main.ts`:

```ts
import { mount } from "svelte";
import App from "./App.svelte";

const app = mount(App, { target: document.getElementById("app")! });
export default app;
```

`app/src/App.svelte`:

```svelte
<main>
  <h1>TheUnduster</h1>
</main>

<style>
  :global(body) {
    margin: 0;
    background: #262626;
    color: #e8e8e8;
    font-family: system-ui, sans-serif;
  }
  main {
    padding: 1rem;
  }
</style>
```

`app/.gitignore`:

```
node_modules/
dist/
```

- [ ] **Step 2: Write the shell crate**

`app/src-tauri/Cargo.toml`:

```toml
[package]
name = "unduster-app"
version = "0.1.0"
edition = "2021"

[lib]
name = "unduster_app"
crate-type = ["staticlib", "cdylib", "rlib"]

[build-dependencies]
tauri-build = { version = "2", features = [] }

[dependencies]
tauri = { version = "2", features = [] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
fd-io = { path = "../../engine/crates/fd-io" }
fd-tiles = { path = "../../engine/crates/fd-tiles" }
```

`app/src-tauri/build.rs`:

```rust
fn main() {
    tauri_build::build()
}
```

`app/src-tauri/src/lib.rs`:

```rust
//! TheUnduster desktop shell: thin Tauri layer over the engine crates.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

`app/src-tauri/src/main.rs`:

```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    unduster_app::run()
}
```

`app/src-tauri/tauri.conf.json`:

```json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "TheUnduster",
  "version": "0.1.0",
  "identifier": "com.theunduster.app",
  "build": {
    "beforeDevCommand": "npm run dev",
    "devUrl": "http://localhost:1420",
    "beforeBuildCommand": "npm run build",
    "frontendDist": "../dist"
  },
  "app": {
    "windows": [
      {
        "title": "TheUnduster",
        "width": 1400,
        "height": 900
      }
    ],
    "security": {
      "csp": "default-src 'self'; img-src 'self' tiles: data:; connect-src 'self' tiles: ipc: http://ipc.localhost; style-src 'self' 'unsafe-inline'"
    }
  }
}
```

Add `"../app/src-tauri"` to `engine/Cargo.toml` workspace members. The app crate lives in the engine workspace so one `cargo test` covers everything and path deps resolve consistently.

- [ ] **Step 3: Verify it builds and launches**

Run: `cd /Users/albert/Development/TheUnduster/app && mise install && npm install && npm run check`
Expected: svelte-check clean.

Run: `cd /Users/albert/Development/TheUnduster/engine && cargo build -p unduster-app`
Expected: compiles. (Tauri's first build downloads and compiles many crates — minutes, once.)

Run: `cd /Users/albert/Development/TheUnduster/app && npm run tauri dev` — confirm a window opens showing "TheUnduster", then close it. If running headless/CI, skip the launch and note it; the build is the gate.

- [ ] **Step 4: Commit**

```bash
git add app engine/Cargo.toml
git commit -m "Scaffold the Tauri app shell"
```

---

### Task 2: Images registry and open_image command

**Files:**
- Create: `app/src-tauri/src/images.rs`
- Modify: `app/src-tauri/src/lib.rs`
- Test: inline `#[cfg(test)]` in `images.rs`

**Interfaces:**
- Consumes: `fd_io::decode`, `fd_tiles::{Pyramid, TileCache, TileKey, Tile, TILE_SIZE}`.
- Produces: `Images::default()`; `Images::open(&mut self, path: &Path) -> Result<ImageInfo, String>`; `Images::tile(&mut self, id: u64, level: u8, tx: u32, ty: u32) -> Option<Arc<Tile>>` (cache-backed); serde-serializable `ImageInfo { id: u64, width: u32, height: u32, levels: Vec<LevelInfo> }`, `LevelInfo { width: u32, height: u32 }`. Tauri command `open_image(path: String) -> Result<ImageInfo, String>` registered on managed state `Mutex<Images>`. Task 3's protocol handler calls `Images::tile`.

- [ ] **Step 1: Write the failing test** (inline in the new module; the module compiles against stubs first is fine — write the whole file, run tests, they must fail only if logic is wrong; the meaningful RED here is the missing module compile error, verify with `cargo test -p unduster-app` before creating the file if strictness is wanted)

- [ ] **Step 2: Implement**

`app/src-tauri/src/images.rs`:

```rust
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use fd_tiles::{Pyramid, Tile, TileCache, TileKey};
use serde::Serialize;

const CACHE_BUDGET_BYTES: usize = 512 * 1024 * 1024;

#[derive(Serialize, Clone, Copy)]
pub struct LevelInfo {
    pub width: u32,
    pub height: u32,
}

#[derive(Serialize, Clone)]
pub struct ImageInfo {
    pub id: u64,
    pub width: u32,
    pub height: u32,
    pub levels: Vec<LevelInfo>,
}

struct Entry {
    info: ImageInfo,
    pyramid: Pyramid,
}

pub struct Images {
    next_id: u64,
    entries: HashMap<u64, Entry>,
    cache: TileCache,
}

impl Default for Images {
    fn default() -> Self {
        Images {
            next_id: 1,
            entries: HashMap::new(),
            cache: TileCache::new(CACHE_BUDGET_BYTES),
        }
    }
}

impl Images {
    pub fn open(&mut self, path: &Path) -> Result<ImageInfo, String> {
        let img = fd_io::decode(path).map_err(|e| e.to_string())?;
        let pyramid = Pyramid::build(&img);
        let id = self.next_id;
        self.next_id += 1;
        let levels = pyramid
            .levels
            .iter()
            .map(|l| LevelInfo { width: l.width, height: l.height })
            .collect();
        let info = ImageInfo { id, width: img.width, height: img.height, levels };
        self.entries.insert(id, Entry { info: info.clone(), pyramid });
        Ok(info)
    }

    pub fn tile(&mut self, id: u64, level: u8, tx: u32, ty: u32) -> Option<Arc<Tile>> {
        let key = TileKey { image_id: id, level, tx, ty };
        let entry = self.entries.get(&id)?;
        let pyramid = &entry.pyramid;
        self.cache.get_or_insert(key, || pyramid.tile(level, tx, ty))
    }

    pub fn close(&mut self, id: u64) {
        self.entries.remove(&id);
        self.cache.evict_image(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fd_io::{encode, ImageBuf, PixelData};

    fn temp_png(dir: &tempfile::TempDir, w: u32, h: u32) -> std::path::PathBuf {
        let path = dir.path().join("t.png");
        let n = (w * h) as usize;
        let img = ImageBuf {
            width: w,
            height: h,
            channels: 1,
            data: PixelData::U8((0..n).map(|i| (i % 251) as u8).collect()),
            icc: None,
            exif: None,
        };
        encode(&path, &img).unwrap();
        path
    }

    #[test]
    fn open_reports_dims_and_levels() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 1100, 600);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        assert_eq!((info.width, info.height), (1100, 600));
        assert_eq!(info.levels.len(), 2); // 1100x600 -> 550x300
        assert_eq!(info.id, 1);
    }

    #[test]
    fn tile_is_cached_and_edge_sized() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 1100, 600);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        let t = images.tile(info.id, 0, 2, 1).unwrap();
        assert_eq!((t.width, t.height), (1100 - 1024, 600 - 512));
        assert!(images.tile(info.id, 0, 9, 0).is_none());
        assert!(images.tile(999, 0, 0, 0).is_none());
    }

    #[test]
    fn open_missing_file_is_an_error_naming_it() {
        let mut images = Images::default();
        let err = images.open(Path::new("/nonexistent/x.png")).unwrap_err();
        assert!(err.contains("x.png"));
    }

    #[test]
    fn close_evicts() {
        let dir = tempfile::tempdir().unwrap();
        let path = temp_png(&dir, 600, 400);
        let mut images = Images::default();
        let info = images.open(&path).unwrap();
        assert!(images.tile(info.id, 0, 0, 0).is_some());
        images.close(info.id);
        assert!(images.tile(info.id, 0, 0, 0).is_none());
    }
}
```

Add `tempfile = "3"` to `[dev-dependencies]` in `app/src-tauri/Cargo.toml`.

Note: `Images::tile` borrows `entries` and `cache` simultaneously — the closure captures `pyramid` (a `&` into `entries`) while `cache` is `&mut`. This compiles because the borrows are disjoint fields accessed through locals; if the borrow checker objects in context, restructure to fetch `entry` first and call a free function taking `&mut self.cache` and `&entry.pyramid`.

Wire the command in `lib.rs`:

```rust
//! TheUnduster desktop shell: thin Tauri layer over the engine crates.

mod images;

use std::sync::Mutex;

use images::{ImageInfo, Images};
use tauri::State;

#[tauri::command]
fn open_image(state: State<'_, Mutex<Images>>, path: String) -> Result<ImageInfo, String> {
    let mut images = state.lock().map_err(|e| e.to_string())?;
    images.open(std::path::Path::new(&path))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(Mutex::new(Images::default()))
        .invoke_handler(tauri::generate_handler![open_image])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/albert/Development/TheUnduster/engine && cargo test -p unduster-app`
Expected: 4 tests pass.

- [ ] **Step 4: Format, lint, commit**

Run: `cargo fmt && cargo clippy -p unduster-app --all-targets -- -D warnings && cargo test -p unduster-app`

```bash
git add app engine
git commit -m "Add Images registry and open_image command"
```

---

### Task 3: tiles:// protocol serving raw RGBA

**Files:**
- Create: `app/src-tauri/src/protocol.rs`
- Modify: `app/src-tauri/src/lib.rs`
- Test: inline `#[cfg(test)]` in `protocol.rs`

**Interfaces:**
- Consumes: `Images::tile` (Task 2).
- Produces: URL contract the viewer fetches: `tiles://localhost/{id}/{level}/{tx}/{ty}` returning status 200, `Content-Type: application/octet-stream`, headers `x-tile-width`/`x-tile-height`, body = RGBA8 bytes row-major; 404 with empty body when out of range; 400 on malformed paths. `parse_tile_path(path: &str) -> Option<(u64, u8, u32, u32)>` is the tested core.

- [ ] **Step 1: Write the failing test + implementation**

`app/src-tauri/src/protocol.rs`:

```rust
use std::sync::Mutex;

use crate::images::Images;

/// Parse "/{id}/{level}/{tx}/{ty}" (leading slash optional).
pub fn parse_tile_path(path: &str) -> Option<(u64, u8, u32, u32)> {
    let mut parts = path.trim_start_matches('/').split('/');
    let id = parts.next()?.parse().ok()?;
    let level = parts.next()?.parse().ok()?;
    let tx = parts.next()?.parse().ok()?;
    let ty = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((id, level, tx, ty))
}

pub fn tile_response(
    images: &Mutex<Images>,
    path: &str,
) -> tauri::http::Response<Vec<u8>> {
    let respond = |status: u16, body: Vec<u8>, w: u32, h: u32| {
        let mut builder = tauri::http::Response::builder()
            .status(status)
            .header("Content-Type", "application/octet-stream")
            .header("Access-Control-Allow-Origin", "*");
        if status == 200 {
            builder = builder
                .header("x-tile-width", w.to_string())
                .header("x-tile-height", h.to_string());
        }
        builder.body(body).expect("static response headers are valid")
    };
    let Some((id, level, tx, ty)) = parse_tile_path(path) else {
        return respond(400, Vec::new(), 0, 0);
    };
    let Ok(mut images) = images.lock() else {
        return respond(500, Vec::new(), 0, 0);
    };
    match images.tile(id, level, tx, ty) {
        Some(tile) => respond(200, tile.rgba.clone(), tile.width, tile.height),
        None => respond(404, Vec::new(), 0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_paths() {
        assert_eq!(parse_tile_path("/3/1/7/2"), Some((3, 1, 7, 2)));
        assert_eq!(parse_tile_path("3/1/7/2"), Some((3, 1, 7, 2)));
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(parse_tile_path("/3/1/7"), None);
        assert_eq!(parse_tile_path("/3/1/7/2/9"), None);
        assert_eq!(parse_tile_path("/a/b/c/d"), None);
        assert_eq!(parse_tile_path(""), None);
        assert_eq!(parse_tile_path("/-1/0/0/0"), None);
    }

    #[test]
    fn missing_tile_is_404_and_malformed_is_400() {
        let images = Mutex::new(Images::default());
        assert_eq!(tile_response(&images, "/1/0/0/0").status(), 404);
        assert_eq!(tile_response(&images, "/nope").status(), 400);
    }
}
```

Register in `lib.rs` `run()` (before `.run(...)`):

```rust
        .register_uri_scheme_protocol("tiles", |ctx, request| {
            let images = ctx.app_handle().state::<Mutex<Images>>();
            protocol::tile_response(&images, request.uri().path())
        })
```

with `mod protocol;` added. (In Tauri 2 the handler receives a `UriSchemeContext`; if the installed minor version passes `&AppHandle` directly, adapt — the tested logic stays in `tile_response`.)

- [ ] **Step 2: Run tests, lint, commit**

Run: `cargo test -p unduster-app && cargo fmt && cargo clippy -p unduster-app --all-targets -- -D warnings`
Expected: 7 tests pass, clean.

```bash
git add app
git commit -m "Serve raw RGBA tiles over a custom protocol"
```

---

### Task 4: Viewport math (pure TS, unit-tested)

**Files:**
- Create: `app/src/lib/viewport.ts`
- Test: `app/src/lib/viewport.test.ts`

**Interfaces:**
- Consumes: nothing (pure).
- Produces: `interface Level { width: number; height: number }`; `pickLevel(levels: Level[], zoom: number): number`; `interface TileRef { level: number; tx: number; ty: number; screenX: number; screenY: number; screenW: number; screenH: number }`; `visibleTiles(levels: Level[], zoom: number, centerX: number, centerY: number, canvasW: number, canvasH: number): TileRef[]` — zoom is screen px per level-0 image px; center is in level-0 image coordinates; returned tiles cover the viewport plus a one-tile prefetch ring, coarse level first (so sharp tiles draw over parent tiles). `fitZoom(level0: Level, canvasW: number, canvasH: number): number`. Task 5 renders exactly this list.

- [ ] **Step 1: Write the failing test**

`app/src/lib/viewport.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { fitZoom, pickLevel, visibleTiles, type Level } from "./viewport";

const LEVELS: Level[] = [
  { width: 2000, height: 1200 },
  { width: 1000, height: 600 },
  { width: 500, height: 300 },
];

describe("pickLevel", () => {
  it("uses level 0 at 100% and above", () => {
    expect(pickLevel(LEVELS, 1)).toBe(0);
    expect(pickLevel(LEVELS, 2.5)).toBe(0);
  });
  it("steps down as zoom halves", () => {
    expect(pickLevel(LEVELS, 0.5)).toBe(1);
    expect(pickLevel(LEVELS, 0.25)).toBe(2);
  });
  it("clamps to the coarsest level", () => {
    expect(pickLevel(LEVELS, 0.01)).toBe(2);
  });
});

describe("fitZoom", () => {
  it("fits the long edge", () => {
    expect(fitZoom(LEVELS[0], 1000, 1000)).toBeCloseTo(0.5);
    expect(fitZoom(LEVELS[0], 4000, 300)).toBeCloseTo(0.25);
  });
});

describe("visibleTiles", () => {
  it("covers the viewport at 100% around the center", () => {
    const tiles = visibleTiles(LEVELS, 1, 1000, 600, 800, 600);
    const sharp = tiles.filter((t) => t.level === 0);
    expect(sharp.length).toBeGreaterThan(0);
    // viewport spans image px 600..1400 x 300..900 -> tiles 1..2 x 0..1
    const keys = new Set(sharp.map((t) => `${t.tx},${t.ty}`));
    for (const k of ["1,0", "2,0", "1,1", "2,1"]) {
      expect(keys.has(k)).toBe(true);
    }
  });

  it("orders coarse level before sharp level", () => {
    const tiles = visibleTiles(LEVELS, 1, 1000, 600, 800, 600);
    const levels = tiles.map((t) => t.level);
    const firstSharp = levels.indexOf(0);
    const lastCoarse = levels.lastIndexOf(1);
    expect(lastCoarse).toBeLessThan(firstSharp === -1 ? Infinity : firstSharp);
  });

  it("never emits tiles outside the grid", () => {
    const tiles = visibleTiles(LEVELS, 0.1, 250, 150, 4000, 4000);
    for (const t of tiles) {
      const l = LEVELS[t.level];
      expect(t.tx).toBeGreaterThanOrEqual(0);
      expect(t.ty).toBeGreaterThanOrEqual(0);
      expect(t.tx).toBeLessThan(Math.ceil(l.width / 512));
      expect(t.ty).toBeLessThan(Math.ceil(l.height / 512));
    }
  });

  it("screen rects scale with zoom", () => {
    const [first] = visibleTiles(LEVELS, 1, 1000, 600, 800, 600).filter(
      (t) => t.level === 0,
    );
    expect(first.screenW).toBeCloseTo(512);
    const [half] = visibleTiles(LEVELS, 0.5, 1000, 600, 800, 600).filter(
      (t) => t.level === 1,
    );
    // level-1 tile is 512 level-1 px = 1024 level-0 px, at zoom 0.5 -> 512 screen px
    expect(half.screenW).toBeCloseTo(512);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/albert/Development/TheUnduster/app && npm run test`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

`app/src/lib/viewport.ts`:

```ts
export const TILE = 512;

export interface Level {
  width: number;
  height: number;
}

export interface TileRef {
  level: number;
  tx: number;
  ty: number;
  screenX: number;
  screenY: number;
  screenW: number;
  screenH: number;
}

/** zoom = screen px per level-0 image px. Level i covers 2^i image px per own px. */
export function pickLevel(levels: Level[], zoom: number): number {
  const ideal = Math.floor(-Math.log2(Math.max(zoom, 1e-6)));
  return Math.min(Math.max(ideal, 0), levels.length - 1);
}

export function fitZoom(level0: Level, canvasW: number, canvasH: number): number {
  return Math.min(canvasW / level0.width, canvasH / level0.height);
}

function tilesForLevel(
  levels: Level[],
  level: number,
  zoom: number,
  centerX: number,
  centerY: number,
  canvasW: number,
  canvasH: number,
  ring: number,
): TileRef[] {
  const l = levels[level];
  const scale = 2 ** level; // level px -> level-0 px
  const screenPerLevelPx = zoom * scale;
  // viewport in level coordinates
  const viewW = canvasW / screenPerLevelPx;
  const viewH = canvasH / screenPerLevelPx;
  const cx = centerX / scale;
  const cy = centerY / scale;
  const x0 = cx - viewW / 2;
  const y0 = cy - viewH / 2;
  const gridX = Math.ceil(l.width / TILE);
  const gridY = Math.ceil(l.height / TILE);
  const tx0 = Math.max(Math.floor(x0 / TILE) - ring, 0);
  const ty0 = Math.max(Math.floor(y0 / TILE) - ring, 0);
  const tx1 = Math.min(Math.ceil((x0 + viewW) / TILE) + ring, gridX);
  const ty1 = Math.min(Math.ceil((y0 + viewH) / TILE) + ring, gridY);
  const out: TileRef[] = [];
  for (let ty = ty0; ty < ty1; ty++) {
    for (let tx = tx0; tx < tx1; tx++) {
      out.push({
        level,
        tx,
        ty,
        screenX: (tx * TILE - x0) * screenPerLevelPx,
        screenY: (ty * TILE - y0) * screenPerLevelPx,
        screenW: TILE * screenPerLevelPx,
        screenH: TILE * screenPerLevelPx,
      });
    }
  }
  return out;
}

/** Tiles to draw, coarse underlay first, then the sharp level, with a
 * one-tile prefetch ring on the sharp level. */
export function visibleTiles(
  levels: Level[],
  zoom: number,
  centerX: number,
  centerY: number,
  canvasW: number,
  canvasH: number,
): TileRef[] {
  const sharp = pickLevel(levels, zoom);
  const out: TileRef[] = [];
  if (sharp + 1 < levels.length) {
    out.push(
      ...tilesForLevel(levels, sharp + 1, zoom, centerX, centerY, canvasW, canvasH, 0),
    );
  }
  out.push(...tilesForLevel(levels, sharp, zoom, centerX, centerY, canvasW, canvasH, 1));
  return out;
}
```

- [ ] **Step 4: Run tests**

Run: `npm run test && npm run check`
Expected: all vitest tests pass, svelte-check clean.

- [ ] **Step 5: Commit**

```bash
git add app/src/lib
git commit -m "Add pure viewport math for the tiled viewer"
```

---

### Task 5: WebGL2 viewer component

**Files:**
- Create: `app/src/lib/renderer.ts`
- Create: `app/src/lib/Viewer.svelte`
- Modify: `app/src/App.svelte`
- Test: `app/src/lib/renderer.test.ts` (texture-cache eviction logic only; GL itself is exercised manually)

**Interfaces:**
- Consumes: `visibleTiles`/`fitZoom`/`pickLevel` (Task 4); `tiles://` protocol (Task 3); `open_image` command (Task 2).
- Produces: `<Viewer info={ImageInfo} />` — pans with pointer drag and arrow keys, zooms with wheel (to cursor) and `+`/`-`, `0` fits, `1` goes to 100%; canvas is focusable (tabindex 0, visible focus outline) with `role="img"` and an `aria-label`. `TextureStore` class in renderer.ts: `get(key) / put(key, tex, bytes) / evictover(budget)` LRU keyed by `id/level/tx/ty`.

- [ ] **Step 1: Write the failing TextureStore test**

`app/src/lib/renderer.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { TextureStore } from "./renderer";

describe("TextureStore", () => {
  it("evicts least-recently-used past the budget", () => {
    const store = new TextureStore<number>(2500);
    const dropped: number[] = [];
    store.onEvict = (t) => dropped.push(t);
    store.put("a", 1, 1000);
    store.put("b", 2, 1000);
    store.get("a");
    store.put("c", 3, 1000);
    expect(store.get("b")).toBeUndefined();
    expect(store.get("a")).toBe(1);
    expect(dropped).toEqual([2]);
  });
});
```

- [ ] **Step 2: Verify failure, then implement**

Run: `npm run test` — FAIL, module/class missing.

`app/src/lib/renderer.ts`:

```ts
const VERT = `#version 300 es
in vec2 pos;
in vec2 uv;
out vec2 vUv;
uniform vec2 viewport;
void main() {
  vec2 clip = (pos / viewport) * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  vUv = uv;
}`;

const FRAG = `#version 300 es
precision mediump float;
in vec2 vUv;
out vec4 color;
uniform sampler2D tile;
void main() {
  color = texture(tile, vUv);
}`;

/** Byte-budgeted LRU keyed by tile URL path. Generic over the texture type
 * so the eviction logic is unit-testable without a GL context. */
export class TextureStore<T> {
  private entries = new Map<string, { value: T; bytes: number }>();
  private used = 0;
  onEvict: (value: T) => void = () => {};

  constructor(private budget: number) {}

  get(key: string): T | undefined {
    const e = this.entries.get(key);
    if (!e) return undefined;
    this.entries.delete(key); // re-insert to refresh recency (Map keeps order)
    this.entries.set(key, e);
    return e.value;
  }

  put(key: string, value: T, bytes: number): void {
    this.entries.set(key, { value, bytes });
    this.used += bytes;
    for (const [k, e] of this.entries) {
      if (this.used <= this.budget || this.entries.size <= 1) break;
      if (k === key) continue;
      this.entries.delete(k);
      this.used -= e.bytes;
      this.onEvict(e.value);
    }
  }
}

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(s) ?? "shader compile failed");
  }
  return s;
}

export class TileRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private buf: WebGLBuffer;
  private textures = new TextureStore<WebGLTexture>(256 * 1024 * 1024);
  private pending = new Set<string>();
  onTileLoaded: () => void = () => {};

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2");
    if (!gl) throw new Error("WebGL2 unavailable");
    this.gl = gl;
    const p = gl.createProgram()!;
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(p) ?? "link failed");
    }
    this.program = p;
    this.buf = gl.createBuffer()!;
    this.textures.onEvict = (t) => gl.deleteTexture(t);
  }

  /** Fetch a tile via tiles:// and upload it; no-op if cached or in flight. */
  private ensure(path: string): WebGLTexture | undefined {
    const hit = this.textures.get(path);
    if (hit) return hit;
    if (!this.pending.has(path)) {
      this.pending.add(path);
      fetch(`tiles://localhost${path}`)
        .then(async (r) => {
          if (!r.ok) return;
          const w = Number(r.headers.get("x-tile-width"));
          const h = Number(r.headers.get("x-tile-height"));
          const rgba = new Uint8Array(await r.arrayBuffer());
          const gl = this.gl;
          const tex = gl.createTexture()!;
          gl.bindTexture(gl.TEXTURE_2D, tex);
          gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
          this.textures.put(path, tex, w * h * 4);
          this.onTileLoaded();
        })
        .finally(() => this.pending.delete(path));
    }
    return undefined;
  }

  /** Draw one frame. tiles come from visibleTiles(), coarse first. */
  draw(
    tiles: {
      path: string;
      screenX: number;
      screenY: number;
      screenW: number;
      screenH: number;
      tileW: number;
      tileH: number;
    }[],
    canvasW: number,
    canvasH: number,
  ): void {
    const gl = this.gl;
    gl.viewport(0, 0, canvasW, canvasH);
    gl.clearColor(0.15, 0.15, 0.15, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.program);
    gl.uniform2f(gl.getUniformLocation(this.program, "viewport"), canvasW, canvasH);
    const posLoc = gl.getAttribLocation(this.program, "pos");
    const uvLoc = gl.getAttribLocation(this.program, "uv");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.enableVertexAttribArray(posLoc);
    gl.enableVertexAttribArray(uvLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 16, 0);
    gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 16, 8);
    for (const t of tiles) {
      const tex = this.ensure(t.path);
      if (!tex) continue;
      // edge tiles are smaller than 512: scale the drawn quad by the real
      // tile fraction so partial tiles are not stretched
      const w = t.screenW * (t.tileW / 512);
      const h = t.screenH * (t.tileH / 512);
      const x0 = t.screenX;
      const y0 = t.screenY;
      const verts = new Float32Array([
        x0, y0, 0, 0,
        x0 + w, y0, 1, 0,
        x0, y0 + h, 0, 1,
        x0 + w, y0, 1, 0,
        x0 + w, y0 + h, 1, 1,
        x0, y0 + h, 0, 1,
      ]);
      gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STREAM_DRAW);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
  }
}
```

`app/src/lib/Viewer.svelte`:

```svelte
<script lang="ts">
  import { onMount } from "svelte";
  import { fitZoom, visibleTiles, TILE, type Level } from "./viewport";
  import { TileRenderer } from "./renderer";

  interface ImageInfo {
    id: number;
    width: number;
    height: number;
    levels: Level[];
  }

  let { info }: { info: ImageInfo } = $props();

  let canvas: HTMLCanvasElement;
  let renderer: TileRenderer | undefined;
  let zoom = 1;
  let centerX = info.width / 2;
  let centerY = info.height / 2;
  let dragging = false;
  let needsFrame = true;

  function requestFrame() {
    needsFrame = true;
  }

  function tilePaths() {
    return visibleTiles(info.levels, zoom, centerX, centerY, canvas.width, canvas.height).map(
      (t) => {
        const l = info.levels[t.level];
        const tileW = Math.min(l.width - t.tx * TILE, TILE);
        const tileH = Math.min(l.height - t.ty * TILE, TILE);
        return {
          path: `/${info.id}/${t.level}/${t.tx}/${t.ty}`,
          screenX: t.screenX,
          screenY: t.screenY,
          screenW: t.screenW,
          screenH: t.screenH,
          tileW,
          tileH,
        };
      },
    );
  }

  function frame() {
    if (renderer && needsFrame) {
      needsFrame = false;
      renderer.draw(tilePaths(), canvas.width, canvas.height);
    }
    requestAnimationFrame(frame);
  }

  function clampCenter() {
    centerX = Math.min(Math.max(centerX, 0), info.width);
    centerY = Math.min(Math.max(centerY, 0), info.height);
  }

  function zoomAt(factor: number, sx: number, sy: number) {
    const next = Math.min(Math.max(zoom * factor, 0.01), 8);
    // keep the image point under (sx, sy) stationary
    const ix = centerX + (sx - canvas.width / 2) / zoom;
    const iy = centerY + (sy - canvas.height / 2) / zoom;
    zoom = next;
    centerX = ix - (sx - canvas.width / 2) / zoom;
    centerY = iy - (sy - canvas.height / 2) / zoom;
    clampCenter();
    requestFrame();
  }

  function onWheel(e: WheelEvent) {
    e.preventDefault();
    zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.offsetX, e.offsetY);
  }

  function onPointerMove(e: PointerEvent) {
    if (!dragging) return;
    centerX -= e.movementX / zoom;
    centerY -= e.movementY / zoom;
    clampCenter();
    requestFrame();
  }

  function onKey(e: KeyboardEvent) {
    const pan = 64 / zoom;
    if (e.key === "ArrowLeft") centerX -= pan;
    else if (e.key === "ArrowRight") centerX += pan;
    else if (e.key === "ArrowUp") centerY -= pan;
    else if (e.key === "ArrowDown") centerY += pan;
    else if (e.key === "+" || e.key === "=") zoomAt(1.25, canvas.width / 2, canvas.height / 2);
    else if (e.key === "-") zoomAt(1 / 1.25, canvas.width / 2, canvas.height / 2);
    else if (e.key === "0") {
      zoom = fitZoom(info.levels[0], canvas.width, canvas.height);
      centerX = info.width / 2;
      centerY = info.height / 2;
    } else if (e.key === "1") zoom = 1;
    else return;
    e.preventDefault();
    clampCenter();
    requestFrame();
  }

  onMount(() => {
    const dpr = window.devicePixelRatio || 1;
    const resize = () => {
      canvas.width = canvas.clientWidth * dpr;
      canvas.height = canvas.clientHeight * dpr;
      requestFrame();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    renderer = new TileRenderer(canvas);
    renderer.onTileLoaded = requestFrame;
    zoom = fitZoom(info.levels[0], canvas.width, canvas.height);
    requestFrame();
    requestAnimationFrame(frame);
    return () => ro.disconnect();
  });
</script>

<canvas
  bind:this={canvas}
  role="img"
  aria-label="Scan viewer: arrows pan, plus and minus zoom, 0 fits, 1 is 100%"
  tabindex="0"
  onwheel={onWheel}
  onpointerdown={(e) => {
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
  }}
  onpointerup={() => (dragging = false)}
  onpointermove={onPointerMove}
  onkeydown={onKey}
></canvas>

<style>
  canvas {
    width: 100%;
    height: 100%;
    display: block;
    touch-action: none;
    cursor: grab;
  }
  canvas:focus-visible {
    outline: 3px solid #6ab0ff;
    outline-offset: -3px;
  }
</style>
```

`app/src/App.svelte`:

```svelte
<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { open } from "@tauri-apps/plugin-dialog";
  import Viewer from "./lib/Viewer.svelte";
  import type { Level } from "./lib/viewport";

  interface ImageInfo {
    id: number;
    width: number;
    height: number;
    levels: Level[];
  }

  let info: ImageInfo | null = $state(null);
  let error: string | null = $state(null);

  async function openScan() {
    error = null;
    const path = await open({
      multiple: false,
      filters: [{ name: "Scans", extensions: ["tif", "tiff", "png", "jpg", "jpeg"] }],
    });
    if (typeof path !== "string") return;
    try {
      info = await invoke<ImageInfo>("open_image", { path });
    } catch (e) {
      error = String(e);
    }
  }
</script>

<div class="shell">
  <header>
    <button onclick={openScan}>Open scan</button>
    {#if error}<p role="alert">{error}</p>{/if}
  </header>
  <section class="stage">
    {#if info}
      <Viewer {info} />
    {:else}
      <p class="hint">Open a scan to begin.</p>
    {/if}
  </section>
</div>

<style>
  .shell {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
  header {
    padding: 0.5rem;
    display: flex;
    gap: 0.75rem;
    align-items: center;
  }
  button {
    font: inherit;
    padding: 0.4rem 0.9rem;
  }
  button:focus-visible {
    outline: 3px solid #6ab0ff;
  }
  .stage {
    flex: 1;
    min-height: 0;
  }
  .hint {
    text-align: center;
    color: #999;
    margin-top: 4rem;
  }
  [role="alert"] {
    color: #ff9c9c;
    margin: 0;
  }
</style>
```

The file dialog needs the plugin: add `"@tauri-apps/plugin-dialog": "^2.0.0"` to package.json dependencies, `tauri-plugin-dialog = "2"` to src-tauri Cargo dependencies, and `.plugin(tauri_plugin_dialog::init())` in `run()`.

- [ ] **Step 3: Run tests and checks**

Run: `npm run test && npm run check` and `cargo test -p unduster-app && cargo clippy -p unduster-app --all-targets -- -D warnings`
Expected: all green (renderer.test.ts covers TextureStore; GL paths compile via svelte-check).

- [ ] **Step 4: Manual verification (documented, not skippable)**

Generate a big test image and open it:

```bash
cd /Users/albert/Development/TheUnduster/training
uv run python -c "
import numpy as np
from unduster_training.io import save_image
rng = np.random.default_rng(0)
big = rng.random((8000, 12000)).astype('float32')  # 96MP grey
save_image('/tmp/big-scan.tif', big)
print('wrote /tmp/big-scan.tif')"
cd ../app && npm run tauri dev
```

Open `/tmp/big-scan.tif`. Verify: fit view appears; wheel-zoom to 100% keeps the point under the cursor; pan by drag and by arrow keys is smooth (no visible stutter); zooming out never flashes white (parent level shows beneath); `0` and `1` work; the canvas shows a focus ring when tabbed to. Note observations in the commit message.

- [ ] **Step 5: Commit**

```bash
git add app
git commit -m "Add WebGL2 tiled viewer with keyboard and pointer navigation"
```

---

### Task 6: App CI

**Files:**
- Create: `.github/workflows/app.yml`

**Interfaces:**
- Produces: CI running vitest + svelte-check + the shell crate's tests on pushes touching `app/**` or `engine/**` (the app depends on engine crates).

- [ ] **Step 1: Write the workflow**

`.github/workflows/app.yml`:

```yaml
name: app

on:
  push:
    paths:
      - "app/**"
      - "engine/**"
      - ".github/workflows/app.yml"
  pull_request:
    paths:
      - "app/**"
      - "engine/**"
      - ".github/workflows/app.yml"

jobs:
  frontend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: app
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
      - run: npm ci
      - run: npm run check
      - run: npm run test

  shell:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - uses: Swatinem/rust-cache@v2
        with:
          workspaces: engine
      - name: Install Tauri linux deps
        run: |
          sudo apt-get update
          sudo apt-get install -y libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf
      - run: cd engine && cargo test -p unduster-app
      - run: cd engine && cargo clippy -p unduster-app --all-targets -- -D warnings
```

- [ ] **Step 2: Verify locally then commit**

Run: `cd app && npm run check && npm run test` and `cd ../engine && cargo test -p unduster-app`
Expected: green.

```bash
git add .github/workflows/app.yml
git commit -m "Add app CI workflow"
```

---

## Definition of done for plan 3a

- App opens 16-bit TIFF/PNG/JPEG scans via the engine and displays them in the tiled viewer.
- Viewport math fully unit-tested; texture cache eviction unit-tested; shell registry and protocol parsing unit-tested; manual smoothness check on a ~100MP image documented in the final commit.
- Deliberately NOT here (plans 3b/3c): detection, mask overlay + sensitivity shader, filmstrip, brush, sidecar persistence, healing, export, before/after (needs healed output), automated frame-time CI.
