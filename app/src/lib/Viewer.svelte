<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { fitZoom, visibleTiles, ringsFor, TILE, type Level } from "./viewport";
  import { TileRenderer, probPathFor, type StrokeSegment } from "./renderer";
  import { screenToImage, stepRadius, pushStroke, type StrokeData } from "./brush";

  interface ImageInfo {
    id: number;
    width: number;
    height: number;
    levels: Level[];
    healed: boolean;
  }

  interface Overlay {
    enabled: boolean;
    threshold: number;
  }

  let {
    info,
    overlay,
    detected,
    healedAvailable,
    onRequestDetect,
    onRequestHeal,
    bboxes = null,
    strokes = [],
    redoStrokes = [],
    onStrokesChange,
  }: {
    info: ImageInfo;
    overlay: Overlay;
    detected: boolean;
    healedAvailable: boolean;
    onRequestDetect: () => void;
    onRequestHeal: () => void;
    bboxes?: [number, number, number, number][] | null;
    strokes?: StrokeData[];
    redoStrokes?: StrokeData[];
    onStrokesChange?: (strokes: StrokeData[], redo: StrokeData[]) => void;
  } = $props();

  let canvas: HTMLCanvasElement;
  let renderer: TileRenderer | undefined;
  let zoom = 1;
  let centerX = info.width / 2;
  let centerY = info.height / 2;
  let dragging = false;
  let needsFrame = true;
  let running = true;
  let rafId = 0;

  let detections: [number, number, number, number][] = $state([]);
  let current = -1;
  let showHealed = $state(false);

  let brushMode: "off" | "paint" | "erase" = $state("off");
  let brushRadius = $state(24);
  let cursorX = $state(0);
  let cursorY = $state(0);
  let painting = false;
  let livePoints: [number, number][] = [];

  // Exposed to App (via bind:this) for its status line. Svelte 5 disallows
  // exporting a $derived value directly from a component; a getter function
  // is the supported instance-API shape, so App reads it each render.
  export function brushStatus(): string | null {
    if (brushMode === "off") return null;
    return `${brushMode === "paint" ? "brush" : "erase"} ${Math.round(brushRadius)}px`;
  }

  function requestFrame() {
    needsFrame = true;
  }

  // The Viewer is a single long-lived instance: remounting per frame switch
  // (the old {#key info.id} approach) tears down the GL context and texture
  // cache on every switch, refetching every visible tile from scratch. React
  // to the image changing instead; the LRU texture store keeps neighboring
  // frames' tiles warm, so switching back is instant.
  let lastInfoId = -1;
  $effect(() => {
    if (info.id === lastInfoId) return;
    lastInfoId = info.id;
    detections = [];
    current = -1;
    showHealed = false;
    // Strokes themselves arrive per-frame via props and belong to App; only
    // the transient in-canvas brush mode resets here.
    brushMode = "off";
    centerX = info.width / 2;
    centerY = info.height / 2;
    if (canvas && canvas.width > 0) {
      zoom = fitZoom(info.levels[0], canvas.width, canvas.height);
    }
    requestFrame();
  });

  $effect(() => {
    // If healed data vanishes (frame evicted and re-decoded), drop the toggle.
    if (!healedAvailable) showHealed = false;
  });

  export async function refreshDetections(threshold: number) {
    try {
      detections = await invoke("components", { id: info.id, threshold });
    } catch {
      // no detection run yet; leave detections empty until `detect` succeeds
      detections = [];
    }
    current = -1;
    // Explicit redraw trigger: the overlay repaints once the component list
    // (used by z/Z cycling) is fetched. This replaces relying on a Viewer
    // remount to force a redraw.
    requestFrame();
  }

  /** One source of truth for defect markers, shared by z/Z navigation and
   * the ring pass: live detections once a detect has run (even when empty --
   * raising the threshold to zero components must not resurrect stale
   * boxes), otherwise the roll queue's stored 0.5-threshold bboxes. */
  function markerSource(): [number, number, number, number][] {
    return detected ? detections : (bboxes ?? []);
  }

  function cycleDetection(dir: 1 | -1) {
    const source = markerSource();
    if (!source.length) return;
    current = (current + dir + source.length) % source.length;
    const [x0, y0, x1, y1] = source[current];
    zoom = 1;
    centerX = (x0 + x1) / 2;
    centerY = (y0 + y1) / 2;
    clampCenter();
    requestFrame();
  }

  // The roll queue delivers bboxes asynchronously mid-scan; redraw when
  // they land so ring markers appear without waiting for a pan or zoom.
  $effect(() => {
    void bboxes;
    requestFrame();
  });

  // Strokes change from outside (undo/redo, a frame switch delivering a
  // seeded list) as well as from local painting; redraw either way.
  $effect(() => {
    void strokes;
    requestFrame();
  });

  let refreshTimer: ReturnType<typeof setTimeout> | undefined;

  $effect(() => {
    // Read both fields so the effect reruns on either change; a slider drag
    // redraws immediately (uniform-only, no refetch) and, debounced, syncs
    // the component list used by z/Z so cycling targets stay current. The
    // redraw is unconditional (it must reflect the slider even before any
    // detection exists), but the debounced refetch is non-load-bearing on
    // mount: only invoke it when a detection actually exists, so a fresh
    // Viewer mount doesn't fire a doomed `components` call before `detect`
    // has ever run.
    const threshold = overlay.threshold;
    void overlay.enabled;
    requestFrame();
    clearTimeout(refreshTimer);
    if (detected) {
      refreshTimer = setTimeout(() => {
        refreshDetections(threshold);
      }, 250);
    }
    return () => clearTimeout(refreshTimer);
  });

  /** Finalizes a stroke: pushes it onto the undo stack and hands the new
   * strokes/redo pair up to App, which owns persistence. */
  function commitStroke(points: [number, number][]) {
    if (points.length === 0) return;
    const s: StrokeData = { erase: brushMode === "erase", radius: brushRadius, points };
    const result = pushStroke(strokes, redoStrokes, s);
    onStrokesChange?.(result.strokes, result.redo);
  }

  /** Image-space stroke -> screen-space capsule segments for drawStrokes(). */
  function strokeSegments(list: StrokeData[]): StrokeSegment[] {
    const segs: StrokeSegment[] = [];
    for (const s of list) {
      const r = s.radius * zoom;
      const toScreen = (p: [number, number]): [number, number] => [
        (p[0] - centerX) * zoom + canvas.width / 2,
        (p[1] - centerY) * zoom + canvas.height / 2,
      ];
      if (s.points.length === 1) {
        const [x, y] = toScreen(s.points[0]);
        segs.push({ ax: x, ay: y, bx: x, by: y, r });
        continue;
      }
      for (let i = 1; i < s.points.length; i++) {
        const [ax, ay] = toScreen(s.points[i - 1]);
        const [bx, by] = toScreen(s.points[i]);
        segs.push({ ax, ay, bx, by, r });
      }
    }
    return segs;
  }

  function tilePaths() {
    return visibleTiles(info.levels, zoom, centerX, centerY, canvas.width, canvas.height).map(
      (t) => {
        const l = info.levels[t.level];
        const tileW = Math.min(l.width - t.tx * TILE, TILE);
        const tileH = Math.min(l.height - t.ty * TILE, TILE);
        const base = `/${info.id}/${t.level}/${t.tx}/${t.ty}`;
        return {
          path: showHealed ? `/healed${base}` : base,
          probPath: probPathFor(base),
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
    if (!running) return; // stopped on unmount; do not re-arm the rAF loop
    if (renderer && needsFrame) {
      needsFrame = false;
      renderer.draw(tilePaths(), canvas.width, canvas.height, overlay);
      // Rings mark defects whenever the red probability tint cannot: always
      // for queue bboxes (no prob tiles exist before a live detect, at any
      // zoom), and below 50% zoom for live detections, where the tint is
      // sub-pixel. Past that, the tint takes over and rings would clutter.
      const source = markerSource();
      const ringsVisible = !detected || zoom < 0.5;
      if (ringsVisible && source.length > 0) {
        const rings = ringsFor(source, zoom, centerX, centerY, canvas.width, canvas.height, 12);
        renderer.drawRings(rings, canvas.width, canvas.height);
      }
      // Strokes are edit state, not a detector overlay: they stay visible
      // regardless of the `m` tint toggle. The in-progress stroke (not yet
      // committed to `strokes`) is appended so painting gives live feedback.
      const allStrokes =
        painting && livePoints.length > 0
          ? [...strokes, { erase: brushMode === "erase", radius: brushRadius, points: livePoints }]
          : strokes;
      const paintSegs = strokeSegments(allStrokes.filter((s) => !s.erase));
      const eraseSegs = strokeSegments(allStrokes.filter((s) => s.erase));
      if (paintSegs.length > 0) {
        renderer.drawStrokes(paintSegs, [1.0, 0.72, 0.24, 0.35], canvas.width, canvas.height);
      }
      if (eraseSegs.length > 0) {
        renderer.drawStrokes(eraseSegs, [0.42, 0.69, 1.0, 0.3], canvas.width, canvas.height);
      }
      if (brushMode !== "off") {
        const cx = (cursorX - centerX) * zoom + canvas.width / 2;
        const cy = (cursorY - centerY) * zoom + canvas.height / 2;
        renderer.drawRings([{ x: cx, y: cy, r: brushRadius * zoom }], canvas.width, canvas.height);
      }
    }
    rafId = requestAnimationFrame(frame);
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
    const dpr = window.devicePixelRatio || 1;
    zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.offsetX * dpr, e.offsetY * dpr);
  }

  function onPointerMove(e: PointerEvent) {
    if (brushMode !== "off") {
      const dpr = window.devicePixelRatio || 1;
      const [ix, iy] = screenToImage(
        e.offsetX * dpr,
        e.offsetY * dpr,
        zoom,
        centerX,
        centerY,
        canvas.width,
        canvas.height,
      );
      cursorX = ix;
      cursorY = iy;
      if (painting) {
        const last = livePoints[livePoints.length - 1];
        // Bound stroke size: skip points that barely moved from the last
        // captured one (2 image px), rather than one per pointermove.
        if (!last || Math.hypot(ix - last[0], iy - last[1]) >= 2) {
          livePoints.push([ix, iy]);
        }
      }
      requestFrame();
      return;
    }
    if (!dragging) return;
    // e.movementX/Y are CSS px but zoom relates image px to device px
    // (canvas.width is device px, see resize()), so convert to device px
    // here too or panning under-shoots by 1/dpr on HiDPI displays.
    const dpr = window.devicePixelRatio || 1;
    centerX -= (e.movementX * dpr) / zoom;
    centerY -= (e.movementY * dpr) / zoom;
    clampCenter();
    requestFrame();
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === "d") {
      e.preventDefault();
      onRequestDetect();
      return;
    } else if (e.key === "m") {
      e.preventDefault();
      overlay.enabled = !overlay.enabled;
      requestFrame();
      return;
    } else if (e.key === "z" || e.key === "Z") {
      e.preventDefault();
      cycleDetection(e.key === "z" ? 1 : -1);
      return;
    } else if (e.key === "h") {
      e.preventDefault();
      onRequestHeal();
      return;
    } else if (e.key === " ") {
      if (healedAvailable) {
        e.preventDefault();
        showHealed = !showHealed;
        requestFrame();
      }
      return;
    } else if (e.key === "b" || e.key === "e") {
      e.preventDefault();
      const mode = e.key === "b" ? "paint" : "erase";
      brushMode = brushMode === mode ? "off" : mode;
      requestFrame();
      return;
    } else if (e.key === "Escape" && brushMode !== "off") {
      e.preventDefault();
      brushMode = "off";
      requestFrame();
      return;
    } else if ((e.key === "[" || e.key === "]") && brushMode !== "off") {
      e.preventDefault();
      brushRadius = stepRadius(brushRadius, e.key === "]" ? 1 : -1);
      requestFrame();
      return;
    } else if (brushMode !== "off" && e.key.startsWith("Arrow")) {
      e.preventDefault();
      const step = (e.shiftKey ? 64 : 16) / zoom;
      if (e.key === "ArrowLeft") cursorX -= step;
      else if (e.key === "ArrowRight") cursorX += step;
      else if (e.key === "ArrowUp") cursorY -= step;
      else if (e.key === "ArrowDown") cursorY += step;
      requestFrame();
      return;
    } else if (brushMode !== "off" && e.key === "Enter") {
      e.preventDefault();
      commitStroke([[cursorX, cursorY]]);
      return;
    }
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

  let glError: string | null = $state(null);

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
    try {
      renderer = new TileRenderer(canvas);
    } catch (e) {
      // A mount-time throw would silently break this component AND freeze
      // sibling template updates; fail visibly instead. The global error
      // hook in main.ts additionally forwards this to the dev terminal.
      glError = String(e);
      throw e;
    }
    renderer.onTileLoaded = requestFrame;
    zoom = fitZoom(info.levels[0], canvas.width, canvas.height);
    requestFrame();
    rafId = requestAnimationFrame(frame);
    return () => {
      // The Viewer unmounts on mode switches (roll opened, everything
      // closed). Without stopping the render loop and dropping the GL
      // context, each unmount leaks a WebGL context and an orphaned rAF
      // loop -- WebKit caps live contexts near 16, then crashes.
      running = false;
      cancelAnimationFrame(rafId);
      ro.disconnect();
      renderer?.dispose();
      renderer = undefined;
    };
  });
</script>

{#if glError}
  <p role="alert" class="gl-error">Viewer failed to start: {glError}</p>
{/if}
<canvas
  bind:this={canvas}
  role="application"
  aria-label="Scan viewer: arrows pan, plus and minus zoom, 0 fits, 1 is 100%, d detects, m toggles overlay, z and shift-z cycle defects, h heals, space toggles before and after, b paints, e erases, bracket keys size the brush, arrows nudge it and enter stamps while brushing, cmd-z undoes"
  tabindex="0"
  onwheel={onWheel}
  onpointerdown={(e) => {
    canvas.setPointerCapture(e.pointerId);
    if (brushMode !== "off") {
      // Compute from the event directly rather than trusting cursorX/Y: a
      // pointerdown with no preceding pointermove over the canvas (the very
      // first click) would otherwise start the stroke at their stale (0,0)
      // init value instead of the actual click position.
      const dpr = window.devicePixelRatio || 1;
      const [ix, iy] = screenToImage(
        e.offsetX * dpr,
        e.offsetY * dpr,
        zoom,
        centerX,
        centerY,
        canvas.width,
        canvas.height,
      );
      cursorX = ix;
      cursorY = iy;
      painting = true;
      livePoints = [[ix, iy]];
      requestFrame();
      return;
    }
    dragging = true;
  }}
  onpointerup={() => {
    dragging = false;
    if (painting) {
      painting = false;
      commitStroke(livePoints);
      livePoints = [];
      requestFrame();
    }
  }}
  onpointercancel={() => {
    dragging = false;
    painting = false;
    livePoints = [];
  }}
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
  .gl-error {
    color: #ff9c9c;
    padding: 1rem;
    margin: 0;
  }

  canvas:focus-visible {
    outline: 3px solid #6ab0ff;
    outline-offset: -3px;
  }
</style>
