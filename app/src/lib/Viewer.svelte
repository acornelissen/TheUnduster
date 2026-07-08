<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { fitZoom, visibleTiles, TILE, type Level } from "./viewport";
  import { TileRenderer } from "./renderer";

  interface ImageInfo {
    id: number;
    width: number;
    height: number;
    levels: Level[];
  }

  interface Overlay {
    enabled: boolean;
    threshold: number;
  }

  let {
    info,
    overlay,
    detected,
    onRequestDetect,
  }: {
    info: ImageInfo;
    overlay: Overlay;
    detected: boolean;
    onRequestDetect: () => void;
  } = $props();

  let canvas: HTMLCanvasElement;
  let renderer: TileRenderer | undefined;
  let zoom = 1;
  let centerX = info.width / 2;
  let centerY = info.height / 2;
  let dragging = false;
  let needsFrame = true;

  let detections: [number, number, number, number][] = $state([]);
  let current = -1;

  function requestFrame() {
    needsFrame = true;
  }

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

  function cycleDetection(dir: 1 | -1) {
    if (!detections.length) return;
    current = (current + dir + detections.length) % detections.length;
    const [x0, y0, x1, y1] = detections[current];
    zoom = 1;
    centerX = (x0 + x1) / 2;
    centerY = (y0 + y1) / 2;
    clampCenter();
    requestFrame();
  }

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
      renderer.draw(tilePaths(), canvas.width, canvas.height, overlay);
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
    const dpr = window.devicePixelRatio || 1;
    zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.offsetX * dpr, e.offsetY * dpr);
  }

  function onPointerMove(e: PointerEvent) {
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
  role="application"
  aria-label="Scan viewer: arrows pan, plus and minus zoom, 0 fits, 1 is 100%, d detects, m toggles overlay, z and shift-z cycle defects"
  tabindex="0"
  onwheel={onWheel}
  onpointerdown={(e) => {
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
  }}
  onpointerup={() => (dragging = false)}
  onpointercancel={() => (dragging = false)}
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
