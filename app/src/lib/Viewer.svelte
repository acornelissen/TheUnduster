<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { fitZoom, visibleTiles, ringsFor, TILE, type Level } from "./viewport";
  import { TileRenderer, probPathFor } from "./renderer";

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
  }: {
    info: ImageInfo;
    overlay: Overlay;
    detected: boolean;
    healedAvailable: boolean;
    onRequestDetect: () => void;
    onRequestHeal: () => void;
    bboxes?: [number, number, number, number][] | null;
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
  aria-label="Scan viewer: arrows pan, plus and minus zoom, 0 fits, 1 is 100%, d detects, m toggles overlay, z and shift-z cycle defects, h heals, space toggles before and after"
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
