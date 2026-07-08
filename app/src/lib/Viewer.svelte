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
