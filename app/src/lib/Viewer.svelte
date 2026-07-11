<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import Icon from "./Icon.svelte";
  import { fitZoom, visibleTiles, ringsFor, ringForBbox, wheelZoomFactor, TILE, type Level } from "./viewport";
  import { TileRenderer, probPathFor, type StrokeSegment } from "./renderer";
  import {
    screenToImage,
    stepRadius,
    pushStroke,
    chunkPoints,
    MAX_POINTS_PER_STROKE,
    MAX_STROKES,
    type StrokeData,
  } from "./brush";
  import { eraseStrokeForBbox, nextCurrentAfterRemoval } from "./detections";

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
    onBrushLimit,
    onDetectionsChange,
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
    onBrushLimit?: (message: string) => void;
    onDetectionsChange?: (count: number) => void;
  } = $props();

  let canvas: HTMLCanvasElement;
  let renderer: TileRenderer | undefined;
  let zoom = $state(1);
  let centerX = info.width / 2;
  let centerY = info.height / 2;
  let dragging = false;
  let needsFrame = true;
  let running = true;
  let rafId = 0;

  let detections: [number, number, number, number][] = $state([]);
  let current = -1;
  let showHealed = $state(false);
  // The threshold `detections` was last populated for -- via a fresh
  // `components` fetch (refreshDetections) or a caller-supplied result
  // (setDetections). Lets the slider $effect below tell "the threshold
  // actually changed" apart from "detected flipped true/false at the same
  // threshold", so a probe-driven detected flip doesn't schedule a redundant
  // refetch of data it was just handed.
  //
  // $state, deliberately: the slider $effect's equality gate reads this, so
  // a refetch that RESOLVES at a threshold the slider has since left (drag
  // A -> B, B's fetch lands after a return to A) reruns the effect, which
  // sees the mismatch and schedules the corrective refetch. As a plain
  // variable the resolve-time write would be invisible and B's boxes would
  // sit undimmed under an A slider with nothing left to fix them.
  let lastFetchedThreshold: number | null = $state(null);
  // True from the moment a slider-driven refetch is (re)scheduled until
  // refreshDetections resolves; drives the dimmed defect-ring paint below so
  // the operator can see the rings are stale rather than trusting circles
  // that no longer match the current threshold.
  let refreshPending = $state(false);

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
    lastFetchedThreshold = null;
    showHealed = false;
    // Strokes themselves arrive per-frame via props and belong to App; only
    // the transient in-canvas brush mode resets here.
    brushMode = "off";
    // An in-progress stroke's coordinates are meaningless on another frame;
    // drop it rather than let a pointerup after the switch commit frame N's
    // geometry onto frame N+1.
    painting = false;
    livePoints = [];
    // A pending refresh belonged to the frame we just left; the new frame
    // hasn't scheduled one yet, so don't leave the rings dimmed for it.
    refreshPending = false;
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
      // Only on success: a failed fetch (no probabilities yet) must not
      // report a count, so the status bar's fallback copy stays in charge.
      lastFetchedThreshold = threshold;
      onDetectionsChange?.(detections.length);
    } catch {
      // no detection run yet; leave detections empty until `detect` succeeds
      detections = [];
    } finally {
      refreshPending = false;
    }
    current = -1;
    // Explicit redraw trigger: the overlay repaints once the component list
    // (used by z/Z cycling) is fetched. This replaces relying on a Viewer
    // remount to force a redraw.
    requestFrame();
  }

  /** Feeds detections the caller already fetched (the activation probe's own
   * `components` call) straight into the viewer, instead of App awaiting a
   * second `refreshDetections` call that would re-invoke `components` with
   * the identical id+threshold. Same net effect as `refreshDetections` on
   * success -- detections replace, cycling resets, the live count updates,
   * a redraw is requested -- minus the invoke. `threshold` records as
   * `lastFetchedThreshold` so the slider $effect below recognizes these
   * detections are already current for that threshold and, when the caller
   * flips `detected` right after this call, does not schedule a redundant
   * debounced refetch. */
  export function setDetections(
    boxes: [number, number, number, number][],
    threshold: number,
  ) {
    detections = boxes;
    current = -1;
    lastFetchedThreshold = threshold;
    onDetectionsChange?.(detections.length);
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
    //
    // This effect also reruns when `detected` flips true with the threshold
    // unchanged (the activation probe's detected-flip): `setDetections`
    // recorded that threshold as `lastFetchedThreshold`, so the refetch below
    // is skipped -- the detections it would fetch are already in hand. A
    // genuine threshold change never matches `lastFetchedThreshold` and still
    // refetches, keeping the slider live.
    const threshold = overlay.threshold;
    void overlay.enabled;
    requestFrame();
    clearTimeout(refreshTimer);
    if (detected && threshold !== lastFetchedThreshold) {
      refreshPending = true;
      refreshTimer = setTimeout(() => {
        refreshDetections(threshold);
      }, 250);
    } else {
      // A slider wiggle back to the fetched threshold within the debounce
      // (A -> B -> A) lands here having cleared B's timer above; the rings
      // aren't stale -- detections match this threshold -- so undim them.
      // Without this the dimmed paint would persist until the next real
      // refetch resolved.
      refreshPending = false;
    }
    return () => clearTimeout(refreshTimer);
  });

  // The rAF loop only repaints when needsFrame is set, so every site that
  // flips refreshPending would otherwise need its own requestFrame() call.
  // A dedicated effect keeps that "pending changed -> repaint" wiring in one
  // place instead of duplicated across the schedule site, refreshDetections,
  // and the frame-switch reset.
  $effect(() => {
    void refreshPending;
    requestFrame();
  });

  /** Finalizes a stroke: chunks it to the backend's per-stroke point cap
   * (a fit-zoom drag on a large scan can easily exceed it), pushes each
   * chunk onto the undo stack in order (so undo pops chunks one at a time --
   * acceptable and predictable), and hands the new strokes/redo pair up to
   * App, which owns persistence. If the chunks would push the total stroke
   * count past the backend's cap, only as many as fit are committed and the
   * excess is surfaced via `onBrushLimit`. */
  function commitStroke(points: [number, number][]) {
    commitStrokeAs(points, brushMode === "erase", brushRadius);
  }

  /** Core of commitStroke, parameterized on erase/radius instead of reading
   * them off brush state. The active-detection delete flow below needs to
   * commit a single-point erase stroke without actually turning brush mode
   * on -- flipping brushMode/brushRadius to fake it would flash the palette
   * buttons and the cursor ring for an action the operator didn't ask for.
   * This is the smallest change that lets it reuse the same chunking/cap/
   * undo-redo pipeline as a real brush stroke. */
  function commitStrokeAs(points: [number, number][], erase: boolean, radius: number) {
    if (points.length === 0) return;
    const chunks = chunkPoints(points, MAX_POINTS_PER_STROKE);
    let curStrokes = strokes;
    let curRedo = redoStrokes;
    let committed = 0;
    for (const chunk of chunks) {
      if (curStrokes.length >= MAX_STROKES) break;
      const s: StrokeData = { erase, radius, points: chunk };
      const result = pushStroke(curStrokes, curRedo, s);
      curStrokes = result.strokes;
      curRedo = result.redo;
      committed++;
    }
    if (committed > 0) {
      onStrokesChange?.(curStrokes, curRedo);
    }
    if (committed < chunks.length) {
      onBrushLimit?.("stroke limit reached for this frame; undo or heal to continue");
    }
  }

  /** Deletes the actively-cycled detection ring (Delete/Backspace while
   * `current >= 0`): paints a single-point erase stroke covering its bbox,
   * riding the existing stroke pipeline for undo/redo, persistence, and
   * heal-mask subtraction for free.
   *
   * An erase stroke only subtracts from the heal mask -- detections are a
   * separate layer that renders independently of strokes, so without more
   * the ring would keep drawing right where it was, reading as "delete
   * didn't work". So this also drops the box from the local `detections`
   * array so the ring itself disappears immediately. That array is
   * display-only: a later refetch or re-detect can restore the box, which
   * is fine, because the erase stroke has already made the heal mask
   * correct regardless of whether this box reappears in a later pass.
   *
   * `detected` only: pre-Detect cycling runs over the roll queue's `bboxes`
   * prop, an array Viewer doesn't own and so can't trim -- an erase stroke
   * committed there would leave the ring drawn exactly where it was, an
   * invisible mutation that reads as "nothing happened" and stacks another
   * stroke on every retry. The onKey branch already gates on `detected`;
   * this early return keeps the invariant local should another caller
   * appear. */
  function deleteActiveDetection() {
    if (!detected) return;
    const source = markerSource();
    if (current < 0 || current >= source.length) return;
    const { cx, cy, radius } = eraseStrokeForBbox(source[current]);
    commitStrokeAs([[cx, cy]], true, radius);
    const removed = current;
    detections = detections.filter((_, i) => i !== removed);
    onDetectionsChange?.(detections.length);
    current = nextCurrentAfterRemoval(removed, detections.length);
    requestFrame();
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
      // Rings mark defects; they follow the overlay toggle (m) at every zoom
      // rather than vanishing above 50% zoom -- the operator twice reported
      // the old zoom cutoff as a bug.
      const source = markerSource();
      const ringsVisible = overlay.enabled;
      if (ringsVisible && source.length > 0) {
        // Dimmed while a threshold refetch is in flight, for both colors
        // below, so the operator can see the drawn circles are stale rather
        // than trusting them at full strength.
        const alpha = refreshPending ? 0.35 : 1.0;
        // The z/shift-z cycling target (`current`) gets its own ring pass in
        // amber so the operator can see which detection is selected before
        // deleting it; every other ring stays the plain --detect red. It is
        // excluded from the list handed to ringsFor (rather than drawn twice
        // on top) so the two passes never fight over the same pixels.
        const rest = source.filter((_, i) => i !== current);
        const rings = ringsFor(rest, zoom, centerX, centerY, canvas.width, canvas.height, 12);
        // --detect red; app.css token is the CSS-side mirror.
        renderer.drawRings(rings, [1.0, 0.05, 0.05, alpha], canvas.width, canvas.height);
        if (current >= 0 && current < source.length) {
          // Project the active box directly with ringForBbox instead of
          // going through ringsFor: ringsFor's offscreen filter would drop
          // the active ring's index out of that array's positions entirely
          // once it (or another ring before it) pans offscreen, and the
          // highlight must still track it, ready to reappear the moment it
          // comes back onscreen.
          const active = ringForBbox(
            source[current],
            zoom,
            centerX,
            centerY,
            canvas.width,
            canvas.height,
            12,
          );
          // Amber: mirrors --accent (app.css) and the paint-stroke amber
          // used for brush strokes below, so "selected" reads consistently
          // across the app. A few px larger than the plain ring so the
          // highlight is visible even where it would otherwise sit flush
          // under the red ring it replaces.
          renderer.drawRings(
            [{ ...active, r: active.r + 4 }],
            [1.0, 0.72, 0.24, alpha],
            canvas.width,
            canvas.height,
          );
        }
      }
      // Strokes are edit state, not a detector overlay: they stay visible
      // regardless of the `m` tint toggle. The in-progress stroke (not yet
      // committed to `strokes`) is appended so painting gives live feedback.
      // The healed "after" view is the exception: strokes annotate the
      // BEFORE state and sit exactly over the healed pixels the operator is
      // trying to inspect, so the space toggle shows the result unobscured.
      if (!showHealed) {
        const allStrokes =
          painting && livePoints.length > 0
            ? [
                ...strokes,
                { erase: brushMode === "erase", radius: brushRadius, points: livePoints },
              ]
            : strokes;
        const paintSegs = strokeSegments(allStrokes.filter((s) => !s.erase));
        const eraseSegs = strokeSegments(allStrokes.filter((s) => s.erase));
        if (paintSegs.length > 0) {
          renderer.drawStrokes(paintSegs, [1.0, 0.72, 0.24, 0.35], canvas.width, canvas.height);
        }
        if (eraseSegs.length > 0) {
          renderer.drawStrokes(eraseSegs, [0.91, 0.9, 0.89, 0.35], canvas.width, canvas.height);
        }
        if (brushMode !== "off") {
          const cx = (cursorX - centerX) * zoom + canvas.width / 2;
          const cy = (cursorY - centerY) * zoom + canvas.height / 2;
          renderer.drawRings(
            [{ x: cx, y: cy, r: brushRadius * zoom }],
            [1.0, 1.0, 1.0, 0.9],
            canvas.width,
            canvas.height,
          );
        }
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

  function zoomFit() {
    zoom = fitZoom(info.levels[0], canvas.width, canvas.height);
    centerX = info.width / 2;
    centerY = info.height / 2;
    clampCenter();
    requestFrame();
  }

  function zoomActual() {
    zoom = 1;
    clampCenter();
    requestFrame();
  }

  function zoomIn() {
    zoomAt(1.25, canvas.width / 2, canvas.height / 2);
  }

  function zoomOut() {
    zoomAt(1 / 1.25, canvas.width / 2, canvas.height / 2);
  }

  /** Toggles paint/erase brush mode. Shared by the b/e key branches and the
   * palette buttons, so it carries the full behavior those keys relied on:
   * flipping the mode off if it's already active, and -- only when turning a
   * brush on from off -- seeding the cursor at the current view center so the
   * brush ring appears somewhere visible immediately. */
  function toggleBrush(mode: "paint" | "erase") {
    const turningOn = brushMode === "off";
    brushMode = brushMode === mode ? "off" : mode;
    if (turningOn && brushMode !== "off") {
      cursorX = centerX;
      cursorY = centerY;
    }
    requestFrame();
  }

  /** Toggles the defect overlay tint. Mutates the `overlay` prop in place --
   * it's a $state object owned by App, and this is the same mutation path
   * the m key has always used, so App's reactivity picks it up the same way. */
  function toggleOverlay() {
    overlay.enabled = !overlay.enabled;
    requestFrame();
  }

  function onWheel(e: WheelEvent) {
    e.preventDefault();
    const dpr = window.devicePixelRatio || 1;
    zoomAt(wheelZoomFactor(e.deltaY, e.ctrlKey), e.offsetX * dpr, e.offsetY * dpr);
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
      toggleOverlay();
      return;
    // Plain z/Z only: with cmd/ctrl held this must fall through untouched so
    // the event bubbles to App's window-level undo/redo handler instead of
    // cycling detections and undoing a stroke on the same keypress.
    } else if ((e.key === "z" || e.key === "Z") && !e.metaKey && !e.ctrlKey) {
      e.preventDefault();
      cycleDetection(e.key === "z" ? 1 : -1);
      return;
    // Only in detected mode with a ring actually selected, and never while
    // brushing -- Delete/Backspace has no brush-mode meaning of its own, but
    // silently deleting a detection out from under an in-progress
    // paint/erase stroke would surprise the operator more than doing
    // nothing. The `detected` gate excludes pre-Detect cycling over the roll
    // queue's bboxes prop, where the ring couldn't be hidden (see
    // deleteActiveDetection).
    } else if ((e.key === "Delete" || e.key === "Backspace") && detected && current >= 0 && brushMode === "off") {
      e.preventDefault();
      deleteActiveDetection();
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
      toggleBrush(e.key === "b" ? "paint" : "erase");
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
      zoomFit();
    } else if (e.key === "1") zoomActual();
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
    // The Viewer only exists in the DOM while `info` is set (App gates it
    // behind `{#if info}`), so mount time is exactly the null->set
    // transition of a fresh open or roll swap -- never a frame-to-frame
    // navigation within an already-open roll, since those reuse this same
    // persistent instance. Focus here, once, so d/h/space/arrows/+/-/0/1
    // work immediately without an initial click.
    canvas.focus();
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
<div class="viewer-canvas-wrap">
  <canvas
    bind:this={canvas}
    role="application"
    aria-label="Scan viewer: arrows pan, plus and minus zoom, 0 fits, 1 is 100%, d detects, m toggles overlay, z and shift-z cycle defects, delete or backspace removes the selected defect, h heals, space toggles before and after, b paints, e erases, bracket keys size the brush, arrows nudge it and enter stamps while brushing, cmd-z undoes, escape exits, shift-cmd-z redoes"
    tabindex="0"
    class:brushing={brushMode !== "off" && !showHealed}
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
  {#if !glError}
    <div class="palette tool-palette">
      <button
          class="btn"
          class:btn-toggle-on={brushMode === "paint"}
          title="Paint mask (b)"
          aria-label="Paint mask"
          aria-pressed={brushMode === "paint"}
          onclick={() => {
            toggleBrush("paint");
            canvas.focus();
          }}><Icon name="paint" /> Paint</button
        >
        <button
          class="btn"
          class:btn-toggle-on={brushMode === "erase"}
          title="Erase mask (e)"
          aria-label="Erase mask"
          aria-pressed={brushMode === "erase"}
          onclick={() => {
            toggleBrush("erase");
            canvas.focus();
          }}><Icon name="erase" /> Erase</button
        >
        <button
          class="btn"
          class:btn-toggle-on={overlay.enabled}
          title="Overlay (m)"
          aria-label="Toggle defect overlay"
          aria-pressed={overlay.enabled}
          onclick={() => {
            toggleOverlay();
            canvas.focus();
          }}><Icon name="overlay" /> Overlay</button
        >
        {#if brushMode !== "off"}
          <span class="radius-readout">{brushRadius}px</span>
        {/if}
    </div>
    <div class="palette zoom-palette">
      <button class="btn" title="Zoom out (-)" aria-label="Zoom out" onclick={zoomOut}>&minus;</button>
      <span class="zoom-readout">{Math.round(zoom * 100)}%</span>
      <button class="btn" title="Zoom in (+)" aria-label="Zoom in" onclick={zoomIn}>+</button>
      <button class="btn" title="Fit (0)" aria-label="Fit to window" onclick={zoomFit}>Fit</button>
      <button class="btn" title="100% (1)" aria-label="Actual size" onclick={zoomActual}>1:1</button>
    </div>
  {/if}
</div>

<style>
  .viewer-canvas-wrap {
    position: relative;
    width: 100%;
    height: 100%;
  }
  canvas {
    width: 100%;
    height: 100%;
    display: block;
    background: var(--surround);
    touch-action: none;
    cursor: grab;
  }
  canvas.brushing {
    cursor: none;
  }
  .palette {
    position: absolute;
    bottom: var(--space-3);
    display: flex;
    align-items: center;
    gap: var(--space-1);
    padding: var(--space-1);
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: var(--radius-1);
  }
  .tool-palette {
    left: var(--space-3);
  }
  .zoom-palette {
    right: var(--space-3);
  }
  .zoom-readout {
    min-width: 4ch;
    text-align: center;
    font-size: var(--text-sm);
    color: var(--text-2);
    font-variant-numeric: tabular-nums;
  }
  .radius-readout {
    min-width: 5ch;
    text-align: center;
    font-size: var(--text-sm);
    color: var(--text-2);
    font-variant-numeric: tabular-nums;
  }
  .gl-error {
    color: var(--err);
    padding: var(--space-4);
    margin: 0;
  }

  canvas:focus-visible {
    outline: 3px solid var(--focus);
    outline-offset: -3px;
  }
</style>
