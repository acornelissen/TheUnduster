import { TILE } from "./viewport";

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
uniform sampler2D probs;
uniform float threshold;
uniform float overlayOn;
void main() {
  vec4 base = texture(tile, vUv);
  float p = texture(probs, vUv).r;
  // p is an 8-bit quantization of the native f32 probability (see
  // build_prob_pyramid), so this GPU compare can disagree with Rust's
  // f32 threshold compare (used by the components command) by up to
  // ~0.002 (half a u8 step out of 255). That's below the slider's step
  // granularity (0.01), so it never produces a visibly different result.
  float hit = overlayOn * step(threshold, p) * step(0.004, p); // never tint p==0
  // Saturated red at high opacity: masks must read at a glance on both
  // grey film and colour scans; subtlety here costs missed defects.
  color = mix(base, vec4(1.0, 0.05, 0.05, 1.0), hit * 0.9);
}`;

/** Maps a tile's rgba path to its probability-layer counterpart. */
export function probPathFor(path: string): string {
  return "/probs" + path;
}

const RING_VERT = `#version 300 es
in vec2 corner;
uniform vec2 viewport;
uniform vec2 center;
uniform float radius;
out vec2 vCorner;
void main() {
  vCorner = corner;
  vec2 pos = center + corner * (radius + 3.0);
  vec2 clip = (pos / viewport) * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
}`;

const RING_FRAG = `#version 300 es
// highp to match the vertex stage: 'radius' is declared in both shaders and
// GLSL ES 300 requires identical precision, or the program fails to link
// (vertex-stage floats default to highp). WebGL2 guarantees fragment highp.
precision highp float;
in vec2 vCorner;
out vec4 color;
uniform float radius;
void main() {
  float d = length(vCorner) * (radius + 3.0);
  // Soft 2px annulus at the ring radius: smoothstep in from both sides so
  // the edge anti-aliases instead of stair-stepping.
  float outer = 1.0 - smoothstep(radius - 1.0, radius + 1.0, d);
  float inner = smoothstep(radius - 3.0, radius - 1.0, d);
  float alpha = outer * inner * 0.9;
  if (alpha <= 0.0) discard;
  color = vec4(1.0, 0.05, 0.05, alpha);
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
    // Invariant: a single entry larger than the whole budget is never evicted
    // (the `entries.size <= 1` guard below keeps it), which is safe only
    // because max tile bytes (512*512*4 = 1MB) stays far below the budget.
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

  /** Evict everything, running onEvict for each so GPU textures are freed. */
  clear(): void {
    for (const e of this.entries.values()) this.onEvict(e.value);
    this.entries.clear();
    this.used = 0;
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
  private ringProgram: WebGLProgram;
  private ringBuf: WebGLBuffer;
  private textures = new TextureStore<WebGLTexture>(256 * 1024 * 1024);
  private pending = new Set<string>();
  private zeroTex: WebGLTexture;
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

    // Static 1x1 zero texture so the "probs" sampler is always bound, even
    // when overlay is off or no prob tile has loaded yet (404 pre-detection).
    const zeroTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, zeroTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([0]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.zeroTex = zeroTex;

    // Ring program: a unit quad in [-1, 1]^2, positioned and scaled per ring
    // via uniforms so one draw call handles one ring (ring counts are small
    // -- dozens, not thousands -- so per-ring draw calls are not a concern).
    const rp = gl.createProgram()!;
    gl.attachShader(rp, compile(gl, gl.VERTEX_SHADER, RING_VERT));
    gl.attachShader(rp, compile(gl, gl.FRAGMENT_SHADER, RING_FRAG));
    gl.linkProgram(rp);
    if (!gl.getProgramParameter(rp, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(rp) ?? "ring link failed");
    }
    this.ringProgram = rp;
    this.ringBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.ringBuf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1]),
      gl.STATIC_DRAW,
    );
  }

  /** Fetch a tile via tiles:// and upload it; no-op if cached or in flight.
   * `single` uploads as a single-channel R8 probability texture instead of
   * RGBA; a 404 (no detection yet) is simply not cached, so a later detect
   * can retry the same path (the pending guard clears in finally either way). */
  private ensure(
    path: string,
    expectedW: number,
    expectedH: number,
    opts: { single?: boolean } = {},
  ): WebGLTexture | undefined {
    const hit = this.textures.get(path);
    if (hit) return hit;
    if (!this.pending.has(path)) {
      this.pending.add(path);
      fetch(`tiles://localhost${path}`)
        .then(async (r) => {
          if (!r.ok) return;
          // Prefer the response headers but never depend on them: CORS hides
          // custom headers unless the server exposes them, and a 0x0 upload
          // is an invisible failure. The caller knows the tile geometry.
          const w = Number(r.headers.get("x-tile-width")) || expectedW;
          const h = Number(r.headers.get("x-tile-height")) || expectedH;
          const bytes = new Uint8Array(await r.arrayBuffer());
          if (bytes.length !== w * h * (opts.single ? 1 : 4)) {
            console.error(`tile ${path}: ${bytes.length} bytes for ${w}x${h}`);
            return;
          }
          const gl = this.gl;
          const tex = gl.createTexture()!;
          gl.bindTexture(gl.TEXTURE_2D, tex);
          gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
          if (opts.single) {
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, w, h, 0, gl.RED, gl.UNSIGNED_BYTE, bytes);
          } else {
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
          }
          // Single-channel probability textures use NEAREST min filtering:
          // LINEAR would interpolate probabilities across texels, softening
          // the threshold edge in the shader (thresholding on a blended
          // value instead of the real per-pixel probability).
          gl.texParameteri(
            gl.TEXTURE_2D,
            gl.TEXTURE_MIN_FILTER,
            opts.single ? gl.NEAREST : gl.LINEAR,
          );
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
          this.textures.put(path, tex, w * h * (opts.single ? 1 : 4));
          this.onTileLoaded();
        })
        .finally(() => this.pending.delete(path));
    }
    return undefined;
  }

  /** Draw one frame. tiles come from visibleTiles(), coarse first.
   * overlay.enabled/threshold drive uniforms only — never triggers a fetch;
   * prob textures already hold raw probabilities fetched via ensure(). */
  draw(
    tiles: {
      path: string;
      probPath: string;
      screenX: number;
      screenY: number;
      screenW: number;
      screenH: number;
      tileW: number;
      tileH: number;
    }[],
    canvasW: number,
    canvasH: number,
    overlay: { enabled: boolean; threshold: number },
  ): void {
    const gl = this.gl;
    gl.viewport(0, 0, canvasW, canvasH);
    gl.clearColor(0.15, 0.15, 0.15, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.program);
    gl.uniform2f(gl.getUniformLocation(this.program, "viewport"), canvasW, canvasH);
    gl.uniform1f(gl.getUniformLocation(this.program, "threshold"), overlay.threshold);
    gl.uniform1f(gl.getUniformLocation(this.program, "overlayOn"), overlay.enabled ? 1 : 0);
    gl.uniform1i(gl.getUniformLocation(this.program, "tile"), 0);
    gl.uniform1i(gl.getUniformLocation(this.program, "probs"), 1);
    const posLoc = gl.getAttribLocation(this.program, "pos");
    const uvLoc = gl.getAttribLocation(this.program, "uv");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.enableVertexAttribArray(posLoc);
    gl.enableVertexAttribArray(uvLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 16, 0);
    gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 16, 8);
    for (const t of tiles) {
      const tex = this.ensure(t.path, t.tileW, t.tileH);
      if (!tex) continue;
      // edge tiles are smaller than 512: scale the drawn quad by the real
      // tile fraction so partial tiles are not stretched
      const w = t.screenW * (t.tileW / TILE);
      const h = t.screenH * (t.tileH / TILE);
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
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      const probTex = overlay.enabled
        ? this.ensure(t.probPath, t.tileW, t.tileH, { single: true })
        : undefined;
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, probTex ?? this.zeroTex);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
  }

  /** Draws ring markers over the already-rendered frame. Call after draw().
   * `rings` are in screen px (see viewport.ts#ringsFor). Uses additive-free
   * alpha blending so overlapping rings don't double-darken past the base
   * 0.9 alpha set in the fragment shader. Blending is disabled again before
   * returning so the next tile pass (which does not itself touch blend
   * state) renders opaquely as before. */
  drawRings(rings: { x: number; y: number; r: number }[], canvasW: number, canvasH: number): void {
    if (rings.length === 0) return;
    const gl = this.gl;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.ringProgram);
    gl.uniform2f(gl.getUniformLocation(this.ringProgram, "viewport"), canvasW, canvasH);
    const centerLoc = gl.getUniformLocation(this.ringProgram, "center");
    const radiusLoc = gl.getUniformLocation(this.ringProgram, "radius");
    const cornerLoc = gl.getAttribLocation(this.ringProgram, "corner");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.ringBuf);
    gl.enableVertexAttribArray(cornerLoc);
    gl.vertexAttribPointer(cornerLoc, 2, gl.FLOAT, false, 8, 0);
    for (const ring of rings) {
      gl.uniform2f(centerLoc, ring.x, ring.y);
      gl.uniform1f(radiusLoc, ring.r);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    gl.disable(gl.BLEND);
  }

  /** Release every GL resource and force the context to be dropped. The
   * Viewer is remounted per frame switch via `{#key info.id}`; without this,
   * each remount leaks a WebGL context (WebKit caps live contexts at ~16),
   * and once the cap is hit new contexts come back lost -- a blank canvas,
   * then a webview crash on the next remount. */
  dispose(): void {
    const gl = this.gl;
    this.textures.clear();
    gl.deleteTexture(this.zeroTex);
    gl.deleteBuffer(this.buf);
    gl.deleteBuffer(this.ringBuf);
    gl.deleteProgram(this.program);
    gl.deleteProgram(this.ringProgram);
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
}
