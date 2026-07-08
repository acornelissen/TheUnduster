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
        ? this.ensure(probPathFor(t.path), t.tileW, t.tileH, { single: true })
        : undefined;
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, probTex ?? this.zeroTex);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
  }
}
