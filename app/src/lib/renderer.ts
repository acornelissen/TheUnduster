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
