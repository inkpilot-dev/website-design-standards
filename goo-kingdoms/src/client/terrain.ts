/**
 * Smooth island terrain for the 3D view.
 *
 * Every tile has a base height. The ground between tile centres is blended
 * with a plateau curve, so each tile keeps a flat middle (where cities and
 * buildings stand) and slopes softly into its neighbours. Beaches fall gently
 * into the shallows, hills roll, and the sea floor drops away from the coast.
 * Small noise keeps it from looking machine-made.
 */
import * as THREE from 'three';
import type { Terrain, WorldMap } from '../shared/types.ts';
import { idx, inBounds, isWater } from '../shared/geo.ts';

/** Extra tiles of scenery around the 20x20 board. */
export const MARGIN = 5;
/** Terrain vertices per tile: finer on small maps, coarser on big ones. */
const resFor = (size: number) => (size <= 24 ? 8 : size <= 36 ? 6 : size <= 52 ? 5 : 4);

const BASE: Record<Terrain, number> = { sea: -0.6, shallows: -0.16, sand: 0.07, grass: 0.17, forest: 0.19, hills: 0.32, mountain: 0.42 };

function hash2(x: number, y: number): number {
  let n = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263)) | 0;
  n = Math.imul(n ^ (n >>> 13), 1274126177);
  return ((n ^ (n >>> 16)) >>> 0) / 4294967296;
}

/** Smooth value noise in [0, 1). */
export function noise(x: number, y: number): number {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = x - x0;
  const ty = y - y0;
  const sx = tx * tx * (3 - 2 * tx);
  const sy = ty * ty * (3 - 2 * ty);
  const a = hash2(x0, y0) + (hash2(x0 + 1, y0) - hash2(x0, y0)) * sx;
  const b = hash2(x0, y0 + 1) + (hash2(x0 + 1, y0 + 1) - hash2(x0, y0 + 1)) * sx;
  return a + (b - a) * sy;
}

const plateau = (t: number) => {
  const k = Math.max(0, Math.min(1, (t - 0.28) / 0.44));
  return k * k * (3 - 2 * k);
};

export class Heightfield {
  readonly map: WorldMap;
  private base: Float32Array;
  private col: Float32Array;
  private kinds: (Terrain | null)[];
  readonly W: number;
  readonly H: number;

  constructor(map: WorldMap) {
    this.map = map;
    this.W = map.width;
    this.H = map.height;
    const n = this.W * this.H;
    this.base = new Float32Array(n);
    this.col = new Float32Array(n * 3);
    this.kinds = new Array(n).fill(null);
    const c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      const t = map.tiles[i].terrain;
      this.kinds[i] = t;
      this.base[i] = t === 'sea' ? -0.25 - map.depth[i] * 0.85 : BASE[t];
      const x = i % this.W;
      const y = Math.floor(i / this.W);
      const v = (noise(x * 0.7, y * 0.7) - 0.5) * 0.06;
      switch (t) {
        case 'grass': c.setHex(0x86b84f).offsetHSL(v * 0.4, 0, v); break;
        case 'forest': c.setHex(0x5f9440).offsetHSL(0, 0, v); break;
        case 'sand': c.setHex(0xe9d9a6).offsetHSL(0, 0, v * 0.5); break;
        case 'hills': c.setHex(0x9fb062).offsetHSL(0, -0.05, v); break;
        case 'mountain': c.setHex(0x9a958b).offsetHSL(0, 0, v); break;
        default: c.setHex(0xd9c89a);
      }
      this.col[i * 3] = c.r;
      this.col[i * 3 + 1] = c.g;
      this.col[i * 3 + 2] = c.b;
    }
  }

  private baseAt(x: number, y: number): number {
    if (x < 0 || y < 0 || x >= this.W || y >= this.H) {
      const d = Math.max(-x, -y, x - this.W + 1, y - this.H + 1);
      return -0.95 - d * 0.04;
    }
    return this.base[y * this.W + x];
  }

  private colAt(x: number, y: number, out: number[]): void {
    const cx = Math.max(0, Math.min(this.W - 1, x));
    const cy = Math.max(0, Math.min(this.H - 1, y));
    const i = (cy * this.W + cx) * 3;
    out[0] = this.col[i];
    out[1] = this.col[i + 1];
    out[2] = this.col[i + 2];
  }

  /** Ground height at a world position (x east, z south; tile centres at integers offset by the board centre). */
  heightAt(wx: number, wz: number): number {
    const u = wx + this.W / 2 - 0.5;
    const v = wz + this.H / 2 - 0.5;
    const x0 = Math.floor(u);
    const y0 = Math.floor(v);
    const h00 = this.baseAt(x0, y0);
    const h10 = this.baseAt(x0 + 1, y0);
    const h01 = this.baseAt(x0, y0 + 1);
    const h11 = this.baseAt(x0 + 1, y0 + 1);
    // Land keeps flat tile middles; open water blends smoothly so the seabed
    // has natural curves instead of square shelves.
    const allWater = h00 < 0 && h10 < 0 && h01 < 0 && h11 < 0;
    const ease = (t: number) => t * t * (3 - 2 * t);
    const sx = allWater ? ease(u - x0) : plateau(u - x0);
    const sy = allWater ? ease(v - y0) : plateau(v - y0);
    const a = h00 + (h10 - h00) * sx;
    const b = h01 + (h11 - h01) * sx;
    let h = a + (b - a) * sy;
    if (allWater) h += (noise(wx * 0.9 + 11, wz * 0.9) - 0.5) * 0.22 * Math.min(1, -h * 2);
    // Where land meets water, wobble the shoreline so beaches don't follow the grid.
    const mixed = !allWater && (h00 < 0 || h10 < 0 || h01 < 0 || h11 < 0);
    if (mixed) h += (noise(wx * 1.7 + 5, wz * 1.7 + 9) - 0.5) * 0.2;
    // Gentle unevenness everywhere, rolling bumps on hills.
    const tx = Math.round(u);
    const ty = Math.round(v);
    const kind = inBounds(this.map, tx, ty) ? this.kinds[ty * this.W + tx] : null;
    h += (noise(wx * 2.1, wz * 2.1) - 0.5) * 0.035;
    if (kind === 'hills') h += (noise(wx * 3.3 + 7, wz * 3.3) - 0.35) * 0.14;
    if (kind === 'mountain') h += (noise(wx * 2.6 + 3, wz * 2.6) - 0.3) * 0.2;
    return h;
  }

  /** Painted ground colour at a world position, given its height. */
  colorAt(wx: number, wz: number, h: number, out: THREE.Color): void {
    const u = wx + this.W / 2 - 0.5;
    const v = wz + this.H / 2 - 0.5;
    const x0 = Math.floor(u);
    const y0 = Math.floor(v);
    const sx = plateau(u - x0);
    const sy = plateau(v - y0);
    const c00 = [0, 0, 0], c10 = [0, 0, 0], c01 = [0, 0, 0], c11 = [0, 0, 0];
    this.colAt(x0, y0, c00);
    this.colAt(x0 + 1, y0, c10);
    this.colAt(x0, y0 + 1, c01);
    this.colAt(x0 + 1, y0 + 1, c11);
    const mix = (k: number) => {
      const a = c00[k] + (c10[k] - c00[k]) * sx;
      const b = c01[k] + (c11[k] - c01[k]) * sx;
      return a + (b - a) * sy;
    };
    out.setRGB(mix(0), mix(1), mix(2));
    // Sand along every shore, wet sand at the waterline, seabed below.
    const sand = new THREE.Color(0xeadcab);
    if (h < 0.11) out.lerp(sand, Math.min(1, (0.11 - h) / 0.08));
    if (h < 0.025) out.lerp(new THREE.Color(0xcdb98a), Math.min(1, (0.025 - h) / 0.04));
    if (h < -0.04) out.lerp(new THREE.Color(0x2f6f7c), Math.min(1, (-0.04 - h) / 0.7));
    // Grain.
    const g = (noise(wx * 9, wz * 9) - 0.5) * 0.07;
    out.offsetHSL(0, 0, g);
  }

  /** Board tile under a world position, or null if off the board. */
  tileAt(wx: number, wz: number): [number, number] | null {
    const x = Math.floor(wx + this.W / 2);
    const y = Math.floor(wz + this.H / 2);
    return inBounds(this.map, x, y) ? [x, y] : null;
  }

  isWaterTile(x: number, y: number): boolean {
    return !inBounds(this.map, x, y) || isWater(this.map.tiles[idx(this.map, x, y)].terrain);
  }
}

/** The ground mesh, with base colours kept for territory tinting. */
export function buildGround(hf: Heightfield): { mesh: THREE.Mesh; base: Float32Array } {
  const ext = hf.W + MARGIN * 2;
  const seg = ext * resFor(hf.W);
  const geo = new THREE.PlaneGeometry(ext, ext, seg, seg);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const colors = new Float32Array(pos.count * 3);
  const c = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);
    const h = hf.heightAt(x, z);
    pos.setY(i, h);
    hf.colorAt(x, z, h, c);
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors.slice(), 3));
  geo.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, metalness: 0 });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.receiveShadow = true;
  return { mesh, base: colors };
}

/** A translucent sea coloured by depth: foam at the shore, turquoise shallows, deep blue beyond. */
export function buildWater(hf: Heightfield, time: { value: number }): THREE.Mesh {
  const ext = hf.W + MARGIN * 2;
  const seg = ext * (hf.W <= 36 ? 5 : 3);
  const geo = new THREE.PlaneGeometry(ext, ext, seg, seg);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const colors = new Float32Array(pos.count * 4);
  const foam = new THREE.Color(0xffffff);
  const shallow = new THREE.Color(0x7fdcd2);
  const mid = new THREE.Color(0x35a6c4);
  const deep = new THREE.Color(0x1d6796);
  const c = new THREE.Color();
  const foamAttr = new Float32Array(pos.count);
  for (let i = 0; i < pos.count; i++) {
    // The plane is rotated flat later: local y becomes world -z.
    const d = -hf.heightAt(pos.getX(i), -pos.getY(i));
    let a: number;
    if (d < 0.035) {
      c.copy(foam);
      a = 0.85;
      foamAttr[i] = 1;
    } else if (d < 0.3) {
      const t = (d - 0.035) / 0.265;
      c.copy(shallow).lerp(mid, t);
      a = 0.38 + t * 0.3;
      foamAttr[i] = Math.max(0, 1 - (d - 0.035) / 0.06);
    } else {
      const t = Math.min(1, (d - 0.3) / 0.7);
      c.copy(mid).lerp(deep, t);
      a = 0.68 + t * 0.3;
    }
    colors[i * 4] = c.r;
    colors[i * 4 + 1] = c.g;
    colors[i * 4 + 2] = c.b;
    colors[i * 4 + 3] = a;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 4));
  geo.setAttribute('foam', new THREE.BufferAttribute(foamAttr, 1));
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, transparent: true, roughness: 0.18, metalness: 0.05, depthWrite: false });
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = time;
    shader.vertexShader = 'uniform float uTime;\nattribute float foam;\nvarying float vFoam;\nvarying vec2 vWorld;\n' + shader.vertexShader.replace(
      '#include <begin_vertex>',
      `#include <begin_vertex>
       transformed.z += sin(position.x * 1.5 + uTime * 1.1) * 0.018 + cos(position.y * 1.8 + uTime * 0.8) * 0.018;
       vFoam = foam;
       vWorld = position.xy;`,
    );
    shader.fragmentShader = 'uniform float uTime;\nvarying float vFoam;\nvarying vec2 vWorld;\n' + shader.fragmentShader.replace(
      '#include <color_fragment>',
      `#include <color_fragment>
       float pulse = 0.55 + 0.45 * sin(uTime * 1.7 + vWorld.x * 2.3 + vWorld.y * 1.9);
       diffuseColor.rgb = mix(diffuseColor.rgb, vec3(1.0), vFoam * pulse * 0.6);
       diffuseColor.a = mix(diffuseColor.a, 0.9, vFoam * pulse * 0.5);`,
    );
  };
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 0.0;
  mesh.receiveShadow = true;
  mesh.renderOrder = 1;
  return mesh;
}

/** The open sea beyond the scenery margin, with a hole where the local water sits. */
export function buildFarSea(hf: Heightfield): THREE.Mesh {
  const half = (hf.W + MARGIN * 2) / 2 - 0.02;
  const shape = new THREE.Shape();
  shape.moveTo(-600, -600);
  shape.lineTo(600, -600);
  shape.lineTo(600, 600);
  shape.lineTo(-600, 600);
  shape.lineTo(-600, -600);
  const hole = new THREE.Path();
  hole.moveTo(-half, -half);
  hole.lineTo(-half, half);
  hole.lineTo(half, half);
  hole.lineTo(half, -half);
  hole.lineTo(-half, -half);
  shape.holes.push(hole);
  const mesh = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshStandardMaterial({ color: 0x1f6a99, roughness: 0.25 }));
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = -0.06;
  return mesh;
}

/** Fog-of-war shroud: a dark veil hovering over the ground with per-vertex darkness. */
export function buildShroud(hf: Heightfield): THREE.Mesh {
  const seg = hf.W <= 36 ? 4 : 2;
  const geo = new THREE.PlaneGeometry(hf.W, hf.H, hf.W * seg, hf.H * seg);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < pos.count; i++) pos.setY(i, Math.max(0, hf.heightAt(pos.getX(i), pos.getZ(i))) + 0.75);
  geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(pos.count * 4), 4));
  const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false }));
  mesh.renderOrder = 6;
  mesh.visible = false;
  return mesh;
}

/** Set shroud darkness from per-tile values (0 = clear, 1 = black), blended smoothly. */
export function paintShroud(mesh: THREE.Mesh, hf: Heightfield, value: (x: number, y: number) => number): void {
  const pos = mesh.geometry.attributes.position as THREE.BufferAttribute;
  const col = mesh.geometry.attributes.color as THREE.BufferAttribute;
  const at = (x: number, y: number) => value(Math.max(0, Math.min(hf.W - 1, x)), Math.max(0, Math.min(hf.H - 1, y)));
  for (let i = 0; i < pos.count; i++) {
    const u = pos.getX(i) + hf.W / 2 - 0.5;
    const v = pos.getZ(i) + hf.H / 2 - 0.5;
    const x0 = Math.floor(u);
    const y0 = Math.floor(v);
    const sx = plateau(u - x0);
    const sy = plateau(v - y0);
    const a = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * sx;
    const b = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * sx;
    col.setXYZW(i, 0.05, 0.07, 0.09, a + (b - a) * sy);
  }
  col.needsUpdate = true;
}
