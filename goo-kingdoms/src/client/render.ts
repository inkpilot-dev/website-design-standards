/**
 * 3D renderer for the island, built with Three.js.
 *
 * The world is a low-poly diorama: terrain columns, a seabed under a moving,
 * translucent sea, mountains, swaying forests, clouds that cast real shadows,
 * cities with faction capitals and waving banners, developments, loot, the
 * marsh boss and the blobs themselves. When a new day arrives, its events
 * play back as animation: hops along paths with squash and stretch, lunges,
 * arrows and cannonballs, damage numbers, deaths, level-up rings and more.
 *
 * Camera: drag to orbit, right-drag (or Shift-drag) to pan, wheel to zoom,
 * WASD to fly over the ground, Q/E to sink and rise, F to follow the
 * selected unit, Home to reset.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import type { Blob, City, Development, Frame, GameEvent, TeamId, Terrain, WorldMap } from '../shared/types.ts';
import { FACTIONS } from '../agents/factions.ts';
import { computeTerritory, exploredSet, idx, inBounds, isWater, visibleTiles, dist } from '../shared/geo.ts';
import { HEARING_RANGE } from '../shared/rules.ts';
import { esc } from './util.ts';
import { Heightfield, buildFarSea, buildGround, buildShroud, buildWater, paintShroud } from './terrain.ts';

export type View = { kind: 'god' } | { kind: 'team'; team: TeamId } | { kind: 'blob'; id: string };
export type Selection =
  | { kind: 'blob'; id: string }
  | { kind: 'city'; id: string }
  | { kind: 'dev'; x: number; y: number }
  | { kind: 'boss' }
  | { kind: 'loot'; id: string }
  | { kind: 'tile'; x: number; y: number }
  | null;

// ---------------------------------------------------------------------------
// Constants and small helpers
// ---------------------------------------------------------------------------

/** Height of the top surface of each terrain column. */
const TOP: Record<Terrain, number> = { sea: -0.5, shallows: -0.18, sand: 0.16, grass: 0.24, forest: 0.26, hills: 0.42, mountain: 0.46 };
const LAND_COLOR: Record<string, number> = { sand: 0xe6d4a0, grass: 0x8db65a, forest: 0x6a9a4a, hills: 0xb0a571, mountain: 0x9a968a };
const WATER_Y = 0.02;
const NEUTRAL_ROOF = 0x8a5a3c;

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const clamp01 = (t: number) => Math.max(0, Math.min(1, t));
const ease = (t: number) => t * t * (3 - 2 * t);

function hash(x: number, y: number, k = 0): number {
  let n = (x * 374761393 + y * 668265263 + k * 2147483647) | 0;
  n = Math.imul(n ^ (n >>> 13), 1274126177);
  return ((n ^ (n >>> 16)) >>> 0) / 4294967296;
}

function std(color: number, extra: Partial<THREE.MeshStandardMaterialParameters> = {}): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.85, metalness: 0, flatShading: true, ...extra });
}

/** Adds a time-driven sway to a material (trees, crops, flags, reeds). */
function addSway(mat: THREE.Material, time: { value: number }, code: string): void {
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = time;
    shader.vertexShader = 'uniform float uTime;\n' + shader.vertexShader.replace('#include <begin_vertex>', `#include <begin_vertex>\n${code}`);
  };
}

function shadowed<T extends THREE.Object3D>(o: T, cast = true, receive = true): T {
  o.traverse((c) => {
    if ((c as THREE.Mesh).isMesh) {
      c.castShadow = cast;
      c.receiveShadow = receive;
    }
  });
  return o;
}

interface Particle {
  mesh: THREE.Mesh;
  vel: THREE.Vector3;
  life: number;
  max: number;
  grow: number;
  gravity: number;
}

interface FloatText { el: HTMLDivElement; pos: THREE.Vector3; life: number; max: number }

interface BlobView {
  root: THREE.Group;
  body: THREE.Group;
  bodyMesh: THREE.Mesh;
  gear: THREE.Group;
  bow: THREE.Group | null;
  boat: THREE.Group | null;
  rank: THREE.Object3D | null;
  sig: string;
  hp: HTMLDivElement;
  facing: number;
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

export class Renderer {
  readonly canvas: HTMLCanvasElement;
  private overlay: HTMLElement;
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private time = { value: 0 };
  private clock = new THREE.Clock();

  map: WorldMap | null = null;
  frame: Frame | null = null;
  private prev: Frame | null = null;
  private animStart = 0;
  private animDur = 700;
  view: View = { kind: 'god' };
  showTerritory = true;
  showSpeech = true;
  showCoords = false;
  selection: Selection = null;
  follow = false;

  private world = new THREE.Group();
  private terrainGroup = new THREE.Group();
  private dynamicGroup = new THREE.Group();
  private hf: Heightfield | null = null;
  private ground: THREE.Mesh | null = null;
  private groundBase: Float32Array | null = null;
  private water: THREE.Mesh | null = null;
  private farSea: THREE.Mesh | null = null;
  private shroud: THREE.Mesh | null = null;
  private territoryLines: THREE.LineSegments | null = null;
  private territorySig = '';
  private hearing: THREE.LineLoop | null = null;
  private grid: THREE.Group | null = null;
  private selectRing: THREE.Mesh;
  private clouds: THREE.Group[] = [];

  private cityViews = new Map<string, { group: THREE.Group; sig: string; label: HTMLDivElement; ring: THREE.Mesh | null; flags: THREE.Mesh[] }>();
  private devViews = new Map<string, THREE.Group>();
  private lootViews = new Map<string, THREE.Group>();
  private blobViews = new Map<string, BlobView>();
  private ghosts: { view: BlobView; start: number }[] = [];
  private boss: { group: THREE.Group; label: HTMLDivElement; hp: HTMLDivElement } | null = null;
  private projectiles: { mesh: THREE.Object3D; from: THREE.Vector3; to: THREE.Vector3; arc: number; dmg: number; done: boolean }[] = [];
  private particles: Particle[] = [];
  private floats: FloatText[] = [];
  private bubbles: HTMLDivElement[] = [];
  private keys = new Set<string>();
  private flagMat = new Map<number, THREE.MeshStandardMaterial>();
  private smokeTimer = 0;

  // Shared geometry and materials.
  private geo = {
    box: new THREE.BoxGeometry(1, 1, 1),
    blob: new THREE.SphereGeometry(0.5, 20, 14),
    eye: new THREE.SphereGeometry(1, 10, 8),
    cone4: new THREE.ConeGeometry(1, 1, 4),
    cyl: new THREE.CylinderGeometry(1, 1, 1, 10),
    sphereLow: new THREE.SphereGeometry(1, 8, 6),
    gable: (() => {
      const sh = new THREE.Shape();
      sh.moveTo(-0.5, 0);
      sh.lineTo(0.5, 0);
      sh.lineTo(0, 0.62);
      sh.lineTo(-0.5, 0);
      const g = new THREE.ExtrudeGeometry(sh, { depth: 1, bevelEnabled: false });
      g.translate(0, 0, -0.5);
      return g;
    })(),
    flag: (() => {
      const g = new THREE.PlaneGeometry(0.26, 0.15, 6, 1);
      g.translate(0.13, 0, 0);
      return g;
    })(),
  };
  private mats = {
    white: std(0xffffff, { flatShading: false, roughness: 0.3 }),
    pupil: std(0x1d1b19, { flatShading: false, roughness: 0.2 }),
    wall: std(0xefe4cc),
    wallShade: std(0xd7c8a8),
    stone: std(0xc9c2b2),
    stoneDark: std(0x9f9786),
    wood: std(0x7a5634),
    woodDark: std(0x5a4030),
    metal: std(0xcfd3d6, { metalness: 0.5, roughness: 0.4 }),
    gold: std(0xf0c84e, { metalness: 0.6, roughness: 0.35 }),
    silver: std(0xdfe6ea, { metalness: 0.6, roughness: 0.3 }),
    leaf: std(0x7f8f45),
    pit: std(0x3a2f22),
    crop: std(0xdcc35a),
    soil: std(0xa9874f),
    rope: new THREE.LineBasicMaterial({ color: 0xefe8d8 }),
  };

  constructor(canvas: HTMLCanvasElement, overlay: HTMLElement) {
    this.canvas = canvas;
    this.overlay = overlay;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 300);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = false;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 70;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.keys = { LEFT: '', UP: '', RIGHT: '', BOTTOM: '' };
    this.resetCamera();

    this.buildSky();
    this.scene.add(this.world);
    this.world.add(this.terrainGroup, this.dynamicGroup);

    this.selectRing = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.03, 8, 40), new THREE.MeshBasicMaterial({ color: 0xfff3c8 }));
    this.selectRing.rotation.x = -Math.PI / 2;
    this.selectRing.visible = false;
    this.scene.add(this.selectRing);

    // Keyboard flying (only when the page, not a form field, has focus).
    window.addEventListener('keydown', (e) => {
      const t = e.target as HTMLElement;
      if (t.closest('input, select, textarea, dialog')) return;
      const k = e.key.toLowerCase();
      if (['w', 'a', 's', 'd', 'q', 'e', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(k)) {
        this.keys.add(k);
        this.follow = false;
        e.preventDefault();
      }
      if (k === 'f') this.follow = !this.follow;
      if (k === 'home') this.resetCamera();
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.key.toLowerCase()));
    window.addEventListener('blur', () => this.keys.clear());

    new ResizeObserver(() => this.resize()).observe(canvas.parentElement!);
    this.resize();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  // -------------------------------------------------------------------------
  // Camera
  // -------------------------------------------------------------------------

  resetCamera(): void {
    this.follow = false;
    this.camera.position.set(-4, 17, 17);
    this.controls.target.set(0, 0, 1);
    this.controls.update();
  }

  focusSelected(): void {
    const p = this.selectionPosition();
    if (!p) return;
    const offset = this.camera.position.clone().sub(this.controls.target);
    offset.setLength(Math.min(offset.length(), 8));
    this.controls.target.copy(p);
    this.camera.position.copy(p).add(offset);
    this.follow = true;
  }

  private flyCamera(dt: number): void {
    if (this.keys.size === 0) return;
    const forward = new THREE.Vector3();
    this.camera.getWorldDirection(forward);
    forward.y = 0;
    forward.normalize();
    const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();
    const move = new THREE.Vector3();
    const k = this.keys;
    if (k.has('w') || k.has('arrowup')) move.add(forward);
    if (k.has('s') || k.has('arrowdown')) move.sub(forward);
    if (k.has('d') || k.has('arrowright')) move.add(right);
    if (k.has('a') || k.has('arrowleft')) move.sub(right);
    if (k.has('e')) move.y += 1;
    if (k.has('q')) move.y -= 1;
    if (move.lengthSq() === 0) return;
    const speed = 4 + this.camera.position.distanceTo(this.controls.target) * 0.6;
    move.normalize().multiplyScalar(speed * dt);
    this.camera.position.add(move);
    this.controls.target.add(move);
    if (this.camera.position.y < 0.4) {
      const d = 0.4 - this.camera.position.y;
      this.camera.position.y += d;
      this.controls.target.y += d;
    }
  }

  private resize(): void {
    const parent = this.canvas.parentElement!;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // -------------------------------------------------------------------------
  // World coordinates
  // -------------------------------------------------------------------------

  private wx(x: number): number {
    return x - (this.map ? this.map.width : 20) / 2 + 0.5;
  }

  private wz(y: number): number {
    return y - (this.map ? this.map.height : 20) / 2 + 0.5;
  }

  /** Height a unit stands at on a tile: the ground, or the sea surface. */
  private top(x: number, y: number): number {
    if (!this.hf || !this.map || !inBounds(this.map, x, y)) return WATER_Y;
    if (isWater(this.map.tiles[idx(this.map, x, y)].terrain)) return WATER_Y;
    return Math.max(WATER_Y, this.hf.heightAt(this.wx(x), this.wz(y)));
  }

  /** Ground height under a world position. */
  private h(wx: number, wz: number): number {
    return this.hf ? this.hf.heightAt(wx, wz) : 0;
  }

  /** Ground height under a fractional tile position (for hops between tiles). */
  private groundAt(fx: number, fy: number): number {
    return Math.max(WATER_Y, this.h(this.wx(fx), this.wz(fy)));
  }

  // -------------------------------------------------------------------------
  // Sky, sea, light
  // -------------------------------------------------------------------------

  private buildSky(): void {
    const skyGeo = new THREE.SphereGeometry(150, 32, 16);
    const colors: number[] = [];
    const top = new THREE.Color(0x5d9ad0);
    const horizon = new THREE.Color(0xd9e9ef);
    const pos = skyGeo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const t = clamp01(pos.getY(i) / 150);
      const c = horizon.clone().lerp(top, Math.pow(t, 0.6));
      colors.push(c.r, c.g, c.b);
    }
    skyGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const sky = new THREE.Mesh(skyGeo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide, fog: false }));
    this.scene.add(sky);
    this.scene.fog = new THREE.Fog(0xcfe2ea, 35, 110);

    this.scene.add(new THREE.HemisphereLight(0xeaf5ff, 0x71824a, 1.75));
    const sun = new THREE.DirectionalLight(0xffefd2, 1.9);
    sun.position.set(-12, 24, -8);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera;
    sc.left = -17; sc.right = 17; sc.top = 17; sc.bottom = -17; sc.near = 1; sc.far = 60;
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.03;
    sun.shadow.radius = 3;
    this.scene.add(sun);
    this.scene.add(sun.target);

    // Clouds drift high above the island. Each one trails a soft patch of
    // shade across the ground below it.
    const cloudMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, transparent: true, opacity: 0.85, flatShading: true, depthWrite: false });
    const shadeTex = (() => {
      const cv = document.createElement('canvas');
      cv.width = 128;
      cv.height = 128;
      const g = cv.getContext('2d')!;
      const grad = g.createRadialGradient(64, 64, 4, 64, 64, 62);
      grad.addColorStop(0, 'rgba(10,25,20,0.55)');
      grad.addColorStop(0.6, 'rgba(10,25,20,0.3)');
      grad.addColorStop(1, 'rgba(10,25,20,0)');
      g.fillStyle = grad;
      g.fillRect(0, 0, 128, 128);
      const tex = new THREE.CanvasTexture(cv);
      tex.colorSpace = THREE.SRGBColorSpace;
      return tex;
    })();
    const shadeGeo = new THREE.PlaneGeometry(1, 1);
    shadeGeo.rotateX(-Math.PI / 2);
    for (let i = 0; i < 6; i++) {
      const g = new THREE.Group();
      const n = 4 + Math.floor(Math.random() * 4);
      for (let k = 0; k < n; k++) {
        const s = 0.8 + Math.random() * 1.0;
        const m = new THREE.Mesh(this.geo.sphereLow, cloudMat);
        m.scale.set(s * 1.3, s * 0.7, s);
        m.position.set((Math.random() - 0.5) * 3, Math.random() * 0.4, (Math.random() - 0.5) * 1.6);
        g.add(m);
      }
      g.position.set(Math.random() * 40 - 20, 21 + Math.random() * 4, Math.random() * 30 - 15);
      g.userData.speed = 0.25 + Math.random() * 0.25;
      const shade = new THREE.Mesh(shadeGeo, new THREE.MeshBasicMaterial({ map: shadeTex, transparent: true, depthWrite: false, opacity: 0.55 }));
      shade.scale.set(5 + Math.random() * 2, 1, 3.5 + Math.random() * 1.5);
      shade.renderOrder = 4;
      g.userData.shade = shade;
      this.clouds.push(g);
      this.scene.add(g, shade);
    }
  }

  // -------------------------------------------------------------------------
  // Map (static terrain)
  // -------------------------------------------------------------------------

  setMap(map: WorldMap): void {
    this.map = map;
    for (const v of this.cityViews.values()) v.label.remove();
    for (const v of this.blobViews.values()) v.hp.remove();
    this.boss?.label.remove();
    this.boss?.hp.remove();
    this.cityViews.clear();
    this.devViews.clear();
    this.lootViews.clear();
    this.blobViews.clear();
    this.ghosts = [];
    this.boss = null;
    this.terrainGroup.clear();
    this.dynamicGroup.clear();
    if (this.farSea) this.scene.remove(this.farSea);
    this.frame = null;
    this.prev = null;
    this.territorySig = '';
    this.lastFogKey = '';

    const hf = new Heightfield(map);
    this.hf = hf;
    const ground = buildGround(hf);
    this.ground = ground.mesh;
    this.groundBase = ground.base;
    this.terrainGroup.add(ground.mesh);
    this.water = buildWater(hf, this.time);
    this.terrainGroup.add(this.water);
    this.farSea = buildFarSea(hf);
    this.scene.add(this.farSea);
    this.shroud = buildShroud(hf);
    this.terrainGroup.add(this.shroud);
    this.buildDecor(map);
    this.buildGrid(map);
  }

  /** A jagged low-poly rock: an icosahedron with every corner nudged. */
  private rockGeometry(seed: number): THREE.BufferGeometry {
    const g = new THREE.IcosahedronGeometry(1, 0);
    const pos = g.attributes.position as THREE.BufferAttribute;
    const moved = new Map<string, [number, number, number]>();
    for (let i = 0; i < pos.count; i++) {
      const key = `${pos.getX(i).toFixed(3)},${pos.getY(i).toFixed(3)},${pos.getZ(i).toFixed(3)}`;
      if (!moved.has(key)) {
        const k = moved.size;
        moved.set(key, [
          pos.getX(i) * (0.8 + hash(seed, k, 1) * 0.4),
          pos.getY(i) * (0.8 + hash(seed, k, 2) * 0.4),
          pos.getZ(i) * (0.8 + hash(seed, k, 3) * 0.4),
        ]);
      }
      const [x, y, z] = moved.get(key)!;
      pos.setXYZ(i, x, y, z);
    }
    g.translate(0, 0.55, 0);
    g.computeVertexNormals();
    return g;
  }

  /** Trees, grass tufts, hill boulders, mountain rocks, fruit bushes and crystals. */
  private buildDecor(map: WorldMap): void {
    const W = map.width;
    const m4 = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const c = new THREE.Color();
    const up = new THREE.Vector3(0, 1, 0);

    // Pines: a trunk and three tiers of foliage, gently swaying in gusts.
    const trunkGeo = new THREE.CylinderGeometry(0.03, 0.045, 0.2, 6);
    trunkGeo.translate(0, 0.1, 0);
    const tiers = [new THREE.ConeGeometry(0.19, 0.3, 8), new THREE.ConeGeometry(0.15, 0.27, 8), new THREE.ConeGeometry(0.1, 0.23, 8)];
    tiers[0].translate(0, 0.3, 0);
    tiers[1].translate(0, 0.45, 0);
    tiers[2].translate(0, 0.6, 0);
    const foliageGeo = mergeGeometries(tiers);

    const spots: { x: number; y: number; s: number; r: number }[] = [];
    const tufts: { x: number; z: number; s: number; r: number }[] = [];
    const boulders: { x: number; z: number; s: number; h: number; r: number }[] = [];
    const peaks: { x: number; z: number; s: number; h: number; r: number }[] = [];
    const bushes: { x: number; z: number }[] = [];
    const crystals: { x: number; z: number; r: number }[] = [];
    for (let y = 0; y < map.height; y++) {
      for (let x = 0; x < W; x++) {
        const t = map.tiles[idx(map, x, y)];
        const cx = this.wx(x);
        const cz = this.wz(y);
        const jit = (k: number) => (hash(x, y, k) - 0.5) * 0.8;
        if (t.terrain === 'forest') {
          const n = 4 + Math.floor(hash(x, y, 5) * 3);
          for (let k = 0; k < n; k++) spots.push({ x: cx + jit(70 + k), y: cz + jit(80 + k), s: 0.7 + hash(x, y, 90 + k) * 0.6, r: hash(x, y, k) * 6 });
        } else if (t.terrain === 'grass') {
          if (hash(x, y, 7) < 0.12) spots.push({ x: cx + jit(71), y: cz + jit(81), s: 0.75, r: 0 });
          for (let k = 0; k < 5; k++) tufts.push({ x: cx + jit(300 + k) * 1.1, z: cz + jit(310 + k) * 1.1, s: 0.6 + hash(x, y, 320 + k) * 0.6, r: hash(x, y, 330 + k) * 6 });
        } else if (t.terrain === 'hills') {
          for (let k = 0; k < 3; k++) boulders.push({ x: cx + jit(50 + k), z: cz + jit(60 + k), s: 0.09 + hash(x, y, 55 + k) * 0.1, h: 0.7 + hash(x, y, 56 + k) * 0.5, r: hash(x, y, 57 + k) * 6 });
          for (let k = 0; k < 3; k++) tufts.push({ x: cx + jit(340 + k), z: cz + jit(350 + k), s: 0.6, r: hash(x, y, 360 + k) * 6 });
        } else if (t.terrain === 'mountain') {
          peaks.push({ x: cx + jit(4) * 0.25, z: cz + jit(5) * 0.25, s: 0.4 + hash(x, y, 3) * 0.1, h: 2.1 + hash(x, y, 6) * 1.0, r: hash(x, y, 8) * 6 });
          for (let k = 0; k < 3; k++) peaks.push({ x: cx + jit(10 + k), z: cz + jit(20 + k), s: 0.2 + hash(x, y, 30 + k) * 0.1, h: 1.4 + hash(x, y, 40 + k) * 1.0, r: hash(x, y, 50 + k) * 6 });
        } else if (t.terrain === 'sand' && hash(x, y, 9) < 0.25) {
          boulders.push({ x: cx + jit(91), z: cz + jit(92), s: 0.06, h: 0.6, r: hash(x, y, 93) * 6 });
        }
        if (t.resource === 'fruit') for (let k = 0; k < 2; k++) bushes.push({ x: cx + jit(30 + k) * 0.6, z: cz + jit(40 + k) * 0.6 });
        if (t.resource === 'crystal') for (let k = 0; k < 3; k++) crystals.push({ x: cx + 0.15 + (k - 1) * 0.08, z: cz + 0.2, r: k * 0.6 - 0.6 });
      }
    }
    this.treeSpots = spots;

    const foliageMat = std(0xffffff);
    addSway(foliageMat, this.time, `
      vec4 wp = instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
      float gust = sin(uTime * 0.9 - wp.x * 0.35 + wp.z * 0.1) * 0.5 + 0.5;
      float bend = max(position.y - 0.15, 0.0);
      transformed.x += (sin(uTime * 2.1 + wp.x * 1.7 + wp.z * 1.3) * 0.05 + gust * 0.06) * bend;
      transformed.z += cos(uTime * 1.7 + wp.z * 1.1) * 0.03 * bend;`);
    const trunks = new THREE.InstancedMesh(trunkGeo, std(0x6b4a2d), spots.length);
    const foliage = new THREE.InstancedMesh(foliageGeo, foliageMat, spots.length);
    spots.forEach((s, k) => {
      q.setFromAxisAngle(up, s.r);
      m4.compose(new THREE.Vector3(s.x, this.h(s.x, s.y) - 0.02, s.y), q, new THREE.Vector3(s.s, s.s * (0.9 + hash(k, 7) * 0.3), s.s));
      trunks.setMatrixAt(k, m4);
      foliage.setMatrixAt(k, m4);
      c.setHex(hash(k, 3) < 0.5 ? 0x2f6e3a : 0x3e8446).offsetHSL((hash(k, 1) - 0.5) * 0.03, 0, (hash(k, 2) - 0.5) * 0.08);
      foliage.setColorAt(k, c);
    });
    shadowed(trunks);
    shadowed(foliage);
    this.trees = { trunks, foliage };
    this.terrainGroup.add(trunks, foliage);

    // Grass tufts.
    const tuftGeo = mergeGeometries([0, 1, 2].map((i) => {
      const g = new THREE.ConeGeometry(0.012, 0.09, 3);
      g.translate(0, 0.045, 0);
      g.rotateZ((i - 1) * 0.35);
      g.translate((i - 1) * 0.015, 0, 0);
      return g;
    }));
    const tuftMat = std(0xffffff);
    addSway(tuftMat, this.time, `
      vec4 wp = instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
      transformed.x += sin(uTime * 2.4 - wp.x * 0.9 + wp.z * 0.4) * 0.25 * max(position.y, 0.0);`);
    const tuftMesh = new THREE.InstancedMesh(tuftGeo, tuftMat, tufts.length);
    tufts.forEach((t, k) => {
      q.setFromAxisAngle(up, t.r);
      m4.compose(new THREE.Vector3(t.x, this.h(t.x, t.z), t.z), q, new THREE.Vector3(t.s, t.s, t.s));
      tuftMesh.setMatrixAt(k, m4);
      tuftMesh.setColorAt(k, c.setHex(0x6fa843).offsetHSL(0, 0, (hash(k, 4) - 0.5) * 0.12));
    });
    tuftMesh.receiveShadow = true;
    this.terrainGroup.add(tuftMesh);

    // Rocks: jagged low-poly shapes in a few variants.
    const variants = [0, 1, 2, 3].map((v) => this.rockGeometry(v + 11));
    const placeRocks = (list: { x: number; z: number; s: number; h: number; r: number }[], base: number, light: number) => {
      variants.forEach((geo, v) => {
        const mine = list.filter((_, k) => k % variants.length === v);
        if (!mine.length) return;
        const mesh = new THREE.InstancedMesh(geo, std(0xffffff), mine.length);
        mine.forEach((r, k) => {
          q.setFromEuler(new THREE.Euler(0, r.r, (hash(k, v, 5) - 0.5) * 0.2));
          m4.compose(new THREE.Vector3(r.x, this.h(r.x, r.z) - r.s * 0.25, r.z), q, new THREE.Vector3(r.s, r.s * r.h, r.s));
          mesh.setMatrixAt(k, m4);
          mesh.setColorAt(k, c.setHex(hash(k, v, 6) < 0.5 ? base : light).offsetHSL(0, 0, (hash(k, v, 7) - 0.5) * 0.06));
        });
        this.terrainGroup.add(shadowed(mesh));
      });
    };
    placeRocks(peaks, 0x8c8880, 0xa9a59c);
    placeRocks(boulders, 0x9c978b, 0xb8b3a6);

    if (bushes.length) {
      const bush = new THREE.InstancedMesh(this.geo.sphereLow, std(0x3f6e2c), bushes.length);
      const berries = new THREE.InstancedMesh(this.geo.sphereLow, std(0xd4553b), bushes.length * 3);
      bushes.forEach((b, k) => {
        const y = this.h(b.x, b.z);
        m4.compose(new THREE.Vector3(b.x, y + 0.06, b.z), q.identity(), new THREE.Vector3(0.12, 0.09, 0.12));
        bush.setMatrixAt(k, m4);
        for (let j = 0; j < 3; j++) {
          m4.compose(new THREE.Vector3(b.x + Math.cos(j * 2.1) * 0.08, y + 0.12, b.z + Math.sin(j * 2.1) * 0.08), q, new THREE.Vector3(0.03, 0.03, 0.03));
          berries.setMatrixAt(k * 3 + j, m4);
        }
      });
      this.terrainGroup.add(shadowed(bush), berries);
    }
    if (crystals.length) {
      const cr = new THREE.InstancedMesh(new THREE.OctahedronGeometry(1, 0), std(0xd9f1ee, { emissive: 0x2a5a56, roughness: 0.2, metalness: 0.2 }), crystals.length);
      crystals.forEach((cc, k) => {
        q.setFromEuler(new THREE.Euler(0, 0, cc.r * 0.4));
        m4.compose(new THREE.Vector3(cc.x, this.h(cc.x, cc.z) + 0.1, cc.z), q, new THREE.Vector3(0.04, 0.12, 0.04));
        cr.setMatrixAt(k, m4);
      });
      this.terrainGroup.add(shadowed(cr));
    }
  }

  private trees: { trunks: THREE.InstancedMesh; foliage: THREE.InstancedMesh } | null = null;
  private treeSpots: { x: number; y: number; s: number; r: number }[] = [];

  private buildGrid(map: WorldMap): void {
    const g = new THREE.Group();
    const pts: number[] = [];
    for (let i = 0; i <= map.width; i++) {
      const x = i - map.width / 2;
      pts.push(x, 0.47, -map.height / 2, x, 0.47, map.height / 2);
      const z = i - map.height / 2;
      pts.push(-map.width / 2, 0.47, z, map.width / 2, 0.47, z);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    g.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25 })));
    g.visible = false;
    this.grid = g;
    this.terrainGroup.add(g);
  }

  // -------------------------------------------------------------------------
  // Frames
  // -------------------------------------------------------------------------

  setFrame(frame: Frame, animate: boolean, duration = 700): void {
    if (!this.map) return;
    this.prev = animate ? this.frame : null;
    this.frame = frame;
    this.animStart = performance.now();
    this.animDur = Math.max(250, Math.min(900, duration));
    this.syncCities(frame);
    this.syncDevs(frame);
    this.syncLoot(frame);
    this.syncBlobs(frame, animate);
    this.syncBoss(frame);
    this.syncTerritory(frame);
    this.syncTrees(frame);
    if (animate) this.spawnEventEffects(frame.events);
    this.updateBubbles(frame);
  }

  /** Hide trees on tiles that now hold a city or a development. */
  private syncTrees(f: Frame): void {
    if (!this.trees || !this.map) return;
    const busy = new Set<string>();
    for (const c of f.cities) busy.add(`${c.x},${c.y}`);
    for (const d of f.devs) busy.add(`${d.x},${d.y}`);
    const m4 = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    this.treeSpots.forEach((s, k) => {
      const tx = Math.floor(s.x + this.map!.width / 2);
      const ty = Math.floor(s.y + this.map!.height / 2);
      const hide = busy.has(`${tx},${ty}`);
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), s.r);
      const sc = hide ? 0.0001 : s.s;
      m4.compose(new THREE.Vector3(s.x, this.h(s.x, s.y) - 0.02, s.y), q, new THREE.Vector3(sc, sc * (0.9 + hash(k, 7) * 0.3), sc));
      this.trees!.trunks.setMatrixAt(k, m4);
      this.trees!.foliage.setMatrixAt(k, m4);
    });
    this.trees.trunks.instanceMatrix.needsUpdate = true;
    this.trees.foliage.instanceMatrix.needsUpdate = true;
  }

  private syncTerritory(f: Frame): void {
    const map = this.map!;
    const terr = computeTerritory(map, f.cities);
    const teamAt = (x: number, y: number): TeamId | null => {
      if (!inBounds(map, x, y)) return null;
      const ci = terr[idx(map, x, y)];
      return ci === null ? null : f.cities[ci].owner;
    };
    const owners: (TeamId | null)[] = [];
    for (let y = 0; y < map.height; y++) for (let x = 0; x < map.width; x++) owners.push(teamAt(x, y));
    const sig = `${this.showTerritory}|${owners.join(',')}`;
    if (sig === this.territorySig) return;
    this.territorySig = sig;

    // Tint the painted ground inside each team's land.
    const ground = this.ground!;
    const col = ground.geometry.attributes.color as THREE.BufferAttribute;
    const pos = ground.geometry.attributes.position as THREE.BufferAttribute;
    const base = this.groundBase!;
    const tint = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      let r = base[i * 3], g = base[i * 3 + 1], b = base[i * 3 + 2];
      if (this.showTerritory) {
        const t = this.hf!.tileAt(pos.getX(i), pos.getZ(i));
        const team = t ? owners[t[1] * map.width + t[0]] : null;
        if (team && pos.getY(i) > -0.02) {
          tint.set(FACTIONS[team].color);
          r = r * 0.72 + tint.r * 0.28;
          g = g * 0.72 + tint.g * 0.28;
          b = b * 0.72 + tint.b * 0.28;
        }
      }
      col.setXYZ(i, r, g, b);
    }
    col.needsUpdate = true;

    // Borders that follow the ground.
    const lines: number[] = [];
    const lineColors: number[] = [];
    const edge = (ax: number, az: number, bx: number, bz: number, color: THREE.Color) => {
      const steps = 6;
      for (let k = 0; k < steps; k++) {
        const t0 = k / steps;
        const t1 = (k + 1) / steps;
        const x0 = ax + (bx - ax) * t0, z0 = az + (bz - az) * t0;
        const x1 = ax + (bx - ax) * t1, z1 = az + (bz - az) * t1;
        lines.push(x0, Math.max(0.01, this.h(x0, z0)) + 0.025, z0, x1, Math.max(0.01, this.h(x1, z1)) + 0.025, z1);
        lineColors.push(color.r, color.g, color.b, color.r, color.g, color.b);
      }
    };
    for (let y = 0; y < map.height; y++) {
      for (let x = 0; x < map.width; x++) {
        const team = owners[y * map.width + x];
        if (!team) continue;
        const color = new THREE.Color(FACTIONS[team].color).offsetHSL(0, 0.1, 0.05);
        const x0 = this.wx(x) - 0.5;
        const z0 = this.wz(y) - 0.5;
        if (teamAt(x, y - 1) !== team) edge(x0, z0, x0 + 1, z0, color);
        if (teamAt(x, y + 1) !== team) edge(x0, z0 + 1, x0 + 1, z0 + 1, color);
        if (teamAt(x - 1, y) !== team) edge(x0, z0, x0, z0 + 1, color);
        if (teamAt(x + 1, y) !== team) edge(x0 + 1, z0, x0 + 1, z0 + 1, color);
      }
    }
    if (this.territoryLines) {
      this.terrainGroup.remove(this.territoryLines);
      this.territoryLines.geometry.dispose();
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(lines, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));
    this.territoryLines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true }));
    this.territoryLines.renderOrder = 3;
    this.territoryLines.visible = this.showTerritory;
    this.terrainGroup.add(this.territoryLines);
  }

  // -------------------------------------------------------------------------
  // Cities
  // -------------------------------------------------------------------------

  private flagMaterial(color: number): THREE.MeshStandardMaterial {
    let m = this.flagMat.get(color);
    if (!m) {
      m = std(color, { side: THREE.DoubleSide, flatShading: false });
      addSway(m, this.time, 'transformed.z += sin(uTime * 5.0 - position.x * 18.0) * 0.035 * (position.x / 0.26);');
      this.flagMat.set(color, m);
    }
    return m;
  }

  private makeFlag(color: number, height: number): THREE.Group {
    const g = new THREE.Group();
    const pole = new THREE.Mesh(this.geo.cyl, this.mats.woodDark);
    pole.scale.set(0.012, height, 0.012);
    pole.position.y = height / 2;
    const flag = new THREE.Mesh(this.geo.flag, this.flagMaterial(color));
    flag.position.set(0.012, height - 0.08, 0);
    g.add(pole, flag);
    return shadowed(g);
  }

  private makeHouse(roof: number, seed: number): THREE.Group {
    const g = new THREE.Group();
    const wallTone = [0xf1e6cf, 0xe9dcc0, 0xf4ecdb][seed % 3];
    const w = 0.2 + hash(seed, 1) * 0.05;
    const d = 0.16 + hash(seed, 2) * 0.03;
    const hgt = 0.14 + hash(seed, 3) * 0.04;
    const walls = new THREE.Mesh(this.geo.box, std(wallTone));
    walls.scale.set(w, hgt + 0.06, d);
    walls.position.y = (hgt - 0.06) / 2;
    const roofMesh = new THREE.Mesh(this.geo.gable, std(new THREE.Color(roof).offsetHSL(0, 0, (hash(seed, 4) - 0.5) * 0.08).getHex()));
    roofMesh.scale.set(d + 0.06, 0.2, w + 0.05);
    roofMesh.rotation.y = Math.PI / 2;
    roofMesh.position.y = hgt;
    const door = new THREE.Mesh(this.geo.box, this.mats.woodDark);
    door.scale.set(0.045, 0.075, 0.01);
    door.position.set(0, 0.0375, d / 2 + 0.002);
    g.add(walls, roofMesh, door);
    for (const sx of [-1, 1]) {
      const win = new THREE.Mesh(this.geo.box, std(0x3b4a57, { roughness: 0.3 }));
      win.scale.set(0.035, 0.035, 0.01);
      win.position.set(sx * w * 0.3, hgt * 0.62, d / 2 + 0.002);
      g.add(win);
    }
    const chimney = new THREE.Mesh(this.geo.box, this.mats.stoneDark);
    chimney.scale.set(0.035, 0.1, 0.035);
    chimney.position.set(w * 0.25, hgt + 0.08, -d * 0.15);
    chimney.name = 'chimney';
    g.add(chimney);
    return shadowed(g);
  }

  private makeCapital(style: string, color: number, dark: number, banner: number): THREE.Group {
    const g = new THREE.Group();
    const add = (geo: THREE.BufferGeometry, mat: THREE.Material, sx: number, sy: number, sz: number, x: number, y: number, z: number, ry = 0) => {
      const m = new THREE.Mesh(geo, mat);
      m.scale.set(sx, sy, sz);
      m.position.set(x, y, z);
      m.rotation.y = ry;
      g.add(m);
      return m;
    };
    if (style === 'keep') {
      add(this.geo.box, this.mats.stone, 0.32, 0.5, 0.32, 0, 0.25, 0);
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
        if (i > 0 && i < 3 && j > 0 && j < 3) continue;
        add(this.geo.box, this.mats.stone, 0.06, 0.06, 0.06, -0.12 + i * 0.08, 0.53, -0.12 + j * 0.08);
      }
      for (const s of [-1, 1]) {
        add(this.geo.cyl, this.mats.stoneDark, 0.07, 0.4, 0.07, s * 0.24, 0.2, 0.12);
        add(this.geo.cone4, std(dark), 0.1, 0.16, 0.1, s * 0.24, 0.48, 0.12, Math.PI / 4);
      }
      add(this.geo.box, this.mats.woodDark, 0.08, 0.12, 0.01, 0, 0.06, 0.165);
      const f = this.makeFlag(banner, 0.32);
      f.position.y = 0.56;
      g.add(f);
    } else if (style === 'citadel') {
      add(this.geo.cyl, this.mats.stone, 0.15, 0.55, 0.15, 0, 0.275, 0);
      const dome = new THREE.Mesh(new THREE.SphereGeometry(0.17, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2), std(color, { flatShading: false, roughness: 0.4, metalness: 0.2 }));
      dome.position.y = 0.55;
      g.add(dome);
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.3, 0.035, 6, 24), this.mats.stoneDark);
      ring.rotation.x = Math.PI / 2;
      ring.position.y = 0.04;
      g.add(ring);
      add(this.geo.box, std(0x3b4d5c), 0.05, 0.09, 0.01, 0, 0.3, 0.151);
      const f = this.makeFlag(banner, 0.26);
      f.position.y = 0.7;
      g.add(f);
    } else if (style === 'guildhall') {
      add(this.geo.box, this.mats.wall, 0.5, 0.24, 0.3, 0, 0.12, 0);
      for (let i = 0; i < 5; i++) add(this.geo.cyl, this.mats.wallShade, 0.02, 0.24, 0.02, -0.2 + i * 0.1, 0.12, 0.165);
      const roofGeo = new THREE.CylinderGeometry(0.2, 0.2, 0.54, 3);
      const roof = new THREE.Mesh(roofGeo, std(color, { metalness: 0.4, roughness: 0.4 }));
      roof.rotation.z = Math.PI / 2;
      roof.rotation.x = Math.PI / 2;
      roof.scale.set(1, 1, 0.75);
      roof.position.y = 0.33;
      g.add(roof);
      add(this.geo.sphereLow, this.mats.gold, 0.035, 0.035, 0.035, 0, 0.47, 0);
      for (const s of [-1, 1]) {
        const f = this.makeFlag(banner, 0.32);
        f.position.set(s * 0.27, 0.24, 0.1);
        g.add(f);
      }
    } else {
      for (let i = 0; i < 18; i++) {
        const a = (i / 18) * Math.PI * 2;
        const h = 0.2 + (i % 2) * 0.03;
        add(this.geo.cyl, this.mats.wood, 0.025, h, 0.025, Math.cos(a) * 0.34, h / 2, Math.sin(a) * 0.28);
      }
      add(this.geo.box, this.mats.woodDark, 0.16, 0.6, 0.16, 0, 0.3, -0.04);
      add(this.geo.cone4, std(color), 0.18, 0.22, 0.18, 0, 0.71, -0.04, Math.PI / 4);
      for (let k = 0; k < 3; k++) add(this.geo.sphereLow, std(0x3c7d41), 0.07, 0.06, 0.07, -0.08 + k * 0.08, 0.83, -0.04);
      const f = this.makeFlag(banner, 0.25);
      f.position.set(0, 0.82, -0.04);
      g.add(f);
    }
    return shadowed(g);
  }

  private makeWarden(): THREE.Group {
    const g = new THREE.Group();
    const body = new THREE.Mesh(this.geo.blob, std(0x9ba19c, { flatShading: false, roughness: 0.6 }));
    body.scale.set(0.34, 0.3, 0.3);
    body.position.y = 0.15;
    const helmet = new THREE.Mesh(new THREE.SphereGeometry(0.5, 12, 6, 0, Math.PI * 2, 0, Math.PI / 2), this.mats.metal);
    helmet.scale.set(0.26, 0.2, 0.26);
    helmet.position.y = 0.2;
    g.add(body, helmet, ...this.makeEyes(0.1, 0.17, 0.13));
    const spear = new THREE.Group();
    const shaft = new THREE.Mesh(this.geo.cyl, this.mats.wood);
    shaft.scale.set(0.012, 0.5, 0.012);
    shaft.position.y = 0.25;
    const tip = new THREE.Mesh(this.geo.cone4, this.mats.metal);
    tip.scale.set(0.03, 0.08, 0.03);
    tip.position.y = 0.54;
    spear.add(shaft, tip);
    spear.position.set(0.2, 0, 0.05);
    g.add(spear);
    return shadowed(g);
  }

  private syncCities(f: Frame): void {
    for (const c of f.cities) {
      const banner = c.owner ? new THREE.Color(FACTIONS[c.owner].color).getHex() : 0xdddddd;
      const roof = c.owner ? new THREE.Color(FACTIONS[c.owner].color).getHex() : NEUTRAL_ROOF;
      const sig = `${c.owner}|${c.level}|${c.walls}|${c.warden ? 1 : 0}`;
      let v = this.cityViews.get(c.id);
      if (v && v.sig === sig) continue;
      if (v) this.dynamicGroup.remove(v.group);
      const group = new THREE.Group();
      const gx = this.wx(c.x);
      const gz = this.wz(c.y);
      const gy = this.h(gx, gz);
      group.position.set(gx, gy, gz);
      /** Put a child on the real ground under it. */
      const seat = (o: THREE.Object3D, lx: number, lz: number, extra = 0) => {
        o.position.set(lx, this.h(gx + lx, gz + lz) - gy + extra, lz);
        group.add(o);
      };

      // Paved square, with a skirt so it never floats on a slope.
      const plaza = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.42, 0.3, 24), std(0xd9c9a0));
      plaza.position.y = -0.13;
      plaza.receiveShadow = true;
      group.add(plaza);
      for (let i = 0; i < 10; i++) {
        const stone = new THREE.Mesh(this.geo.box, std(0xc8b78c));
        const a = hash(c.x, c.y, 400 + i) * Math.PI * 2;
        const r = hash(c.x, c.y, 410 + i) * 0.32;
        stone.scale.set(0.06, 0.01, 0.05);
        stone.rotation.y = a;
        stone.position.set(Math.cos(a) * r, 0.022, Math.sin(a) * r);
        group.add(stone);
      }

      // Houses grow with the city.
      const houses = Math.min(7, c.level + 1 + (c.isCapital ? 1 : 0));
      const spots: [number, number, number][] = [[-0.27, -0.24, 0.2], [0.29, -0.22, -0.3], [-0.31, 0.14, 1.4], [0.31, 0.16, -1.5], [-0.05, -0.33, 0.05], [0.12, 0.33, 3.1], [-0.15, 0.32, 3.3]];
      const capitalHere = c.isCapital && c.founder;
      for (let i = 0; i < houses; i++) {
        let [lx, lz, ry] = spots[i];
        if (capitalHere && Math.hypot(lx, lz) < 0.3) continue;
        if (capitalHere) {
          lx *= 1.15;
          lz *= 1.15;
        }
        const hs = this.makeHouse(roof, c.x * 31 + c.y * 7 + i);
        hs.rotation.y = ry;
        hs.scale.setScalar(capitalHere ? 0.85 : 1);
        seat(hs, lx, lz);
      }

      if (capitalHere) {
        const fa = FACTIONS[c.founder!];
        const cap = this.makeCapital(fa.capitalStyle, new THREE.Color(fa.color).getHex(), new THREE.Color(fa.colorDark).getHex(), banner);
        seat(cap, 0, -0.02);
        // Stone curtain wall with corner towers.
        const R = 0.47;
        for (let i = 0; i < 28; i++) {
          const a = (i / 28) * Math.PI * 2;
          if (Math.abs(a - Math.PI / 2) < 0.18) continue; // gate facing south
          const seg = new THREE.Mesh(this.geo.box, this.mats.stone);
          seg.scale.set(0.11, 0.09, 0.05);
          seg.rotation.y = -a + Math.PI / 2;
          seat(shadowed(seg), Math.cos(a) * R, Math.sin(a) * R, 0.045);
        }
        for (const a of [Math.PI / 4, (3 * Math.PI) / 4, (5 * Math.PI) / 4, (7 * Math.PI) / 4]) {
          const tw = new THREE.Group();
          const body = new THREE.Mesh(this.geo.cyl, this.mats.stone);
          body.scale.set(0.055, 0.2, 0.055);
          body.position.y = 0.1;
          const cone = new THREE.Mesh(new THREE.ConeGeometry(1, 1, 8), std(new THREE.Color(fa.colorDark).getHex()));
          cone.scale.set(0.07, 0.1, 0.07);
          cone.position.y = 0.25;
          tw.add(body, cone);
          seat(shadowed(tw), Math.cos(a) * R, Math.sin(a) * R);
        }
      } else {
        // Villages: a timber fence with a gap for the road, a well and a banner.
        for (let i = 0; i < 26; i++) {
          const a = (i / 26) * Math.PI * 2;
          if (Math.abs(a - Math.PI / 2) < 0.2) continue;
          const post = new THREE.Mesh(this.geo.cyl, this.mats.wood);
          const ph = 0.1 + (i % 2) * 0.02;
          post.scale.set(0.014, ph, 0.014);
          seat(shadowed(post), Math.cos(a) * 0.46, Math.sin(a) * 0.46, ph / 2);
        }
        const well = new THREE.Group();
        const ring = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.055, 0.06, 10, 1, true), this.mats.stone);
        ring.position.y = 0.03;
        const water = new THREE.Mesh(new THREE.CircleGeometry(0.045, 10), std(0x3a8fb0));
        water.rotation.x = -Math.PI / 2;
        water.position.y = 0.04;
        const wroof = new THREE.Mesh(this.geo.cone4, std(roof));
        wroof.scale.set(0.08, 0.05, 0.08);
        wroof.rotation.y = Math.PI / 4;
        wroof.position.y = 0.16;
        for (const sx of [-1, 1]) {
          const p = new THREE.Mesh(this.geo.box, this.mats.wood);
          p.scale.set(0.01, 0.12, 0.01);
          p.position.set(sx * 0.045, 0.08, 0);
          well.add(p);
        }
        well.add(ring, water, wroof);
        seat(shadowed(well), 0.05, 0.05);
        const flag = this.makeFlag(banner, 0.42);
        seat(flag, -0.08, -0.06);
      }
      if (c.walls) {
        for (let i = 0; i < 24; i++) {
          const a = (i / 24) * Math.PI * 2 + 0.13;
          const seg = new THREE.Mesh(this.geo.box, this.mats.stoneDark);
          seg.scale.set(0.12, 0.14 + (i % 2) * 0.03, 0.06);
          seg.rotation.y = -a + Math.PI / 2;
          seat(shadowed(seg), Math.cos(a) * 0.52, Math.sin(a) * 0.52, 0.07);
        }
      }
      if (c.warden) {
        const w = this.makeWarden();
        w.name = 'warden';
        seat(w, 0, 0.3);
        w.userData.baseY = w.position.y;
      }
      this.dynamicGroup.add(group);
      const label = v?.label ?? this.makeLabel('city-label');
      v = { group, sig, label, ring: null, flags: [] };
      this.cityViews.set(c.id, v);
    }
  }

  // -------------------------------------------------------------------------
  // Developments and loot
  // -------------------------------------------------------------------------

  private makeDev(d: Development, team: TeamId | null): THREE.Group {
    const g = new THREE.Group();
    const add = (geo: THREE.BufferGeometry, mat: THREE.Material, sx: number, sy: number, sz: number, x: number, y: number, z: number) => {
      const m = new THREE.Mesh(geo, mat);
      m.scale.set(sx, sy, sz);
      m.position.set(x, y, z);
      g.add(m);
      return m;
    };
    const teamColor = team ? new THREE.Color(FACTIONS[team].color).getHex() : 0xdddddd;
    switch (d.kind) {
      case 'farm': {
        add(this.geo.box, this.mats.soil, 0.8, 0.22, 0.7, 0, -0.08, 0);
        const cropMat = this.mats.crop;
        if (!cropMat.userData.sway) {
          cropMat.userData.sway = true;
          addSway(cropMat, this.time, `
            vec4 wp = modelMatrix * vec4(0.0, 0.0, 0.0, 1.0);
            transformed.x += sin(uTime * 2.6 - wp.x * 0.8 - wp.z * 0.4) * 0.25 * max(position.y + 0.5, 0.0);`);
        }
        for (let r = 0; r < 4; r++) for (let k = 0; k < 6; k++) add(this.geo.box, cropMat, 0.025, 0.12, 0.025, -0.3 + k * 0.12, 0.09, -0.25 + r * 0.17);
        break;
      }
      case 'lumber':
        for (let i = 0; i < 3; i++) {
          const log = add(this.geo.cyl, this.mats.wood, 0.05, 0.36, 0.05, 0, 0.05 + (i === 2 ? 0.08 : 0), -0.05 + (i === 2 ? 0.05 : i * 0.1));
          log.rotation.z = Math.PI / 2;
        }
        add(this.geo.cyl, this.mats.woodDark, 0.08, 0.08, 0.08, 0.25, 0.04, 0.2);
        break;
      case 'mine': {
        const mound = new THREE.Mesh(new THREE.SphereGeometry(0.32, 10, 6, 0, Math.PI * 2, 0, Math.PI / 2), std(0x7a7060));
        mound.scale.y = 0.7;
        g.add(mound);
        add(this.geo.box, std(0x2b2620), 0.16, 0.16, 0.05, 0, 0.08, 0.27);
        add(this.geo.box, this.mats.wood, 0.03, 0.2, 0.03, -0.1, 0.1, 0.29);
        add(this.geo.box, this.mats.wood, 0.03, 0.2, 0.03, 0.1, 0.1, 0.29);
        add(this.geo.box, this.mats.wood, 0.24, 0.03, 0.03, 0, 0.2, 0.29);
        add(this.geo.box, std(0x6f5b40), 0.1, 0.06, 0.08, 0.25, 0.03, 0.3);
        break;
      }
      case 'tower': {
        add(this.geo.cyl, this.mats.stone, 0.11, 0.6, 0.11, 0, 0.3, 0);
        for (let i = 0; i < 6; i++) {
          const a = (i / 6) * Math.PI * 2;
          add(this.geo.box, this.mats.stone, 0.05, 0.06, 0.05, Math.cos(a) * 0.1, 0.63, Math.sin(a) * 0.1);
        }
        const f = this.makeFlag(teamColor, 0.25);
        f.position.y = 0.6;
        g.add(f);
        break;
      }
      case 'harbor': {
        add(this.geo.box, this.mats.wood, 0.9, 0.04, 0.2, 0, 0.0, 0.25);
        for (let i = 0; i < 4; i++) add(this.geo.cyl, this.mats.woodDark, 0.025, 0.3, 0.025, -0.4 + i * 0.27, -0.1, 0.33);
        const boat = this.makeBoat(teamColor, false);
        boat.scale.setScalar(0.7);
        boat.position.set(0.1, -0.12, 0.45);
        boat.name = 'bob';
        g.add(boat);
        break;
      }
      case 'trap':
        if (d.armed) {
          for (let i = 0; i < 8; i++) {
            const leaf = add(this.geo.sphereLow, this.mats.leaf, 0.07, 0.015, 0.04, (hash(d.x, d.y, 100 + i) - 0.5) * 0.45, 0.01, (hash(d.x, d.y, 110 + i) - 0.5) * 0.4);
            leaf.rotation.y = hash(d.x, d.y, i) * 6;
          }
          add(this.geo.cyl, this.mats.woodDark, 0.01, 0.25, 0.01, 0.2, 0.12, 0.1);
          const ribbon = new THREE.Mesh(this.geo.flag, this.flagMaterial(new THREE.Color(FACTIONS[d.builder].color).getHex()));
          ribbon.scale.set(0.5, 0.5, 1);
          ribbon.position.set(0.2, 0.22, 0.1);
          g.add(ribbon);
        } else {
          add(this.geo.cyl, this.mats.pit, 0.25, 0.02, 0.2, 0, 0.005, 0);
          for (let i = 0; i < 5; i++) {
            const st = add(this.geo.cone4, std(0xc9b58a), 0.02, 0.12, 0.02, -0.14 + i * 0.07, 0.05, (i % 2) * 0.06 - 0.03);
            st.rotation.z = (i - 2) * 0.12;
          }
        }
        break;
    }
    return shadowed(g);
  }

  private syncDevs(f: Frame): void {
    const keep = new Set<string>();
    const terr = computeTerritory(this.map!, f.cities);
    for (const d of f.devs) {
      const ci = terr[idx(this.map!, d.x, d.y)];
      const team = ci === null ? null : f.cities[ci].owner;
      const key = `${d.x},${d.y},${d.kind},${d.armed},${team}`;
      keep.add(key);
      if (this.devViews.has(key)) continue;
      const g = this.makeDev(d, team);
      g.position.set(this.wx(d.x), this.top(d.x, d.y), this.wz(d.y));
      g.userData = { x: d.x, y: d.y, kind: d.kind, armed: d.armed, builder: d.builder, born: d.builtOn === f.day ? performance.now() : 0 };
      this.dynamicGroup.add(g);
      this.devViews.set(key, g);
    }
    for (const [k, g] of this.devViews) {
      if (!keep.has(k)) {
        this.dynamicGroup.remove(g);
        this.devViews.delete(k);
      }
    }
  }

  private syncLoot(f: Frame): void {
    const keep = new Set(f.loot.map((l) => l.id));
    for (const l of f.loot) {
      if (this.lootViews.has(l.id)) continue;
      const g = new THREE.Group();
      for (let i = 0; i < 7; i++) {
        const coin = new THREE.Mesh(this.geo.cyl, this.mats.gold);
        coin.scale.set(0.07, 0.015, 0.07);
        coin.position.set((hash(l.x, l.y, 200 + i) - 0.5) * 0.18, 0.01 + i * 0.02, (hash(l.x, l.y, 210 + i) - 0.5) * 0.18);
        g.add(coin);
      }
      const chest = new THREE.Mesh(this.geo.box, this.mats.wood);
      chest.scale.set(0.16, 0.1, 0.11);
      chest.position.set(0.12, 0.05, -0.08);
      g.add(chest);
      g.position.set(this.wx(l.x), this.top(l.x, l.y), this.wz(l.y));
      g.userData = { x: l.x, y: l.y, id: l.id };
      this.dynamicGroup.add(shadowed(g));
      this.lootViews.set(l.id, g);
    }
    for (const [id, g] of this.lootViews) {
      if (!keep.has(id)) {
        this.dynamicGroup.remove(g);
        this.lootViews.delete(id);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Blobs, boats, boss
  // -------------------------------------------------------------------------

  private makeEyes(spread: number, y: number, z: number, size = 1): THREE.Mesh[] {
    const out: THREE.Mesh[] = [];
    for (const s of [-1, 1]) {
      const white = new THREE.Mesh(this.geo.eye, this.mats.white);
      white.scale.set(0.07 * size, 0.08 * size, 0.05 * size);
      white.position.set(s * spread, y, z);
      const pupil = new THREE.Mesh(this.geo.eye, this.mats.pupil);
      pupil.scale.set(0.038 * size, 0.045 * size, 0.025 * size);
      pupil.position.set(s * spread + s * 0.006 * size, y - 0.008 * size, z + 0.04 * size);
      pupil.name = 'pupil';
      const glint = new THREE.Mesh(this.geo.eye, this.mats.white);
      glint.scale.setScalar(0.012 * size);
      glint.position.set(s * spread - 0.012 * size, y + 0.014 * size, z + 0.062 * size);
      out.push(white, pupil, glint);
    }
    return out;
  }

  private makeBoat(sail: number, gunboat: boolean): THREE.Group {
    const g = new THREE.Group();
    const hull = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.62, 10, 1, false, 0, Math.PI), std(gunboat ? 0x4a3a2c : 0x8a6440, { side: THREE.DoubleSide }));
    hull.rotation.z = Math.PI / 2;
    hull.rotation.y = Math.PI / 2;
    hull.scale.set(1, 1, 0.55);
    hull.position.y = 0.05;
    hull.rotation.x = Math.PI;
    const deck = new THREE.Mesh(this.geo.box, this.mats.wood);
    deck.scale.set(0.24, 0.02, 0.6);
    deck.position.y = 0.05;
    const mast = new THREE.Mesh(this.geo.cyl, this.mats.woodDark);
    mast.scale.set(0.012, 0.5, 0.012);
    mast.position.set(0, 0.3, 0.05);
    const sailMesh = new THREE.Mesh(new THREE.PlaneGeometry(0.28, 0.3, 4, 1), this.flagMaterial(sail));
    sailMesh.rotation.y = Math.PI / 2;
    sailMesh.position.set(0.0, 0.36, 0.05);
    g.add(hull, deck, mast, sailMesh);
    if (gunboat) {
      for (const s of [-1, 1]) {
        const cannon = new THREE.Mesh(this.geo.cyl, std(0x222222, { metalness: 0.6 }));
        cannon.scale.set(0.025, 0.14, 0.025);
        cannon.rotation.z = Math.PI / 2;
        cannon.position.set(s * 0.13, 0.08, -0.12);
        g.add(cannon);
      }
    }
    return shadowed(g);
  }

  private blobSig(b: Blob): string {
    return `${b.team}|${b.rank}|${b.gear}|${b.vessel}`;
  }

  private makeBlob(b: Blob): BlobView {
    const f = FACTIONS[b.team];
    const root = new THREE.Group();
    const body = new THREE.Group();
    const scale = b.rank === 'champion' ? 1.12 : b.rank === 'veteran' ? 1.05 : 1;
    const mat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(f.color).offsetHSL(0, 0.08, 0.04),
      roughness: 0.38,
      metalness: 0,
      clearcoat: 0.9,
      clearcoatRoughness: 0.25,
      sheen: 0.4,
      sheenColor: new THREE.Color(f.colorLight),
      transparent: true,
      opacity: 1,
    });
    const bodyMesh = new THREE.Mesh(this.geo.blob, mat);
    bodyMesh.scale.set(0.44 * scale, 0.4 * scale, 0.42 * scale);
    bodyMesh.position.y = 0.19 * scale;
    bodyMesh.castShadow = true;
    body.add(bodyMesh);
    for (const e of this.makeEyes(0.085 * scale, 0.255 * scale, 0.16 * scale, scale)) body.add(e);
    const mouth = new THREE.Mesh(new THREE.TorusGeometry(0.028, 0.007, 6, 12, Math.PI), std(new THREE.Color(f.colorDark).getHex(), { flatShading: false }));
    mouth.rotation.z = Math.PI;
    mouth.position.set(0, 0.175 * scale, 0.205 * scale);
    body.add(mouth);

    const gear = new THREE.Group();
    let bow: THREE.Group | null = null;
    if (!b.vessel && b.gear === 'spear') {
      const shaft = new THREE.Mesh(this.geo.cyl, this.mats.wood);
      shaft.scale.set(0.012, 0.55, 0.012);
      shaft.position.y = 0.2;
      const tip = new THREE.Mesh(this.geo.cone4, this.mats.metal);
      tip.scale.set(0.03, 0.08, 0.03);
      tip.position.y = 0.5;
      gear.add(shaft, tip);
      gear.position.set(0.2, 0, 0.05);
      gear.rotation.x = 0.25;
    } else if (!b.vessel && b.gear === 'shield') {
      const sh = new THREE.Mesh(this.geo.cyl, std(new THREE.Color(f.colorDark).getHex()));
      sh.scale.set(0.11, 0.02, 0.11);
      sh.rotation.z = Math.PI / 2;
      sh.position.set(-0.21, 0.17, 0.04);
      const boss = new THREE.Mesh(this.geo.sphereLow, this.mats.gold);
      boss.scale.setScalar(0.025);
      boss.position.set(-0.225, 0.17, 0.04);
      gear.add(sh, boss);
    } else if (!b.vessel && b.gear === 'bow') {
      // The bow sits at the blob's front; its belly bulges toward the target
      // and the string runs on the archer's side.
      bow = new THREE.Group();
      const arc = new THREE.Mesh(new THREE.TorusGeometry(0.16, 0.012, 6, 16, Math.PI * 0.62), this.mats.wood);
      arc.rotation.z = -Math.PI * 0.31;
      arc.position.x = -0.13;
      const sx = -0.13 + Math.cos(Math.PI * 0.31) * 0.16;
      const sy = Math.sin(Math.PI * 0.31) * 0.16;
      const stringGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(sx, sy, 0), new THREE.Vector3(sx, 0, 0), new THREE.Vector3(sx, -sy, 0)]);
      const str = new THREE.Line(stringGeo, this.mats.rope);
      str.name = 'string';
      bow.add(arc, str);
      // Bow lies in the vertical plane that contains the facing direction.
      bow.rotation.y = -Math.PI / 2;
      bow.position.set(0, 0.17, 0.2);
      gear.add(bow);
    }
    body.add(gear);

    let rank: THREE.Object3D | null = null;
    if (b.rank === 'veteran') {
      rank = new THREE.Mesh(new THREE.OctahedronGeometry(0.05, 0), this.mats.silver);
      rank.position.y = 0.5;
    } else if (b.rank === 'champion') {
      const crown = new THREE.Group();
      const band = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 0.05, 10, 1, true), this.mats.gold);
      crown.add(band);
      for (let i = 0; i < 5; i++) {
        const a = (i / 5) * Math.PI * 2;
        const spike = new THREE.Mesh(this.geo.cone4, this.mats.gold);
        spike.scale.set(0.025, 0.06, 0.025);
        spike.position.set(Math.cos(a) * 0.085, 0.05, Math.sin(a) * 0.085);
        crown.add(spike);
      }
      crown.position.y = 0.4;
      rank = crown;
    }
    if (rank) body.add(rank);

    let boat: THREE.Group | null = null;
    if (b.vessel) {
      boat = this.makeBoat(new THREE.Color(f.color).getHex(), b.vessel === 'gunboat');
      root.add(boat);
      body.scale.setScalar(0.75);
      body.position.y = 0.06;
    }
    root.add(shadowed(body, true, false));
    const hp = document.createElement('div');
    hp.className = 'hp3d';
    hp.innerHTML = '<span></span>';
    this.overlay.appendChild(hp);
    return { root, body, bodyMesh, gear, bow, boat, rank, sig: this.blobSig(b), hp, facing: 0 };
  }

  private syncBlobs(f: Frame, animate: boolean): void {
    const alive = new Set(f.blobs.map((b) => b.id));
    // Blobs that died today linger as fading ghosts.
    for (const [id, v] of this.blobViews) {
      if (alive.has(id)) continue;
      this.blobViews.delete(id);
      v.hp.remove();
      if (animate) this.ghosts.push({ view: v, start: performance.now() });
      else this.dynamicGroup.remove(v.root);
    }
    for (const b of f.blobs) {
      let v = this.blobViews.get(b.id);
      if (v && v.sig !== this.blobSig(b)) {
        this.dynamicGroup.remove(v.root);
        v.hp.remove();
        const facing = v.facing;
        v = this.makeBlob(b);
        v.facing = facing;
        this.blobViews.set(b.id, v);
        this.dynamicGroup.add(v.root);
      }
      if (!v) {
        v = this.makeBlob(b);
        v.root.userData.born = animate ? performance.now() : 0;
        this.blobViews.set(b.id, v);
        this.dynamicGroup.add(v.root);
      }
      v.root.userData.blob = b;
    }
  }

  private syncBoss(f: Frame): void {
    const bo = f.boss;
    if (!(bo.risen && bo.alive)) {
      if (this.boss) {
        this.dynamicGroup.remove(this.boss.group);
        this.boss.label.remove();
        this.boss.hp.remove();
        this.boss = null;
      }
      return;
    }
    if (this.boss) return;
    const g = new THREE.Group();
    const body = new THREE.Mesh(this.geo.blob, new THREE.MeshStandardMaterial({ color: 0x56603a, roughness: 0.7 }));
    body.scale.set(1.3, 0.95, 1.2);
    body.position.y = 0.45;
    g.add(body);
    for (let i = 0; i < 6; i++) {
      const drip = new THREE.Mesh(this.geo.sphereLow, std(0x4a5230));
      const a = (i / 6) * Math.PI * 2;
      drip.scale.set(0.12, 0.08, 0.12);
      drip.position.set(Math.cos(a) * 0.6, 0.06, Math.sin(a) * 0.55);
      g.add(drip);
    }
    const eyes = this.makeEyes(0.2, 0.62, 0.5);
    for (const e of eyes) {
      e.scale.multiplyScalar(2.2);
      if (e.name !== 'pupil') (e as THREE.Mesh).material = std(0xf4e6a0, { flatShading: false });
      g.add(e);
    }
    for (const s of [-1, 1]) {
      const brow = new THREE.Mesh(this.geo.box, std(0x22281a));
      brow.scale.set(0.2, 0.04, 0.04);
      brow.position.set(s * 0.2, 0.8, 0.53);
      brow.rotation.z = s * -0.35;
      g.add(brow);
    }
    const reedMat = std(0x7c8a45);
    addSway(reedMat, this.time, 'transformed.x += sin(uTime * 2.0 + position.y * 3.0) * 0.08 * max(position.y, 0.0);');
    for (let i = -2; i <= 2; i++) {
      const reed = new THREE.Mesh(this.geo.cyl, reedMat);
      reed.scale.set(0.02, 0.35, 0.02);
      reed.position.set(i * 0.15, 1.0, -0.05);
      reed.rotation.z = i * 0.12;
      g.add(reed);
    }
    shadowed(g);
    this.dynamicGroup.add(g);
    const label = this.makeLabel('boss-label');
    label.textContent = bo.name;
    const hp = document.createElement('div');
    hp.className = 'hp3d boss';
    hp.innerHTML = '<span></span>';
    this.overlay.appendChild(hp);
    this.boss = { group: g, label, hp };
  }

  // -------------------------------------------------------------------------
  // Overlay elements (labels, HP bars, speech, floating numbers)
  // -------------------------------------------------------------------------

  private makeLabel(cls: string): HTMLDivElement {
    const el = document.createElement('div');
    el.className = cls;
    this.overlay.appendChild(el);
    return el;
  }

  private project(p: THREE.Vector3): { x: number; y: number; visible: boolean; depth: number } {
    const v = p.clone().project(this.camera);
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    return { x: (v.x * 0.5 + 0.5) * w, y: (-v.y * 0.5 + 0.5) * h, visible: v.z < 1 && v.z > -1 && Math.abs(v.x) < 1.2 && Math.abs(v.y) < 1.2, depth: v.z };
  }

  private place(el: HTMLElement, p: THREE.Vector3, show: boolean): void {
    const s = this.project(p);
    if (!show || !s.visible) {
      el.style.display = 'none';
      return;
    }
    el.style.display = '';
    el.style.transform = `translate(${s.x.toFixed(1)}px, ${s.y.toFixed(1)}px) translate(-50%, -100%)`;
    el.style.zIndex = String(1000 - Math.round(s.depth * 1000));
  }

  private updateBubbles(f: Frame): void {
    for (const b of this.bubbles) b.remove();
    this.bubbles = [];
    for (const s of f.speech.filter((sp) => sp.day === f.day)) {
      const el = this.makeLabel('speech3d');
      el.innerHTML = `${esc(s.text)}`;
      el.style.borderColor = FACTIONS[s.team].color;
      el.dataset.blob = s.blobId;
      el.dataset.x = String(s.x);
      el.dataset.y = String(s.y);
      this.bubbles.push(el);
    }
  }

  private floatText(text: string, x: number, y: number, z: number, color: string): void {
    const el = document.createElement('div');
    el.className = 'float3d';
    el.textContent = text;
    el.style.color = color;
    this.overlay.appendChild(el);
    this.floats.push({ el, pos: new THREE.Vector3(x, y, z), life: 0, max: 1.3 });
  }

  // -------------------------------------------------------------------------
  // Effects
  // -------------------------------------------------------------------------

  private particle(color: number, pos: THREE.Vector3, vel: THREE.Vector3, size: number, max: number, grow = 0, gravity = -2.5, delay = 0): void {
    const m = new THREE.Mesh(this.geo.sphereLow, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 1, depthWrite: false }));
    m.scale.setScalar(size);
    m.position.copy(pos);
    this.scene.add(m);
    this.particles.push({ mesh: m, vel, life: -delay, max, grow, gravity });
    if (this.particles.length > 500) {
      const old = this.particles.shift()!;
      this.scene.remove(old.mesh);
      (old.mesh.material as THREE.Material).dispose();
    }
  }

  private burst(x: number, y: number, z: number, color: number, n: number, speed = 1.5, size = 0.035, delay = 0): void {
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2;
      const s = speed * (0.4 + Math.random() * 0.8);
      this.particle(color, new THREE.Vector3(x, y, z), new THREE.Vector3(Math.cos(a) * s * 0.6, s, Math.sin(a) * s * 0.6), size, 0.9 + Math.random() * 0.5, 0, -3, delay);
    }
  }

  private ring(x: number, y: number, z: number, color: number, size: number, max = 1.4, delay = 0): void {
    const m = new THREE.Mesh(new THREE.TorusGeometry(1, 0.04, 6, 40), new THREE.MeshBasicMaterial({ color, transparent: true, depthWrite: false }));
    m.rotation.x = -Math.PI / 2;
    m.position.set(x, y, z);
    m.scale.setScalar(0.01);
    this.scene.add(m);
    this.particles.push({ mesh: m, vel: new THREE.Vector3(), life: -delay, max, grow: size, gravity: 0 });
  }

  private spawnEventEffects(events: GameEvent[]): void {
    const D = this.animDur / 1000;
    for (const e of events) {
      switch (e.kind) {
        case 'levelup': {
          const y = this.top(e.x, e.y) + 0.1;
          this.ring(this.wx(e.x), y, this.wz(e.y), 0xf3cf63, 0.9, 1.6, D * 0.8);
          this.burst(this.wx(e.x), y + 0.3, this.wz(e.y), 0xf7dc7d, 18, 1.6, 0.03, D * 0.8);
          break;
        }
        case 'capture': {
          const col = new THREE.Color(FACTIONS[e.team].colorLight).getHex();
          this.burst(this.wx(e.x), this.top(e.x, e.y) + 0.4, this.wz(e.y), col, 20, 1.8, 0.035, D * 0.85);
          this.ring(this.wx(e.x), this.top(e.x, e.y) + 0.05, this.wz(e.y), new THREE.Color(FACTIONS[e.team].color).getHex(), 0.8, 1.2, D * 0.85);
          break;
        }
        case 'trap': {
          const y = this.top(e.x, e.y);
          for (let i = 0; i < 16; i++) this.particle(0x9c8a62, new THREE.Vector3(this.wx(e.x), y + 0.05, this.wz(e.y)), new THREE.Vector3((Math.random() - 0.5) * 1.6, 1 + Math.random() * 1.4, (Math.random() - 0.5) * 1.6), 0.04, 0.9, 0.05, -3, D * 0.5);
          for (let i = 0; i < 6; i++) this.particle(0x6b8f3a, new THREE.Vector3(this.wx(e.x), y + 0.05, this.wz(e.y)), new THREE.Vector3((Math.random() - 0.5) * 2, 1.6 + Math.random(), (Math.random() - 0.5) * 2), 0.03, 1.4, 0, -2, D * 0.5);
          setTimeout(() => this.floatText(`-${e.dmg}`, this.wx(e.x), y + 0.6, this.wz(e.y), '#ffb59e'), this.animDur * 0.5);
          break;
        }
        case 'loot':
          this.burst(this.wx(e.x), this.top(e.x, e.y) + 0.15, this.wz(e.y), 0xf5d36b, 12, 1.4, 0.03, D * 0.4);
          setTimeout(() => this.floatText(`+${e.coins}`, this.wx(e.x), this.top(e.x, e.y) + 0.6, this.wz(e.y), '#f5d36b'), this.animDur * 0.4);
          break;
        case 'develop':
          for (let i = 0; i < 8; i++) this.particle(0xc8b78a, new THREE.Vector3(this.wx(e.x), this.top(e.x, e.y) + 0.05, this.wz(e.y)), new THREE.Vector3((Math.random() - 0.5) * 1.2, 0.6 + Math.random() * 0.6, (Math.random() - 0.5) * 1.2), 0.035, 0.7, 0.04, -2, D * 0.6);
          break;
        case 'boss_slam':
          this.ring(this.wx(e.x), this.top(e.x, e.y) + 0.05, this.wz(e.y), 0x5b5a3a, 1.8, 0.8, D * 0.5);
          break;
        case 'boss_rise':
          for (let i = 0; i < 30; i++) this.particle(0x55603a, new THREE.Vector3(this.wx(e.x), this.top(e.x, e.y), this.wz(e.y)), new THREE.Vector3((Math.random() - 0.5) * 2.5, 1 + Math.random() * 2, (Math.random() - 0.5) * 2.5), 0.06, 1.3);
          break;
        case 'promote': {
          const v = this.blobViews.get(e.id);
          if (v) {
            const p = v.root.position;
            this.burst(p.x, p.y + 0.4, p.z, e.rank === 'champion' ? 0xf3cf63 : 0xdfe6ea, 16, 1.6, 0.03, D);
          }
          break;
        }
        case 'recruit':
          this.burst(this.wx(e.x), this.top(e.x, e.y) + 0.2, this.wz(e.y), new THREE.Color(FACTIONS[e.team].colorLight).getHex(), 10, 1.2, 0.03, D * 0.6);
          break;
        case 'attack': {
          if (!e.ranged) break;
          const from = new THREE.Vector3(this.wx(e.fx), this.top(e.fx, e.fy) + 0.3, this.wz(e.fy));
          const to = new THREE.Vector3(this.wx(e.tx), this.top(e.tx, e.ty) + 0.25, this.wz(e.ty));
          let mesh: THREE.Object3D;
          if (e.weapon === 'cannon') {
            mesh = new THREE.Mesh(this.geo.sphereLow, std(0x222222));
            mesh.scale.setScalar(0.05);
          } else {
            const g = new THREE.Group();
            const shaft = new THREE.Mesh(this.geo.cyl, this.mats.wood);
            shaft.scale.set(0.008, 0.26, 0.008);
            shaft.rotation.x = Math.PI / 2;
            const head = new THREE.Mesh(this.geo.cone4, this.mats.metal);
            head.scale.set(0.02, 0.05, 0.02);
            head.rotation.x = Math.PI / 2;
            head.position.z = 0.15;
            g.add(shaft, head);
            mesh = g;
          }
          mesh.visible = false;
          this.scene.add(mesh);
          this.projectiles.push({ mesh, from, to, arc: from.distanceTo(to) * 0.3, dmg: e.dmg, done: false });
          break;
        }
        default:
          break;
      }
    }
  }

  // -------------------------------------------------------------------------
  // Per-frame animation
  // -------------------------------------------------------------------------

  private tick(): void {
    const dt = Math.min(0.05, this.clock.getDelta());
    this.time.value += dt;
    const t = this.time.value;
    const now = performance.now();
    const p = (now - this.animStart) / this.animDur;
    const f = this.frame;
    const map = this.map;

    this.flyCamera(dt);
    if (this.follow) {
      const target = this.selectionPosition();
      if (target) {
        const delta = target.clone().sub(this.controls.target).multiplyScalar(Math.min(1, dt * 4));
        this.controls.target.add(delta);
        this.camera.position.add(delta);
      }
    }
    this.controls.update();

    for (const c of this.clouds) {
      c.position.x += c.userData.speed * dt;
      if (c.position.x > 30) {
        c.position.x = -30;
        c.position.z = Math.random() * 30 - 15;
      }
      // Shade falls where sunlight through the cloud would land.
      const shade = c.userData.shade as THREE.Mesh;
      shade.position.set(c.position.x + c.position.y * 0.5, 0.72, c.position.z + c.position.y * 0.35);
    }

    if (f && map) {
      const viewer = this.viewer();
      const seen = (x: number, y: number) => !viewer.visible || viewer.visible.has(idx(map, x, y));
      const explored = (x: number, y: number) => !viewer.explored || viewer.explored(idx(map, x, y));
      this.animateBlobs(f, p, t, seen);
      this.animateGhosts(now);
      this.animateBoss(f, p, t, seen);
      this.animateProjectiles(p);
      this.animateCities(f, t, dt, explored);
      for (const g of this.devViews.values()) {
        const ud = g.userData as { x: number; y: number; kind: string; armed: boolean; builder: TeamId; born: number };
        const trapHidden = ud.kind === 'trap' && ud.armed && viewer.team !== null && !(viewer.team === ud.builder || f.teams[viewer.team].allies.includes(ud.builder));
        g.visible = !trapHidden && explored(ud.x, ud.y);
        if (ud.born) {
          const bp = clamp01((now - ud.born) / 600);
          const s = bp < 1 ? 0.3 + 0.7 * ease(bp) * (1 + Math.sin(bp * Math.PI * 2.5) * (1 - bp) * 0.35) : 1;
          g.scale.setScalar(s);
        }
        const bob = g.getObjectByName('bob');
        if (bob) bob.position.y = -0.12 + Math.sin(t * 1.8 + ud.x) * 0.015;
      }
      for (const g of this.lootViews.values()) {
        g.visible = seen(g.userData.x, g.userData.y);
        g.rotation.y = Math.sin(t * 0.8) * 0.1;
        if (g.visible && Math.random() < dt * 1.5) this.particle(0xfffbe0, g.position.clone().add(new THREE.Vector3((Math.random() - 0.5) * 0.2, 0.2, (Math.random() - 0.5) * 0.2)), new THREE.Vector3(0, 0.3, 0), 0.02, 0.6, 0, 0);
      }
      this.updateFog(viewer);
      this.updateSpeech(f, viewer);
      this.updateSelectionRing(t);
      if (this.territorySig && !this.territorySig.startsWith(String(this.showTerritory))) this.syncTerritory(f);
      if (this.grid) this.grid.visible = this.showCoords;
    }

    // Particles.
    for (const pt of this.particles) {
      pt.life += dt;
      if (pt.life < 0) {
        pt.mesh.visible = false;
        continue;
      }
      pt.mesh.visible = true;
      const k = pt.life / pt.max;
      if (pt.grow && pt.gravity === 0 && pt.vel.lengthSq() === 0) {
        pt.mesh.scale.setScalar(Math.max(0.01, pt.grow * ease(clamp01(k))));
      } else {
        pt.vel.y += pt.gravity * dt;
        pt.mesh.position.addScaledVector(pt.vel, dt);
        if (pt.grow) pt.mesh.scale.multiplyScalar(1 + pt.grow * dt * 10);
      }
      (pt.mesh.material as THREE.MeshBasicMaterial).opacity = Math.max(0, 1 - k);
    }
    const dead = this.particles.filter((pt) => pt.life >= pt.max);
    for (const pt of dead) {
      this.scene.remove(pt.mesh);
      (pt.mesh.material as THREE.Material).dispose();
    }
    this.particles = this.particles.filter((pt) => pt.life < pt.max);

    for (const fl of this.floats) {
      fl.life += dt;
      fl.pos.y += dt * 0.5;
      this.place(fl.el, fl.pos, true);
      fl.el.style.opacity = String(Math.max(0, 1 - fl.life / fl.max));
    }
    for (const fl of this.floats.filter((x) => x.life >= x.max)) fl.el.remove();
    this.floats = this.floats.filter((x) => x.life < x.max);

    this.renderer.render(this.scene, this.camera);
  }

  private blobTarget(b: Blob, path: [number, number][] | undefined, p: number) {
    // Moves play in the first part of the animation, one hop per tile.
    if (!path || path.length < 2 || p >= 0.55) return { x: b.x, y: b.y, hop: 0, squash: 1, dir: null as [number, number] | null };
    const mp = clamp01(p / 0.55) * (path.length - 1);
    const seg = Math.min(path.length - 2, Math.floor(mp));
    const t = mp - seg;
    const [ax, ay] = path[seg];
    const [bx, by] = path[seg + 1];
    const hop = Math.sin(t * Math.PI) * 0.4;
    const squash = t < 0.15 ? 1 - (0.15 - t) * 1.8 : t > 0.85 ? 1 - (t - 0.85) * 1.8 : 1 + Math.sin(t * Math.PI) * 0.18;
    return { x: lerp(ax, bx, ease(t)), y: lerp(ay, by, ease(t)), hop, squash, dir: [bx - ax, by - ay] as [number, number] };
  }

  private animateBlobs(f: Frame, p: number, t: number, seen: (x: number, y: number) => boolean): void {
    const moves = new Map<string, [number, number][]>();
    const attacks = new Map<string, Extract<GameEvent, { kind: 'attack' }>>();
    if (this.prev) {
      for (const e of f.events) {
        if (e.kind === 'move') moves.set(e.id, e.path);
        if (e.kind === 'attack') attacks.set(e.from, e);
      }
    }
    for (const [id, v] of this.blobViews) {
      const b = v.root.userData.blob as Blob;
      const pos = this.blobTarget(b, moves.get(id), p);
      const visible = seen(Math.round(pos.x), Math.round(pos.y)) || seen(b.x, b.y);
      v.root.visible = visible;
      const ground = this.groundAt(pos.x, pos.y);
      let gx = this.wx(pos.x);
      let gz = this.wz(pos.y);
      let gy = (b.vessel ? WATER_Y - 0.02 + Math.sin(t * 1.8 + b.x) * 0.02 : ground) + pos.hop;

      // Facing: toward movement, or toward the attack target.
      const atk = attacks.get(id);
      let aimDir: [number, number] | null = pos.dir;
      if (atk && p >= 0.4) aimDir = [atk.tx - atk.fx, atk.ty - atk.fy];
      if (aimDir && (aimDir[0] || aimDir[1])) {
        const want = Math.atan2(aimDir[0], aimDir[1]);
        let d = want - v.facing;
        while (d > Math.PI) d -= Math.PI * 2;
        while (d < -Math.PI) d += Math.PI * 2;
        v.facing += d * 0.25;
      }
      v.root.rotation.y = v.facing;

      // Melee lunge.
      if (atk && !atk.ranged) {
        const ap = clamp01((p - 0.45) / 0.35);
        const k = Math.sin(ap * Math.PI) * 0.3;
        const dx = atk.tx - atk.fx;
        const dz = atk.ty - atk.fy;
        const len = Math.hypot(dx, dz) || 1;
        gx += (dx / len) * k;
        gz += (dz / len) * k;
        gy += Math.sin(ap * Math.PI) * 0.08;
      }
      // Bow draw: pull the string back while aiming.
      if (v.bow) {
        const str = v.bow.getObjectByName('string') as THREE.Line | undefined;
        if (str) {
          const drawn = atk && atk.ranged && p > 0.3 && p < 0.5 ? 0.06 * clamp01((p - 0.3) / 0.15) : 0;
          const pos3 = str.geometry.attributes.position as THREE.BufferAttribute;
          pos3.setX(1, pos3.getX(0) - drawn);
          pos3.needsUpdate = true;
        }
      }

      v.root.position.set(gx, gy, gz);
      const breathe = 1 + Math.sin(t * 3 + id.length + b.x) * 0.03;
      const sq = pos.squash * breathe;
      v.body.scale.set((2 - sq) * (b.vessel ? 0.75 : 1), sq * (b.vessel ? 0.75 : 1), (2 - sq) * (b.vessel ? 0.75 : 1));
      if (v.boat) v.boat.rotation.z = Math.sin(t * 1.6 + b.y) * 0.05;
      if (v.rank) v.rank.rotation.y = t * 1.5;
      const born = v.root.userData.born as number | undefined;
      if (born) {
        const bp = clamp01((performance.now() - born) / 500);
        v.root.scale.setScalar(ease(bp));
        if (bp >= 1) v.root.userData.born = 0;
      } else v.root.scale.setScalar(1);

      // HP bar.
      const showHp = visible && b.hp < b.maxHp;
      v.hp.style.display = showHp ? '' : 'none';
      if (showHp) {
        const span = v.hp.firstElementChild as HTMLElement;
        const frac = b.hp / b.maxHp;
        span.style.width = `${Math.round(frac * 100)}%`;
        span.style.background = frac > 0.5 ? '#8fd16a' : frac > 0.25 ? '#e8c14d' : '#e06a4f';
        this.place(v.hp, new THREE.Vector3(gx, gy + 0.62, gz), true);
      }
    }
  }

  private animateGhosts(now: number): void {
    for (const g of this.ghosts) {
      const d = clamp01((now - g.start - this.animDur * 0.75) / 700);
      if (d > 0 && !g.view.root.userData.puffed) {
        g.view.root.userData.puffed = true;
        const pos = g.view.root.position;
        const b = g.view.root.userData.blob as Blob;
        this.burst(pos.x, pos.y + 0.2, pos.z, new THREE.Color(FACTIONS[b.team].colorLight).getHex(), 16, 1.2, 0.05);
      }
      g.view.body.scale.set(1 + d * 0.5, Math.max(0.05, 1 - d), 1 + d * 0.5);
      (g.view.bodyMesh.material as THREE.MeshStandardMaterial).opacity = 1 - d;
    }
    for (const g of this.ghosts.filter((x) => now - x.start > this.animDur * 0.75 + 700)) this.dynamicGroup.remove(g.view.root);
    this.ghosts = this.ghosts.filter((x) => now - x.start <= this.animDur * 0.75 + 700);
  }

  private animateBoss(f: Frame, p: number, t: number, seen: (x: number, y: number) => boolean): void {
    if (!this.boss) return;
    const bo = f.boss;
    let bx = bo.x;
    let by = bo.y;
    const mv = this.prev ? f.events.find((e) => e.kind === 'boss_move') : undefined;
    if (mv && mv.kind === 'boss_move' && p < 0.5) {
      const k = ease(clamp01(p / 0.5));
      bx = lerp(mv.path[0][0], mv.path[1][0], k);
      by = lerp(mv.path[0][1], mv.path[1][1], k);
    }
    const slam = this.prev && f.events.some((e) => e.kind === 'boss_slam') ? Math.sin(clamp01((p - 0.45) / 0.35) * Math.PI) : 0;
    const g = this.boss.group;
    g.position.set(this.wx(bx), this.top(Math.round(bx), Math.round(by)) + slam * 0.3, this.wz(by));
    g.scale.set(1 + Math.sin(t * 1.4) * 0.03 + slam * 0.1, 1 - Math.sin(t * 1.4) * 0.03 - slam * 0.1, 1);
    g.visible = seen(bo.x, bo.y);
    this.place(this.boss.label, new THREE.Vector3(g.position.x, g.position.y + 1.65, g.position.z), g.visible);
    const span = this.boss.hp.firstElementChild as HTMLElement;
    span.style.width = `${Math.round((100 * bo.hp) / bo.maxHp)}%`;
    this.place(this.boss.hp, new THREE.Vector3(g.position.x, g.position.y + 1.45, g.position.z), g.visible);
  }

  private animateProjectiles(p: number): void {
    const ap = clamp01((p - 0.45) / 0.35);
    for (const pr of this.projectiles) {
      if (pr.done) continue;
      if (ap <= 0) continue;
      pr.mesh.visible = ap < 1;
      const pos = pr.from.clone().lerp(pr.to, ap);
      pos.y += Math.sin(ap * Math.PI) * pr.arc;
      const ahead = pr.from.clone().lerp(pr.to, Math.min(1, ap + 0.02));
      ahead.y += Math.sin(Math.min(1, ap + 0.02) * Math.PI) * pr.arc;
      pr.mesh.position.copy(pos);
      pr.mesh.lookAt(ahead);
      if (ap >= 1) {
        pr.done = true;
        this.scene.remove(pr.mesh);
      }
    }
    this.projectiles = this.projectiles.filter((x) => !x.done);
    // Damage numbers for every hit, ranged or not.
    if (this.frame && this.prev && ap >= 0.85 && !this.frame.events.some((e) => (e as { _shown?: boolean })._shown)) {
      for (const e of this.frame.events) {
        if (e.kind !== 'attack') continue;
        (e as { _shown?: boolean })._shown = true;
        const x = this.wx(e.tx);
        const z = this.wz(e.ty);
        const y = this.top(e.tx, e.ty);
        this.floatText(`-${e.dmg}`, x, y + 0.65, z, '#ffb59e');
        this.burst(x, y + 0.3, z, 0xfff3d6, 6, 1.2, 0.02);
      }
    }
  }

  private animateCities(f: Frame, t: number, dt: number, explored: (x: number, y: number) => boolean): void {
    this.smokeTimer += dt;
    const puff = this.smokeTimer > 0.25;
    if (puff) this.smokeTimer = 0;
    for (const c of f.cities) {
      const v = this.cityViews.get(c.id);
      if (!v) continue;
      const vis = explored(c.x, c.y);
      v.group.visible = vis;
      const owner = c.owner ? FACTIONS[c.owner] : null;
      v.label.innerHTML = `${esc(c.name)} <b>${c.level}</b>`;
      v.label.style.borderColor = owner ? owner.color : '#d8d2c2';
      this.place(v.label, new THREE.Vector3(this.wx(c.x), this.top(c.x, c.y) + (c.isCapital ? 1.05 : 0.7), this.wz(c.y)), vis);
      const warden = v.group.getObjectByName('warden');
      if (warden) warden.position.y = (warden.userData.baseY ?? 0) + Math.abs(Math.sin(t * 2 + c.x)) * 0.03;
      // Chimney smoke.
      if (puff && vis && Math.random() < 0.5) {
        const chimneys: THREE.Object3D[] = [];
        v.group.traverse((o) => { if (o.name === 'chimney') chimneys.push(o); });
        const ch = chimneys[Math.floor(Math.random() * chimneys.length)];
        if (ch) {
          const wp = ch.getWorldPosition(new THREE.Vector3());
          wp.y += 0.06;
          this.particle(0xe9e6df, wp, new THREE.Vector3(0.08, 0.25, 0.02), 0.03, 2.4, 0.06, 0);
        }
      }
      // Gold glint ring on the day a city grows.
      if (c.levelUpOn === f.day) {
        if (!v.ring) {
          v.ring = new THREE.Mesh(new THREE.TorusGeometry(0.55, 0.02, 6, 40), new THREE.MeshBasicMaterial({ color: 0xf3cf63, transparent: true }));
          v.ring.rotation.x = -Math.PI / 2;
          v.ring.position.y = 0.08;
          v.group.add(v.ring);
        }
        (v.ring.material as THREE.MeshBasicMaterial).opacity = 0.5 + Math.sin(t * 4) * 0.4;
        v.ring.position.y = 0.08 + Math.abs(Math.sin(t * 2)) * 0.05;
      } else if (v.ring) {
        v.group.remove(v.ring);
        v.ring = null;
      }
    }
  }

  private updateSpeech(f: Frame, viewer: ReturnType<Renderer['viewer']>): void {
    for (const el of this.bubbles) {
      const id = el.dataset.blob!;
      const sx = Number(el.dataset.x);
      const sy = Number(el.dataset.y);
      let show = this.showSpeech;
      if (viewer.blob) show = show && dist(viewer.blob.x, viewer.blob.y, sx, sy) <= HEARING_RANGE;
      else if (viewer.team) show = show && f.blobs.some((b) => b.team === viewer.team && dist(b.x, b.y, sx, sy) <= HEARING_RANGE);
      const v = this.blobViews.get(id);
      const pos = v ? v.root.position.clone().add(new THREE.Vector3(0, 0.75, 0)) : new THREE.Vector3(this.wx(sx), this.top(sx, sy) + 0.75, this.wz(sy));
      this.place(el, pos, show && (!v || v.root.visible));
    }
  }

  // -------------------------------------------------------------------------
  // Fog of war, selection
  // -------------------------------------------------------------------------

  private viewer(): { team: TeamId | null; visible: Set<number> | null; explored: ((i: number) => boolean) | null; blob: Blob | null } {
    const f = this.frame;
    const map = this.map;
    if (!f || !map || this.view.kind === 'god') return { team: null, visible: null, explored: null, blob: null };
    if (this.view.kind === 'blob') {
      const id = this.view.id;
      const b = f.blobs.find((o) => o.id === id) ?? null;
      if (b) return { team: b.team, visible: new Set(visibleTiles(map, b)), explored: exploredSet(f.teams[b.team].explored), blob: b };
      return { team: null, visible: null, explored: null, blob: null };
    }
    const team = this.view.team;
    const vis = new Set<number>();
    for (const b of f.blobs) if (b.team === team) for (const i of visibleTiles(map, b)) vis.add(i);
    for (const c of f.cities) if (c.owner === team) vis.add(idx(map, c.x, c.y));
    return { team, visible: vis, explored: exploredSet(f.teams[team].explored), blob: null };
  }

  private lastFogKey = '';

  private updateFog(viewer: ReturnType<Renderer['viewer']>): void {
    const map = this.map!;
    const key = viewer.visible ? `${this.frame?.day}|${JSON.stringify(this.view)}|${viewer.blob ? `${viewer.blob.x},${viewer.blob.y}` : ''}` : 'none';
    if (key === this.lastFogKey) return;
    this.lastFogKey = key;
    if (this.hearing) {
      this.scene.remove(this.hearing);
      this.hearing = null;
    }
    const shroud = this.shroud!;
    if (!viewer.visible || !viewer.explored) {
      shroud.visible = false;
      return;
    }
    const vis = viewer.visible;
    const exp = viewer.explored;
    paintShroud(shroud, this.hf!, (x, y) => {
      const i = idx(map, x, y);
      return vis.has(i) ? 0 : exp(i) ? 0.5 : 0.9;
    });
    shroud.visible = true;
    if (viewer.blob) {
      const r = HEARING_RANGE + 0.5;
      const cx = this.wx(viewer.blob.x);
      const cz = this.wz(viewer.blob.y);
      const pts: THREE.Vector3[] = [];
      const corners = [[-r, -r], [r, -r], [r, r], [-r, r], [-r, -r]];
      for (let k = 0; k < 4; k++) {
        for (let s2 = 0; s2 < 12; s2++) {
          const t = s2 / 12;
          const x = cx + corners[k][0] + (corners[k + 1][0] - corners[k][0]) * t;
          const z = cz + corners[k][1] + (corners[k + 1][1] - corners[k][1]) * t;
          pts.push(new THREE.Vector3(x, Math.max(0.02, this.h(x, z)) + 0.06, z));
        }
      }
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      this.hearing = new THREE.LineLoop(geo, new THREE.LineDashedMaterial({ color: 0xfff6de, dashSize: 0.2, gapSize: 0.14 }));
      this.hearing.computeLineDistances();
      this.scene.add(this.hearing);
    }
  }

  private selectionPosition(): THREE.Vector3 | null {
    const f = this.frame;
    const sel = this.selection;
    if (!f || !sel) return null;
    if (sel.kind === 'blob') {
      const v = this.blobViews.get(sel.id);
      return v ? v.root.position.clone() : null;
    }
    if (sel.kind === 'boss') return this.boss ? this.boss.group.position.clone() : null;
    let x: number | null = null;
    let y: number | null = null;
    if (sel.kind === 'city') {
      const c = f.cities.find((o) => o.id === sel.id);
      if (c) { x = c.x; y = c.y; }
    } else if (sel.kind === 'loot') {
      const l = f.loot.find((o) => o.id === sel.id);
      if (l) { x = l.x; y = l.y; }
    } else {
      x = sel.x;
      y = sel.y;
    }
    if (x === null || y === null) return null;
    return new THREE.Vector3(this.wx(x), this.top(x, y), this.wz(y));
  }

  private updateSelectionRing(t: number): void {
    const p = this.selectionPosition();
    this.selectRing.visible = !!p;
    if (!p) return;
    this.selectRing.position.set(p.x, (this.selection?.kind === 'blob' ? p.y : p.y) + 0.03, p.z);
    const s = this.selection?.kind === 'boss' ? 1.8 : this.selection?.kind === 'city' ? 1.35 : 1;
    this.selectRing.scale.setScalar(s * (1 + Math.sin(t * 4) * 0.05));
  }

  /** What is under a click, in priority order. */
  pick(clientX: number, clientY: number): Selection {
    const map = this.map;
    const f = this.frame;
    if (!map || !f) return null;
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);

    // Units first: they are small and stand above the ground.
    const unitRoots: THREE.Object3D[] = [];
    for (const v of this.blobViews.values()) if (v.root.visible) unitRoots.push(v.root);
    if (this.boss?.group.visible) unitRoots.push(this.boss.group);
    const hitUnit = ray.intersectObjects(unitRoots, true)[0];
    if (hitUnit) {
      let o: THREE.Object3D | null = hitUnit.object;
      while (o && !unitRoots.includes(o)) o = o.parent;
      if (o === this.boss?.group) return { kind: 'boss' };
      for (const [id, v] of this.blobViews) if (v.root === o) return { kind: 'blob', id };
    }

    // Otherwise work out the tile from the terrain.
    const targets: THREE.Object3D[] = [this.ground!, this.water!];
    const hit = ray.intersectObjects(targets, false)[0];
    if (!hit) return null;
    const t = this.hf!.tileAt(hit.point.x, hit.point.z);
    if (!t) return null;
    const [x, y] = t;
    const tile = idx(map, x, y);
    const viewer = this.viewer();
    const visible = !viewer.visible || viewer.visible.has(tile);
    const blob = visible ? f.blobs.find((b) => b.x === x && b.y === y) : undefined;
    if (blob) return { kind: 'blob', id: blob.id };
    if (visible && f.boss.risen && f.boss.alive && f.boss.x === x && f.boss.y === y) return { kind: 'boss' };
    const city = f.cities.find((c) => c.x === x && c.y === y);
    if (city) return { kind: 'city', id: city.id };
    const loot = visible ? f.loot.find((l) => l.x === x && l.y === y) : undefined;
    if (loot) return { kind: 'loot', id: loot.id };
    const dev = f.devs.find((d) => d.x === x && d.y === y);
    if (dev && (dev.kind !== 'trap' || !dev.armed || viewer.team === null || viewer.team === dev.builder || f.teams[viewer.team].allies.includes(dev.builder))) return { kind: 'dev', x, y };
    return { kind: 'tile', x, y };
  }
}
