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
  private landMesh: THREE.InstancedMesh | null = null;
  private seabedMesh: THREE.InstancedMesh | null = null;
  private tileOfInstance: number[] = [];
  private seabedTileOfInstance: number[] = [];
  private water: THREE.Mesh | null = null;
  private foam: THREE.Mesh | null = null;
  private territoryMesh: THREE.InstancedMesh | null = null;
  private territoryLines: THREE.LineSegments | null = null;
  private fogNear: THREE.InstancedMesh | null = null;
  private fogFar: THREE.InstancedMesh | null = null;
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

  private top(x: number, y: number): number {
    if (!this.map || !inBounds(this.map, x, y)) return WATER_Y;
    const t = this.map.tiles[idx(this.map, x, y)].terrain;
    return isWater(t) ? WATER_Y : TOP[t];
  }

  /** Ground height under a fractional tile position (for hops between tiles). */
  private groundAt(fx: number, fy: number): number {
    return this.top(Math.round(fx), Math.round(fy));
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

    this.scene.add(new THREE.HemisphereLight(0xdfefff, 0x5a6b3a, 1.1));
    const sun = new THREE.DirectionalLight(0xfff1d6, 2.4);
    sun.position.set(-12, 22, -8);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera;
    sc.left = -16; sc.right = 16; sc.top = 16; sc.bottom = -16; sc.near = 1; sc.far = 60;
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.02;
    this.scene.add(sun);
    this.scene.add(sun.target);

    // The open sea beyond the map, fading into the fog.
    const farSea = new THREE.Mesh(new THREE.CircleGeometry(150, 48), std(0x1f4e66, { flatShading: false, roughness: 0.4 }));
    farSea.rotation.x = -Math.PI / 2;
    farSea.position.y = -0.82;
    this.scene.add(farSea);

    // Clouds: clusters of soft spheres that drift and cast shadows on the island.
    const cloudMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, transparent: true, opacity: 0.8, flatShading: true, depthWrite: false });
    for (let i = 0; i < 7; i++) {
      const g = new THREE.Group();
      const n = 4 + Math.floor(Math.random() * 4);
      for (let k = 0; k < n; k++) {
        const s = 0.7 + Math.random() * 0.9;
        const m = new THREE.Mesh(this.geo.sphereLow, cloudMat);
        m.scale.set(s * 1.3, s * 0.7, s);
        m.position.set((Math.random() - 0.5) * 3, Math.random() * 0.4, (Math.random() - 0.5) * 1.6);
        m.castShadow = true;
        g.add(m);
      }
      g.position.set(Math.random() * 40 - 20, 10 + Math.random() * 2.5, Math.random() * 30 - 15);
      g.userData.speed = 0.25 + Math.random() * 0.25;
      this.clouds.push(g);
      this.scene.add(g);
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
    this.frame = null;
    this.prev = null;

    const W = map.width;
    const H = map.height;
    const color = new THREE.Color();

    // Land columns.
    const landTiles: number[] = [];
    const seaTiles: number[] = [];
    map.tiles.forEach((t, i) => (isWater(t.terrain) ? seaTiles : landTiles).push(i));
    const land = new THREE.InstancedMesh(this.geo.box, std(0xffffff), landTiles.length);
    const m4 = new THREE.Matrix4();
    landTiles.forEach((i, k) => {
      const x = i % W;
      const y = Math.floor(i / W);
      const t = map.tiles[i].terrain;
      const hgt = TOP[t] + 0.6;
      m4.compose(new THREE.Vector3(this.wx(x), TOP[t] - hgt / 2, this.wz(y)), new THREE.Quaternion(), new THREE.Vector3(1, hgt, 1));
      land.setMatrixAt(k, m4);
      const j = (hash(x, y, 1) - 0.5) * 0.08;
      color.setHex(LAND_COLOR[t]).offsetHSL(0, 0, j);
      land.setColorAt(k, color);
    });
    land.receiveShadow = true;
    land.castShadow = true;
    this.landMesh = land;
    this.tileOfInstance = landTiles;
    this.terrainGroup.add(land);

    // Seabed: sandy in the shallows, darker further out (seen through the water).
    const seabed = new THREE.InstancedMesh(this.geo.box, std(0xffffff), seaTiles.length);
    const sandC = new THREE.Color(0xd9c99a);
    const deepC = new THREE.Color(0x1f4e66);
    seaTiles.forEach((i, k) => {
      const x = i % W;
      const y = Math.floor(i / W);
      const d = map.depth[i];
      const yTop = -0.12 - d * 0.7;
      m4.compose(new THREE.Vector3(this.wx(x), yTop - 0.3, this.wz(y)), new THREE.Quaternion(), new THREE.Vector3(1, 0.6, 1));
      seabed.setMatrixAt(k, m4);
      seabed.setColorAt(k, sandC.clone().lerp(deepC, ease(clamp01(d * 1.2))));
    });
    seabed.receiveShadow = true;
    this.seabedMesh = seabed;
    this.seabedTileOfInstance = seaTiles;
    this.terrainGroup.add(seabed);

    // Translucent sea with gentle waves.
    const waterMat = new THREE.MeshStandardMaterial({ color: 0x3f93b3, transparent: true, opacity: 0.72, roughness: 0.25, metalness: 0.1 });
    addSway(waterMat, this.time, 'transformed.z += sin(position.x * 1.4 + uTime * 1.1) * 0.025 + cos(position.y * 1.7 + uTime * 0.8) * 0.025;');
    const water = new THREE.Mesh(new THREE.PlaneGeometry(W + 30, H + 30, 100, 100), waterMat);
    water.rotation.x = -Math.PI / 2;
    water.position.y = WATER_Y;
    water.receiveShadow = true;
    this.water = water;
    this.terrainGroup.add(water);

    // Shoreline foam: thin white strips on water tiles that touch land.
    const foamGeos: THREE.BufferGeometry[] = [];
    for (const i of seaTiles) {
      const x = i % W;
      const y = Math.floor(i / W);
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as [number, number][]) {
        if (!inBounds(map, x + dx, y + dy) || isWater(map.tiles[idx(map, x + dx, y + dy)].terrain)) continue;
        const g = new THREE.PlaneGeometry(dx !== 0 ? 0.16 : 1, dx !== 0 ? 1 : 0.16);
        g.rotateX(-Math.PI / 2);
        g.translate(this.wx(x) + dx * 0.42, WATER_Y + 0.03, this.wz(y) + dy * 0.42);
        foamGeos.push(g);
      }
    }
    if (foamGeos.length) {
      const foamMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6, depthWrite: false });
      this.foam = new THREE.Mesh(mergeGeometries(foamGeos), foamMat);
      this.terrainGroup.add(this.foam);
    }

    this.buildDecor(map);

    // Territory overlay (updated per frame).
    const tGeo = new THREE.PlaneGeometry(1, 1);
    tGeo.rotateX(-Math.PI / 2);
    this.territoryMesh = new THREE.InstancedMesh(tGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.22, depthWrite: false }), W * H);
    this.territoryMesh.count = 0;
    this.territoryMesh.renderOrder = 2;
    this.terrainGroup.add(this.territoryMesh);

    // Fog-of-war volumes (updated per frame).
    const fogMatA = new THREE.MeshBasicMaterial({ color: 0x0e1216, transparent: true, opacity: 0.5 });
    const fogMatB = new THREE.MeshBasicMaterial({ color: 0x0e1216, transparent: true, opacity: 0.88 });
    this.fogNear = new THREE.InstancedMesh(this.geo.box, fogMatA, W * H);
    this.fogFar = new THREE.InstancedMesh(this.geo.box, fogMatB, W * H);
    this.fogNear.count = 0;
    this.fogFar.count = 0;
    this.fogNear.renderOrder = 5;
    this.fogFar.renderOrder = 5;
    this.terrainGroup.add(this.fogNear, this.fogFar);

    this.buildGrid(map);
  }

  /** Trees, hill boulders, mountains, fruit bushes and crystals. */
  private buildDecor(map: WorldMap): void {
    const W = map.width;
    const trunkGeo = new THREE.CylinderGeometry(0.035, 0.045, 0.18, 6);
    trunkGeo.translate(0, 0.09, 0);
    const tiers = [new THREE.ConeGeometry(0.2, 0.3, 7), new THREE.ConeGeometry(0.16, 0.26, 7), new THREE.ConeGeometry(0.11, 0.22, 7)];
    tiers[0].translate(0, 0.3, 0);
    tiers[1].translate(0, 0.45, 0);
    tiers[2].translate(0, 0.58, 0);
    const foliageGeo = mergeGeometries(tiers);

    const spots: { x: number; y: number; s: number; r: number }[] = [];
    const boulders: { x: number; y: number; z: number; s: number }[] = [];
    const mountains: { x: number; z: number; y: number; s: number; h: number }[] = [];
    const bushes: { x: number; y: number; z: number }[] = [];
    const crystals: { x: number; y: number; z: number; r: number }[] = [];
    for (let y = 0; y < map.height; y++) {
      for (let x = 0; x < W; x++) {
        const t = map.tiles[idx(map, x, y)];
        const base = TOP[t.terrain];
        if (t.terrain === 'forest') {
          for (let k = 0; k < 3; k++) spots.push({ x: this.wx(x) - 0.3 + hash(x, y, 70 + k) * 0.6, y: this.wz(y) - 0.3 + hash(x, y, 80 + k) * 0.6, s: 0.8 + hash(x, y, 90 + k) * 0.5, r: hash(x, y, k) * 6 });
        } else if (t.terrain === 'grass' && hash(x, y, 7) < 0.12) {
          spots.push({ x: this.wx(x) - 0.3 + hash(x, y, 71) * 0.6, y: this.wz(y) - 0.3 + hash(x, y, 81) * 0.6, s: 0.7, r: 0 });
        } else if (t.terrain === 'hills') {
          for (let k = 0; k < 3; k++) boulders.push({ x: this.wx(x) - 0.3 + hash(x, y, 50 + k) * 0.6, y: base, z: this.wz(y) - 0.3 + hash(x, y, 60 + k) * 0.6, s: 0.12 + hash(x, y, 55 + k) * 0.14 });
        } else if (t.terrain === 'mountain') {
          mountains.push({ x: this.wx(x), z: this.wz(y), y: base, s: 0.52, h: 0.85 + hash(x, y, 3) * 0.45 });
          mountains.push({ x: this.wx(x) - 0.25 + hash(x, y, 4) * 0.5, z: this.wz(y) - 0.25 + hash(x, y, 5) * 0.5, y: base, s: 0.33, h: 0.5 + hash(x, y, 6) * 0.3 });
        }
        if (t.resource === 'fruit') for (let k = 0; k < 2; k++) bushes.push({ x: this.wx(x) - 0.25 + hash(x, y, 30 + k) * 0.5, y: base, z: this.wz(y) - 0.25 + hash(x, y, 40 + k) * 0.5 });
        if (t.resource === 'crystal') for (let k = 0; k < 3; k++) crystals.push({ x: this.wx(x) + 0.15 + (k - 1) * 0.08, y: base, z: this.wz(y) + 0.2, r: k * 0.6 - 0.6 });
      }
    }
    this.treeSpots = spots;

    const foliageMat = std(0x3c7d41);
    addSway(foliageMat, this.time, `
      vec4 wp = instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
      float gust = sin(uTime * 0.9 - wp.x * 0.35 + wp.z * 0.1) * 0.5 + 0.5;
      float bend = max(position.y - 0.15, 0.0);
      transformed.x += (sin(uTime * 2.1 + wp.x * 1.7 + wp.z * 1.3) * 0.05 + gust * 0.06) * bend;
      transformed.z += cos(uTime * 1.7 + wp.z * 1.1) * 0.03 * bend;`);
    const trunks = new THREE.InstancedMesh(trunkGeo, std(0x6b4a2d), spots.length);
    const foliage = new THREE.InstancedMesh(foliageGeo, foliageMat, spots.length);
    const m4 = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const c = new THREE.Color();
    spots.forEach((s, k) => {
      const tx = Math.round(s.x + map.width / 2 - 0.5);
      const ty = Math.round(s.y + map.height / 2 - 0.5);
      const base = TOP[map.tiles[idx(map, Math.max(0, Math.min(W - 1, tx)), Math.max(0, Math.min(map.height - 1, ty)))].terrain];
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), s.r);
      m4.compose(new THREE.Vector3(s.x, base, s.y), q, new THREE.Vector3(s.s, s.s, s.s));
      trunks.setMatrixAt(k, m4);
      foliage.setMatrixAt(k, m4);
      c.setHex(0x3c7d41).offsetHSL((hash(k, 1) - 0.5) * 0.04, 0, (hash(k, 2) - 0.5) * 0.1);
      foliage.setColorAt(k, c);
    });
    shadowed(trunks);
    shadowed(foliage);
    this.trees = { trunks, foliage };
    this.terrainGroup.add(trunks, foliage);

    const rockGeo = new THREE.DodecahedronGeometry(1, 0);
    const rocks = new THREE.InstancedMesh(rockGeo, std(0xa39b7c), boulders.length);
    boulders.forEach((b, k) => {
      q.setFromEuler(new THREE.Euler(hash(k, 3), hash(k, 4) * 6, 0));
      m4.compose(new THREE.Vector3(b.x, b.y + b.s * 0.3, b.z), q, new THREE.Vector3(b.s * 1.3, b.s * 0.8, b.s));
      rocks.setMatrixAt(k, m4);
    });
    this.terrainGroup.add(shadowed(rocks));

    const peakGeo = new THREE.ConeGeometry(1, 1, 6);
    peakGeo.translate(0, 0.5, 0);
    const snowGeo = new THREE.ConeGeometry(1, 1, 6);
    snowGeo.translate(0, 0.5, 0);
    const peaks = new THREE.InstancedMesh(peakGeo, std(0x8b877c), mountains.length);
    const snow = new THREE.InstancedMesh(snowGeo, std(0xf4f2ec), mountains.length);
    mountains.forEach((mt, k) => {
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), hash(k, 9) * 2);
      m4.compose(new THREE.Vector3(mt.x, mt.y, mt.z), q, new THREE.Vector3(mt.s, mt.h, mt.s));
      peaks.setMatrixAt(k, m4);
      m4.compose(new THREE.Vector3(mt.x, mt.y + mt.h * 0.68, mt.z), q, new THREE.Vector3(mt.s * 0.33, mt.h * 0.33, mt.s * 0.33));
      snow.setMatrixAt(k, m4);
    });
    this.terrainGroup.add(shadowed(peaks), shadowed(snow));

    if (bushes.length) {
      const bush = new THREE.InstancedMesh(this.geo.sphereLow, std(0x3f6e2c), bushes.length);
      const berries = new THREE.InstancedMesh(this.geo.sphereLow, std(0xd4553b), bushes.length * 3);
      bushes.forEach((b, k) => {
        m4.compose(new THREE.Vector3(b.x, b.y + 0.06, b.z), q.identity(), new THREE.Vector3(0.12, 0.09, 0.12));
        bush.setMatrixAt(k, m4);
        for (let j = 0; j < 3; j++) {
          m4.compose(new THREE.Vector3(b.x + Math.cos(j * 2.1) * 0.08, b.y + 0.12, b.z + Math.sin(j * 2.1) * 0.08), q, new THREE.Vector3(0.03, 0.03, 0.03));
          berries.setMatrixAt(k * 3 + j, m4);
        }
      });
      this.terrainGroup.add(shadowed(bush), berries);
    }
    if (crystals.length) {
      const cg = new THREE.OctahedronGeometry(1, 0);
      const cr = new THREE.InstancedMesh(cg, std(0xd9f1ee, { emissive: 0x2a5a56, roughness: 0.2, metalness: 0.2 }), crystals.length);
      crystals.forEach((cc, k) => {
        q.setFromEuler(new THREE.Euler(0, 0, cc.r * 0.4));
        m4.compose(new THREE.Vector3(cc.x, cc.y + 0.1, cc.z), q, new THREE.Vector3(0.04, 0.12, 0.04));
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
      pts.push(x, 0.5, -map.height / 2, x, 0.5, map.height / 2);
      const z = i - map.height / 2;
      pts.push(-map.width / 2, 0.5, z, map.width / 2, 0.5, z);
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
      const tx = Math.round(s.x + this.map!.width / 2 - 0.5);
      const ty = Math.round(s.y + this.map!.height / 2 - 0.5);
      const hide = busy.has(`${tx},${ty}`);
      const base = this.top(tx, ty);
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), s.r);
      const sc = hide ? 0.0001 : s.s;
      m4.compose(new THREE.Vector3(s.x, base, s.y), q, new THREE.Vector3(sc, sc, sc));
      this.trees!.trunks.setMatrixAt(k, m4);
      this.trees!.foliage.setMatrixAt(k, m4);
    });
    this.trees.trunks.instanceMatrix.needsUpdate = true;
    this.trees.foliage.instanceMatrix.needsUpdate = true;
  }

  private syncTerritory(f: Frame): void {
    const map = this.map!;
    const mesh = this.territoryMesh!;
    const terr = computeTerritory(map, f.cities);
    const teamAt = (x: number, y: number): TeamId | null => {
      if (!inBounds(map, x, y)) return null;
      const ci = terr[idx(map, x, y)];
      return ci === null ? null : f.cities[ci].owner;
    };
    const m4 = new THREE.Matrix4();
    const c = new THREE.Color();
    let n = 0;
    const lines: number[] = [];
    const lineColors: number[] = [];
    for (let y = 0; y < map.height; y++) {
      for (let x = 0; x < map.width; x++) {
        const team = teamAt(x, y);
        if (!team) continue;
        const h = this.top(x, y) + 0.012;
        m4.makeTranslation(this.wx(x), h, this.wz(y));
        mesh.setMatrixAt(n, m4);
        mesh.setColorAt(n, c.set(FACTIONS[team].color));
        n++;
        const col = new THREE.Color(FACTIONS[team].color);
        const x0 = this.wx(x) - 0.5;
        const z0 = this.wz(y) - 0.5;
        const edge = (ax: number, az: number, bx: number, bz: number) => {
          lines.push(ax, h + 0.01, az, bx, h + 0.01, bz);
          lineColors.push(col.r, col.g, col.b, col.r, col.g, col.b);
        };
        if (teamAt(x, y - 1) !== team) edge(x0, z0, x0 + 1, z0);
        if (teamAt(x, y + 1) !== team) edge(x0, z0 + 1, x0 + 1, z0 + 1);
        if (teamAt(x - 1, y) !== team) edge(x0, z0, x0, z0 + 1);
        if (teamAt(x + 1, y) !== team) edge(x0 + 1, z0, x0 + 1, z0 + 1);
      }
    }
    mesh.count = n;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    if (this.territoryLines) {
      this.terrainGroup.remove(this.territoryLines);
      this.territoryLines.geometry.dispose();
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(lines, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));
    this.territoryLines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true }));
    this.territoryLines.renderOrder = 3;
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

  private makeHouse(roof: number): THREE.Group {
    const g = new THREE.Group();
    const walls = new THREE.Mesh(this.geo.box, this.mats.wall);
    walls.scale.set(0.2, 0.15, 0.17);
    walls.position.y = 0.075;
    const r = new THREE.Mesh(this.geo.cone4, std(roof));
    r.scale.set(0.17, 0.13, 0.15);
    r.rotation.y = Math.PI / 4;
    r.position.y = 0.215;
    const door = new THREE.Mesh(this.geo.box, this.mats.woodDark);
    door.scale.set(0.05, 0.08, 0.01);
    door.position.set(0, 0.04, 0.086);
    const chimney = new THREE.Mesh(this.geo.box, this.mats.stoneDark);
    chimney.scale.set(0.035, 0.1, 0.035);
    chimney.position.set(0.05, 0.25, -0.03);
    chimney.name = 'chimney';
    g.add(walls, r, door, chimney);
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
      const base = this.top(c.x, c.y);
      group.position.set(this.wx(c.x), base, this.wz(c.y));
      const plaza = new THREE.Mesh(new THREE.CylinderGeometry(0.44, 0.46, 0.03, 20), std(0xdccda5));
      plaza.position.y = 0.015;
      plaza.receiveShadow = true;
      group.add(plaza);
      const houses = Math.min(6, c.level + (c.isCapital ? 1 : 0));
      const spots: [number, number, number][] = [[-0.3, -0.25, 0.2], [0.3, -0.25, -0.3], [-0.33, 0.12, 0.5], [0.33, 0.14, -0.6], [-0.1, -0.36, 0.1], [0.12, 0.34, 0.9]];
      for (let i = 0; i < houses; i++) {
        const h = this.makeHouse(roof);
        h.position.set(spots[i][0], 0.02, spots[i][1]);
        h.rotation.y = spots[i][2];
        group.add(h);
      }
      if (c.isCapital && c.founder) {
        const fa = FACTIONS[c.founder];
        const cap = this.makeCapital(fa.capitalStyle, new THREE.Color(fa.color).getHex(), new THREE.Color(fa.colorDark).getHex(), banner);
        cap.position.y = 0.02;
        group.add(cap);
      } else {
        const flag = this.makeFlag(banner, 0.42);
        flag.position.set(0.05, 0.02, 0);
        group.add(flag);
      }
      if (c.walls) {
        for (let i = 0; i < 24; i++) {
          const a = (i / 24) * Math.PI * 2;
          const seg = new THREE.Mesh(this.geo.box, this.mats.stoneDark);
          seg.scale.set(0.12, 0.1 + (i % 2) * 0.03, 0.05);
          seg.position.set(Math.cos(a) * 0.47, 0.06, Math.sin(a) * 0.47);
          seg.rotation.y = -a + Math.PI / 2;
          group.add(shadowed(seg));
        }
      }
      if (c.warden) {
        const w = this.makeWarden();
        w.position.set(0, 0.02, 0.28);
        w.name = 'warden';
        group.add(w);
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
        add(this.geo.box, this.mats.soil, 0.8, 0.03, 0.7, 0, 0.015, 0);
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

  private makeEyes(spread: number, y: number, z: number): THREE.Mesh[] {
    const out: THREE.Mesh[] = [];
    for (const s of [-1, 1]) {
      const white = new THREE.Mesh(this.geo.eye, this.mats.white);
      white.scale.set(0.055, 0.065, 0.04);
      white.position.set(s * spread, y, z);
      const pupil = new THREE.Mesh(this.geo.eye, this.mats.pupil);
      pupil.scale.set(0.028, 0.034, 0.02);
      pupil.position.set(s * spread, y - 0.005, z + 0.03);
      pupil.name = 'pupil';
      out.push(white, pupil);
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
    const mat = new THREE.MeshStandardMaterial({ color: f.color, roughness: 0.35, metalness: 0.05, transparent: true, opacity: 1 });
    const bodyMesh = new THREE.Mesh(this.geo.blob, mat);
    bodyMesh.scale.set(0.42 * scale, 0.36 * scale, 0.4 * scale);
    bodyMesh.position.y = 0.17 * scale;
    bodyMesh.castShadow = true;
    body.add(bodyMesh);
    const shine = new THREE.Mesh(this.geo.eye, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 }));
    shine.scale.set(0.05, 0.03, 0.02);
    shine.position.set(-0.1 * scale, 0.28 * scale, 0.14 * scale);
    body.add(shine);
    for (const e of this.makeEyes(0.075 * scale, 0.22 * scale, 0.15 * scale)) body.add(e);

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
      if (c.position.x > 26) {
        c.position.x = -26;
        c.position.z = Math.random() * 30 - 15;
      }
    }
    if (this.foam) (this.foam.material as THREE.MeshBasicMaterial).opacity = 0.45 + Math.sin(t * 1.6) * 0.2;

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
      if (this.territoryMesh) this.territoryMesh.visible = this.showTerritory;
      if (this.territoryLines) this.territoryLines.visible = this.showTerritory;
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
      const ground = pos.hop > 0 ? lerp(this.groundAt(Math.floor(pos.x), Math.floor(pos.y)), this.groundAt(Math.ceil(pos.x), Math.ceil(pos.y)), 0.5) : this.top(b.x, b.y);
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
      if (warden) warden.position.y = 0.02 + Math.abs(Math.sin(t * 2 + c.x)) * 0.03;
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
    const near = this.fogNear!;
    const far = this.fogFar!;
    const key = viewer.visible ? `${this.frame?.day}|${JSON.stringify(this.view)}|${viewer.blob ? `${viewer.blob.x},${viewer.blob.y}` : ''}` : 'none';
    if (key === this.lastFogKey) return;
    this.lastFogKey = key;
    if (this.hearing) {
      this.scene.remove(this.hearing);
      this.hearing = null;
    }
    if (!viewer.visible || !viewer.explored) {
      near.count = 0;
      far.count = 0;
      return;
    }
    const m4 = new THREE.Matrix4();
    let a = 0;
    let b = 0;
    for (let y = 0; y < map.height; y++) {
      for (let x = 0; x < map.width; x++) {
        const i = idx(map, x, y);
        if (viewer.visible.has(i)) continue;
        const h = 2.2;
        m4.compose(new THREE.Vector3(this.wx(x), -0.6 + h / 2, this.wz(y)), new THREE.Quaternion(), new THREE.Vector3(1.001, h, 1.001));
        if (viewer.explored(i)) near.setMatrixAt(a++, m4);
        else far.setMatrixAt(b++, m4);
      }
    }
    near.count = a;
    far.count = b;
    near.instanceMatrix.needsUpdate = true;
    far.instanceMatrix.needsUpdate = true;
    if (viewer.blob) {
      const r = HEARING_RANGE + 0.5;
      const cx = this.wx(viewer.blob.x);
      const cz = this.wz(viewer.blob.y);
      const pts = [[-r, -r], [r, -r], [r, r], [-r, r]].map(([dx, dz]) => new THREE.Vector3(cx + dx, 0.55, cz + dz));
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      this.hearing = new THREE.LineLoop(geo, new THREE.LineDashedMaterial({ color: 0xebe6da, dashSize: 0.2, gapSize: 0.14 }));
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
    const targets: THREE.Object3D[] = [this.landMesh!, this.seabedMesh!];
    const hit = ray.intersectObjects(targets, false)[0];
    let tile: number | null = null;
    if (hit && hit.instanceId !== undefined) {
      tile = hit.object === this.landMesh ? this.tileOfInstance[hit.instanceId] : this.seabedTileOfInstance[hit.instanceId];
    }
    if (tile === null) return null;
    const x = tile % map.width;
    const y = Math.floor(tile / map.width);
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
