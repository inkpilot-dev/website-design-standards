/**
 * Seeded world generation: one large island, a few islets, terrain, four
 * capitals (one per quarter), neutral villages and resources.
 *
 * The same seed always gives the same world.
 */
import type { City, TeamId, Terrain, Tile, WorldMap } from '../shared/types.ts';
import { Rng, mixSeed } from '../shared/rng.ts';
import { NEIGHBOURS, dist, idx, inBounds, isWalkable, isWater } from '../shared/geo.ts';
import { ISLAND_WARDEN_HP, WARDEN_HP, islandVillages, neutralVillages } from '../shared/rules.ts';

export const CAPITAL_NAMES: Record<TeamId, string> = {
  crimson: 'Scarlet Keep',
  cobalt: 'Azure Citadel',
  amber: 'Goldcrest Hall',
  emerald: 'Verdant Hold',
};

const NAME_HEADS = ['Bram', 'Cole', 'Dun', 'Fen', 'Glim', 'Hollow', 'Kettle', 'Lark', 'Moss', 'Nib', 'Oak', 'Pud', 'Quill', 'Rush', 'Sorrel', 'Tumble', 'Wick', 'Yarrow', 'Bris', 'Thistle', 'Marl', 'Pebble'];
const NAME_TAILS = ['by', 'well', 'stead', 'ford', 'mere', 'holm', 'wick', 'ton', 'dale', 'cot', 'end', 'brook'];

/** Smooth value noise on a coarse random lattice. */
function valueNoise(rng: Rng, size: number, cell: number): (x: number, y: number) => number {
  const n = Math.ceil(size / cell) + 2;
  const lattice: number[] = [];
  for (let i = 0; i < n * n; i++) lattice.push(rng.next());
  const smooth = (t: number) => t * t * (3 - 2 * t);
  return (x: number, y: number) => {
    const gx = x / cell;
    const gy = y / cell;
    const x0 = Math.floor(gx);
    const y0 = Math.floor(gy);
    const tx = smooth(gx - x0);
    const ty = smooth(gy - y0);
    const at = (ix: number, iy: number) => lattice[iy * n + ix];
    const a = at(x0, y0) * (1 - tx) + at(x0 + 1, y0) * tx;
    const b = at(x0, y0 + 1) * (1 - tx) + at(x0 + 1, y0 + 1) * tx;
    return a * (1 - ty) + b * ty;
  };
}

/** Connected groups of walkable tiles (8 directions). */
function components(map: WorldMap): number[][] {
  const seen = new Uint8Array(map.tiles.length);
  const groups: number[][] = [];
  for (let i = 0; i < map.tiles.length; i++) {
    if (seen[i] || !isWalkable(map.tiles[i].terrain)) continue;
    const group: number[] = [];
    const queue = [i];
    seen[i] = 1;
    while (queue.length) {
      const cur = queue.pop()!;
      group.push(cur);
      const cx = cur % map.width;
      const cy = Math.floor(cur / map.width);
      for (const [dx, dy] of NEIGHBOURS) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (!inBounds(map, nx, ny)) continue;
        const ni = idx(map, nx, ny);
        if (seen[ni] || !isWalkable(map.tiles[ni].terrain)) continue;
        seen[ni] = 1;
        queue.push(ni);
      }
    }
    groups.push(group);
  }
  return groups.sort((a, b) => b.length - a.length);
}

function nameGenerator(rng: Rng): () => string {
  const used = new Set<string>(Object.values(CAPITAL_NAMES));
  return () => {
    for (let tries = 0; tries < 200; tries++) {
      const name = rng.pick(NAME_HEADS) + rng.pick(NAME_TAILS);
      if (!used.has(name)) {
        used.add(name);
        return name;
      }
    }
    return `Village ${used.size}`;
  };
}

export interface GeneratedWorld {
  map: WorldMap;
  cities: City[];
  capitals: Record<TeamId, City>;
}

export function generateWorld(seed: number, size: number): GeneratedWorld {
  const rng = new Rng(mixSeed(seed, 1));
  const W = size;
  const H = size;
  const k = size / 20;
  const ISLAND_VILLAGES = islandVillages(size);
  const NEUTRAL_VILLAGES = neutralVillages(size);

  // Feature sizes grow with the map, with extra detail on big maps for bays and headlands.
  const shape = valueNoise(rng, W, 5 * Math.sqrt(k) * 1.2);
  const detail = valueNoise(rng, W, 2.5 * Math.sqrt(k));
  const ridge = valueNoise(rng, W, 4 * Math.sqrt(k));
  const woods = valueNoise(rng, W, 3 * Math.sqrt(k));
  const bays = valueNoise(rng, W, 3.2 * Math.sqrt(k));

  // Island shape: noise plus a soft falloff from a slightly off-centre point.
  const cx = W / 2 - 0.5 + (rng.next() - 0.5) * 2;
  const cy = H / 2 - 0.5 + (rng.next() - 0.5) * 2;
  const height: number[] = [];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const dx = (x - cx) / (W * 0.52);
      const dy = (y - cy) / (H * 0.52);
      const falloff = 1 - Math.sqrt(dx * dx + dy * dy);
      height.push(falloff * 0.9 + shape(x, y) * 0.45 + detail(x, y) * 0.15 + (k > 1.2 ? (bays(x, y) - 0.5) * 0.25 : 0));
    }
  }
  // About 58% land: pick the threshold from the sorted heights.
  const sorted = [...height].sort((a, b) => a - b);
  const seaLevel = sorted[Math.floor(sorted.length * 0.42)];
  const landHeights = height.filter((h) => h > seaLevel).sort((a, b) => a - b);
  const hillLevel = landHeights[Math.floor(landHeights.length * 0.78)];

  const tiles: Tile[] = height.map((h, i) => {
    const x = i % W;
    const y = Math.floor(i / W);
    const edge = x === 0 || y === 0 || x === W - 1 || y === H - 1;
    if (h <= seaLevel || edge) return { terrain: 'sea', resource: null };
    const r = ridge(x, y);
    if (h > hillLevel && r > 0.62) return { terrain: 'mountain', resource: null };
    if (h > hillLevel || r > 0.8) return { terrain: 'hills', resource: null };
    if (woods(x, y) > 0.56) return { terrain: 'forest', resource: null };
    return { terrain: 'grass', resource: null };
  });

  const map: WorldMap = { width: W, height: H, seed, tiles, depth: new Array(W * H).fill(0) };

  // Make sure there are islets for the island villages.
  let groups = components(map);
  const islets = () => groups.slice(1).filter((g) => g.length >= 2);
  for (let attempt = 0; islets().length < ISLAND_VILLAGES && attempt < 200; attempt++) {
    const x = rng.int(2, W - 4);
    const y = rng.int(2, H - 4);
    let clear = true;
    for (let dy = -2; dy <= 3 && clear; dy++) {
      for (let dx = -2; dx <= 3 && clear; dx++) {
        if (!isWater(map.tiles[idx(map, x + dx, y + dy)].terrain)) clear = false;
      }
    }
    if (!clear) continue;
    const isletTiles = k > 1.5 ? [[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2]] : [[0, 0], [1, 0], [0, 1], [1, 1]];
    for (const [dx, dy] of isletTiles) {
      if (x + dx >= W - 1 || y + dy >= H - 1) continue;
      map.tiles[idx(map, x + dx, y + dy)] = { terrain: rng.next() < 0.3 ? 'forest' : 'grass', resource: null };
    }
    groups = components(map);
  }

  // Coastline: shallows next to land, sand on low land next to water.
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const t = map.tiles[idx(map, x, y)];
      let nearLand = false;
      let nearWater = false;
      for (const [dx, dy] of NEIGHBOURS) {
        if (!inBounds(map, x + dx, y + dy)) continue;
        const nt = map.tiles[idx(map, x + dx, y + dy)].terrain;
        if (isWater(nt)) nearWater = true;
        else nearLand = true;
      }
      if (t.terrain === 'sea' && nearLand) t.terrain = 'shallows';
      else if ((t.terrain === 'grass' || t.terrain === 'forest') && nearWater && rng.next() < 0.55) t.terrain = 'sand';
    }
  }

  // Water depth for the renderer (distance from land, 0..1).
  const depthSteps = new Array(W * H).fill(-1);
  const queue: number[] = [];
  map.tiles.forEach((t, i) => {
    if (!isWater(t.terrain)) {
      depthSteps[i] = 0;
      queue.push(i);
    }
  });
  for (let qi = 0; qi < queue.length; qi++) {
    const cur = queue[qi];
    const x = cur % W;
    const y = Math.floor(cur / W);
    for (const [dx, dy] of NEIGHBOURS) {
      if (!inBounds(map, x + dx, y + dy)) continue;
      const ni = idx(map, x + dx, y + dy);
      if (depthSteps[ni] !== -1) continue;
      depthSteps[ni] = depthSteps[cur] + 1;
      queue.push(ni);
    }
  }
  map.depth = depthSteps.map((d) => Math.min(1, Math.max(0, d) / 4));

  groups = components(map);
  const mainland = new Set(groups[0]);

  // Capitals: one per quarter, as close to each quarter's centre as possible,
  // on grass or sand, with some walkable land around them.
  const teamOrder = rng.shuffle<TeamId>(['crimson', 'cobalt', 'amber', 'emerald']);
  const anchors: [number, number][] = [[0.26, 0.26], [0.74, 0.26], [0.74, 0.74], [0.26, 0.74]].map(([ax, ay]) => [ax * (W - 1), ay * (H - 1)] as [number, number]);
  const capitalGap = Math.max(8, Math.round(size * 0.38));
  const capitals = {} as Record<TeamId, City>;
  const cities: City[] = [];
  const okCapital = (i: number) => {
    const t = map.tiles[i].terrain;
    if (!mainland.has(i) || (t !== 'grass' && t !== 'sand' && t !== 'forest')) return false;
    const x = i % W;
    const y = Math.floor(i / W);
    let walk = 0;
    for (const [dx, dy] of NEIGHBOURS) {
      if (inBounds(map, x + dx, y + dy) && isWalkable(map.tiles[idx(map, x + dx, y + dy)].terrain)) walk++;
    }
    return walk >= 5;
  };
  teamOrder.forEach((team, q) => {
    const [ax, ay] = anchors[q];
    let best = -1;
    let bestScore = Infinity;
    for (const i of mainland) {
      if (!okCapital(i)) continue;
      const x = i % W;
      const y = Math.floor(i / W);
      if (cities.some((c) => dist(c.x, c.y, x, y) < capitalGap)) continue;
      const score = Math.hypot(x - ax, y - ay) + rng.next() * 0.5;
      if (score < bestScore) {
        bestScore = score;
        best = i;
      }
    }
    if (best === -1) {
      // Fallback: nearest mainland tile to the anchor, whatever the spacing.
      for (const i of mainland) {
        const s = Math.hypot((i % W) - ax, Math.floor(i / W) - ay);
        if (s < bestScore && !cities.some((c) => idx(map, c.x, c.y) === i)) {
          bestScore = s;
          best = i;
        }
      }
    }
    map.tiles[best] = { terrain: 'grass', resource: null };
    const city: City = {
      id: `c${cities.length}`,
      name: CAPITAL_NAMES[team],
      x: best % W,
      y: Math.floor(best / W),
      owner: team,
      founder: team,
      isCapital: true,
      island: false,
      level: 1,
      pop: 0,
      walls: false,
      warden: null,
      levelUpOn: -1,
      capturedOn: 0,
    };
    capitals[team] = city;
    cities.push(city);
  });

  // Mainland villages, spread out.
  const nextName = nameGenerator(rng);
  const placeVillage = (pool: number[], minGap: number, island: boolean) => {
    const candidates = rng.shuffle([...pool]).filter((i) => {
      const t = map.tiles[i].terrain;
      return t === 'grass' || t === 'sand' || t === 'forest';
    });
    for (const i of candidates) {
      const x = i % W;
      const y = Math.floor(i / W);
      if (cities.some((c) => dist(c.x, c.y, x, y) < minGap)) continue;
      map.tiles[i] = { terrain: map.tiles[i].terrain === 'sand' ? 'sand' : 'grass', resource: null };
      const hp = island ? ISLAND_WARDEN_HP : WARDEN_HP;
      cities.push({
        id: `c${cities.length}`,
        name: nextName(),
        x,
        y,
        owner: null,
        founder: null,
        isCapital: false,
        island,
        level: 1,
        pop: 0,
        walls: false,
        warden: { hp, maxHp: hp },
        levelUpOn: -1,
        capturedOn: -1,
      });
      return true;
    }
    return false;
  };
  for (let v = 0; v < NEUTRAL_VILLAGES; v++) {
    const gap = Math.max(4, Math.round(size / 9));
    if (!placeVillage(groups[0], gap, false) && !placeVillage(groups[0], gap - 1, false)) placeVillage(groups[0], 3, false);
  }
  islets()
    .slice(0, ISLAND_VILLAGES)
    .forEach((g) => placeVillage(g, 2, true));

  // Resources: fruit on some grass, crystal on some hills (never under a city).
  const cityTiles = new Set(cities.map((c) => idx(map, c.x, c.y)));
  map.tiles.forEach((t, i) => {
    if (cityTiles.has(i)) return;
    if (t.terrain === 'grass' && rng.next() < 0.14) t.resource = 'fruit';
    if (t.terrain === 'hills' && rng.next() < 0.3) t.resource = 'crystal';
  });

  return { map, cities, capitals };
}

/** Terrain the renderer and prompts use for single-character map views. */
export const TERRAIN_CHAR: Record<Terrain, string> = {
  sea: '~',
  shallows: '-',
  sand: ':',
  grass: '.',
  forest: 'f',
  hills: 'h',
  mountain: 'M',
};
