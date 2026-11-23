/**
 * Grid helpers shared by the simulation and the renderer: distances,
 * terrain checks, vision and territory.
 */
import type { Blob, City, TeamId, Terrain, WorldMap } from './types.ts';
import { HILL_VISION_BONUS, TERRAIN_COST, VISION, territoryRadius } from './rules.ts';

export const idx = (map: WorldMap, x: number, y: number): number => y * map.width + x;

export const inBounds = (map: WorldMap, x: number, y: number): boolean =>
  x >= 0 && y >= 0 && x < map.width && y < map.height;

/** Chebyshev distance: diagonal steps count as one tile. */
export const dist = (ax: number, ay: number, bx: number, by: number): number =>
  Math.max(Math.abs(ax - bx), Math.abs(ay - by));

export const terrainAt = (map: WorldMap, x: number, y: number): Terrain =>
  map.tiles[idx(map, x, y)].terrain;

export const isWater = (t: Terrain): boolean => t === 'sea' || t === 'shallows';
export const isWalkable = (t: Terrain): boolean => TERRAIN_COST[t] !== null;

export const NEIGHBOURS: readonly [number, number][] = [
  [-1, -1], [0, -1], [1, -1],
  [-1, 0], [1, 0],
  [-1, 1], [0, 1], [1, 1],
];

export function visionRadius(map: WorldMap, b: Pick<Blob, 'x' | 'y' | 'vessel'>): number {
  if (b.vessel) return VISION;
  return terrainAt(map, b.x, b.y) === 'hills' ? VISION + HILL_VISION_BONUS : VISION;
}

/** Tile indices a blob can see this day (a square of its vision radius). */
export function visibleTiles(map: WorldMap, b: Pick<Blob, 'x' | 'y' | 'vessel'>): number[] {
  const r = visionRadius(map, b);
  const out: number[] = [];
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const x = b.x + dx;
      const y = b.y + dy;
      if (inBounds(map, x, y)) out.push(idx(map, x, y));
    }
  }
  return out;
}

/**
 * Which city owns each tile. Water is never territory. When two cities reach
 * the same tile, the nearer one wins and ties go to the older city (lower
 * index in the city list). Neutral villages claim radius 1 with no team.
 */
export function computeTerritory(map: WorldMap, cities: City[]): (number | null)[] {
  const owner: (number | null)[] = new Array(map.tiles.length).fill(null);
  const best: number[] = new Array(map.tiles.length).fill(Infinity);
  cities.forEach((c, ci) => {
    const r = c.owner ? territoryRadius(c.level) : 1;
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const x = c.x + dx;
        const y = c.y + dy;
        if (!inBounds(map, x, y)) continue;
        const i = idx(map, x, y);
        if (isWater(map.tiles[i].terrain)) continue;
        const d = dist(c.x, c.y, x, y);
        if (d < best[i]) {
          best[i] = d;
          owner[i] = ci;
        }
      }
    }
  });
  return owner;
}

/** Team that owns a tile through its territory, or null. */
export function tileTeam(territory: (number | null)[], cities: City[], i: number): TeamId | null {
  const ci = territory[i];
  return ci === null ? null : cities[ci].owner;
}

/** Step-by-step path over walkable tiles (BFS, 8 directions). Returns [] if unreachable. */
export function findPath(
  map: WorldMap,
  fromX: number,
  fromY: number,
  toX: number,
  toY: number,
  passable: (x: number, y: number) => boolean,
): [number, number][] {
  if (fromX === toX && fromY === toY) return [];
  const start = idx(map, fromX, fromY);
  const goal = idx(map, toX, toY);
  const prev = new Int32Array(map.tiles.length).fill(-1);
  prev[start] = start;
  const queue: number[] = [start];
  for (let qi = 0; qi < queue.length; qi++) {
    const cur = queue[qi];
    if (cur === goal) break;
    const cx = cur % map.width;
    const cy = Math.floor(cur / map.width);
    for (const [dx, dy] of NEIGHBOURS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (!inBounds(map, nx, ny)) continue;
      const ni = idx(map, nx, ny);
      if (prev[ni] !== -1) continue;
      if (ni !== goal && !passable(nx, ny)) continue;
      if (ni === goal && !passable(nx, ny)) continue;
      prev[ni] = cur;
      queue.push(ni);
    }
  }
  if (prev[goal] === -1) return [];
  const path: [number, number][] = [];
  for (let cur = goal; cur !== start; cur = prev[cur]) {
    path.push([cur % map.width, Math.floor(cur / map.width)]);
  }
  return path.reverse();
}

/** Decode a team's explored string into a boolean lookup. */
export function exploredSet(explored: string): (i: number) => boolean {
  return (i: number) => explored.charCodeAt(i) === 49;
}
