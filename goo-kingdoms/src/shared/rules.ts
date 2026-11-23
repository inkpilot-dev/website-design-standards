/**
 * Every number in the game lives here, so the engine, the agents' prompts and
 * the in-app rulebook can never disagree.
 */
import type { DevKind, Gear, Rank, Tech, Terrain } from './types.ts';

/** Map edge length in tiles. Matches can be played on any of these sizes. */
export const MAP_SIZES = [20, 32, 48, 64] as const;
export const DEFAULT_MAP_SIZE = 48;
export const DEFAULT_MAX_DAYS = 300;
export const MAX_DAYS_LIMIT = 1000;

// Starting position -------------------------------------------------------
export const START_COINS = 15;
export const START_BLOBS = 3;
/** Neutral villages on the mainland, scaled to the map's area. */
export function neutralVillages(size: number): number {
  return Math.max(6, Math.round((size * size) / 100));
}
/** Villages on islets, reachable only by boat. */
export function islandVillages(size: number): number {
  return Math.max(2, Math.round(size / 12));
}
/** Loot piles that wash up each time loot arrives, scaled to the map's area. */
export function lootPerWave(size: number): number {
  return Math.max(1, Math.round((size * size) / 700));
}

// Blobs -------------------------------------------------------------------
export const BLOB_HP = 10;
export const BLOB_ATTACK = 3;
export const MOVE_POINTS = 2;
export const VESSEL_MOVES = 3;
export const VISION = 2;
export const HILL_VISION_BONUS = 1;
export const HEARING_RANGE = 3;
export const SAY_MAX_CHARS = 120;

/** Movement cost per terrain on foot. `null` means you cannot walk there. */
export const TERRAIN_COST: Record<Terrain, number | null> = {
  sea: null,
  shallows: null,
  sand: 1,
  grass: 1,
  forest: 2,
  hills: 2,
  mountain: null,
};

/** Extra armour a defender gets from the tile it stands on. */
export const TERRAIN_COVER: Record<Terrain, number> = {
  sea: 0, shallows: 0, sand: 0, grass: 0, forest: 1, hills: 1, mountain: 0,
};

export const RANKS: Record<Rank, { kills: number; hpBonus: number; attackBonus: number }> = {
  recruit: { kills: 0, hpBonus: 0, attackBonus: 0 },
  veteran: { kills: 2, hpBonus: 3, attackBonus: 1 },
  champion: { kills: 5, hpBonus: 6, attackBonus: 2 },
};

export const GEAR: Record<Gear, { cost: number; tech: Tech; attack: number; armour: number; range: number; label: string }> = {
  spear: { cost: 9, tech: 'smithing', attack: 2, armour: 0, range: 1, label: 'Spear: +2 attack' },
  shield: { cost: 8, tech: 'smithing', attack: 0, armour: 1, range: 1, label: 'Shield: +1 armour' },
  bow: { cost: 9, tech: 'archery', attack: -1, armour: 0, range: 2, label: 'Bow: attack at range 2, -1 attack' },
};

export const GUNBOAT = { cost: 16, tech: 'navigation' as Tech, attack: 4, range: 2 };
export const RAFT_ATTACK = 2;

// Economy -----------------------------------------------------------------
export const RECRUIT_COST = 10;
/** A team may field at most this many blobs: base + per city, capped. */
export const BLOB_CAP_BASE = 3;
export const BLOB_CAP_PER_CITY = 2;
export const BLOB_CAP_MAX = 16;

export const WALLS_COST = 15;
export const WALLS_ARMOUR = 1;

/** Population needed to reach each level (index = level). Level 1 needs 0. */
export const LEVEL_POP = [0, 0, 3, 7, 12, 18, 25];
export const MAX_CITY_LEVEL = 6;

/** Coins per day. A capital pays its founder more than it pays a conqueror. */
export function cityIncome(level: number, isCapital: boolean, ownerIsFounder: boolean): number {
  if (isCapital && ownerIsFounder) return 1 + level;
  return level;
}

/** Territory radius around a city. */
export function territoryRadius(level: number): number {
  return level >= 3 ? 2 : 1;
}

export const DEVELOPMENTS: Record<DevKind, {
  cost: number;
  tech: Tech | null;
  terrain: Terrain[];
  pop: number;
  resourceBonus: number;
  label: string;
}> = {
  farm: { cost: 5, tech: null, terrain: ['grass', 'sand'], pop: 1, resourceBonus: 1, label: 'Farm: +1 pop (+1 more on fruit)' },
  lumber: { cost: 5, tech: 'forestry', terrain: ['forest'], pop: 1, resourceBonus: 0, label: 'Lumber camp: +1 pop' },
  mine: { cost: 8, tech: 'mining', terrain: ['hills'], pop: 2, resourceBonus: 1, label: 'Mine: +2 pop (+1 more on crystal)' },
  tower: { cost: 12, tech: 'fortification', terrain: ['grass', 'sand', 'hills', 'forest'], pop: 0, resourceBonus: 0, label: 'Tower: shoots the nearest enemy in range 2 for 2' },
  trap: { cost: 4, tech: 'forestry', terrain: ['grass', 'sand', 'forest', 'hills'], pop: 0, resourceBonus: 0, label: 'Trap: hidden from enemies, 5 damage and stops them' },
  harbor: { cost: 8, tech: 'sailing', terrain: ['sand', 'grass'], pop: 1, resourceBonus: 0, label: 'Harbor: +1 pop, lets blobs set sail' },
};

export const TOWER_DAMAGE = 2;
export const TOWER_RANGE = 2;
export const TRAP_DAMAGE = 5;
export const PIT_DAYS = 5;

export const TECHS: Record<Tech, { cost: number; requires: Tech | null; unlocks: string }> = {
  forestry: { cost: 8, requires: null, unlocks: 'lumber camps and traps' },
  smithing: { cost: 12, requires: null, unlocks: 'spears and shields' },
  archery: { cost: 12, requires: null, unlocks: 'bows' },
  sailing: { cost: 12, requires: null, unlocks: 'harbors and rafts' },
  mining: { cost: 14, requires: 'forestry', unlocks: 'mines' },
  fortification: { cost: 18, requires: 'mining', unlocks: 'towers and city walls' },
  navigation: { cost: 18, requires: 'sailing', unlocks: 'gunboat refits' },
};

/** A team with cities but no blobs musters one free blob at its best city every this many days. */
export const MUSTER_EVERY = 3;

// Healing -----------------------------------------------------------------
export const REST_HEAL = 3;
export const REST_HEAL_TERRITORY_BONUS = 2;
export const CITY_PASSIVE_HEAL = 1;

// Neutral wardens ---------------------------------------------------------
export const WARDEN_HP = 8;
export const ISLAND_WARDEN_HP = 10;
export const WARDEN_ATTACK = 2;

// World -------------------------------------------------------------------
export const LOOT_EVERY = 12;
export const LOOT_MIN = 8;
export const LOOT_MAX = 16;

export const BOSS_NAMES = ['Grumblemire', 'Old Sludge', 'The Bog King', 'Mudmaw', 'Brackenbelly'];
export const BOSS_WARNING_DAYS = 8;
export const BOSS_FIRST_RISE = 60;
export const BOSS_HP = 80;
export const BOSS_ARMOUR = 2;
export const BOSS_SLAM = 5;
export const BOSS_RAZE = 2;
export const BOSS_BOUNTY = 30;

// Victory -----------------------------------------------------------------
/** A team that holds at least this share of all cities for SUPREMACY_DAYS straight days wins. */
export const SUPREMACY_SHARE = 2 / 3;
export const SUPREMACY_DAYS = 20;

export function blobCap(cities: number): number {
  return Math.min(BLOB_CAP_MAX, BLOB_CAP_BASE + BLOB_CAP_PER_CITY * cities);
}

export function levelForPop(pop: number): number {
  let level = 1;
  for (let l = 2; l <= MAX_CITY_LEVEL; l++) if (pop >= LEVEL_POP[l]) level = l;
  return level;
}
