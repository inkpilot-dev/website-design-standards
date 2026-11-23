/**
 * Core data model for Goo Kingdoms.
 *
 * Everything the simulation, the agents, the server and the browser share is
 * declared here. All of it is plain JSON-serialisable data so a frame can be
 * sent over the wire or stored in a replay without conversion.
 */

export type TeamId = 'crimson' | 'cobalt' | 'amber' | 'emerald';
export const TEAM_IDS: readonly TeamId[] = ['crimson', 'cobalt', 'amber', 'emerald'];

export type Terrain = 'sea' | 'shallows' | 'sand' | 'grass' | 'forest' | 'hills' | 'mountain';
export type Resource = 'fruit' | 'crystal' | null;

export interface Tile {
  terrain: Terrain;
  /** Fruit sits on grass, crystal on hills. Both make a development worth more. */
  resource: Resource;
}

export interface WorldMap {
  width: number;
  height: number;
  seed: number;
  /** Row-major: index = y * width + x. */
  tiles: Tile[];
  /** Precomputed 0..1 water depth per tile (0 on land) for the renderer. */
  depth: number[];
}

export type DevKind = 'farm' | 'lumber' | 'mine' | 'tower' | 'trap' | 'harbor';

export interface Development {
  x: number;
  y: number;
  kind: DevKind;
  /** Team that built it. Farms, mines, etc. belong to whoever owns the territory. */
  builder: TeamId;
  /** Traps: armed until sprung. After springing they leave a pit for a few days. */
  armed: boolean;
  /** Day the pit closes (traps only, once sprung). */
  pitUntil: number;
  builtOn: number;
}

export interface Warden {
  hp: number;
  maxHp: number;
}

export interface City {
  id: string;
  name: string;
  x: number;
  y: number;
  owner: TeamId | null;
  /** The team whose capital this is, or null for a village. */
  founder: TeamId | null;
  isCapital: boolean;
  island: boolean;
  level: number;
  pop: number;
  walls: boolean;
  /** Neutral villages are watched by a warden until someone beats it. */
  warden: Warden | null;
  /** Day the city last grew a level (drives the gold ring effect). */
  levelUpOn: number;
  capturedOn: number;
}

export type Rank = 'recruit' | 'veteran' | 'champion';
export type Gear = 'spear' | 'bow' | 'shield';
export type Vessel = 'raft' | 'gunboat';

export interface Blob {
  id: string;
  name: string;
  team: TeamId;
  x: number;
  y: number;
  hp: number;
  maxHp: number;
  rank: Rank;
  kills: number;
  gear: Gear | null;
  vessel: Vessel | null;
  bornOn: number;
  /** Team the blob was born into, kept if it ever changes sides. */
  origin: TeamId;
  /** Short memory of the blob's own recent decisions (shown in the inspector). */
  memory: MemoryEntry[];
}

export interface MemoryEntry {
  day: number;
  action: Action;
  ok: boolean;
  note: string;
  thought: string;
  say: string;
}

export interface Boss {
  name: string;
  x: number;
  y: number;
  hp: number;
  maxHp: number;
  alive: boolean;
  /** Day it rises. Before then it is announced but not on the map. */
  risesOn: number;
  risen: boolean;
}

export interface LootPile {
  id: string;
  x: number;
  y: number;
  coins: number;
}

export type Tech =
  | 'forestry'
  | 'mining'
  | 'archery'
  | 'smithing'
  | 'sailing'
  | 'fortification'
  | 'navigation';

export interface TeamStats {
  kills: number;
  deaths: number;
  citiesCaptured: number;
  citiesLost: number;
  coinsEarned: number;
  coinsSpent: number;
  coinsPaid: number;
  messages: number;
  alliancesFormed: number;
  decisions: number;
  validActions: number;
  apiErrors: number;
  inputTokens: number;
  outputTokens: number;
  peakTerritory: number;
}

export interface TeamState {
  id: TeamId;
  coins: number;
  techs: Tech[];
  allies: TeamId[];
  /** Teams that have offered this team an alliance and are waiting for an answer. */
  offersFrom: TeamId[];
  remnant: boolean;
  eliminated: boolean;
  eliminatedOn: number;
  /** Tiles this team has ever seen, as a string of '0'/'1' (compact and JSON friendly). */
  explored: string;
  /** Last known owner of each city this team has seen (city id -> owner or 'none'). */
  cityIntel: Record<string, TeamId | 'none'>;
  stats: TeamStats;
  nextBlobNo: number;
}

/** A line spoken by a blob. Heard by every blob within HEARING_RANGE tiles. */
export interface Speech {
  day: number;
  blobId: string;
  name: string;
  team: TeamId;
  x: number;
  y: number;
  text: string;
}

// ---------------------------------------------------------------------------
// Actions. One per blob per day. Anything malformed becomes `idle`.
// ---------------------------------------------------------------------------

export type Action =
  | { type: 'idle' }
  | { type: 'move'; x: number; y: number }
  | { type: 'attack'; target: string }
  | { type: 'capture' }
  | { type: 'develop'; x: number; y: number; kind: DevKind }
  | { type: 'craft'; item: Gear }
  | { type: 'research'; tech: Tech }
  | { type: 'recruit' }
  | { type: 'fortify' }
  | { type: 'refit' }
  | { type: 'rest' }
  | { type: 'ally_propose'; team: TeamId }
  | { type: 'ally_accept'; team: TeamId }
  | { type: 'ally_leave'; team: TeamId }
  | { type: 'pay'; team: TeamId; amount: number };

export type ActionType = Action['type'];

export const ACTION_TYPES: readonly ActionType[] = [
  'idle', 'move', 'attack', 'capture', 'develop', 'craft', 'research', 'recruit',
  'fortify', 'refit', 'rest', 'ally_propose', 'ally_accept', 'ally_leave', 'pay',
];

/** What an agent hands back for one blob on one day. */
export interface Decision {
  action: Action;
  say: string;
  thought: string;
  /** False when the raw response was missing or malformed and we fell back to idle. */
  wellFormed: boolean;
  source: 'claude' | 'local' | 'fallback';
  inputTokens?: number;
  outputTokens?: number;
  error?: string;
}

// ---------------------------------------------------------------------------
// Events: what happened during a day, used for the log and for animation.
// ---------------------------------------------------------------------------

export type GameEvent =
  | { kind: 'move'; id: string; path: [number, number][]; vessel: Vessel | null }
  | { kind: 'embark'; id: string; x: number; y: number }
  | { kind: 'land'; id: string; x: number; y: number }
  | { kind: 'attack'; from: string; to: string; fx: number; fy: number; tx: number; ty: number; dmg: number; ranged: boolean; weapon: string }
  | { kind: 'death'; id: string; name: string; team: TeamId | null; x: number; y: number; by: string }
  | { kind: 'capture'; city: string; cityName: string; team: TeamId; from: TeamId | null; x: number; y: number }
  | { kind: 'levelup'; city: string; level: number; x: number; y: number }
  | { kind: 'develop'; team: TeamId; devKind: DevKind; x: number; y: number }
  | { kind: 'trap'; x: number; y: number; victim: string; dmg: number }
  | { kind: 'loot'; x: number; y: number; team: TeamId; coins: number; by: string }
  | { kind: 'loot_spawn'; x: number; y: number; coins: number }
  | { kind: 'promote'; id: string; name: string; rank: Rank; team: TeamId }
  | { kind: 'recruit'; id: string; name: string; team: TeamId; x: number; y: number }
  | { kind: 'research'; team: TeamId; tech: Tech }
  | { kind: 'craft'; id: string; item: Gear }
  | { kind: 'fortify'; city: string; team: TeamId }
  | { kind: 'refit'; id: string }
  | { kind: 'ally'; a: TeamId; b: TeamId }
  | { kind: 'ally_offer'; from: TeamId; to: TeamId }
  | { kind: 'ally_break'; a: TeamId; b: TeamId }
  | { kind: 'pay'; from: TeamId; to: TeamId; amount: number }
  | { kind: 'say'; id: string; text: string; x: number; y: number }
  | { kind: 'boss_warning'; name: string; risesOn: number }
  | { kind: 'boss_rise'; name: string; x: number; y: number }
  | { kind: 'boss_move'; path: [number, number][] }
  | { kind: 'boss_slam'; x: number; y: number }
  | { kind: 'boss_dead'; name: string; team: TeamId; by: string; bounty: number }
  | { kind: 'raze'; city: string; pop: number }
  | { kind: 'remnant'; team: TeamId }
  | { kind: 'eliminated'; team: TeamId }
  | { kind: 'supremacy'; team: TeamId; days: number }
  | { kind: 'victory'; team: TeamId | null; reason: VictoryReason };

export type VictoryReason = 'last_standing' | 'supremacy' | 'day_limit' | 'draw';

export interface LogLine {
  day: number;
  text: string;
  team: TeamId | null;
  important: boolean;
}

// ---------------------------------------------------------------------------
// Whole game state and the per-day frame the browser receives.
// ---------------------------------------------------------------------------

export interface GameConfig {
  seed: number;
  maxDays: number;
}

export interface GameState {
  config: GameConfig;
  map: WorldMap;
  day: number;
  cities: City[];
  blobs: Blob[];
  devs: Development[];
  boss: Boss;
  loot: LootPile[];
  teams: Record<TeamId, TeamState>;
  speech: Speech[];
  /** Consecutive days a single team has held supremacy, and which team. */
  supremacy: { team: TeamId | null; days: number };
  winner: TeamId | null;
  victory: VictoryReason | null;
  over: boolean;
  nextId: number;
  log: LogLine[];
}

/** What changes day to day. The map is sent once. */
export interface Frame {
  day: number;
  cities: City[];
  blobs: Blob[];
  devs: Development[];
  boss: Boss;
  loot: LootPile[];
  teams: Record<TeamId, TeamState>;
  speech: Speech[];
  supremacy: { team: TeamId | null; days: number };
  winner: TeamId | null;
  victory: VictoryReason | null;
  over: boolean;
  events: GameEvent[];
  log: LogLine[];
  /** The decision each blob made this day, keyed by blob id (for the inspector). */
  decisions: Record<string, Decision>;
}
