/**
 * The Goo Kingdoms simulation.
 *
 * A day (one tick) resolves every living blob's single action in a freshly
 * shuffled order, through five phases:
 *
 *   1. Council  - alliances and payments.
 *   2. March    - movement, one blob at a time (traps, loot, boarding, landing).
 *   3. Clash    - every attack is planned first, then all damage lands at once,
 *                 so two blobs can kill each other on the same day.
 *   4. Works    - develop, craft, research, recruit, fortify, refit, rest.
 *                 Captures resolve last, so a defender killed in the Clash no
 *                 longer blocks them.
 *   5. Dusk     - speech, the world (boss, loot, pits, promotions), then upkeep
 *                 (income, healing, vision, remnants, eliminations, victory).
 *
 * The engine never trusts an action: anything illegal is recorded as an
 * invalid action and the blob does nothing.
 */
import type {
  Action, Blob, City, Decision, Development, Frame, GameConfig, GameEvent, GameState,
  Gear, LogLine, TeamId, TeamState, TeamStats, Tech, Vessel,
} from '../shared/types.ts';
import { TEAM_IDS } from '../shared/types.ts';
import { Rng, mixSeed } from '../shared/rng.ts';
import {
  NEIGHBOURS, computeTerritory, dist, findPath, idx, inBounds, isWalkable, isWater,
  terrainAt, tileTeam, visibleTiles,
} from '../shared/geo.ts';
import * as R from '../shared/rules.ts';
import { generateWorld } from './mapgen.ts';

export const TEAM_LETTER: Record<TeamId, string> = { crimson: 'R', cobalt: 'B', amber: 'A', emerald: 'E' };
export const TEAM_LABEL: Record<TeamId, string> = {
  crimson: 'Crimson', cobalt: 'Cobalt', amber: 'Amber', emerald: 'Emerald',
};

const BLOB_SYLLABLES_A = ['Bo', 'Gli', 'Mu', 'Pip', 'Squi', 'Wob', 'Glo', 'Bli', 'Plo', 'Dri', 'Fiz', 'Gub', 'Jel', 'Lum', 'Nub', 'Oo', 'Pud', 'Quib', 'Smo', 'Tub', 'Bun', 'Dol'];
const BLOB_SYLLABLES_B = ['bble', 'b', 'sh', 'x', 'mble', 'py', 'gus', 'dle', 'rt', 'zle', 'nk', 'po', 'mo', 'bert', 'ly', 'wick'];

function emptyStats(): TeamStats {
  return {
    kills: 0, deaths: 0, citiesCaptured: 0, citiesLost: 0, coinsEarned: 0, coinsSpent: 0,
    coinsPaid: 0, messages: 0, alliancesFormed: 0, decisions: 0, validActions: 0,
    apiErrors: 0, inputTokens: 0, outputTokens: 0, peakTerritory: 0,
  };
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

export function createGame(config: GameConfig): GameState {
  const { map, cities, capitals } = generateWorld(config.seed, config.mapSize);
  const rng = new Rng(mixSeed(config.seed, 2));
  const teams = {} as Record<TeamId, TeamState>;
  for (const id of TEAM_IDS) {
    teams[id] = {
      id,
      coins: R.START_COINS,
      techs: [],
      allies: [],
      offersFrom: [],
      remnant: false,
      eliminated: false,
      eliminatedOn: -1,
      explored: '0'.repeat(map.tiles.length),
      cityIntel: {},
      stats: emptyStats(),
      nextBlobNo: 1,
    };
  }

  const state: GameState = {
    config,
    map,
    day: 0,
    cities,
    blobs: [],
    devs: [],
    boss: {
      name: rng.pick(R.BOSS_NAMES),
      x: -1,
      y: -1,
      hp: R.BOSS_HP,
      maxHp: R.BOSS_HP,
      alive: true,
      risesOn: R.BOSS_FIRST_RISE + rng.int(0, 20),
      risen: false,
    },
    loot: [],
    teams,
    speech: [],
    supremacy: { team: null, days: 0 },
    winner: null,
    victory: null,
    over: false,
    nextId: 1,
    log: [],
  };

  const usedNames = new Set<string>();
  for (const team of TEAM_IDS) {
    const cap = capitals[team];
    const spots: [number, number][] = [[cap.x, cap.y]];
    for (const [dx, dy] of rng.shuffle([...NEIGHBOURS])) {
      const x = cap.x + dx;
      const y = cap.y + dy;
      if (inBounds(map, x, y) && isWalkable(terrainAt(map, x, y)) && !cities.some((c) => c.x === x && c.y === y)) {
        spots.push([x, y]);
      }
    }
    for (let i = 0; i < R.START_BLOBS && i < spots.length; i++) {
      spawnBlob(state, team, spots[i][0], spots[i][1], rng, usedNames);
    }
  }
  updateVision(state);
  return state;
}

function blobName(rng: Rng, used: Set<string>): string {
  for (let i = 0; i < 100; i++) {
    const n = rng.pick(BLOB_SYLLABLES_A) + rng.pick(BLOB_SYLLABLES_B);
    if (!used.has(n)) {
      used.add(n);
      return n;
    }
  }
  const n = `Blob${used.size}`;
  used.add(n);
  return n;
}

function spawnBlob(state: GameState, team: TeamId, x: number, y: number, rng: Rng, used?: Set<string>): Blob {
  const t = state.teams[team];
  const names = used ?? new Set(state.blobs.map((b) => b.name));
  const blob: Blob = {
    id: `${TEAM_LETTER[team]}${t.nextBlobNo++}`,
    name: blobName(rng, names),
    team,
    x,
    y,
    hp: R.BLOB_HP,
    maxHp: R.BLOB_HP,
    rank: 'recruit',
    kills: 0,
    gear: null,
    vessel: null,
    bornOn: state.day,
    origin: team,
    memory: [],
  };
  state.blobs.push(blob);
  return blob;
}

// ---------------------------------------------------------------------------
// Shared queries (also used by agents to describe the world)
// ---------------------------------------------------------------------------

export function attackOf(b: Blob): number {
  const rankBonus = R.RANKS[b.rank].attackBonus;
  if (b.vessel === 'gunboat') return R.GUNBOAT.attack + rankBonus;
  if (b.vessel === 'raft') return R.RAFT_ATTACK + rankBonus;
  return R.BLOB_ATTACK + rankBonus + (b.gear ? R.GEAR[b.gear].attack : 0);
}

export function rangeOf(b: Blob): number {
  if (b.vessel === 'gunboat') return R.GUNBOAT.range;
  if (b.vessel === 'raft') return 1;
  return b.gear ? R.GEAR[b.gear].range : 1;
}

export function armourOf(state: GameState, b: Blob): number {
  if (b.vessel) return 0;
  let a = b.gear === 'shield' ? R.GEAR.shield.armour : 0;
  a += R.TERRAIN_COVER[terrainAt(state.map, b.x, b.y)];
  const city = cityAt(state, b.x, b.y);
  if (city && city.walls && city.owner === b.team) a += R.WALLS_ARMOUR;
  return a;
}

export function cityAt(state: GameState, x: number, y: number): City | undefined {
  return state.cities.find((c) => c.x === x && c.y === y);
}

export function devAt(state: GameState, x: number, y: number): Development | undefined {
  return state.devs.find((d) => d.x === x && d.y === y);
}

export function blobAt(state: GameState, x: number, y: number): Blob | undefined {
  return state.blobs.find((b) => b.x === x && b.y === y);
}

export function allied(state: GameState, a: TeamId, b: TeamId): boolean {
  return a === b || state.teams[a].allies.includes(b);
}

export function citiesOf(state: GameState, team: TeamId): City[] {
  return state.cities.filter((c) => c.owner === team);
}

export function blobsOf(state: GameState, team: TeamId): Blob[] {
  return state.blobs.filter((b) => b.team === team);
}

export function territoryOf(state: GameState): (number | null)[] {
  return computeTerritory(state.map, state.cities);
}

// ---------------------------------------------------------------------------
// One day
// ---------------------------------------------------------------------------

interface DayContext {
  state: GameState;
  rng: Rng;
  events: GameEvent[];
  log: LogLine[];
  territory: (number | null)[];
  results: Map<string, { ok: boolean; note: string }>;
}

function say(ctx: DayContext, text: string, team: TeamId | null, important = false): void {
  ctx.log.push({ day: ctx.state.day, text, team, important });
}

function result(ctx: DayContext, b: Blob, ok: boolean, note: string): void {
  ctx.results.set(b.id, { ok, note });
}

function spend(state: GameState, team: TeamId, amount: number): boolean {
  const t = state.teams[team];
  if (t.coins < amount) return false;
  t.coins -= amount;
  t.stats.coinsSpent += amount;
  return true;
}

/**
 * Advance the game by one day using the given decisions (keyed by blob id).
 * Blobs without a decision idle. Returns the frame for this day.
 */
export function stepDay(state: GameState, decisions: Record<string, Decision>): Frame {
  if (state.over) return makeFrame(state, [], [], decisions);
  state.day += 1;
  const ctx: DayContext = {
    state,
    rng: new Rng(mixSeed(state.config.seed, 1000 + state.day)),
    events: [],
    log: [],
    territory: territoryOf(state),
    results: new Map(),
  };

  // Freshly shuffled order every day.
  const order = ctx.rng.shuffle([...state.blobs]);
  const actionOf = (b: Blob): Action => decisions[b.id]?.action ?? { type: 'idle' };
  const alive = (b: Blob) => state.blobs.includes(b);

  // Bookkeeping for the benchmark: every living blob owes one decision.
  for (const b of order) {
    const d = decisions[b.id];
    const st = state.teams[b.team].stats;
    st.decisions += 1;
    if (d?.inputTokens) st.inputTokens += d.inputTokens;
    if (d?.outputTokens) st.outputTokens += d.outputTokens;
    if (d?.error) st.apiErrors += 1;
  }

  // Phase 1: Council -------------------------------------------------------
  for (const b of order) {
    const a = actionOf(b);
    if (a.type === 'ally_propose' || a.type === 'ally_accept' || a.type === 'ally_leave' || a.type === 'pay') {
      council(ctx, b, a);
    }
  }

  // Phase 2: March ---------------------------------------------------------
  for (const b of order) {
    const a = actionOf(b);
    if (a.type === 'move' && alive(b)) march(ctx, b, a.x, a.y);
  }

  // Phase 3: Clash ---------------------------------------------------------
  clash(ctx, order.filter(alive), actionOf);

  // Phase 4: Works (captures last) -----------------------------------------
  ctx.territory = territoryOf(state);
  const capturers: Blob[] = [];
  for (const b of order) {
    if (!alive(b)) continue;
    const a = actionOf(b);
    switch (a.type) {
      case 'develop': develop(ctx, b, a.x, a.y, a.kind); break;
      case 'craft': craft(ctx, b, a.item); break;
      case 'research': research(ctx, b, a.tech); break;
      case 'recruit': recruit(ctx, b); break;
      case 'fortify': fortify(ctx, b); break;
      case 'refit': refit(ctx, b); break;
      case 'rest': rest(ctx, b); break;
      case 'capture': capturers.push(b); break;
      case 'idle': result(ctx, b, true, 'Idled.'); break;
      default: break;
    }
  }
  for (const b of capturers) capture(ctx, b);

  // Phase 5: Dusk -----------------------------------------------------------
  speech(ctx, order.filter(alive), decisions);
  world(ctx);
  upkeep(ctx);

  // Record each blob's decision in its own short memory.
  for (const b of order) {
    const d = decisions[b.id];
    const r = ctx.results.get(b.id) ?? { ok: false, note: 'No action resolved.' };
    const wellFormed = d ? d.wellFormed : false;
    if (r.ok && wellFormed) state.teams[b.team].stats.validActions += 1;
    const target = state.blobs.find((x) => x.id === b.id);
    if (!target) continue;
    target.memory.push({
      day: state.day,
      action: actionOf(b),
      ok: r.ok,
      note: r.note,
      thought: d?.thought ?? '',
      say: d?.say ?? '',
    });
    if (target.memory.length > 6) target.memory.shift();
  }

  state.log.push(...ctx.log);
  if (state.log.length > 500) state.log.splice(0, state.log.length - 500);
  return makeFrame(state, ctx.events, ctx.log, decisions);
}

export function makeFrame(state: GameState, events: GameEvent[], log: LogLine[], decisions: Record<string, Decision>): Frame {
  return structuredClone({
    day: state.day,
    cities: state.cities,
    blobs: state.blobs,
    devs: state.devs,
    boss: state.boss,
    loot: state.loot,
    teams: state.teams,
    speech: state.speech,
    supremacy: state.supremacy,
    winner: state.winner,
    victory: state.victory,
    over: state.over,
    events,
    log,
    decisions,
  });
}

// ---------------------------------------------------------------------------
// Phase 1: Council
// ---------------------------------------------------------------------------

function council(ctx: DayContext, b: Blob, a: Action): void {
  const { state } = ctx;
  const me = state.teams[b.team];
  if (!('team' in a)) return;
  const other = state.teams[a.team as TeamId];
  if (!other || other.id === b.team) return result(ctx, b, false, 'That is not another team.');
  if (other.eliminated) return result(ctx, b, false, `${TEAM_LABEL[other.id]} is eliminated.`);

  switch (a.type) {
    case 'ally_propose': {
      if (me.allies.includes(other.id)) return result(ctx, b, false, 'Already allied.');
      if (!other.offersFrom.includes(me.id)) {
        other.offersFrom.push(me.id);
        ctx.events.push({ kind: 'ally_offer', from: me.id, to: other.id });
        say(ctx, `${TEAM_LABEL[me.id]} offers ${TEAM_LABEL[other.id]} an alliance.`, me.id);
      }
      return result(ctx, b, true, `Offered ${TEAM_LABEL[other.id]} an alliance.`);
    }
    case 'ally_accept': {
      if (!me.offersFrom.includes(other.id)) return result(ctx, b, false, `${TEAM_LABEL[other.id]} has made no offer.`);
      me.offersFrom = me.offersFrom.filter((t) => t !== other.id);
      other.offersFrom = other.offersFrom.filter((t) => t !== me.id);
      me.allies.push(other.id);
      other.allies.push(me.id);
      me.stats.alliancesFormed += 1;
      other.stats.alliancesFormed += 1;
      ctx.events.push({ kind: 'ally', a: me.id, b: other.id });
      say(ctx, `${TEAM_LABEL[me.id]} and ${TEAM_LABEL[other.id]} are now allies.`, me.id, true);
      return result(ctx, b, true, `Allied with ${TEAM_LABEL[other.id]}.`);
    }
    case 'ally_leave': {
      if (!me.allies.includes(other.id)) return result(ctx, b, false, 'Not allied with them.');
      me.allies = me.allies.filter((t) => t !== other.id);
      other.allies = other.allies.filter((t) => t !== me.id);
      ctx.events.push({ kind: 'ally_break', a: me.id, b: other.id });
      say(ctx, `${TEAM_LABEL[me.id]} breaks its alliance with ${TEAM_LABEL[other.id]}.`, me.id, true);
      return result(ctx, b, true, `Left the alliance with ${TEAM_LABEL[other.id]}.`);
    }
    case 'pay': {
      const amount = Math.floor(a.amount);
      if (!(amount >= 1)) return result(ctx, b, false, 'Pay at least 1 coin.');
      if (me.coins < amount) return result(ctx, b, false, `Treasury has only ${me.coins} coins.`);
      me.coins -= amount;
      me.stats.coinsPaid += amount;
      other.coins += amount;
      ctx.events.push({ kind: 'pay', from: me.id, to: other.id, amount });
      say(ctx, `${TEAM_LABEL[me.id]} pays ${TEAM_LABEL[other.id]} ${amount} coins.`, me.id);
      return result(ctx, b, true, `Paid ${amount} coins to ${TEAM_LABEL[other.id]}.`);
    }
    default:
      return;
  }
}

// ---------------------------------------------------------------------------
// Phase 2: March
// ---------------------------------------------------------------------------

function occupied(state: GameState, x: number, y: number, except?: Blob): boolean {
  if (state.boss.risen && state.boss.alive && state.boss.x === x && state.boss.y === y) return true;
  const c = cityAt(state, x, y);
  if (c && c.warden) return true;
  return state.blobs.some((o) => o !== except && o.x === x && o.y === y);
}

function ownHarborAt(ctx: DayContext, team: TeamId, x: number, y: number): boolean {
  const d = devAt(ctx.state, x, y);
  return !!d && d.kind === 'harbor' && tileTeam(ctx.territory, ctx.state.cities, idx(ctx.state.map, x, y)) === team;
}

function march(ctx: DayContext, b: Blob, tx: number, ty: number): void {
  const { state } = ctx;
  const map = state.map;
  if (!Number.isInteger(tx) || !Number.isInteger(ty) || !inBounds(map, tx, ty)) {
    return result(ctx, b, false, `(${tx},${ty}) is off the map.`);
  }
  if (tx === b.x && ty === b.y) return result(ctx, b, true, 'Held position.');
  const targetTerrain = terrainAt(map, tx, ty);

  // On foot, heading for water: set sail if standing on an own harbor.
  if (!b.vessel && isWater(targetTerrain)) {
    if (!state.teams[b.team].techs.includes('sailing')) return result(ctx, b, false, 'Cannot walk on water. Research sailing and build a harbor.');
    if (!ownHarborAt(ctx, b.team, b.x, b.y)) return result(ctx, b, false, 'Can only set sail from your own harbor.');
    let best: [number, number] | null = null;
    for (const [dx, dy] of NEIGHBOURS) {
      const nx = b.x + dx;
      const ny = b.y + dy;
      if (!inBounds(map, nx, ny) || !isWater(terrainAt(map, nx, ny)) || occupied(state, nx, ny, b)) continue;
      if (!best || dist(nx, ny, tx, ty) < dist(best[0], best[1], tx, ty)) best = [nx, ny];
    }
    if (!best) return result(ctx, b, false, 'No free water next to the harbor.');
    const from: [number, number] = [b.x, b.y];
    b.x = best[0];
    b.y = best[1];
    b.vessel = 'raft';
    ctx.events.push({ kind: 'move', id: b.id, path: [from, best], vessel: 'raft' });
    ctx.events.push({ kind: 'embark', id: b.id, x: b.x, y: b.y });
    return result(ctx, b, true, `Set sail on a raft at (${b.x},${b.y}).`);
  }

  const onWater = !!b.vessel;
  const passable = (x: number, y: number) => {
    const t = terrainAt(map, x, y);
    if (onWater) return isWater(t) || (x === tx && y === ty && isWalkable(t));
    return isWalkable(t);
  };
  if (!onWater && !isWalkable(targetTerrain)) return result(ctx, b, false, `Cannot walk onto ${targetTerrain}.`);
  // Prefer a route around other units; fall back to the direct route (which may be blocked).
  let path = findPath(map, b.x, b.y, tx, ty, (x, y) => passable(x, y) && (x === tx && y === ty ? true : !occupied(state, x, y, b)));
  if (path.length === 0) path = findPath(map, b.x, b.y, tx, ty, passable);
  if (path.length === 0) return result(ctx, b, false, `No route to (${tx},${ty}).`);

  const travelled: [number, number][] = [[b.x, b.y]];
  let points = onWater ? R.VESSEL_MOVES : R.MOVE_POINTS;
  let note = '';
  for (const [nx, ny] of path) {
    const t = terrainAt(map, nx, ny);
    const cost = onWater ? 1 : (R.TERRAIN_COST[t] ?? 99);
    if (travelled.length > 1 && cost > points) break;
    if (occupied(state, nx, ny, b)) {
      note = ' Path blocked.';
      break;
    }
    points -= cost;
    b.x = nx;
    b.y = ny;
    travelled.push([nx, ny]);

    // Landing from a vessel: the boat is left behind (a gunboat is lost).
    if (b.vessel && !isWater(t)) {
      const was: Vessel = b.vessel;
      b.vessel = null;
      ctx.events.push({ kind: 'land', id: b.id, x: nx, y: ny });
      note += was === 'gunboat' ? ' Landed; the gunboat was wrecked on the shore.' : ' Landed.';
      break;
    }

    // Loot.
    const li = state.loot.findIndex((l) => l.x === nx && l.y === ny);
    if (li !== -1) {
      const pile = state.loot[li];
      state.loot.splice(li, 1);
      state.teams[b.team].coins += pile.coins;
      state.teams[b.team].stats.coinsEarned += pile.coins;
      ctx.events.push({ kind: 'loot', x: nx, y: ny, team: b.team, coins: pile.coins, by: b.id });
      say(ctx, `${b.name} (${TEAM_LABEL[b.team]}) found ${pile.coins} coins.`, b.team);
      note += ` Picked up ${pile.coins} coins.`;
    }

    // Traps: hidden from enemies, harmless to the builder and allies.
    const dev = devAt(state, nx, ny);
    if (dev && dev.kind === 'trap' && dev.armed && !allied(state, dev.builder, b.team)) {
      dev.armed = false;
      dev.pitUntil = state.day + R.PIT_DAYS;
      b.hp -= R.TRAP_DAMAGE;
      ctx.events.push({ kind: 'trap', x: nx, y: ny, victim: b.id, dmg: R.TRAP_DAMAGE });
      say(ctx, `${b.name} (${TEAM_LABEL[b.team]}) stepped on a ${TEAM_LABEL[dev.builder]} trap.`, dev.builder);
      note += ` Stepped on a trap for ${R.TRAP_DAMAGE} damage.`;
      if (b.hp <= 0) {
        kill(ctx, b, `${TEAM_LABEL[dev.builder]} trap`, null, dev.builder);
      }
      break;
    }
    if (points <= 0) break;
  }
  if (travelled.length > 1) {
    ctx.events.push({ kind: 'move', id: b.id, path: travelled, vessel: b.vessel });
  }
  const arrived = b.x === tx && b.y === ty;
  result(ctx, b, travelled.length > 1, travelled.length > 1
    ? `${arrived ? 'Arrived at' : 'Moved toward'} (${tx},${ty}), now at (${b.x},${b.y}).${note}`
    : `Could not move.${note}`);
}

// ---------------------------------------------------------------------------
// Phase 3: Clash
// ---------------------------------------------------------------------------

type Target =
  | { kind: 'blob'; blob: Blob }
  | { kind: 'warden'; city: City }
  | { kind: 'boss' };

interface PlannedHit {
  target: Target;
  dmg: number;
  attackerId: string;
  attackerName: string;
  team: TeamId | null;
  blob: Blob | null;
}

function targetKey(t: Target): string {
  return t.kind === 'blob' ? t.blob.id : t.kind === 'warden' ? `w:${t.city.id}` : 'boss';
}

function clash(ctx: DayContext, blobs: Blob[], actionOf: (b: Blob) => Action): void {
  const { state } = ctx;
  const hits: PlannedHit[] = [];
  const plan = (h: PlannedHit, fx: number, fy: number, tx: number, ty: number, ranged: boolean, weapon: string) => {
    hits.push(h);
    ctx.events.push({ kind: 'attack', from: h.attackerId, to: targetKey(h.target), fx, fy, tx, ty, dmg: h.dmg, ranged, weapon });
  };

  // Blob attacks.
  for (const b of blobs) {
    const a = actionOf(b);
    if (a.type !== 'attack') continue;
    const range = rangeOf(b);
    const weapon = b.vessel === 'gunboat' ? 'cannon' : b.vessel ? 'oar' : b.gear === 'bow' ? 'bow' : b.gear === 'spear' ? 'spear' : 'slap';
    let target: Target | null = null;
    let tx = 0;
    let ty = 0;
    let armour = 0;
    const id = String(a.target);
    if (id === 'boss') {
      if (state.boss.risen && state.boss.alive) {
        target = { kind: 'boss' };
        tx = state.boss.x;
        ty = state.boss.y;
        armour = R.BOSS_ARMOUR;
      }
    } else if (id.startsWith('c')) {
      const city = state.cities.find((c) => c.id === id);
      if (city && city.warden) {
        target = { kind: 'warden', city };
        tx = city.x;
        ty = city.y;
      }
    } else {
      const t = state.blobs.find((o) => o.id === id);
      if (t) {
        if (allied(state, b.team, t.team)) {
          result(ctx, b, false, `${t.name} is a friend.`);
          continue;
        }
        target = { kind: 'blob', blob: t };
        tx = t.x;
        ty = t.y;
        armour = armourOf(state, t);
      }
    }
    if (!target) {
      result(ctx, b, false, `No target "${id}" to attack.`);
      continue;
    }
    if (dist(b.x, b.y, tx, ty) > range) {
      result(ctx, b, false, `Target out of range (range ${range}).`);
      continue;
    }
    const dmg = Math.max(1, attackOf(b) - armour);
    plan({ target, dmg, attackerId: b.id, attackerName: b.name, team: b.team, blob: b }, b.x, b.y, tx, ty, range > 1, weapon);
    result(ctx, b, true, `Attacked ${id} for ${dmg}.`);
  }

  // Towers shoot the nearest enemy of their territory's owner.
  for (const d of state.devs) {
    if (d.kind !== 'tower') continue;
    const owner = tileTeam(ctx.territory, state.cities, idx(state.map, d.x, d.y));
    if (!owner) continue;
    let best: Blob | null = null;
    for (const o of blobs) {
      if (allied(state, owner, o.team)) continue;
      const dd = dist(d.x, d.y, o.x, o.y);
      if (dd <= R.TOWER_RANGE && (!best || dd < dist(d.x, d.y, best.x, best.y))) best = o;
    }
    if (best) {
      const dmg = Math.max(1, R.TOWER_DAMAGE - armourOf(state, best));
      plan({ target: { kind: 'blob', blob: best }, dmg, attackerId: `tower@${d.x},${d.y}`, attackerName: `${TEAM_LABEL[owner]} tower`, team: owner, blob: null }, d.x, d.y, best.x, best.y, true, 'tower');
    }
  }

  // Wardens strike the weakest blob next to their village.
  for (const c of state.cities) {
    if (!c.warden) continue;
    let best: Blob | null = null;
    for (const o of blobs) {
      if (dist(c.x, c.y, o.x, o.y) === 1 && (!best || o.hp < best.hp)) best = o;
    }
    if (best) {
      const dmg = Math.max(1, R.WARDEN_ATTACK - armourOf(state, best));
      plan({ target: { kind: 'blob', blob: best }, dmg, attackerId: `w:${c.id}`, attackerName: `the ${c.name} warden`, team: null, blob: null }, c.x, c.y, best.x, best.y, false, 'warden');
    }
  }

  // The boss slams everything next to it.
  if (state.boss.risen && state.boss.alive) {
    let slammed = false;
    for (const o of blobs) {
      if (dist(state.boss.x, state.boss.y, o.x, o.y) === 1) {
        const dmg = Math.max(1, R.BOSS_SLAM - armourOf(state, o));
        plan({ target: { kind: 'blob', blob: o }, dmg, attackerId: 'boss', attackerName: state.boss.name, team: null, blob: null }, state.boss.x, state.boss.y, o.x, o.y, false, 'slam');
        slammed = true;
      }
    }
    if (slammed) ctx.events.push({ kind: 'boss_slam', x: state.boss.x, y: state.boss.y });
  }

  // Apply all damage at once.
  const byTarget = new Map<string, PlannedHit[]>();
  for (const h of hits) {
    const k = targetKey(h.target);
    if (!byTarget.has(k)) byTarget.set(k, []);
    byTarget.get(k)!.push(h);
  }
  const deaths: { target: Target; killer: PlannedHit }[] = [];
  for (const list of byTarget.values()) {
    const target = list[0].target;
    const total = list.reduce((s, h) => s + h.dmg, 0);
    // The killing blow is credited to whoever dealt the most damage.
    const killer = [...list].sort((a, b) => b.dmg - a.dmg)[0];
    if (target.kind === 'blob') {
      target.blob.hp -= total;
      if (target.blob.hp <= 0) deaths.push({ target, killer });
    } else if (target.kind === 'warden') {
      target.city.warden!.hp -= total;
      if (target.city.warden!.hp <= 0) deaths.push({ target, killer });
    } else {
      state.boss.hp -= total;
      if (state.boss.hp <= 0) deaths.push({ target, killer });
    }
  }
  for (const { target, killer } of deaths) {
    if (target.kind === 'blob') {
      kill(ctx, target.blob, killer.attackerName, killer.blob, killer.team);
    } else if (target.kind === 'warden') {
      target.city.warden = null;
      ctx.events.push({ kind: 'death', id: `w:${target.city.id}`, name: `${target.city.name} warden`, team: null, x: target.city.x, y: target.city.y, by: killer.attackerId });
      say(ctx, `${killer.attackerName} beat the warden of ${target.city.name}. The village can be captured.`, killer.team);
      if (killer.blob) creditKill(ctx, killer.blob);
    } else {
      state.boss.alive = false;
      state.boss.hp = 0;
      const team = killer.team ?? 'crimson';
      ctx.events.push({ kind: 'death', id: 'boss', name: state.boss.name, team: null, x: state.boss.x, y: state.boss.y, by: killer.attackerId });
      if (killer.team) {
        state.teams[killer.team].coins += R.BOSS_BOUNTY;
        state.teams[killer.team].stats.coinsEarned += R.BOSS_BOUNTY;
        ctx.events.push({ kind: 'boss_dead', name: state.boss.name, team, by: killer.attackerId, bounty: R.BOSS_BOUNTY });
        say(ctx, `${killer.attackerName} (${TEAM_LABEL[killer.team]}) slew ${state.boss.name}! +${R.BOSS_BOUNTY} coins.`, killer.team, true);
      }
      if (killer.blob) creditKill(ctx, killer.blob);
    }
  }
}

function creditKill(ctx: DayContext, killer: Blob): void {
  if (!ctx.state.blobs.includes(killer)) return;
  killer.kills += 1;
  ctx.state.teams[killer.team].stats.kills += 1;
}

function kill(ctx: DayContext, b: Blob, byName: string, byBlob: Blob | null, byTeam: TeamId | null): void {
  const { state } = ctx;
  const i = state.blobs.indexOf(b);
  if (i === -1) return;
  state.blobs.splice(i, 1);
  state.teams[b.team].stats.deaths += 1;
  if (byBlob) creditKill(ctx, byBlob);
  else if (byTeam) state.teams[byTeam].stats.kills += 1;
  ctx.events.push({ kind: 'death', id: b.id, name: b.name, team: b.team, x: b.x, y: b.y, by: byBlob?.id ?? byName });
  say(ctx, `${b.name} (${TEAM_LABEL[b.team]}) was defeated by ${byName}.`, b.team);
}

// ---------------------------------------------------------------------------
// Phase 4: Works
// ---------------------------------------------------------------------------

function ownCityUnder(ctx: DayContext, b: Blob): City | null {
  const c = cityAt(ctx.state, b.x, b.y);
  return c && c.owner === b.team && !b.vessel ? c : null;
}

function cityForTile(ctx: DayContext, x: number, y: number): City | null {
  const ci = ctx.territory[idx(ctx.state.map, x, y)];
  return ci === null ? null : ctx.state.cities[ci];
}

function growCity(ctx: DayContext, city: City, pop: number): void {
  const before = city.level;
  city.pop = Math.max(0, city.pop + pop);
  city.level = R.levelForPop(city.pop);
  if (city.level > before) {
    city.levelUpOn = ctx.state.day;
    ctx.events.push({ kind: 'levelup', city: city.id, level: city.level, x: city.x, y: city.y });
    say(ctx, `${city.name} grew to level ${city.level}.`, city.owner, true);
  }
}

function develop(ctx: DayContext, b: Blob, x: number, y: number, kind: Development['kind']): void {
  const { state } = ctx;
  const spec = R.DEVELOPMENTS[kind];
  if (!spec) return result(ctx, b, false, `Unknown development "${kind}".`);
  if (b.vessel) return result(ctx, b, false, 'Cannot build from a boat.');
  if (!Number.isInteger(x) || !Number.isInteger(y) || !inBounds(state.map, x, y)) return result(ctx, b, false, 'Tile is off the map.');
  if (dist(b.x, b.y, x, y) > 1) return result(ctx, b, false, 'Can only build on your tile or next to it.');
  const tile = state.map.tiles[idx(state.map, x, y)];
  if (!spec.terrain.includes(tile.terrain)) return result(ctx, b, false, `A ${kind} cannot go on ${tile.terrain}.`);
  if (cityAt(state, x, y)) return result(ctx, b, false, 'Cannot build on a city tile.');
  if (devAt(state, x, y)) return result(ctx, b, false, 'Tile already developed.');
  const city = cityForTile(ctx, x, y);
  if (!city || city.owner !== b.team) return result(ctx, b, false, 'Tile is not in your territory.');
  if (spec.tech && !state.teams[b.team].techs.includes(spec.tech)) return result(ctx, b, false, `Needs ${spec.tech}.`);
  if (kind === 'harbor' && !NEIGHBOURS.some(([dx, dy]) => inBounds(state.map, x + dx, y + dy) && isWater(terrainAt(state.map, x + dx, y + dy)))) {
    return result(ctx, b, false, 'A harbor must touch water.');
  }
  if (!spend(state, b.team, spec.cost)) return result(ctx, b, false, `Needs ${spec.cost} coins.`);
  state.devs.push({ x, y, kind, builder: b.team, armed: kind === 'trap', pitUntil: -1, builtOn: state.day });
  const bonus = (kind === 'farm' && tile.resource === 'fruit') || (kind === 'mine' && tile.resource === 'crystal') ? spec.resourceBonus : 0;
  ctx.events.push({ kind: 'develop', team: b.team, devKind: kind, x, y });
  if (spec.pop + bonus > 0) growCity(ctx, city, spec.pop + bonus);
  result(ctx, b, true, `Built a ${kind} at (${x},${y}).`);
}

function craft(ctx: DayContext, b: Blob, item: Gear): void {
  const spec = R.GEAR[item];
  if (!spec) return result(ctx, b, false, `Unknown item "${item}".`);
  if (!ownCityUnder(ctx, b)) return result(ctx, b, false, 'Craft while standing in your own city.');
  if (!ctx.state.teams[b.team].techs.includes(spec.tech)) return result(ctx, b, false, `Needs ${spec.tech}.`);
  if (b.gear === item) return result(ctx, b, false, `Already carrying a ${item}.`);
  if (!spend(ctx.state, b.team, spec.cost)) return result(ctx, b, false, `Needs ${spec.cost} coins.`);
  b.gear = item;
  ctx.events.push({ kind: 'craft', id: b.id, item });
  result(ctx, b, true, `Crafted a ${item}.`);
}

function research(ctx: DayContext, b: Blob, tech: Tech): void {
  const spec = R.TECHS[tech];
  const team = ctx.state.teams[b.team];
  if (!spec) return result(ctx, b, false, `Unknown tech "${tech}".`);
  if (!ownCityUnder(ctx, b)) return result(ctx, b, false, 'Research while standing in your own city.');
  if (team.techs.includes(tech)) return result(ctx, b, false, `Already know ${tech}.`);
  if (spec.requires && !team.techs.includes(spec.requires)) return result(ctx, b, false, `Research ${spec.requires} first.`);
  if (!spend(ctx.state, b.team, spec.cost)) return result(ctx, b, false, `Needs ${spec.cost} coins.`);
  team.techs.push(tech);
  ctx.events.push({ kind: 'research', team: b.team, tech });
  say(ctx, `${TEAM_LABEL[b.team]} learned ${tech}.`, b.team);
  result(ctx, b, true, `Researched ${tech}.`);
}

function recruit(ctx: DayContext, b: Blob): void {
  const { state } = ctx;
  const city = ownCityUnder(ctx, b);
  if (!city) return result(ctx, b, false, 'Recruit while standing in your own city.');
  const cap = R.blobCap(citiesOf(state, b.team).length);
  if (blobsOf(state, b.team).length >= cap) return result(ctx, b, false, `Team is at its blob cap (${cap}).`);
  const spots = NEIGHBOURS.map(([dx, dy]) => [city.x + dx, city.y + dy] as [number, number]).filter(
    ([x, y]) => inBounds(state.map, x, y) && isWalkable(terrainAt(state.map, x, y)) && !occupied(state, x, y) && !cityAt(state, x, y),
  );
  if (spots.length === 0) return result(ctx, b, false, 'No free tile next to the city.');
  if (!spend(state, b.team, R.RECRUIT_COST)) return result(ctx, b, false, `Needs ${R.RECRUIT_COST} coins.`);
  const [x, y] = ctx.rng.pick(spots);
  const nb = spawnBlob(state, b.team, x, y, ctx.rng);
  ctx.events.push({ kind: 'recruit', id: nb.id, name: nb.name, team: b.team, x, y });
  say(ctx, `${TEAM_LABEL[b.team]} recruited ${nb.name} at ${city.name}.`, b.team);
  result(ctx, b, true, `Recruited ${nb.name} (${nb.id}).`);
}

function fortify(ctx: DayContext, b: Blob): void {
  const city = ownCityUnder(ctx, b);
  if (!city) return result(ctx, b, false, 'Fortify while standing in your own city.');
  if (city.walls) return result(ctx, b, false, 'City already has walls.');
  if (!ctx.state.teams[b.team].techs.includes('fortification')) return result(ctx, b, false, 'Needs fortification.');
  if (!spend(ctx.state, b.team, R.WALLS_COST)) return result(ctx, b, false, `Needs ${R.WALLS_COST} coins.`);
  city.walls = true;
  ctx.events.push({ kind: 'fortify', city: city.id, team: b.team });
  say(ctx, `${TEAM_LABEL[b.team]} raised walls around ${city.name}.`, b.team);
  result(ctx, b, true, `Built walls at ${city.name}.`);
}

function refit(ctx: DayContext, b: Blob): void {
  const { state } = ctx;
  if (b.vessel !== 'raft') return result(ctx, b, false, 'Only a raft can be refitted.');
  if (!state.teams[b.team].techs.includes('navigation')) return result(ctx, b, false, 'Needs navigation.');
  const nearHarbor = NEIGHBOURS.some(([dx, dy]) => inBounds(state.map, b.x + dx, b.y + dy) && ownHarborAt(ctx, b.team, b.x + dx, b.y + dy));
  if (!nearHarbor) return result(ctx, b, false, 'Refit next to your own harbor.');
  if (!spend(state, b.team, R.GUNBOAT.cost)) return result(ctx, b, false, `Needs ${R.GUNBOAT.cost} coins.`);
  b.vessel = 'gunboat';
  ctx.events.push({ kind: 'refit', id: b.id });
  result(ctx, b, true, 'Refitted into a gunboat.');
}

function rest(ctx: DayContext, b: Blob): void {
  const inTerritory = tileTeam(ctx.territory, ctx.state.cities, idx(ctx.state.map, b.x, b.y)) === b.team;
  const heal = R.REST_HEAL + (inTerritory ? R.REST_HEAL_TERRITORY_BONUS : 0);
  const before = b.hp;
  b.hp = Math.min(b.maxHp, b.hp + heal);
  result(ctx, b, true, `Rested, +${b.hp - before} HP.`);
}

function capture(ctx: DayContext, b: Blob): void {
  const { state } = ctx;
  if (!state.blobs.includes(b)) return;
  const city = cityAt(state, b.x, b.y);
  if (!city || b.vessel) return result(ctx, b, false, 'Stand on a city tile to capture it.');
  if (city.owner === b.team) return result(ctx, b, false, 'Already yours.');
  if (city.owner && allied(state, b.team, city.owner)) return result(ctx, b, false, 'Cannot capture an ally\'s city.');
  if (city.warden) return result(ctx, b, false, 'The warden still stands.');
  if (city.owner) {
    const guard = state.blobs.find((o) => o.team === city.owner && dist(o.x, o.y, city.x, city.y) <= 1);
    if (guard) return result(ctx, b, false, `${guard.name} is defending ${city.name}.`);
  }
  const from = city.owner;
  city.owner = b.team;
  city.capturedOn = state.day;
  state.teams[b.team].stats.citiesCaptured += 1;
  if (from) state.teams[from].stats.citiesLost += 1;
  ctx.territory = territoryOf(state);
  ctx.events.push({ kind: 'capture', city: city.id, cityName: city.name, team: b.team, from, x: city.x, y: city.y });
  say(ctx, `${b.name} captured ${city.name} for ${TEAM_LABEL[b.team]}${from ? ` from ${TEAM_LABEL[from]}` : ''}.`, b.team, true);
  result(ctx, b, true, `Captured ${city.name}.`);
}

// ---------------------------------------------------------------------------
// Phase 5: Dusk (speech, world, upkeep)
// ---------------------------------------------------------------------------

function speech(ctx: DayContext, blobs: Blob[], decisions: Record<string, Decision>): void {
  const { state } = ctx;
  for (const b of blobs) {
    const text = (decisions[b.id]?.say ?? '').trim().slice(0, R.SAY_MAX_CHARS);
    if (!text) continue;
    state.speech.push({ day: state.day, blobId: b.id, name: b.name, team: b.team, x: b.x, y: b.y, text });
    state.teams[b.team].stats.messages += 1;
    ctx.events.push({ kind: 'say', id: b.id, text, x: b.x, y: b.y });
  }
  state.speech = state.speech.filter((s) => s.day > state.day - 3);
}

function world(ctx: DayContext): void {
  const { state, rng } = ctx;
  const boss = state.boss;

  // The boss: announced, then rises, then marches on the nearest city.
  if (!boss.risen && boss.alive && state.day === boss.risesOn - R.BOSS_WARNING_DAYS) {
    ctx.events.push({ kind: 'boss_warning', name: boss.name, risesOn: boss.risesOn });
    say(ctx, `The marsh stirs. ${boss.name} will rise near the centre on day ${boss.risesOn}.`, null, true);
  }
  if (!boss.risen && boss.alive && state.day >= boss.risesOn) {
    const cx = Math.floor(state.map.width / 2);
    const cy = Math.floor(state.map.height / 2);
    let best: [number, number] | null = null;
    for (let r = 0; r < 8 && !best; r++) {
      for (let dy = -r; dy <= r && !best; dy++) {
        for (let dx = -r; dx <= r && !best; dx++) {
          const x = cx + dx;
          const y = cy + dy;
          if (!inBounds(state.map, x, y) || !isWalkable(terrainAt(state.map, x, y))) continue;
          if (occupied(state, x, y) || cityAt(state, x, y)) continue;
          best = [x, y];
        }
      }
    }
    if (best) {
      boss.risen = true;
      boss.x = best[0];
      boss.y = best[1];
      ctx.events.push({ kind: 'boss_rise', name: boss.name, x: boss.x, y: boss.y });
      say(ctx, `${boss.name} rises from the mud at (${boss.x},${boss.y})!`, null, true);
    }
  } else if (boss.risen && boss.alive) {
    const targets = state.cities.filter((c) => c.owner);
    targets.sort((a, b) => dist(boss.x, boss.y, a.x, a.y) - dist(boss.x, boss.y, b.x, b.y));
    const target = targets[0];
    if (target && !(boss.x === target.x && boss.y === target.y)) {
      const path = findPath(state.map, boss.x, boss.y, target.x, target.y, (x, y) => isWalkable(terrainAt(state.map, x, y)));
      const next = path[0];
      if (next && !occupied(state, next[0], next[1])) {
        const from: [number, number] = [boss.x, boss.y];
        boss.x = next[0];
        boss.y = next[1];
        ctx.events.push({ kind: 'boss_move', path: [from, next] });
      }
    }
    const under = cityAt(state, boss.x, boss.y);
    if (under && under.pop > 0) {
      const before = under.pop;
      under.pop = Math.max(0, under.pop - R.BOSS_RAZE);
      under.level = R.levelForPop(under.pop);
      ctx.events.push({ kind: 'raze', city: under.id, pop: before - under.pop });
      say(ctx, `${boss.name} tramples ${under.name}.`, under.owner, true);
    }
  }

  // Loot piles wash up now and then.
  if (state.day % R.LOOT_EVERY === 0) for (let wave = 0; wave < R.lootPerWave(state.map.width); wave++) {
    for (let tries = 0; tries < 40; tries++) {
      const x = rng.int(1, state.map.width - 2);
      const y = rng.int(1, state.map.height - 2);
      if (!isWalkable(terrainAt(state.map, x, y)) || occupied(state, x, y) || cityAt(state, x, y) || devAt(state, x, y)) continue;
      if (state.loot.some((l) => l.x === x && l.y === y)) continue;
      const coins = rng.int(R.LOOT_MIN, R.LOOT_MAX);
      state.loot.push({ id: `L${state.nextId++}`, x, y, coins });
      ctx.events.push({ kind: 'loot_spawn', x, y, coins });
      break;
    }
  }

  // Pits from sprung traps close up.
  state.devs = state.devs.filter((d) => !(d.kind === 'trap' && !d.armed && d.pitUntil <= state.day));

  // Promotions.
  for (const b of state.blobs) {
    const newRank = b.kills >= R.RANKS.champion.kills ? 'champion' : b.kills >= R.RANKS.veteran.kills ? 'veteran' : 'recruit';
    if (newRank !== b.rank) {
      const gain = R.RANKS[newRank].hpBonus - R.RANKS[b.rank].hpBonus;
      b.rank = newRank;
      b.maxHp += gain;
      b.hp += gain;
      ctx.events.push({ kind: 'promote', id: b.id, name: b.name, rank: newRank, team: b.team });
      say(ctx, `${b.name} (${TEAM_LABEL[b.team]}) is now a ${newRank}.`, b.team, true);
    }
  }
}

function upkeep(ctx: DayContext): void {
  const { state } = ctx;
  ctx.territory = territoryOf(state);

  // Remnants and eliminations.
  for (const id of TEAM_IDS) {
    const t = state.teams[id];
    if (t.eliminated) continue;
    const cities = citiesOf(state, id).length;
    const blobs = blobsOf(state, id).length;
    if (cities === 0 && blobs === 0) {
      t.eliminated = true;
      t.eliminatedOn = state.day;
      t.remnant = false;
      for (const other of TEAM_IDS) {
        const o = state.teams[other];
        o.allies = o.allies.filter((a) => a !== id);
        o.offersFrom = o.offersFrom.filter((a) => a !== id);
      }
      t.allies = [];
      t.offersFrom = [];
      ctx.events.push({ kind: 'eliminated', team: id });
      say(ctx, `${TEAM_LABEL[id]} has been eliminated.`, id, true);
    } else if (cities === 0 && !t.remnant) {
      t.remnant = true;
      ctx.events.push({ kind: 'remnant', team: id });
      say(ctx, `${TEAM_LABEL[id]} has lost every city. Its blobs fight on as wanderers with no income.`, id, true);
    } else if (cities > 0 && t.remnant) {
      t.remnant = false;
      say(ctx, `${TEAM_LABEL[id]} holds a city again.`, id, true);
    }
  }

  // Muster: a team that still holds cities but has no blobs left raises one.
  if (state.day % R.MUSTER_EVERY === 0) {
    for (const id of TEAM_IDS) {
      if (state.teams[id].eliminated || blobsOf(state, id).length > 0) continue;
      const city = [...citiesOf(state, id)].sort((a, b) => b.level - a.level).find((c) => !occupied(state, c.x, c.y));
      if (!city) continue;
      const nb = spawnBlob(state, id, city.x, city.y, ctx.rng);
      ctx.events.push({ kind: 'recruit', id: nb.id, name: nb.name, team: id, x: city.x, y: city.y });
      say(ctx, `${city.name} musters ${nb.name} to defend ${TEAM_LABEL[id]}.`, id);
    }
  }

  // Income.
  for (const c of state.cities) {
    if (!c.owner) continue;
    const coins = R.cityIncome(c.level, c.isCapital, c.owner === c.founder);
    state.teams[c.owner].coins += coins;
    state.teams[c.owner].stats.coinsEarned += coins;
  }

  // Healing in own cities; wardens recover too.
  for (const b of state.blobs) {
    const c = cityAt(state, b.x, b.y);
    if (c && c.owner === b.team && !b.vessel) b.hp = Math.min(b.maxHp, b.hp + R.CITY_PASSIVE_HEAL);
  }
  for (const c of state.cities) {
    if (c.warden && c.warden.hp < c.warden.maxHp) c.warden.hp += 1;
  }

  updateVision(state);

  // Territory peaks for the stats tab.
  for (const id of TEAM_IDS) {
    const size = ctx.territory.filter((ci) => ci !== null && state.cities[ci].owner === id).length;
    const st = state.teams[id].stats;
    st.peakTerritory = Math.max(st.peakTerritory, size);
  }

  // Victory.
  const standing = TEAM_IDS.filter((id) => !state.teams[id].eliminated);
  if (standing.length <= 1) {
    finish(ctx, standing[0] ?? null, standing.length === 1 ? 'last_standing' : 'draw');
    return;
  }
  const need = Math.ceil(state.cities.length * R.SUPREMACY_SHARE);
  const leader = standing.find((id) => citiesOf(state, id).length >= need) ?? null;
  if (leader && leader === state.supremacy.team) state.supremacy.days += 1;
  else state.supremacy = { team: leader, days: leader ? 1 : 0 };
  if (leader && (state.supremacy.days === 1 || state.supremacy.days % 5 === 0)) {
    ctx.events.push({ kind: 'supremacy', team: leader, days: state.supremacy.days });
    say(ctx, `${TEAM_LABEL[leader]} holds ${need}+ cities: supremacy day ${state.supremacy.days} of ${R.SUPREMACY_DAYS}.`, leader, true);
  }
  if (leader && state.supremacy.days >= R.SUPREMACY_DAYS) {
    finish(ctx, leader, 'supremacy');
    return;
  }
  if (state.day >= state.config.maxDays) {
    const ranked = [...standing].sort((a, b) => {
      const ca = citiesOf(state, a);
      const cb = citiesOf(state, b);
      if (cb.length !== ca.length) return cb.length - ca.length;
      const la = ca.reduce((s, c) => s + c.level, 0);
      const lb = cb.reduce((s, c) => s + c.level, 0);
      if (lb !== la) return lb - la;
      return state.teams[b].coins - state.teams[a].coins;
    });
    const [first, second] = ranked;
    const tie = second
      && citiesOf(state, first).length === citiesOf(state, second).length
      && citiesOf(state, first).reduce((s, c) => s + c.level, 0) === citiesOf(state, second).reduce((s, c) => s + c.level, 0)
      && state.teams[first].coins === state.teams[second].coins;
    finish(ctx, tie ? null : first, tie ? 'draw' : 'day_limit');
  }
}

function finish(ctx: DayContext, winner: TeamId | null, reason: GameState['victory']): void {
  const { state } = ctx;
  state.over = true;
  state.winner = winner;
  state.victory = reason;
  ctx.events.push({ kind: 'victory', team: winner, reason: reason ?? 'draw' });
  const why = reason === 'last_standing' ? 'as the last team standing'
    : reason === 'supremacy' ? `by supremacy (${R.SUPREMACY_DAYS} days holding two thirds of all cities)`
      : reason === 'day_limit' ? `on day ${state.config.maxDays} with the most cities` : '';
  say(ctx, winner ? `${TEAM_LABEL[winner]} wins ${why}.` : 'The match ends in a draw.', winner, true);
}

/** Mark what every blob can see as explored and refresh city intel. */
export function updateVision(state: GameState): void {
  const explored: Record<TeamId, string[]> = {} as Record<TeamId, string[]>;
  for (const id of TEAM_IDS) explored[id] = state.teams[id].explored.split('');
  for (const b of state.blobs) {
    for (const i of visibleTiles(state.map, b)) explored[b.team][i] = '1';
  }
  for (const id of TEAM_IDS) {
    // A team always knows its own cities.
    for (const c of citiesOf(state, id)) explored[id][idx(state.map, c.x, c.y)] = '1';
    state.teams[id].explored = explored[id].join('');
    for (const c of state.cities) {
      if (explored[id][idx(state.map, c.x, c.y)] === '1') {
        const seenNow = c.owner === id || state.blobs.some((b) => b.team === id && dist(b.x, b.y, c.x, c.y) <= 3);
        if (seenNow || !(c.id in state.teams[id].cityIntel)) state.teams[id].cityIntel[c.id] = c.owner ?? 'none';
      }
    }
  }
}
