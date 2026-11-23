/**
 * Builds what one blob knows on a given day: its own state, what it can see,
 * what its team remembers of the map, and what it heard within earshot.
 *
 * Both agent types (Claude and the local bot) decide from this view only.
 * They never see the whole game state.
 */
import type { Blob, City, DevKind, Gear, Rank, Speech, TeamId, Tech, Terrain, Vessel } from '../shared/types.ts';
import { TEAM_IDS } from '../shared/types.ts';
import type { GameState } from '../shared/types.ts';
import { NEIGHBOURS, computeTerritory, dist, exploredSet, idx, inBounds, isWater, terrainAt, tileTeam, visionRadius } from '../shared/geo.ts';
import * as R from '../shared/rules.ts';
import { allied, armourOf, attackOf, blobsOf, citiesOf, rangeOf } from './engine.ts';
import { TERRAIN_CHAR } from './mapgen.ts';

export interface SeenBlob {
  id: string;
  name: string;
  team: TeamId;
  x: number;
  y: number;
  hp: number;
  maxHp: number;
  rank: Rank;
  gear: Gear | null;
  vessel: Vessel | null;
  distance: number;
  friendly: boolean;
}

export interface SeenCity {
  id: string;
  name: string;
  x: number;
  y: number;
  owner: TeamId | 'none';
  isCapital: boolean;
  island: boolean;
  level: number;
  walls: boolean;
  wardenHp: number | null;
  distance: number;
  visible: boolean;
}

export interface Observation {
  day: number;
  maxDays: number;
  self: {
    id: string; name: string; team: TeamId; x: number; y: number; hp: number; maxHp: number;
    rank: Rank; kills: number; gear: Gear | null; vessel: Vessel | null;
    attack: number; range: number; armour: number; terrain: Terrain; vision: number;
    inOwnTerritory: boolean;
    onCity: { id: string; name: string; owner: TeamId | 'none'; warden: boolean } | null;
    onHarbor: boolean;
  };
  team: {
    coins: number; techs: Tech[]; allies: TeamId[]; offersFrom: TeamId[]; remnant: boolean;
    cities: { id: string; name: string; x: number; y: number; level: number; pop: number; walls: boolean; isCapital: boolean }[];
    blobs: { id: string; name: string; x: number; y: number; hp: number }[];
    blobCap: number;
    income: number;
  };
  seenBlobs: SeenBlob[];
  knownCities: SeenCity[];
  seenDevs: { x: number; y: number; kind: DevKind; team: TeamId | null; armed: boolean }[];
  seenLoot: { x: number; y: number; coins: number; distance: number }[];
  boss: { name: string; x: number; y: number; hp: number; distance: number } | null;
  bossNews: string | null;
  heard: Speech[];
  standings: { team: TeamId; cities: number; eliminated: boolean; remnant: boolean }[];
  supremacy: { team: TeamId | null; days: number; need: number };
  /** Free tiles next to the blob that are in its own territory (handy for building). */
  buildSpots: { x: number; y: number; terrain: Terrain; resource: string | null; developed: DevKind | null; coastal: boolean }[];
  mapWindow: string;
  memory: Blob['memory'];
}

const WINDOW = 5;

export function observe(state: GameState, b: Blob): Observation {
  const map = state.map;
  const team = state.teams[b.team];
  const known = exploredSet(team.explored);
  const vision = visionRadius(map, b);
  const sees = (x: number, y: number) => dist(b.x, b.y, x, y) <= vision;
  const territory = computeTerritory(map, state.cities);
  const myCities = citiesOf(state, b.team);
  const cityUnder = state.cities.find((c) => c.x === b.x && c.y === b.y);
  const devUnder = state.devs.find((d) => d.x === b.x && d.y === b.y);

  const seenBlobs: SeenBlob[] = state.blobs
    .filter((o) => o !== b && sees(o.x, o.y))
    .map((o) => ({
      id: o.id, name: o.name, team: o.team, x: o.x, y: o.y, hp: o.hp, maxHp: o.maxHp,
      rank: o.rank, gear: o.gear, vessel: o.vessel, distance: dist(b.x, b.y, o.x, o.y),
      friendly: allied(state, b.team, o.team),
    }))
    .sort((p, q) => p.distance - q.distance);

  const knownCities: SeenCity[] = state.cities
    .filter((c) => known(idx(map, c.x, c.y)))
    .map((c: City) => {
      const visible = sees(c.x, c.y) || c.owner === b.team;
      const owner = visible ? (c.owner ?? 'none') : (team.cityIntel[c.id] ?? 'none');
      return {
        id: c.id, name: c.name, x: c.x, y: c.y, owner, isCapital: c.isCapital, island: c.island,
        level: c.level, walls: c.walls, wardenHp: visible ? (c.warden?.hp ?? null) : (owner === 'none' ? (c.warden ? c.warden.maxHp : null) : null),
        distance: dist(b.x, b.y, c.x, c.y), visible,
      };
    })
    .sort((p, q) => p.distance - q.distance);

  const seenDevs = state.devs
    .filter((d) => sees(d.x, d.y))
    .filter((d) => d.kind !== 'trap' || !d.armed || allied(state, d.builder, b.team))
    .map((d) => ({ x: d.x, y: d.y, kind: d.kind, team: tileTeam(territory, state.cities, idx(map, d.x, d.y)), armed: d.armed }));

  const seenLoot = state.loot
    .filter((l) => sees(l.x, l.y))
    .map((l) => ({ x: l.x, y: l.y, coins: l.coins, distance: dist(b.x, b.y, l.x, l.y) }));

  const boss = state.boss.risen && state.boss.alive && sees(state.boss.x, state.boss.y)
    ? { name: state.boss.name, x: state.boss.x, y: state.boss.y, hp: state.boss.hp, distance: dist(b.x, b.y, state.boss.x, state.boss.y) }
    : null;
  let bossNews: string | null = null;
  if (state.boss.alive && !state.boss.risen && state.day >= state.boss.risesOn - R.BOSS_WARNING_DAYS) {
    bossNews = `${state.boss.name}, a huge marsh blob (${R.BOSS_HP} HP, armour ${R.BOSS_ARMOUR}, slams every adjacent blob for ${R.BOSS_SLAM}), rises near the map centre on day ${state.boss.risesOn}. Killing it pays ${R.BOSS_BOUNTY} coins.`;
  } else if (state.boss.risen && state.boss.alive) {
    bossNews = `${state.boss.name} is loose (${state.boss.hp}/${state.boss.maxHp} HP), trampling the nearest city. Killing it pays ${R.BOSS_BOUNTY} coins.`;
  }

  const heard = state.speech.filter((s) => s.day >= state.day - 1 && dist(s.x, s.y, b.x, b.y) <= R.HEARING_RANGE);

  const buildSpots: Observation['buildSpots'] = [];
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const x = b.x + dx;
      const y = b.y + dy;
      if (!inBounds(map, x, y)) continue;
      const i = idx(map, x, y);
      if (tileTeam(territory, state.cities, i) !== b.team) continue;
      if (state.cities.some((c) => c.x === x && c.y === y)) continue;
      const t = map.tiles[i];
      if (isWater(t.terrain) || t.terrain === 'mountain') continue;
      const dev = state.devs.find((d) => d.x === x && d.y === y);
      const coastal = NEIGHBOURS.some(([nx, ny]) => inBounds(map, x + nx, y + ny) && isWater(terrainAt(map, x + nx, y + ny)));
      buildSpots.push({ x, y, terrain: t.terrain, resource: t.resource, developed: dev?.kind ?? null, coastal });
    }
  }

  // Text map around the blob. Unexplored tiles are '?'.
  const rows: string[] = [];
  const header = ['     '];
  for (let x = b.x - WINDOW; x <= b.x + WINDOW; x++) header.push(inBounds(map, x, 0) ? String(x).padStart(2) : '  ');
  rows.push(header.join(' '));
  for (let y = b.y - WINDOW; y <= b.y + WINDOW; y++) {
    if (y < 0 || y >= map.height) continue;
    const row = [`y=${String(y).padStart(2)}`];
    for (let x = b.x - WINDOW; x <= b.x + WINDOW; x++) {
      if (!inBounds(map, x, y)) {
        row.push('  ');
        continue;
      }
      const i = idx(map, x, y);
      let ch = known(i) ? TERRAIN_CHAR[terrainAt(map, x, y)] : '?';
      if (known(i) && state.cities.some((c) => c.x === x && c.y === y)) ch = 'C';
      if (sees(x, y)) {
        if (state.loot.some((l) => l.x === x && l.y === y)) ch = '$';
        const o = state.blobs.find((ob) => ob.x === x && ob.y === y);
        if (o) ch = o.team === b.team ? 'a' : allied(state, b.team, o.team) ? 'y' : 'E';
        if (boss && boss.x === x && boss.y === y) ch = 'X';
      }
      if (x === b.x && y === b.y) ch = '@';
      row.push(` ${ch}`);
    }
    rows.push(row.join(' '));
  }

  const need = Math.ceil(state.cities.length * R.SUPREMACY_SHARE);
  return {
    day: state.day,
    maxDays: state.config.maxDays,
    self: {
      id: b.id, name: b.name, team: b.team, x: b.x, y: b.y, hp: b.hp, maxHp: b.maxHp,
      rank: b.rank, kills: b.kills, gear: b.gear, vessel: b.vessel,
      attack: attackOf(b), range: rangeOf(b), armour: armourOf(state, b),
      terrain: terrainAt(map, b.x, b.y), vision,
      inOwnTerritory: tileTeam(territory, state.cities, idx(map, b.x, b.y)) === b.team,
      onCity: cityUnder ? { id: cityUnder.id, name: cityUnder.name, owner: cityUnder.owner ?? 'none', warden: !!cityUnder.warden } : null,
      onHarbor: !!devUnder && devUnder.kind === 'harbor' && tileTeam(territory, state.cities, idx(map, b.x, b.y)) === b.team,
    },
    team: {
      coins: team.coins,
      techs: [...team.techs],
      allies: [...team.allies],
      offersFrom: [...team.offersFrom],
      remnant: team.remnant,
      cities: myCities.map((c) => ({ id: c.id, name: c.name, x: c.x, y: c.y, level: c.level, pop: c.pop, walls: c.walls, isCapital: c.isCapital })),
      blobs: blobsOf(state, b.team).map((o) => ({ id: o.id, name: o.name, x: o.x, y: o.y, hp: o.hp })),
      blobCap: R.blobCap(myCities.length),
      income: myCities.reduce((s, c) => s + R.cityIncome(c.level, c.isCapital, c.founder === b.team), 0),
    },
    seenBlobs,
    knownCities,
    seenDevs,
    seenLoot,
    boss,
    bossNews,
    heard,
    standings: TEAM_IDS.map((t) => ({
      team: t, cities: citiesOf(state, t).length, eliminated: state.teams[t].eliminated, remnant: state.teams[t].remnant,
    })),
    supremacy: { ...state.supremacy, need },
    buildSpots,
    mapWindow: rows.join('\n'),
    memory: b.memory.slice(-4),
  };
}
