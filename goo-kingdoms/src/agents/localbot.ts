/**
 * Local bot: a fast, deterministic heuristic policy that imitates each
 * faction's doctrine. It runs with no API key so the game always works
 * offline. It is not Claude, and the stats tab keeps its results separate.
 *
 * Like the Claude agents, it decides from the blob's own Observation (plus
 * its team's map memory for scouting).
 */
import type { Action, Blob, Decision, DevKind, GameState, Gear, Tech } from '../shared/types.ts';
import { Rng, mixSeed } from '../shared/rng.ts';
import { NEIGHBOURS, dist, exploredSet, idx, inBounds, isWalkable, terrainAt } from '../shared/geo.ts';
import * as R from '../shared/rules.ts';
import { observe } from '../sim/observe.ts';
import type { Observation } from '../sim/observe.ts';
import { FACTIONS } from './factions.ts';

const LINES: Record<string, string[]> = {
  crimson: ['Hold the line.', 'On my mark.', 'Converge on the weak point.', 'Steady. Pairs, not singles.', 'This village is ours by noon.'],
  cobalt: ['Measure twice, build once.', 'Another farm, another level.', 'Walls first, questions later.', 'Growth is the best defence.', 'Let them come to our towers.'],
  amber: ['Everything has a price.', 'Shall we talk terms?', 'Gold speaks every language.', 'A fine day for business.', 'Friends get a discount.'],
  emerald: ['Saw you first!', 'Mind your step.', 'Race you to the next hill.', 'Too slow!', 'The woods are on our side.'],
};

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

export function decideLocal(state: GameState, b: Blob): Decision {
  const obs = observe(state, b);
  const f = FACTIONS[b.team];
  const w = f.bot;
  const rng = new Rng(mixSeed(state.config.seed, state.day * 7919 + hash(b.id)));
  const out = (action: Action, thought: string, sayChance = 0.06, sayText?: string): Decision => ({
    action,
    thought: `Local bot: ${thought}`,
    say: sayText ?? (rng.next() < sayChance ? rng.pick(LINES[b.team]) : ''),
    wellFormed: true,
    source: 'local',
  });
  const s = obs.self;
  const team = obs.team;
  const blocked = (x: number, y: number) =>
    obs.seenBlobs.some((o) => o.x === x && o.y === y) ||
    (obs.boss !== null && obs.boss.x === x && obs.boss.y === y) ||
    obs.knownCities.some((c) => c.x === x && c.y === y && c.wardenHp !== null && c.visible);
  /** Move toward a tile, unless we are already next to it and it is occupied. */
  const goTo = (x: number, y: number, thought: string, sayChance = 0.04): Decision | null =>
    dist(s.x, s.y, x, y) <= 1 && blocked(x, y) ? null : out({ type: 'move', x, y }, thought, sayChance);
  const enemies = obs.seenBlobs.filter((o) => !o.friendly);
  const inRange = enemies.filter((e) => e.distance <= s.range).sort((p, q) => p.hp - q.hp);

  // 1. Diplomacy, rarely and by doctrine.
  if (team.offersFrom.length && rng.next() < w.diplomacy * 0.5) {
    const leader = [...obs.standings].sort((p, q) => q.cities - p.cities)[0];
    const pick = team.offersFrom.find((t) => t !== leader.team) ?? (rng.next() < w.diplomacy * 0.4 ? team.offersFrom[0] : null);
    if (pick) return out({ type: 'ally_accept', team: pick }, `accepting ${pick}'s alliance offer.`, 0.5, 'Agreed. Let us not waste blobs on each other.');
  }
  if (team.allies.length === 0 && rng.next() < w.diplomacy * 0.03) {
    const others = obs.standings.filter((st) => st.team !== s.team && !st.eliminated);
    const leader = [...others].sort((p, q) => q.cities - p.cities)[0];
    const candidates = others.filter((st) => st !== leader);
    if (candidates.length) {
      const t = rng.pick(candidates).team;
      return out({ type: 'ally_propose', team: t }, `offering ${t} an alliance against the leader.`, 0.6, `${t}, an alliance would serve us both.`);
    }
  }
  if (b.team === 'amber' && team.allies.length && team.coins > 40 && rng.next() < 0.04) {
    return out({ type: 'pay', team: team.allies[0], amount: 5 }, `paying our ally ${team.allies[0]} to keep the friendship warm.`, 0.5, 'A small token of goodwill.');
  }

  // 2. Badly hurt: rest at home or fall back.
  const home = nearestOwnCity(obs);
  if (s.hp <= Math.max(3, s.maxHp * 0.35)) {
    if (inRange.length && s.hp > 2 && inRange[0].hp <= s.attack) {
      return out({ type: 'attack', target: inRange[0].id }, `wounded, but ${inRange[0].name} is one hit from defeat.`);
    }
    if (enemies.length && home && !(s.x === home.x && s.y === home.y)) {
      const d = goTo(home.x, home.y, `wounded (${s.hp} HP), falling back to ${home.name}.`, 0.1);
      if (d) return d;
    }
    return out({ type: 'rest' }, `resting to recover (${s.hp}/${s.maxHp} HP).`, 0.02);
  }

  // 3. Fight what is in reach.
  if (inRange.length && rng.next() < 0.55 + w.aggression * 0.45) {
    const t = inRange[0];
    return out({ type: 'attack', target: t.id }, `attacking ${t.name} (${t.team}, ${t.hp} HP) in range.`, 0.08);
  }
  if (obs.boss && obs.boss.distance <= s.range && s.hp >= 8 && rng.next() < w.aggression + w.lootHunting * 0.3) {
    return out({ type: 'attack', target: 'boss' }, `striking ${obs.boss.name} for the bounty.`, 0.15, `For the bounty!`);
  }
  const warden = obs.knownCities.find((c) => c.owner === 'none' && c.wardenHp !== null && c.distance <= s.range && c.visible);
  if (warden && s.hp >= 5) {
    return out({ type: 'attack', target: warden.id }, `hitting the warden of ${warden.name} (${warden.wardenHp} HP).`);
  }

  // 4. Capture the city underfoot.
  if (s.onCity && s.onCity.owner !== s.team && !s.onCity.warden && !(s.onCity.owner !== 'none' && team.allies.includes(s.onCity.owner))) {
    return out({ type: 'capture' }, `capturing ${s.onCity.name}.`, 0.4, `${s.onCity.name} is ours now.`);
  }

  // 5. In our own city: recruit, research, craft, fortify.
  if (s.onCity && s.onCity.owner === s.team) {
    const threatened = enemies.some((e) => e.distance <= 2);
    if (team.blobs.length < team.blobCap && team.coins >= R.RECRUIT_COST + 2 && rng.next() < 0.35 + w.expansion * 0.4) {
      return out({ type: 'recruit' }, `recruiting a new blob (${team.blobs.length}/${team.blobCap}).`);
    }
    const tech = pickTech(obs, w.research, rng);
    if (tech && rng.next() < 0.3 + w.economy * 0.4) {
      return out({ type: 'research', tech }, `researching ${tech}.`, 0.03);
    }
    const gear = pickGear(obs, b.team);
    if (gear && rng.next() < 0.5) {
      return out({ type: 'craft', item: gear }, `crafting a ${gear}.`);
    }
    if (team.techs.includes('fortification') && team.coins >= R.WALLS_COST + 4 && (threatened || b.team === 'cobalt') && rng.next() < 0.5) {
      const city = team.cities.find((c) => c.x === s.x && c.y === s.y);
      if (city && !city.walls) return out({ type: 'fortify' }, `raising walls at ${city.name}.`);
    }
    if (threatened && rng.next() < 0.7) return out({ type: 'rest' }, 'holding the city against nearby enemies.', 0.1, 'Not one step further.');
  }

  // 6. Develop our land.
  const build = pickBuild(obs, b.team, w.traps, enemies.length > 0, rng);
  if (build && rng.next() < 0.25 + w.economy * 0.6) {
    return out({ type: 'develop', x: build.x, y: build.y, kind: build.kind }, `building a ${build.kind} at (${build.x},${build.y}).`, 0.03);
  }

  // 7. Choose somewhere to go.
  const loot = obs.seenLoot.sort((p, q) => p.distance - q.distance)[0];
  if (loot && rng.next() < 0.3 + w.lootHunting * 0.7) {
    const d = goTo(loot.x, loot.y, `going for ${loot.coins} coins of loot.`, 0.05);
    if (d) return d;
  }
  if (obs.boss && s.hp >= 9 && rng.next() < w.aggression * 0.6) {
    const d = goTo(obs.boss.x, obs.boss.y, `closing in on ${obs.boss.name}.`);
    if (d) return d;
  }
  const weakest = [...enemies].sort((p, q) => p.hp - q.hp)[0];
  if (weakest && weakest.distance > s.range && rng.next() < w.aggression) {
    const d = goTo(weakest.x, weakest.y, `pursuing ${weakest.name} (${weakest.hp} HP).`, 0.05);
    if (d) return d;
  }
  if (inRange.length) {
    return out({ type: 'attack', target: inRange[0].id }, `engaging ${inRange[0].name} rather than leave it be.`);
  }

  // Defend: architects keep one blob at home when enemies are near.
  if (home && b.team === 'cobalt' && enemies.length && dist(s.x, s.y, home.x, home.y) <= 2) {
    if (s.x === home.x && s.y === home.y) return out({ type: 'rest' }, `guarding ${home.name}.`);
    const d = goTo(home.x, home.y, `guarding ${home.name}.`);
    if (d) return d;
  }

  const targets = obs.knownCities
    .filter((c) => c.owner !== s.team && !(c.owner !== 'none' && team.allies.includes(c.owner)))
    .filter((c) => c.owner === 'none' || rng.next() < 0.35 + w.aggression * 0.6)
    .sort((p, q) => p.distance - q.distance);
  if (targets.length && rng.next() < 0.4 + w.expansion * 0.6) {
    const t = targets[Math.min(targets.length - 1, rng.int(0, Math.min(2, targets.length - 1)))];
    const d = goTo(t.x, t.y, `marching on ${t.name} (${t.owner === 'none' ? 'neutral village' : t.owner}, ${t.distance} tiles).`, 0.05);
    if (d) return d;
  }

  const frontier = nearestFrontier(state, b, rng);
  if (frontier && rng.next() < 0.3 + w.scouting * 0.7) {
    const d = goTo(frontier[0], frontier[1], `scouting unexplored land toward (${frontier[0]},${frontier[1]}).`, 0.03);
    if (d) return d;
  }
  if (home && dist(s.x, s.y, home.x, home.y) > 2) {
    const d = goTo(home.x, home.y, `returning toward ${home.name}.`);
    if (d) return d;
  }
  if (s.hp < s.maxHp) return out({ type: 'rest' }, 'nothing pressing, resting.');
  return out({ type: 'idle' }, 'waiting for an opening.', 0.04);
}

function nearestOwnCity(obs: Observation) {
  return [...obs.team.cities].sort((p, q) => dist(obs.self.x, obs.self.y, p.x, p.y) - dist(obs.self.x, obs.self.y, q.x, q.y))[0] ?? null;
}

function pickTech(obs: Observation, weights: Record<string, number>, rng: Rng): Tech | null {
  const known = new Set(obs.team.techs);
  const options = (Object.keys(R.TECHS) as Tech[])
    .filter((t) => !known.has(t))
    .filter((t) => !R.TECHS[t].requires || known.has(R.TECHS[t].requires!))
    .filter((t) => obs.team.coins >= R.TECHS[t].cost + 4)
    .map((t) => ({ t, score: (weights[t] ?? 0.3) + rng.next() }))
    .sort((p, q) => q.score - p.score);
  return options[0]?.t ?? null;
}

function pickGear(obs: Observation, team: string): Gear | null {
  if (obs.self.gear || obs.self.vessel) return null;
  const has = (t: Tech) => obs.team.techs.includes(t);
  const order: Gear[] = team === 'emerald' ? ['bow', 'spear', 'shield'] : team === 'cobalt' ? ['shield', 'spear', 'bow'] : ['spear', 'bow', 'shield'];
  for (const g of order) {
    if (has(R.GEAR[g].tech) && obs.team.coins >= R.GEAR[g].cost + 6) return g;
  }
  return null;
}

function pickBuild(obs: Observation, team: string, trapWeight: number, enemiesNear: boolean, rng: Rng): { x: number; y: number; kind: DevKind } | null {
  const has = (t: Tech | null) => !t || obs.team.techs.includes(t);
  const coins = obs.team.coins;
  const spots = obs.buildSpots.filter((s) => !s.developed);
  const options: { x: number; y: number; kind: DevKind; score: number }[] = [];
  for (const s of spots) {
    for (const kind of Object.keys(R.DEVELOPMENTS) as DevKind[]) {
      const d = R.DEVELOPMENTS[kind];
      if (!d.terrain.includes(s.terrain) || !has(d.tech) || coins < d.cost + 3) continue;
      if (kind === 'harbor' && !s.coastal) continue;
      let score = d.pop + (kind === 'farm' && s.resource === 'fruit' ? 1.5 : 0) + (kind === 'mine' && s.resource === 'crystal' ? 1.5 : 0);
      if (kind === 'trap') score = trapWeight * (enemiesNear ? 3 : 1);
      if (kind === 'tower') score = enemiesNear ? 2.5 : team === 'cobalt' ? 0.8 : 0.2;
      if (kind === 'harbor') score = team === 'amber' ? 2.5 : 0.6;
      options.push({ x: s.x, y: s.y, kind, score: score + rng.next() * 0.5 });
    }
  }
  options.sort((p, q) => q.score - p.score);
  return options[0] ?? null;
}

/** The nearest explored, walkable tile that borders unexplored land. */
function nearestFrontier(state: GameState, b: Blob, rng: Rng): [number, number] | null {
  const map = state.map;
  const known = exploredSet(state.teams[b.team].explored);
  let best: [number, number] | null = null;
  let bestScore = Infinity;
  for (let y = 0; y < map.height; y++) {
    for (let x = 0; x < map.width; x++) {
      const i = idx(map, x, y);
      if (!known(i) || !isWalkable(terrainAt(map, x, y))) continue;
      const frontier = NEIGHBOURS.some(([dx, dy]) => inBounds(map, x + dx, y + dy) && !known(idx(map, x + dx, y + dy)));
      if (!frontier) continue;
      const score = dist(b.x, b.y, x, y) + rng.next() * 3;
      if (score < bestScore) {
        bestScore = score;
        best = [x, y];
      }
    }
  }
  return best;
}

/** Decide for every listed blob (instant). */
export function decideAllLocal(state: GameState, blobs: Blob[]): Record<string, Decision> {
  const out: Record<string, Decision> = {};
  for (const b of blobs) out[b.id] = decideLocal(state, b);
  return out;
}

