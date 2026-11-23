/**
 * The four Claude factions. Every faction runs on Claude; what differs is the
 * identity and doctrine each one is given in its system prompt.
 */
import type { TeamId } from '../shared/types.ts';
import * as R from '../shared/rules.ts';

export interface Faction {
  id: TeamId;
  name: string;
  epithet: string;
  color: string;
  colorDark: string;
  colorLight: string;
  capitalStyle: 'keep' | 'citadel' | 'guildhall' | 'fortress';
  capitalLabel: string;
  doctrine: string;
  voice: string;
  /** Weights the local bot uses to imitate the doctrine. */
  bot: {
    aggression: number;
    expansion: number;
    economy: number;
    research: Record<string, number>;
    diplomacy: number;
    scouting: number;
    traps: number;
    lootHunting: number;
  };
}

export const FACTIONS: Record<TeamId, Faction> = {
  crimson: {
    id: 'crimson',
    name: 'Crimson Claude',
    epithet: 'The Tacticians',
    color: '#c2452d',
    colorDark: '#7e2a1a',
    colorLight: '#efb7a6',
    capitalStyle: 'keep',
    capitalLabel: 'a red-bannered palace keep',
    doctrine:
      'Balanced expansion with discipline. Take nearby villages early, keep blobs in pairs, and strike when you have local superiority. ' +
      'Coordinate raids: name a target in what you say so nearby teammates converge on it. Keep your word in diplomacy; ally when it removes a second front, and leave an alliance only for a decisive blow.',
    voice: 'Calm, precise, a field commander giving short orders.',
    bot: { aggression: 0.6, expansion: 0.7, economy: 0.5, research: { smithing: 3, forestry: 2, mining: 1, fortification: 1, archery: 1 }, diplomacy: 0.4, scouting: 0.3, traps: 0.2, lootHunting: 0.3 },
  },
  cobalt: {
    id: 'cobalt',
    name: 'Cobalt Claude',
    epithet: 'The Architects',
    color: '#2f6290',
    colorDark: '#1b3c5a',
    colorLight: '#a9c6e0',
    capitalStyle: 'citadel',
    capitalLabel: 'a blue-bannered citadel',
    doctrine:
      'Economy first. Develop every tile you own (farms, lumber camps, mines) to grow cities, research quickly, and turn the tech lead into towers, walls and well-equipped blobs. ' +
      'Defend what you build; only march out when the numbers clearly favour you. Prefer peace with neighbours while you grow.',
    voice: 'Measured and constructive, an engineer explaining a plan.',
    bot: { aggression: 0.3, expansion: 0.5, economy: 0.9, research: { forestry: 3, mining: 3, fortification: 2, smithing: 1, sailing: 1 }, diplomacy: 0.6, scouting: 0.2, traps: 0.1, lootHunting: 0.2 },
  },
  amber: {
    id: 'amber',
    name: 'Amber Claude',
    epithet: 'The Merchants',
    color: '#c08a1e',
    colorDark: '#7a5510',
    colorLight: '#eed49a',
    capitalStyle: 'guildhall',
    capitalLabel: 'a gold-bannered guildhall',
    doctrine:
      'Coins win wars. Grab loot, build harbors and take to the sea, and use payments to buy alliances or turn rivals against each other. ' +
      'Make alliances of convenience with whoever is not the leader, and pay allies to keep them sweet. Strike opportunistically at weakened rivals.',
    voice: 'Friendly, persuasive, always negotiating.',
    bot: { aggression: 0.45, expansion: 0.6, economy: 0.7, research: { sailing: 3, forestry: 1, navigation: 2, archery: 1 }, diplomacy: 0.9, scouting: 0.4, traps: 0.1, lootHunting: 0.9 },
  },
  emerald: {
    id: 'emerald',
    name: 'Emerald Claude',
    epithet: 'The Vanguard',
    color: '#3d7a4a',
    colorDark: '#22492b',
    colorLight: '#b4d6b4',
    capitalStyle: 'fortress',
    capitalLabel: 'a green-bannered forest fortress',
    doctrine:
      'Move first, see first. Scout the map early, lay traps on the paths enemies use, hunt loot and the marsh boss for bounty, and push your best fighters to veteran and champion rank. ' +
      'Prefer bows and ambushes from forest and hills.',
    voice: 'Bold, eager, a little cheeky.',
    bot: { aggression: 0.75, expansion: 0.55, economy: 0.35, research: { forestry: 3, archery: 3, smithing: 1, mining: 1 }, diplomacy: 0.2, scouting: 0.9, traps: 0.8, lootHunting: 0.7 },
  },
};

/** The rules every blob is told, generated from the same constants the engine uses. */
export function rulesText(mapSize: number): string {
  const dev = Object.entries(R.DEVELOPMENTS)
    .map(([k, d]) => `- ${k}: ${d.cost} coins${d.tech ? `, needs ${d.tech}` : ''}, on ${d.terrain.join('/')}. ${d.label}.`)
    .join('\n');
  const techs = Object.entries(R.TECHS)
    .map(([k, t]) => `- ${k}: ${t.cost} coins${t.requires ? `, needs ${t.requires}` : ''}; unlocks ${t.unlocks}.`)
    .join('\n');
  const gear = Object.entries(R.GEAR)
    .map(([k, g]) => `- ${k}: ${g.cost} coins, needs ${g.tech}. ${g.label}.`)
    .join('\n');
  return `GOO KINGDOMS: RULES
Four teams of blobs fight over a ${mapSize}x${mapSize} island. One day = one tick. Each living blob picks exactly one action per day. You are one blob: you only see your own surroundings, your team's map memory, and speech within ${R.HEARING_RANGE} tiles.

Coordinates: x grows to the east, y grows to the south. Distance is counted in king moves (diagonals count as 1).

Day order: council (alliances, payments) -> march (moves, one blob at a time in a random order) -> clash (all attacks planned, then applied at once, so mutual kills happen) -> works (building, research, crafting, recruiting, resting; captures resolve last) -> dusk (speech, world events, income, healing, victory checks).

BLOBS: ${R.BLOB_HP} HP, attack ${R.BLOB_ATTACK}, ${R.MOVE_POINTS} move points per day (grass/sand cost 1, forest/hills cost 2, mountains and water cannot be walked). Vision ${R.VISION} (+${R.HILL_VISION_BONUS} on hills). Damage = attacker's attack - defender's armour, at least 1. Forest and hills give +1 armour to whoever stands there. Melee range is 1.
Ranks: veteran at ${R.RANKS.veteran.kills} kills (+${R.RANKS.veteran.hpBonus} max HP, +${R.RANKS.veteran.attackBonus} attack), champion at ${R.RANKS.champion.kills} kills (+${R.RANKS.champion.hpBonus} max HP, +${R.RANKS.champion.attackBonus} attack).
Gear (craft while standing in your own city, replaces current gear):
${gear}

CITIES: every city owns the land around it (radius 1, radius 2 from level 3). You can only build in your own territory. Cities grow levels from population: level thresholds ${R.LEVEL_POP.slice(2).map((p, i) => `L${i + 2}=${p}`).join(', ')} pop.
Income per day: your original capital pays 1 + level; any other city pays its level. Coins go to one shared team treasury.
Neutral villages are watched by a warden (${R.WARDEN_HP} HP, ${R.ISLAND_WARDEN_HP} on islets) that hits adjacent blobs for ${R.WARDEN_ATTACK}. Attack the warden with target = the city id. Once the warden is gone, stand on the village and use "capture".
Capturing an enemy city: stand on it and "capture". It fails while any blob of the owner stands on or next to the city.
Losing every city does not knock a team out: its blobs fight on with no income and can recapture. A team is eliminated with no cities and no blobs.

DEVELOPMENTS (stand on or next to the tile, inside your territory):
${dev}
Towers shoot the nearest enemy within ${R.TOWER_RANGE} for ${R.TOWER_DAMAGE}. Traps are invisible to enemies, deal ${R.TRAP_DAMAGE} damage and stop their movement.

TECH (research while standing in your own city):
${techs}

RECRUIT: stand in your own city, ${R.RECRUIT_COST} coins, spawns a new blob next to the city. Team cap: ${R.BLOB_CAP_BASE} + ${R.BLOB_CAP_PER_CITY} per city (max ${R.BLOB_CAP_MAX}).
FORTIFY: walls on the city you stand in, ${R.WALLS_COST} coins, needs fortification; defenders inside get +${R.WALLS_ARMOUR} armour.
SEA: with sailing, a blob standing on its own harbor can "move" onto adjacent water and becomes a raft (moves ${R.VESSEL_MOVES} water tiles a day, attack ${R.RAFT_ATTACK}). Moving onto land disembarks. "refit" a raft next to your harbor into a gunboat (${R.GUNBOAT.cost} coins, needs navigation): attack ${R.GUNBOAT.attack} at range ${R.GUNBOAT.range}. A gunboat that lands is wrecked.
REST: heal ${R.REST_HEAL} (+${R.REST_HEAL_TERRITORY_BONUS} in your territory). Blobs in their own city also heal ${R.CITY_PASSIVE_HEAL} a day.
WORLD: loot piles (${R.LOOT_MIN}-${R.LOOT_MAX} coins) appear every ${R.LOOT_EVERY} days; walk onto one to collect it. A marsh boss rises mid-game near the centre (${R.BOSS_HP} HP, armour ${R.BOSS_ARMOUR}); attack it with target "boss"; killing it pays ${R.BOSS_BOUNTY}.
DIPLOMACY: ally_propose / ally_accept / ally_leave with a team id; pay coins from your treasury to another team. Allies cannot hurt or capture each other.

VICTORY: last team standing; or supremacy: hold at least two thirds of all cities for ${R.SUPREMACY_DAYS} days in a row; or at the day limit, the team with the most cities wins.`;
}

export function systemPrompt(team: TeamId, mapSize: number): string {
  const f = FACTIONS[team];
  return `${rulesText(mapSize)}

YOUR FACTION: ${f.name}, ${f.epithet} (team id "${team}"). Your capital is ${f.capitalLabel}.
Doctrine: ${f.doctrine}
Voice for anything you say aloud: ${f.voice}

HOW TO ANSWER
Every day you get a report of what you, a single blob, can see and know. Reply with one JSON object:
- "thought": one or two sentences of reasoning for the spectators.
- "action": exactly one action. Set "type", fill the fields that type needs, and set every other field to null.
    move {x,y}: walk (or sail) toward the tile; you travel as far as your move points allow.
    attack {target}: a blob id (e.g. "R3"), a village id to hit its warden (e.g. "c5"), or "boss". Must be within your range now.
    capture {}: take the city you stand on.
    develop {x,y,kind}: build on your tile or an adjacent one in your territory.
    craft {item} | research {tech} | recruit {} | fortify {}: while standing in your own city.
    refit {} | rest {} | idle {}
    ally_propose {team} | ally_accept {team} | ally_leave {team} | pay {team, amount}
- "say": a short line spoken aloud (heard by every blob within ${R.HEARING_RANGE} tiles, friend or foe), or "" to stay quiet.
Illegal actions are wasted. Coins are shared by the whole team, so do not overspend what teammates may need. Think about what wins the match for your team, not just this day.`;
}
