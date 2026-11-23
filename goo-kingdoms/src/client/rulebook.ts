/**
 * The in-app rulebook, generated from the same constants the engine uses so
 * it is always accurate.
 */
import * as R from '../shared/rules.ts';
import { FACTIONS } from '../agents/factions.ts';
import { TEAM_IDS } from '../shared/types.ts';
import { esc } from './util.ts';

export function renderRulebook(): string {
  const sections: [string, string, string][] = [];
  const add = (id: string, title: string, body: string) => sections.push([id, title, body]);

  add('glance', 'At a glance', `
    <p>Four factions of blobs share one ${R.MAP_SIZE} by ${R.MAP_SIZE} island. Every faction is played by Claude, and every blob is its own Claude agent: each day it reads a report of what it can see, what its team remembers and what it heard nearby, then picks one action. You watch.</p>
    <ul>
      <li>Each team starts with a capital, ${R.START_BLOBS} blobs and ${R.START_COINS} coins.</li>
      <li>One day is one tick. A match lasts up to the day limit (default ${R.DEFAULT_MAX_DAYS}, at most ${R.MAX_DAYS_LIMIT}).</li>
      <li>Win by being the last team standing, by holding two thirds of all cities for ${R.SUPREMACY_DAYS} days in a row, or by holding the most cities when the day limit arrives.</li>
    </ul>`);

  add('day', 'How a day resolves', `
    <p>All blobs act in a freshly shuffled order every day, phase by phase:</p>
    <ol class="phases">
      <li><strong>Council.</strong> Alliance offers, acceptances, breaks and payments.</li>
      <li><strong>March.</strong> Moves resolve one blob at a time. Blobs spring traps, collect loot, set sail and land here.</li>
      <li><strong>Clash.</strong> Every attack (blobs, towers, wardens, the boss) is planned first, then all damage lands at once, so two blobs can defeat each other on the same day.</li>
      <li><strong>Works.</strong> Building, crafting, research, recruiting, walls, refits and rest. Captures resolve last, so a defender beaten in the Clash no longer blocks them.</li>
      <li><strong>Dusk.</strong> Speech is heard, the world moves (boss, loot, closing pits, promotions), then upkeep: musters, income, healing, vision, remnant and elimination checks, and victory.</li>
    </ol>
    <p>An action that is malformed or illegal is wasted: the blob does nothing that day. The faction stats track how often each faction's actions were valid.</p>`);

  const terrainRows = (Object.keys(R.TERRAIN_COST) as (keyof typeof R.TERRAIN_COST)[])
    .map((t) => `<tr><td>${esc(t)}</td><td>${R.TERRAIN_COST[t] ?? 'impassable on foot'}</td><td>${R.TERRAIN_COVER[t] ? `+${R.TERRAIN_COVER[t]}` : 'none'}</td></tr>`).join('');
  add('island', 'The island', `
    <p>Each seed generates the same island: one large landmass with mountain ridges, forests and hills, plus islets that only boats can reach. Capitals sit one per quarter of the map, with ${R.NEUTRAL_VILLAGES} neutral villages on the mainland and up to ${R.ISLAND_VILLAGES} on islets.</p>
    <table><thead><tr><th>Terrain</th><th>Move cost</th><th>Armour for the defender</th></tr></thead><tbody>${terrainRows}</tbody></table>
    <p>Blobs have ${R.MOVE_POINTS} move points a day and always manage at least one step. They see ${R.VISION} tiles around them (${R.VISION + R.HILL_VISION_BONUS} from hills). Fruit grows on some grass and crystal shows in some hills; both make a development worth more.</p>`);

  const levels = R.LEVEL_POP.slice(2).map((p, i) => `<tr><td>Level ${i + 2}</td><td>${p} pop</td><td>${R.cityIncome(i + 2, true, true)}</td><td>${R.cityIncome(i + 2, false, false)}</td></tr>`).join('');
  add('cities', 'Cities and territory', `
    <p>Each city owns the land around it: radius 1, and radius 2 from level 3. Neutral villages claim radius 1. Where territories overlap, the nearer city wins, and ties go to the older city. Water is never territory. Blobs may only build inside their own territory.</p>
    <table><thead><tr><th>Level</th><th>Needs</th><th>Original capital pays</th><th>Other cities pay</th></tr></thead><tbody>
      <tr><td>Level 1</td><td>0 pop</td><td>${R.cityIncome(1, true, true)}</td><td>${R.cityIncome(1, false, false)}</td></tr>${levels}</tbody></table>
    <p>Neutral villages are watched by a warden (${R.WARDEN_HP} HP, ${R.ISLAND_WARDEN_HP} on islets) that hits the weakest adjacent blob for ${R.WARDEN_ATTACK} each day and recovers 1 HP a day. Beat the warden, then stand on the village and capture it.</p>
    <p>To take an enemy city, stand on it and capture. The capture fails while any blob of the owner stands on the city or next to it. A team that loses every city is not out: its blobs fight on as wanderers with no income and can recapture. A team is eliminated only when it has no cities and no blobs. A team that still holds a city but has no blobs musters a free blob there every ${R.MUSTER_EVERY} days.</p>`);

  const gearRows = (Object.keys(R.GEAR) as (keyof typeof R.GEAR)[]).map((g) => `<tr><td>${g}</td><td>${R.GEAR[g].cost}</td><td>${R.GEAR[g].tech}</td><td>${esc(R.GEAR[g].label)}</td></tr>`).join('');
  add('blobs', 'Blobs, combat and ranks', `
    <p>Blobs start with ${R.BLOB_HP} HP and ${R.BLOB_ATTACK} attack, with melee range 1. Damage is the attacker's attack minus the defender's armour, never less than 1. Armour comes from shields, forest and hill cover, and city walls.</p>
    <table><thead><tr><th>Gear</th><th>Cost</th><th>Needs</th><th>Effect</th></tr></thead><tbody>${gearRows}</tbody></table>
    <p>Gear is crafted while standing in your own city and replaces whatever the blob carried. Ranks come from kills: veteran at ${R.RANKS.veteran.kills} (+${R.RANKS.veteran.hpBonus} max HP, +${R.RANKS.veteran.attackBonus} attack) and champion at ${R.RANKS.champion.kills} (+${R.RANKS.champion.hpBonus} max HP, +${R.RANKS.champion.attackBonus} attack). A promotion heals by the HP it adds.</p>
    <p>Resting heals ${R.REST_HEAL} HP, or ${R.REST_HEAL + R.REST_HEAL_TERRITORY_BONUS} inside your territory. Blobs standing in their own city also heal ${R.CITY_PASSIVE_HEAL} a day. Recruiting costs ${R.RECRUIT_COST} coins from inside a city; a team may field ${R.BLOB_CAP_BASE} blobs plus ${R.BLOB_CAP_PER_CITY} per city, up to ${R.BLOB_CAP_MAX}.</p>`);

  const devRows = (Object.keys(R.DEVELOPMENTS) as (keyof typeof R.DEVELOPMENTS)[]).map((k) => {
    const d = R.DEVELOPMENTS[k];
    return `<tr><td>${k}</td><td>${d.cost}</td><td>${d.tech ?? 'none'}</td><td>${d.terrain.join(', ')}</td><td>${esc(d.label)}</td></tr>`;
  }).join('');
  add('build', 'Developments', `
    <p>A blob builds on its own tile or an adjacent one, inside its own territory. Population from developments grows the city that owns the tile.</p>
    <table><thead><tr><th>Building</th><th>Cost</th><th>Needs</th><th>Terrain</th><th>Effect</th></tr></thead><tbody>${devRows}</tbody></table>
    <p>Towers fire every day at the nearest enemy within ${R.TOWER_RANGE} for ${R.TOWER_DAMAGE}. Traps are invisible to enemies. An enemy that steps on one takes ${R.TRAP_DAMAGE} damage and stops, and the trap leaves an open pit for ${R.PIT_DAYS} days. City walls cost ${R.WALLS_COST} and give defenders inside +${R.WALLS_ARMOUR} armour.</p>`);

  const techRows = (Object.keys(R.TECHS) as (keyof typeof R.TECHS)[]).map((k) => `<tr><td>${k}</td><td>${R.TECHS[k].cost}</td><td>${R.TECHS[k].requires ?? 'none'}</td><td>${esc(R.TECHS[k].unlocks)}</td></tr>`).join('');
  add('tech', 'Tech tree', `
    <p>Research happens while a blob stands in one of its team's cities. Techs belong to the whole team.</p>
    <table><thead><tr><th>Tech</th><th>Cost</th><th>Requires</th><th>Unlocks</th></tr></thead><tbody>${techRows}</tbody></table>`);

  add('sea', 'Seas and boats', `
    <p>With sailing, a blob on its own harbor can move onto adjacent water and becomes a raft. Rafts sail ${R.VESSEL_MOVES} water tiles a day and fight with attack ${R.RAFT_ATTACK}. Moving onto land disembarks. A raft next to its team's harbor can be refitted into a gunboat (${R.GUNBOAT.cost} coins, needs navigation) that fires at range ${R.GUNBOAT.range} for ${R.GUNBOAT.attack}. A gunboat that lands is wrecked on the shore.</p>`);

  add('talk', 'Diplomacy and speech', `
    <p>Any blob can offer an alliance to another team, accept a standing offer, or break an alliance. Allies cannot attack or capture each other, and their traps do not trigger for each other. Any blob can pay coins from its treasury to another team.</p>
    <p>Each blob can also say one short line a day (up to ${R.SAY_MAX_CHARS} characters). Every blob within ${R.HEARING_RANGE} tiles hears it the next day, friend or foe, so plans can leak.</p>`);

  add('world', 'Loot and the marsh boss', `
    <p>Every ${R.LOOT_EVERY} days a loot pile of ${R.LOOT_MIN} to ${R.LOOT_MAX} coins washes up somewhere on land. The first blob to step on it collects it for its team.</p>
    <p>Between day ${R.BOSS_FIRST_RISE} and day ${R.BOSS_FIRST_RISE + 20}, a marsh boss rises near the centre, announced ${R.BOSS_WARNING_DAYS} days ahead. It has ${R.BOSS_HP} HP and armour ${R.BOSS_ARMOUR}, never heals, slams every adjacent blob for ${R.BOSS_SLAM} each day, and crawls toward the nearest city, trampling ${R.BOSS_RAZE} population a day from any city it stands on. The team that lands the killing blow earns ${R.BOSS_BOUNTY} coins.</p>`);

  add('victory', 'Victory', `
    <ol>
      <li><strong>Last team standing.</strong> Every other team is eliminated.</li>
      <li><strong>Supremacy.</strong> One team holds at least two thirds of all cities for ${R.SUPREMACY_DAYS} days in a row.</li>
      <li><strong>Day limit.</strong> When the limit arrives, the team with the most cities wins. Ties go to total city levels, then coins.</li>
    </ol>`);

  const factions = TEAM_IDS.map((t) => {
    const f = FACTIONS[t];
    return `<div class="faction"><span class="bar" style="background:${f.color}"></span><div><h3>${esc(f.name)}, ${esc(f.epithet)}</h3><p>Capital: ${esc(f.capitalLabel)}.</p><p>${esc(f.doctrine)}</p><p class="muted">Voice: ${esc(f.voice)}</p></div></div>`;
  }).join('');
  add('factions', 'The four Claude factions', `
    <p>All four teams run on the same Claude model. They differ only in the identity and doctrine their system prompt gives them, so the stats tab shows how much a doctrine changes play.</p>${factions}`);

  add('brains', 'Who is actually playing', `
    <p><strong>Claude mode.</strong> Every living blob is a separate request to the Anthropic API each day. The request carries the faction's system prompt (the rules above plus its doctrine) and that blob's own report: its stats, a map window around it with fog of war, the blobs and cities it can see or remembers, what it heard, and its last few actions and their results. Claude answers with a JSON action that must match a strict schema. Anything missing or malformed becomes an idle day.</p>
    <p><strong>Local bot.</strong> A built-in rule-based policy imitates each doctrine so the game runs with no API key and no cost. It is not Claude. Matches it plays are recorded separately in the stats.</p>
    <p><strong>Your data.</strong> In Claude mode, each blob's report is sent to the Anthropic API. Your API key stays in the server's memory on your machine (or your <code>.env</code> file) and is never written anywhere by the app. Match results are saved locally in <code>data/results.json</code>.</p>`);

  const toc = sections.map(([id, title]) => `<li><a href="#rb-${id}">${esc(title)}</a></li>`).join('');
  const body = sections.map(([id, title, html]) => `<h2 id="rb-${id}">${esc(title)}</h2>${html}`).join('');
  return `<h1>Rulebook</h1><p class="lede">Everything here is generated from the game's own rule constants, so it always matches what the engine does.</p><ul class="toc">${toc}</ul>${body}`;
}
