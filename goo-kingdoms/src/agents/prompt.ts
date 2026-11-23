/**
 * Turns one blob's Observation into the daily report Claude reads.
 */
import type { Observation } from '../sim/observe.ts';
import { describeAction } from './actions.ts';
import { DEVELOPMENTS, GEAR, TECHS } from '../shared/rules.ts';

export function renderObservation(o: Observation): string {
  const s = o.self;
  const lines: string[] = [];
  lines.push(`DAY ${o.day + 1} of ${o.maxDays}.`);
  lines.push(
    `You are ${s.name} (id ${s.id}) of team ${s.team}, a ${s.rank} with ${s.kills} kills. HP ${s.hp}/${s.maxHp}, attack ${s.attack}, range ${s.range}, armour ${s.armour}. ` +
    `Position (${s.x},${s.y}) on ${s.terrain}${s.vessel ? `, sailing a ${s.vessel}` : ''}. Gear: ${s.gear ?? 'none'}. Vision ${s.vision}.` +
    `${s.inOwnTerritory ? ' You are inside your own territory.' : ''}` +
    `${s.onCity ? ` You are standing on city ${s.onCity.name} (${s.onCity.id}, owner ${s.onCity.owner}${s.onCity.warden ? ', warden alive' : ''}).` : ''}` +
    `${s.onHarbor ? ' You are on your own harbor.' : ''}`,
  );

  const t = o.team;
  lines.push('');
  lines.push(`TEAM: ${t.coins} coins (+${t.income}/day). Techs: ${t.techs.join(', ') || 'none'}. Allies: ${t.allies.join(', ') || 'none'}. Pending alliance offers from: ${t.offersFrom.join(', ') || 'none'}.${t.remnant ? ' Your team holds no cities: you are a wanderer band with no income.' : ''}`);
  lines.push(`Your cities: ${t.cities.map((c) => `${c.name} ${c.id} (${c.x},${c.y}) L${c.level} pop ${c.pop}${c.walls ? ' walled' : ''}${c.isCapital ? ' capital' : ''}`).join('; ') || 'none'}.`);
  lines.push(`Your blobs (${t.blobs.length}/${t.blobCap}): ${t.blobs.map((b) => `${b.id} ${b.name} (${b.x},${b.y}) ${b.hp}hp`).join('; ')}.`);
  const unknownTechs = Object.entries(TECHS).filter(([k]) => !t.techs.includes(k as never)).map(([k, v]) => `${k} ${v.cost}c${v.requires ? ` (needs ${v.requires})` : ''}`);
  lines.push(`Techs you could research: ${unknownTechs.join(', ') || 'all known'}.`);

  lines.push('');
  lines.push('STANDINGS (public): ' + o.standings.map((st) => `${st.team} ${st.eliminated ? 'eliminated' : `${st.cities} cities${st.remnant ? ', wanderers' : ''}`}`).join('; ') + '.');
  if (o.supremacy.team) lines.push(`Supremacy: ${o.supremacy.team} holds ${o.supremacy.need}+ cities, day ${o.supremacy.days}.`);
  if (o.bossNews) lines.push(`NEWS: ${o.bossNews}`);

  lines.push('');
  lines.push('MAP around you (@ you, a teammate, y ally, E enemy blob, X boss, C city, $ loot, ~ sea, - shallows, : sand, . grass, f forest, h hills, M mountain, ? unexplored):');
  lines.push(o.mapWindow);

  lines.push('');
  lines.push('BLOBS YOU SEE:');
  if (o.seenBlobs.length === 0) lines.push('- none');
  for (const b of o.seenBlobs) {
    lines.push(`- ${b.id} ${b.name}, ${b.team}${b.friendly ? (b.team === s.team ? ' (teammate)' : ' (ally)') : ' (ENEMY)'}, (${b.x},${b.y}) distance ${b.distance}, ${b.hp}/${b.maxHp} HP, ${b.rank}${b.gear ? `, ${b.gear}` : ''}${b.vessel ? `, on a ${b.vessel}` : ''}`);
  }
  if (o.boss) lines.push(`- BOSS ${o.boss.name} at (${o.boss.x},${o.boss.y}) distance ${o.boss.distance}, ${o.boss.hp} HP. Target id "boss".`);

  lines.push('');
  lines.push('CITIES YOUR TEAM KNOWS:');
  if (o.knownCities.length === 0) lines.push('- none');
  for (const c of o.knownCities.slice(0, 14)) {
    lines.push(`- ${c.id} ${c.name} (${c.x},${c.y}) distance ${c.distance}, owner ${c.owner}${c.isCapital ? ', capital' : ''}${c.island ? ', on an islet (reach it by boat)' : ''}, L${c.level}${c.walls ? ', walled' : ''}${c.wardenHp !== null ? `, warden ${c.wardenHp} HP` : ''}${c.visible ? '' : ' (last seen)'}`);
  }

  if (o.seenDevs.length) {
    lines.push('');
    lines.push('DEVELOPMENTS YOU SEE: ' + o.seenDevs.map((d) => `${d.kind}${d.kind === 'trap' ? (d.armed ? ' (armed, yours/ally)' : ' (sprung pit)') : ''} at (${d.x},${d.y}) [${d.team ?? 'no territory'}]`).join('; '));
  }
  if (o.seenLoot.length) {
    lines.push('LOOT YOU SEE: ' + o.seenLoot.map((l) => `${l.coins} coins at (${l.x},${l.y})`).join('; '));
  }
  if (o.buildSpots.length) {
    lines.push('');
    lines.push('YOUR TERRITORY NEXT TO YOU: ' + o.buildSpots.map((b) => `(${b.x},${b.y}) ${b.terrain}${b.resource ? `+${b.resource}` : ''}${b.coastal ? ' coastal' : ''}${b.developed ? ` [${b.developed}]` : ''}`).join('; '));
    lines.push('Build costs: ' + Object.entries(DEVELOPMENTS).map(([k, d]) => `${k} ${d.cost}`).join(', ') + '. Gear: ' + Object.entries(GEAR).map(([k, g]) => `${k} ${g.cost}`).join(', ') + '.');
  }

  lines.push('');
  lines.push('HEARD NEARBY:');
  if (o.heard.length === 0) lines.push('- nothing');
  for (const h of o.heard) lines.push(`- day ${h.day}, ${h.name} (${h.blobId}, ${h.team}) at (${h.x},${h.y}): "${h.text}"`);

  if (o.memory.length) {
    lines.push('');
    lines.push('YOUR RECENT DAYS:');
    for (const m of o.memory) lines.push(`- day ${m.day}: ${describeAction(m.action)} -> ${m.ok ? 'ok' : 'FAILED'}: ${m.note}`);
  }

  lines.push('');
  lines.push('Choose your one action for today.');
  return lines.join('\n');
}
