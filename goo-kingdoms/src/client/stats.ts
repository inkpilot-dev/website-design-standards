/**
 * Faction stats: compares the four Claude factions across every finished
 * match on this machine, plus a live view of the current match.
 */
import type { Frame, TeamId } from '../shared/types.ts';
import { TEAM_IDS } from '../shared/types.ts';
import { FACTIONS } from '../agents/factions.ts';
import type { MatchRecord } from '../server/results.ts';
import { esc } from './util.ts';

let filter = 'claude';

const num = (v: unknown) => (typeof v === 'number' ? v : 0);
const pct = (a: number, b: number) => (b > 0 ? `${Math.round((100 * a) / b)}%` : 'n/a');
const fmt = (v: number, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : 'n/a');

function swatch(t: TeamId): string {
  return `<span class="sw" style="background:${FACTIONS[t].color}"></span>${esc(FACTIONS[t].name)}`;
}

function liveTable(frame: Frame | null): string {
  if (!frame) return '';
  const rows = TEAM_IDS.map((t) => {
    const s = frame.teams[t].stats;
    const cities = frame.cities.filter((c) => c.owner === t).length;
    return `<tr><td>${swatch(t)}</td><td>${cities}</td><td>${frame.blobs.filter((b) => b.team === t).length}</td><td>${frame.teams[t].coins}</td><td>${s.peakTerritory}</td><td>${s.kills} / ${s.deaths}</td><td>${s.coinsEarned}</td><td>${s.coinsPaid}</td><td>${s.messages}</td><td>${s.alliancesFormed}</td><td>${pct(s.validActions, s.decisions)}</td><td>${s.apiErrors}</td></tr>`;
  }).join('');
  return `<h2>This match, day ${frame.day}</h2><div class="table-wrap"><table><thead><tr><th>Faction</th><th>Cities</th><th>Blobs</th><th>Coins</th><th>Peak territory</th><th>Kills / deaths</th><th>Coins earned</th><th>Coins paid out</th><th>Lines spoken</th><th>Alliances</th><th>Valid actions</th><th>API errors</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function aggregate(records: MatchRecord[]): string {
  if (records.length === 0) {
    return `<p class="muted">No finished matches with this filter yet. Results appear here when a match ends.</p>`;
  }
  const rows = TEAM_IDS.map((t) => {
    let wins = 0, cities = 0, territory = 0, kills = 0, deaths = 0, earned = 0, paid = 0, messages = 0, alliances = 0, decisions = 0, valid = 0, errors = 0, tokens = 0;
    for (const r of records) {
      const s = r.teams[t] ?? {};
      if (r.winner === t) wins++;
      cities += num(s.cities);
      territory += num(s.peakTerritory);
      kills += num(s.kills);
      deaths += num(s.deaths);
      earned += num(s.coinsEarned);
      paid += num(s.coinsPaid);
      messages += num(s.messages);
      alliances += num(s.alliancesFormed);
      decisions += num(s.decisions);
      valid += num(s.validActions);
      errors += num(s.apiErrors);
      tokens += num(s.inputTokens) + num(s.outputTokens);
    }
    const n = records.length;
    return `<tr><td>${swatch(t)}</td><td>${wins}</td><td>${pct(wins, n)}</td><td>${fmt(cities / n)}</td><td>${fmt(territory / n)}</td><td>${fmt(kills / Math.max(1, deaths), 2)}</td><td>${fmt(earned / n, 0)}</td><td>${fmt(paid / n, 0)}</td><td>${fmt(messages / n, 0)}</td><td>${fmt(alliances / n)}</td><td>${pct(valid, decisions)}</td><td>${errors}</td><td>${decisions ? fmt(tokens / decisions, 0) : 'n/a'}</td></tr>`;
  }).join('');
  const draws = records.filter((r) => !r.winner).length;
  const spend = records.reduce((s, r) => s + (r.spendUsd ?? 0), 0);
  return `<div class="table-wrap"><table><thead><tr><th>Faction</th><th>Wins</th><th>Win rate</th><th>Cities at end</th><th>Peak territory</th><th>K/D</th><th>Coins earned</th><th>Coins paid out</th><th>Lines spoken</th><th>Alliances</th><th>Valid actions</th><th>API errors</th><th>Tokens per decision</th></tr></thead><tbody>${rows}</tbody></table></div>
  <p class="muted">${records.length} matches, ${draws} drawn. Averages are per match. Estimated API spend across these matches: $${spend.toFixed(2)}.</p>`;
}

function recentTable(records: MatchRecord[]): string {
  if (!records.length) return '';
  const rows = [...records].reverse().slice(0, 20).map((r) => `<tr><td>${esc(new Date(r.finishedAt).toLocaleString())}</td><td>${r.seed}</td><td>${r.winner ? swatch(r.winner as TeamId) : 'Draw'}</td><td>${esc(r.reason ?? '')}</td><td>${r.days}</td><td>${esc(r.brain === 'claude' ? r.model ?? 'Claude' : 'Local bot')}</td></tr>`).join('');
  return `<h2>Recent matches</h2><div class="table-wrap"><table><thead><tr><th>Finished</th><th>Seed</th><th>Winner</th><th>How</th><th>Days</th><th>Played by</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

export async function renderStats(el: HTMLElement, frame: Frame | null): Promise<void> {
  let records: MatchRecord[] = [];
  try {
    records = (await (await fetch('/api/results')).json()) as MatchRecord[];
  } catch {
    records = [];
  }
  const models = [...new Set(records.filter((r) => r.brain === 'claude').map((r) => r.model ?? ''))].filter(Boolean);
  const options = [
    ['claude', 'All Claude matches'],
    ...models.map((m) => [`model:${m}`, `Claude matches on ${m}`]),
    ['local', 'Local bot matches (not Claude)'],
    ['all', 'Every match'],
  ];
  const chosen = records.filter((r) =>
    filter === 'all' ? true : filter === 'local' ? r.brain === 'local' : filter === 'claude' ? r.brain === 'claude' : r.model === filter.slice(6));
  el.innerHTML = `
    <h1>Faction stats</h1>
    <p class="lede">How the four Claude factions compare across every finished match on this machine. Same model, different doctrines: wins, territory, combat, economy, diplomacy, talk, and how often each faction's chosen actions were legal.</p>
    ${liveTable(frame)}
    <h2>All finished matches</h2>
    <div class="filters"><label for="stats-filter">Show</label><select id="stats-filter">${options.map(([v, l]) => `<option value="${esc(v)}"${v === filter ? ' selected' : ''}>${esc(l)}</option>`).join('')}</select></div>
    ${aggregate(chosen)}
    ${recentTable(chosen)}`;
  el.querySelector<HTMLSelectElement>('#stats-filter')!.addEventListener('change', (e) => {
    filter = (e.target as HTMLSelectElement).value;
    void renderStats(el, frame);
  });
}
