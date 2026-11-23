/**
 * Spectator app: connects to the server's event stream, drives the renderer,
 * and fills the standings, inspector, chronicle, replay and settings.
 */
import type { Blob, Frame, GameConfig, LogLine, TeamId, WorldMap } from '../shared/types.ts';
import { TEAM_IDS } from '../shared/types.ts';
import type { PublicStatus } from '../server/match.ts';
import type { ModelInfo } from '../agents/claude.ts';
import { FACTIONS } from '../agents/factions.ts';
import { describeAction } from '../agents/actions.ts';
import * as R from '../shared/rules.ts';
import { computeTerritory, idx } from '../shared/geo.ts';
import { Renderer } from './render.ts';
import type { Selection } from './render.ts';
import { renderRulebook } from './rulebook.ts';
import { renderStats } from './stats.ts';
import { $, blobIcon, cap, esc, post } from './util.ts';

interface InitPayload {
  map: WorldMap;
  config: GameConfig;
  frame: Frame;
  status: PublicStatus;
  models: ModelInfo[];
  log: LogLine[];
}

const app = {
  map: null as WorldMap | null,
  config: null as GameConfig | null,
  status: null as PublicStatus | null,
  models: [] as ModelInfo[],
  frames: new Map<number, Frame>(),
  latest: null as Frame | null,
  shown: null as Frame | null,
  live: true,
  log: [] as LogLine[],
  statsOpen: false,
};

const icon = (t: TeamId) => blobIcon(FACTIONS[t].color, FACTIONS[t].colorDark, FACTIONS[t].colorLight);

const renderer = new Renderer($('board') as HTMLCanvasElement, $('overlay'));
const TEAM_NAME = (t: TeamId) => FACTIONS[t].name;
const SPEED_MS: Record<string, number> = { '1x': 1200, '2x': 600, '5x': 240, max: 120 };

// ---------------------------------------------------------------------------
// Event stream
// ---------------------------------------------------------------------------

function connect(): void {
  const es = new EventSource('/api/events');
  es.addEventListener('init', (e) => onInit(JSON.parse((e as MessageEvent).data) as InitPayload));
  es.addEventListener('frame', (e) => onFrame(JSON.parse((e as MessageEvent).data) as Frame));
  es.addEventListener('status', (e) => onStatus(JSON.parse((e as MessageEvent).data) as PublicStatus));
  es.onerror = () => showBanner('Lost connection to the server. Retrying...');
  es.onopen = () => hideBannerIf('Lost connection');
}

function onInit(p: InitPayload): void {
  app.map = p.map;
  app.config = p.config;
  app.models = p.models;
  app.frames.clear();
  app.frames.set(p.frame.day, p.frame);
  app.latest = p.frame;
  app.live = true;
  app.log = p.log ?? [];
  renderer.setMap(p.map);
  renderer.selection = null;
  showFrame(p.frame, false);
  (document.getElementById('seed-input') as HTMLInputElement).value = String(p.config.seed);
  onStatus(p.status);
  renderChronicle();
}

function onFrame(f: Frame): void {
  app.frames.set(f.day, f);
  app.latest = f;
  app.log.push(...f.log);
  if (app.log.length > 600) app.log.splice(0, app.log.length - 600);
  if (app.live) showFrame(f, true);
  updateReplay();
  renderChronicle();
  if (app.statsOpen) void renderStats($('stats'), app.latest);
}

function onStatus(s: PublicStatus): void {
  app.status = s;
  const play = $('btn-play') as HTMLButtonElement;
  play.textContent = s.status === 'running' ? 'Pause' : 'Play';
  play.disabled = s.status === 'over';
  ($('btn-step') as HTMLButtonElement).disabled = s.status !== 'paused' || !!s.thinking;
  for (const b of document.querySelectorAll<HTMLButtonElement>('#speed button')) {
    b.setAttribute('role', 'radio');
    b.setAttribute('aria-checked', String(b.dataset.speed === s.speed));
  }
  const model = app.models.find((m) => m.id === s.settings.model);
  const chip = $('brain-chip');
  chip.className = `brain-chip ${s.settings.brain}`;
  chip.innerHTML = s.settings.brain === 'claude'
    ? `<span class="dot"></span>${esc(model?.label ?? s.settings.model)} · $${s.spend.usd.toFixed(2)} of $${s.settings.budgetUsd.toFixed(2)}`
    : '<span class="dot"></span>Local bot (no API key, not Claude)';

  const th = $('thinking');
  if (s.thinking) {
    th.hidden = false;
    $('thinking-fill').style.width = `${(100 * s.thinking.done) / Math.max(1, s.thinking.total)}%`;
    $('thinking-text').textContent = `Claude is deciding for each blob: ${s.thinking.done} of ${s.thinking.total} answered`;
  } else th.hidden = true;

  if (s.error) showBanner(s.error);
  else hideBannerIf('');
}

// ---------------------------------------------------------------------------
// Showing a frame
// ---------------------------------------------------------------------------

function showFrame(f: Frame, animate: boolean): void {
  app.shown = f;
  const speed = app.status?.speed ?? '2x';
  renderer.setFrame(f, animate, Math.min(900, SPEED_MS[speed] * 0.85));
  $('day').textContent = `Day ${f.day}`;
  $('day-of').textContent = `of ${app.config?.maxDays ?? '?'}`;
  renderStandings(f);
  renderInspector();
  renderVictory(f);
  updateReplay();
}

function renderVictory(f: Frame): void {
  const el = $('victory');
  if (!f.over) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  const how = f.victory === 'last_standing' ? 'Last team standing'
    : f.victory === 'supremacy' ? `Supremacy: held two thirds of the cities for ${R.SUPREMACY_DAYS} days`
      : f.victory === 'day_limit' ? 'Most cities at the day limit' : 'No single winner';
  el.innerHTML = f.winner
    ? `${icon(f.winner)}<span style="color:${FACTIONS[f.winner].colorLight}">${esc(TEAM_NAME(f.winner))} wins</span><small>${esc(how)}, day ${f.day}. Start a new match from the side panel.</small>`
    : `The match is a draw<small>${esc(how)}, day ${f.day}.</small>`;
}

function renderStandings(f: Frame): void {
  const map = app.map!;
  const territory = computeTerritory(map, f.cities);
  const rows = TEAM_IDS.map((t) => {
    const team = f.teams[t];
    const fa = FACTIONS[t];
    const cities = f.cities.filter((c) => c.owner === t).length;
    const blobs = f.blobs.filter((b) => b.team === t).length;
    const land = territory.filter((ci) => ci !== null && f.cities[ci].owner === t).length;
    const tags = [
      team.eliminated ? '<span class="tag">eliminated</span>' : '',
      team.remnant ? '<span class="tag">wanderers</span>' : '',
      ...team.allies.map((a) => `<span class="tag" title="Allied">allied with ${esc(cap(a))}</span>`),
    ].join('');
    return `<li class="${team.eliminated ? 'out' : ''}" data-team="${t}" title="Show ${esc(fa.name)}'s fog of war">
      ${icon(t)}
      <span class="name">${esc(fa.name)} <span class="epithet">${esc(fa.epithet)}</span>${tags}</span>
      <span class="nums">${cities} cities · ${blobs} blobs</span>
      <span class="detail">${team.coins} coins · ${land} tiles · ${team.techs.length} techs · ${team.stats.kills} kills</span>
    </li>`;
  }).join('');
  $('standings').innerHTML = rows;
  const sup = f.supremacy;
  const need = Math.ceil(f.cities.length * R.SUPREMACY_SHARE);
  $('supremacy').textContent = sup.team
    ? `${TEAM_NAME(sup.team)} holds ${need}+ of ${f.cities.length} cities: supremacy day ${sup.days} of ${R.SUPREMACY_DAYS}.`
    : `Supremacy needs ${need} of ${f.cities.length} cities for ${R.SUPREMACY_DAYS} days in a row.`;
  if (f.boss.alive && !f.boss.risen && f.day >= f.boss.risesOn - R.BOSS_WARNING_DAYS) {
    $('supremacy').textContent += ` ${f.boss.name} rises on day ${f.boss.risesOn}.`;
  }
}

function renderChronicle(): void {
  const items = app.log.slice(-250).reverse().map((l) => {
    const color = l.team ? FACTIONS[l.team].color : 'var(--muted)';
    return `<li class="${l.important ? 'important' : ''}"><span class="d">${l.day}</span><span class="t" style="--c:${color}">${esc(l.text)}</span></li>`;
  }).join('');
  // column-reverse keeps the newest line in view at the bottom.
  $('chronicle').innerHTML = items;
}

// ---------------------------------------------------------------------------
// Inspector
// ---------------------------------------------------------------------------

function hpBar(frac: number, color: string): string {
  return `<div class="hpbar"><span style="width:${Math.max(0, Math.min(100, frac * 100))}%;background:${color}"></span></div>`;
}

function blobStats(b: Blob, f: Frame): { attack: number; range: number; armour: number } {
  const rankAtk = R.RANKS[b.rank].attackBonus;
  const attack = b.vessel === 'gunboat' ? R.GUNBOAT.attack + rankAtk : b.vessel === 'raft' ? R.RAFT_ATTACK + rankAtk : R.BLOB_ATTACK + rankAtk + (b.gear ? R.GEAR[b.gear].attack : 0);
  const range = b.vessel === 'gunboat' ? R.GUNBOAT.range : b.vessel ? 1 : b.gear ? R.GEAR[b.gear].range : 1;
  let armour = 0;
  if (!b.vessel) {
    if (b.gear === 'shield') armour += 1;
    armour += R.TERRAIN_COVER[app.map!.tiles[idx(app.map!, b.x, b.y)].terrain];
    const c = f.cities.find((o) => o.x === b.x && o.y === b.y);
    if (c && c.walls && c.owner === b.team) armour += R.WALLS_ARMOUR;
  }
  return { attack, range, armour };
}

function renderInspector(): void {
  const el = $('inspector');
  const f = app.shown;
  const sel = renderer.selection;
  if (!f || !sel) {
    el.innerHTML = '<p class="muted">Click a blob, city, building, loot pile or the boss to inspect it.</p>';
    return;
  }
  if (sel.kind === 'blob') {
    const b = f.blobs.find((o) => o.id === sel.id);
    if (!b) {
      el.innerHTML = '<p class="muted">This blob is no longer on the island.</p>';
      return;
    }
    const fa = FACTIONS[b.team];
    const st = blobStats(b, f);
    const d = f.decisions[b.id];
    const last = b.memory[b.memory.length - 1];
    const src = d ? (d.source === 'claude' ? 'Claude' : d.source === 'local' ? 'Local bot' : 'No answer: idled') : null;
    el.innerHTML = `
      <h3>${icon(b.team)}${esc(b.name)} <span class="muted mono">${esc(b.id)}</span></h3>
      <div>${esc(fa.name)} · ${esc(cap(b.rank))}${b.origin !== b.team ? ` · born ${esc(cap(b.origin))}` : ''}</div>
      ${hpBar(b.hp / b.maxHp, fa.color)}
      <dl>
        <dt>HP</dt><dd>${b.hp} / ${b.maxHp}</dd>
        <dt>Position</dt><dd>(${b.x}, ${b.y})${b.vessel ? ` on a ${b.vessel}` : ''}</dd>
        <dt>Attack</dt><dd>${st.attack} at range ${st.range}</dd>
        <dt>Armour</dt><dd>${st.armour}</dd>
        <dt>Gear</dt><dd>${b.gear ? esc(R.GEAR[b.gear].label) : 'none'}</dd>
        <dt>Kills</dt><dd>${b.kills}</dd>
      </dl>
      ${d ? `<div><strong>Day ${f.day} action</strong> <span class="source">${esc(src)}</span></div>
      <div>${esc(describeAction(d.action))}${last && last.day === f.day ? ` → ${last.ok ? '' : '<span style="color:var(--danger)">failed: </span>'}${esc(last.note)}` : ''}</div>
      ${d.thought ? `<p class="thought">${esc(d.thought)}</p>` : ''}
      ${d.say ? `<p class="said">Said: “${esc(d.say)}”</p>` : ''}
      ${d.error ? `<p class="muted">Error: ${esc(d.error)}</p>` : ''}
      ${d.inputTokens ? `<p class="muted mono">${d.inputTokens} in · ${d.outputTokens} out tokens</p>` : ''}` : '<p class="muted">No decision recorded for this day yet.</p>'}
      ${b.memory.length ? `<div><strong>Recent days</strong></div><ol class="memory">${[...b.memory].reverse().map((m) => `<li class="${m.ok ? '' : 'fail'}">Day ${m.day}: ${esc(describeAction(m.action))}: ${esc(m.note)}</li>`).join('')}</ol>` : ''}
      <p><button class="btn small" type="button" id="btn-follow">See what ${esc(b.name)} sees</button></p>`;
    document.getElementById('btn-follow')?.addEventListener('click', () => {
      renderer.view = { kind: 'blob', id: b.id };
      ($('view-select') as HTMLSelectElement).value = 'blob';
    });
    return;
  }
  if (sel.kind === 'city') {
    const c = f.cities.find((o) => o.id === sel.id)!;
    const owner = c.owner ? FACTIONS[c.owner] : null;
    const next = R.LEVEL_POP[c.level + 1];
    const income = c.owner ? R.cityIncome(c.level, c.isCapital, c.owner === c.founder) : 0;
    el.innerHTML = `
      <h3>${esc(c.name)} <span class="muted mono">${esc(c.id)}</span></h3>
      <div>${c.isCapital ? `Capital of ${esc(FACTIONS[c.founder!].name)}` : c.island ? 'Island village' : 'Village'}</div>
      <dl>
        <dt>Owner</dt><dd>${owner ? `<span style="color:${owner.colorLight}">${esc(owner.name)}</span>` : 'Neutral'}</dd>
        <dt>Level</dt><dd>${c.level}${next !== undefined ? ` (pop ${c.pop}, next level at ${next})` : ` (pop ${c.pop}, max level)`}</dd>
        <dt>Income</dt><dd>${owner ? `${income} coins a day` : 'none while neutral'}</dd>
        <dt>Territory</dt><dd>radius ${c.owner ? R.territoryRadius(c.level) : 1}</dd>
        <dt>Walls</dt><dd>${c.walls ? `yes (+${R.WALLS_ARMOUR} armour)` : 'no'}</dd>
        <dt>Warden</dt><dd>${c.warden ? `${c.warden.hp} / ${c.warden.maxHp} HP` : 'none'}</dd>
        <dt>Position</dt><dd>(${c.x}, ${c.y})</dd>
      </dl>`;
    return;
  }
  if (sel.kind === 'boss') {
    const bo = f.boss;
    el.innerHTML = `<h3>${esc(bo.name)}</h3><div>The marsh boss</div>${hpBar(bo.hp / bo.maxHp, '#e06a4f')}
      <dl><dt>HP</dt><dd>${bo.hp} / ${bo.maxHp} (never heals)</dd><dt>Armour</dt><dd>${R.BOSS_ARMOUR}</dd><dt>Slam</dt><dd>${R.BOSS_SLAM} to every adjacent blob</dd><dt>Bounty</dt><dd>${R.BOSS_BOUNTY} coins</dd><dt>Position</dt><dd>(${bo.x}, ${bo.y})</dd></dl>`;
    return;
  }
  if (sel.kind === 'loot') {
    const l = f.loot.find((o) => o.id === sel.id);
    el.innerHTML = l ? `<h3>Loot pile</h3><p>${l.coins} coins at (${l.x}, ${l.y}). The first blob to step here takes them.</p>` : '<p class="muted">Someone already took it.</p>';
    return;
  }
  if (sel.kind === 'dev') {
    const d = f.devs.find((o) => o.x === sel.x && o.y === sel.y);
    if (!d) {
      el.innerHTML = '<p class="muted">Nothing is built here any more.</p>';
      return;
    }
    const spec = R.DEVELOPMENTS[d.kind];
    el.innerHTML = `<h3>${esc(cap(d.kind))}</h3><p>${esc(spec.label)}.</p><dl><dt>Built by</dt><dd>${esc(FACTIONS[d.builder].name)}, day ${d.builtOn}</dd>${d.kind === 'trap' ? `<dt>State</dt><dd>${d.armed ? 'armed (hidden from enemies)' : `sprung, pit closes on day ${d.pitUntil}`}</dd>` : ''}<dt>Position</dt><dd>(${d.x}, ${d.y})</dd></dl>`;
    return;
  }
  const t = app.map!.tiles[idx(app.map!, sel.x, sel.y)];
  const ci = computeTerritory(app.map!, f.cities)[idx(app.map!, sel.x, sel.y)];
  const owner = ci === null ? null : f.cities[ci];
  el.innerHTML = `<h3>${esc(cap(t.terrain))}</h3><dl><dt>Position</dt><dd>(${sel.x}, ${sel.y})</dd><dt>Move cost</dt><dd>${R.TERRAIN_COST[t.terrain] ?? 'impassable on foot'}</dd><dt>Cover</dt><dd>${R.TERRAIN_COVER[t.terrain] ? `+${R.TERRAIN_COVER[t.terrain]} armour` : 'none'}</dd><dt>Resource</dt><dd>${t.resource ?? 'none'}</dd><dt>Territory</dt><dd>${owner ? `${esc(owner.name)}${owner.owner ? ` (${esc(FACTIONS[owner.owner].name)})` : ' (neutral)'}` : 'none'}</dd></dl>`;
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

function updateReplay(): void {
  const range = $('replay-range') as HTMLInputElement;
  const max = app.latest?.day ?? 0;
  range.max = String(max);
  if (app.live) range.value = String(max);
  $('replay-day').textContent = app.live ? `Day ${max} (live)` : `Day ${app.shown?.day ?? 0} of ${max}`;
  ($('btn-live') as HTMLButtonElement).disabled = app.live;
}

async function seek(day: number): Promise<void> {
  let f = app.frames.get(day);
  if (!f) {
    const res = await fetch(`/api/frame?day=${day}`);
    if (!res.ok) return;
    f = (await res.json()) as Frame;
    app.frames.set(day, f);
  }
  app.live = day >= (app.latest?.day ?? 0);
  showFrame(f, true);
}

// ---------------------------------------------------------------------------
// Banner, settings, controls
// ---------------------------------------------------------------------------

function showBanner(text: string): void {
  const b = $('banner');
  b.textContent = text;
  b.hidden = false;
}

function hideBannerIf(prefix: string): void {
  const b = $('banner');
  if (b.textContent?.startsWith(prefix)) b.hidden = true;
}

function openSettings(): void {
  const s = app.status;
  if (!s) return;
  const dlg = $('settings') as HTMLDialogElement;
  const form = $('settings-form') as HTMLFormElement;
  const modelSel = $('model-select') as HTMLSelectElement;
  modelSel.innerHTML = app.models.map((m) => `<option value="${esc(m.id)}">${esc(m.label)} ($${m.inputPrice} / $${m.outputPrice} per million tokens)</option>`).join('');
  (form.elements.namedItem('brain') as RadioNodeList).value = s.settings.brain;
  (form.elements.namedItem('apiKey') as HTMLInputElement).value = '';
  modelSel.value = s.settings.model;
  (form.elements.namedItem('effort') as HTMLSelectElement).value = s.settings.effort;
  (form.elements.namedItem('concurrency') as HTMLInputElement).value = String(s.settings.concurrency);
  (form.elements.namedItem('budgetUsd') as HTMLInputElement).value = String(s.settings.budgetUsd);
  (form.elements.namedItem('maxDays') as HTMLInputElement).value = String(s.settings.maxDays);
  (form.elements.namedItem('mapSize') as HTMLSelectElement).value = String(s.settings.mapSize);
  $('key-hint').textContent = s.hasKey
    ? `A key is set (from ${s.keySource === 'env' ? 'the ANTHROPIC_API_KEY environment variable or .env' : 'this dialog'}). Leave blank to keep it.`
    : 'No key set. Paste one to let Claude play. It is kept in the server\'s memory only.';
  const updateCost = () => {
    const m = app.models.find((x) => x.id === modelSel.value);
    const blobs = app.latest?.blobs.length ?? 12;
    if (!m) return;
    // Rough: ~3,500 input and ~400 output tokens per blob per day at low effort.
    const perDay = blobs * ((3500 * m.inputPrice + 400 * m.outputPrice) / 1_000_000);
    $('cost-hint').textContent = `Rough cost with ${blobs} blobs: about $${perDay.toFixed(3)} a day, $${(perDay * 100).toFixed(2)} per 100 days. Higher effort costs more. The match pauses when the budget is reached.`;
  };
  modelSel.onchange = updateCost;
  updateCost();
  dlg.showModal();
  dlg.onclose = async () => {
    if (dlg.returnValue !== 'save') return;
    const data = new FormData(form);
    const body: Record<string, unknown> = {
      brain: data.get('brain'),
      model: data.get('model'),
      effort: data.get('effort'),
      concurrency: Number(data.get('concurrency')),
      budgetUsd: Number(data.get('budgetUsd')),
      maxDays: Number(data.get('maxDays')),
      mapSize: Number(data.get('mapSize')),
    };
    const key = String(data.get('apiKey') ?? '').trim();
    if (key) body.apiKey = key;
    await post('/api/settings', body);
  };
}

function bindControls(): void {
  $('btn-play').addEventListener('click', () => {
    void post(app.status?.status === 'running' ? '/api/pause' : '/api/play');
  });
  $('btn-step').addEventListener('click', () => void post('/api/step'));
  for (const b of document.querySelectorAll<HTMLButtonElement>('#speed button')) {
    b.addEventListener('click', () => void post('/api/speed', { speed: b.dataset.speed }));
  }
  $('seed-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const v = ($('seed-input') as HTMLInputElement).value;
    void post('/api/new', { seed: v === '' ? undefined : Number(v) });
  });
  $('btn-settings').addEventListener('click', openSettings);
  $('brain-chip').addEventListener('click', openSettings);

  // Board interaction: a click selects, a drag moves the camera.
  let downAt: [number, number] | null = null;
  $('board').addEventListener('pointerdown', (e) => {
    downAt = [e.clientX, e.clientY];
    $('camera-help').hidden = true;
  });
  $('board').addEventListener('click', (e) => {
    const me = e as MouseEvent;
    if (downAt && Math.hypot(me.clientX - downAt[0], me.clientY - downAt[1]) > 5) return;
    const sel: Selection = renderer.pick(me.clientX, me.clientY);
    renderer.selection = sel;
    if (sel?.kind === 'blob' && renderer.view.kind === 'blob') renderer.view = { kind: 'blob', id: sel.id };
    renderInspector();
  });
  $('view-select').addEventListener('change', (e) => {
    const v = (e.target as HTMLSelectElement).value;
    if (v === 'god') renderer.view = { kind: 'god' };
    else if (v === 'blob') {
      const sel = renderer.selection;
      if (sel?.kind === 'blob') renderer.view = { kind: 'blob', id: sel.id };
      else {
        renderer.view = { kind: 'god' };
        (e.target as HTMLSelectElement).value = 'god';
        showBanner('Select a blob on the map first, then choose this view.');
        setTimeout(() => hideBannerIf('Select a blob'), 3000);
      }
    } else renderer.view = { kind: 'team', team: v as TeamId };
  });
  $('standings').addEventListener('click', (e) => {
    const li = (e.target as HTMLElement).closest('li');
    const t = li?.dataset.team as TeamId | undefined;
    if (!t) return;
    renderer.view = { kind: 'team', team: t };
    ($('view-select') as HTMLSelectElement).value = t;
  });
  $('board').addEventListener('dblclick', () => renderer.focusSelected());
  $('btn-follow-cam').addEventListener('click', () => renderer.focusSelected());
  $('btn-reset-cam').addEventListener('click', () => renderer.resetCamera());
  $('opt-territory').addEventListener('change', (e) => { renderer.showTerritory = (e.target as HTMLInputElement).checked; });
  $('opt-speech').addEventListener('change', (e) => { renderer.showSpeech = (e.target as HTMLInputElement).checked; });
  $('opt-coords').addEventListener('change', (e) => { renderer.showCoords = (e.target as HTMLInputElement).checked; });
  $('replay-range').addEventListener('input', (e) => void seek(Number((e.target as HTMLInputElement).value)));
  $('btn-live').addEventListener('click', () => {
    app.live = true;
    if (app.latest) showFrame(app.latest, false);
  });

  // Rulebook and stats open as sheets over the map.
  const openSheet = (id: string) => {
    $(id).hidden = false;
    if (id === 'sheet-stats') {
      app.statsOpen = true;
      void renderStats($('stats'), app.latest);
    }
    ($(id).querySelector('[data-close]') as HTMLButtonElement).focus();
  };
  const closeSheets = () => {
    for (const id of ['sheet-rulebook', 'sheet-stats']) $(id).hidden = true;
    app.statsOpen = false;
  };
  $('btn-rulebook').addEventListener('click', () => openSheet('sheet-rulebook'));
  $('btn-stats').addEventListener('click', () => openSheet('sheet-stats'));
  for (const sheet of document.querySelectorAll<HTMLElement>('.sheet')) {
    sheet.addEventListener('click', (e) => {
      if (e.target === sheet || (e.target as HTMLElement).closest('[data-close]')) closeSheets();
    });
  }
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeSheets();
  });

  // Keyboard: space plays or pauses, period steps.
  document.addEventListener('keydown', (e) => {
    const target = e.target as HTMLElement;
    if (target.closest('input, select, textarea, dialog') || !$('sheet-rulebook').hidden || !$('sheet-stats').hidden) return;
    if (e.key === ' ') {
      e.preventDefault();
      $('btn-play').click();
    } else if (e.key === '.') $('btn-step').click();
  });
}

$('rulebook').innerHTML = renderRulebook();
bindControls();
connect();
