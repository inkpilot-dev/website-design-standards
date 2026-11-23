/**
 * Run matches without the UI, using the local bot, and print a summary.
 * Usage: npm run headless -- [matches] [maxDays] [firstSeed]
 */
import { createGame, stepDay, citiesOf, TEAM_LABEL } from '../sim/engine.ts';
import { decideAllLocal } from '../agents/localbot.ts';
import { TEAM_IDS } from '../shared/types.ts';

const matches = Number(process.argv[2] ?? 5);
const maxDays = Number(process.argv[3] ?? 300);
const firstSeed = Number(process.argv[4] ?? 1);
const mapSize = Number(process.argv[5] ?? 48);

for (let m = 0; m < matches; m++) {
  const seed = firstSeed + m;
  const state = createGame({ seed, maxDays, mapSize });
  const started = Date.now();
  while (!state.over) stepDay(state, decideAllLocal(state, state.blobs));
  const parts = TEAM_IDS.map((t) => {
    const s = state.teams[t].stats;
    return `${TEAM_LABEL[t]} ${citiesOf(state, t).length}c ${state.blobs.filter((b) => b.team === t).length}b ${s.kills}k/${s.deaths}d ${Math.round((100 * s.validActions) / Math.max(1, s.decisions))}%valid ${state.teams[t].techs.length}tech`;
  });
  console.log(`seed ${seed}: day ${state.day} winner ${state.winner ?? 'none'} (${state.victory}) in ${Date.now() - started}ms | ${parts.join(' | ')}`);
}
