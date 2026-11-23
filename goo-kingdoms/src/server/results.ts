/**
 * Finished matches are appended to data/results.json so the Stats tab can
 * compare the four factions across many matches.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const FILE = path.join(here, '..', '..', 'data', 'results.json');

export interface MatchRecord {
  id: string;
  seed: number;
  finishedAt: string;
  days: number;
  maxDays: number;
  winner: string | null;
  reason: string | null;
  brain: string;
  model: string | null;
  spendUsd: number;
  teams: Record<string, Record<string, number | boolean>>;
}

export function loadResults(): MatchRecord[] {
  try {
    return JSON.parse(fs.readFileSync(FILE, 'utf8')) as MatchRecord[];
  } catch {
    return [];
  }
}

export function recordResult(record: MatchRecord): void {
  const all = loadResults();
  if (all.some((r) => r.id === record.id)) return;
  all.push(record);
  fs.mkdirSync(path.dirname(FILE), { recursive: true });
  fs.writeFileSync(FILE, JSON.stringify(all, null, 1));
}
