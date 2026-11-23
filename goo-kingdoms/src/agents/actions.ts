/**
 * The JSON schema every agent answers with, and a strict parser that turns a
 * raw response into a typed Decision. Anything that does not fit becomes
 * `idle` and is flagged as not well formed.
 */
import type { Action, Decision, DevKind, Gear, TeamId, Tech } from '../shared/types.ts';
import { ACTION_TYPES, TEAM_IDS } from '../shared/types.ts';
import { DEVELOPMENTS, GEAR, SAY_MAX_CHARS, TECHS } from '../shared/rules.ts';

const nullable = (schema: Record<string, unknown>) => ({ anyOf: [schema, { type: 'null' }] });

const DEV_KINDS = Object.keys(DEVELOPMENTS) as DevKind[];
const GEARS = Object.keys(GEAR) as Gear[];
const TECH_IDS = Object.keys(TECHS) as Tech[];

/** Structured-output schema sent to Claude. Every field is required; unused ones are null. */
export const DECISION_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    thought: { type: 'string', description: 'One or two sentences of reasoning, shown to spectators.' },
    action: {
      type: 'object',
      additionalProperties: false,
      properties: {
        type: { type: 'string', enum: [...ACTION_TYPES] },
        x: nullable({ type: 'integer' }),
        y: nullable({ type: 'integer' }),
        target: nullable({ type: 'string' }),
        kind: nullable({ type: 'string', enum: DEV_KINDS }),
        item: nullable({ type: 'string', enum: GEARS }),
        tech: nullable({ type: 'string', enum: TECH_IDS }),
        team: nullable({ type: 'string', enum: [...TEAM_IDS] }),
        amount: nullable({ type: 'integer' }),
      },
      required: ['type', 'x', 'y', 'target', 'kind', 'item', 'tech', 'team', 'amount'],
    },
    say: { type: 'string', description: 'A short line spoken aloud, or an empty string.' },
  },
  required: ['thought', 'action', 'say'],
} as const;

const isInt = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v);
const isTeam = (v: unknown): v is TeamId => typeof v === 'string' && (TEAM_IDS as readonly string[]).includes(v);

/** Turn the `action` object of a response into a typed Action, or null if it does not fit. */
export function parseAction(raw: unknown): Action | null {
  if (!raw || typeof raw !== 'object') return null;
  const a = raw as Record<string, unknown>;
  switch (a.type) {
    case 'idle': case 'capture': case 'recruit': case 'fortify': case 'refit': case 'rest':
      return { type: a.type };
    case 'move':
      return isInt(a.x) && isInt(a.y) ? { type: 'move', x: a.x, y: a.y } : null;
    case 'attack':
      return typeof a.target === 'string' && a.target ? { type: 'attack', target: a.target } : null;
    case 'develop':
      return isInt(a.x) && isInt(a.y) && DEV_KINDS.includes(a.kind as DevKind)
        ? { type: 'develop', x: a.x, y: a.y, kind: a.kind as DevKind } : null;
    case 'craft':
      return GEARS.includes(a.item as Gear) ? { type: 'craft', item: a.item as Gear } : null;
    case 'research':
      return TECH_IDS.includes(a.tech as Tech) ? { type: 'research', tech: a.tech as Tech } : null;
    case 'ally_propose': case 'ally_accept': case 'ally_leave':
      return isTeam(a.team) ? { type: a.type, team: a.team } : null;
    case 'pay':
      return isTeam(a.team) && isInt(a.amount) && a.amount > 0 ? { type: 'pay', team: a.team, amount: a.amount } : null;
    default:
      return null;
  }
}

/** Parse a whole response (already JSON-decoded). */
export function parseDecision(raw: unknown, source: Decision['source']): Decision {
  const obj = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {};
  const action = parseAction(obj.action);
  const thought = typeof obj.thought === 'string' ? obj.thought.slice(0, 400) : '';
  const say = typeof obj.say === 'string' ? obj.say.slice(0, SAY_MAX_CHARS) : '';
  return {
    action: action ?? { type: 'idle' },
    say,
    thought,
    wellFormed: action !== null,
    source,
  };
}

export function idleDecision(source: Decision['source'], error?: string): Decision {
  return { action: { type: 'idle' }, say: '', thought: '', wellFormed: false, source, error };
}

/** One-line description of an action for logs and the inspector. */
export function describeAction(a: Action): string {
  switch (a.type) {
    case 'move': return `move to (${a.x},${a.y})`;
    case 'attack': return `attack ${a.target}`;
    case 'develop': return `build ${a.kind} at (${a.x},${a.y})`;
    case 'craft': return `craft ${a.item}`;
    case 'research': return `research ${a.tech}`;
    case 'ally_propose': return `propose alliance to ${a.team}`;
    case 'ally_accept': return `accept alliance with ${a.team}`;
    case 'ally_leave': return `leave alliance with ${a.team}`;
    case 'pay': return `pay ${a.amount} to ${a.team}`;
    default: return a.type;
  }
}
