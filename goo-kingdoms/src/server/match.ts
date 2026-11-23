/**
 * Runs one match at a time: asks the selected brain for every blob's action,
 * advances the simulation, keeps every frame for replay, and tells listeners
 * (the browser) what happened.
 */
import type { Decision, Frame, GameState, TeamId } from '../shared/types.ts';
import { TEAM_IDS } from '../shared/types.ts';
import { createGame, makeFrame, stepDay, territoryOf } from '../sim/engine.ts';
import { decideAllLocal } from '../agents/localbot.ts';
import { ClaudeBrain, DEFAULT_MODEL, FatalBrainError, MODELS } from '../agents/claude.ts';
import type { Effort } from '../agents/claude.ts';
import { DEFAULT_MAX_DAYS, MAX_DAYS_LIMIT } from '../shared/rules.ts';
import { recordResult } from './results.ts';

export type Brain = 'claude' | 'local';
export type Speed = '1x' | '2x' | '5x' | 'max';
export type Status = 'paused' | 'running' | 'over';

const SPEED_DELAY: Record<Speed, number> = { '1x': 1200, '2x': 600, '5x': 240, max: 0 };

export interface Settings {
  brain: Brain;
  model: string;
  effort: Effort;
  concurrency: number;
  budgetUsd: number;
  maxDays: number;
}

export interface PublicStatus {
  status: Status;
  speed: Speed;
  thinking: { done: number; total: number } | null;
  settings: Settings;
  hasKey: boolean;
  keySource: 'env' | 'settings' | null;
  spend: { usd: number; inputTokens: number; outputTokens: number; calls: number };
  error: string | null;
  matchId: string;
  seed: number;
  days: number;
}

export type Listener = (event: string, data: unknown) => void;

export class MatchController {
  state!: GameState;
  frames: Frame[] = [];
  status: Status = 'paused';
  speed: Speed = '2x';
  thinking: { done: number; total: number } | null = null;
  error: string | null = null;
  matchId = '';
  spend = { usd: 0, inputTokens: 0, outputTokens: 0, calls: 0 };
  settings: Settings;
  private apiKey: string | null;
  private keySource: 'env' | 'settings' | null;
  private listeners = new Set<Listener>();
  private loopToken = 0;
  private stepping = false;

  constructor() {
    const envKey = process.env.ANTHROPIC_API_KEY?.trim() || null;
    this.apiKey = envKey;
    this.keySource = envKey ? 'env' : null;
    const envModel = process.env.CLAUDE_MODEL?.trim();
    this.settings = {
      brain: envKey ? 'claude' : 'local',
      model: envModel && MODELS.some((m) => m.id === envModel) ? envModel : DEFAULT_MODEL,
      effort: 'low',
      concurrency: 8,
      budgetUsd: Number(process.env.BUDGET_USD ?? 5),
      maxDays: DEFAULT_MAX_DAYS,
    };
    this.newMatch(Math.floor(Math.random() * 100000));
  }

  on(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(event: string, data: unknown): void {
    for (const l of this.listeners) l(event, data);
  }

  publicStatus(): PublicStatus {
    return {
      status: this.status,
      speed: this.speed,
      thinking: this.thinking,
      settings: this.settings,
      hasKey: !!this.apiKey,
      keySource: this.keySource,
      spend: this.spend,
      error: this.error,
      matchId: this.matchId,
      seed: this.state.config.seed,
      days: this.frames.length,
    };
  }

  initPayload() {
    return {
      map: this.state.map,
      config: this.state.config,
      frame: this.frames[this.frames.length - 1],
      status: this.publicStatus(),
      models: MODELS,
      log: this.state.log.slice(-300),
    };
  }

  newMatch(seed: number, maxDays?: number): void {
    this.loopToken++;
    const days = Math.max(10, Math.min(MAX_DAYS_LIMIT, Math.floor(maxDays ?? this.settings.maxDays)));
    this.settings.maxDays = days;
    this.state = createGame({ seed: Math.abs(Math.floor(seed)) % 1_000_000, maxDays: days });
    this.frames = [makeFrame(this.state, [], [], {})];
    this.status = 'paused';
    this.thinking = null;
    this.error = null;
    this.spend = { usd: 0, inputTokens: 0, outputTokens: 0, calls: 0 };
    this.matchId = `${Date.now().toString(36)}-${this.state.config.seed}`;
    this.emit('init', this.initPayload());
  }

  setKey(key: string | null): void {
    const k = key?.trim() || null;
    if (k) {
      this.apiKey = k;
      this.keySource = 'settings';
    } else if (this.keySource === 'settings') {
      this.apiKey = process.env.ANTHROPIC_API_KEY?.trim() || null;
      this.keySource = this.apiKey ? 'env' : null;
    }
    this.error = null;
    this.pushStatus();
  }

  updateSettings(patch: Partial<Settings>): void {
    if (patch.brain === 'claude' || patch.brain === 'local') this.settings.brain = patch.brain;
    if (patch.model && MODELS.some((m) => m.id === patch.model)) this.settings.model = patch.model;
    if (patch.effort === 'low' || patch.effort === 'medium' || patch.effort === 'high') this.settings.effort = patch.effort;
    if (patch.concurrency !== undefined) this.settings.concurrency = Math.max(1, Math.min(32, Math.floor(Number(patch.concurrency) || 8)));
    if (patch.budgetUsd !== undefined) this.settings.budgetUsd = Math.max(0, Number(patch.budgetUsd) || 0);
    if (patch.maxDays !== undefined) this.settings.maxDays = Math.max(10, Math.min(MAX_DAYS_LIMIT, Math.floor(Number(patch.maxDays) || DEFAULT_MAX_DAYS)));
    this.error = null;
    this.pushStatus();
  }

  pushStatus(): void {
    this.emit('status', this.publicStatus());
  }

  setSpeed(speed: Speed): void {
    if (speed in SPEED_DELAY) this.speed = speed;
    this.pushStatus();
  }

  play(): void {
    if (this.status === 'over') return;
    if (!this.canRun()) return;
    this.status = 'running';
    this.error = null;
    this.pushStatus();
    void this.loop(++this.loopToken);
  }

  pause(): void {
    if (this.status === 'running') this.status = 'paused';
    this.loopToken++;
    this.pushStatus();
  }

  async step(): Promise<void> {
    if (this.status !== 'paused' || this.stepping || !this.canRun()) return;
    await this.advance(this.loopToken);
  }

  private canRun(): boolean {
    if (this.settings.brain === 'claude' && !this.apiKey) {
      this.error = 'Claude mode needs an Anthropic API key. Add one in Settings, or switch the brain to the local bot.';
      this.pushStatus();
      return false;
    }
    return true;
  }

  private async loop(token: number): Promise<void> {
    while (this.loopToken === token && this.status === 'running') {
      const started = Date.now();
      const ok = await this.advance(token);
      if (!ok || this.loopToken !== token) return;
      if (this.state.over) return;
      const wait = SPEED_DELAY[this.speed] - (Date.now() - started);
      await new Promise((r) => (wait > 0 ? setTimeout(r, wait) : setImmediate(r)));
    }
  }

  /** Decide and resolve one day. Returns false if the match had to stop. */
  private async advance(_token: number): Promise<boolean> {
    if (this.state.over) return false;
    this.stepping = true;
    const state = this.state;
    try {
      const blobs = [...state.blobs];
      let decisions: Record<string, Decision>;
      if (this.settings.brain === 'claude') {
        if (this.settings.budgetUsd > 0 && this.spend.usd >= this.settings.budgetUsd) {
          this.status = 'paused';
          this.error = `Spending reached the $${this.settings.budgetUsd.toFixed(2)} budget for this match. Raise it in Settings to continue.`;
          this.pushStatus();
          return false;
        }
        const brain = new ClaudeBrain({
          apiKey: this.apiKey!,
          model: this.settings.model,
          effort: this.settings.effort,
          concurrency: this.settings.concurrency,
        });
        this.thinking = { done: 0, total: blobs.length };
        this.pushStatus();
        let lastPush = 0;
        try {
          const res = await brain.decideAll(state, blobs, (done, total) => {
            this.thinking = { done, total };
            if (Date.now() - lastPush > 250 || done === total) {
              lastPush = Date.now();
              this.pushStatus();
            }
          });
          decisions = res.decisions;
          this.spend.usd += res.cost.usd;
          this.spend.inputTokens += res.cost.inputTokens;
          this.spend.outputTokens += res.cost.outputTokens;
          this.spend.calls += blobs.length;
        } catch (err) {
          if (state !== this.state) return false;
          this.thinking = null;
          this.status = 'paused';
          this.error = err instanceof FatalBrainError ? err.message : `Claude request failed: ${err instanceof Error ? err.message : String(err)}`;
          this.pushStatus();
          return false;
        }
        this.thinking = null;
        // A new match may have started while we waited for Claude.
        if (state !== this.state) return false;
      } else {
        decisions = decideAllLocal(state, blobs);
      }
      const frame = stepDay(this.state, decisions);
      this.frames.push(frame);
      this.emit('frame', frame);
      if (this.state.over) {
        this.status = 'over';
        recordResult(this.summary());
      }
      this.pushStatus();
      return true;
    } finally {
      this.stepping = false;
    }
  }

  summary() {
    const territory = territoryOf(this.state);
    const teams = {} as Record<TeamId, Record<string, number | boolean>>;
    for (const id of TEAM_IDS) {
      const t = this.state.teams[id];
      teams[id] = {
        ...t.stats,
        cities: this.state.cities.filter((c) => c.owner === id).length,
        blobs: this.state.blobs.filter((b) => b.team === id).length,
        territory: territory.filter((ci) => ci !== null && this.state.cities[ci].owner === id).length,
        coins: t.coins,
        techs: t.techs.length,
        eliminated: t.eliminated,
      };
    }
    return {
      id: this.matchId,
      seed: this.state.config.seed,
      finishedAt: new Date().toISOString(),
      days: this.state.day,
      maxDays: this.state.config.maxDays,
      winner: this.state.winner,
      reason: this.state.victory,
      brain: this.settings.brain,
      model: this.settings.brain === 'claude' ? this.settings.model : null,
      spendUsd: Math.round(this.spend.usd * 10000) / 10000,
      teams,
    };
  }

  replay() {
    return { map: this.state.map, config: this.state.config, frames: this.frames };
  }
}
