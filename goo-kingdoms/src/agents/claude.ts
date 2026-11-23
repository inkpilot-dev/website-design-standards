/**
 * Claude brain: every living blob is its own Claude call each day, made
 * concurrently with a cap on parallel requests. Each call gets the blob's
 * faction system prompt (cached) and its local observation, and must answer
 * with structured JSON that matches DECISION_SCHEMA.
 */
import Anthropic from '@anthropic-ai/sdk';
import type { Blob, Decision, GameState } from '../shared/types.ts';
import { observe } from '../sim/observe.ts';
import { DECISION_SCHEMA, idleDecision, parseDecision } from './actions.ts';
import { renderObservation } from './prompt.ts';
import { systemPrompt } from './factions.ts';

export interface ModelInfo {
  id: string;
  label: string;
  /** USD per million tokens, for the spend estimate shown in the UI. */
  inputPrice: number;
  outputPrice: number;
  /** Whether the model supports server-side refusal fallbacks. */
  fallbacks: boolean;
}

export const MODELS: ModelInfo[] = [
  { id: 'claude-haiku-5-5', label: 'Claude Haiku 5.5', inputPrice: 0.10, outputPrice: 0.50, fallbacks: false },
  { id: 'claude-sonnet-5-5', label: 'Claude Sonnet 5.5', inputPrice: 2, outputPrice: 10, fallbacks: true },
  { id: 'claude-opus-5-5', label: 'Claude Opus 5.5', inputPrice: 4, outputPrice: 20, fallbacks: true },
];
export const DEFAULT_MODEL = 'claude-haiku-5-5';

export type Effort = 'low' | 'medium' | 'high';

export interface ClaudeOptions {
  apiKey: string;
  model: string;
  effort: Effort;
  concurrency: number;
}

/** Thrown when continuing would only repeat the same failure (bad key, bad model). */
export class FatalBrainError extends Error {}

export interface CallCost {
  inputTokens: number;
  outputTokens: number;
  usd: number;
}

export class ClaudeBrain {
  readonly options: ClaudeOptions;
  private client: Anthropic;

  constructor(options: ClaudeOptions) {
    this.options = options;
    this.client = new Anthropic({ apiKey: options.apiKey, maxRetries: 2, timeout: 90_000 });
  }

  get model(): ModelInfo {
    return MODELS.find((m) => m.id === this.options.model) ?? MODELS[0];
  }

  /** Ask every blob for its action, at most `concurrency` requests at a time. */
  async decideAll(
    state: GameState,
    blobs: Blob[],
    onProgress?: (done: number, total: number) => void,
  ): Promise<{ decisions: Record<string, Decision>; cost: CallCost }> {
    const decisions: Record<string, Decision> = {};
    const cost: CallCost = { inputTokens: 0, outputTokens: 0, usd: 0 };
    let next = 0;
    let done = 0;
    let fatal: unknown = null;
    const worker = async () => {
      while (next < blobs.length && !fatal) {
        const b = blobs[next++];
        try {
          const { decision, usd } = await this.decide(state, b);
          decisions[b.id] = decision;
          cost.inputTokens += decision.inputTokens ?? 0;
          cost.outputTokens += decision.outputTokens ?? 0;
          cost.usd += usd;
        } catch (err) {
          if (err instanceof FatalBrainError) fatal = err;
          else decisions[b.id] = idleDecision('fallback', err instanceof Error ? err.message : String(err));
        }
        done++;
        onProgress?.(done, blobs.length);
      }
    };
    const workers = Array.from({ length: Math.max(1, Math.min(this.options.concurrency, blobs.length)) }, worker);
    await Promise.all(workers);
    if (fatal) throw fatal;
    return { decisions, cost };
  }

  private async decide(state: GameState, b: Blob): Promise<{ decision: Decision; usd: number }> {
    const obs = observe(state, b);
    const model = this.model;
    let response;
    try {
      response = await this.client.beta.messages.create({
        model: model.id,
        max_tokens: 8000,
        system: [{ type: 'text', text: systemPrompt(b.team), cache_control: { type: 'ephemeral' } }],
        messages: [{ role: 'user', content: renderObservation(obs) }],
        output_config: {
          effort: this.options.effort,
          format: { type: 'json_schema', schema: DECISION_SCHEMA as unknown as Record<string, unknown> },
        },
        // On a policy decline, Sonnet and Opus retry on their default fallback model.
        ...(model.fallbacks ? { betas: ['server-side-fallback-2026-07-01'], fallbacks: 'default' as const } : {}),
      });
    } catch (err) {
      if (err instanceof Anthropic.AuthenticationError) throw new FatalBrainError('The Anthropic API rejected the API key (401). Check the key in Settings.');
      if (err instanceof Anthropic.PermissionDeniedError) throw new FatalBrainError(`This API key cannot use ${model.label} (403).`);
      if (err instanceof Anthropic.NotFoundError) throw new FatalBrainError(`Model ${model.id} was not found (404).`);
      if (err instanceof Anthropic.RateLimitError) throw new Error('Rate limited (429); this blob idles today.');
      if (err instanceof Anthropic.APIError) throw new Error(`API error ${err.status ?? ''}: ${err.message}`);
      throw err;
    }

    const u = response.usage;
    const cacheWrite = u.cache_creation_input_tokens ?? 0;
    const cacheRead = u.cache_read_input_tokens ?? 0;
    const inputTokens = u.input_tokens + cacheWrite + cacheRead;
    const outputTokens = u.output_tokens;
    const usd = ((u.input_tokens + cacheWrite * 1.25 + cacheRead * 0.1) * model.inputPrice + outputTokens * model.outputPrice) / 1_000_000;

    if (response.stop_reason === 'refusal') {
      return { decision: { ...idleDecision('claude', 'Claude declined this request.'), inputTokens, outputTokens }, usd };
    }
    let text = '';
    for (const block of response.content) if (block.type === 'text') text += block.text;
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(text);
    } catch {
      return { decision: { ...idleDecision('claude', response.stop_reason === 'max_tokens' ? 'Response cut off at max_tokens.' : 'Response was not valid JSON.'), inputTokens, outputTokens }, usd };
    }
    const decision = parseDecision(parsed, 'claude');
    decision.inputTokens = inputTokens;
    decision.outputTokens = outputTokens;
    return { decision, usd };
  }
}
