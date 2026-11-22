import { useEffect, useState } from "react";
import { Check, Gauge, Hourglass, Rocket, TriangleAlert, X } from "lucide-react";
import type { AppError, EngineEvent, ProjectInfo } from "../lib/contract";
import { api, isMac, isWindows, on, subscribe, toAppError } from "../lib/ipc";
import { type Bench, type Preset, type TrainingState, verdictReady } from "../lib/training";
import type { MemoryNote } from "../lib/memory";
import { Badge, Button, Card, ChoiceCard, Drawer, ErrorNote, fmtEta, Icon, LiftGauge, Note, PageHeader, ProgressBar, SectionTitle, Sparkline, TextInput } from "../components/ui";

const STAGES: { id: string; label: string; what: string }[] = [
  { id: "verify_data", label: "Check data", what: "Confirms that no final-exam question leaked into the training set." },
  { id: "baseline", label: "Test the original", what: "The untouched model sits the hidden final exam, to set the score to beat." },
  { id: "sft", label: "Learn", what: "Your model studies your examples and is checked on practice questions as it goes." },
  { id: "stage2", label: "Practice", what: "It answers practice questions and keeps learning from the ones it gets right." },
  { id: "gate", label: "Final exam", what: "The trained model sits the same hidden exam as the original." },
  { id: "promote", label: "Save", what: "A new version is saved only if it clearly beat the original." },
];

interface Props {
  project: ProjectInfo;
  /** Set when this computer has little memory for training this project's model. */
  memory?: MemoryNote | null;
  /** The one training run the app knows about (it may belong to another project). */
  training: TrainingState;
  trainingName: string;
  onStart: (preset: Preset) => void | Promise<void>;
  onStop: () => void | Promise<void>;
  onDismissError: () => void;
  onDeploy: () => void | Promise<void>;
  onEditData: () => void;
}

/** A question about a document starts with a long excerpt; on a small card show the question itself. */
function cardQuestion(prompt: string): string {
  const cut = prompt.lastIndexOf('"""\n\n');
  return prompt.startsWith("Excerpt from ") && cut > 0 ? prompt.slice(cut + 5) : prompt;
}

export function TrainingArena({ project, memory, training, trainingName, onStart, onStop, onDismissError, onDeploy, onEditData }: Props) {
  const mine = training.projectId === project.id;
  const t = mine ? training : null;
  const running = !!t?.running;
  const elsewhere = training.running && !mine;
  const [stopping, setStopping] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [asking, setAsking] = useState(false);
  const [arena, setArena] = useState<Bench[]>([]);
  const [arenaError, setArenaError] = useState<AppError | null>(null);

  useEffect(() => {
    if (!training.running) setStopping(false);
  }, [training.running]);

  // "Ask both models yourself" results (a separate, short job).
  useEffect(
    () =>
      subscribe(
        on<EngineEvent>("benchmark://result", (e) => {
          if (e.type === "benchmark" && e.source === "arena" && (!e.project_id || e.project_id === project.id)) setArena((a) => [e, ...a].slice(0, 6));
        }),
      ),
    [project.id],
  );

  const stop = async () => {
    setStopping(true);
    await onStop();
  };

  const ask = async () => {
    if (asking) return;
    setAsking(true);
    setArenaError(null);
    try {
      await api.runBenchmarkArena(project.id, prompt || undefined, prompt ? 1 : 3);
    } catch (e) {
      setArenaError(toAppError(e));
    } finally {
      setAsking(false);
    }
  };

  const gate = t?.gate ?? null;
  const promoted = gate?.decision === "PROMOTE";
  const saving = promoted && !gate?.version && running;
  const verdict = t && verdictReady(t) ? gate : null;
  const showRun = running || t?.ended === "ok";
  const canStart = !training.running;
  const report = project.latest_report;
  const preset = t?.preset ?? "standard";
  // A verdict from this session already says everything the "current version" card would.
  const showCurrent = !!project.active_version && !running && !(verdict && promoted);

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow={project.name}
        title="Train and prove"
        lead="Your AI studies your examples, then sits a final exam it has never seen. It is only kept if it beats the original."
        action={running ? <Button variant="danger" onClick={stop} busy={stopping}>{stopping ? "Stopping…" : "Stop training"}</Button> : undefined}
      />

      {elsewhere && (
        <Note icon={Hourglass}>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <span>
              Training is running for {trainingName ? <span className="font-semibold">“{trainingName}”</span> : "another project"} ({Math.round(training.overall)}%). Only one project can train at a time.
            </span>
            <Button variant="danger" onClick={stop} busy={stopping}>{stopping ? "Stopping…" : "Stop that training"}</Button>
          </div>
        </Note>
      )}

      {t?.ended === "stopped" && <Note icon={X}>Training was stopped. Nothing was saved from that run{project.active_version ? `; version ${project.active_version} is still the one in use` : ""}.</Note>}
      {t?.error && (
        <div className="space-y-3">
          <ErrorNote error={t.error} title={t.error.code === "OUT_OF_MEMORY" ? "Training ran out of memory" : t.error.code === "BUSY" ? "Training could not start" : "Training did not finish"} onRetry={t.error.retryable && canStart ? () => onStart(preset) : undefined} />
          <div className="flex gap-3">
            {(t.error.code === "BAD_INPUT" || t.error.code === "LEAKAGE" || t.error.code === "NOT_FOUND") && <Button variant="secondary" onClick={onEditData}>Go to Data &amp; rules</Button>}
            <Button variant="ghost" onClick={onDismissError}>Dismiss</Button>
          </div>
        </div>
      )}

      {showRun && t && (
        <Card className="space-y-5">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <div className="min-w-0 text-[15px] text-muted">{running ? (stopping ? "Stopping…" : saving ? "Saving your new version…" : t.detail || "Starting…") : "Finished"}</div>
            <div className="text-[15px] font-medium tabular-nums">{Math.round(t.overall)}%<span className="ml-2 font-normal text-muted">{running ? (t.eta === null ? "" : `${fmtEta(t.eta)} left in this step`) : "done"}</span></div>
          </div>
          <ProgressBar value={t.overall} label="Overall training progress" />
          {running && <p className="text-sm leading-relaxed text-muted">You can keep using your computer, or leave it running.{isWindows || isMac ? " Inkport keeps it awake until training has finished (closing a laptop’s lid still puts it to sleep)." : " If it is set to go to sleep when idle, training pauses until it wakes up."}</p>}
          <ol className="flex flex-wrap gap-x-6 gap-y-2 text-sm">
            {STAGES.map((s) => {
              const st = t.stages[s.id];
              return (
                <li key={s.id} className={`inline-flex items-center gap-1.5 ${st === "done" ? "text-ink" : st === "active" ? "font-semibold text-ink" : st === "failed" ? "text-bad" : "text-faint"}`}>
                  {st === "done" ? <Icon icon={Check} size={14} className="text-good" /> : st === "failed" ? <Icon icon={X} size={14} /> : <span className={`h-1.5 w-1.5 rounded-full ${st === "active" && running ? "animate-pulse bg-ink" : "bg-line"}`} aria-hidden />}
                  {s.label}
                </li>
              );
            })}
          </ol>
        </Card>
      )}

      {saving && <Note icon={Hourglass}>The trained model beat the original on the hidden exam. Saving it as a new version; this takes a moment.</Note>}

      {verdict && (
        <Card className={`space-y-4 ${promoted ? "border-ink" : ""}`}>
          <div className="flex flex-wrap items-center gap-2">
            {promoted ? <Badge tone="good"><Icon icon={Check} size={13} /> Better: new version {verdict.version} saved</Badge> : verdict.decision === "INSUFFICIENT_DATA" ? <Badge tone="warn">Need more examples</Badge> : <Badge tone="warn">{project.active_version ? `Not better yet: version ${project.active_version} stays in use` : "Not better yet: nothing was saved"}</Badge>}
            {verdict.label && <Badge>{verdict.label === "VERIFIED" ? "Verified result" : "Preliminary result"}</Badge>}
          </div>
          <p className="max-w-[70ch] text-[17px] leading-[1.65]">{verdict.explanation}</p>
          <div className="flex flex-wrap gap-3 pt-1">
            {promoted && <Button onClick={onDeploy}>Use it in your apps</Button>}
            {!promoted && <Button variant="secondary" onClick={onEditData}>Add more examples</Button>}
          </div>
        </Card>
      )}

      {showRun && t && (
        <div className="grid gap-5 lg:grid-cols-5">
          <Card className="lg:col-span-2 lg:self-start">
            <LiftGauge label={t.candRate !== null ? "Final exam, hidden questions" : "Practice score so far"} before={t.baseRate} after={t.candRate ?? t.devPass} />
            <p className="mt-6 text-sm leading-relaxed text-muted">“Original” is the untouched model on the hidden exam. {t.candRate === null ? "“Yours” is your model on practice questions while it learns." : "“Yours” is your trained model on the same hidden exam."}</p>
          </Card>
          <Card className="space-y-4 lg:col-span-3">
            <h2 className="text-[17px] font-semibold tracking-[-0.01em]">Same question, both models</h2>
            {t.duels.length === 0 && <div className="text-[15px] leading-relaxed text-muted">Side-by-side answers appear here as training checks its progress.</div>}
            <div className="divide-y divide-line">
              {t.duels.slice(0, 3).map((d) => (
                <Duel key={d.case_id} d={d} />
              ))}
            </div>
          </Card>
        </div>
      )}

      {showCurrent && (
        <Card className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="good"><Icon icon={Check} size={13} /> Version {project.active_version} is in use</Badge>
            {report?.label && report.promoted_version === project.active_version && <Badge>{report.label === "VERIFIED" ? "Verified result" : "Preliminary result"}</Badge>}
          </div>
          {report?.explanation && report.promoted_version === project.active_version && <p className="max-w-[70ch] text-[16px] leading-[1.65]">{report.explanation}</p>}
          <div className="flex flex-wrap gap-3 pt-1">
            <Button onClick={onDeploy}>Use it in your apps</Button>
          </div>
        </Card>
      )}

      {!running && (
        <section className="space-y-5">
          {(project.active_version || t?.ended) && <SectionTitle title={project.active_version ? "Train a newer version" : "Train again"} lead={project.active_version ? `A new run only replaces version ${project.active_version} if it scores clearly higher on the hidden exam.` : undefined} />}
          {memory && <Note icon={TriangleAlert} tone={memory.level === "short" ? "warn" : "neutral"}>{memory.text}</Note>}
          <div className="grid gap-5 sm:grid-cols-2">
            <ChoiceCard icon={Rocket} title="Best results" onClick={() => onStart("standard")} disabled={!canStart}>
              Trains for as long as it keeps improving, up to about 8 hours. Choose this when the result matters.
            </ChoiceCard>
            <ChoiceCard icon={Gauge} title="Quick run" onClick={() => onStart("fast")} disabled={!canStart}>
              A shorter run, up to about 90 minutes. Good for a first look at whether your examples teach the skill.
            </ChoiceCard>
          </div>
        </section>
      )}

      {!running && !t?.ended && !project.active_version && (
        <section className="space-y-5 pt-4">
          <SectionTitle title="What happens" lead="Six steps run on this computer, in order. You can stop at any time and nothing is saved unless the last step passes." />
          <ol className="divide-y divide-line border-y border-line">
            {STAGES.map((s, i) => (
              <li key={s.id} className="flex gap-5 py-4 text-[15px]">
                <span className="w-5 flex-none font-mono text-[13px] leading-6 text-faint">{i + 1}</span>
                <span className="w-40 flex-none font-semibold">{s.label}</span>
                <span className="leading-relaxed text-muted">{s.what}</span>
              </li>
            ))}
          </ol>
        </section>
      )}

      {!!project.active_version && !training.running && (
        <section className="space-y-5 pt-4">
          <SectionTitle title="Ask both models yourself" lead="Type a new example, or leave the box empty to draw from your practice questions." />
          <div className="flex gap-2">
            <TextInput value={prompt} onChange={setPrompt} onEnter={ask} placeholder="e.g. a new line from your records" label="Question for both models" />
            <Button onClick={ask} busy={asking}>Compare</Button>
          </div>
          {asking && <div className="text-sm text-muted">Loading both models; the first comparison takes the longest.</div>}
          <ErrorNote error={arenaError} title="The comparison could not run" />
          {arena.length > 0 && (
            <Card className="divide-y divide-line py-1">
              {arena.map((d, i) => (
                <Duel key={d.case_id + i} d={d} />
              ))}
            </Card>
          )}
        </section>
      )}

      {t && (running || t.ended) && (
        <Drawer title="Advanced metrics" open={advanced} onToggle={() => setAdvanced(!advanced)}>
          <div className="grid gap-6 md:grid-cols-2">
            <div>
              <div className="mb-2 text-xs text-muted">Training loss</div>
              <Sparkline values={t.loss} />
            </div>
            <dl className="grid grid-cols-2 gap-1.5 font-mono text-xs">
              {Object.entries(t.metrics).map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="text-muted">{k}</dt>
                  <dd>{v === null ? "–" : Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(2) : Number(v.toFixed(4))}</dd>
                </div>
              ))}
            </dl>
          </div>
          <pre className="mt-5 max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-xl bg-canvas p-4 font-mono text-[11px] leading-relaxed">{t.logs.join("\n") || "Logs appear here."}</pre>
        </Drawer>
      )}
    </div>
  );
}

function Duel({ d }: { d: Bench }) {
  return (
    <div className="py-4 text-[15px]">
      <div className="mb-3 line-clamp-2 break-words font-medium leading-relaxed">{cardQuestion(d.prompt)}</div>
      <div className="grid gap-3 sm:grid-cols-2">
        {([["Original", d.baseline], ["Yours", d.trained]] as const).map(([name, side]) => (
          <div key={name} className={`min-w-0 rounded-xl px-3.5 py-3 ${side.pass ? "bg-good-soft" : side.pass === false ? "bg-bad-soft" : "bg-canvas"}`}>
            <div className={`flex items-center gap-1.5 text-[13px] font-semibold ${side.pass ? "text-good" : side.pass === false ? "text-bad" : "text-muted"}`}>
              {side.pass ? <Icon icon={Check} size={14} /> : side.pass === false ? <Icon icon={X} size={14} /> : null}
              {name}
            </div>
            <div className="mt-1.5 line-clamp-4 whitespace-pre-wrap break-words font-mono text-[13px] leading-relaxed">{side.output || "(no answer)"}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
