import { useEffect, useRef, useState } from "react";
import type { AppError, EngineEvent, ProjectInfo } from "../lib/contract";
import { api, on, subscribe, toAppError } from "../lib/ipc";
import { Check, Gauge, Rocket, X } from "lucide-react";
import { Badge, Button, Card, ChoiceCard, Drawer, ErrorNote, fmtEta, Icon, LiftGauge, PageHeader, ProgressBar, SectionTitle, Sparkline, TextInput } from "../components/ui";

type Bench = Extract<EngineEvent, { type: "benchmark" }>;
type Gate = Extract<EngineEvent, { type: "gate" }>;

const STAGES: { id: string; label: string; what: string }[] = [
  { id: "verify_data", label: "Check data", what: "Confirms that no final-exam question leaked into the training set." },
  { id: "baseline", label: "Test the original", what: "The untouched model sits the hidden final exam, to set the score to beat." },
  { id: "sft", label: "Learn", what: "Your model studies your examples and is checked on practice questions as it goes." },
  { id: "stage2", label: "Practice", what: "It answers practice questions and keeps learning from the ones it gets right." },
  { id: "gate", label: "Final exam", what: "The trained model sits the same hidden exam as the original." },
  { id: "promote", label: "Save", what: "A new version is saved only if it clearly beat the original." },
];

export function TrainingArena({ project, onDeploy }: { project: ProjectInfo; onDeploy: () => void }) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [overall, setOverall] = useState(0);
  const [eta, setEta] = useState<number | null>(null);
  const [stageState, setStageState] = useState<Record<string, string>>({});
  const [stageDetail, setStageDetail] = useState("");
  const [baseRate, setBaseRate] = useState<number | null>(null);
  const [devPass, setDevPass] = useState<number | null>(null);
  const [candRate, setCandRate] = useState<number | null>(null);
  const [loss, setLoss] = useState<number[]>([]);
  const [metrics, setMetrics] = useState<Record<string, number | null>>({});
  const [duels, setDuels] = useState<Bench[]>([]);
  const [arena, setArena] = useState<Bench[]>([]);
  const [gate, setGate] = useState<Gate | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<AppError | null>(null);
  const [advanced, setAdvanced] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [asking, setAsking] = useState(false);
  const jobRef = useRef<string | null>(null);

  useEffect(() => {
    const mine = (e: EngineEvent) => !jobRef.current || e.job_id === jobRef.current || e.job_kind === "arena";
    const progress = on<EngineEvent>("training://progress", (e) => {
      if (!mine(e)) return;
      if (e.type === "progress") {
        setOverall(e.overall_pct);
        setEta(e.eta_s);
        setMetrics(e.metrics);
        const l = e.metrics.loss;
        if (typeof l === "number") setLoss((prev) => [...prev.slice(-119), l]);
        if (typeof e.metrics.dev_pass === "number") setDevPass(e.metrics.dev_pass);
      } else if (e.type === "stage") {
        const key = e.stage.startsWith("rft") || e.stage === "grpo" ? "stage2" : e.stage;
        setStageState((s) => ({ ...s, [key]: e.status === "completed" || e.status === "skipped" ? "done" : e.status === "failed" ? "failed" : s[key] === "done" ? "done" : "active" }));
        if (e.detail) setStageDetail(e.detail);
      } else if (e.type === "eval" && e.split === "test") {
        if (e.model === "base") setBaseRate(e.pass_rate);
        if (e.model === "candidate") setCandRate(e.pass_rate);
      } else if (e.type === "gate") {
        setGate(e);
      } else if (e.type === "error") {
        setError({ code: e.code, message: e.message, retryable: e.retryable });
      } else if (e.type === "done" && e.job_kind !== "arena") {
        setRunning(false);
        setOverall(100);
        setEta(0);
      }
    });
    const results = on<EngineEvent>("benchmark://result", (e) => {
      if (e.type !== "benchmark") return;
      if (e.source === "training_dev") setDuels((d) => [e, ...d.filter((x) => x.case_id !== e.case_id)].slice(0, 6));
      else setArena((a) => [e, ...a].slice(0, 6));
    });
    const logLines = on<EngineEvent>("training://log", (e) => {
      if (e.type === "log" && e.level !== "debug") setLogs((l) => [...l.slice(-200), `${e.level.toUpperCase()}  ${e.message}`]);
    });
    return subscribe(progress, results, logLines);
  }, []);

  const start = async (preset: "standard" | "fast" | "smoke") => {
    setError(null);
    setGate(null);
    setDuels([]);
    setLoss([]);
    setStageState({});
    setOverall(0);
    setCandRate(null);
    setDevPass(null);
    setRunning(true);
    try {
      const id = await api.startTrainingJob(project.id, preset);
      jobRef.current = id;
      setJobId(id);
    } catch (e) {
      setRunning(false);
      setError(toAppError(e));
    }
  };

  const cancel = async () => {
    if (jobId) await api.cancelTrainingJob(jobId).catch((e) => setError(toAppError(e)));
  };

  const ask = async () => {
    setAsking(true);
    try {
      await api.runBenchmarkArena(project.id, prompt || undefined, prompt ? 1 : 3);
    } catch (e) {
      setError(toAppError(e));
    } finally {
      setAsking(false);
    }
  };

  const promoted = gate?.decision === "PROMOTE";
  const finished = !running && (gate !== null || overall >= 100);

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow={project.name}
        title="Train and prove"
        lead="Your AI studies your examples, then sits a final exam it has never seen. It is only kept if it beats the original."
        action={running ? <Button variant="danger" onClick={cancel}>Stop training</Button> : undefined}
      />
      <ErrorNote error={error} onRetry={error?.retryable ? () => start("standard") : undefined} />

      {!running && !finished && (
        <>
          <div className="grid gap-5 sm:grid-cols-2">
            <ChoiceCard icon={Rocket} title="Best results" onClick={() => start("standard")}>
              Trains for as long as it keeps improving, up to about 8 hours. Choose this when the result matters.
            </ChoiceCard>
            <ChoiceCard icon={Gauge} title="Quick run" onClick={() => start("fast")}>
              A shorter run, up to about 90 minutes. Good for a first look at whether your examples teach the skill.
            </ChoiceCard>
          </div>
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
        </>
      )}

      {(running || finished) && (
        <Card className="space-y-5">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <div className="text-[15px] text-muted">{running ? stageDetail || "Starting…" : "Finished"}</div>
            <div className="text-[15px] font-medium tabular-nums">{Math.round(overall)}%<span className="ml-2 font-normal text-muted">{running ? `${fmtEta(eta)} left in this step` : "done"}</span></div>
          </div>
          <ProgressBar value={overall} label="Overall training progress" />
          <ol className="flex flex-wrap gap-x-6 gap-y-2 text-sm">
            {STAGES.map((s) => {
              const st = stageState[s.id];
              return (
                <li key={s.id} className={`inline-flex items-center gap-1.5 ${st === "done" ? "text-ink" : st === "active" ? "font-semibold text-ink" : st === "failed" ? "text-bad" : "text-faint"}`}>
                  {st === "done" ? <Icon icon={Check} size={14} className="text-good" /> : st === "failed" ? <Icon icon={X} size={14} /> : <span className={`h-1.5 w-1.5 rounded-full ${st === "active" ? "animate-pulse bg-ink" : "bg-line"}`} aria-hidden />}
                  {s.label}
                </li>
              );
            })}
          </ol>
        </Card>
      )}

      {gate && (
        <Card className={`space-y-4 ${promoted ? "border-ink" : ""}`}>
          <div className="flex flex-wrap items-center gap-2">
            {promoted ? <Badge tone="good"><Icon icon={Check} size={13} /> Better: new version {gate.version ?? ""} saved</Badge> : gate.decision === "INSUFFICIENT_DATA" ? <Badge tone="warn">Need more examples</Badge> : <Badge tone="warn">{project.active_version ? "Not better yet: your current version stays active" : "Not better yet: nothing was saved"}</Badge>}
            {gate.label && <Badge>{gate.label === "VERIFIED" ? "Verified result" : "Preliminary result"}</Badge>}
          </div>
          <p className="max-w-[70ch] text-[17px] leading-[1.65]">{gate.explanation}</p>
          <div className="flex flex-wrap gap-3 pt-1">
            {promoted && <Button onClick={onDeploy}>Use it in your apps</Button>}
            <Button variant="secondary" onClick={() => start("standard")}>Train again</Button>
          </div>
        </Card>
      )}

      {(running || finished) && (
        <div className="grid gap-5 lg:grid-cols-5">
          <Card className="lg:col-span-2 lg:self-start">
            <LiftGauge label={candRate !== null ? "Final exam, hidden questions" : "Practice score so far"} before={baseRate} after={candRate ?? devPass} />
            <p className="mt-6 text-sm leading-relaxed text-muted">“Original” is the untouched model on the hidden exam. {candRate === null ? "“Yours” is your model on practice questions while it learns." : "“Yours” is your trained model on the same hidden exam."}</p>
          </Card>
          <Card className="space-y-4 lg:col-span-3">
            <h2 className="text-[17px] font-semibold tracking-[-0.01em]">Same question, both models</h2>
            {duels.length === 0 && <div className="text-[15px] leading-relaxed text-muted">Side-by-side answers appear here as training checks its progress.</div>}
            <div className="divide-y divide-line">
              {duels.slice(0, 3).map((d) => (
                <Duel key={d.case_id} d={d} />
              ))}
            </div>
          </Card>
        </div>
      )}

      {finished && promoted && (
        <section className="space-y-5 pt-4">
          <SectionTitle title="Ask both models yourself" lead="Type a new example, or leave the box empty to draw from your practice questions." />
          <div className="flex gap-2">
            <TextInput value={prompt} onChange={setPrompt} onEnter={ask} placeholder="e.g. a new line from your records" label="Question for both models" />
            <Button onClick={ask} busy={asking}>Compare</Button>
          </div>
          {arena.length > 0 && (
            <Card className="divide-y divide-line py-1">
              {arena.map((d, i) => (
                <Duel key={d.case_id + i} d={d} />
              ))}
            </Card>
          )}
        </section>
      )}

      {(running || finished) && (
        <Drawer title="Advanced metrics" open={advanced} onToggle={() => setAdvanced(!advanced)}>
          <div className="grid gap-6 md:grid-cols-2">
            <div>
              <div className="mb-2 text-xs text-muted">Training loss</div>
              <Sparkline values={loss} />
            </div>
            <dl className="grid grid-cols-2 gap-1.5 font-mono text-xs">
              {Object.entries(metrics).map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="text-muted">{k}</dt>
                  <dd>{v === null ? "–" : Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(2) : Number(v.toFixed(4))}</dd>
                </div>
              ))}
            </dl>
          </div>
          <pre className="mt-5 max-h-56 overflow-auto rounded-xl bg-canvas p-4 font-mono text-[11px] leading-relaxed">{logs.join("\n") || "Logs appear here."}</pre>
        </Drawer>
      )}
    </div>
  );
}

function Duel({ d }: { d: Bench }) {
  return (
    <div className="py-4 text-[15px]">
      <div className="mb-3 line-clamp-2 font-medium leading-relaxed">{d.prompt}</div>
      <div className="grid gap-3 sm:grid-cols-2">
        {([["Original", d.baseline], ["Yours", d.trained]] as const).map(([name, side]) => (
          <div key={name} className={`rounded-xl px-3.5 py-3 ${side.pass ? "bg-good-soft" : side.pass === false ? "bg-bad-soft" : "bg-canvas"}`}>
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
