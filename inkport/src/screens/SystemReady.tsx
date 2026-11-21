import { useEffect, useState } from "react";
import type { AppError, HardwareReport, PackProgress, PackState, PreflightProgress, PreflightReport, RuntimeProgress } from "../lib/contract";
import { api, on, subscribe, toAppError } from "../lib/ipc";
import { Badge, Button, Card, Drawer, ErrorNote, ProgressBar, Spinner } from "../components/ui";

const gb = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`;

export function SystemReady({ onReady }: { onReady: (hw: HardwareReport) => void }) {
  const [hw, setHw] = useState<HardwareReport | null>(null);
  const [report, setReport] = useState<PreflightReport | null>(null);
  const [checking, setChecking] = useState(true);
  const [progress, setProgress] = useState<PreflightProgress | null>(null);
  const [error, setError] = useState<AppError | null>(null);
  const [advanced, setAdvanced] = useState(false);
  // Development builds only: online setup of a private runtime (installed apps never need this).
  const [devSteps, setDevSteps] = useState<RuntimeProgress[]>([]);
  const [devBusy, setDevBusy] = useState(false);

  const check = async (forceFull = false) => {
    setChecking(true);
    setError(null);
    try {
      const [h, r] = await Promise.all([api.detectHardware(), api.preflight(forceFull)]);
      setHw(h);
      setReport(r);
    } catch (e) {
      setError(toAppError(e));
    } finally {
      setChecking(false);
      setProgress(null);
    }
  };

  useEffect(() => {
    void check();
    return subscribe(on<PreflightProgress | RuntimeProgress>("runtime://progress", (p) => {
      if (p.type === "preflight") setProgress(p);
      else setDevSteps((prev) => [...prev.filter((s) => s.step !== p.step), p].sort((a, b) => a.step - b.step));
    }));
  }, []);

  const devSetup = async () => {
    setDevBusy(true);
    try {
      await api.bootstrapRuntime(true);
      await check();
    } catch (e) {
      setError(toAppError(e));
    } finally {
      setDevBusy(false);
    }
  };

  const ready = report?.ok ?? false;
  const starter = report?.starter_model;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">{ready ? "Inkport is ready" : "Getting Inkport ready"}</h1>
        <p className="mt-2 text-muted">Everything Inkport needs is already installed. Training, testing and your local AI all work offline; nothing you add leaves this computer.</p>
      </header>
      <ErrorNote error={error} onRetry={() => check()} />

      <Card className="space-y-4">
        {checking ? (
          <div className="space-y-3">
            <div className="flex items-center gap-3 text-muted">
              <Spinner /> {progress?.phase === "warmup" ? "Starting the AI engine…" : "Checking the installed files…"}
            </div>
            {progress && progress.phase === "verify" && <ProgressBar value={progress.pct} label="Checking installed files" />}
          </div>
        ) : report && ready ? (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone="good">✓ Ready in {report.seconds.toFixed(1)} s</Badge>
              {report.bundled && <Badge tone="good">✓ {report.verify.files_checked.toLocaleString()} files verified{report.verify.mode === "full" ? ` (${gb(report.verify.bytes_checked)})` : ""}</Badge>}
              {report.offline_core && <Badge>Works offline</Badge>}
            </div>
            <ul className="space-y-2 text-[15px]">
              {starter && <li>🧠 <span className="font-semibold">{starter.label}</span> is included ({(starter.size_mb / 1000).toFixed(1)} GB, {starter.license})</li>}
              <li>⚙️ Training runs on: <span className="font-semibold">{report.training_accelerator}</span></li>
              <li>⚡ Answers run on: <span className="font-semibold">{report.inference_devices.length ? report.inference_devices.map((d) => d.split(":")[1]?.split("(")[0]?.trim() || d).join(", ") : "Processor (CPU)"}</span></li>
              {hw && <li>💻 {hw.summary[0]}</li>}
            </ul>
            <div className="flex flex-wrap gap-3">
              <Button onClick={() => hw && onReady(hw)}>Start a project</Button>
            </div>
          </>
        ) : report && !report.bundled ? (
          <div className="space-y-3">
            <Badge tone="warn">Developer build</Badge>
            <p className="text-sm">This development build has no bundled runtime. Installed versions of Inkport never show this step.</p>
            {devSteps.map((s) => (
              <div key={s.step} className="text-sm">{s.status === "completed" ? "✓" : "…"} {s.label} <span className="text-muted">{s.detail}</span></div>
            ))}
            <Button onClick={devSetup} busy={devBusy}>Set up developer runtime (downloads)</Button>
          </div>
        ) : (
          <div className="space-y-3">
            <Badge tone="bad">Needs repair</Badge>
            <p className="text-sm">Some of Inkport’s installed files are missing or changed. Reinstall Inkport from the same installer (your projects are kept), then open it again.</p>
            <ul className="list-disc space-y-1 pl-5 font-mono text-xs">{report?.problems.map((p) => <li key={p}>{p}</li>)}</ul>
            <Button variant="secondary" onClick={() => check(true)}>Check again</Button>
          </div>
        )}
      </Card>

      {report?.bundled && <AddOns />}

      {report && (
        <Drawer title="Advanced details" open={advanced} onToggle={() => setAdvanced(!advanced)}>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-xs">
            {Object.entries({
              version: report.app_version,
              platform: `${report.platform} (${report.flavor})`,
              engine: report.engine_version ?? "–",
              python: report.python ?? "–",
              "integrity check": `${report.verify.mode}, ${report.verify.files_checked} files in ${report.verify.seconds.toFixed(2)} s${report.verify.manifest_pinned ? ", manifest pinned to this build" : ""}`,
              "inference devices": report.inference_devices.join("; ") || "CPU",
              ...(hw ? { "recommended tier": `${hw.recommendation.tier} · ${hw.recommendation.backend}`, ram: `${hw.ram_gb} GB`, disk: `${hw.disk_free_gb} GB free` } : {}),
            }).map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-muted">{k}</dt>
                <dd className="break-all">{String(v)}</dd>
              </div>
            ))}
          </dl>
          {report.bundled && (
            <Button variant="secondary" className="mt-3" onClick={() => check(true)}>
              Verify every file again
            </Button>
          )}
        </Drawer>
      )}
    </div>
  );
}

function AddOns() {
  const [packs, setPacks] = useState<PackState[]>([]);
  const [live, setLive] = useState<Record<string, PackProgress>>({});
  const [error, setError] = useState<AppError | null>(null);

  const refresh = () => api.listPacks().then(setPacks).catch((e) => setError(toAppError(e)));

  useEffect(() => {
    void refresh();
    return subscribe(on<PackProgress>("packs://progress", (p) => {
      setLive((l) => ({ ...l, [p.pack_id]: p }));
      if (p.status === "installed" || p.status === "failed" || p.status === "cancelled") void refresh();
    }));
  }, []);

  if (packs.length === 0) return null;
  return (
    <Card className="space-y-3">
      <div>
        <div className="text-sm font-semibold">Optional add-ons</div>
        <p className="text-sm text-muted">Not needed for anything above. Downloads are checked file by file and can be paused and resumed.</p>
      </div>
      <ErrorNote error={error} />
      <ul className="space-y-3">
        {packs.map((p) => {
          const lp = live[p.id];
          const busy = p.installing || (lp && ["resolving", "downloading", "verifying"].includes(lp.status));
          return (
            <li key={p.id} className="rounded-xl border border-line p-3">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="font-medium">{p.title}</div>
                  <div className="text-sm text-muted">{p.description}</div>
                  <div className="mt-1 text-xs text-muted">{(p.approx_size_mb / 1000).toFixed(1)} GB · {p.license}</div>
                </div>
                {p.installed ? (
                  <div className="flex items-center gap-2">
                    <Badge tone="good">✓ Installed</Badge>
                    <Button variant="ghost" onClick={() => api.removePack(p.id).then(refresh).catch((e) => setError(toAppError(e)))}>Remove</Button>
                  </div>
                ) : busy ? (
                  <Button variant="secondary" onClick={() => api.cancelPack(p.id)}>Pause</Button>
                ) : (
                  <Button variant="secondary" onClick={() => api.installPack(p.id).then(refresh).catch((e) => setError(toAppError(e)))}>
                    {p.partial_bytes > 0 ? `Resume (${(p.partial_bytes / 1e9).toFixed(1)} GB done)` : "Download"}
                  </Button>
                )}
              </div>
              {busy && lp && (
                <div className="mt-2 space-y-1">
                  <ProgressBar value={lp.pct} label={`${p.title} download`} />
                  <div className="text-xs text-muted">{lp.status === "downloading" ? `${(lp.done / 1e9).toFixed(2)} of ${(lp.total / 1e9).toFixed(2)} GB` : lp.status === "verifying" ? "Checking every file…" : "Preparing…"}</div>
                </div>
              )}
              {lp && (lp.status === "failed" || lp.status === "cancelled") && <div className="mt-2 text-xs text-muted">{lp.message}</div>}
            </li>
          );
        })}
      </ul>
    </Card>
  );
}
