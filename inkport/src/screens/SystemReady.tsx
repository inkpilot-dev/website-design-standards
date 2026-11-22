import { useEffect, useState } from "react";
import type { AppError, HardwareReport, PackProgress, PackState, PreflightProgress, PreflightReport, RuntimeProgress } from "../lib/contract";
import { api, on, subscribe, toAppError } from "../lib/ipc";
import { Box, Check, Cpu, Monitor, Sparkles, Zap } from "lucide-react";
import { Badge, Button, Card, Drawer, ErrorNote, Icon, PageHeader, ProgressBar, SectionTitle, Spinner } from "../components/ui";

const gb = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`;

export function SystemReady({ onReady, onContinue }: { onReady: () => void; onContinue: () => void }) {
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
      await check(true);
    } catch (e) {
      setError(toAppError(e));
    } finally {
      setDevBusy(false);
    }
  };

  const ready = report?.ok ?? false;
  const starter = report?.starter_model;

  useEffect(() => {
    if (ready && hw) onReady();
  }, [ready, hw]);

  return (
    <div className="space-y-10">
      <PageHeader eyebrow="Get started" title={ready ? "Inkport is ready" : "Getting Inkport ready"} lead="Everything Inkport needs is already installed. Training, testing and your local AI all work offline; nothing you add leaves this computer." />
      <ErrorNote error={error} onRetry={() => check()} />

      <Card className="space-y-6">
        {checking ? (
          <div className="space-y-4">
            <div className="flex items-center gap-3 text-[15px] text-muted">
              <Spinner /> {progress?.phase === "warmup" ? "Starting the AI engine…" : "Checking the installed files…"}
            </div>
            {progress && progress.phase === "verify" && <ProgressBar value={progress.pct} label="Checking installed files" />}
          </div>
        ) : report && ready ? (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone="good"><Icon icon={Check} size={13} /> Ready in {report.seconds.toFixed(1)} s</Badge>
              {report.bundled && <Badge tone="good"><Icon icon={Check} size={13} /> {report.verify.files_checked.toLocaleString()} files verified{report.verify.mode === "full" ? ` (${gb(report.verify.bytes_checked)})` : ""}</Badge>}
              {report.offline_core && <Badge>Works offline</Badge>}
            </div>
            <dl className="divide-y divide-line border-y border-line text-[15px]">
              {[
                ...(starter ? [{ icon: Sparkles, k: "Included model", v: starter.label, note: `${starter.size_mb >= 100 ? `${(starter.size_mb / 1000).toFixed(1)} GB` : `${Math.max(1, Math.round(starter.size_mb))} MB`} · ${starter.license}` }] : []),
                { icon: Cpu, k: "Training runs on", v: report.training_accelerator, note: "" },
                { icon: Zap, k: "Answers run on", v: report.inference_devices.length ? report.inference_devices.map((d) => d.split(":")[1]?.split("(")[0]?.trim() || d).join(", ") : "Processor (CPU)", note: "" },
                ...(hw ? [{ icon: Monitor, k: "This computer", v: hw.summary[0], note: "" }] : []),
              ].map(({ icon, k, v, note }) => (
                <div key={k} className="flex flex-wrap items-center gap-x-4 gap-y-1 py-3.5">
                  <dt className="flex w-48 flex-none items-center gap-3 text-muted"><Icon icon={icon} size={18} className="text-ink" /> {k}</dt>
                  <dd className="min-w-0 flex-1 font-medium">{v}{note && <span className="ml-2 font-normal text-muted">{note}</span>}</dd>
                </div>
              ))}
            </dl>
            <div className="flex flex-wrap gap-3">
              <Button onClick={onContinue}>Continue</Button>
            </div>
          </>
        ) : report && !report.bundled ? (
          <div className="space-y-3">
            <Badge tone="warn">Developer build</Badge>
            <p className="text-[15px] leading-relaxed">This development build has no bundled runtime. Installed versions of Inkport never show this step.</p>
            {devSteps.map((s) => (
              <div key={s.step} className="flex items-center gap-2 text-[15px]">{s.status === "completed" ? <Icon icon={Check} /> : <Spinner />} {s.label} <span className="text-muted">{s.detail}</span></div>
            ))}
            <Button onClick={devSetup} busy={devBusy}>Set up developer runtime (downloads)</Button>
          </div>
        ) : !report ? (
          <div className="space-y-3">
            <p className="text-[15px] leading-relaxed">Inkport could not finish checking this computer. The details are above.</p>
            <Button variant="secondary" onClick={() => check()}>Check again</Button>
          </div>
        ) : (
          <div className="space-y-3">
            <Badge tone="bad">Needs repair</Badge>
            <p className="text-[15px] leading-relaxed">Some of Inkport’s installed files are missing or changed. Reinstall Inkport from the same installer (your projects are kept), then open it again. If this keeps happening, an antivirus program may be removing Inkport’s files: add Inkport’s folder to its exceptions.</p>
            <ul className="list-disc space-y-1 pl-5 font-mono text-xs">{report?.problems.map((p) => <li key={p}>{p}</li>)}</ul>
            <Button variant="secondary" onClick={() => check(true)}>Check again</Button>
          </div>
        )}
      </Card>

      {report?.bundled && <AddOns />}

      {report && (
        <Drawer title="Advanced details" open={advanced} onToggle={() => setAdvanced(!advanced)}>
          <dl className="grid grid-cols-[auto_1fr] gap-x-8 gap-y-1.5 font-mono text-xs">
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
            <Button variant="secondary" className="mt-5" onClick={() => check(true)}>
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
    <section className="space-y-6 pt-4">
      <SectionTitle title="Optional add-ons" lead="Not needed for anything above. Downloads are checked file by file and can be paused and resumed." />
      <ErrorNote error={error} />
      <ul className="grid gap-5 sm:grid-cols-2">
        {packs.map((p) => {
          const lp = live[p.id];
          const busy = p.installing || (lp && ["resolving", "downloading", "verifying"].includes(lp.status));
          return (
            <li key={p.id} className="flex flex-col rounded-2xl border border-line p-7">
              <Icon icon={Box} size={26} />
              <div className="mt-5 text-[17px] font-semibold tracking-[-0.01em]">{p.title}</div>
              <div className="mt-1.5 flex-1 text-[15px] leading-[1.6] text-muted">{p.description}</div>
              <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
                <div className="text-sm text-muted">{(p.approx_size_mb / 1000).toFixed(1)} GB · {p.license}</div>
                {p.installed ? (
                  <div className="flex items-center gap-2">
                    <Badge tone="good"><Icon icon={Check} size={13} /> Installed</Badge>
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
                <div className="mt-4 space-y-2">
                  <ProgressBar value={lp.pct} label={`${p.title} download`} />
                  <div className="text-sm text-muted">{lp.status === "downloading" ? `${(lp.done / 1e9).toFixed(2)} of ${(lp.total / 1e9).toFixed(2)} GB` : lp.status === "verifying" ? "Checking every file…" : "Preparing…"}</div>
                </div>
              )}
              {lp && (lp.status === "failed" || lp.status === "cancelled") && <div className="mt-3 text-sm text-muted">{lp.message}</div>}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
