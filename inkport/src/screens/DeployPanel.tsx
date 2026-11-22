import { useEffect, useRef, useState } from "react";
import { Check, Eye, EyeOff, Hourglass, TriangleAlert } from "lucide-react";
import type { ApiStatus, AppError, ProjectInfo } from "../lib/contract";
import { api, confirmDialog, isWindows, pickFiles, pickSavePath, toAppError } from "../lib/ipc";
import { Badge, Button, Card, CopyButton, ErrorNote, Icon, Note, PageHeader, SectionTitle, Spinner, TextInput } from "../components/ui";

type Tab = "python" | "javascript" | "shell";
type Line = { ok: boolean | null; text: string };

/** "Vulkan0: NVIDIA GeForce RTX 4060 (8188 MiB, 7700 MiB free)" -> "NVIDIA GeForce RTX 4060". */
function shortEngine(engine: string): string {
  const m = /^([^()]+)\((.*)\)\s*$/.exec(engine);
  if (!m) return engine;
  const devices = m[2].split(/,\s*(?=[A-Za-z]+\d*:)/).map((d) => (d.split(":")[1] ?? d).split("(")[0].trim()).filter(Boolean);
  return devices.length ? `${m[1].trim()} on ${devices.join(", ")}` : m[1].trim();
}

interface Props {
  project: ProjectInfo;
  /** The local API as the app knows it (it may be serving another project). */
  status: ApiStatus;
  onStatus: (s: ApiStatus) => void;
  trainingRunning: boolean;
  onTrain: () => void;
  onImported: () => void | Promise<void>;
}

export function DeployPanel({ project, status, onStatus, trainingRunning, onTrain, onImported }: Props) {
  const mine = status.running && status.project_id === project.id;
  const info = mine ? status.info : null;
  const other = status.running && !mine;
  const [busy, setBusy] = useState<null | "start" | "stop">(null);
  const [error, setError] = useState<AppError | null>(null);
  const [showKey, setShowKey] = useState(false);
  const [tab, setTab] = useState<Tab>("python");
  const [chatOpen, setChatOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState<{ role: "user" | "assistant"; content: string }[]>([]);
  const [sending, setSending] = useState(false);
  const [chatError, setChatError] = useState<AppError | null>(null);
  const [transfer, setTransfer] = useState<Line[]>([]);
  const [transferring, setTransferring] = useState<null | "export" | "import">(null);
  const [transferError, setTransferError] = useState<AppError | null>(null);
  const chatEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ block: "nearest" });
  }, [chat.length, sending]);

  const start = async () => {
    if (busy) return;
    setBusy("start");
    setError(null);
    try {
      const started = await api.startLocalApi(project.id);
      onStatus({ running: true, project_id: project.id, info: started });
      setChat([]);
    } catch (e) {
      setError(toAppError(e));
      onStatus(await api.apiStatus().catch(() => ({ running: false, project_id: null, info: null })));
    } finally {
      setBusy(null);
    }
  };

  const stop = async () => {
    if (busy) return;
    setBusy("stop");
    setError(null);
    try {
      await api.stopLocalApi();
    } catch (e) {
      setError(toAppError(e));
    } finally {
      onStatus(await api.apiStatus().catch(() => ({ running: false, project_id: null, info: null })));
      setBusy(null);
    }
  };

  const send = async () => {
    const text = message.trim();
    if (!info || !text || sending) return;
    const msgs = [...chat, { role: "user" as const, content: text }];
    setChat(msgs);
    setMessage("");
    setSending(true);
    setChatError(null);
    try {
      const reply = await api.chatLocalApi(msgs);
      setChat((cur) => (cur === msgs ? [...msgs, { role: "assistant", content: reply || "(the model returned an empty answer)" }] : cur));
    } catch (e) {
      setChatError(toAppError(e));
      setChat((cur) => (cur === msgs ? chat : cur));
      setMessage(text);
    } finally {
      setSending(false);
    }
  };

  const doExport = async () => {
    if (transferring) return;
    setTransferError(null);
    const dest = await pickSavePath(`${project.slug}-${project.active_version ?? "v1"}.inkpatch`);
    if (!dest) return;
    setTransferring("export");
    setTransfer([]);
    try {
      const r = await api.exportAdapter(project.id, dest);
      setTransfer([{ ok: true, text: `Saved version ${r.version} to ${r.path}` }, { ok: null, text: `Fingerprint (SHA-256): ${r.sha256}` }, { ok: null, text: `Signed with your key ${r.publisher.slice(0, 16)}…` }]);
    } catch (e) {
      setTransferError(toAppError(e));
    } finally {
      setTransferring(null);
    }
  };

  const doImport = async () => {
    if (transferring) return;
    setTransferError(null);
    const file = (await pickFiles([".inkpatch"]))[0];
    if (!file) return;
    setTransferring("import");
    setTransfer([]);
    try {
      let result;
      try {
        result = await api.importAdapter(project.id, file, false);
      } catch (e) {
        const err = toAppError(e);
        if (err.code !== "UNTRUSTED_PUBLISHER") throw e;
        // The answer must be awaited: in the desktop app a dialog is asynchronous.
        const trust = await confirmDialog("This adapter was signed by someone you have not trusted before.\n\nTrust this publisher and continue? The adapter still has to beat your current version on your own hidden exam before it is used.", "Unknown publisher");
        if (!trust) {
          setTransfer([{ ok: null, text: "Import cancelled. Nothing was changed." }]);
          return;
        }
        result = await api.importAdapter(project.id, file, true);
      }
      const lines: Line[] = [
        { ok: true, text: `Signature verified (publisher ${result.import.publisher.slice(0, 16)}…)` },
        { ok: true, text: "Every file matches the signed fingerprint" },
        { ok: true, text: `${result.import.checks.tensors} tensors fit your model (rank ${result.import.checks.rank})` },
      ];
      if (result.gate) lines.push(result.gate.decision === "PROMOTE" ? { ok: true, text: `It beat your current version on your hidden exam and is now in use as ${result.gate.version}` } : { ok: false, text: `Not used: ${result.gate.explanation}` });
      setTransfer(lines);
      await onImported();
    } catch (e) {
      setTransferError(toAppError(e));
    } finally {
      setTransferring(null);
    }
  };

  const key = showKey && info ? info.token : "YOUR_KEY";
  const snippets: Record<Tab, string> = info
    ? {
        python: `from openai import OpenAI\n\nclient = OpenAI(base_url="${info.base_url}", api_key="${key}")\nreply = client.chat.completions.create(\n    model="${info.model}",\n    messages=[{"role": "user", "content": "Paid $42 at Shell"}],\n)\nprint(reply.choices[0].message.content)`,
        javascript: `const res = await fetch("${info.url}", {\n  method: "POST",\n  headers: { Authorization: "Bearer ${key}", "Content-Type": "application/json" },\n  body: JSON.stringify({ model: "${info.model}", messages: [{ role: "user", content: "Paid $42 at Shell" }] }),\n});\nconsole.log((await res.json()).choices[0].message.content);`,
        shell: isWindows
          ? `$body = @{ model = "${info.model}"; messages = @(@{ role = "user"; content = 'Paid $42 at Shell' }) } | ConvertTo-Json -Depth 5\n$reply = Invoke-RestMethod -Uri "${info.url}" -Method Post -ContentType "application/json" -Headers @{ Authorization = "Bearer ${key}" } -Body $body\n$reply.choices[0].message.content`
          : `curl ${info.url} \\\n  -H "Authorization: Bearer ${key}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"model": "${info.model}", "messages": [{"role": "user", "content": "Paid $42 at Shell"}]}'`,
      }
    : { python: "", javascript: "", shell: "" };
  const tabLabel: Record<Tab, string> = { python: "Python", javascript: "JavaScript", shell: isWindows ? "PowerShell" : "curl" };
  const newer = info && project.active_version && info.version !== project.active_version ? project.active_version : null;

  if (!project.active_version) {
    return (
      <div className="space-y-10">
        <PageHeader eyebrow={project.name} title="Use your AI anywhere on this computer" lead="This project has no trained version yet. Train one first; it is saved here as soon as it beats the original on the hidden exam." />
        <Button onClick={onTrain}>Go to Train &amp; prove</Button>
      </div>
    );
  }

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow={project.name}
        title="Use your AI anywhere on this computer"
        lead="Inkport runs a private, OpenAI-compatible endpoint. Any app that lets you set a “custom OpenAI URL” can use your model."
        action={info ? <Button variant="secondary" onClick={stop} busy={busy === "stop"}>Stop local API</Button> : <Button onClick={start} busy={busy === "start"}>{busy === "start" ? "Starting…" : "Start local API"}</Button>}
      />
      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-3 text-[15px] text-muted">
          {info ? <Badge tone="good"><span className="h-1.5 w-1.5 rounded-full bg-good" aria-hidden /> Running · version {info.version}</Badge> : <Badge>Stopped</Badge>}
          {info ? <span className="min-w-0 break-words">Answering with {shortEngine(info.engine)}. Only apps on this computer can reach it.</span> : <span>Only apps on this computer can reach it.</span>}
        </div>
        {busy === "start" && <div className="flex items-center gap-2 text-sm text-muted"><Spinner /> Loading the model. This can take a minute the first time.</div>}
        {other && !busy && <Note icon={Hourglass}>The local API is currently serving another project. Starting it here switches it to “{project.name}”; apps using the other model will get this one instead.</Note>}
        {newer && <Note tone="warn" icon={TriangleAlert}><div className="flex flex-wrap items-center justify-between gap-3"><span>A newer version ({newer}) has been saved, but the API is still answering with {info?.version}.</span><Button variant="secondary" onClick={start} busy={busy === "start"}>Switch to {newer}</Button></div></Note>}
        {trainingRunning && !info && <Note icon={Hourglass}>Training is running. You can start the API now, but both will share this computer’s memory and each will be slower.</Note>}
        <ErrorNote error={error} title="The local API could not be started" />
      </div>

      {info && (
        <>
          <dl className="divide-y divide-line border-y border-line">
            {([["API address", info.base_url, info.base_url], ["Model name", info.model, info.model], ["API key", showKey ? info.token : "ik_" + "•".repeat(24), info.token]] as const).map(([label, shown, value]) => (
              <div key={label} className="flex flex-wrap items-center gap-x-4 gap-y-2 py-3.5">
                <dt className="w-32 flex-none text-[15px] text-muted">{label}</dt>
                <dd className="min-w-0 flex-1"><code className="block truncate font-mono text-sm" title={label === "API key" ? undefined : shown}>{shown}</code></dd>
                <div className="flex flex-none gap-2">
                  {label === "API key" && (
                    <button className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-panel px-2.5 py-1.5 text-[13px] font-medium transition-colors hover:border-ink" onClick={() => setShowKey(!showKey)}>
                      <Icon icon={showKey ? EyeOff : Eye} size={14} /> {showKey ? "Hide" : "Show"}
                    </button>
                  )}
                  <CopyButton text={value} />
                </div>
              </div>
            ))}
          </dl>

          <section className="space-y-5 pt-4">
            <SectionTitle title="Call it from your code" lead="The endpoint speaks the OpenAI chat format, so existing client libraries work unchanged." />
            <div>
              <div className="flex gap-7 border-b border-line" role="tablist">
                {(["python", "javascript", "shell"] as Tab[]).map((t) => (
                  <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)} className={`-mb-px border-b-2 pb-3 text-[15px] transition-colors ${tab === t ? "border-ink font-semibold text-ink" : "border-transparent text-muted hover:text-ink"}`}>{tabLabel[t]}</button>
                ))}
              </div>
              <div className="relative mt-5">
                <pre className="overflow-auto rounded-2xl bg-canvas p-5 pr-28 font-mono text-[13px] leading-[1.7]">{snippets[tab]}</pre>
                <div className="absolute right-3 top-3"><CopyButton text={snippets[tab].split("YOUR_KEY").join(info.token)} /></div>
              </div>
              {!showKey && <p className="mt-3 text-sm text-muted">The example shows YOUR_KEY in place of the key; “Copy” puts the real key in.</p>}
            </div>
          </section>

          <section className="space-y-5 pt-4">
            <SectionTitle title="Try it here" lead="Send a message to your trained model through the same endpoint your apps will use." />
            {!chatOpen ? (
              <Button variant="secondary" onClick={() => setChatOpen(true)}>Open test chat</Button>
            ) : (
              <Card className="space-y-4">
                <div className="max-h-72 space-y-2.5 overflow-auto">
                  {chat.length === 0 && !sending && <div className="text-[15px] text-muted">Send a message to your model.</div>}
                  {chat.map((m, i) => (
                    <div key={i} className={`w-fit max-w-[85%] whitespace-pre-wrap break-words rounded-2xl px-4 py-2.5 text-[15px] ${m.role === "user" ? "ml-auto bg-ink text-panel" : "bg-canvas font-mono text-[13px]"}`}>{m.content}</div>
                  ))}
                  {sending && <div className="flex w-fit items-center gap-2 rounded-2xl bg-canvas px-4 py-2.5 text-sm text-muted"><Spinner /> thinking…</div>}
                  <div ref={chatEnd} />
                </div>
                <ErrorNote error={chatError} title="The model did not answer" />
                <div className="flex gap-2">
                  <TextInput value={message} onChange={setMessage} onEnter={send} placeholder="Type a message" label="Message" />
                  <Button onClick={send} busy={sending} disabled={!message.trim()}>Send</Button>
                  <Button variant="ghost" onClick={() => setChatOpen(false)}>Close</Button>
                </div>
              </Card>
            )}
          </section>
        </>
      )}

      <section className="space-y-5 pt-4">
        <SectionTitle title="Share or back up your trained skill" lead="Adapters are small files (megabytes, not gigabytes). Each one is signed and fingerprinted; imports are verified and must beat your current version on your own hidden exam before they are switched on." />
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="secondary" onClick={doExport} busy={transferring === "export"} disabled={!!transferring}>Export adapter…</Button>
          <Button variant="secondary" onClick={doImport} busy={transferring === "import"} disabled={!!transferring || trainingRunning}>Import adapter…</Button>
        </div>
        {transferring === "import" && <div className="flex items-start gap-2 text-sm leading-relaxed text-muted"><Spinner /> Checking the file, then running it through your hidden exam against the current version. This takes a few minutes.</div>}
        {trainingRunning && <p className="text-sm text-muted">Importing is available when training has finished.</p>}
        <ErrorNote error={transferError} title={transferError?.code === "PATCH_REJECTED" ? "This adapter was not accepted" : "That did not work"} />
        {transfer.length > 0 && (
          <ul className="space-y-2 rounded-2xl bg-canvas p-5 text-[14px] leading-relaxed">
            {transfer.map((l) => (
              <li key={l.text} className="flex items-start gap-2.5 break-words">
                {l.ok === true ? <Icon icon={Check} size={15} className="mt-1 text-good" /> : l.ok === false ? <Icon icon={TriangleAlert} size={15} className="mt-1 text-warn" /> : <span className="w-[15px] flex-none" />}
                <span className="min-w-0 break-all">{l.text}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
