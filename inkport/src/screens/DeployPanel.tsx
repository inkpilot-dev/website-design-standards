import { useState } from "react";
import type { ApiInfo, AppError, ProjectInfo } from "../lib/contract";
import { api, inTauri, pickFiles, pickSavePath, toAppError } from "../lib/ipc";
import { Badge, Button, Card, ErrorNote } from "../components/ui";

type Tab = "curl" | "python" | "javascript";

export function DeployPanel({ project }: { project: ProjectInfo }) {
  const [info, setInfo] = useState<ApiInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<AppError | null>(null);
  const [showKey, setShowKey] = useState(false);
  const [tab, setTab] = useState<Tab>("python");
  const [copied, setCopied] = useState("");
  const [chatOpen, setChatOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState<{ role: "user" | "assistant"; content: string }[]>([]);
  const [sending, setSending] = useState(false);
  const [transfer, setTransfer] = useState<string[]>([]);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      setInfo(await api.startLocalApi(project.id));
    } catch (e) {
      setError(toAppError(e));
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    await api.stopLocalApi().catch((e) => setError(toAppError(e)));
    setInfo(null);
  };

  const copy = async (label: string, text: string) => {
    await navigator.clipboard.writeText(text).catch(() => {});
    setCopied(label);
    setTimeout(() => setCopied(""), 1500);
  };

  const send = async () => {
    if (!info || !message.trim()) return;
    const msgs = [...chat, { role: "user" as const, content: message.trim() }];
    setChat(msgs);
    setMessage("");
    setSending(true);
    try {
      let reply: string;
      if (inTauri) {
        const r = await fetch(`${info.base_url}/chat/completions`, { method: "POST", headers: { Authorization: `Bearer ${info.token}`, "Content-Type": "application/json" }, body: JSON.stringify({ model: info.model, messages: msgs, max_tokens: 256 }) });
        const j = await r.json();
        reply = r.ok ? j.choices[0].message.content : `Error: ${j.error?.message ?? r.status}`;
      } else {
        reply = '{"category": "fuel", "amount": 42}';
      }
      setChat([...msgs, { role: "assistant", content: reply }]);
    } catch (e) {
      setError(toAppError(e));
    } finally {
      setSending(false);
    }
  };

  const doExport = async () => {
    const dest = await pickSavePath(`${project.slug}-${project.active_version ?? "v1"}.inkpatch`);
    if (!dest) return;
    try {
      const r = await api.exportAdapter(project.id, dest);
      setTransfer([`Exported ${r.version} to ${r.path}`, `SHA-256 ${r.sha256}`, `Signed by publisher key ${r.publisher.slice(0, 16)}…`]);
    } catch (e) {
      setError(toAppError(e));
    }
  };

  const doImport = async (trust = false, file?: string) => {
    const f = file ?? (await pickFiles([".inkpatch"]))[0];
    if (!f) return;
    setTransfer(["Checking signature, SHA-256 hashes and every tensor…"]);
    try {
      const r = await api.importAdapter(project.id, f, trust);
      const lines = [`✓ Signature verified (publisher ${r.import.publisher.slice(0, 16)}…)`, "✓ Every file's SHA-256 matches the signed manifest", `✓ ${r.import.checks.tensors} tensors checked against your base model (rank ${r.import.checks.rank})`];
      if (r.gate) lines.push(r.gate.decision === "PROMOTE" ? `✓ Beat your current version on your hidden exam, now active as ${r.gate.version}` : `Kept your current version: ${r.gate.explanation}`);
      setTransfer(lines);
    } catch (e) {
      const err = toAppError(e);
      if (err.code === "UNTRUSTED_PUBLISHER" && confirm("This adapter was signed by someone you haven't trusted before. Trust this publisher and continue? It will still have to pass your own hidden exam.")) {
        return doImport(true, f);
      }
      setTransfer([]);
      setError(err);
    }
  };

  const snippets: Record<Tab, string> = info
    ? {
        curl: `curl ${info.url} \\\n  -H "Authorization: Bearer ${showKey ? info.token : "YOUR_KEY"}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"model": "${info.model}", "messages": [{"role": "user", "content": "Paid $42 at Shell"}]}'`,
        python: `from openai import OpenAI\n\nclient = OpenAI(base_url="${info.base_url}", api_key="${showKey ? info.token : "YOUR_KEY"}")\nreply = client.chat.completions.create(\n    model="${info.model}",\n    messages=[{"role": "user", "content": "Paid $42 at Shell"}],\n)\nprint(reply.choices[0].message.content)`,
        javascript: `const res = await fetch("${info.url}", {\n  method: "POST",\n  headers: { Authorization: "Bearer ${showKey ? info.token : "YOUR_KEY"}", "Content-Type": "application/json" },\n  body: JSON.stringify({ model: "${info.model}", messages: [{ role: "user", content: "Paid $42 at Shell" }] }),\n});\nconsole.log((await res.json()).choices[0].message.content);`,
      }
    : { curl: "", python: "", javascript: "" };

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">Use your AI anywhere on this computer</h1>
        <p className="mt-2 text-muted">Inkport runs a private, OpenAI-compatible endpoint. Any app that lets you set a “custom OpenAI URL” can use your model.</p>
      </header>
      <ErrorNote error={error} />
      <Card className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            {info ? <Badge tone="good">● Running · version {info.version} · {info.engine}</Badge> : <Badge>Stopped</Badge>}
            <span className="text-sm text-muted">Only apps on this computer can reach it.</span>
          </div>
          {info ? <Button variant="secondary" onClick={stop}>Stop</Button> : <Button onClick={start} busy={busy}>Start local API</Button>}
        </div>
        {info && (
          <>
            <div className="grid gap-3 md:grid-cols-3">
              {([["API address", info.base_url], ["Model name", info.model], ["API key", showKey ? info.token : "ik_" + "•".repeat(20)]] as const).map(([label, value]) => (
                <div key={label} className="rounded-xl bg-canvas p-3">
                  <div className="text-xs text-muted">{label}</div>
                  <div className="mt-1 flex items-center justify-between gap-2">
                    <code className="truncate text-sm">{value}</code>
                    <button className="text-xs font-semibold text-accent" onClick={() => copy(label, label === "API key" ? info.token : value)}>{copied === label ? "Copied" : "Copy"}</button>
                  </div>
                  {label === "API key" && <button className="mt-1 text-xs text-muted underline" onClick={() => setShowKey(!showKey)}>{showKey ? "Hide" : "Show"}</button>}
                </div>
              ))}
            </div>
            <div>
              <div className="flex gap-1" role="tablist">
                {(["python", "javascript", "curl"] as Tab[]).map((t) => (
                  <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)} className={`rounded-t-lg px-3 py-1.5 text-xs font-semibold ${tab === t ? "bg-canvas text-ink" : "text-muted"}`}>{t === "javascript" ? "JavaScript" : t === "python" ? "Python" : "curl"}</button>
                ))}
              </div>
              <div className="relative">
                <pre className="overflow-auto rounded-b-xl rounded-tr-xl bg-canvas p-3 font-mono text-xs leading-relaxed">{snippets[tab]}</pre>
                <button className="absolute right-2 top-2 text-xs font-semibold text-accent" onClick={() => copy("snippet", snippets[tab])}>{copied === "snippet" ? "Copied" : "Copy"}</button>
              </div>
            </div>
            <Button variant="secondary" onClick={() => setChatOpen(!chatOpen)}>{chatOpen ? "Close test chat" : "Try it: test chat"}</Button>
          </>
        )}
      </Card>

      {info && chatOpen && (
        <Card className="space-y-3">
          <div className="max-h-72 space-y-2 overflow-auto">
            {chat.length === 0 && <div className="text-sm text-muted">Send a message to your model.</div>}
            {chat.map((m, i) => (
              <div key={i} className={`max-w-[85%] rounded-xl px-3 py-2 text-sm ${m.role === "user" ? "ml-auto bg-accent text-on-accent" : "bg-canvas font-mono"}`}>{m.content}</div>
            ))}
          </div>
          <div className="flex gap-2">
            <input value={message} onChange={(e) => setMessage(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send()} placeholder="Type a message" className="flex-1 rounded-xl border border-line bg-canvas px-3 py-2 text-sm outline-none focus:border-accent" />
            <Button onClick={send} busy={sending}>Send</Button>
          </div>
        </Card>
      )}

      <Card className="space-y-3">
        <div className="text-sm font-semibold">Share or back up your trained skill</div>
        <p className="text-sm text-muted">Adapters are small files (megabytes, not gigabytes). Each one is signed and fingerprinted; imports are verified and must beat your current version on your own hidden exam before they are switched on.</p>
        <div className="flex flex-wrap gap-3">
          <Button variant="secondary" onClick={doExport} disabled={!project.active_version && !info}>Export adapter…</Button>
          <Button variant="secondary" onClick={() => doImport()}>Import adapter…</Button>
        </div>
        {transfer.length > 0 && (
          <ul className="space-y-1 rounded-xl bg-canvas p-3 font-mono text-xs">
            {transfer.map((l) => (
              <li key={l} className="break-all">{l}</li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
