//! Supervises `python -m engine <command>` child processes.
//!
//! * stdout is the NDJSON protocol: each line is parsed as [`EngineEvent`] and forwarded
//!   (raw JSON plus `job_id`, `job_kind`, `project_id`) to the UI channel for the job kind;
//!   `log` events always go to `training://log`.
//! * stderr (library noise, tracebacks) is forwarded as debug logs, rate-limited.
//! * stdin is held open; the engine exits by itself if the supervisor dies (EOF watch).
//! * a watchdog kills a job that has been silent for longer than `stall_secs`
//!   (the engine emits heartbeats every 15 s while busy).
//! * cancel kills the whole process group (Unix) / process tree (Windows).

use crate::contract::{EngineError, EngineEvent, JobKind, JobOutcome};
use crate::events::EventSink;
use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::BufReader;
use tokio::process::{Child, Command};
use tokio::sync::{oneshot, watch};

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub python: PathBuf,
    /// Set in development so `python -m engine` resolves to the checkout.
    pub pythonpath: Option<PathBuf>,
    pub home: PathBuf,
    pub stall_secs: u64,
    /// Interpreter flags (bundled runtime: `-I -B`, fully isolated from any user Python setup).
    pub python_flags: Vec<String>,
    /// Extra environment for every engine process (offline switches, bundle paths).
    pub env: Vec<(String, String)>,
    /// Variables removed from the inherited environment.
    pub env_remove: Vec<String>,
}

pub struct JobSpec {
    pub kind: JobKind,
    pub args: Vec<String>,
    pub project_id: Option<String>,
    pub env: Vec<(String, String)>,
}

struct Running {
    pid: Option<u32>,
    cancel: watch::Sender<bool>,
    kind: JobKind,
    project_id: Option<String>,
    /// The events a window needs to redraw this job's screen after it was away (stages, results,
    /// the latest progress and side-by-side answers). Bounded; kept only while the job runs.
    history: Vec<Value>,
}

/// A running job as reported to the UI: enough to re-attach to it and replay what happened so far.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunningJob {
    pub job_id: String,
    pub kind: JobKind,
    pub project_id: Option<String>,
    pub events: Vec<Value>,
}

pub struct Supervisor {
    pub cfg: EngineConfig,
    sink: Arc<dyn EventSink>,
    jobs: Mutex<HashMap<String, Running>>,
}

pub struct Spawned {
    pub job_id: String,
    pub ready: oneshot::Receiver<Value>,
    pub finished: tokio::task::JoinHandle<JobOutcome>,
}

impl Supervisor {
    pub fn new(cfg: EngineConfig, sink: Arc<dyn EventSink>) -> Self {
        Self { cfg, sink, jobs: Mutex::new(HashMap::new()) }
    }

    pub fn running(&self) -> Vec<(String, JobKind)> {
        self.jobs.lock().unwrap().iter().map(|(k, v)| (k.clone(), v.kind)).collect()
    }

    /// The running job of this kind, if any, with its replayable events.
    pub fn running_job(&self, kind: JobKind) -> Option<RunningJob> {
        self.jobs.lock().unwrap().iter().find(|(_, v)| v.kind == kind).map(|(k, v)| RunningJob { job_id: k.clone(), kind: v.kind, project_id: v.project_id.clone(), events: v.history.clone() })
    }

    /// Is a job of one of these kinds running for this project (or for any project if `None`)?
    pub fn busy(&self, kinds: &[JobKind], project_id: Option<&str>) -> bool {
        self.jobs.lock().unwrap().values().any(|v| kinds.contains(&v.kind) && (project_id.is_none() || v.project_id.as_deref() == project_id))
    }

    fn remember(&self, job_id: &str, v: &Value) {
        let kind = v.get("type").and_then(Value::as_str).unwrap_or("");
        if !matches!(kind, "stage" | "progress" | "eval" | "gate" | "benchmark" | "error") {
            return;
        }
        let mut jobs = self.jobs.lock().unwrap();
        let Some(job) = jobs.get_mut(job_id) else { return };
        if !matches!(job.kind, JobKind::Train | JobKind::Import) {
            return;
        }
        // One progress event is enough (the latest); a handful of the most recent comparisons.
        if kind == "progress" {
            job.history.retain(|e| e["type"] != "progress");
        } else if kind == "benchmark" {
            let n = job.history.iter().filter(|e| e["type"] == "benchmark").count();
            if n >= 8 {
                if let Some(i) = job.history.iter().position(|e| e["type"] == "benchmark") {
                    job.history.remove(i);
                }
            }
        }
        if job.history.len() < 400 {
            job.history.push(v.clone());
        }
    }

    fn command(&self, job_id: &str, spec: &JobSpec) -> Command {
        let mut cmd = Command::new(&self.cfg.python);
        cmd.args(&self.cfg.python_flags).arg("-u").arg("-m").arg("engine").args(&spec.args);
        for k in &self.cfg.env_remove {
            cmd.env_remove(k);
        }
        cmd.envs(self.cfg.env.iter().cloned());
        cmd.env("INKPORT_JOB_ID", job_id)
            .env("INKPORT_WATCH_STDIN", "1")
            .env("INKPORT_HOME", &self.cfg.home)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .env("TOKENIZERS_PARALLELISM", "false")
            .env("HF_HUB_DISABLE_TELEMETRY", "1")
            .env("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1");
        if let Some(pp) = &self.cfg.pythonpath {
            cmd.env("PYTHONPATH", pp);
        }
        for (k, v) in &spec.env {
            cmd.env(k, v);
        }
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000 | 0x0000_0200); // CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        cmd
    }

    /// Start a job in the background. `ready` resolves on the first `ready` event (serve jobs).
    pub fn spawn(self: &Arc<Self>, spec: JobSpec) -> Result<Spawned> {
        let job_id = format!("job-{}", &uuid::Uuid::new_v4().simple().to_string()[..12]);
        let mut child: Child = self
            .command(&job_id, &spec)
            .spawn()
            .with_context(|| format!("could not start the Inkport engine ({}). Run setup first.", self.cfg.python.display()))?;
        let pid = child.id();
        let (cancel_tx, cancel_rx) = watch::channel(false);
        crate::procguard::adopt_tokio(&child);
        self.jobs.lock().unwrap().insert(job_id.clone(), Running { pid, cancel: cancel_tx, kind: spec.kind, project_id: spec.project_id.clone(), history: vec![] });
        let (ready_tx, ready_rx) = oneshot::channel();
        let me = Arc::clone(self);
        let jid = job_id.clone();
        let finished = tokio::spawn(async move {
            let outcome = me.drive(&jid, &spec, &mut child, cancel_rx, ready_tx).await;
            me.jobs.lock().unwrap().remove(&jid);
            outcome
        });
        Ok(Spawned { job_id, ready: ready_rx, finished })
    }

    /// Start a job and wait for it to finish.
    pub async fn run(self: &Arc<Self>, spec: JobSpec) -> Result<JobOutcome> {
        let sp = self.spawn(spec)?;
        Ok(sp.finished.await?)
    }

    fn forward(&self, spec: &JobSpec, job_id: &str, mut v: Value) {
        if let Some(obj) = v.as_object_mut() {
            obj.insert("job_id".into(), json!(job_id));
            obj.insert("job_kind".into(), json!(spec.kind));
            if let Some(p) = &spec.project_id {
                obj.insert("project_id".into(), json!(p));
            }
        }
        let kind = v.get("type").and_then(Value::as_str).unwrap_or("");
        let channel = match kind {
            "log" => "training://log",
            "benchmark" => "benchmark://result",
            _ => spec.kind.channel(),
        };
        self.remember(job_id, &v);
        self.sink.emit(channel, &v);
    }

    async fn drive(&self, job_id: &str, spec: &JobSpec, child: &mut Child, mut cancel: watch::Receiver<bool>, ready_tx: oneshot::Sender<Value>) -> JobOutcome {
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let _stdin = child.stdin.take(); // keep open for the engine's parent-death watch
        // Read raw lines and decode leniently: one byte that is not valid UTF-8 (a legacy Windows
        // code page, a library printing binary) must never stop us draining the pipes, or the
        // engine would block forever on a full pipe.
        let (mut out_r, mut out_buf) = (BufReader::new(stdout), Vec::new());
        let (mut err_r, mut err_buf) = (BufReader::new(stderr), Vec::new());
        let mut ready_tx = Some(ready_tx);
        let mut done: Option<(bool, Value)> = None;
        let mut error: Option<EngineError> = None;
        let mut error_was_sent = false; // the engine's own `error` event already reached the UI
        let mut last = Instant::now();
        let mut stderr_tail: Vec<String> = vec![];
        let mut stderr_budget: u32 = 0;
        let mut stderr_window = Instant::now();
        let stall = Duration::from_secs(self.cfg.stall_secs);
        let mut stdout_open = true;
        let mut stderr_open = true;
        let cancelled = loop {
            if !stdout_open && !stderr_open {
                break false;
            }
            tokio::select! {
                line = next_line_lossy(&mut out_r, &mut out_buf), if stdout_open => match line {
                    Ok(Some(line)) => {
                        last = Instant::now();
                        let trimmed = line.trim();
                        if trimmed.is_empty() { continue; }
                        match serde_json::from_str::<Value>(trimmed) {
                            Ok(v) => {
                                match serde_json::from_value::<EngineEvent>(v.clone()) {
                                    Ok(EngineEvent::Ready { .. }) => { if let Some(tx) = ready_tx.take() { let _ = tx.send(v.clone()); } }
                                    Ok(EngineEvent::Done { ok, result }) => done = Some((ok, result)),
                                    Ok(EngineEvent::Error { code, message, retryable }) => {
                                        error = Some(EngineError { code, message, retryable });
                                        error_was_sent = true;
                                    }
                                    Ok(_) => {}
                                    Err(e) => self.sink.emit("training://log", &json!({"type":"log","level":"warn","message":format!("protocol: unexpected event shape ({e})"),"job_id":job_id})),
                                }
                                self.forward(spec, job_id, v);
                            }
                            Err(_) => self.sink.emit("training://log", &json!({"type":"log","level":"debug","message":trimmed,"job_id":job_id})),
                        }
                    }
                    _ => stdout_open = false,
                },
                line = next_line_lossy(&mut err_r, &mut err_buf), if stderr_open => match line {
                    Ok(Some(line)) => {
                        last = Instant::now();
                        if stderr_tail.len() >= 40 { stderr_tail.remove(0); }
                        stderr_tail.push(line.clone());
                        if stderr_window.elapsed() > Duration::from_secs(1) { stderr_window = Instant::now(); stderr_budget = 0; }
                        stderr_budget += 1;
                        if stderr_budget <= 20 {
                            self.sink.emit("training://log", &json!({"type":"log","level":"debug","message":line,"job_id":job_id,"job_kind":spec.kind}));
                        }
                    }
                    _ => stderr_open = false,
                },
                _ = cancel.changed() => break true,
                _ = tokio::time::sleep(Duration::from_secs(5)) => {
                    if last.elapsed() > stall && spec.kind != JobKind::Serve {
                        error = Some(EngineError { code: "ENGINE_STALLED".into(), message: format!("Inkport's engine stopped responding for {} minutes, so it was stopped. Try again; if it happens again, restart the computer first.", stall.as_secs() / 60), retryable: true });
                        error_was_sent = false;
                        break true;
                    }
                }
            }
        };
        if cancelled {
            kill_tree(child).await;
            if error.is_none() || error_was_sent {
                // A stop request wins over whatever the engine printed while it was being stopped.
                if error.as_ref().map(|e| e.code != "ENGINE_STALLED").unwrap_or(true) {
                    error = Some(EngineError { code: "CANCELLED".into(), message: "Stopped. Nothing was saved from this run.".into(), retryable: true });
                    error_was_sent = false;
                }
            }
        }
        let status = child.wait().await.ok();
        let had_done = done.is_some();
        let (ok, result) = done.unwrap_or_else(|| (false, json!({})));
        if !ok && error.is_none() {
            error = Some(crash_report(spec.kind, status, &stderr_tail, &self.cfg.home, job_id));
            error_was_sent = false;
        }
        let outcome = JobOutcome { job_id: job_id.to_string(), ok: ok && !cancelled, result, error };
        if !outcome.ok && !error_was_sent {
            // Jobs started in the background (training) have nobody waiting for the outcome, so the
            // reason must travel as an event: a stop, a crash, running out of memory, a stall.
            if let Some(e) = &outcome.error {
                self.forward(spec, job_id, json!({"v":1,"type":"error","ts":0,"code":e.code,"message":e.message,"retryable":e.retryable}));
            }
        }
        if cancelled || !had_done {
            // The engine could not report; synthesize a terminal event so the UI never hangs.
            self.forward(spec, job_id, json!({"v":1,"type":"done","ts":0,"ok":false,"result":{"code": outcome.error.as_ref().map(|e| e.code.clone())}}));
        }
        outcome
    }

    pub async fn cancel(&self, job_id: &str) -> Result<()> {
        let tx = {
            let jobs = self.jobs.lock().unwrap();
            jobs.get(job_id).map(|r| (r.cancel.clone(), r.pid))
        };
        match tx {
            Some((tx, _pid)) => {
                let _ = tx.send(true);
                Ok(())
            }
            None => Err(anyhow!("no running job {job_id}")),
        }
    }

    pub async fn cancel_all(&self) {
        let ids: Vec<String> = self.jobs.lock().unwrap().keys().cloned().collect();
        for id in ids {
            let _ = self.cancel(&id).await;
        }
    }
}

async fn kill_tree(child: &mut Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        // SAFETY: pid is our own child's process-group id (we set process_group(0)).
        unsafe {
            libc::killpg(pid as i32, libc::SIGTERM);
        }
        if tokio::time::timeout(Duration::from_secs(5), child.wait()).await.is_ok() {
            return;
        }
        unsafe {
            libc::killpg(pid as i32, libc::SIGKILL);
        }
    }
    #[cfg(windows)]
    if let Some(pid) = child.id() {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW: without it a console window flashes on screen when the user presses Stop.
        let _ = std::process::Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).creation_flags(0x0800_0000).stdin(Stdio::null()).output();
    }
    let _ = child.kill().await;
}

/// What to tell the user when the engine process died without reporting an error itself.
/// Running out of memory is by far the most common cause, and the only one they can act on.
fn crash_report(kind: JobKind, status: Option<std::process::ExitStatus>, stderr_tail: &[String], home: &std::path::Path, job_id: &str) -> EngineError {
    // Progress bars redraw one line with carriage returns; they are noise in an error report.
    let lines: Vec<String> = stderr_tail
        .iter()
        .flat_map(|l| l.split('\r').map(str::trim).filter(|p| !p.is_empty()).map(String::from).collect::<Vec<_>>())
        .filter(|l| !(l.contains("it/s]") || l.contains("s/it]") || (l.contains('%') && l.contains('|') && l.contains('['))))
        .collect();
    let text = lines.join("\n");
    let lower = text.to_lowercase();
    let status_text = status.map(|s| s.to_string()).unwrap_or_else(|| "unknown exit".into());
    #[cfg(unix)]
    let killed = {
        use std::os::unix::process::ExitStatusExt;
        status.and_then(|s| s.signal()) == Some(9)
    };
    #[cfg(not(unix))]
    // STATUS_NO_MEMORY, STATUS_COMMITMENT_LIMIT, and the C++ runtime's abort after std::bad_alloc.
    let killed = matches!(status.and_then(|s| s.code()).map(|c| c as u32), Some(0xC000_0017) | Some(0xC000_012D));
    let memory_words = ["memoryerror", "out of memory", "not enough memory", "cannot allocate memory", "bad_alloc", "paging file is too small", "unable to allocate", "alloc_cpu", "defaultcpuallocator"];
    let out_of_memory = killed || memory_words.iter().any(|w| lower.contains(w));
    let doing = match kind {
        JobKind::Train => "training",
        JobKind::Serve => "loading the model",
        JobKind::Ingest => "reading your files",
        JobKind::Import => "checking the adapter",
        JobKind::Arena => "comparing the models",
        _ => "working",
    };
    let log_path = home.join("logs").join("last-engine-error.log");
    let saved = std::fs::create_dir_all(home.join("logs")).and_then(|_| std::fs::write(&log_path, format!("job {job_id} ({kind:?}) ended with {status_text}\n\n{}\n", stderr_tail.join("\n")))).is_ok();
    let where_ = if saved { format!(" Technical details were saved to {}.", log_path.display()) } else { String::new() };
    if out_of_memory {
        return EngineError {
            code: "OUT_OF_MEMORY".into(),
            message: format!("Your computer ran out of memory while {doing}. Close other programs (web browsers use a lot of memory) and try again. If it keeps happening, restart the computer first, or use Quick run.{where_}"),
            retryable: true,
        };
    }
    let detail = lines.iter().rev().take(4).rev().cloned().collect::<Vec<_>>().join("\n");
    EngineError {
        code: "ENGINE_CRASHED".into(),
        message: format!("Inkport's engine stopped unexpectedly while {doing} ({status_text}). Try again.{where_}{}", if detail.is_empty() { String::new() } else { format!("\n\n{detail}") }),
        retryable: true,
    }
}
/// Next line from a child pipe, decoded leniently (invalid UTF-8 becomes U+FFFD). Cancel-safe:
/// bytes read before a `select!` cancellation stay in `buf` and are continued by the next call.
async fn next_line_lossy<R: tokio::io::AsyncBufRead + Unpin>(reader: &mut R, buf: &mut Vec<u8>) -> std::io::Result<Option<String>> {
    use tokio::io::AsyncBufReadExt;
    let n = reader.read_until(b'\n', buf).await?;
    if n == 0 && buf.is_empty() {
        return Ok(None);
    }
    let line = String::from_utf8_lossy(buf).trim_end_matches(['\r', '\n']).to_string();
    buf.clear();
    Ok(Some(line))
}

