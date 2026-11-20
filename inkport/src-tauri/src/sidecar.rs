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
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{oneshot, watch};

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub python: PathBuf,
    /// Set in development so `python -m engine` resolves to the checkout.
    pub pythonpath: Option<PathBuf>,
    pub home: PathBuf,
    pub stall_secs: u64,
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

    fn command(&self, job_id: &str, spec: &JobSpec) -> Command {
        let mut cmd = Command::new(&self.cfg.python);
        cmd.arg("-u").arg("-m").arg("engine").args(&spec.args);
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
        self.jobs.lock().unwrap().insert(job_id.clone(), Running { pid, cancel: cancel_tx, kind: spec.kind });
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
        self.sink.emit(channel, &v);
    }

    async fn drive(&self, job_id: &str, spec: &JobSpec, child: &mut Child, mut cancel: watch::Receiver<bool>, ready_tx: oneshot::Sender<Value>) -> JobOutcome {
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let _stdin = child.stdin.take(); // keep open for the engine's parent-death watch
        let mut lines = BufReader::new(stdout).lines();
        let mut errs = BufReader::new(stderr).lines();
        let mut ready_tx = Some(ready_tx);
        let mut done: Option<(bool, Value)> = None;
        let mut error: Option<EngineError> = None;
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
                line = lines.next_line(), if stdout_open => match line {
                    Ok(Some(line)) => {
                        last = Instant::now();
                        let trimmed = line.trim();
                        if trimmed.is_empty() { continue; }
                        match serde_json::from_str::<Value>(trimmed) {
                            Ok(v) => {
                                match serde_json::from_value::<EngineEvent>(v.clone()) {
                                    Ok(EngineEvent::Ready { .. }) => { if let Some(tx) = ready_tx.take() { let _ = tx.send(v.clone()); } }
                                    Ok(EngineEvent::Done { ok, result }) => done = Some((ok, result)),
                                    Ok(EngineEvent::Error { code, message, retryable }) => error = Some(EngineError { code, message, retryable }),
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
                line = errs.next_line(), if stderr_open => match line {
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
                        error = Some(EngineError { code: "ENGINE_STALLED".into(), message: format!("The engine stopped responding for {}s; it was stopped and can be resumed.", stall.as_secs()), retryable: true });
                        break true;
                    }
                }
            }
        };
        if cancelled {
            kill_tree(child).await;
            if error.is_none() {
                error = Some(EngineError { code: "CANCELLED".into(), message: "Stopped by the user".into(), retryable: true });
            }
        }
        let status = child.wait().await.ok();
        let had_done = done.is_some();
        let (ok, result) = done.unwrap_or_else(|| (false, json!({})));
        if !ok && error.is_none() {
            let tail = stderr_tail.iter().rev().take(6).rev().cloned().collect::<Vec<_>>().join("\n");
            error = Some(EngineError { code: "ENGINE_CRASHED".into(), message: format!("The engine exited unexpectedly ({}).\n{tail}", status.map(|s| s.to_string()).unwrap_or_default()), retryable: true });
        }
        let outcome = JobOutcome { job_id: job_id.to_string(), ok: ok && !cancelled, result, error };
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
        let _ = std::process::Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).output();
    }
    let _ = child.kill().await;
}
