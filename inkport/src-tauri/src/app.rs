//! The application core shared by the Tauri commands and the headless CLI.

use crate::api_server::{load_or_create_token, Gateway, Upstream, UpstreamKind, DEFAULT_PORT};
use crate::bootstrap::{self, Bundle, PreflightReport};
use crate::packs::{PackState, Packs};
use crate::contract::{templates, ApiInfo, HardwareReport, JobKind, JobOutcome, ProjectInfo, RuntimeStatus, Template, TIERS_JSON};
use crate::events::EventSink;
use crate::hardware;
use crate::project;
use crate::runtime;
use crate::sidecar::{EngineConfig, JobSpec, Supervisor};
use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

impl From<anyhow::Error> for AppError {
    fn from(e: anyhow::Error) -> Self {
        let msg = e.to_string();
        let (code, message) = match msg.split_once(": ") {
            Some((c, m)) if c.chars().all(|ch| ch.is_ascii_uppercase() || ch == '_') && c.len() > 2 => (c.to_string(), m.to_string()),
            _ => ("ERROR".to_string(), format!("{e:#}")),
        };
        AppError { code, message, retryable: false }
    }
}

impl From<&crate::contract::EngineError> for AppError {
    fn from(e: &crate::contract::EngineError) -> Self {
        AppError { code: e.code.clone(), message: e.message.clone(), retryable: e.retryable }
    }
}

pub type AppResult<T> = std::result::Result<T, AppError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateProjectRequest {
    pub name: String,
    pub template: String,
    /// Optional overrides (tests, power users). Defaults come from the hardware recommendation.
    pub base_model_id: Option<String>,
    pub base_model_path: Option<String>,
    /// GGUF file inside base_model_path (when the explicit base is a GGUF model).
    #[serde(default)]
    pub base_model_gguf: Option<String>,
    pub backend: Option<String>,
    pub quant: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainRequest {
    pub project_id: String,
    /// "standard" | "fast" | "smoke"
    pub preset: Option<String>,
    pub resume_run_id: Option<String>,
    pub overrides: Option<Value>,
}

pub struct App {
    pub home: PathBuf,
    pub resource_dir: Option<PathBuf>,
    pub sink: Arc<dyn EventSink>,
    pub sup: Arc<Supervisor>,
    pub gateway: Arc<Gateway>,
    pub bundle: Option<Bundle>,
    bundle_error: Option<String>,
    pub packs: Arc<Packs>,
    serve_job: Mutex<Option<String>>,
    llama: Mutex<Option<tokio::process::Child>>,
    hardware: Mutex<Option<HardwareReport>>,
    preflight_cache: Mutex<Option<PreflightReport>>,
}

fn engine_err(o: &JobOutcome) -> AppError {
    o.error.as_ref().map(AppError::from).unwrap_or(AppError { code: "ENGINE_FAILED".into(), message: "The engine reported a failure".into(), retryable: true })
}

impl App {
    pub fn new(home: PathBuf, resource_dir: Option<PathBuf>, sink: Arc<dyn EventSink>) -> Result<Arc<Self>> {
        std::fs::create_dir_all(&home)?;
        // Installed app: everything comes from the bundled payload. Development: INKPORT_PYTHON or
        // the checkout. A damaged installation is reported by preflight, never "fixed" by downloading.
        let (bundle, bundle_error) = match bootstrap::locate(resource_dir.as_deref()) {
            Some(dir) => match bootstrap::load(&dir) {
                Ok(b) => (Some(b), None),
                Err(e) => (None, Some(format!("{e:#}"))),
            },
            None => (None, None),
        };
        let cfg = match &bundle {
            Some(b) => {
                let (env, env_remove) = b.engine_env(&home);
                EngineConfig { python: b.python(), pythonpath: None, home: home.clone(), stall_secs: 600, python_flags: vec!["-I".into(), "-B".into()], env, env_remove }
            }
            None => EngineConfig { python: runtime::python_path(&home), pythonpath: runtime::dev_engine_root(), home: home.clone(), stall_secs: 600, python_flags: vec![], env: vec![], env_remove: vec![] },
        };
        let catalog = bundle.as_ref().map(|b| b.packs_catalog()).or_else(|| Some(Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("packaging").join("packs").join("catalog.json")));
        let packs = Arc::new(Packs::new(&home, catalog, Arc::clone(&sink))?);
        let sup = Arc::new(Supervisor::new(cfg, Arc::clone(&sink)));
        let gateway = Arc::new(Gateway::new(load_or_create_token(&home)?));
        Ok(Arc::new(Self { home, resource_dir, sink, sup, gateway, bundle, bundle_error, packs, serve_job: Mutex::new(None), llama: Mutex::new(None), hardware: Mutex::new(None), preflight_cache: Mutex::new(None) }))
    }

    fn pythonpath(&self) -> Option<PathBuf> {
        self.sup.cfg.pythonpath.clone()
    }

    // ------------------------------------------------------------------ system --------
    pub async fn detect_hardware(&self) -> AppResult<HardwareReport> {
        let home = self.home.clone();
        let report = tokio::task::spawn_blocking(move || hardware::detect(&home)).await.map_err(|e| anyhow!(e))?;
        std::fs::write(self.home.join("hardware.json"), serde_json::to_vec_pretty(&report).map_err(|e| anyhow!(e))?).map_err(|e| anyhow!(e))?;
        *self.hardware.lock().await = Some(report.clone());
        Ok(report)
    }

    async fn hw(&self) -> AppResult<HardwareReport> {
        if let Some(h) = self.hardware.lock().await.clone() {
            return Ok(h);
        }
        self.detect_hardware().await
    }

    /// Launch-time readiness: integrity check + engine warm-up + accelerator detection.
    /// Installed builds never download anything here.
    pub async fn preflight(&self, force_full: bool) -> PreflightReport {
        if !force_full {
            if let Some(r) = self.preflight_cache.lock().await.clone() {
                return r;
            }
        }
        let report = match (&self.bundle, &self.bundle_error) {
            (Some(b), _) => {
                let sink = Arc::clone(&self.sink);
                let progress: Arc<dyn Fn(&str, u64, u64) + Send + Sync> = Arc::new(move |phase: &str, done: u64, total: u64| {
                    let pct = if total > 0 { done as f64 * 100.0 / total as f64 } else { 0.0 };
                    sink.emit("runtime://progress", &json!({"type": "preflight", "phase": phase, "done": done, "total": total, "pct": pct}));
                });
                bootstrap::preflight(b, &self.home, force_full, progress).await
            }
            (None, Some(err)) => PreflightReport { ok: false, bundled: true, offline_core: true, platform: bootstrap::current_platform().into(), problems: vec![format!("The installation is damaged: {err}. Reinstall Inkport from the same installer; your projects are kept.")], ..Default::default() },
            (None, None) => {
                let st = runtime::status(&self.home, self.pythonpath().as_deref()).await;
                PreflightReport { ok: st.ready, bundled: false, offline_core: false, platform: bootstrap::current_platform().into(), engine_version: st.engine_version.clone(), python: st.python.clone(), training_accelerator: bootstrap::training_accelerator(&st.backends.iter().map(|(k, v)| (k.clone(), *v)).collect()), inference_devices: vec![], problems: if st.ready { vec![] } else { vec![format!("Development build without a bundled runtime: {}", st.message)] }, ..Default::default() }
            }
        };
        *self.preflight_cache.lock().await = Some(report.clone());
        report
    }

    pub async fn runtime_status(&self) -> RuntimeStatus {
        if self.bundle.is_some() || self.bundle_error.is_some() {
            let p = self.preflight(false).await;
            return RuntimeStatus { ready: p.ok, python: p.python, engine_version: p.engine_version, backends: Default::default(), home: self.home.display().to_string(), message: if p.ok { "Ready (bundled, works offline)".into() } else { p.problems.join("; ") } };
        }
        runtime::status(&self.home, self.pythonpath().as_deref()).await
    }

    /// Developer-only online setup. Installed builds already contain the runtime, so this is a no-op there.
    pub async fn bootstrap_runtime(&self, download_model: bool) -> AppResult<RuntimeStatus> {
        if self.bundle.is_some() {
            return Ok(self.runtime_status().await);
        }
        let hw = self.hw().await?;
        Ok(runtime::bootstrap(&self.home, &hw, self.resource_dir.as_deref(), self.pythonpath().as_deref(), download_model, Arc::clone(&self.sink)).await?)
    }

    // ------------------------------------------------------------------ models & packs -
    /// Base models usable right now: the bundled starter plus installed model packs.
    pub fn list_models(&self) -> Vec<Value> {
        let mut out = vec![];
        if let Some(b) = &self.bundle {
            out.push(b.starter_base_model());
        }
        out.extend(self.packs.installed_models());
        out
    }

    pub fn list_packs(&self) -> Vec<PackState> {
        self.packs.list()
    }

    /// Starts a background download; progress arrives on `packs://progress`.
    pub fn install_pack(&self, id: &str) -> AppResult<()> {
        if !self.packs.catalog().iter().any(|p| p.id == id) {
            return Err(AppError { code: "BAD_INPUT".into(), message: format!("unknown pack {id}"), retryable: false });
        }
        let packs = Arc::clone(&self.packs);
        let id = id.to_string();
        tokio::spawn(async move {
            let _ = packs.install(&id).await;
        });
        Ok(())
    }

    pub async fn install_pack_blocking(&self, id: &str) -> AppResult<Value> {
        let dir = self.packs.install(id).await?;
        Ok(json!({"pack": id, "path": dir}))
    }

    pub fn cancel_pack(&self, id: &str) -> bool {
        self.packs.cancel(id)
    }

    pub fn remove_pack(&self, id: &str) -> AppResult<()> {
        Ok(self.packs.remove(id)?)
    }

    /// Bundled sample files (a ready-made first project that works offline).
    pub fn sample_data(&self) -> Option<String> {
        self.bundle.as_ref().map(|b| b.samples_dir()).filter(|p| p.is_dir()).map(|p| p.display().to_string())
    }

    pub fn list_templates(&self) -> Vec<Template> {
        templates()
    }

    // ------------------------------------------------------------------ projects ------
    pub async fn create_project(&self, req: CreateProjectRequest) -> AppResult<ProjectInfo> {
        let hw = self.hw().await?;
        let rec = &hw.recommendation;
        let models: Value = serde_json::from_str(TIERS_JSON).map_err(|e| anyhow!(e))?;
        let base: Value = if let Some(path) = &req.base_model_path {
            let id = req.base_model_id.clone().unwrap_or_else(|| "custom".into());
            let ck = models["models"][&id]["chat_kwargs"].clone();
            let mut v = json!({"id": id, "path": PathBuf::from(path).display().to_string(), "chat_kwargs": if ck.is_null() { json!({}) } else { ck }});
            if let Some(g) = &req.base_model_gguf {
                v["gguf"] = json!(g);
                v["format"] = json!("gguf");
            }
            v
        } else if let Some(m) = req.base_model_id.as_ref().and_then(|id| self.list_models().into_iter().find(|m| m["id"].as_str() == Some(id.as_str()))) {
            m
        } else if let Some(b) = &self.bundle {
            b.starter_base_model()
        } else {
            let id = req.base_model_id.clone().unwrap_or_else(|| rec.model_id.clone());
            let ck = models["models"][&id]["chat_kwargs"].clone();
            json!({"id": id, "path": runtime::model_dir(&self.home, &id, &rec.backend).display().to_string(), "chat_kwargs": if ck.is_null() { json!({}) } else { ck }})
        };
        let backend = req.backend.clone().or_else(|| base["backend"].as_str().map(String::from)).unwrap_or_else(|| rec.backend.clone());
        // GGUF / MLX bases pick their own precision; only explicit requests pin it.
        let quant = req.quant.clone();
        Ok(project::create(&self.home, &req.name, &req.template, base, Some(&backend), quant.as_deref())?)
    }

    pub fn list_projects(&self) -> AppResult<Vec<ProjectInfo>> {
        Ok(project::list(&self.home)?)
    }

    pub fn get_project(&self, id: &str) -> AppResult<ProjectInfo> {
        Ok(project::read(&project::project_dir(&self.home, id)?)?)
    }

    fn dir(&self, id: &str) -> AppResult<String> {
        Ok(project::project_dir(&self.home, id)?.display().to_string())
    }

    async fn run_engine(&self, kind: JobKind, project_id: Option<&str>, args: Vec<String>) -> AppResult<Value> {
        let out = self.sup.run(JobSpec { kind, args, project_id: project_id.map(String::from), env: vec![] }).await?;
        if out.ok {
            Ok(out.result)
        } else {
            Err(engine_err(&out))
        }
    }

    // ------------------------------------------------------------------ data ----------
    pub async fn ingest_dataset(&self, project_id: &str, paths: Vec<String>, rules: Vec<String>) -> AppResult<Value> {
        if paths.is_empty() {
            return Err(AppError { code: "BAD_INPUT".into(), message: "Add at least one file".into(), retryable: false });
        }
        let mut args = vec!["ingest".into(), "--project".into(), self.dir(project_id)?, "--inputs".into()];
        args.extend(paths);
        args.extend(["--rules".into(), serde_json::to_string(&rules).unwrap()]);
        self.run_engine(JobKind::Ingest, Some(project_id), args).await
    }

    pub async fn preview_items(&self, project_id: &str, limit: u32) -> AppResult<Value> {
        self.run_engine(JobKind::Preview, Some(project_id), vec!["preview".into(), "--project".into(), self.dir(project_id)?, "--limit".into(), limit.to_string()]).await
    }

    pub async fn compile_rules(&self, project_id: Option<&str>, texts: Vec<String>) -> AppResult<Value> {
        let mut args = vec!["rules".into(), "--texts".into(), serde_json::to_string(&texts).unwrap()];
        if let Some(p) = project_id {
            args.extend(["--project".into(), self.dir(p)?]);
        }
        self.run_engine(JobKind::Rules, project_id, args).await
    }

    // ------------------------------------------------------------------ training ------
    fn hardware_args(&self, preset: &str) -> Vec<String> {
        let mut a = vec!["--preset".to_string(), preset.to_string()];
        let hw = self.home.join("hardware.json");
        if hw.is_file() {
            a.extend(["--hardware".into(), hw.display().to_string()]);
        }
        a
    }

    pub async fn start_training_job(&self, req: TrainRequest) -> AppResult<String> {
        if self.sup.running().iter().any(|(_, k)| *k == JobKind::Train) {
            return Err(AppError { code: "BUSY".into(), message: "A training job is already running".into(), retryable: true });
        }
        if self.hardware.lock().await.is_none() {
            self.detect_hardware().await?;
        }
        let mut args = vec!["train".into(), "--project".into(), self.dir(&req.project_id)?];
        if let Some(r) = &req.resume_run_id {
            args.extend(["--resume".into(), r.clone()]);
        }
        args.extend(self.hardware_args(req.preset.as_deref().unwrap_or("standard")));
        if let Some(o) = &req.overrides {
            args.extend(["--overrides".into(), o.to_string()]);
        }
        let sp = self.sup.spawn(JobSpec { kind: JobKind::Train, args, project_id: Some(req.project_id.clone()), env: vec![] })?;
        Ok(sp.job_id)
    }

    /// Train and wait (CLI).
    pub async fn train_blocking(&self, req: TrainRequest) -> AppResult<Value> {
        let mut args = vec!["train".into(), "--project".into(), self.dir(&req.project_id)?];
        if let Some(r) = &req.resume_run_id {
            args.extend(["--resume".into(), r.clone()]);
        }
        args.extend(self.hardware_args(req.preset.as_deref().unwrap_or("standard")));
        if let Some(o) = &req.overrides {
            args.extend(["--overrides".into(), o.to_string()]);
        }
        self.run_engine(JobKind::Train, Some(&req.project_id), args).await
    }

    pub async fn cancel_training_job(&self, job_id: &str) -> AppResult<()> {
        Ok(self.sup.cancel(job_id).await?)
    }

    pub async fn run_benchmark_arena(&self, project_id: &str, prompt: Option<String>, n: u32) -> AppResult<Value> {
        let mut args = vec!["arena".into(), "--project".into(), self.dir(project_id)?, "--n".into(), n.to_string()];
        if let Some(p) = prompt.filter(|p| !p.trim().is_empty()) {
            args.extend(["--prompt".into(), p]);
        }
        self.run_engine(JobKind::Arena, Some(project_id), args).await
    }

    // ------------------------------------------------------------------ serving -------
    pub async fn start_local_api(&self, project_id: &str, port: Option<u16>) -> AppResult<ApiInfo> {
        self.stop_worker().await;
        let info = self.get_project(project_id)?;
        let worker_token = format!("wk_{}", uuid::Uuid::new_v4().simple());
        let model = format!("inkport/{}", info.slug);
        let version = info.active_version.clone().unwrap_or_else(|| "base".into());
        match self.try_start_llama(&info, &worker_token).await {
            Ok(Some((lport, has_lora))) => {
                let default_system = default_system_prompt(Path::new(&info.dir));
                self.gateway.set_upstream(Some(Upstream { base_url: format!("http://127.0.0.1:{lport}"), worker_token, models: vec![model.clone(), format!("{model}@base")], project_id: project_id.into(), version: version.clone(), kind: UpstreamKind::Llama { has_lora }, default_system })).await;
                let gport = self.gateway.start(port.unwrap_or(DEFAULT_PORT)).await?;
                let devices = self.preflight_cache.lock().await.as_ref().map(|p| p.inference_devices.join(", ")).unwrap_or_default();
                let info = ApiInfo { url: format!("http://127.0.0.1:{gport}/v1/chat/completions"), base_url: format!("http://127.0.0.1:{gport}/v1"), token: self.gateway.token.clone(), model: model.clone(), base_model: format!("{model}@base"), version, engine: format!("llama.cpp{}", if devices.is_empty() { String::new() } else { format!(" ({devices})") }) };
                self.sink.emit("api://status", &json!({"type": "api", "running": true, "info": info}));
                return Ok(info);
            }
            Ok(None) => {}
            Err(e) => self.sink.emit("training://log", &json!({"type": "log", "level": "warn", "message": format!("llama.cpp serving unavailable, using the Python worker instead: {e}")})),
        }
        let sp = self.sup.spawn(JobSpec {
            kind: JobKind::Serve,
            args: vec!["serve".into(), "--project".into(), info.dir.clone(), "--port".into(), "0".into()],
            project_id: Some(project_id.to_string()),
            env: vec![("INKPORT_WORKER_TOKEN".into(), worker_token.clone())],
        })?;
        *self.serve_job.lock().await = Some(sp.job_id.clone());
        let ready = tokio::select! {
            r = sp.ready => r.map_err(|_| anyhow!("WORKER_FAILED: the model worker exited before it was ready (see the log)"))?,
            _ = tokio::time::sleep(std::time::Duration::from_secs(900)) => return Err(anyhow!("WORKER_TIMEOUT: the model took too long to load").into()),
        };
        let wport = ready["port"].as_u64().ok_or_else(|| anyhow!("worker did not report a port"))? as u16;
        self.gateway.set_upstream(Some(Upstream { base_url: format!("http://127.0.0.1:{wport}"), worker_token, models: vec![model.clone(), format!("{model}@base")], project_id: project_id.into(), version: version.clone(), kind: UpstreamKind::Python, default_system: None })).await;
        let gport = self.gateway.start(port.unwrap_or(DEFAULT_PORT)).await?;
        let engine = if info.base_model["format"] == "mlx" { "MLX (Apple GPU)" } else { "PyTorch" };
        let info = ApiInfo { url: format!("http://127.0.0.1:{gport}/v1/chat/completions"), base_url: format!("http://127.0.0.1:{gport}/v1"), token: self.gateway.token.clone(), model: model.clone(), base_model: format!("{model}@base"), version, engine: engine.into() };
        self.sink.emit("api://status", &json!({"type": "api", "running": true, "info": info}));
        Ok(info)
    }

    /// Serve with the bundled llama.cpp when the base is GGUF and the active version has a GGUF
    /// adapter (or there is no trained version yet). Returns (port, adapter_loaded).
    async fn try_start_llama(&self, info: &ProjectInfo, key: &str) -> Result<Option<(u16, bool)>> {
        let Some(b) = &self.bundle else { return Ok(None) };
        let Some(server) = b.llama_server() else { return Ok(None) };
        let Some(gguf) = info.base_model["gguf"].as_str() else { return Ok(None) };
        let base = PathBuf::from(info.base_model["path"].as_str().unwrap_or_default()).join(gguf);
        let adapter = info.active_version.as_ref().map(|v| Path::new(&info.dir).join("versions").join(v).join("gguf").join("adapter-f16.gguf"));
        if matches!(&adapter, Some(a) if !a.is_file()) {
            return Ok(None); // trained version without a GGUF adapter: the Python worker serves it
        }
        let port = std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
        let log_dir = self.home.join("logs");
        std::fs::create_dir_all(&log_dir)?;
        let log = std::fs::File::create(log_dir.join("llama-server.log"))?;
        let mut cmd = tokio::process::Command::new(&server);
        cmd.args(["--model", &base.display().to_string(), "--host", "127.0.0.1", "--port", &port.to_string(), "--api-key", key, "--no-webui", "--jinja", "--ctx-size", "4096", "--parallel", "2", "--alias", &format!("inkport/{}", info.slug)]);
        if let Some(a) = &adapter {
            cmd.args(["--lora", &a.display().to_string()]);
        }
        let ck = &info.base_model["chat_kwargs"];
        if ck.as_object().map(|o| !o.is_empty()).unwrap_or(false) {
            cmd.args(["--chat-template-kwargs", &ck.to_string()]);
        }
        cmd.envs(b.llama_env()).stdin(std::process::Stdio::null()).stdout(log.try_clone()?).stderr(log).kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);
        #[cfg(target_os = "linux")]
        // SAFETY: prctl is async-signal-safe; the server dies if Inkport is killed outright.
        unsafe {
            cmd.pre_exec(|| {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                Ok(())
            });
        }
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000);
        let mut child = cmd.spawn().with_context_msg("could not start the bundled llama-server")?;
        let client = reqwest::Client::builder().no_proxy().build()?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(180);
        loop {
            if let Some(status) = child.try_wait()? {
                bail!("llama-server exited during startup ({status}); see {}", log_dir.join("llama-server.log").display());
            }
            if let Ok(r) = client.get(format!("http://127.0.0.1:{port}/health")).send().await {
                if r.status().is_success() {
                    break;
                }
            }
            if std::time::Instant::now() > deadline {
                let _ = child.kill().await;
                bail!("llama-server did not become ready in time");
            }
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        }
        *self.llama.lock().await = Some(child);
        Ok(Some((port, adapter.is_some())))
    }

    async fn stop_worker(&self) {
        if let Some(id) = self.serve_job.lock().await.take() {
            let _ = self.sup.cancel(&id).await;
        }
        if let Some(mut child) = self.llama.lock().await.take() {
            #[cfg(unix)]
            if let Some(pid) = child.id() {
                // SAFETY: our own child's process group (process_group(0) above).
                unsafe {
                    libc::killpg(pid as i32, libc::SIGTERM);
                }
            }
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), child.wait()).await;
            let _ = child.kill().await;
        }
        self.gateway.set_upstream(None).await;
    }

    pub async fn stop_local_api(&self) -> AppResult<()> {
        self.stop_worker().await;
        self.gateway.stop().await;
        self.sink.emit("api://status", &json!({"type": "api", "running": false}));
        Ok(())
    }

    // ------------------------------------------------------------------ adapters ------
    pub async fn export_adapter(&self, project_id: &str, dest: &str) -> AppResult<Value> {
        self.run_engine(JobKind::Export, Some(project_id), vec!["export".into(), "--project".into(), self.dir(project_id)?, "--out".into(), dest.into()]).await
    }

    pub async fn import_adapter(&self, project_id: &str, file: &str, trust: bool, preset: Option<String>) -> AppResult<Value> {
        let mut args = vec!["import".into(), "--project".into(), self.dir(project_id)?, "--file".into(), file.into()];
        if trust {
            args.push("--trust".into());
        }
        args.extend(self.hardware_args(preset.as_deref().unwrap_or("standard")));
        self.run_engine(JobKind::Import, Some(project_id), args).await
    }

    pub async fn shutdown(&self) {
        self.stop_worker().await;
        self.sup.cancel_all().await;
        self.gateway.stop().await;
    }
}

/// The system prompt the project's examples were trained with (inserted when a client sends none).
pub fn default_system_prompt(project_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(project_dir.join("data").join("train.jsonl")).ok()?;
    let first: Value = serde_json::from_str(text.lines().next()?).ok()?;
    first["messages"].as_array()?.iter().find(|m| m["role"] == "system").and_then(|m| m["content"].as_str()).map(String::from)
}

trait ContextMsg<T> {
    fn with_context_msg(self, msg: &str) -> Result<T>;
}

impl<T> ContextMsg<T> for std::io::Result<T> {
    fn with_context_msg(self, msg: &str) -> Result<T> {
        self.map_err(|e| anyhow!("{msg}: {e}"))
    }
}

pub fn validate_path_exists(p: &str) -> Result<()> {
    if !Path::new(p).exists() {
        bail!("BAD_INPUT: {p} does not exist");
    }
    Ok(())
}
