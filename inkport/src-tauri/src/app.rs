//! The application core shared by the Tauri commands and the headless CLI.

use crate::api_server::{load_or_create_token, Gateway, Upstream, DEFAULT_PORT};
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
    serve_job: Mutex<Option<String>>,
    hardware: Mutex<Option<HardwareReport>>,
}

fn engine_err(o: &JobOutcome) -> AppError {
    o.error.as_ref().map(AppError::from).unwrap_or(AppError { code: "ENGINE_FAILED".into(), message: "The engine reported a failure".into(), retryable: true })
}

impl App {
    pub fn new(home: PathBuf, resource_dir: Option<PathBuf>, sink: Arc<dyn EventSink>) -> Result<Arc<Self>> {
        std::fs::create_dir_all(&home)?;
        let cfg = EngineConfig { python: runtime::python_path(&home), pythonpath: runtime::dev_engine_root(), home: home.clone(), stall_secs: 600 };
        let sup = Arc::new(Supervisor::new(cfg, Arc::clone(&sink)));
        let gateway = Arc::new(Gateway::new(load_or_create_token(&home)?));
        Ok(Arc::new(Self { home, resource_dir, sink, sup, gateway, serve_job: Mutex::new(None), hardware: Mutex::new(None) }))
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

    pub async fn runtime_status(&self) -> RuntimeStatus {
        runtime::status(&self.home, self.pythonpath().as_deref()).await
    }

    pub async fn bootstrap_runtime(&self, download_model: bool) -> AppResult<RuntimeStatus> {
        let hw = self.hw().await?;
        Ok(runtime::bootstrap(&self.home, &hw, self.resource_dir.as_deref(), self.pythonpath().as_deref(), download_model, Arc::clone(&self.sink)).await?)
    }

    pub fn list_templates(&self) -> Vec<Template> {
        templates()
    }

    // ------------------------------------------------------------------ projects ------
    pub async fn create_project(&self, req: CreateProjectRequest) -> AppResult<ProjectInfo> {
        let hw = self.hw().await?;
        let rec = &hw.recommendation;
        let models: Value = serde_json::from_str(TIERS_JSON).map_err(|e| anyhow!(e))?;
        let base_id = req.base_model_id.clone().unwrap_or_else(|| rec.model_id.clone());
        let backend = req.backend.clone().unwrap_or_else(|| rec.backend.clone());
        let base_path = req.base_model_path.clone().map(PathBuf::from).unwrap_or_else(|| runtime::model_dir(&self.home, &base_id, &backend));
        let chat_kwargs = models["models"][&base_id]["chat_kwargs"].clone();
        let chat_kwargs = if chat_kwargs.is_null() { json!({}) } else { chat_kwargs };
        Ok(project::create(&self.home, &req.name, &req.template, &base_id, &base_path, chat_kwargs, Some(&backend), req.quant.as_deref())?)
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
        let model = format!("inkport/{}", info.slug);
        let version = info.active_version.clone().unwrap_or_else(|| "base".into());
        self.gateway.set_upstream(Some(Upstream { base_url: format!("http://127.0.0.1:{wport}"), worker_token, models: vec![model.clone(), format!("{model}@base")], project_id: project_id.into(), version: version.clone() })).await;
        let gport = self.gateway.start(port.unwrap_or(DEFAULT_PORT)).await?;
        let info = ApiInfo { url: format!("http://127.0.0.1:{gport}/v1/chat/completions"), base_url: format!("http://127.0.0.1:{gport}/v1"), token: self.gateway.token.clone(), model: model.clone(), base_model: format!("{model}@base"), version };
        self.sink.emit("api://status", &json!({"type": "api", "running": true, "info": info}));
        Ok(info)
    }

    async fn stop_worker(&self) {
        if let Some(id) = self.serve_job.lock().await.take() {
            let _ = self.sup.cancel(&id).await;
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
        self.sup.cancel_all().await;
        self.gateway.stop().await;
    }
}

pub fn validate_path_exists(p: &str) -> Result<()> {
    if !Path::new(p).exists() {
        bail!("BAD_INPUT: {p} does not exist");
    }
    Ok(())
}
