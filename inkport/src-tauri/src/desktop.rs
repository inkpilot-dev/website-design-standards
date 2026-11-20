//! Tauri v2 shell: typed commands + event streaming to the WebView.

use crate::app::{App, AppError, AppResult, CreateProjectRequest, TrainRequest};
use crate::contract::{ApiInfo, HardwareReport, ProjectInfo, RuntimeStatus, Template};
use crate::events::EventSink;
use serde_json::Value;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager, State};

struct TauriSink(AppHandle);

impl EventSink for TauriSink {
    fn emit(&self, channel: &str, payload: &Value) {
        let _ = self.0.emit(channel, payload.clone());
    }
}

pub struct AppState(pub Arc<App>);

#[tauri::command]
async fn detect_hardware(s: State<'_, AppState>) -> AppResult<HardwareReport> {
    s.0.detect_hardware().await
}

#[tauri::command]
async fn runtime_status(s: State<'_, AppState>) -> AppResult<RuntimeStatus> {
    Ok(s.0.runtime_status().await)
}

#[tauri::command]
async fn bootstrap_runtime(s: State<'_, AppState>, download_model: Option<bool>) -> AppResult<RuntimeStatus> {
    s.0.bootstrap_runtime(download_model.unwrap_or(true)).await
}

#[tauri::command]
fn list_templates(s: State<'_, AppState>) -> Vec<Template> {
    s.0.list_templates()
}

#[tauri::command]
async fn create_project(s: State<'_, AppState>, request: CreateProjectRequest) -> AppResult<ProjectInfo> {
    s.0.create_project(request).await
}

#[tauri::command]
fn list_projects(s: State<'_, AppState>) -> AppResult<Vec<ProjectInfo>> {
    s.0.list_projects()
}

#[tauri::command]
fn get_project(s: State<'_, AppState>, project_id: String) -> AppResult<ProjectInfo> {
    s.0.get_project(&project_id)
}

#[tauri::command]
async fn ingest_dataset(s: State<'_, AppState>, project_id: String, paths: Vec<String>, rules: Vec<String>) -> AppResult<Value> {
    s.0.ingest_dataset(&project_id, paths, rules).await
}

#[tauri::command]
async fn preview_items(s: State<'_, AppState>, project_id: String, limit: Option<u32>) -> AppResult<Value> {
    s.0.preview_items(&project_id, limit.unwrap_or(12)).await
}

#[tauri::command]
async fn compile_rules(s: State<'_, AppState>, project_id: Option<String>, texts: Vec<String>) -> AppResult<Value> {
    s.0.compile_rules(project_id.as_deref(), texts).await
}

#[tauri::command]
async fn start_training_job(s: State<'_, AppState>, request: TrainRequest) -> AppResult<String> {
    s.0.start_training_job(request).await
}

#[tauri::command]
async fn cancel_training_job(s: State<'_, AppState>, job_id: String) -> AppResult<()> {
    s.0.cancel_training_job(&job_id).await
}

#[tauri::command]
async fn run_benchmark_arena(s: State<'_, AppState>, project_id: String, prompt: Option<String>, n: Option<u32>) -> AppResult<Value> {
    s.0.run_benchmark_arena(&project_id, prompt, n.unwrap_or(4)).await
}

#[tauri::command]
async fn start_local_api(s: State<'_, AppState>, project_id: String, port: Option<u16>) -> AppResult<ApiInfo> {
    s.0.start_local_api(&project_id, port).await
}

#[tauri::command]
async fn stop_local_api(s: State<'_, AppState>) -> AppResult<()> {
    s.0.stop_local_api().await
}

#[tauri::command]
async fn export_adapter(s: State<'_, AppState>, project_id: String, dest: String) -> AppResult<Value> {
    s.0.export_adapter(&project_id, &dest).await
}

#[tauri::command]
async fn import_adapter(s: State<'_, AppState>, project_id: String, file: String, trust: Option<bool>) -> AppResult<Value> {
    s.0.import_adapter(&project_id, &file, trust.unwrap_or(false), None).await
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let handle = app.handle().clone();
            let resource_dir = app.path().resource_dir().ok();
            let core = App::new(crate::runtime::home_dir(), resource_dir, Arc::new(TauriSink(handle))).map_err(|e| Box::<dyn std::error::Error>::from(e.to_string()))?;
            app.manage(AppState(core));
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(state) = window.try_state::<AppState>() {
                    let core = Arc::clone(&state.0);
                    tauri::async_runtime::block_on(async move { core.shutdown().await });
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            detect_hardware,
            runtime_status,
            bootstrap_runtime,
            list_templates,
            create_project,
            list_projects,
            get_project,
            ingest_dataset,
            preview_items,
            compile_rules,
            start_training_job,
            cancel_training_job,
            run_benchmark_arena,
            start_local_api,
            stop_local_api,
            export_adapter,
            import_adapter
        ])
        .run(tauri::generate_context!())
        .expect("error while running Barnraise");
}

#[allow(dead_code)]
fn _assert_error_serializes(e: AppError) -> String {
    serde_json::to_string(&e).unwrap()
}
