//! Tauri v2 shell: typed commands + event streaming to the WebView.

use crate::app::{ApiStatus, App, AppError, AppResult, CreateProjectRequest, TrainRequest};
use crate::bootstrap::PreflightReport;
use crate::contract::{ApiInfo, HardwareReport, ProjectInfo, RuntimeStatus, Template};
use crate::packs::PackState;
use crate::events::EventSink;
use crate::sidecar::RunningJob;
use serde_json::Value;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

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

/// Launch-time readiness (integrity + warm-up + accelerators). Never downloads.
#[tauri::command]
async fn preflight(s: State<'_, AppState>, force_full: Option<bool>) -> AppResult<PreflightReport> {
    Ok(s.0.preflight(force_full.unwrap_or(false)).await)
}

#[tauri::command]
fn list_models(s: State<'_, AppState>) -> Vec<Value> {
    s.0.list_models()
}

#[tauri::command]
fn list_packs(s: State<'_, AppState>) -> Vec<PackState> {
    s.0.list_packs()
}

/// Optional, user-initiated download of an add-on pack; progress on `packs://progress`.
/// Async so it runs on Tauri's tokio runtime, where the background download task is spawned.
#[tauri::command]
async fn install_pack(s: State<'_, AppState>, pack_id: String) -> AppResult<()> {
    s.0.install_pack(&pack_id)
}

#[tauri::command]
fn cancel_pack(s: State<'_, AppState>, pack_id: String) -> bool {
    s.0.cancel_pack(&pack_id)
}

#[tauri::command]
fn remove_pack(s: State<'_, AppState>, pack_id: String) -> AppResult<()> {
    s.0.remove_pack(&pack_id)
}

#[tauri::command]
fn sample_data(s: State<'_, AppState>) -> Option<String> {
    s.0.sample_data()
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

/// The training run in progress (if any) with the events needed to redraw its screen, so a window
/// that was showing another step, or another project, can pick the run up again.
#[tauri::command]
fn training_status(s: State<'_, AppState>) -> Option<RunningJob> {
    s.0.training_status()
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

/// Whether the local API is up and for which project (also notices a model process that died).
#[tauri::command]
async fn api_status(s: State<'_, AppState>) -> AppResult<ApiStatus> {
    Ok(s.0.api_status().await)
}

/// One chat turn through the local API, the same way another app on this computer would call it.
#[tauri::command]
async fn chat_local_api(s: State<'_, AppState>, messages: Value, use_base: Option<bool>) -> AppResult<String> {
    s.0.chat_local_api(messages, use_base.unwrap_or(false)).await
}

#[tauri::command]
async fn export_adapter(s: State<'_, AppState>, project_id: String, dest: String) -> AppResult<Value> {
    s.0.export_adapter(&project_id, &dest).await
}

#[tauri::command]
async fn import_adapter(s: State<'_, AppState>, project_id: String, file: String, trust: Option<bool>) -> AppResult<Value> {
    s.0.import_adapter(&project_id, &file, trust.unwrap_or(false), None).await
}

/// Record a crash where it can be found afterwards. The app has no console on Windows, so without
/// this a crash leaves no trace at all; with it there is `logs/app-crash.log` in the data folder.
fn record_crashes() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let dir = crate::runtime::home_dir().join("logs");
        let _ = std::fs::create_dir_all(&dir);
        let when = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let thread = std::thread::current().name().unwrap_or("unnamed").to_string();
        let text = format!("--- Inkport {} crashed (unix time {when}, thread {thread}) ---\n{info}\n{}\n\n", env!("CARGO_PKG_VERSION"), std::backtrace::Backtrace::force_capture());
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("app-crash.log")) {
            use std::io::Write;
            let _ = f.write_all(text.as_bytes());
        }
        previous(info);
    }));
}

pub fn run() {
    record_crashes();
    let app = tauri::Builder::default()
        // A second launch (double-clicking the icon again while the first window is busy) brings
        // the existing window forward instead of starting a second copy that would fight over the
        // same projects and the same port.
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.unminimize();
                let _ = w.show();
                let _ = w.set_focus();
            }
        }))
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let handle = app.handle().clone();
            let resource_dir = app.path().resource_dir().ok();
            let home = crate::runtime::home_dir();
            match App::new(home.clone(), resource_dir, Arc::new(TauriSink(handle))) {
                Ok(core) => {
                    app.manage(AppState(core));
                    // Logging out, shutting the computer down or `kill` ask the app to end with a
                    // signal. Leave the way the window's close button does, so that training and
                    // the answer engine are stopped instead of being left behind.
                    #[cfg(unix)]
                    {
                        let handle = app.handle().clone();
                        tauri::async_runtime::spawn(async move {
                            use tokio::signal::unix::{signal, SignalKind};
                            let (Ok(mut term), Ok(mut int), Ok(mut hup)) = (signal(SignalKind::terminate()), signal(SignalKind::interrupt()), signal(SignalKind::hangup())) else { return };
                            tokio::select! {
                                _ = term.recv() => {}
                                _ = int.recv() => {}
                                _ = hup.recv() => {}
                            }
                            handle.exit(0);
                        });
                    }
                }
                Err(e) => {
                    // Without its data folder the app cannot do anything; say why instead of
                    // showing a window whose every button fails (or no window at all).
                    if let Some(w) = app.get_webview_window("main") {
                        let _ = w.hide();
                    }
                    app.dialog()
                        .message(format!("Inkport could not start.\n\n{e:#}\n\nInkport keeps your projects in:\n{}\n\nMake sure that folder exists and can be written to, then open Inkport again.", home.display()))
                        .title("Inkport")
                        .kind(MessageDialogKind::Error)
                        .show(|_| std::process::exit(1));
                }
            }
            Ok(())
        })
        .on_window_event(|window, event| match event {
            tauri::WindowEvent::CloseRequested { api, .. } => {
                // Training can take hours; closing the window by accident must not silently end it.
                let training = window.try_state::<AppState>().map(|s| s.0.training_status().is_some()).unwrap_or(false);
                if training {
                    api.prevent_close();
                    let w = window.clone();
                    window
                        .app_handle()
                        .dialog()
                        .message("Training is still running. If you close Inkport now, training stops and this run's progress is lost.")
                        .title("Close Inkport?")
                        .kind(MessageDialogKind::Warning)
                        .buttons(MessageDialogButtons::OkCancelCustom("Stop training and close".into(), "Keep training".into()))
                        .show(move |close| {
                            if close {
                                let _ = w.destroy();
                            }
                        });
                }
            }
            tauri::WindowEvent::Destroyed => {
                if let Some(state) = window.try_state::<AppState>() {
                    let core = Arc::clone(&state.0);
                    tauri::async_runtime::block_on(async move { core.shutdown().await });
                }
            }
            _ => {}
        })
        .invoke_handler(tauri::generate_handler![
            detect_hardware,
            preflight,
            list_models,
            list_packs,
            install_pack,
            cancel_pack,
            remove_pack,
            sample_data,
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
            training_status,
            run_benchmark_arena,
            start_local_api,
            stop_local_api,
            api_status,
            chat_local_api,
            export_adapter,
            import_adapter
        ])
        .build(tauri::generate_context!())
        .expect("error while starting Inkport");
    app.run(|handle, event| {
        // Quitting from the menu or the dock (macOS) does not always destroy the window first;
        // stopping the helpers here as well is harmless (it does nothing the second time).
        if let tauri::RunEvent::Exit = event {
            if let Some(state) = handle.try_state::<AppState>() {
                let core = Arc::clone(&state.0);
                tauri::async_runtime::block_on(async move { core.shutdown().await });
            }
        }
    });
}

#[allow(dead_code)]
fn _assert_error_serializes(e: AppError) -> String {
    serde_json::to_string(&e).unwrap()
}
