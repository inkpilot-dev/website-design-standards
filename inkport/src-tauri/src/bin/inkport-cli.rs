//! Headless Inkport: the same supervisor, engine and gateway as the desktop app, driven
//! from a terminal. Used by scripts/verify_e2e.sh and handy for servers and CI.
//! Prints the final result as one JSON line prefixed with `RESULT `.

use inkport_lib::app::{App, AppError, CreateProjectRequest, TrainRequest};
use inkport_lib::events::StdoutSink;
use inkport_lib::runtime;
use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "inkport-cli", version, about = "Train, verify and serve specialist models locally")]
struct Cli {
    /// Print every raw engine event
    #[arg(long, global = true)]
    raw: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Detect GPU / Apple Silicon / CPU / RAM / disk and the recommended model
    Hardware,
    /// Show whether the Python engine is installed
    Status,
    /// Pre-flight: verify the bundled files, warm up the engine, detect accelerators (offline)
    Doctor {
        /// Hash every bundled file instead of the quick check
        #[arg(long)]
        full: bool,
    },
    /// Base models available right now (bundled starter + installed packs)
    Models,
    /// Optional add-on packs and their state
    Packs,
    /// Download, verify and install an optional pack (resume-safe)
    InstallPack {
        id: String,
    },
    /// Print the folder with the bundled sample data
    Samples,
    /// Install the private Python runtime and the engine
    Bootstrap {
        #[arg(long)]
        download_model: bool,
    },
    /// List the starter templates
    Templates,
    /// List projects
    Projects,
    /// Create a project
    CreateProject {
        #[arg(long)]
        name: String,
        #[arg(long)]
        template: String,
        #[arg(long)]
        base_id: Option<String>,
        #[arg(long)]
        base_path: Option<String>,
        /// GGUF file inside --base-path
        #[arg(long)]
        base_gguf: Option<String>,
        #[arg(long)]
        backend: Option<String>,
        #[arg(long)]
        quant: Option<String>,
    },
    /// Turn files into a practice exam with a hidden test set
    Ingest {
        #[arg(long)]
        project: String,
        #[arg(long, num_args = 1..)]
        inputs: Vec<String>,
        /// JSON list of plain-English rules
        #[arg(long, default_value = "[]")]
        rules: String,
    },
    /// Show sample practice questions
    Preview {
        #[arg(long)]
        project: String,
    },
    /// Train, run the before-vs-after gate, promote or roll back
    Train {
        #[arg(long)]
        project: String,
        #[arg(long, default_value = "standard")]
        preset: String,
        #[arg(long)]
        resume: Option<String>,
        #[arg(long)]
        overrides: Option<String>,
    },
    /// Original vs trained, side by side
    Arena {
        #[arg(long)]
        project: String,
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long, default_value_t = 4)]
        n: u32,
    },
    /// Serve the active version at http://127.0.0.1:<port>/v1 until Ctrl+C
    Serve {
        #[arg(long)]
        project: String,
        #[arg(long, default_value_t = 11435)]
        port: u16,
    },
    /// Start the local API, exercise it over HTTP (auth, models, chat, @base, streaming), stop it
    ApiSmoke {
        #[arg(long)]
        project: String,
        #[arg(long, default_value_t = 11541)]
        port: u16,
        #[arg(long, default_value = "Paid $42 at Shell for the delivery van")]
        prompt: String,
    },
    /// Export a signed .inkpatch
    Export {
        #[arg(long)]
        project: String,
        #[arg(long)]
        out: String,
    },
    /// Verify a .inkpatch (signature + SHA-256 + tensors) and gate it before activation
    Import {
        #[arg(long)]
        project: String,
        #[arg(long)]
        file: String,
        #[arg(long)]
        trust: bool,
        #[arg(long, default_value = "standard")]
        preset: String,
    },
}

/// Ctrl+C everywhere, plus SIGTERM on Unix (service managers, `kill`), so children are cleaned up.
async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("signal handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
}

/// HTTP checks against the running gateway; uses no external tools (works on minimal systems).
async fn api_smoke(info: &inkport_lib::contract::ApiInfo, prompt: &str) -> anyhow::Result<Value> {
    use anyhow::{bail, Context};
    let c = reqwest::Client::builder().no_proxy().build()?;
    let unauth = c.get(format!("{}/models", info.base_url)).send().await?.status().as_u16();
    if unauth != 401 {
        bail!("expected 401 without an API key, got {unauth}");
    }
    let models: Value = c.get(format!("{}/models", info.base_url)).bearer_auth(&info.token).send().await?.json().await?;
    let ids: Vec<String> = models["data"].as_array().context("models list")?.iter().filter_map(|m| m["id"].as_str().map(String::from)).collect();
    if !ids.contains(&info.model) {
        bail!("{} not listed in /v1/models: {ids:?}", info.model);
    }
    let mut answers = serde_json::Map::new();
    for model in [&info.model, &info.base_model] {
        let body = json!({"model": model, "temperature": 0, "max_tokens": 64, "messages": [{"role": "user", "content": prompt}]});
        let r: Value = c.post(format!("{}/chat/completions", info.base_url)).bearer_auth(&info.token).json(&body).send().await?.error_for_status()?.json().await?;
        let text = r["choices"][0]["message"]["content"].as_str().context("chat completion content")?.to_string();
        answers.insert(model.clone(), json!(text));
    }
    let body = json!({"model": info.model, "stream": true, "max_tokens": 32, "messages": [{"role": "user", "content": prompt}]});
    let stream = c.post(format!("{}/chat/completions", info.base_url)).bearer_auth(&info.token).json(&body).send().await?.error_for_status()?.text().await?;
    if !stream.contains("data:") || !stream.contains("[DONE]") {
        bail!("streaming response was not server-sent events");
    }
    Ok(json!({"unauthorized_status": unauth, "models": ids, "answers": answers, "streaming": true, "engine": info.engine, "base_url": info.base_url}))
}

fn print_result(v: &Value) {
    println!("RESULT {v}");
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let sink = Arc::new(StdoutSink::new(cli.raw));
    let app = match App::new(runtime::home_dir(), None, sink) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::exit(2);
        }
    };
    let res: Result<Value, AppError> = async {
        Ok(match cli.cmd {
            Cmd::Hardware => serde_json::to_value(app.detect_hardware().await?).unwrap(),
            Cmd::Status => serde_json::to_value(app.runtime_status().await).unwrap(),
            Cmd::Bootstrap { download_model } => serde_json::to_value(app.bootstrap_runtime(download_model).await?).unwrap(),
            Cmd::Templates => serde_json::to_value(app.list_templates()).unwrap(),
            Cmd::Projects => serde_json::to_value(app.list_projects()?).unwrap(),
            Cmd::Doctor { full } => {
                let r = app.preflight(full).await;
                let v = serde_json::to_value(&r).unwrap();
                if !r.ok {
                    return Err(AppError { code: "PREFLIGHT_FAILED".into(), message: r.problems.join("; "), retryable: false });
                }
                v
            }
            Cmd::Models => serde_json::to_value(app.list_models()).unwrap(),
            Cmd::Packs => serde_json::to_value(app.list_packs()).unwrap(),
            Cmd::InstallPack { id } => app.install_pack_blocking(&id).await?,
            Cmd::Samples => json!({"samples": app.sample_data()}),
            Cmd::CreateProject { name, template, base_id, base_path, base_gguf, backend, quant } => serde_json::to_value(app.create_project(CreateProjectRequest { name, template, base_model_id: base_id, base_model_path: base_path, base_model_gguf: base_gguf, backend, quant }).await?).unwrap(),
            Cmd::Ingest { project, inputs, rules } => {
                let rules: Vec<String> = serde_json::from_str(&rules).map_err(|e| AppError { code: "BAD_INPUT".into(), message: format!("--rules must be a JSON list: {e}"), retryable: false })?;
                app.ingest_dataset(&project, inputs, rules).await?
            }
            Cmd::Preview { project } => app.preview_items(&project, 12).await?,
            Cmd::Train { project, preset, resume, overrides } => {
                app.detect_hardware().await?;
                let overrides = overrides.map(|o| serde_json::from_str(&o)).transpose().map_err(|e| AppError { code: "BAD_INPUT".into(), message: format!("--overrides: {e}"), retryable: false })?;
                app.train_blocking(TrainRequest { project_id: project, preset: Some(preset), resume_run_id: resume, overrides }).await?
            }
            Cmd::Arena { project, prompt, n } => app.run_benchmark_arena(&project, prompt, n).await?,
            Cmd::Serve { project, port } => {
                let info = app.start_local_api(&project, Some(port)).await?;
                eprintln!("Serving {} at {}\nAPI key: {}\nPress Ctrl+C to stop.", info.model, info.base_url, info.token);
                print_result(&serde_json::to_value(&info).unwrap());
                wait_for_shutdown_signal().await;
                app.stop_local_api().await?;
                json!({"stopped": true})
            }
            Cmd::ApiSmoke { project, port, prompt } => {
                let info = app.start_local_api(&project, Some(port)).await?;
                let r = api_smoke(&info, &prompt).await;
                app.stop_local_api().await?;
                r.map_err(|e| AppError { code: "API_SMOKE_FAILED".into(), message: format!("{e:#}"), retryable: false })?
            }
            Cmd::Export { project, out } => app.export_adapter(&project, &out).await?,
            Cmd::Import { project, file, trust, preset } => app.import_adapter(&project, &file, trust, Some(preset)).await?,
        })
    }
    .await;
    match res {
        Ok(v) => {
            print_result(&v);
            app.shutdown().await;
        }
        Err(e) => {
            eprintln!("error [{}]: {}", e.code, e.message);
            println!("RESULT {}", json!({"error": e}));
            app.shutdown().await;
            std::process::exit(1);
        }
    }
}
