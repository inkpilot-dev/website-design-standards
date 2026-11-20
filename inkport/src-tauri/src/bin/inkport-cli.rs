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
            Cmd::CreateProject { name, template, base_id, base_path, backend, quant } => serde_json::to_value(app.create_project(CreateProjectRequest { name, template, base_model_id: base_id, base_model_path: base_path, backend, quant }).await?).unwrap(),
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
                let _ = tokio::signal::ctrl_c().await;
                app.stop_local_api().await?;
                json!({"stopped": true})
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
