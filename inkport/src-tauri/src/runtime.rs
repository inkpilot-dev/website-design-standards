//! Zero-terminal Python runtime bootstrap.
//!
//! Uses `uv` (bundled as a resource, or found on PATH) to create an isolated CPython 3.12
//! virtualenv under the Barnraise home directory, installs the right PyTorch build for the
//! detected hardware (CUDA on NVIDIA, ROCm on AMD Linux, CPU elsewhere; none on Apple
//! Silicon, which uses MLX), installs the engine with the matching extra, verifies it with
//! `python -m engine hello`, and optionally downloads the recommended base model.
//! Progress is streamed on `runtime://progress`.

use crate::contract::{HardwareReport, RuntimeStatus};
use crate::events::EventSink;
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

pub fn home_dir() -> PathBuf {
    if let Ok(h) = std::env::var("BARNRAISE_HOME") {
        return PathBuf::from(h);
    }
    dirs::data_dir().unwrap_or_else(|| dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))).join("Barnraise")
}

pub fn venv_dir(home: &Path) -> PathBuf {
    home.join("runtime").join("venv")
}

pub fn venv_python(home: &Path) -> PathBuf {
    let v = venv_dir(home);
    if cfg!(windows) {
        v.join("Scripts").join("python.exe")
    } else {
        v.join("bin").join("python")
    }
}

/// The interpreter to run the engine with. BARNRAISE_PYTHON overrides (development, CI).
pub fn python_path(home: &Path) -> PathBuf {
    std::env::var("BARNRAISE_PYTHON").map(PathBuf::from).unwrap_or_else(|_| venv_python(home))
}

/// In development the engine is imported straight from the checkout (no reinstall needed).
pub fn dev_engine_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BARNRAISE_ENGINE_SRC") {
        return Some(PathBuf::from(p));
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent()?.to_path_buf();
    root.join("engine").join("cli.py").is_file().then_some(root)
}

fn find_uv(resource_dir: Option<&Path>) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BARNRAISE_UV") {
        return Some(PathBuf::from(p));
    }
    let exe = if cfg!(windows) { "uv.exe" } else { "uv" };
    if let Some(r) = resource_dir {
        for cand in [r.join("resources").join("bin").join(exe), r.join("bin").join(exe)] {
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(exe)).find(|p| p.is_file()).or_else(|| dirs::home_dir().map(|h| h.join(".local").join("bin").join(exe)).filter(|p| p.is_file()))
}

/// The engine package to install: a bundled wheel in release builds, the checkout in dev.
fn engine_source(resource_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(r) = resource_dir {
        for dir in [r.join("resources").join("engine"), r.join("engine")] {
            if let Ok(entries) = std::fs::read_dir(&dir) {
                if let Some(whl) = entries.flatten().map(|e| e.path()).find(|p| p.extension().map(|e| e == "whl").unwrap_or(false)) {
                    return Some(whl);
                }
            }
        }
    }
    dev_engine_root()
}

/// (extra, torch index URL or None for PyPI default / no torch)
pub fn install_plan(hw: &HardwareReport) -> (&'static str, Option<&'static str>) {
    let vendor = hw.recommendation.vendor.as_str();
    match (hw.os.as_str(), vendor) {
        (_, "apple") => ("mlx", None),
        ("linux", "nvidia") => ("cuda", Some("https://pypi.org/simple")),
        ("windows", "nvidia") => ("cuda", Some("https://download.pytorch.org/whl/cu128")),
        ("linux", "amd") => ("torch", Some("https://download.pytorch.org/whl/rocm6.3")),
        ("linux", "intel") | ("windows", "intel") => ("torch", Some("https://download.pytorch.org/whl/xpu")),
        _ => ("cpu", Some("https://download.pytorch.org/whl/cpu")),
    }
}

struct Reporter {
    sink: Arc<dyn EventSink>,
    total: usize,
}

impl Reporter {
    fn step(&self, i: usize, label: &str, status: &str, detail: &str) {
        let pct = (100.0 * (i as f64 + if status == "completed" { 1.0 } else { 0.0 }) / self.total as f64).min(100.0);
        self.sink.emit("runtime://progress", &json!({"type": "runtime", "step": i + 1, "total": self.total, "label": label, "status": status, "detail": detail, "pct": pct}));
    }
}

async fn run_streaming(program: &Path, args: &[String], rep: &Reporter, i: usize, label: &str, env: &[(&str, String)]) -> Result<()> {
    let mut cmd = Command::new(program);
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    for (k, v) in env {
        cmd.env(k, v);
    }
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000);
    let mut child = cmd.spawn().with_context(|| format!("could not run {}", program.display()))?;
    let mut err = BufReader::new(child.stderr.take().unwrap()).lines();
    let mut out = BufReader::new(child.stdout.take().unwrap()).lines();
    let mut tail: Vec<String> = vec![];
    let mut last = std::time::Instant::now();
    loop {
        tokio::select! {
            l = err.next_line() => match l { Ok(Some(l)) => { if tail.len() > 20 { tail.remove(0); } if last.elapsed().as_millis() > 300 { rep.step(i, label, "running", l.trim()); last = std::time::Instant::now(); } tail.push(l); }, _ => break },
            l = out.next_line() => if let Ok(Some(l)) = l { if tail.len() > 20 { tail.remove(0); } tail.push(l); },
        }
    }
    let status = child.wait().await?;
    if !status.success() {
        bail!("{label} failed:\n{}", tail.join("\n"));
    }
    Ok(())
}

async fn engine_hello(python: &Path, pythonpath: Option<&Path>) -> Result<(String, HashMap<String, bool>)> {
    let mut cmd = Command::new(python);
    cmd.args(["-m", "engine", "hello"]).stdin(Stdio::null()).stderr(Stdio::null());
    if let Some(pp) = pythonpath {
        cmd.env("PYTHONPATH", pp);
    }
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000);
    let out = tokio::time::timeout(std::time::Duration::from_secs(120), cmd.output()).await.map_err(|_| anyhow!("engine check timed out"))??;
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            if v["type"] == "hello" {
                let backends = serde_json::from_value(v["backends"].clone()).unwrap_or_default();
                return Ok((v["engine_version"].as_str().unwrap_or("").to_string(), backends));
            }
        }
    }
    bail!("the engine did not start: {}", String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or(""))
}

pub async fn status(home: &Path, pythonpath: Option<&Path>) -> RuntimeStatus {
    let python = python_path(home);
    let mut st = RuntimeStatus { home: home.display().to_string(), python: Some(python.display().to_string()), ..Default::default() };
    if !python.is_file() && std::env::var("BARNRAISE_PYTHON").is_err() {
        st.message = "Not set up yet".into();
        return st;
    }
    match engine_hello(&python, pythonpath).await {
        Ok((ver, backends)) => {
            st.ready = true;
            st.engine_version = Some(ver);
            st.backends = backends;
            st.message = "Ready".into();
        }
        Err(e) => st.message = e.to_string(),
    }
    st
}

pub fn model_dir(home: &Path, model_id: &str, backend: &str) -> PathBuf {
    home.join("models").join(format!("{model_id}-{}", if backend == "mlx" { "mlx" } else { "hf" }))
}

pub async fn bootstrap(home: &Path, hw: &HardwareReport, resource_dir: Option<&Path>, pythonpath: Option<&Path>, download_model: bool, sink: Arc<dyn EventSink>) -> Result<RuntimeStatus> {
    let rep = Reporter { sink: Arc::clone(&sink), total: if download_model { 5 } else { 4 } };
    std::fs::create_dir_all(home.join("runtime"))?;
    let (extra, torch_index) = install_plan(hw);
    let python = python_path(home);

    if std::env::var("BARNRAISE_PYTHON").is_ok() {
        rep.step(0, "Using the developer's Python", "completed", &python.display().to_string());
        rep.step(1, "Installing PyTorch", "completed", "skipped (BARNRAISE_PYTHON is set)");
        rep.step(2, "Installing the Barnraise engine", "completed", "skipped (BARNRAISE_PYTHON is set)");
    } else {
        let uv = find_uv(resource_dir).ok_or_else(|| anyhow!("The installer helper (uv) is missing. Reinstall Barnraise."))?;
        rep.step(0, "Creating a private Python 3.12", "running", "");
        if !python.is_file() {
            run_streaming(&uv, &["venv".into(), "--python".into(), "3.12".into(), "--seed".into(), venv_dir(home).display().to_string()], &rep, 0, "Creating a private Python 3.12", &[]).await?;
        }
        rep.step(0, "Creating a private Python 3.12", "completed", "");
        let py = python.display().to_string();
        rep.step(1, "Installing PyTorch", "running", "");
        if let Some(idx) = torch_index {
            let mut args = vec!["pip".into(), "install".into(), "--python".into(), py.clone(), "torch>=2.6".into()];
            if idx != "https://pypi.org/simple" {
                args.extend(["--index-url".into(), idx.into()]);
            }
            run_streaming(&uv, &args, &rep, 1, "Installing PyTorch", &[]).await?;
        }
        rep.step(1, "Installing PyTorch", "completed", if torch_index.is_none() { "not needed on Apple Silicon (uses MLX)" } else { "" });
        rep.step(2, "Installing the Barnraise engine", "running", "");
        let src = engine_source(resource_dir).ok_or_else(|| anyhow!("The Barnraise engine package is missing from this installation."))?;
        let spec = format!("{}[{extra}]", src.display());
        run_streaming(&uv, &["pip".into(), "install".into(), "--python".into(), py, spec], &rep, 2, "Installing the Barnraise engine", &[]).await?;
        rep.step(2, "Installing the Barnraise engine", "completed", extra);
    }

    rep.step(3, "Checking everything works", "running", "");
    let st = status(home, pythonpath).await;
    if !st.ready {
        bail!("Setup finished but the engine does not start: {}", st.message);
    }
    rep.step(3, "Checking everything works", "completed", &format!("engine {}", st.engine_version.clone().unwrap_or_default()));

    if download_model {
        let rec = &hw.recommendation;
        let dest = model_dir(home, &rec.model_id, &rec.backend);
        rep.step(4, &format!("Downloading {}", rec.model_label), "running", "");
        if !dest.join("config.json").is_file() {
            let mut cmd = Command::new(&python);
            cmd.args(["-m", "engine", "fetch-model", "--id", &rec.model_id, "--backend", &rec.backend, "--dest", &dest.display().to_string()]).stdout(Stdio::piped()).stderr(Stdio::null());
            if let Some(pp) = pythonpath {
                cmd.env("PYTHONPATH", pp);
            }
            let mut child = cmd.spawn()?;
            let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                if let Ok(v) = serde_json::from_str::<Value>(&l) {
                    if v["type"] == "progress" {
                        rep.step(4, &format!("Downloading {}", rec.model_label), "running", &format!("{:.0}%", v["pct"].as_f64().unwrap_or(0.0)));
                    } else if v["type"] == "error" {
                        bail!("Model download failed: {}", v["message"].as_str().unwrap_or(""));
                    }
                }
            }
            if !child.wait().await?.success() {
                bail!("Model download failed");
            }
        }
        rep.step(4, &format!("Downloading {}", rec.model_label), "completed", &dest.display().to_string());
    }
    std::fs::write(home.join("runtime").join("runtime.json"), serde_json::to_vec_pretty(&json!({"python": python, "extra": extra, "engine_version": st.engine_version}))?)?;
    Ok(st)
}
