//! First-launch readiness for the all-in-one installer. Nothing here downloads anything.
//!
//! 1. Locate the bundled payload (Python runtime, llama.cpp, starter model) inside the installed app.
//! 2. Integrity: the payload's manifest.json must match the SHA-256 compiled into this (signed)
//!    binary, and every file must match the manifest. The first launch after an install or
//!    update hashes everything (multi-threaded, ~1-3 s for ~2 GB on an SSD); later launches check
//!    sizes plus the critical files, with a full re-check available on demand.
//! 3. Environment: an isolated, offline-by-construction environment for the engine (bundled
//!    interpreter run with -I -B, Hugging Face/transformers offline switches, caches under the
//!    Inkport home, library paths for the bundled llama.cpp).
//! 4. Warm-up and accelerator detection: `python -m engine hello` (training backends: MLX,
//!    CUDA or CPU) and `llama-server --list-devices` (inference: Metal, CUDA, Vulkan or CPU).
//!    A missing GPU is not an error: everything falls back to the CPU.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// SHA-256 of payload/manifest.json, embedded by build.rs from the all-in-one build.
/// Empty in development builds, where the manifest is still checked file-by-file.
pub const EXPECTED_MANIFEST_SHA256: &str = env!("INKPORT_EXPECTED_MANIFEST_SHA256");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonInfo {
    pub exe: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarterModel {
    pub dir: String,
    pub id: String,
    pub label: String,
    pub format: String,
    #[serde(default)]
    pub gguf: Option<String>,
    pub backend: String,
    pub quant: String,
    #[serde(default)]
    pub chat_kwargs: Value,
    #[serde(default)]
    pub params_b: f64,
    #[serde(default)]
    pub license: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Component {
    pub files: u64,
    pub bytes: u64,
    pub tree_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    pub app_version: String,
    pub platform: String,
    pub flavor: String,
    pub created_at: u64,
    pub python: PythonInfo,
    pub llama_server: Option<String>,
    pub llama_convert: Option<String>,
    pub starter_model: StarterModel,
    pub samples: String,
    pub packs_catalog: String,
    pub components: BTreeMap<String, Component>,
    pub critical: Vec<String>,
    pub files: BTreeMap<String, (u64, String)>,
}

#[derive(Debug, Clone)]
pub struct Bundle {
    pub root: PathBuf,
    pub manifest: Manifest,
    pub manifest_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VerifyReport {
    pub ok: bool,
    pub mode: String,
    pub files_checked: u64,
    pub bytes_checked: u64,
    pub seconds: f64,
    pub manifest_pinned: bool,
    pub problems: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PreflightReport {
    pub ok: bool,
    pub bundled: bool,
    pub offline_core: bool,
    pub platform: String,
    pub flavor: String,
    pub app_version: String,
    pub engine_version: Option<String>,
    pub python: Option<String>,
    pub training_accelerator: String,
    pub inference_devices: Vec<String>,
    pub starter_model: Option<Value>,
    pub verify: VerifyReport,
    pub seconds: f64,
    pub problems: Vec<String>,
}

impl Bundle {
    pub fn path(&self, rel: &str) -> PathBuf {
        rel.split('/').fold(self.root.clone(), |p, part| p.join(part))
    }

    pub fn python(&self) -> PathBuf {
        self.path(&self.manifest.python.exe)
    }

    pub fn llama_server(&self) -> Option<PathBuf> {
        self.manifest.llama_server.as_deref().map(|r| self.path(r)).filter(|p| p.is_file())
    }

    pub fn starter_dir(&self) -> PathBuf {
        self.path(&self.manifest.starter_model.dir)
    }

    pub fn samples_dir(&self) -> PathBuf {
        self.path(&self.manifest.samples)
    }

    pub fn packs_catalog(&self) -> PathBuf {
        self.path(&self.manifest.packs_catalog)
    }

    /// project.json `base_model` for the bundled starter model.
    pub fn starter_base_model(&self) -> Value {
        let m = &self.manifest.starter_model;
        let mut v = json!({"id": m.id, "label": m.label, "path": self.starter_dir().display().to_string(), "format": m.format, "backend": m.backend, "chat_kwargs": if m.chat_kwargs.is_null() { json!({}) } else { m.chat_kwargs.clone() }, "bundled": true});
        if let Some(g) = &m.gguf {
            v["gguf"] = json!(g);
        }
        v
    }

    /// Environment for every engine process: isolated, offline, caches under the Inkport home.
    pub fn engine_env(&self, home: &Path) -> (Vec<(String, String)>, Vec<String>) {
        let cache = home.join("cache");
        let mut set: Vec<(String, String)> = vec![
            ("INKPORT_BUNDLE".into(), self.root.display().to_string()),
            ("INKPORT_BUNDLED".into(), "1".into()),
            ("HF_HUB_OFFLINE".into(), "1".into()),
            ("TRANSFORMERS_OFFLINE".into(), "1".into()),
            ("HF_DATASETS_OFFLINE".into(), "1".into()),
            ("HF_HUB_DISABLE_TELEMETRY".into(), "1".into()),
            ("DO_NOT_TRACK".into(), "1".into()),
            ("HF_HOME".into(), cache.join("hf").display().to_string()),
            ("TORCH_HOME".into(), cache.join("torch").display().to_string()),
            ("XDG_CACHE_HOME".into(), cache.display().to_string()),
            ("PYTHONDONTWRITEBYTECODE".into(), "1".into()),
            ("PYTHONNOUSERSITE".into(), "1".into()),
            ("PYTHONUTF8".into(), "1".into()),
            ("TOKENIZERS_PARALLELISM".into(), "false".into()),
        ];
        if let Some(conv) = &self.manifest.llama_convert {
            set.push(("INKPORT_LLAMA_CPP".into(), self.path(conv).display().to_string()));
        }
        if let Some(server) = self.llama_server() {
            set.push(("INKPORT_LLAMA_SERVER".into(), server.display().to_string()));
        }
        // Never let a user's own Python/conda setup leak into the bundled interpreter.
        let remove = ["PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONEXECUTABLE", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PIP_REQUIRE_VIRTUALENV", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES"].iter().map(|s| s.to_string()).collect();
        (set, remove)
    }

    /// Extra environment for llama-server (its libraries live next to it).
    pub fn llama_env(&self) -> Vec<(String, String)> {
        let dir = self.path("llama");
        let mut env = vec![];
        if cfg!(target_os = "linux") {
            let prev = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
            env.push(("LD_LIBRARY_PATH".into(), if prev.is_empty() { dir.display().to_string() } else { format!("{}:{prev}", dir.display()) }));
        } else if cfg!(windows) {
            let prev = std::env::var("PATH").unwrap_or_default();
            env.push(("PATH".into(), format!("{};{prev}", dir.display())));
        }
        env
    }
}

/// Where an installed Inkport keeps its payload, per packaging format.
pub fn candidate_dirs(resource_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut c = vec![];
    if let Ok(p) = std::env::var("INKPORT_BUNDLE_DIR") {
        c.push(PathBuf::from(p));
    }
    if let Some(r) = resource_dir {
        c.push(r.join("payload"));
    }
    if let Ok(exe) = std::env::current_exe() {
        let exe = std::fs::canonicalize(&exe).unwrap_or(exe);
        if let Some(dir) = exe.parent() {
            c.push(dir.join("payload")); // Windows (NSIS/MSI): next to Inkport.exe
            c.push(dir.join("resources").join("payload"));
            c.push(dir.join("..").join("Resources").join("payload")); // macOS .app
            c.push(dir.join("..").join("lib").join("Inkport").join("payload")); // .deb, AppImage, portable tarball
            c.push(dir.join("..").join("lib").join("inkport").join("payload"));
        }
    }
    if let Ok(appdir) = std::env::var("APPDIR") {
        c.push(PathBuf::from(appdir).join("usr").join("lib").join("Inkport").join("payload"));
    }
    c
}

pub fn locate(resource_dir: Option<&Path>) -> Option<PathBuf> {
    candidate_dirs(resource_dir).into_iter().find(|d| d.join("manifest.json").is_file()).map(|d| std::fs::canonicalize(&d).unwrap_or(d))
}

pub fn load(root: &Path) -> Result<Bundle> {
    let raw = std::fs::read(root.join("manifest.json")).with_context(|| format!("cannot read {}/manifest.json", root.display()))?;
    let manifest_sha256 = hex(&Sha256::digest(&raw));
    let manifest: Manifest = serde_json::from_slice(&raw).context("manifest.json is not a valid Inkport payload manifest")?;
    if manifest.format != "inkport-payload" {
        bail!("unexpected payload format {}", manifest.format);
    }
    let expected_platform = current_platform();
    if manifest.platform != expected_platform {
        bail!("this installation contains files for {}, but this computer is {expected_platform}", manifest.platform);
    }
    Ok(Bundle { root: root.to_path_buf(), manifest, manifest_sha256 })
}

pub fn current_platform() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => "linux-x86_64",
        ("macos", "aarch64") => "macos-aarch64",
        ("windows", "x86_64") => "windows-x86_64",
        _ => "unsupported",
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// Verify the payload. `full` hashes every file; otherwise sizes for all + hashes for critical files.
pub fn verify(bundle: &Bundle, full: bool, progress: &(dyn Fn(u64, u64) + Sync)) -> VerifyReport {
    verify_with_pin(bundle, full, EXPECTED_MANIFEST_SHA256, progress)
}

/// `pin`: the manifest SHA-256 this binary was built for ("" = development build, not pinned).
pub fn verify_with_pin(bundle: &Bundle, full: bool, pin: &str, progress: &(dyn Fn(u64, u64) + Sync)) -> VerifyReport {
    let t0 = Instant::now();
    let mut problems = vec![];
    let pinned = !pin.is_empty();
    if pinned && bundle.manifest_sha256 != pin {
        problems.push("manifest.json does not match this app build (the installation was modified or mixed with another version)".to_string());
    }
    let files: Vec<(&String, &(u64, String))> = bundle.manifest.files.iter().collect();
    let total_bytes: u64 = files.iter().filter(|(k, _)| full || bundle.manifest.critical.contains(k)).map(|(_, v)| v.0).sum();
    let done = AtomicU64::new(0);
    let checked = AtomicU64::new(0);
    let bad: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(vec![]);
    let next = AtomicUsize::new(0);
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(2, 16);
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some((rel, (size, sha))) = files.get(i) else { break };
                let path = bundle.path(rel);
                let meta = match std::fs::metadata(&path) {
                    Ok(m) => m,
                    Err(_) => {
                        bad.lock().unwrap().push(format!("missing: {rel}"));
                        continue;
                    }
                };
                if meta.len() != *size {
                    bad.lock().unwrap().push(format!("wrong size: {rel}"));
                    continue;
                }
                if full || bundle.manifest.critical.contains(rel) {
                    match sha256_file(&path) {
                        Ok(h) if &h == sha => {}
                        _ => bad.lock().unwrap().push(format!("checksum mismatch: {rel}")),
                    }
                    let d = done.fetch_add(*size, Ordering::Relaxed) + size;
                    progress(d, total_bytes);
                }
                checked.fetch_add(1, Ordering::Relaxed);
            });
        }
    });
    let mut bad = bad.into_inner().unwrap();
    bad.sort();
    if bad.len() > 20 {
        let n = bad.len();
        bad.truncate(20);
        bad.push(format!("…and {} more", n - 20));
    }
    problems.extend(bad);
    VerifyReport { ok: problems.is_empty(), mode: if full { "full".into() } else { "quick".into() }, files_checked: checked.load(Ordering::Relaxed), bytes_checked: done.load(Ordering::Relaxed), seconds: t0.elapsed().as_secs_f64(), manifest_pinned: pinned, problems }
}

fn stamp_path(home: &Path) -> PathBuf {
    home.join("state").join("bundle-verified.json")
}

/// Full verification once per installed build (first launch, after updates), quick afterwards.
pub fn needs_full_verify(home: &Path, bundle: &Bundle) -> bool {
    let Ok(raw) = std::fs::read(stamp_path(home)) else { return true };
    let Ok(v) = serde_json::from_slice::<Value>(&raw) else { return true };
    v["manifest_sha256"].as_str() != Some(bundle.manifest_sha256.as_str()) || v["root"].as_str() != Some(&bundle.root.display().to_string())
}

pub fn write_stamp(home: &Path, bundle: &Bundle, report: &VerifyReport) {
    let _ = std::fs::create_dir_all(home.join("state"));
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let _ = std::fs::write(stamp_path(home), serde_json::to_vec_pretty(&json!({"manifest_sha256": bundle.manifest_sha256, "root": bundle.root.display().to_string(), "verified_at": now, "seconds": report.seconds})).unwrap_or_default());
}

fn command(program: &Path) -> tokio::process::Command {
    let mut c = tokio::process::Command::new(program);
    c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    #[cfg(windows)]
    c.creation_flags(0x0800_0000);
    c
}

/// `python -m engine hello` with the bundled interpreter. Returns (engine_version, backends).
pub async fn engine_hello(bundle: &Bundle, home: &Path) -> Result<(String, BTreeMap<String, bool>)> {
    let (set, remove) = bundle.engine_env(home);
    let mut c = command(&bundle.python());
    c.args(["-I", "-B", "-m", "engine", "hello"]);
    for k in &remove {
        c.env_remove(k);
    }
    c.envs(set);
    let out = tokio::time::timeout(Duration::from_secs(90), c.output()).await.map_err(|_| anyhow!("the bundled engine took too long to start"))??;
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            if v["type"] == "hello" {
                return Ok((v["engine_version"].as_str().unwrap_or_default().to_string(), serde_json::from_value(v["backends"].clone()).unwrap_or_default()));
            }
        }
    }
    bail!("the bundled engine did not start: {}", String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or("no output"))
}

/// Parse `llama-server --list-devices` (e.g. "  Vulkan0: NVIDIA GeForce RTX 4060 (8188 MiB, 7700 MiB free)").
pub fn parse_devices(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| l.contains(':') && !l.to_lowercase().starts_with("available devices") && !l.starts_with("load_backend") && !l.starts_with("ggml_"))
        .filter(|l| ["vulkan", "cuda", "metal", "mtl", "rocm", "hip", "sycl", "cpu", "blas"].iter().any(|k| l.to_lowercase().starts_with(k)))
        .map(String::from)
        .collect()
}

pub async fn inference_devices(bundle: &Bundle) -> Vec<String> {
    let Some(server) = bundle.llama_server() else { return vec![] };
    let mut c = command(&server);
    c.arg("--list-devices").envs(bundle.llama_env());
    match tokio::time::timeout(Duration::from_secs(30), c.output()).await {
        Ok(Ok(out)) => parse_devices(&(String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr))),
        _ => vec![],
    }
}

pub fn training_accelerator(backends: &BTreeMap<String, bool>) -> String {
    if *backends.get("mlx").unwrap_or(&false) {
        "Apple GPU (MLX)".into()
    } else if *backends.get("cuda").unwrap_or(&false) {
        "NVIDIA GPU (CUDA)".into()
    } else if *backends.get("mps").unwrap_or(&false) {
        "Apple GPU (Metal)".into()
    } else {
        "Processor (CPU)".into()
    }
}

/// Full preflight: integrity, warm-up, accelerators. Never downloads, never prompts.
pub async fn preflight(bundle: &Bundle, home: &Path, force_full: bool, progress: Arc<dyn Fn(&str, u64, u64) + Send + Sync>) -> PreflightReport {
    let t0 = Instant::now();
    let full = force_full || needs_full_verify(home, bundle);
    let b2 = bundle.clone();
    let p2 = Arc::clone(&progress);
    let verify_report = tokio::task::spawn_blocking(move || verify(&b2, full, &move |d, t| p2("verify", d, t))).await.unwrap_or_default();
    if verify_report.ok && full {
        write_stamp(home, bundle, &verify_report);
    }
    progress("warmup", 0, 1);
    let (hello, devices) = tokio::join!(engine_hello(bundle, home), inference_devices(bundle));
    progress("warmup", 1, 1);
    let mut problems = verify_report.problems.clone();
    let (engine_version, accel) = match hello {
        Ok((v, b)) => (Some(v), training_accelerator(&b)),
        Err(e) => {
            problems.push(e.to_string());
            (None, "unavailable".into())
        }
    };
    let starter_size: u64 = bundle.manifest.files.iter().filter(|(k, _)| k.starts_with(&bundle.manifest.starter_model.dir)).map(|(_, v)| v.0).sum();
    let mut starter = serde_json::to_value(&bundle.manifest.starter_model).unwrap_or(Value::Null);
    starter["size_mb"] = json!(starter_size / 1_000_000);
    PreflightReport {
        ok: problems.is_empty(),
        bundled: true,
        offline_core: true,
        platform: bundle.manifest.platform.clone(),
        flavor: bundle.manifest.flavor.clone(),
        app_version: bundle.manifest.app_version.clone(),
        engine_version,
        python: Some(bundle.python().display().to_string()),
        training_accelerator: accel,
        inference_devices: if devices.is_empty() { vec!["CPU".into()] } else { devices },
        starter_model: Some(starter),
        verify: verify_report,
        seconds: t0.elapsed().as_secs_f64(),
        problems,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_bundle(dir: &Path) -> Bundle {
        std::fs::create_dir_all(dir.join("python/bin")).unwrap();
        std::fs::create_dir_all(dir.join("models/starter")).unwrap();
        std::fs::write(dir.join("python/bin/python3"), b"#!fake").unwrap();
        std::fs::write(dir.join("models/starter/model.json"), b"{}").unwrap();
        std::fs::write(dir.join("models/starter/w.gguf"), vec![7u8; 3_000_000]).unwrap();
        let mut files = BTreeMap::new();
        for rel in ["python/bin/python3", "models/starter/model.json", "models/starter/w.gguf"] {
            let p = dir.join(rel);
            files.insert(rel.to_string(), (std::fs::metadata(&p).unwrap().len(), sha256_file(&p).unwrap()));
        }
        let manifest = json!({
            "format": "inkport-payload", "version": 1, "app_version": "0.1.0", "platform": current_platform(), "flavor": "cpu", "created_at": 0,
            "python": {"exe": "python/bin/python3", "version": "3.12.12"}, "llama_server": null, "llama_convert": null,
            "starter_model": {"dir": "models/starter", "id": "t", "label": "t", "format": "gguf", "gguf": "w.gguf", "backend": "torch", "quant": "fp32", "chat_kwargs": {}, "params_b": 0.1, "license": "x"},
            "samples": "samples", "packs_catalog": "packs/catalog.json", "components": {}, "critical": ["python/bin/python3", "models/starter/model.json"], "files": files
        });
        std::fs::write(dir.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
        load(dir).unwrap()
    }

    #[test]
    fn verify_detects_tampering_and_missing_files() {
        let t = tempfile::tempdir().unwrap();
        let b = fake_bundle(t.path());
        assert!(verify_with_pin(&b, true, "", &|_, _| {}).ok);
        std::fs::write(t.path().join("models/starter/w.gguf"), vec![8u8; 3_000_000]).unwrap(); // same size, different bytes
        let quick = verify_with_pin(&b, false, "", &|_, _| {});
        assert!(quick.ok, "quick mode only hashes critical files");
        let full = verify_with_pin(&b, true, "", &|_, _| {});
        assert!(!full.ok && full.problems.iter().any(|p| p.contains("checksum mismatch: models/starter/w.gguf")));
        std::fs::remove_file(t.path().join("python/bin/python3")).unwrap();
        assert!(verify_with_pin(&b, false, "", &|_, _| {}).problems.iter().any(|p| p.contains("missing: python/bin/python3")));
    }

    #[test]
    fn manifest_pin_mismatch_is_reported() {
        let t = tempfile::tempdir().unwrap();
        let b = fake_bundle(t.path());
        let good = verify_with_pin(&b, false, &b.manifest_sha256, &|_, _| {});
        assert!(good.ok && good.manifest_pinned);
        let bad = verify_with_pin(&b, false, &"0".repeat(64), &|_, _| {});
        assert!(!bad.ok && bad.problems[0].contains("does not match this app build"));
    }

    #[test]
    fn stamp_controls_full_verification() {
        let t = tempfile::tempdir().unwrap();
        let home = tempfile::tempdir().unwrap();
        let b = fake_bundle(t.path());
        assert!(needs_full_verify(home.path(), &b));
        let r = verify_with_pin(&b, true, "", &|_, _| {});
        write_stamp(home.path(), &b, &r);
        assert!(!needs_full_verify(home.path(), &b));
    }

    #[test]
    fn parses_llama_devices() {
        let out = "load_backend: loaded Vulkan backend\nAvailable devices:\n  Vulkan0: NVIDIA GeForce RTX 4060 (8188 MiB, 7700 MiB free)\n  CPU: AMD Ryzen 7 (31000 MiB, 20000 MiB free)\n";
        assert_eq!(parse_devices(out), vec!["Vulkan0: NVIDIA GeForce RTX 4060 (8188 MiB, 7700 MiB free)", "CPU: AMD Ryzen 7 (31000 MiB, 20000 MiB free)"]);
    }

    #[test]
    fn engine_env_is_offline_and_isolated() {
        let t = tempfile::tempdir().unwrap();
        let b = fake_bundle(t.path());
        let (set, remove) = b.engine_env(Path::new("/home/x/.inkport"));
        let get = |k: &str| set.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
        assert_eq!(get("HF_HUB_OFFLINE").as_deref(), Some("1"));
        assert_eq!(get("TRANSFORMERS_OFFLINE").as_deref(), Some("1"));
        assert!(remove.contains(&"PYTHONPATH".to_string()) && remove.contains(&"PYTHONHOME".to_string()));
        assert_eq!(b.starter_base_model()["gguf"], "w.gguf");
    }
}
