//! Optional add-on packs (bigger base models, dataset/verifier packs). Strictly opt-in: core
//! offline training, verification and the local API never call into this module.
//!
//! * Sources: Hugging Face repos (resolved to an exact commit; files checked against the
//!   SHA-256 / git-blob SHA-1 the Hub publishes for that commit) or plain URLs with a pinned
//!   SHA-256 and size.
//! * Resume-safe: bytes go to `<home>/packs/<id>/.staging/<file>.part` and continue with an
//!   HTTP Range request after a crash, cancel or network drop.
//! * Atomic: the pack only appears (`files/` + `pack.json`) after every file has been verified.
//! * Progress on `packs://progress`.

use crate::events::EventSink;
use anyhow::{anyhow, bail, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha1::Sha1;
use sha2::Digest;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackSource {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub repo: Option<String>,
    #[serde(default)]
    pub repo_type: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub include: Vec<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pack {
    pub id: String,
    pub title: String,
    pub description: String,
    pub kind: String,
    pub platforms: Vec<String>,
    #[serde(default)]
    pub flavors: Option<Vec<String>>,
    pub approx_size_mb: u64,
    pub license: String,
    pub sources: Vec<PackSource>,
    #[serde(default)]
    pub model: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackState {
    #[serde(flatten)]
    pub pack: Pack,
    pub installed: bool,
    pub installing: bool,
    pub partial_bytes: u64,
    pub path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlannedFile {
    path: String,
    url: String,
    size: u64,
    sha256: Option<String>,
    git_sha1: Option<String>,
}

pub struct Packs {
    home: PathBuf,
    catalog_path: Option<PathBuf>,
    sink: Arc<dyn EventSink>,
    client: reqwest::Client,
    running: Mutex<HashMap<String, Arc<AtomicBool>>>,
    hf_endpoint: String,
}

fn glob_match(pat: &str, s: &str) -> bool {
    let (p, t): (Vec<char>, Vec<char>) = (pat.chars().collect(), s.chars().collect());
    let (mut i, mut j, mut star, mut mark) = (0usize, 0usize, None::<usize>, 0usize);
    while j < t.len() {
        if i < p.len() && (p[i] == '?' || p[i] == t[j]) {
            i += 1;
            j += 1;
        } else if i < p.len() && p[i] == '*' {
            star = Some(i);
            mark = j;
            i += 1;
        } else if let Some(si) = star {
            i = si + 1;
            mark += 1;
            j = mark;
        } else {
            return false;
        }
    }
    while i < p.len() && p[i] == '*' {
        i += 1;
    }
    i == p.len()
}

fn safe_rel(path: &str) -> Result<PathBuf> {
    let p = Path::new(path);
    if p.is_absolute() || path.contains('\\') || p.components().any(|c| !matches!(c, std::path::Component::Normal(_))) {
        bail!("unsafe file path in pack: {path}");
    }
    Ok(p.to_path_buf())
}

impl Packs {
    pub fn new(home: &Path, catalog_path: Option<PathBuf>, sink: Arc<dyn EventSink>) -> Result<Self> {
        Ok(Self {
            home: home.to_path_buf(),
            catalog_path,
            sink,
            client: reqwest::Client::builder().user_agent("Inkport/0.1 (optional pack download)").connect_timeout(std::time::Duration::from_secs(20)).build()?,
            running: Mutex::new(HashMap::new()),
            hf_endpoint: std::env::var("INKPORT_HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into()),
        })
    }

    fn dir(&self, id: &str) -> PathBuf {
        self.home.join("packs").join(id)
    }

    pub fn catalog(&self) -> Vec<Pack> {
        let Some(p) = &self.catalog_path else { return vec![] };
        let Ok(raw) = std::fs::read(p) else { return vec![] };
        let all: Vec<Pack> = serde_json::from_slice::<Value>(&raw).ok().and_then(|v| serde_json::from_value(v["packs"].clone()).ok()).unwrap_or_default();
        all.into_iter().filter(|p| p.platforms.iter().any(|x| x == crate::bootstrap::current_platform())).collect()
    }

    pub fn list(&self) -> Vec<PackState> {
        let running = self.running.lock().unwrap();
        self.catalog()
            .into_iter()
            .map(|pack| {
                let dir = self.dir(&pack.id);
                let installed = dir.join("pack.json").is_file();
                let partial_bytes = walk_size(&dir.join(".staging"));
                PackState { installing: running.contains_key(&pack.id), installed, partial_bytes, path: installed.then(|| dir.join("files").display().to_string()), pack }
            })
            .collect()
    }

    /// project.json `base_model` entries for installed model packs.
    pub fn installed_models(&self) -> Vec<Value> {
        self.list()
            .into_iter()
            .filter(|s| s.installed && s.pack.kind == "model")
            .filter_map(|s| {
                let mut m = s.pack.model.clone()?;
                m["path"] = json!(s.path.clone()?);
                m["pack"] = json!(s.pack.id);
                Some(m)
            })
            .collect()
    }

    fn emit(&self, id: &str, status: &str, file: &str, done: u64, total: u64, message: &str) {
        let pct = if total > 0 { (done as f64 * 100.0 / total as f64).min(100.0) } else { 0.0 };
        self.sink.emit("packs://progress", &json!({"type": "pack", "pack_id": id, "status": status, "file": file, "done": done, "total": total, "pct": pct, "message": message}));
    }

    pub fn cancel(&self, id: &str) -> bool {
        if let Some(flag) = self.running.lock().unwrap().get(id) {
            flag.store(true, Ordering::Relaxed);
            return true;
        }
        false
    }

    pub fn remove(&self, id: &str) -> Result<()> {
        if self.running.lock().unwrap().contains_key(id) {
            bail!("the pack is still downloading; cancel it first");
        }
        let dir = self.dir(id);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }

    async fn get_json(&self, url: &str) -> Result<Value> {
        let r = self.client.get(url).send().await.with_context(|| format!("cannot reach {url}"))?;
        if !r.status().is_success() {
            bail!("{url} returned HTTP {}", r.status());
        }
        Ok(r.json().await?)
    }

    async fn plan(&self, pack: &Pack) -> Result<(Vec<PlannedFile>, Vec<Value>)> {
        let mut files = vec![];
        let mut pins = vec![];
        for src in &pack.sources {
            match src.kind.as_str() {
                "hf" => {
                    let repo = src.repo.as_deref().ok_or_else(|| anyhow!("hf source without repo"))?;
                    let api = if src.repo_type.as_deref() == Some("dataset") { "datasets" } else { "models" };
                    let prefix = if api == "datasets" { "datasets/" } else { "" };
                    let rev = src.revision.as_deref().unwrap_or("main");
                    let commit = self.get_json(&format!("{}/api/{api}/{repo}/revision/{rev}", self.hf_endpoint)).await?["sha"].as_str().ok_or_else(|| anyhow!("{repo}: no commit"))?.to_string();
                    let tree = self.get_json(&format!("{}/api/{api}/{repo}/tree/{commit}?recursive=true", self.hf_endpoint)).await?;
                    let mut matched = 0;
                    for e in tree.as_array().cloned().unwrap_or_default() {
                        let path = e["path"].as_str().unwrap_or_default().to_string();
                        if e["type"] != "file" || !src.include.iter().any(|p| glob_match(p, &path)) {
                            continue;
                        }
                        safe_rel(&path)?;
                        matched += 1;
                        let lfs = &e["lfs"];
                        files.push(PlannedFile {
                            url: format!("{}/{prefix}{repo}/resolve/{commit}/{path}", self.hf_endpoint),
                            size: lfs["size"].as_u64().or(e["size"].as_u64()).unwrap_or(0),
                            sha256: lfs["oid"].as_str().map(String::from),
                            git_sha1: if lfs.is_null() { e["oid"].as_str().map(String::from) } else { None },
                            path,
                        });
                    }
                    if matched == 0 {
                        bail!("{repo}@{commit}: no files match {:?}", src.include);
                    }
                    pins.push(json!({"repo": repo, "commit": commit}));
                }
                "url" => {
                    let url = src.url.clone().ok_or_else(|| anyhow!("url source without url"))?;
                    let path = src.path.clone().ok_or_else(|| anyhow!("url source without path"))?;
                    safe_rel(&path)?;
                    let sha256 = src.sha256.clone().ok_or_else(|| anyhow!("url sources must pin a sha256"))?;
                    files.push(PlannedFile { url: url.clone(), size: src.size.unwrap_or(0), sha256: Some(sha256), git_sha1: None, path });
                    pins.push(json!({"url": url}));
                }
                other => bail!("unknown pack source type {other}"),
            }
        }
        Ok((files, pins))
    }

    /// Download, verify and atomically install a pack. Safe to call again after a failure or cancel.
    pub async fn install(&self, id: &str) -> Result<PathBuf> {
        let pack = self.catalog().into_iter().find(|p| p.id == id).ok_or_else(|| anyhow!("unknown pack {id}"))?;
        let cancel = Arc::new(AtomicBool::new(false));
        {
            let mut running = self.running.lock().unwrap();
            if running.contains_key(id) {
                bail!("{} is already downloading", pack.title);
            }
            running.insert(id.to_string(), Arc::clone(&cancel));
        }
        let res = self.install_inner(&pack, &cancel).await;
        self.running.lock().unwrap().remove(id);
        match &res {
            Ok(p) => self.emit(id, "installed", "", 1, 1, &p.display().to_string()),
            Err(e) if cancel.load(Ordering::Relaxed) => self.emit(id, "cancelled", "", 0, 0, &format!("Paused. Downloaded parts are kept; press Resume to continue. ({e})")),
            Err(e) => self.emit(id, "failed", "", 0, 0, &format!("{e:#}")),
        }
        res
    }

    async fn install_inner(&self, pack: &Pack, cancel: &AtomicBool) -> Result<PathBuf> {
        let dir = self.dir(&pack.id);
        let staging = dir.join(".staging");
        std::fs::create_dir_all(&staging)?;
        self.emit(&pack.id, "resolving", "", 0, 0, "Looking up the exact files");
        let (files, pins) = self.plan(pack).await?;
        let total: u64 = files.iter().map(|f| f.size).sum();
        let mut done_before: u64 = 0;
        for f in &files {
            let dest = staging.join(safe_rel(&f.path)?);
            if dest.is_file() && verify_file(&dest, f).is_ok() {
                done_before += f.size;
                continue;
            }
            self.download(pack, f, &dest, done_before, total, cancel).await?;
            done_before += f.size;
        }
        self.emit(&pack.id, "verifying", "", total, total, "Checking every file");
        for f in &files {
            verify_file(&staging.join(safe_rel(&f.path)?), f).with_context(|| format!("{} failed verification", f.path))?;
        }
        let files_dir = dir.join("files");
        if files_dir.exists() {
            std::fs::remove_dir_all(&files_dir)?;
        }
        std::fs::rename(&staging, &files_dir)?;
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let record = json!({"id": pack.id, "installed_at": now, "sources": pins, "files": files.iter().map(|f| json!({"path": f.path, "size": f.size, "sha256": f.sha256, "git_sha1": f.git_sha1})).collect::<Vec<_>>(), "model": pack.model});
        std::fs::write(dir.join("pack.json"), serde_json::to_vec_pretty(&record)?)?;
        Ok(files_dir)
    }

    async fn download(&self, pack: &Pack, f: &PlannedFile, dest: &Path, base: u64, total: u64, cancel: &AtomicBool) -> Result<()> {
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let part = dest.with_file_name(format!("{}.part", dest.file_name().unwrap().to_string_lossy()));
        for attempt in 1..=5u32 {
            let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
            let mut req = self.client.get(&f.url);
            if have > 0 {
                req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
            }
            let result: Result<()> = async {
                let resp = req.send().await?;
                let status = resp.status();
                if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && have >= f.size && f.size > 0 {
                    return Ok(());
                }
                if !status.is_success() {
                    bail!("HTTP {status} for {}", f.path);
                }
                let append = have > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
                let mut out = std::fs::OpenOptions::new().create(true).write(true).append(append).truncate(!append).open(&part)?;
                let mut written = if append { have } else { 0 };
                let mut stream = resp.bytes_stream();
                let mut last = std::time::Instant::now();
                while let Some(chunk) = stream.next().await {
                    if cancel.load(Ordering::Relaxed) {
                        bail!("cancelled");
                    }
                    let chunk = chunk?;
                    out.write_all(&chunk)?;
                    written += chunk.len() as u64;
                    if last.elapsed().as_millis() > 250 {
                        self.emit(&pack.id, "downloading", &f.path, base + written, total, "");
                        last = std::time::Instant::now();
                    }
                }
                out.flush()?;
                Ok(())
            }
            .await;
            match result {
                Ok(()) => break,
                Err(e) if cancel.load(Ordering::Relaxed) => return Err(e),
                Err(e) if attempt == 5 => return Err(e.context(format!("download of {} failed after 5 attempts", f.path))),
                Err(_) => tokio::time::sleep(std::time::Duration::from_secs(2u64.pow(attempt))).await,
            }
        }
        verify_file(&part, f).map_err(|e| {
            let _ = std::fs::remove_file(&part);
            e
        })?;
        std::fs::rename(&part, dest)?;
        Ok(())
    }
}

fn verify_file(path: &Path, f: &PlannedFile) -> Result<()> {
    let meta = std::fs::metadata(path)?;
    if f.size > 0 && meta.len() != f.size {
        bail!("{}: size {} != {}", f.path, meta.len(), f.size);
    }
    if let Some(expected) = &f.sha256 {
        let got = crate::bootstrap::sha256_file(path)?;
        if &got != expected {
            bail!("{}: SHA-256 mismatch", f.path);
        }
    } else if let Some(expected) = &f.git_sha1 {
        let mut h = Sha1::new();
        h.update(format!("blob {}\0", meta.len()).as_bytes());
        let mut file = std::fs::File::open(path)?;
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if &got != expected {
            bail!("{}: git blob SHA-1 mismatch", f.path);
        }
    } else {
        bail!("{}: no checksum published; refusing to install", f.path);
    }
    Ok(())
}

fn walk_size(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else { return 0 };
    entries.flatten().map(|e| if e.path().is_dir() { walk_size(&e.path()) } else { e.metadata().map(|m| m.len()).unwrap_or(0) }).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::Path as AxPath;
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::IntoResponse;
    use axum::routing::get;
    use axum::{Json, Router};

    #[test]
    fn globbing_and_path_safety() {
        assert!(glob_match("*.safetensors", "model-00001-of-00002.safetensors"));
        assert!(glob_match("Qwen3-8B-Q4_K_M.gguf", "Qwen3-8B-Q4_K_M.gguf"));
        assert!(!glob_match("*.json", "model.safetensors"));
        assert!(safe_rel("../etc/passwd").is_err() && safe_rel("/abs").is_err() && safe_rel("a/b.json").is_ok());
    }

    /// A fake Hugging Face endpoint: resolve, tree listing, and Range-capable file serving.
    async fn fake_hub(weights: Vec<u8>, small: Vec<u8>) -> String {
        let w = Arc::new(weights);
        let s = Arc::new(small);
        let w_sha = crate::bootstrap::sha256_file(&{
            let p = std::env::temp_dir().join(format!("ink-w-{}", std::process::id()));
            std::fs::write(&p, &*w).unwrap();
            p
        })
        .unwrap();
        let mut h = Sha1::new();
        h.update(format!("blob {}\0", s.len()).as_bytes());
        h.update(&*s);
        let s_sha1: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        let (wl, sl) = (w.len(), s.len());
        let app = Router::new()
            .route("/api/models/{org}/{name}/revision/{rev}", get(|| async { Json(json!({"sha": "abc123"})) }))
            .route(
                "/api/models/{org}/{name}/tree/{rev}",
                get(move || {
                    let (w_sha, s_sha1) = (w_sha.clone(), s_sha1.clone());
                    async move { Json(json!([{"type": "file", "path": "model.gguf", "size": wl, "oid": "x", "lfs": {"oid": w_sha, "size": wl}}, {"type": "file", "path": "config.json", "size": sl, "oid": s_sha1}, {"type": "file", "path": "README.md", "size": 1, "oid": "y"}])) }
                }),
            )
            .route(
                "/{org}/{name}/resolve/{rev}/{file}",
                get(move |AxPath((_o, _n, _r, file)): AxPath<(String, String, String, String)>, headers: HeaderMap| {
                    let (w, s) = (Arc::clone(&w), Arc::clone(&s));
                    async move {
                        let body: Vec<u8> = if file == "model.gguf" { (*w).clone() } else { (*s).clone() };
                        if let Some(r) = headers.get("range").and_then(|v| v.to_str().ok()) {
                            let start: usize = r.trim_start_matches("bytes=").trim_end_matches('-').parse().unwrap();
                            return (StatusCode::PARTIAL_CONTENT, body[start..].to_vec()).into_response();
                        }
                        (StatusCode::OK, body).into_response()
                    }
                }),
            );
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
        format!("http://127.0.0.1:{port}")
    }

    #[tokio::test]
    async fn installs_resumes_and_verifies() {
        let weights: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
        let endpoint = fake_hub(weights.clone(), b"{\"a\":1}".to_vec()).await;
        let home = tempfile::tempdir().unwrap();
        let cat = home.path().join("catalog.json");
        std::fs::write(&cat, serde_json::to_vec(&json!({"packs": [{"id": "p1", "title": "P1", "description": "d", "kind": "model", "platforms": ["linux-x86_64"], "approx_size_mb": 3, "license": "MIT", "sources": [{"type": "hf", "repo": "org/name", "revision": "main", "include": ["*.gguf", "config.json"]}], "model": {"id": "m", "label": "M", "format": "gguf", "gguf": "model.gguf"}}]})).unwrap()).unwrap();
        let sink = Arc::new(crate::events::MemorySink::default());
        let mut packs = Packs::new(home.path(), Some(cat), sink.clone()).unwrap();
        packs.hf_endpoint = endpoint;
        // Simulate an interrupted earlier download: first 1 MB already on disk.
        let part = home.path().join("packs/p1/.staging/model.gguf.part");
        std::fs::create_dir_all(part.parent().unwrap()).unwrap();
        std::fs::write(&part, &weights[..1_000_000]).unwrap();
        let dir = packs.install("p1").await.unwrap();
        assert_eq!(std::fs::read(dir.join("model.gguf")).unwrap(), weights);
        assert!(!dir.join("README.md").exists(), "only included files are downloaded");
        let models = packs.installed_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["pack"], "p1");
        assert!(sink.events.lock().unwrap().iter().any(|(c, v)| c == "packs://progress" && v["status"] == "installed"));
    }
}
