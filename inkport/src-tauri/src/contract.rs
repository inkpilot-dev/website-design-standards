//! Rust mirror of contracts/schemas/*.json. Every engine stdout line is parsed into
//! [`EngineEvent`]; the raw JSON is also forwarded untouched to the UI.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

pub const TIERS_JSON: &str = include_str!("../../contracts/tiers.json");
pub const TEMPLATES_JSON: &str = include_str!("../../contracts/templates.json");

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuInfo {
    pub vendor: String,
    pub name: String,
    pub vram_gb: f64,
    #[serde(default)]
    pub vram_free_gb: Option<f64>,
    #[serde(default)]
    pub driver: Option<String>,
    #[serde(default)]
    pub compute_capability: Option<String>,
    pub unified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Recommendation {
    pub tier: String,
    pub vendor: String,
    pub memory_gb: f64,
    pub model_id: String,
    pub model_label: String,
    pub backend: String,
    pub quant: String,
    pub lora_r: u32,
    pub micro_batch: u32,
    pub grad_accum: u32,
    pub max_seq: u32,
    pub stage2: String,
    pub rft_k: u32,
    pub plain: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareReport {
    pub os: String,
    pub arch: String,
    pub cpu_brand: String,
    pub cpu_cores: u32,
    pub ram_gb: f64,
    pub disk_free_gb: f64,
    pub gpus: Vec<GpuInfo>,
    pub apple_silicon: bool,
    pub recommendation: Recommendation,
    /// Plain-English summary lines for the System Ready screen.
    #[serde(default)]
    pub summary: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Side {
    pub output: String,
    pub pass: Option<bool>,
    pub score: Option<f64>,
}

/// Typed view of the engine protocol (contracts/schemas/event.schema.json).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineEvent {
    Hello { engine_version: String, python: String, platform: String, backends: HashMap<String, bool> },
    Heartbeat {},
    Log { level: String, message: String },
    Stage { stage: String, status: String, #[serde(default)] detail: Option<String> },
    Progress { stage: String, step: u64, total: u64, pct: f64, overall_pct: f64, eta_s: Option<f64>, metrics: HashMap<String, Option<f64>> },
    Eval { split: String, model: String, n: u64, passed: u64, pass_rate: f64, ci95: [f64; 2] },
    Benchmark { source: String, case_id: String, prompt: String, reference: Option<String>, baseline: Side, trained: Side, #[serde(default)] step: Option<u64>, #[serde(default)] judge: Option<Value> },
    Gate { decision: String, label: Option<String>, delta_pts: f64, baseline_pass_rate: f64, candidate_pass_rate: f64, checks: HashMap<String, bool>, version: Option<String>, report_path: String, explanation: String },
    Artifact { kind: String, path: String, sha256: String },
    Ready { host: String, port: u16, model: String },
    Error { code: String, message: String, retryable: bool },
    Done { ok: bool, result: Value },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Ingest,
    Train,
    Arena,
    Serve,
    Export,
    Import,
    Rules,
    Preview,
    Hello,
    Fetch,
}

impl JobKind {
    /// Which UI channel carries this job's non-log events.
    pub fn channel(&self) -> &'static str {
        match self {
            JobKind::Arena => "benchmark://result",
            JobKind::Serve => "api://status",
            JobKind::Fetch => "runtime://progress",
            _ => "training://progress",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobOutcome {
    pub job_id: String,
    pub ok: bool,
    pub result: Value,
    pub error: Option<EngineError>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectInfo {
    pub id: String,
    pub slug: String,
    pub name: String,
    pub template: String,
    pub dir: String,
    pub base_model: Value,
    pub backend: Option<String>,
    pub quant: Option<String>,
    pub active_version: Option<String>,
    pub counts: Option<Value>,
    pub latest_report: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiInfo {
    pub url: String,
    pub base_url: String,
    pub token: String,
    pub model: String,
    pub base_model: String,
    pub version: String,
    /// What is answering: "llama.cpp (Vulkan0: …)", "MLX (Apple GPU)" or "PyTorch".
    pub engine: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeStatus {
    pub ready: bool,
    pub python: Option<String>,
    pub engine_version: Option<String>,
    pub backends: HashMap<String, bool>,
    pub home: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Template {
    pub id: String,
    pub title: String,
    pub tagline: String,
    pub proof: String,
    pub proof_label: String,
    pub accepts: Vec<String>,
    pub needs: String,
    pub example_rules: Vec<String>,
}

pub fn templates() -> Vec<Template> {
    #[derive(Deserialize)]
    struct File {
        templates: Vec<Template>,
    }
    serde_json::from_str::<File>(TEMPLATES_JSON).expect("contracts/templates.json is valid").templates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_fixture_event_parses() {
        let fixtures = include_str!("../../contracts/fixtures/events.ndjson");
        let mut kinds = 0;
        for line in fixtures.lines().filter(|l| !l.trim().is_empty()) {
            let ev: EngineEvent = serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}"));
            if let EngineEvent::Gate { decision, .. } = &ev {
                assert_eq!(decision, "PROMOTE");
            }
            kinds += 1;
        }
        assert_eq!(kinds, 12, "one fixture per event type");
    }

    #[test]
    fn templates_load() {
        let t = templates();
        assert_eq!(t.iter().map(|t| t.id.as_str()).collect::<Vec<_>>(), ["business_rules", "code_sql", "docs_legal", "custom"]);
    }
}
