//! Hardware detection: NVIDIA (nvidia-smi), AMD/Intel (sysfs on Linux, registry on Windows),
//! Apple Silicon unified memory (sysctl), CPU, RAM and free disk. Tier selection mirrors
//! engine/hardware_profile.py::select_tier and is tested against the same fixture.

use crate::contract::{GpuInfo, HardwareReport, Recommendation, TIERS_JSON};
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;
use std::process::Command;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Deserialize)]
struct Tier {
    id: String,
    vendor: String,
    min_gb: f64,
    max_gb: f64,
    model: String,
    backend: String,
    quant: String,
    lora_r: u32,
    micro_batch: u32,
    grad_accum: u32,
    max_seq: u32,
    stage2: String,
    rft_k: u32,
    plain: String,
}

#[derive(Deserialize)]
struct TierFile {
    models: serde_json::Map<String, Value>,
    tiers: Vec<Tier>,
}

fn tiers() -> TierFile {
    serde_json::from_str(TIERS_JSON).expect("contracts/tiers.json is valid")
}

/// (vendor, memory_gb) of the accelerator used for training.
pub fn primary_accelerator(gpus: &[GpuInfo], apple_silicon: bool, ram_gb: f64) -> (String, f64) {
    if apple_silicon {
        return ("apple".into(), ram_gb);
    }
    let rank = |v: &str| match v {
        "nvidia" => 0,
        "amd" => 1,
        "intel" => 2,
        _ => 9,
    };
    let mut best: Option<(String, f64)> = None;
    for g in gpus.iter().filter(|g| rank(&g.vendor) < 9) {
        let better = match &best {
            None => true,
            Some((bv, bm)) => g.vram_gb > *bm || (g.vram_gb == *bm && rank(&g.vendor) < rank(bv)),
        };
        if better {
            best = Some((g.vendor.clone(), g.vram_gb));
        }
    }
    best.unwrap_or(("cpu".into(), 0.0))
}

pub fn select_tier(gpus: &[GpuInfo], apple_silicon: bool, ram_gb: f64) -> Recommendation {
    let file = tiers();
    let (vendor, mem) = primary_accelerator(gpus, apple_silicon, ram_gb);
    let tier = file
        .tiers
        .iter()
        .find(|t| t.vendor == vendor && t.min_gb <= mem && mem < t.max_gb)
        .or_else(|| file.tiers.iter().find(|t| t.id == "X"))
        .expect("tier X exists");
    let label = file.models.get(&tier.model).and_then(|m| m.get("label")).and_then(Value::as_str).unwrap_or(&tier.model).to_string();
    Recommendation {
        tier: tier.id.clone(),
        vendor,
        memory_gb: (mem * 100.0).round() / 100.0,
        model_id: tier.model.clone(),
        model_label: label,
        backend: tier.backend.clone(),
        quant: tier.quant.clone(),
        lora_r: tier.lora_r,
        micro_batch: tier.micro_batch,
        grad_accum: tier.grad_accum,
        max_seq: tier.max_seq,
        stage2: tier.stage2.clone(),
        rft_k: tier.rft_k,
        plain: tier.plain.clone(),
    }
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let mut c = Command::new(cmd);
    c.args(args);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let out = c.output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).to_string())
}

/// Parse `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader,nounits`.
pub fn parse_nvidia_smi(text: &str) -> Vec<GpuInfo> {
    text.lines()
        .filter_map(|line| {
            let cols: Vec<&str> = line.split(',').map(str::trim).collect();
            if cols.len() < 3 {
                return None;
            }
            let total: f64 = cols[1].parse().ok()?;
            let free: Option<f64> = cols[2].parse().ok();
            Some(GpuInfo {
                vendor: "nvidia".into(),
                name: cols[0].to_string(),
                vram_gb: round2(total / 1024.0),
                vram_free_gb: free.map(|f| round2(f / 1024.0)),
                driver: cols.get(3).map(|s| s.to_string()),
                compute_capability: cols.get(4).map(|s| s.to_string()).filter(|s| !s.is_empty() && s != "[N/A]"),
                unified: false,
            })
        })
        .collect()
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn nvidia() -> Vec<GpuInfo> {
    run("nvidia-smi", &["--query-gpu=name,memory.total,memory.free,driver_version,compute_cap", "--format=csv,noheader,nounits"])
        .map(|t| parse_nvidia_smi(&t))
        .unwrap_or_default()
}

#[cfg(target_os = "linux")]
fn linux_sysfs_gpus() -> Vec<GpuInfo> {
    let mut out = vec![];
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else { return out };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let dev = e.path().join("device");
        let vendor = std::fs::read_to_string(dev.join("vendor")).unwrap_or_default();
        let vendor = match vendor.trim() {
            "0x1002" => "amd",
            "0x8086" => "intel",
            _ => continue,
        };
        let total = std::fs::read_to_string(dev.join("mem_info_vram_total")).ok().and_then(|s| s.trim().parse::<f64>().ok());
        let used = std::fs::read_to_string(dev.join("mem_info_vram_used")).ok().and_then(|s| s.trim().parse::<f64>().ok());
        if let Some(total) = total {
            if total < 1.0 * GIB {
                continue; // integrated graphics with a tiny carve-out
            }
            out.push(GpuInfo {
                vendor: vendor.into(),
                name: format!("{} GPU ({name})", if vendor == "amd" { "AMD" } else { "Intel" }),
                vram_gb: round2(total / GIB),
                vram_free_gb: used.map(|u| round2((total - u) / GIB)),
                driver: None,
                compute_capability: None,
                unified: false,
            });
        }
    }
    out
}

#[cfg(windows)]
fn windows_registry_gpus() -> Vec<GpuInfo> {
    let script = r#"Get-ItemProperty 'HKLM:\SYSTEM\ControlSet001\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0*' -ErrorAction SilentlyContinue | Select-Object DriverDesc,ProviderName,@{n='mem';e={$_.'HardwareInformation.qwMemorySize'}} | ConvertTo-Json -Compress"#;
    let Some(text) = run("powershell", &["-NoProfile", "-NonInteractive", "-Command", script]) else { return vec![] };
    let v: Value = serde_json::from_str(text.trim()).unwrap_or(Value::Null);
    let items = match v {
        Value::Array(a) => a,
        Value::Object(_) => vec![v],
        _ => vec![],
    };
    items
        .into_iter()
        .filter_map(|it| {
            let desc = it.get("DriverDesc")?.as_str()?.to_string();
            let provider = it.get("ProviderName").and_then(Value::as_str).unwrap_or("").to_lowercase();
            let mem = it.get("mem").and_then(Value::as_f64)?;
            let d = desc.to_lowercase();
            let vendor = if d.contains("nvidia") || provider.contains("nvidia") {
                return None; // nvidia-smi covers NVIDIA with accurate free memory
            } else if d.contains("amd") || d.contains("radeon") || provider.contains("advanced micro") {
                "amd"
            } else if d.contains("intel") || provider.contains("intel") {
                "intel"
            } else {
                "other"
            };
            if mem < 1.0 * GIB {
                return None;
            }
            Some(GpuInfo { vendor: vendor.into(), name: desc, vram_gb: round2(mem / GIB), vram_free_gb: None, driver: None, compute_capability: None, unified: false })
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn apple_info() -> (bool, Option<String>) {
    let arm = run("sysctl", &["-n", "hw.optional.arm64"]).map(|s| s.trim() == "1").unwrap_or(false);
    let chip = run("sysctl", &["-n", "machdep.cpu.brand_string"]).map(|s| s.trim().to_string());
    (arm, chip)
}

fn disk_free_gb(path: &Path) -> f64 {
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let target = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    disks
        .list()
        .iter()
        .filter(|d| target.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| round2(d.available_space() as f64 / GIB))
        .unwrap_or(0.0)
}

pub fn detect(data_dir: &Path) -> HardwareReport {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();
    let ram_gb = round2(sys.total_memory() as f64 / GIB);
    let mut cpu_brand = sys.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default();
    let cpu_cores = sys.physical_core_count().unwrap_or_else(|| sys.cpus().len()) as u32;
    let os = match std::env::consts::OS {
        "macos" => "macos",
        "windows" => "windows",
        "linux" => "linux",
        _ => "other",
    }
    .to_string();
    let arch = std::env::consts::ARCH.to_string();
    let mut gpus = nvidia();
    #[cfg(target_os = "linux")]
    gpus.extend(linux_sysfs_gpus());
    #[cfg(windows)]
    gpus.extend(windows_registry_gpus());
    #[allow(unused_mut)]
    let mut apple_silicon = false;
    #[cfg(target_os = "macos")]
    {
        let (arm, chip) = apple_info();
        apple_silicon = arm;
        if let Some(c) = chip {
            cpu_brand = c;
        }
        if arm {
            gpus.push(GpuInfo { vendor: "apple".into(), name: format!("{cpu_brand} GPU"), vram_gb: ram_gb, vram_free_gb: None, driver: None, compute_capability: None, unified: true });
        }
    }
    if cpu_brand.is_empty() {
        cpu_brand = arch.clone();
    }
    std::fs::create_dir_all(data_dir).ok();
    let disk_free_gb = disk_free_gb(data_dir);
    let recommendation = select_tier(&gpus, apple_silicon, ram_gb);
    let summary = plain_summary(&gpus, apple_silicon, ram_gb, disk_free_gb, &recommendation);
    HardwareReport { os, arch, cpu_brand, cpu_cores, ram_gb, disk_free_gb, gpus, apple_silicon, recommendation, summary }
}

fn plain_summary(gpus: &[GpuInfo], apple: bool, ram: f64, disk: f64, rec: &Recommendation) -> Vec<String> {
    let mut lines = vec![];
    if apple {
        lines.push(format!("Apple Silicon Mac with {ram:.0} GB of shared memory"));
    } else if let Some(g) = gpus.iter().filter(|g| g.vendor != "other").max_by(|a, b| a.vram_gb.total_cmp(&b.vram_gb)) {
        lines.push(format!("{} with {:.0} GB of graphics memory", g.name, g.vram_gb));
    } else {
        lines.push("No supported graphics card found; training will run on the processor (slow)".into());
    }
    lines.push(format!("{ram:.0} GB of system memory"));
    lines.push(if disk >= 30.0 { format!("{disk:.0} GB free on disk") } else { format!("Only {disk:.0} GB free on disk; at least 30 GB is recommended") });
    lines.push(format!("Recommended AI brain: {} ({})", rec.model_label, rec.plain));
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_cases_match_python() {
        let cases: Value = serde_json::from_str(include_str!("../../contracts/fixtures/tier_cases.json")).unwrap();
        for c in cases["cases"].as_array().unwrap() {
            let gpus: Vec<GpuInfo> = c["gpus"]
                .as_array()
                .unwrap()
                .iter()
                .map(|g| GpuInfo { vendor: g["vendor"].as_str().unwrap().into(), name: "x".into(), vram_gb: g["vram_gb"].as_f64().unwrap(), vram_free_gb: None, driver: None, compute_capability: None, unified: g["unified"].as_bool().unwrap() })
                .collect();
            let rec = select_tier(&gpus, c["apple_silicon"].as_bool().unwrap(), c["ram_gb"].as_f64().unwrap());
            assert_eq!(rec.tier, c["expect"].as_str().unwrap(), "case {}", c["name"]);
        }
    }

    #[test]
    fn parses_nvidia_smi() {
        let g = parse_nvidia_smi("NVIDIA GeForce RTX 4090, 24564, 23012, 572.16, 8.9\nNVIDIA GeForce RTX 3060, 12288, 11000, 572.16, 8.6\n");
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].vram_gb, 23.99);
        assert_eq!(g[0].compute_capability.as_deref(), Some("8.9"));
        assert_eq!(select_tier(&g, false, 64.0).tier, "N24");
    }

    #[test]
    fn detect_runs_here() {
        let dir = tempfile::tempdir().unwrap();
        let r = detect(dir.path());
        assert!(r.ram_gb > 0.0);
        assert!(!r.recommendation.tier.is_empty());
        assert!(r.summary.len() >= 3);
    }
}
