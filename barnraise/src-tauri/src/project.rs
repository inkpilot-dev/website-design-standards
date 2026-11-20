//! Project folders: <home>/projects/<slug>/project.json (same layout the engine reads).

use crate::contract::ProjectInfo;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub fn projects_dir(home: &Path) -> PathBuf {
    home.join("projects")
}

pub fn slugify(name: &str) -> String {
    let mut s = String::new();
    let mut dash = false;
    for ch in name.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            s.push(ch);
            dash = false;
        } else if !dash && !s.is_empty() {
            s.push('-');
            dash = true;
        }
    }
    let s = s.trim_matches('-').chars().take(48).collect::<String>();
    if s.is_empty() {
        "project".into()
    } else {
        s
    }
}

pub fn project_dir(home: &Path, id_or_slug: &str) -> Result<PathBuf> {
    let dir = projects_dir(home).join(slugify(id_or_slug));
    if dir.join("project.json").is_file() {
        return Ok(dir);
    }
    for p in list(home)? {
        if p.id == id_or_slug || p.slug == id_or_slug {
            return Ok(PathBuf::from(p.dir));
        }
    }
    bail!("project '{id_or_slug}' not found")
}

#[allow(clippy::too_many_arguments)]
pub fn create(home: &Path, name: &str, template: &str, base_id: &str, base_path: &Path, chat_kwargs: Value, backend: Option<&str>, quant: Option<&str>) -> Result<ProjectInfo> {
    if !crate::contract::templates().iter().any(|t| t.id == template) {
        bail!("unknown template '{template}'");
    }
    let slug = slugify(name);
    let dir = projects_dir(home).join(&slug);
    if dir.join("project.json").exists() {
        bail!("A project named '{name}' already exists");
    }
    std::fs::create_dir_all(&dir)?;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs_f64();
    let project = json!({
        "id": uuid::Uuid::new_v4().simple().to_string()[..12],
        "slug": slug,
        "name": name,
        "template": template,
        "created_at": now,
        "base_model": {"id": base_id, "path": base_path.display().to_string(), "chat_kwargs": chat_kwargs},
        "backend": backend,
        "quant": quant,
        "active_version": null,
    });
    std::fs::write(dir.join("project.json"), serde_json::to_vec_pretty(&project)?)?;
    read(&dir)
}

fn read_json(p: &Path) -> Option<Value> {
    std::fs::read(p).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

pub fn read(dir: &Path) -> Result<ProjectInfo> {
    let p = read_json(&dir.join("project.json")).with_context(|| format!("unreadable project at {}", dir.display()))?;
    let active = read_json(&dir.join("active.json")).and_then(|a| a["version"].as_str().map(String::from));
    let counts = read_json(&dir.join("data").join("manifest.json")).map(|m| m["counts"].clone());
    let latest_report = active.as_ref().and_then(|v| read_json(&dir.join("versions").join(v).join("report.json"))).or_else(|| {
        let runs = std::fs::read_dir(dir.join("runs")).ok()?;
        let newest = runs.flatten().filter(|e| e.path().join("report.json").is_file()).max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())?;
        read_json(&newest.path().join("report.json"))
    });
    Ok(ProjectInfo {
        id: p["id"].as_str().unwrap_or_default().to_string(),
        slug: p["slug"].as_str().unwrap_or_default().to_string(),
        name: p["name"].as_str().unwrap_or_default().to_string(),
        template: p["template"].as_str().unwrap_or_default().to_string(),
        dir: dir.display().to_string(),
        base_model: p["base_model"].clone(),
        backend: p["backend"].as_str().map(String::from),
        quant: p["quant"].as_str().map(String::from),
        active_version: active,
        counts,
        latest_report,
    })
}

pub fn list(home: &Path) -> Result<Vec<ProjectInfo>> {
    let mut out = vec![];
    if let Ok(entries) = std::fs::read_dir(projects_dir(home)) {
        for e in entries.flatten() {
            if e.path().join("project.json").is_file() {
                if let Ok(p) = read(&e.path()) {
                    out.push(p);
                }
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_matches_engine() {
        assert_eq!(slugify("Expense Sorter"), "expense-sorter");
        assert_eq!(slugify("  Ohio PT -- Claims!! "), "ohio-pt-claims");
        assert_eq!(slugify("***"), "project");
    }

    #[test]
    fn create_and_list() {
        let home = tempfile::tempdir().unwrap();
        let p = create(home.path(), "My Books", "business_rules", "qwen3-0.6b", Path::new("/models/x"), json!({}), Some("torch"), None).unwrap();
        assert_eq!(p.slug, "my-books");
        assert!(create(home.path(), "My Books", "business_rules", "x", Path::new("/m"), json!({}), None, None).is_err());
        assert!(create(home.path(), "Other", "nope", "x", Path::new("/m"), json!({}), None, None).is_err());
        assert_eq!(list(home.path()).unwrap().len(), 1);
        assert_eq!(project_dir(home.path(), &p.id).unwrap(), PathBuf::from(&p.dir));
    }
}
