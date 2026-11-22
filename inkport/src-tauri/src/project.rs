//! Project folders: <home>/projects/<slug>/project.json (same layout the engine reads).

use crate::contract::ProjectInfo;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub fn projects_dir(home: &Path) -> PathBuf {
    home.join("projects")
}

/// Folder-safe, URL-safe name: lower-case ASCII letters, digits and single dashes.
/// Accents and other scripts are transliterated ("Résumé" -> "resume", "経費" -> "jing-fei") so
/// names that are not written in Latin letters still get distinct, readable folders.
pub fn slugify(name: &str) -> String {
    let mut s = String::new();
    let mut dash = false;
    // Only letters and digits are transliterated; punctuation, symbols and emoji separate words.
    let latin: String = name.chars().map(|c| if c.is_alphanumeric() { deunicode::deunicode_char(c).unwrap_or("").to_string() } else { " ".to_string() }).collect::<Vec<_>>().join("");
    for ch in latin.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            s.push(ch);
            dash = false;
        } else if !dash && !s.is_empty() {
            s.push('-');
            dash = true;
        }
    }
    let s = s.chars().take(48).collect::<String>().trim_matches('-').to_string();
    if s.is_empty() {
        "project".into()
    } else if is_reserved_on_windows(&s) {
        format!("{s}-project")
    } else {
        s
    }
}

/// Device names Windows refuses as file or folder names (CON, NUL, COM1, ...), in any letter case.
fn is_reserved_on_windows(s: &str) -> bool {
    matches!(s, "con" | "prn" | "aux" | "nul") || ((s.starts_with("com") || s.starts_with("lpt")) && s.len() == 4 && s.as_bytes()[3].is_ascii_digit())
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

/// `base_model` is the full project.json entry: {id, path, chat_kwargs, [gguf, format, label, bundled, pack]}.
pub fn create(home: &Path, name: &str, template: &str, base_model: Value, backend: Option<&str>, quant: Option<&str>) -> Result<ProjectInfo> {
    if !crate::contract::templates().iter().any(|t| t.id == template) {
        bail!("unknown template '{template}'");
    }
    let name = name.trim();
    if name.is_empty() {
        bail!("BAD_INPUT: Give the project a name");
    }
    // Two different names can shorten to the same folder name ("My Books" / "my-books!"); those get
    // a numbered folder. Only a project with the same name (ignoring case) is a real duplicate.
    let base_slug = slugify(name);
    let mut slug = base_slug.clone();
    let mut n = 1;
    loop {
        let dir = projects_dir(home).join(&slug);
        if !dir.exists() {
            break;
        }
        let existing = read_json(&dir.join("project.json")).and_then(|p| p["name"].as_str().map(|s| s.trim().to_lowercase()));
        if existing.as_deref() == Some(name.to_lowercase().as_str()) {
            bail!("PROJECT_EXISTS: A project named '{name}' already exists. Open it from the list of projects, or choose another name.");
        }
        n += 1;
        if n > 200 {
            bail!("PROJECT_EXISTS: Too many projects with names like '{name}'. Choose a different name.");
        }
        slug = format!("{base_slug}-{n}");
    }
    let dir = projects_dir(home).join(&slug);
    std::fs::create_dir_all(&dir).with_context(|| format!("could not create the project folder {}", dir.display()))?;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs_f64();
    let project = json!({
        "id": uuid::Uuid::new_v4().simple().to_string()[..12],
        "slug": slug,
        "name": name,
        "template": template,
        "created_at": now,
        "base_model": base_model,
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
    fn slugify_other_scripts_accents_and_reserved_names() {
        assert_eq!(slugify("Résumé – été"), "resume-ete");
        assert_ne!(slugify("経費"), "project");
        assert_ne!(slugify("Расходы"), "project");
        assert_ne!(slugify("経費"), slugify("Расходы"));
        assert_eq!(slugify("Expenses 💸 2026"), "expenses-2026");
        for reserved in ["CON", "nul", "Aux", "COM1", "lpt9", "PRN"] {
            let s = slugify(reserved);
            assert!(s.ends_with("-project"), "{reserved} -> {s}");
        }
        assert_eq!(slugify("command"), "command");
        assert_eq!(slugify("a/b\\c:d*e?"), "a-b-c-d-e");
        assert!(slugify(&"long name ".repeat(30)).len() <= 48);
        assert!(!slugify(&"x ".repeat(40)).ends_with('-'));
    }

    #[test]
    fn similar_names_get_their_own_folders_and_true_duplicates_are_refused() {
        let home = tempfile::tempdir().unwrap();
        let base = json!({"id": "m", "path": "/m", "chat_kwargs": {}});
        let a = create(home.path(), "My Books", "custom", base.clone(), None, None).unwrap();
        let b = create(home.path(), "MY-BOOKS!", "custom", base.clone(), None, None).unwrap();
        assert_eq!((a.slug.as_str(), b.slug.as_str()), ("my-books", "my-books-2"));
        let dup = create(home.path(), "  my books ", "custom", base.clone(), None, None).unwrap_err().to_string();
        assert!(dup.starts_with("PROJECT_EXISTS: "), "{dup}");
        assert!(create(home.path(), "   ", "custom", base.clone(), None, None).unwrap_err().to_string().starts_with("BAD_INPUT: "));
        let c = create(home.path(), "経費", "custom", base.clone(), None, None).unwrap();
        let d = create(home.path(), "Расходы", "custom", base.clone(), None, None).unwrap();
        assert_ne!(c.slug, d.slug);
        assert_eq!(c.name, "経費");
        assert_eq!(list(home.path()).unwrap().len(), 4);
        assert_eq!(project_dir(home.path(), &b.id).unwrap(), PathBuf::from(&b.dir));
        assert_eq!(project_dir(home.path(), "my-books-2").unwrap(), PathBuf::from(&b.dir));
    }

    #[test]
    fn create_and_list() {
        let home = tempfile::tempdir().unwrap();
        let base = json!({"id": "qwen3-1.7b", "path": "/models/x", "gguf": "m.gguf", "chat_kwargs": {}});
        let p = create(home.path(), "My Books", "business_rules", base.clone(), Some("torch"), None).unwrap();
        assert_eq!(p.slug, "my-books");
        assert_eq!(p.base_model["gguf"], "m.gguf");
        assert!(create(home.path(), "My Books", "business_rules", base.clone(), None, None).is_err());
        assert!(create(home.path(), "Other", "nope", base, None, None).is_err());
        assert_eq!(list(home.path()).unwrap().len(), 1);
        assert_eq!(project_dir(home.path(), &p.id).unwrap(), PathBuf::from(&p.dir));
    }
}
