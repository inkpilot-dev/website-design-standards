//! Embeds the SHA-256 of the bundled payload manifest into the binary, so a signed app can tell
//! whether the files installed next to it are exactly the ones it was built with.
//! Source: $INKPORT_MANIFEST_SHA256, else build/payload-<platform>.manifest.sha256 written by
//! scripts/lib/finalize_payload.py, else empty (development builds).
use std::path::Path;

fn main() {
    let platform = match (std::env::var("CARGO_CFG_TARGET_OS").as_deref(), std::env::var("CARGO_CFG_TARGET_ARCH").as_deref()) {
        (Ok("linux"), Ok("x86_64")) => "linux-x86_64",
        (Ok("macos"), Ok("aarch64")) => "macos-aarch64",
        (Ok("windows"), Ok("x86_64")) => "windows-x86_64",
        _ => "unsupported",
    };
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("build").join(format!("payload-{platform}.manifest.sha256"));
    println!("cargo:rerun-if-env-changed=INKPORT_MANIFEST_SHA256");
    println!("cargo:rerun-if-changed={}", file.display());
    let sha = std::env::var("INKPORT_MANIFEST_SHA256").ok().filter(|s| !s.trim().is_empty()).or_else(|| std::fs::read_to_string(&file).ok()).map(|s| s.trim().to_string()).unwrap_or_default();
    if !sha.is_empty() && (sha.len() != 64 || !sha.chars().all(|c| c.is_ascii_hexdigit())) {
        panic!("INKPORT_MANIFEST_SHA256 must be a 64-character hex SHA-256, got {sha:?}");
    }
    println!("cargo:rustc-env=INKPORT_EXPECTED_MANIFEST_SHA256={sha}");
    #[cfg(feature = "desktop")]
    tauri_build::build();
}
