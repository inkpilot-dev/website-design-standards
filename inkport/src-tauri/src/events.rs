//! Event sinks: the Tauri window in the desktop app, stdout in the headless CLI.

use serde_json::Value;
use std::sync::Mutex;

pub trait EventSink: Send + Sync + 'static {
    fn emit(&self, channel: &str, payload: &Value);
}

/// Prints `[channel] {...}` lines; with `quiet`, only logs, stages, evals, practice answers, gates and errors.
pub struct StdoutSink {
    pub raw: bool,
    lock: Mutex<()>,
}

impl StdoutSink {
    pub fn new(raw: bool) -> Self {
        Self { raw, lock: Mutex::new(()) }
    }
}

impl EventSink for StdoutSink {
    fn emit(&self, channel: &str, payload: &Value) {
        let _g = self.lock.lock().unwrap();
        if self.raw {
            println!("[{channel}] {payload}");
            return;
        }
        let kind = payload.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
            "log" => {
                let level = payload["level"].as_str().unwrap_or("info");
                if level != "debug" {
                    eprintln!("  {level:>5} | {}", payload["message"].as_str().unwrap_or(""));
                }
            }
            "stage" => eprintln!(
                "  stage | {} {} {}",
                payload["stage"].as_str().unwrap_or(""),
                payload["status"].as_str().unwrap_or(""),
                payload.get("detail").and_then(Value::as_str).unwrap_or("")
            ),
            "eval" => eprintln!(
                "   eval | {} {}: {}/{} passed",
                payload["model"].as_str().unwrap_or(""),
                payload["split"].as_str().unwrap_or(""),
                payload["passed"],
                payload["n"]
            ),
            "gate" => eprintln!("   gate | {} - {}", payload["decision"].as_str().unwrap_or(""), payload["explanation"].as_str().unwrap_or("")),
            // What the model being trained answers on a few practice questions, each time it is
            // checked: the quickest way to see a run that has broken (empty or garbled answers).
            "benchmark" if payload["source"].as_str() == Some("training_dev") => {
                let answer: String = payload["trained"]["output"].as_str().unwrap_or("").split_whitespace().collect::<Vec<_>>().join(" ");
                let shown: String = answer.chars().take(110).collect();
                eprintln!(
                    "    dev | step {}: {} | {}{}",
                    payload["step"],
                    if payload["trained"]["pass"].as_bool() == Some(true) { "pass" } else { "FAIL" },
                    shown,
                    if answer.chars().count() > 110 { "…" } else { "" }
                );
            }
            "error" => eprintln!("  error | {}: {}", payload["code"].as_str().unwrap_or(""), payload["message"].as_str().unwrap_or("")),
            "ready" => eprintln!("  ready | worker on 127.0.0.1:{}", payload["port"]),
            "runtime" => eprintln!("runtime | {}", payload.get("label").and_then(Value::as_str).unwrap_or("")),
            _ => {}
        }
    }
}

/// Collects events in memory (tests).
#[derive(Default)]
pub struct MemorySink {
    pub events: Mutex<Vec<(String, Value)>>,
}

impl EventSink for MemorySink {
    fn emit(&self, channel: &str, payload: &Value) {
        self.events.lock().unwrap().push((channel.to_string(), payload.clone()));
    }
}
