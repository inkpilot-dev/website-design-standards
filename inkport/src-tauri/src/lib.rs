//! Inkport desktop supervisor.
//!
//! Layers: [`hardware`] detection -> [`runtime`] bootstrap -> [`sidecar`] engine supervision
//! (NDJSON events) -> [`api_server`] local OpenAI-compatible gateway, tied together by
//! [`app::App`], which both the Tauri commands ([`desktop`]) and the headless CLI use.

pub mod api_server;
pub mod app;
pub mod bootstrap;
pub mod contract;
pub mod events;
pub mod hardware;
pub mod packs;
pub mod procguard;
pub mod project;
pub mod runtime;
pub mod sidecar;

#[cfg(feature = "desktop")]
pub mod desktop;
