//! Local OpenAI-compatible gateway on http://127.0.0.1:11435/v1.
//!
//! Security model (localhost is not a trust boundary on its own):
//! * binds 127.0.0.1 only;
//! * `Authorization: Bearer <token>` required on /v1/* (token file in the Inkport home dir,
//!   created with 0600 permissions on Unix);
//! * Host header must be 127.0.0.1/localhost/[::1] (blocks DNS-rebinding attacks);
//! * browser Origins are rejected unless allow-listed (the desktop app, browser extensions);
//! * the upstream Python worker has its own secret and listens on an ephemeral port.
//! Streaming responses (SSE) are proxied byte-for-byte.

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::{oneshot, RwLock};

pub const DEFAULT_PORT: u16 = 11435;
const MAX_BODY: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub enum UpstreamKind {
    /// The Python inference worker (PyTorch / MLX); authenticated with X-Inkport-Worker-Token.
    Python,
    /// The bundled llama.cpp server (Metal / CUDA / Vulkan / CPU); authenticated with a bearer key.
    /// `has_lora`: a trained adapter is loaded, so `<model>@base` turns it off per request.
    Llama { has_lora: bool },
}

#[derive(Clone, Debug)]
pub struct Upstream {
    pub base_url: String,
    pub worker_token: String,
    pub models: Vec<String>,
    pub project_id: String,
    pub version: String,
    pub kind: UpstreamKind,
    /// Inserted when a request has no system message (the prompt the model was trained with).
    pub default_system: Option<String>,
}

#[derive(Default)]
struct GwState {
    upstream: Option<Upstream>,
}

#[derive(Clone)]
struct Shared {
    state: Arc<RwLock<GwState>>,
    token: Arc<String>,
    port: u16,
    client: reqwest::Client,
    allowed_origins: Arc<Vec<String>>,
}

pub struct Gateway {
    state: Arc<RwLock<GwState>>,
    pub token: String,
    pub port: RwLock<Option<u16>>,
    shutdown: tokio::sync::Mutex<Option<oneshot::Sender<()>>>,
}

pub fn load_or_create_token(home: &Path) -> std::io::Result<String> {
    let path = home.join("api_token");
    if let Ok(t) = std::fs::read_to_string(&path) {
        let t = t.trim().to_string();
        if t.len() >= 32 {
            return Ok(t);
        }
    }
    std::fs::create_dir_all(home)?;
    let token = format!("ik_{}{}", uuid::Uuid::new_v4().simple(), uuid::Uuid::new_v4().simple());
    std::fs::write(&path, &token)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(token)
}

impl Gateway {
    pub fn new(token: String) -> Self {
        Self { state: Arc::new(RwLock::new(GwState::default())), token, port: RwLock::new(None), shutdown: tokio::sync::Mutex::new(None) }
    }

    pub async fn set_upstream(&self, up: Option<Upstream>) {
        self.state.write().await.upstream = up;
    }

    pub async fn upstream(&self) -> Option<Upstream> {
        self.state.read().await.upstream.clone()
    }

    pub async fn is_running(&self) -> bool {
        self.port.read().await.is_some()
    }

    /// Bind and serve in the background. Returns the bound port (pass 0 for an ephemeral port in tests).
    pub async fn start(&self, port: u16) -> anyhow::Result<u16> {
        if let Some(p) = *self.port.read().await {
            return Ok(p);
        }
        let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], port)))
            .await
            .map_err(|e| anyhow::anyhow!("PORT_IN_USE: could not listen on 127.0.0.1:{port} ({e}). Another app may be using it."))?;
        let bound = listener.local_addr()?.port();
        let shared = Shared {
            state: Arc::clone(&self.state),
            token: Arc::new(self.token.clone()),
            port: bound,
            client: reqwest::Client::builder().no_proxy().build()?,
            allowed_origins: Arc::new(vec![
                "tauri://localhost".into(),
                "http://tauri.localhost".into(),
                "https://tauri.localhost".into(),
                "http://localhost:1420".into(),
                "chrome-extension://".into(),
                "moz-extension://".into(),
                "safari-web-extension://".into(),
            ]),
        };
        let app = Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(models).options(preflight))
            .route("/v1/chat/completions", any(chat))
            .fallback(not_found)
            .with_state(shared);
        let (tx, rx) = oneshot::channel::<()>();
        *self.shutdown.lock().await = Some(tx);
        *self.port.write().await = Some(bound);
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
        });
        Ok(bound)
    }

    pub async fn stop(&self) {
        if let Some(tx) = self.shutdown.lock().await.take() {
            let _ = tx.send(());
        }
        *self.port.write().await = None;
        self.set_upstream(None).await;
    }
}

fn err(status: StatusCode, code: &str, msg: &str) -> Response {
    (status, Json(json!({"error": {"message": msg, "type": code, "code": code}}))).into_response()
}

/// Host must be loopback; Origin (if any) must be allow-listed. Returns the CORS origin to echo.
fn check_origin(h: &HeaderMap, s: &Shared) -> Result<Option<HeaderValue>, Response> {
    let host = h.get(header::HOST).and_then(|v| v.to_str().ok()).unwrap_or("");
    let host_name = host.rsplit_once(':').map(|(n, _)| n).unwrap_or(host);
    if !matches!(host_name, "127.0.0.1" | "localhost" | "[::1]") {
        return Err(err(StatusCode::FORBIDDEN, "forbidden_host", "Requests must be addressed to 127.0.0.1 or localhost"));
    }
    match h.get(header::ORIGIN).and_then(|v| v.to_str().ok()) {
        None => Ok(None),
        Some(o) if s.allowed_origins.iter().any(|a| if a.ends_with("://") { o.starts_with(a.as_str()) } else { o == a }) => Ok(HeaderValue::from_str(o).ok()),
        Some(o) if o == format!("http://127.0.0.1:{}", s.port) || o == format!("http://localhost:{}", s.port) => Ok(HeaderValue::from_str(o).ok()),
        Some(_) => Err(err(StatusCode::FORBIDDEN, "forbidden_origin", "This website is not allowed to use your local model")),
    }
}

fn check_auth(h: &HeaderMap, s: &Shared) -> Result<(), Response> {
    let got = h.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")).unwrap_or("");
    if constant_time_eq(got.as_bytes(), s.token.as_bytes()) {
        Ok(())
    } else {
        Err(err(StatusCode::UNAUTHORIZED, "invalid_api_key", "Missing or wrong API key. Copy it from the Inkport app."))
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn with_cors(mut resp: Response, origin: Option<HeaderValue>) -> Response {
    if let Some(o) = origin {
        let h = resp.headers_mut();
        h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, o);
        h.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, HeaderValue::from_static("authorization, content-type"));
        h.insert(header::ACCESS_CONTROL_ALLOW_METHODS, HeaderValue::from_static("GET, POST, OPTIONS"));
        h.insert(header::VARY, HeaderValue::from_static("Origin"));
    }
    resp
}

async fn health(State(s): State<Shared>, h: HeaderMap) -> Response {
    if let Err(r) = check_origin(&h, &s) {
        return r;
    }
    let up = s.state.read().await.upstream.clone();
    Json(json!({"ok": true, "model_loaded": up.is_some(), "models": up.map(|u| u.models).unwrap_or_default()})).into_response()
}

async fn preflight(State(s): State<Shared>, h: HeaderMap) -> Response {
    match check_origin(&h, &s) {
        Ok(o) => with_cors(StatusCode::NO_CONTENT.into_response(), o),
        Err(r) => r,
    }
}

async fn models(State(s): State<Shared>, h: HeaderMap) -> Response {
    let origin = match check_origin(&h, &s) {
        Ok(o) => o,
        Err(r) => return r,
    };
    if let Err(r) = check_auth(&h, &s) {
        return with_cors(r, origin);
    }
    let up = s.state.read().await.upstream.clone();
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let data: Vec<Value> = up.map(|u| u.models).unwrap_or_default().into_iter().map(|id| json!({"id": id, "object": "model", "created": now, "owned_by": "inkport"})).collect();
    with_cors(Json(json!({"object": "list", "data": data})).into_response(), origin)
}

async fn chat(State(s): State<Shared>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let origin = match check_origin(&parts.headers, &s) {
        Ok(o) => o,
        Err(r) => return r,
    };
    if parts.method == Method::OPTIONS {
        return with_cors(StatusCode::NO_CONTENT.into_response(), origin);
    }
    if parts.method != Method::POST {
        return with_cors(err(StatusCode::METHOD_NOT_ALLOWED, "method_not_allowed", "use POST"), origin);
    }
    if let Err(r) = check_auth(&parts.headers, &s) {
        return with_cors(r, origin);
    }
    let Some(up) = s.state.read().await.upstream.clone() else {
        return with_cors(err(StatusCode::SERVICE_UNAVAILABLE, "model_not_loaded", "No model is running. Press 'Start local API' in Inkport."), origin);
    };
    let bytes = match axum::body::to_bytes(body, MAX_BODY).await {
        Ok(b) => b,
        Err(_) => return with_cors(err(StatusCode::PAYLOAD_TOO_LARGE, "too_large", "request body too large"), origin),
    };
    let mut payload: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => return with_cors(err(StatusCode::BAD_REQUEST, "invalid_json", "body must be JSON"), origin),
    };
    // Map friendly aliases onto the served model ids.
    let requested = payload.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    let model = if requested.is_empty() || requested == "inkport" || requested == "default" { up.models[0].clone() } else { requested };
    if !up.models.contains(&model) {
        return with_cors(err(StatusCode::NOT_FOUND, "model_not_found", &format!("Unknown model '{model}'. Available: {}", up.models.join(", "))), origin);
    }
    payload["model"] = json!(model);
    let mut request = s.client.post(format!("{}/v1/chat/completions", up.base_url));
    if let UpstreamKind::Llama { has_lora } = up.kind {
        if let (Some(sys), Some(msgs)) = (&up.default_system, payload.get_mut("messages").and_then(Value::as_array_mut)) {
            if !msgs.iter().any(|m| m["role"] == "system") {
                msgs.insert(0, json!({"role": "system", "content": sys}));
            }
        }
        if has_lora && model.ends_with("@base") {
            payload["lora"] = json!([{"id": 0, "scale": 0.0}]);
        }
        if payload.get("max_tokens").is_none() && payload.get("max_completion_tokens").is_none() {
            payload["max_tokens"] = json!(512);
        }
        request = request.bearer_auth(&up.worker_token);
    } else {
        request = request.header("X-Inkport-Worker-Token", &up.worker_token);
    }
    let resp = request.json(&payload).send().await;
    let resp = match resp {
        Ok(r) => r,
        Err(e) => return with_cors(err(StatusCode::BAD_GATEWAY, "worker_unreachable", &format!("The model worker is not responding: {e}")), origin),
    };
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let ctype = resp.headers().get(reqwest::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("application/json").to_string();
    let stream = resp.bytes_stream();
    let mut out = Response::new(Body::from_stream(stream));
    *out.status_mut() = status;
    out.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_str(&ctype).unwrap_or(HeaderValue::from_static("application/json")));
    if ctype.starts_with("text/event-stream") {
        out.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    }
    with_cors(out, origin)
}

async fn not_found() -> Response {
    err(StatusCode::NOT_FOUND, "not_found", "Inkport serves /v1/models and /v1/chat/completions")
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn fake_worker() -> String {
        async fn chat(h: HeaderMap, Json(body): Json<Value>) -> Response {
            let python_ok = h.get("X-Inkport-Worker-Token").and_then(|v| v.to_str().ok()) == Some("wt");
            let llama_ok = h.get("authorization").and_then(|v| v.to_str().ok()) == Some("Bearer wt");
            if !python_ok && !llama_ok {
                return StatusCode::UNAUTHORIZED.into_response();
            }
            if body["stream"].as_bool() == Some(true) {
                let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";
                return ([(header::CONTENT_TYPE, "text/event-stream")], sse).into_response();
            }
            Json(json!({"object":"chat.completion","model":body["model"],"echo":body,"choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}]})).into_response()
        }
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(l, Router::new().route("/v1/chat/completions", axum::routing::post(chat))).await.unwrap() });
        format!("http://127.0.0.1:{port}")
    }

    #[tokio::test]
    async fn auth_host_origin_and_proxy() {
        let gw = Gateway::new("tok_0123456789abcdef0123456789abcdef".into());
        let port = gw.start(0).await.unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let c = reqwest::Client::builder().no_proxy().build().unwrap();
        let tok = "Bearer tok_0123456789abcdef0123456789abcdef";
        // No model yet -> 503 (after auth).
        let r = c.post(format!("{base}/v1/chat/completions")).header("Authorization", tok).json(&json!({"messages":[]})).send().await.unwrap();
        assert_eq!(r.status(), 503);
        gw.set_upstream(Some(Upstream { base_url: fake_worker().await, worker_token: "wt".into(), models: vec!["inkport/x".into(), "inkport/x@base".into()], project_id: "p".into(), version: "v1".into(), kind: UpstreamKind::Python, default_system: None })).await;
        assert_eq!(c.get(format!("{base}/v1/models")).send().await.unwrap().status(), 401);
        assert_eq!(c.get(format!("{base}/v1/models")).header("Authorization", "Bearer nope").send().await.unwrap().status(), 401);
        let r = c.get(format!("{base}/v1/models")).header("Authorization", tok).send().await.unwrap();
        assert_eq!(r.status(), 200);
        assert_eq!(r.json::<Value>().await.unwrap()["data"][0]["id"], "inkport/x");
        // DNS rebinding and hostile web origins are refused.
        assert_eq!(c.get(format!("{base}/v1/models")).header("Authorization", tok).header("Host", "evil.example").send().await.unwrap().status(), 403);
        assert_eq!(c.get(format!("{base}/v1/models")).header("Authorization", tok).header("Origin", "https://evil.example").send().await.unwrap().status(), 403);
        let r = c.get(format!("{base}/v1/models")).header("Authorization", tok).header("Origin", "chrome-extension://abc").send().await.unwrap();
        assert_eq!(r.headers()["access-control-allow-origin"], "chrome-extension://abc");
        // Proxy (plain + streaming) and unknown-model handling.
        let r: Value = c.post(format!("{base}/v1/chat/completions")).header("Authorization", tok).json(&json!({"model":"inkport","messages":[{"role":"user","content":"ping"}]})).send().await.unwrap().json().await.unwrap();
        assert_eq!(r["choices"][0]["message"]["content"], "pong");
        assert_eq!(r["model"], "inkport/x");
        let r = c.post(format!("{base}/v1/chat/completions")).header("Authorization", tok).json(&json!({"model":"inkport/x","stream":true,"messages":[]})).send().await.unwrap();
        assert!(r.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
        assert!(r.text().await.unwrap().contains("[DONE]"));
        assert_eq!(c.post(format!("{base}/v1/chat/completions")).header("Authorization", tok).json(&json!({"model":"gpt-4","messages":[]})).send().await.unwrap().status(), 404);
        // llama.cpp upstream: bearer auth, trained system prompt, adapter switched off for @base.
        let mut up = gw.upstream().await.unwrap();
        up.kind = UpstreamKind::Llama { has_lora: true };
        up.default_system = Some("SYS".into());
        gw.set_upstream(Some(up)).await;
        let r: Value = c.post(format!("{base}/v1/chat/completions")).header("Authorization", tok).json(&json!({"model":"inkport/x@base","messages":[{"role":"user","content":"hi"}]})).send().await.unwrap().json().await.unwrap();
        assert_eq!(r["echo"]["messages"][0]["content"], "SYS");
        assert_eq!(r["echo"]["lora"][0]["scale"], 0.0);
        assert_eq!(r["echo"]["max_tokens"], 512);
        let r: Value = c.post(format!("{base}/v1/chat/completions")).header("Authorization", tok).json(&json!({"model":"inkport/x","messages":[{"role":"system","content":"mine"},{"role":"user","content":"hi"}]})).send().await.unwrap().json().await.unwrap();
        assert_eq!(r["echo"]["messages"][0]["content"], "mine");
        assert!(r["echo"].get("lora").is_none());
        gw.stop().await;
    }
}
