// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! HTTP server: OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/models`,
//! `/v1/responses`), Anthropic (`/v1/messages`, `/v1/messages/count_tokens`) and control
//! endpoints (`/health`, `/v1/stats`, `/v1/requests`, `/v1/cache/*`, `/generate`,
//! `/v1/admin/prepare-stop`).

pub mod anthropic;
pub mod effort;
pub mod inline_system;
pub mod metrics;
pub mod openai;
pub mod parsers;
pub mod request_log;
pub mod responses;

use crate::images::ImageSpan;
use crate::sampling::SamplingParams;
use crate::scheduler::{EngineHandle, Event, FinishReason, Usage};
use crate::tokenizer::{DecodeStream, StopChecker, TokenizerBundle};
use axum::extract::{ConnectInfo, Query, Request, State};
use axum::http::{HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router};
use effort::ThinkingProfile;
use inline_system::InlineSystem;
use metrics::{Metrics, RequestRecord};
use serde_json::{json, Map, Value};
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

#[derive(Clone, Debug, Default)]
pub struct SamplingDefaults {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

impl SamplingDefaults {
    /// Read `generation_config.json` (the checkpoint's recommended sampling).
    pub fn from_model_dir(dir: &Path) -> Self {
        let v: Value = std::fs::read_to_string(dir.join("generation_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(Value::Null);
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64()).map(|x| x as f32);
        let sample = v.get("do_sample").and_then(|x| x.as_bool()).unwrap_or(true);
        Self {
            temperature: if sample { f("temperature") } else { Some(0.0) },
            top_p: f("top_p"),
            top_k: v.get("top_k").and_then(|x| x.as_i64()).map(|x| x as i32),
            min_p: f("min_p"),
            repetition_penalty: f("repetition_penalty"),
        }
    }
}

impl SamplingDefaults {
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        if let Some(t) = self.temperature {
            m.insert("temperature".into(), json!(t));
        }
        if let Some(t) = self.top_p {
            m.insert("top_p".into(), json!(t));
        }
        if let Some(t) = self.top_k {
            m.insert("top_k".into(), json!(t));
        }
        if let Some(t) = self.min_p {
            m.insert("min_p".into(), json!(t));
        }
        if let Some(t) = self.repetition_penalty {
            m.insert("repetition_penalty".into(), json!(t));
        }
        Value::Object(m)
    }
}

/// Server behaviour switches (the `serve` flags that are not engine settings).
#[derive(Clone, Debug)]
pub struct ServerOptions {
    /// Default output budget for requests that omit one.
    pub max_output_tokens: usize,
    /// Split `<think>` blocks into reasoning content.
    pub reasoning: bool,
    /// Parse `<tool_call>` blocks into tool calls.
    pub tools: bool,
    /// Report prefix-cache hits in usage (cached_tokens / cache_read_input_tokens).
    pub cache_report: bool,
    /// Late system messages in Anthropic requests: None = probe the template.
    pub inline_system: Option<InlineSystem>,
    /// Browser origins allowed by CORS (`*` = any; empty = no CORS headers).
    pub cors_origins: Vec<String>,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            max_output_tokens: 4096,
            reasoning: true,
            tools: true,
            cache_report: true,
            inline_system: None,
            cors_origins: vec![],
        }
    }
}

pub struct AppState {
    pub engine: EngineHandle,
    pub tok: Arc<TokenizerBundle>,
    pub model_name: String,
    pub defaults: SamplingDefaults,
    pub max_output_tokens: usize,
    pub reasoning: bool,
    pub tools: bool,
    pub cache_report: bool,
    pub cors_origins: Vec<String>,
    pub started: Instant,
    /// What the chat template does with effort / thinking kwargs (probed at start-up).
    pub thinking: ThinkingProfile,
    pub inline_system: InlineSystem,
    pub metrics: Metrics,
    pub instance_id: String,
    /// Set by `/v1/admin/prepare-stop`: new generations are refused with 503.
    pub admission_closed: AtomicBool,
    sealed: tokio::sync::Mutex<Option<Value>>,
    pub request_log: Option<request_log::RequestLog>,
}

impl AppState {
    pub fn new(
        engine: EngineHandle,
        tok: Arc<TokenizerBundle>,
        model_name: String,
        defaults: SamplingDefaults,
        opts: ServerOptions,
    ) -> Self {
        let thinking = ThinkingProfile::probe(&|kw: &Map<String, Value>, tools: bool| {
            let tools = tools.then(|| {
                json!([{"type": "function", "function": {"name": "noop",
                    "description": "No-op probe tool.",
                    "parameters": {"type": "object", "properties": {}}}}])
            });
            tok.apply_chat_template(
                &[json!({"role": "user", "content": "Hello"})],
                tools.as_ref(),
                true,
                kw,
            )
            .map_err(|e| e.to_string())
        });
        if thinking.efforts.consumes_effort {
            tracing::info!(
                "reasoning effort levels: {:?} (default {:?})",
                thinking.efforts.effective(),
                thinking.efforts.default
            );
        }
        let inline_system = opts.inline_system.unwrap_or_else(|| {
            let (mode, why) = inline_system::probe(&tok);
            tracing::debug!("inline system messages: {mode:?} ({why})");
            mode
        });
        Self {
            engine,
            tok,
            model_name,
            defaults,
            max_output_tokens: opts.max_output_tokens,
            reasoning: opts.reasoning,
            tools: opts.tools,
            cache_report: opts.cache_report,
            cors_origins: opts.cors_origins,
            started: Instant::now(),
            thinking,
            inline_system,
            metrics: Metrics::default(),
            instance_id: uuid::Uuid::new_v4().to_string(),
            admission_closed: AtomicBool::new(false),
            sealed: tokio::sync::Mutex::new(None),
            request_log: request_log::RequestLog::from_env(),
        }
    }

    /// Append the request body to the JSONL request log (no-op unless enabled).
    pub fn log_request(&self, endpoint: &str, body: &Value) {
        if let Some(l) = &self.request_log {
            l.log(endpoint, None, body);
        }
    }

    /// Cached tokens as reported to clients (0 unless cache reporting is on).
    pub fn reported_cached(&self, u: &Usage) -> usize {
        if self.cache_report {
            u.cached_tokens
        } else {
            0
        }
    }
}

pub type SharedState = Arc<AppState>;

/// Paths that run a generation: refused while admission is closed, recorded in the ring.
const GENERATION_PATHS: [&str; 5] = [
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
    "/v1/responses",
    "/generate",
];

pub fn router(state: SharedState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1", any(v1_root))
        .route("/v1/models", get(openai::models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/responses", post(responses::create))
        .route("/v1/responses/{id}", get(responses::retrieve))
        .route("/v1/responses/{id}/cancel", post(responses::cancel))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/v1/stats", get(stats))
        .route("/v1/requests", get(requests))
        .route("/v1/cache/status", get(cache_status))
        .route("/v1/cache/rebuild", post(cache_rebuild))
        .route("/v1/admin/prepare-stop", post(prepare_stop))
        .route("/generate", post(generate))
        .layer(axum::middleware::from_fn_with_state(state.clone(), gate))
        .layer(axum::middleware::from_fn_with_state(state.clone(), cors))
        .with_state(state)
}

/// Serve on `listener` (with peer addresses, which the loopback-only admin endpoint needs).
pub async fn serve_listener(
    state: SharedState,
    listener: tokio::net::TcpListener,
) -> anyhow::Result<()> {
    axum::serve(
        listener,
        router(state).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

pub async fn serve(state: SharedState, addr: &str) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(
        "serving '{}' on http://{}",
        state.model_name,
        listener.local_addr()?
    );
    serve_listener(state, listener).await
}

/// CORS for browser / webview clients: allowed origins get the headers, preflights are answered.
async fn cors(State(s): State<SharedState>, req: Request, next: Next) -> Response {
    let origin = req
        .headers()
        .get("origin")
        .and_then(|o| o.to_str().ok())
        .map(String::from);
    let allowed = origin.as_ref().and_then(|o| {
        if s.cors_origins.iter().any(|x| x == "*") {
            Some("*".to_string())
        } else {
            s.cors_origins.iter().any(|x| x == o).then(|| o.clone())
        }
    });
    let Some(allow) = allowed else {
        return next.run(req).await;
    };
    let preflight = req.method() == Method::OPTIONS
        && req.headers().contains_key("access-control-request-method");
    let req_headers = req
        .headers()
        .get("access-control-request-headers")
        .cloned();
    let mut resp = if preflight {
        StatusCode::OK.into_response()
    } else {
        next.run(req).await
    };
    let h = resp.headers_mut();
    if let Ok(v) = HeaderValue::from_str(&allow) {
        h.insert("access-control-allow-origin", v);
    }
    if allow != "*" {
        h.insert("vary", HeaderValue::from_static("Origin"));
    }
    if preflight {
        h.insert(
            "access-control-allow-methods",
            HeaderValue::from_static("GET, POST, PUT, DELETE, OPTIONS, HEAD, PATCH"),
        );
        h.insert(
            "access-control-allow-headers",
            req_headers.unwrap_or(HeaderValue::from_static("*")),
        );
        h.insert("access-control-max-age", HeaderValue::from_static("600"));
    }
    resp
}

/// Admission gate (503 once prepare-stop closed it) and ring records for refused requests.
async fn gate(State(s): State<SharedState>, req: Request, next: Next) -> Response {
    let path = req.uri().path().to_string();
    let tracked = req.method() == Method::POST
        && GENERATION_PATHS.contains(&path.as_str());
    if !tracked {
        return next.run(req).await;
    }
    let t0 = Instant::now();
    let resp = if s.admission_closed.load(Ordering::SeqCst) {
        ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "server is stopping".into(),
        }
        .into_response()
    } else {
        next.run(req).await
    };
    // successful generations record themselves (with tokens and TTFT) when they finish
    if !resp.status().is_success() {
        s.metrics.record(RequestRecord {
            ts: metrics::utc_timestamp(),
            method: "POST".into(),
            path,
            status: resp.status().as_u16(),
            model: Some(s.model_name.clone()),
            duration_ms: t0.elapsed().as_millis() as u64,
            ttft_ms: None,
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
            stream: None,
            error: Some(resp.status().canonical_reason().unwrap_or("error").into()),
        });
    }
    resp
}

async fn v1_root() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }
    pub fn unavailable(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: msg.into(),
        }
    }
    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
        }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        Self::bad_request(format!("{e:#}"))
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let kind = if self.status == StatusCode::BAD_REQUEST {
            "invalid_request_error"
        } else {
            "server_error"
        };
        (
            self.status,
            Json(json!({"error": {"message": self.message, "type": kind, "code": null}})),
        )
            .into_response()
    }
}

impl AppState {
    /// Sampling parameters from an OpenAI/Anthropic style JSON body.
    pub fn sampling(&self, body: &Value, max_tokens: Option<u64>) -> SamplingParams {
        let f = |k: &str| body.get(k).and_then(|x| x.as_f64()).map(|x| x as f32);
        let d = &self.defaults;
        SamplingParams {
            temperature: f("temperature").or(d.temperature).unwrap_or(1.0),
            top_p: f("top_p").or(d.top_p).unwrap_or(1.0),
            top_k: body
                .get("top_k")
                .and_then(|x| x.as_i64())
                .map(|x| x as i32)
                .or(d.top_k)
                .unwrap_or(-1),
            min_p: f("min_p").or(d.min_p).unwrap_or(0.0),
            max_tokens: max_tokens
                .map(|x| x as usize)
                .unwrap_or(self.max_output_tokens)
                .max(1),
            ignore_eos: body
                .get("ignore_eos")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            stop_token_ids: body
                .get("stop_token_ids")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64())
                        .map(|x| x as u32)
                        .collect()
                })
                .unwrap_or_default(),
            seed: body.get("seed").and_then(|x| x.as_u64()),
            presence_penalty: f("presence_penalty").unwrap_or(0.0),
            frequency_penalty: f("frequency_penalty").unwrap_or(0.0),
            repetition_penalty: f("repetition_penalty")
                .or(d.repetition_penalty)
                .unwrap_or(1.0),
        }
    }

    /// Whether the rendered prompt leaves the model inside an open `<think>` block.
    pub fn prompt_opens_think(&self, prompt: &str) -> bool {
        self.reasoning && {
            let t = prompt.trim_end();
            t.ends_with("<think>")
        }
    }
}

pub fn stop_strings(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect(),
        _ => vec![],
    }
}

#[derive(Debug)]
pub enum Chunk {
    Text(String),
    Done {
        reason: FinishReason,
        usage: Usage,
        stop: Option<String>,
    },
    Error(String),
}

/// What the request ring records about a generation.
#[derive(Clone, Copy, Debug)]
pub struct ReqTag {
    pub path: &'static str,
    pub stream: bool,
}

/// One request in flight: detokenizes engine tokens, applies stop strings, and aborts the
/// engine request if dropped early (e.g. the client disconnected).
pub struct Generation {
    state: SharedState,
    id: u64,
    rx: mpsc::UnboundedReceiver<Event>,
    detok: DecodeStream,
    stop: StopChecker,
    matched: Option<String>,
    pending: VecDeque<Chunk>,
    finished: bool,
    tag: ReqTag,
    started: Instant,
    first_token: Option<Instant>,
    prompt_len: usize,
    generated: usize,
}

impl Generation {
    pub fn start(
        state: SharedState,
        prompt: Vec<u32>,
        params: SamplingParams,
        stops: Vec<String>,
        tag: ReqTag,
    ) -> Self {
        Self::start_with_images(state, prompt, vec![], params, stops, tag)
    }

    /// `start` for a prompt from `with_images`.
    pub fn start_with_images(
        state: SharedState,
        prompt: Vec<u32>,
        images: Vec<ImageSpan>,
        params: SamplingParams,
        stops: Vec<String>,
        tag: ReqTag,
    ) -> Self {
        let prompt_len = prompt.len();
        let (id, rx) = state.engine.submit_with_images(prompt, images, params);
        state.metrics.start(id);
        Self {
            state,
            id,
            rx,
            detok: DecodeStream::new(true),
            stop: StopChecker::new(stops),
            matched: None,
            pending: VecDeque::new(),
            finished: false,
            tag,
            started: Instant::now(),
            first_token: None,
            prompt_len,
            generated: 0,
        }
    }

    /// Close the books on this request: metrics + request-ring record.
    fn record(&self, status: u16, usage: Option<&Usage>, error: Option<String>) {
        let s = &self.state;
        let (prompt, completion) = usage
            .map(|u| (u.prompt_tokens - u.cached_tokens, u.completion_tokens))
            .unwrap_or((0, self.generated));
        s.metrics
            .finish(self.id, status == 200 && error.is_none(), prompt, completion);
        s.metrics.record(RequestRecord {
            ts: metrics::utc_timestamp(),
            method: "POST".into(),
            path: self.tag.path.into(),
            status,
            model: Some(s.model_name.clone()),
            duration_ms: self.started.elapsed().as_millis() as u64,
            ttft_ms: self
                .first_token
                .map(|t| t.duration_since(self.started).as_millis() as u64),
            prompt_tokens: Some(usage.map(|u| u.prompt_tokens).unwrap_or(self.prompt_len)),
            completion_tokens: Some(usage.map(|u| u.completion_tokens).unwrap_or(self.generated)),
            cached_tokens: usage.map(|u| u.cached_tokens),
            stream: Some(self.tag.stream),
            error,
        });
    }

    pub fn output_ids(&self) -> &[u32] {
        self.detok.ids()
    }

    pub async fn next(&mut self) -> Option<Chunk> {
        loop {
            if let Some(c) = self.pending.pop_front() {
                return Some(c);
            }
            if self.finished {
                return None;
            }
            match self.rx.recv().await {
                None => {
                    self.finished = true;
                    self.record(500, None, Some("engine stopped".into()));
                    return Some(Chunk::Error("engine stopped".into()));
                }
                Some(Event::Token(t)) => {
                    self.first_token.get_or_insert_with(Instant::now);
                    self.generated += 1;
                    if self.matched.is_some() {
                        continue;
                    }
                    if let Some(delta) = self.detok.step(&self.state.tok, t) {
                        let (emit, matched) = self.stop.push(&delta);
                        if !emit.is_empty() {
                            self.pending.push_back(Chunk::Text(emit));
                        }
                        if matched.is_some() {
                            self.matched = matched;
                            self.state.engine.abort(self.id);
                        }
                    }
                }
                Some(Event::Done { reason, usage }) => {
                    self.finished = true;
                    self.record(200, Some(&usage), None);
                    let rest = self.stop.flush();
                    if !rest.is_empty() && self.matched.is_none() {
                        self.pending.push_back(Chunk::Text(rest));
                    }
                    let reason = if self.matched.is_some() {
                        FinishReason::Stop
                    } else {
                        reason
                    };
                    self.pending.push_back(Chunk::Done {
                        reason,
                        usage,
                        stop: self.matched.take(),
                    });
                }
                Some(Event::Error(e)) => {
                    self.finished = true;
                    self.record(500, None, Some(e.clone()));
                    self.pending.push_back(Chunk::Error(e));
                }
            }
        }
    }

    /// Drain the whole generation.
    pub async fn collect(
        mut self,
    ) -> Result<(String, FinishReason, Usage, Option<String>), String> {
        let mut text = String::new();
        while let Some(c) = self.next().await {
            match c {
                Chunk::Text(t) => text.push_str(&t),
                Chunk::Done {
                    reason,
                    usage,
                    stop,
                } => return Ok((text, reason, usage, stop)),
                Chunk::Error(e) => return Err(e),
            }
        }
        Err("generation ended without a result".into())
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        if !self.finished {
            self.state.engine.abort(self.id);
            self.record(499, None, Some("client disconnected".into()));
        }
    }
}

async fn health(State(s): State<SharedState>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "model": s.model_name,
        "instance_id": s.instance_id,
        "uptime_s": s.started.elapsed().as_secs(),
        "maintenance": if s.admission_closed.load(Ordering::SeqCst) { "stopping" } else { "serving" },
        "version": env!("CARGO_PKG_VERSION"),
        "engine": &*s.engine.info,
    }))
}

async fn stats(State(s): State<SharedState>) -> Result<Json<Value>, ApiError> {
    let st = s
        .engine
        .stats()
        .await
        .ok_or_else(|| ApiError::internal("engine is not running"))?;
    let info = &s.engine.info;
    let (gpus, vram) = tokio::task::spawn_blocking(metrics::gpu_info)
        .await
        .unwrap_or_default();
    let state_total: usize = st.stages.first().map(|x| x.state_slots).unwrap_or(0);
    let round1 = |x: f64| (x * 10.0).round() / 10.0;
    Ok(Json(json!({
        "instance_id": s.instance_id,
        "model": {
            "id": s.model_name,
            "ctx": info.max_seq_len,
            "attn": info.attn,
            "moe": info.moe,
            "input_modalities": if info.image_processor.is_some() { json!(["text", "image"]) } else { json!(["text"]) },
            "sampling": s.defaults.to_json(),
        },
        "uptime_s": s.started.elapsed().as_secs(),
        "kv": {
            "used_pages": st.kv_capacity.saturating_sub(st.kv_free),
            "total_pages": st.kv_capacity,
            "page_size": 1,
        },
        "mamba": (state_total > 0).then(|| json!({
            "used_slots": (st.running + st.prefix_snapshots).min(state_total),
            "total_slots": state_total,
        })),
        "swa": null,
        "vram_bytes": vram,
        "gpus": gpus,
        "throughput": {
            "decode_tps": round1(st.decode_tps),
            "prefill_tps": round1(st.prefill_tps),
            "host_gap_us": round1(st.host_gap_us),
        },
        "requests": s.metrics.snapshot(),
        "engine": &**info,
        "scheduler": st,
    })))
}

#[derive(serde::Deserialize)]
struct RequestsQuery {
    #[serde(default)]
    since: u64,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    100
}

/// Recent requests, polled incrementally with `?since=<next_cursor>`.
async fn requests(State(s): State<SharedState>, Query(q): Query<RequestsQuery>) -> Json<Value> {
    let (records, next) = s.metrics.requests_since(q.since, q.limit.clamp(1, 1000));
    Json(json!({"requests": records, "next_cursor": next}))
}

fn is_loopback(addr: Option<&SocketAddr>) -> bool {
    addr.is_some_and(|a| match a.ip() {
        std::net::IpAddr::V4(v) => v.is_loopback(),
        std::net::IpAddr::V6(v) => {
            v.is_loopback() || v.to_ipv4_mapped().is_some_and(|m| m.is_loopback())
        }
    })
}

/// Fail-closed stop barrier for supervisors: close admission, let running work drain, abort
/// what is left, then return one sealed (idempotent) token-accounting snapshot. Loopback only.
async fn prepare_stop(State(s): State<SharedState>, req: Request) -> Response {
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0);
    if !is_loopback(peer.as_ref()) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"error": "loopback access required"})),
        )
            .into_response();
    }
    let body: Value = axum::body::to_bytes(req.into_body(), 1 << 16)
        .await
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(json!({}));
    let secs = |k: &str, d: f64, max: f64| body[k].as_f64().unwrap_or(d).clamp(0.0, max);
    let drain = Duration::from_secs_f64(secs("drain_timeout_s", 5.0, 10.0));
    let abort = Duration::from_secs_f64(secs("abort_timeout_s", 3.0, 4.0));
    let mut sealed = s.sealed.lock().await;
    if let Some(v) = sealed.as_ref() {
        return Json(v.clone()).into_response();
    }
    s.admission_closed.store(true, Ordering::SeqCst);
    let wait_idle = |limit: Duration| {
        let s = s.clone();
        async move {
            let t0 = Instant::now();
            while s.metrics.active() > 0 {
                if t0.elapsed() >= limit {
                    return false;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            true
        }
    };
    if !wait_idle(drain).await {
        for id in s.metrics.inflight() {
            s.engine.abort(id);
        }
        if !wait_idle(abort).await {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "error": format!("accounting abort barrier timed out with {} request(s) still active", s.metrics.active()),
                    "drain_complete": false,
                    "engine_preserved": true,
                })),
            )
                .into_response();
        }
    }
    let (p, c) = s.metrics.totals();
    let v = json!({
        "instance_id": s.instance_id,
        "model_id": s.model_name,
        "prompt_tokens_total": p,
        "completion_tokens_total": c,
        "uptime_s": s.started.elapsed().as_secs(),
        "drain_complete": true,
    });
    *sealed = Some(v.clone());
    Json(v).into_response()
}

async fn cache_status(State(s): State<SharedState>) -> Result<Json<Value>, ApiError> {
    let st = s
        .engine
        .stats()
        .await
        .ok_or_else(|| ApiError::internal("engine is not running"))?;
    Ok(Json(json!({
        "kv": {
            "capacity_tokens": st.kv_capacity,
            "free_tokens": st.kv_free,
            "prefix_cached_tokens": st.prefix_cached_tokens,
            "prefix_evictable_tokens": st.prefix_evictable_tokens,
        },
        "stages": st.stages,
    })))
}

/// Live resize of the KV cache (`kv_tokens`) and/or the MoE expert cache (`moe_cache_size`).
/// Waits for running requests to finish; new requests queue meanwhile.
async fn cache_rebuild(
    State(s): State<SharedState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let get = |k: &str| body.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
    let req = crate::scheduler::RebuildRequest {
        kv_tokens: get("kv_tokens"),
        expert_cache: get("moe_cache_size"),
    };
    if req.kv_tokens.is_none() && req.expert_cache.is_none() {
        return Err(ApiError::bad_request(
            "pass kv_tokens and/or moe_cache_size",
        ));
    }
    let wait = body.get("wait").and_then(|x| x.as_u64()).unwrap_or(300);
    match tokio::time::timeout(std::time::Duration::from_secs(wait), s.engine.rebuild(req)).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(ApiError::internal(e)),
        Err(_) => {
            return Err(ApiError::internal(
                "timed out waiting for running requests to drain",
            ))
        }
    }
    cache_status(State(s)).await
}

/// Raw completion without a chat template (`ft ctl generate` equivalent).
async fn generate(
    State(s): State<SharedState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let prompt: Vec<u32> = if let Some(ids) = body.get("input_ids").and_then(|x| x.as_array()) {
        ids.iter()
            .filter_map(|x| x.as_u64())
            .map(|x| x as u32)
            .collect()
    } else {
        let text = body
            .get("text")
            .and_then(|x| x.as_str())
            .ok_or_else(|| ApiError::bad_request("'text' or 'input_ids' is required"))?;
        s.tok.encode(text, true)?
    };
    let sp = body.get("sampling_params").cloned().unwrap_or(json!({}));
    let max = sp
        .get("max_new_tokens")
        .or_else(|| sp.get("max_tokens"))
        .and_then(|x| x.as_u64());
    let mut params = s.sampling(&sp, max);
    if sp.get("temperature").is_none() {
        params.temperature = 0.0;
    }
    let stops = stop_strings(sp.get("stop"));
    s.log_request("/generate", &body);
    let t0 = Instant::now();
    let generation = Generation::start(
        s.clone(),
        prompt,
        params,
        stops,
        ReqTag {
            path: "/generate",
            stream: false,
        },
    );
    let mut g = generation;
    let mut text = String::new();
    let (reason, usage) = loop {
        match g.next().await {
            Some(Chunk::Text(t)) => text.push_str(&t),
            Some(Chunk::Done { reason, usage, .. }) => break (reason, usage),
            Some(Chunk::Error(e)) => return Err(ApiError::bad_request(e)),
            None => return Err(ApiError::internal("generation ended unexpectedly")),
        }
    };
    let ids = g.output_ids().to_vec();
    let secs = t0.elapsed().as_secs_f64();
    Ok(Json(json!({
        "text": text,
        "output_ids": ids,
        "meta_info": {
            "finish_reason": reason,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cached_tokens": usage.cached_tokens,
            "e2e_latency_s": secs,
        }
    })))
}

/// An image content part as chat templates take it.
pub fn image_part(url: &str) -> Value {
    json!({"type": "image", "image": url})
}

/// The images of template messages (`image_part`s), in the order the template renders them;
/// an error when there are some and the model takes no images.
pub fn message_images(s: &SharedState, msgs: &[Value]) -> Result<Vec<String>, ApiError> {
    let images: Vec<String> = msgs
        .iter()
        .filter_map(|m| m.get("content")?.as_array())
        .flatten()
        .filter(|p| p["type"] == "image")
        .filter_map(|p| p["image"].as_str().map(String::from))
        .collect();
    if !images.is_empty() && s.engine.info.image_processor.is_none() {
        return Err(ApiError::bad_request("this model does not take image input"));
    }
    Ok(images)
}

/// Fetch and preprocess `sources` (data: or http(s) URLs, or bare base64) and expand the image placeholders of
/// the templated prompt `ids` (one per image, in order) into the image tokens.
pub async fn with_images(
    s: &SharedState,
    ids: Vec<u32>,
    sources: Vec<String>,
) -> Result<(Vec<u32>, Vec<ImageSpan>), ApiError> {
    if sources.is_empty() {
        return Ok((ids, vec![]));
    }
    let info = &s.engine.info;
    let (Some(proc), Some(image_id)) = (info.image_processor.clone(), info.image_token_id) else {
        return Err(ApiError::bad_request("this model does not take image input"));
    };
    if sources.len() > MAX_IMAGES {
        return Err(ApiError::bad_request(format!("at most {MAX_IMAGES} images per request")));
    }
    tokio::task::spawn_blocking(move || {
        let mut images = Vec::with_capacity(sources.len());
        for (i, src) in sources.iter().enumerate() {
            // no local files from the API
            let bytes = if src.starts_with("data:") || src.starts_with("http://") || src.starts_with("https://") {
                crate::images::fetch(src)
            } else if let Some(b) = crate::images::bare_base64_image(src) {
                Ok(b)
            } else {
                return Err(ApiError::bad_request(format!(
                    "image {i}: expected a data: or http(s) URL or base64 image data"
                )));
            };
            let img = bytes
                .and_then(|b| proc.prepare(&b))
                .map_err(|e| ApiError::bad_request(format!("image {i}: {e:#}")))?;
            images.push(img);
        }
        crate::images::expand_image_tokens(&ids, image_id, images, proc.merge_size)
            .map_err(|e| ApiError::bad_request(format!("{e:#}")))
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
}

const MAX_IMAGES: usize = 64;

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn short_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..24].to_string()
}
