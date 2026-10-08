// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages API (`/v1/messages`), enough for Claude Code style agents.

use super::parsers::{OutputParser, Piece};
use super::inline_system;
use super::{
    image_part, message_images, short_id, stop_strings, with_images, ApiError, Chunk, Generation, ReqTag,
    SharedState,
};
use crate::images::ImageSpan;
use crate::scheduler::FinishReason;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Map, Value};
use std::convert::Infallible;

struct AnthropicError(ApiError);

impl From<ApiError> for AnthropicError {
    fn from(e: ApiError) -> Self {
        Self(e)
    }
}

impl From<anyhow::Error> for AnthropicError {
    fn from(e: anyhow::Error) -> Self {
        Self(e.into())
    }
}

impl IntoResponse for AnthropicError {
    fn into_response(self) -> Response {
        let kind = if self.0.status == StatusCode::BAD_REQUEST {
            "invalid_request_error"
        } else {
            "api_error"
        };
        (
            self.0.status,
            Json(json!({"type": "error", "error": {"type": kind, "message": self.0.message}})),
        )
            .into_response()
    }
}

fn blocks_text(v: &Value) -> Result<String, ApiError> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Array(blocks) => {
            let mut out = String::new();
            for b in blocks {
                match b.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        out.push_str(b.get("text").and_then(|t| t.as_str()).unwrap_or(""))
                    }
                    Some("image") | Some("document") => {
                        return Err(ApiError::bad_request(
                            "image/document input is not supported by this server (text only)",
                        ))
                    }
                    _ => {}
                }
            }
            Ok(out)
        }
        Value::Null => Ok(String::new()),
        other => Err(ApiError::bad_request(format!("invalid content: {other}"))),
    }
}

/// An `image` block's source as a URL: `{"type": "base64", media_type, data}` or
/// `{"type": "url", url}`.
fn image_url(b: &Value) -> Result<String, ApiError> {
    let src = b.get("source").ok_or_else(|| ApiError::bad_request("image block without a source"))?;
    match src.get("type").and_then(|t| t.as_str()) {
        Some("base64") => Ok(format!(
            "data:{};base64,{}",
            src.get("media_type").and_then(|t| t.as_str()).unwrap_or("image/png"),
            src.get("data").and_then(|d| d.as_str()).unwrap_or("")
        )),
        Some("url") => src
            .get("url")
            .and_then(|u| u.as_str())
            .map(String::from)
            .ok_or_else(|| ApiError::bad_request("image source without a url")),
        t => Err(ApiError::bad_request(format!("image source type {t:?} is not supported"))),
    }
}

/// Text and image blocks -> template content: the text, or text and `image_part`s.
struct Content {
    parts: Vec<Value>,
    text: String,
}

impl Content {
    fn new() -> Self {
        Self { parts: vec![], text: String::new() }
    }

    fn image(&mut self, b: &Value) -> Result<(), ApiError> {
        if !self.text.is_empty() {
            self.parts.push(json!({"type": "text", "text": std::mem::take(&mut self.text)}));
        }
        self.parts.push(image_part(&image_url(b)?));
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.parts.is_empty() && self.text.is_empty()
    }

    fn take(&mut self) -> Value {
        let mut c = std::mem::replace(self, Self::new());
        if c.parts.is_empty() {
            return json!(c.text);
        }
        if !c.text.is_empty() {
            c.parts.push(json!({"type": "text", "text": c.text}));
        }
        Value::Array(c.parts)
    }
}

/// A tool result's content (string or text/image blocks).
fn tool_result_content(v: &Value) -> Result<Content, ApiError> {
    let mut c = Content::new();
    match v {
        Value::Array(blocks) => {
            for b in blocks {
                match b.get("type").and_then(|t| t.as_str()) {
                    Some("text") => c.text.push_str(b.get("text").and_then(|t| t.as_str()).unwrap_or("")),
                    Some("image") => c.image(b)?,
                    Some("document") => return Err(ApiError::bad_request("document input is not supported")),
                    _ => {}
                }
            }
        }
        other => c.text = blocks_text(other)?,
    }
    Ok(c)
}

/// Convert an Anthropic request into OpenAI-style template messages + tools.
fn convert(body: &Value) -> Result<(Vec<Value>, Option<Value>), ApiError> {
    let mut out = vec![];
    if let Some(sys) = body.get("system") {
        let text = blocks_text(sys)?;
        if !text.is_empty() {
            out.push(json!({"role": "system", "content": text}));
        }
    }
    let messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or_else(|| ApiError::bad_request("'messages' is required"))?;
    for m in messages {
        let role = m
            .get("role")
            .and_then(|r| r.as_str())
            .ok_or_else(|| ApiError::bad_request("message without role"))?;
        let content = m.get("content").cloned().unwrap_or(Value::Null);
        let blocks = match content {
            Value::String(s) => vec![json!({"type": "text", "text": s})],
            Value::Array(a) => a,
            _ => vec![],
        };
        if role == "assistant" {
            let mut text = String::new();
            let mut thinking = String::new();
            let mut calls = vec![];
            for b in &blocks {
                match b.get("type").and_then(|t| t.as_str()) {
                    Some("text") => text.push_str(b.get("text").and_then(|t| t.as_str()).unwrap_or("")),
                    Some("thinking") => thinking.push_str(b.get("thinking").and_then(|t| t.as_str()).unwrap_or("")),
                    Some("tool_use") => calls.push(json!({
                        "id": b.get("id").cloned().unwrap_or(Value::Null),
                        "type": "function",
                        "function": {"name": b.get("name").cloned().unwrap_or(Value::Null), "arguments": b.get("input").cloned().unwrap_or(json!({}))},
                    })),
                    _ => {}
                }
            }
            let mut msg = Map::new();
            msg.insert("role".into(), json!("assistant"));
            msg.insert("content".into(), json!(text));
            if !thinking.is_empty() {
                msg.insert("reasoning_content".into(), json!(thinking));
            }
            if !calls.is_empty() {
                msg.insert("tool_calls".into(), json!(calls));
            }
            out.push(Value::Object(msg));
        } else {
            // tool results become tool messages, text and images accumulate into a user message
            let mut user = Content::new();
            for b in &blocks {
                match b.get("type").and_then(|t| t.as_str()) {
                    Some("tool_result") => {
                        if !user.is_empty() {
                            out.push(json!({"role": "user", "content": user.take()}));
                        }
                        let mut c = tool_result_content(b.get("content").unwrap_or(&Value::Null))?;
                        if b.get("is_error").and_then(|e| e.as_bool()).unwrap_or(false) {
                            c.text = format!("Error: {}", c.text);
                        }
                        out.push(json!({"role": "tool", "tool_call_id": b.get("tool_use_id").cloned().unwrap_or(Value::Null), "content": c.take()}));
                    }
                    Some("text") => {
                        user.text.push_str(b.get("text").and_then(|t| t.as_str()).unwrap_or(""))
                    }
                    Some("image") => user.image(b)?,
                    Some("document") => {
                        return Err(ApiError::bad_request("document input is not supported"))
                    }
                    _ => {}
                }
            }
            if !user.is_empty() {
                out.push(json!({"role": "user", "content": user.take()}));
            }
        }
    }
    let tool_choice_none =
        body.pointer("/tool_choice/type").and_then(|t| t.as_str()) == Some("none");
    let tools = body.get("tools").and_then(|t| t.as_array()).filter(|_| !tool_choice_none).map(|list| {
        Value::Array(
            list.iter()
                .filter(|t| t.get("input_schema").is_some() || t.get("type").and_then(|x| x.as_str()) == Some("custom"))
                .map(|t| {
                    json!({"type": "function", "function": {
                        "name": t.get("name").cloned().unwrap_or(Value::Null),
                        "description": t.get("description").cloned().unwrap_or(json!("")),
                        "parameters": t.get("input_schema").cloned().unwrap_or(json!({"type": "object"})),
                    }})
                })
                .collect(),
        )
    });
    Ok((out, tools))
}

async fn prepare(
    s: &SharedState,
    body: &Value,
) -> Result<(Vec<u32>, Vec<ImageSpan>, String, bool), ApiError> {
    let (msgs, tools) = convert(body)?;
    let msgs =
        inline_system::normalize(msgs, s.inline_system).map_err(ApiError::bad_request)?;
    let effort = body
        .pointer("/output_config/effort")
        .or_else(|| body.get("reasoning_effort"))
        .and_then(|e| e.as_str());
    let kwargs = s.thinking.apply(
        effort,
        body.pointer("/thinking/type").and_then(|t| t.as_str()),
        Map::new(),
    );
    let has_tools = tools
        .as_ref()
        .and_then(|t| t.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    let sources = message_images(s, &msgs)?;
    let prompt = s
        .tok
        .apply_chat_template(&msgs, tools.as_ref(), true, &kwargs)?;
    let ids = s.tok.encode_prompt(&prompt)?;
    let (ids, images) = with_images(s, ids, sources).await?;
    Ok((ids, images, prompt, has_tools))
}

pub async fn count_tokens(State(s): State<SharedState>, Json(body): Json<Value>) -> Response {
    match prepare(&s, &body).await {
        Ok((ids, _, _, _)) => Json(json!({"input_tokens": ids.len()})).into_response(),
        Err(e) => AnthropicError(e).into_response(),
    }
}

fn stop_reason(r: FinishReason, stop: &Option<String>, tool_use: bool) -> &'static str {
    if tool_use {
        "tool_use"
    } else if stop.is_some() {
        "stop_sequence"
    } else if r == FinishReason::Length {
        "max_tokens"
    } else {
        "end_turn"
    }
}

pub async fn messages(State(s): State<SharedState>, Json(body): Json<Value>) -> Response {
    match messages_inner(s, body).await {
        Ok(r) => r,
        Err(e) => e.into_response(),
    }
}

async fn messages_inner(s: SharedState, body: Value) -> Result<Response, AnthropicError> {
    let (ids, images, prompt, has_tools) = prepare(&s, &body).await?;
    let max = body.get("max_tokens").and_then(|x| x.as_u64());
    let params = s.sampling(&body, max);
    let stops = stop_strings(body.get("stop_sequences"));
    let parser = OutputParser::new(
        s.reasoning,
        s.tools && has_tools,
        s.prompt_opens_think(&prompt),
    );
    let input_tokens = ids.len();
    s.log_request("/v1/messages", &body);
    let stream = body
        .get("stream")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    let gen = Generation::start_with_images(
        s.clone(),
        ids,
        images,
        params,
        stops,
        ReqTag {
            path: "/v1/messages",
            stream,
        },
    );
    let id = format!("msg_{}", short_id());

    if stream {
        let stream = message_stream(s, gen, parser, id, input_tokens);
        // Anthropic clients expect data-bearing `ping` events, not SSE comments
        return Ok(Sse::new(stream)
            .keep_alive(
                KeepAlive::new()
                    .interval(std::time::Duration::from_secs(15))
                    .event(Event::default().event("ping").data(r#"{"type": "ping"}"#)),
            )
            .into_response());
    }

    let mut parser = parser;
    let (text, reason, usage, stop) = gen.collect().await.map_err(ApiError::internal)?;
    let mut pieces = parser.push(&text);
    pieces.extend(parser.finish());
    let mut content: Vec<Value> = vec![];
    let mut tool_use = false;
    for p in pieces {
        match p {
            Piece::Reasoning(r) => match content.last_mut() {
                Some(b) if b["type"] == "thinking" => {
                    b["thinking"] = json!(format!("{}{r}", b["thinking"].as_str().unwrap_or("")))
                }
                _ => content.push(json!({"type": "thinking", "thinking": r, "signature": ""})),
            },
            Piece::Content(c) => match content.last_mut() {
                Some(b) if b["type"] == "text" => {
                    b["text"] = json!(format!("{}{c}", b["text"].as_str().unwrap_or("")))
                }
                _ => content.push(json!({"type": "text", "text": c})),
            },
            Piece::ToolCall { name, arguments } => {
                tool_use = true;
                let input: Value = serde_json::from_str(&arguments).unwrap_or(json!({}));
                content.push(json!({"type": "tool_use", "id": format!("toolu_{}", short_id()), "name": name, "input": input}));
            }
        }
    }
    Ok(Json(json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": s.model_name,
        "content": content,
        "stop_reason": stop_reason(reason, &stop, tool_use),
        "stop_sequence": stop,
        "usage": {
            "input_tokens": usage.prompt_tokens - s.reported_cached(&usage),
            "cache_read_input_tokens": s.reported_cached(&usage),
            "cache_creation_input_tokens": 0,
            "output_tokens": usage.completion_tokens,
        },
    }))
    .into_response())
}

fn message_stream(
    s: SharedState,
    mut gen: Generation,
    mut parser: OutputParser,
    id: String,
    input_tokens: usize,
) -> impl futures::Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let ev = |name: &str, data: Value| Event::default().event(name).data(data.to_string());
        yield Ok(ev("message_start", json!({"type": "message_start", "message": {
            "id": id, "type": "message", "role": "assistant", "model": s.model_name, "content": [],
            "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }})));
        let mut index: i64 = -1;
        let mut open: Option<&'static str> = None;
        let mut tool_use = false;
        loop {
            let (pieces, done) = match gen.next().await {
                Some(Chunk::Text(t)) => (parser.push(&t), None),
                Some(Chunk::Done { reason, usage, stop }) => (parser.finish(), Some((reason, usage, stop))),
                Some(Chunk::Error(e)) => {
                    yield Ok(ev("error", json!({"type": "error", "error": {"type": "api_error", "message": e}})));
                    break;
                }
                None => break,
            };
            for p in pieces {
                let kind = match &p {
                    Piece::Reasoning(_) => "thinking",
                    Piece::Content(_) => "text",
                    Piece::ToolCall { .. } => "tool_use",
                };
                if open != Some(kind) || kind == "tool_use" {
                    if open.is_some() {
                        yield Ok(ev("content_block_stop", json!({"type": "content_block_stop", "index": index})));
                    }
                    index += 1;
                    let block = match &p {
                        Piece::Reasoning(_) => json!({"type": "thinking", "thinking": "", "signature": ""}),
                        Piece::Content(_) => json!({"type": "text", "text": ""}),
                        Piece::ToolCall { name, .. } => json!({"type": "tool_use", "id": format!("toolu_{}", short_id()), "name": name, "input": {}}),
                    };
                    yield Ok(ev("content_block_start", json!({"type": "content_block_start", "index": index, "content_block": block})));
                    open = Some(kind);
                }
                let delta = match p {
                    Piece::Reasoning(r) => json!({"type": "thinking_delta", "thinking": r}),
                    Piece::Content(c) => json!({"type": "text_delta", "text": c}),
                    Piece::ToolCall { arguments, .. } => {
                        tool_use = true;
                        json!({"type": "input_json_delta", "partial_json": arguments})
                    }
                };
                yield Ok(ev("content_block_delta", json!({"type": "content_block_delta", "index": index, "delta": delta})));
            }
            if let Some((reason, usage, stop)) = done {
                if open.is_some() {
                    yield Ok(ev("content_block_stop", json!({"type": "content_block_stop", "index": index})));
                }
                yield Ok(ev("message_delta", json!({"type": "message_delta",
                    "delta": {"stop_reason": stop_reason(reason, &stop, tool_use), "stop_sequence": stop},
                    "usage": {"output_tokens": usage.completion_tokens, "cache_read_input_tokens": s.reported_cached(&usage)}})));
                yield Ok(ev("message_stop", json!({"type": "message_stop"})));
                break;
            }
        }
    }
}
