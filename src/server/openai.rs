// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! OpenAI-compatible endpoints.

use super::parsers::{OutputParser, Piece};
use super::{
    now_secs, short_id, stop_strings, ApiError, Chunk, Generation, ReqTag, SharedState, image_part, message_images, with_images};
use crate::scheduler::{FinishReason, Usage};
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::Stream;
use serde_json::{json, Map, Value};
use std::convert::Infallible;

pub async fn models(State(s): State<SharedState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": s.model_name,
            "object": "model",
            "created": now_secs(),
            "owned_by": "carbon-engine",
            "max_model_len": s.engine.info.max_seq_len,
            "context_length": s.engine.info.max_seq_len,
            "supported_reasoning_efforts": s.thinking.efforts.effective(),
            "default_reasoning_effort": s.thinking.efforts.default,
        }]
    }))
}

fn finish_str(r: FinishReason, has_tools: bool) -> &'static str {
    match r {
        _ if has_tools => "tool_calls",
        FinishReason::Length => "length",
        FinishReason::Stop | FinishReason::Abort => "stop",
    }
}

fn usage_json(s: &SharedState, u: &Usage) -> Value {
    json!({
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.prompt_tokens + u.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": s.reported_cached(u)},
    })
}

/// `tool_choice`: None = no tools for the template, Some(name) = only that function.
pub enum ToolChoice {
    Auto,
    None,
    Named(String),
}

pub fn tool_choice(v: Option<&Value>) -> ToolChoice {
    match v {
        Some(Value::String(s)) if s == "none" => ToolChoice::None,
        Some(Value::Object(o)) => o
            .get("function")
            .and_then(|f| f.get("name"))
            .or_else(|| o.get("name"))
            .and_then(|n| n.as_str())
            .map(|n| ToolChoice::Named(n.to_string()))
            .unwrap_or(ToolChoice::Auto),
        _ => ToolChoice::Auto,
    }
}

/// The tool list the chat template sees under `choice` (a named choice keeps only that tool).
pub fn template_tools(tools: Option<&Value>, choice: &ToolChoice) -> Option<Value> {
    let list = tools?.as_array()?;
    match choice {
        ToolChoice::None => None,
        ToolChoice::Auto => Some(Value::Array(list.clone())),
        ToolChoice::Named(n) => Some(Value::Array(
            list.iter()
                .filter(|t| t.pointer("/function/name").and_then(|x| x.as_str()) == Some(n))
                .cloned()
                .collect(),
        )),
    }
    .filter(|t| t.as_array().is_some_and(|a| !a.is_empty()))
}

/// OpenAI message content (string or parts) -> template content: the text, or text and
/// `image_part`s when it has images (`image_url` / Responses `input_image` parts).
fn content_value(content: Option<&Value>) -> Result<Value, ApiError> {
    match content {
        None | Some(Value::Null) => Ok(json!("")),
        Some(Value::String(s)) => Ok(json!(s)),
        Some(Value::Array(parts)) => {
            let (mut out, mut text) = (vec![], String::new());
            for p in parts {
                match p.get("type").and_then(|t| t.as_str()) {
                    Some("text") | Some("input_text") | Some("output_text") => {
                        text.push_str(p.get("text").and_then(|t| t.as_str()).unwrap_or(""))
                    }
                    Some("image_url") | Some("input_image") => {
                        let url = p
                            .get("image_url")
                            .and_then(|u| u.as_str().or_else(|| u.get("url")?.as_str()))
                            .ok_or_else(|| ApiError::bad_request("image part without an image_url"))?;
                        if !text.is_empty() {
                            out.push(json!({"type": "text", "text": std::mem::take(&mut text)}));
                        }
                        out.push(image_part(url));
                    }
                    Some(t) => {
                        return Err(ApiError::bad_request(format!(
                            "content part type '{t}' is not supported"
                        )))
                    }
                    None => {}
                }
            }
            if out.is_empty() {
                return Ok(json!(text));
            }
            if !text.is_empty() {
                out.push(json!({"type": "text", "text": text}));
            }
            Ok(Value::Array(out))
        }
        Some(other) => Err(ApiError::bad_request(format!(
            "invalid message content: {other}"
        ))),
    }
}

/// Normalise OpenAI messages into what chat templates expect.
pub fn template_messages(messages: &[Value]) -> Result<Vec<Value>, ApiError> {
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        let role = m
            .get("role")
            .and_then(|r| r.as_str())
            .ok_or_else(|| ApiError::bad_request("message without role"))?;
        let role = if role == "developer" { "system" } else { role };
        let mut msg = Map::new();
        msg.insert("role".into(), json!(role));
        msg.insert("content".into(), content_value(m.get("content"))?);
        if let Some(calls) = m.get("tool_calls").and_then(|t| t.as_array()) {
            let calls: Vec<Value> = calls
                .iter()
                .map(|c| {
                    let mut c = c.clone();
                    // templates commonly expect arguments as a mapping
                    if let Some(args) = c.pointer("/function/arguments").and_then(|a| a.as_str()) {
                        if let Ok(parsed) = serde_json::from_str::<Value>(args) {
                            c["function"]["arguments"] = parsed;
                        }
                    }
                    c
                })
                .collect();
            msg.insert("tool_calls".into(), Value::Array(calls));
        }
        for k in ["tool_call_id", "name", "reasoning_content"] {
            if let Some(v) = m.get(k) {
                msg.insert(k.into(), v.clone());
            }
        }
        out.push(Value::Object(msg));
    }
    Ok(out)
}

fn tool_call_json(index: usize, name: &str, args: &str) -> Value {
    json!({
        "index": index,
        "id": format!("call_{}", short_id()),
        "type": "function",
        "function": {"name": name, "arguments": args},
    })
}

pub async fn chat_completions(
    State(s): State<SharedState>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or_else(|| ApiError::bad_request("'messages' is required"))?;
    if body.get("n").and_then(|n| n.as_u64()).unwrap_or(1) != 1 {
        return Err(ApiError::bad_request("only n=1 is supported"));
    }
    if body.get("logit_bias").is_some_and(|b| b.as_object().is_some_and(|o| !o.is_empty())) {
        return Err(ApiError::bad_request("logit_bias is not supported"));
    }
    if let Some(t) = body.pointer("/response_format/type").and_then(|t| t.as_str()) {
        if t != "text" {
            return Err(ApiError::bad_request(format!(
                "response_format '{t}' is not supported (no constrained decoding)"
            )));
        }
    }
    s.log_request("/v1/chat/completions", &body);
    let msgs = template_messages(messages)?;
    let tools = template_tools(body.get("tools"), &tool_choice(body.get("tool_choice")));
    let has_tools = tools.is_some();
    let kwargs = body
        .get("chat_template_kwargs")
        .and_then(|k| k.as_object())
        .cloned()
        .unwrap_or_default();
    let kwargs = s.thinking.apply(
        body.get("reasoning_effort").and_then(|e| e.as_str()),
        body.pointer("/thinking/type").and_then(|t| t.as_str()),
        kwargs,
    );
    let sources = message_images(&s, &msgs)?;
    let prompt = s.tok.apply_chat_template(&msgs, tools.as_ref(), true, &kwargs)?;
    let ids = s.tok.encode_prompt(&prompt)?;
    let (ids, images) = with_images(&s, ids, sources).await?;
    let max = body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(|x| x.as_u64());
    let params = s.sampling(&body, max);
    let stops = stop_strings(body.get("stop"));
    let start_think = s.prompt_opens_think(&prompt);
    let parser = OutputParser::new(s.reasoning, s.tools && has_tools, start_think);
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
            path: "/v1/chat/completions",
            stream,
        },
    );
    let id = format!("chatcmpl-{}", short_id());

    if stream {
        let include_usage = body
            .pointer("/stream_options/include_usage")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        return Ok(Sse::new(chat_stream(s, gen, parser, id, include_usage))
            .keep_alive(KeepAlive::default())
            .into_response());
    }

    let mut parser = parser;
    let (text, reason, usage, _) = gen.collect().await.map_err(ApiError::internal)?;
    let mut pieces = parser.push(&text);
    pieces.extend(parser.finish());
    let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), vec![]);
    for p in pieces {
        match p {
            Piece::Reasoning(r) => reasoning.push_str(&r),
            Piece::Content(c) => content.push_str(&c),
            Piece::ToolCall { name, arguments } => {
                let mut c = tool_call_json(calls.len(), &name, &arguments);
                c.as_object_mut().unwrap().remove("index");
                calls.push(c);
            }
        }
    }
    let mut message = json!({"role": "assistant", "content": if content.is_empty() && !calls.is_empty() { Value::Null } else { json!(content) }});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    if !calls.is_empty() {
        message["tool_calls"] = json!(calls);
    }
    Ok(Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": now_secs(),
        "model": s.model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_str(reason, !calls.is_empty()), "logprobs": null}],
        "usage": usage_json(&s, &usage),
    }))
    .into_response())
}

fn chat_stream(
    s: SharedState,
    mut gen: Generation,
    mut parser: OutputParser,
    id: String,
    include_usage: bool,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let created = now_secs();
        let model = s.model_name.clone();
        let chunk = |delta: Value, finish: Value| -> Event {
            Event::default().data(json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": null}],
            }).to_string())
        };
        yield Ok(chunk(json!({"role": "assistant", "content": ""}), Value::Null));
        let mut n_calls = 0usize;
        loop {
            let (pieces, done) = match gen.next().await {
                Some(Chunk::Text(t)) => (parser.push(&t), None),
                Some(Chunk::Done { reason, usage, .. }) => (parser.finish(), Some((reason, usage))),
                Some(Chunk::Error(e)) => {
                    yield Ok(Event::default().data(json!({"error": {"message": e, "type": "server_error"}}).to_string()));
                    break;
                }
                None => break,
            };
            for p in pieces {
                let delta = match p {
                    Piece::Content(c) => json!({"content": c}),
                    Piece::Reasoning(r) => json!({"reasoning_content": r}),
                    Piece::ToolCall { name, arguments } => {
                        n_calls += 1;
                        json!({"tool_calls": [tool_call_json(n_calls - 1, &name, &arguments)]})
                    }
                };
                yield Ok(chunk(delta, Value::Null));
            }
            if let Some((reason, usage)) = done {
                yield Ok(chunk(json!({}), json!(finish_str(reason, n_calls > 0))));
                if include_usage {
                    yield Ok(Event::default().data(json!({
                        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [], "usage": usage_json(&s, &usage),
                    }).to_string()));
                }
                break;
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    }
}

pub async fn completions(
    State(s): State<SharedState>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let ids: Vec<u32> = match body.get("prompt") {
        Some(Value::String(p)) => s.tok.encode(p, true)?,
        Some(Value::Array(a)) if a.iter().all(|x| x.is_u64()) => a
            .iter()
            .filter_map(|x| x.as_u64())
            .map(|x| x as u32)
            .collect(),
        Some(Value::Array(a)) if a.len() == 1 && a[0].is_string() => {
            s.tok.encode(a[0].as_str().unwrap(), true)?
        }
        _ => {
            return Err(ApiError::bad_request(
                "'prompt' must be a string or a list of token ids",
            ))
        }
    };
    let max = body.get("max_tokens").and_then(|x| x.as_u64()).or(Some(16));
    let params = s.sampling(&body, max);
    let stops = stop_strings(body.get("stop"));
    s.log_request("/v1/completions", &body);
    let stream = body
        .get("stream")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    let gen = Generation::start(
        s.clone(),
        ids,
        params,
        stops,
        ReqTag {
            path: "/v1/completions",
            stream,
        },
    );
    let id = format!("cmpl-{}", short_id());
    if stream {
        let include_usage = body
            .pointer("/stream_options/include_usage")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let stream = async_stream::stream! {
            let mut gen = gen;
            let created = now_secs();
            let model = s.model_name.clone();
            loop {
                match gen.next().await {
                    Some(Chunk::Text(t)) => yield Ok::<_, Infallible>(Event::default().data(json!({
                        "id": id, "object": "text_completion", "created": created, "model": model,
                        "choices": [{"index": 0, "text": t, "finish_reason": null, "logprobs": null}],
                    }).to_string())),
                    Some(Chunk::Done { reason, usage, .. }) => {
                        let mut v = json!({
                            "id": id, "object": "text_completion", "created": created, "model": model,
                            "choices": [{"index": 0, "text": "", "finish_reason": finish_str(reason, false), "logprobs": null}],
                        });
                        if include_usage {
                            v["usage"] = usage_json(&s, &usage);
                        }
                        yield Ok(Event::default().data(v.to_string()));
                        break;
                    }
                    Some(Chunk::Error(e)) => {
                        yield Ok(Event::default().data(json!({"error": {"message": e}}).to_string()));
                        break;
                    }
                    None => break,
                }
            }
            yield Ok(Event::default().data("[DONE]"));
        };
        return Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response());
    }
    let (text, reason, usage, _) = gen.collect().await.map_err(ApiError::internal)?;
    Ok(Json(json!({
        "id": id,
        "object": "text_completion",
        "created": now_secs(),
        "model": s.model_name,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_str(reason, false), "logprobs": null}],
        "usage": usage_json(&s, &usage),
    }))
    .into_response())
}
