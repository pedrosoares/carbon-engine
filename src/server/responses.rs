// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! OpenAI Responses API (`/v1/responses`), the endpoint codex talks to.
//!
//! A stateless subset: `store` is ignored, `previous_response_id` / `background` are refused and
//! `GET /v1/responses/{id}` / `cancel` answer 404, so clients resend the full context as `input`.
//! Reasoning streams as a first-class `reasoning` item; reasoning items replayed as input fold
//! back into their assistant turn as `reasoning_content`.

use super::openai::{template_messages, template_tools, tool_choice};
use super::parsers::{OutputParser, Piece};
use super::{message_images, now_secs, short_id, with_images, ApiError, Chunk, Generation, ReqTag, SharedState};
use crate::scheduler::{FinishReason, Usage};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Map, Value};
use std::convert::Infallible;
use std::time::Duration;

fn error(status: StatusCode, msg: impl Into<String>) -> Response {
    (
        status,
        Json(json!({"error": {"message": msg.into(), "type": "invalid_request_error", "code": null}})),
    )
        .into_response()
}

pub async fn retrieve(Path(id): Path<String>) -> Response {
    error(
        StatusCode::NOT_FOUND,
        format!("response '{id}' not found (stateless server)"),
    )
}

pub async fn cancel(Path(id): Path<String>) -> Response {
    retrieve(Path(id)).await
}

fn text_of(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p {
                Value::String(s) => Some(s.clone()),
                Value::Object(_) => p.get("text").and_then(|t| t.as_str()).map(String::from),
                _ => None,
            })
            .collect(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn has_images(content: &Value) -> bool {
    content.as_array().is_some_and(|parts| parts.iter().any(|p| p["type"] == "input_image"))
}

/// One input item -> chat messages.
fn convert_item(item: &Value) -> Result<Vec<Value>, ApiError> {
    let kind = item
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("message");
    Ok(match kind {
        "message" => {
            let role = match item.get("role").and_then(|r| r.as_str()).unwrap_or("user") {
                "developer" => "system",
                r => r,
            };
            let content = item.get("content").cloned().unwrap_or(Value::Null);
            // parts with images go to `template_messages` as they are
            let content = if has_images(&content) && role != "system" { content } else { json!(text_of(&content)) };
            vec![json!({"role": role, "content": content})]
        }
        "function_call" => vec![json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": item.get("call_id").or_else(|| item.get("id")).cloned()
                    .unwrap_or_else(|| json!(format!("call_{}", short_id()))),
                "type": "function",
                "function": {
                    "name": item.get("name").cloned().unwrap_or(json!("")),
                    "arguments": item.get("arguments").and_then(|a| a.as_str()).unwrap_or(""),
                },
            }],
        })],
        "function_call_output" => {
            let out = item.get("output").cloned().unwrap_or(Value::Null);
            let content = match &out {
                Value::Array(_) if has_images(&out) => out.clone(),
                Value::Array(_) => json!(text_of(&out)),
                Value::String(s) => json!(s),
                Value::Null => json!(""),
                o => json!(o.to_string()),
            };
            vec![json!({"role": "tool", "tool_call_id": item.get("call_id").cloned().unwrap_or(json!("")), "content": content})]
        }
        "reasoning" => {
            let text: String = item
                .get("content")
                .and_then(|c| c.as_array())
                .map(|a| {
                    a.iter()
                        .filter(|c| c["type"] == "reasoning_text")
                        .filter_map(|c| c.get("text").and_then(|t| t.as_str()))
                        .collect()
                })
                .unwrap_or_default();
            if text.is_empty() {
                vec![]
            } else {
                vec![json!({"role": "assistant", "content": "", "reasoning_content": text})]
            }
        }
        // item references, built-in tool calls, ...: nothing a chat template can render
        _ => vec![],
    })
}

fn nonempty(m: &Value, k: &str) -> bool {
    match m.get(k) {
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Null) | None => false,
        Some(_) => true,
    }
}

/// Coalesce one assistant turn's items (reasoning / message / function calls) into a single
/// message; a reasoning item always opens a new turn.
fn merge_assistant_runs(messages: Vec<Value>) -> Vec<Value> {
    let mut out: Vec<Value> = vec![];
    for m in messages {
        let mergeable = out.last().is_some_and(|prev| {
            m["role"] == "assistant"
                && prev["role"] == "assistant"
                && !nonempty(&m, "reasoning_content")
                && (nonempty(&m, "tool_calls")
                    || (nonempty(&m, "content")
                        && !nonempty(prev, "content")
                        && !nonempty(prev, "tool_calls")))
        });
        if !mergeable {
            out.push(m);
            continue;
        }
        let prev = out.last_mut().unwrap();
        if nonempty(&m, "content") {
            prev["content"] = m["content"].clone();
        }
        if let Some(calls) = m.get("tool_calls").and_then(|c| c.as_array()) {
            let mut all = prev
                .get("tool_calls")
                .and_then(|c| c.as_array())
                .cloned()
                .unwrap_or_default();
            all.extend(calls.iter().cloned());
            prev["tool_calls"] = Value::Array(all);
        }
    }
    out
}

/// Responses request -> (chat messages, template tools).
pub fn convert(body: &Value) -> Result<(Vec<Value>, Option<Value>), ApiError> {
    let mut system: Vec<String> = vec![];
    if let Some(i) = body.get("instructions").and_then(|i| i.as_str()) {
        system.push(i.to_string());
    }
    let mut other = vec![];
    match body.get("input") {
        Some(Value::String(s)) => other.push(json!({"role": "user", "content": s})),
        Some(Value::Array(items)) => {
            for item in items {
                for m in convert_item(item)? {
                    if m["role"] == "system" {
                        system.push(m["content"].as_str().unwrap_or("").to_string());
                    } else {
                        other.push(m);
                    }
                }
            }
            other = merge_assistant_runs(other);
        }
        _ => return Err(ApiError::bad_request("'input' is required")),
    }
    // strict templates want exactly one leading system message
    let mut messages = vec![];
    let sys = system
        .into_iter()
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if !sys.is_empty() {
        messages.push(json!({"role": "system", "content": sys}));
    }
    messages.extend(other);
    let tools: Vec<Value> = body
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|list| {
            list.iter()
                .filter(|t| matches!(t.get("type").and_then(|x| x.as_str()), None | Some("function")))
                .map(|t| {
                    let f = t.get("function").filter(|f| f.is_object()).unwrap_or(t);
                    json!({"type": "function", "function": {
                        "name": f.get("name").cloned().unwrap_or(json!("")),
                        "description": f.get("description").cloned().unwrap_or(Value::Null),
                        "parameters": f.get("parameters").cloned().filter(|p| !p.is_null())
                            .unwrap_or(json!({"type": "object"})),
                    }})
                })
                .collect()
        })
        .unwrap_or_default();
    let tools = template_tools(
        (!tools.is_empty()).then_some(&Value::Array(tools)),
        &tool_choice(body.get("tool_choice")),
    );
    Ok((messages, tools))
}

fn usage(s: &SharedState, u: &Usage) -> Value {
    json!({
        "input_tokens": u.prompt_tokens,
        "output_tokens": u.completion_tokens,
        "total_tokens": u.prompt_tokens + u.completion_tokens,
        "input_tokens_details": {"cached_tokens": s.reported_cached(u)},
        "output_tokens_details": {"reasoning_tokens": 0},
    })
}

#[allow(clippy::too_many_arguments)]
fn response_obj(
    id: &str,
    created: u64,
    model: &str,
    status: &str,
    output: &[Value],
    usage: Value,
    incomplete: bool,
    error: Value,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": created,
        "model": model,
        "status": status,
        "output": output,
        "usage": usage,
        "error": error,
        "incomplete_details": if incomplete { json!({"reason": "max_output_tokens"}) } else { Value::Null },
        "parallel_tool_calls": true,
        "tool_choice": "auto",
        "tools": [],
    })
}

fn reasoning_item(id: &str, text: &str, status: &str) -> Value {
    json!({"id": id, "type": "reasoning", "summary": [], "status": status,
           "content": [{"type": "reasoning_text", "text": text}]})
}

fn message_item(id: &str, text: &str, status: &str) -> Value {
    json!({"id": id, "type": "message", "role": "assistant", "status": status,
           "content": [{"type": "output_text", "text": text, "annotations": []}]})
}

fn call_item(id: &str, call_id: &str, name: &str, args: &str, status: &str) -> Value {
    json!({"id": id, "type": "function_call", "call_id": call_id, "name": name,
           "arguments": args, "status": status})
}

pub async fn create(State(s): State<SharedState>, Json(body): Json<Value>) -> Response {
    match create_inner(s, body).await {
        Ok(r) => r,
        Err(e) => error(e.status, e.message),
    }
}

async fn create_inner(s: SharedState, body: Value) -> Result<Response, ApiError> {
    if body.get("background").and_then(|b| b.as_bool()) == Some(true) {
        return Err(ApiError::bad_request("background mode is not supported"));
    }
    if body
        .get("previous_response_id")
        .is_some_and(|p| !p.is_null())
    {
        return Err(ApiError::bad_request(
            "previous_response_id is not supported (stateless server); resend full context in 'input'",
        ));
    }
    let max = body.get("max_output_tokens").and_then(|x| x.as_u64());
    if max == Some(0) {
        return Err(ApiError::bad_request(
            "max_output_tokens must be a positive integer",
        ));
    }
    s.log_request("/v1/responses", &body);
    let (msgs, tools) = convert(&body)?;
    let msgs = template_messages(&msgs)?;
    let kwargs = body
        .get("chat_template_kwargs")
        .and_then(|k| k.as_object())
        .cloned()
        .unwrap_or_default();
    let kwargs = s.thinking.apply(
        body.pointer("/reasoning/effort").and_then(|e| e.as_str()),
        None,
        kwargs,
    );
    let sources = message_images(&s, &msgs)?;
    let prompt = s
        .tok
        .apply_chat_template(&msgs, tools.as_ref(), true, &kwargs)?;
    let ids = s.tok.encode_prompt(&prompt)?;
    let (ids, images) = with_images(&s, ids, sources).await?;
    let params = s.sampling(&body, max);
    let parser = OutputParser::new(s.reasoning, s.tools && tools.is_some(), s.prompt_opens_think(&prompt));
    let stream = body.get("stream").and_then(|x| x.as_bool()).unwrap_or(false);
    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(&s.model_name)
        .to_string();
    let gen = Generation::start_with_images(
        s.clone(),
        ids,
        images,
        params,
        vec![],
        ReqTag {
            path: "/v1/responses",
            stream,
        },
    );
    let id = format!("resp_{}", short_id());
    let created = now_secs();
    if stream {
        return Ok(Sse::new(response_stream(s, gen, parser, id, created, model))
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response());
    }
    let mut parser = parser;
    let (text, reason, u, _) = gen.collect().await.map_err(ApiError::internal)?;
    let mut pieces = parser.push(&text);
    pieces.extend(parser.finish());
    let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), vec![]);
    for p in pieces {
        match p {
            Piece::Reasoning(r) => reasoning.push_str(&r),
            Piece::Content(c) => content.push_str(&c),
            Piece::ToolCall { name, arguments } => calls.push((name, arguments)),
        }
    }
    let truncated = reason == FinishReason::Length;
    let status = if truncated { "incomplete" } else { "completed" };
    let mut output = vec![];
    if !reasoning.is_empty() {
        output.push(reasoning_item(&format!("rs_{}", short_id()), &reasoning, "completed"));
    }
    if !content.is_empty() {
        output.push(message_item(&format!("msg_{}", short_id()), &content, status));
    }
    for (name, args) in calls {
        output.push(call_item(
            &format!("fc_{}", short_id()),
            &format!("call_{}", short_id()),
            &name,
            &args,
            "completed",
        ));
    }
    Ok(Json(response_obj(
        &id,
        created,
        &model,
        status,
        &output,
        usage(&s, &u),
        truncated,
        Value::Null,
    ))
    .into_response())
}

/// Open output item being streamed.
enum Open {
    Reasoning { id: String, text: String },
    Message { id: String, text: String },
}

fn response_stream(
    s: SharedState,
    mut gen: Generation,
    mut parser: OutputParser,
    id: String,
    created: u64,
    model: String,
) -> impl futures::Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let mut seq: u64 = 0;
        let mut ev = |kind: &str, mut data: Map<String, Value>| -> Event {
            data.insert("type".into(), json!(kind));
            data.insert("sequence_number".into(), json!(seq));
            seq += 1;
            Event::default().event(kind).data(Value::Object(data).to_string())
        };
        macro_rules! obj {
            ($($t:tt)*) => { json!({$($t)*}).as_object().unwrap().clone() };
        }
        let snapshot = |status: &str, output: &[Value], u: Value, incomplete: bool, err: Value| {
            response_obj(&id, created, &model, status, output, u, incomplete, err)
        };
        yield Ok(ev("response.created", obj!("response": snapshot("in_progress", &[], Value::Null, false, Value::Null))));
        yield Ok(ev("response.in_progress", obj!("response": snapshot("in_progress", &[], Value::Null, false, Value::Null))));
        let mut output: Vec<Value> = vec![];
        let mut open: Option<Open> = None;
        // closes the open item, emitting its done events
        macro_rules! close {
            ($finish:expr) => {
                if let Some(o) = open.take() {
                    let idx = output.len();
                    let item = match o {
                        Open::Reasoning { id: iid, text } => {
                            yield Ok(ev("response.reasoning_text.done", obj!("item_id": iid, "output_index": idx, "content_index": 0, "text": text)));
                            reasoning_item(&iid, &text, "completed")
                        }
                        Open::Message { id: iid, text } => {
                            yield Ok(ev("response.output_text.done", obj!("item_id": iid, "output_index": idx, "content_index": 0, "text": text, "logprobs": [])));
                            yield Ok(ev("response.content_part.done", obj!("item_id": iid, "output_index": idx, "content_index": 0,
                                "part": {"type": "output_text", "text": text, "annotations": []})));
                            message_item(&iid, &text, if $finish == Some(FinishReason::Length) { "incomplete" } else { "completed" })
                        }
                    };
                    yield Ok(ev("response.output_item.done", obj!("output_index": idx, "item": item.clone())));
                    output.push(item);
                }
            };
        }
        loop {
            let (pieces, done) = match gen.next().await {
                Some(Chunk::Text(t)) => (parser.push(&t), None),
                Some(Chunk::Done { reason, usage, .. }) => (parser.finish(), Some((reason, usage))),
                Some(Chunk::Error(e)) => {
                    close!(None::<FinishReason>);
                    yield Ok(ev("response.failed", obj!("response": snapshot("failed", &output, Value::Null, false,
                        json!({"code": "server_error", "message": e})))));
                    break;
                }
                None => break,
            };
            for p in pieces {
                match p {
                    Piece::Reasoning(r) => {
                        if !matches!(open, Some(Open::Reasoning { .. })) {
                            close!(None::<FinishReason>);
                            let iid = format!("rs_{}", short_id());
                            yield Ok(ev("response.output_item.added", obj!("output_index": output.len(),
                                "item": {"id": iid, "type": "reasoning", "summary": [], "status": "in_progress"})));
                            open = Some(Open::Reasoning { id: iid, text: String::new() });
                        }
                        if let Some(Open::Reasoning { id: iid, text }) = open.as_mut() {
                            text.push_str(&r);
                            let iid = iid.clone();
                            yield Ok(ev("response.reasoning_text.delta", obj!("item_id": iid, "output_index": output.len(), "content_index": 0, "delta": r)));
                        }
                    }
                    Piece::Content(c) => {
                        if !matches!(open, Some(Open::Message { .. })) {
                            close!(None::<FinishReason>);
                            let iid = format!("msg_{}", short_id());
                            yield Ok(ev("response.output_item.added", obj!("output_index": output.len(),
                                "item": {"id": iid, "type": "message", "role": "assistant", "status": "in_progress", "content": []})));
                            yield Ok(ev("response.content_part.added", obj!("item_id": iid, "output_index": output.len(), "content_index": 0,
                                "part": {"type": "output_text", "text": "", "annotations": []})));
                            open = Some(Open::Message { id: iid, text: String::new() });
                        }
                        if let Some(Open::Message { id: iid, text }) = open.as_mut() {
                            text.push_str(&c);
                            let iid = iid.clone();
                            yield Ok(ev("response.output_text.delta", obj!("item_id": iid, "output_index": output.len(), "content_index": 0, "delta": c, "logprobs": [])));
                        }
                    }
                    Piece::ToolCall { name, arguments } => {
                        close!(None::<FinishReason>);
                        let idx = output.len();
                        let (iid, cid) = (format!("fc_{}", short_id()), format!("call_{}", short_id()));
                        yield Ok(ev("response.output_item.added", obj!("output_index": idx,
                            "item": call_item(&iid, &cid, &name, "", "in_progress"))));
                        yield Ok(ev("response.function_call_arguments.delta", obj!("item_id": iid, "output_index": idx, "delta": arguments)));
                        yield Ok(ev("response.function_call_arguments.done", obj!("item_id": iid, "output_index": idx, "name": name, "arguments": arguments)));
                        let item = call_item(&iid, &cid, &name, &arguments, "completed");
                        yield Ok(ev("response.output_item.done", obj!("output_index": idx, "item": item.clone())));
                        output.push(item);
                    }
                }
            }
            if let Some((reason, u)) = done {
                close!(Some(reason));
                let truncated = reason == FinishReason::Length;
                let (kind, status) = if truncated {
                    ("response.incomplete", "incomplete")
                } else {
                    ("response.completed", "completed")
                };
                yield Ok(ev(kind, obj!("response": snapshot(status, &output, usage(&s, &u), truncated, Value::Null))));
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_codex_style_input() {
        let body = json!({
            "instructions": "You are codex.",
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "perms"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list files"}]},
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "need ls"}]},
                {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
                {"type": "function_call_output", "call_id": "c1", "output": "a.txt"},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "a.txt"}]},
            ],
            "tools": [
                {"type": "function", "name": "shell", "parameters": {"type": "object"}},
                {"type": "web_search"},
            ],
        });
        let (msgs, tools) = convert(&body).unwrap();
        assert_eq!(msgs[0], json!({"role": "system", "content": "You are codex.\n\nperms"}));
        assert_eq!(msgs[1], json!({"role": "user", "content": "list files"}));
        // reasoning + function call merge into one assistant turn
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["reasoning_content"], "need ls");
        assert_eq!(msgs[2]["tool_calls"][0]["function"]["name"], "shell");
        assert_eq!(msgs[3], json!({"role": "tool", "tool_call_id": "c1", "content": "a.txt"}));
        assert_eq!(msgs[4]["content"], "a.txt");
        assert_eq!(msgs.len(), 5);
        let tools = tools.unwrap();
        assert_eq!(tools.as_array().unwrap().len(), 1, "built-in tools are skipped");
        assert_eq!(tools[0]["function"]["name"], "shell");
        // a forced tool narrows the template's list; none removes it
        let mut b = body.clone();
        b["tool_choice"] = json!({"type": "function", "name": "nope"});
        assert!(convert(&b).unwrap().1.is_none());
        b["tool_choice"] = json!("none");
        assert!(convert(&b).unwrap().1.is_none());
        // images become template image parts
        let (msgs, _) = convert(&json!({"input": [{"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "see"}, {"type": "input_image", "image_url": "x"}]}]})).unwrap();
        assert_eq!(template_messages(&msgs).unwrap()[0]["content"],
            json!([{"type": "text", "text": "see"}, {"type": "image", "image": "x"}]));
        assert_eq!(convert(&json!({"input": "hi"})).unwrap().0, vec![json!({"role": "user", "content": "hi"})]);
    }
}
