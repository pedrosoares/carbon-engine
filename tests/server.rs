// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! HTTP API tests against a tiny model: OpenAI chat/completions (+ streaming),
//! Anthropic messages (+ streaming), stop strings, control endpoints and aborts.

use carbon_engine::engine::{Engine, EngineConfig};
use carbon_engine::scheduler::{self, SchedulerConfig};
use carbon_engine::server::{serve_listener, AppState, SamplingDefaults, ServerOptions};
use carbon_engine::testing::{write_tiny_model, TinySpec};
use carbon_engine::tokenizer::TokenizerBundle;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::sync::Arc;

struct Server {
    base: String,
    _dir: tempfile::TempDir,
    _rt: tokio::runtime::Runtime,
}

fn start() -> Server {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("m");
    write_tiny_model(&model, &TinySpec::new("qwen3")).unwrap();
    let mut ec = EngineConfig::new(&model);
    ec.kv_tokens = Some(8192);
    let engine = Engine::load(&ec).unwrap();
    let eos = engine.cfg.eos_token_ids.clone();
    let handle = scheduler::spawn(
        engine,
        SchedulerConfig {
            max_running: 8,
            max_prefill_tokens: 512,
            eos_token_ids: eos,
            anchor_token: None,
        },
    )
    .unwrap();
    let tok = Arc::new(TokenizerBundle::load(&model, None).unwrap());
    let state = Arc::new(AppState::new(
        handle,
        tok,
        "tiny".into(),
        SamplingDefaults::from_model_dir(&model),
        ServerOptions {
            max_output_tokens: 64,
            cors_origins: vec!["http://localhost:1420".into()],
            ..Default::default()
        },
    ));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let listener = rt
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    rt.spawn(async move { serve_listener(state, listener).await.unwrap() });
    Server {
        base,
        _dir: dir,
        _rt: rt,
    }
}

fn post(s: &Server, path: &str, body: Value) -> Value {
    ureq::post(&format!("{}{path}", s.base))
        .send_json(body)
        .unwrap()
        .into_json()
        .unwrap()
}

fn sse(s: &Server, path: &str, body: Value) -> Vec<(Option<String>, String)> {
    let resp = ureq::post(&format!("{}{path}", s.base))
        .send_json(body)
        .unwrap();
    assert!(resp.content_type().starts_with("text/event-stream"));
    let mut events = vec![];
    let mut name = None;
    for line in BufReader::new(resp.into_reader()).lines() {
        let line = line.unwrap();
        if let Some(n) = line.strip_prefix("event: ") {
            name = Some(n.to_string());
        } else if let Some(d) = line.strip_prefix("data: ") {
            events.push((name.take(), d.to_string()));
        }
    }
    events
}

#[test]
fn http_api() {
    let s = start();
    let models: Value = ureq::get(&format!("{}/v1/models", s.base))
        .call()
        .unwrap()
        .into_json()
        .unwrap();
    assert_eq!(models["data"][0]["id"], "tiny");
    let health: Value = ureq::get(&format!("{}/health", s.base))
        .call()
        .unwrap()
        .into_json()
        .unwrap();
    assert_eq!(health["status"], "ok");

    // --- OpenAI chat, non-streaming
    let req = json!({"model": "tiny", "messages": [{"role": "user", "content": "hi there"}], "max_tokens": 12, "ignore_eos": true});
    let r = post(&s, "/v1/chat/completions", req.clone());
    assert_eq!(r["object"], "chat.completion");
    assert_eq!(r["choices"][0]["finish_reason"], "length");
    assert_eq!(r["usage"]["completion_tokens"], 12);
    let full = r["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string()
        + r["choices"][0]["message"]["reasoning_content"]
            .as_str()
            .unwrap_or("");
    let prompt_tokens = r["usage"]["prompt_tokens"].as_u64().unwrap();
    assert!(prompt_tokens > 5);

    // same request again: the prompt prefix is served from the radix cache
    let r2 = post(&s, "/v1/chat/completions", req.clone());
    assert_eq!(
        r2["choices"][0]["message"], r["choices"][0]["message"],
        "greedy decoding must be deterministic"
    );
    assert_eq!(
        r2["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap(),
        prompt_tokens - 1
    );

    // --- streaming yields the same text
    let mut sreq = req.clone();
    sreq["stream"] = json!(true);
    sreq["stream_options"] = json!({"include_usage": true});
    let events = sse(&s, "/v1/chat/completions", sreq);
    assert_eq!(events.last().unwrap().1, "[DONE]");
    let mut streamed = String::new();
    let mut saw_usage = false;
    let mut finish = None;
    for (_, d) in &events[..events.len() - 1] {
        let v: Value = serde_json::from_str(d).unwrap();
        if let Some(c) = v["choices"].get(0) {
            streamed += c["delta"]["content"].as_str().unwrap_or("");
            streamed += c["delta"]["reasoning_content"].as_str().unwrap_or("");
            if !c["finish_reason"].is_null() {
                finish = Some(c["finish_reason"].clone());
            }
        } else if v.get("usage").is_some() {
            saw_usage = true;
        }
    }
    assert_eq!(streamed, full);
    assert_eq!(finish, Some(json!("length")));
    assert!(saw_usage);

    // --- stop strings truncate the output before the match
    let chars: Vec<char> = full.chars().collect();
    if chars.len() > 6 {
        let stop: String = chars[3..5].iter().collect();
        let cut = full.find(&stop).unwrap();
        let mut stop_req = req.clone();
        stop_req["stop"] = json!([stop]);
        let r = post(&s, "/v1/chat/completions", stop_req);
        let got = r["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string()
            + r["choices"][0]["message"]["reasoning_content"]
                .as_str()
                .unwrap_or("");
        assert_eq!(
            got.len(),
            cut,
            "output should end right before the stop string"
        );
        assert_eq!(r["choices"][0]["finish_reason"], "stop");
    }

    // --- legacy completions
    let r = post(
        &s,
        "/v1/completions",
        json!({"prompt": "abc", "max_tokens": 5, "ignore_eos": true}),
    );
    assert_eq!(r["usage"]["completion_tokens"], 5);
    let r = post(
        &s,
        "/v1/completions",
        json!({"prompt": [97, 98, 99], "max_tokens": 3, "ignore_eos": true}),
    );
    assert_eq!(r["usage"]["prompt_tokens"], 3);

    // --- Anthropic messages
    let areq = json!({"model": "tiny", "max_tokens": 10, "system": "be brief", "ignore_eos": true,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]});
    let r = post(&s, "/v1/messages", areq.clone());
    assert_eq!(r["type"], "message");
    assert_eq!(r["stop_reason"], "max_tokens");
    assert_eq!(r["usage"]["output_tokens"], 10);
    let mut sreq = areq.clone();
    sreq["stream"] = json!(true);
    let events = sse(&s, "/v1/messages", sreq);
    let names: Vec<String> = events.iter().filter_map(|e| e.0.clone()).collect();
    assert_eq!(names.first().map(String::as_str), Some("message_start"));
    assert_eq!(names.last().map(String::as_str), Some("message_stop"));
    assert!(names.iter().any(|n| n == "message_delta"));
    let c = post(&s, "/v1/messages/count_tokens", areq.clone());
    assert!(c["input_tokens"].as_u64().unwrap() > 3);

    // tool results and tool definitions render through the chat template
    let tool_req = json!({"model": "tiny", "max_tokens": 4, "ignore_eos": true,
    "tools": [{"name": "get_weather", "description": "weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
    "messages": [
        {"role": "user", "content": "weather in Paris?"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}]}
    ]});
    let r = post(&s, "/v1/messages", tool_req);
    assert_eq!(r["usage"]["output_tokens"], 4);

    // images are rejected with an Anthropic-shaped error
    let bad = ureq::post(&format!("{}/v1/messages", s.base)).send_json(json!({"max_tokens": 4,
        "messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]}));
    match bad {
        Err(ureq::Error::Status(400, resp)) => {
            let v: Value = resp.into_json().unwrap();
            assert_eq!(v["type"], "error");
        }
        other => panic!("expected 400, got {other:?}"),
    }

    // --- raw generate + stats
    let g = post(
        &s,
        "/generate",
        json!({"text": "xyz", "sampling_params": {"max_new_tokens": 4, "ignore_eos": true}}),
    );
    assert_eq!(g["output_ids"].as_array().unwrap().len(), 4);
    let st: Value = ureq::get(&format!("{}/v1/stats", s.base))
        .call()
        .unwrap()
        .into_json()
        .unwrap();
    assert!(st["scheduler"]["total_generated_tokens"].as_u64().unwrap() > 40);
    let cache: Value = ureq::get(&format!("{}/v1/cache/status", s.base))
        .call()
        .unwrap()
        .into_json()
        .unwrap();
    assert!(cache["kv"]["prefix_cached_tokens"].as_u64().unwrap() > 0);
}

#[test]
fn client_disconnect_aborts_generation() {
    let s = start();
    let resp = ureq::post(&format!("{}/v1/chat/completions", s.base))
        .send_json(json!({"messages": [{"role": "user", "content": "go"}], "max_tokens": 100000, "ignore_eos": true, "stream": true}))
        .unwrap();
    let mut lines = BufReader::new(resp.into_reader()).lines();
    for _ in 0..4 {
        lines.next().unwrap().unwrap();
    }
    drop(lines);
    let mut running = 1;
    for _ in 0..100 {
        let st: Value = ureq::get(&format!("{}/v1/stats", s.base))
            .call()
            .unwrap()
            .into_json()
            .unwrap();
        running = st["scheduler"]["running"].as_u64().unwrap();
        if running == 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert_eq!(running, 0, "dropped stream should abort the request");
}

fn get(s: &Server, path: &str) -> Value {
    ureq::get(&format!("{}{path}", s.base))
        .call()
        .unwrap()
        .into_json()
        .unwrap()
}

#[test]
fn responses_api() {
    let s = start();
    let req = json!({"model": "tiny", "instructions": "be brief", "max_output_tokens": 6,
        "ignore_eos": true, "input": [{"type": "message", "role": "user",
        "content": [{"type": "input_text", "text": "hello"}]}]});
    let r = post(&s, "/v1/responses", req.clone());
    assert_eq!(r["object"], "response");
    assert_eq!(r["status"], "incomplete");
    assert_eq!(r["incomplete_details"]["reason"], "max_output_tokens");
    assert_eq!(r["usage"]["output_tokens"], 6);
    let text: String = r["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| match item["type"].as_str().unwrap() {
            "message" => item["content"][0]["text"].as_str().unwrap().to_string(),
            "reasoning" => item["content"][0]["text"].as_str().unwrap().to_string(),
            _ => String::new(),
        })
        .collect();

    // streaming: typed events, a terminal response.incomplete, same text
    let mut sreq = req.clone();
    sreq["stream"] = json!(true);
    let events = sse(&s, "/v1/responses", sreq);
    let names: Vec<String> = events.iter().filter_map(|e| e.0.clone()).collect();
    assert_eq!(names[0], "response.created");
    assert_eq!(names.last().unwrap(), "response.incomplete");
    let mut streamed = String::new();
    let mut seqs = vec![];
    for (name, d) in &events {
        let v: Value = serde_json::from_str(d).unwrap();
        seqs.push(v["sequence_number"].as_u64().unwrap());
        if matches!(name.as_deref(), Some("response.output_text.delta" | "response.reasoning_text.delta")) {
            streamed += v["delta"].as_str().unwrap();
        }
    }
    assert_eq!(streamed, text);
    assert!(seqs.windows(2).all(|w| w[1] == w[0] + 1), "sequence numbers are consecutive");

    // stateless: no retrieval, no chaining
    match ureq::get(&format!("{}/v1/responses/resp_x", s.base)).call() {
        Err(ureq::Error::Status(404, _)) => {}
        o => panic!("expected 404, got {o:?}"),
    }
    let mut chained = req.clone();
    chained["previous_response_id"] = json!("resp_x");
    assert!(matches!(
        ureq::post(&format!("{}/v1/responses", s.base)).send_json(chained),
        Err(ureq::Error::Status(400, _))
    ));
}

#[test]
fn control_plane() {
    let s = start();
    // /v1 status probe, model card extras
    assert_eq!(get(&s, "/v1")["status"], "ok");
    let m = get(&s, "/v1/models");
    assert!(m["data"][0]["context_length"].as_u64().unwrap() > 0);
    assert!(m["data"][0]["supported_reasoning_efforts"].is_array());

    // unsupported request features are refused up front, and the refusal is recorded
    for bad in [
        json!({"messages": [{"role": "user", "content": "x"}], "logit_bias": {"5": 10}}),
        json!({"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_object"}}),
    ] {
        assert!(matches!(
            ureq::post(&format!("{}/v1/chat/completions", s.base)).send_json(bad),
            Err(ureq::Error::Status(400, _))
        ));
    }
    // reasoning_effort and a named tool_choice render without errors
    let r = post(&s, "/v1/chat/completions", json!({"max_tokens": 3, "ignore_eos": true,
        "reasoning_effort": "low",
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}},
                  {"type": "function", "function": {"name": "b", "parameters": {"type": "object"}}}],
        "tool_choice": {"type": "function", "function": {"name": "b"}}}));
    assert_eq!(r["usage"]["completion_tokens"], 3);

    // request ring: newest records with tokens and TTFT, paged by cursor
    let rq = get(&s, "/v1/requests?since=0&limit=100");
    let recs = rq["requests"].as_array().unwrap();
    assert_eq!(recs.len(), 3);
    assert_eq!(recs[0]["status"], 400);
    let ok = &recs[2];
    assert_eq!(ok["path"], "/v1/chat/completions");
    assert_eq!(ok["status"], 200);
    assert_eq!(ok["completion_tokens"], 3);
    assert!(ok["ttft_ms"].is_u64());
    assert_eq!(rq["next_cursor"], 3);
    assert_eq!(get(&s, "/v1/requests?since=3")["requests"].as_array().unwrap().len(), 0);

    // stats in FreeToken's shape
    let st = get(&s, "/v1/stats");
    assert_eq!(st["model"]["id"], "tiny");
    assert_eq!(st["model"]["attn"], "mha");
    assert_eq!(st["requests"]["completed"], 1);
    assert_eq!(st["requests"]["completion_tokens_total"], 3);
    assert!(st["kv"]["total_pages"].as_u64().unwrap() > 0);
    assert!(st["throughput"]["decode_tps"].as_f64().unwrap() > 0.0);
    assert!(st["instance_id"].is_string());

    // CORS: allowed origins get headers and preflights, others do not
    let r = ureq::request("OPTIONS", &format!("{}/v1/chat/completions", s.base))
        .set("Origin", "http://localhost:1420")
        .set("Access-Control-Request-Method", "POST")
        .call()
        .unwrap();
    assert_eq!(r.header("access-control-allow-origin"), Some("http://localhost:1420"));
    assert!(r.header("access-control-allow-methods").unwrap().contains("POST"));
    let r = ureq::get(&format!("{}/health", s.base))
        .set("Origin", "http://evil.example")
        .call()
        .unwrap();
    assert_eq!(r.header("access-control-allow-origin"), None);

    // prepare-stop: drains, seals totals (idempotent), then refuses new work
    let sealed = post(&s, "/v1/admin/prepare-stop", json!({"drain_timeout_s": 2}));
    assert_eq!(sealed["drain_complete"], true);
    assert_eq!(sealed["completion_tokens_total"], 3);
    assert_eq!(post(&s, "/v1/admin/prepare-stop", json!({})), sealed);
    assert_eq!(get(&s, "/health")["maintenance"], "stopping");
    assert!(matches!(
        ureq::post(&format!("{}/v1/chat/completions", s.base))
            .send_json(json!({"messages": [{"role": "user", "content": "x"}]})),
        Err(ureq::Error::Status(503, _))
    ));
}
