// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Image input against HF transformers. The references come from
//! `scripts/qwen_vl_reference.py --out DIR` and the tests read DIR from `CARBON_VL_REF`; without
//! it they return early (generating the references needs Python + transformers).

use carbon_engine::config::ModelConfig;
use carbon_engine::model::layers::Placement;
use carbon_engine::model::vision::VisionTower;
use candle_core::{DType, Device, Tensor};
use std::path::PathBuf;

fn reference() -> Option<PathBuf> {
    std::env::var("CARBON_VL_REF").ok().map(PathBuf::from)
}

fn device() -> Device {
    match std::env::var("CARBON_TEST_DEVICE").as_deref() {
        Ok("cuda") => Device::new_cuda(0).unwrap(),
        _ => Device::Cpu,
    }
}

fn rel_err(a: &Tensor, b: &Tensor) -> f32 {
    let d = (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    let m = b.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    d / m.max(1e-6)
}

#[test]
fn vision_tower_matches_hf() {
    let Some(root) = reference() else { return };
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("reference.json")).unwrap()).unwrap();
    let grid: Vec<usize> = r["grid"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    for variant in ["q35", "q38"] {
        let dir = root.join(variant);
        let cfg = ModelConfig::from_dir_with(&dir, None).unwrap();
        let vcfg = cfg.vision.clone().expect("vision_config");
        let src = carbon_engine::checkpoint::open_weights(&dir).unwrap();
        let dev = device();
        let tower = VisionTower::load(&*src, &vcfg, &Placement::single(dev.clone(), DType::F32)).unwrap();
        let refs = candle_core::safetensors::load(root.join(format!("{variant}.safetensors")), &dev).unwrap();
        let out = tower.forward(&refs["pixel_values"], (grid[0], grid[1], grid[2])).unwrap();
        let want = &refs["image_embeds"];
        assert_eq!(out.dims(), want.dims(), "{variant}");
        let e = rel_err(&out.to_dtype(DType::F32).unwrap(), want);
        assert!(e < 1e-4, "{variant}: vision tower differs from HF by {e:.3e}");
    }
}

/// carbon's image processor against HF's on the reference image: same grid, pixel values within
/// 8-bit rounding of the resampling (the resize is reimplemented, not shared).
#[test]
fn image_preprocessing_matches_hf() {
    let Some(root) = reference() else { return };
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("reference.json")).unwrap()).unwrap();
    let grid: Vec<usize> = r["grid"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let dir = root.join("q35");
    let cfg = ModelConfig::from_dir_with(&dir, None).unwrap();
    let proc = carbon_engine::images::ImageProcessor::load(&dir, cfg.vision.as_ref().unwrap());
    let img = proc.prepare(&std::fs::read(root.join("image.png")).unwrap()).unwrap();
    assert_eq!([img.grid.0, img.grid.1, img.grid.2], [grid[0], grid[1], grid[2]]);
    let refs = candle_core::safetensors::load(root.join("q35.safetensors"), &Device::Cpu).unwrap();
    let want: Vec<f32> = refs["pixel_values"].flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(img.pixels.len(), want.len());
    let step = 2.0 / 255.0; // one 8-bit level after normalising with std 0.5
    let max = img.pixels.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let mean = img.pixels.iter().zip(&want).map(|(a, b)| (a - b).abs()).sum::<f32>() / want.len() as f32;
    assert!(max <= 1.01 * step && mean < 0.1 * step, "pixels differ: max {max}, mean {mean}");
}

/// mRoPE positions of the reference prompt equal HF's `get_rope_index`.
#[test]
fn mrope_positions_match_hf() {
    let Some(root) = reference() else { return };
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("reference.json")).unwrap()).unwrap();
    let prompt: Vec<u32> = r["prompt"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let grid: Vec<usize> = r["grid"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    // the reference prompt has the placeholders expanded: collapse them to one, as a template renders it
    let image_id = 259;
    let mut collapsed = vec![];
    for &t in &prompt {
        if !(t == image_id && collapsed.last() == Some(&image_id)) {
            collapsed.push(t);
        }
    }
    let img = carbon_engine::images::PreparedImage { pixels: vec![], grid: (grid[0], grid[1], grid[2]), hash: 1 };
    let (expanded, spans) = carbon_engine::images::expand_image_tokens(&collapsed, image_id, vec![img], 2).unwrap();
    assert_eq!(expanded.len(), prompt.len());
    let m = carbon_engine::images::MropePositions::new(&spans);
    let refs = candle_core::safetensors::load(root.join("q35.safetensors"), &Device::Cpu).unwrap();
    let want: Vec<Vec<i64>> = refs["position_ids"].to_vec2().unwrap();
    for i in 0..prompt.len() {
        let got = m.at(i);
        for a in 0..3 {
            assert_eq!(got[a] as i64, want[a][i], "token {i} axis {a}");
        }
    }
}

/// Greedy generation with an image through the scheduler equals HF's (`generate` with
/// pixel_values), for the Qwen3.5-MoE and Qwen3.8-Flash-Next layouts (`q38s`: QSA selecting
/// blocks). The engine gets HF's
/// pixel values so the comparison isolates the model; near-ties (margin < 1e-3) may flip.
#[test]
fn image_generation_matches_hf() {
    use carbon_engine::engine::{Engine, EngineConfig};
    use carbon_engine::sampling::SamplingParams;
    use carbon_engine::scheduler::{self, Event, SchedulerConfig};
    let Some(root) = reference() else { return };
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root.join("reference.json")).unwrap()).unwrap();
    let prompt: Vec<u32> = r["prompt"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let grid: Vec<usize> = r["grid"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let image_id = 259;
    let mut collapsed = vec![];
    for &t in &prompt {
        if !(t == image_id && collapsed.last() == Some(&image_id)) {
            collapsed.push(t);
        }
    }
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    for variant in ["q35", "q38", "q38s"] {
        let dir = root.join(variant);
        let refs = candle_core::safetensors::load(root.join(format!("{variant}.safetensors")), &Device::Cpu).unwrap();
        let pixels: Vec<f32> = refs["pixel_values"].flatten_all().unwrap().to_vec1().unwrap();
        let img = carbon_engine::images::PreparedImage { pixels, grid: (grid[0], grid[1], grid[2]), hash: 42 };
        let (expanded, spans) = carbon_engine::images::expand_image_tokens(&collapsed, image_id, vec![img], 2).unwrap();
        let mut ec = EngineConfig::new(&dir);
        if std::env::var("CARBON_TEST_DEVICE").as_deref() == Ok("cuda") {
            ec.devices = vec!["cuda:0".into()];
        }
        ec.dtype = Some(DType::F32);
        ec.kv_tokens = Some(1024);
        ec.max_running = 2;
        let engine = Engine::load(&ec).unwrap();
        assert!(engine.has_vision(), "{variant}: no vision tower");
        let handle = scheduler::spawn(
            engine,
            SchedulerConfig { max_running: 2, max_prefill_tokens: 16, eos_token_ids: vec![], anchor_token: None },
        )
        .unwrap();
        let want: Vec<u32> = r[variant]["tokens"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let margins: Vec<f64> = r[variant]["margins"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        // twice: the second run hits the prefix cache through the image's pseudo ids
        for run in 0..2 {
            let (_, mut rx) = handle.submit_with_images(
                expanded.clone(),
                spans.clone(),
                SamplingParams { max_tokens: want.len(), ignore_eos: true, ..Default::default() },
            );
            let got = rt.block_on(async {
                let mut toks = vec![];
                loop {
                    match rx.recv().await.unwrap() {
                        Event::Token(t) => toks.push(t),
                        Event::Done { .. } => return toks,
                        Event::Error(e) => panic!("{e}"),
                    }
                }
            });
            for i in 0..want.len() {
                if got[i] != want[i] {
                    assert!(margins[i] < 1e-3, "{variant} run {run}: token {i} {} != HF {} (margin {}): {got:?} vs {want:?}", got[i], want[i], margins[i]);
                    break; // a near-tie flipped; the rest follows another path
                }
            }
        }
    }
}

/// The q38 reference checkpoint with vision tokens in its tokenizer and a template that renders
/// image parts the way Qwen's does.
fn vision_server_model(root: &std::path::Path, dir: &std::path::Path) {
    let src = root.join("q38");
    std::fs::create_dir_all(dir).unwrap();
    for f in ["config.json", "model.safetensors", "preprocessor_config.json"] {
        std::fs::copy(src.join(f), dir.join(f)).unwrap();
    }
    let mut tok: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(src.join("tokenizer.json")).unwrap()).unwrap();
    for a in tok["added_tokens"].as_array_mut().unwrap() {
        let name = match a["id"].as_u64().unwrap() {
            259 => "<|image_pad|>",
            261 => "<|vision_start|>",
            262 => "<|vision_end|>",
            _ => continue,
        };
        a["content"] = serde_json::json!(name);
        a["special"] = serde_json::json!(true);
    }
    std::fs::write(dir.join("tokenizer.json"), tok.to_string()).unwrap();
    let mut cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(src.join("tokenizer_config.json")).unwrap()).unwrap();
    cfg["chat_template"] = serde_json::json!(concat!(
        "{%- for message in messages %}<|im_start|>{{ message.role }}\n",
        "{%- if message.content is string %}{{ message.content }}{%- else %}{%- for item in message.content %}",
        "{%- if item.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{%- else %}{{ item.text }}{%- endif %}",
        "{%- endfor %}{%- endif %}<|im_end|>\n{%- endfor %}<|im_start|>assistant\n"
    ));
    cfg.as_object_mut().unwrap().remove("added_tokens_decoder");
    std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
}

struct TestServer {
    base: String,
    _rt: tokio::runtime::Runtime,
}

fn serve(model: &std::path::Path) -> TestServer {
    use carbon_engine::engine::{Engine, EngineConfig};
    use carbon_engine::scheduler::{self, SchedulerConfig};
    use carbon_engine::server::{serve_listener, AppState, SamplingDefaults, ServerOptions};
    let mut ec = EngineConfig::new(model);
    ec.dtype = Some(DType::F32);
    ec.kv_tokens = Some(2048);
    let engine = Engine::load(&ec).unwrap();
    let handle = scheduler::spawn(
        engine,
        SchedulerConfig { max_running: 4, max_prefill_tokens: 64, eos_token_ids: vec![], anchor_token: None },
    )
    .unwrap();
    let tok = std::sync::Arc::new(carbon_engine::tokenizer::TokenizerBundle::load(model, None).unwrap());
    let state = std::sync::Arc::new(AppState::new(
        handle,
        tok,
        "tiny".into(),
        SamplingDefaults::from_model_dir(model),
        ServerOptions { max_output_tokens: 64, ..Default::default() },
    ));
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let listener = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    rt.spawn(async move { serve_listener(state, listener).await.unwrap() });
    TestServer { base, _rt: rt }
}

fn post(s: &TestServer, path: &str, body: serde_json::Value) -> Result<serde_json::Value, (u16, String)> {
    match ureq::post(&format!("{}{path}", s.base)).send_json(body) {
        Ok(r) => Ok(r.into_json().unwrap()),
        Err(ureq::Error::Status(code, r)) => Err((code, r.into_string().unwrap())),
        Err(e) => panic!("{e}"),
    }
}

/// OpenAI `image_url`, Anthropic `image` and Responses `input_image` parts reach the model the
/// same way: same prompt length (the image's tokens) and the same greedy output.
#[test]
fn image_requests_over_http() {
    use base64::Engine as _;
    use serde_json::json;
    let Some(root) = reference() else { return };
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m");
    vision_server_model(&root, &model);
    let s = serve(&model);
    let png = std::fs::read(root.join("image.png")).unwrap();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
    let url = format!("data:image/png;base64,{b64}");
    let text = "describe";

    let plain = post(&s, "/v1/chat/completions", json!({"messages": [{"role": "user", "content": text}], "max_tokens": 6, "temperature": 0})).unwrap();
    let openai = post(&s, "/v1/chat/completions", json!({
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]}],
        "max_tokens": 6, "temperature": 0,
    }))
    .unwrap();
    // 20 image tokens + vision start/end in place of the template's 3 placeholder tokens
    let image_tokens = openai["usage"]["prompt_tokens"].as_u64().unwrap() - plain["usage"]["prompt_tokens"].as_u64().unwrap();
    assert_eq!(image_tokens, 22);
    let want = openai["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    assert_ne!(want, plain["choices"][0]["message"]["content"].as_str().unwrap(), "the image changed nothing");

    let anthropic = post(&s, "/v1/messages", json!({
        "max_tokens": 6, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": text}]}],
    }))
    .unwrap();
    assert_eq!(anthropic["content"][0]["text"].as_str().unwrap(), want);
    assert_eq!(anthropic["usage"]["input_tokens"], openai["usage"]["prompt_tokens"]);

    let responses = post(&s, "/v1/responses", json!({
        "max_output_tokens": 6, "temperature": 0,
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_image", "image_url": url}, {"type": "input_text", "text": text}]}],
    }))
    .unwrap();
    assert_eq!(responses["output"].as_array().unwrap().last().unwrap()["content"][0]["text"].as_str().unwrap(), want);

    // bare base64 (no data: prefix) works as well
    let bare = post(&s, "/v1/chat/completions", json!({
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": b64}}, {"type": "text", "text": text}]}],
        "max_tokens": 6, "temperature": 0,
    }))
    .unwrap();
    assert_eq!(bare["choices"][0]["message"]["content"].as_str().unwrap(), want);

    let stats: serde_json::Value = ureq::get(&format!("{}/v1/stats", s.base)).call().unwrap().into_json().unwrap();
    assert_eq!(stats["model"]["input_modalities"], json!(["text", "image"]));

    // no local files from the API; undecodable data is a client error
    let file = post(&s, "/v1/chat/completions", json!({
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": root.join("image.png").to_str().unwrap()}}]}],
    }));
    assert_eq!(file.unwrap_err().0, 400);
    let junk = post(&s, "/v1/chat/completions", json!({
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}],
    }));
    assert_eq!(junk.unwrap_err().0, 400);
}

/// A model without a vision tower refuses images with a 400.
#[test]
fn text_only_model_rejects_images() {
    use serde_json::json;
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m");
    carbon_engine::testing::write_tiny_model(&model, &carbon_engine::testing::TinySpec::new("qwen3")).unwrap();
    let s = serve(&model);
    let r = post(&s, "/v1/chat/completions", json!({
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}, {"type": "text", "text": "hi"}]}],
    }));
    let (code, msg) = r.unwrap_err();
    assert_eq!(code, 400);
    assert!(msg.contains("image"), "{msg}");
}
