// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Checks against the real Qwen3.6 / Qwen3.8 config + tokenizer files committed next to the
//! engine (no weights). Skipped when the files are absent.

use carbon_engine::config::{Arch, LayerKind, ModelConfig};
use carbon_engine::server::parsers::parse_all;
use carbon_engine::tokenizer::TokenizerBundle;
use serde_json::json;
use std::path::PathBuf;

fn dir(name: &str) -> Option<PathBuf> {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(name);
    d.join("config.json").exists().then_some(d)
}

#[test]
fn qwen36_35b_a3b_fp8_config() {
    let Some(d) = dir("Qwen3.6-35B-A3B-FP8-tokenizer") else {
        return;
    };
    let c = ModelConfig::from_dir(&d).unwrap();
    assert_eq!(c.arch, Arch::Qwen35);
    assert_eq!(c.num_layers, 40);
    assert_eq!(
        c.layer_types
            .iter()
            .filter(|t| **t == LayerKind::Full)
            .count(),
        10
    );
    assert_eq!((c.hidden_size, c.head_dim, c.rotary_dim), (2048, 256, 64));
    let m = c.moe.as_ref().unwrap();
    assert_eq!(
        (
            m.num_experts,
            m.top_k,
            m.intermediate_size,
            m.shared_expert_intermediate_size
        ),
        (256, 8, 512, 512)
    );
    let l = c.linear_attn.as_ref().unwrap();
    assert_eq!(
        (l.num_k_heads, l.num_v_heads, l.k_head_dim, l.v_head_dim),
        (16, 32, 128, 128)
    );
    assert!(c.attn_output_gate && c.norm_zero_centered);
    // per sequence: 30 linear layers of recurrent state in f32
    let per_seq = 30 * l.state_elems() * 4;
    assert!(per_seq < 80 << 20, "{per_seq}");
}

#[test]
fn qwen36_tokenizer_template_and_tool_calls() {
    let Some(d) = dir("Qwen3.6-35B-A3B-FP8-tokenizer") else {
        return;
    };
    let tok = TokenizerBundle::load(&d, None).unwrap();
    for t in ["<think>", "</think>", "<tool_call>", "<|im_end|>"] {
        assert!(tok.token_id(t).is_some(), "{t}");
    }
    let tools =
        json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]);
    let p = tok
        .apply_chat_template(
            &[json!({"role": "user", "content": "hi"})],
            Some(&tools),
            true,
            &Default::default(),
        )
        .unwrap();
    assert!(
        p.trim_end().ends_with("<think>"),
        "generation prompt opens a think block"
    );
    // what the model emits after that prompt
    let out = "plan the call\n</think>\n\n<tool_call>\n<function=f>\n<parameter=x>\n3\n</parameter>\n</function>\n</tool_call>";
    let (r, c, calls) = parse_all(out, true, true, true);
    assert_eq!(r, "plan the call\n");
    assert_eq!(c, "");
    assert_eq!(calls, vec![("f".to_string(), "{\"x\":3}".to_string())]);
}

#[test]
fn qwen38_flash_next_config() {
    let Some(d) = dir("Qwen3.8-Flash-Next-NVFP4-tokenizer") else {
        return;
    };
    let c = ModelConfig::from_dir(&d).unwrap();
    assert_eq!(c.arch, Arch::Qwen35);
    assert_eq!((c.hidden_size, c.num_layers, c.num_heads, c.num_kv_heads, c.head_dim), (2560, 48, 24, 2, 256));
    assert_eq!(c.rotary_dim, 64);
    assert!(c.gdn_gate_sigmoid && c.attn_output_gate && c.norm_zero_centered);
    let full: Vec<usize> = (0..48).filter(|&l| c.layer_types[l] == LayerKind::Full).collect();
    assert_eq!(full, (3..48).step_by(4).collect::<Vec<_>>());
    let moe = c.moe.as_ref().unwrap();
    assert_eq!((moe.num_experts, moe.top_k, moe.intermediate_size, moe.shared_expert_intermediate_size), (512, 10, 640, 640));
    assert!(moe.norm_topk_prob);
    let q = c.qwen38.as_ref().unwrap();
    assert_eq!((q.hc_count, q.hc_lowrank), (4, 320));
    assert_eq!(q.ple_layers, vec![1]);
    assert_eq!((q.ngram_heads(), q.ngram_head_dim(), q.split_ngram_parts), (16, 160, 128));
    assert_eq!(q.ngram_boundary, 248044);
    assert_eq!((q.index_heads, q.index_head_dim, q.index_budget, q.index_ratio, q.index_topk_blocks()), (4, 128, 2048, 4, 512));
    assert_eq!(q.ple_conv_state_len(), 9);
    // the n-gram table is 128 shards of 2,500,012 rows (FreeToken's checkpoint test)
    let h = carbon_engine::model::hyper::NgramHasher::new(q, c.vocab_size, 0);
    assert_eq!(h.total_rows, 128 * 2_500_012);
}

#[test]
fn qwen38_template_efforts_and_tool_calls() {
    use carbon_engine::server::effort::ThinkingProfile;
    use serde_json::{Map, Value};
    let Some(d) = dir("Qwen3.8-Flash-Next-NVFP4-tokenizer") else {
        return;
    };
    let tok = TokenizerBundle::load(&d, None).unwrap();
    for t in ["<think>", "</think>", "<tool_call>", "<|im_end|>"] {
        assert!(tok.token_id(t).is_some(), "{t}");
    }
    let render = |kw: &Map<String, Value>, tools: bool| {
        let tools = tools.then(|| json!([{"type": "function", "function": {"name": "noop", "parameters": {"type": "object", "properties": {}}}}]));
        tok.apply_chat_template(&[json!({"role": "user", "content": "Hello"})], tools.as_ref(), true, kw)
            .map_err(|e| e.to_string())
    };
    let p = ThinkingProfile::probe(&render);
    // the template grades effort itself and raises on anything else
    let e = &p.efforts;
    assert!(e.consumes_effort && e.validates);
    let mut levels = e.effective();
    levels.sort();
    assert_eq!(levels, vec!["low", "medium", "xhigh"]);
    assert_eq!(e.default.as_deref(), Some("xhigh"));
    // OpenAI's "high" lands on a level the template accepts and renders
    let kw = p.apply(Some("high"), None, Map::new());
    let lvl = kw["reasoning_effort"].as_str().unwrap().to_string();
    assert!(["xhigh", "medium"].contains(&lvl.as_str()), "{lvl}");
    assert!(render(&kw, true).is_ok());
    let out = "plan\n</think>\n\n<tool_call>\n<function=f>\n<parameter=x>\n3\n</parameter>\n</function>\n</tool_call>";
    let (r, c, calls) = parse_all(out, true, true, true);
    assert_eq!((r.as_str(), c.as_str()), ("plan\n", ""));
    assert_eq!(calls, vec![("f".to_string(), "{\"x\":3}".to_string())]);
}
