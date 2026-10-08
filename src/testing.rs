// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Tiny random checkpoints for tests, demos and benchmarking the parallel paths
//! (`carbon tiny-model`). They follow the Hugging Face layout exactly, so the same
//! files load in `transformers` for cross-checking.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;

pub const SPECIAL_TOKENS: [(&str, bool); 7] = [
    ("<|endoftext|>", true),
    ("<|im_start|>", true),
    ("<|im_end|>", true),
    ("<think>", false),
    ("</think>", false),
    ("<tool_call>", false),
    ("</tool_call>", false),
];

const CHAT_TEMPLATE: &str = r##"{%- if tools %}
{{- '<|im_start|>system\n' }}
{%- if messages[0].role == 'system' %}{{- messages[0].content + '\n\n' }}{%- endif %}
{{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n<tools>" }}
{%- for tool in tools %}{{- "\n" }}{{- tool | tojson }}{%- endfor %}
{{- "\n</tools>\n\nFor each function call, return a json object within <tool_call></tool_call> XML tags.<|im_end|>\n" }}
{%- elif messages[0].role == 'system' %}
{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
{%- if message.role == 'system' %}
{%- elif message.role == 'tool' %}
{{- '<|im_start|>user\n<tool_response>\n' + message.content + '\n</tool_response><|im_end|>\n' }}
{%- else %}
{{- '<|im_start|>' + message.role + '\n' + (message.content or '') }}
{%- if message.tool_calls %}{%- for tc in message.tool_calls %}
{{- '<tool_call>\n{"name": "' + tc.function.name + '", "arguments": ' + (tc.function.arguments | tojson) + '}\n</tool_call>' }}
{%- endfor %}{%- endif %}
{{- '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- if enable_thinking is defined and enable_thinking %}{{- '<think>\n' }}{%- endif %}
{%- endif %}"##;

/// GPT-2 byte-to-unicode table used by byte-level BPE.
fn byte_chars() -> Vec<char> {
    let mut bs: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut table = vec![' '; 256];
    for (b, c) in bs.iter().zip(cs) {
        table[*b as usize] = char::from_u32(c).unwrap();
    }
    table
}

/// Byte-level tokenizer: one token per byte plus the special tokens (ids 256..263).
pub fn tokenizer_json() -> Value {
    let chars = byte_chars();
    let mut vocab = serde_json::Map::new();
    for (i, c) in chars.iter().enumerate() {
        vocab.insert(c.to_string(), json!(i));
    }
    let added: Vec<Value> = SPECIAL_TOKENS
        .iter()
        .enumerate()
        .map(|(i, (t, special))| {
            json!({"id": 256 + i, "content": t, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": special})
        })
        .collect();
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                  "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                  "vocab": vocab, "merges": []}
    })
}

pub const TINY_VOCAB: usize = 256 + SPECIAL_TOKENS.len();
pub const IM_END: u32 = 258;

struct Gen {
    rng: StdRng,
    map: HashMap<String, Tensor>,
    dtype: DType,
}

impl Gen {
    fn rand(&mut self, name: &str, shape: &[usize], std: f32) -> Result<()> {
        let n: usize = shape.iter().product();
        // sum of uniforms: cheap, deterministic, roughly normal
        let v: Vec<f32> = (0..n)
            .map(|_| {
                (self.rng.gen::<f32>() + self.rng.gen::<f32>() + self.rng.gen::<f32>() - 1.5)
                    * 2.0
                    * std
            })
            .collect();
        self.map.insert(
            name.into(),
            Tensor::from_vec(v, shape, &Device::Cpu)?.to_dtype(self.dtype)?,
        );
        Ok(())
    }

    fn norm(&mut self, name: &str, n: usize) -> Result<()> {
        let v: Vec<f32> = (0..n)
            .map(|_| 1.0 + (self.rng.gen::<f32>() - 0.5) * 0.2)
            .collect();
        self.map.insert(
            name.into(),
            Tensor::from_vec(v, n, &Device::Cpu)?.to_dtype(self.dtype)?,
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TinySpec {
    pub arch: String,
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub experts: usize,
    pub top_k: usize,
    pub moe_intermediate: usize,
    pub seed: u64,
    pub dtype: DType,
}

impl TinySpec {
    pub fn new(arch: &str) -> Self {
        Self {
            arch: arch.into(),
            layers: 4,
            hidden: 64,
            heads: 4,
            kv_heads: 2,
            head_dim: 16,
            intermediate: 128,
            experts: 8,
            top_k: 2,
            moe_intermediate: 32,
            seed: 0,
            dtype: DType::F32,
        }
    }
}

pub fn write_tiny_model(dir: &Path, spec: &TinySpec) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    if spec.arch == "qwen3_5_moe" || spec.arch == "qwen4_exp" {
        return write_tiny_qwen35(dir, spec);
    }
    let (model_type, arch_name) = match spec.arch.as_str() {
        "llama" => ("llama", "LlamaForCausalLM"),
        "qwen2" => ("qwen2", "Qwen2ForCausalLM"),
        "qwen3" => ("qwen3", "Qwen3ForCausalLM"),
        "qwen3_moe" => ("qwen3_moe", "Qwen3MoeForCausalLM"),
        "mixtral" => ("mixtral", "MixtralForCausalLM"),
        a => {
            bail!("unknown tiny-model arch '{a}' (llama|qwen2|qwen3|qwen3_moe|mixtral|qwen3_5_moe|qwen4_exp)")
        }
    };
    let (h, nh, kvh, hd, inter) = (
        spec.hidden,
        spec.heads,
        spec.kv_heads,
        spec.head_dim,
        spec.intermediate,
    );
    let is_moe = matches!(model_type, "qwen3_moe" | "mixtral");
    let mut config = json!({
        "architectures": [arch_name],
        "model_type": model_type,
        "hidden_size": h,
        "intermediate_size": if model_type == "mixtral" { spec.moe_intermediate } else { inter },
        "num_hidden_layers": spec.layers,
        "num_attention_heads": nh,
        "num_key_value_heads": kvh,
        "head_dim": hd,
        "vocab_size": TINY_VOCAB,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": false,
        "bos_token_id": 256,
        "eos_token_id": IM_END,
        "torch_dtype": "float32",
        "attention_bias": model_type == "qwen2",
    });
    if model_type == "qwen3_moe" {
        config["num_experts"] = json!(spec.experts);
        config["num_experts_per_tok"] = json!(spec.top_k);
        config["moe_intermediate_size"] = json!(spec.moe_intermediate);
        config["norm_topk_prob"] = json!(true);
        config["decoder_sparse_step"] = json!(1);
        config["mlp_only_layers"] = json!([0]);
    }
    if model_type == "mixtral" {
        config["num_local_experts"] = json!(spec.experts);
        config["num_experts_per_tok"] = json!(spec.top_k);
    }
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config)?,
    )?;
    std::fs::write(
        dir.join("generation_config.json"),
        serde_json::to_string_pretty(&json!({"eos_token_id": [IM_END, 256], "do_sample": false}))?,
    )?;
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string(&tokenizer_json())?,
    )?;
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_string_pretty(&json!({
            "chat_template": CHAT_TEMPLATE, "eos_token": "<|im_end|>", "bos_token": null,
            "model_max_length": 4096, "tokenizer_class": "PreTrainedTokenizerFast"
        }))?,
    )?;

    let mut g = Gen {
        rng: StdRng::seed_from_u64(spec.seed),
        map: HashMap::new(),
        dtype: spec.dtype,
    };
    let s_in = |n: usize| 1.0 / (n as f32).sqrt();
    g.rand("model.embed_tokens.weight", &[TINY_VOCAB, h], 1.0)?;
    g.rand("lm_head.weight", &[TINY_VOCAB, h], s_in(h) * 2.0)?;
    g.norm("model.norm.weight", h)?;
    for l in 0..spec.layers {
        let p = format!("model.layers.{l}");
        g.norm(&format!("{p}.input_layernorm.weight"), h)?;
        g.norm(&format!("{p}.post_attention_layernorm.weight"), h)?;
        g.rand(
            &format!("{p}.self_attn.q_proj.weight"),
            &[nh * hd, h],
            s_in(h),
        )?;
        g.rand(
            &format!("{p}.self_attn.k_proj.weight"),
            &[kvh * hd, h],
            s_in(h),
        )?;
        g.rand(
            &format!("{p}.self_attn.v_proj.weight"),
            &[kvh * hd, h],
            s_in(h),
        )?;
        g.rand(
            &format!("{p}.self_attn.o_proj.weight"),
            &[h, nh * hd],
            s_in(nh * hd),
        )?;
        if model_type == "qwen2" {
            g.rand(&format!("{p}.self_attn.q_proj.bias"), &[nh * hd], 0.1)?;
            g.rand(&format!("{p}.self_attn.k_proj.bias"), &[kvh * hd], 0.1)?;
            g.rand(&format!("{p}.self_attn.v_proj.bias"), &[kvh * hd], 0.1)?;
        }
        if matches!(model_type, "qwen3" | "qwen3_moe") {
            g.norm(&format!("{p}.self_attn.q_norm.weight"), hd)?;
            g.norm(&format!("{p}.self_attn.k_norm.weight"), hd)?;
        }
        let moe_layer = is_moe && !(model_type == "qwen3_moe" && l == 0);
        if moe_layer {
            let mi = spec.moe_intermediate;
            let (block, names) = if model_type == "mixtral" {
                (format!("{p}.block_sparse_moe"), ["w1", "w3", "w2"])
            } else {
                (format!("{p}.mlp"), ["gate_proj", "up_proj", "down_proj"])
            };
            g.rand(
                &format!("{block}.gate.weight"),
                &[spec.experts, h],
                s_in(h) * 3.0,
            )?;
            for e in 0..spec.experts {
                g.rand(
                    &format!("{block}.experts.{e}.{}.weight", names[0]),
                    &[mi, h],
                    s_in(h),
                )?;
                g.rand(
                    &format!("{block}.experts.{e}.{}.weight", names[1]),
                    &[mi, h],
                    s_in(h),
                )?;
                g.rand(
                    &format!("{block}.experts.{e}.{}.weight", names[2]),
                    &[h, mi],
                    s_in(mi),
                )?;
            }
        } else {
            g.rand(&format!("{p}.mlp.gate_proj.weight"), &[inter, h], s_in(h))?;
            g.rand(&format!("{p}.mlp.up_proj.weight"), &[inter, h], s_in(h))?;
            g.rand(
                &format!("{p}.mlp.down_proj.weight"),
                &[h, inter],
                s_in(inter),
            )?;
        }
    }
    candle_core::safetensors::save(&g.map, dir.join("model.safetensors"))?;
    Ok(())
}

fn write_tokenizer_files(dir: &Path) -> Result<()> {
    std::fs::write(
        dir.join("generation_config.json"),
        serde_json::to_string_pretty(&json!({"eos_token_id": [IM_END, 256], "do_sample": false}))?,
    )?;
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string(&tokenizer_json())?,
    )?;
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_string_pretty(&json!({
            "chat_template": CHAT_TEMPLATE, "eos_token": "<|im_end|>", "bos_token": null,
            "model_max_length": 4096, "tokenizer_class": "PreTrainedTokenizerFast"
        }))?,
    )?;
    Ok(())
}

/// Qwen3.5-MoE in the release layout: multimodal config wrapper, `model.language_model.*`
/// names, alternating Gated-DeltaNet / gated full attention, routed + shared experts.
/// `qwen4_exp` (Qwen3.8-Flash-Next) adds hyper-connections (3 streams), PLE on layers 0 and 2
/// and a QSA indexer with a 16-token budget in blocks of 4.
fn write_tiny_qwen35(dir: &Path, spec: &TinySpec) -> Result<()> {
    let q38 = spec.arch == "qwen4_exp";
    let (h, nh, kvh, hd) = (spec.hidden, spec.heads, spec.kv_heads, spec.head_dim);
    let (lk, lv, lkd, lvd, kernel) = (2usize, 4usize, 8usize, 8usize, 4usize);
    let (mi, shared) = (spec.moe_intermediate, 48usize);
    let (hc, lowrank, ple_dim, idx_heads, idx_dim) = (3usize, 12usize, 16usize, 4usize, 16usize);
    let layer_types: Vec<&str> = (0..spec.layers)
        .map(|i| match (i % 2 == 1, q38) {
            (true, true) => "indexed_attention",
            (true, false) => "full_attention",
            _ => "linear_attention",
        })
        .collect();
    let ple_layers: Vec<usize> = if q38 { (0..spec.layers).filter(|l| l % 2 == 0).take(2).collect() } else { vec![] };
    let mut text = json!({
        "model_type": "qwen3_5_moe_text", "hidden_size": h, "num_hidden_layers": spec.layers,
        "num_attention_heads": nh, "num_key_value_heads": kvh, "head_dim": hd, "vocab_size": TINY_VOCAB,
        "linear_num_key_heads": lk, "linear_num_value_heads": lv, "linear_key_head_dim": lkd,
        "linear_value_head_dim": lvd, "linear_conv_kernel_dim": kernel, "layer_types": layer_types,
        "num_experts": spec.experts, "num_experts_per_tok": spec.top_k, "moe_intermediate_size": mi,
        "shared_expert_intermediate_size": shared, "rms_norm_eps": 1e-6, "hidden_act": "silu",
        "max_position_embeddings": 4096, "eos_token_id": IM_END, "attn_output_gate": true,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                            "mrope_section": [1, 1, 0], "mrope_interleaved": true},
    });
    if q38 {
        let o = text.as_object_mut().unwrap();
        o.insert("model_type".into(), json!("qwen4_exp_text"));
        o.remove("attn_output_gate");
        o.insert("rope_parameters".into(), json!({"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5}));
        for (k, v) in [
            ("hc_count", json!(hc)), ("hc_lowrank", json!(lowrank)),
            ("ple_layer_ids", json!(ple_layers.iter().map(|l| l + 1).collect::<Vec<_>>())),
            ("ple_embed_dim", json!(ple_dim)), ("ple_conv_kernel_size", json!(3)), ("ngram_size", json!(3)),
            ("heads_per_ngram", json!(2)), ("ngram_vocab_size_base", json!(97)),
            ("make_ngram_vocab_size_divisible_by", json!(8)), ("split_ngram_parts", json!(4)), ("seed", json!(1234)),
            ("indexer_n_heads", json!(idx_heads)), ("indexer_kv_heads", json!(1)), ("indexer_head_dim", json!(idx_dim)),
            ("indexer_budget", json!(16)), ("indexer_compress_ratio", json!(4)),
            ("output_gate_type", json!("sigmoid")),
        ] {
            o.insert(k.into(), v);
        }
    }
    let config = if q38 {
        json!({"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
               "tie_word_embeddings": false, "text_config": text})
    } else {
        json!({"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
               "tie_word_embeddings": false, "text_config": text})
    };
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config)?,
    )?;
    write_tokenizer_files(dir)?;

    let mut g = Gen {
        rng: StdRng::seed_from_u64(spec.seed),
        map: HashMap::new(),
        dtype: spec.dtype,
    };
    let s_in = |n: usize| 1.0 / (n as f32).sqrt();
    let pre = "model.language_model.";
    g.rand(&format!("{pre}embed_tokens.weight"), &[TINY_VOCAB, h], 1.0)?;
    g.rand("lm_head.weight", &[TINY_VOCAB, h], s_in(h) * 2.0)?;
    let hyper = |g: &mut Gen, p: &str, inject: bool| -> Result<()> {
        g.rand(&format!("{p}.hc_norm.weight"), &[hc * h], 0.1)?;
        g.rand(&format!("{p}.input_mix_weight_down.weight"), &[lowrank, hc * h], s_in(hc * h))?;
        g.rand(&format!("{p}.input_mix_weight_up.weight"), &[hc * h, lowrank], s_in(lowrank))?;
        if inject {
            g.rand(&format!("{p}.block_inject_weight.weight"), &[hc, hc * h], s_in(hc * h))?;
        }
        Ok(())
    };
    if q38 {
        hyper(&mut g, &format!("{pre}hyper_connection_mixer"), false)?;
    } else {
        g.rand(&format!("{pre}norm.weight"), &[h], 0.1)?;
    }
    let conv_dim = 2 * lk * lkd + lv * lvd;
    for (l, kind) in layer_types.iter().enumerate() {
        let p = format!("{pre}layers.{l}");
        if q38 {
            hyper(&mut g, &format!("{p}.attn_hyper_connection"), true)?;
            hyper(&mut g, &format!("{p}.mlp_hyper_connection"), true)?;
            if let Some(i) = ple_layers.iter().position(|&x| x == l) {
                let e = format!("{p}.ple");
                g.rand(&format!("{e}.conv1d.weight"), &[hc * h, 1, 3], 0.5)?;
                g.rand(&format!("{e}.key_proj.weight"), &[hc * h, ple_dim], s_in(ple_dim))?;
                g.rand(&format!("{e}.value_proj.weight"), &[h, ple_dim], s_in(ple_dim))?;
                for n in ["norm_key", "norm_query", "norm_conv"] {
                    g.rand(&format!("{e}.{n}.weight"), &[hc * h], 0.1)?;
                }
                let qc = crate::config::ModelConfig::from_hf(&config)?.qwen38.unwrap();
                let rows = crate::model::hyper::NgramHasher::new(&qc, TINY_VOCAB, i).total_rows as usize;
                for part in 0..4 {
                    g.rand(
                        &format!("{e}.ple_embedding.ngram_embedding.shard_{part}.weight"),
                        &[rows / 4, qc.ngram_head_dim()],
                        1.0,
                    )?;
                }
            }
        } else {
            // zero-centred norms: stored as offsets from 1
            g.rand(&format!("{p}.input_layernorm.weight"), &[h], 0.1)?;
            g.rand(&format!("{p}.post_attention_layernorm.weight"), &[h], 0.1)?;
        }
        if *kind != "linear_attention" {
            if q38 {
                let ix = format!("{p}.self_attn.indexer");
                g.rand(&format!("{ix}.index_qk_proj.weight"), &[(idx_heads + 1) * idx_dim, h], s_in(h))?;
                g.rand(&format!("{ix}.q_layernorm.weight"), &[idx_dim], 0.1)?;
                g.rand(&format!("{ix}.k_layernorm.weight"), &[idx_dim], 0.1)?;
            }
            g.rand(
                &format!("{p}.self_attn.q_proj.weight"),
                &[2 * nh * hd, h],
                s_in(h),
            )?;
            g.rand(
                &format!("{p}.self_attn.k_proj.weight"),
                &[kvh * hd, h],
                s_in(h),
            )?;
            g.rand(
                &format!("{p}.self_attn.v_proj.weight"),
                &[kvh * hd, h],
                s_in(h),
            )?;
            g.rand(
                &format!("{p}.self_attn.o_proj.weight"),
                &[h, nh * hd],
                s_in(nh * hd),
            )?;
            g.rand(&format!("{p}.self_attn.q_norm.weight"), &[hd], 0.1)?;
            g.rand(&format!("{p}.self_attn.k_norm.weight"), &[hd], 0.1)?;
        } else {
            let a = format!("{p}.linear_attn");
            g.rand(&format!("{a}.in_proj_qkv.weight"), &[conv_dim, h], s_in(h))?;
            g.rand(&format!("{a}.in_proj_z.weight"), &[lv * lvd, h], s_in(h))?;
            g.rand(&format!("{a}.in_proj_b.weight"), &[lv, h], s_in(h))?;
            g.rand(&format!("{a}.in_proj_a.weight"), &[lv, h], s_in(h))?;
            g.rand(&format!("{a}.conv1d.weight"), &[conv_dim, 1, kernel], 0.5)?;
            g.rand(&format!("{a}.dt_bias"), &[lv], 0.5)?;
            g.norm(&format!("{a}.A_log"), lv)?;
            g.norm(&format!("{a}.norm.weight"), lvd)?;
            g.rand(
                &format!("{a}.out_proj.weight"),
                &[h, lv * lvd],
                s_in(lv * lvd),
            )?;
        }
        let m = format!("{p}.mlp");
        g.rand(
            &format!("{m}.gate.weight"),
            &[spec.experts, h],
            s_in(h) * 3.0,
        )?;
        for e in 0..spec.experts {
            g.rand(
                &format!("{m}.experts.{e}.gate_proj.weight"),
                &[mi, h],
                s_in(h),
            )?;
            g.rand(
                &format!("{m}.experts.{e}.up_proj.weight"),
                &[mi, h],
                s_in(h),
            )?;
            g.rand(
                &format!("{m}.experts.{e}.down_proj.weight"),
                &[h, mi],
                s_in(mi),
            )?;
        }
        g.rand(
            &format!("{m}.shared_expert.gate_proj.weight"),
            &[shared, h],
            s_in(h),
        )?;
        g.rand(
            &format!("{m}.shared_expert.up_proj.weight"),
            &[shared, h],
            s_in(h),
        )?;
        g.rand(
            &format!("{m}.shared_expert.down_proj.weight"),
            &[h, shared],
            s_in(shared),
        )?;
        g.rand(&format!("{m}.shared_expert_gate.weight"), &[1, h], s_in(h))?;
    }
    if q38 {
        // the MTP head (generated last: the other tensors do not change): one indexed-attention
        // layer with stacked experts, as the Qwen3.8 checkpoints store it
        g.rand("mtp.pre_fc_norm_embedding.weight", &[h], 0.1)?;
        g.rand("mtp.pre_fc_norm_hidden.weight", &[hc * h], 0.1)?;
        g.rand("mtp.fc_embedding.weight", &[h, h], s_in(h))?;
        g.rand("mtp.fc_hidden.weight", &[h, h], s_in(h))?;
        hyper(&mut g, "mtp.hyper_connection_mixer", false)?;
        let p = "mtp.layers.0";
        hyper(&mut g, &format!("{p}.attn_hyper_connection"), true)?;
        hyper(&mut g, &format!("{p}.mlp_hyper_connection"), true)?;
        let ix = format!("{p}.self_attn.indexer");
        g.rand(&format!("{ix}.index_qk_proj.weight"), &[(idx_heads + 1) * idx_dim, h], s_in(h))?;
        g.rand(&format!("{ix}.q_layernorm.weight"), &[idx_dim], 0.1)?;
        g.rand(&format!("{ix}.k_layernorm.weight"), &[idx_dim], 0.1)?;
        let a = format!("{p}.self_attn");
        g.rand(&format!("{a}.q_proj.weight"), &[2 * nh * hd, h], s_in(h))?;
        g.rand(&format!("{a}.k_proj.weight"), &[kvh * hd, h], s_in(h))?;
        g.rand(&format!("{a}.v_proj.weight"), &[kvh * hd, h], s_in(h))?;
        g.rand(&format!("{a}.o_proj.weight"), &[h, nh * hd], s_in(nh * hd))?;
        g.rand(&format!("{a}.q_norm.weight"), &[hd], 0.1)?;
        g.rand(&format!("{a}.k_norm.weight"), &[hd], 0.1)?;
        let m = format!("{p}.mlp");
        g.rand(&format!("{m}.gate.weight"), &[spec.experts, h], s_in(h) * 3.0)?;
        g.rand(&format!("{m}.experts.gate_up_proj"), &[spec.experts, 2 * mi, h], s_in(h))?;
        g.rand(&format!("{m}.experts.down_proj"), &[spec.experts, h, mi], s_in(mi))?;
        g.rand(&format!("{m}.shared_expert.gate_proj.weight"), &[shared, h], s_in(h))?;
        g.rand(&format!("{m}.shared_expert.up_proj.weight"), &[shared, h], s_in(h))?;
        g.rand(&format!("{m}.shared_expert.down_proj.weight"), &[h, shared], s_in(shared))?;
        g.rand(&format!("{m}.shared_expert_gate.weight"), &[1, h], s_in(h))?;
    }
    candle_core::safetensors::save(&g.map, dir.join("model.safetensors"))?;
    Ok(())
}

/// Weight-quantization formats the tiny-model generator can produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TinyQuant {
    /// 16x16 block FP8 with `weight_scale_inv` (DeepSeek / Qwen FP8 release style).
    Fp8Block,
    /// ModelOpt NVFP4: `weight` (packed u8), `weight_scale` (e4m3 per 16), `weight_scale_2`.
    Nvfp4ModelOpt,
    /// compressed-tensors NVFP4: `weight_packed`, `weight_scale`, `weight_global_scale` (reciprocal).
    Nvfp4CompressedTensors,
}

fn quantizable(name: &str) -> bool {
    const PROJ: [&str; 12] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
        "w1",
        "w3",
    ];
    name.ends_with(".weight") && PROJ.iter().any(|p| name.ends_with(&format!(".{p}.weight")))
        || name.ends_with(".w2.weight")
}

/// Write a quantized tiny checkpoint to `qdir` and its exact dequantized dense twin to `ddir`.
pub fn write_tiny_quantized(
    qdir: &Path,
    ddir: &Path,
    spec: &TinySpec,
    quant: TinyQuant,
) -> Result<()> {
    write_tiny_model(ddir, spec)?;
    std::fs::create_dir_all(qdir)?;
    for f in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ] {
        std::fs::copy(ddir.join(f), qdir.join(f))?;
    }
    let mut cfg: Value = serde_json::from_str(&std::fs::read_to_string(qdir.join("config.json"))?)?;
    cfg["quantization_config"] = match quant {
        TinyQuant::Fp8Block => {
            json!({"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [16, 16]})
        }
        TinyQuant::Nvfp4ModelOpt => json!({"quant_method": "modelopt", "quant_algo": "NVFP4"}),
        TinyQuant::Nvfp4CompressedTensors => {
            json!({"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized"})
        }
    };
    std::fs::write(
        qdir.join("config.json"),
        serde_json::to_string_pretty(&cfg)?,
    )?;

    let dense = candle_core::safetensors::load(ddir.join("model.safetensors"), &Device::Cpu)?;
    let mut qmap = HashMap::new();
    let mut dmap = HashMap::new();
    for (name, t) in dense {
        if !(quantizable(&name) && t.rank() == 2) {
            qmap.insert(name.clone(), t.clone());
            dmap.insert(name, t);
            continue;
        }
        let (rows, cols) = t.dims2()?;
        let w = t.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let base = name.strip_suffix(".weight").unwrap().to_string();
        let deq: Vec<f32> = match quant {
            TinyQuant::Fp8Block => {
                let (sr, sc) = (rows.div_ceil(16), cols.div_ceil(16));
                let mut scale = vec![0f32; sr * sc];
                for r in 0..rows {
                    for c in 0..cols {
                        let s = &mut scale[(r / 16) * sc + c / 16];
                        *s = s.max(w[r][c].abs() / 448.0);
                    }
                }
                scale.iter_mut().for_each(|s| *s = s.max(1e-12));
                let scaled: Vec<f32> = (0..rows * cols)
                    .map(|i| w[i / cols][i % cols] / scale[(i / cols / 16) * sc + (i % cols) / 16])
                    .collect();
                let q = Tensor::from_vec(scaled, (rows, cols), &Device::Cpu)?
                    .to_dtype(DType::F8E4M3)?;
                let st = Tensor::from_vec(scale, (sr, sc), &Device::Cpu)?;
                let d = crate::weights::dequant_fp8(&q, &st, DType::F32)?;
                qmap.insert(name.clone(), q);
                qmap.insert(format!("{name}_scale_inv"), st);
                d.flatten_all()?.to_vec1()?
            }
            TinyQuant::Nvfp4ModelOpt | TinyQuant::Nvfp4CompressedTensors => {
                let amax = w
                    .iter()
                    .flatten()
                    .fold(0f32, |a, x| a.max(x.abs()))
                    .max(1e-12);
                let global = amax / (6.0 * 448.0);
                let nb = cols / 16;
                let mut scales = vec![0f32; rows * nb];
                for r in 0..rows {
                    for b in 0..nb {
                        let m = w[r][b * 16..b * 16 + 16]
                            .iter()
                            .fold(0f32, |a, x| a.max(x.abs()));
                        scales[r * nb + b] = m / 6.0 / global;
                    }
                }
                let s8 =
                    Tensor::from_vec(scales, (rows, nb), &Device::Cpu)?.to_dtype(DType::F8E4M3)?;
                let sf: Vec<f32> = s8.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                let mut packed = vec![0u8; rows * cols / 2];
                for r in 0..rows {
                    for c in 0..cols {
                        let s = sf[r * nb + c / 16] * global;
                        let x = if s > 0.0 { w[r][c] / s } else { 0.0 };
                        let code = (0..16u8)
                            .min_by(|&a, &b| {
                                (crate::weights::E2M1[a as usize] - x)
                                    .abs()
                                    .total_cmp(&(crate::weights::E2M1[b as usize] - x).abs())
                            })
                            .unwrap();
                        packed[(r * cols + c) / 2] |= if c % 2 == 0 { code } else { code << 4 };
                    }
                }
                let packed = Tensor::from_vec(packed, (rows, cols / 2), &Device::Cpu)?;
                let q = crate::weights::Nvfp4 {
                    packed: packed.clone(),
                    scale: s8.clone(),
                    global,
                };
                let d = crate::weights::dequant_nvfp4(&q, DType::F32)?;
                if quant == TinyQuant::Nvfp4ModelOpt {
                    qmap.insert(name.clone(), packed);
                    qmap.insert(
                        format!("{base}.weight_scale_2"),
                        Tensor::new(&[global], &Device::Cpu)?,
                    );
                } else {
                    qmap.insert(format!("{base}.weight_packed"), packed);
                    qmap.insert(
                        format!("{base}.weight_global_scale"),
                        Tensor::new(&[1.0 / global], &Device::Cpu)?,
                    );
                }
                qmap.insert(format!("{base}.weight_scale"), s8);
                d.flatten_all()?.to_vec1()?
            }
        };
        dmap.insert(
            name,
            Tensor::from_vec(deq, (rows, cols), &Device::Cpu)?.to_dtype(t.dtype())?,
        );
    }
    candle_core::safetensors::save(&qmap, qdir.join("model.safetensors"))?;
    candle_core::safetensors::save(&dmap, ddir.join("model.safetensors"))?;
    Ok(())
}
