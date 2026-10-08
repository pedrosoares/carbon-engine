// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Model configuration parsed from a Hugging Face `config.json`.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Arch {
    /// Llama, Mistral, SmolLM, ... (no qk-norm, optional attention bias)
    Llama,
    /// Qwen2 / Qwen2.5 (q/k/v bias)
    Qwen2,
    /// Qwen3 dense (per-head q/k RMSNorm)
    Qwen3,
    /// Qwen3-MoE (qk-norm + routed experts)
    Qwen3Moe,
    /// Mixtral (routed experts with w1/w2/w3 naming)
    Mixtral,
    /// Qwen3.5 / Qwen3.6 (dense or MoE): hybrid Gated-DeltaNet + gated full attention,
    /// partial rope, zero-centred RMSNorm, sigmoid-gated shared expert
    Qwen35,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerKind {
    /// Softmax attention over the paged KV cache.
    Full,
    /// Gated DeltaNet linear attention with a per-sequence recurrent state.
    Linear,
}

/// Gated DeltaNet (linear attention) dimensions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearAttnConfig {
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
    pub conv_kernel: usize,
}

impl LinearAttnConfig {
    /// Channels of the fused q/k/v projection that runs through the causal conv.
    pub fn conv_dim(&self) -> usize {
        2 * self.num_k_heads * self.k_head_dim + self.num_v_heads * self.v_head_dim
    }

    /// f32 elements of recurrent + conv state one sequence holds per layer.
    pub fn state_elems(&self) -> usize {
        self.num_v_heads * self.k_head_dim * self.v_head_dim
            + self.conv_dim() * (self.conv_kernel - 1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoeNaming {
    /// `mlp.gate`, `mlp.experts.{e}.{gate,up,down}_proj`
    Qwen,
    /// `block_sparse_moe.gate`, `block_sparse_moe.experts.{e}.{w1,w3,w2}`
    Mixtral,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MoeConfig {
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub decoder_sparse_step: usize,
    pub mlp_only_layers: Vec<usize>,
    pub naming: MoeNaming,
    /// Size of the always-on shared expert (0 = none).
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RopeScaling {
    Linear {
        factor: f64,
    },
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position: usize,
    },
    /// YaRN (NTK-by-parts interpolation + attention temperature).
    Yarn {
        factor: f64,
        original_max_position: usize,
        beta_fast: f64,
        beta_slow: f64,
        /// cos/sin multiplier (`attention_factor`, or derived from `mscale` / `mscale_all_dim`).
        attention_factor: f64,
    },
}

/// YaRN's attention temperature `0.1 * mscale * ln(scale) + 1`.
pub fn yarn_mscale(scale: f64, mscale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

/// Qwen3.8-Flash-Next (`qwen4_exp`) on top of the Qwen3.5 layer stack: hyper-connections
/// (`hc_count` residual streams), PLE n-gram embeddings on some linear layers and QSA sparse
/// attention (an indexer selects `index_budget` tokens in blocks of `index_ratio`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Qwen38Config {
    pub hc_count: usize,
    pub hc_lowrank: usize,
    /// Zero-based decoder layers with a PLE block (all linear-attention layers).
    pub ple_layers: Vec<usize>,
    pub ple_embed_dim: usize,
    pub ple_conv_kernel: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ngram_vocab_size_base: u64,
    pub ngram_vocab_divisor: u64,
    pub split_ngram_parts: usize,
    pub seed: u64,
    /// n-gram windows restart after this token (the first eos id).
    pub ngram_boundary: u32,
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_budget: usize,
    pub index_ratio: usize,
}

impl Qwen38Config {
    pub fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }
    pub fn ngram_head_dim(&self) -> usize {
        self.ple_embed_dim / self.ngram_heads()
    }
    /// The PLE conv is dilated by the n-gram size; it keeps this many past inputs.
    pub fn ple_conv_state_len(&self) -> usize {
        (self.ple_conv_kernel - 1) * self.ngram_size
    }
    pub fn index_topk_blocks(&self) -> usize {
        self.index_budget / self.index_ratio
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub arch: Arch,
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
    pub qk_norm: bool,
    pub hidden_act: String,
    pub moe: Option<MoeConfig>,
    pub eos_token_ids: Vec<u32>,
    pub bos_token_id: Option<u32>,
    /// Per-layer token mixer (all `Full` for classic transformers).
    pub layer_types: Vec<LayerKind>,
    pub linear_attn: Option<LinearAttnConfig>,
    /// Rotary dims per head (`head_dim * partial_rotary_factor`).
    pub rotary_dim: usize,
    /// q_proj also emits a per-head sigmoid gate applied to the attention output.
    pub attn_output_gate: bool,
    /// RMSNorm weights are stored as offsets from 1 (`x * (1 + w)`).
    pub norm_zero_centered: bool,
    /// Tensor-name prefix of the decoder (`model.` or `model.language_model.`); set from the weights.
    pub prefix: String,
    /// The Gated DeltaNet output gate is `sigmoid(z)` instead of `silu(z)`.
    #[serde(default)]
    pub gdn_gate_sigmoid: bool,
    #[serde(default)]
    pub qwen38: Option<Qwen38Config>,
    /// The image encoder (`vision_config` of multimodal checkpoints).
    #[serde(default)]
    pub vision: Option<VisionConfig>,
    /// Interleaved mRoPE: rotary frequencies per position axis (t, h, w), summing to
    /// `rotary_dim / 2` (`rope_parameters.mrope_section`). Text-only positions give all three
    /// axes the same index, so it only matters with image tokens.
    #[serde(default)]
    pub mrope_section: Option<[usize; 3]>,
}

/// The Qwen3.5 / 3.6 / 3.8 vision tower (a ViT with a 2x2 patch merger).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    /// Learned position table (a square grid, resampled to each image's grid).
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    pub image_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

impl VisionConfig {
    fn from_hf(root: &Value) -> Option<Self> {
        let v = root.get("vision_config")?;
        let id = |k: &str| root.get(k).and_then(|x| x.as_u64()).map(|x| x as u32);
        Some(Self {
            depth: get_usize(v, "depth")?,
            hidden_size: get_usize(v, "hidden_size")?,
            num_heads: get_usize(v, "num_heads")?,
            intermediate_size: get_usize(v, "intermediate_size")?,
            hidden_act: v.get("hidden_act").and_then(|x| x.as_str()).unwrap_or("gelu_pytorch_tanh").to_string(),
            in_channels: get_usize(v, "in_channels").unwrap_or(3),
            patch_size: get_usize(v, "patch_size")?,
            temporal_patch_size: get_usize(v, "temporal_patch_size").unwrap_or(2),
            spatial_merge_size: get_usize(v, "spatial_merge_size").unwrap_or(2),
            num_position_embeddings: get_usize(v, "num_position_embeddings")?,
            out_hidden_size: get_usize(v, "out_hidden_size")?,
            image_token_id: id("image_token_id")?,
            vision_start_token_id: id("vision_start_token_id")?,
            vision_end_token_id: id("vision_end_token_id")?,
        })
    }

    /// Patch values per patch: channels x temporal x patch x patch.
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }
}

fn get_usize(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
}

fn get_f64(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(|x| x.as_f64())
}

fn token_ids(v: Option<&Value>) -> Vec<u32> {
    match v {
        Some(Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_u64())
            .map(|x| x as u32)
            .collect(),
        _ => vec![],
    }
}

/// vLLM `--hf-overrides` semantics: nested config sections (`*_config` objects) merge key by
/// key, every other value is replaced whole.
pub fn apply_overrides(config: &mut Value, overrides: &Value) -> Result<()> {
    let Some(o) = overrides.as_object() else {
        bail!("--hf-overrides must be a JSON object");
    };
    let Some(c) = config.as_object_mut() else {
        bail!("config.json is not an object");
    };
    for (k, v) in o {
        match c.get_mut(k) {
            Some(section) if k.ends_with("_config") && section.is_object() && v.is_object() => {
                apply_overrides(section, v)?
            }
            _ => {
                c.insert(k.clone(), v.clone());
            }
        }
    }
    Ok(())
}

/// Refuse weight formats the loader cannot decode, so they fail at start-up instead of
/// producing shape errors mid-request. Supported: block / per-tensor FP8 and NVFP4
/// (ModelOpt and compressed-tensors).
fn check_quantization(config: &Value, dir: &Path) -> Result<()> {
    let modelopt: Option<Value> = std::fs::read_to_string(dir.join("hf_quant_config.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok());
    let q = config.get("quantization_config").or_else(|| {
        config
            .get("text_config")
            .and_then(|t| t.get("quantization_config"))
    });
    let method = q
        .and_then(|q| q.get("quant_method"))
        .and_then(|m| m.as_str())
        .unwrap_or("");
    let algo = q
        .and_then(|q| q.get("quant_algo"))
        .or_else(|| {
            modelopt
                .as_ref()
                .and_then(|m| m.pointer("/quantization/quant_algo"))
        })
        .and_then(|a| a.as_str())
        .unwrap_or("")
        .to_ascii_uppercase();
    let format = q
        .and_then(|q| q.get("format"))
        .and_then(|f| f.as_str())
        .unwrap_or("");
    let ok = match method {
        "" if modelopt.is_none() => true,
        "fp8" => true,
        "modelopt" | "" => matches!(algo.as_str(), "NVFP4" | "W4A16_NVFP4" | "FP8" | ""),
        "compressed-tensors" => matches!(
            format,
            "nvfp4-pack-quantized" | "float-quantized" | "naive-quantized" | ""
        ),
        _ => false,
    };
    if !ok {
        let what = [method, algo.as_str(), format]
            .iter()
            .filter(|x| !x.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join("/");
        bail!("unsupported weight quantization '{what}' (supported: bf16/f16/f32, FP8 block or per-tensor, NVFP4)");
    }
    Ok(())
}

fn qwen38_config(
    v: &Value,
    root: &Value,
    model_type: &str,
    num_layers: usize,
) -> Result<Option<Qwen38Config>> {
    if !model_type.starts_with("qwen4_exp") {
        return Ok(None);
    }
    let req = |k: &str| get_usize(v, k).with_context(|| format!("qwen4_exp config needs '{k}'"));
    let hidden = req("hidden_size")?;
    let ple_layers: Vec<usize> = v
        .get("ple_layer_ids")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
        .unwrap_or_default();
    if let Some(bad) = ple_layers.iter().find(|&&l| l == 0 || l > num_layers) {
        bail!("ple_layer_ids are one-indexed in [1, {num_layers}], got {bad}");
    }
    let mut ple_layers: Vec<usize> = ple_layers.into_iter().map(|l| l - 1).collect();
    ple_layers.sort_unstable();
    ple_layers.dedup();
    let eos = token_ids(v.get("eos_token_id").or_else(|| root.get("eos_token_id")));
    let q = Qwen38Config {
        hc_count: req("hc_count")?,
        hc_lowrank: req("hc_lowrank")?,
        ple_embed_dim: get_usize(v, "ple_embed_dim").unwrap_or(hidden),
        ple_conv_kernel: get_usize(v, "ple_conv_kernel_size").unwrap_or(4),
        ngram_size: get_usize(v, "ngram_size").unwrap_or(3),
        heads_per_ngram: get_usize(v, "heads_per_ngram").unwrap_or(8),
        ngram_vocab_size_base: v
            .get("ngram_vocab_size_base")
            .and_then(|x| x.as_u64())
            .unwrap_or(20_000_000),
        ngram_vocab_divisor: v
            .get("make_ngram_vocab_size_divisible_by")
            .and_then(|x| x.as_u64())
            .unwrap_or(128),
        split_ngram_parts: get_usize(v, "split_ngram_parts").unwrap_or(1).max(1),
        seed: v.get("seed").and_then(|x| x.as_u64()).unwrap_or(1234),
        ngram_boundary: match (eos.first(), ple_layers.is_empty()) {
            (Some(&e), _) => e,
            (None, true) => 0,
            (None, false) => bail!("qwen4_exp with PLE layers needs eos_token_id"),
        },
        index_heads: req("indexer_n_heads")?,
        index_head_dim: req("indexer_head_dim")?,
        index_budget: req("indexer_budget")?,
        index_ratio: req("indexer_compress_ratio")?,
        ple_layers,
    };
    if get_usize(v, "indexer_kv_heads").unwrap_or(1) != 1 {
        bail!("qwen4_exp QSA needs indexer_kv_heads = 1");
    }
    if q.hc_count < 2 || q.index_ratio == 0 || !q.index_budget.is_multiple_of(q.index_ratio) {
        bail!("bad qwen4_exp config (hc_count > 1, indexer_budget divisible by indexer_compress_ratio)");
    }
    if !q.ple_layers.is_empty() && (q.ngram_size < 2 || !q.ple_embed_dim.is_multiple_of(q.ngram_heads())) {
        bail!("ple_embed_dim must split evenly over the n-gram heads");
    }
    Ok(Some(q))
}

impl ModelConfig {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::from_dir_with(dir, None)
    }

    /// `from_dir` with `--hf-overrides` applied to `config.json` first.
    pub fn from_dir_with(dir: &Path, overrides: Option<&Value>) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let mut v: Value = serde_json::from_str(&text)?;
        if let Some(o) = overrides {
            apply_overrides(&mut v, o)?;
        }
        check_quantization(&v, dir)?;
        let mut cfg = Self::from_hf(&v)?;
        // generation_config.json usually carries the full EOS list (e.g. <|im_end|> + <|endoftext|>)
        if let Ok(text) = std::fs::read_to_string(dir.join("generation_config.json")) {
            if let Ok(g) = serde_json::from_str::<Value>(&text) {
                for id in token_ids(g.get("eos_token_id")) {
                    if !cfg.eos_token_ids.contains(&id) {
                        cfg.eos_token_ids.push(id);
                    }
                }
            }
        }
        Ok(cfg)
    }

    pub fn from_hf(root: &Value) -> Result<Self> {
        // multimodal wrappers keep the LM config under text_config
        let v = root
            .get("text_config")
            .filter(|t| t.get("hidden_size").is_some())
            .unwrap_or(root);
        let model_type = v
            .get("model_type")
            .and_then(|x| x.as_str())
            .unwrap_or("llama")
            .to_string();
        let arch = match model_type.as_str() {
            "llama" | "mistral" | "smollm" | "smollm3" | "granite" | "internlm2" | "yi" => Arch::Llama,
            "qwen2" => Arch::Qwen2,
            "qwen3" => Arch::Qwen3,
            "qwen3_moe" => Arch::Qwen3Moe,
            "mixtral" => Arch::Mixtral,
            "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_5" | "qwen3_5_text" => Arch::Qwen35,
            // Qwen3.8-Flash-Next: the Qwen3.5 stack plus `qwen38` (parsed below)
            "qwen4_exp" | "qwen4_exp_text" => Arch::Qwen35,
            other => bail!("unsupported model_type '{other}' (supported: llama, mistral, qwen2, qwen3, qwen3_moe, mixtral, qwen3_5, qwen3_5_moe, qwen4_exp)"),
        };
        let hidden_size = get_usize(v, "hidden_size").context("hidden_size")?;
        let num_heads = get_usize(v, "num_attention_heads").context("num_attention_heads")?;
        let num_kv_heads = get_usize(v, "num_key_value_heads").unwrap_or(num_heads);
        let head_dim = get_usize(v, "head_dim").unwrap_or(hidden_size / num_heads);
        let rope_params = v
            .get("rope_scaling")
            .or_else(|| v.get("rope_parameters"))
            .filter(|x| !x.is_null());
        let rope_theta = get_f64(v, "rope_theta")
            .or_else(|| rope_params.and_then(|r| get_f64(r, "rope_theta")))
            .unwrap_or(10000.0);
        let partial_rotary = get_f64(v, "partial_rotary_factor")
            .or_else(|| rope_params.and_then(|r| get_f64(r, "partial_rotary_factor")))
            .unwrap_or(if arch == Arch::Qwen35 { 0.25 } else { 1.0 });
        let rotary_dim = ((head_dim as f64 * partial_rotary) as usize) & !1;
        let rope_scaling = match rope_params {
            Some(r) => {
                let kind = r
                    .get("rope_type")
                    .or_else(|| r.get("type"))
                    .and_then(|x| x.as_str())
                    .unwrap_or("default");
                let factor = get_f64(r, "factor").unwrap_or(1.0);
                match kind {
                    "linear" => Some(RopeScaling::Linear { factor }),
                    "llama3" => Some(RopeScaling::Llama3 {
                        factor,
                        low_freq_factor: get_f64(r, "low_freq_factor").unwrap_or(1.0),
                        high_freq_factor: get_f64(r, "high_freq_factor").unwrap_or(4.0),
                        original_max_position: get_usize(r, "original_max_position_embeddings")
                            .unwrap_or(8192),
                    }),
                    "yarn" => {
                        let ms = get_f64(r, "mscale");
                        let ms_all = get_f64(r, "mscale_all_dim");
                        let attention_factor = get_f64(r, "attention_factor").unwrap_or(
                            match (ms, ms_all) {
                                (Some(a), Some(b)) => yarn_mscale(factor, a) / yarn_mscale(factor, b),
                                _ => yarn_mscale(factor, 1.0),
                            },
                        );
                        Some(RopeScaling::Yarn {
                            factor,
                            original_max_position: get_usize(r, "original_max_position_embeddings")
                                .or_else(|| get_usize(v, "max_position_embeddings"))
                                .unwrap_or(4096),
                            beta_fast: get_f64(r, "beta_fast").unwrap_or(32.0),
                            beta_slow: get_f64(r, "beta_slow").unwrap_or(1.0),
                            attention_factor,
                        })
                    }
                    "default" => None,
                    other => {
                        tracing::warn!(
                            "rope scaling type '{other}' is not implemented, using unscaled rope"
                        );
                        None
                    }
                }
            }
            None => None,
        };
        let moe = match arch {
            Arch::Qwen3Moe => Some(MoeConfig {
                num_experts: get_usize(v, "num_experts").context("num_experts")?,
                top_k: get_usize(v, "num_experts_per_tok").context("num_experts_per_tok")?,
                intermediate_size: get_usize(v, "moe_intermediate_size")
                    .context("moe_intermediate_size")?,
                norm_topk_prob: v
                    .get("norm_topk_prob")
                    .and_then(|x| x.as_bool())
                    .unwrap_or(false),
                decoder_sparse_step: get_usize(v, "decoder_sparse_step").unwrap_or(1).max(1),
                mlp_only_layers: v
                    .get("mlp_only_layers")
                    .and_then(|x| x.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_u64())
                            .map(|x| x as usize)
                            .collect()
                    })
                    .unwrap_or_default(),
                naming: MoeNaming::Qwen,
                shared_expert_intermediate_size: 0,
            }),
            Arch::Qwen35 if get_usize(v, "num_experts").unwrap_or(0) > 0 => Some(MoeConfig {
                num_experts: get_usize(v, "num_experts").unwrap(),
                top_k: get_usize(v, "num_experts_per_tok").context("num_experts_per_tok")?,
                intermediate_size: get_usize(v, "moe_intermediate_size")
                    .context("moe_intermediate_size")?,
                norm_topk_prob: v
                    .get("norm_topk_prob")
                    .and_then(|x| x.as_bool())
                    .unwrap_or(true),
                decoder_sparse_step: 1,
                mlp_only_layers: vec![],
                naming: MoeNaming::Qwen,
                shared_expert_intermediate_size: get_usize(v, "shared_expert_intermediate_size")
                    .unwrap_or(0),
            }),
            Arch::Mixtral => Some(MoeConfig {
                num_experts: get_usize(v, "num_local_experts").context("num_local_experts")?,
                top_k: get_usize(v, "num_experts_per_tok").unwrap_or(2),
                intermediate_size: get_usize(v, "intermediate_size")
                    .context("intermediate_size")?,
                norm_topk_prob: true,
                decoder_sparse_step: 1,
                mlp_only_layers: vec![],
                naming: MoeNaming::Mixtral,
                shared_expert_intermediate_size: 0,
            }),
            _ => None,
        };
        let num_layers = get_usize(v, "num_hidden_layers").context("num_hidden_layers")?;
        let layer_types = match v.get("layer_types").and_then(|x| x.as_array()) {
            Some(list) => list
                .iter()
                .map(|t| match t.as_str() {
                    Some("linear_attention") => Ok(LayerKind::Linear),
                    // qwen4_exp: full attention restricted by the QSA indexer
                    Some("full_attention") | Some("attention") | Some("indexed_attention")
                    | Some("qwen_sparse_attention") => Ok(LayerKind::Full),
                    other => bail!("unsupported layer type {other:?}"),
                })
                .collect::<Result<Vec<_>>>()?,
            None if arch == Arch::Qwen35 => {
                let every = get_usize(v, "full_attention_interval").unwrap_or(4).max(1);
                (0..num_layers)
                    .map(|i| {
                        if (i + 1) % every == 0 {
                            LayerKind::Full
                        } else {
                            LayerKind::Linear
                        }
                    })
                    .collect()
            }
            None => vec![LayerKind::Full; num_layers],
        };
        if layer_types.len() != num_layers {
            bail!(
                "layer_types has {} entries for {num_layers} layers",
                layer_types.len()
            );
        }
        let linear_attn = if layer_types.contains(&LayerKind::Linear) {
            Some(LinearAttnConfig {
                num_k_heads: get_usize(v, "linear_num_key_heads")
                    .context("linear_num_key_heads")?,
                num_v_heads: get_usize(v, "linear_num_value_heads")
                    .context("linear_num_value_heads")?,
                k_head_dim: get_usize(v, "linear_key_head_dim").context("linear_key_head_dim")?,
                v_head_dim: get_usize(v, "linear_value_head_dim")
                    .context("linear_value_head_dim")?,
                conv_kernel: get_usize(v, "linear_conv_kernel_dim").unwrap_or(4),
            })
        } else {
            None
        };
        let qwen38 = qwen38_config(v, root, &model_type, num_layers)?;
        let cfg = Self {
            layer_types,
            linear_attn,
            rotary_dim,
            attn_output_gate: v
                .get("attn_output_gate")
                .and_then(|x| x.as_bool())
                .unwrap_or(arch == Arch::Qwen35),
            norm_zero_centered: arch == Arch::Qwen35,
            prefix: "model.".into(),
            arch,
            model_type,
            hidden_size,
            intermediate_size: get_usize(v, "intermediate_size").unwrap_or(4 * hidden_size),
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size: get_usize(v, "vocab_size").context("vocab_size")?,
            rms_norm_eps: get_f64(v, "rms_norm_eps").unwrap_or(1e-6),
            rope_theta,
            rope_scaling,
            max_position_embeddings: get_usize(v, "max_position_embeddings").unwrap_or(4096),
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .or_else(|| root.get("tie_word_embeddings"))
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            attention_bias: v
                .get("attention_bias")
                .and_then(|x| x.as_bool())
                .unwrap_or(arch == Arch::Qwen2),
            qk_norm: matches!(arch, Arch::Qwen3 | Arch::Qwen3Moe | Arch::Qwen35),
            hidden_act: v
                .get("hidden_act")
                .and_then(|x| x.as_str())
                .unwrap_or("silu")
                .to_string(),
            moe,
            eos_token_ids: token_ids(v.get("eos_token_id").or_else(|| root.get("eos_token_id"))),
            bos_token_id: token_ids(v.get("bos_token_id")).first().copied(),
            gdn_gate_sigmoid: v
                .get("output_gate_type")
                .and_then(|x| x.as_str())
                .is_some_and(|g| g == "sigmoid"),
            qwen38,
            vision: VisionConfig::from_hf(root),
            mrope_section: rope_params
                .and_then(|r| r.get("mrope_section"))
                .and_then(|m| m.as_array())
                .and_then(|a| {
                    let s: Vec<usize> = a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect();
                    (s.len() == 3).then(|| [s[0], s[1], s[2]])
                }),
        };
        if let Some(q) = &cfg.qwen38 {
            if let Some(&l) = q.ple_layers.iter().find(|&&l| cfg.layer_types[l] != LayerKind::Linear) {
                bail!("qwen4_exp PLE must sit on a linear_attention layer, layer {l} is not one");
            }
            if cfg.moe.is_none() {
                bail!("qwen4_exp needs num_experts > 0");
            }
        }
        Ok(cfg)
    }

    pub fn layer_is_moe(&self, layer: usize) -> bool {
        match &self.moe {
            Some(m) => {
                !m.mlp_only_layers.contains(&layer)
                    && (layer + 1).is_multiple_of(m.decoder_sparse_step)
            }
            None => false,
        }
    }

    /// Bytes of KV cache one token occupies in `num_layers` layers at `dtype_size` bytes/elem.
    pub fn kv_bytes_per_token(&self, num_layers: usize, dtype_size: usize) -> usize {
        2 * num_layers * self.num_kv_heads * self.head_dim * dtype_size
    }

    pub fn is_hybrid(&self) -> bool {
        self.layer_types.contains(&LayerKind::Linear)
    }

    /// Name of decoder layer `l`'s tensors, e.g. `model.layers.3`.
    pub fn layer_prefix(&self, l: usize) -> String {
        format!("{}layers.{l}", self.prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_overrides_merge_sections() {
        let mut v = serde_json::json!({
            "model_type": "qwen3", "max_position_embeddings": 4096,
            "rope_scaling": {"rope_type": "linear", "factor": 2.0},
            "text_config": {"hidden_size": 8, "vocab_size": 10},
        });
        apply_overrides(&mut v, &serde_json::json!({
            "max_position_embeddings": 131072,
            "rope_scaling": {"rope_type": "yarn"},
            "text_config": {"vocab_size": 12},
        }))
        .unwrap();
        assert_eq!(v["max_position_embeddings"], 131072);
        // plain dicts are replaced whole, config sections merge
        assert_eq!(v["rope_scaling"], serde_json::json!({"rope_type": "yarn"}));
        assert_eq!(v["text_config"], serde_json::json!({"hidden_size": 8, "vocab_size": 12}));
        assert!(apply_overrides(&mut v, &serde_json::json!([1])).is_err());
    }

    #[test]
    fn parses_qwen3_moe() {
        let v = serde_json::json!({
            "model_type": "qwen3_moe", "hidden_size": 64, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "num_hidden_layers": 3, "vocab_size": 100,
            "intermediate_size": 128, "num_experts": 8, "num_experts_per_tok": 2,
            "moe_intermediate_size": 32, "norm_topk_prob": true, "mlp_only_layers": [0],
            "eos_token_id": 5, "rope_theta": 1000000.0
        });
        let c = ModelConfig::from_hf(&v).unwrap();
        assert_eq!(c.arch, Arch::Qwen3Moe);
        assert!(c.qk_norm);
        assert!(!c.layer_is_moe(0));
        assert!(c.layer_is_moe(1));
        assert_eq!(c.eos_token_ids, vec![5]);
        assert_eq!(c.kv_bytes_per_token(3, 2), 2 * 3 * 2 * 16 * 2);
    }

    #[test]
    fn parses_qwen35_moe_wrapper() {
        let v = serde_json::json!({
            "model_type": "qwen3_5_moe", "tie_word_embeddings": false,
            "text_config": {
                "model_type": "qwen3_5_moe_text", "hidden_size": 2048, "num_attention_heads": 16,
                "num_key_value_heads": 2, "head_dim": 256, "num_hidden_layers": 8, "vocab_size": 248320,
                "num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512, "linear_num_key_heads": 16,
                "linear_num_value_heads": 32, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4, "full_attention_interval": 4, "eos_token_id": 248044,
                "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0,
                                    "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]}
            }
        });
        let c = ModelConfig::from_hf(&v).unwrap();
        assert_eq!(c.arch, Arch::Qwen35);
        assert_eq!(c.rotary_dim, 64);
        assert!(c.attn_output_gate && c.norm_zero_centered && c.qk_norm && c.is_hybrid());
        assert_eq!(c.layer_types[3], LayerKind::Full);
        assert_eq!(c.layer_types[0], LayerKind::Linear);
        let m = c.moe.as_ref().unwrap();
        assert_eq!(
            (m.num_experts, m.shared_expert_intermediate_size),
            (256, 512)
        );
        assert!(c.layer_is_moe(0));
        assert_eq!(c.linear_attn.as_ref().unwrap().conv_dim(), 8192);
        assert_eq!(c.eos_token_ids, vec![248044]);
    }

    #[test]
    fn rejects_unknown_quantization() {
        let d = std::env::temp_dir();
        let gptq = serde_json::json!({"quantization_config": {"quant_method": "gptq", "bits": 4}});
        assert!(check_quantization(&gptq, &d).is_err());
        let nv = serde_json::json!({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}});
        assert!(check_quantization(&nv, &d).is_ok());
        let fp8 = serde_json::json!({"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}});
        assert!(check_quantization(&fp8, &d).is_ok());
        assert!(check_quantization(&serde_json::json!({}), &d).is_ok());
    }

    #[test]
    fn parses_llama3_rope() {
        let v = serde_json::json!({
            "model_type": "llama", "hidden_size": 64, "num_attention_heads": 4,
            "num_hidden_layers": 2, "vocab_size": 100, "eos_token_id": [1, 2],
            "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                             "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
        });
        let c = ModelConfig::from_hf(&v).unwrap();
        assert!(matches!(c.rope_scaling, Some(RopeScaling::Llama3 { .. })));
        assert_eq!(c.num_kv_heads, 4);
        assert_eq!(c.head_dim, 16);
        assert_eq!(c.eos_token_ids, vec![1, 2]);
    }
}
