// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! GGUF checkpoints (llama.cpp's format): the file's metadata becomes an HF-style config and
//! tokenizer, its tensors are served under the HF names the model loaders ask for.
//!
//! Supported: architecture `qwen35` (dense Qwen3.5 / 3.6 / 3.8), tensors stored F32 / F16 /
//! BF16 / Q4_K / Q6_K. K-quant projections stay packed (`WeightSource::load_kquant`) and are
//! decoded per GEMM by the kernels; `load_raw` decodes any tensor to f32 on the CPU.
//!
//! The converter (`conversion/qwen.py` in llama.cpp) rewrites a few tensors; `load_raw` undoes
//! what it can so the loaders see HF values:
//!
//! - RMSNorm weights are stored as `1 + w` (all but the Gated DeltaNet norm): 1 is subtracted.
//! - `A_log` is stored as `-exp(A_log)`: the log of its negation is returned.
//! - `conv1d.weight` is stored `[C, K]` instead of `[C, 1, K]` (the loader reshapes either).
//! - Gated DeltaNet value heads are stored tiled (`k0 k1 .. k0 k1 ..`) instead of grouped by key
//!   head. That permutes columns of a quantized projection, so it is not undone: the config
//!   says `linear_v_tiled` and the layer maps value head `h` to key head `h % num_k_heads`.

use crate::weights::WeightSource;
use anyhow::{anyhow, bail, Context, Result};
use candle_core::{Device, Tensor};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Weights per K-quant super-block.
pub const QK_K: usize = 256;
/// Bytes of a Q4_K super-block: f16 `d`, f16 `dmin`, 12 bytes of 6-bit scales / mins, 128 bytes
/// of 4-bit codes.
pub const Q4K_BLOCK: usize = 144;
/// Bytes of a Q6_K super-block in the file: 128 low-nibble bytes, 64 bytes of high bit pairs,
/// 16 i8 scales, f16 `d`.
const Q6K_FILE_BLOCK: usize = 210;
/// Bytes of a Q6_K super-block in memory: the file's block padded to a multiple of 16, so the
/// kernels read aligned words.
pub const Q6K_BLOCK: usize = 224;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Bf16,
    Q4K,
    Q6K,
}

impl GgmlType {
    fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::F32,
            1 => Self::F16,
            30 => Self::Bf16,
            12 => Self::Q4K,
            14 => Self::Q6K,
            _ => return None,
        })
    }

    /// Bytes of `n` elements (`n` a multiple of 256 for the K-quants).
    fn bytes(self, n: usize) -> usize {
        match self {
            Self::F32 => 4 * n,
            Self::F16 | Self::Bf16 => 2 * n,
            Self::Q4K => n / QK_K * Q4K_BLOCK,
            Self::Q6K => n / QK_K * Q6K_FILE_BLOCK,
        }
    }
}

/// The packed format of a K-quant projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KKind {
    Q4K,
    Q6K,
}

impl KKind {
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q4K => Q4K_BLOCK,
            Self::Q6K => Q6K_BLOCK,
        }
    }
}

/// A packed K-quant weight `[rows, cols]`: `data` is U8 `[rows, cols / 256 * block_bytes]`
/// (Q4_K blocks as in the file, Q6_K blocks padded to 224 bytes).
#[derive(Clone, Debug)]
pub struct KQuant {
    pub kind: KKind,
    pub data: Tensor,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Debug)]
pub enum Meta {
    U(u64),
    I(i64),
    F(f64),
    B(bool),
    S(String),
    A(Vec<Meta>),
}

impl Meta {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U(x) => Some(*x),
            Self::I(x) if *x >= 0 => Some(*x as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F(x) => Some(*x),
            Self::U(x) => Some(*x as f64),
            Self::I(x) => Some(*x as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::S(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Meta]> {
        match self {
            Self::A(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    /// Dimensions in torch order (GGUF stores them reversed).
    pub shape: Vec<usize>,
    pub ty: GgmlType,
    /// Absolute offset of the data in the file.
    pub offset: usize,
}

impl TensorInfo {
    pub fn elems(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A parsed GGUF file, memory mapped.
pub struct Gguf {
    pub path: PathBuf,
    map: Arc<memmap2::Mmap>,
    pub meta: HashMap<String, Meta>,
    pub tensors: HashMap<String, TensorInfo>,
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.p.checked_add(n).filter(|&e| e <= self.b.len()).context("truncated GGUF header")?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Meta> {
        Ok(match ty {
            0 => Meta::U(self.take(1)?[0] as u64),
            1 => Meta::I(self.take(1)?[0] as i8 as i64),
            2 => Meta::U(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => Meta::I(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Meta::U(self.u32()? as u64),
            5 => Meta::I(self.u32()? as i32 as i64),
            6 => Meta::F(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => Meta::B(self.take(1)?[0] != 0),
            8 => Meta::S(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                if et == 9 {
                    bail!("nested GGUF arrays are not supported");
                }
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                Meta::A(v)
            }
            10 => Meta::U(self.u64()?),
            11 => Meta::I(self.u64()? as i64),
            12 => Meta::F(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            other => bail!("unknown GGUF metadata type {other}"),
        })
    }
}

/// The GGUF file `model` names: the file itself, or the one model file of a directory that has
/// no `config.json` (`mmproj-*` projector files are skipped).
pub fn find(model: &Path) -> Option<PathBuf> {
    let is_gguf = |p: &Path| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
    if model.is_file() {
        return is_gguf(model).then(|| model.to_path_buf());
    }
    if !model.is_dir() || model.join("config.json").exists() {
        return None;
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(model)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| is_gguf(p))
        .filter(|p| !p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("mmproj")))
        .collect();
    files.sort();
    (files.len() == 1).then(|| files.remove(0))
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: mapped read-only; the checkpoint is not modified while serving
        let map = unsafe { memmap2::Mmap::map(&file)? };
        let mut r = Reader { b: &map, p: 0 };
        if r.take(4)? != b"GGUF" {
            bail!("{} is not a GGUF file", path.display());
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            bail!("GGUF version {version} is not supported");
        }
        let (nt, nkv) = (r.u64()? as usize, r.u64()? as usize);
        let mut meta = HashMap::with_capacity(nkv);
        for _ in 0..nkv {
            let k = r.string()?;
            let ty = r.u32()?;
            let v = r.value(ty).with_context(|| format!("metadata key '{k}'"))?;
            meta.insert(k, v);
        }
        let mut infos = Vec::with_capacity(nt);
        for _ in 0..nt {
            let name = r.string()?;
            let nd = r.u32()? as usize;
            let mut shape = Vec::with_capacity(nd);
            for _ in 0..nd {
                shape.push(r.u64()? as usize);
            }
            shape.reverse();
            let ty = r.u32()?;
            let off = r.u64()? as usize;
            infos.push((name, shape, ty, off));
        }
        let align = meta.get("general.alignment").and_then(Meta::as_u64).unwrap_or(32) as usize;
        let base = r.p.div_ceil(align) * align;
        let mut tensors = HashMap::with_capacity(nt);
        for (name, shape, ty, off) in infos {
            let ty = GgmlType::from_id(ty).with_context(|| {
                format!("tensor '{name}' has GGML type {ty}: only F32, F16, BF16, Q4_K and Q6_K are supported")
            })?;
            let info = TensorInfo { shape, ty, offset: base + off };
            if info.offset + ty.bytes(info.elems()) > map.len() {
                bail!("tensor '{name}' lies outside {}", path.display());
            }
            tensors.insert(name, info);
        }
        Ok(Self { path: path.to_path_buf(), map: Arc::new(map), meta, tensors })
    }

    pub fn arch(&self) -> &str {
        self.meta.get("general.architecture").and_then(Meta::as_str).unwrap_or("")
    }

    fn key(&self, k: &str) -> Option<&Meta> {
        self.meta.get(&format!("{}.{k}", self.arch()))
    }

    fn usize_key(&self, k: &str) -> Result<usize> {
        self.key(k)
            .and_then(Meta::as_u64)
            .map(|x| x as usize)
            .with_context(|| format!("GGUF metadata '{}.{k}' is missing", self.arch()))
    }

    pub fn bytes(&self, info: &TensorInfo) -> &[u8] {
        &self.map[info.offset..info.offset + info.ty.bytes(info.elems())]
    }

    /// MTP (NextN) blocks stored after the decoder layers.
    pub fn nextn_layers(&self) -> usize {
        self.key("nextn_predict_layers").and_then(Meta::as_u64).unwrap_or(0) as usize
    }

    /// The model's configuration as an HF `config.json`.
    pub fn hf_config(&self) -> Result<Value> {
        if self.arch() != "qwen35" {
            bail!("GGUF architecture '{}' is not supported (supported: qwen35)", self.arch());
        }
        let hidden = self.usize_key("embedding_length")?;
        let heads = self.usize_key("attention.head_count")?;
        let head_dim = self.usize_key("attention.key_length").unwrap_or(hidden / heads);
        let layers = self.usize_key("block_count")? - self.nextn_layers();
        let vocab = self
            .meta
            .get("tokenizer.ggml.tokens")
            .and_then(Meta::as_array)
            .map(|a| a.len())
            .or_else(|| self.tensors.get("token_embd.weight").map(|t| t.shape[0]))
            .context("GGUF has no vocabulary")?;
        let rot = self.usize_key("rope.dimension_count").unwrap_or(head_dim / 4);
        let state = self.usize_key("ssm.state_size")?;
        let v_heads = self.usize_key("ssm.time_step_rank")?;
        let k_heads = self.usize_key("ssm.group_count")?;
        let mut sections: Vec<u64> = self
            .key("rope.dimension_sections")
            .and_then(Meta::as_array)
            .map(|a| a.iter().filter_map(Meta::as_u64).collect())
            .unwrap_or_else(|| vec![11, 11, 10]);
        sections.truncate(3);
        let tok = |k: &str| self.meta.get(&format!("tokenizer.ggml.{k}")).and_then(Meta::as_u64);
        let mut eos: Vec<u64> = tok("eos_token_id").into_iter().collect();
        // ChatML checkpoints also stop on <|endoftext|>
        if let Some(id) = self.token_id("<|endoftext|>") {
            if !eos.contains(&id) {
                eos.push(id);
            }
        }
        Ok(json!({
            "model_type": "qwen3_5_text",
            "hidden_size": hidden,
            "intermediate_size": self.usize_key("feed_forward_length")?,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": self.usize_key("attention.head_count_kv").unwrap_or(heads),
            "head_dim": head_dim,
            "vocab_size": vocab,
            "rms_norm_eps": self.key("attention.layer_norm_rms_epsilon").and_then(Meta::as_f64).unwrap_or(1e-6),
            "max_position_embeddings": self.usize_key("context_length")?,
            "full_attention_interval": self.usize_key("full_attention_interval").unwrap_or(4),
            "rope_parameters": {
                "rope_theta": self.key("rope.freq_base").and_then(Meta::as_f64).unwrap_or(1e7),
                "partial_rotary_factor": rot as f64 / head_dim as f64,
                "mrope_section": sections,
            },
            "linear_num_key_heads": k_heads,
            "linear_num_value_heads": v_heads,
            "linear_key_head_dim": state,
            "linear_value_head_dim": self.usize_key("ssm.inner_size")? / v_heads,
            "linear_conv_kernel_dim": self.usize_key("ssm.conv_kernel")?,
            "linear_v_tiled": k_heads != v_heads,
            "mtp_num_hidden_layers": self.nextn_layers(),
            "tie_word_embeddings": !self.tensors.contains_key("output.weight"),
            "eos_token_id": eos,
            "bos_token_id": tok("bos_token_id"),
        }))
    }

    fn token_id(&self, token: &str) -> Option<u64> {
        self.meta
            .get("tokenizer.ggml.tokens")?
            .as_array()?
            .iter()
            .position(|t| t.as_str() == Some(token))
            .map(|i| i as u64)
    }

    /// Sampling defaults in `generation_config.json` form (`general.sampling.*`).
    pub fn generation_config(&self) -> Value {
        let mut g = serde_json::Map::new();
        for (k, name) in [("temp", "temperature"), ("top_p", "top_p"), ("min_p", "min_p")] {
            if let Some(x) = self.meta.get(&format!("general.sampling.{k}")).and_then(Meta::as_f64) {
                // f32 metadata: 0.95 arrives as 0.949999988
                g.insert(name.into(), json!((x * 1e6).round() / 1e6));
            }
        }
        if let Some(x) = self.meta.get("general.sampling.top_k").and_then(Meta::as_u64) {
            g.insert("top_k".into(), json!(x));
        }
        Value::Object(g)
    }

    pub fn chat_template(&self) -> Option<String> {
        self.meta.get("tokenizer.chat_template").and_then(Meta::as_str).map(String::from)
    }

    /// The BOS / EOS token strings.
    pub fn special_token(&self, which: &str) -> Option<String> {
        let id = self.meta.get(&format!("tokenizer.ggml.{which}_token_id")).and_then(Meta::as_u64)? as usize;
        self.meta.get("tokenizer.ggml.tokens")?.as_array()?.get(id)?.as_str().map(String::from)
    }

    /// The vocabulary as an HF `tokenizer.json` (byte-level BPE with the Qwen pre-tokenizer).
    pub fn tokenizer_json(&self) -> Result<Value> {
        let model = self.meta.get("tokenizer.ggml.model").and_then(Meta::as_str).unwrap_or("");
        let pre = self.meta.get("tokenizer.ggml.pre").and_then(Meta::as_str).unwrap_or("");
        if model != "gpt2" {
            bail!("GGUF tokenizer model '{model}' is not supported (supported: gpt2 byte-level BPE)");
        }
        // llama.cpp's LLAMA_VOCAB_PRE_TYPE_QWEN35 / QWEN2, as in the checkpoints' tokenizer.json
        let regex = match pre {
            "qwen35" => r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            "qwen2" => r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            other => bail!("GGUF pre-tokenizer '{other}' is not supported (supported: qwen35, qwen2)"),
        };
        let tokens = self
            .meta
            .get("tokenizer.ggml.tokens")
            .and_then(Meta::as_array)
            .context("tokenizer.ggml.tokens")?;
        let types = self.meta.get("tokenizer.ggml.token_type").and_then(Meta::as_array);
        let mut vocab = serde_json::Map::with_capacity(tokens.len());
        let mut added = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            let t = t.as_str().context("token is not a string")?;
            // 1 normal, 3 control, 4 user defined, 5 unused
            let ty = types.and_then(|a| a.get(i)).and_then(Meta::as_u64).unwrap_or(1);
            if ty == 1 {
                vocab.insert(t.to_string(), json!(i));
            } else {
                added.push(json!({
                    "id": i, "content": t, "single_word": false, "lstrip": false, "rstrip": false,
                    "normalized": false, "special": ty != 4,
                }));
            }
        }
        let merges: Vec<Value> = self
            .meta
            .get("tokenizer.ggml.merges")
            .and_then(Meta::as_array)
            .context("tokenizer.ggml.merges")?
            .iter()
            .filter_map(|m| m.as_str().map(|s| json!(s)))
            .collect();
        let byte_level = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
        Ok(json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": regex}, "behavior": "Isolated", "invert": false},
                byte_level,
            ]},
            "post_processor": byte_level,
            "decoder": byte_level,
            "model": {
                "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
                "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                "vocab": vocab, "merges": merges,
            },
        }))
    }
}

/// The HF name of GGUF tensor `name` (architecture `qwen35`); blocks from `layers` on are the
/// MTP head (`mtp.*`).
fn hf_name(name: &str, layers: usize) -> Option<String> {
    match name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        _ => {}
    }
    let rest = name.strip_prefix("blk.")?;
    let (idx, tail) = rest.split_once('.')?;
    let idx: usize = idx.parse().ok()?;
    let sub = match tail {
        "attn_norm.weight" => "input_layernorm.weight",
        "post_attention_norm.weight" => "post_attention_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "attn_qkv.weight" => "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight" => "linear_attn.in_proj_z.weight",
        "ssm_alpha.weight" => "linear_attn.in_proj_a.weight",
        "ssm_beta.weight" => "linear_attn.in_proj_b.weight",
        "ssm_conv1d.weight" => "linear_attn.conv1d.weight",
        "ssm_dt.bias" => "linear_attn.dt_bias",
        "ssm_a" => "linear_attn.A_log",
        "ssm_norm.weight" => "linear_attn.norm.weight",
        "ssm_out.weight" => "linear_attn.out_proj.weight",
        "nextn.eh_proj.weight" => return Some("mtp.fc.weight".into()),
        "nextn.enorm.weight" => return Some("mtp.pre_fc_norm_embedding.weight".into()),
        "nextn.hnorm.weight" => return Some("mtp.pre_fc_norm_hidden.weight".into()),
        "nextn.shared_head_norm.weight" => return Some("mtp.norm.weight".into()),
        _ => return None,
    };
    Some(if idx >= layers {
        format!("mtp.layers.{}.{sub}", idx - layers)
    } else {
        format!("model.layers.{idx}.{sub}")
    })
}

#[inline]
fn f16_at(b: &[u8], i: usize) -> f32 {
    half::f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// Scale and min of sub-block `j` (0..8) from a Q4_K block's 12 packed bytes.
#[inline]
pub fn q4k_scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Decode Q4_K blocks (`src.len() / 144` of them) into `dst`.
pub fn dequant_q4k(src: &[u8], dst: &mut [f32]) {
    for (blk, out) in src.chunks_exact(Q4K_BLOCK).zip(dst.chunks_exact_mut(QK_K)) {
        let (d, dmin) = (f16_at(blk, 0), f16_at(blk, 2));
        let (scales, qs) = (&blk[4..16], &blk[16..]);
        for pair in 0..4 {
            let (s1, m1) = q4k_scale_min(2 * pair, scales);
            let (s2, m2) = q4k_scale_min(2 * pair + 1, scales);
            let (d1, mm1) = (d * s1 as f32, dmin * m1 as f32);
            let (d2, mm2) = (d * s2 as f32, dmin * m2 as f32);
            let q = &qs[32 * pair..32 * pair + 32];
            let o = &mut out[64 * pair..64 * pair + 64];
            for l in 0..32 {
                o[l] = d1 * (q[l] & 0xF) as f32 - mm1;
                o[l + 32] = d2 * (q[l] >> 4) as f32 - mm2;
            }
        }
    }
}

/// Decode Q6_K blocks of `stride` bytes each (210 in a file, 224 in memory) into `dst`.
pub fn dequant_q6k(src: &[u8], dst: &mut [f32], stride: usize) {
    for (blk, out) in src.chunks_exact(stride).zip(dst.chunks_exact_mut(QK_K)) {
        let d = f16_at(blk, 208);
        for half in 0..2 {
            let ql = &blk[64 * half..64 * half + 64];
            let qh = &blk[128 + 32 * half..128 + 32 * half + 32];
            let sc = &blk[192 + 8 * half..192 + 8 * half + 8];
            let o = &mut out[128 * half..128 * half + 128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                o[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                o[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                o[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                o[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
}

/// Decode packed K-quant rows (the in-memory layout) to f32 `[rows * cols]`.
pub fn dequant_kquant(kind: KKind, data: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    dequant_rows(kind, data, rows, cols, kind.block_bytes())
}

fn dequant_rows(kind: KKind, data: &[u8], rows: usize, cols: usize, stride: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    let row_bytes = cols / QK_K * stride;
    out.par_chunks_mut(cols).zip(data.par_chunks(row_bytes)).for_each(|(o, s)| match kind {
        KKind::Q4K => dequant_q4k(s, o),
        KKind::Q6K => dequant_q6k(s, o, stride),
    });
    out
}

/// Q6_K blocks as stored in a file, padded to the in-memory block size.
fn pad_q6k(file: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; file.len() / Q6K_FILE_BLOCK * Q6K_BLOCK];
    out.par_chunks_mut(Q6K_BLOCK)
        .zip(file.par_chunks(Q6K_FILE_BLOCK))
        .for_each(|(o, s)| o[..Q6K_FILE_BLOCK].copy_from_slice(s));
    out
}

/// A GGUF file's tensors under their HF names.
pub struct GgufSource {
    gguf: Gguf,
    /// HF name -> GGUF name
    names: HashMap<String, String>,
}

impl GgufSource {
    pub fn open(path: &Path) -> Result<Self> {
        let gguf = Gguf::open(path)?;
        if gguf.arch() != "qwen35" {
            bail!("GGUF architecture '{}' is not supported (supported: qwen35)", gguf.arch());
        }
        let layers = gguf.usize_key("block_count")? - gguf.nextn_layers();
        let mut names = HashMap::with_capacity(gguf.tensors.len());
        for name in gguf.tensors.keys() {
            match hf_name(name, layers) {
                Some(hf) => {
                    names.insert(hf, name.clone());
                }
                None => tracing::warn!("GGUF tensor '{name}' has no known role, ignored"),
            }
        }
        Ok(Self { gguf, names })
    }

    fn info(&self, name: &str) -> Result<&TensorInfo> {
        let g = self.names.get(name).ok_or_else(|| anyhow!("weight '{name}' is not in the GGUF file"))?;
        Ok(&self.gguf.tensors[g])
    }
}

impl WeightSource for GgufSource {
    fn load_raw(&self, name: &str) -> Result<Tensor> {
        let info = self.info(name)?;
        let bytes = self.gguf.bytes(info);
        let n = info.elems();
        let mut data: Vec<f32> = match info.ty {
            GgmlType::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            GgmlType::F16 => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            GgmlType::Bf16 => bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            GgmlType::Q4K | GgmlType::Q6K => {
                let cols = *info.shape.last().context("scalar K-quant tensor")?;
                if cols % QK_K != 0 {
                    bail!("K-quant tensor '{name}' has {cols} columns, not a multiple of {QK_K}");
                }
                match info.ty {
                    GgmlType::Q4K => dequant_rows(KKind::Q4K, bytes, n / cols, cols, Q4K_BLOCK),
                    _ => dequant_rows(KKind::Q6K, bytes, n / cols, cols, Q6K_FILE_BLOCK),
                }
            }
        };
        // undo the converter's rewrites (see the module docs)
        if name.ends_with("norm.weight") && !name.ends_with("linear_attn.norm.weight") {
            data.iter_mut().for_each(|x| *x -= 1.0);
        } else if name.ends_with(".A_log") {
            data.iter_mut().for_each(|x| *x = (-*x).ln());
        }
        Ok(Tensor::from_vec(data, info.shape.as_slice(), &Device::Cpu)?)
    }

    fn load_kquant(&self, name: &str) -> Result<Option<KQuant>> {
        let Some(g) = self.names.get(name) else {
            return Ok(None);
        };
        let info = &self.gguf.tensors[g];
        let kind = match info.ty {
            GgmlType::Q4K => KKind::Q4K,
            GgmlType::Q6K => KKind::Q6K,
            _ => return Ok(None),
        };
        let &[rows, cols] = info.shape.as_slice() else {
            return Ok(None);
        };
        if cols % QK_K != 0 {
            return Ok(None);
        }
        let row_bytes = cols / QK_K * kind.block_bytes();
        let data = match kind {
            KKind::Q4K => Tensor::from_slice(self.gguf.bytes(info), (rows, row_bytes), &Device::Cpu)?,
            KKind::Q6K => Tensor::from_vec(pad_q6k(self.gguf.bytes(info)), (rows, row_bytes), &Device::Cpu)?,
        };
        Ok(Some(KQuant { kind, data, rows, cols }))
    }

    fn is_kquant(&self, name: &str) -> bool {
        self.names.get(name).is_some_and(|g| {
            let t = &self.gguf.tensors[g];
            matches!(t.ty, GgmlType::Q4K | GgmlType::Q6K) && t.shape.len() == 2 && t.shape[1] % QK_K == 0
        })
    }

    fn contains(&self, name: &str) -> bool {
        self.names.contains_key(name)
    }

    fn names(&self) -> Vec<String> {
        self.names.keys().cloned().collect()
    }
}

/// Encode f32 weights as Q4_K blocks (tests and tiny checkpoints; a plain min / max fit per
/// sub-block, not llama.cpp's error-minimising search).
pub fn quantize_q4k(w: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(w.len() / QK_K * Q4K_BLOCK);
    for blk in w.chunks_exact(QK_K) {
        let mut scales = [0f32; 8];
        let mut mins = [0f32; 8];
        for (j, sub) in blk.chunks_exact(32).enumerate() {
            let lo = sub.iter().cloned().fold(f32::INFINITY, f32::min).min(0.0);
            let hi = sub.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            scales[j] = (hi - lo) / 15.0;
            mins[j] = -lo;
        }
        let d = scales.iter().cloned().fold(0f32, f32::max) / 63.0;
        let dmin = mins.iter().cloned().fold(0f32, f32::max) / 63.0;
        let (d16, dmin16) = (half::f16::from_f32(d), half::f16::from_f32(dmin));
        let (d, dmin) = (d16.to_f32(), dmin16.to_f32());
        let code = |x: f32, unit: f32| if unit > 0.0 { (x / unit).round().clamp(0.0, 63.0) as u8 } else { 0 };
        let sc: Vec<u8> = scales.iter().map(|&s| code(s, d)).collect();
        let mn: Vec<u8> = mins.iter().map(|&m| code(m, dmin)).collect();
        let mut packed = [0u8; 12];
        for j in 0..8 {
            if j < 4 {
                packed[j] = sc[j];
                packed[j + 4] = mn[j];
            } else {
                packed[j + 4] = (sc[j] & 0xF) | ((mn[j] & 0xF) << 4);
                packed[j - 4] |= (sc[j] >> 4) << 6;
                packed[j] |= (mn[j] >> 4) << 6;
            }
        }
        out.extend_from_slice(&d16.to_le_bytes());
        out.extend_from_slice(&dmin16.to_le_bytes());
        out.extend_from_slice(&packed);
        let mut qs = [0u8; 128];
        for pair in 0..4 {
            for l in 0..32 {
                let q = |j: usize, x: f32| {
                    let (s, m) = (d * sc[j] as f32, dmin * mn[j] as f32);
                    if s > 0.0 { ((x + m) / s).round().clamp(0.0, 15.0) as u8 } else { 0 }
                };
                let lo = q(2 * pair, blk[64 * pair + l]);
                let hi = q(2 * pair + 1, blk[64 * pair + 32 + l]);
                qs[32 * pair + l] = lo | (hi << 4);
            }
        }
        out.extend_from_slice(&qs);
    }
    out
}

/// Encode f32 weights as Q6_K blocks (tests and tiny checkpoints).
pub fn quantize_q6k(w: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(w.len() / QK_K * Q6K_BLOCK);
    for blk in w.chunks_exact(QK_K) {
        let mut scales = [0f32; 16];
        for (j, sub) in blk.chunks_exact(16).enumerate() {
            scales[j] = sub.iter().fold(0f32, |a, x| a.max(x.abs())) / 31.0;
        }
        let d16 = half::f16::from_f32(scales.iter().cloned().fold(0f32, f32::max) / 127.0);
        let d = d16.to_f32();
        let sc: Vec<i8> = scales.iter().map(|&s| if d > 0.0 { (s / d).ceil().clamp(0.0, 127.0) as i8 } else { 0 }).collect();
        let mut codes = [0u8; QK_K];
        for (i, &x) in blk.iter().enumerate() {
            let s = d * sc[i / 16] as f32;
            codes[i] = if s > 0.0 { ((x / s).round().clamp(-32.0, 31.0) as i32 + 32) as u8 } else { 32 };
        }
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for half in 0..2 {
            for l in 0..32 {
                let c = &codes[128 * half..];
                let (q1, q2, q3, q4) = (c[l], c[l + 32], c[l + 64], c[l + 96]);
                ql[64 * half + l] = (q1 & 0xF) | ((q3 & 0xF) << 4);
                ql[64 * half + l + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4);
                qh[32 * half + l] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
            }
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend(sc.iter().map(|&s| s as u8));
        out.extend_from_slice(&d16.to_le_bytes());
        out.extend_from_slice(&[0u8; Q6K_BLOCK - Q6K_FILE_BLOCK]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(n: usize) -> Vec<f32> {
        // deterministic, zero-mean, varying scale per sub-block
        (0..n)
            .map(|i| {
                let x = ((i as u64).wrapping_mul(0x9E3779B97F4A7C15) >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                x * (0.02 + 0.01 * ((i / 32) % 7) as f32)
            })
            .collect()
    }

    #[test]
    fn q4k_round_trips() {
        let w = weights(4 * QK_K);
        let q = quantize_q4k(&w);
        assert_eq!(q.len(), 4 * Q4K_BLOCK);
        let mut back = vec![0f32; w.len()];
        dequant_q4k(&q, &mut back);
        let err = w.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err < 4e-3, "max error {err}");
    }

    #[test]
    fn q6k_round_trips() {
        let w = weights(4 * QK_K);
        let q = quantize_q6k(&w);
        assert_eq!(q.len(), 4 * Q6K_BLOCK);
        let mut back = vec![0f32; w.len()];
        dequant_q6k(&q, &mut back, Q6K_BLOCK);
        let err = w.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err < 1.5e-3, "max error {err}");
    }

    #[test]
    fn maps_names() {
        assert_eq!(hf_name("blk.3.attn_q.weight", 64).unwrap(), "model.layers.3.self_attn.q_proj.weight");
        assert_eq!(hf_name("blk.0.ssm_a", 64).unwrap(), "model.layers.0.linear_attn.A_log");
        assert_eq!(hf_name("blk.64.ffn_up.weight", 64).unwrap(), "mtp.layers.0.mlp.up_proj.weight");
        assert_eq!(hf_name("blk.64.nextn.eh_proj.weight", 64).unwrap(), "mtp.fc.weight");
        assert!(hf_name("blk.0.unknown", 64).is_none());
    }
}
