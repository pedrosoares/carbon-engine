// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! `--dummy-weight`: random weights shaped from `config.json` alone, generated lazily per
//! tensor (nothing is read from disk), for benchmarking and smoke-testing a model layout
//! without its checkpoint.

use crate::config::{Arch, LayerKind, ModelConfig, MoeNaming};
use crate::weights::WeightSource;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
enum Init {
    /// N(0, 1/sqrt(fan_in)) style projections.
    Proj,
    /// Norm weights: 1 (or 0 for zero-centred norms).
    Ones,
    Zeros,
    /// Small values, for biases / log-decays.
    Small,
}

pub struct DummySource {
    shapes: BTreeMap<String, (Vec<usize>, Init)>,
    seed: u64,
}

impl DummySource {
    pub fn new(cfg: &ModelConfig, seed: u64) -> Self {
        Self {
            shapes: weight_shapes(cfg),
            seed,
        }
    }

    pub fn shapes(&self) -> impl Iterator<Item = (&String, &Vec<usize>)> {
        self.shapes.iter().map(|(k, v)| (k, &v.0))
    }
}

fn fnv(s: &str, seed: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h | 1
}

impl WeightSource for DummySource {
    fn load_raw(&self, name: &str) -> Result<Tensor> {
        let (shape, init) = self
            .shapes
            .get(name)
            .with_context(|| format!("dummy checkpoint has no tensor '{name}'"))?;
        let n: usize = shape.iter().product();
        let fan_in = *shape.last().unwrap_or(&1) as f32;
        let data: Vec<f32> = match init {
            Init::Ones => vec![1.0; n],
            Init::Zeros => vec![0.0; n],
            Init::Proj | Init::Small => {
                let scale = match init {
                    Init::Proj => 1.0 / fan_in.sqrt(),
                    _ => 0.02,
                };
                let mut x = fnv(name, self.seed);
                (0..n)
                    .map(|_| {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        // uniform(-1, 1) * sqrt(3) has unit variance
                        ((x >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 1.732 * scale
                    })
                    .collect()
            }
        };
        Ok(Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)?.to_dtype(DType::BF16)?)
    }

    fn contains(&self, name: &str) -> bool {
        self.shapes.contains_key(name)
    }

    fn names(&self) -> Vec<String> {
        self.shapes.keys().cloned().collect()
    }
}

/// Every tensor the loaders read for `cfg`, with its shape.
fn weight_shapes(cfg: &ModelConfig) -> BTreeMap<String, (Vec<usize>, Init)> {
    let mut m = BTreeMap::new();
    let mut add = |name: String, shape: Vec<usize>, init: Init| {
        m.insert(name, (shape, init));
    };
    let h = cfg.hidden_size;
    let norm = if cfg.norm_zero_centered {
        Init::Zeros
    } else {
        Init::Ones
    };
    let pre = &cfg.prefix;
    add(format!("{pre}embed_tokens.weight"), vec![cfg.vocab_size, h], Init::Proj);
    add(format!("{pre}norm.weight"), vec![h], norm);
    if !cfg.tie_word_embeddings {
        add("lm_head.weight".into(), vec![cfg.vocab_size, h], Init::Proj);
    }
    let (nh, kvh, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    for l in 0..cfg.num_layers {
        let p = cfg.layer_prefix(l);
        add(format!("{p}.input_layernorm.weight"), vec![h], norm);
        add(format!("{p}.post_attention_layernorm.weight"), vec![h], norm);
        match cfg.layer_types[l] {
            LayerKind::Full => {
                let a = format!("{p}.self_attn");
                let q_rows = nh * hd * if cfg.attn_output_gate { 2 } else { 1 };
                add(format!("{a}.q_proj.weight"), vec![q_rows, h], Init::Proj);
                add(format!("{a}.k_proj.weight"), vec![kvh * hd, h], Init::Proj);
                add(format!("{a}.v_proj.weight"), vec![kvh * hd, h], Init::Proj);
                add(format!("{a}.o_proj.weight"), vec![h, nh * hd], Init::Proj);
                if cfg.attention_bias {
                    add(format!("{a}.q_proj.bias"), vec![q_rows], Init::Small);
                    add(format!("{a}.k_proj.bias"), vec![kvh * hd], Init::Small);
                    add(format!("{a}.v_proj.bias"), vec![kvh * hd], Init::Small);
                }
                if cfg.qk_norm {
                    add(format!("{a}.q_norm.weight"), vec![hd], norm);
                    add(format!("{a}.k_norm.weight"), vec![hd], norm);
                }
            }
            LayerKind::Linear => {
                let lc = cfg.linear_attn.as_ref().expect("linear attention config");
                let a = format!("{p}.linear_attn");
                let (kd, vd) = (lc.num_k_heads * lc.k_head_dim, lc.num_v_heads * lc.v_head_dim);
                add(format!("{a}.in_proj_qkv.weight"), vec![2 * kd + vd, h], Init::Proj);
                add(format!("{a}.in_proj_z.weight"), vec![vd, h], Init::Proj);
                add(format!("{a}.in_proj_b.weight"), vec![lc.num_v_heads, h], Init::Proj);
                add(format!("{a}.in_proj_a.weight"), vec![lc.num_v_heads, h], Init::Proj);
                add(
                    format!("{a}.conv1d.weight"),
                    vec![2 * kd + vd, 1, lc.conv_kernel],
                    Init::Proj,
                );
                add(format!("{a}.dt_bias"), vec![lc.num_v_heads], Init::Ones);
                add(format!("{a}.A_log"), vec![lc.num_v_heads], Init::Small);
                add(format!("{a}.norm.weight"), vec![lc.v_head_dim], Init::Ones);
                add(format!("{a}.out_proj.weight"), vec![h, vd], Init::Proj);
            }
        }
        if cfg.layer_is_moe(l) {
            let moe = cfg.moe.as_ref().unwrap();
            let (block, names) = match moe.naming {
                MoeNaming::Mixtral => (format!("{p}.block_sparse_moe"), ["w1", "w3", "w2"]),
                MoeNaming::Qwen => (format!("{p}.mlp"), ["gate_proj", "up_proj", "down_proj"]),
            };
            let mi = moe.intermediate_size;
            add(format!("{block}.gate.weight"), vec![moe.num_experts, h], Init::Proj);
            for e in 0..moe.num_experts {
                add(format!("{block}.experts.{e}.{}.weight", names[0]), vec![mi, h], Init::Proj);
                add(format!("{block}.experts.{e}.{}.weight", names[1]), vec![mi, h], Init::Proj);
                add(format!("{block}.experts.{e}.{}.weight", names[2]), vec![h, mi], Init::Proj);
            }
            let si = moe.shared_expert_intermediate_size;
            if si > 0 {
                let s = format!("{block}.shared_expert");
                add(format!("{s}.gate_proj.weight"), vec![si, h], Init::Proj);
                add(format!("{s}.up_proj.weight"), vec![si, h], Init::Proj);
                add(format!("{s}.down_proj.weight"), vec![h, si], Init::Proj);
                if cfg.arch == Arch::Qwen35 || cfg.arch == Arch::Qwen3Moe {
                    add(format!("{block}.shared_expert_gate.weight"), vec![1, h], Init::Proj);
                }
            }
        } else {
            let i = cfg.intermediate_size;
            add(format!("{p}.mlp.gate_proj.weight"), vec![i, h], Init::Proj);
            add(format!("{p}.mlp.up_proj.weight"), vec![i, h], Init::Proj);
            add(format!("{p}.mlp.down_proj.weight"), vec![h, i], Init::Proj);
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{write_tiny_model, TinySpec};
    use crate::weights::SafetensorsSource;

    /// The dummy layout matches a real checkpoint's names and shapes for every tiny arch.
    #[test]
    fn matches_real_checkpoint_layout() {
        for arch in ["llama", "qwen2", "qwen3", "qwen3_moe", "mixtral", "qwen3_5_moe"] {
            let dir = tempfile::tempdir().unwrap();
            write_tiny_model(dir.path(), &TinySpec::new(arch)).unwrap();
            let mut cfg = ModelConfig::from_dir(dir.path()).unwrap();
            let real = SafetensorsSource::open(dir.path()).unwrap();
            if real.contains("model.language_model.embed_tokens.weight") {
                cfg.prefix = "model.language_model.".into();
            }
            let dummy = DummySource::new(&cfg, 0);
            let mut real_names = real.names();
            real_names.sort();
            assert_eq!(dummy.names(), real_names, "{arch}");
            for (name, shape) in dummy.shapes() {
                assert_eq!(
                    real.load_raw(name).unwrap().dims(),
                    shape.as_slice(),
                    "{arch} {name}"
                );
            }
            let t = dummy.load_raw(&dummy.names()[0]).unwrap();
            assert!(t.to_dtype(DType::F32).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().is_finite());
        }
    }
}
