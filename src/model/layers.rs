// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Basic building blocks: linear layers (with tensor-parallel sharding), RMSNorm,
//! rotary embeddings and the gated MLP.

use crate::config::{ModelConfig, RopeScaling};
use crate::weights::WeightSource;
use anyhow::{bail, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::Linear;

/// Where a layer runs and which tensor-parallel shard it holds.
#[derive(Clone, Debug)]
pub struct Placement {
    pub device: Device,
    pub dtype: DType,
    pub rank: usize,
    pub world: usize,
    /// Keep FP8 / NVFP4 projections quantized (decoded per call) instead of dequantizing at load.
    pub keep_quantized: bool,
}

impl Placement {
    pub fn single(device: Device, dtype: DType) -> Self {
        Self {
            device,
            dtype,
            rank: 0,
            world: 1,
            keep_quantized: false,
        }
    }
}

/// Column-parallel projection, kept quantized when the placement allows (single rank only).
pub fn load_column_q(src: &dyn WeightSource, prefix: &str, p: &Placement) -> Result<QLinear> {
    if p.keep_quantized && p.world == 1 {
        return QLinear::load(src, prefix, true, p);
    }
    Ok(QLinear::Dense(load_column_parallel(src, prefix, p)?))
}

/// Row-parallel projection, kept quantized when the placement allows (single rank only).
pub fn load_row_q(src: &dyn WeightSource, prefix: &str, p: &Placement) -> Result<QLinear> {
    if p.keep_quantized && p.world == 1 {
        return QLinear::load(src, prefix, true, p);
    }
    Ok(QLinear::Dense(load_row_parallel(src, prefix, p)?))
}

pub fn load_linear(src: &dyn WeightSource, prefix: &str, p: &Placement) -> Result<Linear> {
    let w = src.get(&format!("{prefix}.weight"), &p.device, p.dtype)?;
    let bname = format!("{prefix}.bias");
    let b = if src.contains(&bname) {
        Some(src.get(&bname, &p.device, p.dtype)?)
    } else {
        None
    };
    Ok(Linear::new(w, b))
}

/// Output features split across ranks (q/k/v, gate/up).
pub fn load_column_parallel(src: &dyn WeightSource, prefix: &str, p: &Placement) -> Result<Linear> {
    let w = src.get_shard(
        &format!("{prefix}.weight"),
        0,
        p.rank,
        p.world,
        &p.device,
        p.dtype,
    )?;
    let bname = format!("{prefix}.bias");
    let b = if src.contains(&bname) {
        Some(src.get_shard(&bname, 0, p.rank, p.world, &p.device, p.dtype)?)
    } else {
        None
    };
    Ok(Linear::new(w, b))
}

/// Input features split across ranks (o_proj, down); outputs are partial sums.
pub fn load_row_parallel(src: &dyn WeightSource, prefix: &str, p: &Placement) -> Result<Linear> {
    let w = src.get_shard(
        &format!("{prefix}.weight"),
        1,
        p.rank,
        p.world,
        &p.device,
        p.dtype,
    )?;
    let bname = format!("{prefix}.bias");
    // the bias is added once, by rank 0, so the all-reduce sums it exactly once
    let b = if src.contains(&bname) && p.rank == 0 {
        Some(src.get(&bname, &p.device, p.dtype)?)
    } else {
        None
    };
    Ok(Linear::new(w, b))
}

#[derive(Clone, Debug)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    pub fn load(src: &dyn WeightSource, name: &str, eps: f64, p: &Placement) -> Result<Self> {
        Ok(Self {
            weight: src.get(name, &p.device, p.dtype)?,
            eps: eps as f32,
        })
    }

    /// A model norm: zero-centred checkpoints (Qwen3.5) store `w - 1`.
    pub fn load_model(
        src: &dyn WeightSource,
        name: &str,
        cfg: &ModelConfig,
        p: &Placement,
    ) -> Result<Self> {
        let w = src.get(name, &p.device, DType::F32)?;
        let w = if cfg.norm_zero_centered {
            (w + 1.0)?
        } else {
            w
        };
        Ok(Self {
            weight: w.to_dtype(p.dtype)?,
            eps: cfg.rms_norm_eps as f32,
        })
    }

    /// The (already 1 + w for zero-centred checkpoints) weight and epsilon.
    pub fn parts(&self) -> (&Tensor, f32) {
        (&self.weight, self.eps)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::ops::rms_norm(
            &x.contiguous()?,
            &self.weight,
            self.eps,
        )?)
    }
}

pub fn activation(name: &str, x: &Tensor) -> Result<Tensor> {
    Ok(match name {
        "silu" | "swish" => candle_nn::ops::silu(x)?,
        "gelu" => x.gelu_erf()?,
        "gelu_pytorch_tanh" | "gelu_tanh" | "gelu_new" => x.gelu()?,
        "relu" => x.relu()?,
        other => bail!("unsupported activation '{other}'"),
    })
}

/// Rotary embedding tables (NeoX / rotate-half layout, as used by HF Llama and Qwen).
#[derive(Clone)]
pub struct Rotary {
    cos: Tensor,
    sin: Tensor,
    max_pos: usize,
    /// Interleaved mRoPE: per frequency, 1.0 where it reads the (t, h, w) axis, `[1, half]`.
    mrope: Option<[Tensor; 3]>,
}

pub fn rope_inv_freq(cfg: &ModelConfig) -> Vec<f64> {
    let d = cfg.rotary_dim;
    let mut inv: Vec<f64> = (0..d / 2)
        .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f64 / d as f64))
        .collect();
    match &cfg.rope_scaling {
        Some(RopeScaling::Linear { factor }) => inv.iter_mut().for_each(|f| *f /= factor),
        Some(RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position,
        }) => {
            let old = *original_max_position as f64;
            let low_wl = old / low_freq_factor;
            let high_wl = old / high_freq_factor;
            for f in inv.iter_mut() {
                let wl = 2.0 * std::f64::consts::PI / *f;
                if wl > low_wl {
                    *f /= factor;
                } else if wl >= high_wl {
                    let smooth =
                        (old / wl - low_freq_factor) / (high_freq_factor - low_freq_factor);
                    *f = (1.0 - smooth) * *f / factor + smooth * *f;
                }
            }
        }
        Some(RopeScaling::Yarn {
            factor,
            original_max_position,
            beta_fast,
            beta_slow,
            ..
        }) => {
            // HF `_compute_yarn_parameters` (truncated correction range)
            let base = cfg.rope_theta;
            let dim = d as f64;
            let corr = |rot: f64| {
                dim * (*original_max_position as f64 / (rot * 2.0 * std::f64::consts::PI)).ln()
                    / (2.0 * base.ln())
            };
            let low = corr(*beta_fast).floor().max(0.0);
            let mut high = corr(*beta_slow).ceil().min(dim - 1.0);
            if high == low {
                high += 0.001;
            }
            for (i, f) in inv.iter_mut().enumerate() {
                let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
                let extrapolation = 1.0 - ramp;
                *f = *f / factor * (1.0 - extrapolation) + *f * extrapolation;
            }
        }
        None => {}
    }
    inv
}

/// cos/sin multiplier of the rope scheme (YaRN's attention temperature, else 1).
pub fn rope_attention_factor(cfg: &ModelConfig) -> f64 {
    match &cfg.rope_scaling {
        Some(RopeScaling::Yarn {
            attention_factor, ..
        }) => *attention_factor,
        _ => 1.0,
    }
}

impl Rotary {
    pub fn new(cfg: &ModelConfig, max_pos: usize, p: &Placement) -> Result<Self> {
        let inv = rope_inv_freq(cfg);
        let mscale = rope_attention_factor(cfg);
        let half = inv.len();
        let mut cos = Vec::with_capacity(max_pos * half);
        let mut sin = Vec::with_capacity(max_pos * half);
        for pos in 0..max_pos {
            for f in &inv {
                let a = pos as f64 * f;
                cos.push((a.cos() * mscale) as f32);
                sin.push((a.sin() * mscale) as f32);
            }
        }
        let cos = Tensor::from_vec(cos, (max_pos, half), &p.device)?.to_dtype(p.dtype)?;
        let sin = Tensor::from_vec(sin, (max_pos, half), &p.device)?.to_dtype(p.dtype)?;
        // HF `apply_interleaved_mrope`: frequency f reads h if f % 3 == 1 and f < 3 * sec[1],
        // w if f % 3 == 2 and f < 3 * sec[2], else t
        let mrope = match cfg.mrope_section {
            Some(sec) => {
                let axis = |f: usize| {
                    if f % 3 == 1 && f < 3 * sec[1] {
                        1
                    } else if f % 3 == 2 && f < 3 * sec[2] {
                        2
                    } else {
                        0
                    }
                };
                let mask = |a: usize| -> Result<Tensor> {
                    let m: Vec<f32> = (0..half).map(|f| f32::from(axis(f) == a)).collect();
                    Ok(Tensor::from_vec(m, (1, half), &p.device)?.to_dtype(p.dtype)?)
                };
                Some([mask(0)?, mask(1)?, mask(2)?])
            }
            None => None,
        };
        Ok(Self { cos, sin, max_pos, mrope })
    }

    /// cos/sin rows for (t, h, w) positions `[T, 3]`: each frequency from its axis' position.
    pub fn select_mrope(&self, pos: &[[u32; 3]], device: &Device) -> Result<(Tensor, Tensor)> {
        let Some(masks) = &self.mrope else {
            bail!("this model has no mRoPE sections (image positions)");
        };
        let max = pos.iter().flatten().copied().max().unwrap_or(0);
        let t = pos.len();
        let (mut cos, mut sin): (Option<Tensor>, Option<Tensor>) = (None, None);
        for (a, mask) in masks.iter().enumerate() {
            let ids = Tensor::from_vec(pos.iter().map(|p| p[a]).collect::<Vec<u32>>(), t, device)?;
            let (c, s) = self.select(&ids, max)?;
            let (c, s) = (c.broadcast_mul(mask)?, s.broadcast_mul(mask)?);
            cos = Some(match cos { Some(x) => (x + c)?, None => c });
            sin = Some(match sin { Some(x) => (x + s)?, None => s });
        }
        Ok((cos.unwrap(), sin.unwrap()))
    }

    /// The whole `[max_pos, rotary_dim/2]` cos and sin tables.
    pub fn tables(&self) -> (&Tensor, &Tensor) {
        (&self.cos, &self.sin)
    }

    /// cos/sin rows for the given positions: `[T, head_dim/2]` each.
    pub fn select(&self, positions: &Tensor, max_position: u32) -> Result<(Tensor, Tensor)> {
        if max_position as usize >= self.max_pos {
            bail!(
                "position {max_position} exceeds the rope table ({}); raise --max-seq-len",
                self.max_pos
            );
        }
        Ok((
            self.cos.index_select(positions, 0)?,
            self.sin.index_select(positions, 0)?,
        ))
    }

    /// Apply to `x: [T, heads, head_dim]`; with partial rope only the first `2 * cos.dim(1)`
    /// dims of each head rotate.
    pub fn apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let d = x.dim(2)?;
        let rd = cos.dim(1)? * 2;
        if rd == d {
            let x = x.unsqueeze(0)?.contiguous()?;
            return Ok(candle_nn::rotary_emb::rope_thd(&x, cos, sin)?.squeeze(0)?);
        }
        let rot = x.narrow(2, 0, rd)?.unsqueeze(0)?.contiguous()?;
        let rot = candle_nn::rotary_emb::rope_thd(&rot, cos, sin)?.squeeze(0)?;
        Ok(Tensor::cat(&[rot, x.narrow(2, rd, d - rd)?], 2)?)
    }
}

pub use crate::weights::{dequant_fp8, dequant_nvfp4, Nvfp4};

/// Run one of carbon's CUDA kernels; on failure warn once and return None so the caller falls
/// back to the generic tensor-op path.
#[cfg(feature = "cuda")]
fn cuda_quant(
    x: &Tensor,
    what: &str,
    f: impl FnOnce() -> candle_core::Result<Tensor>,
) -> Option<Tensor> {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    match f() {
        Ok(t) => Some(t),
        Err(e) => {
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    "carbon {what} CUDA kernel failed on {:?}, using the generic path: {e}",
                    x.device().location()
                );
            }
            None
        }
    }
}

/// Copies a host tensor to a device.
pub type Upload<'a> = dyn Fn(&Tensor) -> candle_core::Result<Tensor> + 'a;

/// A bias-free projection stored dense, or as block-FP8 codes + scales that are
/// dequantized per call (half the memory and PCIe traffic of bf16).
#[derive(Clone)]
pub enum QLinear {
    Dense(Linear),
    Fp8 { w: Tensor, scale: Tensor },
    Nvfp4(Nvfp4),
}

impl QLinear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fused =
            x.device().is_cpu() && x.rank() == 2 && x.dim(0)? <= crate::kernels::MAX_FUSED_ROWS;
        match self {
            Self::Dense(l) => Ok(l.forward(x)?),
            Self::Nvfp4(q)
                if fused
                    && q.packed.device().is_cpu()
                    && matches!(q.scale.dtype(), DType::U8 | DType::F8E4M3) =>
            {
                crate::kernels::nvfp4_matmul(x, &q.packed, &q.scale, q.global)
            }
            Self::Fp8 { w, scale } if fused && w.device().is_cpu() => {
                crate::kernels::fp8_matmul(x, w, scale)
            }
            Self::Fp8 { w, scale } => {
                #[cfg(feature = "cuda")]
                if x.device().is_cuda() && x.rank() == 2 && w.dtype() == DType::U8 {
                    use crate::kernels::cuda as k;
                    if let Some(y) = cuda_quant(x, "fp8", || {
                        if x.dim(0)? <= k::MAX_GEMV_ROWS {
                            k::fp8_gemv(x, w, scale)
                        } else {
                            Linear::new(k::fp8_dequant(w, scale, x.dtype())?, None).forward(x)
                        }
                    }) {
                        return Ok(y);
                    }
                }
                let w = dequant_fp8(w, scale, x.dtype())?;
                Ok(Linear::new(w, None).forward(x)?)
            }
            Self::Nvfp4(q) => {
                #[cfg(feature = "cuda")]
                if x.device().is_cuda() && x.rank() == 2 && q.scale.dtype() == DType::U8 {
                    use crate::kernels::cuda as k;
                    if let Some(y) = cuda_quant(x, "nvfp4", || {
                        if x.dim(0)? <= k::MAX_GEMV_ROWS {
                            k::nvfp4_gemv(x, &q.packed, &q.scale, q.global)
                        } else {
                            let w = k::nvfp4_dequant(&q.packed, &q.scale, q.global, x.dtype())?;
                            Linear::new(w, None).forward(x)
                        }
                    }) {
                        return Ok(y);
                    }
                }
                Ok(Linear::new(dequant_nvfp4(q, x.dtype())?, None).forward(x)?)
            }
        }
    }

    /// Move to `device`; dense weights are cast to `dtype`, FP8 codes stay FP8.
    pub fn to_device(&self, device: &Device, dtype: DType) -> Result<Self> {
        self.to_device_with(dtype, &|t| t.to_device(device))
    }

    /// Like `to_device`, with `up` copying each tensor to the device.
    pub fn to_device_with(&self, dtype: DType, up: &Upload<'_>) -> Result<Self> {
        Ok(match self {
            Self::Dense(l) => Self::Dense(Linear::new(
                up(l.weight())?.to_dtype(dtype)?,
                l.bias()
                    .map(|b| up(b).and_then(|b| b.to_dtype(dtype)))
                    .transpose()?,
            )),
            Self::Fp8 { w, scale } => Self::Fp8 {
                w: up(w)?,
                scale: up(scale)?,
            },
            Self::Nvfp4(q) => Self::Nvfp4(Nvfp4 {
                packed: up(&q.packed)?,
                scale: up(&q.scale)?,
                global: q.global,
            }),
        })
    }

    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::Dense(l) => l.weight().elem_count() * l.weight().dtype().size_in_bytes(),
            Self::Fp8 { w, scale } => {
                w.elem_count() + scale.elem_count() * scale.dtype().size_in_bytes()
            }
            Self::Nvfp4(q) => {
                q.packed.elem_count() + q.scale.elem_count() * q.scale.dtype().size_in_bytes()
            }
        }
    }

    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::Dense(_))
    }

    /// Load `prefix.weight`; FP8 / NVFP4 weights stay quantized when `keep_quantized` is set.
    pub fn load(
        src: &dyn WeightSource,
        prefix: &str,
        keep_quantized: bool,
        p: &Placement,
    ) -> Result<Self> {
        let name = format!("{prefix}.weight");
        // a bias only exists on dense projections here; keep those dense
        if keep_quantized && p.world == 1 && !src.contains(&format!("{prefix}.bias")) {
            if let Some(q) = src.load_nvfp4(&name)? {
                return Ok(Self::Nvfp4(Nvfp4 {
                    packed: q.packed.to_device(&p.device)?,
                    scale: q.scale.to_device(&p.device)?,
                    global: q.global,
                }));
            }
            if let Some((w, scale)) = src.load_fp8(&name)? {
                return Ok(Self::Fp8 {
                    w: w.to_device(&p.device)?,
                    scale: scale.to_device(&p.device)?,
                });
            }
        }
        Ok(Self::Dense(load_linear(src, prefix, p)?))
    }
}

/// SwiGLU-style MLP: `down(act(gate(x)) * up(x))`.
#[derive(Clone)]
pub struct Mlp {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
    act: String,
}

impl Mlp {
    pub fn new(gate: Linear, up: Linear, down: Linear, act: &str) -> Self {
        Self::new_q(
            QLinear::Dense(gate),
            QLinear::Dense(up),
            QLinear::Dense(down),
            act,
        )
    }

    pub fn new_q(gate: QLinear, up: QLinear, down: QLinear, act: &str) -> Self {
        Self {
            gate,
            up,
            down,
            act: act.to_string(),
        }
    }

    /// Dense MLP, column/row parallel under tensor parallelism (output is a partial sum).
    pub fn load(
        src: &dyn WeightSource,
        prefix: &str,
        names: [&str; 3],
        act: &str,
        p: &Placement,
    ) -> Result<Self> {
        Ok(Self::new_q(
            load_column_q(src, &format!("{prefix}.{}", names[0]), p)?,
            load_column_q(src, &format!("{prefix}.{}", names[1]), p)?,
            load_row_q(src, &format!("{prefix}.{}", names[2]), p)?,
            act,
        ))
    }

    /// An unsharded expert; block-FP8 weights stay quantized when `keep_fp8`.
    pub fn load_expert(
        src: &dyn WeightSource,
        prefix: &str,
        names: [&str; 3],
        act: &str,
        keep_fp8: bool,
        p: &Placement,
    ) -> Result<Self> {
        let l = |n: &str| QLinear::load(src, &format!("{prefix}.{n}"), keep_fp8, p);
        Ok(Self::new_q(l(names[0])?, l(names[1])?, l(names[2])?, act))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = activation(&self.act, &self.gate.forward(x)?)?;
        let u = self.up.forward(x)?;
        self.down.forward(&(g * u)?)
    }

    /// `[gate, up, down]` in the form carbon's MoE kernels read, or None when they can't
    /// (non-SiLU activation, biased projections).
    #[cfg(feature = "cuda")]
    pub fn kernel_projections(&self) -> Option<[crate::kernels::cuda::ExpertProj; 3]> {
        use crate::kernels::cuda::ExpertProj;
        if !matches!(self.act.as_str(), "silu" | "swish") {
            return None;
        }
        let one = |q: &QLinear| -> Option<ExpertProj> {
            Some(match q {
                QLinear::Dense(l) if l.bias().is_none() => ExpertProj::Dense(l.weight().clone()),
                QLinear::Dense(_) => return None,
                QLinear::Fp8 { w, scale } if w.dtype() == DType::U8 => ExpertProj::Fp8 {
                    w: w.clone(),
                    scale: scale.clone(),
                },
                QLinear::Fp8 { .. } => return None,
                QLinear::Nvfp4(q) if q.scale.dtype() == DType::U8 => ExpertProj::Nvfp4 {
                    packed: q.packed.clone(),
                    scale: q.scale.clone(),
                    global: q.global,
                },
                QLinear::Nvfp4(_) => return None,
            })
        };
        Some([one(&self.gate)?, one(&self.up)?, one(&self.down)?])
    }

    pub fn to_device(&self, device: &Device, dtype: DType) -> Result<Self> {
        self.to_device_with(dtype, &|t| t.to_device(device))
    }

    /// Like `to_device`, with `up` copying each tensor to the device.
    pub fn to_device_with(&self, dtype: DType, up: &Upload<'_>) -> Result<Self> {
        Ok(Self {
            gate: self.gate.to_device_with(dtype, up)?,
            up: self.up.to_device_with(dtype, up)?,
            down: self.down.to_device_with(dtype, up)?,
            act: self.act.clone(),
        })
    }

    /// Forward in f32 regardless of the stored dtype (weights are upcast per call).
    pub fn forward_f32(&self, x: &Tensor) -> Result<Tensor> {
        let up = |l: &QLinear| -> Result<QLinear> {
            Ok(match l {
                QLinear::Dense(l) if l.weight().dtype() != DType::F32 => {
                    QLinear::Dense(Linear::new(
                        l.weight().to_dtype(DType::F32)?,
                        l.bias().map(|b| b.to_dtype(DType::F32)).transpose()?,
                    ))
                }
                other => other.clone(),
            })
        };
        let x = x.to_dtype(DType::F32)?;
        let g = activation(&self.act, &up(&self.gate)?.forward(&x)?)?;
        let u = up(&self.up)?.forward(&x)?;
        up(&self.down)?.forward(&(g * u)?)
    }

    pub fn is_quantized(&self) -> bool {
        self.gate.is_quantized()
    }

    pub fn size_in_bytes(&self) -> usize {
        self.gate.size_in_bytes() + self.up.size_in_bytes() + self.down.size_in_bytes()
    }
}

/// Softmax over the last dim computed in f32.
pub fn softmax_f32(x: &Tensor) -> Result<Tensor> {
    let dt = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let max = x.max_keepdim(D::Minus1)?;
    let e = x.broadcast_sub(&max)?.exp()?;
    let s = e.sum_keepdim(D::Minus1)?;
    Ok(e.broadcast_div(&s)?.to_dtype(dt)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    #[test]
    fn yarn_rope() {
        let mut v = serde_json::json!({"model_type": "qwen3", "hidden_size": 64, "num_attention_heads": 1,
            "head_dim": 64, "num_hidden_layers": 1, "vocab_size": 8, "rope_theta": 1000000.0,
            "max_position_embeddings": 131072});
        let plain = rope_inv_freq(&ModelConfig::from_hf(&v).unwrap());
        v["rope_scaling"] = serde_json::json!({"rope_type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 32768});
        let cfg = ModelConfig::from_hf(&v).unwrap();
        let yarn = rope_inv_freq(&cfg);
        // high-frequency dims extrapolate (unchanged), low-frequency ones interpolate (/factor)
        assert_eq!(yarn[0], plain[0]);
        assert!((yarn[31] - plain[31] / 4.0).abs() < 1e-15);
        assert!(yarn.iter().zip(&plain).all(|(y, p)| *y <= *p && *y >= p / 4.0 - 1e-18));
        assert!((rope_attention_factor(&cfg) - (0.1 * 4f64.ln() + 1.0)).abs() < 1e-12);
    }
}

