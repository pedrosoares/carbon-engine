// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! The Qwen3.5 / 3.6 / 3.8 vision tower (`model.visual.*`), after HF transformers'
//! `Qwen3_5VisionModel` (identical to `Qwen4ExpVisionModel`).
//!
//! An image arrives as `pixel_values [N, C*T*P*P]`: its `grid_h x grid_w` patches of
//! `P x P` pixels (each frame duplicated over `T`), listed in 2x2 merge-block order (see
//! `preprocess.rs`). The tower embeds the patches (a linear over each patch), adds a learned
//! position table resampled bilinearly to the image's grid, runs pre-LayerNorm transformer
//! blocks with 2-D rotary attention over the whole image (bidirectional), and merges every
//! 2x2 block of patches into one token for the language model: `[N / 4, out_hidden_size]`.

use super::layers::Placement;
use crate::config::VisionConfig;
use crate::weights::WeightSource;
use anyhow::{bail, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{LayerNorm, Linear};

const PREFIX: &str = "model.visual";
const LN_EPS: f64 = 1e-6;
/// Attention score elements per query tile (f32): bounds the memory of large images.
const SCORE_BUDGET: usize = 1 << 26;

struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
}

pub struct VisionTower {
    cfg: VisionConfig,
    /// The patch Conv3d as a linear: `[hidden, C*T*P*P]` and its bias.
    patch: Linear,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    merger_norm: LayerNorm,
    merger_fc1: Linear,
    merger_fc2: Linear,
    /// Rotary inverse frequencies over `head_dim / 2` (h and w halves).
    inv_freq: Vec<f32>,
    pub device: Device,
    pub dtype: DType,
}

fn linear(src: &dyn WeightSource, name: &str, p: &Placement) -> Result<Linear> {
    Ok(Linear::new(
        src.get(&format!("{name}.weight"), &p.device, p.dtype)?,
        Some(src.get(&format!("{name}.bias"), &p.device, p.dtype)?),
    ))
}

fn layer_norm(src: &dyn WeightSource, name: &str, p: &Placement) -> Result<LayerNorm> {
    // computed in f32 (see `ln`)
    Ok(LayerNorm::new(
        src.get(&format!("{name}.weight"), &p.device, DType::F32)?,
        src.get(&format!("{name}.bias"), &p.device, DType::F32)?,
        LN_EPS,
    ))
}

fn ln(norm: &LayerNorm, x: &Tensor) -> Result<Tensor> {
    Ok(norm.forward(&x.to_dtype(DType::F32)?)?.to_dtype(x.dtype())?)
}

impl VisionTower {
    pub fn load(src: &dyn WeightSource, cfg: &VisionConfig, p: &Placement) -> Result<Self> {
        let h = cfg.hidden_size;
        if !h.is_multiple_of(cfg.num_heads) || !(h / cfg.num_heads).is_multiple_of(4) {
            bail!("vision tower: head dim {} must be a multiple of 4", h / cfg.num_heads);
        }
        let side = (cfg.num_position_embeddings as f64).sqrt() as usize;
        if side * side != cfg.num_position_embeddings {
            bail!("vision tower: {} position embeddings are not a square grid", cfg.num_position_embeddings);
        }
        let w = src.get(&format!("{PREFIX}.patch_embed.proj.weight"), &p.device, p.dtype)?;
        let patch = Linear::new(
            w.reshape((h, cfg.patch_dim()))?,
            Some(src.get(&format!("{PREFIX}.patch_embed.proj.bias"), &p.device, p.dtype)?),
        );
        let blocks = (0..cfg.depth)
            .map(|i| {
                let b = format!("{PREFIX}.blocks.{i}");
                Ok(Block {
                    norm1: layer_norm(src, &format!("{b}.norm1"), p)?,
                    norm2: layer_norm(src, &format!("{b}.norm2"), p)?,
                    qkv: linear(src, &format!("{b}.attn.qkv"), p)?,
                    proj: linear(src, &format!("{b}.attn.proj"), p)?,
                    fc1: linear(src, &format!("{b}.mlp.linear_fc1"), p)?,
                    fc2: linear(src, &format!("{b}.mlp.linear_fc2"), p)?,
                })
            })
            .collect::<Result<_>>()?;
        // rotary over head_dim / 2 positions-dims: inv_freq has head_dim / 4 entries per axis
        let rot = h / cfg.num_heads / 2;
        let inv_freq = (0..rot / 2).map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / rot as f32)).collect();
        Ok(Self {
            cfg: cfg.clone(),
            patch,
            pos_embed: src.get(&format!("{PREFIX}.pos_embed.weight"), &p.device, p.dtype)?,
            blocks,
            merger_norm: layer_norm(src, &format!("{PREFIX}.merger.norm"), p)?,
            merger_fc1: linear(src, &format!("{PREFIX}.merger.linear_fc1"), p)?,
            merger_fc2: linear(src, &format!("{PREFIX}.merger.linear_fc2"), p)?,
            inv_freq,
            device: p.device.clone(),
            dtype: p.dtype,
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    /// One image: `pixels [t*h*w, C*T*P*P]` with its patch grid `(t, h, w)`; returns the
    /// language model's image tokens `[t*h*w / m^2, out_hidden_size]`.
    pub fn forward(&self, pixels: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (t, gh, gw) = grid;
        let m = self.cfg.spatial_merge_size;
        let n = t * gh * gw;
        if pixels.dims() != [n, self.cfg.patch_dim()] || !gh.is_multiple_of(m) || !gw.is_multiple_of(m) {
            bail!("vision tower: pixels {:?} do not match grid {grid:?}", pixels.dims());
        }
        let (rows, cols) = merge_order(t, gh, gw, m);
        let x = self.patch.forward(&pixels.to_device(&self.device)?.to_dtype(self.dtype)?)?;
        let x = (x + self.positions(&rows, &cols, gh, gw)?)?;
        let (cos, sin) = self.rotary(&rows, &cols)?;
        let mut x = x;
        for b in &self.blocks {
            x = (&x + self.attention(b, &ln(&b.norm1, &x)?, &cos, &sin)?)?;
            let h = b.fc1.forward(&ln(&b.norm2, &x)?)?;
            let h = match self.cfg.hidden_act.as_str() {
                "gelu" => h.gelu_erf()?,
                _ => h.gelu()?, // gelu_pytorch_tanh
            };
            x = (&x + b.fc2.forward(&h)?)?;
        }
        let x = ln(&self.merger_norm, &x)?.reshape((n / (m * m), m * m * self.cfg.hidden_size))?;
        let x = self.merger_fc1.forward(&x)?.gelu_erf()?;
        Ok(self.merger_fc2.forward(&x)?)
    }

    /// The learned `side x side` position table resampled to the image's grid (bilinear,
    /// align_corners), for every patch.
    fn positions(&self, rows: &[usize], cols: &[usize], gh: usize, gw: usize) -> Result<Tensor> {
        let side = (self.cfg.num_position_embeddings as f64).sqrt() as usize;
        let taps = |i: usize, size: usize| -> [(usize, f32); 2] {
            let src = i as f32 * (side - 1) as f32 / (size.max(2) - 1) as f32;
            let f = src.floor();
            let d = src - f;
            let a = (f as usize).min(side - 1);
            let b = (f as usize + 1).min(side - 1);
            [(a, 1.0 - d), (b, d)]
        };
        let n = rows.len();
        let (mut idx, mut wts) = (Vec::with_capacity(4 * n), Vec::with_capacity(4 * n));
        for (&r, &c) in rows.iter().zip(cols) {
            for (hi, hw) in taps(r, gh) {
                for (wi, ww) in taps(c, gw) {
                    idx.push((hi * side + wi) as u32);
                    wts.push(hw * ww);
                }
            }
        }
        let dev = &self.device;
        let g = self
            .pos_embed
            .index_select(&Tensor::from_vec(idx, 4 * n, dev)?, 0)?
            .to_dtype(DType::F32)?
            .reshape((n, 4, self.cfg.hidden_size))?;
        let w = Tensor::from_vec(wts, (n, 4, 1), dev)?;
        let g = g.broadcast_mul(&w)?;
        let mut acc = g.narrow(1, 0, 1)?;
        for k in 1..4 {
            acc = (acc + g.narrow(1, k, 1)?)?;
        }
        Ok(acc.squeeze(1)?.to_dtype(self.dtype)?)
    }

    /// cos / sin `[N, head_dim]` (f32): frequencies of the patch's row then of its column,
    /// repeated for the two rotate-half halves.
    fn rotary(&self, rows: &[usize], cols: &[usize]) -> Result<(Tensor, Tensor)> {
        let f = self.inv_freq.len();
        let mut ang = Vec::with_capacity(rows.len() * 4 * f);
        for (&r, &c) in rows.iter().zip(cols) {
            let half: Vec<f32> = self.inv_freq.iter().map(|w| r as f32 * w)
                .chain(self.inv_freq.iter().map(|w| c as f32 * w)).collect();
            ang.extend_from_slice(&half);
            ang.extend_from_slice(&half);
        }
        let a = Tensor::from_vec(ang, (rows.len(), 4 * f), &self.device)?;
        Ok((a.cos()?, a.sin()?))
    }

    fn attention(&self, b: &Block, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        let (nh, h) = (self.cfg.num_heads, self.cfg.hidden_size);
        let d = h / nh;
        let qkv = b.qkv.forward(x)?.reshape((n, 3, nh, d))?;
        let rope = |t: Tensor| -> Result<Tensor> {
            let t = t.to_dtype(DType::F32)?; // [n, nh, d]
            let (c, s) = (cos.unsqueeze(1)?, sin.unsqueeze(1)?);
            let rot = Tensor::cat(&[t.narrow(2, d / 2, d / 2)?.neg()?, t.narrow(2, 0, d / 2)?], 2)?;
            Ok((t.broadcast_mul(&c)? + rot.broadcast_mul(&s)?)?)
        };
        let q = rope(qkv.narrow(1, 0, 1)?.squeeze(1)?)?.transpose(0, 1)?.contiguous()?; // [nh, n, d]
        let k = rope(qkv.narrow(1, 1, 1)?.squeeze(1)?)?.transpose(0, 1)?.contiguous()?;
        let v = qkv.narrow(1, 2, 1)?.squeeze(1)?.to_dtype(DType::F32)?.transpose(0, 1)?.contiguous()?;
        let kt = k.t()?.contiguous()?;
        let scale = (d as f64).powf(-0.5);
        let tile = (SCORE_BUDGET / (nh * n).max(1)).clamp(1, n);
        let mut outs = Vec::with_capacity(n.div_ceil(tile));
        let mut a = 0;
        while a < n {
            let len = tile.min(n - a);
            let s = (q.narrow(1, a, len)?.matmul(&kt)? * scale)?;
            let p = candle_nn::ops::softmax(&s, D::Minus1)?;
            outs.push(p.matmul(&v)?); // [nh, len, d]
            a += len;
        }
        let o = Tensor::cat(&outs, 1)?.transpose(0, 1)?.reshape((n, h))?.to_dtype(x.dtype())?;
        Ok(b.proj.forward(&o)?)
    }
}

/// Grid row and column of every patch in 2x2 merge-block order: block row, block column, then
/// the block's rows and columns (`reshape(h/m, m, w/m, m).transpose(1, 2)`), per frame.
pub fn merge_order(t: usize, gh: usize, gw: usize, m: usize) -> (Vec<usize>, Vec<usize>) {
    let mut rows = Vec::with_capacity(t * gh * gw);
    let mut cols = Vec::with_capacity(t * gh * gw);
    for _ in 0..t {
        for br in 0..gh / m {
            for bc in 0..gw / m {
                for ir in 0..m {
                    for ic in 0..m {
                        rows.push(br * m + ir);
                        cols.push(bc * m + ic);
                    }
                }
            }
        }
    }
    (rows, cols)
}
