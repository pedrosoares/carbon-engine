// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Inkling 2's image encoder (llama.cpp `inkling2v`, the `mmproj` GGUF file).
//!
//! A ViT over `patch x patch` patches of the image as it is (padded to a multiple of
//! `patch * merge`): patch embedding, learned position embeddings resized bilinearly to the
//! patch grid, a CLS token in front (its own position embedding), a pre-LayerNorm, then
//! pre-norm blocks with biased q / k / v / out projections, 2D rope on queries and keys (the
//! first half of each head rotated by the patch column, the second by the row; CLS at 0, patches
//! from 1), layer-scaled attention and GELU MLP branches. The CLS output is dropped, each 2x2
//! block of patches is concatenated (`[dy][dx][channel]`) and a two-layer GELU MLP projects it to
//! the language model's width: one language-model token per block.
//!
//! Weights are bf16 on the device (matmuls in bf16), activations and norms in f32. Attention
//! is chunked over queries so large images (up to 16k patches) fit.

use crate::config::VisionConfig;
use crate::gguf::Gguf;
use anyhow::{bail, ensure, Result};
use candle_core::{DType, Device, Tensor};

struct Linear {
    w: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    fn load(g: &Gguf, name: &str, dev: &Device) -> Result<Self> {
        let w = g.f32_tensor(&format!("{name}.weight"), dev)?.to_dtype(DType::BF16)?;
        let bias = format!("{name}.bias");
        let b = if g.tensors.contains_key(&bias) { Some(g.f32_tensor(&bias, dev)?) } else { None };
        Ok(Self { w, b })
    }

    /// `x [T, in]` f32 -> `[T, out]` f32.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.to_dtype(DType::BF16)?.matmul(&self.w.t()?)?.to_dtype(DType::F32)?;
        Ok(match &self.b {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        })
    }
}

struct Norm {
    w: Tensor,
    b: Tensor,
}

impl Norm {
    fn load(g: &Gguf, name: &str, dev: &Device) -> Result<Self> {
        Ok(Self { w: g.f32_tensor(&format!("{name}.weight"), dev)?, b: g.f32_tensor(&format!("{name}.bias"), dev)? })
    }

    fn forward(&self, x: &Tensor, eps: f32) -> Result<Tensor> {
        Ok(candle_nn::ops::layer_norm(&x.contiguous()?, &self.w, &self.b, eps)?)
    }
}

struct Block {
    ln1: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ls1: Tensor,
    ln2: Norm,
    up: Linear,
    down: Linear,
    ls2: Tensor,
}

pub struct InklingVision {
    patch: Linear,
    /// Learned positions, row-major over a `side x side` grid: `[side * side, D]` on the host.
    pos: Vec<f32>,
    side: usize,
    /// CLS embedding plus its position embedding, `[1, D]`.
    cls: Tensor,
    pre_ln: Norm,
    blocks: Vec<Block>,
    mm1: Linear,
    mm2: Linear,
    hidden: usize,
    heads: usize,
    patch_size: usize,
    merge: usize,
    eps: f32,
    pub device: Device,
}

/// Most f32 score elements one attention chunk materializes.
const SCORE_BUDGET: usize = 1 << 26;

impl InklingVision {
    pub fn load(g: &Gguf, vc: &VisionConfig, dev: &Device) -> Result<Self> {
        ensure!(vc.is_inkling(), "not an inkling2v projector");
        let d = vc.hidden_size;
        ensure!(d % vc.num_heads == 0 && (d / vc.num_heads) % 4 == 0, "inkling2v: {} heads for width {d}", vc.num_heads);
        let p = vc.patch_size;
        let pw = g.f32_tensor("v.patch_embd.weight", dev)?; // [D, 3, P, P]
        ensure!(pw.dims() == [d, 3, p, p], "inkling2v: patch embedding {:?}", pw.dims());
        let patch = Linear {
            w: pw.reshape((d, 3 * p * p))?.to_dtype(DType::BF16)?,
            b: if g.tensors.contains_key("v.patch_embd.bias") { Some(g.f32_tensor("v.patch_embd.bias", dev)?) } else { None },
        };
        let pos_t = g.f32_tensor("v.position_embd.weight", &Device::Cpu)?;
        let (n_pos, d2) = pos_t.dims2()?;
        let side = (n_pos as f64).sqrt().round() as usize;
        ensure!(side * side == n_pos && d2 == d, "inkling2v: position table {:?}", pos_t.dims());
        let cls = (g.f32_tensor("v.class_embd", dev)? + g.f32_tensor("v.class_position_embd.weight", dev)?)?.reshape((1, d))?;
        let mut blocks = Vec::with_capacity(vc.depth);
        for l in 0..vc.depth {
            let n = |s: &str| format!("v.blk.{l}.{s}");
            blocks.push(Block {
                ln1: Norm::load(g, &n("ln1"), dev)?,
                q: Linear::load(g, &n("attn_q"), dev)?,
                k: Linear::load(g, &n("attn_k"), dev)?,
                v: Linear::load(g, &n("attn_v"), dev)?,
                o: Linear::load(g, &n("attn_out"), dev)?,
                ls1: g.f32_tensor(&n("ls1.weight"), dev)?,
                ln2: Norm::load(g, &n("ln2"), dev)?,
                up: Linear::load(g, &n("ffn_up"), dev)?,
                down: Linear::load(g, &n("ffn_down"), dev)?,
                ls2: g.f32_tensor(&n("ls2.weight"), dev)?,
            });
        }
        let mm1 = Linear::load(g, "mm.1", dev)?;
        let mm2 = Linear::load(g, "mm.2", dev)?;
        let merge = vc.spatial_merge_size;
        ensure!(mm1.w.dim(1)? == d * merge * merge, "inkling2v: projector input {}", mm1.w.dim(1)?);
        ensure!(mm2.w.dim(0)? == vc.out_hidden_size, "inkling2v: projector output {}", mm2.w.dim(0)?);
        tracing::info!("inkling2v image encoder: {} layers, width {d}, patch {p}, merge {merge}", vc.depth);
        Ok(Self {
            patch,
            pos: pos_t.flatten_all()?.to_vec1::<f32>()?,
            side,
            cls,
            pre_ln: Norm::load(g, "v.pre_ln", dev)?,
            blocks,
            mm1,
            mm2,
            hidden: d,
            heads: vc.num_heads,
            patch_size: p,
            merge,
            eps: vc.layer_norm_eps as f32,
            device: dev.clone(),
        })
    }

    /// The position table resized to a `gh x gw` patch grid (bilinear, half-pixel centres,
    /// edges clamped: ggml_interpolate's bilinear mode), `[gh * gw, D]`.
    fn positions(&self, gh: usize, gw: usize) -> Vec<f32> {
        let (s, d) = (self.side, self.hidden);
        if gh == s && gw == s {
            return self.pos.clone();
        }
        let axis = |i: usize, n_out: usize| -> (usize, usize, f32) {
            let x = (i as f32 + 0.5) * s as f32 / n_out as f32 - 0.5;
            let x0 = x.floor() as isize;
            let lo = x0.clamp(0, s as isize - 1) as usize;
            let hi = (x0 + 1).clamp(0, s as isize - 1) as usize;
            (lo, hi, (x - lo as f32).clamp(0.0, 1.0))
        };
        let mut out = vec![0f32; gh * gw * d];
        for y in 0..gh {
            let (y0, y1, dy) = axis(y, gh);
            for x in 0..gw {
                let (x0, x1, dx) = axis(x, gw);
                let (a, b, c, e) = (&self.pos[(y0 * s + x0) * d..], &self.pos[(y0 * s + x1) * d..], &self.pos[(y1 * s + x0) * d..], &self.pos[(y1 * s + x1) * d..]);
                let o = &mut out[(y * gw + x) * d..(y * gw + x + 1) * d];
                for k in 0..d {
                    let top = a[k] * (1.0 - dx) + b[k] * dx;
                    let bot = c[k] * (1.0 - dx) + e[k] * dx;
                    o[k] = top * (1.0 - dy) + bot * dy;
                }
            }
        }
        out
    }

    /// cos / sin `[T, hd / 2]` of the 2D rope: for pair i of a head, the column (first half of the
    /// pairs) or row (second half) position times 10000^(-2 i' / (hd / 2)).
    fn rope_tables(&self, gh: usize, gw: usize) -> Result<(Tensor, Tensor)> {
        let hd = self.hidden / self.heads;
        let pairs = hd / 2;
        let half = pairs / 2;
        let t = gh * gw + 1;
        let (mut cos, mut sin) = (Vec::with_capacity(t * pairs), Vec::with_capacity(t * pairs));
        for i in 0..t {
            let (px, py) = if i == 0 { (0.0, 0.0) } else { (((i - 1) % gw + 1) as f64, ((i - 1) / gw + 1) as f64) };
            for j in 0..pairs {
                let (pos, k) = if j < half { (px, j) } else { (py, j - half) };
                let a = pos * 10000f64.powf(-2.0 * k as f64 / (hd / 2) as f64);
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
        Ok((Tensor::from_vec(cos, (t, 1, pairs, 1), &self.device)?, Tensor::from_vec(sin, (t, 1, pairs, 1), &self.device)?))
    }

    /// Adjacent pairs `(x[2i], x[2i+1])` rotated: `x [T, H, hd]`.
    fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (t, h, hd) = x.dims3()?;
        let x = x.reshape((t, h, hd / 2, 2))?;
        let (x0, x1) = (x.narrow(3, 0, 1)?, x.narrow(3, 1, 1)?);
        let r0 = (x0.broadcast_mul(cos)? - x1.broadcast_mul(sin)?)?;
        let r1 = (x1.broadcast_mul(cos)? + x0.broadcast_mul(sin)?)?;
        Ok(Tensor::cat(&[r0, r1], 3)?.reshape((t, h, hd))?)
    }

    /// Bidirectional attention of `q, k, v [H, T, hd]`, chunked over queries: `[T, H * hd]`.
    fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (h, t, hd) = q.dims3()?;
        let scale = 1.0 / (hd as f64).sqrt();
        let kt = k.t()?.contiguous()?;
        let chunk = (SCORE_BUDGET / (h * t).max(1)).clamp(1, t);
        let mut outs = Vec::with_capacity(t.div_ceil(chunk));
        let mut i = 0;
        while i < t {
            let n = chunk.min(t - i);
            let qc = q.narrow(1, i, n)?.contiguous()?;
            let s = (qc.matmul(&kt)? * scale)?;
            let p = candle_nn::ops::softmax_last_dim(&s)?;
            outs.push(p.matmul(v)?);
            i += n;
        }
        let o = if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 1)? };
        Ok(o.transpose(0, 1)?.contiguous()?.reshape((t, h * hd))?)
    }

    /// `pixels [gh * gw, 3 * P * P]` (raster-order patches, `images::ImageProcessor` with `pad`)
    /// -> language-model tokens `[gh * gw / merge^2, out]` (f32).
    pub fn forward(&self, pixels: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (gt, gh, gw) = grid;
        let m = self.merge;
        if gt != 1 || gh % m != 0 || gw % m != 0 {
            bail!("inkling2v: patch grid {grid:?} is not a single frame in whole {m}x{m} blocks");
        }
        let n = gh * gw;
        let (d, heads) = (self.hidden, self.heads);
        let hd = d / heads;
        ensure!(pixels.dims2()? == (n, 3 * self.patch_size * self.patch_size), "inkling2v: pixels {:?}", pixels.dims());
        let pos = Tensor::from_vec(self.positions(gh, gw), (n, d), &self.device)?;
        let x = (self.patch.forward(&pixels.to_dtype(DType::F32)?)? + pos)?;
        let mut x = self.pre_ln.forward(&Tensor::cat(&[&self.cls, &x], 0)?, self.eps)?;
        let (cos, sin) = self.rope_tables(gh, gw)?;
        let t = n + 1;
        for b in &self.blocks {
            let h = b.ln1.forward(&x, self.eps)?;
            let heads_of = |y: Tensor, rope: bool| -> Result<Tensor> {
                let y = y.reshape((t, heads, hd))?;
                let y = if rope { Self::rope(&y, &cos, &sin)? } else { y };
                Ok(y.transpose(0, 1)?.contiguous()?)
            };
            let q = heads_of(b.q.forward(&h)?, true)?;
            let k = heads_of(b.k.forward(&h)?, true)?;
            let v = heads_of(b.v.forward(&h)?, false)?;
            let a = b.o.forward(&Self::attention(&q, &k, &v)?)?;
            x = (x + a.broadcast_mul(&b.ls1)?)?;
            let h = b.ln2.forward(&x, self.eps)?;
            let f = b.down.forward(&b.up.forward(&h)?.gelu_erf()?)?;
            x = (x + f.broadcast_mul(&b.ls2)?)?;
        }
        // drop CLS; each m x m block of patches becomes one row [dy][dx][channel]
        let x = x
            .narrow(0, 1, n)?
            .reshape((gh / m, m, gw / m, m, d))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .reshape((n / (m * m), m * m * d))?;
        let y = self.mm2.forward(&self.mm1.forward(&x)?.gelu_erf()?)?;
        Ok(y.contiguous()?)
    }
}
