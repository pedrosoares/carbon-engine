// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! LFM2 gated short convolution (`Lfm2ShortConv` in HF transformers, `shortconv` in llama.cpp).
//!
//! `in_proj` splits each token into gates `b`, `c` and values `x`; the causal depthwise conv
//! (kernel `K`, no bias, no activation) runs over `b * x` and its output is gated by `c` before
//! `out_proj`. Each sequence owns a state slot with its last `K - 1` conv inputs (`b * x`).

use super::attention::DeviceBatch;
use super::layers::{load_row_parallel, Placement, QLinear};
use crate::config::ModelConfig;
use crate::weights::WeightSource;
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::Linear;

pub struct ShortConv {
    in_proj: QLinear,
    out_proj: QLinear,
    /// Conv taps (f32), one `[1, 1, channels]` row per kernel position, oldest input first.
    taps: Vec<Tensor>,
    /// Channels on this rank.
    channels: usize,
    kernel: usize,
}

impl ShortConv {
    pub fn load(
        src: &dyn WeightSource,
        cfg: &ModelConfig,
        prefix: &str,
        p: &Placement,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        if !h.is_multiple_of(p.world) {
            bail!(
                "tensor-parallel size {} must divide the hidden size {h}",
                p.world
            );
        }
        let kernel = cfg.short_conv_kernel;
        // `b`, `c`, `x` segments of the input projection, each sharded over the ranks
        let in_w =
            super::gdn::load_segments(src, &format!("{prefix}.in_proj.weight"), &[h, h, h], p)?;
        let f32p = Placement {
            dtype: DType::F32,
            ..p.clone()
        };
        let conv = super::gdn::load_segments(src, &format!("{prefix}.conv.weight"), &[h], &f32p)?;
        let channels = h / p.world;
        let conv = conv.reshape((channels, kernel))?;
        let taps = (0..kernel)
            .map(|j| {
                Ok(conv
                    .narrow(1, j, 1)?
                    .reshape((1, 1, channels))?
                    .contiguous()?)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            in_proj: QLinear::Dense(Linear::new(in_w, None)),
            out_proj: QLinear::Dense(load_row_parallel(src, &format!("{prefix}.out_proj"), p)?),
            taps,
            channels,
            kernel,
        })
    }

    /// A state pool of `slots` sequences: `[slots, (kernel - 1) * channels]`.
    pub fn new_state(
        cfg: &ModelConfig,
        world: usize,
        slots: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        Ok(Tensor::zeros(
            (slots, (cfg.short_conv_kernel - 1) * cfg.hidden_size / world),
            dtype,
            device,
        )?)
    }

    /// `x: [T, hidden]` -> `[T, hidden]` (partial sum under tensor parallelism).
    pub fn forward(&self, x: &Tensor, b: &DeviceBatch, pool: &Tensor) -> Result<Tensor> {
        let bcx = self.in_proj.forward(x)?;
        let c = self.channels;
        let gate_b = bcx.narrow(1, 0, c)?;
        let gate_c = bcx.narrow(1, c, c)?;
        let bx = (gate_b * bcx.narrow(1, 2 * c, c)?)?;
        let conv = self.conv(&bx, b, pool)?.to_dtype(x.dtype())?;
        self.out_proj.forward(&(gate_c * conv)?)
    }

    /// The causal conv of `bx [T, C]` over the batch's sequences, advancing their states.
    fn conv(&self, bx: &Tensor, b: &DeviceBatch, pool: &Tensor) -> Result<Tensor> {
        // multi-token decode steps: one token of every sequence per sub-step, in order
        let multi = b.per_substep(&[bx], |r, sub| {
            if let Some(prev) = &sub.prev_idx {
                copy_rows(pool, prev, &sub.state_idx)?;
            }
            let keep = keep_from_fresh(&sub.fresh)?;
            Ok(Some(self.step(&r[0], &sub.state_idx, &keep, pool)?))
        })?;
        if let Some(y) = multi {
            return Ok(y);
        }
        if b.seqs.iter().all(|s| s.q_len == 1) {
            let keep = match &b.fresh {
                // device flags: what a captured decode step reads
                Some(f) => keep_from_fresh(f)?,
                None => {
                    let k: Vec<f32> = b
                        .starts
                        .iter()
                        .map(|&s| if s { 0.0 } else { 1.0 })
                        .collect();
                    Tensor::from_vec(k, b.starts.len(), bx.device())?
                }
            };
            return self.step(bx, &b.state_idx, &keep, pool);
        }
        let mut outs = Vec::with_capacity(b.seqs.len());
        for (i, s) in b.seqs.iter().enumerate() {
            let idx = b.state_idx.narrow(0, i, 1)?;
            let x = bx
                .narrow(0, s.q_start, s.q_len)?
                .to_dtype(DType::F32)?
                .unsqueeze(0)?;
            let prev = if b.starts[i] {
                Tensor::zeros((1, self.kernel - 1, self.channels), DType::F32, bx.device())?
            } else {
                self.read_state(pool, &idx)?
            };
            let (y, state) = self.run(&prev, &x)?;
            write_rows(pool, &idx, &state.reshape((1, ()))?)?;
            outs.push(y.squeeze(0)?);
        }
        Ok(if outs.len() == 1 {
            outs.pop().unwrap()
        } else {
            Tensor::cat(&outs, 0)?
        })
    }

    /// One token per sequence: `bx [n, C]`, state slots `idx [n]`, `keep [n]` (0 = start fresh).
    fn step(&self, bx: &Tensor, idx: &Tensor, keep: &Tensor, pool: &Tensor) -> Result<Tensor> {
        let n = bx.dim(0)?;
        let prev = self
            .read_state(pool, idx)?
            .broadcast_mul(&keep.reshape((n, 1, 1))?)?;
        let (y, state) = self.run(&prev, &bx.to_dtype(DType::F32)?.unsqueeze(1)?)?;
        write_rows(pool, idx, &state.reshape((n, ()))?)?;
        Ok(y.squeeze(1)?)
    }

    /// States of slots `idx` as f32 `[n, kernel - 1, C]`.
    fn read_state(&self, pool: &Tensor, idx: &Tensor) -> Result<Tensor> {
        let n = idx.dim(0)?;
        Ok(pool.index_select(idx, 0)?.to_dtype(DType::F32)?.reshape((
            n,
            self.kernel - 1,
            self.channels,
        ))?)
    }

    /// Conv over `prev ++ x` (f32 `[B, kernel-1, C]`, `[B, T, C]`): the outputs `[B, T, C]` and
    /// the new state `[B, kernel-1, C]`.
    fn run(&self, prev: &Tensor, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let t = x.dim(1)?;
        let full = Tensor::cat(&[prev, x], 1)?;
        let mut acc = full.narrow(1, 0, t)?.broadcast_mul(&self.taps[0])?;
        for (j, tap) in self.taps.iter().enumerate().skip(1) {
            acc = (acc + full.narrow(1, j, t)?.broadcast_mul(tap)?)?;
        }
        Ok((acc, full.narrow(1, t, self.kernel - 1)?))
    }
}

/// `1 - fresh` as f32 (u32 flags, 1 = the sequence starts at position 0).
fn keep_from_fresh(fresh: &Tensor) -> Result<Tensor> {
    Ok((1.0 - fresh.to_dtype(DType::F32)?)?)
}

/// Copy pool rows `src` (u32 `[n]`) to rows `dst` (device-side, capturable).
pub fn copy_rows(pool: &Tensor, src: &Tensor, dst: &Tensor) -> Result<()> {
    write_rows(pool, dst, &pool.index_select(src, 0)?)
}

fn write_rows(dst: &Tensor, idx: &Tensor, rows: &Tensor) -> Result<()> {
    let (n, w) = rows.dims2()?;
    let idx = idx.reshape((n, 1))?.broadcast_as((n, w))?.contiguous()?;
    dst.scatter_set(&idx, &rows.to_dtype(dst.dtype())?.contiguous()?, 0)?;
    Ok(())
}
