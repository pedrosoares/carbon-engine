// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Qwen3.8-Flash-Next (`qwen4_exp`) pieces: hyper-connections and PLE n-gram embeddings.
//!
//! The residual stream is `hc_count` streams wide (`[T, hc * hidden]`). Before each mixer and
//! each MoE block a [`GatedResidual`] normalises the streams, mixes them into one `[T, hidden]`
//! input and later injects the block's output back into every stream with per-stream weights.
//!
//! PLE ("per-layer embeddings") adds hashed n-gram features on some linear-attention layers:
//! the previous `ngram_size - 1` tokens are hashed per head into large prime-sized tables, the
//! looked-up rows gate one value per stream, and a dilated depthwise conv adds local context.
//! The tables are tens of GiB, so rows are gathered from the memory-mapped checkpoint on the
//! host instead of being loaded.

use super::layers::{load_linear, Placement};
use crate::config::{ModelConfig, Qwen38Config};
use crate::weights::{RowReader, WeightSource};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::Linear;
use std::sync::Arc;

/// RMSNorm over groups of `group` channels with zero-centred weights (`x * (1 + w)`), in f32.
pub struct GroupedNorm {
    /// `1 + w`, f32, `[groups, group]`.
    weight: Tensor,
    group: usize,
    eps: f64,
}

impl GroupedNorm {
    pub fn load(src: &dyn WeightSource, name: &str, group: usize, cfg: &ModelConfig, p: &Placement) -> Result<Self> {
        let w = src.get(name, &p.device, DType::F32)?;
        let n = w.dim(0)?;
        if !n.is_multiple_of(group) {
            bail!("{name}: {n} channels are not groups of {group}");
        }
        Ok(Self {
            weight: (w + 1.0)?.reshape((n / group, group))?,
            group,
            eps: cfg.rms_norm_eps,
        })
    }

    /// `[T, groups * group]` -> same shape and dtype.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (t, c) = x.dims2()?;
        let g = c / self.group;
        let xf = x.to_dtype(DType::F32)?.reshape((t, g, self.group))?;
        let inv = (xf.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?.recip()?;
        Ok(xf
            .broadcast_mul(&inv)?
            .broadcast_mul(&self.weight)?
            .reshape((t, c))?
            .to_dtype(x.dtype())?)
    }
}

/// One hyper-connection: mixes the streams into a block input and (with `inject`) gives the
/// per-stream weights its output is added back with.
pub struct GatedResidual {
    norm: GroupedNorm,
    down: Linear,
    up: Linear,
    inject: Option<Linear>,
    hc: usize,
    hidden: usize,
}

impl GatedResidual {
    pub fn load(src: &dyn WeightSource, prefix: &str, cfg: &ModelConfig, p: &Placement, inject: bool) -> Result<Self> {
        let q = cfg.qwen38.as_ref().context("qwen4_exp config")?;
        Ok(Self {
            norm: GroupedNorm::load(src, &format!("{prefix}.hc_norm.weight"), cfg.hidden_size, cfg, p)?,
            down: load_linear(src, &format!("{prefix}.input_mix_weight_down"), p)?,
            up: load_linear(src, &format!("{prefix}.input_mix_weight_up"), p)?,
            inject: if inject {
                Some(load_linear(src, &format!("{prefix}.block_inject_weight"), p)?)
            } else {
                None
            },
            hc: q.hc_count,
            hidden: cfg.hidden_size,
        })
    }

    /// `x: [T, hc*hidden]` -> (block input `[T, hidden]`, injection weights `[T, hc]`).
    pub fn mix(&self, x: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let t = x.dim(0)?;
        let hc = self.hc as f64;
        let n = self.norm.forward(x)?;
        let w = candle_nn::ops::silu(&(self.down.forward(&n)? / hc)?)?;
        let w = candle_nn::ops::sigmoid(&self.up.forward(&w)?)?;
        // mean over the hc streams as hc - 1 slice additions in f32: a reduction over the
        // middle axis is a slow strided kernel in candle
        let wn = (w * &n)?.reshape((t, self.hc, self.hidden))?;
        let mut acc = wn.narrow(1, 0, 1)?.to_dtype(DType::F32)?;
        for i in 1..self.hc {
            acc = (acc + wn.narrow(1, i, 1)?.to_dtype(DType::F32)?)?;
        }
        let mixed = (acc.squeeze(1)? / hc)?.to_dtype(n.dtype())?;
        let inj = match &self.inject {
            Some(l) => Some((candle_nn::ops::sigmoid(&(l.forward(&n)? / hc)?)? * 2.0)?),
            None => None,
        };
        Ok((mixed, inj))
    }

    /// `x + y` injected into every stream: `x [T, hc*hidden]`, `y [T, hidden]`, `inj [T, hc]`.
    pub fn combine(&self, x: &Tensor, y: &Tensor, inj: &Tensor) -> Result<Tensor> {
        let t = x.dim(0)?;
        let y = y.to_dtype(x.dtype())?.unsqueeze(1)?;
        let add = y
            .broadcast_mul(&inj.to_dtype(x.dtype())?.unsqueeze(2)?)?
            .reshape((t, self.hc * self.hidden))?;
        Ok((x + add)?)
    }
}

// ------------------------------------------------------------------ n-gram hashing

const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

fn splitmix64(v: u64) -> u64 {
    let v = v.wrapping_add(GAMMA);
    let v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}

fn is_prime(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    if v.is_multiple_of(2) {
        return v == 2;
    }
    let mut d = 3;
    while d * d <= v {
        if v.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// HF `Qwen4ExpTextNGramEmbedding`'s ids: per PLE layer, multipliers and one prime-sized
/// vocabulary per head (orders 2..=ngram_size, `heads_per_ngram` heads each).
pub struct NgramHasher {
    ngram: usize,
    heads_per_ngram: usize,
    multipliers: Vec<u64>,
    head_sizes: Vec<u64>,
    head_offsets: Vec<u64>,
    /// Rows of the (padded) table.
    pub total_rows: u64,
    boundary: u32,
}

impl NgramHasher {
    pub fn new(q: &Qwen38Config, vocab: usize, ple_index: usize) -> Self {
        let heads = q.ngram_heads();
        // head g of this layer gets the (global + 1)-th prime >= base
        let mut p = q.ngram_vocab_size_base.saturating_sub(1);
        for _ in 0..ple_index * heads {
            p += 1;
            while !is_prime(p) {
                p += 1;
            }
        }
        let mut head_sizes = Vec::with_capacity(heads);
        let mut head_offsets = Vec::with_capacity(heads);
        let mut total = 0u64;
        for _ in 0..heads {
            p += 1;
            while !is_prime(p) {
                p += 1;
            }
            head_sizes.push(p);
            head_offsets.push(total);
            total += p;
        }
        let half_bound = ((i64::MAX as u64) / vocab.max(1) as u64 / 2).max(1);
        let base_seed = q.seed.wrapping_add(10007u64.wrapping_mul(ple_index as u64));
        let multipliers = (0..q.ngram_size as u64)
            .map(|i| 2 * (splitmix64(base_seed.wrapping_add(GAMMA.wrapping_mul(i + 1))) % half_bound) + 1)
            .collect();
        let d = q.ngram_vocab_divisor.max(1);
        Self {
            ngram: q.ngram_size,
            heads_per_ngram: q.heads_per_ngram,
            multipliers,
            head_sizes,
            head_offsets,
            total_rows: total.div_ceil(d) * d,
            boundary: q.ngram_boundary,
        }
    }

    /// Table rows `[tokens.len() * heads]` for the new `tokens` of one sequence, given the
    /// (up to `ngram - 1`) tokens right before them. n-gram windows never cross the boundary
    /// token; positions before the sequence start count as boundary tokens.
    pub fn rows(&self, prev: &[u32], tokens: &[u32], out: &mut Vec<u64>) {
        let ctx = self.ngram - 1;
        let prev = &prev[prev.len().saturating_sub(ctx)..];
        let hist: Vec<u32> = prev.iter().chain(tokens).copied().collect();
        let p0 = prev.len();
        let mut shifted = vec![0u64; self.ngram];
        for j in p0..hist.len() {
            for (s, slot) in shifted.iter_mut().enumerate() {
                // the token s back, unless the window [j-s, j) holds a boundary (or the start)
                let tok = if s > j || hist[j - s..j].contains(&self.boundary) {
                    self.boundary
                } else {
                    hist[j - s]
                };
                *slot = tok as u64;
            }
            for order in 2..=self.ngram {
                let mixed = shifted[..order]
                    .iter()
                    .zip(&self.multipliers)
                    .map(|(t, m)| t * m)
                    .fold(0, |a, x| a ^ x);
                let h0 = (order - 2) * self.heads_per_ngram;
                for h in h0..h0 + self.heads_per_ngram {
                    out.push(mixed % self.head_sizes[h] + self.head_offsets[h]);
                }
            }
        }
    }
}

// ------------------------------------------------------------------ PLE table

enum Part {
    Mapped(RowReader),
    Loaded(Tensor),
}

/// One PLE layer's n-gram table: `split_ngram_parts` row blocks, FP8 (with a scalar scale) or
/// float. Rows are read from the memory-mapped checkpoint when the source supports it.
pub struct PleTable {
    parts: Vec<Part>,
    rows_per_part: u64,
    cols: usize,
    scale: f32,
}

impl PleTable {
    pub fn open(src: &dyn WeightSource, stem: &str, q: &Qwen38Config, total_rows: u64) -> Result<Self> {
        let single = format!("{stem}.weight");
        let names: Vec<String> = if src.contains(&single) {
            vec![single]
        } else {
            (0..q.split_ngram_parts)
                .map(|i| format!("{stem}.shard_{i}.weight"))
                .collect()
        };
        let mut parts = Vec::with_capacity(names.len());
        let mut shape = None;
        for n in &names {
            if !src.contains(n) {
                bail!("PLE table part '{n}' is missing");
            }
            let part = match src.row_reader(n) {
                Some(r) => Part::Mapped(r?),
                None => Part::Loaded(src.load_raw(n)?),
            };
            let s = match &part {
                Part::Mapped(r) => (r.shape[0], r.shape[1]),
                Part::Loaded(t) => t.dims2()?,
            };
            if shape.is_some_and(|x| x != s) {
                bail!("PLE table parts of {stem} differ in shape");
            }
            shape = Some(s);
            parts.push(part);
        }
        let (rows, cols) = shape.context("empty PLE table")?;
        if (rows * parts.len()) as u64 != total_rows {
            bail!("{stem}: {} rows, the n-gram hash needs {total_rows}", rows * parts.len());
        }
        if cols != q.ngram_head_dim() {
            bail!("{stem}: rows are {cols} wide, expected {}", q.ngram_head_dim());
        }
        let scale_name = format!("{stem}.weight_scale");
        let scale = if src.contains(&scale_name) {
            src.load_raw(&scale_name)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?[0]
        } else {
            1.0
        };
        Ok(Self {
            parts,
            rows_per_part: rows as u64,
            cols,
            scale,
        })
    }

    /// Start reading the mapped rows `ids` from disk (returns before they arrive).
    pub fn prefetch(&self, ids: &[u64]) {
        for &id in ids {
            let (p, r) = ((id / self.rows_per_part) as usize, (id % self.rows_per_part) as usize);
            if let Some(Part::Mapped(m)) = self.parts.get(p) {
                m.prefetch_row(r);
            }
        }
    }

    /// The rows `ids` as f32 `[ids.len(), cols]` on the host (dequantized). Mapped rows are read
    /// in parallel: a cold row is a random read from disk.
    pub fn gather(&self, ids: &[u64]) -> Result<Tensor> {
        use rayon::prelude::*;
        // all reads requested up front: the disk serves them at queue depth (~5x faster on NVMe
        // than one fault per worker thread at a time)
        if ids.len() > 1 {
            ids.par_chunks(1024).for_each(|c| self.prefetch(c));
        }
        let mut out = vec![0f32; ids.len() * self.cols];
        let row = |id: u64, dst: &mut [f32]| -> Result<()> {
            let (p, r) = ((id / self.rows_per_part) as usize, (id % self.rows_per_part) as usize);
            match self.parts.get(p).context("PLE row out of range")? {
                Part::Mapped(m) => m.row_into(r, dst),
                Part::Loaded(t) => {
                    dst.copy_from_slice(&t.get(r)?.to_dtype(DType::F32)?.to_vec1::<f32>()?);
                    Ok(())
                }
            }
        };
        // a cold row is a page fault on a random disk block: issue them all at once so the
        // reads overlap (a decode step needs one row per n-gram head, all independent)
        if ids.len() > 1 {
            ids.par_iter()
                .zip(out.par_chunks_mut(self.cols))
                .try_for_each(|(&id, dst)| row(id, dst))?;
        } else {
            for (&id, dst) in ids.iter().zip(out.chunks_mut(self.cols)) {
                row(id, dst)?;
            }
        }
        let t = Tensor::from_vec(out, (ids.len(), self.cols), &Device::Cpu)?;
        Ok(if self.scale != 1.0 {
            (t * self.scale as f64)?
        } else {
            t
        })
    }
}

/// Reads one PLE layer's table rows for a prompt ahead of its prefill (page cache only).
#[derive(Clone)]
pub struct PlePrefetch {
    hasher: Arc<NgramHasher>,
    table: Arc<PleTable>,
}

impl PlePrefetch {
    /// Request the rows of `tokens` (following `prev`) from disk.
    pub fn run(&self, prev: &[u32], tokens: &[u32]) {
        let mut ids = Vec::with_capacity(tokens.len() * self.hasher.head_sizes.len());
        self.hasher.rows(prev, tokens, &mut ids);
        self.table.prefetch(&ids);
    }
}

/// PLE on one layer.
pub struct Ple {
    hasher: Arc<NgramHasher>,
    table: Arc<PleTable>,
    key: Linear,
    value: Linear,
    norm_key: GroupedNorm,
    norm_query: GroupedNorm,
    norm_conv: GroupedNorm,
    /// Depthwise conv taps `[kernel, channels]`, model dtype.
    conv: Tensor,
    hc: usize,
    hidden: usize,
    dilation: usize,
}

impl Ple {
    pub fn load(src: &dyn WeightSource, prefix: &str, cfg: &ModelConfig, ple_index: usize, p: &Placement) -> Result<Self> {
        let q = cfg.qwen38.as_ref().context("qwen4_exp config")?;
        let hasher = NgramHasher::new(q, cfg.vocab_size, ple_index);
        let table = PleTable::open(src, &format!("{prefix}.ple_embedding.ngram_embedding"), q, hasher.total_rows)?;
        let h = cfg.hidden_size;
        let w = src.get(&format!("{prefix}.conv1d.weight"), &p.device, p.dtype)?; // [C, 1, K]
        let (c, _, k) = w.dims3()?;
        if c != q.hc_count * h || k != q.ple_conv_kernel {
            bail!("{prefix}.conv1d: shape {:?}", w.dims());
        }
        Ok(Self {
            hasher: Arc::new(hasher),
            table: Arc::new(table),
            key: load_linear(src, &format!("{prefix}.key_proj"), p)?,
            value: load_linear(src, &format!("{prefix}.value_proj"), p)?,
            norm_key: GroupedNorm::load(src, &format!("{prefix}.norm_key.weight"), h, cfg, p)?,
            norm_query: GroupedNorm::load(src, &format!("{prefix}.norm_query.weight"), h, cfg, p)?,
            norm_conv: GroupedNorm::load(src, &format!("{prefix}.norm_conv.weight"), h, cfg, p)?,
            conv: w.squeeze(1)?.t()?.contiguous()?,
            hc: q.hc_count,
            hidden: h,
            dilation: q.ngram_size,
        })
    }

    /// A handle that reads this layer's rows for tokens ahead of time.
    pub fn prefetcher(&self) -> PlePrefetch {
        PlePrefetch { hasher: self.hasher.clone(), table: self.table.clone() }
    }

    /// Conv state per sequence: `[(kernel - 1) * dilation, hc*hidden]`.
    pub fn state_shape(q: &Qwen38Config, hidden: usize) -> (usize, usize) {
        (q.ple_conv_state_len(), q.hc_count * hidden)
    }

    /// `x [T, hc*hidden]` -> `x + ple(x)`; `state [slots, S, C]` holds each sequence's conv input
    /// history (read at `state_slot`, written back).
    /// The batch's n-gram embeddings `[T, ple_dim]` on `device` (hashing and the table gather
    /// run on the host; done before the layers so a CUDA graph only reads the result).
    pub fn embed(&self, b: &super::attention::DeviceBatch, device: &Device, dtype: DType) -> Result<Tensor> {
        let t = b.num_tokens;
        let mut ids = Vec::with_capacity(t * self.hasher.head_sizes.len());
        for s in &b.seqs {
            self.hasher.rows(&s.prev_tokens, &b.tokens[s.q_start..s.q_start + s.q_len], &mut ids);
        }
        if ids.len() != t * self.hasher.head_sizes.len() {
            bail!("PLE needs the batch's token ids");
        }
        Ok(self
            .table
            .gather(&ids)?
            .reshape((t, ()))?
            .to_dtype(dtype)?
            .to_device(device)?)
    }

    /// `x [T, hc*hidden]` -> `x + ple(x)`, given this layer's embeddings from [`Self::embed`];
    /// `state [slots, S, C]` holds each sequence's conv input history.
    pub fn forward(&self, x: &Tensor, emb: &Tensor, b: &super::attention::DeviceBatch, state: &Tensor) -> Result<Tensor> {
        let (t, c) = x.dims2()?;
        let emb = emb.to_dtype(x.dtype())?;
        let key = self.norm_key.forward(&self.key.forward(&emb)?)?.reshape((t, self.hc, self.hidden))?;
        let value = self.value.forward(&emb)?;
        let query = self.norm_query.forward(x)?.reshape((t, self.hc, self.hidden))?;
        let gate = ((key * query)?.sum(2)? / (self.hidden as f64).sqrt())?; // [T, hc]
        // sign(g) * sqrt(max(|g|, 1e-6))
        let gate = (gate.abs()?.maximum(1e-6)?.sqrt()? * gate.sign()?)?;
        let gated = candle_nn::ops::sigmoid(&gate)?
            .unsqueeze(2)?
            .broadcast_mul(&value.unsqueeze(1)?)?
            .reshape((t, c))?;
        let normed = self.norm_conv.forward(&gated)?;
        let multi = b.per_substep(&[&normed], |r, sub| {
            let read = sub.prev_idx.as_ref().unwrap_or(&sub.state_idx);
            Ok(Some(self.conv_decode(&r[0], read, &sub.state_idx, &sub.fresh, state)?))
        })?;
        let conv = match (&b.fresh, multi) {
            (_, Some(c)) => c,
            // decode on CUDA: one batched, branch-free step (CUDA-graph safe)
            (Some(fresh), None) if b.seqs.iter().all(|s| s.q_len == 1) => {
                self.conv_decode(&normed, &b.state_idx, &b.state_idx, fresh, state)?
            }
            _ => self.conv_seqs(&normed, b, state)?,
        };
        Ok((x + (gated + candle_nn::ops::silu(&conv)?)?)?)
    }

    /// One token per sequence: `x [B, C]`; `fresh [B]` u32 flags restart a sequence's history.
    /// One token per row: `x [B, C]`; each row's history is read from slot `read_idx` and the
    /// updated one written to `state_idx` (u32), `fresh` rows start from zero.
    fn conv_decode(&self, x: &Tensor, read_idx: &Tensor, state_idx: &Tensor, fresh: &Tensor, state: &Tensor) -> Result<Tensor> {
        let (_, sl, c) = state.dims3()?;
        let bsz = x.dim(0)?;
        let k = self.conv.dim(0)?;
        let keep = (1.0 - fresh.to_dtype(DType::F32)?)?.reshape((bsz, 1, 1))?;
        let prev = state
            .index_select(read_idx, 0)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&keep)?; // [B, S, C]
        let full = Tensor::cat(&[prev, x.to_dtype(DType::F32)?.unsqueeze(1)?], 1)?; // [B, S+1, C]
        let mut y = Tensor::zeros((bsz, c), DType::F32, x.device())?;
        for j in 0..k {
            let tap = self.conv.narrow(0, j, 1)?.to_dtype(DType::F32)?;
            y = (y + full.narrow(1, j * self.dilation, 1)?.squeeze(1)?.broadcast_mul(&tap)?)?;
        }
        let tail = full.narrow(1, 1, sl)?.reshape((bsz, sl * c))?.to_dtype(state.dtype())?.contiguous()?;
        let flat = state.reshape((state.dim(0)?, sl * c))?;
        let idx = state_idx.reshape((bsz, 1))?.broadcast_as((bsz, sl * c))?.contiguous()?;
        flat.scatter_set(&idx, &tail, 0)?;
        Ok(y.to_dtype(x.dtype())?)
    }

    /// Dilated depthwise causal conv over each sequence, continuing from its state.
    fn conv_seqs(&self, x: &Tensor, b: &super::attention::DeviceBatch, state: &Tensor) -> Result<Tensor> {
        let (_, sl, c) = state.dims3()?;
        let k = self.conv.dim(0)?;
        let mut outs = Vec::with_capacity(b.seqs.len());
        for (i, s) in b.seqs.iter().enumerate() {
            let slot = b.state_idx.narrow(0, i, 1)?;
            let prev = if b.starts[i] {
                Tensor::zeros((sl, c), x.dtype(), x.device())?
            } else {
                state.index_select(&slot, 0)?.squeeze(0)?.to_dtype(x.dtype())?
            };
            let cur = x.narrow(0, s.q_start, s.q_len)?;
            let full = Tensor::cat(&[prev, cur], 0)?; // [S + L, C]
            let mut y = Tensor::zeros((s.q_len, c), DType::F32, x.device())?;
            for j in 0..k {
                let tap = self.conv.narrow(0, j, 1)?.to_dtype(DType::F32)?;
                let xs = full.narrow(0, j * self.dilation, s.q_len)?.to_dtype(DType::F32)?;
                y = (y + xs.broadcast_mul(&tap)?)?;
            }
            outs.push(y.to_dtype(x.dtype())?);
            let tail = full.narrow(0, full.dim(0)? - sl, sl)?;
            write_state(state, &slot, &tail)?;
        }
        Ok(if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 0)? })
    }
}

fn write_state(state: &Tensor, slot: &Tensor, rows: &Tensor) -> Result<()> {
    let (_, sl, c) = state.dims3()?;
    let flat = state.reshape((state.dim(0)?, sl * c))?;
    let idx = slot.reshape((1, 1))?.broadcast_as((1, sl * c))?.contiguous()?;
    flat.scatter_set(&idx, &rows.to_dtype(state.dtype())?.reshape((1, sl * c))?.contiguous()?, 0)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Qwen38Config {
        Qwen38Config {
            hc_count: 3,
            hc_lowrank: 4,
            ple_layers: vec![0, 2],
            ple_embed_dim: 16,
            ple_conv_kernel: 3,
            ngram_size: 3,
            heads_per_ngram: 2,
            ngram_vocab_size_base: 97,
            ngram_vocab_divisor: 8,
            split_ngram_parts: 4,
            seed: 1234,
            ngram_boundary: 258,
            index_heads: 2,
            index_head_dim: 16,
            index_budget: 16,
            index_ratio: 4,
        }
    }

    /// Values from HF's `_build_layer_multipliers` / `_find_nth_prime_after` (vocab 263).
    #[test]
    fn hasher_matches_hf_tables() {
        let h0 = NgramHasher::new(&cfg(), 263, 0);
        let h1 = NgramHasher::new(&cfg(), 263, 1);
        assert_eq!(h0.head_sizes, vec![97, 101, 103, 107]);
        assert_eq!(h1.head_sizes, vec![109, 113, 127, 131]);
        assert_eq!(h0.total_rows, 408);
        assert_eq!(h0.head_offsets, vec![0, 97, 198, 301]);
        assert!(h0.multipliers.iter().all(|m| m % 2 == 1));
        assert_ne!(h0.multipliers, h1.multipliers);
    }

    #[test]
    fn boundary_and_start_are_eos() {
        let h = NgramHasher::new(&cfg(), 263, 0);
        // a chunk split anywhere gives the same rows as the whole sequence
        let toks = [5u32, 7, 258, 9, 11, 13];
        let mut whole = vec![];
        h.rows(&[], &toks, &mut whole);
        for cut in 1..toks.len() {
            let mut a = vec![];
            h.rows(&[], &toks[..cut], &mut a);
            h.rows(&toks[..cut], &toks[cut..], &mut a);
            assert_eq!(a, whole, "cut {cut}");
        }
        // right after the boundary the window restarts: same rows as at the sequence start
        let mut fresh = vec![];
        h.rows(&[], &[9, 11], &mut fresh);
        assert_eq!(&whole[3 * 4..5 * 4], &fresh[..]);
    }
}
