// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Paged attention over the token-granular KV pool.
//!
//! New K/V rows are scattered into the pool at `out_loc`; each sequence then
//! gathers its full history through its slot table. Prefill runs per sequence
//! with query tiling (bounded score memory); decode batches all sequences into
//! a single padded gather when that fits, falling back to per-sequence work.

use super::layers::{load_column_q, load_row_q, Placement, QLinear, RmsNorm, Rotary};
use crate::batch::BatchMeta;
use crate::config::ModelConfig;
use crate::kvcache::LayerKv;
use crate::weights::WeightSource;
use anyhow::{bail, Result};
use candle_core::{DType, Device, Module, Tensor, D};

const MASK_VALUE: f32 = -1e30;
/// Max score elements materialised at once per query tile.
const SCORE_BUDGET: usize = 1 << 26;
/// Max gathered K+V elements for the batched decode path.
const DECODE_GATHER_BUDGET: usize = 1 << 28;

static FUSED_DECODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// carbon's CUDA attention kernels (decode and prefill) are on unless `CARBON_FUSED_ATTN=0`
/// or one failed once.
pub(crate) fn fused_decode_enabled() -> bool {
    use std::sync::atomic::Ordering;
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var("CARBON_FUSED_ATTN").map_or(true, |v| v != "0"))
        && FUSED_DECODE.load(Ordering::Relaxed)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn disable_fused_decode(e: &anyhow::Error) {
    if FUSED_DECODE.swap(false, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!("fused attention kernels disabled, using the generic path: {e:#}");
    }
}

pub struct SeqCtx {
    pub q_start: usize,
    pub q_len: usize,
    pub kv_len: usize,
    /// Tokens right before the new ones (PLE n-gram context).
    pub prev_tokens: Vec<u32>,
    /// Host slot table; its device copy is uploaded on first use by the generic path.
    slots: std::sync::Arc<Vec<u32>>,
    kv_idx: std::sync::OnceLock<Tensor>,
}

impl SeqCtx {
    /// Device slot table `[kv_len]` (u32) for the generic gather path.
    pub fn kv_idx(&self, device: &Device) -> Result<&Tensor> {
        if let Some(t) = self.kv_idx.get() {
            return Ok(t);
        }
        let t = Tensor::from_slice(&self.slots[..self.kv_len], self.kv_len, device)?;
        let _ = self.kv_idx.set(t);
        Ok(self.kv_idx.get().expect("just set"))
    }
}

/// Slot tables for carbon's fused CUDA decode kernel (all sequences have one query token).
pub struct PagedCtx {
    /// Concatenated slot tables, u32.
    pub slots: Tensor,
    /// `[B + 1]` offsets into `slots`, u32.
    pub starts: Tensor,
}

/// Tables for carbon's CUDA prefill attention kernel (batches with any multi-token sequence).
pub struct PrefillCtx {
    /// Concatenated slot tables, u32.
    pub slots: Tensor,
    /// `[B * 4]`: q_start, q_len, slot_start, kv_len per sequence, u32.
    pub seqinfo: Tensor,
    /// `[n * 2]`: sequence, first query of each tile of `PREFILL_TILE` queries, u32.
    pub tiles: Tensor,
}

pub struct DecodeCtx {
    pub kv_idx: Tensor,
    pub mask: Tensor,
    pub lmax: usize,
}

/// Most new tokens per sequence a step may have and still run on the fused decode kernels
/// (one decode row per token): MTP verify steps and short prefill chunks.
pub const MULTI_DECODE_MAX: usize = 8;

/// The j-th new token of every sequence that has one (multi-token decode steps): recurrent
/// layers (GDN, PLE conv) advance one token per sub-step, in order.
pub struct SubStep {
    /// Token rows of the batch, u32 `[n]`.
    pub rows: Tensor,
    /// Recurrent-state slot and fresh flag (j == 0 of a sequence starting at position 0), u32.
    pub state_idx: Tensor,
    pub fresh: Tensor,
    /// j > 0: the slot token j - 1 left its state in (u32 `[n]`); layers start token j from it,
    /// copying it into `state_idx` first where the two differ (`SeqMeta::token_state_slots`).
    pub prev_idx: Option<Tensor>,
}

/// Per-device tensors describing a batch, built once and shared by all layers.
pub struct DeviceBatch {
    pub num_tokens: usize,
    pub cos: Tensor,
    pub sin: Tensor,
    pub scatter_idx: Tensor,
    pub seqs: Vec<SeqCtx>,
    pub decode: Option<DecodeCtx>,
    pub paged: Option<PagedCtx>,
    pub prefill: Option<PrefillCtx>,
    /// Recurrent-state slot of each sequence (linear-attention layers).
    pub state_idx: Tensor,
    /// Per sequence: the forward starts at position 0, so its recurrent state starts from zero.
    pub starts: Vec<bool>,
    /// `starts` as u32 flags on the device, for fused decode kernels (CUDA decode steps only).
    pub fresh: Option<Tensor>,
    /// Token ids of the new positions (when the batch carries them).
    pub tokens: Vec<u32>,
    /// The stage's rope tables (QSA rotates pooled block keys at their own positions).
    pub rotary: Rotary,
    /// `BatchMeta::mrope_layout`: per sequence, where QSA block keys rotate when it has images.
    pub mrope_layout: Vec<crate::images::MropePositions>,
    /// Qwen3.8 PLE n-gram embeddings `[T, ple_dim]`, one per PLE layer of the stage.
    pub ple_emb: Vec<Tensor>,
    /// Multi-token decode steps (some sequence has 2..=MULTI_DECODE_MAX new tokens, on the
    /// fused CUDA kernels): the sub-steps and, u32 `[T]`, where each token's row lands when the
    /// sub-steps' outputs are concatenated (to put them back in token order).
    pub substeps: Option<(Vec<SubStep>, Tensor)>,
}

impl DeviceBatch {
    /// Run `f` once per sub-step on that sub-step's rows of `x` and return the outputs in token
    /// order; `None` when `f` declines the first sub-step (a later refusal is an error: the
    /// recurrent state has already advanced).
    pub fn per_substep(
        &self,
        x: &[&Tensor],
        mut f: impl FnMut(&[Tensor], &SubStep) -> Result<Option<Tensor>>,
    ) -> Result<Option<Tensor>> {
        let Some((steps, order)) = &self.substeps else {
            return Ok(None);
        };
        let mut outs = Vec::with_capacity(steps.len());
        for (j, st) in steps.iter().enumerate() {
            let rows: Vec<Tensor> = x.iter().map(|t| t.index_select(&st.rows, 0)).collect::<candle_core::Result<_>>()?;
            match f(&rows, st)? {
                Some(y) => outs.push(y),
                None if j == 0 => return Ok(None),
                None => bail!("a fused kernel fell back in the middle of a multi-token step"),
            }
        }
        Ok(Some(Tensor::cat(&outs, 0)?.index_select(order, 0)?))
    }
}

impl DeviceBatch {
    /// A copy whose tensors a CUDA graph can capture: fused-decode tables only, the slot table
    /// sized to `slot_capacity` (the KV pool) so later steps fit. Refresh with [`Self::copy_from`].
    pub fn graph_static(&self, slot_capacity: usize) -> Result<Self> {
        let Some(pc) = &self.paged else {
            bail!("batch has no fused decode tables");
        };
        if self.fresh.is_none() && self.substeps.is_none() {
            bail!("batch has no fused decode tables");
        }
        let n = pc.slots.dim(0)?;
        let slots = Tensor::zeros(slot_capacity.max(n).max(1), DType::U32, pc.slots.device())?;
        if n > 0 {
            slots.slice_set(&pc.slots, 0, 0)?;
        }
        Ok(Self {
            num_tokens: self.num_tokens,
            cos: self.cos.copy()?,
            sin: self.sin.copy()?,
            scatter_idx: self.scatter_idx.copy()?,
            seqs: self
                .seqs
                .iter()
                .map(|s| SeqCtx {
                    q_start: s.q_start,
                    q_len: s.q_len,
                    kv_len: s.kv_len,
                    prev_tokens: s.prev_tokens.clone(),
                    slots: s.slots.clone(),
                    kv_idx: std::sync::OnceLock::new(),
                })
                .collect(),
            decode: None,
            paged: Some(PagedCtx {
                slots,
                starts: pc.starts.copy()?,
            }),
            prefill: None,
            state_idx: self.state_idx.copy()?,
            starts: self.starts.clone(),
            fresh: self.fresh.as_ref().map(|f| f.copy()).transpose()?,
            tokens: self.tokens.clone(),
            rotary: self.rotary.clone(),
            mrope_layout: self.mrope_layout.clone(),
            ple_emb: self.ple_emb.iter().map(|t| t.copy()).collect::<candle_core::Result<_>>()?,
            substeps: match &self.substeps {
                Some((steps, order)) => Some((
                    steps
                        .iter()
                        .map(|s| {
                            Ok(SubStep {
                                rows: s.rows.copy()?,
                                state_idx: s.state_idx.copy()?,
                                fresh: s.fresh.copy()?,
                                prev_idx: s.prev_idx.as_ref().map(|p| p.copy()).transpose()?,
                            })
                        })
                        .collect::<Result<_>>()?,
                    order.copy()?,
                )),
                None => None,
            },
        })
    }

    /// Whether `o` (same batch size) fits this static batch's slot table.
    pub fn graph_fits(&self, o: &DeviceBatch) -> bool {
        match (&self.paged, &o.paged) {
            (Some(a), Some(b)) => {
                o.num_tokens == self.num_tokens
                    && b.slots.dim(0).unwrap_or(usize::MAX) <= a.slots.dim(0).unwrap_or(0)
            }
            _ => false,
        }
    }

    /// Overwrite this static batch's device tables in place with `o`'s (same batch size).
    pub fn copy_from(&self, o: &DeviceBatch) -> Result<()> {
        let (Some(pc), Some(opc)) = (&self.paged, &o.paged) else {
            bail!("batch has no fused decode tables");
        };
        match (&self.fresh, &o.fresh) {
            (Some(a), Some(b)) => a.slice_set(b, 0, 0)?,
            (None, None) => {}
            _ => bail!("batch does not fit the captured graph"),
        }
        match (&self.substeps, &o.substeps) {
            (Some((a, ao)), Some((b, bo))) if a.len() == b.len() => {
                for (x, y) in a.iter().zip(b) {
                    if x.rows.dims() != y.rows.dims() {
                        bail!("batch does not fit the captured graph");
                    }
                    x.rows.slice_set(&y.rows, 0, 0)?;
                    x.state_idx.slice_set(&y.state_idx, 0, 0)?;
                    x.fresh.slice_set(&y.fresh, 0, 0)?;
                    match (&x.prev_idx, &y.prev_idx) {
                        (Some(a), Some(b)) => a.slice_set(b, 0, 0)?,
                        (None, None) => {}
                        _ => bail!("batch does not fit the captured graph"),
                    }
                }
                ao.slice_set(bo, 0, 0)?;
            }
            (None, None) => {}
            _ => bail!("batch does not fit the captured graph"),
        }
        let n = opc.slots.dim(0)?;
        if o.num_tokens != self.num_tokens || n > pc.slots.dim(0)? {
            bail!("batch does not fit the captured graph");
        }
        self.cos.slice_set(&o.cos, 0, 0)?;
        self.sin.slice_set(&o.sin, 0, 0)?;
        self.scatter_idx.slice_set(&o.scatter_idx, 0, 0)?;
        self.state_idx.slice_set(&o.state_idx, 0, 0)?;
        for (a, b) in self.ple_emb.iter().zip(&o.ple_emb) {
            a.slice_set(b, 0, 0)?;
        }
        pc.starts.slice_set(&opc.starts, 0, 0)?;
        if n > 0 {
            pc.slots.slice_set(&opc.slots, 0, 0)?;
        }
        Ok(())
    }

    pub fn build(
        meta: &BatchMeta,
        rotary: &Rotary,
        kv_width: usize,
        device: &Device,
    ) -> Result<Self> {
        let t = meta.num_tokens();
        let positions = Tensor::from_slice(&meta.positions, t, device)?;
        let max_pos = meta.positions.iter().copied().max().unwrap_or(0);
        let (cos, sin) = if meta.mrope.is_empty() {
            rotary.select(&positions, max_pos)?
        } else {
            rotary.select_mrope(&meta.mrope, device)?
        };
        let scatter_idx = Tensor::from_slice(&meta.out_loc, (t, 1), device)?
            .broadcast_as((t, kv_width))?
            .contiguous()?;

        let b = meta.seqs.len();
        let lmax = meta.seqs.iter().map(|s| s.kv_len).max().unwrap_or(0);
        let all_decode = meta.seqs.iter().all(|s| s.q_len == 1);
        // carbon's CUDA kernels read the slot tables directly; the generic tables are then only
        // built if a layer falls back (lazily, per sequence)
        let fused = device.is_cuda() && fused_decode_enabled();
        let batched_decode =
            !fused && all_decode && b > 1 && b * lmax * kv_width * 2 <= DECODE_GATHER_BUDGET;
        let mut seqs = Vec::with_capacity(b);
        let mut q_start = 0;
        for s in &meta.seqs {
            let seq = SeqCtx {
                q_start,
                q_len: s.q_len,
                kv_len: s.kv_len,
                prev_tokens: s.prev_tokens.clone(),
                slots: s.slots.clone(),
                kv_idx: std::sync::OnceLock::new(),
            };
            if !batched_decode && !fused {
                seq.kv_idx(device)?;
            }
            seqs.push(seq);
            q_start += s.q_len;
        }
        let decode = if batched_decode {
            let mut idx = Vec::with_capacity(b * lmax);
            let mut mask = Vec::with_capacity(b * lmax);
            for s in &meta.seqs {
                idx.extend_from_slice(&s.slots[..s.kv_len]);
                idx.extend(std::iter::repeat_n(s.slots[0], lmax - s.kv_len));
                mask.extend(std::iter::repeat_n(0f32, s.kv_len));
                mask.extend(std::iter::repeat_n(MASK_VALUE, lmax - s.kv_len));
            }
            Some(DecodeCtx {
                kv_idx: Tensor::from_vec(idx, b * lmax, device)?,
                mask: Tensor::from_vec(mask, (b, 1, 1, lmax), device)?,
                lmax,
            })
        } else {
            None
        };
        // one decode row per new token: token j of a sequence sees its causal prefix
        let multi = !all_decode
            && device.is_cuda()
            && fused_decode_enabled()
            && meta.seqs.iter().all(|s| s.q_len <= MULTI_DECODE_MAX);
        let paged = if device.is_cuda() && fused_decode_enabled() && (all_decode || multi) {
            let mut slots = Vec::with_capacity(meta.seqs.iter().map(|s| s.kv_len * s.q_len).sum());
            let mut starts = Vec::with_capacity(t + 1);
            starts.push(0u32);
            for s in &meta.seqs {
                for j in 0..s.q_len {
                    slots.extend_from_slice(&s.slots[..s.kv_len - s.q_len + j + 1]);
                    starts.push(slots.len() as u32);
                }
            }
            Some(PagedCtx {
                slots: Tensor::from_vec(slots.clone(), slots.len(), device)?,
                starts: Tensor::from_vec(starts, t + 1, device)?,
            })
        } else {
            None
        };
        if !multi && meta.seqs.iter().any(|s| !s.token_state_slots.is_empty()) {
            bail!("per-token state slots need the fused multi-token decode path");
        }
        let substeps = if multi {
            let kmax = meta.seqs.iter().map(|s| s.q_len).max().unwrap_or(0);
            let mut steps = Vec::with_capacity(kmax);
            let mut order = vec![0u32; t];
            let mut placed = 0u32;
            for j in 0..kmax {
                let (mut rows, mut st, mut fr, mut prev) = (vec![], vec![], vec![], vec![]);
                let mut q0 = 0usize;
                for s in &meta.seqs {
                    if j < s.q_len {
                        let slot = |j: usize| s.token_state_slots.get(j).copied().unwrap_or(s.state_slot);
                        rows.push((q0 + j) as u32);
                        st.push(slot(j));
                        prev.push(slot(j.saturating_sub(1)));
                        fr.push(u32::from(j == 0 && s.kv_len == s.q_len));
                        order[q0 + j] = placed;
                        placed += 1;
                    }
                    q0 += s.q_len;
                }
                let n = rows.len();
                steps.push(SubStep {
                    rows: Tensor::from_vec(rows, n, device)?,
                    state_idx: Tensor::from_vec(st, n, device)?,
                    fresh: Tensor::from_vec(fr, n, device)?,
                    // always present for j > 0 (a captured graph needs the same shape every step)
                    prev_idx: (j > 0).then(|| Tensor::from_vec(prev, n, device)).transpose()?,
                });
            }
            Some((steps, Tensor::from_vec(order, t, device)?))
        } else {
            None
        };
        let prefill = if device.is_cuda()
            && fused_decode_enabled()
            && !all_decode
            && !multi
        {
            const TILE: usize = 64; // kernels::cuda::PREFILL_TILE
            let mut slots = Vec::new();
            let mut seqinfo = Vec::with_capacity(b * 4);
            let mut tiles = Vec::new();
            let mut q_start = 0u32;
            for (i, s) in meta.seqs.iter().enumerate() {
                seqinfo.extend_from_slice(&[
                    q_start,
                    s.q_len as u32,
                    slots.len() as u32,
                    s.kv_len as u32,
                ]);
                slots.extend_from_slice(&s.slots[..s.kv_len]);
                for q0 in (0..s.q_len).step_by(TILE) {
                    tiles.extend_from_slice(&[i as u32, q0 as u32]);
                }
                q_start += s.q_len as u32;
            }
            let n = slots.len();
            let nt = tiles.len();
            Some(PrefillCtx {
                slots: Tensor::from_vec(slots, n, device)?,
                seqinfo: Tensor::from_vec(seqinfo, b * 4, device)?,
                tiles: Tensor::from_vec(tiles, nt, device)?,
            })
        } else {
            None
        };
        // decode batches (also on the CPU: the batched PLE conv reads it)
        let fresh = if meta.seqs.iter().all(|s| s.q_len == 1) {
            let f: Vec<u32> = meta
                .seqs
                .iter()
                .map(|s| u32::from(s.kv_len == s.q_len))
                .collect();
            Some(Tensor::from_vec(f, b, device)?)
        } else {
            None
        };
        let state_slots: Vec<u32> = meta.seqs.iter().map(|s| s.state_slot).collect();
        Ok(Self {
            num_tokens: t,
            cos,
            sin,
            scatter_idx,
            seqs,
            decode,
            paged,
            prefill,
            state_idx: Tensor::from_vec(state_slots, b, device)?,
            starts: meta.seqs.iter().map(|s| s.kv_len == s.q_len).collect(),
            fresh,
            tokens: meta.tokens.clone(),
            rotary: rotary.clone(),
            mrope_layout: meta.mrope_layout.clone(),
            ple_emb: vec![],
            substeps,
        })
    }
}

pub struct Attention {
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f64,
    /// q_proj emits `[q, gate]` per head; the output is multiplied by `sigmoid(gate)`.
    output_gate: bool,
    /// QSA (Qwen3.8): attend only to the blocks an indexer selects.
    indexer: Option<Indexer>,
}

/// What a sequence's queries may attend to under QSA.
enum Selection {
    /// Every token (no more complete blocks than the budget).
    All,
    /// Multi-token chunks: per query, its selected blocks (`None`: every earlier token).
    Blocks(QsaRows),
    /// The KV slots of the selected tokens (single-query decode: attend over these only).
    Slots(Vec<u32>),
}

/// A chunk's QSA selection, turned into mask rows one attention tile at a time.
pub struct QsaRows {
    blocks: Vec<Option<Vec<u32>>>,
    /// CUDA: the selection as `[q_len, ceil(blocks / 32)]` u32 bitmasks on the device
    /// (`blocks` is then empty).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    bits: Option<Tensor>,
    ratio: usize,
    /// Position of the chunk's first query.
    first: usize,
}

impl QsaRows {
    /// Additive mask rows `[n, kv_len]` (f32) for queries `a..a+n`.
    fn mask(&self, a: usize, n: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
        let r = self.ratio;
        #[cfg(feature = "cuda")]
        if let Some(bits) = &self.bits {
            let rows = bits.narrow(0, a, n)?;
            return Ok(crate::kernels::cuda::qsa_bits_mask(&rows, kv_len, r, self.first + a, MASK_VALUE)?);
        }
        let mut m = vec![MASK_VALUE; n * kv_len];
        for i in 0..n {
            let pos = self.first + a + i;
            let row = &mut m[i * kv_len..(i + 1) * kv_len];
            match &self.blocks[a + i] {
                None => row[..=pos].fill(0.0),
                Some(bl) => {
                    for &blk in bl {
                        let b = blk as usize;
                        row[b * r..(b + 1) * r].fill(0.0);
                    }
                    row[(pos + 1) / r * r..=pos].fill(0.0);
                }
            }
        }
        Ok(Tensor::from_vec(m, (n, kv_len), device)?)
    }
}

/// Queries scored against the pooled block keys at once.
const QSA_SCORE_TILE: usize = 256;

/// QSA indexer: scores blocks of `ratio` tokens (their mean-pooled keys) per query and keeps
/// the `topk` best blocks plus the trailing incomplete block.
pub struct Indexer {
    qk: candle_nn::Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    dim: usize,
    ratio: usize,
    topk: usize,
}

impl Indexer {
    fn load(src: &dyn WeightSource, cfg: &ModelConfig, prefix: &str, p: &Placement) -> Result<Self> {
        let q = cfg.qwen38.as_ref().expect("qwen4_exp config");
        Ok(Self {
            qk: super::layers::load_linear(src, &format!("{prefix}.index_qk_proj"), p)?,
            q_norm: RmsNorm::load_model(src, &format!("{prefix}.q_layernorm.weight"), cfg, p)?,
            k_norm: RmsNorm::load_model(src, &format!("{prefix}.k_layernorm.weight"), cfg, p)?,
            heads: q.index_heads,
            dim: q.index_head_dim,
            ratio: q.index_ratio,
            topk: q.index_topk_blocks(),
        })
    }

    /// Writes this batch's raw indexer keys into `pool` and returns the indexer queries
    /// `[T, heads, d]` (normed and roped).
    fn queries(&self, x: &Tensor, b: &DeviceBatch, pool: &Tensor) -> Result<Tensor> {
        let t = b.num_tokens;
        let (nh, d) = (self.heads, self.dim);
        let qk = self.qk.forward(x)?;
        let q = self.q_norm.forward(&qk.narrow(1, 0, nh * d)?.reshape((t, nh, d))?)?;
        let q = Rotary::apply(&q, &b.cos, &b.sin)?;
        let k = qk.narrow(1, nh * d, d)?.contiguous()?;
        let idx = b.scatter_idx.narrow(1, 0, 1)?.broadcast_as((t, d))?.contiguous()?;
        pool.scatter_set(&idx, &k.to_dtype(pool.dtype())?, 0)?;
        Ok(q)
    }

    /// Decode on CUDA: the selection kernel builds the attended slot tables on the device (no
    /// host round trip, CUDA-graph capturable). `None` when unavailable.
    #[cfg(feature = "cuda")]
    fn select_device(&self, q: &Tensor, b: &DeviceBatch, pool: &Tensor) -> Option<Result<PagedCtx>> {
        use crate::kernels::cuda;
        let pc = b.paged.as_ref()?;
        // the kernel rotates block keys at 1-D positions
        if !q.device().is_cuda() || !cuda::qsa_select_supported(self.heads, self.dim) || !b.mrope_layout.is_empty() {
            return None;
        }
        let (cos, sin) = b.rotary.tables();
        let (knorm, eps) = self.k_norm.parts();
        let t = cuda::QsaTables {
            pool,
            knorm,
            cos,
            sin,
            slots: &pc.slots,
            starts: &pc.starts,
            topk: self.topk,
            ratio: self.ratio,
            max_blocks: pool.dim(0).ok()? / self.ratio + 1,
            eps,
        };
        Some(
            cuda::qsa_select(&q.to_dtype(pool.dtype()).ok()?, &t)
                .map(|(slots, starts)| PagedCtx { slots, starts })
                .map_err(anyhow::Error::from),
        )
    }

    /// What each sequence's queries may attend to, chosen on the host.
    fn select(&self, q: &Tensor, b: &DeviceBatch, pool: &Tensor) -> Result<Vec<Selection>> {
        let (nh, d, r) = (self.heads, self.dim, self.ratio);
        let device = q.device();
        let mut out = Vec::with_capacity(b.seqs.len());
        for (si, s) in b.seqs.iter().enumerate() {
            let nb = s.kv_len / r;
            if nb <= self.topk {
                out.push(Selection::All); // at most `topk` complete blocks are visible: dense
                continue;
            }
            let slots = Tensor::from_slice(&s.slots[..nb * r], nb * r, device)?;
            let pooled = pool
                .index_select(&slots, 0)?
                .to_dtype(DType::F32)?
                .reshape((nb, r, d))?
                .mean(1)?
                .to_dtype(pool.dtype())?;
            let keys = self.k_norm.forward(&pooled)?.reshape((nb, 1, d))?;
            let (cos, sin) = match b.mrope_layout.get(si).filter(|m| !m.is_empty()) {
                Some(m) => {
                    let starts: Vec<[u32; 3]> = (0..nb).map(|i| m.at(i * r)).collect();
                    b.rotary.select_mrope(&starts, device)?
                }
                None => {
                    let starts: Vec<u32> = (0..nb as u32).map(|i| i * r as u32).collect();
                    b.rotary.select(&Tensor::from_vec(starts, nb, device)?, ((nb - 1) * r) as u32)?
                }
            };
            let keys = Rotary::apply(&keys, &cos, &sin)?.reshape((nb, d))?.to_dtype(DType::F32)?;
            let keys_t = keys.t()?.contiguous()?;
            let first = s.kv_len - s.q_len;
            #[cfg(feature = "cuda")]
            if device.is_cuda() {
                // scores and the top-k stay on the device; only a decode step reads its row back
                let mut tiles = Vec::with_capacity(s.q_len.div_ceil(QSA_SCORE_TILE));
                let mut a = 0;
                while a < s.q_len {
                    let n = QSA_SCORE_TILE.min(s.q_len - a);
                    let qs = q.narrow(0, s.q_start + a, n)?.to_dtype(DType::F32)?;
                    let scores = qs
                        .reshape((n * nh, d))?
                        .matmul(&keys_t)?
                        .relu()?
                        .reshape((n, nh, nb))?
                        .sum(1)?;
                    let scores = (scores / (d as f64).sqrt())?;
                    tiles.push(crate::kernels::cuda::qsa_topk_bits(&scores, self.topk, r, first + a)?);
                    a += n;
                }
                let bits = if tiles.len() == 1 { tiles.pop().unwrap() } else { Tensor::cat(&tiles, 0)? };
                if s.q_len == 1 {
                    let words: Vec<u32> = bits.flatten_all()?.to_vec1()?;
                    let visible = s.kv_len / r;
                    let mut sel = Vec::with_capacity(self.topk * r + r);
                    for blk in (0..visible).filter(|&b| words[b / 32] >> (b % 32) & 1 == 1) {
                        sel.extend_from_slice(&s.slots[blk * r..(blk + 1) * r]);
                    }
                    sel.extend_from_slice(&s.slots[visible * r..s.kv_len]);
                    out.push(Selection::Slots(sel));
                } else {
                    out.push(Selection::Blocks(QsaRows { blocks: vec![], bits: Some(bits), ratio: r, first }));
                }
                continue;
            }
            // best `topk` of the blocks query `pos` sees, ties to the lower index
            let pick = |row: &[f32], visible: usize| -> Vec<u32> {
                let mut order: Vec<u32> = (0..visible as u32).collect();
                order.sort_by(|&a, &c| row[c as usize].total_cmp(&row[a as usize]).then(a.cmp(&c)));
                order.truncate(self.topk);
                order.sort_unstable();
                order
            };
            let mut blocks = Vec::with_capacity(s.q_len);
            let mut a = 0;
            while a < s.q_len {
                let n = QSA_SCORE_TILE.min(s.q_len - a);
                let qs = q.narrow(0, s.q_start + a, n)?.to_dtype(DType::F32)?; // [n, nh, d]
                let scores = qs
                    .reshape((n * nh, d))?
                    .matmul(&keys_t)?
                    .relu()?
                    .reshape((n, nh, nb))?
                    .sum(1)?;
                let scores = (scores / (d as f64).sqrt())?.to_vec2::<f32>()?;
                for (i, row) in scores.iter().enumerate() {
                    let visible = (first + a + i + 1) / r;
                    blocks.push((visible > self.topk).then(|| pick(row, visible)));
                }
                a += n;
            }
            if s.q_len == 1 {
                // decode: attend over the selected blocks' slots and the trailing partial block
                let visible = s.kv_len / r;
                let mut sel = Vec::with_capacity(self.topk * r + r);
                for &blk in blocks[0].as_ref().expect("more blocks than the budget") {
                    let b = blk as usize;
                    sel.extend_from_slice(&s.slots[b * r..(b + 1) * r]);
                }
                sel.extend_from_slice(&s.slots[visible * r..s.kv_len]);
                out.push(Selection::Slots(sel));
                continue;
            }
            out.push(Selection::Blocks(QsaRows { blocks, bits: None, ratio: r, first }));
        }
        Ok(out)
    }
}

impl Attention {
    pub fn load(
        src: &dyn WeightSource,
        cfg: &ModelConfig,
        prefix: &str,
        p: &Placement,
    ) -> Result<Self> {
        if !cfg.num_heads.is_multiple_of(p.world) || !cfg.num_kv_heads.is_multiple_of(p.world) {
            bail!(
                "tensor-parallel size {} must divide attention heads ({}) and kv heads ({})",
                p.world,
                cfg.num_heads,
                cfg.num_kv_heads
            );
        }
        let norm = |n: &str| -> Result<Option<RmsNorm>> {
            if cfg.qk_norm {
                Ok(Some(RmsNorm::load_model(
                    src,
                    &format!("{prefix}.{n}.weight"),
                    cfg,
                    p,
                )?))
            } else {
                Ok(None)
            }
        };
        Ok(Self {
            q: load_column_q(src, &format!("{prefix}.q_proj"), p)?,
            k: load_column_q(src, &format!("{prefix}.k_proj"), p)?,
            v: load_column_q(src, &format!("{prefix}.v_proj"), p)?,
            o: load_row_q(src, &format!("{prefix}.o_proj"), p)?,
            q_norm: norm("q_norm")?,
            k_norm: norm("k_norm")?,
            heads: cfg.num_heads / p.world,
            kv_heads: cfg.num_kv_heads / p.world,
            head_dim: cfg.head_dim,
            scale: 1.0 / (cfg.head_dim as f64).sqrt(),
            output_gate: cfg.attn_output_gate,
            indexer: match &cfg.qwen38 {
                Some(_) => Some(Indexer::load(src, cfg, &format!("{prefix}.indexer"), p)?),
                None => None,
            },
        })
    }

    /// Indexer key width (QSA layers), for the extra per-token pool.
    pub fn index_dim(&self) -> Option<usize> {
        self.indexer.as_ref().map(|i| i.dim)
    }

    /// Decode steps of this layer run on carbon's fused CUDA kernel.
    pub fn fused_decode_ready(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            fused_decode_enabled()
                && crate::kernels::cuda::paged_decode_supported(
                    self.heads,
                    self.kv_heads,
                    self.head_dim,
                )
                // QSA: the selection must run on the device too
                && self
                    .indexer
                    .as_ref()
                    .is_none_or(|i| crate::kernels::cuda::qsa_select_supported(i.heads, i.dim))
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// `x: [T, hidden]` -> `[T, hidden]` (a partial sum under tensor parallelism).
    /// Only this step's KV (and QSA indexer keys) into the pools, without attending: keeps the
    /// layer's cache current when its output is not needed (the MTP head while it cannot
    /// speculate).
    pub fn write_kv(&self, x: &Tensor, b: &DeviceBatch, kv: &LayerKv) -> Result<()> {
        let t = b.num_tokens;
        let (kvh, d) = (self.kv_heads, self.head_dim);
        let mut k = self.k.forward(x)?.reshape((t, kvh, d))?;
        let v = self.v.forward(x)?.reshape((t, kvh * d))?;
        if let (Some(_), Some(kn)) = (&self.q_norm, &self.k_norm) {
            k = kn.forward(&k)?;
        }
        let k = Rotary::apply(&k, &b.cos, &b.sin)?.reshape((t, kvh * d))?;
        kv.k.scatter_set(&b.scatter_idx, &k.to_dtype(kv.k.dtype())?.contiguous()?, 0)?;
        kv.v.scatter_set(&b.scatter_idx, &v.to_dtype(kv.v.dtype())?.contiguous()?, 0)?;
        if let (Some(ix), Some(pool)) = (&self.indexer, &kv.idx) {
            ix.queries(x, b, pool)?;
        }
        Ok(())
    }

    pub fn forward(&self, x: &Tensor, b: &DeviceBatch, kv: &LayerKv) -> Result<Tensor> {
        let t = b.num_tokens;
        let (h, kvh, d) = (self.heads, self.kv_heads, self.head_dim);
        let (mut q, gate) = if self.output_gate {
            let qg = self.q.forward(x)?.reshape((t, h, 2 * d))?;
            (
                qg.narrow(2, 0, d)?,
                Some(qg.narrow(2, d, d)?.reshape((t, h * d))?),
            )
        } else {
            (self.q.forward(x)?.reshape((t, h, d))?, None)
        };
        let mut k = self.k.forward(x)?.reshape((t, kvh, d))?;
        let v = self.v.forward(x)?.reshape((t, kvh * d))?;
        if let (Some(qn), Some(kn)) = (&self.q_norm, &self.k_norm) {
            q = qn.forward(&q)?;
            k = kn.forward(&k)?;
        }
        let q = Rotary::apply(&q, &b.cos, &b.sin)?;
        let k = Rotary::apply(&k, &b.cos, &b.sin)?.reshape((t, kvh * d))?;
        kv.k.scatter_set(&b.scatter_idx, &k.to_dtype(kv.k.dtype())?.contiguous()?, 0)?;
        kv.v.scatter_set(&b.scatter_idx, &v.to_dtype(kv.v.dtype())?.contiguous()?, 0)?;
        let out = match (&self.indexer, &kv.idx) {
            (Some(ix), Some(pool)) => {
                let qi = ix.queries(x, b, pool)?;
                match self.qsa_decode_fused(ix, &qi, &q, b, kv, pool) {
                    Some(o) => o,
                    None => {
                        let sel = ix.select(&qi, b, pool)?;
                        if sel.iter().all(|s| matches!(s, Selection::All)) {
                            self.attend_all(&q, b, kv)?
                        } else {
                            self.attend_selected(&q, b, kv, &sel)?
                        }
                    }
                }
            }
            (Some(_), None) => bail!("QSA layer without an indexer key pool"),
            _ => self.attend_all(&q, b, kv)?,
        };        let out = out.to_dtype(x.dtype())?;
        let out = match gate {
            Some(g) => (out * candle_nn::ops::sigmoid(&g)?)?,
            None => out,
        };
        self.o.forward(&out)
    }

    /// QSA decode entirely on the device: selection kernel, then the paged decode kernel over
    /// the selected slots. `None` (the host path runs) when either is unavailable.
    #[cfg(feature = "cuda")]
    fn qsa_decode_fused(&self, ix: &Indexer, qi: &Tensor, q: &Tensor, b: &DeviceBatch, kv: &LayerKv, pool: &Tensor) -> Option<Tensor> {
        if !fused_decode_enabled() {
            return None;
        }
        match ix.select_device(qi, b, pool)? {
            Ok(pc) => self.decode_fused(q, &pc, kv),
            Err(e) => {
                disable_fused_decode(&e);
                None
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn qsa_decode_fused(&self, _: &Indexer, _: &Tensor, _: &Tensor, _: &DeviceBatch, _: &LayerKv, _: &Tensor) -> Option<Tensor> {
        None
    }

    /// Every query over its whole causal history.
    fn attend_all(&self, q: &Tensor, b: &DeviceBatch, kv: &LayerKv) -> Result<Tensor> {
        let fused = match (&b.paged, &b.prefill) {
            (Some(pc), _) => self.decode_fused(q, pc, kv),
            (None, Some(pf)) => self.prefill_fused(q, pf, kv),
            (None, None) => None,
        };
        Ok(match (fused, &b.decode) {
            (Some(o), _) => o,
            (None, Some(dc)) => self.decode_batched(q, dc, b.seqs.len(), kv)?,
            (None, None) => {
                let mut outs = Vec::with_capacity(b.seqs.len());
                for s in &b.seqs {
                    outs.push(self.attend_seq(&q.narrow(0, s.q_start, s.q_len)?, s, kv, None)?);
                }
                if outs.len() == 1 {
                    outs.pop().unwrap()
                } else {
                    Tensor::cat(&outs, 0)?
                }
            }
        })
    }

    /// QSA: queries restricted to the indexer's selection.
    fn attend_selected(&self, q: &Tensor, b: &DeviceBatch, kv: &LayerKv, sel: &[Selection]) -> Result<Tensor> {
        let device = q.device();
        // a decode batch goes through the fused paged kernel with per-sequence slot lists
        if b.paged.is_some() {
            let mut slots = Vec::new();
            let mut starts = vec![0u32];
            for (s, x) in b.seqs.iter().zip(sel) {
                match x {
                    Selection::Slots(v) => slots.extend_from_slice(v),
                    _ => slots.extend_from_slice(&s.slots[..s.kv_len]),
                }
                starts.push(slots.len() as u32);
            }
            let n = slots.len();
            let pc = PagedCtx {
                slots: Tensor::from_vec(slots, n, device)?,
                starts: Tensor::from_vec(starts, b.seqs.len() + 1, device)?,
            };
            if let Some(o) = self.decode_fused(q, &pc, kv) {
                return Ok(o);
            }
        }
        let mut outs = Vec::with_capacity(b.seqs.len());
        for (s, x) in b.seqs.iter().zip(sel) {
            let qs = q.narrow(0, s.q_start, s.q_len)?;
            outs.push(match x {
                Selection::All => self.attend_seq(&qs, s, kv, None)?,
                Selection::Blocks(rows) => self.attend_seq(&qs, s, kv, Some(rows))?,
                Selection::Slots(v) => {
                    // one query over the selected tokens only (all of them precede it)
                    let sub = SeqCtx {
                        q_start: 0,
                        q_len: 1,
                        kv_len: v.len(),
                        prev_tokens: vec![],
                        slots: std::sync::Arc::new(v.clone()),
                        kv_idx: std::sync::OnceLock::new(),
                    };
                    self.attend_seq(&qs, &sub, kv, None)?
                }
            });
        }
        Ok(if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 0)? })
    }

    fn gather(&self, cache: &Tensor, idx: &Tensor, rows: usize) -> Result<Tensor> {
        // [rows, kvh*d] -> [kvh, rows, d]
        Ok(cache
            .index_select(idx, 0)?
            .reshape((rows, self.kv_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?)
    }

    /// `extra`: a QSA selection on top of the causal mask.
    fn attend_seq(&self, q: &Tensor, s: &SeqCtx, kv: &LayerKv, extra: Option<&QsaRows>) -> Result<Tensor> {
        let (h, kvh, d) = (self.heads, self.kv_heads, self.head_dim);
        let g = h / kvh;
        let (q_len, kv_len) = (s.q_len, s.kv_len);
        let idx = s.kv_idx(kv.k.device())?;
        let k = self.gather(&kv.k, idx, kv_len)?;
        let v = self.gather(&kv.v, idx, kv_len)?;
        let kt = k.t()?.contiguous()?; // [kvh, d, kv_len]
        let q = q
            .to_dtype(k.dtype())?
            .reshape((q_len, kvh, g, d))?
            .permute((1, 2, 0, 3))?
            .contiguous()?; // [kvh, g, q_len, d]
        let prefix = kv_len - q_len;
        let tile = (SCORE_BUDGET / (h * kv_len).max(1)).clamp(1, q_len);
        let device = q.device();
        let mut tiles = Vec::with_capacity(q_len.div_ceil(tile));
        let mut a = 0;
        while a < q_len {
            let n = tile.min(q_len - a);
            let qt = q.narrow(2, a, n)?.reshape((kvh, g * n, d))?;
            let scores = (qt.matmul(&kt)? * self.scale)?
                .to_dtype(DType::F32)?
                .reshape((kvh, g, n, kv_len))?;
            // causal: query at absolute position prefix+a+i sees keys 0..=prefix+a+i
            let last_pos = prefix + a + n - 1;
            let scores = match extra {
                Some(rows) => scores.broadcast_add(&rows.mask(a, n, kv_len, device)?)?,
                None => scores,
            };
            let scores = if last_pos + 1 < kv_len || n > 1 {
                let j = Tensor::arange(0u32, kv_len as u32, device)?.reshape((1, kv_len))?;
                let p = Tensor::arange((prefix + a) as u32, (prefix + a + n) as u32, device)?
                    .reshape((n, 1))?;
                let mask = (j.broadcast_gt(&p)?.to_dtype(DType::F32)? * MASK_VALUE as f64)?;
                scores.broadcast_add(&mask)?
            } else {
                scores
            };
            let probs =
                candle_nn::ops::softmax_last_dim(&scores.contiguous()?)?.to_dtype(v.dtype())?;
            let o = probs.reshape((kvh, g * n, kv_len))?.matmul(&v)?; // [kvh, g*n, d]
            tiles.push(o.reshape((kvh, g, n, d))?);
            a += n;
        }
        let o = if tiles.len() == 1 {
            tiles.pop().unwrap()
        } else {
            Tensor::cat(&tiles, 2)?
        };
        Ok(o.permute((2, 0, 1, 3))?.reshape((q_len, h * d))?)
    }

    /// carbon's fused paged-attention kernel; `None` when unavailable (the caller falls back).
    #[cfg(feature = "cuda")]
    fn decode_fused(&self, q: &Tensor, pc: &PagedCtx, kv: &LayerKv) -> Option<Tensor> {
        use crate::kernels::cuda;
        let (h, kvh, d) = (self.heads, self.kv_heads, self.head_dim);
        if !cuda::paged_decode_supported(h, kvh, d) {
            return None;
        }
        let run = || -> Result<Tensor> {
            let b = q.dim(0)?;
            let q = q.to_dtype(kv.k.dtype())?.reshape((b, h, d))?;
            let o = cuda::paged_decode(
                &q,
                &kv.k,
                &kv.v,
                &pc.slots,
                &pc.starts,
                kvh,
                self.scale as f32,
            )?;
            Ok(o.reshape((b, h * d))?)
        };
        match run() {
            Ok(o) => Some(o),
            Err(e) => {
                disable_fused_decode(&e);
                None
            }
        }
    }

    /// carbon's tiled CUDA prefill attention; `None` when unavailable (the caller falls back).
    #[cfg(feature = "cuda")]
    fn prefill_fused(&self, q: &Tensor, pf: &PrefillCtx, kv: &LayerKv) -> Option<Tensor> {
        use crate::kernels::cuda;
        let (h, kvh, d) = (self.heads, self.kv_heads, self.head_dim);
        if !cuda::paged_decode_supported(h, kvh, d) {
            return None;
        }
        let run = || -> Result<Tensor> {
            let t = q.dim(0)?;
            let q = q.to_dtype(kv.k.dtype())?.reshape((t, h, d))?;
            let o = cuda::flash_prefill(
                &q,
                &kv.k,
                &kv.v,
                &pf.slots,
                &pf.seqinfo,
                &pf.tiles,
                kvh,
                self.scale as f32,
            )?;
            Ok(o.reshape((t, h * d))?)
        };
        match run() {
            Ok(o) => Some(o),
            Err(e) => {
                disable_fused_decode(&e);
                None
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn prefill_fused(&self, _: &Tensor, _: &PrefillCtx, _: &LayerKv) -> Option<Tensor> {
        None
    }

    #[cfg(not(feature = "cuda"))]
    fn decode_fused(&self, _: &Tensor, _: &PagedCtx, _: &LayerKv) -> Option<Tensor> {
        None
    }

    fn decode_batched(&self, q: &Tensor, dc: &DecodeCtx, b: usize, kv: &LayerKv) -> Result<Tensor> {
        let (h, kvh, d) = (self.heads, self.kv_heads, self.head_dim);
        let g = h / kvh;
        let l = dc.lmax;
        let k = kv.k.index_select(&dc.kv_idx, 0)?.reshape((b, l, kvh, d))?;
        let v = kv.v.index_select(&dc.kv_idx, 0)?.reshape((b, l, kvh, d))?;
        let kt = k.permute((0, 2, 3, 1))?.contiguous()?; // [b, kvh, d, l]
        let v = v.permute((0, 2, 1, 3))?.contiguous()?; // [b, kvh, l, d]
        let q = q
            .to_dtype(kt.dtype())?
            .reshape((b, kvh, g, d))?
            .contiguous()?;
        let scores = (q.matmul(&kt)? * self.scale)?
            .to_dtype(DType::F32)?
            .broadcast_add(&dc.mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores.contiguous()?)?.to_dtype(v.dtype())?;
        let o = probs.matmul(&v)?; // [b, kvh, g, d]
        Ok(o.reshape((b, h * d))?)
    }
}

/// Reference attention without a KV pool (used by tests): `q:[T,h,d] k,v:[T,kvh,d]`, causal.
pub fn naive_causal_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (t, h, d) = q.dims3()?;
    let kvh = k.dim(1)?;
    let g = h / kvh;
    let k = k.unsqueeze(2)?.repeat((1, 1, g, 1))?.reshape((t, h, d))?;
    let v = v.unsqueeze(2)?.repeat((1, 1, g, 1))?.reshape((t, h, d))?;
    let q = q.transpose(0, 1)?.contiguous()?;
    let k = k.transpose(0, 1)?.contiguous()?;
    let v = v.transpose(0, 1)?.contiguous()?;
    let s = (q.matmul(&k.t()?)? / (d as f64).sqrt())?;
    let mask: Vec<f32> = (0..t)
        .flat_map(|i| (0..t).map(move |j| if j > i { MASK_VALUE } else { 0.0 }))
        .collect();
    let mask = Tensor::from_vec(mask, (t, t), q.device())?;
    let p = candle_nn::ops::softmax(&s.broadcast_add(&mask)?, D::Minus1)?;
    Ok(p.matmul(&v)?.transpose(0, 1)?.reshape((t, h * d))?)
}
