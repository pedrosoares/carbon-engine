// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Qwen3.8-Flash-Next multi-token prediction head (`mtp.*`), after vLLM's
//! `vllm/models/qwen4_exp/nvidia/mtp.py`.
//!
//! At position `i` it takes the main model's last-layer state before the final mixer `H_i`
//! (`[T, hc * H]`) and the embedding of token `i + 1`, runs one QSA decoder layer (hyper
//! connections, attention with an indexer, MoE; no PLE) and mixes the streams for the shared LM
//! head, which then predicts token `i + 2`:
//!
//! ```text
//! e = fc_embedding(norm_e(embed(tok[i+1])))                     [T, H]
//! h = fc_hidden(norm_h(H_i) viewed [T, hc, H])                  [T, hc*H]   (fc shared per stream)
//! h = h + e in every stream; attention and MoE blocks as in the main layers
//! single = hyper_connection_mixer.mix(h)                        [T, H] -> LM head
//! ```
//!
//! Its attention keeps its own KV (and indexer key) pool, addressed with the main model's KV
//! slots: position `i` of a sequence lives in the same slot in both, so it runs over every
//! position the main model computes, prompt included.
//!
//! Phase 1 (shadow mode): the routed experts (bf16, stacked, 9.8 MB each) stay in the mapped
//! checkpoint; each step uploads the experts it routes to.

use super::attention::{Attention, DeviceBatch};
use super::hyper::GatedResidual;
use super::layers::{load_linear, Mlp, Placement, RmsNorm, Rotary};
use crate::batch::BatchMeta;
use crate::config::ModelConfig;
use crate::kvcache::LayerKv;
use crate::weights::{RowReader, WeightSource};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::Linear;
use parking_lot::RwLock;

pub struct MtpHead {
    norm_e: RmsNorm,
    fc_e: Linear,
    norm_h: RmsNorm,
    fc_h: Linear,
    attn_hc: GatedResidual,
    mlp_hc: GatedResidual,
    attn: Attention,
    gate: Linear,
    shared: Mlp,
    shared_gate: Linear,
    /// `[E * 2I, H]` (gate rows then up rows per expert) and `[E * H, I]`, mapped.
    gate_up: RowReader,
    down: RowReader,
    num_experts: usize,
    inter: usize,
    top_k: usize,
    norm_topk: bool,
    mixer: GatedResidual,
    /// The routed experts in the stage's slot cache (`attach_bank`), NVFP4-quantized.
    #[cfg(feature = "cuda")]
    banked: Option<super::moe::MoeBlock>,
    rotary: Rotary,
    kv: RwLock<Option<LayerKv>>,
    kv_width: usize,
    index_dim: usize,
    head_dim: usize,
    hidden: usize,
    hc: usize,
    device: Device,
    dtype: DType,
}

impl MtpHead {
    /// The checkpoint's MTP head, or `None` when it has none (or it is not a Qwen3.8 model).
    pub fn load(src: &dyn WeightSource, cfg: &ModelConfig, max_seq_len: usize, p: &Placement) -> Result<Option<Self>> {
        let (Some(q), Some(moe)) = (cfg.qwen38.as_ref(), cfg.moe.as_ref()) else {
            return Ok(None);
        };
        if !src.contains("mtp.fc_embedding.weight") {
            return Ok(None);
        }
        if p.world != 1 {
            bail!("the MTP head runs on one device");
        }
        let l = "mtp.layers.0";
        let reader = |name: &str| -> Result<RowReader> {
            src.row_reader(name)
                .with_context(|| format!("{name}: the MTP experts must be memory-mapped"))?
        };
        let gate_up = reader(&format!("{l}.mlp.experts.gate_up_proj"))?;
        let down = reader(&format!("{l}.mlp.experts.down_proj"))?;
        let (e, i, h) = (moe.num_experts, moe.intermediate_size, cfg.hidden_size);
        if gate_up.shape != [e * 2 * i, h] || down.shape != [e * h, i] {
            bail!("unexpected MTP expert shapes {:?} / {:?}", gate_up.shape, down.shape);
        }
        let mlp_names = ["gate_proj", "up_proj", "down_proj"];
        Ok(Some(Self {
            norm_e: RmsNorm::load_model(src, "mtp.pre_fc_norm_embedding.weight", cfg, p)?,
            fc_e: load_linear(src, "mtp.fc_embedding", p)?,
            norm_h: RmsNorm::load_model(src, "mtp.pre_fc_norm_hidden.weight", cfg, p)?,
            fc_h: load_linear(src, "mtp.fc_hidden", p)?,
            attn_hc: GatedResidual::load(src, &format!("{l}.attn_hyper_connection"), cfg, p, true)?,
            mlp_hc: GatedResidual::load(src, &format!("{l}.mlp_hyper_connection"), cfg, p, true)?,
            attn: Attention::load(src, cfg, &format!("{l}.self_attn"), p)?,
            gate: load_linear(src, &format!("{l}.mlp.gate"), p)?,
            shared: Mlp::load(src, &format!("{l}.mlp.shared_expert"), mlp_names, &cfg.hidden_act, p)?,
            shared_gate: load_linear(src, &format!("{l}.mlp.shared_expert_gate"), p)?,
            gate_up,
            down,
            num_experts: e,
            inter: i,
            top_k: moe.top_k,
            norm_topk: moe.norm_topk_prob,
            mixer: GatedResidual::load(src, "mtp.hyper_connection_mixer", cfg, p, false)?,
            #[cfg(feature = "cuda")]
            banked: None,
            rotary: Rotary::new(cfg, max_seq_len, p)?,
            kv: RwLock::new(None),
            kv_width: cfg.num_kv_heads * cfg.head_dim,
            index_dim: q.index_head_dim,
            head_dim: cfg.head_dim,
            hidden: h,
            hc: q.hc_count,
            device: p.device.clone(),
            dtype: p.dtype,
        }))
    }

    /// (Re)allocate the KV and indexer-key pools; `num_slots` = the main model's KV slots.
    pub fn alloc_kv(&self, num_slots: usize) -> Result<()> {
        let mut kv = self.kv.write();
        *kv = None;
        *kv = Some(
            LayerKv::new(num_slots, self.kv_width / self.head_dim, self.head_dim, self.dtype, &self.device)?
                .with_index(self.index_dim)?,
        );
        Ok(())
    }

    /// Bytes the KV pools take per token.
    pub fn kv_bytes_per_token(&self) -> usize {
        (2 * self.kv_width + self.index_dim) * self.dtype.size_in_bytes()
    }

    /// Run the head over every token of `meta`: `hidden [T, hc*H]` is the main model's
    /// pre-mixer output for those tokens (or the head's own, for a further draft step) and
    /// `next_emb [T, H]` the embeddings of each token's successor. Writes the head's KV for every
    /// token and returns, for `rows`, the mixed state `[R, H]` (for the LM head) and the
    /// pre-mixer streams `[R, hc*H]` (the next draft step's input).
    pub fn forward(&self, meta: &BatchMeta, hidden: &Tensor, next_emb: &Tensor, rows: &[u32]) -> Result<(Tensor, Tensor)> {
        let kv = self.kv.read();
        let kv = kv.as_ref().context("MTP KV pool not allocated")?;
        let dev = &self.device;
        let t = hidden.dim(0)?;
        let hidden = hidden.to_device(dev)?.to_dtype(self.dtype)?;
        let e = self.fc_e.forward(&self.norm_e.forward(&next_emb.to_device(dev)?.to_dtype(self.dtype)?)?)?;
        let h = self.norm_h.forward(&hidden)?.reshape((t, self.hc, self.hidden))?;
        let h = self.fc_h.forward(&h)?; // [T, hc, H], shared across streams
        // the embedding branch enters as the previous block's output: added to every stream
        let mut x = h.broadcast_add(&e.unsqueeze(1)?)?.reshape((t, self.hc * self.hidden))?;
        let b = DeviceBatch::build(meta, &self.rotary, self.kv_width, dev)?;
        let (blk, inj) = self.attn_hc.mix(&x)?;
        let a = self.attn.forward(&blk, &b, kv)?;
        x = self.attn_hc.combine(&x, &a, &inj.context("MTP attention injection")?)?;
        let (blk, inj) = self.mlp_hc.mix(&x)?;
        let m = self.moe(&blk)?;
        x = self.mlp_hc.combine(&x, &m, &inj.context("MTP MoE injection")?)?;
        if rows.is_empty() {
            // a prompt chunk that samples nothing: only the KV mattered
            let w = self.hc * self.hidden;
            return Ok((Tensor::zeros((0, self.hidden), self.dtype, dev)?, Tensor::zeros((0, w), self.dtype, dev)?));
        }
        let idx = Tensor::from_slice(rows, rows.len(), dev)?;
        let multi = x.index_select(&idx, 0)?;
        Ok((self.mixer.mix(&multi)?.0, multi))
    }

    /// Quantize the routed experts to NVFP4 into `bank` (the main stage's slot cache, which then
    /// serves them like a 49th layer: routing, caching and the fused kernels on the GPU).
    #[cfg(feature = "cuda")]
    pub fn attach_bank(&mut self, bank: super::moe::Bank, moe: &crate::config::MoeConfig, act: &str) -> Result<()> {
        use super::layers::QLinear;
        use crate::weights::{quantize_nvfp4, Nvfp4};
        use rayon::prelude::*;
        let (i, h) = (self.inter, self.hidden);
        // the head's experts take the bank's format
        let (fmt, dense_dtype) = bank.format().context("the expert slot cache is empty")?;
        if fmt != "nvfp4" && fmt != "dense" {
            tracing::warn!("mtp: experts stay in the mapped checkpoint ({fmt} expert cache)");
            return Ok(());
        }
        let q = |w: Tensor| -> Result<QLinear> {
            if fmt == "dense" {
                return Ok(QLinear::Dense(Linear::new(w.to_dtype(dense_dtype)?.contiguous()?, None)));
            }
            let (n, k) = w.dims2()?;
            let v = w.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let (packed, scale, global) = quantize_nvfp4(&v, n, k);
            Ok(QLinear::Nvfp4(Nvfp4 {
                packed: Tensor::from_vec(packed, (n, k / 2), &Device::Cpu)?,
                scale: Tensor::from_vec(scale, (n, k / 16), &Device::Cpu)?,
                global,
            }))
        };
        let list: Vec<Option<std::sync::Arc<Mlp>>> = (0..self.num_experts)
            .into_par_iter()
            .map(|e| -> Result<_> {
                let gu = self.gate_up.rows_tensor(e * 2 * i, 2 * i)?;
                let dn = self.down.rows_tensor(e * h, h)?;
                let mlp = Mlp::new_q(q(gu.narrow(0, 0, i)?)?, q(gu.narrow(0, i, i)?)?, q(dn)?, act);
                Ok(Some(std::sync::Arc::new(mlp)))
            })
            .collect::<Result<_>>()?;
        let (row, table) = bank.register(&list)?;
        self.banked = Some(super::moe::MoeBlock::from_bank(
            usize::MAX,
            self.gate.clone(),
            Some((self.shared.clone(), self.shared_gate.clone())),
            bank,
            row,
            table,
            moe,
            act,
            &self.device,
            self.dtype,
        ));
        Ok(())
    }

    /// Only the head's KV for every token of `meta` (same inputs as `forward`): the attention
    /// block's input and its K/V/indexer projections, no attention, MoE or mixer. Keeps later
    /// drafts' context intact while the head cannot speculate.
    pub fn fill_kv(&self, meta: &BatchMeta, hidden: &Tensor, next_emb: &Tensor) -> Result<()> {
        let kv = self.kv.read();
        let kv = kv.as_ref().context("MTP KV pool not allocated")?;
        let dev = &self.device;
        let t = hidden.dim(0)?;
        let hidden = hidden.to_device(dev)?.to_dtype(self.dtype)?;
        let e = self.fc_e.forward(&self.norm_e.forward(&next_emb.to_device(dev)?.to_dtype(self.dtype)?)?)?;
        let h = self.fc_h.forward(&self.norm_h.forward(&hidden)?.reshape((t, self.hc, self.hidden))?)?;
        let x = h.broadcast_add(&e.unsqueeze(1)?)?.reshape((t, self.hc * self.hidden))?;
        let b = DeviceBatch::build(meta, &self.rotary, self.kv_width, dev)?;
        let (blk, _) = self.attn_hc.mix(&x)?;
        self.attn.write_kv(&blk, &b, kv)
    }

    /// Routed experts + shared expert: through the slot cache once attached, else the experts
    /// are uploaded from the mapping per step (host routing).
    fn moe(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(m) = &self.banked {
            return m.forward(x);
        }
        let (t, h) = x.dims2()?;
        let i = self.inter;
        let logits = self.gate.forward(x)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let mut per_expert: Vec<Vec<(u32, f32)>> = vec![vec![]; self.num_experts];
        let mut order: Vec<usize> = (0..self.num_experts).collect();
        for (tok, row) in logits.iter().enumerate() {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
            order.sort_unstable_by(|&a, &b| exp[b].total_cmp(&exp[a]).then(a.cmp(&b)));
            let top = &order[..self.top_k];
            let denom = if self.norm_topk {
                top.iter().map(|&e| exp[e]).sum::<f32>()
            } else {
                exp.iter().sum()
            };
            for &e in top {
                per_expert[e].push((tok as u32, exp[e] / denom));
            }
        }
        let dev = &self.device;
        let mut out = Tensor::zeros((t, h), self.dtype, dev)?;
        for (e, r) in per_expert.iter().enumerate() {
            if r.is_empty() {
                continue;
            }
            let gu = self.gate_up.rows_tensor(e * 2 * i, 2 * i)?.to_device(dev)?.to_dtype(self.dtype)?;
            let dn = self.down.rows_tensor(e * h, h)?.to_device(dev)?.to_dtype(self.dtype)?;
            let n = r.len();
            let idx = Tensor::from_vec(r.iter().map(|(t, _)| *t).collect::<Vec<_>>(), n, dev)?;
            let w = Tensor::from_vec(r.iter().map(|(_, w)| *w).collect::<Vec<_>>(), (n, 1), dev)?.to_dtype(self.dtype)?;
            let xs = x.index_select(&idx, 0)?;
            let gu_out = xs.matmul(&gu.t()?)?; // [n, 2I]
            let act = (candle_nn::ops::silu(&gu_out.narrow(1, 0, i)?)? * gu_out.narrow(1, i, i)?)?;
            let y = act.matmul(&dn.t()?)?.broadcast_mul(&w)?;
            out = out.index_add(&idx, &y, 0)?;
        }
        let g = candle_nn::ops::sigmoid(&self.shared_gate.forward(x)?)?;
        Ok((out + self.shared.forward(x)?.broadcast_mul(&g)?)?)
    }
}
