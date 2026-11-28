// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Inkling 2 / Inkling 2 Mini (llama.cpp `inkling2`, `inkling2_loop`): a looped MoE transformer.
//!
//! Layers `0..loop_start` (prelude) run once, the core `loop_start..loop_end` runs `loop_count`
//! times with shared weights, then the coda. Every layer visit has its own caches (a "cache
//! slot"), so a token makes `visits()` layer visits. Per visit:
//!
//! - Inputs come from block attention residuals: the layer mixes the RMS-normalized outputs of
//!   earlier blocks (a new block every `block_size` layers) and its current stream with learned
//!   queries (an attention over the depth). From the second pass on, a cross-pass message (the
//!   same kind of mix over the previous pass) enters each input through a tanh gain / forget
//!   controller.
//! - Attention has no rope: queries / keys are L2-clipped, values RMS-normalized, and the
//!   logits get a relative-position bias (a learned projection of the query against a table
//!   over distances) and a per-query sink. Keys and values pass a causal short conv first (with
//!   input-dependent taps on the full-attention layers), and so do both layer outputs. Sliding
//!   window layers keep their last `window` keys in per-sequence rings (recurrent state); full
//!   attention layers use the paged KV pool.
//! - MoE: sigmoid routing with a selection bias, SiTU-gated experts in GGUF NVFP4 or Q8_0.
//!
//! The stack runs on one CUDA device in f32 activations; weights stay quantized (Q8 dense
//! projections, NVFP4 / Q8 experts) and are read by the `kernels::inkling` kernels.

use crate::batch::BatchMeta;
use crate::config::{InklingConfig, ModelConfig};
use crate::gguf::{Gguf, GgmlType, NVFP4_BLOCK};
use crate::kernels::inkling::{self as k, ExpertWeights, KvSource, RowNorm, Tables, Q8};
use crate::weights::WeightSource;
use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use parking_lot::RwLock;
use rayon::prelude::*;

/// Norm epsilon of the depth mixes and loop controllers (fixed in the reference).
const MIX_EPS: f32 = 1e-6;

/// Batches up to this many tokens use the decode kernels for the loop controllers.
const SMALL_T: usize = 16;

struct Layer {
    swa: bool,
    /// SWA: [attn_norm]; full attention: q, k, v input norms.
    norms: Vec<Tensor>,
    /// Rows q | rel | gate (input 0), k (input 1), v (input 2).
    proj: Q8,
    wo: Q8,
    /// Rows: sink logits (input 0), and on full attention layers the K / V conv taps (inputs
    /// 1 and 2, `nkv * 4` each).
    small: Tensor,
    k_conv: Tensor,
    v_conv: Tensor,
    post_conv: Tensor,
    ffn_conv: Tensor,
    q_clip: Tensor,
    rel_clip: Tensor,
    rel_bias: Tensor,
    attn_query: Option<Tensor>,
    ffn_query: Tensor,
    ffn_norm: Tensor,
    router: Tensor,
    router_bias: Option<Tensor>,
    experts: ExpertWeights,
}

/// Cross-pass controllers of a core layer (q, k, v, ffn).
struct LoopLayer {
    msg_query: Tensor,
    down: [Tensor; 4],
    gain: [Tensor; 4],
    forget: [Tensor; 4],
}

#[derive(Clone, Copy)]
enum Cache {
    /// Index into the global KV pools.
    Global(usize),
    /// Index into the sliding-window rings.
    Ring(usize),
}

#[derive(Default)]
struct Caches {
    kv: Vec<crate::kvcache::LayerKv>,
    /// (k, v) bf16 `[states * window, nkv * 128]` per sliding-window cache slot.
    rings: Vec<(Tensor, Tensor)>,
    /// f32 `[states, 3, conv_channels]` per cache slot.
    conv: Vec<Tensor>,
    num_slots: usize,
    num_states: usize,
}

pub struct Body {
    ic: InklingConfig,
    hidden: usize,
    nh: usize,
    nkv: usize,
    eps: f32,
    layers: Vec<Layer>,
    loops: Vec<LoopLayer>,
    out_query: Tensor,
    /// The core layers' message queries stacked `[core layers * 4, H]`.
    msg_queries: Tensor,
    slots: Vec<Cache>,
    n_global: usize,
    n_ring: usize,
    caches: RwLock<Caches>,
    graphs: parking_lot::Mutex<Graphs>,
    device: Device,
}

/// Most sequences of a decode step replayed from a CUDA graph.
const GRAPH_MAX_BATCH: usize = 16;

/// Decode graphs are on unless `CARBON_INKLING_GRAPHS=0`.
fn graphs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CARBON_INKLING_GRAPHS").map_or(true, |v| v != "0"))
}

struct GraphEntry {
    graph: candle_core::cuda_backend::cudarc::driver::CudaGraph,
    x_in: Tensor,
    y_out: Tensor,
    batch: Batch,
}

#[derive(Default)]
struct Graphs {
    disabled: bool,
    seen: std::collections::HashMap<usize, u32>,
    entries: std::collections::HashMap<usize, GraphEntry>,
}

// CudaGraph holds raw driver handles; every use goes through the mutex.
unsafe impl Send for Graphs {}

fn info<'a>(g: &'a Gguf, name: &str) -> Result<&'a crate::gguf::TensorInfo> {
    g.tensors.get(name).with_context(|| format!("GGUF tensor '{name}' is missing"))
}

/// Q8_0 tensors stacked by rows into the Q8 layout.
fn load_q8(g: &Gguf, names: &[String], dev: &Device) -> Result<Q8> {
    let (mut codes, mut scales, mut rows, mut cols) = (Vec::new(), Vec::new(), 0, 0);
    for n in names {
        let t = info(g, n)?;
        ensure!(t.ty == GgmlType::Q8_0 && t.shape.len() == 2, "inkling: '{n}' is {:?} {:?}, expected a Q8_0 matrix", t.ty, t.shape);
        ensure!(cols == 0 || cols == t.shape[1], "inkling: '{n}' has {} columns, not {cols}", t.shape[1]);
        cols = t.shape[1];
        rows += t.shape[0];
        for blk in g.bytes(t).chunks_exact(34) {
            scales.push(half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
            codes.extend_from_slice(&blk[2..]);
        }
    }
    Ok(Q8 {
        codes: Tensor::from_vec(codes, (rows, cols), &Device::Cpu)?.to_device(dev)?,
        scales: Tensor::from_vec(scales, (rows, cols / 32), &Device::Cpu)?.to_device(dev)?,
        rows,
        cols,
    })
}

/// A 3-D expert tensor `[E, rows, cols]` in the kernels' layout: (codes, scales, format).
fn load_expert_tensor(g: &Gguf, name: &str, dev: &Device) -> Result<(Tensor, Tensor, u32, [usize; 3])> {
    let t = info(g, name)?;
    let &[e, rows, cols] = t.shape.as_slice() else {
        bail!("inkling: '{name}' has shape {:?}", t.shape);
    };
    let b = g.bytes(t);
    match t.ty {
        GgmlType::Q8_0 => {
            let n = e * rows * cols;
            let mut codes = vec![0u8; n];
            let mut scales = vec![half::f16::ZERO; n / 32];
            codes
                .par_chunks_exact_mut(32)
                .zip(scales.par_iter_mut())
                .zip(b.par_chunks_exact(34))
                .for_each(|((c, s), blk)| {
                    *s = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]]));
                    c.copy_from_slice(&blk[2..]);
                });
            Ok((
                Tensor::from_vec(codes, n, &Device::Cpu)?.to_device(dev)?,
                Tensor::from_vec(scales, n / 32, &Device::Cpu)?.to_device(dev)?,
                0,
                [e, rows, cols],
            ))
        }
        GgmlType::Nvfp4 => {
            let n = e * rows * cols;
            let mut codes = vec![0u8; n / 2];
            let mut scales = vec![0u8; n / 16];
            codes
                .par_chunks_exact_mut(32)
                .zip(scales.par_chunks_exact_mut(4))
                .zip(b.par_chunks_exact(NVFP4_BLOCK))
                .for_each(|((c, s), blk)| {
                    // codes keep the GGUF nibble order (the kernels' int8 expansion wants it)
                    c.copy_from_slice(&blk[4..]);
                    for sub in 0..4 {
                        s[sub] = if blk[sub] == 0x7F { 0 } else { blk[sub] };
                    }
                });
            Ok((
                Tensor::from_vec(codes, n / 2, &Device::Cpu)?.to_device(dev)?,
                Tensor::from_vec(scales, n / 16, &Device::Cpu)?.to_device(dev)?,
                1,
                [e, rows, cols],
            ))
        }
        other => bail!("inkling: expert tensor '{name}' is {other:?} (supported: NVFP4, Q8_0)"),
    }
}

impl Body {
    pub fn load(src: &dyn WeightSource, cfg: &ModelConfig, device: &Device) -> Result<Self> {
        let ic = cfg.inkling.clone().context("not an inkling2 config")?;
        let g = src.gguf().context("inkling2 loads from a GGUF file")?;
        ensure!(device.is_cuda(), "inkling2 runs on a CUDA device");
        k::warm(device)?;
        let (h, nh, nkv, hd) = (cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        ensure!(hd == 128, "inkling2: head dim {hd} (the kernels take 128)");
        ensure!(ic.conv_kernel == 4, "inkling2: conv kernel {} (the kernels take 4)", ic.conv_kernel);
        ensure!(ic.n_rel <= 16 && ic.window > 0, "inkling2: relative width {} / window {}", ic.n_rel, ic.window);
        let f32_of = |name: &str| -> Result<Tensor> {
            Ok(src.load_raw(name).with_context(|| format!("loading '{name}'"))?.to_dtype(DType::F32)?.to_device(device)?)
        };
        let flat = |name: &str| -> Result<Tensor> { Ok(f32_of(name)?.flatten_all()?) };
        let mut layers = Vec::with_capacity(ic.layers);
        let mut total = 0usize;
        for l in 0..ic.layers {
            let pfx = format!("blk.{l}.");
            let n = |s: &str| format!("{pfx}{s}");
            let swa = ic.swa[l];
            let norms = if swa {
                vec![flat(&n("attn_norm.weight"))?]
            } else {
                ["q", "k", "v"].iter().map(|x| flat(&n(&format!("attn_{x}_inp_norm.weight")))).collect::<Result<_>>()?
            };
            let proj = load_q8(
                g,
                &["attn_q", "attn_rel", "attn_gate", "attn_k", "attn_v"].map(|x| n(&format!("{x}.weight"))),
                device,
            )?;
            ensure!(proj.rows == 2 * nh * hd + nh * ic.n_rel + 2 * nkv * hd, "inkling2: layer {l} projection rows {}", proj.rows);
            let wo = load_q8(g, &[n("attn_output.weight")], device)?;
            let sink = f32_of(&n("attn_sink_proj.weight"))?;
            let small = if swa {
                sink
            } else {
                Tensor::cat(&[sink, f32_of(&n("attn_k_conv_proj.weight"))?, f32_of(&n("attn_v_conv_proj.weight"))?], 0)?
            };
            let (gu_c, gu_s, fmt, gdims) = load_expert_tensor(g, &n("ffn_gate_up_exps.weight"), device)?;
            let (dn_c, dn_s, fmt2, ddims) = load_expert_tensor(g, &n("ffn_down_exps.weight"), device)?;
            ensure!(fmt == fmt2, "inkling2: layer {l} gate_up and down experts differ in format");
            ensure!(
                gdims == [ic.num_experts, 2 * ic.expert_ff, h] && ddims == [ic.num_experts, h, ic.expert_ff],
                "inkling2: layer {l} expert shapes {gdims:?} / {ddims:?}"
            );
            let scale = |s: &str| -> Result<Vec<f32>> {
                let name = n(s);
                if g.tensors.contains_key(&name) {
                    Ok(src.load_raw(&name)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
                } else {
                    Ok(vec![1.0; ic.num_experts])
                }
            };
            let (gs_host, ds_host) = (scale("ffn_gate_up_exps.scale")?, scale("ffn_down_exps.scale")?);
            let experts = ExpertWeights {
                fmt,
                gu_codes: gu_c,
                gu_scales: gu_s,
                dn_codes: dn_c,
                dn_scales: dn_s,
                gs: Tensor::from_slice(&gs_host, gs_host.len(), device)?,
                ds: Tensor::from_slice(&ds_host, ds_host.len(), device)?,
                gs_host,
                ds_host,
                hidden: h,
                inter: ic.expert_ff,
            };
            total += proj.bytes() + wo.bytes() + experts.bytes();
            let attn_query = if g.tensors.contains_key(&n("attn_res_query.weight")) {
                Some(f32_of(&n("attn_res_query.weight"))?.reshape(((), h))?)
            } else {
                None
            };
            layers.push(Layer {
                swa,
                norms,
                proj,
                wo,
                small,
                k_conv: f32_of(&n("attn_k_conv.weight"))?,
                v_conv: f32_of(&n("attn_v_conv.weight"))?,
                post_conv: f32_of(&n("attn_post_conv.weight"))?,
                ffn_conv: f32_of(&n("ffn_post_conv.weight"))?,
                q_clip: flat(&n("attn_q_clip.weight"))?,
                rel_clip: flat(&n("attn_rel_clip.weight"))?,
                rel_bias: f32_of(&n("attn_rel_bias.weight"))?,
                attn_query,
                ffn_query: f32_of(&n("ffn_res_query.weight"))?.reshape((1, h))?,
                ffn_norm: flat(&n("ffn_norm.weight"))?,
                router: f32_of(&n("ffn_gate_inp.weight"))?,
                router_bias: if g.tensors.contains_key(&n("exp_probs_b.bias")) { Some(flat(&n("exp_probs_b.bias"))?) } else { None },
                experts,
            });
            if l % 4 == 3 {
                tracing::info!("inkling2: loaded layers 0..{} ({:.1} GiB)", l + 1, total as f64 / (1u64 << 30) as f64);
            }
        }
        let mut loops = vec![];
        if ic.loop_count > 1 {
            for l in ic.loop_start..ic.loop_end {
                let n = |s: &str| format!("blk.{l}.loop_{s}.weight");
                let ctl = |part: &str| -> Result<[Tensor; 4]> {
                    Ok([
                        f32_of(&n(&format!("q_ctrl_{part}")))?,
                        f32_of(&n(&format!("k_ctrl_{part}")))?,
                        f32_of(&n(&format!("v_ctrl_{part}")))?,
                        f32_of(&n(&format!("ffn_ctrl_{part}")))?,
                    ])
                };
                loops.push(LoopLayer {
                    msg_query: f32_of(&n("msg_query"))?.reshape((4, h))?,
                    down: ctl("down")?,
                    gain: ctl("gain")?,
                    forget: ctl("forget")?,
                });
            }
        }
        let msg_queries = if loops.is_empty() {
            Tensor::zeros((4, h), DType::F32, device)?
        } else {
            Tensor::cat(&loops.iter().map(|l| l.msg_query.clone()).collect::<Vec<_>>(), 0)?
        };
        let mut slots = Vec::with_capacity(ic.visits());
        let (mut n_global, mut n_ring) = (0, 0);
        for il in 0..ic.visits() {
            if ic.swa[ic.phys(il)] {
                slots.push(Cache::Ring(n_ring));
                n_ring += 1;
            } else {
                slots.push(Cache::Global(n_global));
                n_global += 1;
            }
        }
        tracing::info!(
            "inkling2: {} layers, core {}..{} x{} ({} visits: {n_global} full attention, {n_ring} sliding window {}), {:.1} GiB of weights",
            ic.layers,
            ic.loop_start,
            ic.loop_end,
            ic.loop_count,
            ic.visits(),
            ic.window,
            total as f64 / (1u64 << 30) as f64
        );
        Ok(Self {
            hidden: h,
            nh,
            nkv,
            eps: cfg.rms_norm_eps as f32,
            layers,
            loops,
            out_query: f32_of("output_res_query.weight")?.reshape((1, h))?,
            msg_queries,
            slots,
            n_global,
            n_ring,
            caches: RwLock::new(Caches::default()),
            graphs: Default::default(),
            device: device.clone(),
            ic,
        })
    }

    /// The KV pool format: the activation-dtype pools are bf16 here.
    fn q8_pool() -> Result<bool> {
        match crate::kvcache::kv_format() {
            crate::kvcache::KvFormat::Model => Ok(false),
            crate::kvcache::KvFormat::Q8 => Ok(true),
            f => bail!("inkling2: --kv-dtype {f:?} is not supported (use bf16 or q8)"),
        }
    }

    /// Full-attention cache slots (one KV pool each).
    pub fn global_slots(&self) -> usize {
        self.n_global
    }

    pub fn kv_bytes_per_token(&self) -> usize {
        let per = if Self::q8_pool().unwrap_or(false) { 128 + 4 } else { 256 };
        self.n_global * 2 * self.nkv * per
    }

    /// Bytes of one sequence's recurrent state: the rings and the conv states.
    pub fn state_bytes(&self) -> usize {
        let ring = 2 * self.ic.window * self.nkv * 128 * 2;
        let conv = 3 * (2 * self.nkv * 128 + 2 * self.hidden) * 4;
        self.n_ring * ring + self.slots.len() * conv
    }

    pub fn alloc(&self, num_slots: usize, num_states: usize) -> Result<()> {
        *self.graphs.lock() = Graphs::default();
        let mut c = self.caches.write();
        *c = Caches::default();
        let q8 = Self::q8_pool()?;
        let states = num_states.max(1);
        let w = self.nkv * 128;
        for _ in 0..self.n_global {
            let kv = if q8 {
                let z = || Tensor::zeros((num_slots, self.nkv * 132), DType::U8, &self.device);
                crate::kvcache::LayerKv::new_raw(z()?, z()?, DType::BF16, crate::kvcache::KvFormat::Q8)
            } else {
                let z = || Tensor::zeros((num_slots, w), DType::BF16, &self.device);
                crate::kvcache::LayerKv::new_raw(z()?, z()?, DType::BF16, crate::kvcache::KvFormat::Model)
            };
            c.kv.push(kv);
        }
        for _ in 0..self.n_ring {
            let z = || Tensor::zeros((states * self.ic.window, w), DType::BF16, &self.device);
            c.rings.push((z()?, z()?));
        }
        let ctot = 2 * w + 2 * self.hidden;
        for _ in 0..self.slots.len() {
            c.conv.push(Tensor::zeros((states, 3, ctot), DType::F32, &self.device)?);
        }
        c.num_slots = num_slots;
        c.num_states = states;
        Ok(())
    }

    pub fn kv_slots(&self) -> usize {
        self.caches.read().num_slots
    }

    pub fn state_slots(&self) -> usize {
        self.caches.read().num_states
    }

    pub fn copy_state(&self, src: u32, dst: u32) -> Result<()> {
        if src == dst {
            return Ok(());
        }
        let c = self.caches.read();
        let win = self.ic.window;
        for (kr, vr) in &c.rings {
            for t in [kr, vr] {
                let rows = t.narrow(0, src as usize * win, win)?;
                t.slice_set(&rows.copy()?, 0, dst as usize * win)?;
            }
        }
        for t in &c.conv {
            let row = t.narrow(0, src as usize, 1)?.copy()?;
            t.slice_set(&row, 0, dst as usize)?;
        }
        Ok(())
    }

    /// Hidden states `[T, H]` (f32) after the stack and the output block mix, ready for the
    /// final norm; `x` holds the raw token embeddings.
    pub fn forward(&self, meta: &BatchMeta, x: &Tensor) -> Result<Tensor> {
        let c = self.caches.read();
        ensure!(c.conv.len() == self.slots.len(), "inkling2: KV cache not allocated");
        // `x`: token embeddings already normalized by the head (image rows as they are)
        let x = x.to_device(&self.device)?.to_dtype(DType::F32)?.contiguous()?;
        if let Some(y) = self.forward_graph(meta, &x, &c)? {
            return Ok(y);
        }
        let b = Batch::build(meta, &self.device)?;
        self.run(&x, &b, &c)
    }

    /// Decode steps (one new token per sequence, up to `GRAPH_MAX_BATCH` sequences) replay a
    /// CUDA graph captured per batch size over static tables: the kernels read lengths from
    /// device tables only, the slot table is sized to the KV pool and attention always splits
    /// for the pool's length. `Ok(None)`: run eagerly.
    fn forward_graph(&self, meta: &BatchMeta, x: &Tensor, c: &Caches) -> Result<Option<Tensor>> {
        let bsz = meta.seqs.len();
        if !graphs_enabled() || bsz == 0 || bsz > GRAPH_MAX_BATCH || meta.seqs.iter().any(|s| s.q_len != 1) {
            return Ok(None);
        }
        let Device::Cuda(dev) = &self.device else { return Ok(None) };
        let mut g = self.graphs.lock();
        if g.disabled {
            return Ok(None);
        }
        let total: usize = meta.seqs.iter().map(|s| s.kv_len).sum();
        if total > c.num_slots {
            return Ok(None); // shared prefixes outgrow the static slot table: eagerly
        }
        if let Some(e) = g.entries.get(&bsz) {
            let t0 = std::time::Instant::now();
            e.batch.refresh(meta)?;
            e.x_in.slice_set(x, 0, 0)?;
            e.graph.launch()?;
            let y = e.y_out.copy()?;
            if std::env::var("CARBON_INK_TIMING").is_ok() {
                dev.cuda_stream().synchronize()?;
                tracing::info!("inkling2: graph step {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
            }
            return Ok(Some(y));
        }
        let _htod = dev.enable_cuda_graph_htod_cache();
        let seen = g.seen.entry(bsz).or_insert(0);
        *seen += 1;
        let static_batch = Batch::graph_static(meta, c.num_slots, &self.device)?;
        if *seen < 2 {
            // warm-up: modules load, the upload cache fills
            return Ok(Some(self.run(x, &static_batch, c)?));
        }
        // the eager result of this step (its conv state rolled back after) to check the capture
        let saved: Vec<Tensor> = c.conv.iter().map(|t| t.copy()).collect::<candle_core::Result<_>>()?;
        let reference = self.run(x, &static_batch, c)?;
        for (t, s) in c.conv.iter().zip(&saved) {
            t.slice_set(s, 0, 0)?;
        }
        let x_in = x.copy()?;
        let y_out = reference.zeros_like()?;
        let stream = dev.cuda_stream();
        stream.synchronize()?;
        stream.begin_capture(candle_core::cuda_backend::cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        let recorded = self.run(&x_in, &static_batch, c).and_then(|y| Ok(y_out.slice_set(&y, 0, 0)?));
        let graph = stream.end_capture(candle_core::cuda_backend::cudarc::driver::sys::CUgraphInstantiate_flags(0));
        let outcome = (|| -> Result<_> {
            recorded?;
            let graph = graph?.context("empty capture")?;
            graph.upload()?;
            // the graph recomputes the step the reference ran: same state in, same output
            for (t, s) in c.conv.iter().zip(&saved) {
                t.slice_set(s, 0, 0)?;
            }
            graph.launch()?;
            let y = y_out.copy()?;
            let d = (&y - &reference)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let m = reference.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
            ensure!(d / m <= 1e-3, "the captured decode graph (batch {bsz}) differs from eager decoding by {:.3e}", d / m);
            Ok((graph, y))
        })();
        match outcome {
            Ok((graph, y)) => {
                tracing::info!("inkling2: captured a cuda graph for decode batch {bsz}");
                g.entries.insert(bsz, GraphEntry { graph, x_in, y_out, batch: static_batch });
                Ok(Some(y))
            }
            Err(e) => {
                tracing::warn!("inkling2: cuda graphs disabled, decoding eagerly: {e:#}");
                g.disabled = true;
                g.entries.clear();
                for (t, s) in c.conv.iter().zip(&saved) {
                    t.slice_set(s, 0, 0)?;
                }
                Ok(Some(self.run(x, &static_batch, c)?))
            }
        }
    }

    fn run(&self, x: &Tensor, b: &Batch, c: &Caches) -> Result<Tensor> {
        let tb = b.tables();
        let mut cur = x.clone();
        let mut blocks: Vec<Tensor> = vec![];
        let ic = &self.ic;
        let (start, end) = if ic.loop_count > 1 { (ic.loop_start, ic.loop_end) } else { (ic.layers, ic.layers) };
        for ip in 0..start {
            cur = self.layer(ip, ip, cur, &mut blocks, None, &b, &tb, &c)?;
        }
        let prelude = cur.clone();
        let prelude_blocks = blocks.clone();
        for pass in 0..ic.loop_count.max(1) {
            if start == end {
                break;
            }
            // the previous pass's blocks after its first (the prelude output) and its output
            let bank = if pass > 0 {
                let states: Vec<Tensor> = blocks[prelude_blocks.len() + 1..].to_vec();
                let out = std::mem::replace(&mut cur, prelude.clone());
                blocks = prelude_blocks.clone();
                Some((states, out))
            } else {
                None
            };
            // every core layer's messages of this pass in one mix: [layers * 4, T, H]
            let all_msgs = match &bank {
                Some((states, out)) => Some(k::mix(states, out, &self.msg_queries, false)?.0),
                None => None,
            };
            for ip in start..end {
                let il = if pass == 0 { ip } else { ic.layers + (pass - 1) * (end - start) + (ip - start) };
                let msgs = match &all_msgs {
                    Some(m) => Some(m.narrow(0, 4 * (ip - start), 4)?),
                    None => None,
                };
                cur = self.layer(ip, il, cur, &mut blocks, msgs.as_ref(), &b, &tb, &c)?;
            }
        }
        for ip in end..ic.layers {
            cur = self.layer(ip, ip, cur, &mut blocks, None, &b, &tb, &c)?;
        }
        Ok(k::mix(&blocks, &cur, &self.out_query, false)?.0.get(0)?)
    }

    /// Cross-pass controller: `(1 - forget) cur + gain msg`, gated on both.
    fn control(&self, cur: &Tensor, msg: &Tensor, lp: &LoopLayer, j: usize) -> Result<Tensor> {
        let gate = Tensor::cat(&[k::rms(cur, None, MIX_EPS)?, k::rms(msg, None, MIX_EPS)?], 1)?;
        let gate = gate.matmul(&lp.down[j].t()?)?;
        let gain = gate.matmul(&lp.gain[j].t()?)?.tanh()?;
        let forget = gate.matmul(&lp.forget[j].t()?)?.tanh()?;
        Ok(((cur - forget.mul(cur)?)? + gain.mul(msg)?)?)
    }

    /// Loop controllers `j` of a core layer on (input, message) pairs.
    fn controls(&self, lp: &LoopLayer, pairs: &[(usize, &Tensor, Tensor)]) -> Result<Vec<Tensor>> {
        let t = pairs[0].1.dim(0)?;
        if t > SMALL_T {
            return pairs.iter().map(|(j, x, m)| self.control(x, m, lp, *j)).collect();
        }
        let items: Vec<k::CtrlIn<'_>> = pairs
            .iter()
            .map(|(j, x, m)| k::CtrlIn { cur: x, msg: m, down: &lp.down[*j], gain: &lp.gain[*j], forget: &lp.forget[*j] })
            .collect();
        k::control(&items)
    }

    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        ip: usize,
        il: usize,
        cur: Tensor,
        blocks: &mut Vec<Tensor>,
        msgs: Option<&Tensor>,
        b: &Batch,
        tb: &Tables<'_>,
        c: &Caches,
    ) -> Result<Tensor> {
        let ly = &self.layers[ip];
        let (nh, nkv, ic) = (self.nh, self.nkv, &self.ic);
        let t = cur.dim(0)?;
        let state = &c.conv[il];
        let boundary = ip % ic.block_size == 0;
        let same = ly.swa && msgs.is_none();
        // without a message the input norms are applied by the mix kernel itself
        let fused_norms: Vec<&Tensor> = if msgs.is_none() { ly.norms.iter().collect() } else { vec![] };
        let decode = t <= k::GEMV_MAX_T;
        let mut qins: Vec<k::QAct> = vec![];
        let (mut ins, cur_n): ([Tensor; 3], Option<Tensor>) = match &ly.attn_query {
            Some(q) => {
                let (m, cn, qs) = k::mix_full(blocks, &cur, q, boundary, &fused_norms, self.eps, decode && msgs.is_none())?;
                qins = qs;
                let pick = |j: usize| m.get(if ly.swa { 0 } else { j });
                ([pick(0)?, pick(1)?, pick(2)?], cn)
            }
            None => {
                let n = |j: usize| k::rms(&cur, Some(&ly.norms[if ly.swa { 0 } else { j }]), self.eps);
                let ins = if same { let x = n(0)?; [x.clone(), x.clone(), x] } else { [n(0)?, n(1)?, n(2)?] };
                (ins, if boundary { Some(k::rms(&cur, None, MIX_EPS)?) } else { None })
            }
        };
        let lp = msgs.map(|_| &self.loops[ip - ic.loop_start]);
        if let (Some(m), Some(lp)) = (msgs, lp) {
            let pairs = [(0, &ins[0], m.get(0)?), (1, &ins[1], m.get(1)?), (2, &ins[2], m.get(2)?)];
            let out = self.controls(lp, &pairs)?;
            ins = [out[0].clone(), out[1].clone(), out[2].clone()];
            for (j, x) in ins.iter_mut().enumerate() {
                *x = k::rms(x, Some(&ly.norms[if ly.swa { 0 } else { j }]), self.eps)?;
            }
        }
        // projections: q | rel | gate from input 0, k from 1, v from 2
        let (qw, rw, kw) = (nh * 128, nh * ic.n_rel, nkv * 128);
        let (q_off, r_off, g_off, k_off, v_off) = (0, qw, qw + rw, 2 * qw + rw, 2 * qw + rw + kw);
        // the small f32 rows: sinks (input 0) and, on full attention layers, the K (1) and V (2) conv taps
        let taps = nkv * ic.conv_kernel;
        let log = (!ly.swa).then_some((ic.log_alpha, ic.log_floor));
        let (y, small, q, r, kk, vv) = if decode {
            let y = k::empty((t, ly.proj.rows), DType::F32, &self.device)?;
            let small = k::empty((t, ly.small.dim(0)?), DType::F32, &self.device)?;
            if qins.is_empty() {
                qins.push(k::quant_x(&ins[0])?);
                if !same {
                    qins.push(k::quant_x(&ins[1])?);
                    qins.push(k::quant_x(&ins[2])?);
                }
            }
            let (q0, q1, q2) = if qins.len() == 1 { (&qins[0], &qins[0], &qins[0]) } else { (&qins[0], &qins[1], &qins[2]) };
            let tail = k::F32Tail { w: &ly.small, xs: [&ins[0], &ins[1], &ins[2]], r1: nh, r2: nh + taps, y: &small };
            k::q8_gemv_tail(&ly.proj, [q0, q1, q2], k_off, v_off, &y, Some(tail))?;
            let dst = match self.slots[il] {
                Cache::Global(gi) => k::KvDst::Pool { k: &c.kv[gi].k, v: &c.kv[gi].v, q8: c.kv[gi].packed() },
                Cache::Ring(ri) => k::KvDst::Ring { k: &c.rings[ri].0, v: &c.rings[ri].1, len: ic.window },
            };
            let tp = (!ly.swa).then_some((&small, nh, nh + taps));
            let (kk, vv) = k::kv_prep(&y, k_off, v_off, tp, &ly.k_conv, &ly.v_conv, state, nkv, ic.qk_clip, self.eps, dst, tb)?;
            let (q, r) = k::qr_prep(&y, q_off, r_off, &ly.q_clip, &ly.rel_clip, nh, ic.n_rel, ic.qk_clip, ic.rel_clip, log, tb.positions)?;
            (y, small, q, r, kk, vv)
        } else {
            let wd = k::q8_dequant(&ly.proj)?;
            let part = |x: &Tensor, a: usize, n: usize| -> Result<Tensor> {
                Ok(x.to_dtype(DType::BF16)?.matmul(&wd.narrow(0, a, n)?.t()?)?.to_dtype(DType::F32)?)
            };
            let y = if same {
                part(&ins[0], 0, ly.proj.rows)?
            } else {
                Tensor::cat(&[part(&ins[0], 0, k_off)?, part(&ins[1], k_off, kw)?, part(&ins[2], v_off, kw)?], 1)?
            };
            let small = if ly.swa {
                ins[0].matmul(&ly.small.t()?)?
            } else {
                Tensor::cat(
                    &[
                        ins[0].matmul(&ly.small.narrow(0, 0, nh)?.t()?)?,
                        ins[1].matmul(&ly.small.narrow(0, nh, taps)?.t()?)?,
                        ins[2].matmul(&ly.small.narrow(0, nh + taps, taps)?.t()?)?,
                    ],
                    1,
                )?
            };
            let (kd, vd) = if ly.swa {
                (None, None)
            } else {
                (Some(small.narrow(1, nh, taps)?.contiguous()?), Some(small.narrow(1, nh + taps, taps)?.contiguous()?))
            };
            let kc = k::conv_par(&y, k_off, kw, &ly.k_conv, kd.as_ref(), nkv, state, 0, None, 0, tb)?;
            let vc = k::conv_par(&y, v_off, kw, &ly.v_conv, vd.as_ref(), nkv, state, kw, None, 0, tb)?;
            let q = k::rownorm(&y, q_off, nh, 128, Some(&ly.q_clip), RowNorm::Clip(ic.qk_clip), log, tb.positions)?;
            let kk = k::rownorm(&kc, 0, nkv, 128, None, RowNorm::Clip(ic.qk_clip), None, tb.positions)?;
            let vv = k::rownorm(&vc, 0, nkv, 128, None, RowNorm::Rms(self.eps), None, tb.positions)?;
            let r = k::rownorm(&y, r_off, nh, ic.n_rel, Some(&ly.rel_clip), RowNorm::Clip(ic.rel_clip), log, tb.positions)?;
            (y, small.contiguous()?, q, r, kk, vv)
        };
        let (old, window, max_keys) = match self.slots[il] {
            Cache::Global(gi) => {
                let kv = &c.kv[gi];
                (KvSource::Pool { k: &kv.k, v: &kv.v, q8: kv.packed() }, 0, b.max_kv)
            }
            Cache::Ring(ri) => {
                let (kr, vr) = &c.rings[ri];
                (KvSource::Ring { k: kr, v: vr, len: ic.window }, ic.window, b.max_kv.min(ic.window))
            }
        };
        let args = k::AttnArgs {
                q: &q,
                r: &r,
                sink: &small,
                k: &kk,
                v: &vv,
                old,
                relb: &ly.rel_bias,
                nkv,
                window,
                max_keys,
                gate: Some((&y, g_off)),
                quant: decode,
            };
        let (a, aq) = if !decode && k::flash_supported(&self.device) {
            (k::flash(&args, tb)?, None)
        } else {
            k::attention_q(&args, tb)?
        };
        if !decode {
            match self.slots[il] {
                Cache::Global(gi) => {
                    let kv = &c.kv[gi];
                    k::pool_write(&kk, &kv.k, kv.packed(), nkv, tb)?;
                    k::pool_write(&vv, &kv.v, kv.packed(), nkv, tb)?;
                }
                Cache::Ring(ri) => {
                    let (kr, vr) = &c.rings[ri];
                    k::ring_write(&kk, kr, ic.window, tb)?;
                    k::ring_write(&vv, vr, ic.window, tb)?;
                }
            }
        }
        let o = match &aq {
            Some(aq) => {
                let o = k::empty((t, self.hidden), DType::F32, &self.device)?;
                k::q8_gemv(&ly.wo, [aq, aq, aq], usize::MAX >> 1, usize::MAX >> 1, &o)?;
                o
            }
            None => k::q8_matmul(&ly.wo, &a)?,
        };
        let res = if boundary { None } else { Some(&cur) };
        let post = if decode {
            k::conv(&o, 0, self.hidden, &ly.post_conv, None, 1, state, 2 * kw, res, tb)?
        } else {
            k::conv_par(&o, 0, self.hidden, &ly.post_conv, None, 1, state, 2 * kw, res, 0, tb)?
        };
        if boundary {
            // the layer input closes a block: it joins the mixed states, the stream restarts
            blocks.push(cur_n.context("boundary layer without its normalized input")?);
        }
        let cur = post;
        let (hn, hq) = match (msgs, lp) {
            (Some(m), Some(lp)) => {
                let hn = k::mix(blocks, &cur, &ly.ffn_query, false)?.0.get(0)?;
                let hn = self.controls(lp, &[(3, &hn, m.get(3)?)])?.remove(0);
                (k::rms(&hn, Some(&ly.ffn_norm), self.eps)?, None)
            }
            _ => {
                let (o, _, mut qs) = k::mix_full(blocks, &cur, &ly.ffn_query, false, &[&ly.ffn_norm], self.eps, decode)?;
                (o.get(0)?, qs.pop())
            }
        };
        let (ys, nsum) = self.moe(ly, &hn, hq.as_ref())?;
        if decode {
            k::conv_n(&ys, 0, self.hidden, &ly.ffn_conv, None, 1, state, 2 * kw + self.hidden, Some(&cur), nsum, tb)
        } else {
            k::conv_par(&ys, 0, self.hidden, &ly.ffn_conv, None, 1, state, 2 * kw + self.hidden, Some(&cur), nsum, tb)
        }
    }

    /// Routed experts: the output, or (decode) the K weighted parts per token to be summed (the
    /// second value is K then).
    fn moe(&self, ly: &Layer, x: &Tensor, xq: Option<&k::QAct>) -> Result<(Tensor, usize)> {
        let ic = &self.ic;
        let t = x.dim(0)?;
        let logits = if t <= k::GEMV_MAX_T { k::f32_gemv(&ly.router, [x, x, x], usize::MAX >> 1, usize::MAX >> 1)? } else { x.matmul(&ly.router.t()?)? };
        let (ids, wts) = k::route(&logits, ly.router_bias.as_ref(), ic.top_k, ic.expert_weights_scale)?;
        if t * ic.top_k <= k::MOE_GEMV_MAX_PAIRS {
            let parts = k::moe_gemv_parts(&ly.experts, x, xq, &ids, &wts, ic.situ_beta, ic.situ_linear_beta, true)?;
            return Ok((parts, ic.top_k));
        }
        // prefill: the tokens of each expert through dense GEMMs on its dequantized weights
        let ew = &ly.experts;
        let ids_h = ids.flatten_all()?.to_vec1::<u32>()?;
        let wts_h = wts.flatten_all()?.to_vec1::<f32>()?;
        // per expert: (token, routed pair j = token * K + slot, weight)
        let mut per: Vec<Vec<(u32, u32, f32)>> = vec![vec![]; ic.num_experts];
        for (j, (&e, &w)) in ids_h.iter().zip(&wts_h).enumerate() {
            per[e as usize].push(((j / ic.top_k) as u32, j as u32, w));
        }
        let gu_w = k::empty((2 * ew.inter, ew.hidden), DType::BF16, &self.device)?;
        let dn_w = k::empty((ew.hidden, ew.inter), DType::BF16, &self.device)?;
        let xb = x.to_dtype(DType::BF16)?;
        // every pair belongs to one expert, so the parts are all written
        let parts = k::empty((t * ic.top_k, self.hidden), DType::F32, &self.device)?;
        for (e, list) in per.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            let rows: Vec<u32> = list.iter().map(|r| r.0).collect();
            let pairs: Vec<u32> = list.iter().map(|r| r.1).collect();
            let w: Vec<f32> = list.iter().map(|r| r.2).collect();
            let idx = Tensor::from_vec(rows, list.len(), &self.device)?;
            let pidx = Tensor::from_vec(pairs, list.len(), &self.device)?;
            let xe = xb.index_select(&idx, 0)?;
            k::expert_dequant(ew, e, false, &gu_w)?;
            let gu = xe.matmul(&gu_w.t()?)?.to_dtype(DType::F32)?;
            let act = k::situ_rows(&gu, ic.situ_beta, ic.situ_linear_beta)?;
            k::expert_dequant(ew, e, true, &dn_w)?;
            let d = act.to_dtype(DType::BF16)?.matmul(&dn_w.t()?)?.to_dtype(DType::F32)?;
            let wv = Tensor::from_vec(w, list.len(), &self.device)?;
            k::scatter_rows(&d, &pidx, &wv, &parts)?;
        }
        Ok((parts, ic.top_k))
    }
}

/// The batch's device tables.
struct Batch {
    conv_seqs: Tensor,
    seqs: Tensor,
    rowseq: Tensor,
    slots: Tensor,
    positions: Tensor,
    out_loc: Tensor,
    tiles: Tensor,
    num_tiles: usize,
    num_seqs: usize,
    max_kv: usize,
}

impl Batch {
    fn build(meta: &BatchMeta, dev: &Device) -> Result<Self> {
        let t = meta.num_tokens();
        let (mut conv, mut seqs, mut rowseq, mut slots) = (vec![], vec![], Vec::with_capacity(t), vec![]);
        let mut tiles = vec![];
        let mut q0 = 0u32;
        for (i, s) in meta.seqs.iter().enumerate() {
            if !s.token_state_slots.is_empty() {
                bail!("inkling2: per-token state slots (speculative decoding) are not supported");
            }
            let fresh = u32::from(s.kv_len == s.q_len);
            conv.extend_from_slice(&[q0, s.q_len as u32, s.state_slot, fresh]);
            seqs.extend_from_slice(&[q0, s.q_len as u32, s.kv_len as u32, slots.len() as u32, s.state_slot]);
            slots.extend_from_slice(&s.slots[..s.kv_len]);
            rowseq.extend(std::iter::repeat_n(i as u32, s.q_len));
            for j in (0..s.q_len).step_by(64) {
                tiles.extend_from_slice(&[i as u32, j as u32]);
            }
            q0 += s.q_len as u32;
        }
        ensure!(rowseq.len() == t, "inkling2: batch rows");
        if slots.is_empty() {
            slots.push(0);
        }
        let b = meta.seqs.len();
        Ok(Self {
            conv_seqs: Tensor::from_vec(conv, b * 4, dev)?,
            seqs: Tensor::from_vec(seqs, b * 5, dev)?,
            rowseq: Tensor::from_vec(rowseq, t, dev)?,
            slots: Tensor::from_slice(&slots, slots.len(), dev)?,
            positions: Tensor::from_slice(&meta.positions, t, dev)?,
            out_loc: Tensor::from_slice(&meta.out_loc, t, dev)?,
            num_tiles: tiles.len() / 2,
            tiles: Tensor::from_slice(&tiles, tiles.len(), dev)?,
            num_seqs: b,
            max_kv: meta.seqs.iter().map(|s| s.kv_len).max().unwrap_or(0),
        })
    }

    /// Host contents of a decode batch's tables, the slot table padded to `cap`.
    fn host(meta: &BatchMeta, cap: usize) -> Result<[Vec<u32>; 6]> {
        let (mut conv, mut seqs, mut rowseq, mut slots) = (vec![], vec![], vec![], vec![]);
        for (i, s) in meta.seqs.iter().enumerate() {
            ensure!(s.q_len == 1 && s.token_state_slots.is_empty(), "inkling2: not a plain decode step");
            conv.extend_from_slice(&[i as u32, 1, s.state_slot, u32::from(s.kv_len == 1)]);
            seqs.extend_from_slice(&[i as u32, 1, s.kv_len as u32, slots.len() as u32, s.state_slot]);
            slots.extend_from_slice(&s.slots[..s.kv_len]);
            rowseq.push(i as u32);
        }
        ensure!(slots.len() <= cap, "inkling2: slot table larger than the pool");
        slots.resize(cap.max(1), 0);
        Ok([conv, seqs, rowseq, slots, meta.positions.clone(), meta.out_loc.clone()])
    }

    /// A decode batch whose tables a captured graph reads (`refresh` rewrites them in place).
    fn graph_static(meta: &BatchMeta, cap: usize, dev: &Device) -> Result<Self> {
        let [conv, seqs, rowseq, slots, positions, out_loc] = Self::host(meta, cap)?;
        let v = |x: Vec<u32>| -> Result<Tensor> {
            let n = x.len();
            Ok(Tensor::from_vec(x, n, dev)?)
        };
        Ok(Self {
            conv_seqs: v(conv)?,
            seqs: v(seqs)?,
            rowseq: v(rowseq)?,
            slots: v(slots)?,
            positions: v(positions)?,
            out_loc: v(out_loc)?,
            tiles: Tensor::zeros(2, DType::U32, dev)?,
            num_tiles: 0,
            num_seqs: meta.seqs.len(),
            max_kv: cap,
        })
    }

    fn refresh(&self, meta: &BatchMeta) -> Result<()> {
        let host = Self::host(meta, self.slots.dim(0)?)?;
        let dev = self.slots.device();
        for (t, x) in [&self.conv_seqs, &self.seqs, &self.rowseq, &self.slots, &self.positions, &self.out_loc].into_iter().zip(host) {
            ensure!(t.dim(0)? == x.len(), "inkling2: batch does not fit the captured graph");
            let n = x.len();
            t.slice_set(&Tensor::from_vec(x, n, dev)?, 0, 0)?;
        }
        Ok(())
    }

    fn tables(&self) -> Tables<'_> {
        Tables {
            conv_seqs: &self.conv_seqs,
            seqs: &self.seqs,
            rowseq: &self.rowseq,
            slots: &self.slots,
            positions: &self.positions,
            out_loc: &self.out_loc,
            tiles: &self.tiles,
            num_tiles: self.num_tiles,
            num_seqs: self.num_seqs,
        }
    }
}

/// The LM head: final norm, Q8 projection, softcap, the padded vocabulary dropped.
pub struct Head {
    /// The token embeddings' RMS norm (image rows skip it).
    tok_norm: Tensor,
    norm: Tensor,
    eps: f32,
    lm: Q8,
    softcap: f32,
    vocab: usize,
}

impl Head {
    pub fn load(src: &dyn WeightSource, cfg: &ModelConfig, device: &Device) -> Result<Self> {
        let ic = cfg.inkling.as_ref().context("not an inkling2 config")?;
        let g = src.gguf().context("inkling2 loads from a GGUF file")?;
        let lm = load_q8(g, &["output.weight".to_string()], device)?;
        ensure!(lm.rows == ic.vocab_padded && cfg.vocab_size <= lm.rows, "inkling2: LM head has {} rows", lm.rows);
        Ok(Self {
            tok_norm: src.load_raw("token_embd_norm.weight")?.to_dtype(DType::F32)?.flatten_all()?.to_device(device)?,
            norm: src.load_raw("output_norm.weight")?.to_dtype(DType::F32)?.to_device(device)?,
            eps: cfg.rms_norm_eps as f32,
            lm,
            softcap: ic.softcap,
            vocab: cfg.vocab_size,
        })
    }

    /// Token embeddings `[T, H]` through the embedding norm.
    pub fn embed_norm(&self, e: &Tensor) -> Result<Tensor> {
        k::rms(&e.to_dtype(DType::F32)?, Some(&self.tok_norm), self.eps)
    }

    /// f32 logits `[R, vocab]` of rows `[R, H]` of the stack's output.
    pub fn logits(&self, rows: &Tensor) -> Result<Tensor> {
        let x = k::rms(&rows.to_dtype(DType::F32)?, Some(&self.norm), self.eps)?;
        let r = x.dim(0)?;
        let mut parts = vec![];
        let mut i = 0;
        while i < r {
            let n = (r - i).min(k::GEMV_MAX_T);
            parts.push(k::q8_matmul(&self.lm, &x.narrow(0, i, n)?.contiguous()?)?);
            i += n;
        }
        let z = if parts.len() == 1 { parts.pop().unwrap() } else { Tensor::cat(&parts, 0)? };
        let z = z.narrow(D::Minus1, 0, self.vocab)?;
        let z = if self.softcap > 0.0 { ((z / self.softcap as f64)?.tanh()? * self.softcap as f64)? } else { z };
        Ok(z.contiguous()?)
    }
}
