// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Transformer model pieces: a contiguous range of decoder layers on one device
//! (one tensor-parallel shard) and the embedding / LM head.

pub mod attention;
#[cfg(feature = "cuda")]
pub mod bank;
pub mod gdn;
pub mod graph;
pub mod hyper;
pub mod layers;
pub mod moe;
pub mod mtp;
pub mod vision;

use crate::batch::BatchMeta;
use crate::config::{LayerKind, ModelConfig};
use crate::kvcache::LayerKv;
use crate::parallel::comm::Communicator;
use crate::weights::WeightSource;
use anyhow::{bail, Result};
use attention::{Attention, DeviceBatch};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::Linear;
use gdn::{GatedDeltaNet, LinearState};
use layers::{Mlp, Placement, RmsNorm, Rotary};
use moe::{ExpertCache, ExpertStats, MoeBlock, MoeOptions, MoeStrategy};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::sync::Arc;

pub enum Ffn {
    Dense(Mlp),
    Moe(Box<MoeBlock>),
}

pub enum Mixer {
    /// Softmax attention; the index selects this layer's KV pool.
    Attn(Attention, usize),
    /// Gated DeltaNet; the index selects this layer's recurrent-state pool.
    Linear(GatedDeltaNet, usize),
}

/// How a layer reads and writes the residual stream.
pub enum Residual {
    /// Pre-norm transformer: `x += mixer(norm(x))`, `x += ffn(norm(x))`.
    Plain { input_ln: RmsNorm, post_ln: RmsNorm },
    /// Qwen3.8 hyper-connections over `hc_count` streams, PLE first on some layers (the index
    /// selects the stack's PLE conv-state pool).
    Hyper {
        attn: hyper::GatedResidual,
        mlp: hyper::GatedResidual,
        ple: Option<Box<(hyper::Ple, usize)>>,
    },
}

pub struct DecoderLayer {
    residual: Residual,
    mixer: Mixer,
    ffn: Ffn,
}

/// Device caches of a stack: KV pools for attention layers, state pools for linear layers.
#[derive(Default)]
struct Caches {
    kv: Vec<LayerKv>,
    states: Vec<LinearState>,
    /// PLE conv inputs `[slots, len, hc*hidden]` per PLE layer (Qwen3.8).
    ple: Vec<Tensor>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StageStats {
    pub name: String,
    pub layers: (usize, usize),
    pub devices: Vec<String>,
    pub kv_slots: usize,
    #[serde(default)]
    pub state_slots: usize,
    pub moe_strategy: Option<String>,
    pub experts: Option<ExpertStats>,
    /// Decode batch sizes currently replayed from captured CUDA graphs.
    #[serde(default)]
    pub cuda_graphs: usize,
    /// Why CUDA graphs were turned off for this stage, if they were.
    #[serde(default)]
    pub cuda_graphs_disabled: Option<String>,
}

pub struct LayerStack {
    pub range: Range<usize>,
    cfg: Arc<ModelConfig>,
    layers: Vec<DecoderLayer>,
    rotary: Rotary,
    caches: RwLock<Caches>,
    n_attn: usize,
    n_linear: usize,
    n_ple: usize,
    /// Indexer key width of QSA layers (Qwen3.8), 0 otherwise.
    index_dim: usize,
    pub placement: Placement,
    comm: Arc<dyn Communicator>,
    expert_cache: Option<Arc<ExpertCache>>,
    /// Offloaded experts behind a device slot cache (`bank.rs`), instead of `expert_cache`.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    expert_bank: Option<moe::Bank>,
    moe_strategy: Option<MoeStrategy>,
    kv_width: usize,
    /// Bytes of one expert as the device cache holds it.
    expert_bytes: usize,
    /// Bytes of one recurrent-state slot across this stage's GDN and PLE layers.
    state_bytes: usize,
    /// State slots the engine will allocate (`MoeOptions::state_slots`), kept out of the KV pool.
    planned_states: usize,
    /// Captured decode steps (`--cuda-graphs`).
    graphs: graph::DecodeGraphs,
}

/// Device memory an expert slot cache leaves unassigned (allocator slack): 2% of the card.
#[cfg(feature = "cuda")]
fn bank_margin(device: &Device) -> usize {
    crate::device::total_memory(device).unwrap_or(0) / 50
}

/// Bytes one sequence's recurrent state takes across the GDN and PLE layers of `range`.
fn state_bytes_per_slot(cfg: &ModelConfig, range: &Range<usize>, p: &Placement) -> usize {
    let n_linear = range
        .clone()
        .filter(|&l| cfg.layer_types[l] == LayerKind::Linear)
        .count();
    let gdn = cfg.linear_attn.as_ref().map_or(0, |lc| {
        let ssm = lc.num_v_heads / p.world * lc.k_head_dim * lc.v_head_dim;
        let conv = (lc.conv_kernel - 1) * lc.conv_dim() / p.world;
        ssm * DType::F32.size_in_bytes() + conv * p.dtype.size_in_bytes()
    });
    let ple = cfg.qwen38.as_ref().map_or(0, |q| {
        let (len, width) = hyper::Ple::state_shape(q, cfg.hidden_size);
        let layers = q.ple_layers.iter().filter(|l| range.contains(l)).count();
        layers * len * width * p.dtype.size_in_bytes()
    });
    n_linear * gdn + ple
}

impl LayerStack {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        src: &dyn WeightSource,
        cfg: Arc<ModelConfig>,
        range: Range<usize>,
        p: Placement,
        comm: Arc<dyn Communicator>,
        moe_opts: &MoeOptions,
        max_seq_len: usize,
    ) -> Result<Self> {
        if range.end > cfg.num_layers {
            bail!(
                "layer range {range:?} exceeds the model's {} layers",
                cfg.num_layers
            );
        }
        // CPU decode is memory-bound: keep quantized projections packed and use the fused kernels
        let mut p = p;
        p.keep_quantized = p.device.is_cpu() && p.world == 1 && !moe_opts.dequant_fp8;
        let has_moe = range.clone().any(|l| cfg.layer_is_moe(l));
        let n_attn_stage = range
            .clone()
            .filter(|&l| cfg.layer_types[l] == LayerKind::Full)
            .count();
        let kv_width = cfg.num_kv_heads / p.world * cfg.head_dim;
        let index_dim = cfg.qwen38.as_ref().map_or(0, |q| q.index_head_dim);
        let state_bytes = state_bytes_per_slot(&cfg, &range, &p);
        // the KV floor and the recurrent-state pools allocated after loading
        let kv_reserve_bytes = moe_opts.kv_reserve_tokens.unwrap_or(max_seq_len)
            * n_attn_stage
            * (2 * kv_width + index_dim)
            * p.dtype.size_in_bytes()
            + moe_opts.state_slots * state_bytes;
        // per-expert bytes as stored (quantized experts stay quantized unless dequant_fp8)
        let (expert_bytes, local_experts, format) = match (has_moe, cfg.moe.as_ref()) {
            (true, Some(moe)) => {
                let first_moe = range.clone().find(|&l| cfg.layer_is_moe(l)).unwrap();
                let quant = moe::expert_quant_bytes(src, &cfg, first_moe)
                    .filter(|_| !moe_opts.dequant_fp8);
                let per_weight = quant.unwrap_or(p.dtype.size_in_bytes() as f64);
                (
                    ((3 * moe.intermediate_size * cfg.hidden_size) as f64 * per_weight) as usize,
                    range.clone().filter(|&l| cfg.layer_is_moe(l)).count()
                        * moe.num_experts.div_ceil(p.world),
                    crate::benchbw::ExpertFormat::from_quant_bytes(quant),
                )
            }
            _ => (0, 0, crate::benchbw::ExpertFormat::Bf16),
        };
        let strategy = moe_opts.resolve(
            &p.device,
            has_moe
                .then(|| moe::StageMoeNeeds {
                    expert_bytes: expert_bytes * local_experts,
                    kv_reserve_bytes,
                    format,
                })
                .as_ref(),
        );
        // an auto-sized cache starts small and is sized once the stage's other weights are
        // loaded (it fills lazily), so those weights cannot be squeezed out
        let auto_cache = moe_opts.cache_size.is_none() && moe_opts.cache_rate.is_none();
        // offload on one GPU: experts in pinned host memory behind a device slot cache
        let expert_bank = Self::make_bank(&cfg, &p, strategy, has_moe, expert_bytes, &range, &moe_opts.cpu_layers);
        let expert_cache = if has_moe
            && expert_bank.is_none()
            && matches!(strategy, MoeStrategy::Offload | MoeStrategy::Hybrid)
        {
            let moe = cfg.moe.as_ref().unwrap();
            let cap = match (moe_opts.cache_size, moe_opts.cache_rate) {
                (Some(n), _) => n,
                (None, Some(r)) => (r * local_experts as f64).round() as usize,
                (None, None) => moe.top_k * 2,
            }
            .clamp(moe.top_k * 2, local_experts.max(1));
            Some(ExpertCache::new(cap, p.device.clone(), p.dtype))
        } else {
            None
        };
        let fetch = match moe_opts.hybrid_max_fetch {
            Some(n) => moe::FetchPolicy::Max(n),
            None => match moe_opts.fetch_fraction(&p.device, format) {
                Some(f) if strategy == MoeStrategy::Hybrid => {
                    tracing::info!("hybrid: uploading {:.0}% of each step's misses (bench profile)", f * 100.0);
                    moe::FetchPolicy::Fraction(f)
                }
                _ => moe::FetchPolicy::Max(cfg.moe.as_ref().map(|m| m.top_k).unwrap_or(1)),
            },
        };
        // --moe-cpu-layers: these MoE layers skip the device cache and run on the CPU
        let cpu_layers = if matches!(strategy, MoeStrategy::Offload | MoeStrategy::Hybrid) {
            let moe_layers: Vec<usize> = (0..cfg.num_layers).filter(|&l| cfg.layer_is_moe(l)).collect();
            moe_opts.cpu_layers.select(&moe_layers)
        } else {
            vec![]
        };
        let mut layers = Vec::with_capacity(range.len());
        let (mut n_attn, mut n_linear, mut n_ple) = (0, 0, 0);
        for l in range.clone() {
            let prefix = cfg.layer_prefix(l);
            let ffn = if cfg.layer_is_moe(l) {
                let on_cpu = cpu_layers.contains(&l);
                Ffn::Moe(Box::new(MoeBlock::load(
                    src,
                    &cfg,
                    l,
                    &p,
                    if on_cpu { MoeStrategy::Cpu } else { strategy },
                    if on_cpu { None } else { expert_cache.clone() },
                    if on_cpu { None } else { expert_bank.clone() },
                    fetch,
                    !moe_opts.dequant_fp8,
                )?))
            } else {
                Ffn::Dense(Mlp::load(
                    src,
                    &format!("{prefix}.mlp"),
                    ["gate_proj", "up_proj", "down_proj"],
                    &cfg.hidden_act,
                    &p,
                )?)
            };
            let mixer = match cfg.layer_types[l] {
                LayerKind::Full => {
                    n_attn += 1;
                    Mixer::Attn(
                        Attention::load(src, &cfg, &format!("{prefix}.self_attn"), &p)?,
                        n_attn - 1,
                    )
                }
                LayerKind::Linear => {
                    n_linear += 1;
                    Mixer::Linear(
                        GatedDeltaNet::load(src, &cfg, &format!("{prefix}.linear_attn"), &p)?,
                        n_linear - 1,
                    )
                }
            };
            let residual = match &cfg.qwen38 {
                Some(q) => Residual::Hyper {
                    attn: hyper::GatedResidual::load(src, &format!("{prefix}.attn_hyper_connection"), &cfg, &p, true)?,
                    mlp: hyper::GatedResidual::load(src, &format!("{prefix}.mlp_hyper_connection"), &cfg, &p, true)?,
                    ple: match q.ple_layers.iter().position(|&x| x == l) {
                        Some(ple_index) => {
                            n_ple += 1;
                            Some(Box::new((
                                hyper::Ple::load(src, &format!("{prefix}.ple"), &cfg, ple_index, &p)?,
                                n_ple - 1,
                            )))
                        }
                        None => None,
                    },
                },
                None => Residual::Plain {
                    input_ln: RmsNorm::load_model(
                        src,
                        &format!("{prefix}.input_layernorm.weight"),
                        &cfg,
                        &p,
                    )?,
                    post_ln: RmsNorm::load_model(
                        src,
                        &format!("{prefix}.post_attention_layernorm.weight"),
                        &cfg,
                        &p,
                    )?,
                },
            };
            layers.push(DecoderLayer {
                residual,
                mixer,
                ffn,
            });
        }
        if let (Some(cache), Some(moe)) = (&expert_cache, cfg.moe.as_ref()) {
            if auto_cache {
                // the free memory now, less the KV reserve and the activation workspace
                let cap = crate::device::free_memory(&p.device)
                    .map(|free| {
                        ((free as f64 * 0.9) as usize)
                            .saturating_sub(kv_reserve_bytes + crate::device::workspace_reserve(&p.device))
                            / expert_bytes.max(1)
                    })
                    .unwrap_or(local_experts / 4)
                    .clamp(moe.top_k * 2, local_experts.max(1));
                cache.set_capacity(cap);
            }
            let cap = cache.stats().capacity;
            tracing::info!(
                "expert cache on {}: {cap} of {local_experts} experts ({:.2} GiB), strategy {strategy:?}",
                crate::device::device_name(&p.device),
                (cap * expert_bytes) as f64 / (1u64 << 30) as f64
            );
        }
        #[cfg(feature = "cuda")]
        if let (Some(bank), Some(moe)) = (&expert_bank, cfg.moe.as_ref()) {
            let per = bank.expert_bytes().max(1);
            let cap = match (moe_opts.cache_size, moe_opts.cache_rate) {
                (Some(n), _) => n,
                (None, Some(r)) => (r * local_experts as f64).round() as usize,
                // the free memory now, less the KV reserve (which the KV pool then gets), the
                // activation workspace and a small allocator margin
                (None, None) => crate::device::free_memory(&p.device)
                    .map(|free| {
                        free.saturating_sub(
                            // + 128 MiB: allocations made after this point (graphs, cuBLAS)
                            kv_reserve_bytes
                                + crate::device::workspace_reserve(&p.device)
                                + bank_margin(&p.device)
                                + (128 << 20),
                        ) / per
                    })
                    .unwrap_or(local_experts / 4),
            }
            // a prefill chunk may route to every expert of a layer at once
            .clamp(moe.num_experts, local_experts.max(moe.num_experts));
            bank.set_capacity(cap)?;
            tracing::info!(
                "expert slot cache on {}: {cap} of {local_experts} experts ({:.2} GiB), {:.1} GiB pinned host memory",
                crate::device::device_name(&p.device),
                (cap * per) as f64 / (1u64 << 30) as f64,
                bank.host_bytes() as f64 / (1u64 << 30) as f64
            );
        }
        let rotary = Rotary::new(&cfg, max_seq_len, &p)?;
        Ok(Self {
            range,
            cfg,
            layers,
            rotary,
            caches: RwLock::new(Caches::default()),
            n_attn,
            n_linear,
            n_ple,
            index_dim,
            placement: p,
            comm,
            expert_cache,
            expert_bank,
            moe_strategy: has_moe.then_some(strategy),
            kv_width,
            expert_bytes,
            state_bytes,
            planned_states: moe_opts.state_slots,
            graphs: Default::default(),
        })
    }

    pub fn kv_bytes_per_token(&self) -> usize {
        self.n_attn * (2 * self.kv_width + self.index_dim) * self.placement.dtype.size_in_bytes()
    }

    /// How many KV tokens fit in `memory_ratio` of the device's free memory (None when the
    /// stack has no attention layers or the memory cannot be queried).
    pub fn kv_capacity(&self, memory_ratio: f64) -> Option<usize> {
        if self.n_attn == 0 {
            return None;
        }
        let per_token = self.kv_bytes_per_token().max(1);
        // the expert cache fills lazily: keep room for the part not resident yet
        let cache_reserve = self
            .expert_cache
            .as_ref()
            .map(|c| {
                let s = c.stats();
                s.capacity.saturating_sub(s.resident) * self.expert_bytes
            })
            .unwrap_or(0);
        // the recurrent-state pools are allocated next to the KV pool
        let states = self.planned_states * self.state_bytes;
        // activations need room too (CUDA; host memory is not pre-reserved)
        let workspace = if self.placement.device.is_cuda() {
            crate::device::workspace_reserve(&self.placement.device)
        } else {
            0
        };
        // with an expert slot cache the bank was sized to leave the KV reserve free: the pool
        // takes that, less a small margin, instead of a share of what is left
        #[cfg(feature = "cuda")]
        if self.expert_bank.is_some() {
            let margin = bank_margin(&self.placement.device);
            return crate::device::free_memory(&self.placement.device)
                .map(|free| free.saturating_sub(states + workspace + margin) / per_token);
        }
        crate::device::free_memory(&self.placement.device).map(|free| {
            ((free as f64 * memory_ratio) as usize).saturating_sub(cache_reserve + states + workspace) / per_token
        })
    }

    /// (Re)allocate `num_slots` KV tokens per attention layer and `num_states` recurrent-state
    /// slots per linear-attention layer. Old pools are freed first.
    pub fn alloc_kv(&self, num_slots: usize, num_states: usize) -> Result<()> {
        self.graphs.clear();
        let mut c = self.caches.write();
        *c = Caches::default();
        let p = &self.placement;
        for _ in 0..self.n_attn {
            let kv = LayerKv::new(
                num_slots,
                self.kv_width / self.cfg.head_dim,
                self.cfg.head_dim,
                p.dtype,
                &p.device,
            )?;
            c.kv.push(if self.index_dim > 0 {
                kv.with_index(self.index_dim)?
            } else {
                kv
            });
        }
        if let Some(q) = &self.cfg.qwen38 {
            let (len, width) = hyper::Ple::state_shape(q, self.cfg.hidden_size);
            for _ in 0..self.n_ple {
                c.ple.push(Tensor::zeros((num_states.max(1), len, width), p.dtype, &p.device)?);
            }
        }
        if let Some(lc) = &self.cfg.linear_attn {
            for _ in 0..self.n_linear {
                c.states.push(GatedDeltaNet::new_state(
                    lc,
                    p.world,
                    num_states.max(1),
                    p.dtype,
                    &p.device,
                )?);
            }
        }
        Ok(())
    }

    /// Returns false when this stack has no expert cache.
    pub fn resize_expert_cache(&self, experts: usize) -> bool {
        #[cfg(feature = "cuda")]
        if let Some(b) = &self.expert_bank {
            // captured graphs point into the old bank
            self.graphs.clear();
            if let Err(e) = b.set_capacity(experts) {
                tracing::error!("resizing the expert slot cache failed: {e:#}");
            }
            return true;
        }
        match &self.expert_cache {
            Some(c) => {
                c.set_capacity(experts);
                true
            }
            None => false,
        }
    }

    /// The device slot cache for `--moe-strategy offload` on a single CUDA device, unless
    /// `CARBON_MOE_BANK=0`, the experts are not in a format the fused kernels read, or the
    /// host cannot pin them all (then the host-routed LRU path serves the stage).
    #[allow(clippy::too_many_arguments)]
    fn make_bank(
        cfg: &ModelConfig,
        p: &Placement,
        strategy: MoeStrategy,
        has_moe: bool,
        expert_bytes: usize,
        range: &Range<usize>,
        cpu_layers: &moe::CpuLayers,
    ) -> Option<moe::Bank> {
        #[cfg(feature = "cuda")]
        {
            let moe = cfg.moe.as_ref()?;
            if !has_moe
                || strategy != MoeStrategy::Offload
                || !p.device.is_cuda()
                || p.world != 1
                || std::env::var("CARBON_MOE_BANK").is_ok_and(|v| v == "0")
                || !matches!(cfg.hidden_act.as_str(), "silu" | "swish")
                || moe.num_experts > 1024
            {
                return None;
            }
            let moe_layers: Vec<usize> = (0..cfg.num_layers).filter(|&l| cfg.layer_is_moe(l)).collect();
            let on_cpu = cpu_layers.select(&moe_layers);
            let banked = range.clone().filter(|l| cfg.layer_is_moe(*l) && !on_cpu.contains(l)).count();
            let need = banked * moe.num_experts * expert_bytes;
            let avail = crate::device::host_available_memory().unwrap_or(usize::MAX);
            // keep room for the rest of the process and the page cache of mapped weights
            let keep = (8usize << 30).max(avail / 10);
            if need + keep > avail {
                tracing::warn!(
                    "offload: {:.1} GiB of experts cannot be pinned ({:.1} GiB available): using the host-routed expert cache",
                    need as f64 / (1u64 << 30) as f64,
                    avail as f64 / (1u64 << 30) as f64
                );
                return None;
            }
            match bank::ExpertBank::new(&p.device, moe.num_experts) {
                Ok(b) => Some(b),
                Err(e) => {
                    tracing::warn!("offload: no expert slot cache: {e:#}");
                    None
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (cfg, p, strategy, has_moe, expert_bytes, range, cpu_layers);
            None
        }
    }

    pub fn kv_slots(&self) -> usize {
        self.caches
            .read()
            .kv
            .first()
            .map(|k| k.k.dim(0).unwrap_or(0))
            .unwrap_or(0)
    }

    /// Copy recurrent state `src` -> `dst` in every linear-attention layer (prefix snapshots).
    pub fn copy_state(&self, src: u32, dst: u32) -> Result<()> {
        let c = self.caches.read();
        for st in &c.states {
            st.copy_slot(src, dst)?;
        }
        for t in &c.ple {
            let (n, len, w) = t.dims3()?;
            let flat = t.reshape((n, len * w))?;
            let row = flat.index_select(&Tensor::new(&[src], t.device())?, 0)?;
            let idx = Tensor::new(&[[dst]], t.device())?.broadcast_as((1, len * w))?.contiguous()?;
            flat.scatter_set(&idx, &row.contiguous()?, 0)?;
        }
        Ok(())
    }

    pub fn state_slots(&self) -> usize {
        self.caches
            .read()
            .states
            .first()
            .map(|s| s.ssm.dim(0).unwrap_or(0))
            .unwrap_or(0)
    }

    pub fn forward(&self, meta: &BatchMeta, hidden: &Tensor) -> Result<Tensor> {
        let p = &self.placement;
        let x = hidden.to_device(&p.device)?.to_dtype(p.dtype)?;
        if self.layers.is_empty() {
            return Ok(x);
        }
        let c = self.caches.read();
        if c.kv.len() != self.n_attn || c.states.len() != self.n_linear || c.ple.len() != self.n_ple {
            bail!("KV cache not allocated for layers {:?}", self.range);
        }
        let mut b = DeviceBatch::build(meta, &self.rotary, self.kv_width, &p.device)?;
        for l in &self.layers {
            if let Residual::Hyper { ple: Some(pi), .. } = &l.residual {
                b.ple_emb.push(pi.0.embed(&b, &p.device, p.dtype)?);
            }
        }
        #[cfg(feature = "cuda")]
        if let Some(y) = self.forward_graph(meta, &x, &c, &b)? {
            return Ok(y);
        }
        self.run_layers(x, &b, &c)
    }

    /// Capture single-device decode steps as CUDA graphs (no effect elsewhere).
    pub fn enable_cuda_graphs(&self, on: bool) {
        self.graphs.set_requested(on);
    }

    /// The decoder layers over a prepared batch (also what a CUDA graph captures).
    fn run_layers(&self, mut x: Tensor, b: &DeviceBatch, c: &Caches) -> Result<Tensor> {
        let world = self.comm.world();
        let mixer = |layer: &DecoderLayer, h: &Tensor| -> Result<Tensor> {
            let a = match &layer.mixer {
                Mixer::Attn(attn, i) => attn.forward(h, b, &c.kv[*i])?,
                Mixer::Linear(gdn, i) => gdn.forward(h, b, &c.states[*i])?,
            };
            if world > 1 {
                self.comm.all_reduce(&a)
            } else {
                Ok(a)
            }
        };
        let ffn = |layer: &DecoderLayer, h: &Tensor| -> Result<Tensor> {
            let m = match &layer.ffn {
                Ffn::Dense(mlp) => mlp.forward(h)?,
                Ffn::Moe(moe) => moe.forward(h)?,
            };
            if world > 1 {
                self.comm.all_reduce(&m)
            } else {
                Ok(m)
            }
        };
        for layer in &self.layers {
            match &layer.residual {
                Residual::Plain { input_ln, post_ln } => {
                    x = (&x + mixer(layer, &input_ln.forward(&x)?)?)?;
                    x = (&x + ffn(layer, &post_ln.forward(&x)?)?)?;
                }
                Residual::Hyper { attn, mlp, ple } => {
                    if let Some(pi) = ple {
                        let (p, i) = &**pi;
                        x = p.forward(&x, &b.ple_emb[*i], b, &c.ple[*i])?;
                    }
                    let (h, inj) = attn.mix(&x)?;
                    x = attn.combine(&x, &mixer(layer, &h)?, &inj.expect("injection"))?;
                    let (h, inj) = mlp.mix(&x)?;
                    x = mlp.combine(&x, &ffn(layer, &h)?, &inj.expect("injection"))?;
                }
            }
        }
        Ok(x)
    }

    pub fn expert_bank(&self) -> Option<moe::Bank> {
        self.expert_bank.clone()
    }

    /// Prefetch handles of the stage's PLE layers.
    pub fn ple_prefetchers(&self) -> Vec<hyper::PlePrefetch> {
        self.layers
            .iter()
            .filter_map(|l| match &l.residual {
                Residual::Hyper { ple: Some(pi), .. } => Some(pi.0.prefetcher()),
                _ => None,
            })
            .collect()
    }

    fn expert_stats(&self) -> Option<moe::ExpertStats> {
        #[cfg(feature = "cuda")]
        if let Some(b) = &self.expert_bank {
            return Some(b.stats());
        }
        self.expert_cache.as_ref().map(|c| c.stats())
    }

    pub fn stats(&self) -> StageStats {
        StageStats {
            name: "local".into(),
            layers: (self.range.start, self.range.end),
            devices: vec![crate::device::device_name(&self.placement.device)],
            kv_slots: self.kv_slots(),
            state_slots: self.state_slots(),
            moe_strategy: self.moe_strategy.map(|s| format!("{s:?}").to_lowercase()),
            experts: self.expert_stats(),
            cuda_graphs: self.graphs.captured(),
            cuda_graphs_disabled: self.graphs.disabled_reason(),
        }
    }
}

/// What turns the last hidden state into the LM head's input.
enum FinalNorm {
    Rms(RmsNorm),
    /// Qwen3.8: mix the hyper-connection streams back into one.
    Hyper(hyper::GatedResidual),
}

/// Token embedding, final norm and LM head (lives on the coordinator's head device).
pub struct ModelHead {
    embed: Tensor,
    norm: FinalNorm,
    /// Qwen3.8: the embedding is copied into this many residual streams.
    streams: usize,
    lm_head: Linear,
    pub device: Device,
    pub dtype: DType,
}

impl ModelHead {
    pub fn load(src: &dyn WeightSource, cfg: &ModelConfig, p: &Placement) -> Result<Self> {
        let name = format!("{}embed_tokens.weight", cfg.prefix);
        let untied = !cfg.tie_word_embeddings && src.contains("lm_head.weight");
        // an untied input embedding is only gathered from (a few rows per step): on a GPU it
        // stays in host memory and leaves the VRAM to the KV and expert caches
        let host_embed = untied && p.device.is_cuda();
        let embed = src.get(&name, if host_embed { &Device::Cpu } else { &p.device }, p.dtype)?;
        let lm_head = if untied {
            Linear::new(src.get("lm_head.weight", &p.device, p.dtype)?, None)
        } else {
            Linear::new(embed.clone(), None)
        };
        let (norm, streams) = match &cfg.qwen38 {
            Some(q) => (
                FinalNorm::Hyper(hyper::GatedResidual::load(
                    src,
                    &format!("{}hyper_connection_mixer", cfg.prefix),
                    cfg,
                    p,
                    false,
                )?),
                q.hc_count,
            ),
            None => (
                FinalNorm::Rms(RmsNorm::load_model(src, &format!("{}norm.weight", cfg.prefix), cfg, p)?),
                1,
            ),
        };
        Ok(Self {
            embed,
            norm,
            streams,
            lm_head,
            device: p.device.clone(),
            dtype: p.dtype,
        })
    }

    pub fn embed(&self, tokens: &[u32]) -> Result<Tensor> {
        self.embed_with(tokens, &[])
    }

    /// `embed` with rows `row..row + n` replaced by `rows [n, hidden]` (image features).
    pub fn embed_with(&self, tokens: &[u32], inject: &[(usize, Tensor)]) -> Result<Tensor> {
        let mut e = self.embed_single(tokens)?;
        if !inject.is_empty() {
            let h = e.dim(1)?;
            for (row, rows) in inject {
                let rows = rows.to_device(&self.device)?.to_dtype(e.dtype())?;
                e = e.slice_assign(&[*row..*row + rows.dim(0)?, 0..h], &rows)?;
            }
        }
        Ok(if self.streams > 1 { e.repeat((1, self.streams))? } else { e })
    }

    /// Token embeddings `[T, H]`, one stream (the MTP head's input).
    pub fn embed_single(&self, tokens: &[u32]) -> Result<Tensor> {
        let ids = Tensor::from_slice(tokens, tokens.len(), self.embed.device())?;
        Ok(self.embed.index_select(&ids, 0)?.to_device(&self.device)?)
    }

    /// Logits in f32 for already mixed states `[R, H]` (the MTP head's output).
    pub fn project(&self, mixed: &Tensor) -> Result<Tensor> {
        let h = mixed.to_device(&self.device)?.to_dtype(self.dtype)?;
        Ok(self.lm_head.forward(&h)?.to_dtype(DType::F32)?)
    }

    /// Logits in f32 for the selected rows of `hidden`.
    pub fn logits(&self, hidden: &Tensor, rows: &[u32]) -> Result<Tensor> {
        let hidden = hidden.to_device(&self.device)?.to_dtype(self.dtype)?;
        let idx = Tensor::from_slice(rows, rows.len(), &self.device)?;
        let rows = hidden.index_select(&idx, 0)?;
        let h = match &self.norm {
            FinalNorm::Rms(n) => n.forward(&rows)?,
            FinalNorm::Hyper(m) => m.mix(&rows)?.0,
        };
        Ok(self.lm_head.forward(&h)?.to_dtype(DType::F32)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The memory planning's per-slot state size is what `alloc_kv` allocates (GDN conv + ssm,
    /// PLE conv inputs), so the expert cache and KV pool leave the right room for it.
    #[test]
    fn state_bytes_match_the_allocated_pools() {
        for arch in ["qwen3_5_moe", "qwen4_exp"] {
            let dir = tempfile::tempdir().unwrap();
            crate::testing::write_tiny_model(dir.path(), &crate::testing::TinySpec::new(arch))
                .unwrap();
            let src = crate::checkpoint::open_weights(dir.path()).unwrap();
            let mut cfg = ModelConfig::from_dir_with(dir.path(), None).unwrap();
            if src.contains("model.language_model.embed_tokens.weight") {
                cfg.prefix = "model.language_model.".into();
            }
            let cfg = Arc::new(cfg);
            let p = Placement::single(Device::Cpu, DType::BF16);
            let range = 0..cfg.num_layers;
            let stack = LayerStack::load(
                &*src,
                cfg.clone(),
                range.clone(),
                p.clone(),
                Arc::new(crate::parallel::comm::NoComm),
                &MoeOptions::default(),
                64,
            )
            .unwrap();
            let slots = 5;
            stack.alloc_kv(16, slots).unwrap();
            let c = stack.caches.read();
            let bytes = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
            let allocated: usize = c.states.iter().map(|s| bytes(&s.conv) + bytes(&s.ssm)).sum::<usize>()
                + c.ple.iter().map(bytes).sum::<usize>();
            assert!(allocated > 0, "{arch}");
            assert_eq!(allocated, slots * state_bytes_per_slot(&cfg, &range, &p), "{arch}");
        }
    }
}
