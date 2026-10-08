// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Mixture-of-Experts block with FreeToken-style expert placement strategies.
//!
//! * `fused`   - every expert resident on the device.
//! * `offload` - experts live in host RAM; a global LRU cache of expert slots
//!   lives on the device and misses are copied over PCIe.
//! * `cpu`     - experts live in host RAM and are computed on the CPU.
//! * `hybrid`  - per layer step, cache hits run on the device, up to
//!   `max_fetch` misses are copied, the remaining misses run on the CPU.
//!
//! Under tensor parallelism experts are distributed (expert parallelism):
//! rank `r` owns experts `e % world == r` and the block output is a partial sum.

use super::layers::{load_linear, Mlp, Placement};
use crate::config::{ModelConfig, MoeNaming};
use crate::weights::WeightSource;
use anyhow::{bail, Result};
#[cfg(feature = "cuda")]
use anyhow::Context;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::Linear;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MoeStrategy {
    #[default]
    Auto,
    Fused,
    Offload,
    Cpu,
    Hybrid,
}

impl FromStr for MoeStrategy {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "auto" => Self::Auto,
            "fused" => Self::Fused,
            "offload" => Self::Offload,
            "cpu" => Self::Cpu,
            "hybrid" => Self::Hybrid,
            o => bail!("unknown moe strategy '{o}' (auto|fused|offload|cpu|hybrid)"),
        })
    }
}

/// `--moe-cpu-layers`: MoE layers computed on the CPU under offload / hybrid.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum CpuLayers {
    #[default]
    None,
    /// Explicit layer ids.
    List(Vec<usize>),
    /// This many MoE layers, evenly strided.
    Count(usize),
    /// This fraction of the MoE layers, evenly strided.
    Fraction(f64),
}

impl FromStr for CpuLayers {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim();
        Ok(if s.is_empty() || s == "none" || s == "auto" {
            // `auto` exists for pinned-memory quotas (WSL); carbon does not pin host experts
            Self::None
        } else if s.contains(',') {
            Self::List(
                s.split(',')
                    .map(|x| x.trim().parse::<usize>())
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|e| anyhow::anyhow!("bad --moe-cpu-layers list '{s}': {e}"))?,
            )
        } else if s.contains('.') {
            let f: f64 = s.parse()?;
            if !(0.0..=1.0).contains(&f) {
                bail!("--moe-cpu-layers fraction must be in [0, 1]");
            }
            Self::Fraction(f)
        } else {
            Self::Count(s.parse()?)
        })
    }
}

impl CpuLayers {
    /// Which of the model's `moe_layers` (ascending ids) run on the CPU.
    pub fn select(&self, moe_layers: &[usize]) -> Vec<usize> {
        let n = moe_layers.len();
        let strided = |k: usize| -> Vec<usize> {
            let k = k.min(n);
            (0..k).map(|i| moe_layers[i * n / k.max(1)]).collect()
        };
        match self {
            Self::None => vec![],
            Self::List(l) => l.iter().copied().filter(|x| moe_layers.contains(x)).collect(),
            Self::Count(k) => strided(*k),
            Self::Fraction(f) => strided((f * n as f64).round() as usize),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MoeOptions {
    pub strategy: MoeStrategy,
    /// Device expert-cache capacity (experts across all layers of a stage); None = auto.
    pub cache_size: Option<usize>,
    /// Device expert-cache capacity as a fraction of the stage's experts.
    #[serde(default)]
    pub cache_rate: Option<f64>,
    /// Hybrid: max misses copied to the device per layer step (None = the bench profile's
    /// split, else top-k).
    pub hybrid_max_fetch: Option<usize>,
    /// Dequantize block-FP8 experts to the model dtype at load instead of keeping them FP8.
    #[serde(default)]
    pub dequant_fp8: bool,
    /// KV tokens kept free when sizing the expert cache and deciding `auto` placement
    /// (None = one full context; the engine fills it in).
    #[serde(default)]
    pub kv_reserve_tokens: Option<usize>,
    /// Recurrent-state slots each stage will allocate (live sequences + prefix snapshots); the
    /// engine fills it in so the expert cache and KV pool leave room for them.
    #[serde(default)]
    pub state_slots: usize,
    /// MoE layers computed on the CPU under offload / hybrid.
    #[serde(default)]
    pub cpu_layers: CpuLayers,
    /// Keep an offload / hybrid strategy on CPU stages too (tests exercise the cache machinery
    /// there); normally a CPU stage keeps its experts resident.
    #[serde(default)]
    pub offload_on_cpu: bool,
}

impl Default for MoeOptions {
    fn default() -> Self {
        Self {
            strategy: MoeStrategy::Auto,
            cache_size: None,
            cache_rate: None,
            hybrid_max_fetch: None,
            dequant_fp8: false,
            kv_reserve_tokens: None,
            state_slots: 0,
            cpu_layers: CpuLayers::None,
            offload_on_cpu: false,
        }
    }
}

/// What `MoeOptions::resolve` knows about a stage.
pub struct StageMoeNeeds {
    /// Bytes of this stage's (local) routed experts.
    pub expert_bytes: usize,
    /// Bytes the KV reserve and the recurrent-state pools take on this stage.
    pub kv_reserve_bytes: usize,
    pub format: crate::benchbw::ExpertFormat,
}

impl MoeOptions {
    /// Resolve `auto`: experts stay resident when they fit next to the KV reserve (always on
    /// the CPU); otherwise the bench profile's pick for this expert format, else `offload`.
    /// Explicit strategies are honoured on every device (on a CPU device they only exercise the
    /// code paths, which the tests rely on).
    pub fn resolve(&self, device: &Device, needs: Option<&StageMoeNeeds>) -> MoeStrategy {
        // a CPU stage's experts already live in host memory: an offload cache there would
        // only duplicate them
        if device.is_cpu() && !(self.offload_on_cpu && self.strategy != MoeStrategy::Auto) {
            return MoeStrategy::Fused;
        }
        if self.strategy != MoeStrategy::Auto {
            return self.strategy;
        }
        let (Some(n), false) = (needs, device.is_cpu()) else {
            return MoeStrategy::Fused;
        };
        let Some(free) = crate::device::free_memory(device) else {
            return MoeStrategy::Fused;
        };
        // non-expert weights, activations and allocator slack: keep 10% of free memory
        if (n.expert_bytes + n.kv_reserve_bytes + crate::device::workspace_reserve(device)) as f64
            <= free as f64 * 0.9
        {
            return MoeStrategy::Fused;
        }
        let name = crate::device::device_name(device);
        let pick = crate::benchbw::recommendation(&name, n.format)
            .map(|r| r.strategy)
            .unwrap_or_else(|| "offload".into());
        let s = if pick == "hybrid" {
            MoeStrategy::Hybrid
        } else {
            MoeStrategy::Offload
        };
        tracing::info!(
            "moe auto: {:.1} GiB of {} experts do not fit in {:.1} GiB free on {name}: {s:?}{}",
            n.expert_bytes as f64 / (1u64 << 30) as f64,
            n.format.key(),
            free as f64 / (1u64 << 30) as f64,
            if pick == "hybrid" { " (bench profile)" } else { "" }
        );
        s
    }

    /// Hybrid's share of a step's misses that is uploaded (the rest runs on the CPU).
    pub fn fetch_fraction(&self, device: &Device, format: crate::benchbw::ExpertFormat) -> Option<f64> {
        crate::benchbw::recommendation(&crate::device::device_name(device), format)
            .and_then(|r| r.fetch_fraction)
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct ExpertStats {
    pub capacity: usize,
    pub resident: usize,
    pub hits: u64,
    pub misses: u64,
    pub cpu_computed: u64,
    pub bytes_copied: u64,
}

struct CacheInner {
    map: HashMap<(usize, usize), (Arc<Mlp>, u64)>,
    tick: u64,
}

/// Global LRU cache of experts on one device, shared by every MoE layer of a stage.
pub struct ExpertCache {
    capacity: AtomicUsize,
    device: Device,
    dtype: DType,
    /// Where uploaded experts live, apart from the activations (see `DevicePool`).
    pool: crate::device::DevicePool,
    inner: Mutex<CacheInner>,
    hits: AtomicU64,
    misses: AtomicU64,
    cpu_computed: AtomicU64,
    bytes_copied: AtomicU64,
}

impl ExpertCache {
    pub fn new(capacity: usize, device: Device, dtype: DType) -> Arc<Self> {
        Arc::new(Self {
            capacity: AtomicUsize::new(capacity.max(1)),
            pool: crate::device::DevicePool::new(&device),
            device,
            dtype,
            inner: Mutex::new(CacheInner {
                map: HashMap::new(),
                tick: 0,
            }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            cpu_computed: AtomicU64::new(0),
            bytes_copied: AtomicU64::new(0),
        })
    }

    pub fn lookup(&self, key: (usize, usize)) -> Option<Arc<Mlp>> {
        let mut g = self.inner.lock();
        g.tick += 1;
        let tick = g.tick;
        let hit = g.map.get_mut(&key).map(|e| {
            e.1 = tick;
            e.0.clone()
        });
        if hit.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }

    pub fn contains(&self, key: (usize, usize)) -> bool {
        self.inner.lock().map.contains_key(&key)
    }

    /// Copy a host expert to the device, evicting the least recently used entry if full.
    pub fn fetch(&self, key: (usize, usize), host: &Mlp) -> Result<Arc<Mlp>> {
        self.misses.fetch_add(1, Ordering::Relaxed);
        let mut g = self.inner.lock();
        // evict first so the victim's blocks are free for this upload
        Self::evict_to(&mut g, self.capacity() - 1);
        drop(g);
        let dev = Arc::new(host.to_device_with(self.dtype, &|t| self.pool.upload(t, &self.device))?);
        self.bytes_copied
            .fetch_add(host.size_in_bytes() as u64, Ordering::Relaxed);
        let mut g = self.inner.lock();
        Self::evict_to(&mut g, self.capacity() - 1);
        g.tick += 1;
        let tick = g.tick;
        g.map.insert(key, (dev.clone(), tick));
        Ok(dev)
    }

    pub fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Relaxed)
    }

    /// Resize at runtime; shrinking evicts least recently used experts.
    pub fn set_capacity(&self, n: usize) {
        let n = n.max(1);
        self.capacity.store(n, Ordering::Relaxed);
        Self::evict_to(&mut self.inner.lock(), n);
    }

    fn evict_to(g: &mut CacheInner, n: usize) {
        while g.map.len() > n {
            let victim = g
                .map
                .iter()
                .min_by_key(|(_, (_, t))| *t)
                .map(|(k, _)| *k)
                .unwrap();
            g.map.remove(&victim);
        }
    }

    pub fn note_cpu(&self, n: u64) {
        self.cpu_computed.fetch_add(n, Ordering::Relaxed);
    }

    pub fn stats(&self) -> ExpertStats {
        ExpertStats {
            capacity: self.capacity(),
            resident: self.inner.lock().map.len(),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            cpu_computed: self.cpu_computed.load(Ordering::Relaxed),
            bytes_copied: self.bytes_copied.load(Ordering::Relaxed),
        }
    }
}

/// A stage's device slot cache for offloaded experts (CUDA builds only).
#[cfg(feature = "cuda")]
pub type Bank = Arc<super::bank::ExpertBank>;
/// Without CUDA there is no slot cache: an uninhabited placeholder.
#[cfg(not(feature = "cuda"))]
#[derive(Clone)]
pub enum Bank {}

enum Experts {
    /// Experts on the compute device (None = owned by another expert-parallel rank).
    Resident(Vec<Option<Arc<Mlp>>>),
    /// Offloaded experts in pinned host memory behind the stage's device slot cache; routing,
    /// caching and expert compute stay on the GPU (see `bank.rs`).
    #[cfg(feature = "cuda")]
    Banked {
        bank: Bank,
        row: usize,
        table: crate::kernels::cuda::MoeTable,
    },
    /// Experts in host memory, executed according to `strategy`.
    Host {
        experts: Vec<Option<Arc<Mlp>>>,
        strategy: MoeStrategy,
        cache: Option<Arc<ExpertCache>>,
        fetch: FetchPolicy,
    },
}

/// How many of a hybrid step's cache misses are uploaded (the rest run on the CPU).
#[derive(Clone, Copy, Debug)]
pub enum FetchPolicy {
    /// At most this many per layer step (`--moe-hybrid-max-fetch`).
    Max(usize),
    /// This fraction of the step's misses, rounded up (bench-profile split).
    Fraction(f64),
}

impl FetchPolicy {
    pub fn limit(&self, misses: usize) -> usize {
        match *self {
            Self::Max(n) => n,
            Self::Fraction(f) => (f * misses as f64).ceil() as usize,
        }
    }
}

pub struct MoeBlock {
    layer: usize,
    gate: Linear,
    experts: Experts,
    top_k: usize,
    norm_topk: bool,
    num_experts: usize,
    device: Device,
    dtype: DType,
    /// Always-on shared expert and its sigmoid gate (Qwen3.5 / Qwen2-MoE style).
    shared: Option<(Mlp, Linear)>,
    /// The experts' activation.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    act: String,
    /// Resident experts on CUDA: pointer table for carbon's fused routing + expert kernels.
    #[cfg(feature = "cuda")]
    kernel_table: Option<crate::kernels::cuda::MoeTable>,
}

/// Experts per copy batch of a prefill chunk's misses (each batch computes once it lands).
#[cfg(feature = "cuda")]
const COPY_BATCH: usize = 32;

/// Largest step (in tokens) the fused GPU MoE path takes; it re-reads each routed expert's
/// weights per token, so prefill chunks use the grouped per-expert path instead. Multi-token
/// decode steps (B sequences x up to 8 tokens) stay below it.
#[cfg(feature = "cuda")]
const FUSED_MOE_MAX_TOKENS: usize = 64;

static FUSED_MOE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// The fused GPU MoE path is on unless `CARBON_FUSED_MOE=0` or it failed once.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn fused_moe_enabled() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var("CARBON_FUSED_MOE").map_or(true, |v| v != "0"))
        && FUSED_MOE.load(Ordering::Relaxed)
}

/// Stored bytes per weight of layer `layer`'s routed experts when kept quantized
/// (1 for FP8, ~0.56 for NVFP4), or None when they are dense.
pub fn expert_quant_bytes(src: &dyn WeightSource, cfg: &ModelConfig, layer: usize) -> Option<f64> {
    let (prefix, names, _) = expert_names(cfg, layer);
    let w = format!("{prefix}.experts.0.{}", names[0]);
    if src.contains(&format!("{w}.weight_scale_inv")) {
        Some(1.0)
    } else if src.contains(&format!("{w}.weight_scale_2"))
        || src.contains(&format!("{w}.weight_packed"))
    {
        Some(0.5 + 1.0 / 16.0)
    } else if src.contains(&format!("{w}.weight_scale")) {
        Some(1.0)
    } else {
        None
    }
}

pub fn expert_names(cfg: &ModelConfig, layer: usize) -> (String, [&'static str; 3], String) {
    match cfg.moe.as_ref().map(|m| m.naming) {
        Some(MoeNaming::Mixtral) => (
            format!("{}.block_sparse_moe", cfg.layer_prefix(layer)),
            ["w1", "w3", "w2"],
            format!("{}.block_sparse_moe.gate", cfg.layer_prefix(layer)),
        ),
        _ => (
            format!("{}.mlp", cfg.layer_prefix(layer)),
            ["gate_proj", "up_proj", "down_proj"],
            format!("{}.mlp.gate", cfg.layer_prefix(layer)),
        ),
    }
}

impl MoeBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        src: &dyn WeightSource,
        cfg: &ModelConfig,
        layer: usize,
        p: &Placement,
        strategy: MoeStrategy,
        cache: Option<Arc<ExpertCache>>,
        bank: Option<Bank>,
        fetch: FetchPolicy,
        keep_fp8: bool,
    ) -> Result<Self> {
        let moe = cfg.moe.as_ref().expect("moe config");
        let (prefix, names, gate_name) = expert_names(cfg, layer);
        let gate = load_linear(src, &gate_name, p)?;
        let single = |dev: &Device, dtype: DType| Placement {
            device: dev.clone(),
            dtype,
            rank: 0,
            world: 1,
            keep_quantized: p.keep_quantized,
        };
        // host experts keep the model dtype: f32 copies of a large MoE would double host RAM
        let host_dtype = p.dtype;
        // HF-native checkpoints may store the experts stacked: gate_up_proj [E, 2I, H], down_proj [E, H, I]
        let stacked_name = format!("{prefix}.experts.gate_up_proj");
        let stacked = if src.contains(&stacked_name) {
            Some((
                src.load_cpu(&stacked_name)?,
                src.load_cpu(&format!("{prefix}.experts.down_proj"))?,
            ))
        } else {
            None
        };
        let mut list = Vec::with_capacity(moe.num_experts);
        for e in 0..moe.num_experts {
            if e % p.world != p.rank {
                list.push(None);
                continue;
            }
            let place = match strategy {
                MoeStrategy::Fused | MoeStrategy::Auto => single(&p.device, p.dtype),
                _ => single(&Device::Cpu, host_dtype),
            };
            let mlp = match &stacked {
                Some((gu, dn)) => {
                    let i = moe.intermediate_size;
                    let gu = gu.get(e)?;
                    let lin = |w: Tensor| -> Result<Linear> {
                        Ok(Linear::new(
                            w.contiguous()?
                                .to_dtype(place.dtype)?
                                .to_device(&place.device)?,
                            None,
                        ))
                    };
                    Mlp::new(
                        lin(gu.narrow(0, 0, i)?)?,
                        lin(gu.narrow(0, i, i)?)?,
                        lin(dn.get(e)?)?,
                        &cfg.hidden_act,
                    )
                }
                None => Mlp::load_expert(
                    src,
                    &format!("{prefix}.experts.{e}"),
                    names,
                    &cfg.hidden_act,
                    keep_fp8,
                    &place,
                )?,
            };
            list.push(Some(Arc::new(mlp)));
        }
        let shared = if moe.shared_expert_intermediate_size > 0 {
            Some((
                Mlp::load(
                    src,
                    &format!("{prefix}.shared_expert"),
                    names,
                    &cfg.hidden_act,
                    p,
                )?,
                load_linear(src, &format!("{prefix}.shared_expert_gate"), p)?,
            ))
        } else {
            None
        };
        #[cfg(feature = "cuda")]
        let kernel_table =
            if p.device.is_cuda() && matches!(strategy, MoeStrategy::Fused | MoeStrategy::Auto) {
                let projs: Option<Vec<_>> = list
                    .iter()
                    .map(|m| match m {
                        None => Some(None),
                        Some(m) => m.kernel_projections().map(Some),
                    })
                    .collect();
                projs.and_then(|projs| {
                    crate::kernels::cuda::MoeTable::build(&projs, &p.device)
                        .map_err(|e| tracing::debug!("layer {layer}: fused moe unavailable: {e}"))
                        .ok()
                })
            } else {
                None
            };
        let experts = match (strategy, bank) {
            (MoeStrategy::Fused | MoeStrategy::Auto, _) => Experts::Resident(list),
            #[cfg(feature = "cuda")]
            (MoeStrategy::Offload, Some(bank)) => {
                // the host experts move into pinned memory; `list` is dropped here
                let (row, table) = bank
                    .register(&list)
                    .with_context(|| format!("layer {layer}: banking the offloaded experts (CARBON_MOE_BANK=0 keeps the host-routed path)"))?;
                Experts::Banked { bank, row, table }
            }
            (s, _) => Experts::Host {
                experts: list,
                strategy: s,
                cache,
                fetch,
            },
        };
        Ok(Self {
            layer,
            gate,
            experts,
            top_k: moe.top_k,
            norm_topk: moe.norm_topk_prob,
            num_experts: moe.num_experts,
            device: p.device.clone(),
            dtype: p.dtype,
            shared,
            act: cfg.hidden_act.clone(),
            #[cfg(feature = "cuda")]
            kernel_table,
        })
    }

    /// Top-k routing on the host: per expert, the (token, weight) pairs routed to it.
    pub fn route(&self, logits: &[Vec<f32>]) -> Vec<Vec<(u32, f32)>> {
        let mut per_expert: Vec<Vec<(u32, f32)>> = vec![vec![]; self.num_experts];
        let mut order: Vec<usize> = (0..self.num_experts).collect();
        for (t, row) in logits.iter().enumerate() {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = row.iter().map(|x| (x - max).exp()).collect();
            let sum: f32 = exp.iter().sum();
            order.sort_unstable_by(|&a, &b| exp[b].total_cmp(&exp[a]).then(a.cmp(&b)));
            let top = &order[..self.top_k];
            let denom = if self.norm_topk {
                top.iter().map(|&e| exp[e]).sum::<f32>()
            } else {
                sum
            };
            for &e in top {
                per_expert[e].push((t as u32, exp[e] / denom));
            }
        }
        per_expert
    }

    /// Small decode steps of this block run on carbon's fused CUDA kernels.
    pub fn fused_ready(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            (self.kernel_table.is_some() || matches!(self.experts, Experts::Banked { .. }))
                && fused_moe_enabled()
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    /// Shared expert (if any) added to the routed output.
    fn add_shared(&self, x: &Tensor, out: Tensor) -> Result<Tensor> {
        Ok(match &self.shared {
            Some((mlp, gate)) => {
                let g = candle_nn::ops::sigmoid(&gate.forward(x)?)?;
                (out + mlp.forward(x)?.broadcast_mul(&g)?)?
            }
            None => out,
        })
    }

    /// Routing and routed experts in three GPU launches, no host round trip.
    #[cfg(feature = "cuda")]
    fn forward_fused(&self, x: &Tensor) -> Option<Tensor> {
        use crate::kernels::cuda as k;
        if x.dim(0).ok()? > FUSED_MOE_MAX_TOKENS || !fused_moe_enabled() {
            return None;
        }
        let run = || -> Result<Tensor> {
            let logits = self.gate.forward(x)?;
            let y = match (&self.experts, &self.kernel_table) {
                (Experts::Banked { bank, row, table }, _) => {
                    let route = k::moe_route(&logits, self.top_k, self.norm_topk)?;
                    bank.ensure(*row, table, &route, x.dim(0)? * self.top_k)?;
                    k::moe_compute(table, x, &route, self.top_k)?
                }
                (_, Some(table)) => k::moe_experts(table, x, &logits, self.top_k, self.norm_topk)?,
                _ => return Err(anyhow::anyhow!("no fused expert table")),
            };
            self.add_shared(x, y)
        };
        if self.kernel_table.is_none() && !matches!(self.experts, Experts::Banked { .. }) {
            return None;
        }
        match run() {
            Ok(y) => Some(y),
            Err(e) => {
                if FUSED_MOE.swap(false, Ordering::Relaxed) {
                    tracing::warn!("fused moe disabled, using the generic path: {e:#}");
                }
                None
            }
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(y) = self.forward_fused(x) {
            return Ok(y);
        }
        #[cfg(feature = "cuda")]
        if let Experts::Banked { bank, row, table } = &self.experts {
            return self.forward_banked(x, bank, *row, table);
        }
        let (t, h) = x.dims2()?;
        let logits = self
            .gate
            .forward(x)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let routes = self.route(&logits);

        // which experts run where this step
        let mut work: Vec<(usize, &Vec<(u32, f32)>)> = routes
            .iter()
            .enumerate()
            .filter(|(e, r)| !r.is_empty() && self.local(*e))
            .collect();
        // most-loaded experts first: hybrid fetches the ones that pay off most
        work.sort_by_key(|(e, r)| (std::cmp::Reverse(r.len()), *e));

        let mut idx_parts = Vec::with_capacity(work.len());
        let mut out_parts = Vec::with_capacity(work.len());
        let mut fetched = 0usize;
        // hybrid: this step's upload budget, from its number of cache misses
        let max_fetch = match &self.experts {
            Experts::Host {
                fetch,
                cache: Some(c),
                strategy: MoeStrategy::Hybrid,
                ..
            } => fetch.limit(
                work.iter()
                    .filter(|(e, _)| !c.contains((self.layer, *e)))
                    .count(),
            ),
            _ => 0,
        };
        for (e, r) in work {
            let tokens: Vec<u32> = r.iter().map(|(t, _)| *t).collect();
            let weights: Vec<f32> = r.iter().map(|(_, w)| *w).collect();
            let n = tokens.len();
            let idx = Tensor::from_vec(tokens, n, &self.device)?;
            let xs = x.index_select(&idx, 0)?;
            let y = match &self.experts {
                Experts::Resident(list) => {
                    let w =
                        Tensor::from_vec(weights, (n, 1), &self.device)?.to_dtype(self.dtype)?;
                    list[e].as_ref().unwrap().forward(&xs)?.broadcast_mul(&w)?
                }
                #[cfg(feature = "cuda")]
                Experts::Banked { .. } => unreachable!("banked layers take forward_banked"),
                Experts::Host {
                    experts,
                    strategy,
                    cache,
                    ..
                } => {
                    let host = experts[e].as_ref().unwrap();
                    let key = (self.layer, e);
                    let on_device = match strategy {
                        MoeStrategy::Offload => {
                            Some(match cache.as_ref().and_then(|c| c.lookup(key)) {
                                Some(m) => m,
                                None => cache.as_ref().unwrap().fetch(key, host)?,
                            })
                        }
                        MoeStrategy::Hybrid => match cache.as_ref().and_then(|c| c.lookup(key)) {
                            Some(m) => Some(m),
                            None if fetched < max_fetch => {
                                fetched += 1;
                                Some(cache.as_ref().unwrap().fetch(key, host)?)
                            }
                            None => None,
                        },
                        _ => None,
                    };
                    match on_device {
                        Some(m) => {
                            let w = Tensor::from_vec(weights, (n, 1), &self.device)?
                                .to_dtype(self.dtype)?;
                            m.forward(&xs)?.broadcast_mul(&w)?
                        }
                        None => {
                            if let Some(c) = cache {
                                c.note_cpu(1);
                            }
                            // the CPU GEMM has no bf16 kernel: compute in f32
                            let w = Tensor::from_vec(weights, (n, 1), &Device::Cpu)?;
                            let xc = xs.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                            host.forward_f32(&xc)?
                                .broadcast_mul(&w)?
                                .to_dtype(self.dtype)?
                                .to_device(&self.device)?
                        }
                    }
                }
            };
            idx_parts.push(idx);
            out_parts.push(y);
        }
        let mut out = Tensor::zeros((t, h), self.dtype, &self.device)?;
        if !idx_parts.is_empty() {
            let idx = Tensor::cat(&idx_parts, 0)?;
            let ys = Tensor::cat(&out_parts, 0)?;
            out = out.index_add(&idx, &ys, 0)?;
        }
        self.add_shared(x, out)
    }

    /// A block whose routed experts are row `row` of `bank` (already registered), e.g. the MTP
    /// head's.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn from_bank(
        layer: usize,
        gate: Linear,
        shared: Option<(Mlp, Linear)>,
        bank: Bank,
        row: usize,
        table: crate::kernels::cuda::MoeTable,
        moe: &crate::config::MoeConfig,
        act: &str,
        device: &Device,
        dtype: DType,
    ) -> Self {
        Self {
            layer,
            gate,
            experts: Experts::Banked { bank, row, table },
            top_k: moe.top_k,
            norm_topk: moe.norm_topk_prob,
            num_experts: moe.num_experts,
            device: device.clone(),
            dtype,
            shared,
            act: act.to_string(),
            kernel_table: None,
        }
    }

    fn local(&self, e: usize) -> bool {
        match &self.experts {
            Experts::Resident(l) => l[e].is_some(),
            Experts::Host { experts, .. } => experts[e].is_some(),
            #[cfg(feature = "cuda")]
            Experts::Banked { .. } => true,
        }
    }

    /// A multi-token step (prefill chunk) over banked experts: route and cache on the GPU, read
    /// the route and slot map back once, then run each routed expert on its contiguous run of
    /// expert-sorted pairs (views into the bank) and fold the pairs back per token.
    #[cfg(feature = "cuda")]
    fn forward_banked(&self, x: &Tensor, bank: &Bank, row: usize, table: &crate::kernels::cuda::MoeTable) -> Result<Tensor> {
        use crate::kernels::cuda as k;
        let t = x.dim(0)?;
        let kk = self.top_k;
        let pairs = t * kk;
        let logits = self.gate.forward(x)?;
        let route = k::moe_route(&logits, kk, self.norm_topk)?;
        // assign slots, read the route, slot map and copy queue back, then copy the misses in
        // batches on the bank's copy stream while the resident experts compute; each batch's
        // experts start as soon as it lands
        bank.assign(row, table, &route, pairs)?;
        let rt: Vec<u32> = route.to_vec1()?;
        let slots = bank.slot_row(row)?;
        let missing = bank.copy_queue()?;
        let batches: Vec<(Vec<u32>, _)> = missing
            .chunks(COPY_BATCH)
            .enumerate()
            .map(|(i, c)| Ok((c.to_vec(), bank.copy_batch(row, i * COPY_BATCH, c.len())?)))
            .collect::<Result<_>>()?;
        // pairs grouped by expert (stable: token order within an expert)
        let mut order: Vec<u32> = (0..pairs as u32).collect();
        order.sort_by_key(|&p| rt[p as usize]);
        let mut inverse = vec![0u32; pairs];
        for (pos, &p) in order.iter().enumerate() {
            inverse[p as usize] = pos as u32;
        }
        // where each sorted pair lands in the K-major [K * T] layout the fold below sums
        let kmajor: Vec<u32> = (0..pairs as u32)
            .map(|q| {
                let p = (q % t as u32) * kk as u32 + q / t as u32;
                inverse[p as usize]
            })
            .collect();
        let tokens: Vec<u32> = order.iter().map(|&p| p / kk as u32).collect();
        let weights: Vec<f32> = order.iter().map(|&p| f32::from_bits(rt[pairs + p as usize])).collect();
        let dev = &self.device;
        let xs = x.index_select(&Tensor::from_vec(tokens, pairs, dev)?, 0)?;
        // runs of expert-sorted pairs: (expert, start, end)
        let mut runs = Vec::new();
        let mut a = 0;
        while a < pairs {
            let e = rt[order[a] as usize] as usize;
            let mut b = a + 1;
            while b < pairs && rt[order[b] as usize] as usize == e {
                b += 1;
            }
            runs.push((e, a, b));
            a = b;
        }
        let mut run_of = vec![usize::MAX; self.num_experts];
        for (i, &(e, _, _)) in runs.iter().enumerate() {
            run_of[e] = i;
        }
        let mut queued = vec![false; self.num_experts];
        for &e in &missing {
            queued[e as usize] = true;
        }
        if let Some(&(e, _, _)) = runs.iter().find(|&&(e, _, _)| slots[e] < 0) {
            anyhow::bail!("layer {}: expert {e} found no cache slot", self.layer);
        }
        // resident experts first, then each copy batch once it has landed
        let ys = if k::moe_grouped_supported(table, x.dtype(), dev) {
            // grouped tensor-core GEMMs: two launches per batch of experts
            let (h, inter) = table.dims();
            let act = Tensor::zeros((pairs, inter), DType::BF16, dev)?;
            let ys = Tensor::zeros((pairs, h), DType::BF16, dev)?;
            let wts = Tensor::from_vec(weights, pairs, dev)?;
            let launch = |pick: &mut dyn Iterator<Item = usize>| -> Result<()> {
                let mut tl = Vec::new();
                for i in pick {
                    let (e, a, b) = runs[i];
                    for r0 in (a..b).step_by(k::MOE_TILE) {
                        tl.extend([e as u32, r0 as u32, (k::MOE_TILE.min(b - r0)) as u32]);
                    }
                }
                let n = tl.len() / 3;
                if n > 0 {
                    let tiles = Tensor::from_vec(tl, 3 * n, dev)?;
                    k::moe_grouped(table, &xs, &wts, &tiles, n, &act, &ys)?;
                }
                Ok(())
            };
            launch(&mut (0..runs.len()).filter(|&i| !queued[runs[i].0]))?;
            for (experts, ev) in &batches {
                bank.wait(ev)?;
                launch(&mut experts.iter().map(|&e| run_of[e as usize]).filter(|&i| i != usize::MAX))?;
            }
            ys
        } else {
            let mut outs: Vec<Option<Tensor>> = vec![None; runs.len()];
            let compute = |i: usize, outs: &mut Vec<Option<Tensor>>| -> Result<()> {
                let (e, a, b) = runs[i];
                let mlp = bank.expert(row, e, slots[e] as usize, &self.act)?;
                outs[i] = Some(mlp.forward(&xs.narrow(0, a, b - a)?)?);
                Ok(())
            };
            for (i, &(e, _, _)) in runs.iter().enumerate() {
                if !queued[e] {
                    compute(i, &mut outs)?;
                }
            }
            for (experts, ev) in &batches {
                bank.wait(ev)?;
                for &e in experts {
                    if run_of[e as usize] != usize::MAX {
                        compute(run_of[e as usize], &mut outs)?;
                    }
                }
            }
            let outs: Vec<Tensor> = outs.into_iter().map(|o| o.context("an expert run was not computed")).collect::<Result<_>>()?;
            let w = Tensor::from_vec(weights, (pairs, 1), dev)?.to_dtype(self.dtype)?;
            Tensor::cat(&outs, 0)?.broadcast_mul(&w)?
        };
        // K-major [K * T, H]: each token's K expert outputs summed as K contiguous slices
        let ys = ys.index_select(&Tensor::from_vec(kmajor, pairs, dev)?, 0)?;
        let mut out = ys.narrow(0, 0, t)?;
        for r in 1..kk {
            out = (out + ys.narrow(0, r * t, t)?)?;
        }
        self.add_shared(x, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_layer_selection() {
        let moe: Vec<usize> = (1..9).collect(); // layer 0 dense
        assert_eq!("4".parse::<CpuLayers>().unwrap().select(&moe), vec![1, 3, 5, 7]);
        assert_eq!("0.25".parse::<CpuLayers>().unwrap().select(&moe), vec![1, 5]);
        assert_eq!("0,3,7".parse::<CpuLayers>().unwrap().select(&moe), vec![3, 7]);
        assert_eq!("auto".parse::<CpuLayers>().unwrap(), CpuLayers::None);
        assert_eq!("99".parse::<CpuLayers>().unwrap().select(&moe).len(), 8);
        assert!("1.5".parse::<CpuLayers>().is_err());
        assert_eq!(FetchPolicy::Fraction(0.25).limit(6), 2);
        assert_eq!(FetchPolicy::Max(3).limit(10), 3);
    }

    #[test]
    fn lru_cache_evicts_oldest() {
        let dev = Device::Cpu;
        let w = |v: f32| Linear::new(Tensor::full(v, (2, 2), &dev).unwrap(), None);
        let m = Mlp::new(w(1.0), w(1.0), w(1.0), "silu");
        let c = ExpertCache::new(2, dev.clone(), DType::F32);
        c.fetch((0, 0), &m).unwrap();
        c.fetch((0, 1), &m).unwrap();
        assert!(c.lookup((0, 0)).is_some()); // (0,1) is now the LRU entry
        c.fetch((1, 0), &m).unwrap();
        assert!(c.contains((0, 0)));
        assert!(!c.contains((0, 1)));
        let s = c.stats();
        assert_eq!((s.hits, s.misses, s.resident), (1, 3, 2));
        c.set_capacity(1);
        assert_eq!(c.stats().resident, 1);
        assert!(c.contains((1, 0)));
    }
}
