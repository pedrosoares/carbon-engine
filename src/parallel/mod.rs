// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Parallel execution: pipeline stages over devices and machines, and tensor
//! (+ expert) parallel groups inside a stage.

pub mod comm;

use crate::batch::BatchMeta;
use crate::config::ModelConfig;
use crate::model::layers::Placement;
use crate::model::moe::MoeOptions;
use crate::model::{LayerStack, StageStats};
use crate::weights::WeightSource;
use anyhow::{anyhow, bail, Result};
use candle_core::{DType, Device, Tensor};
use comm::{Communicator, HostGroup, NoComm};
use std::ops::Range;
use std::sync::mpsc;
use std::sync::Arc;

/// One pipeline stage: a contiguous range of decoder layers.
pub trait Stage: Send + Sync {
    fn layers(&self) -> Range<usize>;
    /// The stage's device slot cache for offloaded experts, when it is local and has one.
    fn expert_bank(&self) -> Option<crate::model::moe::Bank> {
        None
    }
    /// Handles that read the stage's PLE table rows ahead of a prefill (local stages).
    fn ple_prefetchers(&self) -> Vec<crate::model::hyper::PlePrefetch> {
        vec![]
    }
    /// Run the stage's layers on `hidden: [T, hidden_size]` (any device) and return the result.
    fn forward(&self, meta: &Arc<BatchMeta>, hidden: Tensor) -> Result<Tensor>;
    /// Sequences that finished; remote stages drop their slot tables.
    fn release(&self, _seq_ids: &[u64]) -> Result<()> {
        Ok(())
    }
    /// KV tokens this stage could hold in `memory_ratio` of its free memory (None = unknown).
    fn kv_capacity(&self, memory_ratio: f64) -> Result<Option<usize>>;
    /// Allocate `num_slots` KV tokens and `num_states` recurrent-state slots per layer.
    fn alloc_kv(&self, num_slots: usize, num_states: usize) -> Result<()>;
    /// Resize the device expert cache (offload/hybrid MoE); a no-op for stages without one.
    fn resize_expert_cache(&self, experts: usize) -> Result<()>;
    /// Copy recurrent-state slot `src` into `dst` (hybrid models' prefix snapshots).
    fn copy_state(&self, src: u32, dst: u32) -> Result<()>;
    fn stats(&self) -> Vec<StageStats>;
}

/// Weights, config and options needed to build stages.
#[derive(Clone)]
pub struct BuildCtx {
    pub src: Arc<dyn WeightSource>,
    pub cfg: Arc<ModelConfig>,
    pub dtype: Option<DType>,
    pub moe: MoeOptions,
    pub max_seq_len: usize,
    /// Single-device stages capture decode steps as CUDA graphs.
    pub cuda_graphs: bool,
}

impl BuildCtx {
    pub fn dtype_for(&self, device: &Device) -> DType {
        self.dtype
            .unwrap_or_else(|| crate::device::default_dtype(device))
    }
}

pub struct LocalStage {
    stack: LayerStack,
}

impl LocalStage {
    pub fn build(ctx: &BuildCtx, range: Range<usize>, device: Device) -> Result<Self> {
        let p = Placement::single(device.clone(), ctx.dtype_for(&device));
        let stack = LayerStack::load(
            &*ctx.src,
            ctx.cfg.clone(),
            range,
            p,
            Arc::new(NoComm),
            &ctx.moe,
            ctx.max_seq_len,
        )?;
        stack.enable_cuda_graphs(ctx.cuda_graphs);
        Ok(Self { stack })
    }
}

impl Stage for LocalStage {
    fn expert_bank(&self) -> Option<crate::model::moe::Bank> {
        self.stack.expert_bank()
    }
    fn ple_prefetchers(&self) -> Vec<crate::model::hyper::PlePrefetch> {
        self.stack.ple_prefetchers()
    }
    fn layers(&self) -> Range<usize> {
        self.stack.range.clone()
    }
    fn forward(&self, meta: &Arc<BatchMeta>, hidden: Tensor) -> Result<Tensor> {
        self.stack.forward(meta, &hidden)
    }
    fn kv_capacity(&self, memory_ratio: f64) -> Result<Option<usize>> {
        Ok(self.stack.kv_capacity(memory_ratio))
    }
    fn alloc_kv(&self, num_slots: usize, num_states: usize) -> Result<()> {
        self.stack.alloc_kv(num_slots, num_states)
    }
    fn resize_expert_cache(&self, experts: usize) -> Result<()> {
        self.stack.resize_expert_cache(experts);
        Ok(())
    }
    fn copy_state(&self, src: u32, dst: u32) -> Result<()> {
        self.stack.copy_state(src, dst)
    }
    fn stats(&self) -> Vec<StageStats> {
        vec![self.stack.stats()]
    }
}

enum TpCmd {
    Forward(Arc<BatchMeta>, Tensor, mpsc::Sender<Result<Tensor>>),
    Capacity(f64, mpsc::Sender<Option<usize>>),
    Alloc(usize, usize, mpsc::Sender<Result<()>>),
    ResizeExperts(usize, mpsc::Sender<()>),
    CopyState(u32, u32, mpsc::Sender<Result<()>>),
    Stats(mpsc::Sender<StageStats>),
}

/// Tensor-parallel stage: one thread per rank/device, attention heads and dense MLPs are
/// sharded, MoE experts are distributed across ranks, partial sums are all-reduced.
pub struct TpStage {
    range: Range<usize>,
    ranks: Vec<mpsc::Sender<TpCmd>>,
    poison: Vec<Arc<dyn Communicator>>,
}

/// Deferred communicator constructor; NCCL ranks must initialise concurrently on their threads.
type CommFactory = Box<dyn FnOnce() -> Result<Arc<dyn Communicator>> + Send>;

fn make_comms(devices: &[Device]) -> Result<Vec<CommFactory>> {
    let world = devices.len();
    #[cfg(feature = "nccl")]
    if devices.iter().all(|d| d.is_cuda()) {
        let id = comm::nccl::new_id()?;
        return Ok(devices
            .iter()
            .enumerate()
            .map(|(rank, d)| {
                let d = d.clone();
                Box::new(move || -> Result<Arc<dyn Communicator>> {
                    Ok(Arc::new(comm::nccl::NcclComm::new(&d, rank, world, id)?))
                }) as CommFactory
            })
            .collect());
    }
    let group = HostGroup::new(world);
    Ok((0..world)
        .map(|rank| {
            let c = group.comm(rank);
            Box::new(move || -> Result<Arc<dyn Communicator>> { Ok(Arc::new(c)) }) as CommFactory
        })
        .collect())
}

impl TpStage {
    pub fn build(ctx: &BuildCtx, range: Range<usize>, devices: Vec<Device>) -> Result<Self> {
        let world = devices.len();
        let comms = make_comms(&devices)?;
        let (ready_tx, ready_rx) = mpsc::channel::<(usize, Result<Arc<dyn Communicator>>)>();
        let mut ranks = Vec::with_capacity(world);
        for (rank, (device, make_comm)) in devices.into_iter().zip(comms).enumerate() {
            let (tx, rx) = mpsc::channel::<TpCmd>();
            ranks.push(tx);
            let ctx = ctx.clone();
            let range = range.clone();
            let ready_tx = ready_tx.clone();
            std::thread::Builder::new()
                .name(format!("carbon-tp{rank}"))
                .spawn(move || {
                    let built = (|| -> Result<(LayerStack, Arc<dyn Communicator>)> {
                        let comm = make_comm()?;
                        let p = Placement {
                            device: device.clone(),
                            dtype: ctx.dtype_for(&device),
                            rank,
                            world,
                            keep_quantized: false,
                        };
                        let stack = LayerStack::load(
                            &*ctx.src,
                            ctx.cfg.clone(),
                            range,
                            p,
                            comm.clone(),
                            &ctx.moe,
                            ctx.max_seq_len,
                        )?;
                        Ok((stack, comm))
                    })();
                    let (stack, comm) = match built {
                        Ok(x) => x,
                        Err(e) => {
                            let _ = ready_tx.send((rank, Err(e)));
                            return;
                        }
                    };
                    let _ = ready_tx.send((rank, Ok(comm.clone())));
                    drop(ready_tx);
                    while let Ok(cmd) = rx.recv() {
                        match cmd {
                            TpCmd::Forward(meta, hidden, reply) => {
                                let r = stack.forward(&meta, &hidden);
                                if r.is_err() {
                                    comm.poison();
                                }
                                let _ = reply.send(r);
                            }
                            TpCmd::Capacity(ratio, reply) => {
                                let _ = reply.send(stack.kv_capacity(ratio));
                            }
                            TpCmd::Alloc(n, m, reply) => {
                                let _ = reply.send(stack.alloc_kv(n, m));
                            }
                            TpCmd::CopyState(src, dst, reply) => {
                                let _ = reply.send(stack.copy_state(src, dst));
                            }
                            TpCmd::ResizeExperts(n, reply) => {
                                // each rank holds 1/world of the experts
                                stack.resize_expert_cache(n.div_ceil(world));
                                let _ = reply.send(());
                            }
                            TpCmd::Stats(reply) => {
                                let mut s = stack.stats();
                                s.name = format!("tp-rank{rank}");
                                let _ = reply.send(s);
                            }
                        }
                    }
                })?;
        }
        drop(ready_tx);
        let mut poison = Vec::with_capacity(world);
        let mut first_err = None;
        for _ in 0..world {
            match ready_rx.recv() {
                Ok((_, Ok(c))) => poison.push(c),
                Ok((rank, Err(e))) => {
                    first_err.get_or_insert(anyhow!(
                        "tensor-parallel rank {rank} failed to load: {e:#}"
                    ));
                }
                Err(_) => {
                    first_err
                        .get_or_insert(anyhow!("tensor-parallel rank thread died during load"));
                }
            }
        }
        if let Some(e) = first_err {
            for c in &poison {
                c.poison();
            }
            return Err(e);
        }
        Ok(Self {
            range,
            ranks,
            poison,
        })
    }

    fn broadcast<T>(&self, make: impl Fn(mpsc::Sender<T>) -> TpCmd) -> Result<Vec<T>> {
        let mut rxs = Vec::with_capacity(self.ranks.len());
        for tx in &self.ranks {
            let (rtx, rrx) = mpsc::channel();
            tx.send(make(rtx))
                .map_err(|_| anyhow!("tensor-parallel rank thread is gone"))?;
            rxs.push(rrx);
        }
        rxs.into_iter()
            .map(|r| {
                r.recv()
                    .map_err(|_| anyhow!("tensor-parallel rank thread is gone"))
            })
            .collect()
    }
}

impl Stage for TpStage {
    fn layers(&self) -> Range<usize> {
        self.range.clone()
    }

    fn forward(&self, meta: &Arc<BatchMeta>, hidden: Tensor) -> Result<Tensor> {
        // each rank thread reads `hidden` on its own CUDA stream (candle's per-thread default
        // stream), which is not ordered after the work that produced it on this thread's
        hidden.device().synchronize()?;
        let results = self.broadcast(|tx| TpCmd::Forward(meta.clone(), hidden.clone(), tx))?;
        let mut out = None;
        for r in results {
            match r {
                Ok(t) => {
                    out.get_or_insert(t);
                }
                Err(e) => {
                    for c in &self.poison {
                        c.poison();
                    }
                    return Err(e);
                }
            }
        }
        out.ok_or_else(|| anyhow!("empty tensor-parallel group"))
    }

    fn kv_capacity(&self, memory_ratio: f64) -> Result<Option<usize>> {
        let caps = self.broadcast(|tx| TpCmd::Capacity(memory_ratio, tx))?;
        Ok(caps
            .into_iter()
            .try_fold(usize::MAX, |acc, c| c.map(|c| acc.min(c))))
    }

    fn alloc_kv(&self, num_slots: usize, num_states: usize) -> Result<()> {
        self.broadcast(|tx| TpCmd::Alloc(num_slots, num_states, tx))?
            .into_iter()
            .collect()
    }

    fn resize_expert_cache(&self, experts: usize) -> Result<()> {
        self.broadcast(|tx| TpCmd::ResizeExperts(experts, tx))
            .map(|_| ())
    }

    fn copy_state(&self, src: u32, dst: u32) -> Result<()> {
        self.broadcast(|tx| TpCmd::CopyState(src, dst, tx))?
            .into_iter()
            .collect()
    }

    fn stats(&self) -> Vec<StageStats> {
        self.broadcast(TpCmd::Stats).unwrap_or_default()
    }
}

/// Several stages run back to back (used by RPC workers that own more than one device group).
pub struct PipelineStage {
    pub stages: Vec<Box<dyn Stage>>,
}

impl Stage for PipelineStage {
    fn layers(&self) -> Range<usize> {
        let start = self.stages.first().map(|s| s.layers().start).unwrap_or(0);
        let end = self.stages.last().map(|s| s.layers().end).unwrap_or(start);
        start..end
    }
    fn forward(&self, meta: &Arc<BatchMeta>, mut hidden: Tensor) -> Result<Tensor> {
        for s in &self.stages {
            hidden = s.forward(meta, hidden)?;
        }
        Ok(hidden)
    }
    fn ple_prefetchers(&self) -> Vec<crate::model::hyper::PlePrefetch> {
        self.stages.iter().flat_map(|s| s.ple_prefetchers()).collect()
    }
    fn release(&self, seq_ids: &[u64]) -> Result<()> {
        self.stages.iter().try_for_each(|s| s.release(seq_ids))
    }
    fn kv_capacity(&self, memory_ratio: f64) -> Result<Option<usize>> {
        let mut acc: Option<usize> = Some(usize::MAX);
        for s in &self.stages {
            acc = match (acc, s.kv_capacity(memory_ratio)?) {
                (Some(a), Some(c)) => Some(a.min(c)),
                _ => None,
            };
        }
        Ok(acc)
    }
    fn alloc_kv(&self, num_slots: usize, num_states: usize) -> Result<()> {
        self.stages
            .iter()
            .try_for_each(|s| s.alloc_kv(num_slots, num_states))
    }
    fn resize_expert_cache(&self, experts: usize) -> Result<()> {
        self.stages
            .iter()
            .try_for_each(|s| s.resize_expert_cache(experts))
    }
    fn copy_state(&self, src: u32, dst: u32) -> Result<()> {
        self.stages.iter().try_for_each(|s| s.copy_state(src, dst))
    }
    fn stats(&self) -> Vec<StageStats> {
        self.stages.iter().flat_map(|s| s.stats()).collect()
    }
}

/// Build the stage for one device group: a plain local stage for one device,
/// a tensor-parallel stage for several.
pub fn build_group(
    ctx: &BuildCtx,
    range: Range<usize>,
    devices: &[String],
) -> Result<Box<dyn Stage>> {
    let devs: Vec<Device> = devices
        .iter()
        .map(|d| crate::device::parse_device(d))
        .collect::<Result<_>>()?;
    match devs.len() {
        0 => bail!("empty device group"),
        1 => Ok(Box::new(LocalStage::build(
            ctx,
            range,
            devs.into_iter().next().unwrap(),
        )?)),
        _ => Ok(Box::new(TpStage::build(ctx, range, devs)?)),
    }
}

/// Split `num_layers` across stages. `split` holds either exact per-stage counts
/// (summing to `num_layers`) or relative weights; empty = as even as possible.
pub fn split_layers(num_layers: usize, stages: usize, split: &[f64]) -> Result<Vec<Range<usize>>> {
    if stages == 0 {
        bail!("no pipeline stages");
    }
    let counts: Vec<usize> = if split.is_empty() {
        (0..stages)
            .map(|i| num_layers / stages + usize::from(i < num_layers % stages))
            .collect()
    } else {
        if split.len() != stages {
            bail!(
                "--layer-split has {} entries but there are {stages} stages",
                split.len()
            );
        }
        if split.iter().any(|w| *w < 0.0) {
            bail!("--layer-split entries must be >= 0");
        }
        let sum: f64 = split.iter().sum();
        if split.iter().all(|w| w.fract() == 0.0) && sum as usize == num_layers {
            split.iter().map(|w| *w as usize).collect()
        } else {
            // largest-remainder apportionment of the weights
            let exact: Vec<f64> = split.iter().map(|w| w / sum * num_layers as f64).collect();
            let mut c: Vec<usize> = exact.iter().map(|x| x.floor() as usize).collect();
            let mut rest = num_layers - c.iter().sum::<usize>();
            let mut order: Vec<usize> = (0..stages).collect();
            order.sort_by(|&a, &b| {
                (exact[b] - exact[b].floor()).total_cmp(&(exact[a] - exact[a].floor()))
            });
            for i in order {
                if rest == 0 {
                    break;
                }
                c[i] += 1;
                rest -= 1;
            }
            c
        }
    };
    let mut ranges = Vec::with_capacity(stages);
    let mut start = 0;
    for c in counts {
        ranges.push(start..start + c);
        start += c;
    }
    Ok(ranges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_splits() {
        assert_eq!(split_layers(10, 3, &[]).unwrap(), vec![0..4, 4..7, 7..10]);
        assert_eq!(split_layers(10, 2, &[6.0, 4.0]).unwrap(), vec![0..6, 6..10]);
        assert_eq!(split_layers(10, 2, &[3.0, 1.0]).unwrap(), vec![0..8, 8..10]);
        assert_eq!(
            split_layers(4, 3, &[0.0, 2.0, 2.0]).unwrap(),
            vec![0..0, 0..2, 2..4]
        );
        assert!(split_layers(4, 3, &[1.0]).is_err());
    }
}
