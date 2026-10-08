// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! CUDA graphs for decode steps (`--cuda-graphs`).
//!
//! A decode step of B sequences on one GPU, with every layer on carbon's fused kernels, launches
//! the same kernels with the same shapes whatever the sequences' lengths: lengths only reach the
//! kernels through device tables. Such a step is captured once per B into a CUDA graph over
//! static input buffers (hidden states, rotary rows, KV/state slots, a slot table sized to the
//! whole KV pool) and replayed after refreshing those buffers, which removes the per-launch CPU
//! cost (hundreds of launches per step).
//!
//! Safety nets: a batch size is captured the second time it is seen (the first run loads
//! modules and warms cuBLAS); a capture during which any fused path fell back to a generic one
//! is discarded (the generic paths have length-dependent shapes); any capture or launch error
//! turns graphs off for this stage and the step runs eagerly. Graphs are dropped whenever the
//! KV / state pools are reallocated.

#[cfg(not(feature = "cuda"))]
#[derive(Default)]
pub struct DecodeGraphs;

#[cfg(not(feature = "cuda"))]
impl DecodeGraphs {
    pub fn clear(&self) {}
    pub fn set_requested(&self, _: bool) {}
    pub fn captured(&self) -> usize {
        0
    }
    pub fn disabled_reason(&self) -> Option<String> {
        None
    }
}

#[cfg(feature = "cuda")]
pub use imp::DecodeGraphs;

#[cfg(feature = "cuda")]
mod imp {
    use super::super::attention::DeviceBatch;
    use super::super::{Caches, Ffn, LayerStack, Mixer};
    use crate::batch::BatchMeta;
    use anyhow::{anyhow, bail, Result};
    use candle_core::cuda_backend::cudarc::driver::{sys, CudaGraph};
    use candle_core::{Device, Tensor};
    use parking_lot::Mutex;
    use std::collections::HashMap;

    /// Largest decode batch captured, in sequences and in tokens (the fused MoE path's limit).
    const MAX_GRAPH_BATCH: usize = 16;
    const MAX_GRAPH_TOKENS: usize = 64;

    struct Entry {
        graph: CudaGraph,
        x_in: Tensor,
        y_out: Tensor,
        batch: DeviceBatch,
        /// Replays still to check against an eager step.
        validate: u32,
    }

    /// Replays per batch size checked against eager execution (`CARBON_GRAPH_VALIDATE`,
    /// default 1: the capture step and the first replay).
    fn validate_count() -> u32 {
        std::env::var("CARBON_GRAPH_VALIDATE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
    }

    /// The pool rows a decode step writes (KV at out_loc, recurrent / PLE state at the
    /// sequences' slots), saved so an eager reference step can be undone.
    struct Touched(Vec<(Tensor, Tensor, Tensor)>);

    impl Touched {
        fn save(c: &Caches, b: &DeviceBatch) -> Result<Self> {
            let mut v = vec![];
            let mut add = |pool: &Tensor, idx: &Tensor| -> Result<()> {
                let n = pool.dim(0)?;
                let flat = pool.reshape((n, pool.elem_count() / n.max(1)))?;
                let rows = flat.index_select(idx, 0)?.contiguous()?;
                v.push((flat, idx.clone(), rows));
                Ok(())
            };
            let out_loc = b.scatter_idx.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?;
            for kv in &c.kv {
                add(&kv.k, &out_loc)?;
                add(&kv.v, &out_loc)?;
                if let Some(i) = &kv.idx {
                    add(i, &out_loc)?;
                }
            }
            for st in &c.states {
                add(&st.conv, &b.state_idx)?;
                add(&st.ssm, &b.state_idx)?;
            }
            for t in &c.ple {
                add(t, &b.state_idx)?;
            }
            Ok(Self(v))
        }

        fn restore(&self) -> Result<()> {
            for (flat, idx, rows) in &self.0 {
                let (n, w) = rows.dims2()?;
                let i = idx.reshape((n, 1))?.broadcast_as((n, w))?.contiguous()?;
                flat.scatter_set(&i, rows, 0)?;
            }
            Ok(())
        }
    }

    /// Largest |a - b| relative to max |b|.
    fn rel_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a = a.to_dtype(candle_core::DType::F32)?;
        let b = b.to_dtype(candle_core::DType::F32)?;
        let d = (&a - &b)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let m = b.abs()?.max_all()?.to_scalar::<f32>()?;
        Ok(d / m.max(1e-6))
    }

    #[derive(Default)]
    struct State {
        requested: bool,
        disabled: bool,
        reason: Option<String>,
        /// Keyed by (sequences, new tokens per sequence).
        seen: HashMap<(usize, usize), u32>,
        entries: HashMap<(usize, usize), Entry>,
    }

    #[derive(Default)]
    pub struct DecodeGraphs(Mutex<State>);

    // CudaGraph holds raw driver handles; every use goes through the mutex.
    unsafe impl Send for DecodeGraphs {}
    unsafe impl Sync for DecodeGraphs {}

    impl DecodeGraphs {
        pub fn captured(&self) -> usize {
            self.0.lock().entries.len()
        }

        pub fn disabled_reason(&self) -> Option<String> {
            self.0.lock().reason.clone()
        }

        pub fn set_requested(&self, on: bool) {
            self.0.lock().requested = on;
        }

        /// Drop every graph (they point into the pools about to be freed).
        pub fn clear(&self) {
            let mut s = self.0.lock();
            s.entries.clear();
            s.seen.clear();
        }

        fn disable(&self, s: &mut State, why: &anyhow::Error) {
            if !s.disabled {
                tracing::warn!("cuda graphs disabled, decoding eagerly: {why:#}");
                s.reason = Some(format!("{why:#}"));
            }
            s.disabled = true;
            s.entries.clear();
        }
    }

    /// Fused-path switches; a capture is valid only if none flipped while it was recorded.
    fn fused_flags() -> [bool; 3] {
        [
            super::super::attention::fused_decode_enabled(),
            super::super::gdn::fused_gdn_enabled(),
            super::super::moe::fused_moe_enabled(),
        ]
    }

    impl LayerStack {
        /// Every layer of this stage runs a decode step on fused kernels.
        fn graph_compatible(&self) -> bool {
            fused_flags().iter().all(|&f| f)
                && self.layers.iter().all(|l| {
                    (match &l.mixer {
                        Mixer::Attn(a, _) => a.fused_decode_ready(),
                        Mixer::Linear(g, _) => g.fused_decode_ready(),
                    }) && match &l.ffn {
                        Ffn::Dense(_) => true,
                        Ffn::Moe(m) => m.fused_ready(),
                    }
                })
        }

        /// Run this decode step through a CUDA graph. `Ok(None)`: not applicable, run eagerly.
        pub(in crate::model) fn forward_graph(
            &self,
            meta: &BatchMeta,
            x: &Tensor,
            c: &Caches,
            b: &DeviceBatch,
        ) -> Result<Option<Tensor>> {
            let p = &self.placement;
            let bsz = meta.seqs.len();
            if !p.device.is_cuda()
                || self.comm.world() > 1
                || bsz == 0
                || bsz > MAX_GRAPH_BATCH
                || !meta.seqs.iter().all(|s| s.q_len == meta.seqs[0].q_len)
                || meta.seqs[0].q_len > super::super::attention::MULTI_DECODE_MAX
                || bsz * meta.seqs[0].q_len > MAX_GRAPH_TOKENS
                || !meta.mrope_layout.is_empty() // QSA with images: host block selection
            {
                return Ok(None);
            }
            let key = (bsz, meta.seqs[0].q_len);
            let mut s = self.graphs.0.lock();
            if !s.requested || s.disabled {
                return Ok(None);
            }
            if !self.graph_compatible() {
                self.graphs
                    .disable(&mut s, &anyhow!("a layer has no fused decode path"));
                return Ok(None);
            }
            if let Some(e) = s.entries.get(&key) {
                if !e.batch.graph_fits(b) {
                    // shared prefixes can make the slot tables outgrow the pool: run this one eagerly
                    return Ok(Some(self.run_layers(x.clone(), b, c)?));
                }
                let check = e.validate > 0;
                let saved = if check { Some(Touched::save(c, b)?) } else { None };
                let reference = match &saved {
                    Some(t) => {
                        let y = self.run_layers(x.clone(), b, c)?;
                        t.restore()?;
                        Some(y)
                    }
                    None => None,
                };
                let run = || -> Result<Tensor> {
                    e.x_in.slice_set(&x.contiguous()?, 0, 0)?;
                    e.batch.copy_from(b)?;
                    e.graph.launch()?;
                    Ok(e.y_out.copy()?)
                };
                let out = run().and_then(|y| match &reference {
                    Some(r) => {
                        let d = rel_diff(&y, r)?;
                        if d > 1e-3 {
                            bail!("a replayed cuda graph (decode batch {bsz}x{}) differs from eager decoding by {d:.3e}", key.1);
                        }
                        Ok(y)
                    }
                    None => Ok(y),
                });
                return match out {
                    Ok(y) => {
                        if let Some(e) = s.entries.get_mut(&key) {
                            e.validate = e.validate.saturating_sub(1);
                        }
                        Ok(Some(y))
                    }
                    Err(err) => {
                        self.graphs.disable(&mut s, &err);
                        // the graph may have written the pools: undo it and decode eagerly
                        if let Some(t) = &saved {
                            t.restore()?;
                        }
                        Ok(Some(self.run_layers(x.clone(), b, c)?))
                    }
                };
            }
            // candle uploads the dims/strides of non-contiguous operands (broadcasts) from a
            // temporary host buffer per op; captured, that copy would replay from freed host
            // memory. Under this guard such uploads come from a device cache keyed by content,
            // filled by the eager runs of this size below, and a capture that misses it (or
            // reads back to the host) fails instead of recording garbage.
            let Device::Cuda(dev) = &p.device else {
                return Ok(None);
            };
            let _htod_cache = dev.enable_cuda_graph_htod_cache();
            let seen = s.seen.entry(key).or_insert(0);
            *seen += 1;
            if *seen < 2 {
                return Ok(Some(self.run_layers(x.clone(), b, c)?)); // warm-up
            }
            // the eager result of this very step, to check the capture against
            let saved = if validate_count() > 0 { Some(Touched::save(c, b)?) } else { None };
            let reference = match &saved {
                Some(t) => {
                    let y = self.run_layers(x.clone(), b, c)?;
                    t.restore()?;
                    Some(y)
                }
                None => None,
            };
            match self.capture(x, b, c) {
                Ok(mut e) => {
                    let launched = e.graph.launch().map_err(anyhow::Error::from).and_then(|_| {
                        let y = e.y_out.copy()?;
                        if let Some(r) = &reference {
                            let d = rel_diff(&y, r)?;
                            if d > 1e-3 {
                                bail!("the cuda graph captured for decode batch {bsz}x{} differs from eager decoding by {d:.3e}", key.1);
                            }
                        }
                        Ok(y)
                    });
                    match launched {
                        Ok(y) => {
                            tracing::info!("captured a cuda graph for decode batch {bsz}x{}", key.1);
                            e.validate = validate_count().saturating_sub(1);
                            s.entries.insert(key, e);
                            Ok(Some(y))
                        }
                        Err(err) => {
                            self.graphs.disable(&mut s, &err);
                            if let Some(t) = &saved {
                                t.restore()?;
                            }
                            Ok(Some(self.run_layers(x.clone(), b, c)?))
                        }
                    }
                }
                Err(err) => {
                    // nothing ran during the capture: the step still has to be computed
                    self.graphs.disable(&mut s, &err);
                    Ok(Some(self.run_layers(x.clone(), b, c)?))
                }
            }
        }

        fn capture(&self, x: &Tensor, b: &DeviceBatch, c: &Caches) -> Result<Entry> {
            let Device::Cuda(dev) = &self.placement.device else {
                bail!("not a cuda device")
            };
            let capacity = c.kv.first().map(|kv| kv.k.dim(0)).transpose()?.unwrap_or(0);
            // static inputs, allocated before the capture so the graph only reads them
            let batch = b.graph_static(capacity)?;
            let x_in = x.contiguous()?.copy()?;
            let y_out = x_in.zeros_like()?;
            let stream = dev.cuda_stream();
            stream.synchronize()?;
            let flags = fused_flags();
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
            let recorded = self
                .run_layers(x_in.clone(), &batch, c)
                .and_then(|y| Ok(y_out.slice_set(&y, 0, 0)?));
            let graph = stream.end_capture(sys::CUgraphInstantiate_flags(0));
            recorded?;
            let graph = graph?.ok_or_else(|| anyhow!("empty capture"))?;
            if fused_flags() != flags {
                bail!("a fused kernel fell back while the step was being captured");
            }
            graph.upload()?;
            Ok(Entry {
                graph,
                x_in,
                y_out,
                batch,
                validate: 0,
            })
        }
    }
}
