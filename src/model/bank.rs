// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Offloaded experts on the GPU without host round trips (`--moe-strategy offload` on CUDA).
//!
//! Every MoE layer of a stage keeps its experts in pinned, device-mapped host memory, one array
//! per expert component (gate/up/down weight and scale, expert-major). A shared device bank of
//! `S` slots per component caches them. A decode step routes on the GPU (`moe_route`), makes
//! the routed experts resident with `moe_slot_ensure` (least recently used slots, misses copied
//! straight from pinned memory by a kernel at PCIe speed) and runs the fused expert kernels
//! through the layer's pointer table, which the ensure kernel points at the slots. Nothing waits
//! on the host, so offloaded decode steps can be captured as CUDA graphs.
//!
//! Prefill chunks use the same cache: the route and the layer's slot map are read back once per
//! layer and each routed expert runs as a grouped projection over views into the bank.

use super::layers::{Mlp, QLinear};
use crate::kernels::cuda::{self as k, ExpertProj, MoeTable, PinnedHost, SlotState};
use crate::weights::Nvfp4;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::Linear;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Pieces of one expert as the kernels read them: gate, up, down (weight, scale) each.
const COMPONENTS: usize = 6;

/// One expert component's per-expert shape and type (identical for every banked layer).
#[derive(Clone, Debug, PartialEq)]
struct Component {
    dtype: DType,
    shape: Vec<usize>,
}

impl Component {
    fn bytes(&self) -> usize {
        self.shape.iter().product::<usize>() * self.dtype.size_in_bytes()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Schema {
    fmt: &'static str,
    comps: [Option<Component>; COMPONENTS],
}

impl Schema {
    fn bytes(&self, c: usize) -> usize {
        self.comps[c].as_ref().map_or(0, Component::bytes)
    }
    fn expert_bytes(&self) -> usize {
        (0..COMPONENTS).map(|c| self.bytes(c)).sum()
    }
}

/// An expert's projections split into components; FP8 scales become f32, as the kernels read them.
fn components(p: &[ExpertProj; 3]) -> Result<([Option<Tensor>; COMPONENTS], [f32; 3])> {
    let mut out: [Option<Tensor>; COMPONENTS] = Default::default();
    let mut glob = [1f32; 3];
    for (j, proj) in p.iter().enumerate() {
        let (w, s) = match proj {
            ExpertProj::Dense(w) => (w.clone(), None),
            ExpertProj::Fp8 { w, scale } => (w.clone(), Some(scale.to_dtype(DType::F32)?)),
            ExpertProj::Nvfp4 {
                packed,
                scale,
                global,
            } => {
                glob[j] = *global;
                (packed.clone(), Some(scale.clone()))
            }
        };
        out[2 * j] = Some(w.contiguous()?);
        out[2 * j + 1] = s.map(|s| s.contiguous()).transpose()?;
    }
    Ok((out, glob))
}

fn fmt_name(p: &ExpertProj) -> &'static str {
    match p {
        ExpertProj::Dense(_) => "dense",
        ExpertProj::Fp8 { .. } => "fp8",
        ExpertProj::Nvfp4 { .. } => "nvfp4",
    }
}

/// Copy the bytes of a contiguous host tensor into `dst` (exactly its size).
fn copy_bytes(t: &Tensor, dst: &mut [u8]) -> Result<()> {
    use candle_core::{CpuStorage, Storage};
    let (st, l) = t.storage_and_layout();
    let (a, b) = l.contiguous_offsets().context("expert tensor is not contiguous")?;
    let Storage::Cpu(cpu) = &*st else {
        bail!("expert tensor is not in host memory")
    };
    macro_rules! raw {
        ($v:expr) => {{
            let v = &$v[a..b];
            // SAFETY: plain-old-data elements viewed as their bytes while the guard is held
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
        }};
    }
    let src: &[u8] = match cpu {
        CpuStorage::U8(v) => raw!(v),
        CpuStorage::U32(v) => raw!(v),
        CpuStorage::BF16(v) => raw!(v),
        CpuStorage::F16(v) => raw!(v),
        CpuStorage::F32(v) => raw!(v),
        CpuStorage::F8E4M3(v) => raw!(v),
        _ => bail!("unsupported expert dtype {:?}", t.dtype()),
    };
    if src.len() != dst.len() {
        bail!("expert tensor is {} bytes, expected {}", src.len(), dst.len());
    }
    dst.copy_from_slice(src);
    Ok(())
}

/// One banked layer's host copy: component `c` of expert `e` at `off[c] + e * bytes(c)`.
struct HostLayer {
    mem: PinnedHost,
    off: [usize; COMPONENTS],
    globals: Vec<[f32; 3]>,
}

/// The device side, (re)allocated by `set_capacity`.
struct DeviceSide {
    slots: usize,
    banks: [Option<Tensor>; COMPONENTS],
    slot_of: Tensor,
    owner: Tensor,
    usage: Tensor,
    step: Tensor,
    stats: Tensor,
    copy_n: Tensor,
    copy_slot: Tensor,
    copy_exp: Tensor,
    /// Per layer row: src[6], dst[6], bytes[6] as i64.
    descs: Vec<Tensor>,
    copy_blocks: u32,
    /// Prefill copies run here, in batches, while the layer computes the experts it has.
    copy_stream: k::RawStream,
}

enum Step {
    Both,
    Assign,
    Copy,
}

/// The slot cache shared by every banked MoE layer of a stage.
pub struct ExpertBank {
    device: Device,
    cuda: candle_core::CudaDevice,
    num_experts: usize,
    schema: Mutex<Option<Schema>>,
    layers: Mutex<Vec<Arc<HostLayer>>>,
    side: Mutex<Option<DeviceSide>>,
    capacity: AtomicUsize,
}

impl ExpertBank {
    pub fn new(device: &Device, num_experts: usize) -> Result<Arc<Self>> {
        let Device::Cuda(cuda) = device else {
            bail!("the expert slot cache needs a CUDA device")
        };
        if num_experts > 1024 {
            bail!("the expert slot cache handles at most 1024 experts per layer");
        }
        Ok(Arc::new(Self {
            device: device.clone(),
            cuda: cuda.clone(),
            num_experts,
            schema: Mutex::new(None),
            layers: Mutex::new(Vec::new()),
            side: Mutex::new(None),
            capacity: AtomicUsize::new(0),
        }))
    }

    /// Whether this expert can be banked (the fused kernels read its format).
    pub fn supports(expert: &Mlp) -> bool {
        expert.kernel_projections().is_some()
    }

    /// The banked experts' format ("nvfp4", "fp8", "dense") and, for dense ones, their dtype.
    pub fn format(&self) -> Option<(&'static str, DType)> {
        self.schema
            .lock()
            .as_ref()
            .and_then(|s| s.comps[0].as_ref().map(|c| (s.fmt, c.dtype)))
    }

    /// Bytes one expert takes in a slot.
    pub fn expert_bytes(&self) -> usize {
        self.schema.lock().as_ref().map_or(0, Schema::expert_bytes)
    }

    /// Copy a layer's host experts (`None` = not local) into pinned memory; returns its row and
    /// the pointer table its kernels read. The host experts can be dropped afterwards.
    pub fn register(&self, experts: &[Option<Arc<Mlp>>]) -> Result<(usize, MoeTable)> {
        if experts.len() != self.num_experts || experts.iter().any(Option::is_none) {
            bail!("the expert slot cache needs every expert of the layer local");
        }
        let projs: Vec<[ExpertProj; 3]> = experts
            .iter()
            .map(|m| {
                m.as_ref()
                    .unwrap()
                    .kernel_projections()
                    .context("expert format not supported by the fused kernels")
            })
            .collect::<Result<_>>()?;
        let mut comps = Vec::with_capacity(projs.len());
        let mut globals = Vec::with_capacity(projs.len());
        for p in &projs {
            let (c, g) = components(p)?;
            comps.push(c);
            globals.push(g);
        }
        let shape_of = |c: &[Option<Tensor>; COMPONENTS]| -> [Option<Component>; COMPONENTS] {
            std::array::from_fn(|i| {
                c[i].as_ref().map(|t| Component {
                    dtype: t.dtype(),
                    shape: t.dims().to_vec(),
                })
            })
        };
        let schema = Schema {
            fmt: fmt_name(&projs[0][0]),
            comps: shape_of(&comps[0]),
        };
        if comps.iter().any(|c| shape_of(c) != schema.comps) {
            bail!("experts of a layer differ in shape or type");
        }
        {
            let mut s = self.schema.lock();
            match &*s {
                Some(prev) if *prev != schema => bail!("MoE layers differ in expert shape or format"),
                Some(_) => {}
                None => *s = Some(schema.clone()),
            }
        }
        // component arrays, each 256-byte aligned
        let mut off = [0usize; COMPONENTS];
        let mut total = 0usize;
        for (c, o) in off.iter_mut().enumerate() {
            *o = total;
            total = (total + schema.bytes(c) * self.num_experts).next_multiple_of(256);
        }
        let mut mem = PinnedHost::new(&self.cuda, total)
            .with_context(|| format!("allocating {:.2} GiB of pinned host memory", total as f64 / (1u64 << 30) as f64))?;
        {
            let buf = mem.as_mut_slice();
            for (e, c) in comps.iter().enumerate() {
                for (i, t) in c.iter().enumerate() {
                    let Some(t) = t else { continue };
                    let n = schema.bytes(i);
                    copy_bytes(t, &mut buf[off[i] + e * n..off[i] + (e + 1) * n])?;
                }
            }
        }
        let table = MoeTable::for_slots(&projs[0], &globals, &self.device)?;
        let mut layers = self.layers.lock();
        layers.push(Arc::new(HostLayer { mem, off, globals }));
        if self.side.lock().is_some() {
            drop(layers);
            // a layer added after sizing: rebuild the device side so it gets a descriptor
            self.set_capacity(self.capacity())?;
            return Ok((self.layers.lock().len() - 1, table));
        }
        Ok((layers.len() - 1, table))
    }

    pub fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Relaxed)
    }

    /// Allocate (or reallocate, emptying it) the device bank with `slots` slots; at least one
    /// layer's worth, so a prefill chunk that routes to every expert still fits.
    pub fn set_capacity(&self, slots: usize) -> Result<()> {
        let schema = self.schema.lock().clone().context("no banked MoE layer")?;
        let slots = slots.max(self.num_experts);
        let dev = &self.device;
        let mut side = self.side.lock();
        *side = None; // free the old bank first
        let mut banks: [Option<Tensor>; COMPONENTS] = Default::default();
        for (b, comp) in banks.iter_mut().zip(&schema.comps) {
            if let Some(comp) = comp {
                let elems = comp.shape.iter().product::<usize>();
                *b = Some(Tensor::zeros((slots, elems), comp.dtype, dev)?);
            }
        }
        let layers = self.layers.lock();
        let rows = layers.len();
        let mut descs = Vec::with_capacity(rows);
        for l in layers.iter() {
            let mut d = [0i64; 18];
            for c in 0..COMPONENTS {
                if let Some(b) = &banks[c] {
                    d[c] = (l.mem.dptr + l.off[c] as u64) as i64;
                    d[6 + c] = k::device_address(b)? as i64;
                    d[12 + c] = schema.bytes(c) as i64;
                }
            }
            descs.push(Tensor::from_vec(d.to_vec(), 18, dev)?);
        }
        let e = self.num_experts;
        let sms = self
            .cuda
            .cuda_stream()
            .context()
            .attribute(candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap_or(64) as u32;
        *side = Some(DeviceSide {
            slots,
            banks,
            // (candle has no CUDA fill kernel for i32)
            slot_of: Tensor::from_vec(vec![-1i32; rows.max(1) * e], rows.max(1) * e, dev)?,
            owner: Tensor::from_vec(vec![-1i32; slots], slots, dev)?,
            usage: Tensor::zeros(slots, DType::I64, dev)?,
            step: Tensor::zeros(1, DType::I64, dev)?,
            stats: Tensor::zeros(3, DType::I64, dev)?,
            copy_n: Tensor::zeros(1, DType::U32, dev)?,
            copy_slot: Tensor::zeros(e, DType::U32, dev)?,
            copy_exp: Tensor::zeros(e, DType::U32, dev)?,
            descs,
            copy_blocks: 2 * sms,
            copy_stream: k::RawStream::new(&self.cuda)?,
        });
        self.capacity.store(slots, Ordering::Relaxed);
        Ok(())
    }

    /// Make the experts `route` selects resident for layer `row` and point `table` at them.
    pub fn ensure(&self, row: usize, table: &MoeTable, route: &Tensor, pairs: usize) -> Result<()> {
        self.run(row, table, route, pairs, Step::Both)
    }

    /// Assign slots for `route` (misses queued, not yet copied).
    pub fn assign(&self, row: usize, table: &MoeTable, route: &Tensor, pairs: usize) -> Result<()> {
        self.run(row, table, route, pairs, Step::Assign)
    }

    /// Copy the misses the last `assign` queued.
    pub fn copy_queued(&self, row: usize, table: &MoeTable, route: &Tensor, pairs: usize) -> Result<()> {
        self.run(row, table, route, pairs, Step::Copy)
    }

    fn run(&self, row: usize, table: &MoeTable, route: &Tensor, pairs: usize, step: Step) -> Result<()> {
        let side = self.side.lock();
        let s = side.as_ref().context("the expert slot cache is not allocated")?;
        let state = SlotState {
            slot_of: &s.slot_of,
            owner: &s.owner,
            usage: &s.usage,
            step: &s.step,
            stats: &s.stats,
            copy_n: &s.copy_n,
            copy_slot: &s.copy_slot,
            copy_exp: &s.copy_exp,
            slots: s.slots,
            copy_blocks: s.copy_blocks,
        };
        match step {
            Step::Both => k::moe_slot_ensure(&state, table, &s.descs[row], row, route, pairs, true)?,
            Step::Assign => k::moe_slot_ensure(&state, table, &s.descs[row], row, route, pairs, false)?,
            Step::Copy => k::moe_slot_copy(&state, table, &s.descs[row], route, pairs)?,
        }
        Ok(())
    }

    /// The experts the last `assign` queued for copy, in queue order (reads the device).
    pub fn copy_queue(&self) -> Result<Vec<u32>> {
        let side = self.side.lock();
        let s = side.as_ref().context("the expert slot cache is not allocated")?;
        let n = s.copy_n.to_vec1::<u32>()?[0] as usize;
        Ok(if n == 0 { vec![] } else { s.copy_exp.narrow(0, 0, n)?.to_vec1()? })
    }

    /// Copy queue entries `start..start + n` of layer `row` on the bank's copy stream; the
    /// returned event marks them resident. The assignment must be complete (`copy_queue` read).
    pub fn copy_batch(&self, row: usize, start: usize, n: usize) -> Result<k::RawEvent> {
        let side = self.side.lock();
        let s = side.as_ref().context("the expert slot cache is not allocated")?;
        let state = SlotState {
            slot_of: &s.slot_of,
            owner: &s.owner,
            usage: &s.usage,
            step: &s.step,
            stats: &s.stats,
            copy_n: &s.copy_n,
            copy_slot: &s.copy_slot,
            copy_exp: &s.copy_exp,
            slots: s.slots,
            copy_blocks: s.copy_blocks,
        };
        let stream = &s.copy_stream;
        k::moe_slot_copy_range_on(stream, &self.cuda, &state, &s.descs[row], start, n, s.copy_blocks / 2)?;
        Ok(stream.record()?)
    }

    /// Make the device's stream wait for `ev` (a [`Self::copy_batch`]).
    pub fn wait(&self, ev: &k::RawEvent) -> Result<()> {
        Ok(k::wait_event(&self.cuda, ev)?)
    }

    /// Layer `row`'s slot per expert (-1 = not resident); reads the device (synchronises).
    pub fn slot_row(&self, row: usize) -> Result<Vec<i32>> {
        let side = self.side.lock();
        let s = side.as_ref().context("the expert slot cache is not allocated")?;
        Ok(s.slot_of.narrow(0, row * self.num_experts, self.num_experts)?.to_vec1()?)
    }

    /// Expert `e` of layer `row` as an MLP over views of slot `slot` (valid until the slot is
    /// reused by a later `ensure`).
    pub fn expert(&self, row: usize, e: usize, slot: usize, act: &str) -> Result<Mlp> {
        let schema = self.schema.lock().clone().context("no banked MoE layer")?;
        let globals = self.layers.lock()[row].globals[e];
        let side = self.side.lock();
        let s = side.as_ref().context("the expert slot cache is not allocated")?;
        let view = |c: usize| -> Result<Option<Tensor>> {
            match (&s.banks[c], &schema.comps[c]) {
                (Some(b), Some(comp)) => Ok(Some(b.get(slot)?.reshape(comp.shape.as_slice())?)),
                _ => Ok(None),
            }
        };
        let proj = |j: usize| -> Result<QLinear> {
            let w = view(2 * j)?.context("missing expert weight")?;
            Ok(match schema.fmt {
                "nvfp4" => QLinear::Nvfp4(Nvfp4 {
                    packed: w,
                    scale: view(2 * j + 1)?.context("missing nvfp4 scale")?,
                    global: globals[j],
                }),
                "fp8" => QLinear::Fp8 {
                    w,
                    scale: view(2 * j + 1)?.context("missing fp8 scale")?,
                },
                _ => QLinear::Dense(Linear::new(w, None)),
            })
        };
        Ok(Mlp::new_q(proj(0)?, proj(1)?, proj(2)?, act))
    }

    /// Hits, copies (misses), misses that found no slot.
    pub fn counters(&self) -> [u64; 3] {
        let side = self.side.lock();
        side.as_ref()
            .and_then(|s| s.stats.to_vec1::<i64>().ok())
            .map(|v| [v[0] as u64, v[1] as u64, v[2] as u64])
            .unwrap_or_default()
    }

    pub fn stats(&self) -> super::moe::ExpertStats {
        let [hits, misses, _] = self.counters();
        let cap = self.capacity();
        super::moe::ExpertStats {
            capacity: cap,
            resident: (misses as usize).min(cap),
            hits,
            misses,
            cpu_computed: 0,
            bytes_copied: misses * self.expert_bytes() as u64,
        }
    }

    /// Pinned host bytes held for the banked layers.
    pub fn host_bytes(&self) -> usize {
        self.layers.lock().iter().map(|l| l.mem.len()).sum()
    }
}
