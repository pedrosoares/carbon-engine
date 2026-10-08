// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Collective communication for tensor / expert parallelism.
//!
//! * [`NoComm`]: world size 1.
//! * [`HostComm`]: all-reduce through host memory. Works between any devices
//!   (CPU threads, GPUs without peer access) and is what the test-suite uses.
//! * `NcclComm` (feature `nccl`): NCCL all-reduce directly between CUDA devices.

use anyhow::{bail, Result};
use candle_core::{DType, Tensor};
use parking_lot::{Condvar, Mutex};
use std::sync::Arc;

pub trait Communicator: Send + Sync {
    fn rank(&self) -> usize;
    fn world(&self) -> usize;
    /// Sum `x` across every rank; every rank receives the same result.
    fn all_reduce(&self, x: &Tensor) -> Result<Tensor>;
    /// Mark the group as failed so peers blocked in a collective return an error.
    fn poison(&self) {}
}

pub struct NoComm;

impl Communicator for NoComm {
    fn rank(&self) -> usize {
        0
    }
    fn world(&self) -> usize {
        1
    }
    fn all_reduce(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}

struct BarrierState {
    count: usize,
    generation: u64,
    poisoned: bool,
}

/// A reusable barrier that can be poisoned so a failing rank never deadlocks its peers.
struct PoisonBarrier {
    n: usize,
    state: Mutex<BarrierState>,
    cv: Condvar,
}

impl PoisonBarrier {
    fn new(n: usize) -> Self {
        Self {
            n,
            state: Mutex::new(BarrierState {
                count: 0,
                generation: 0,
                poisoned: false,
            }),
            cv: Condvar::new(),
        }
    }

    fn wait(&self) -> Result<()> {
        let mut s = self.state.lock();
        if s.poisoned {
            bail!("tensor-parallel group poisoned by a failed rank");
        }
        let gen = s.generation;
        s.count += 1;
        if s.count == self.n {
            s.count = 0;
            s.generation += 1;
            self.cv.notify_all();
            return Ok(());
        }
        while s.generation == gen && !s.poisoned {
            self.cv.wait(&mut s);
        }
        if s.poisoned {
            bail!("tensor-parallel group poisoned by a failed rank");
        }
        Ok(())
    }

    fn poison(&self) {
        let mut s = self.state.lock();
        s.poisoned = true;
        self.cv.notify_all();
    }
}

pub struct HostGroup {
    world: usize,
    buffers: Mutex<Vec<Vec<f32>>>,
    barrier: PoisonBarrier,
}

impl HostGroup {
    pub fn new(world: usize) -> Arc<Self> {
        Arc::new(Self {
            world,
            buffers: Mutex::new(vec![vec![]; world]),
            barrier: PoisonBarrier::new(world),
        })
    }

    pub fn comm(self: &Arc<Self>, rank: usize) -> HostComm {
        assert!(rank < self.world);
        HostComm {
            group: self.clone(),
            rank,
        }
    }
}

pub struct HostComm {
    group: Arc<HostGroup>,
    rank: usize,
}

impl Communicator for HostComm {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world(&self) -> usize {
        self.group.world
    }

    fn all_reduce(&self, x: &Tensor) -> Result<Tensor> {
        if self.group.world == 1 {
            return Ok(x.clone());
        }
        let (dtype, device, shape) = (x.dtype(), x.device().clone(), x.shape().clone());
        let local: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        self.group.buffers.lock()[self.rank] = local;
        self.group.barrier.wait()?;
        // every rank sums in rank order so the results are bit-identical across ranks
        let sum = {
            let bufs = self.group.buffers.lock();
            let mut acc = bufs[0].clone();
            for b in bufs.iter().skip(1) {
                if b.len() != acc.len() {
                    drop(bufs);
                    self.group.barrier.poison();
                    bail!("all_reduce size mismatch between ranks");
                }
                for (a, v) in acc.iter_mut().zip(b) {
                    *a += v;
                }
            }
            acc
        };
        // nobody may overwrite its buffer before all ranks have read
        self.group.barrier.wait()?;
        Ok(Tensor::from_vec(sum, shape, &device)?.to_dtype(dtype)?)
    }

    fn poison(&self) {
        self.group.barrier.poison();
    }
}

#[cfg(feature = "nccl")]
pub mod nccl {
    use super::Communicator;
    use anyhow::{bail, Result};
    use candle_core::backend::BackendStorage;
    use candle_core::{CpuStorage, CudaStorage, CustomOp1, DType, Layout, Shape, Tensor};
    use cudarc::nccl::safe::{Comm, Id, ReduceOp};
    use half::{bf16, f16};
    use std::sync::Arc;

    pub use cudarc::nccl::safe::Id as NcclId;

    pub fn new_id() -> Result<Id> {
        Id::new().map_err(|e| anyhow::anyhow!("nccl id: {e:?}"))
    }

    /// NCCL communicators are used from one rank thread at a time.
    struct SharedComm(Comm);
    // SAFETY: each rank's Comm is only driven by that rank's thread; the Arc exists so the
    // custom op can borrow it while the tensor op runs on the same thread.
    unsafe impl Send for SharedComm {}
    unsafe impl Sync for SharedComm {}

    pub struct NcclComm {
        comm: Arc<SharedComm>,
        rank: usize,
        world: usize,
    }

    impl NcclComm {
        /// Must be called concurrently from every rank (the call blocks until all ranks join).
        pub fn new(
            device: &candle_core::Device,
            rank: usize,
            world: usize,
            id: Id,
        ) -> Result<Self> {
            let candle_core::Device::Cuda(dev) = device else {
                bail!("NCCL requires a CUDA device")
            };
            let comm = Comm::from_rank(dev.cuda_stream(), rank, world, id)
                .map_err(|e| anyhow::anyhow!("nccl init: {e:?}"))?;
            Ok(Self {
                comm: Arc::new(SharedComm(comm)),
                rank,
                world,
            })
        }
    }

    struct AllReduce {
        comm: Arc<SharedComm>,
    }

    impl CustomOp1 for AllReduce {
        fn name(&self) -> &'static str {
            "nccl-allreduce"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("nccl all-reduce is only available on CUDA tensors")
        }

        fn cuda_fwd(
            &self,
            s: &CudaStorage,
            l: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let elem = l.shape().elem_count();
            let dev = s.device().clone();
            let (start, end) = match l.contiguous_offsets() {
                Some(o) => o,
                None => candle_core::bail!("all-reduce input must be contiguous"),
            };
            macro_rules! reduce {
                ($t:ty) => {{
                    let src = s.as_cuda_slice::<$t>()?.slice(start..end);
                    let mut dst = unsafe { dev.alloc::<$t>(elem) }?;
                    self.comm
                        .0
                        .all_reduce(&src, &mut dst, &ReduceOp::Sum)
                        .map_err(|e| candle_core::Error::Msg(format!("nccl all-reduce: {e:?}")))?;
                    CudaStorage::wrap_cuda_slice(dst, dev)
                }};
            }
            let out = match s.dtype() {
                DType::BF16 => reduce!(bf16),
                DType::F16 => reduce!(f16),
                DType::F32 => reduce!(f32),
                d => candle_core::bail!("nccl all-reduce: unsupported dtype {d:?}"),
            };
            Ok((out, l.shape().clone()))
        }
    }

    impl Communicator for NcclComm {
        fn rank(&self) -> usize {
            self.rank
        }
        fn world(&self) -> usize {
            self.world
        }
        fn all_reduce(&self, x: &Tensor) -> Result<Tensor> {
            Ok(x.contiguous()?.apply_op1_no_bwd(&AllReduce {
                comm: self.comm.clone(),
            })?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn host_all_reduce_sums() {
        let g = HostGroup::new(3);
        let handles: Vec<_> = (0..3)
            .map(|r| {
                let c = g.comm(r);
                std::thread::spawn(move || {
                    let mut out = vec![];
                    for step in 0..4 {
                        let x = Tensor::new(&[r as f32 + step as f32, 1.0], &Device::Cpu).unwrap();
                        out.push(c.all_reduce(&x).unwrap().to_vec1::<f32>().unwrap());
                    }
                    out
                })
            })
            .collect();
        for h in handles {
            let out = h.join().unwrap();
            for (step, v) in out.iter().enumerate() {
                assert_eq!(v, &vec![3.0 + 3.0 * step as f32, 3.0]);
            }
        }
    }

    #[test]
    fn poison_unblocks_peers() {
        let g = HostGroup::new(2);
        let c0 = g.comm(0);
        let c1 = g.comm(1);
        let h = std::thread::spawn(move || {
            c0.all_reduce(&Tensor::new(&[1f32], &Device::Cpu).unwrap())
                .is_err()
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        c1.poison();
        assert!(h.join().unwrap());
    }
}
