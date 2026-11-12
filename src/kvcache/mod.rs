// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! KV cache: slot allocator, radix prefix cache and the per-device KV tensors.

pub mod allocator;
pub mod radix;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub use allocator::SlotAllocator;
pub use radix::RadixCache;

static KV_Q8: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Bytes after each int8 KV vector: its f32 scale (`Q8_PAD` in attention_tc.cu).
pub const Q8_PAD: usize = 4;

/// Keep KV pools on CUDA devices as int8 with a scale per vector (`--kv-dtype q8`): half the
/// memory and half the attention traffic of bf16. Set before the pools are allocated.
pub fn set_kv_q8(on: bool) {
    KV_Q8.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn kv_q8() -> bool {
    KV_Q8.load(std::sync::atomic::Ordering::Relaxed)
}

/// Bytes one token's K and V take in one layer's pool on `device`.
pub fn kv_layer_bytes(kv_heads: usize, head_dim: usize, dtype: DType, device: &Device) -> usize {
    if kv_q8() && device.is_cuda() {
        2 * kv_heads * (head_dim + Q8_PAD)
    } else {
        2 * kv_heads * head_dim * dtype.size_in_bytes()
    }
}

/// KV tensors for one layer on one device: `[num_slots, kv_heads * head_dim]` each, or with
/// `q8` U8 `[num_slots, kv_heads * (head_dim + 4)]` (int8 values, then the vector's f32 scale).
pub struct LayerKv {
    pub k: Tensor,
    pub v: Tensor,
    /// The activations' dtype: what attention reads the pools as.
    pub dtype: DType,
    /// The pools hold int8 vectors with scales (CUDA only, read by the tensor-core kernels).
    pub q8: bool,
    /// Per-token indexer keys `[slots, index_dim]` (QSA sparse attention layers only).
    pub idx: Option<Tensor>,
}

impl LayerKv {
    pub fn new(
        num_slots: usize,
        kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let q8 = kv_q8() && device.is_cuda();
        let (shape, pool_dtype) = if q8 {
            ((num_slots, kv_heads * (head_dim + Q8_PAD)), DType::U8)
        } else {
            ((num_slots, kv_heads * head_dim), dtype)
        };
        Ok(Self {
            k: Tensor::zeros(shape, pool_dtype, device)?,
            v: Tensor::zeros(shape, pool_dtype, device)?,
            dtype,
            q8,
            idx: None,
        })
    }

    /// Store the new tokens' keys and values `[T, kv_heads * head_dim]` in the slots of
    /// `scatter_idx` (`[T, kv_heads * head_dim]` u32, every element of a row its slot).
    pub fn write(&self, scatter_idx: &Tensor, k: &Tensor, v: &Tensor, kv_heads: usize) -> Result<()> {
        if self.q8 {
            #[cfg(feature = "cuda")]
            {
                crate::kernels::cuda::kv_q8_write(&k.to_dtype(self.dtype)?, &self.k, scatter_idx, kv_heads)?;
                crate::kernels::cuda::kv_q8_write(&v.to_dtype(self.dtype)?, &self.v, scatter_idx, kv_heads)?;
                return Ok(());
            }
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("int8 KV pools need the cuda build");
        }
        let _ = kv_heads;
        self.k.scatter_set(scatter_idx, &k.to_dtype(self.dtype)?.contiguous()?, 0)?;
        self.v.scatter_set(scatter_idx, &v.to_dtype(self.dtype)?.contiguous()?, 0)?;
        Ok(())
    }

    /// Also keep a `[num_slots, index_dim]` pool of indexer keys.
    pub fn with_index(mut self, index_dim: usize) -> Result<Self> {
        let n = self.k.dim(0)?;
        self.idx = Some(Tensor::zeros((n, index_dim), self.dtype, self.k.device())?);
        Ok(self)
    }
}
