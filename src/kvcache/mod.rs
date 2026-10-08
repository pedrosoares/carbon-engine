// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! KV cache: slot allocator, radix prefix cache and the per-device KV tensors.

pub mod allocator;
pub mod radix;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub use allocator::SlotAllocator;
pub use radix::RadixCache;

/// KV tensors for one layer on one device: `[num_slots, kv_heads * head_dim]` each.
pub struct LayerKv {
    pub k: Tensor,
    pub v: Tensor,
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
        let shape = (num_slots, kv_heads * head_dim);
        Ok(Self {
            k: Tensor::zeros(shape, dtype, device)?,
            v: Tensor::zeros(shape, dtype, device)?,
            idx: None,
        })
    }

    /// Also keep a `[num_slots, index_dim]` pool of indexer keys.
    pub fn with_index(mut self, index_dim: usize) -> Result<Self> {
        let n = self.k.dim(0)?;
        self.idx = Some(Tensor::zeros((n, index_dim), self.k.dtype(), self.k.device())?);
        Ok(self)
    }
}
