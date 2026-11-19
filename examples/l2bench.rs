// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Micro-benchmark: how fast the K-quant GEMV reads a weight that was just read (L2 cache)
//! against one that was not, by weight size.
//! cargo run --release --features cuda --example l2bench
use candle_core::{DType, Device, Tensor};
use carbon_engine::gguf::{KKind, KQuant};
use carbon_engine::kernels::cuda as k;
use std::time::Instant;

fn weight(rows: usize, cols: usize, dev: &Device) -> anyhow::Result<KQuant> {
    let block = KKind::Q4K.block_bytes();
    let row_bytes = cols / 256 * block;
    let mut bytes = Tensor::rand(0f32, 255.99, rows * row_bytes, &Device::Cpu)?.to_dtype(DType::U8)?.to_vec1::<u8>()?;
    for b in bytes.chunks_exact_mut(block) {
        let d = half::f16::from_f32(0.001).to_le_bytes();
        b[0..2].copy_from_slice(&d);
        b[2..4].copy_from_slice(&d);
    }
    Ok(KQuant { kind: KKind::Q4K, data: Tensor::from_vec(bytes, (rows, row_bytes), &Device::Cpu)?.to_device(dev)?, rows, cols })
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let cols = 5120;
    let x = Tensor::rand(-1f32, 1.0, (1, cols), &dev)?.to_dtype(DType::BF16)?;
    let prep = k::kquant_prep(&x)?;
    for mb in [4usize, 8, 16, 32, 48, 64, 96, 128, 256] {
        let rows = mb * 1_000_000 / (cols / 256 * 144);
        // enough distinct weights to exceed any cache several times over
        let n = (2000 / mb).clamp(4, 64);
        let ws: Vec<KQuant> = (0..n).map(|_| weight(rows, cols, &dev)).collect::<anyhow::Result<_>>()?;
        let gb = ws[0].data.elem_count() as f64 / 1e9;
        let time = |same: bool| -> anyhow::Result<f64> {
            let reps = 40;
            let mut y = k::kquant_gemv_prepared(&prep, &ws[0], DType::BF16)?;
            y.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
            let t0 = Instant::now();
            for r in 0..reps {
                for i in 0..n {
                    y = k::kquant_gemv_prepared(&prep, &ws[if same { 0 } else { i }], DType::BF16)?;
                }
                let _ = r;
            }
            y.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
            Ok(t0.elapsed().as_secs_f64() / (reps * n) as f64)
        };
        let (cold, warm) = (time(false)?, time(true)?);
        println!(
            "{mb:4} MB x {n:2}: other weights in between {:7.1} us {:5.0} GB/s | same weight again {:7.1} us {:5.0} GB/s",
            cold * 1e6, gb / cold, warm * 1e6, gb / warm
        );
    }
    Ok(())
}
