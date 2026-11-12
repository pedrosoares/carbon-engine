// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Micro-benchmark: prefill attention of one layer at the Qwen3.8-27B geometry (24 heads on 4
//! KV heads, head dim 256) over an int8 KV pool: one chunk of queries after a cached prefix.
//! cargo run --release --features cuda --example attnbench [prefix tokens] [chunk tokens] [bf16|q8]
use candle_core::{DType, Device, Tensor};
use carbon_engine::kernels::cuda as k;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let arg = |i: usize, d: usize| std::env::args().nth(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let (prefix, t) = (arg(1, 32768), arg(2, 4096));
    let q8 = std::env::args().nth(3).is_none_or(|v| v == "q8");
    let dev = Device::new_cuda(0)?;
    let (h, kvh, d) = (24usize, 4usize, 256usize);
    let n = prefix + t;
    let q = Tensor::randn(0f32, 1.0, (t, h, d), &dev)?.to_dtype(DType::BF16)?;
    let pool = |seed: f32| -> anyhow::Result<Tensor> {
        if !q8 {
            return Ok((Tensor::randn(0f32, 1.0, (n, kvh * d), &dev)? * seed as f64)?.to_dtype(DType::BF16)?);
        }
        // codes in [0, 255] read as int8, then a positive f32 scale per vector
        let codes = Tensor::rand(0f32, 255.99, (n * kvh, d), &Device::Cpu)?.to_dtype(DType::U8)?;
        let scale = Tensor::from_vec(
            (0..n * kvh).flat_map(|_| (seed / 127.0).to_le_bytes()).collect::<Vec<u8>>(),
            (n * kvh, 4),
            &Device::Cpu,
        )?;
        Ok(Tensor::cat(&[codes, scale], 1)?.reshape((n, kvh * (d + 4)))?.to_device(&dev)?)
    };
    let (kp, vp) = (pool(1.0)?, pool(0.5)?);
    let slots: Vec<u32> = (0..n as u32).collect();
    let seqinfo = [0u32, t as u32, 0, n as u32];
    let tiles: Vec<u32> = (0..t).step_by(k::PREFILL_TILE).flat_map(|q0| [0u32, q0 as u32]).collect();
    let (slots, seqinfo, tiles) =
        (Tensor::new(&slots[..], &dev)?, Tensor::new(&seqinfo[..], &dev)?, Tensor::new(&tiles[..], &dev)?);
    let run = || k::flash_prefill(&q, &kp, &vp, &slots, &seqinfo, &tiles, kvh, 1.0 / (d as f32).sqrt());
    let sync = |y: &Tensor| -> anyhow::Result<f32> { Ok(y.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?) };
    sync(&run()?)?;
    let reps = 3;
    let t0 = Instant::now();
    let mut y = run()?;
    for _ in 1..reps {
        y = run()?;
    }
    let check = sync(&y)?;
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    // useful work: Q K^T and P V over the causal part, 2 flops per multiply-add
    let pairs = (t * prefix + t * (t + 1) / 2) as f64 * h as f64;
    println!(
        "prefill attention, {t} queries after {prefix}: {:.1} ms, {:.0} TFLOPS, checksum {check:.3}",
        dt * 1e3,
        pairs * (4 * d) as f64 / dt / 1e12
    );
    Ok(())
}
