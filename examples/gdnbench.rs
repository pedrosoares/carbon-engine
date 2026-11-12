// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Micro-benchmark: the fused Gated DeltaNet prefill (conv, gates, recurrence, gated norm) of
//! one layer at the Qwen3.8-27B geometry, in ms per chunk.
//! cargo run --release --features cuda --example gdnbench [tokens]
use candle_core::{DType, Device, Tensor};
use carbon_engine::kernels::cuda as k;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let t: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(4096);
    let dev = Device::new_cuda(0)?;
    let (hk, hv, dk, dv, kernel) = (16usize, 48usize, 128usize, 128usize, 4usize);
    let c = 2 * hk * dk + hv * dv;
    let dtype = DType::BF16;
    let r = |shape: &[usize], s: f64| -> anyhow::Result<Tensor> {
        Ok((Tensor::randn(0f32, 1.0, shape, &dev)? * s)?.to_dtype(dtype)?)
    };
    let (taps, norm_w) = (r(&[kernel, c], 0.5)?, r(&[dv], 1.0)?);
    let dt_bias = Tensor::randn(0f32, 1.0, hv, &dev)?;
    let a_neg = Tensor::randn(0f32, 0.5, hv, &dev)?.exp()?.neg()?;
    let conv_pool = Tensor::zeros((2, (kernel - 1) * c), dtype, &dev)?;
    let ssm_pool = Tensor::zeros((2, hv * dk * dv), DType::F32, &dev)?;
    let layer = k::GdnLayer {
        taps: &taps,
        dt_bias: &dt_bias,
        a_neg: &a_neg,
        norm_w: &norm_w,
        conv_pool: &conv_pool,
        ssm_pool: &ssm_pool,
        hk,
        hv,
        dk,
        dv,
        eps: 1e-6,
        gate_sigmoid: true,
        v_tiled: true,
    };
    let (qkv, z, braw, araw) = (r(&[t, c], 1.0)?, r(&[t, hv * dv], 1.0)?, r(&[t, hv], 1.0)?, r(&[t, hv], 1.0)?);
    let slot = Tensor::new(&[1u32], &dev)?;
    let run = |start: bool| k::gdn_prefill(&layer, &qkv, &z, &braw, &araw, &slot, start);
    let sync = |y: &Tensor| -> anyhow::Result<f32> { Ok(y.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?) };
    sync(&run(true)?)?;
    let n = 20;
    let t0 = Instant::now();
    let mut y = run(true)?;
    for _ in 1..n {
        y = run(false)?;
    }
    let check = sync(&y)?;
    let dt = t0.elapsed().as_secs_f64() / n as f64;
    println!("gdn prefill, {t} tokens: {:.3} ms per chunk ({:.0} ns per token), checksum {check:.3}", dt * 1e3, dt * 1e9 / t as f64);
    Ok(())
}
