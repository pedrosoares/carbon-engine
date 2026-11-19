// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Micro-benchmark: prefill attention of one layer at the Qwen3.8-27B geometry (24 heads on 4
//! KV heads, head dim 256) over an int8 KV pool: one chunk of queries after a cached prefix.
//! cargo run --release --features cuda --example attnbench [prefix tokens] [chunk tokens] [bf16|q8|tq4]
//! cargo run --release --features cuda --example attnbench decode [context tokens] [rows] [bf16|q8|tq4]
use candle_core::{DType, Device, Tensor};
use carbon_engine::kernels::cuda as k;
use std::time::Instant;

/// One decode step of `rows` token rows (a speculative verify step) over `ctx` cached positions.
fn decode(ctx: usize, rows: usize, fmt: &str) -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let (h, kvh, d) = (24usize, 4usize, 256usize);
    let n = (ctx + rows).max(131_072); // pools this large use the row-shared kernel
    let q = Tensor::randn(0f32, 1.0, (rows, h, d), &dev)?.to_dtype(DType::BF16)?;
    let pool = |seed: f32| -> anyhow::Result<Tensor> {
        if fmt == "bf16" {
            return Ok((Tensor::randn(0f32, 1.0, (n, kvh * d), &dev)? * seed as f64)?.to_dtype(DType::BF16)?);
        }
        // q8: D codes read as int8; tq4: D / 2 bytes of two 4-bit codes; then an f32 scale
        let d = if fmt == "tq4" { d / 2 } else { d };
        let codes = Tensor::rand(0f32, 255.99, (n * kvh, d), &Device::Cpu)?.to_dtype(DType::U8)?;
        let scale = Tensor::from_vec(
            (0..n * kvh).flat_map(|_| (seed / 127.0).to_le_bytes()).collect::<Vec<u8>>(),
            (n * kvh, 4),
            &Device::Cpu,
        )?;
        Ok(Tensor::cat(&[codes, scale], 1)?.reshape((n, kvh * (d + 4)))?.to_device(&dev)?)
    };
    let (kp, vp) = (pool(1.0)?, pool(0.5)?);
    let slots: Vec<u32> = (0..(ctx + rows) as u32).collect();
    // the rows share one table: row j sees its first ctx + j + 1 slots
    let starts: Vec<u32> = if rows == 1 {
        vec![0, (ctx + 1) as u32]
    } else {
        (0..rows).flat_map(|j| [0u32, (ctx + j + 1) as u32]).collect()
    };
    let links: Vec<u32> = (0..rows).map(|j| if j == 0 { ((rows - 1) as u32) << 8 } else { 2 }).collect();
    let (slots, starts, links) =
        (Tensor::new(&slots[..], &dev)?, Tensor::new(&starts[..], &dev)?, Tensor::new(&links[..], &dev)?);
    let run =
        || k::paged_decode_with(&q, &kp, &vp, &slots, &starts, kvh, 1.0 / (d as f32).sqrt(), Some(&links), rows, true);
    let sync = |y: &Tensor| -> anyhow::Result<f32> { Ok(y.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?) };
    sync(&run()?)?;
    let reps = 50;
    let t0 = Instant::now();
    let mut y = run()?;
    for _ in 1..reps {
        y = run()?;
    }
    let check = sync(&y)?;
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    let gb = (ctx * kvh * 2 * match fmt { "bf16" => 2 * d, "tq4" => d / 2 + 4, _ => d + 4 }) as f64 / 1e9;
    println!(
        "decode attention, {rows} rows over {ctx}: {:.3} ms per layer ({:.1} ms for 16), {:.0} GB/s of K/V, checksum {check:.3}",
        dt * 1e3,
        dt * 16e3,
        gb / dt
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let arg = |i: usize, d: usize| std::env::args().nth(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    if std::env::args().nth(1).is_some_and(|v| v == "decode") {
        // attnbench decode [context tokens] [rows] [bf16|q8|tq4]
        return decode(arg(2, 131_072), arg(3, 4), &std::env::args().nth(4).unwrap_or("q8".into()));
    }
    let (prefix, t) = (arg(1, 32768), arg(2, 4096));
    let fmt = std::env::args().nth(3).unwrap_or("q8".into());
    let fmt = fmt.as_str();
    let dev = Device::new_cuda(0)?;
    let (h, kvh, d) = (24usize, 4usize, 256usize);
    let n = prefix + t;
    let q = Tensor::randn(0f32, 1.0, (t, h, d), &dev)?.to_dtype(DType::BF16)?;
    let pool = |seed: f32| -> anyhow::Result<Tensor> {
        if fmt == "bf16" {
            return Ok((Tensor::randn(0f32, 1.0, (n, kvh * d), &dev)? * seed as f64)?.to_dtype(DType::BF16)?);
        }
        // q8: D codes read as int8; tq4: D / 2 bytes of two 4-bit codes; then an f32 scale
        let d = if fmt == "tq4" { d / 2 } else { d };
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
