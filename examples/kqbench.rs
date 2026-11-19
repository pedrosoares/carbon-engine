// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Micro-benchmark: GGUF K-quant GEMV on a GPU, in GB/s of packed weights read.
//! cargo run --release --features cuda --example kqbench
use candle_core::{DType, Device, Tensor};
use carbon_engine::gguf::{KKind, KQuant};
use carbon_engine::model::layers::QLinear;
use std::time::Instant;

fn weight(kind: KKind, rows: usize, cols: usize, dev: &Device) -> anyhow::Result<QLinear> {
    let block = kind.block_bytes();
    let row_bytes = cols / 256 * block;
    let mut bytes = Tensor::rand(0f32, 255.99, rows * row_bytes, &Device::Cpu)?.to_dtype(DType::U8)?.to_vec1::<u8>()?;
    for b in bytes.chunks_exact_mut(block) {
        let d = half::f16::from_f32(0.001).to_le_bytes();
        match kind {
            KKind::Q4K => {
                b[0..2].copy_from_slice(&d);
                b[2..4].copy_from_slice(&d);
            }
            KKind::Q6K => b[208..210].copy_from_slice(&d),
        }
    }
    let q = KQuant { kind, data: Tensor::from_vec(bytes, (rows, row_bytes), &Device::Cpu)?, rows, cols };
    Ok(QLinear::KQuant(q).to_device(dev, DType::BF16)?)
}

fn main() -> anyhow::Result<()> {
    use carbon_engine::kernels::cuda as k;
    let dev = Device::new_cuda(0)?;
    println!("tensor-core kernel enabled: {}", k::kquant_tc_enabled());
    for (name, kind, rows, cols) in [
        ("q4_k ffn gate/up 17408x5120", KKind::Q4K, 17408usize, 5120usize),
        ("q6_k ffn down    5120x17408", KKind::Q6K, 5120, 17408),
        ("q4_k attn k      1024x5120", KKind::Q4K, 1024, 5120),
        ("q6_k lm_head   248320x5120", KKind::Q6K, 248320, 5120),
        // far larger than the L2 cache: what the kernels do when the weights come from memory
        ("q4_k 2000000x5120 (5.8 GB)", KKind::Q4K, 2_000_000, 5120),
        ("q6_k 1000000x5120 (4.5 GB)", KKind::Q6K, 1_000_000, 5120),
    ] {
        let QLinear::KQuant(q) = weight(kind, rows, cols, &dev)? else { unreachable!() };
        let gb = q.data.elem_count() as f64 / 1e9;
        for t in [1usize, 2, 4, 8] {
            let x = Tensor::rand(-1f32, 1.0, (t, cols), &dev)?.to_dtype(DType::BF16)?;
            let prep = k::kquant_prep(&x)?;
            let mut line = format!("{name}  T={t}");
            for tc in [false, true] {
                let run = || if tc { k::kquant_gemm_tc(&prep, &q, DType::BF16) } else { k::kquant_gemv_prepared(&prep, &q, DType::BF16) };
                run()?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
                let n = if rows > 100_000 { 20 } else { 200 };
                let t0 = Instant::now();
                let mut y = run()?;
                for _ in 1..n {
                    y = run()?;
                }
                y.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
                let dt = t0.elapsed().as_secs_f64() / n as f64;
                line += &format!("  {} {:7.3} ms {:6.0} GB/s", if tc { "tc  " } else { "dp4a" }, dt * 1e3, gb / dt);
            }
            println!("{line}");
        }
    }
    // the model's pattern: many 50 MB weights one after the other, each its own launch
    let mats: Vec<KQuant> = (0..60)
        .map(|_| match weight(KKind::Q4K, 17408, 5120, &dev) {
            Ok(QLinear::KQuant(q)) => Ok(q),
            Ok(_) => unreachable!(),
            Err(e) => Err(e),
        })
        .collect::<anyhow::Result<_>>()?;
    let gb = mats.iter().map(|q| q.data.elem_count()).sum::<usize>() as f64 / 1e9;
    for t in [1usize, 3] {
        let x = Tensor::rand(-1f32, 1.0, (t, 5120), &dev)?.to_dtype(DType::BF16)?;
        let prep = k::kquant_prep(&x)?;
        for tc in [false, true] {
            let mut y = None;
            let t0 = Instant::now();
            let n = 20;
            for _ in 0..n {
                for q in &mats {
                    y = Some(if tc { k::kquant_gemm_tc(&prep, q, DType::BF16)? } else { k::kquant_gemv_prepared(&prep, q, DType::BF16)? });
                }
            }
            y.unwrap().to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
            let dt = t0.elapsed().as_secs_f64() / n as f64;
            println!("60 x q4_k 17408x5120 in sequence  T={t}  {}  {:7.3} ms  {:6.0} GB/s", if tc { "tc  " } else { "dp4a" }, dt * 1e3, gb / dt);
        }
    }
    Ok(())
}
