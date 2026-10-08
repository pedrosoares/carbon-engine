// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Micro-benchmark: quantized expert projections on the CPU (decode-sized batches).
//! cargo run --release --example qbench
use candle_core::{DType, Device, Tensor};
use carbon_engine::model::layers::QLinear;
use carbon_engine::weights::Nvfp4;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let (rows, cols) = (768usize, 2048usize);
    let dev = Device::Cpu;
    let packed = Tensor::rand(0f32, 255.0, (rows, cols / 2), &dev)?.to_dtype(DType::U8)?;
    let scale = Tensor::rand(0.5f32, 2.0, (rows, cols / 16), &dev)?.to_dtype(DType::F8E4M3)?;
    let nv = QLinear::Nvfp4(Nvfp4 {
        packed,
        scale,
        global: 0.01,
    });
    let fw = Tensor::rand(-1f32, 1.0, (rows, cols), &dev)?;
    let fp8 = QLinear::Fp8 {
        w: fw.to_dtype(DType::F8E4M3)?,
        scale: Tensor::rand(0.5f32, 2.0, (rows / 128, cols / 128), &dev)?,
    };
    let dense = QLinear::Dense(candle_nn::Linear::new(fw, None));
    for t in [1usize, 4, 64] {
        let x = Tensor::rand(-1f32, 1.0, (t, cols), &dev)?;
        for (name, l) in [("nvfp4", &nv), ("fp8", &fp8), ("f32", &dense)] {
            l.forward(&x)?;
            let n = 50;
            let t0 = Instant::now();
            for _ in 0..n {
                l.forward(&x)?;
            }
            println!(
                "T={t:3} {name:6} {:8.3} ms",
                t0.elapsed().as_secs_f64() * 1e3 / n as f64
            );
        }
    }
    Ok(())
}
