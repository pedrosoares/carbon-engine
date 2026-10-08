// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! `carbon bench bw`: measure what the offload-vs-hybrid MoE choice rides on, on this machine.
//!
//! * ceilings: a STREAM-style host DRAM read and a plain host -> device copy;
//! * per expert format (bf16, fp8, nvfp4): the real CPU expert kernel (what `hybrid` / `cpu`
//!   compute on the host) and the real expert upload (what `offload` streams on a cache miss),
//!   each alone and both at once (they contend for host DRAM).
//!
//! Per format, `hybrid` is recommended when the CPU kernel moves more than `threshold` x the
//! bytes per second of the upload, else `offload`. The overlapped pair sets hybrid's fetch
//! split: of a step's cache misses, `pcie / (pcie + cpu)` are uploaded and the rest computed on
//! the CPU, so both sides finish together. The profile is written per GPU to
//! `$XDG_CACHE_HOME/carbon/benchbw/<gpu-uuid>.json`; `--moe-strategy auto` and hybrid's fetch
//! split read it.

use crate::model::layers::{Mlp, QLinear};
use crate::weights::Nvfp4;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::Linear;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertFormat {
    Bf16,
    Fp8,
    Nvfp4,
}

impl ExpertFormat {
    pub fn key(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
            Self::Nvfp4 => "nvfp4",
        }
    }

    pub fn parse(s: &str) -> Result<Vec<Self>> {
        Ok(match s.trim() {
            "" | "all" => vec![Self::Bf16, Self::Fp8, Self::Nvfp4],
            list => list
                .split(',')
                .map(|f| match f.trim() {
                    "bf16" => Ok(Self::Bf16),
                    "fp8" | "fp8_block" => Ok(Self::Fp8),
                    "nvfp4" => Ok(Self::Nvfp4),
                    o => bail!("unknown expert format '{o}' (bf16 | fp8 | nvfp4)"),
                })
                .collect::<Result<_>>()?,
        })
    }

    /// The format of experts stored with `bytes_per_weight` (see `moe::expert_quant_bytes`).
    pub fn from_quant_bytes(bytes: Option<f64>) -> Self {
        match bytes {
            Some(b) if b < 0.9 => Self::Nvfp4,
            Some(_) => Self::Fp8,
            None => Self::Bf16,
        }
    }
}

pub struct BenchOptions {
    pub formats: Vec<ExpertFormat>,
    /// Device the experts are uploaded to (`cuda:N`); None = CPU-only measurements.
    pub device: Option<String>,
    pub threshold: f64,
    pub hidden: usize,
    pub intermediate: usize,
    /// Host memory per format's synthetic expert bank (must exceed the last-level cache).
    pub bank_bytes: usize,
    /// Seconds each measurement runs.
    pub seconds: f64,
    pub out: Option<PathBuf>,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            formats: ExpertFormat::parse("all").unwrap(),
            device: None,
            threshold: 2.0,
            hidden: 2048,
            intermediate: 768,
            bank_bytes: 1 << 30,
            seconds: 2.0,
            out: None,
        }
    }
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn random_u8(n: usize, seed: u64, f: impl Fn(u8) -> u8) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n).map(|_| f((xorshift(&mut s) >> 24) as u8)).collect()
}

/// One synthetic expert with real shapes and storage for `fmt`, on the CPU.
pub fn synthetic_expert(fmt: ExpertFormat, hidden: usize, inter: usize, seed: u64) -> Result<Mlp> {
    let proj = |rows: usize, cols: usize, s: u64| -> Result<QLinear> {
        let dev = Device::Cpu;
        Ok(match fmt {
            ExpertFormat::Bf16 => {
                let v: Vec<f32> = random_u8(rows * cols, s, |b| b)
                    .into_iter()
                    .map(|b| (b as f32 - 128.0) / 4096.0)
                    .collect();
                QLinear::Dense(Linear::new(
                    Tensor::from_vec(v, (rows, cols), &dev)?.to_dtype(DType::BF16)?,
                    None,
                ))
            }
            ExpertFormat::Fp8 => {
                // e4m3 codes with small exponents (no NaNs), 128x128 block scales
                let w = Tensor::from_vec(random_u8(rows * cols, s, |b| b & 0x3f), (rows, cols), &dev)?;
                let scale = Tensor::full(1e-3f32, (rows.div_ceil(128), cols.div_ceil(128)), &dev)?;
                QLinear::Fp8 { w, scale }
            }
            ExpertFormat::Nvfp4 => QLinear::Nvfp4(Nvfp4 {
                packed: Tensor::from_vec(random_u8(rows * cols / 2, s, |b| b), (rows, cols / 2), &dev)?,
                scale: Tensor::from_vec(
                    random_u8(rows * cols / 16, s ^ 7, |b| 0x30 | (b & 0x0f)),
                    (rows, cols / 16),
                    &dev,
                )?,
                global: 1e-2,
            }),
        })
    };
    Ok(Mlp::new_q(
        proj(inter, hidden, seed)?,
        proj(inter, hidden, seed ^ 0x55)?,
        proj(hidden, inter, seed ^ 0xaa)?,
        "silu",
    ))
}

fn gbs(bytes: f64, secs: f64) -> f64 {
    bytes / secs.max(1e-9) / 1e9
}

/// STREAM-style read of a 1 GiB buffer with all threads.
pub fn host_read_gbs(seconds: f64) -> f64 {
    use rayon::prelude::*;
    let buf = vec![1.0f32; 1 << 28];
    let t0 = Instant::now();
    let mut bytes = 0f64;
    let mut sink = 0f32;
    while t0.elapsed().as_secs_f64() < seconds {
        sink += buf.par_chunks(1 << 16).map(|c| c.iter().sum::<f32>()).sum::<f32>();
        bytes += (buf.len() * 4) as f64;
    }
    std::hint::black_box(sink);
    gbs(bytes, t0.elapsed().as_secs_f64())
}

/// Run `f(i)` repeatedly for `seconds` (or until `stop`), returning bytes/s of `per_call` bytes.
fn timed(seconds: f64, per_call: usize, stop: Option<&AtomicBool>, mut f: impl FnMut(usize) -> Result<()>) -> Result<f64> {
    f(0)?; // warm-up (kernel compilation, allocator)
    let t0 = Instant::now();
    let mut calls = 0usize;
    while t0.elapsed().as_secs_f64() < seconds && !stop.is_some_and(|s| s.load(Ordering::Relaxed)) {
        f(calls + 1)?;
        calls += 1;
    }
    Ok(gbs((calls * per_call) as f64, t0.elapsed().as_secs_f64()))
}

/// One token through expert `i % n` on the CPU, as `hybrid` / `cpu` run a decode miss.
fn cpu_moe_step(bank: &[Arc<Mlp>], i: usize, x: &Tensor) -> Result<()> {
    std::hint::black_box(bank[i % bank.len()].forward_f32(x)?);
    Ok(())
}

/// Upload expert `i % n` to `dev`, as `offload` serves a cache miss.
fn upload_step(bank: &[Arc<Mlp>], i: usize, dev: &Device) -> Result<()> {
    let m = bank[i % bank.len()].to_device(dev, DType::BF16)?;
    dev.synchronize()?;
    drop(m);
    Ok(())
}

/// GPU name and UUID (`nvidia-smi` index = CUDA ordinal under the default device order).
pub fn gpu_identity(device: &str) -> Option<(String, String)> {
    let idx: u64 = device.strip_prefix("cuda:")?.parse().ok()?;
    let (gpus, _) = crate::server::metrics::gpu_info();
    gpus.iter().find(|g| g["index"] == idx).map(|g| {
        (
            g["name"].as_str().unwrap_or("").to_string(),
            g["uuid"].as_str().unwrap_or("").to_string(),
        )
    })
}

pub fn profile_dir() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("carbon/benchbw")
}

pub fn run(opts: &BenchOptions) -> Result<Value> {
    let device = match &opts.device {
        Some(d) => {
            let dev = crate::device::parse_device(d)?;
            if dev.is_cpu() {
                None
            } else {
                Some(dev)
            }
        }
        None => None,
    };
    let identity = opts.device.as_deref().and_then(gpu_identity);
    eprintln!("host DRAM read ...");
    let host_read = host_read_gbs(opts.seconds);
    let h2d = match &device {
        Some(dev) => {
            eprintln!("host -> device copy ...");
            let t = Tensor::zeros(1 << 26, DType::F32, &Device::Cpu)?;
            Some(timed(opts.seconds, 1 << 28, None, |_| {
                let d = t.to_device(dev)?;
                dev.synchronize()?;
                drop(d);
                Ok(())
            })?)
        }
        None => None,
    };
    let x = Tensor::ones((1, opts.hidden), DType::F32, &Device::Cpu)?;
    let mut dtypes = serde_json::Map::new();
    let mut kernels = serde_json::Map::new();
    for &fmt in &opts.formats {
        let one = synthetic_expert(fmt, opts.hidden, opts.intermediate, 1)?;
        let bytes = one.size_in_bytes();
        let n = (opts.bank_bytes / bytes).clamp(2, 4096);
        eprintln!("{}: {n} experts of {:.1} MiB ...", fmt.key(), bytes as f64 / (1 << 20) as f64);
        let bank: Vec<Arc<Mlp>> = (0..n)
            .map(|i| synthetic_expert(fmt, opts.hidden, opts.intermediate, i as u64 + 1).map(Arc::new))
            .collect::<Result<_>>()?;
        let cpu = timed(opts.seconds, bytes, None, |i| cpu_moe_step(&bank, i, &x))?;
        let mut entry = json!({"expert_bytes": bytes, "cpu_moe_gbs": cpu});
        if let Some(dev) = &device {
            let pcie = timed(opts.seconds, bytes, None, |i| upload_step(&bank, i, dev))?;
            // both at once: they compete for host DRAM bandwidth
            let stop = AtomicBool::new(false);
            let (cpu_ov, pcie_ov) = std::thread::scope(|sc| -> Result<(f64, f64)> {
                let h = sc.spawn(|| timed(opts.seconds, bytes, Some(&stop), |i| cpu_moe_step(&bank, i + n / 2, &x)));
                let p = timed(opts.seconds, bytes, None, |i| upload_step(&bank, i, dev));
                stop.store(true, Ordering::Relaxed);
                Ok((h.join().unwrap()?, p?))
            })?;
            let rec = if cpu > opts.threshold * pcie { "hybrid" } else { "offload" };
            entry["pcie_gather_gbs"] = json!(pcie);
            entry["cpu_moe_overlap_gbs"] = json!(cpu_ov);
            entry["pcie_gather_overlap_gbs"] = json!(pcie_ov);
            entry["hybrid_fetch_fraction"] = json!((pcie_ov / (pcie_ov + cpu_ov)).min(1.0));
            entry["recommended"] = json!(rec);
            dtypes.insert(fmt.key().into(), json!(rec));
        }
        kernels.insert(fmt.key().into(), entry);
    }
    let threads = rayon::current_num_threads();
    Ok(json!({
        "version": 1,
        "created": crate::server::metrics::utc_timestamp(),
        "host": {"threads": threads, "physical_cores": crate::device::physical_cores()},
        "gpu": identity.as_ref().map(|(name, uuid)| json!({"name": name, "uuid": uuid, "device": opts.device})),
        "threshold": opts.threshold,
        "geometry": {"hidden": opts.hidden, "intermediate": opts.intermediate},
        "ceilings": {"host_read_gbs": host_read, "h2d_copy_gbs": h2d},
        "dtypes": dtypes,
        "dtype_kernels": kernels,
    }))
}

/// Where `run`'s profile is written for `device` (one file per GPU).
pub fn default_profile_path(device: Option<&str>) -> PathBuf {
    let id = device
        .and_then(gpu_identity)
        .map(|g| g.1)
        .filter(|u| !u.is_empty())
        .unwrap_or_else(|| "cpu".into());
    profile_dir().join(format!("{id}.json"))
}

pub fn write_profile(path: &Path, profile: &Value) -> Result<()> {
    std::fs::create_dir_all(path.parent().context("profile path has no parent")?)?;
    std::fs::write(path, serde_json::to_string_pretty(profile)? + "\n")?;
    Ok(())
}

/// What a bench profile says about one expert format on one GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct Recommendation {
    /// "hybrid" or "offload".
    pub strategy: String,
    /// Fraction of a step's misses hybrid uploads (the rest run on the CPU).
    pub fetch_fraction: Option<f64>,
}

/// The profile for `device` (`CARBON_BENCHBW_PATH`, else the per-GPU file), ignored when it
/// was measured on a different GPU model.
pub fn recommendation(device: &str, fmt: ExpertFormat) -> Option<Recommendation> {
    let path = std::env::var_os("CARBON_BENCHBW_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| default_profile_path(Some(device)));
    let prof: Value = serde_json::from_str(&std::fs::read_to_string(&path).ok()?).ok()?;
    if let (Some((name, _)), Some(pname)) = (gpu_identity(device), prof["gpu"]["name"].as_str()) {
        if name != pname {
            tracing::warn!(
                "bench profile {} was measured on {pname}, not {name}; ignoring it",
                path.display()
            );
            return None;
        }
    }
    recommendation_from(&prof, fmt)
}

pub fn recommendation_from(prof: &Value, fmt: ExpertFormat) -> Option<Recommendation> {
    let entry = &prof["dtype_kernels"][fmt.key()];
    let strategy = prof["dtypes"][fmt.key()].as_str()?.to_string();
    let fetch_fraction = entry["hybrid_fetch_fraction"].as_f64().or_else(|| {
        let (c, p) = (entry["cpu_moe_gbs"].as_f64()?, entry["pcie_gather_gbs"].as_f64()?);
        Some((p / c).min(1.0))
    });
    Some(Recommendation {
        strategy,
        fetch_fraction,
    })
}

/// Print a profile as a table.
pub fn summary(prof: &Value) -> String {
    let mut s = String::new();
    let c = &prof["ceilings"];
    s += &format!(
        "host DRAM read {:.1} GB/s, host->device copy {}\n",
        c["host_read_gbs"].as_f64().unwrap_or(0.0),
        c["h2d_copy_gbs"].as_f64().map(|v| format!("{v:.1} GB/s")).unwrap_or("n/a".into())
    );
    s += "format   cpu-moe GB/s   upload GB/s   (overlapped cpu / upload)   fetch   recommended\n";
    if let Some(k) = prof["dtype_kernels"].as_object() {
        for (fmt, e) in k {
            let f = |key: &str| e[key].as_f64().map(|v| format!("{v:.2}")).unwrap_or("-".into());
            s += &format!(
                "{fmt:<8} {:>12}   {:>11}   ({:>6} / {:>6})            {:>5}   {}\n",
                f("cpu_moe_gbs"),
                f("pcie_gather_gbs"),
                f("cpu_moe_overlap_gbs"),
                f("pcie_gather_overlap_gbs"),
                f("hybrid_fetch_fraction"),
                e["recommended"]
                    .as_str()
                    .or(prof["dtypes"][fmt].as_str())
                    .unwrap_or("-"),
            );
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_experts_run_on_cpu_kernels() {
        let x = Tensor::ones((1, 256), DType::F32, &Device::Cpu).unwrap();
        for fmt in ExpertFormat::parse("all").unwrap() {
            let m = synthetic_expert(fmt, 256, 128, 3).unwrap();
            let y = m.forward_f32(&x).unwrap();
            assert_eq!(y.dims(), &[1, 256]);
            let v: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
            assert!(v.iter().all(|x| x.is_finite()), "{fmt:?}");
        }
        let n = synthetic_expert(ExpertFormat::Nvfp4, 256, 128, 3).unwrap().size_in_bytes();
        let b = synthetic_expert(ExpertFormat::Bf16, 256, 128, 3).unwrap().size_in_bytes();
        assert!(n * 3 < b);
    }

    #[test]
    fn cpu_only_profile_and_recommendations() {
        let opts = BenchOptions {
            formats: vec![ExpertFormat::Nvfp4],
            hidden: 256,
            intermediate: 128,
            bank_bytes: 1 << 20,
            seconds: 0.05,
            ..Default::default()
        };
        let p = run(&opts).unwrap();
        assert!(p["ceilings"]["host_read_gbs"].as_f64().unwrap() > 0.0);
        assert!(p["dtype_kernels"]["nvfp4"]["cpu_moe_gbs"].as_f64().unwrap() > 0.0);
        // no device: nothing to recommend
        assert_eq!(recommendation_from(&p, ExpertFormat::Nvfp4), None);
        let prof = json!({"dtypes": {"nvfp4": "hybrid"}, "dtype_kernels": {"nvfp4": {
            "cpu_moe_gbs": 40.0, "pcie_gather_gbs": 10.0,
            "cpu_moe_overlap_gbs": 30.0, "pcie_gather_overlap_gbs": 10.0}}});
        let r = recommendation_from(&prof, ExpertFormat::Nvfp4).unwrap();
        assert_eq!(r.strategy, "hybrid");
        // no stored fraction: falls back to the standalone ratio
        assert_eq!(r.fetch_fraction, Some(0.25));
        assert!(summary(&prof).contains("hybrid"));
        assert_eq!(ExpertFormat::from_quant_bytes(Some(0.5625)), ExpertFormat::Nvfp4);
    }
}
