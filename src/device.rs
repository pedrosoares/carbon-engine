// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Device parsing, dtype selection and memory queries.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};

/// Parse a device spec: `cpu`, `cuda`, `cuda:N`, `metal`, `metal:N`.
pub fn parse_device(spec: &str) -> Result<Device> {
    let spec = spec.trim().to_ascii_lowercase();
    let (kind, idx) = match spec.split_once(':') {
        Some((k, i)) => (
            k.to_string(),
            i.parse::<usize>()
                .with_context(|| format!("bad device index in '{spec}'"))?,
        ),
        None => (spec.clone(), 0),
    };
    match kind.as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" | "gpu" => new_cuda(idx, &spec),
        "metal" => new_metal(idx, &spec),
        _ => bail!("unknown device '{spec}' (expected cpu, cuda:N or metal:N)"),
    }
}

#[cfg(feature = "cuda")]
fn new_cuda(idx: usize, _spec: &str) -> Result<Device> {
    Ok(Device::new_cuda(idx)?)
}

#[cfg(not(feature = "cuda"))]
fn new_cuda(_idx: usize, spec: &str) -> Result<Device> {
    bail!("device '{spec}' requested but carbon-engine was built without the `cuda` feature")
}

#[cfg(feature = "metal")]
fn new_metal(idx: usize, _spec: &str) -> Result<Device> {
    Ok(Device::new_metal(idx)?)
}

#[cfg(not(feature = "metal"))]
fn new_metal(_idx: usize, spec: &str) -> Result<Device> {
    bail!("device '{spec}' requested but carbon-engine was built without the `metal` feature")
}

/// Expand a comma separated device list. `cuda:all` expands to every visible GPU.
pub fn parse_device_list(spec: &str) -> Result<Vec<String>> {
    let mut out = vec![];
    for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if part.eq_ignore_ascii_case("cuda:all") || part.eq_ignore_ascii_case("all") {
            let n = cuda_device_count();
            if n == 0 {
                bail!("'{part}' requested but no CUDA devices are visible");
            }
            out.extend((0..n).map(|i| format!("cuda:{i}")));
        } else {
            parse_device(part)?;
            out.push(part.to_ascii_lowercase());
        }
    }
    if out.is_empty() {
        out.push(default_device_name());
    }
    Ok(out)
}

pub fn default_device_name() -> String {
    if cuda_device_count() > 0 {
        "cuda:0".into()
    } else {
        "cpu".into()
    }
}

pub fn cuda_device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        candle_core::cuda_backend::cudarc::driver::result::init()
            .and_then(|_| candle_core::cuda_backend::cudarc::driver::result::device::get_count())
            .map(|n| n as usize)
            .unwrap_or(0)
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

pub fn parse_dtype(s: &str) -> Result<Option<DType>> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "auto" | "" => None,
        "f32" | "float32" | "float" => Some(DType::F32),
        "f16" | "float16" | "half" => Some(DType::F16),
        "bf16" | "bfloat16" => Some(DType::BF16),
        other => bail!("unsupported dtype '{other}'"),
    })
}

/// Default activation/weight dtype for a device: bf16 on CUDA when this GPU and build have the
/// kernels (candle compiles bf16 only for compute capability >= 8.0), else f16, else f32;
/// f16 on Metal; f32 on CPU.
pub fn default_dtype(device: &Device) -> DType {
    if device.is_cuda() {
        static CHOSEN: std::sync::OnceLock<
            parking_lot::Mutex<std::collections::HashMap<String, DType>>,
        > = std::sync::OnceLock::new();
        let key = device_name(device);
        let cache = CHOSEN.get_or_init(Default::default);
        if let Some(d) = cache.lock().get(&key) {
            return *d;
        }
        let mut chosen = DType::F32;
        for d in [DType::BF16, DType::F16, DType::F32] {
            let failures: Vec<_> = kernel_selftest(device, d)
                .into_iter()
                .filter_map(|(op, r)| r.err().map(|e| (op, e)))
                .collect();
            if failures.is_empty() {
                chosen = d;
                break;
            }
            for (op, e) in &failures {
                tracing::warn!("{key} [{}]: kernel check '{op}' failed: {e}", dtype_name(d));
            }
        }
        if chosen != DType::BF16 {
            tracing::warn!(
                "{key}: bf16 kernels are unavailable (GPU older than compute capability 8.0, or carbon was built with a lower CUDA_COMPUTE_CAP); using {}",
                dtype_name(chosen)
            );
        }
        cache.lock().insert(key, chosen);
        chosen
    } else if device.is_metal() {
        DType::F16
    } else {
        DType::F32
    }
}

/// Run the ops a forward pass needs in `dtype` on `device`; false if any kernel is missing.
pub fn dtype_works(device: &Device, dtype: DType) -> bool {
    kernel_selftest(device, dtype)
        .iter()
        .all(|(_, r)| r.is_ok())
}

/// Run every tensor op the engine uses, one at a time, on `device` with activations in `dtype`.
/// Returns each op's name and its error, so a missing GPU kernel can be named precisely.
pub fn kernel_selftest(device: &Device, dtype: DType) -> Vec<(&'static str, Result<(), String>)> {
    use candle_core::D;
    type Op<'a> = Box<dyn Fn() -> candle_core::Result<()> + 'a>;
    let d = device;
    let t = |shape: (usize, usize)| -> candle_core::Result<Tensor> {
        Tensor::arange(0f32, (shape.0 * shape.1) as f32, d)?
            .reshape(shape)?
            .affine(0.01, -0.1)?
            .to_dtype(dtype)
    };
    let f = |shape: (usize, usize)| -> candle_core::Result<Tensor> {
        Tensor::arange(0f32, (shape.0 * shape.1) as f32, d)?
            .reshape(shape)?
            .affine(0.01, -0.1)
    };
    let idx = || Tensor::new(&[0u32, 2], d);
    let ops: Vec<(&'static str, Op)> = vec![
        (
            "cast f32->dtype->f32",
            Box::new(|| t((4, 8))?.to_dtype(DType::F32).map(|_| ())),
        ),
        (
            "zeros/ones",
            Box::new(|| {
                Tensor::zeros((4, 8), dtype, d)?
                    .add(&Tensor::ones((4, 8), dtype, d)?)
                    .map(|_| ())
            }),
        ),
        (
            "matmul (linear)",
            Box::new(|| t((4, 8))?.matmul(&t((6, 8))?.t()?).map(|_| ())),
        ),
        (
            "batched matmul 3d",
            Box::new(|| {
                t((8, 4))?
                    .reshape((2, 4, 4))?
                    .matmul(&t((8, 4))?.reshape((2, 4, 4))?)
                    .map(|_| ())
            }),
        ),
        (
            "batched matmul 4d",
            Box::new(|| {
                t((8, 4))?
                    .reshape((2, 1, 4, 4))?
                    .matmul(&t((8, 4))?.reshape((2, 1, 4, 4))?)
                    .map(|_| ())
            }),
        ),
        (
            "broadcast add/mul",
            Box::new(|| {
                t((4, 8))?
                    .broadcast_add(&t((1, 8))?)?
                    .broadcast_mul(&t((4, 1))?)
                    .map(|_| ())
            }),
        ),
        (
            "index_select u32 (embed/gather)",
            Box::new(|| t((4, 8))?.index_select(&idx()?, 0).map(|_| ())),
        ),
        (
            "index_select u8 on f32 (fp8/nvfp4 lut)",
            Box::new(|| {
                f((16, 1))?
                    .flatten_all()?
                    .index_select(&Tensor::new(&[1u8, 7, 15], d)?, 0)
                    .map(|_| ())
            }),
        ),
        (
            "cast u8->f32",
            Box::new(|| Tensor::new(&[1u8, 2], d)?.to_dtype(DType::F32).map(|_| ())),
        ),
        (
            "scatter_set u32 (kv write)",
            Box::new(|| {
                let pool = Tensor::zeros((4, 8), dtype, d)?;
                pool.scatter_set(
                    &idx()?.reshape((2, 1))?.broadcast_as((2, 8))?.contiguous()?,
                    &t((2, 8))?,
                    0,
                )
            }),
        ),
        (
            "scatter_set f32 (recurrent state)",
            Box::new(|| {
                let pool = Tensor::zeros((4, 8), DType::F32, d)?;
                pool.scatter_set(
                    &idx()?.reshape((2, 1))?.broadcast_as((2, 8))?.contiguous()?,
                    &f((2, 8))?,
                    0,
                )
            }),
        ),
        (
            "index_add u32 (moe combine)",
            Box::new(|| {
                Tensor::zeros((4, 8), dtype, d)?
                    .index_add(&idx()?, &t((2, 8))?, 0)
                    .map(|_| ())
            }),
        ),
        (
            "rms_norm",
            Box::new(|| {
                candle_nn::ops::rms_norm(&t((4, 8))?, &Tensor::ones(8, dtype, d)?, 1e-6).map(|_| ())
            }),
        ),
        (
            "rope_thd",
            Box::new(|| {
                let cs = t((2, 4))?;
                candle_nn::rotary_emb::rope_thd(
                    &t((2, 8))?.reshape((1, 2, 1, 8))?.contiguous()?,
                    &cs,
                    &cs,
                )
                .map(|_| ())
            }),
        ),
        (
            "softmax_last_dim f32",
            Box::new(|| candle_nn::ops::softmax_last_dim(&f((4, 8))?).map(|_| ())),
        ),
        (
            "silu / sigmoid / gelu",
            Box::new(|| {
                let x = t((4, 8))?;
                candle_nn::ops::silu(&x)?;
                candle_nn::ops::sigmoid(&x)?;
                x.gelu().map(|_| ())
            }),
        ),
        (
            "arange u32 + broadcast_gt + cast (masks)",
            Box::new(|| {
                let j = Tensor::arange(0u32, 8, d)?.reshape((1, 8))?;
                let p = Tensor::arange(2u32, 6, d)?.reshape((4, 1))?;
                j.broadcast_gt(&p)?.to_dtype(DType::F32).map(|_| ())
            }),
        ),
        (
            "argmax f32 (greedy sampling)",
            Box::new(|| f((4, 8))?.argmax(1)?.to_vec1::<u32>().map(|_| ())),
        ),
        (
            "cat / narrow / transpose / permute",
            Box::new(|| {
                let x = Tensor::cat(&[t((2, 8))?, t((2, 8))?], 0)?;
                x.narrow(1, 2, 4)?.contiguous()?.t()?.contiguous()?;
                x.reshape((2, 2, 8))?
                    .permute((2, 0, 1))?
                    .contiguous()
                    .map(|_| ())
            }),
        ),
        (
            "stack",
            Box::new(|| Tensor::stack(&[f((2, 2))?, f((2, 2))?], 1).map(|_| ())),
        ),
        (
            "reductions f32",
            Box::new(|| {
                let x = f((4, 8))?;
                x.sum_keepdim(D::Minus1)?;
                x.mean_keepdim(D::Minus1)?;
                x.max_keepdim(D::Minus1).map(|_| ())
            }),
        ),
        (
            "unary f32 (exp/log/sqrt/recip/abs/neg/relu)",
            Box::new(|| {
                let x = (f((4, 8))?.abs()? + 0.5)?;
                x.exp()?.log()?.sqrt()?.recip()?.neg()?.relu().map(|_| ())
            }),
        ),
        (
            "cumsum / eye f32 (linear attention)",
            Box::new(|| {
                f((4, 8))?.cumsum(1)?;
                Tensor::eye(8, DType::F32, d).map(|_| ())
            }),
        ),
        (
            "device -> host copy",
            Box::new(|| f((4, 8))?.to_vec2::<f32>().map(|_| ())),
        ),
    ];
    #[cfg(feature = "cuda")]
    let ops = {
        let mut ops = ops;
        if d.is_cuda() {
            ops.push((
                "carbon nvfp4 kernels (gemv + dequant vs cpu)",
                Box::new(move || check_quant_kernels(d, dtype, true)),
            ));
            ops.push((
                "carbon fp8 kernels (gemv + dequant vs cpu)",
                Box::new(move || check_quant_kernels(d, dtype, false)),
            ));
            for (name, fmt) in [
                ("carbon moe kernels, dense experts (vs cpu)", 0),
                ("carbon moe kernels, fp8 experts (vs cpu)", 1),
                ("carbon moe kernels, nvfp4 experts (vs cpu)", 2),
            ] {
                ops.push((name, Box::new(move || check_moe_kernels(d, dtype, fmt))));
            }
            ops.push((
                "carbon gated deltanet decode (vs generic path)",
                Box::new(move || {
                    let (ey, es) = crate::model::gdn::fused_decode_selftest(d, dtype)
                        .map_err(|e| candle_core::Error::Msg(format!("{e:#}")))?;
                    let tol = if dtype == DType::F32 { 1e-4 } else { 5e-2 };
                    if ey > tol || es > tol {
                        candle_core::bail!(
                            "mismatch: output rel err {ey:.2e}, state rel err {es:.2e}"
                        );
                    }
                    Ok(())
                }),
            ));
            ops.push((
                "carbon prefill attention (vs cpu)",
                Box::new(move || check_flash_prefill(d, dtype)),
            ));
            ops.push((
                "carbon gated deltanet prefill recurrence (vs cpu)",
                Box::new(move || check_gdn_sequence(d)),
            ));
            ops.push((
                "carbon paged decode attention (vs cpu)",
                Box::new(move || check_paged_decode(d, dtype)),
            ));
        }
        ops
    };
    ops.into_iter()
        .map(|(name, op)| (name, op().map_err(|e| e.to_string())))
        .collect()
}

/// Compare carbon's CUDA quantized kernels with the CPU reference on random data.
#[cfg(feature = "cuda")]
fn check_quant_kernels(d: &Device, dtype: DType, nvfp4: bool) -> candle_core::Result<()> {
    use crate::kernels::cuda as k;
    let cpu = Device::Cpu;
    let (n, kk) = (40usize, 256usize);
    let err = |e: anyhow::Error| candle_core::Error::Msg(format!("{e:#}"));
    let x = Tensor::rand(-1f32, 1.0, (3, kk), &cpu)?.to_dtype(dtype)?;
    let (w_ref, y_gpu, w_gpu) = if nvfp4 {
        let packed = Tensor::rand(0f32, 255.0, (n, kk / 2), &cpu)?.to_dtype(DType::U8)?;
        let scale = Tensor::rand(0.3f32, 2.0, (n, kk / 16), &cpu)?.to_dtype(DType::F8E4M3)?;
        let scale = crate::weights::fp8_codes(&scale).map_err(err)?;
        let q = crate::weights::Nvfp4 {
            packed: packed.clone(),
            scale: scale.clone(),
            global: 0.5,
        };
        let w_ref = crate::weights::dequant_nvfp4(&q, DType::F32).map_err(err)?;
        let (p, s) = (packed.to_device(d)?, scale.to_device(d)?);
        (
            w_ref,
            k::nvfp4_gemv(&x.to_device(d)?, &p, &s, 0.5)?,
            k::nvfp4_dequant(&p, &s, 0.5, dtype)?,
        )
    } else {
        let w = Tensor::rand(-2f32, 2.0, (n, kk), &cpu)?.to_dtype(DType::F8E4M3)?;
        let w = crate::weights::fp8_codes(&w).map_err(err)?;
        let s = Tensor::rand(0.5f32, 1.5, (n.div_ceil(16), kk / 128), &cpu)?;
        let w_ref = crate::weights::dequant_fp8(&w, &s, DType::F32).map_err(err)?;
        let (wd, sd) = (w.to_device(d)?, s.to_device(d)?);
        (
            w_ref,
            k::fp8_gemv(&x.to_device(d)?, &wd, &sd)?,
            k::fp8_dequant(&wd, &sd, dtype)?,
        )
    };
    let y_ref = x.to_dtype(DType::F32)?.matmul(&w_ref.t()?)?;
    let rel = |a: &Tensor, b: &Tensor| -> candle_core::Result<f32> {
        let a = a.to_device(&cpu)?.to_dtype(DType::F32)?;
        let diff = (&a - b)?.abs()?.max_all()?.to_scalar::<f32>()?;
        Ok(diff / (b.abs()?.max_all()?.to_scalar::<f32>()? + 1e-6))
    };
    let tol = if dtype == DType::F32 { 1e-4 } else { 2e-2 };
    let (ew, ey) = (rel(&w_gpu, &w_ref)?, rel(&y_gpu, &y_ref)?);
    if ew > tol || ey > tol {
        candle_core::bail!("mismatch vs cpu: dequant rel err {ew:.2e}, gemv rel err {ey:.2e}");
    }
    Ok(())
}

/// Compare carbon's fused MoE kernels (device routing + expert GEMVs) with a CPU reference.
/// Expert 5 is marked remote (expert parallel) and must contribute nothing.
#[cfg(feature = "cuda")]
fn check_moe_kernels(d: &Device, dtype: DType, fmt: u8) -> candle_core::Result<()> {
    use crate::kernels::cuda::{moe_experts, ExpertProj, MoeTable};
    let cpu = Device::Cpu;
    let err = |e: anyhow::Error| candle_core::Error::Msg(format!("{e:#}"));
    let (e_n, k, t, h, i) = (8usize, 3usize, 3usize, 128usize, 64usize);
    // returns (device projection, f32 reference weight on the cpu)
    let proj = |n: usize, kk: usize| -> candle_core::Result<(ExpertProj, Tensor)> {
        Ok(match fmt {
            0 => {
                let w = (Tensor::rand(-1f32, 1.0, (n, kk), &cpu)? * 0.2)?.to_dtype(dtype)?;
                (ExpertProj::Dense(w.to_device(d)?), w.to_dtype(DType::F32)?)
            }
            1 => {
                let w = Tensor::rand(-2f32, 2.0, (n, kk), &cpu)?.to_dtype(DType::F8E4M3)?;
                let w = crate::weights::fp8_codes(&w).map_err(err)?;
                let s = Tensor::rand(0.02f32, 0.06, (n.div_ceil(32), kk.div_ceil(32)), &cpu)?;
                let r = crate::weights::dequant_fp8(&w, &s, DType::F32).map_err(err)?;
                let (w, scale) = (w.to_device(d)?, s.to_device(d)?);
                (ExpertProj::Fp8 { w, scale }, r)
            }
            _ => {
                let packed = Tensor::rand(0f32, 255.0, (n, kk / 2), &cpu)?.to_dtype(DType::U8)?;
                let scale =
                    Tensor::rand(0.05f32, 0.2, (n, kk / 16), &cpu)?.to_dtype(DType::F8E4M3)?;
                let scale = crate::weights::fp8_codes(&scale).map_err(err)?;
                let q = crate::weights::Nvfp4 {
                    packed: packed.clone(),
                    scale: scale.clone(),
                    global: 0.5,
                };
                let r = crate::weights::dequant_nvfp4(&q, DType::F32).map_err(err)?;
                let (packed, scale) = (packed.to_device(d)?, scale.to_device(d)?);
                (
                    ExpertProj::Nvfp4 {
                        packed,
                        scale,
                        global: 0.5,
                    },
                    r,
                )
            }
        })
    };
    let mut projs = Vec::new();
    let mut refs = Vec::new();
    for e in 0..e_n {
        let (g, gr) = proj(i, h)?;
        let (u, ur) = proj(i, h)?;
        let (dn, dr) = proj(h, i)?;
        projs.push((e != 5).then_some([g, u, dn]));
        refs.push([gr, ur, dr]);
    }
    let table = MoeTable::build(&projs, d)?;
    let x = Tensor::randn(0f32, 1.0, (t, h), &cpu)?.to_dtype(dtype)?;
    let logits = Tensor::randn(0f32, 2.0, (t, e_n), &cpu)?;
    let got = moe_experts(&table, &x.to_device(d)?, &logits.to_device(d)?, k, true)?
        .to_device(&cpu)?
        .to_dtype(DType::F32)?;
    let x = x.to_dtype(DType::F32)?;
    let lg = logits.to_vec2::<f32>()?;
    let mut worst = 0f32;
    for (ti, row) in lg.iter().enumerate() {
        let mut order: Vec<usize> = (0..e_n).collect();
        order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        let mx = row[order[0]];
        let denom: f32 = order[..k].iter().map(|&e| (row[e] - mx).exp()).sum();
        let xt = x.get(ti)?.unsqueeze(0)?;
        let mut want = Tensor::zeros((1, h), DType::F32, &cpu)?;
        for &e in &order[..k] {
            if e == 5 {
                continue;
            }
            let [gr, ur, dr] = &refs[e];
            let a = (candle_nn::ops::silu(&xt.matmul(&gr.t()?)?)? * xt.matmul(&ur.t()?)?)?;
            let y = a.matmul(&dr.t()?)?;
            want = (want + (y * ((row[e] - mx).exp() / denom) as f64)?)?;
        }
        let want = want.squeeze(0)?;
        let scale = want.abs()?.max_all()?.to_scalar::<f32>()? + 1e-6;
        let diff = (got.get(ti)? - &want)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let rel = diff / scale;
        worst = if rel.is_nan() {
            f32::INFINITY
        } else {
            worst.max(rel)
        };
    }
    let tol = if dtype == DType::F32 { 1e-4 } else { 3e-2 };
    if worst > tol {
        candle_core::bail!("mismatch vs cpu: max rel err {worst:.2e}");
    }
    Ok(())
}

/// Compare carbon's tiled prefill attention with a CPU causal softmax (cached prefixes, GQA,
/// ragged tiles, two sequences in one launch).
#[cfg(feature = "cuda")]
fn check_flash_prefill(d: &Device, dtype: DType) -> candle_core::Result<()> {
    let cpu = Device::Cpu;
    let (h, kvh, hd, pool) = (8usize, 2usize, 128usize, 200usize);
    let seqs = [(5usize, 37usize), (0, 20)]; // (cached prefix, new queries)
    let t: usize = seqs.iter().map(|s| s.1).sum();
    let q = Tensor::randn(0f32, 1.0, (t, h, hd), &cpu)?.to_dtype(dtype)?;
    let k = Tensor::randn(0f32, 1.0, (pool, kvh * hd), &cpu)?.to_dtype(dtype)?;
    let v = Tensor::randn(0f32, 1.0, (pool, kvh * hd), &cpu)?.to_dtype(dtype)?;
    let (mut slots, mut seqinfo, mut tiles) = (Vec::new(), Vec::new(), Vec::new());
    let mut q_start = 0u32;
    for (i, &(pre, n)) in seqs.iter().enumerate() {
        seqinfo.extend_from_slice(&[q_start, n as u32, slots.len() as u32, (pre + n) as u32]);
        let base = slots.len();
        slots.extend((0..pre + n).map(|j| (pool - 1 - (base + j) * 3 % pool) as u32));
        for q0 in (0..n).step_by(crate::kernels::cuda::PREFILL_TILE) {
            tiles.extend_from_slice(&[i as u32, q0 as u32]);
        }
        q_start += n as u32;
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let got = crate::kernels::cuda::flash_prefill(
        &q.to_device(d)?,
        &k.to_device(d)?,
        &v.to_device(d)?,
        &Tensor::new(&slots[..], d)?,
        &Tensor::new(&seqinfo[..], d)?,
        &Tensor::new(&tiles[..], d)?,
        kvh,
        scale,
    )?
    .to_device(&cpu)?
    .to_dtype(DType::F32)?;
    let (q, k, v) = (
        q.to_dtype(DType::F32)?,
        k.to_dtype(DType::F32)?,
        v.to_dtype(DType::F32)?,
    );
    let mut worst = 0f32;
    for (i, &(pre, n)) in seqs.iter().enumerate() {
        let (qs, ss) = (seqinfo[4 * i] as usize, seqinfo[4 * i + 2] as usize);
        let idx = Tensor::new(&slots[ss..ss + pre + n], &cpu)?;
        let kb = k.index_select(&idx, 0)?.reshape(((), kvh, hd))?;
        let vb = v.index_select(&idx, 0)?.reshape(((), kvh, hd))?;
        for qi in 0..n {
            for hh in 0..h {
                let g = hh / (h / kvh);
                let len = pre + qi + 1;
                let qh = q.get(qs + qi)?.get(hh)?.reshape((1, hd))?;
                let kh = kb.narrow(0, 0, len)?.narrow(1, g, 1)?.squeeze(1)?;
                let vh = vb.narrow(0, 0, len)?.narrow(1, g, 1)?.squeeze(1)?;
                let p = candle_nn::ops::softmax_last_dim(&(qh.matmul(&kh.t()?)? * scale as f64)?)?;
                let want = p.matmul(&vh)?.squeeze(0)?;
                let diff = (got.get(qs + qi)?.get(hh)? - want)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?;
                worst = if diff.is_nan() {
                    f32::INFINITY
                } else {
                    worst.max(diff)
                };
            }
        }
    }
    let tol = if dtype == DType::F32 { 1e-4 } else { 3e-2 };
    if worst > tol {
        candle_core::bail!("mismatch vs cpu: max abs err {worst:.2e}");
    }
    Ok(())
}

/// Compare carbon's whole-sequence gated delta rule with the token-by-token CPU reference.
#[cfg(feature = "cuda")]
fn check_gdn_sequence(d: &Device) -> candle_core::Result<()> {
    let cpu = Device::Cpu;
    let err = |e: anyhow::Error| candle_core::Error::Msg(format!("{e:#}"));
    let (b, h, t, dk, dv) = (1usize, 3usize, 45usize, 128usize, 64usize);
    let l2 = |x: Tensor| -> candle_core::Result<Tensor> {
        x.broadcast_div(&x.sqr()?.sum_keepdim(3)?.sqrt()?)
    };
    let q = (l2(Tensor::randn(0f32, 1.0, (b, h, t, dk), &cpu)?)? * (dk as f64).powf(-0.5))?;
    let k = l2(Tensor::randn(0f32, 1.0, (b, h, t, dk), &cpu)?)?;
    let v = Tensor::randn(0f32, 1.0, (b, h, t, dv), &cpu)?;
    let beta = Tensor::rand(0.1f32, 0.9, (b, h, t), &cpu)?;
    let g = Tensor::rand(-0.5f32, -0.01, (b, h, t), &cpu)?;
    let s0 = Tensor::randn(0f32, 0.1, (b, h, dk, dv), &cpu)?;
    let (o_ref, s_ref) =
        crate::model::gdn::recurrent(&q, &k, &v, &beta, &g, s0.clone()).map_err(err)?;
    let on = |x: &Tensor| x.to_device(d);
    let (o, s1) = crate::kernels::cuda::gdn_sequence(
        &on(&q)?,
        &on(&k)?,
        &on(&v)?,
        &on(&beta)?,
        &on(&g)?,
        &on(&s0)?,
    )?;
    let diff = |a: &Tensor, b: &Tensor| -> candle_core::Result<f32> {
        let e = (a.to_device(&cpu)? - b)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        Ok(if e.is_nan() { f32::INFINITY } else { e })
    };
    let (eo, es) = (diff(&o, &o_ref)?, diff(&s1, &s_ref)?);
    if eo > 1e-4 || es > 1e-4 {
        candle_core::bail!("mismatch vs cpu: output err {eo:.2e}, state err {es:.2e}");
    }
    Ok(())
}

/// Compare carbon's fused paged decode attention with a CPU softmax reference (GQA, ragged).
#[cfg(feature = "cuda")]
fn check_paged_decode(d: &Device, dtype: DType) -> candle_core::Result<()> {
    let cpu = Device::Cpu;
    let (h, kvh, hd, pool) = (8usize, 2usize, 128usize, 300usize);
    let lens = [1usize, 17, 200];
    let q = Tensor::randn(0f32, 1.0, (lens.len(), h, hd), &cpu)?.to_dtype(dtype)?;
    let k = Tensor::randn(0f32, 1.0, (pool, kvh * hd), &cpu)?.to_dtype(dtype)?;
    let v = Tensor::randn(0f32, 1.0, (pool, kvh * hd), &cpu)?.to_dtype(dtype)?;
    // Reversed, interleaved slot tables so the gather order matters.
    let mut slots = Vec::new();
    let mut starts = vec![0u32];
    for (i, &l) in lens.iter().enumerate() {
        slots.extend((0..l).map(|j| (pool - 1 - (j * lens.len() + i) % pool) as u32));
        starts.push(slots.len() as u32);
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let got = crate::kernels::cuda::paged_decode(
        &q.to_device(d)?,
        &k.to_device(d)?,
        &v.to_device(d)?,
        &Tensor::new(&slots[..], d)?,
        &Tensor::new(&starts[..], d)?,
        kvh,
        scale,
    )?
    .to_device(&cpu)?
    .to_dtype(DType::F32)?;
    let (q, k, v) = (
        q.to_dtype(DType::F32)?,
        k.to_dtype(DType::F32)?,
        v.to_dtype(DType::F32)?,
    );
    let mut worst = 0f32;
    for (b, _) in lens.iter().enumerate() {
        let idx = Tensor::new(&slots[starts[b] as usize..starts[b + 1] as usize], &cpu)?;
        let kb = k.index_select(&idx, 0)?.reshape(((), kvh, hd))?;
        let vb = v.index_select(&idx, 0)?.reshape(((), kvh, hd))?;
        for hh in 0..h {
            let g = hh / (h / kvh);
            let qh = q.get(b)?.get(hh)?.reshape((1, hd))?;
            let kh = kb.narrow(1, g, 1)?.squeeze(1)?;
            let vh = vb.narrow(1, g, 1)?.squeeze(1)?;
            let p = candle_nn::ops::softmax_last_dim(&(qh.matmul(&kh.t()?)? * scale as f64)?)?;
            let want = p.matmul(&vh)?.squeeze(0)?;
            let diff = (got.get(b)?.get(hh)? - want)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            worst = worst.max(diff);
        }
    }
    let tol = if dtype == DType::F32 { 1e-4 } else { 3e-2 };
    if worst > tol {
        candle_core::bail!("mismatch vs cpu: max abs err {worst:.2e}");
    }
    Ok(())
}

pub fn dtype_name(d: DType) -> &'static str {
    match d {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::U32 => "u32",
        DType::U8 => "u8",
        DType::I64 => "i64",
        DType::F64 => "f64",
        _ => "other",
    }
}

pub fn device_name(d: &Device) -> String {
    match d.location() {
        candle_core::DeviceLocation::Cpu => "cpu".into(),
        candle_core::DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
        candle_core::DeviceLocation::Metal { gpu_id } => format!("metal:{gpu_id}"),
    }
}

/// Free memory in bytes available to this device, if it can be queried.
pub fn free_memory(device: &Device) -> Option<usize> {
    match device.location() {
        candle_core::DeviceLocation::Cpu => host_available_memory(),
        #[cfg(feature = "cuda")]
        candle_core::DeviceLocation::Cuda { .. } => {
            if let Device::Cuda(dev) = device {
                let _ = dev.cuda_stream().context().bind_to_thread();
                candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
                    .ok()
                    .map(|(free, _)| free)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Total memory of a CUDA device (None elsewhere).
pub fn total_memory(device: &Device) -> Option<usize> {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => {
            let _ = dev.cuda_stream().context().bind_to_thread();
            candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
                .ok()
                .map(|(_, total)| total)
        }
        _ => None,
    }
}

/// A CUDA memory pool of its own for long-lived allocations of a few fixed sizes (the expert
/// cache). Interleaving thousands of ~1 MB experts with short-lived activations in the default
/// stream-ordered pool fragments it: under offload churn its reserved memory grows by tens of MB
/// per layer while the bytes in use stay flat, until allocations fail. In a pool of their own an
/// evicted expert's blocks are reused by the next upload of the same shape.
pub struct DevicePool {
    #[cfg(feature = "cuda")]
    pool: Option<candle_core::cuda_backend::cudarc::driver::sys::CUmemoryPool>,
}

// SAFETY: a CUmemoryPool handle is a process-wide driver object, usable from any thread.
unsafe impl Send for DevicePool {}
unsafe impl Sync for DevicePool {}

impl DevicePool {
    /// A new pool on `device`; on a CPU device, or if the pool cannot be created,
    /// [`DevicePool::upload`] is a plain `to_device`.
    pub fn new(device: &Device) -> Self {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = device {
            use candle_core::cuda_backend::cudarc::driver::{result, sys};
            let ctx = dev.cuda_stream().context().clone();
            let _ = ctx.bind_to_thread();
            let pool = unsafe {
                let mut props: sys::CUmemPoolProps = std::mem::zeroed();
                props.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
                // CUmemLocation is `{ int type; int id; }` in every CUDA version (bindgen wraps
                // `id` in a union from 13.2 on)
                let loc = &mut props.location as *mut _ as *mut [i32; 2];
                (*loc)[0] = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE as i32;
                (*loc)[1] = ctx.ordinal() as i32;
                result::mem_pool::create(&props)
            };
            return match pool {
                Ok(pool) => {
                    // keep freed blocks for the next upload instead of returning them at syncs
                    let mut keep = u64::MAX;
                    let _ = unsafe {
                        result::mem_pool::set_attribute(
                            pool,
                            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                            &mut keep as *mut u64 as *mut _,
                        )
                    };
                    Self { pool: Some(pool) }
                }
                Err(e) => {
                    tracing::warn!("cannot create a CUDA memory pool, using the default one: {e}");
                    Self { pool: None }
                }
            };
        }
        let _ = device;
        Self {
            #[cfg(feature = "cuda")]
            pool: None,
        }
    }

    /// Copy host tensor `t` to `device`, allocating from this pool. The device's current pool
    /// is left alone (it is per device, shared by every thread), and the slice's stream-ordered
    /// free returns the blocks here. Falls back to a plain `to_device` without a pool.
    pub fn upload(&self, t: &Tensor, device: &Device) -> candle_core::Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let (Some(pool), Device::Cuda(dev)) = (self.pool, device) {
            if t.device().is_cpu() && t.elem_count() > 0 {
                use candle_core::cuda_backend::cudarc::driver::result::mem_pool;
                use candle_core::cuda_backend::WrapErr;
                use candle_core::{CpuStorage, CudaStorage, Storage};
                let t = t.contiguous()?;
                let (start, n) = (t.layout().start_offset(), t.elem_count());
                let stream = dev.cuda_stream();
                macro_rules! up {
                    ($v:expr) => {{
                        let host = &$v[start..start + n];
                        let bytes = std::mem::size_of_val(host);
                        // SAFETY: a fresh `bytes`-long allocation of the pool on this stream,
                        // fully overwritten by the copy before any use
                        let mut slice = unsafe {
                            let ptr = mem_pool::alloc_async(pool, bytes, stream.cu_stream()).w()?;
                            stream.upgrade_device_ptr(ptr, n)
                        };
                        stream.memcpy_htod(host, &mut slice).w()?;
                        CudaStorage::wrap_cuda_slice(slice, dev.clone())
                    }};
                }
                let (storage, _) = t.storage_and_layout();
                let cuda = match &*storage {
                    Storage::Cpu(CpuStorage::U8(v)) => up!(v),
                    Storage::Cpu(CpuStorage::U32(v)) => up!(v),
                    Storage::Cpu(CpuStorage::I64(v)) => up!(v),
                    Storage::Cpu(CpuStorage::BF16(v)) => up!(v),
                    Storage::Cpu(CpuStorage::F16(v)) => up!(v),
                    Storage::Cpu(CpuStorage::F32(v)) => up!(v),
                    Storage::Cpu(CpuStorage::F8E4M3(v)) => up!(v),
                    _ => {
                        drop(storage);
                        return t.to_device(device);
                    }
                };
                drop(storage);
                return Ok(Tensor::from_storage(
                    Storage::Cuda(cuda),
                    t.shape().clone(),
                    candle_core::op::BackpropOp::none(),
                    false,
                ));
            }
        }
        t.to_device(device)
    }
}

impl Drop for DevicePool {
    fn drop(&mut self) {
        // outstanding allocations keep the pool alive until they are freed
        #[cfg(feature = "cuda")]
        if let Some(pool) = self.pool.take() {
            let _ = unsafe { candle_core::cuda_backend::cudarc::driver::result::mem_pool::destroy(pool) };
        }
    }
}

/// Device memory kept free for activations and allocator slack (prefill chunks, MoE uploads,
/// attention scratch): 1.5 GiB plus 5% of the card.
pub fn workspace_reserve(device: &Device) -> usize {
    let total = total_memory(device).unwrap_or(0);
    (3usize << 29) + total / 20
}

/// Physical cores (hyper-threads collapsed) from /proc/cpuinfo; logical CPUs elsewhere.
pub fn physical_cores() -> usize {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") else {
        return logical;
    };
    let mut cores = std::collections::HashSet::new();
    let mut phys = "0".to_string();
    for line in text.lines() {
        if let Some((k, v)) = line.split_once(':') {
            match k.trim() {
                "physical id" => phys = v.trim().to_string(),
                "core id" => {
                    cores.insert((phys.clone(), v.trim().to_string()));
                }
                _ => {}
            }
        }
    }
    if cores.is_empty() {
        logical
    } else {
        cores.len().min(logical)
    }
}

pub fn host_available_memory() -> Option<usize> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    text.lines()
        .find(|l| l.starts_with("MemAvailable:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|kb| kb.parse::<usize>().ok())
        .map(|kb| kb * 1024)
}

/// A human readable summary of the visible compute devices.
pub fn describe_devices() -> Vec<String> {
    let mut out = vec![format!(
        "cpu ({} threads, {:.1} GiB available)",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        host_available_memory().unwrap_or(0) as f64 / (1u64 << 30) as f64
    )];
    for i in 0..cuda_device_count() {
        let name = format!("cuda:{i}");
        let free = parse_device(&name).ok().and_then(|d| free_memory(&d));
        out.push(match free {
            Some(f) => format!("{name} ({:.1} GiB free)", f as f64 / (1u64 << 30) as f64),
            None => name,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cpu() {
        assert!(parse_device("cpu").unwrap().is_cpu());
        assert_eq!(parse_device_list("cpu, cpu").unwrap(), vec!["cpu", "cpu"]);
        assert!(parse_device("tpu:0").is_err());
        assert_eq!(parse_dtype("bf16").unwrap(), Some(DType::BF16));
        assert_eq!(parse_dtype("auto").unwrap(), None);
        assert!(dtype_works(&Device::Cpu, DType::F32));
        assert_eq!(default_dtype(&Device::Cpu), DType::F32);
    }
}
