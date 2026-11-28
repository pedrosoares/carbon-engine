// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Launchers for `inkling.cu` (the Inkling 2 layer stack, `model::inkling`).
//!
//! The kernels take raw device addresses of contiguous tensors and run on candle's stream, so
//! they order with candle ops and are captured by CUDA graphs like them. Every launcher checks
//! the shapes it relies on; the caller owns all outputs (allocated with candle).

use super::cuda::{nvrtc_arch, raw_ptr};
use anyhow::{bail, ensure, Result};
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use candle_core::cuda_backend::{CudaDevice, WrapErr};
use candle_core::{DType, Device, Tensor};
use std::sync::OnceLock;

fn ptx() -> Result<&'static str> {
    static PTX: OnceLock<Result<String, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        let opts = CompileOptions {
            arch: Some(nvrtc_arch(70)),
            ..Default::default()
        };
        compile_ptx_with_opts(include_str!("inkling.cu"), opts)
            .map(|p| p.to_src())
            .map_err(|e| format!("{e:?}"))
    })
    .as_ref()
    .map(|s| s.as_str())
    .map_err(|e| anyhow::anyhow!("NVRTC failed to compile the inkling kernels: {e}"))
}

/// The tensor-core prefill attention (`inkling_tc.cu`), for compute capability 8.0 and newer.
fn tc_ptx() -> Result<&'static str> {
    static PTX: OnceLock<Result<String, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        let opts = CompileOptions {
            arch: Some(nvrtc_arch(80)),
            ..Default::default()
        };
        compile_ptx_with_opts(include_str!("inkling_tc.cu"), opts)
            .map(|p| p.to_src())
            .map_err(|e| format!("{e:?}"))
    })
    .as_ref()
    .map(|s| s.as_str())
    .map_err(|e| anyhow::anyhow!("NVRTC failed to compile the inkling tensor-core kernels: {e}"))
}

/// Whether the tensor-core prefill attention runs here (sm_80+, unless `CARBON_INKLING_FLASH=0`).
pub fn flash_supported(device: &Device) -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    if !*ENV.get_or_init(|| std::env::var("CARBON_INKLING_FLASH").map_or(true, |v| v != "0")) {
        return false;
    }
    let Ok(d) = cuda(device) else { return false };
    use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute as Attr;
    d.cuda_stream()
        .context()
        .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .is_ok_and(|m| m >= 8)
        && tc_ptx().is_ok()
}

/// Compile the kernels now (load time), so a failure shows before serving.
pub fn warm(device: &Device) -> Result<()> {
    let d = cuda(device)?;
    d.get_or_load_custom_func("ink_rms", "carbon_inkling", ptx()?)?;
    Ok(())
}

fn cuda(device: &Device) -> Result<&CudaDevice> {
    match device {
        Device::Cuda(d) => Ok(d),
        _ => bail!("the inkling kernels need a CUDA device"),
    }
}

/// A kernel argument.
#[derive(Clone, Copy)]
pub enum A {
    P(u64),
    I(i32),
    F(f32),
    Ctrl(Ctrl4),
    Mix(Mix7),
}

/// `Mix7` of inkling.cu: 7 state pointers, 4 norm weights, 4 int8 outputs and their scales.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Mix7([u64; 19]);

unsafe impl candle_core::cuda_backend::cudarc::driver::DeviceRepr for Mix7 {}

/// `Ctrl4` of inkling.cu: per controller cur, msg, down, gain, forget, out (4 slots each).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Ctrl4([u64; 24]);

unsafe impl candle_core::cuda_backend::cudarc::driver::DeviceRepr for Ctrl4 {}

/// Address of `t`'s element `off` (contiguous tensors only).
pub fn at(t: &Tensor, off: usize) -> Result<A> {
    Ok(A::P(raw_ptr(t)? + (off * t.dtype().size_in_bytes()) as u64))
}

pub fn p(t: &Tensor) -> Result<A> {
    at(t, 0)
}

pub fn opt(t: Option<&Tensor>) -> Result<A> {
    t.map_or(Ok(A::P(0)), p)
}

fn i(x: usize) -> A {
    A::I(x as i32)
}

fn launch(device: &Device, name: &str, grid: (u32, u32, u32), block: u32, smem: u32, args: &[A]) -> Result<()> {
    launch_in(device, "carbon_inkling", ptx()?, name, grid, block, smem, args)
}

#[allow(clippy::too_many_arguments)]
fn launch_in(device: &Device, module: &str, src: &str, name: &str, grid: (u32, u32, u32), block: u32, smem: u32, args: &[A]) -> Result<()> {
    if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
        return Ok(());
    }
    let d = cuda(device)?;
    let f = d.get_or_load_custom_func(name, module, src)?;
    let mut b = f.builder();
    for a in args {
        match a {
            A::P(x) => b.arg(x),
            A::I(x) => b.arg(x),
            A::F(x) => b.arg(x),
            A::Ctrl(x) => b.arg(x),
            A::Mix(x) => b.arg(x),
        };
    }
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (block, 1, 1),
        shared_mem_bytes: smem,
    };
    unsafe { b.launch(cfg) }.w()?;
    Ok(())
}

fn grid_stride(n: usize) -> (u32, u32, u32) {
    (n.div_ceil(256).clamp(1, 8192) as u32, 1, 1)
}

/// An output buffer the kernel writes completely (uninitialized: no memset node in a graph).
pub fn empty<S: Into<candle_core::Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Tensor> {
    // SAFETY: every caller hands the buffer to a kernel that writes all of it before any read
    Ok(unsafe { Tensor::empty(shape, dtype, device)? })
}

fn f32_out(rows: usize, cols: usize, device: &Device) -> Result<Tensor> {
    empty((rows, cols), DType::F32, device)
}

/// A weight in the Q8 layout: i8 codes `[rows, cols]` (stored U8) and f16 scales `[rows, cols / 32]`.
#[derive(Clone)]
pub struct Q8 {
    pub codes: Tensor,
    pub scales: Tensor,
    pub rows: usize,
    pub cols: usize,
}

impl Q8 {
    pub fn bytes(&self) -> usize {
        self.rows * self.cols + self.rows * self.cols / 16
    }
}

/// Most tokens [`q8_gemv`] takes in one launch.
pub const GEMV_MAX_T: usize = 8;

/// Activations quantized for the dp4a GEMVs: int8 codes `[T, K]` (U8) and f32 scales `[T, K / 32]`.
pub struct QAct {
    pub q: Tensor,
    pub d: Tensor,
}

pub fn quant_x(x: &Tensor) -> Result<QAct> {
    let (t, kk) = x.dims2()?;
    ensure!(kk % 32 == 0 && x.dtype() == DType::F32, "quant: {:?} {:?}", x.shape(), x.dtype());
    let x = x.contiguous()?;
    let dev = x.device();
    let q = empty((t, kk), DType::U8, dev)?;
    let d = f32_out(t, kk / 32, dev)?;
    launch(dev, "ink_quant_x", ((t * kk).div_ceil(256) as u32, 1, 1), 256, 0, &[p(&x)?, i(kk), p(&q)?, p(&d)?, i(t), i(kk)])?;
    Ok(QAct { q, d })
}

/// `y [T, rows]` (f32, contiguous) = `x_sel(row) [T, cols]` . `W[row]`, for T <= 8. Rows below
/// `r1` read `xs[0]`, below `r2` `xs[1]`, the rest `xs[2]` (quantized activations).
pub fn q8_gemv(w: &Q8, xs: [&QAct; 3], r1: usize, r2: usize, y: &Tensor) -> Result<()> {
    q8_gemv_tail(w, xs, r1, r2, y, None)
}

/// Extra f32 rows a [`q8_gemv_tail`] launch computes: weight `[nf, K]`, f32 inputs and their row
/// ranges (as the Q8 rows), output `[T, nf]`.
pub struct F32Tail<'a> {
    pub w: &'a Tensor,
    pub xs: [&'a Tensor; 3],
    pub r1: usize,
    pub r2: usize,
    pub y: &'a Tensor,
}

/// [`q8_gemv`] plus f32 rows in the same launch.
pub fn q8_gemv_tail(w: &Q8, xs: [&QAct; 3], r1: usize, r2: usize, y: &Tensor, tail: Option<F32Tail<'_>>) -> Result<()> {
    let t = xs[0].q.dim(0)?;
    ensure!((1..=GEMV_MAX_T).contains(&t), "q8 gemv: {t} tokens");
    ensure!(w.cols % 32 == 0, "q8 gemv: {} columns", w.cols);
    for x in xs {
        ensure!(x.q.dims2()? == (t, w.cols), "q8 gemv: input {:?}", x.q.shape());
    }
    ensure!(y.dims2()? == (t, w.rows) && y.dtype() == DType::F32, "q8 gemv: output {:?}", y.shape());
    let mut args = vec![p(&w.codes)?, p(&w.scales)?, p(&xs[0].q)?, p(&xs[0].d)?, p(&xs[1].q)?, p(&xs[1].d)?, p(&xs[2].q)?, p(&xs[2].d)?, i(r1), i(r2), p(y)?, i(w.rows), i(w.rows), i(w.cols)];
    let nf = match &tail {
        Some(tl) => {
            let (nf, kk) = tl.w.dims2()?;
            ensure!(kk == w.cols && tl.y.dims2()? == (t, nf), "q8 gemv: f32 tail {:?}", tl.w.shape());
            for x in tl.xs {
                ensure!(x.dims2()? == (t, kk) && x.is_contiguous(), "q8 gemv: f32 tail input {:?}", x.shape());
            }
            args.extend_from_slice(&[p(tl.w)?, i(nf), p(tl.xs[0])?, p(tl.xs[1])?, p(tl.xs[2])?, i(tl.r1), i(tl.r2), p(tl.y)?, i(nf)]);
            nf
        }
        None => {
            args.extend_from_slice(&[A::P(0), i(0), A::P(0), A::P(0), A::P(0), i(0), i(0), A::P(0), i(0)]);
            0
        }
    };
    // the unrolled variants for the model's widths (2304 and 3072)
    if w.cols == 2304 || w.cols == 3072 {
        let nit = w.cols / 256;
        return launch(y.device(), &format!("ink_q8_gemv_u_{t}_{nit}"), ((w.rows + nf).div_ceil(8) as u32, 1, 1), 256, 0, &args);
    }
    launch(y.device(), &format!("ink_q8_gemv_q_{t}"), ((w.rows + nf).div_ceil(16) as u32, 1, 1), 256, 0, &args)
}

/// The whole weight as bf16 `[rows, cols]`.
pub fn q8_dequant(w: &Q8) -> Result<Tensor> {
    let out = empty((w.rows, w.cols), DType::BF16, w.codes.device())?;
    let n = w.rows * w.cols;
    launch(w.codes.device(), "ink_q8_dequant", grid_stride(n), 256, 0, &[p(&w.codes)?, p(&w.scales)?, p(&out)?, A::P(n as u64)])?;
    Ok(out)
}

/// `x [T, cols] f32` . W^T for any T: the GEMV kernel for a few tokens, else bf16 dequant + GEMM.
pub fn q8_matmul(w: &Q8, x: &Tensor) -> Result<Tensor> {
    let t = x.dim(0)?;
    let x = x.contiguous()?;
    if t <= GEMV_MAX_T {
        let y = f32_out(t, w.rows, x.device())?;
        let xq = quant_x(&x)?;
        q8_gemv(w, [&xq, &xq, &xq], w.rows, w.rows, &y)?;
        return Ok(y);
    }
    let wd = q8_dequant(w)?;
    Ok(x.to_dtype(DType::BF16)?.matmul(&wd.t()?)?.to_dtype(DType::F32)?)
}

/// Batch tables the sequence-aware kernels read.
pub struct Tables<'a> {
    /// Per sequence: q_start, q_len, state slot, fresh (u32 `[B * 4]`).
    pub conv_seqs: &'a Tensor,
    /// Per sequence: q_start, q_len, kv_len, slot offset, state slot (u32 `[B * 5]`).
    pub seqs: &'a Tensor,
    /// Sequence of each token row (u32 `[T]`).
    pub rowseq: &'a Tensor,
    /// Every sequence's KV slot table, concatenated (u32).
    pub slots: &'a Tensor,
    /// Position of each token row (u32 `[T]`).
    pub positions: &'a Tensor,
    /// KV slot each new token is written to (u32 `[T]`).
    pub out_loc: &'a Tensor,
    /// Prefill tiles of 64 query rows: (sequence, first row) pairs, u32 `[n * 2]`.
    pub tiles: &'a Tensor,
    pub num_tiles: usize,
    pub num_seqs: usize,
}

/// Causal short conv over `C` channels starting at column `col` of `x [T, ldx]`, into a new
/// `[T, C]` tensor; `state [slots, 3, ctot]` f32 at channel offset `off` advances.
#[allow(clippy::too_many_arguments)]
pub fn conv(
    x: &Tensor,
    col: usize,
    c: usize,
    w: &Tensor,
    dyn_taps: Option<&Tensor>,
    heads: usize,
    state: &Tensor,
    off: usize,
    res: Option<&Tensor>,
    tb: &Tables<'_>,
) -> Result<Tensor> {
    conv_n(x, col, c, w, dyn_taps, heads, state, off, res, 0, tb)
}

/// [`conv`] whose input is the sum of `nsum` parts per row (`x [T * nsum, C]`) when nsum > 0.
#[allow(clippy::too_many_arguments)]
pub fn conv_n(
    x: &Tensor,
    col: usize,
    c: usize,
    w: &Tensor,
    dyn_taps: Option<&Tensor>,
    heads: usize,
    state: &Tensor,
    off: usize,
    res: Option<&Tensor>,
    nsum: usize,
    tb: &Tables<'_>,
) -> Result<Tensor> {
    let (rows, ldx) = x.dims2()?;
    let t = if nsum > 0 { rows / nsum } else { rows };
    let (_, three, ctot) = state.dims3()?;
    ensure!(three == 3 && off + c <= ctot && col + c <= ldx && w.dims2()? == (c, 4), "conv: bad shapes");
    let out = f32_out(t, c, x.device())?;
    let ldd = match dyn_taps {
        Some(d) => {
            ensure!(d.dims2()? == (t, heads * 4) && c % heads == 0, "conv: taps {:?}", d.shape());
            heads * 4
        }
        None => 0,
    };
    let inv = 1.0 / (((if dyn_taps.is_some() { 2 } else { 1 }) * 4 + 1) as f32).sqrt();
    launch(
        x.device(),
        "ink_conv",
        (c.div_ceil(128) as u32, tb.num_seqs as u32, 1),
        128,
        0,
        &[at(x, col)?, i(ldx), p(&out)?, i(c), p(w)?, opt(dyn_taps)?, i(ldd), i(heads.max(1)), p(state)?, i(ctot), i(off), p(tb.conv_seqs)?, i(c), A::F(inv), opt(res)?, i(c), i(nsum)],
    )?;
    if let Some(r) = res {
        ensure!(r.dims2()? == (t, c) && r.is_contiguous(), "conv: residual {:?}", r.shape());
    }
    Ok(out)
}

/// Mode of [`rownorm`].
pub enum RowNorm {
    /// Clip each row to this L2 norm.
    Clip(f32),
    /// RMS-normalize each row (eps).
    Rms(f32),
}

/// Per token and group of `d` values at column `col` of `x [T, ldx]`: times `scale` (`[g * d]`),
/// then clip or RMS-normalize, then the log position factor when `log` is (alpha, floor).
/// Returns a new `[T, g * d]` tensor.
#[allow(clippy::too_many_arguments)]
pub fn rownorm(
    x: &Tensor,
    col: usize,
    g: usize,
    d: usize,
    scale: Option<&Tensor>,
    mode: RowNorm,
    log: Option<(f32, f32)>,
    positions: &Tensor,
) -> Result<Tensor> {
    let (t, ldx) = x.dims2()?;
    ensure!(d <= 128 && col + g * d <= ldx, "rownorm: bad shapes");
    let out = f32_out(t, g * d, x.device())?;
    let (m, param) = match mode {
        RowNorm::Clip(c) => (0, c),
        RowNorm::Rms(e) => (1, e),
    };
    let (alpha, floor) = log.unwrap_or((0.0, 1.0));
    launch(
        x.device(),
        "ink_rownorm",
        ((t * g) as u32, 1, 1),
        128,
        0,
        &[at(x, col)?, i(ldx), p(&out)?, i(g * d), i(g), i(d), opt(scale)?, A::I(m), A::F(param), p(positions)?, A::F(alpha), A::F(floor)],
    )?;
    Ok(out)
}

/// RMS norm of `x [T, H]` (f32, contiguous) with an optional weight.
pub fn rms(x: &Tensor, w: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let (t, h) = x.dims2()?;
    let x = x.contiguous()?;
    ensure!(x.dtype() == DType::F32, "rms: {:?}", x.dtype());
    let out = f32_out(t, h, x.device())?;
    launch(x.device(), "ink_rms", (t as u32, 1, 1), 256, 0, &[p(&x)?, i(h), p(&out)?, i(h), opt(w)?, i(h), A::F(eps)])?;
    Ok(out)
}

/// Where a layer's earlier keys and values live.
pub enum KvSource<'a> {
    /// The paged pool through the slot tables: bf16 rows (`q8` false) or int8 + scale rows.
    Pool { k: &'a Tensor, v: &'a Tensor, q8: bool },
    /// Per-sequence rings of `len` positions (bf16 rows `[states * len, W]`).
    Ring { k: &'a Tensor, v: &'a Tensor, len: usize },
}

pub struct AttnArgs<'a> {
    /// `[T, nh * 128]`, `[T, nh * nrel]`, `[T, nh]`.
    pub q: &'a Tensor,
    pub r: &'a Tensor,
    pub sink: &'a Tensor,
    /// The batch's keys and values `[T, nkv * 128]`.
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub old: KvSource<'a>,
    /// Relative bias table `[extent, nrel]`.
    pub relb: &'a Tensor,
    pub nkv: usize,
    /// 0 = full attention.
    pub window: usize,
    /// Longest key range of any row (picks the split count).
    pub max_keys: usize,
    /// Output gate: `out *= sigmoid(gate)`, the `[T, nh * 128]` columns at `.1` of `.0`.
    pub gate: Option<(&'a Tensor, usize)>,
    /// Also store the output as int8 per 32 (the next GEMV's input).
    pub quant: bool,
}

pub fn attention(a: &AttnArgs<'_>, tb: &Tables<'_>) -> Result<Tensor> {
    Ok(attention_q(a, tb)?.0)
}

/// [`attention`], and its output as int8 when `a.quant`.
pub fn attention_q(a: &AttnArgs<'_>, tb: &Tables<'_>) -> Result<(Tensor, Option<QAct>)> {
    let (t, qw) = a.q.dims2()?;
    let nh = qw / 128;
    let nrel = a.r.dim(1)? / nh.max(1);
    let (extent, nrel2) = a.relb.dims2()?;
    ensure!(qw == nh * 128 && nh % a.nkv == 0 && nrel == nrel2 && nrel <= 16, "attention: bad shapes");
    ensure!(a.k.dims2()? == (t, a.nkv * 128) && a.v.dims2()? == (t, a.nkv * 128), "attention: k/v shapes");
    let lds = a.sink.dim(1)?;
    ensure!(a.sink.dim(0)? == t && lds >= nh && a.sink.is_contiguous(), "attention: sink shape {:?}", a.sink.shape());
    let rep = nh / a.nkv;
    ensure!((1..=4).contains(&rep), "attention: {rep} query heads per KV head");
    let dev = a.q.device();
    let (kold, vold, slots, ring, fmt) = match &a.old {
        KvSource::Pool { k, v, q8 } => (*k, *v, p(tb.slots)?, 0usize, usize::from(*q8)),
        KvSource::Ring { k, v, len } => (*k, *v, A::P(0), *len, 0),
    };
    let blocks = t * a.nkv;
    // split long key ranges (at most 64 shares, sized per row from its own length) when the
    // grid would not fill the GPU
    let (nsplit, per) = if blocks >= 1024 { (1, a.max_keys.max(1)) } else { (a.max_keys.div_ceil(256).clamp(1, 256), 0) };
    let (gate, ldg) = match a.gate {
        Some((g, col)) => (at(g, col)?, g.dim(1)?),
        None => (A::P(0), 0),
    };
    let out = f32_out(t, qw, dev)?;
    let oq = if a.quant { Some(QAct { q: empty((t, qw), DType::U8, dev)?, d: f32_out(t, qw / 32, dev)? }) } else { None };
    let (oqa, oda) = match &oq {
        Some(x) => (p(&x.q)?, p(&x.d)?),
        None => (A::P(0), A::P(0)),
    };
    let (pml, pacc) = if nsplit > 1 {
        (f32_out(t * nh * nsplit, 2, dev)?, f32_out(t * nh * nsplit, 128, dev)?)
    } else {
        (empty(1, DType::F32, dev)?, empty(1, DType::F32, dev)?)
    };
    launch(
        dev,
        &format!("ink_attn_{rep}_{fmt}"),
        (blocks as u32, nsplit as u32, 1),
        128,
        0,
        &[
            p(a.q)?, i(qw), p(a.r)?, i(nh * nrel), p(a.sink)?, i(lds), p(a.k)?, p(a.v)?, i(a.nkv * 128),
            p(kold)?, p(vold)?, slots, i(ring), p(tb.seqs)?, p(tb.rowseq)?, p(a.relb)?, i(extent), i(nrel),
            i(a.nkv), i(a.window), A::F(1.0 / (128f32).sqrt()), p(&out)?, i(qw), p(&pml)?, p(&pacc)?, i(nsplit),
            i(per), gate, i(ldg), oqa, oda,
        ],
    )?;
    if nsplit > 1 {
        launch(dev, "ink_attn_combine", ((t * nh) as u32, 1, 1), 128, 0, &[p(&pml)?, p(&pacc)?, p(a.sink)?, i(lds), p(&out)?, i(qw), i(nh), i(nsplit), gate, i(ldg), oqa, oda])?;
    }
    Ok((out, oq))
}

/// The last `len` positions of each sequence's new rows `x [T, W]` into its ring (bf16).
pub fn ring_write(x: &Tensor, ring: &Tensor, len: usize, tb: &Tables<'_>) -> Result<()> {
    let (t, w) = x.dims2()?;
    ensure!(ring.dim(1)? == w && ring.dtype() == DType::BF16, "ring write: bad shapes");
    launch(x.device(), "ink_ring_write", (t as u32, w.div_ceil(256) as u32, 1), 256, 0, &[p(x)?, i(w), p(ring)?, i(w), i(len), p(tb.seqs)?, p(tb.rowseq)?])
}

/// New rows `x [T, nkv * 128]` into a pool at `out_loc`: bf16 rows, or the `q8` layout.
pub fn pool_write(x: &Tensor, pool: &Tensor, q8: bool, nkv: usize, tb: &Tables<'_>) -> Result<()> {
    let (t, w) = x.dims2()?;
    ensure!(w == nkv * 128, "pool write: bad shapes");
    if q8 {
        ensure!(pool.dim(1)? == nkv * 132, "pool write: q8 pool {:?}", pool.shape());
        launch(x.device(), "ink_pool_write_q8", ((t * nkv) as u32, 1, 1), 128, 0, &[p(x)?, i(w), p(pool)?, p(tb.out_loc)?, i(nkv)])
    } else {
        ensure!(pool.dim(1)? == w && pool.dtype() == DType::BF16, "pool write: pool {:?} {:?}", pool.shape(), pool.dtype());
        launch(x.device(), "ink_pool_write_bf16", (t as u32, w.div_ceil(256) as u32, 1), 256, 0, &[p(x)?, i(w), p(pool)?, p(tb.out_loc)?, i(w)])
    }
}

/// Sigmoid top-k routing: (ids u32 `[T, K]`, weights f32 `[T, K]`).
pub fn route(logits: &Tensor, bias: Option<&Tensor>, k: usize, scale: f32) -> Result<(Tensor, Tensor)> {
    let (t, e) = logits.dims2()?;
    ensure!(e <= 256 && k <= 32 && k <= e, "route: {e} experts, top {k}");
    let logits = logits.contiguous()?;
    let dev = logits.device();
    let ids = empty((t, k), DType::U32, dev)?;
    let wts = f32_out(t, k, dev)?;
    launch(dev, "ink_route", (t as u32, 1, 1), 32, 0, &[p(&logits)?, opt(bias)?, p(&ids)?, p(&wts)?, i(e), i(k), A::F(scale)])?;
    Ok((ids, wts))
}

/// Routed experts of one layer as stored: `fmt` 0 Q8, 1 NVFP4; gate_up `[E, 2I, H]` (gate rows
/// first), down `[E, H, I]`; per-expert f32 scales.
pub struct ExpertWeights {
    pub fmt: u32,
    pub gu_codes: Tensor,
    pub gu_scales: Tensor,
    pub dn_codes: Tensor,
    pub dn_scales: Tensor,
    pub gs: Tensor,
    pub ds: Tensor,
    pub gs_host: Vec<f32>,
    pub ds_host: Vec<f32>,
    pub hidden: usize,
    pub inter: usize,
}

impl ExpertWeights {
    pub fn bytes(&self) -> usize {
        [&self.gu_codes, &self.gu_scales, &self.dn_codes, &self.dn_scales]
            .iter()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum()
    }
}

/// Most routed (token, expert) pairs the GEMV expert kernels take.
pub const MOE_GEMV_MAX_PAIRS: usize = 128;

/// Decode-sized MoE: `y [T, H]` f32 for routed `ids` / `wts` `[T, K]`.
pub fn moe_gemv(ew: &ExpertWeights, x: &Tensor, xq: Option<&QAct>, ids: &Tensor, wts: &Tensor, beta: f32, lb: f32) -> Result<Tensor> {
    moe_gemv_parts(ew, x, xq, ids, wts, beta, lb, false)
}

/// [`moe_gemv`]; with `parts_only` the weighted expert outputs `[T * K, H]` unsummed.
#[allow(clippy::too_many_arguments)]
pub fn moe_gemv_parts(ew: &ExpertWeights, x: &Tensor, xq: Option<&QAct>, ids: &Tensor, wts: &Tensor, beta: f32, lb: f32, parts_only: bool) -> Result<Tensor> {
    let (t, h) = x.dims2()?;
    let k = ids.dim(1)?;
    let (ii, pairs) = (ew.inter, t * k);
    ensure!(h == ew.hidden && pairs <= MOE_GEMV_MAX_PAIRS && h % 32 == 0 && ii % 32 == 0, "moe: bad shapes");
    let dev = x.device();
    let own;
    let xq = match xq {
        Some(q) => q,
        None => {
            own = quant_x(x)?;
            &own
        }
    };
    let aq = empty((pairs, ii), DType::U8, dev)?;
    let ad = f32_out(pairs, ii / 32, dev)?;
    let part = f32_out(pairs, h, dev)?;
    let y = f32_out(t, h, dev)?;
    let f = ew.fmt;
    launch(
        dev,
        &format!("ink_moe_gate_up_{f}"),
        ((ii / 32) as u32, pairs as u32, 1),
        256,
        0,
        &[p(&ew.gu_codes)?, p(&ew.gu_scales)?, p(&ew.gs)?, p(&xq.q)?, p(&xq.d)?, p(ids)?, p(&aq)?, p(&ad)?, i(k), i(ii), i(h), A::F(beta), A::F(lb)],
    )?;
    launch(
        dev,
        &format!("ink_moe_down_{f}"),
        (h.div_ceil(32) as u32, t as u32, k as u32),
        256,
        0,
        &[p(&ew.dn_codes)?, p(&ew.dn_scales)?, p(&ew.ds)?, p(&aq)?, p(&ad)?, p(ids)?, p(wts)?, p(&part)?, i(h), i(k), i(ii), i(h)],
    )?;
    if parts_only {
        return Ok(part);
    }
    launch(dev, "ink_moe_sum", grid_stride(t * h), 256, 0, &[p(&part)?, p(&y)?, i(t), i(k), i(h)])?;
    Ok(y)
}

/// Expert `e`'s gate_up (`down` false, `[2I, H]`) or down (`[H, I]`) weight as bf16, times its
/// per-expert scale, written into `out` (bf16, that shape).
pub fn expert_dequant(ew: &ExpertWeights, e: usize, down: bool, out: &Tensor) -> Result<()> {
    let (rows, cols, codes, scales, g) = if down {
        (ew.hidden, ew.inter, &ew.dn_codes, &ew.dn_scales, ew.ds_host[e])
    } else {
        (2 * ew.inter, ew.hidden, &ew.gu_codes, &ew.gu_scales, ew.gs_host[e])
    };
    ensure!(out.dims2()? == (rows, cols) && out.dtype() == DType::BF16, "expert dequant: output {:?}", out.shape());
    launch(
        out.device(),
        &format!("ink_expert_dequant_{}", ew.fmt),
        grid_stride(rows * cols / 2),
        256,
        0,
        &[p(codes)?, p(scales)?, i(e), A::P(rows as u64), i(cols), A::F(g), p(out)?],
    )
}

/// `gu [n, 2I]` (gate | up) -> SiTU `[n, I]`.
pub fn situ_rows(gu: &Tensor, beta: f32, lb: f32) -> Result<Tensor> {
    let (n, w) = gu.dims2()?;
    let gu = gu.contiguous()?;
    let act = f32_out(n, w / 2, gu.device())?;
    launch(gu.device(), "ink_situ_rows", grid_stride(n * w / 2), 256, 0, &[p(&gu)?, p(&act)?, A::P(n as u64), i(w / 2), A::F(beta), A::F(lb)])?;
    Ok(act)
}

/// Block attention residual: `[nq, T, H]` mixes of the normalized states `pre` (each `[T, H]`)
/// and of `cur` normalized (returned too when `want_curn`).
pub fn mix(pre: &[Tensor], cur: &Tensor, query: &Tensor, want_curn: bool) -> Result<(Tensor, Option<Tensor>)> {
    let (o, c, _) = mix_full(pre, cur, query, want_curn, &[], 0.0, false)?;
    Ok((o, c))
}

/// [`mix`] with output row `q` RMS-normalized and scaled by `norms[q]` (when given).
pub fn mix_norm(
    pre: &[Tensor],
    cur: &Tensor,
    query: &Tensor,
    want_curn: bool,
    norms: &[&Tensor],
    eps: f32,
) -> Result<(Tensor, Option<Tensor>)> {
    let (o, c, _) = mix_full(pre, cur, query, want_curn, norms, eps, false)?;
    Ok((o, c))
}

/// [`mix_norm`], also returning every output row quantized when `quant` (the dp4a GEMV input).
pub fn mix_full(
    pre: &[Tensor],
    cur: &Tensor,
    query: &Tensor,
    want_curn: bool,
    norms: &[&Tensor],
    eps: f32,
    quant: bool,
) -> Result<(Tensor, Option<Tensor>, Vec<QAct>)> {
    let (t, h) = cur.dims2()?;
    let (nq, h2) = query.dims2()?;
    ensure!(h == h2 && pre.len() <= 7 && norms.len() <= nq.min(4), "mix: {nq} queries over {} states", pre.len());
    ensure!(nq <= 4 || (norms.is_empty() && !quant), "mix: more than 4 queries take no norms or int8 outputs");
    ensure!(h % 128 == 0 && h <= 4096, "mix: hidden {h}");
    let addr = |a: A| match a {
        A::P(x) => x,
        _ => 0,
    };
    let mut m = Mix7([0; 19]);
    for (j, s) in pre.iter().enumerate() {
        ensure!(s.dims2()? == (t, h) && s.is_contiguous(), "mix: state {:?}", s.shape());
        m.0[j] = addr(p(s)?);
    }
    for (q, w) in norms.iter().enumerate() {
        m.0[7 + q] = addr(p(w)?);
    }
    let cur = cur.contiguous()?;
    let dev = cur.device();
    let out = empty((nq, t, h), DType::F32, dev)?;
    let curn = if want_curn { Some(f32_out(t, h, dev)?) } else { None };
    let mut qs = vec![];
    if quant {
        for q in 0..nq {
            let qa = QAct { q: empty((t, h), DType::U8, dev)?, d: f32_out(t, h / 32, dev)? };
            m.0[11 + q] = addr(p(&qa.q)?);
            m.0[15 + q] = addr(p(&qa.d)?);
            qs.push(qa);
        }
    }
    launch(
        dev,
        "ink_mix",
        (t as u32, nq.div_ceil(4) as u32, 1),
        1024,
        (h * 4) as u32,
        &[A::Mix(m), i(pre.len()), p(&cur)?, opt(curn.as_ref())?, p(query)?, i(nq), p(&out)?, i(t), i(h), A::F(1e-6), A::F(eps)],
    )?;
    Ok((out, curn, qs))
}

/// One loop controller's inputs: cur, msg `[T, H]`, down `[R, 2H]`, gain, forget `[H, R]`.
pub struct CtrlIn<'a> {
    pub cur: &'a Tensor,
    pub msg: &'a Tensor,
    pub down: &'a Tensor,
    pub gain: &'a Tensor,
    pub forget: &'a Tensor,
}

/// Up to 4 loop controllers in two launches: `(1 - tanh(forget g)) cur + tanh(gain g) msg`.
pub fn control(items: &[CtrlIn<'_>]) -> Result<Vec<Tensor>> {
    ensure!(!items.is_empty() && items.len() <= 4, "control: {} controllers", items.len());
    let (t, h) = items[0].cur.dims2()?;
    let r = items[0].down.dim(0)?;
    ensure!(r <= 64, "control: rank {r}");
    let dev = items[0].cur.device();
    let mut c = Ctrl4::default();
    let mut outs = vec![];
    let mut keep = vec![];
    for (z, it) in items.iter().enumerate() {
        let cur = it.cur.contiguous()?;
        let msg = it.msg.contiguous()?;
        ensure!(cur.dims2()? == (t, h) && msg.dims2()? == (t, h), "control: shapes");
        ensure!(it.down.dims2()? == (r, 2 * h) && it.gain.dims2()? == (h, r) && it.forget.dims2()? == (h, r), "control: weights");
        let out = f32_out(t, h, dev)?;
        let addr = |a: A| match a {
            A::P(x) => x,
            _ => 0,
        };
        c.0[z] = addr(p(&cur)?);
        c.0[4 + z] = addr(p(&msg)?);
        c.0[8 + z] = addr(p(it.down)?);
        c.0[12 + z] = addr(p(it.gain)?);
        c.0[16 + z] = addr(p(it.forget)?);
        c.0[20 + z] = addr(p(&out)?);
        keep.push((cur, msg));
        outs.push(out);
    }
    let nc = items.len() as u32;
    let gate = f32_out(items.len() * t, r, dev)?;
    launch(dev, "ink_ctrl_down", (r as u32, t as u32, nc), 256, 0, &[A::Ctrl(c), p(&gate)?, i(t), i(h), i(r), A::F(1e-6)])?;
    launch(dev, "ink_ctrl_up", (h.div_ceil(8) as u32, t as u32, nc), 256, 0, &[A::Ctrl(c), p(&gate)?, i(t), i(h), i(r)])?;
    drop(keep);
    Ok(outs)
}

/// f32 GEMV for up to 8 tokens: `y [T, rows]` = `x_sel(row)` . `W[row]` (W f32 `[rows, K]`),
/// rows below `r1` from `xs[0]`, below `r2` from `xs[1]`, the rest from `xs[2]`.
pub fn f32_gemv(w: &Tensor, xs: [&Tensor; 3], r1: usize, r2: usize) -> Result<Tensor> {
    let (n, kk) = w.dims2()?;
    let t = xs[0].dim(0)?;
    ensure!((1..=GEMV_MAX_T).contains(&t) && kk % 4 == 0, "f32 gemv: {t} tokens, {kk} columns");
    for x in xs {
        ensure!(x.dims2()? == (t, kk) && x.is_contiguous(), "f32 gemv: input {:?}", x.shape());
    }
    let y = f32_out(t, n, w.device())?;
    launch(w.device(), &format!("ink_f32_gemv_{t}"), (n.div_ceil(8) as u32, 1, 1), 256, 0, &[p(w)?, p(xs[0])?, p(xs[1])?, p(xs[2])?, i(r1), i(r2), i(kk), p(&y)?, i(n), i(n), i(kk)])?;
    Ok(y)
}

/// Where [`kv_prep`] stores the new keys and values.
pub enum KvDst<'a> {
    Pool { k: &'a Tensor, v: &'a Tensor, q8: bool },
    Ring { k: &'a Tensor, v: &'a Tensor, len: usize },
}

/// Decode-sized K / V: conv (taps `kd` / `vd` when given: `[T, ldd]` tensors at column offsets),
/// clip / normalize, store in the cache. Returns (k, v) `[T, nkv * 128]`.
#[allow(clippy::too_many_arguments)]
pub fn kv_prep(
    y: &Tensor,
    kc: usize,
    vc: usize,
    taps: Option<(&Tensor, usize, usize)>,
    wk: &Tensor,
    wv: &Tensor,
    state: &Tensor,
    nkv: usize,
    qk_clip: f32,
    eps: f32,
    dst: KvDst<'_>,
    tb: &Tables<'_>,
) -> Result<(Tensor, Tensor)> {
    let (t, ldy) = y.dims2()?;
    let c = nkv * 128;
    let (_, _, ctot) = state.dims3()?;
    ensure!(kc + c <= ldy && vc + c <= ldy && wk.dims2()? == (c, 4) && wv.dims2()? == (c, 4), "kv prep: bad shapes");
    let dev = y.device();
    let (kn, vn) = (f32_out(t, c, dev)?, f32_out(t, c, dev)?);
    let (kd, vd, ldd, inv_k) = match taps {
        Some((tt, ko, vo)) => (at(tt, ko)?, at(tt, vo)?, tt.dim(1)?, 1.0 / 9f32.sqrt()),
        None => (A::P(0), A::P(0), 0, 1.0 / 5f32.sqrt()),
    };
    let (mode, kp, vp, ring) = match dst {
        KvDst::Pool { k, v, q8 } => (i32::from(q8), p(k)?, p(v)?, 0),
        KvDst::Ring { k, v, len } => (2, p(k)?, p(v)?, len),
    };
    launch(
        dev,
        "ink_kv_prep",
        ((2 * nkv) as u32, tb.num_seqs as u32, 1),
        128,
        0,
        &[
            p(y)?, i(ldy), i(kc), i(vc), kd, vd, i(ldd), p(wk)?, p(wv)?, p(state)?, i(ctot), p(tb.conv_seqs)?, p(tb.seqs)?,
            p(tb.out_loc)?, p(&kn)?, p(&vn)?, i(nkv), A::F(qk_clip), A::F(eps), A::F(inv_k), A::F(inv_k), A::I(mode), kp, vp, i(ring),
        ],
    )?;
    Ok((kn, vn))
}

/// Decode-sized queries and relative projections from the projections y (see ink_qr_prep).
#[allow(clippy::too_many_arguments)]
pub fn qr_prep(
    y: &Tensor,
    qc: usize,
    rc: usize,
    qs: &Tensor,
    rs: &Tensor,
    nh: usize,
    nrel: usize,
    qk_clip: f32,
    rel_clip: f32,
    log: Option<(f32, f32)>,
    positions: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (t, ldy) = y.dims2()?;
    ensure!(qc + nh * 128 <= ldy && rc + nh * nrel <= ldy && nrel <= 32, "qr prep: bad shapes");
    let dev = y.device();
    let (qo, ro) = (f32_out(t, nh * 128, dev)?, f32_out(t, nh * nrel, dev)?);
    let (alpha, floor) = log.unwrap_or((0.0, 1.0));
    launch(
        dev,
        "ink_qr_prep",
        (t as u32, nh as u32, 1),
        128,
        0,
        &[p(y)?, i(ldy), i(qc), i(rc), p(qs)?, p(rs)?, i(nh), i(nrel), A::F(qk_clip), A::F(rel_clip), p(positions)?, A::F(alpha), A::F(floor), p(&qo)?, p(&ro)?],
    )?;
    Ok((qo, ro))
}

/// Tensor-core prefill attention (same arguments and result as [`attention`], no int8 output).
pub fn flash(a: &AttnArgs<'_>, tb: &Tables<'_>) -> Result<Tensor> {
    let (t, qw) = a.q.dims2()?;
    let nh = qw / 128;
    let nrel = a.r.dim(1)? / nh.max(1);
    let (extent, nrel2) = a.relb.dims2()?;
    ensure!(qw == nh * 128 && nh % a.nkv == 0 && nrel == nrel2 && nrel <= 16, "flash: bad shapes");
    ensure!(a.k.dims2()? == (t, a.nkv * 128) && a.v.dims2()? == (t, a.nkv * 128), "flash: k/v shapes");
    let lds = a.sink.dim(1)?;
    let dev = a.q.device();
    let (kold, vold, slots, ring, fmt) = match &a.old {
        KvSource::Pool { k, v, q8 } => (*k, *v, p(tb.slots)?, 0usize, usize::from(*q8)),
        KvSource::Ring { k, v, len } => (*k, *v, A::P(0), *len, 0),
    };
    let (gate, ldg) = match a.gate {
        Some((g, col)) => (at(g, col)?, g.dim(1)?),
        None => (A::P(0), 0),
    };
    let out = f32_out(t, qw, dev)?;
    launch_in(
        dev,
        "carbon_inkling_tc",
        tc_ptx()?,
        &format!("ink_flash_{fmt}"),
        (tb.num_tiles as u32, nh as u32, 1),
        128,
        0,
        &[
            p(a.q)?, i(qw), p(a.r)?, i(nh * nrel), p(a.sink)?, i(lds), p(a.k)?, p(a.v)?, i(a.nkv * 128), p(kold)?, p(vold)?,
            slots, i(ring), p(tb.seqs)?, p(tb.tiles)?, p(a.relb)?, i(extent), i(nrel), i(a.nkv), i(nh / a.nkv), i(a.window),
            gate, i(ldg), p(&out)?, i(qw),
        ],
    )?;
    Ok(out)
}

/// [`conv_n`] with every row in parallel (prefill), then the state advanced.
#[allow(clippy::too_many_arguments)]
pub fn conv_par(
    x: &Tensor,
    col: usize,
    c: usize,
    w: &Tensor,
    dyn_taps: Option<&Tensor>,
    heads: usize,
    state: &Tensor,
    off: usize,
    res: Option<&Tensor>,
    nsum: usize,
    tb: &Tables<'_>,
) -> Result<Tensor> {
    let (rows, ldx) = x.dims2()?;
    let t = if nsum > 0 { rows / nsum } else { rows };
    let (_, three, ctot) = state.dims3()?;
    ensure!(three == 3 && off + c <= ctot && w.dims2()? == (c, 4), "conv: bad shapes");
    ensure!(nsum > 0 || col + c <= ldx, "conv: columns");
    let out = f32_out(t, c, x.device())?;
    let ldd = match dyn_taps {
        Some(d) => {
            ensure!(d.dims2()? == (t, heads * 4) && c % heads == 0, "conv: taps {:?}", d.shape());
            heads * 4
        }
        None => 0,
    };
    if let Some(r) = res {
        ensure!(r.dims2()? == (t, c) && r.is_contiguous(), "conv: residual {:?}", r.shape());
    }
    let inv = 1.0 / (((if dyn_taps.is_some() { 2 } else { 1 }) * 4 + 1) as f32).sqrt();
    launch(
        x.device(),
        "ink_conv_par",
        (c.div_ceil(128) as u32, t as u32, 1),
        128,
        0,
        &[at(x, col)?, i(ldx), p(&out)?, i(c), p(w)?, opt(dyn_taps)?, i(ldd), i(heads.max(1)), p(state)?, i(ctot), i(off), p(tb.conv_seqs)?, p(tb.rowseq)?, i(c), A::F(inv), opt(res)?, i(c), i(nsum)],
    )?;
    launch(
        x.device(),
        "ink_conv_state",
        (c.div_ceil(128) as u32, tb.num_seqs as u32, 1),
        128,
        0,
        &[at(x, col)?, i(ldx), p(state)?, i(ctot), i(off), p(tb.conv_seqs)?, i(c), i(nsum)],
    )?;
    Ok(out)
}

/// `dst[idx[i]] = src[i] * wts[i]` (rows of width H).
pub fn scatter_rows(src: &Tensor, idx: &Tensor, wts: &Tensor, dst: &Tensor) -> Result<()> {
    let (n, h) = src.dims2()?;
    ensure!(dst.dim(1)? == h && idx.dim(0)? == n && wts.dim(0)? == n, "scatter: bad shapes");
    let src = src.contiguous()?;
    launch(src.device(), "ink_scatter_rows", (n as u32, h.div_ceil(256) as u32, 1), 256, 0, &[p(&src)?, p(idx)?, p(wts)?, p(dst)?, i(h)])
}
