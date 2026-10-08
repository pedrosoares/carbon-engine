// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! carbon's own CUDA kernels for quantized weights (`quant.cu`), compiled at runtime with NVRTC
//! to PTX for the oldest virtual architecture the installed NVRTC accepts (compute_70 on
//! CUDA 12, compute_75 on CUDA 13), so they JIT onto every newer GPU regardless of how
//! candle-kernels was built.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{sys, CudaView, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use candle_core::cuda_backend::WrapErr;
use candle_core::{CpuStorage, CudaStorage, CustomOp2, CustomOp3, DType, Layout, Shape, Tensor};
use half::{bf16, f16};
use std::sync::OnceLock;

type CResult<T> = candle_core::Result<T>;

/// Virtual architectures NVRTC may target, oldest first.
const ARCHS: [(u32, &str); 10] = [
    (70, "compute_70"),
    (75, "compute_75"),
    (80, "compute_80"),
    (86, "compute_86"),
    (87, "compute_87"),
    (89, "compute_89"),
    (90, "compute_90"),
    (100, "compute_100"),
    (110, "compute_110"),
    (120, "compute_120"),
];

/// The oldest virtual architecture >= `min` (e.g. 70, 80) that this NVRTC compiles for.
/// Each CUDA release drops old targets (CUDA 13 starts at compute_75), so this is probed at
/// runtime with an empty kernel instead of hard-coded; `CARBON_NVRTC_ARCH=compute_XY` forces
/// a choice.
pub fn nvrtc_arch(min: u32) -> &'static str {
    static CACHE: OnceLock<parking_lot::Mutex<std::collections::HashMap<u32, &'static str>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(a) = cache.lock().get(&min) {
        return a;
    }
    let forced = std::env::var("CARBON_NVRTC_ARCH").ok().and_then(|v| {
        ARCHS
            .iter()
            .find(|(_, a)| *a == v.trim())
            .map(|(_, a)| *a)
    });
    let arch = forced.unwrap_or_else(|| {
        ARCHS
            .iter()
            .filter(|(cc, _)| *cc >= min)
            .find(|(_, a)| {
                compile_ptx_with_opts(
                    "extern \"C\" __global__ void carbon_probe() {}",
                    CompileOptions {
                        arch: Some(a),
                        ..Default::default()
                    },
                )
                .is_ok()
            })
            .map(|(_, a)| *a)
            // nothing compiled: keep the requested one so the real error is reported
            .unwrap_or_else(|| {
                tracing::warn!(
                    "NVRTC compiled none of {}; carbon's CUDA kernels will be unavailable \
                     (set CARBON_NVRTC_ARCH=compute_XY to force a target)",
                    ARCHS.iter().filter(|(cc, _)| *cc >= min).map(|x| x.1).collect::<Vec<_>>().join(", ")
                );
                ARCHS
                    .iter()
                    .find(|(cc, _)| *cc >= min)
                    .map(|x| x.1)
                    .unwrap_or("compute_70")
            })
    });
    if arch != ARCHS.iter().find(|(cc, _)| *cc >= min).map(|x| x.1).unwrap_or("") {
        tracing::info!("NVRTC: compiling carbon kernels for {arch}");
    }
    cache.lock().insert(min, arch);
    arch
}

fn ptx() -> CResult<&'static str> {
    static PTX: OnceLock<Result<String, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        let opts = CompileOptions {
            arch: Some(nvrtc_arch(70)),
            ..Default::default()
        };
        compile_ptx_with_opts(include_str!("quant.cu"), opts)
            .map(|p| p.to_src())
            .map_err(|e| format!("{e:?}"))
    })
    .as_ref()
    .map(|s| s.as_str())
    .map_err(|e| candle_core::Error::Msg(format!("NVRTC failed to compile carbon kernels: {e}")))
}

fn suffix(d: DType) -> CResult<&'static str> {
    Ok(match d {
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        d => candle_core::bail!("carbon kernels: unsupported dtype {d:?}"),
    })
}

fn view<'a, T: candle_core::cuda_backend::CudaDType>(
    s: &'a CudaStorage,
    l: &Layout,
) -> CResult<CudaView<'a, T>> {
    let (a, b) = l
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("carbon kernels need contiguous inputs".into()))?;
    Ok(s.as_cuda_slice::<T>()?.slice(a..b))
}

fn grid_stride(total: usize) -> LaunchConfig {
    let blocks = total.div_ceil(256).clamp(1, 4096) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn warp_per_row(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(4) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Allocate `len` outputs of `dtype`, run `launch` with the output pointer pushed last-but-args.
macro_rules! with_out {
    ($dev:expr, $dtype:expr, $len:expr, |$out:ident| $body:block) => {{
        match $dtype {
            DType::F32 => {
                let $out = unsafe { $dev.alloc::<f32>($len)? };
                $body;
                CudaStorage::wrap_cuda_slice($out, $dev.clone())
            }
            DType::BF16 => {
                let $out = unsafe { $dev.alloc::<bf16>($len)? };
                $body;
                CudaStorage::wrap_cuda_slice($out, $dev.clone())
            }
            DType::F16 => {
                let $out = unsafe { $dev.alloc::<f16>($len)? };
                $body;
                CudaStorage::wrap_cuda_slice($out, $dev.clone())
            }
            d => candle_core::bail!("carbon kernels: unsupported dtype {d:?}"),
        }
    }};
}

/// Load one of carbon's kernels on a device (candle caches the PTX module per device).
macro_rules! kernel {
    ($dev:expr, $name:expr) => {
        $dev.get_or_load_custom_func(&$name, "carbon_quant", ptx()?)?
    };
}

struct Nvfp4Dequant {
    global: f32,
    dtype: DType,
}

impl CustomOp2 for Nvfp4Dequant {
    fn name(&self) -> &'static str {
        "carbon-nvfp4-dequant"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(
        &self,
        p: &CudaStorage,
        pl: &Layout,
        s: &CudaStorage,
        sl: &Layout,
    ) -> CResult<(CudaStorage, Shape)> {
        let (n, half) = pl.shape().dims2()?;
        let k = half * 2;
        let dev = p.device().clone();
        let f = kernel!(dev, format!("nvfp4_dequant_{}", suffix(self.dtype)?));
        let (pv, sv) = (view::<u8>(p, pl)?, view::<u8>(s, sl)?);
        let out = with_out!(dev, self.dtype, n * k, |out| {
            let mut b = f.builder();
            b.arg(&pv);
            b.arg(&sv);
            b.arg(&self.global);
            b.arg(&out);
            let (n32, k32) = (n as i32, k as i32);
            b.arg(&n32);
            b.arg(&k32);
            unsafe { b.launch(grid_stride(n * half)) }.w()?;
        });
        Ok((out, Shape::from((n, k))))
    }
}

struct Nvfp4Gemv {
    global: f32,
}

impl CustomOp3 for Nvfp4Gemv {
    fn name(&self) -> &'static str {
        "carbon-nvfp4-gemv"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(
        &self,
        p: &CudaStorage,
        pl: &Layout,
        s: &CudaStorage,
        sl: &Layout,
        x: &CudaStorage,
        xl: &Layout,
    ) -> CResult<(CudaStorage, Shape)> {
        let (n, half) = pl.shape().dims2()?;
        let (t, k) = xl.shape().dims2()?;
        if half * 2 != k || t > 8 {
            candle_core::bail!("nvfp4 gemv: bad shapes {:?} x {:?}", pl.shape(), xl.shape());
        }
        let dev = p.device().clone();
        let dtype = x.dtype();
        let f = kernel!(dev, format!("nvfp4_gemv_{}", suffix(dtype)?));
        let (pv, sv) = (view::<u8>(p, pl)?, view::<u8>(s, sl)?);
        let (n32, k32, t32) = (n as i32, k as i32, t as i32);
        macro_rules! go {
            ($ty:ty) => {{
                let xv = view::<$ty>(x, xl)?;
                let out = unsafe { dev.alloc::<$ty>(t * n)? };
                let mut b = f.builder();
                b.arg(&pv);
                b.arg(&sv);
                b.arg(&self.global);
                b.arg(&xv);
                b.arg(&out);
                b.arg(&n32);
                b.arg(&k32);
                b.arg(&t32);
                unsafe { b.launch(warp_per_row(n)) }.w()?;
                CudaStorage::wrap_cuda_slice(out, dev.clone())
            }};
        }
        let out = match dtype {
            DType::F32 => go!(f32),
            DType::BF16 => go!(bf16),
            DType::F16 => go!(f16),
            d => candle_core::bail!("nvfp4 gemv: unsupported dtype {d:?}"),
        };
        Ok((out, Shape::from((t, n))))
    }
}

/// FP8 block geometry: scale grid (rows, cols) and block size (b0, b1).
fn fp8_geometry(n: usize, k: usize, scale_shape: &Shape) -> CResult<(i32, i32, i32)> {
    let (sr, sc) = match scale_shape.dims() {
        [] | [1] | [1, 1] => (1, 1),
        [r, c] => (*r, *c),
        d => candle_core::bail!("fp8 scale shape {d:?} unsupported"),
    };
    Ok((sc as i32, n.div_ceil(sr) as i32, k.div_ceil(sc) as i32))
}

struct Fp8Dequant {
    dtype: DType,
}

impl CustomOp2 for Fp8Dequant {
    fn name(&self) -> &'static str {
        "carbon-fp8-dequant"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(
        &self,
        w: &CudaStorage,
        wl: &Layout,
        s: &CudaStorage,
        sl: &Layout,
    ) -> CResult<(CudaStorage, Shape)> {
        let (n, k) = wl.shape().dims2()?;
        let (sc, b0, b1) = fp8_geometry(n, k, sl.shape())?;
        let dev = w.device().clone();
        let f = kernel!(dev, format!("fp8_dequant_{}", suffix(self.dtype)?));
        let (wv, sv) = (view::<u8>(w, wl)?, view::<f32>(s, sl)?);
        let out = with_out!(dev, self.dtype, n * k, |out| {
            let mut b = f.builder();
            b.arg(&wv);
            b.arg(&sv);
            b.arg(&out);
            let (n32, k32) = (n as i32, k as i32);
            b.arg(&n32);
            b.arg(&k32);
            b.arg(&sc);
            b.arg(&b0);
            b.arg(&b1);
            unsafe { b.launch(grid_stride(n * k)) }.w()?;
        });
        Ok((out, Shape::from((n, k))))
    }
}

struct Fp8Gemv;

impl CustomOp3 for Fp8Gemv {
    fn name(&self) -> &'static str {
        "carbon-fp8-gemv"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(
        &self,
        w: &CudaStorage,
        wl: &Layout,
        s: &CudaStorage,
        sl: &Layout,
        x: &CudaStorage,
        xl: &Layout,
    ) -> CResult<(CudaStorage, Shape)> {
        let (n, k) = wl.shape().dims2()?;
        let (t, kx) = xl.shape().dims2()?;
        if kx != k || t > 8 {
            candle_core::bail!("fp8 gemv: bad shapes {:?} x {:?}", wl.shape(), xl.shape());
        }
        let (sc, b0, b1) = fp8_geometry(n, k, sl.shape())?;
        let dev = w.device().clone();
        let dtype = x.dtype();
        let f = kernel!(dev, format!("fp8_gemv_{}", suffix(dtype)?));
        let (wv, sv) = (view::<u8>(w, wl)?, view::<f32>(s, sl)?);
        let (n32, k32, t32) = (n as i32, k as i32, t as i32);
        macro_rules! go {
            ($ty:ty) => {{
                let xv = view::<$ty>(x, xl)?;
                let out = unsafe { dev.alloc::<$ty>(t * n)? };
                let mut b = f.builder();
                b.arg(&wv);
                b.arg(&sv);
                b.arg(&xv);
                b.arg(&out);
                b.arg(&n32);
                b.arg(&k32);
                b.arg(&t32);
                b.arg(&sc);
                b.arg(&b0);
                b.arg(&b1);
                unsafe { b.launch(warp_per_row(n)) }.w()?;
                CudaStorage::wrap_cuda_slice(out, dev.clone())
            }};
        }
        let out = match dtype {
            DType::F32 => go!(f32),
            DType::BF16 => go!(bf16),
            DType::F16 => go!(f16),
            d => candle_core::bail!("fp8 gemv: unsupported dtype {d:?}"),
        };
        Ok((out, Shape::from((t, n))))
    }
}

/// Largest batch the GPU GEMV kernels take; bigger batches dequantize and use cuBLAS.
pub const MAX_GEMV_ROWS: usize = 8;

pub fn nvfp4_dequant(
    packed: &Tensor,
    scale: &Tensor,
    global: f32,
    dtype: DType,
) -> CResult<Tensor> {
    packed
        .contiguous()?
        .apply_op2_no_bwd(&scale.contiguous()?, &Nvfp4Dequant { global, dtype })
}

pub fn nvfp4_gemv(x: &Tensor, packed: &Tensor, scale: &Tensor, global: f32) -> CResult<Tensor> {
    packed.contiguous()?.apply_op3_no_bwd(
        &scale.contiguous()?,
        &x.contiguous()?,
        &Nvfp4Gemv { global },
    )
}

pub fn fp8_dequant(w: &Tensor, scale: &Tensor, dtype: DType) -> CResult<Tensor> {
    w.contiguous()?.apply_op2_no_bwd(
        &scale.to_dtype(DType::F32)?.contiguous()?,
        &Fp8Dequant { dtype },
    )
}

pub fn fp8_gemv(x: &Tensor, w: &Tensor, scale: &Tensor) -> CResult<Tensor> {
    w.contiguous()?.apply_op3_no_bwd(
        &scale.to_dtype(DType::F32)?.contiguous()?,
        &x.contiguous()?,
        &Fp8Gemv,
    )
}

fn attn_ptx() -> CResult<&'static str> {
    static PTX: OnceLock<Result<String, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        let opts = CompileOptions {
            arch: Some(nvrtc_arch(70)),
            ..Default::default()
        };
        compile_ptx_with_opts(include_str!("attention.cu"), opts)
            .map(|p| p.to_src())
            .map_err(|e| format!("{e:?}"))
    })
    .as_ref()
    .map(|s| s.as_str())
    .map_err(|e| candle_core::Error::Msg(format!("NVRTC failed to compile carbon attention: {e}")))
}

/// Paged decode attention: q `[B, H, D]` against the KV pool `[S, KVH*D]` through the
/// concatenated slot tables (`slots`, with `starts[b]..starts[b+1]` per sequence).
struct PagedDecode {
    slots: Tensor,
    starts: Tensor,
    kv_heads: usize,
    scale: f32,
}

/// Contiguous range of a u32 CUDA tensor that lives outside the op's inputs.
fn u32_range(t: &Tensor) -> CResult<(usize, usize)> {
    t.layout()
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("paged decode: non-contiguous index".into()))
}

fn cuda_of(s: &candle_core::Storage) -> CResult<&CudaStorage> {
    match s {
        candle_core::Storage::Cuda(s) => Ok(s),
        _ => candle_core::bail!("paged decode: index tensor is not on CUDA"),
    }
}

impl CustomOp3 for PagedDecode {
    fn name(&self) -> &'static str {
        "carbon-paged-decode"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        v: &CudaStorage,
        vl: &Layout,
    ) -> CResult<(CudaStorage, Shape)> {
        let (b, h, d) = ql.shape().dims3()?;
        let kvh = self.kv_heads;
        if kl.shape() != vl.shape() || kl.shape().dims2()?.1 != kvh * d {
            candle_core::bail!("paged decode: bad kv shape {:?}", kl.shape());
        }
        if !paged_decode_supported(h, kvh, d) {
            candle_core::bail!("paged decode: unsupported head geometry h={h} kvh={kvh} d={d}");
        }
        let dtype = q.dtype();
        if k.dtype() != dtype || v.dtype() != dtype {
            candle_core::bail!("paged decode: q/k/v dtypes differ");
        }
        let dev = q.device().clone();
        let f = dev.get_or_load_custom_func(
            &format!("paged_decode_{}", suffix(dtype)?),
            "carbon_attn",
            attn_ptx()?,
        )?;
        // Read guards on the index tensors (distinct from q/k/v, so no lock conflict).
        let (sg, _) = self.slots.storage_and_layout();
        let (tg, _) = self.starts.storage_and_layout();
        let (sa, sb) = u32_range(&self.slots)?;
        let (ta, tb) = u32_range(&self.starts)?;
        let slots = cuda_of(&sg)?.as_cuda_slice::<u32>()?.slice(sa..sb);
        let starts = cuda_of(&tg)?.as_cuda_slice::<u32>()?.slice(ta..tb);
        if tb - ta != b + 1 {
            candle_core::bail!(
                "paged decode: starts has {} entries for {b} sequences",
                tb - ta
            );
        }
        let (h32, kvh32, d32) = (h as i32, kvh as i32, d as i32);
        let cfg = LaunchConfig {
            grid_dim: ((b * h) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        macro_rules! go {
            ($ty:ty) => {{
                let (qv, kv, vv) = (
                    view::<$ty>(q, ql)?,
                    view::<$ty>(k, kl)?,
                    view::<$ty>(v, vl)?,
                );
                let out = unsafe { dev.alloc::<$ty>(b * h * d)? };
                let mut bld = f.builder();
                bld.arg(&qv);
                bld.arg(&kv);
                bld.arg(&vv);
                bld.arg(&slots);
                bld.arg(&starts);
                bld.arg(&out);
                bld.arg(&h32);
                bld.arg(&kvh32);
                bld.arg(&d32);
                bld.arg(&self.scale);
                unsafe { bld.launch(cfg) }.w()?;
                CudaStorage::wrap_cuda_slice(out, dev.clone())
            }};
        }
        let out = match dtype {
            DType::F32 => go!(f32),
            DType::BF16 => go!(bf16),
            DType::F16 => go!(f16),
            d => candle_core::bail!("paged decode: unsupported dtype {d:?}"),
        };
        Ok((out, Shape::from((b, h, d))))
    }
}

/// Whether [`paged_decode`] handles this head geometry.
pub fn paged_decode_supported(heads: usize, kv_heads: usize, head_dim: usize) -> bool {
    head_dim.is_multiple_of(32) && head_dim <= 256 && kv_heads > 0 && heads.is_multiple_of(kv_heads)
}

/// One decode step of attention for `B` sequences: `q [B, H, D]`, pools `[S, KVH*D]`,
/// `slots`/`starts` u32 on the same device. Returns `[B, H, D]` in q's dtype.
pub fn paged_decode(
    q: &Tensor,
    k_pool: &Tensor,
    v_pool: &Tensor,
    slots: &Tensor,
    starts: &Tensor,
    kv_heads: usize,
    scale: f32,
) -> CResult<Tensor> {
    q.contiguous()?.apply_op3_no_bwd(
        &k_pool.contiguous()?,
        &v_pool.contiguous()?,
        &PagedDecode {
            slots: slots.contiguous()?,
            starts: starts.contiguous()?,
            kv_heads,
            scale,
        },
    )
}

// ---- MoE decode: device routing + grouped expert GEMVs over raw weight pointers (moe.cu) ----

fn moe_ptx() -> CResult<&'static str> {
    static PTX: OnceLock<Result<String, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        let opts = CompileOptions {
            arch: Some(nvrtc_arch(70)),
            ..Default::default()
        };
        let src = format!("{}\n{}", include_str!("quant.cu"), include_str!("moe.cu"));
        compile_ptx_with_opts(src, opts)
            .map(|p| p.to_src())
            .map_err(|e| format!("{e:?}"))
    })
    .as_ref()
    .map(|s| s.as_str())
    .map_err(|e| candle_core::Error::Msg(format!("NVRTC failed to compile carbon moe: {e}")))
}

/// Raw device address of a contiguous CUDA tensor's first element.
fn raw_ptr(t: &Tensor) -> CResult<u64> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    let (s, l) = t.storage_and_layout();
    let candle_core::Storage::Cuda(c) = &*s else {
        candle_core::bail!("moe table: weight is not on CUDA")
    };
    if !l.is_contiguous() {
        candle_core::bail!("moe table: weight is not contiguous");
    }
    let stream = c.device().cuda_stream();
    macro_rules! addr {
        ($ty:ty) => {{
            let (p, _sync) = c.as_cuda_slice::<$ty>()?.device_ptr(&stream);
            p + (l.start_offset() * std::mem::size_of::<$ty>()) as u64
        }};
    }
    Ok(match t.dtype() {
        DType::U8 => addr!(u8),
        DType::F32 => addr!(f32),
        DType::BF16 => addr!(bf16),
        DType::F16 => addr!(f16),
        DType::U32 => addr!(u32),
        DType::I32 => addr!(i32),
        DType::I64 => addr!(i64),
        d => candle_core::bail!("moe table: unsupported weight dtype {d:?}"),
    })
}

/// One expert projection as the MoE kernels read it.
pub enum ExpertProj {
    Dense(Tensor),
    Fp8 {
        w: Tensor,
        scale: Tensor,
    },
    Nvfp4 {
        packed: Tensor,
        scale: Tensor,
        global: f32,
    },
}

impl ExpertProj {
    fn fmt(&self) -> (i32, &'static str) {
        match self {
            Self::Dense(_) => (0, "dense"),
            Self::Fp8 { .. } => (1, "fp8"),
            Self::Nvfp4 { .. } => (2, "nvfp4"),
        }
    }
    /// (rows, cols) of the logical weight.
    fn dims(&self) -> CResult<(usize, usize)> {
        Ok(match self {
            Self::Dense(w) | Self::Fp8 { w, .. } => w.dims2()?,
            Self::Nvfp4 { packed, .. } => {
                let (n, half) = packed.dims2()?;
                (n, half * 2)
            }
        })
    }
}

/// Per-layer device table of every local expert's weight pointers, built once at load.
pub struct MoeTable {
    fname: &'static str,
    dense_dtype: Option<DType>,
    tab: Tensor,
    glob: Tensor,
    geo_gu: [i32; 3],
    geo_dn: [i32; 3],
    hidden: usize,
    inter: usize,
    num_experts: usize,
    /// Tensors the table points into that the caller does not own (f32 copies of FP8 scales).
    _keep: Vec<Tensor>,
}

impl MoeTable {
    /// (hidden, intermediate) of the experts.
    pub fn dims(&self) -> (usize, usize) {
        (self.hidden, self.inter)
    }

    /// `experts[e]` = `[gate, up, down]` for local experts, `None` for experts another rank owns.
    /// Fails (and the caller keeps the generic path) on mixed formats or geometries.
    pub fn build(
        experts: &[Option<[ExpertProj; 3]>],
        device: &candle_core::Device,
    ) -> CResult<Self> {
        let first = experts
            .iter()
            .flatten()
            .next()
            .ok_or_else(|| candle_core::Error::Msg("moe table: no local experts".into()))?;
        let (fmt, fname) = first[0].fmt();
        let (inter, hidden) = first[0].dims()?;
        let dense_dtype = match &first[0] {
            ExpertProj::Dense(w) => Some(w.dtype()),
            _ => None,
        };
        if fmt == 2 && (hidden % 16 != 0 || inter % 16 != 0) {
            candle_core::bail!("moe table: nvfp4 dims must be multiples of 16");
        }
        let mut tab = vec![0i64; experts.len() * 6];
        let mut glob = vec![1f32; experts.len() * 3];
        let mut keep = Vec::new();
        let mut geo: [Option<[i32; 3]>; 2] = [None, None];
        for (e, ex) in experts.iter().enumerate() {
            let Some(ex) = ex else { continue };
            for (j, p) in ex.iter().enumerate() {
                let want = if j == 2 {
                    (hidden, inter)
                } else {
                    (inter, hidden)
                };
                if p.fmt().0 != fmt || p.dims()? != want {
                    candle_core::bail!("moe table: experts differ in format or shape");
                }
                let (w, s) = match p {
                    ExpertProj::Dense(w) => {
                        if Some(w.dtype()) != dense_dtype {
                            candle_core::bail!("moe table: experts differ in dtype");
                        }
                        (raw_ptr(w)?, 0)
                    }
                    ExpertProj::Fp8 { w, scale } => {
                        let (n, k) = w.dims2()?;
                        let (sc, b0, b1) = fp8_geometry(n, k, scale.shape())?;
                        let g = &mut geo[usize::from(j == 2)];
                        if g.is_some_and(|g| g != [sc, b0, b1]) {
                            candle_core::bail!("moe table: experts differ in fp8 block layout");
                        }
                        *g = Some([sc, b0, b1]);
                        let s32 = scale.to_dtype(DType::F32)?.contiguous()?;
                        let ptrs = (raw_ptr(w)?, raw_ptr(&s32)?);
                        keep.push(s32);
                        ptrs
                    }
                    ExpertProj::Nvfp4 {
                        packed,
                        scale,
                        global,
                    } => {
                        glob[e * 3 + j] = *global;
                        (raw_ptr(packed)?, raw_ptr(scale)?)
                    }
                };
                tab[e * 6 + 2 * j] = w as i64;
                tab[e * 6 + 2 * j + 1] = s as i64;
            }
        }
        let n = experts.len();
        Ok(Self {
            fname,
            dense_dtype,
            tab: Tensor::from_vec(tab, n * 6, device)?,
            glob: Tensor::from_vec(glob, n * 3, device)?,
            geo_gu: geo[0].unwrap_or([1, 1, 1]),
            geo_dn: geo[1].unwrap_or([1, 1, 1]),
            hidden,
            inter,
            num_experts: n,
            _keep: keep,
        })
    }
}

impl MoeTable {
    /// A table for `globals.len()` experts held in a slot cache: every entry starts null and
    /// `moe_slot_ensure` points the routed ones at their slots. `like` is any one expert's
    /// projections (format and geometry; it may live on the host), `globals[e]` an expert's
    /// NVFP4 global scales (gate, up, down).
    pub fn for_slots(
        like: &[ExpertProj; 3],
        globals: &[[f32; 3]],
        device: &candle_core::Device,
    ) -> CResult<Self> {
        let (_, fname) = like[0].fmt();
        let (inter, hidden) = like[0].dims()?;
        let dense_dtype = match &like[0] {
            ExpertProj::Dense(w) => Some(w.dtype()),
            _ => None,
        };
        let mut geo = [[1, 1, 1]; 2];
        for (j, p) in like.iter().enumerate() {
            let want = if j == 2 { (hidden, inter) } else { (inter, hidden) };
            if p.fmt().1 != fname || p.dims()? != want {
                candle_core::bail!("moe table: projections differ in format or shape");
            }
            if let ExpertProj::Fp8 { w, scale } = p {
                let (n, k) = w.dims2()?;
                let (sc, b0, b1) = fp8_geometry(n, k, scale.shape())?;
                let g = &mut geo[usize::from(j == 2)];
                if j == 1 && *g != [sc, b0, b1] {
                    candle_core::bail!("moe table: gate and up differ in fp8 block layout");
                }
                *g = [sc, b0, b1];
            }
        }
        if fname == "nvfp4" && (hidden % 16 != 0 || inter % 16 != 0) {
            candle_core::bail!("moe table: nvfp4 dims must be multiples of 16");
        }
        let n = globals.len();
        let glob: Vec<f32> = globals.iter().flatten().copied().collect();
        Ok(Self {
            fname,
            dense_dtype,
            tab: Tensor::zeros(n * 6, DType::I64, device)?,
            glob: Tensor::from_vec(glob, n * 3, device)?,
            geo_gu: geo[0],
            geo_dn: geo[1],
            hidden,
            inter,
            num_experts: n,
            _keep: vec![],
        })
    }
}

/// Softmax top-k routing: `logits [T, E]` f32 -> u32 `[2 * T * K]`, the expert ids then the
/// routing weights (as f32 bits), the layout every MoE kernel reads.
struct MoeRoute {
    top_k: usize,
    norm: bool,
}

impl candle_core::CustomOp1 for MoeRoute {
    fn name(&self) -> &'static str {
        "carbon-moe-route"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(&self, lg: &CudaStorage, ll: &Layout) -> CResult<(CudaStorage, Shape)> {
        let (t, e) = ll.shape().dims2()?;
        let k = self.top_k;
        if e > 1024 || k == 0 || k > e || lg.dtype() != DType::F32 {
            candle_core::bail!("moe route: bad logits {:?} for top-{k}", ll.shape());
        }
        let p = t * k;
        let dev = lg.device().clone();
        let f = dev.get_or_load_custom_func("moe_topk", "carbon_moe", moe_ptx()?)?;
        let logits = view::<f32>(lg, ll)?;
        let route = unsafe { dev.alloc::<u32>(2 * p)? };
        {
            let ids = route.slice(0..p);
            let wts = route.slice(p..2 * p);
            let (e32, k32, norm) = (e as i32, k as i32, i32::from(self.norm));
            let mut b = f.builder();
            b.arg(&logits);
            b.arg(&ids);
            b.arg(&wts);
            b.arg(&e32);
            b.arg(&k32);
            b.arg(&norm);
            let cfg = LaunchConfig {
                grid_dim: (t.max(1) as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            if t > 0 {
                unsafe { b.launch(cfg) }.w()?;
            }
        }
        Ok((CudaStorage::wrap_cuda_slice(route, dev), Shape::from(2 * p)))
    }
}

/// Routed experts for a precomputed route (`MoeRoute`'s layout), reading weights through
/// `table`.
struct MoeCompute<'a> {
    table: &'a MoeTable,
    top_k: usize,
}

impl CustomOp2 for MoeCompute<'_> {
    fn name(&self) -> &'static str {
        "carbon-moe-compute"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        rt: &CudaStorage,
        rl: &Layout,
    ) -> CResult<(CudaStorage, Shape)> {
        let tb = self.table;
        let (t, h) = xl.shape().dims2()?;
        let (k, i) = (self.top_k, tb.inter);
        let p = t * k;
        if h != tb.hidden || rl.shape().elem_count() != 2 * p || k == 0 {
            candle_core::bail!(
                "moe experts: bad shapes x {:?} route {:?}",
                xl.shape(),
                rl.shape()
            );
        }
        if p > 65535 {
            candle_core::bail!("moe experts: too many routed pairs ({p})");
        }
        let dtype = x.dtype();
        if tb.dense_dtype.is_some_and(|d| d != dtype) {
            candle_core::bail!("moe experts: dense expert dtype differs from activations");
        }
        let dev = x.device().clone();
        let sfx = suffix(dtype)?;
        // NVFP4 with the activations staged in shared memory (48 KiB without opt-in)
        let fast = tb.fname == "nvfp4" && h * 4 <= 48 * 1024 && k * i * 4 <= 48 * 1024;
        let fname = if fast { "nvfp4v" } else { tb.fname };
        let f_gu = dev.get_or_load_custom_func(
            &format!("moe_gate_up_{fname}_{sfx}"),
            "carbon_moe",
            moe_ptx()?,
        )?;
        let f_dn = dev.get_or_load_custom_func(
            &format!("moe_down_{fname}_{sfx}"),
            "carbon_moe",
            moe_ptx()?,
        )?;
        let tab = Side::new(&tb.tab, "moe table")?;
        let glob = Side::new(&tb.glob, "moe globals")?;
        let (tab, glob) = (tab.view::<i64>()?, glob.view::<f32>()?);
        let route = view::<u32>(rt, rl)?;
        let ids = route.slice(0..p);
        let wts = route.slice(p..2 * p);
        let act = unsafe { dev.alloc::<f32>(p * i)? };
        let (k32, i32_, h32) = (k as i32, i as i32, h as i32);
        let [gsc, gb0, gb1] = tb.geo_gu;
        let [dsc, db0, db1] = tb.geo_dn;
        macro_rules! go {
            ($ty:ty) => {{
                let xv = view::<$ty>(x, xl)?;
                let out = unsafe { dev.alloc::<$ty>(t * h)? };
                if t > 0 && fast {
                    let mut b = f_gu.builder();
                    b.arg(&tab);
                    b.arg(&glob);
                    b.arg(&xv);
                    b.arg(&ids);
                    b.arg(&act);
                    b.arg(&k32);
                    b.arg(&i32_);
                    b.arg(&h32);
                    let cfg = LaunchConfig {
                        grid_dim: (i.div_ceil(8) as u32, p as u32, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: (h * 4) as u32,
                    };
                    unsafe { b.launch(cfg) }.w()?;
                    let mut b = f_dn.builder();
                    b.arg(&tab);
                    b.arg(&glob);
                    b.arg(&act);
                    b.arg(&ids);
                    b.arg(&wts);
                    b.arg(&out);
                    b.arg(&k32);
                    b.arg(&i32_);
                    b.arg(&h32);
                    let cfg = LaunchConfig {
                        grid_dim: (h.div_ceil(8) as u32, t as u32, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: (k * i * 4) as u32,
                    };
                    unsafe { b.launch(cfg) }.w()?;
                } else if t > 0 {
                    let mut b = f_gu.builder();
                    b.arg(&tab);
                    b.arg(&glob);
                    b.arg(&xv);
                    b.arg(&ids);
                    b.arg(&act);
                    b.arg(&k32);
                    b.arg(&i32_);
                    b.arg(&h32);
                    b.arg(&gsc);
                    b.arg(&gb0);
                    b.arg(&gb1);
                    let cfg = LaunchConfig {
                        grid_dim: (i.div_ceil(4) as u32, p as u32, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe { b.launch(cfg) }.w()?;
                    let mut b = f_dn.builder();
                    b.arg(&tab);
                    b.arg(&glob);
                    b.arg(&act);
                    b.arg(&ids);
                    b.arg(&wts);
                    b.arg(&out);
                    b.arg(&k32);
                    b.arg(&i32_);
                    b.arg(&h32);
                    b.arg(&dsc);
                    b.arg(&db0);
                    b.arg(&db1);
                    let cfg = LaunchConfig {
                        grid_dim: (h.div_ceil(4) as u32, t as u32, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe { b.launch(cfg) }.w()?;
                }
                CudaStorage::wrap_cuda_slice(out, dev.clone())
            }};
        }
        let out = match dtype {
            DType::F32 => go!(f32),
            DType::BF16 => go!(bf16),
            DType::F16 => go!(f16),
            d => candle_core::bail!("moe experts: unsupported dtype {d:?}"),
        };
        Ok((out, Shape::from((t, h))))
    }
}

/// Softmax top-k over `logits [T, E]`: u32 `[2 * T * K]`, ids then f32 weight bits.
pub fn moe_route(logits: &Tensor, top_k: usize, norm: bool) -> CResult<Tensor> {
    logits
        .to_dtype(DType::F32)?
        .contiguous()?
        .apply_op1_no_bwd(&MoeRoute { top_k, norm })
}

/// The routed experts' weighted sum for `route` (from [`moe_route`]); `x: [T, H]` -> `[T, H]`.
pub fn moe_compute(table: &MoeTable, x: &Tensor, route: &Tensor, top_k: usize) -> CResult<Tensor> {
    x.contiguous()?
        .apply_op2_no_bwd(&route.contiguous()?, &MoeCompute { table, top_k })
}

/// Routed-expert output of one MoE layer, entirely on the GPU: softmax top-k over `logits`
/// (`[T, E]`, f32), SiLU-gated expert MLPs, weighted sum. `x: [T, H]`; returns `[T, H]`.
/// Experts the table marks as remote contribute zero (expert parallel sums across ranks).
pub fn moe_experts(
    table: &MoeTable,
    x: &Tensor,
    logits: &Tensor,
    top_k: usize,
    norm: bool,
) -> CResult<Tensor> {
    let (t, e) = logits.dims2()?;
    if x.dim(0)? != t || e != table.num_experts {
        candle_core::bail!("moe experts: logits {:?} do not match the table", logits.shape());
    }
    moe_compute(table, x, &moe_route(logits, top_k, norm)?, top_k)
}

// ---- Offloaded experts: device slot cache (moe.cu `moe_slot_*`) ----

/// Host memory allocated pinned and mapped into the device address space, so kernels read it
/// over PCIe directly (`moe_slot_copy`).
pub struct PinnedHost {
    ptr: *mut u8,
    len: usize,
    /// The device address of `ptr`.
    pub dptr: u64,
}

// SAFETY: plain host memory owned by this value; the driver allows any thread to use or free it.
unsafe impl Send for PinnedHost {}
unsafe impl Sync for PinnedHost {}

impl PinnedHost {
    pub fn new(dev: &candle_core::CudaDevice, len: usize) -> CResult<Self> {
        use candle_core::cuda_backend::cudarc::driver::{result, sys};
        let _ = dev.cuda_stream().context().bind_to_thread();
        let flags = sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP;
        let ptr = unsafe { result::malloc_host(len.max(1), flags) }.w()? as *mut u8;
        let mut dptr = 0;
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dptr, ptr as *mut _, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            unsafe {
                let _ = result::free_host(ptr as *mut _);
            }
            candle_core::bail!("pinned host memory has no device mapping: {r:?}");
        }
        Ok(Self { ptr, len, dptr })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `len` bytes allocated by `new`, exclusively borrowed
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for PinnedHost {
    fn drop(&mut self) {
        unsafe {
            let _ = candle_core::cuda_backend::cudarc::driver::result::free_host(self.ptr as *mut _);
        }
    }
}

/// The device-side state of a stage's expert slot cache.
pub struct SlotState<'a> {
    /// i32 `[rows * E]`: slot of (layer row, expert), -1 when not resident.
    pub slot_of: &'a Tensor,
    /// i32 `[S]`: `row * E + expert` held by each slot, -1 when empty.
    pub owner: &'a Tensor,
    /// u64 (as i64) `[S]`: the step each slot was last used.
    pub usage: &'a Tensor,
    /// i64 `[1]` step counter, i64 `[3]` hits / copies / misses left without a slot.
    pub step: &'a Tensor,
    pub stats: &'a Tensor,
    /// u32 scratch: queue length `[1]`, slots and experts `[E]`.
    pub copy_n: &'a Tensor,
    pub copy_slot: &'a Tensor,
    pub copy_exp: &'a Tensor,
    pub slots: usize,
    /// Blocks of the copy kernel.
    pub copy_blocks: u32,
}

struct SlotEnsure<'a> {
    state: &'a SlotState<'a>,
    table: &'a MoeTable,
    /// u64 (as i64) `[18]`: src[6], dst[6], bytes[6].
    desc: &'a Tensor,
    row: usize,
    pairs: usize,
    /// Launch the slot assignment / the copy of the queued misses.
    ensure: bool,
    copy: bool,
}

impl candle_core::InplaceOp1 for SlotEnsure<'_> {
    fn name(&self) -> &'static str {
        "carbon-moe-slot-ensure"
    }
    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> CResult<()> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(&self, rt: &mut CudaStorage, rl: &Layout) -> CResult<()> {
        let st = self.state;
        let e = self.table.num_experts;
        if rl.shape().elem_count() != 2 * self.pairs || e > 1024 {
            candle_core::bail!("moe slot ensure: bad route {:?}", rl.shape());
        }
        let dev = rt.device().clone();
        let f_ens = dev.get_or_load_custom_func("moe_slot_ensure", "carbon_moe", moe_ptx()?)?;
        let f_cp = dev.get_or_load_custom_func("moe_slot_copy", "carbon_moe", moe_ptx()?)?;
        let (a, _) = rl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("moe slot ensure: route not contiguous".into()))?;
        let route = rt.as_cuda_slice::<u32>()?.slice(a..a + self.pairs);
        let sides = [
            Side::new(st.slot_of, "slot map")?,
            Side::new(st.owner, "slot owners")?,
            Side::new(st.usage, "slot usage")?,
            Side::new(st.step, "slot step")?,
            Side::new(st.stats, "slot stats")?,
            Side::new(&self.table.tab, "moe table")?,
            Side::new(self.desc, "copy descriptor")?,
            Side::new(st.copy_n, "copy count")?,
            Side::new(st.copy_slot, "copy slots")?,
            Side::new(st.copy_exp, "copy experts")?,
        ];
        let [slot_of, owner, usage, step, stats, tab, desc, copy_n, copy_slot, copy_exp] = &sides;
        let (slot_of, owner) = (slot_of.view::<i32>()?, owner.view::<i32>()?);
        let (usage, step, stats) = (usage.view::<i64>()?, step.view::<i64>()?, stats.view::<i64>()?);
        let (tab, desc) = (tab.view::<i64>()?, desc.view::<i64>()?);
        let (copy_n, copy_slot, copy_exp) =
            (copy_n.view::<u32>()?, copy_slot.view::<u32>()?, copy_exp.view::<u32>()?);
        let (p32, e32, s32, r32) = (self.pairs as i32, e as i32, st.slots as i32, self.row as i32);
        if self.ensure {
        let mut b = f_ens.builder();
        b.arg(&route);
        b.arg(&p32);
        b.arg(&e32);
        b.arg(&s32);
        b.arg(&r32);
        b.arg(&slot_of);
        b.arg(&owner);
        b.arg(&usage);
        b.arg(&step);
        b.arg(&stats);
        b.arg(&tab);
        b.arg(&desc);
        b.arg(&copy_n);
        b.arg(&copy_slot);
        b.arg(&copy_exp);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.w()?;
        }
        if self.copy {
            let mut b = f_cp.builder();
            b.arg(&copy_n);
            b.arg(&copy_slot);
            b.arg(&copy_exp);
            b.arg(&desc);
            let cfg = LaunchConfig {
                grid_dim: (st.copy_blocks.max(1), 1, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { b.launch(cfg) }.w()?;
        }
        Ok(())
    }
}

/// Make the experts `route` (from [`moe_route`], `pairs` = T*K) selects resident in the slot
/// cache for stage-local MoE layer `row`, copying misses from pinned host memory, and point
/// `table` at them. Device-side only: no host synchronisation, capturable. With `copy` false
/// only the slots are assigned and queued; [`moe_slot_copy`] then moves the bytes.
pub fn moe_slot_ensure(
    state: &SlotState<'_>,
    table: &MoeTable,
    desc: &Tensor,
    row: usize,
    route: &Tensor,
    pairs: usize,
    copy: bool,
) -> CResult<()> {
    route.inplace_op1(&SlotEnsure {
        state,
        table,
        desc,
        row,
        pairs,
        ensure: true,
        copy,
    })
}

/// Copy the misses the last `moe_slot_ensure(.., copy: false)` queued.
pub fn moe_slot_copy(
    state: &SlotState<'_>,
    table: &MoeTable,
    desc: &Tensor,
    route: &Tensor,
    pairs: usize,
) -> CResult<()> {
    route.inplace_op1(&SlotEnsure {
        state,
        table,
        desc,
        row: 0,
        pairs,
        ensure: false,
        copy: true,
    })
}

/// A CUDA stream (and its events) that cudarc does not manage. A second cudarc stream would put
/// the context in multi-stream mode, where every buffer records events on use: overhead, and an
/// event wait inside a CUDA-graph capture invalidates it.
pub struct RawStream(sys::CUstream);

/// An event recorded on a [`RawStream`].
pub struct RawEvent(sys::CUevent);

// SAFETY: driver handles, usable from any thread with the context current.
unsafe impl Send for RawStream {}
unsafe impl Sync for RawStream {}
unsafe impl Send for RawEvent {}
unsafe impl Sync for RawEvent {}

fn check(r: sys::CUresult, what: &str) -> CResult<()> {
    if r == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        candle_core::bail!("{what}: {r:?}")
    }
}

impl RawStream {
    pub fn new(dev: &candle_core::CudaDevice) -> CResult<Self> {
        dev.cuda_stream().context().bind_to_thread().w()?;
        let mut s = std::ptr::null_mut();
        check(unsafe { sys::cuStreamCreate(&mut s, sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32) }, "creating a stream")?;
        Ok(Self(s))
    }

    /// An event at the stream's current end.
    pub fn record(&self) -> CResult<RawEvent> {
        let mut e = std::ptr::null_mut();
        check(unsafe { sys::cuEventCreate(&mut e, sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32) }, "creating an event")?;
        let ev = RawEvent(e);
        check(unsafe { sys::cuEventRecord(ev.0, self.0) }, "recording an event")?;
        Ok(ev)
    }
}

impl Drop for RawStream {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuStreamDestroy_v2(self.0);
        }
    }
}

impl Drop for RawEvent {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuEventDestroy_v2(self.0);
        }
    }
}

/// Make `dev`'s stream wait for `ev` (not inside a CUDA-graph capture).
pub fn wait_event(dev: &candle_core::CudaDevice, ev: &RawEvent) -> CResult<()> {
    check(unsafe { sys::cuStreamWaitEvent(dev.cuda_stream().cu_stream(), ev.0, 0) }, "waiting for an event")
}

/// Copy the queued misses `start..start + n` (of the last `moe_slot_ensure(.., copy: false)`)
/// on `stream`, with `blocks` blocks; the caller orders `stream` after the assignment.
#[allow(clippy::too_many_arguments)]
pub fn moe_slot_copy_range_on(
    stream: &RawStream,
    dev: &candle_core::CudaDevice,
    state: &SlotState<'_>,
    desc: &Tensor,
    start: usize,
    n: usize,
    blocks: u32,
) -> CResult<()> {
    let f = dev.get_or_load_custom_func("moe_slot_copy_range", "carbon_moe", moe_ptx()?)?;
    let (mut cs, mut ce, mut d) = (raw_ptr(state.copy_slot)?, raw_ptr(state.copy_exp)?, raw_ptr(desc)?);
    let (mut s32, mut n32) = (start as u32, n as u32);
    let mut params: [*mut std::ffi::c_void; 5] = [
        &mut cs as *mut u64 as *mut _,
        &mut ce as *mut u64 as *mut _,
        &mut d as *mut u64 as *mut _,
        &mut s32 as *mut u32 as *mut _,
        &mut n32 as *mut u32 as *mut _,
    ];
    check(
        unsafe {
            sys::cuLaunchKernel(f.cu_function(), blocks.max(1), 1, 1, 512, 1, 1, 0, stream.0, params.as_mut_ptr(), std::ptr::null_mut())
        },
        "launching moe_slot_copy_range",
    )
}

// ---- grouped prefill MoE on tensor cores (moe_tc.cu) ----

fn moe_tc_ptx() -> CResult<&'static str> {
    static PTX: OnceLock<Result<String, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        let opts = CompileOptions {
            arch: Some(nvrtc_arch(80)),
            ..Default::default()
        };
        let src = format!("{}\n{}", include_str!("quant.cu"), include_str!("moe_tc.cu"));
        compile_ptx_with_opts(src, opts)
            .map(|p| p.to_src())
            .map_err(|e| format!("{e:?}"))
    })
    .as_ref()
    .map(|s| s.as_str())
    .map_err(|e| candle_core::Error::Msg(format!("NVRTC failed to compile carbon grouped moe: {e}")))
}

/// Pairs per tile of [`moe_grouped`] (MT_M in moe_tc.cu).
pub const MOE_TILE: usize = 64;

/// Whether prefill MoE over `table`'s experts runs as grouped tensor-core GEMMs: NVFP4 experts,
/// bf16 activations, dims multiples of 32, compute capability 8.0+, `CARBON_MOE_TC` not `0`.
pub fn moe_grouped_supported(table: &MoeTable, dtype: DType, dev: &candle_core::Device) -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    let on = *ENV.get_or_init(|| std::env::var("CARBON_MOE_TC").map_or(true, |v| v != "0"));
    let candle_core::Device::Cuda(d) = dev else { return false };
    on && table.fname == "nvfp4"
        && dtype == DType::BF16
        && table.hidden % 32 == 0
        && table.inter % 32 == 0
        && d.cuda_stream().context().compute_capability().is_ok_and(|(major, _)| major >= 8)
}

/// Grouped prefill MoE for the `ntiles` tiles of `tiles` (u32 `[n, 3]`: expert, first pair,
/// pairs): `act [P, I]` = silu(gate) * up of the expert-sorted pairs `xs [P, H]`, then
/// `y [P, H]` = `wts[p]` * down(act). Writes the tiles' rows of `act` and `y` in place, on the
/// device's stream.
pub fn moe_grouped(
    table: &MoeTable,
    xs: &Tensor,
    wts: &Tensor,
    tiles: &Tensor,
    ntiles: usize,
    act: &Tensor,
    y: &Tensor,
) -> CResult<()> {
    if ntiles == 0 {
        return Ok(());
    }
    let candle_core::Device::Cuda(dev) = xs.device() else {
        candle_core::bail!("moe grouped: not on CUDA")
    };
    let stream = dev.cuda_stream();
    let (h, i) = (table.hidden as i32, table.inter as i32);
    let (tab, glob) = (raw_ptr(&table.tab)?, raw_ptr(&table.glob)?);
    let (px, pw, pt, pa, py) = (raw_ptr(xs)?, raw_ptr(wts)?, raw_ptr(tiles)?, raw_ptr(act)?, raw_ptr(y)?);
    let f = dev.get_or_load_custom_func("moe_grouped_gate_up_bf16", "carbon_moe_tc", moe_tc_ptx()?)?;
    let mut b = stream.launch_builder(&f);
    b.arg(&tab);
    b.arg(&glob);
    b.arg(&px);
    b.arg(&pt);
    b.arg(&pa);
    b.arg(&i);
    b.arg(&h);
    let cfg = LaunchConfig {
        grid_dim: (ntiles as u32, (table.inter as u32).div_ceil(64), 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { b.launch(cfg) }.w()?;
    let f = dev.get_or_load_custom_func("moe_grouped_down_bf16", "carbon_moe_tc", moe_tc_ptx()?)?;
    let mut b = stream.launch_builder(&f);
    b.arg(&tab);
    b.arg(&glob);
    b.arg(&pa);
    b.arg(&pw);
    b.arg(&pt);
    b.arg(&py);
    b.arg(&i);
    b.arg(&h);
    let cfg = LaunchConfig {
        grid_dim: (ntiles as u32, (table.hidden as u32).div_ceil(64), 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { b.launch(cfg) }.w()?;
    Ok(())
}

/// Device address of a contiguous CUDA tensor (start of its view).
pub fn device_address(t: &Tensor) -> CResult<u64> {
    raw_ptr(t)
}

// ---- Gated DeltaNet decode (gdn.cu) ----

fn gdn_ptx() -> CResult<&'static str> {
    static PTX: OnceLock<Result<String, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        let opts = CompileOptions {
            arch: Some(nvrtc_arch(70)),
            ..Default::default()
        };
        let src = format!("{}\n{}", include_str!("quant.cu"), include_str!("gdn.cu"));
        compile_ptx_with_opts(src, opts)
            .map(|p| p.to_src())
            .map_err(|e| format!("{e:?}"))
    })
    .as_ref()
    .map(|s| s.as_str())
    .map_err(|e| candle_core::Error::Msg(format!("NVRTC failed to compile carbon gdn: {e}")))
}

/// Weights and state pools of one Gated DeltaNet layer, as the decode kernels read them.
pub struct GdnLayer<'a> {
    /// `[kernel, conv_dim]`, model dtype.
    pub taps: &'a Tensor,
    /// `[hv]` f32 each.
    pub dt_bias: &'a Tensor,
    pub a_neg: &'a Tensor,
    /// `[dv]`, model dtype.
    pub norm_w: &'a Tensor,
    /// `[slots, (kernel-1) * conv_dim]` model dtype and `[slots, hv*dk*dv]` f32, updated in place.
    pub conv_pool: &'a Tensor,
    pub ssm_pool: &'a Tensor,
    pub hk: usize,
    pub hv: usize,
    pub dk: usize,
    pub dv: usize,
    pub eps: f32,
    /// Output gate `sigmoid(z)` instead of `silu(z)`.
    pub gate_sigmoid: bool,
}

/// Whether [`gdn_decode`] handles these head dims.
pub fn gdn_decode_supported(dk: usize, dv: usize) -> bool {
    dk <= 256 && dv <= 256
}

struct GdnDecode<'a> {
    layer: &'a GdnLayer<'a>,
    /// Inputs besides qkv: z, b, a projections, state slots (u32), fresh flags (u32).
    z: &'a Tensor,
    braw: &'a Tensor,
    araw: &'a Tensor,
    idx: &'a Tensor,
    fresh: &'a Tensor,
}

/// A read guard plus contiguous element range of a tensor that is not one of the op's inputs.
struct Side<'a> {
    guard: std::sync::RwLockReadGuard<'a, candle_core::Storage>,
    range: (usize, usize),
}

impl<'a> Side<'a> {
    fn new(t: &'a Tensor, what: &str) -> CResult<Self> {
        let (guard, layout) = t.storage_and_layout();
        let range = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg(format!("gdn decode: {what} not contiguous")))?;
        if !matches!(&*guard, candle_core::Storage::Cuda(_)) {
            candle_core::bail!("gdn decode: {what} not on CUDA");
        }
        Ok(Self { guard, range })
    }
    fn view<T: candle_core::cuda_backend::CudaDType>(&self) -> CResult<CudaView<'_, T>> {
        let candle_core::Storage::Cuda(s) = &*self.guard else {
            unreachable!()
        };
        Ok(s.as_cuda_slice::<T>()?.slice(self.range.0..self.range.1))
    }
    fn len(&self) -> usize {
        self.range.1 - self.range.0
    }
}

impl candle_core::CustomOp1 for GdnDecode<'_> {
    fn name(&self) -> &'static str {
        "carbon-gdn-decode"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(&self, qkv: &CudaStorage, ql: &Layout) -> CResult<(CudaStorage, Shape)> {
        let l = self.layer;
        let (b, c) = ql.shape().dims2()?;
        let (hk, hv, dk, dv) = (l.hk, l.hv, l.dk, l.dv);
        if c != 2 * hk * dk + hv * dv || hk == 0 || hv % hk != 0 || !gdn_decode_supported(dk, dv) {
            candle_core::bail!("gdn decode: bad geometry");
        }
        let dtype = qkv.dtype();
        let taps = Side::new(l.taps, "taps")?;
        let kernel = taps.len() / c;
        let km1 = kernel - 1;
        let sides = [
            (Side::new(self.z, "z")?, b * hv * dv),
            (Side::new(self.braw, "b")?, b * hv),
            (Side::new(self.araw, "a")?, b * hv),
            (Side::new(self.idx, "slots")?, b),
            (Side::new(self.fresh, "fresh")?, b),
            (Side::new(l.dt_bias, "dt_bias")?, hv),
            (Side::new(l.a_neg, "a_neg")?, hv),
            (Side::new(l.norm_w, "norm")?, dv),
        ];
        if taps.len() != kernel * c || kernel == 0 || sides.iter().any(|(s, n)| s.len() != *n) {
            candle_core::bail!("gdn decode: input sizes do not match the layer");
        }
        let [(z, _), (braw, _), (araw, _), (idx, _), (fresh, _), (dt_bias, _), (a_neg, _), (norm_w, _)] =
            sides;
        let conv_pool = Side::new(l.conv_pool, "conv pool")?;
        let ssm_pool = Side::new(l.ssm_pool, "ssm pool")?;
        if conv_pool.len() % (km1 * c).max(1) != 0 || ssm_pool.len() % (hv * dk * dv) != 0 {
            candle_core::bail!("gdn decode: state pools do not match the layer");
        }
        let dev = qkv.device().clone();
        let sfx = suffix(dtype)?;
        let f_conv =
            dev.get_or_load_custom_func(&format!("gdn_conv_{sfx}"), "carbon_gdn", gdn_ptx()?)?;
        let f_step =
            dev.get_or_load_custom_func(&format!("gdn_step_{sfx}"), "carbon_gdn", gdn_ptx()?)?;
        let act = unsafe { dev.alloc::<f32>(b * c)? };
        let (idx, fresh) = (idx.view::<u32>()?, fresh.view::<u32>()?);
        let (dt_bias, a_neg) = (dt_bias.view::<f32>()?, a_neg.view::<f32>()?);
        let ssm = ssm_pool.view::<f32>()?;
        let (b32, c32, km1_32) = (b as i32, c as i32, km1 as i32);
        let (hk32, hv32, dk32, dv32) = (hk as i32, hv as i32, dk as i32, dv as i32);
        macro_rules! go {
            ($ty:ty) => {{
                let xv = view::<$ty>(qkv, ql)?;
                let pool = conv_pool.view::<$ty>()?;
                let tv = taps.view::<$ty>()?;
                let mut bld = f_conv.builder();
                bld.arg(&xv);
                bld.arg(&pool);
                bld.arg(&idx);
                bld.arg(&fresh);
                bld.arg(&tv);
                bld.arg(&act);
                bld.arg(&b32);
                bld.arg(&c32);
                bld.arg(&km1_32);
                unsafe { bld.launch(grid_stride(b * c)) }.w()?;

                let (zv, bv, av, nv) = (
                    z.view::<$ty>()?,
                    braw.view::<$ty>()?,
                    araw.view::<$ty>()?,
                    norm_w.view::<$ty>()?,
                );
                let out = unsafe { dev.alloc::<$ty>(b * hv * dv)? };
                let mut bld = f_step.builder();
                bld.arg(&act);
                bld.arg(&zv);
                bld.arg(&bv);
                bld.arg(&av);
                bld.arg(&ssm);
                bld.arg(&idx);
                bld.arg(&fresh);
                bld.arg(&dt_bias);
                bld.arg(&a_neg);
                bld.arg(&nv);
                bld.arg(&out);
                bld.arg(&hk32);
                bld.arg(&hv32);
                bld.arg(&dk32);
                bld.arg(&dv32);
                bld.arg(&l.eps);
                let gs = i32::from(l.gate_sigmoid);
                bld.arg(&gs);
                let cfg = LaunchConfig {
                    grid_dim: ((b * hv) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { bld.launch(cfg) }.w()?;
                CudaStorage::wrap_cuda_slice(out, dev.clone())
            }};
        }
        let out = match dtype {
            DType::F32 => go!(f32),
            DType::BF16 => go!(bf16),
            DType::F16 => go!(f16),
            d => candle_core::bail!("gdn decode: unsupported dtype {d:?}"),
        };
        Ok((out, Shape::from((b, hv * dv))))
    }
}

/// One Gated DeltaNet decode step for `B` sequences (one token each): `qkv [B, conv_dim]`,
/// `z [B, hv*dv]`, `braw`/`araw [B, hv]` (projection outputs, model dtype), `slots`/`fresh`
/// u32 `[B]` (fresh = the sequence starts here, its state is treated as zero). Updates both
/// state pools in place and returns the gated, normalised core output `[B, hv*dv]`, ready for
/// the output projection.
pub fn gdn_decode(
    layer: &GdnLayer<'_>,
    qkv: &Tensor,
    z: &Tensor,
    braw: &Tensor,
    araw: &Tensor,
    slots: &Tensor,
    fresh: &Tensor,
) -> CResult<Tensor> {
    let (z, braw, araw) = (z.contiguous()?, braw.contiguous()?, araw.contiguous()?);
    qkv.contiguous()?.apply_op1_no_bwd(&GdnDecode {
        layer,
        z: &z,
        braw: &braw,
        araw: &araw,
        idx: slots,
        fresh,
    })
}

// ---- prefill attention (attention.cu) ----

/// Queries per entry of the prefill tile table (FP_TILE in attention.cu, TC_BQ in attention_tc.cu).
pub const PREFILL_TILE: usize = 64;

fn attn_tc_ptx() -> CResult<&'static str> {
    static PTX: OnceLock<Result<String, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        let opts = CompileOptions {
            arch: Some(nvrtc_arch(80)),
            ..Default::default()
        };
        let src = format!(
            "{}\n{}",
            include_str!("attention.cu"),
            include_str!("attention_tc.cu")
        );
        compile_ptx_with_opts(src, opts)
            .map(|p| p.to_src())
            .map_err(|e| format!("{e:?}"))
    })
    .as_ref()
    .map(|s| s.as_str())
    .map_err(|e| {
        candle_core::Error::Msg(format!(
            "NVRTC failed to compile carbon tensor-core attention: {e}"
        ))
    })
}

/// Whether prefill attention on `dev` runs on tensor cores for this dtype / head dim:
/// bf16 or f16, head dim 64 / 128 / 256, compute capability 8.0+, `CARBON_ATTN_TC` not `0`.
fn prefill_on_tensor_cores(dev: &candle_core::CudaDevice, dtype: DType, d: usize) -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    let on = *ENV.get_or_init(|| std::env::var("CARBON_ATTN_TC").map_or(true, |v| v != "0"));
    on && matches!(dtype, DType::BF16 | DType::F16)
        && matches!(d, 64 | 128 | 256)
        && dev
            .cuda_stream()
            .context()
            .compute_capability()
            .is_ok_and(|(major, _)| major >= 8)
}

struct FlashPrefill<'a> {
    slots: &'a Tensor,
    seqinfo: &'a Tensor,
    tiles: &'a Tensor,
    kv_heads: usize,
    scale: f32,
}

impl CustomOp3 for FlashPrefill<'_> {
    fn name(&self) -> &'static str {
        "carbon-flash-prefill"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(
        &self,
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        v: &CudaStorage,
        vl: &Layout,
    ) -> CResult<(CudaStorage, Shape)> {
        let (t, h, d) = ql.shape().dims3()?;
        let kvh = self.kv_heads;
        if kl.shape() != vl.shape() || kl.shape().dims2()?.1 != kvh * d {
            candle_core::bail!("flash prefill: bad kv shape {:?}", kl.shape());
        }
        if !paged_decode_supported(h, kvh, d) {
            candle_core::bail!("flash prefill: unsupported head geometry h={h} kvh={kvh} d={d}");
        }
        let dtype = q.dtype();
        if k.dtype() != dtype || v.dtype() != dtype {
            candle_core::bail!("flash prefill: q/k/v dtypes differ");
        }
        let slots = Side::new(self.slots, "slots")?;
        let seqinfo = Side::new(self.seqinfo, "seqinfo")?;
        let tiles = Side::new(self.tiles, "tiles")?;
        let n_tiles = tiles.len() / 2;
        let dev = q.device().clone();
        let tc = prefill_on_tensor_cores(&dev, dtype, d);
        let (f, blocks) = if tc {
            let f = dev.get_or_load_custom_func(
                &format!("flash_prefill_tc_{}_d{d}", suffix(dtype)?),
                "carbon_attn_tc",
                attn_tc_ptx()?,
            )?;
            // one block per (tile, head, slice of up to 128 output dims)
            (f, n_tiles * h * d.div_ceil(128))
        } else {
            let f = dev.get_or_load_custom_func(
                &format!("flash_prefill_{}", suffix(dtype)?),
                "carbon_attn",
                attn_ptx()?,
            )?;
            // one block per 16-query quarter of each tile and head
            (f, n_tiles * h * (PREFILL_TILE / 16))
        };
        let (sv, iv, tv) = (
            slots.view::<u32>()?,
            seqinfo.view::<u32>()?,
            tiles.view::<u32>()?,
        );
        let (h32, kvh32, d32) = (h as i32, kvh as i32, d as i32);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        macro_rules! go {
            ($ty:ty) => {{
                let (qv, kv, vv) = (
                    view::<$ty>(q, ql)?,
                    view::<$ty>(k, kl)?,
                    view::<$ty>(v, vl)?,
                );
                // zeroed: rows no tile covers (none, when the tables are consistent) stay finite
                let out = dev.alloc_zeros::<$ty>(t * h * d)?;
                let mut b = f.builder();
                b.arg(&qv);
                b.arg(&kv);
                b.arg(&vv);
                b.arg(&sv);
                b.arg(&iv);
                b.arg(&tv);
                b.arg(&out);
                b.arg(&h32);
                b.arg(&kvh32);
                if !tc {
                    b.arg(&d32); // the tensor-core kernels are compiled per head dim
                }
                b.arg(&self.scale);
                unsafe { b.launch(cfg) }.w()?;
                CudaStorage::wrap_cuda_slice(out, dev.clone())
            }};
        }
        let out = match dtype {
            DType::F32 => go!(f32),
            DType::BF16 => go!(bf16),
            DType::F16 => go!(f16),
            d => candle_core::bail!("flash prefill: unsupported dtype {d:?}"),
        };
        Ok((out, Shape::from((t, h, d))))
    }
}

/// Causal attention for a batch of sequences whose K/V are already in the pool:
/// `q [T, H, D]` (all sequences' new queries, concatenated), pools `[S, KVH*D]`,
/// `slots` (concatenated slot tables), `seqinfo [B*4]` = (q_start, q_len, slot_start, kv_len),
/// `tiles [n*2]` = (sequence, first query) per [`PREFILL_TILE`] queries; u32 on the device.
#[allow(clippy::too_many_arguments)]
pub fn flash_prefill(
    q: &Tensor,
    k_pool: &Tensor,
    v_pool: &Tensor,
    slots: &Tensor,
    seqinfo: &Tensor,
    tiles: &Tensor,
    kv_heads: usize,
    scale: f32,
) -> CResult<Tensor> {
    q.contiguous()?.apply_op3_no_bwd(
        &k_pool.contiguous()?,
        &v_pool.contiguous()?,
        &FlashPrefill {
            slots,
            seqinfo,
            tiles,
            kv_heads,
            scale,
        },
    )
}

// ---- prefill gated delta rule (gdn.cu: gdn_seq) ----

struct GdnSeq<'a> {
    k: &'a Tensor,
    v: &'a Tensor,
    beta: &'a Tensor,
    g: &'a Tensor,
    s0: &'a Tensor,
    dims: (usize, usize, usize, usize),
}

impl candle_core::CustomOp1 for GdnSeq<'_> {
    fn name(&self) -> &'static str {
        "carbon-gdn-seq"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(&self, q: &CudaStorage, ql: &Layout) -> CResult<(CudaStorage, Shape)> {
        let (bh, t, dk, dv) = self.dims;
        if q.dtype() != DType::F32 || ql.shape().elem_count() != bh * t * dk {
            candle_core::bail!("gdn seq: q must be f32 [BH, T, DK]");
        }
        let sides = [
            (Side::new(self.k, "k")?, bh * t * dk),
            (Side::new(self.v, "v")?, bh * t * dv),
            (Side::new(self.beta, "beta")?, bh * t),
            (Side::new(self.g, "g")?, bh * t),
            (Side::new(self.s0, "state")?, bh * dk * dv),
        ];
        if sides.iter().any(|(s, n)| s.len() != *n) {
            candle_core::bail!("gdn seq: input sizes do not match");
        }
        let dev = q.device().clone();
        let f = dev.get_or_load_custom_func("gdn_seq", "carbon_gdn", gdn_ptx()?)?;
        let qv = view::<f32>(q, ql)?;
        let [(k, _), (v, _), (beta, _), (g, _), (s0, _)] = sides;
        let (k, v, beta, g, s0) = (
            k.view::<f32>()?,
            v.view::<f32>()?,
            beta.view::<f32>()?,
            g.view::<f32>()?,
            s0.view::<f32>()?,
        );
        let n_out = bh * t * dv;
        let out = unsafe { dev.alloc::<f32>(n_out + bh * dk * dv)? };
        let (o, s1) = (out.slice(0..n_out), out.slice(n_out..));
        let (bh32, t32, dk32, dv32) = (bh as i32, t as i32, dk as i32, dv as i32);
        let mut b = f.builder();
        b.arg(&qv);
        b.arg(&k);
        b.arg(&v);
        b.arg(&beta);
        b.arg(&g);
        b.arg(&s0);
        b.arg(&o);
        b.arg(&s1);
        b.arg(&bh32);
        b.arg(&t32);
        b.arg(&dk32);
        b.arg(&dv32);
        let cfg = LaunchConfig {
            grid_dim: ((bh * dv).div_ceil(4) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.w()?;
        drop(b);
        Ok((
            CudaStorage::wrap_cuda_slice(out, dev.clone()),
            Shape::from(n_out + bh * dk * dv),
        ))
    }
}

/// Whether [`gdn_sequence`] handles these head dims.
pub fn gdn_sequence_supported(dk: usize) -> bool {
    dk.is_multiple_of(32) && dk <= 256
}

/// The gated delta rule over whole sequences, as [`crate::model::gdn::recurrent`]:
/// `q,k [B,H,T,Dk]`, `v [B,H,T,Dv]`, `beta,g [B,H,T]`, `s [B,H,Dk,Dv]` (f32) ->
/// `([B,H,T,Dv], final state)`.
pub fn gdn_sequence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    s0: &Tensor,
) -> CResult<(Tensor, Tensor)> {
    let (b, h, t, dk) = q.dims4()?;
    let dv = v.dim(3)?;
    if !gdn_sequence_supported(dk) {
        candle_core::bail!("gdn seq: unsupported k head dim {dk}");
    }
    let c = |x: &Tensor| -> CResult<Tensor> { x.to_dtype(DType::F32)?.contiguous() };
    let (k, v, beta, g, s0) = (c(k)?, c(v)?, c(beta)?, c(g)?, c(s0)?);
    let out = c(q)?.apply_op1_no_bwd(&GdnSeq {
        k: &k,
        v: &v,
        beta: &beta,
        g: &g,
        s0: &s0,
        dims: (b * h, t, dk, dv),
    })?;
    let n_out = b * h * t * dv;
    Ok((
        out.narrow(0, 0, n_out)?.reshape((b, h, t, dv))?,
        out.narrow(0, n_out, b * h * dk * dv)?
            .reshape((b, h, dk, dv))?,
    ))
}

#[cfg(test)]
mod tests {
    /// Every kernel source compiles with NVRTC exactly as the engine compiles it at start-up
    /// (needs libnvrtc, not a GPU).
    #[test]
    fn kernels_compile_with_nvrtc() {
        for (name, ptx) in [
            ("quant", super::ptx()),
            ("attention", super::attn_ptx()),
            ("attention_tc", super::attn_tc_ptx()),
            ("moe", super::moe_ptx()),
            ("gdn", super::gdn_ptx()),
            ("sampling", super::sample_ptx()),
            ("qsa", super::qsa_ptx()),
        ] {
            let ptx = ptx.unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(ptx.contains(".entry"), "{name}: no kernels in the PTX");
        }
    }
}

#[cfg(test)]
mod arch_tests {
    use super::*;

    /// Whatever NVRTC is installed (CUDA 12 takes compute_70, CUDA 13 starts at compute_75),
    /// the probed target must compile carbon's kernels. Needs NVRTC, not a GPU.
    #[test]
    fn probed_arch_compiles_the_kernels() {
        let arch = nvrtc_arch(70);
        eprintln!("NVRTC target: {arch}");
        compile_ptx_with_opts(
            include_str!("quant.cu"),
            CompileOptions {
                arch: Some(arch),
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("quant.cu does not compile for {arch}: {e:?}"));
        // the tensor-core prefill path asks for >= 80
        let tc = nvrtc_arch(80);
        assert!(ARCHS.iter().any(|(cc, a)| *a == tc && *cc >= 80), "{tc}");
    }
}

// ---- sampling (sampling.cu) ----

fn sample_ptx() -> CResult<&'static str> {
    static PTX: OnceLock<Result<String, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        let opts = CompileOptions {
            arch: Some(nvrtc_arch(70)),
            ..Default::default()
        };
        compile_ptx_with_opts(include_str!("sampling.cu"), opts)
            .map(|p| p.to_src())
            .map_err(|e| format!("{e:?}"))
    })
    .as_ref()
    .map(|s| s.as_str())
    .map_err(|e| candle_core::Error::Msg(format!("NVRTC failed to compile carbon sampling: {e}")))
}

/// Per-row sampling parameters for [`sample_on_gpu`].
#[derive(Clone, Copy, Debug)]
pub struct RowSampling {
    pub temperature: f32,
    /// 0 = off.
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
}

struct GpuSample;

impl candle_core::CustomOp2 for GpuSample {
    fn name(&self) -> &'static str {
        "carbon-sample"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(
        &self,
        logits: &CudaStorage,
        ll: &Layout,
        params: &CudaStorage,
        pl: &Layout,
    ) -> CResult<(CudaStorage, Shape)> {
        let (rows, v) = ll.shape().dims2()?;
        if pl.shape().dims2()? != (rows, 6) {
            candle_core::bail!("sample: params shape {:?} for {rows} rows", pl.shape());
        }
        let dev = logits.device().clone();
        let f = dev.get_or_load_custom_func("carbon_sample", "carbon_sample", sample_ptx()?)?;
        let (lv, pv) = (view::<f32>(logits, ll)?, view::<u32>(params, pl)?);
        let out = unsafe { dev.alloc::<u32>(rows * 2)? };
        let v32 = v as i32;
        let mut b = f.builder();
        b.arg(&lv);
        b.arg(&v32);
        b.arg(&pv);
        b.arg(&out);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.w()?;
        Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from((rows, 2))))
    }
}

/// Sample one token per row of `logits` `[rows, V]` (on CUDA) with temperature / top-k / top-p
/// / min-p. Only `rows * 2` u32 come back to the host: the token and a flag set when the row's
/// top-p nucleus is wider than the kernel's candidate set (the caller samples it on the host).
pub fn sample_on_gpu(logits: &Tensor, rows: &[RowSampling]) -> CResult<Vec<(u32, bool)>> {
    let params: Vec<u32> = rows
        .iter()
        .flat_map(|r| {
            [
                r.temperature.to_bits(),
                r.top_k,
                r.top_p.to_bits(),
                r.min_p.to_bits(),
                r.seed as u32,
                (r.seed >> 32) as u32,
            ]
        })
        .collect();
    let params = Tensor::from_vec(params, (rows.len(), 6), logits.device())?;
    let out = logits
        .to_dtype(DType::F32)?
        .contiguous()?
        .apply_op2_no_bwd(&params, &GpuSample)?
        .to_vec2::<u32>()?;
    Ok(out.into_iter().map(|r| (r[0], r[1] != 0)).collect())
}

// ---- QSA decode selection (qsa.cu) ----

fn qsa_ptx() -> CResult<&'static str> {
    static PTX: OnceLock<Result<String, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        let opts = CompileOptions {
            arch: Some(nvrtc_arch(70)),
            ..Default::default()
        };
        let src = format!("{}\n{}", include_str!("quant.cu"), include_str!("qsa.cu"));
        compile_ptx_with_opts(src, opts)
            .map(|p| p.to_src())
            .map_err(|e| format!("{e:?}"))
    })
    .as_ref()
    .map(|s| s.as_str())
    .map_err(|e| candle_core::Error::Msg(format!("NVRTC failed to compile carbon qsa: {e}")))
}

/// Whether [`qsa_select`] handles this indexer geometry.
pub fn qsa_select_supported(heads: usize, dim: usize) -> bool {
    heads <= 8 && dim <= 256 && dim > 0
}

/// Inputs of a QSA decode selection besides the queries.
pub struct QsaTables<'a> {
    /// `[slots, d]` raw indexer keys.
    pub pool: &'a Tensor,
    /// `[d]` k_layernorm weight (1 + w).
    pub knorm: &'a Tensor,
    /// `[positions, half]` rope tables.
    pub cos: &'a Tensor,
    pub sin: &'a Tensor,
    /// Every sequence's slot table and `[B + 1]` offsets (u32).
    pub slots: &'a Tensor,
    pub starts: &'a Tensor,
    pub topk: usize,
    pub ratio: usize,
    /// Capacity of the per-sequence score scratch (blocks); >= any sequence's block count.
    pub max_blocks: usize,
    pub eps: f32,
}

struct QsaSelect<'a>(&'a QsaTables<'a>);

impl candle_core::CustomOp1 for QsaSelect<'_> {
    fn name(&self) -> &'static str {
        "carbon-qsa-select"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(&self, q: &CudaStorage, ql: &Layout) -> CResult<(CudaStorage, Shape)> {
        let t = self.0;
        let (b, nh, d) = ql.shape().dims3()?;
        if !qsa_select_supported(nh, d) {
            candle_core::bail!("qsa select: unsupported geometry nh={nh} d={d}");
        }
        let half = t.cos.dim(1)?;
        let dtype = q.dtype();
        let (pool, knorm, cos, sin, slots, starts) = (
            Side::new(t.pool, "index pool")?,
            Side::new(t.knorm, "k norm")?,
            Side::new(t.cos, "cos")?,
            Side::new(t.sin, "sin")?,
            Side::new(t.slots, "slots")?,
            Side::new(t.starts, "starts")?,
        );
        if starts.len() != b + 1 {
            candle_core::bail!("qsa select: {} offsets for {b} sequences", starts.len());
        }
        let dev = q.device().clone();
        let f = dev.get_or_load_custom_func(&format!("qsa_select_{}", suffix(dtype)?), "carbon_qsa", qsa_ptx()?)?;
        let cap = (t.topk + 1) * t.ratio;
        let scores = unsafe { dev.alloc::<f32>((b * t.max_blocks).max(1))? };
        let out = dev.alloc_zeros::<u32>(b + 1 + b * cap)?;
        let (sv, ss) = (slots.view::<u32>()?, starts.view::<u32>()?);
        let (o_starts, o_slots) = (out.slice(0..b + 1), out.slice(b + 1..));
        let ints = [nh as i32, d as i32, half as i32, t.topk as i32, t.ratio as i32, t.max_blocks as i32];
        let cfg = LaunchConfig {
            grid_dim: (b as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        macro_rules! go {
            ($ty:ty) => {{
                let qv = view::<$ty>(q, ql)?;
                let (pv, kv, cv, sn) = (pool.view::<$ty>()?, knorm.view::<$ty>()?, cos.view::<$ty>()?, sin.view::<$ty>()?);
                let mut bld = f.builder();
                bld.arg(&qv);
                bld.arg(&pv);
                bld.arg(&kv);
                bld.arg(&cv);
                bld.arg(&sn);
                bld.arg(&sv);
                bld.arg(&ss);
                bld.arg(&scores);
                bld.arg(&o_slots);
                bld.arg(&o_starts);
                for i in &ints {
                    bld.arg(i);
                }
                bld.arg(&t.eps);
                unsafe { bld.launch(cfg) }.w()?;
            }};
        }
        match dtype {
            DType::F32 => go!(f32),
            DType::BF16 => go!(bf16),
            DType::F16 => go!(f16),
            d => candle_core::bail!("qsa select: unsupported dtype {d:?}"),
        }
        drop(scores);
        Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from(b + 1 + b * cap)))
    }
}

struct QsaTopkBits {
    topk: usize,
    ratio: usize,
    first: usize,
}

impl candle_core::CustomOp1 for QsaTopkBits {
    fn name(&self) -> &'static str {
        "carbon-qsa-topk-bits"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(&self, s: &CudaStorage, l: &Layout) -> CResult<(CudaStorage, Shape)> {
        let (n, nb) = l.shape().dims2()?;
        let w = nb.div_ceil(32).max(1);
        let dev = s.device().clone();
        let f = dev.get_or_load_custom_func("qsa_topk_bits", "carbon_qsa", qsa_ptx()?)?;
        let out = unsafe { dev.alloc::<u32>((n * w).max(1))? };
        if n > 0 {
            let sv = view::<f32>(s, l)?;
            let ints = [nb as i32, w as i32, self.topk as i32, self.ratio as i32, self.first as i32];
            let mut bld = f.builder();
            bld.arg(&sv);
            bld.arg(&out);
            for i in &ints {
                bld.arg(i);
            }
            let cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { bld.launch(cfg) }.w()?;
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from((n, w))))
    }
}

/// QSA prefill selection: `scores [N, NB]` (f32, block scores of queries at positions
/// `first..first + N`) -> `[N, ceil(NB / 32)]` u32 bitmasks of each query's best `topk` visible
/// blocks (all of them when it sees no more than `topk`), ties to the lower block.
pub fn qsa_topk_bits(scores: &Tensor, topk: usize, ratio: usize, first: usize) -> CResult<Tensor> {
    scores.contiguous()?.apply_op1_no_bwd(&QsaTopkBits { topk, ratio, first })
}

struct QsaBitsMask {
    kv_len: usize,
    ratio: usize,
    first: usize,
    neg: f32,
}

impl candle_core::CustomOp1 for QsaBitsMask {
    fn name(&self) -> &'static str {
        "carbon-qsa-bits-mask"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("cuda only")
    }
    fn cuda_fwd(&self, s: &CudaStorage, l: &Layout) -> CResult<(CudaStorage, Shape)> {
        let (n, w) = l.shape().dims2()?;
        let len = self.kv_len;
        let dev = s.device().clone();
        let f = dev.get_or_load_custom_func("qsa_bits_mask", "carbon_qsa", qsa_ptx()?)?;
        let out = unsafe { dev.alloc::<f32>((n * len).max(1))? };
        if n > 0 && len > 0 {
            let bv = view::<u32>(s, l)?;
            let ints = [w as i32, len as i32, self.ratio as i32, self.first as i32];
            let mut bld = f.builder();
            bld.arg(&bv);
            bld.arg(&out);
            for i in &ints {
                bld.arg(i);
            }
            bld.arg(&self.neg);
            let cfg = LaunchConfig {
                grid_dim: (len.div_ceil(256) as u32, n as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { bld.launch(cfg) }.w()?;
        }
        Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from((n, len))))
    }
}

/// Additive mask `[N, kv_len]` (f32: 0 or `neg`) for queries at `first..first + N` from their
/// [`qsa_topk_bits`] rows: the selected blocks plus the trailing partial block, causal.
pub fn qsa_bits_mask(bits: &Tensor, kv_len: usize, ratio: usize, first: usize, neg: f32) -> CResult<Tensor> {
    bits.contiguous()?.apply_op1_no_bwd(&QsaBitsMask { kv_len, ratio, first, neg })
}

/// QSA decode selection for `q [B, heads, d]` (normed, roped indexer queries): returns the
/// slot table (`[B * (topk + 1) * ratio]`, filled up to the last offset) and its `[B + 1]`
/// offsets for [`paged_decode`]. Everything stays on the device.
pub fn qsa_select(q: &Tensor, t: &QsaTables<'_>) -> CResult<(Tensor, Tensor)> {
    let b = q.dim(0)?;
    let out = q.contiguous()?.apply_op1_no_bwd(&QsaSelect(t))?;
    Ok((out.narrow(0, b + 1, out.dim(0)? - b - 1)?, out.narrow(0, 0, b + 1)?))
}
