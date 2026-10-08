// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Fused CPU kernels for quantized projections.
//!
//! Decode-sized batches multiply directly from the packed codes (FP8 e4m3 / NVFP4 e2m1) through
//! small lookup tables, so no dense weight matrix is materialised per call. Work is split across
//! output rows with rayon.

use anyhow::{bail, Result};
use candle_core::{DType, Storage, Tensor};
use rayon::prelude::*;
use std::sync::OnceLock;

use crate::weights::E2M1;

#[cfg(feature = "cuda")]
pub mod cuda;

/// Largest batch the fused kernels handle; bigger batches dequantize once and use the GEMM.
pub const MAX_FUSED_ROWS: usize = 16;

fn e4m3_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| std::array::from_fn(|b| float8::F8E4M3::from_bits(b as u8).to_f32()))
}

fn e2m1_pair_lut() -> &'static [[f32; 2]; 256] {
    static LUT: OnceLock<[[f32; 2]; 256]> = OnceLock::new();
    LUT.get_or_init(|| std::array::from_fn(|b| [E2M1[b & 15], E2M1[b >> 4]]))
}

/// Run `f` on the contiguous CPU bytes of `t` (u8 or f8e4m3 storage).
fn with_bytes<R>(t: &Tensor, f: impl FnOnce(&[u8]) -> Result<R>) -> Result<R> {
    let (st, layout) = t.storage_and_layout();
    let Storage::Cpu(cpu) = &*st else {
        bail!("fused kernel needs a CPU tensor")
    };
    let Some((a, b)) = layout.contiguous_offsets() else {
        bail!("fused kernel needs a contiguous tensor")
    };
    match t.dtype() {
        DType::U8 => f(&cpu.as_slice::<u8>()?[a..b]),
        DType::F8E4M3 => {
            let s = &cpu.as_slice::<float8::F8E4M3>()?[a..b];
            // SAFETY: F8E4M3 is a transparent one-byte wrapper
            let bytes = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len()) };
            f(bytes)
        }
        d => bail!("fused kernel: unexpected dtype {d:?}"),
    }
}

fn input_rows(x: &Tensor) -> Result<(usize, usize, Vec<f32>)> {
    let (t, k) = x.dims2()?;
    Ok((
        t,
        k,
        x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
    ))
}

/// Rows of output computed per rayon task.
const ROWS_PER_TASK: usize = 8;

/// `y[t, n] = sum_k x[t, k] * W[n, k]` with NVFP4 `W` (codes `[n, k/2]`, e4m3 scales `[n, k/16]`).
pub fn nvfp4_matmul(x: &Tensor, packed: &Tensor, scale: &Tensor, global: f32) -> Result<Tensor> {
    let (t, k, xs) = input_rows(x)?;
    let (n, half) = packed.dims2()?;
    if half * 2 != k || scale.dims2()? != (n, k / 16) || t > MAX_FUSED_ROWS {
        bail!(
            "nvfp4 fused kernel: shape mismatch x {:?} w {:?} s {:?}",
            x.dims(),
            packed.dims(),
            scale.dims()
        );
    }
    let nb = k / 16;
    let pair = e2m1_pair_lut();
    let s8 = e4m3_lut();
    let mut out = vec![0f32; n * t];
    with_bytes(packed, |codes| {
        with_bytes(scale, |scales| {
            out.par_chunks_mut(ROWS_PER_TASK * t)
                .enumerate()
                .for_each(|(chunk, dst)| {
                    let r0 = chunk * ROWS_PER_TASK;
                    for (ri, row_out) in dst.chunks_mut(t).enumerate() {
                        let r = r0 + ri;
                        let row_codes = &codes[r * half..(r + 1) * half];
                        let row_scales = &scales[r * nb..(r + 1) * nb];
                        let mut acc = [0f32; MAX_FUSED_ROWS];
                        for b in 0..nb {
                            let s = s8[row_scales[b] as usize];
                            let bc = &row_codes[b * 8..b * 8 + 8];
                            for ti in 0..t {
                                let xb = &xs[ti * k + b * 16..ti * k + b * 16 + 16];
                                let mut blk = 0f32;
                                for j in 0..8 {
                                    let [lo, hi] = pair[bc[j] as usize];
                                    blk += lo * xb[2 * j] + hi * xb[2 * j + 1];
                                }
                                acc[ti] += s * blk;
                            }
                        }
                        for ti in 0..t {
                            row_out[ti] = acc[ti] * global;
                        }
                    }
                });
            Ok(())
        })
    })?;
    // out is [n, t] (row-major per output feature); transpose to [t, n]
    Ok(Tensor::from_vec(out, (n, t), x.device())?
        .t()?
        .contiguous()?
        .to_dtype(x.dtype())?)
}

/// `y = x W^T` with FP8 e4m3 `W [n, k]` and f32 block scales `[ceil(n/b0), ceil(k/b1)]`
/// (or one per-tensor scale).
pub fn fp8_matmul(x: &Tensor, w: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let (t, k, xs) = input_rows(x)?;
    let (n, kw) = w.dims2()?;
    if kw != k || t > MAX_FUSED_ROWS {
        bail!(
            "fp8 fused kernel: shape mismatch x {:?} w {:?}",
            x.dims(),
            w.dims()
        );
    }
    let sc: Vec<f32> = scale.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let (sr, scols) = if sc.len() == 1 {
        (1, 1)
    } else {
        scale.dims2()?
    };
    let (b0, b1) = (n.div_ceil(sr), k.div_ceil(scols));
    let lut = e4m3_lut();
    let mut out = vec![0f32; n * t];
    with_bytes(w, |codes| {
        out.par_chunks_mut(ROWS_PER_TASK * t)
            .enumerate()
            .for_each(|(chunk, dst)| {
                let r0 = chunk * ROWS_PER_TASK;
                for (ri, row_out) in dst.chunks_mut(t).enumerate() {
                    let r = r0 + ri;
                    let row = &codes[r * k..(r + 1) * k];
                    let srow = &sc[(r / b0) * scols..(r / b0 + 1) * scols];
                    let mut acc = [0f32; MAX_FUSED_ROWS];
                    for (bi, s) in srow.iter().enumerate() {
                        let (c0, c1) = (bi * b1, ((bi + 1) * b1).min(k));
                        for ti in 0..t {
                            let xb = &xs[ti * k + c0..ti * k + c1];
                            let blk: f32 = row[c0..c1]
                                .iter()
                                .zip(xb)
                                .map(|(c, xv)| lut[*c as usize] * xv)
                                .sum();
                            acc[ti] += s * blk;
                        }
                    }
                    row_out[..t].copy_from_slice(&acc[..t]);
                }
            });
        Ok(())
    })?;
    Ok(Tensor::from_vec(out, (n, t), x.device())?
        .t()?
        .contiguous()?
        .to_dtype(x.dtype())?)
}

/// Decode NVFP4 to a dense f32 `[n, k]` matrix on the CPU.
pub fn nvfp4_dequant(packed: &Tensor, scale: &Tensor, global: f32) -> Result<Tensor> {
    let (n, half) = packed.dims2()?;
    let k = half * 2;
    let nb = scale.dim(1)?;
    if scale.dim(0)? != n || k % nb != 0 {
        bail!(
            "nvfp4 scale shape {:?} does not match weight {n}x{k}",
            scale.dims()
        );
    }
    let per = k / nb;
    let (pair, s8) = (e2m1_pair_lut(), e4m3_lut());
    let mut out = vec![0f32; n * k];
    with_bytes(packed, |codes| {
        with_bytes(scale, |scales| {
            out.par_chunks_mut(k).enumerate().for_each(|(r, row)| {
                for (c, v) in row.chunks_mut(2).enumerate() {
                    // (code * scale) * global, the generic decoder's rounding order
                    let s = s8[scales[r * nb + 2 * c / per] as usize];
                    let [lo, hi] = pair[codes[r * half + c] as usize];
                    v[0] = lo * s * global;
                    v[1] = hi * s * global;
                }
            });
            Ok(())
        })
    })?;
    Ok(Tensor::from_vec(out, (n, k), packed.device())?)
}

/// Decode FP8 e4m3 with block (or per-tensor) f32 scales to a dense f32 `[n, k]` matrix on the CPU.
pub fn fp8_dequant(w: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let (n, k) = w.dims2()?;
    let sc: Vec<f32> = scale.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let (sr, scols) = if sc.len() == 1 {
        (1, 1)
    } else {
        scale.dims2()?
    };
    let (b0, b1) = (n.div_ceil(sr), k.div_ceil(scols));
    let lut = e4m3_lut();
    let mut out = vec![0f32; n * k];
    with_bytes(w, |codes| {
        out.par_chunks_mut(k).enumerate().for_each(|(r, row)| {
            for (c, v) in row.iter_mut().enumerate() {
                *v = lut[codes[r * k + c] as usize] * sc[(r / b0) * scols + c / b1];
            }
        });
        Ok(())
    })?;
    Ok(Tensor::from_vec(out, (n, k), w.device())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{dequant_fp8, dequant_nvfp4, Nvfp4};
    use candle_core::Device;

    fn close(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn fused_kernels_match_dequant_gemm() {
        let dev = Device::Cpu;
        let (n, k) = (37, 96);
        let packed = Tensor::rand(0f32, 255.0, (n, k / 2), &dev)
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap();
        let scale = Tensor::rand(0.25f32, 2.0, (n, k / 16), &dev)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let q = Nvfp4 {
            packed: packed.clone(),
            scale: scale.clone(),
            global: 0.3,
        };
        let w8 = Tensor::rand(-2f32, 2.0, (n, k), &dev)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let s8 = Tensor::rand(0.5f32, 1.5, (n.div_ceil(16), k.div_ceil(32)), &dev).unwrap();
        for t in [1, 3, 16] {
            let x = Tensor::rand(-1f32, 1.0, (t, k), &dev).unwrap();
            let want = x
                .matmul(&dequant_nvfp4(&q, DType::F32).unwrap().t().unwrap())
                .unwrap();
            let got = nvfp4_matmul(&x, &packed, &scale, 0.3).unwrap();
            assert!(
                close(&want, &got) < 1e-3,
                "nvfp4 t={t}: {}",
                close(&want, &got)
            );
            let want = x
                .matmul(&dequant_fp8(&w8, &s8, DType::F32).unwrap().t().unwrap())
                .unwrap();
            let got = fp8_matmul(&x, &w8, &s8).unwrap();
            assert!(
                close(&want, &got) < 1e-3,
                "fp8 t={t}: {}",
                close(&want, &got)
            );
        }
        // CPU dequant kernels agree exactly with the generic tensor-op decoders
        let generic = |q: &Nvfp4| {
            let q2 = Nvfp4 {
                packed: q.packed.clone(),
                scale: q.scale.clone(),
                global: q.global,
            };
            crate::weights::dequant_nvfp4_generic(&q2, DType::F32).unwrap()
        };
        assert_eq!(
            close(&nvfp4_dequant(&packed, &scale, 0.3).unwrap(), &generic(&q)),
            0.0
        );
        assert_eq!(
            close(
                &fp8_dequant(&w8, &s8).unwrap(),
                &crate::weights::dequant_fp8_generic(&w8, &s8, DType::F32).unwrap()
            ),
            0.0
        );
    }
}
