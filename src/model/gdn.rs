// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Gated DeltaNet linear attention (Qwen3.5 / Qwen3.6 / Qwen3-Next hybrid layers).
//!
//! Each sequence owns a recurrent state slot holding the last `kernel - 1` conv inputs and
//! the `[heads, k_dim, v_dim]` delta-rule state. Prefill uses the chunked (UT-transform)
//! algorithm, decode the per-token recurrence; both follow the HF reference exactly.

use super::attention::DeviceBatch;
use super::layers::{load_row_parallel, Placement};
use crate::config::{LinearAttnConfig, ModelConfig};
use crate::weights::WeightSource;
use anyhow::{bail, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::Linear;

const CHUNK: usize = 64;
const MASK_VALUE: f64 = -1e30;

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
static FUSED_GDN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// The fused GDN decode kernels are on unless `CARBON_FUSED_GDN=0` or they failed once.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn fused_gdn_enabled() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var("CARBON_FUSED_GDN").map_or(true, |v| v != "0"))
        && FUSED_GDN.load(std::sync::atomic::Ordering::Relaxed)
}

/// Per-layer recurrent state pool: `conv [slots, (kernel-1) * conv_dim]` (model dtype) and
/// `ssm [slots, heads * k_dim * v_dim]` (f32).
pub struct LinearState {
    pub conv: Tensor,
    pub ssm: Tensor,
}

pub struct GatedDeltaNet {
    in_qkv: Linear,
    in_z: Linear,
    in_b: Linear,
    in_a: Linear,
    out: Linear,
    /// Conv taps, one `[1, 1, conv_dim]` row per kernel position.
    conv_taps: Vec<Tensor>,
    /// The same taps stacked `[kernel, conv_dim]` for carbon's fused decode kernels.
    #[cfg(feature = "cuda")]
    taps_stacked: Tensor,
    dt_bias: Tensor,
    a_neg: Tensor,
    norm_w: Tensor,
    eps: f64,
    /// Output gate `sigmoid(z)` instead of `silu(z)` (Qwen3.8).
    gate_sigmoid: bool,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
    kernel: usize,
}

/// Load `name` and keep this rank's share of each consecutive segment along dim 0.
fn load_segments(
    src: &dyn WeightSource,
    name: &str,
    segments: &[usize],
    p: &Placement,
) -> Result<Tensor> {
    let t = src.load_cpu(name)?;
    let t = if p.world > 1 {
        let mut parts = Vec::with_capacity(segments.len());
        let mut off = 0;
        for &seg in segments {
            let chunk = seg / p.world;
            parts.push(t.narrow(0, off + p.rank * chunk, chunk)?);
            off += seg;
        }
        Tensor::cat(&parts, 0)?
    } else {
        t
    };
    Ok(t.to_dtype(p.dtype)?.to_device(&p.device)?)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    // log(1 + e^x) = relu(x) + log(1 + e^-|x|), stable for large |x|
    Ok((x.relu()? + ((x.abs()?.neg()?.exp()? + 1.0)?.log()?))?)
}

fn l2norm(x: &Tensor) -> Result<Tensor> {
    let inv = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?.recip()?;
    Ok(x.broadcast_mul(&inv)?)
}

impl LinearState {
    /// Copy the conv and recurrent state of slot `src` into slot `dst`.
    pub fn copy_slot(&self, src: u32, dst: u32) -> Result<()> {
        let dev = self.ssm.device();
        let s = Tensor::new(&[src], dev)?;
        let d = Tensor::new(&[dst], dev)?;
        write_rows(&self.conv, &d, &self.conv.index_select(&s, 0)?)?;
        write_rows(&self.ssm, &d, &self.ssm.index_select(&s, 0)?)?;
        Ok(())
    }
}

impl LinearState {
    /// Copy the states in slots `src` (u32 `[n]`) to slots `dst` (device-side, capturable).
    pub fn copy_rows(&self, src: &Tensor, dst: &Tensor) -> Result<()> {
        write_rows(&self.conv, dst, &self.conv.index_select(src, 0)?)?;
        write_rows(&self.ssm, dst, &self.ssm.index_select(src, 0)?)?;
        Ok(())
    }
}

fn write_rows(dst: &Tensor, idx: &Tensor, rows: &Tensor) -> Result<()> {
    let (n, w) = rows.dims2()?;
    let idx = idx.reshape((n, 1))?.broadcast_as((n, w))?.contiguous()?;
    dst.scatter_set(&idx, &rows.to_dtype(dst.dtype())?.contiguous()?, 0)?;
    Ok(())
}

impl GatedDeltaNet {
    pub fn load(
        src: &dyn WeightSource,
        cfg: &ModelConfig,
        prefix: &str,
        p: &Placement,
    ) -> Result<Self> {
        let lc = cfg.linear_attn.as_ref().expect("linear attention config");
        if !lc.num_k_heads.is_multiple_of(p.world) || !lc.num_v_heads.is_multiple_of(p.world) {
            bail!(
                "tensor-parallel size {} must divide linear-attention heads ({} k, {} v)",
                p.world,
                lc.num_k_heads,
                lc.num_v_heads
            );
        }
        let (kd, vd) = (
            lc.num_k_heads * lc.k_head_dim,
            lc.num_v_heads * lc.v_head_dim,
        );
        let qkv_segments = [kd, kd, vd];
        let w = load_segments(
            src,
            &format!("{prefix}.in_proj_qkv.weight"),
            &qkv_segments,
            p,
        )?;
        let in_qkv = Linear::new(w, None);
        let lin = |n: &str, seg: usize| -> Result<Linear> {
            Ok(Linear::new(
                load_segments(src, &format!("{prefix}.{n}.weight"), &[seg], p)?,
                None,
            ))
        };
        let in_z = lin("in_proj_z", vd)?;
        let in_b = lin("in_proj_b", lc.num_v_heads)?;
        let in_a = lin("in_proj_a", lc.num_v_heads)?;
        let conv = load_segments(src, &format!("{prefix}.conv1d.weight"), &qkv_segments, p)?;
        let c_local = conv.dim(0)?;
        let conv = conv.reshape((c_local, lc.conv_kernel))?;
        let conv_taps = (0..lc.conv_kernel)
            .map(|j| {
                Ok(conv
                    .narrow(1, j, 1)?
                    .reshape((1, 1, c_local))?
                    .contiguous()?)
            })
            .collect::<Result<Vec<_>>>()?;
        let f32p = Placement {
            dtype: DType::F32,
            ..p.clone()
        };
        let dt_bias = load_segments(src, &format!("{prefix}.dt_bias"), &[lc.num_v_heads], &f32p)?;
        let a_neg = load_segments(src, &format!("{prefix}.A_log"), &[lc.num_v_heads], &f32p)?
            .exp()?
            .neg()?;
        Ok(Self {
            in_qkv,
            in_z,
            in_b,
            in_a,
            out: load_row_parallel(src, &format!("{prefix}.out_proj"), p)?,
            #[cfg(feature = "cuda")]
            taps_stacked: conv.t()?.contiguous()?,
            conv_taps,
            dt_bias,
            a_neg,
            norm_w: src.get(&format!("{prefix}.norm.weight"), &p.device, p.dtype)?,
            eps: cfg.rms_norm_eps,
            gate_sigmoid: cfg.gdn_gate_sigmoid,
            hk: lc.num_k_heads / p.world,
            hv: lc.num_v_heads / p.world,
            dk: lc.k_head_dim,
            dv: lc.v_head_dim,
            kernel: lc.conv_kernel,
        })
    }

    /// Decode steps of this layer run on carbon's fused CUDA kernels.
    pub fn fused_decode_ready(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            fused_gdn_enabled() && crate::kernels::cuda::gdn_decode_supported(self.dk, self.dv)
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    fn conv_dim(&self) -> usize {
        2 * self.hk * self.dk + self.hv * self.dv
    }

    pub fn new_state(
        lc: &LinearAttnConfig,
        world: usize,
        slots: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<LinearState> {
        let conv_dim = lc.conv_dim() / world;
        let ssm = lc.num_v_heads / world * lc.k_head_dim * lc.v_head_dim;
        Ok(LinearState {
            conv: Tensor::zeros((slots, (lc.conv_kernel - 1) * conv_dim), dtype, device)?,
            ssm: Tensor::zeros((slots, ssm), DType::F32, device)?,
        })
    }

    /// `x: [T, hidden]` -> `[T, hidden]` (partial sum under tensor parallelism).
    pub fn forward(&self, x: &Tensor, b: &DeviceBatch, st: &LinearState) -> Result<Tensor> {
        let qkv = self.in_qkv.forward(x)?;
        let z = self.in_z.forward(x)?;
        let braw = self.in_b.forward(x)?;
        let araw = self.in_a.forward(x)?;

        // multi-token decode: the fused step once per token position, in order
        let multi = b.per_substep(&[&qkv, &z, &braw, &araw], |r, sub| {
            if let Some(prev) = &sub.prev_idx {
                st.copy_rows(prev, &sub.state_idx)?;
            }
            Ok(self.decode_fused(&r[0], &r[1], &r[2], &r[3], &sub.state_idx, &Some(sub.fresh.clone()), st))
        })?;
        if let Some(y) = multi {
            return Ok(self.out.forward(&y)?);
        }
        let decode = b.seqs.iter().all(|s| s.q_len == 1);
        if decode {
            if let Some(y) = self.decode_fused(&qkv, &z, &braw, &araw, &b.state_idx, &b.fresh, st) {
                return Ok(self.out.forward(&y)?);
            }
            let y = self.decode_generic(&qkv, &z, &braw, &araw, &b.state_idx, &b.starts, st)?;
            return Ok(self.out.forward(&y)?);
        }
        let (beta, g) = self.beta_g(&braw, &araw)?;
        let mut outs = Vec::with_capacity(b.seqs.len());
        for (i, s) in b.seqs.iter().enumerate() {
            let idx = b.state_idx.narrow(0, i, 1)?;
            let r = |x: &Tensor| x.narrow(0, s.q_start, s.q_len);
            outs.push(self.seq_forward(&r(&qkv)?, &r(&beta)?, &r(&g)?, &idx, b.starts[i], st)?);
        }
        let core = if outs.len() == 1 {
            outs.pop().unwrap()
        } else {
            Tensor::cat(&outs, 0)?
        }; // [T, hv, dv] f32
        Ok(self.out.forward(&self.gated_norm(&core, &z)?)?)
    }

    /// Write strength `beta = sigmoid(b)` and log-decay `g = -exp(A_log) * softplus(a + dt_bias)`.
    fn beta_g(&self, braw: &Tensor, araw: &Tensor) -> Result<(Tensor, Tensor)> {
        let beta = candle_nn::ops::sigmoid(&braw.to_dtype(DType::F32)?)?;
        let a = araw.to_dtype(DType::F32)?.broadcast_add(&self.dt_bias)?;
        let g = softplus(&a)?.broadcast_mul(&self.a_neg)?;
        Ok((beta, g))
    }

    /// Gated RMSNorm over each head's values: `core [T, hv, dv]` f32 -> `[T, hv*dv]` model dtype.
    fn gated_norm(&self, core: &Tensor, z: &Tensor) -> Result<Tensor> {
        let t = core.dim(0)?;
        let dtype = z.dtype();
        let var = core.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = core
            .broadcast_mul(&(var + self.eps)?.sqrt()?.recip()?)?
            .to_dtype(dtype)?;
        let normed = normed.broadcast_mul(&self.norm_w)?.to_dtype(DType::F32)?;
        let z = z.to_dtype(DType::F32)?.reshape((t, self.hv, self.dv))?;
        let gate = if self.gate_sigmoid {
            candle_nn::ops::sigmoid(&z)?
        } else {
            candle_nn::ops::silu(&z)?
        };
        Ok((normed * gate)?
            .to_dtype(dtype)?
            .reshape((t, self.hv * self.dv))?)
    }

    /// Decode step with candle ops; returns the input of the output projection.
    #[allow(clippy::too_many_arguments)]
    fn decode_generic(
        &self,
        qkv: &Tensor,
        z: &Tensor,
        braw: &Tensor,
        araw: &Tensor,
        state_idx: &Tensor,
        starts: &[bool],
        st: &LinearState,
    ) -> Result<Tensor> {
        let (beta, g) = self.beta_g(braw, araw)?;
        let core = self.step_batched(qkv, &beta, &g, state_idx, starts, st)?;
        self.gated_norm(&core, z)
    }

    /// Decode step in carbon's two CUDA kernels (conv, then delta rule + gated norm), updating
    /// the state pools in place. `None` when unavailable; the caller then uses the generic path.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn decode_fused(
        &self,
        qkv: &Tensor,
        z: &Tensor,
        braw: &Tensor,
        araw: &Tensor,
        state_idx: &Tensor,
        fresh: &Option<Tensor>,
        st: &LinearState,
    ) -> Option<Tensor> {
        use crate::kernels::cuda;
        let fresh = fresh.as_ref()?;
        if !qkv.device().is_cuda()
            || !cuda::gdn_decode_supported(self.dk, self.dv)
            || !fused_gdn_enabled()
        {
            return None;
        }
        let layer = cuda::GdnLayer {
            taps: &self.taps_stacked,
            dt_bias: &self.dt_bias,
            a_neg: &self.a_neg,
            norm_w: &self.norm_w,
            conv_pool: &st.conv,
            ssm_pool: &st.ssm,
            hk: self.hk,
            hv: self.hv,
            dk: self.dk,
            dv: self.dv,
            eps: self.eps as f32,
            gate_sigmoid: self.gate_sigmoid,
        };
        match cuda::gdn_decode(&layer, qkv, z, braw, araw, state_idx, fresh) {
            Ok(y) => Some(y),
            Err(e) => {
                if FUSED_GDN.swap(false, std::sync::atomic::Ordering::Relaxed) {
                    tracing::warn!("fused gdn decode disabled, using the generic path: {e:#}");
                }
                None
            }
        }
    }

    /// The whole-sequence recurrence in carbon's CUDA kernel (one warp per state column).
    #[cfg(feature = "cuda")]
    fn sequence_fused(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        beta: &Tensor,
        g: &Tensor,
        s0: &Tensor,
    ) -> Option<(Tensor, Tensor)> {
        use crate::kernels::cuda;
        if !q.device().is_cuda() || !cuda::gdn_sequence_supported(self.dk) || !fused_gdn_enabled() {
            return None;
        }
        match cuda::gdn_sequence(q, k, v, beta, g, s0) {
            Ok(r) => Some(r),
            Err(e) => {
                if FUSED_GDN.swap(false, std::sync::atomic::Ordering::Relaxed) {
                    tracing::warn!("fused gdn kernels disabled, using the generic path: {e:#}");
                }
                None
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn sequence_fused(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Option<(Tensor, Tensor)> {
        None
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    fn decode_fused(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Option<Tensor>,
        _: &LinearState,
    ) -> Option<Tensor> {
        None
    }

    /// Causal depthwise conv over `prev ++ x` (`[B, kernel-1, C]`, `[B, T, C]`) followed by SiLU.
    /// Returns the activations `[B, T, C]` and the new conv state `[B, kernel-1, C]`.
    fn conv(&self, prev: &Tensor, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let t = x.dim(1)?;
        let full = Tensor::cat(&[prev, x], 1)?;
        let mut acc = full.narrow(1, 0, t)?.broadcast_mul(&self.conv_taps[0])?;
        for (j, tap) in self.conv_taps.iter().enumerate().skip(1) {
            acc = (acc + full.narrow(1, j, t)?.broadcast_mul(tap)?)?;
        }
        let state = full.narrow(1, t, self.kernel - 1)?;
        Ok((candle_nn::ops::silu(&acc)?, state))
    }

    /// Split conv output `[B, T, C]` into l2-normalised, head-expanded q/k and v,
    /// each `[B, hv, T, d]` in f32.
    fn split_qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, t, _) = x.dims3()?;
        let (kd, r) = (self.hk * self.dk, self.hv / self.hk);
        let x = x.to_dtype(DType::F32)?;
        let expand = |y: Tensor| -> Result<Tensor> {
            let y = l2norm(&y.reshape((b, t, self.hk, 1, self.dk))?)?;
            Ok(y.broadcast_as((b, t, self.hk, r, self.dk))?
                .reshape((b, t, self.hv, self.dk))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let q = (expand(x.narrow(2, 0, kd)?)? * (self.dk as f64).powf(-0.5))?;
        let k = expand(x.narrow(2, kd, kd)?)?;
        let v = x
            .narrow(2, 2 * kd, self.hv * self.dv)?
            .reshape((b, t, self.hv, self.dv))?
            .transpose(1, 2)?
            .contiguous()?;
        Ok((q, k, v))
    }

    fn load_state(
        &self,
        idx: &Tensor,
        keep: Option<&Tensor>,
        st: &LinearState,
    ) -> Result<(Tensor, Tensor)> {
        let n = idx.dim(0)?;
        let conv = st
            .conv
            .index_select(idx, 0)?
            .reshape((n, self.kernel - 1, self.conv_dim()))?;
        let ssm = st
            .ssm
            .index_select(idx, 0)?
            .reshape((n, self.hv, self.dk, self.dv))?;
        Ok(match keep {
            Some(k) => (
                conv.broadcast_mul(&k.to_dtype(conv.dtype())?.reshape((n, 1, 1))?)?,
                ssm.broadcast_mul(&k.reshape((n, 1, 1, 1))?)?,
            ),
            None => (conv, ssm),
        })
    }

    /// Decode: one token per sequence, all sequences at once.
    fn step_batched(
        &self,
        qkv: &Tensor,
        beta: &Tensor,
        g: &Tensor,
        state_idx: &Tensor,
        starts: &[bool],
        st: &LinearState,
    ) -> Result<Tensor> {
        let n = starts.len();
        let keep: Vec<f32> = starts.iter().map(|&s| if s { 0.0 } else { 1.0 }).collect();
        let keep = Tensor::from_vec(keep, n, qkv.device())?;
        let (prev, s0) = self.load_state(state_idx, Some(&keep), st)?;
        let (act, conv_state) = self.conv(&prev, &qkv.unsqueeze(1)?)?;
        let (q, k, v) = self.split_qkv(&act)?; // [n, hv, 1, d]
        let (o, s1) = recurrent(
            &q,
            &k,
            &v,
            &beta.reshape((n, self.hv, 1))?,
            &g.reshape((n, self.hv, 1))?,
            s0,
        )?;
        write_rows(&st.conv, state_idx, &conv_state.reshape((n, ()))?)?;
        write_rows(&st.ssm, state_idx, &s1.reshape((n, ()))?)?;
        Ok(o.squeeze(2)?) // [n, hv, dv]
    }

    fn seq_forward(
        &self,
        qkv: &Tensor,
        beta: &Tensor,
        g: &Tensor,
        idx: &Tensor,
        start: bool,
        st: &LinearState,
    ) -> Result<Tensor> {
        let t = qkv.dim(0)?;
        let (prev, s0) = if start {
            (
                Tensor::zeros(
                    (1, self.kernel - 1, self.conv_dim()),
                    qkv.dtype(),
                    qkv.device(),
                )?,
                Tensor::zeros((1, self.hv, self.dk, self.dv), DType::F32, qkv.device())?,
            )
        } else {
            self.load_state(idx, None, st)?
        };
        let (act, conv_state) = self.conv(&prev.to_dtype(qkv.dtype())?, &qkv.unsqueeze(0)?)?;
        let (q, k, v) = self.split_qkv(&act)?;
        let beta = beta.t()?.unsqueeze(0)?.contiguous()?; // [1, hv, T]
        let g = g.t()?.unsqueeze(0)?.contiguous()?;
        let (o, s1) = match self.sequence_fused(&q, &k, &v, &beta, &g, &s0) {
            Some(r) => r,
            None if t == 1 => recurrent(&q, &k, &v, &beta, &g, s0)?,
            None => chunked(&q, &k, &v, &beta, &g, s0)?,
        };
        write_rows(&st.conv, idx, &conv_state.reshape((1, ()))?)?;
        write_rows(&st.ssm, idx, &s1.reshape((1, ()))?)?;
        Ok(o.squeeze(0)?.transpose(0, 1)?.contiguous()?) // [T, hv, dv]
    }
}

/// Token-by-token gated delta rule. `q,k: [B,H,T,Dk]`, `v: [B,H,T,Dv]`, `beta,g: [B,H,T]`,
/// `s: [B,H,Dk,Dv]`; returns `([B,H,T,Dv], s)`.
pub fn recurrent(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    mut s: Tensor,
) -> Result<(Tensor, Tensor)> {
    let t = q.dim(2)?;
    let mut outs = Vec::with_capacity(t);
    for i in 0..t {
        let qt = q.narrow(2, i, 1)?; // [B,H,1,Dk]
        let kt = k.narrow(2, i, 1)?;
        let vt = v.narrow(2, i, 1)?; // [B,H,1,Dv]
        let decay = g.narrow(2, i, 1)?.exp()?.unsqueeze(3)?; // [B,H,1,1]
        let bt = beta.narrow(2, i, 1)?.unsqueeze(3)?;
        s = s.broadcast_mul(&decay)?;
        let mem = kt.matmul(&s)?; // [B,H,1,Dv]
        let delta = (vt - mem)?.broadcast_mul(&bt)?;
        s = (s + kt.transpose(2, 3)?.contiguous()?.matmul(&delta)?)?;
        outs.push(qt.matmul(&s)?);
    }
    Ok((Tensor::cat(&outs, 2)?, s))
}

/// Chunked gated delta rule (UT transform per 64-token chunk, forward substitution instead of
/// a triangular solve). Shapes as in [`recurrent`] with `B = 1`.
pub fn chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    s0: Tensor,
) -> Result<(Tensor, Tensor)> {
    let (bsz, h, t, dk) = q.dims4()?;
    let dv = v.dim(3)?;
    let bh = bsz * h;
    let dev = q.device().clone();
    let pad = (CHUNK - t % CHUNK) % CHUNK;
    // flatten batch x heads so every matmul has at most two batch dims
    let padded = |x: &Tensor, d: Option<usize>| -> Result<Tensor> {
        let x = match d {
            Some(d) => x.reshape((bh, t, d))?,
            None => x.reshape((bh, t))?,
        };
        if pad == 0 {
            return Ok(x);
        }
        let mut shape = x.dims().to_vec();
        shape[1] = pad;
        Ok(Tensor::cat(
            &[x.clone(), Tensor::zeros(shape, x.dtype(), &dev)?],
            1,
        )?)
    };
    let (q, k, v) = (
        padded(q, Some(dk))?,
        padded(k, Some(dk))?,
        padded(v, Some(dv))?,
    );
    let (beta, g) = (padded(beta, None)?, padded(g, None)?);
    let n = (t + pad) / CHUNK;
    let c = CHUNK;
    let k_beta = k.broadcast_mul(&beta.unsqueeze(2)?)?;
    let v_beta = v.broadcast_mul(&beta.unsqueeze(2)?)?;
    let chunks = |x: Tensor, d: usize| x.reshape((bh, n, c, d));
    let (q, k, k_beta, v_beta) = (
        chunks(q, dk)?,
        chunks(k, dk)?,
        chunks(k_beta, dk)?,
        chunks(v_beta, dv)?,
    );
    let cum = g.reshape((bh, n, c))?.cumsum(2)?;

    // strictly-upper (j > i) and strictly-lower (j < i) masks
    let i_idx = Tensor::arange(0u32, c as u32, &dev)?.reshape((c, 1))?;
    let j_idx = Tensor::arange(0u32, c as u32, &dev)?.reshape((1, c))?;
    let upper = j_idx.broadcast_gt(&i_idx)?.to_dtype(DType::F32)?;
    let lower = j_idx.broadcast_lt(&i_idx)?.to_dtype(DType::F32)?;
    let pairwise = cum.unsqueeze(3)?.broadcast_sub(&cum.unsqueeze(2)?)?;
    let pairwise = pairwise.broadcast_add(&(upper * MASK_VALUE)?)?.exp()?;

    let kt = k.transpose(2, 3)?.contiguous()?;
    let ut = k_beta.matmul(&kt)?.mul(&pairwise)?;
    let intra = q.matmul(&kt)?.mul(&pairwise)?;
    let decayed_k_beta = k_beta.broadcast_mul(&cum.exp()?.unsqueeze(3)?)?;

    // (I + strict_lower(ut))^-1 by forward substitution, all chunks and heads at once
    let a = ut.broadcast_mul(&lower)?.neg()?;
    let mut acc = a.narrow(2, 0, 1)?;
    for i in 1..c {
        let row = a.narrow(2, i, 1)?.narrow(3, 0, i)?.contiguous()?;
        let sub = acc.narrow(3, 0, i)?.contiguous()?;
        let new = (&row + row.matmul(&sub)?)?;
        let new = Tensor::cat(
            &[new, Tensor::zeros((bh, n, 1, c - i), DType::F32, &dev)?],
            3,
        )?;
        acc = Tensor::cat(&[acc, new], 2)?;
    }
    let tri = acc.broadcast_add(&Tensor::eye(c, DType::F32, &dev)?)?;
    let new_values = tri.matmul(&v_beta)?;
    let k_cumdecay = tri.matmul(&decayed_k_beta)?;

    let last = cum.narrow(2, c - 1, 1)?; // [bh, n, 1]
    let q = q.broadcast_mul(&cum.exp()?.unsqueeze(3)?)?;
    let k = k.broadcast_mul(&last.broadcast_sub(&cum)?.exp()?.unsqueeze(3)?)?;
    let chunk_decay = last.exp()?;

    let mut s = s0.reshape((bh, dk, dv))?;
    let mut outs = Vec::with_capacity(n);
    for ci in 0..n {
        let at =
            |x: &Tensor| -> Result<Tensor> { Ok(x.narrow(1, ci, 1)?.squeeze(1)?.contiguous()?) };
        let v_new = (at(&new_values)? - at(&k_cumdecay)?.matmul(&s)?)?;
        let inter = at(&q)?.matmul(&s)?;
        outs.push((inter + at(&intra)?.matmul(&v_new)?)?);
        let decay = at(&chunk_decay)?.unsqueeze(2)?; // [bh, 1, 1]
        s = (s.broadcast_mul(&decay)? + at(&k)?.transpose(1, 2)?.contiguous()?.matmul(&v_new)?)?;
    }
    let out = Tensor::cat(&outs, 1)?
        .narrow(1, 0, t)?
        .reshape((bsz, h, t, dv))?;
    Ok((out, s.reshape((bsz, h, dk, dv))?))
}

/// Run the generic and the fused CUDA decode step side by side on a random layer for a few
/// steps (fresh and continuing sequences, GQA heads) and return the worst relative error of the
/// outputs and of the recurrent state.
#[cfg(feature = "cuda")]
pub fn fused_decode_selftest(device: &Device, dtype: DType) -> Result<(f32, f32)> {
    let (hk, hv, dk, dv, kernel, slots) = (2usize, 4usize, 128usize, 128usize, 4usize, 6usize);
    let c = 2 * hk * dk + hv * dv;
    let r = |shape: &[usize], s: f64| -> Result<Tensor> {
        Ok((Tensor::randn(0f32, 1.0, shape, device)? * s)?.to_dtype(dtype)?)
    };
    let conv = r(&[c, kernel], 0.5)?;
    let lin = || -> Result<Linear> { Ok(Linear::new(r(&[1, 1], 1.0)?, None)) };
    let layer = GatedDeltaNet {
        in_qkv: lin()?,
        in_z: lin()?,
        in_b: lin()?,
        in_a: lin()?,
        out: lin()?,
        conv_taps: (0..kernel)
            .map(|j| Ok(conv.narrow(1, j, 1)?.reshape((1, 1, c))?.contiguous()?))
            .collect::<Result<Vec<_>>>()?,
        taps_stacked: conv.t()?.contiguous()?,
        dt_bias: Tensor::randn(0f32, 1.0, hv, device)?,
        a_neg: Tensor::randn(0f32, 0.5, hv, device)?.exp()?.neg()?,
        norm_w: r(&[dv], 1.0)?,
        eps: 1e-6,
        gate_sigmoid: false,
        hk,
        hv,
        dk,
        dv,
        kernel,
    };
    let lc = LinearAttnConfig {
        num_k_heads: hk,
        num_v_heads: hv,
        k_head_dim: dk,
        v_head_dim: dv,
        conv_kernel: kernel,
    };
    let a = GatedDeltaNet::new_state(&lc, 1, slots, dtype, device)?;
    a.conv
        .slice_set(&r(&[slots, (kernel - 1) * c], 1.0)?, 0, 0)?;
    a.ssm.slice_set(
        &(Tensor::randn(0f32, 0.1, (slots, hv * dk * dv), device)?),
        0,
        0,
    )?;
    let b = LinearState {
        conv: a.conv.copy()?,
        ssm: a.ssm.copy()?,
    };
    let idx = Tensor::new(&[4u32, 1, 3], device)?;
    let (mut worst_y, mut worst_s) = (0f32, 0f32);
    let rel = |x: &Tensor, y: &Tensor| -> Result<f32> {
        let x = x.to_dtype(DType::F32)?;
        let y = y.to_dtype(DType::F32)?;
        let d = (&x - &y)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let e = d / (y.abs()?.max_all()?.to_scalar::<f32>()? + 1e-6);
        Ok(if e.is_nan() { f32::INFINITY } else { e })
    };
    for step in 0..3 {
        let starts = [step == 0, false, step == 1];
        let fresh = Tensor::new(&starts.map(u32::from), device)?;
        let (qkv, z, braw, araw) = (
            r(&[3, c], 1.0)?,
            r(&[3, hv * dv], 1.0)?,
            r(&[3, hv], 1.0)?,
            r(&[3, hv], 1.0)?,
        );
        let want = layer.decode_generic(&qkv, &z, &braw, &araw, &idx, &starts, &a)?;
        let got = layer
            .decode_fused(&qkv, &z, &braw, &araw, &idx, &Some(fresh), &b)
            .ok_or_else(|| {
                anyhow::anyhow!("fused gdn decode unavailable (see the warning above)")
            })?;
        worst_y = worst_y.max(rel(&got, &want)?);
        worst_s = worst_s
            .max(rel(&b.ssm, &a.ssm)?)
            .max(rel(&b.conv, &a.conv)?);
    }
    Ok((worst_y, worst_s))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand(shape: &[usize], seed: u64) -> Tensor {
        use rand::{Rng, SeedableRng};
        let mut r = rand::rngs::StdRng::seed_from_u64(seed);
        let n: usize = shape.iter().product();
        Tensor::from_vec(
            (0..n).map(|_| r.gen::<f32>() - 0.5).collect::<Vec<_>>(),
            shape,
            &Device::Cpu,
        )
        .unwrap()
    }

    #[test]
    fn chunked_matches_recurrent() {
        let (h, t, dk, dv) = (3, 150, 8, 6);
        let q = l2norm(&rand(&[1, h, t, dk], 1)).unwrap();
        let k = l2norm(&rand(&[1, h, t, dk], 2)).unwrap();
        let v = rand(&[1, h, t, dv], 3);
        let beta = (rand(&[1, h, t], 4) + 0.5).unwrap();
        let g = (rand(&[1, h, t], 5) - 0.5)
            .unwrap()
            .clamp(-1.0, 0.0)
            .unwrap();
        let s0 = rand(&[1, h, dk, dv], 6);
        let (o1, s1) = recurrent(&q, &k, &v, &beta, &g, s0.clone()).unwrap();
        let (o2, s2) = chunked(&q, &k, &v, &beta, &g, s0).unwrap();
        let d = |a: &Tensor, b: &Tensor| {
            (a - b)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };
        assert!(d(&o1, &o2) < 1e-4, "out diff {}", d(&o1, &o2));
        assert!(d(&s1, &s2) < 1e-4, "state diff {}", d(&s1, &s2));
    }
}
