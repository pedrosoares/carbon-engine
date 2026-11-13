// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! KV cache: slot allocator, radix prefix cache and the per-device KV tensors.

pub mod allocator;
pub mod radix;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub use allocator::SlotAllocator;
pub use radix::RadixCache;

/// How KV pools on CUDA devices store keys and values (`--kv-dtype`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvFormat {
    /// The activation dtype.
    #[default]
    Model,
    /// int8 with an f32 scale per vector: half the memory and attention traffic of bf16.
    Q8,
    /// TurboQuant at 4 bits: vectors rotated by a random orthogonal matrix, each coordinate an
    /// index into the 16-level Lloyd-Max codebook of a Gaussian, an f32 scale per vector. A
    /// quarter of bf16.
    Turbo4,
    /// RotorQuant at 4 bits: the storage of `Turbo4`, the rotation made of independent random
    /// 3D rotations (Clifford rotors) on consecutive coordinate triples.
    Rotor4,
    /// RotorQuant's planar variant: 2D (Givens) blocks.
    Planar4,
    /// RotorQuant's isoclinic variant: 4D (quaternion pair) blocks.
    Iso4,
}

impl KvFormat {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "model" | "auto" => Self::Model,
            "q8" | "int8" => Self::Q8,
            "turbo4" | "turboquant" | "tq4" => Self::Turbo4,
            "rotor4" | "rotorquant" | "rq4" => Self::Rotor4,
            "planar4" | "planarquant" => Self::Planar4,
            "iso4" | "isoquant" => Self::Iso4,
            _ => return None,
        })
    }

    /// Bytes of one stored vector of `head_dim` values whose activations are `dtype`.
    pub fn vector_bytes(self, head_dim: usize, dtype: DType) -> usize {
        match self {
            Self::Model => head_dim * dtype.size_in_bytes(),
            Self::Q8 => head_dim + Q8_PAD,
            Self::Turbo4 | Self::Rotor4 | Self::Planar4 | Self::Iso4 => head_dim / 2 + TQ_PAD,
        }
    }

    /// What the kernels see: the rotated formats share the storage of `Turbo4`.
    pub fn storage(self) -> Self {
        match self {
            Self::Rotor4 | Self::Planar4 | Self::Iso4 => Self::Turbo4,
            f => f,
        }
    }

    /// Side of the rotation's diagonal blocks: `None` without a rotation, 0 for a dense one.
    pub fn rotation_block(self) -> Option<usize> {
        match self {
            Self::Model | Self::Q8 => None,
            Self::Turbo4 => Some(0),
            Self::Planar4 => Some(2),
            Self::Rotor4 => Some(3),
            Self::Iso4 => Some(4),
        }
    }
}

static KV_FORMAT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Bytes after each int8 KV vector: its f32 scale (`Q8_PAD` in attention_tc.cu).
pub const Q8_PAD: usize = 4;
/// Bytes after each TurboQuant vector: its f32 scale (`TQ_PAD` in attention_tc.cu).
pub const TQ_PAD: usize = 4;

/// The format of KV pools on CUDA devices. Set before the pools are allocated.
pub fn set_kv_format(f: KvFormat) {
    KV_FORMAT.store(f as u8, std::sync::atomic::Ordering::Relaxed);
}

pub fn kv_format() -> KvFormat {
    match KV_FORMAT.load(std::sync::atomic::Ordering::Relaxed) {
        1 => KvFormat::Q8,
        2 => KvFormat::Turbo4,
        3 => KvFormat::Rotor4,
        4 => KvFormat::Planar4,
        5 => KvFormat::Iso4,
        _ => KvFormat::Model,
    }
}

/// The format a pool on `device` gets: the packed ones are read by the CUDA kernels only.
fn format_on(device: &Device) -> KvFormat {
    if device.is_cuda() {
        kv_format()
    } else {
        KvFormat::Model
    }
}

/// Bytes one token's K and V take in one layer's pool on `device`.
pub fn kv_layer_bytes(kv_heads: usize, head_dim: usize, dtype: DType, device: &Device) -> usize {
    2 * kv_heads * format_on(device).vector_bytes(head_dim, dtype)
}

/// The 16-level Lloyd-Max quantizer of N(0, 1) that TurboQuant pools use (`tq4_centroid` /
/// `tq4_code` in attention_tc.cu): the positive centroids and the boundaries between them.
pub const TQ4_CENTROIDS: [f32; 8] = [0.128395, 0.388048, 0.656759, 0.942340, 1.256231, 1.618046, 2.069017, 2.732590];
pub const TQ4_BOUNDS: [f32; 7] = [0.258222, 0.522404, 0.799550, 1.099286, 1.437139, 1.843532, 2.400803];

/// Factor on a stored vector's scale: a Lloyd-Max reconstruction has `1 - MSE` of the inner
/// product with its original, so plain centroids would bias scores and outputs low.
pub const TQ4_UNBIAS: f32 = 1.009592;

/// Centroid of code `c` (0..16).
pub fn tq4_centroid(c: u8) -> f32 {
    if c >= 8 {
        TQ4_CENTROIDS[c as usize - 8]
    } else {
        -TQ4_CENTROIDS[7 - c as usize]
    }
}

/// Code of a value in units of the vector's scale.
pub fn tq4_code(u: f32) -> u8 {
    let m = TQ4_BOUNDS.iter().filter(|b| u.abs() > **b).count() as u8;
    if u >= 0.0 {
        8 + m
    } else {
        7 - m
    }
}

/// A `d x d` orthogonal matrix, the same for a given `d` and `seed` everywhere (snapshots and
/// restarts must agree): Gaussian entries, orthonormalised row by row.
pub fn random_rotation(d: usize, seed: u64) -> Vec<f64> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut uniform = move || {
        // splitmix64
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut m = vec![0f64; d * d];
    for x in m.iter_mut() {
        // Box-Muller
        let (u1, u2) = (uniform().max(1e-300), uniform());
        *x = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    }
    for i in 0..d {
        // twice: the second pass removes what rounding left of the first
        for _ in 0..2 {
            for j in 0..i {
                let dot: f64 = (0..d).map(|c| m[i * d + c] * m[j * d + c]).sum();
                for c in 0..d {
                    m[i * d + c] -= dot * m[j * d + c];
                }
            }
        }
        let norm = (0..d).map(|c| m[i * d + c] * m[i * d + c]).sum::<f64>().sqrt();
        for c in 0..d {
            m[i * d + c] /= norm;
        }
    }
    m
}

/// Determinant of a row-major `n x n` matrix (elimination with partial pivoting).
fn determinant(m: &[f64], n: usize) -> f64 {
    let mut a = m.to_vec();
    let mut det = 1.0;
    for c in 0..n {
        let p = (c..n).max_by(|x, y| a[x * n + c].abs().total_cmp(&a[y * n + c].abs())).unwrap();
        if a[p * n + c] == 0.0 {
            return 0.0;
        }
        if p != c {
            for j in 0..n {
                a.swap(p * n + j, c * n + j);
            }
            det = -det;
        }
        det *= a[c * n + c];
        for r in c + 1..n {
            let f = a[r * n + c] / a[c * n + c];
            for j in c..n {
                a[r * n + j] -= f * a[c * n + j];
            }
        }
    }
    det
}

/// RotorQuant's rotation as a `d x d` matrix: independent random rotations (determinant +1,
/// what a rotor sandwich `R x R~` or a quaternion pair gives) on consecutive groups of `block`
/// coordinates; the `d % block` coordinates left at the end are not rotated.
pub fn block_rotation(d: usize, block: usize, seed: u64) -> Vec<f64> {
    let mut m = vec![0f64; d * d];
    for g in 0..d / block {
        let mut r = random_rotation(block, seed.wrapping_add(g as u64).wrapping_mul(0x2545_f491_4f6c_dd1d));
        if determinant(&r, block) < 0.0 {
            r[..block].iter_mut().for_each(|x| *x = -*x);
        }
        for i in 0..block {
            for j in 0..block {
                m[(g * block + i) * d + g * block + j] = r[i * block + j];
            }
        }
    }
    for i in d - d % block..d {
        m[i * d + i] = 1.0;
    }
    m
}

/// The rotation of a rotated pool as two `[d, d]` tensors: `x.matmul(to)` rotates row
/// vectors, `y.matmul(back)` rotates them back. One matrix per format and head dim for the
/// whole engine.
fn pool_rotation(block: usize, d: usize, dtype: DType, device: &Device) -> Result<(Tensor, Tensor)> {
    let m = match block {
        0 => random_rotation(d, 0x7475_7262_6f34),
        b => block_rotation(d, b, 0x726f_746f_7234 + b as u64),
    };
    let m: Vec<f32> = m.into_iter().map(|x| x as f32).collect();
    let back = Tensor::from_vec(m, (d, d), &Device::Cpu)?; // rows are the rotation's rows
    let to = back.t()?.contiguous()?;
    Ok((to.to_dtype(dtype)?.to_device(device)?, back.to_dtype(dtype)?.to_device(device)?))
}

/// KV tensors for one layer on one device: `[num_slots, kv_heads * head_dim]` each in the
/// activation dtype, or U8 `[num_slots, kv_heads * vector_bytes]` for the packed formats.
pub struct LayerKv {
    pub k: Tensor,
    pub v: Tensor,
    /// The activations' dtype: what attention reads the pools as.
    pub dtype: DType,
    /// How the pools store vectors; anything but `Model` is read by the CUDA kernels only.
    pub fmt: KvFormat,
    /// TurboQuant / RotorQuant: the rotation stored vectors went through, as (`to`, `back`) `[d, d]`
    /// matrices in the activation dtype. Queries take `to`, attention outputs `back`.
    pub rot: Option<(Tensor, Tensor)>,
    /// Per-token indexer keys `[slots, index_dim]` (QSA sparse attention layers only).
    pub idx: Option<Tensor>,
}

impl LayerKv {
    pub fn new(
        num_slots: usize,
        kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let fmt = format_on(device);
        let (shape, pool_dtype) = match fmt {
            KvFormat::Model => ((num_slots, kv_heads * head_dim), dtype),
            _ => ((num_slots, kv_heads * fmt.vector_bytes(head_dim, dtype)), DType::U8),
        };
        let rot = match fmt.rotation_block() {
            Some(block) => Some(pool_rotation(block, head_dim, dtype, device)?),
            None => None,
        };
        Ok(Self {
            k: Tensor::zeros(shape, pool_dtype, device)?,
            v: Tensor::zeros(shape, pool_dtype, device)?,
            dtype,
            fmt,
            rot,
            idx: None,
        })
    }

    /// Whether the pools hold packed vectors (not readable as activations).
    pub fn packed(&self) -> bool {
        self.fmt != KvFormat::Model
    }

    /// Queries `[n, heads, head_dim]` in the space the pool's vectors are stored in.
    pub fn rotate_queries(&self, q: &Tensor) -> Result<Tensor> {
        match &self.rot {
            Some((to, _)) => {
                let (n, h, d) = q.dims3()?;
                Ok(q.reshape((n * h, d))?.matmul(to)?.reshape((n, h, d))?)
            }
            None => Ok(q.clone()),
        }
    }

    /// Attention outputs `[n, heads, head_dim]` (sums of stored values) back in model space.
    pub fn unrotate_outputs(&self, o: &Tensor) -> Result<Tensor> {
        match &self.rot {
            Some((_, back)) => {
                let (n, h, d) = o.dims3()?;
                Ok(o.reshape((n * h, d))?.matmul(back)?.reshape((n, h, d))?)
            }
            None => Ok(o.clone()),
        }
    }

    /// Store the new tokens' keys and values `[T, kv_heads * head_dim]` in the slots of
    /// `scatter_idx` (`[T, kv_heads * head_dim]` u32, every element of a row its slot).
    pub fn write(&self, scatter_idx: &Tensor, k: &Tensor, v: &Tensor, kv_heads: usize) -> Result<()> {
        if self.packed() {
            #[cfg(feature = "cuda")]
            {
                use crate::kernels::cuda::kv_packed_write;
                for (x, pool) in [(k, &self.k), (v, &self.v)] {
                    let mut x = x.to_dtype(self.dtype)?;
                    if let Some((to, _)) = &self.rot {
                        let (t, width) = x.dims2()?;
                        let d = width / kv_heads;
                        x = x.reshape((t * kv_heads, d))?.matmul(to)?.reshape((t, width))?;
                    }
                    kv_packed_write(&x, pool, scatter_idx, kv_heads, self.fmt.storage())?;
                }
                return Ok(());
            }
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("packed KV pools need the cuda build");
        }
        let _ = kv_heads;
        self.k.scatter_set(scatter_idx, &k.to_dtype(self.dtype)?.contiguous()?, 0)?;
        self.v.scatter_set(scatter_idx, &v.to_dtype(self.dtype)?.contiguous()?, 0)?;
        Ok(())
    }

    /// Also keep a `[num_slots, index_dim]` pool of indexer keys.
    pub fn with_index(mut self, index_dim: usize) -> Result<Self> {
        let n = self.k.dim(0)?;
        self.idx = Some(Tensor::zeros((n, index_dim), self.dtype, self.k.device())?);
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_is_orthogonal_and_reproducible() {
        let d = 64;
        let m = random_rotation(d, 7);
        assert_eq!(m, random_rotation(d, 7));
        for i in 0..d {
            for j in 0..d {
                let dot: f64 = (0..d).map(|c| m[i * d + c] * m[j * d + c]).sum();
                assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12, "rows {i}, {j}: {dot}");
            }
        }
    }

    /// Block rotations are orthogonal, touch only their own coordinates, have determinant +1
    /// per block, and leave the coordinates past the last whole block alone.
    #[test]
    fn block_rotation_is_block_diagonal() {
        for (d, b) in [(256, 3), (256, 2), (256, 4), (64, 3), (128, 3)] {
            let m = block_rotation(d, b, 11);
            assert_eq!(m, block_rotation(d, b, 11));
            for i in 0..d {
                for j in 0..d {
                    let dot: f64 = (0..d).map(|c| m[i * d + c] * m[j * d + c]).sum();
                    assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
                    let whole = i < d - d % b && j < d - d % b;
                    if !(whole && i / b == j / b) && i != j {
                        assert_eq!(m[i * d + j], 0.0, "d {d} block {b}: ({i}, {j})");
                    }
                }
            }
            for g in 0..d / b {
                let blk: Vec<f64> = (0..b * b).map(|x| m[(g * b + x / b) * d + g * b + x % b]).collect();
                assert!((determinant(&blk, b) - 1.0).abs() < 1e-12);
                // a rotation, not a permutation of the axes
                assert!(blk.iter().all(|x| x.abs() < 1.0));
            }
            for i in d - d % b..d {
                assert_eq!(m[i * d + i], 1.0);
            }
            // blocks differ from one another
            assert_ne!(m[0], m[b * d + b]);
        }
        for s in ["rotor4", "planar4", "iso4"] {
            let f = KvFormat::parse(s).unwrap();
            assert_eq!(f.storage(), KvFormat::Turbo4);
            assert_eq!(f.vector_bytes(256, DType::BF16), 132);
        }
    }

    /// The codebook is a fixed point of Lloyd's iteration on N(0, 1) (each centroid is the mean
    /// of its cell) and its distortion is the 0.0095 the TurboQuant paper gives for 4 bits.
    #[test]
    fn tq4_codebook_is_lloyd_max() {
        let pdf = |x: f64| (-x * x / 2.0).exp() / (std::f64::consts::TAU).sqrt();
        let mut edges = vec![0.0f64];
        edges.extend(TQ4_BOUNDS.iter().map(|b| *b as f64));
        edges.push(12.0);
        let mut mse = 0.0;
        for (i, c) in TQ4_CENTROIDS.iter().enumerate() {
            let (a, b, n) = (edges[i], edges[i + 1], 200_000);
            let h = (b - a) / n as f64;
            let (mut mass, mut mean, mut err) = (0.0, 0.0, 0.0);
            for k in 0..n {
                let x = a + (k as f64 + 0.5) * h;
                mass += pdf(x) * h;
                mean += x * pdf(x) * h;
                err += (x - *c as f64).powi(2) * pdf(x) * h;
            }
            assert!((mean / mass - *c as f64).abs() < 2e-5, "centroid {i}: {} vs {c}", mean / mass);
            mse += 2.0 * err;
        }
        assert!((mse - 0.009501).abs() < 1e-5, "mse {mse}");
        // codes and centroids agree, and boundaries sit halfway between centroids
        for c in 0..16u8 {
            assert_eq!(tq4_code(tq4_centroid(c)), c);
        }
        for (i, b) in TQ4_BOUNDS.iter().enumerate() {
            assert!((b - (TQ4_CENTROIDS[i] + TQ4_CENTROIDS[i + 1]) / 2.0).abs() < 2e-6);
        }
    }
}
