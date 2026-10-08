// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Weight sources: memory-mapped safetensors on disk, or an in-memory map
//! (used by RPC workers that receive their shard over the network).

use anyhow::{bail, Context, Result};
use candle_core::{safetensors::MmapedSafetensors, DType, Device, Tensor};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

pub trait WeightSource: Send + Sync {
    /// Load a tensor on the CPU exactly as stored (FP8 codes stay FP8).
    fn load_raw(&self, name: &str) -> Result<Tensor>;
    fn contains(&self, name: &str) -> bool;
    fn names(&self) -> Vec<String>;

    /// Row access to a stored 2-D tensor without loading it (memory-mapped sources only), for
    /// tables too large to hold (PLE). `None` when the source cannot map it.
    fn row_reader(&self, _name: &str) -> Option<Result<RowReader>> {
        None
    }

    /// Load a tensor on the CPU; quantized weights (block/per-tensor FP8, NVFP4) are decoded to
    /// f32 (a transient, per-tensor copy: callers cast to their compute dtype right away).
    fn load_cpu(&self, name: &str) -> Result<Tensor> {
        if let Some(q) = self.load_nvfp4(name)? {
            return dequant_nvfp4(&q, DType::F32).with_context(|| format!("dequantizing '{name}'"));
        }
        match self.load_fp8(name)? {
            Some((w, s)) => {
                dequant_fp8(&w, &s, DType::F32).with_context(|| format!("dequantizing '{name}'"))
            }
            None => self.load_raw(name),
        }
    }

    /// FP8 codes and f32 scale multipliers, if `name` is stored FP8: block scales
    /// (`<name>_scale_inv`, DeepSeek / Qwen FP8) or a per-tensor `weight_scale` (ModelOpt FP8).
    fn load_fp8(&self, name: &str) -> Result<Option<(Tensor, Tensor)>> {
        let block = format!("{name}_scale_inv");
        let tensor = name
            .strip_suffix(".weight")
            .map(|b| format!("{b}.weight_scale"));
        let scale = if self.contains(&block) {
            block
        } else {
            match tensor.filter(|t| self.contains(t)) {
                Some(t) => t,
                None => return Ok(None),
            }
        };
        if !self.contains(name) {
            return Ok(None);
        }
        let w = self.load_raw(name)?;
        if w.dtype() != DType::F8E4M3 {
            return Ok(None);
        }
        Ok(Some((
            fp8_codes(&w)?,
            self.load_raw(&scale)?.to_dtype(DType::F32)?,
        )))
    }

    /// NVFP4 weight `name` (`<base>.weight`): ModelOpt (`weight` u8 + `weight_scale` +
    /// `weight_scale_2`) or compressed-tensors (`weight_packed` + `weight_scale` +
    /// `weight_global_scale`, a reciprocal).
    fn load_nvfp4(&self, name: &str) -> Result<Option<Nvfp4>> {
        let Some(base) = name.strip_suffix(".weight") else {
            return Ok(None);
        };
        let scale_name = format!("{base}.weight_scale");
        if !self.contains(&scale_name) {
            return Ok(None);
        }
        let (packed, global) = if self.contains(&format!("{base}.weight_packed")) {
            let g = scalar(&self.load_raw(&format!("{base}.weight_global_scale"))?)?;
            (self.load_raw(&format!("{base}.weight_packed"))?, 1.0 / g)
        } else if self.contains(&format!("{base}.weight_scale_2")) && self.contains(name) {
            (
                self.load_raw(name)?,
                scalar(&self.load_raw(&format!("{base}.weight_scale_2"))?)?,
            )
        } else {
            return Ok(None);
        };
        if packed.dtype() != DType::U8 {
            return Ok(None);
        }
        let scale = self.load_raw(&scale_name)?;
        let scale = if scale.dtype() == DType::F8E4M3 {
            fp8_codes(&scale)?
        } else {
            scale.to_dtype(DType::F32)?
        };
        Ok(Some(Nvfp4 {
            packed,
            scale,
            global,
        }))
    }

    /// Load, cast and move a tensor.
    fn get(&self, name: &str, device: &Device, dtype: DType) -> Result<Tensor> {
        let t = self.load_cpu(name)?;
        Ok(t.to_dtype(dtype)?.to_device(device)?)
    }

    /// Load a tensor shard: split `dim` into `world` equal parts and keep part `rank`.
    fn get_shard(
        &self,
        name: &str,
        dim: usize,
        rank: usize,
        world: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let t = self.load_cpu(name)?;
        let t = if world > 1 {
            let size = t.dim(dim)?;
            if size % world != 0 {
                bail!("cannot shard {name} dim {dim} of size {size} into {world} parts");
            }
            let chunk = size / world;
            t.narrow(dim, rank * chunk, chunk)?.contiguous()?
        } else {
            t
        };
        Ok(t.to_dtype(dtype)?.to_device(device)?)
    }
}

/// NVFP4 weight: E2M1 codes packed two per byte (low nibble first) `[rows, cols/2]`,
/// FP8-E4M3 scales per 16 inputs `[rows, cols/16]` and a global multiplier.
#[derive(Clone, Debug)]
pub struct Nvfp4 {
    pub packed: Tensor,
    pub scale: Tensor,
    pub global: f32,
}

/// Quantize a row-major `[rows, cols]` matrix to NVFP4 (`cols % 16 == 0`): packed E2M1 codes
/// `[rows, cols / 2]` (low nibble = even column), E4M3 block scales `[rows, cols / 16]` (bits) and
/// the global multiplier, so `w ~= E2M1[code] * e4m3(scale) * global` (ModelOpt's layout).
pub fn quantize_nvfp4(w: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>, f32) {
    assert!(cols.is_multiple_of(16) && w.len() == rows * cols);
    let amax = w.iter().fold(0f32, |m, v| m.max(v.abs()));
    // block scales up to E4M3's max (448) times the largest E2M1 value (6) cover amax
    let global = if amax > 0.0 { amax / (6.0 * 448.0) } else { 1.0 };
    let mut packed = vec![0u8; rows * cols / 2];
    let mut scales = vec![0u8; rows * cols / 16];
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for r in 0..rows {
        for b in 0..cols / 16 {
            let blk = &w[r * cols + b * 16..r * cols + b * 16 + 16];
            let bmax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let s8 = float8::F8E4M3::from_f32((bmax / 6.0 / global).min(448.0));
            scales[r * cols / 16 + b] = s8.to_bits();
            let step = s8.to_f32() * global;
            for (j, &v) in blk.iter().enumerate() {
                let q = if step > 0.0 { (v / step).abs() } else { 0.0 };
                // nearest magnitude (ties to the lower one)
                let mut code = 0u8;
                for (c, m) in MAG.iter().enumerate().skip(1) {
                    if q > (MAG[c - 1] + m) / 2.0 {
                        code = c as u8;
                    }
                }
                if v < 0.0 && code != 0 {
                    code |= 8;
                }
                let byte = &mut packed[r * cols / 2 + b * 8 + j / 2];
                *byte |= if j % 2 == 0 { code } else { code << 4 };
            }
        }
    }
    (packed, scales, global)
}

pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// FP8 e4m3 values as their raw bytes (U8). Decoding goes through a lookup table, so no
/// FP8 GPU kernels are needed (candle only builds those for compute capability >= 8.0).
pub fn fp8_codes(t: &Tensor) -> Result<Tensor> {
    match t.dtype() {
        DType::U8 => Ok(t.clone()),
        DType::F8E4M3 => {
            let bytes: Vec<u8> = t
                .flatten_all()?
                .to_vec1::<float8::F8E4M3>()?
                .iter()
                .map(|x| x.to_bits())
                .collect();
            Ok(Tensor::from_vec(bytes, t.shape(), &Device::Cpu)?.to_device(t.device())?)
        }
        d => bail!("expected fp8 codes, got {d:?}"),
    }
}

/// Decode e4m3 bytes (U8, or an F8E4M3 tensor) to f32 on the tensor's device.
pub fn e4m3_to_f32(codes: &Tensor) -> Result<Tensor> {
    let codes = fp8_codes(codes)?;
    let lut: Vec<f32> = (0..=255u8)
        .map(|b| float8::F8E4M3::from_bits(b).to_f32())
        .collect();
    let lut = Tensor::from_vec(lut, 256, codes.device())?;
    Ok(lut
        .index_select(&codes.flatten_all()?, 0)?
        .reshape(codes.shape())?)
}

/// Scales stored as e4m3 bytes or as floats, decoded to f32.
fn scales_f32(s: &Tensor) -> Result<Tensor> {
    match s.dtype() {
        DType::U8 | DType::F8E4M3 => e4m3_to_f32(s),
        _ => Ok(s.to_dtype(DType::F32)?),
    }
}

fn scalar(t: &Tensor) -> Result<f32> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0])
}

/// Decode NVFP4 on the tensors' device: `w = E2M1[code] * scale[row, col/16] * global`.
pub fn dequant_nvfp4(q: &Nvfp4, dtype: DType) -> Result<Tensor> {
    if q.packed.device().is_cpu() && q.packed.is_contiguous() && q.scale.is_contiguous() {
        return Ok(crate::kernels::nvfp4_dequant(&q.packed, &q.scale, q.global)?.to_dtype(dtype)?);
    }
    dequant_nvfp4_generic(q, dtype)
}

/// Tensor-op NVFP4 decoder (any device).
pub fn dequant_nvfp4_generic(q: &Nvfp4, dtype: DType) -> Result<Tensor> {
    let (rows, half) = q.packed.dims2()?;
    let cols = half * 2;
    let dev = q.packed.device();
    let lo: Vec<f32> = (0..256).map(|b| E2M1[b & 15]).collect();
    let hi: Vec<f32> = (0..256).map(|b| E2M1[b >> 4]).collect();
    let codes = q.packed.flatten_all()?;
    let lo = Tensor::from_vec(lo, 256, dev)?.index_select(&codes, 0)?;
    let hi = Tensor::from_vec(hi, 256, dev)?.index_select(&codes, 0)?;
    let vals = Tensor::stack(&[lo, hi], 1)?.reshape((rows, cols))?;
    let s = scales_f32(&q.scale)?;
    let nb = s.dim(1)?;
    if s.dim(0)? != rows || cols % nb != 0 {
        bail!(
            "nvfp4 scale shape {:?} does not match weight {rows}x{cols}",
            s.dims()
        );
    }
    let w = vals
        .reshape((rows, nb, cols / nb))?
        .broadcast_mul(&s.reshape((rows, nb, 1))?)?
        .reshape((rows, cols))?;
    Ok((w * q.global as f64)?.to_dtype(dtype)?)
}

/// Rows of a 2-D tensor in a memory-mapped file.
pub struct RowReader {
    map: std::sync::Arc<memmap2::Mmap>,
    offset: usize,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl RowReader {
    /// An N-D tensor is read as 2-D: `[product of the leading dims, last dim]`.
    pub fn new(map: std::sync::Arc<memmap2::Mmap>, offset: usize, shape: Vec<usize>, dtype: DType) -> Result<Self> {
        let shape = match shape.as_slice() {
            [] | [_] => bail!("row access needs at least a 2-D tensor, got {shape:?}"),
            [lead @ .., last] => vec![lead.iter().product(), *last],
        };
        let end = offset + shape[0] * shape[1] * dtype.size_in_bytes();
        if end > map.len() {
            bail!("tensor data runs past the end of the file");
        }
        Ok(Self { map, offset, shape, dtype })
    }

    /// Rows `start..start + n` as a host tensor in the stored dtype `[n, cols]` (one copy out of
    /// the mapping; the range is read ahead first).
    pub fn rows_tensor(&self, start: usize, n: usize) -> Result<Tensor> {
        let (rows, cols) = (self.shape[0], self.shape[1]);
        if start + n > rows {
            bail!("rows {start}..{} out of {rows}", start + n);
        }
        let es = self.dtype.size_in_bytes();
        let a = self.offset + start * cols * es;
        let len = n * cols * es;
        let _ = self.map.advise_range(memmap2::Advice::WillNeed, a, len);
        Ok(Tensor::from_raw_buffer(&self.map[a..a + len], self.dtype, &[n, cols], &Device::Cpu)?)
    }

    /// Ask the kernel to read row `r` in (asynchronous: many outstanding requests keep the disk's
    /// queue deep, where faulting rows one by one waits for each).
    pub fn prefetch_row(&self, r: usize) {
        let (rows, cols) = (self.shape[0], self.shape[1]);
        if r >= rows {
            return;
        }
        let es = self.dtype.size_in_bytes();
        let start = self.offset + r * cols * es;
        let a = start & !4095;
        let _ = self.map.advise_range(memmap2::Advice::WillNeed, a, start + cols * es - a);
    }

    /// Append row `r` as f32 to `out`.
    pub fn row_f32(&self, r: usize, out: &mut Vec<f32>) -> Result<()> {
        let n = out.len();
        out.resize(n + self.shape[1], 0.0);
        self.row_into(r, &mut out[n..])
    }

    /// Row `r` as f32 into `dst` (`cols` long).
    pub fn row_into(&self, r: usize, dst: &mut [f32]) -> Result<()> {
        let (rows, cols) = (self.shape[0], self.shape[1]);
        if r >= rows || dst.len() != cols {
            bail!("row {r} out of {rows} (or a {}-wide destination for {cols} columns)", dst.len());
        }
        let es = self.dtype.size_in_bytes();
        let start = self.offset + r * cols * es;
        let b = &self.map[start..start + cols * es];
        match self.dtype {
            DType::F8E4M3 => {
                for (d, &x) in dst.iter_mut().zip(b) {
                    *d = float8::F8E4M3::from_bits(x).to_f32();
                }
            }
            DType::BF16 => {
                for (d, c) in dst.iter_mut().zip(b.chunks_exact(2)) {
                    *d = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
                }
            }
            DType::F16 => {
                for (d, c) in dst.iter_mut().zip(b.chunks_exact(2)) {
                    *d = half::f16::from_le_bytes([c[0], c[1]]).to_f32();
                }
            }
            DType::F32 => {
                for (d, c) in dst.iter_mut().zip(b.chunks_exact(4)) {
                    *d = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
            d => bail!("row access does not support {d:?}"),
        }
        Ok(())
    }
}

fn safetensors_dtype(s: &str) -> Option<DType> {
    Some(match s {
        "F8_E4M3" => DType::F8E4M3,
        "BF16" => DType::BF16,
        "F16" => DType::F16,
        "F32" => DType::F32,
        "U8" => DType::U8,
        _ => return None,
    })
}

/// name -> (file, absolute data offset, dtype, shape)
type StIndex = HashMap<String, (usize, usize, DType, Vec<usize>)>;

pub struct SafetensorsSource {
    inner: MmapedSafetensors,
    names: HashSet<String>,
    files: Vec<PathBuf>,
    index: std::sync::OnceLock<StIndex>,
    maps: Mutex<HashMap<usize, std::sync::Arc<memmap2::Mmap>>>,
}

impl SafetensorsSource {
    /// Parse every file's header once (offsets of each tensor's data).
    fn index(&self) -> &StIndex {
        self.index.get_or_init(|| {
            let mut idx = HashMap::new();
            for (fi, path) in self.files.iter().enumerate() {
                let Ok(mut f) = std::fs::File::open(path) else {
                    continue;
                };
                use std::io::Read;
                let mut n = [0u8; 8];
                if f.read_exact(&mut n).is_err() {
                    continue;
                }
                let n = u64::from_le_bytes(n) as usize;
                let mut h = vec![0u8; n];
                if f.read_exact(&mut h).is_err() {
                    continue;
                }
                let Ok(v) = serde_json::from_slice::<serde_json::Value>(&h) else {
                    continue;
                };
                for (name, meta) in v.as_object().into_iter().flatten() {
                    let (Some(dt), Some(off), Some(shape)) = (
                        meta["dtype"].as_str().and_then(safetensors_dtype),
                        meta["data_offsets"][0].as_u64(),
                        meta["shape"].as_array(),
                    ) else {
                        continue;
                    };
                    let shape = shape.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect();
                    idx.insert(name.clone(), (fi, 8 + n + off as usize, dt, shape));
                }
            }
            idx
        })
    }
}

pub fn safetensor_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index)?)?;
        let map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .context("weight_map missing in index")?;
        let mut files: Vec<String> = map
            .values()
            .filter_map(|x| x.as_str().map(String::from))
            .collect();
        files.sort();
        files.dedup();
        return Ok(files.into_iter().map(|f| dir.join(f)).collect());
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no .safetensors files found in {}", dir.display());
    }
    Ok(files)
}

impl SafetensorsSource {
    pub fn open(dir: &Path) -> Result<Self> {
        let files = safetensor_files(dir)?;
        // SAFETY: the files are memory mapped read-only; modifying them while serving is unsupported.
        let inner = unsafe { MmapedSafetensors::multi(&files)? };
        let names = inner.tensors().into_iter().map(|(n, _)| n).collect();
        Ok(Self {
            inner,
            names,
            files,
            index: Default::default(),
            maps: Default::default(),
        })
    }
}

impl WeightSource for SafetensorsSource {
    fn row_reader(&self, name: &str) -> Option<Result<RowReader>> {
        let (fi, off, dt, shape) = self.index().get(name)?.clone();
        let map = {
            let mut maps = self.maps.lock();
            match maps.get(&fi) {
                Some(m) => m.clone(),
                None => {
                    let m = match std::fs::File::open(&self.files[fi])
                        // SAFETY: read-only map; the checkpoint is not modified while serving
                        .and_then(|f| unsafe { memmap2::Mmap::map(&f) })
                    {
                        Ok(m) => m,
                        Err(e) => return Some(Err(e.into())),
                    };
                    // rows are hashed n-grams: no readahead, unless asked to warm the cache
                    let preload = std::env::var("CARBON_PLE_PRELOAD").is_ok_and(|v| v == "1");
                    let _ = m.advise(if preload {
                        memmap2::Advice::WillNeed
                    } else {
                        memmap2::Advice::Random
                    });
                    let m = std::sync::Arc::new(m);
                    maps.insert(fi, m.clone());
                    m
                }
            }
        };
        Some(RowReader::new(map, off, shape, dt))
    }

    fn load_raw(&self, name: &str) -> Result<Tensor> {
        self.inner
            .load(name, &Device::Cpu)
            .with_context(|| format!("loading weight '{name}'"))
    }
    fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }
    fn names(&self) -> Vec<String> {
        self.names.iter().cloned().collect()
    }
}

/// Tensors held in host memory, keyed by name.
#[derive(Default)]
pub struct MapSource {
    map: Mutex<HashMap<String, Tensor>>,
}

impl MapSource {
    pub fn insert(&self, name: String, t: Tensor) {
        self.map.lock().insert(name, t);
    }
    pub fn len(&self) -> usize {
        self.map.lock().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl WeightSource for MapSource {
    fn load_raw(&self, name: &str) -> Result<Tensor> {
        self.map
            .lock()
            .get(name)
            .cloned()
            .with_context(|| format!("weight '{name}' was not received"))
    }
    fn contains(&self, name: &str) -> bool {
        self.map.lock().contains_key(name)
    }
    fn names(&self) -> Vec<String> {
        self.map.lock().keys().cloned().collect()
    }
}

/// Dequantize block-FP8 `w: [rows, cols]` with inverse scales `[ceil(rows/b0), ceil(cols/b1)]`
/// (or one per-tensor scale) on `w`'s device, producing `dtype`.
pub fn dequant_fp8(w: &Tensor, scale: &Tensor, dtype: DType) -> Result<Tensor> {
    if w.device().is_cpu() && w.is_contiguous() && w.rank() == 2 {
        return Ok(crate::kernels::fp8_dequant(w, scale)?.to_dtype(dtype)?);
    }
    dequant_fp8_generic(w, scale, dtype)
}

/// Tensor-op FP8 decoder (any device).
pub fn dequant_fp8_generic(w: &Tensor, scale: &Tensor, dtype: DType) -> Result<Tensor> {
    let wf = e4m3_to_f32(w)?;
    let s = scale.to_dtype(DType::F32)?;
    if s.elem_count() == 1 {
        return Ok(wf.broadcast_mul(&s.flatten_all()?)?.to_dtype(dtype)?);
    }
    let (rows, cols) = wf.dims2()?;
    let (sr, sc) = s.dims2()?;
    let (b0, b1) = (rows.div_ceil(sr), cols.div_ceil(sc));
    if rows.div_ceil(b0) != sr || cols.div_ceil(b1) != sc {
        bail!("fp8 scale shape {sr}x{sc} does not tile weight {rows}x{cols}");
    }
    let out = if b0 * sr == rows && b1 * sc == cols {
        // exact tiling: scale broadcasts over each block without materialising a full scale map
        wf.reshape((sr, b0, sc, b1))?
            .broadcast_mul(&s.reshape((sr, 1, sc, 1))?)?
            .reshape((rows, cols))?
    } else {
        let full = s
            .unsqueeze(1)?
            .broadcast_as((sr, b0, sc))?
            .reshape((sr * b0, sc))?
            .unsqueeze(2)?
            .broadcast_as((sr * b0, sc, b1))?
            .reshape((sr * b0, sc * b1))?
            .narrow(0, 0, rows)?
            .narrow(1, 0, cols)?;
        (wf * full)?
    };
    Ok(out.to_dtype(dtype)?)
}

/// Block-FP8 to bf16 (see [`dequant_fp8`]).
pub fn dequant_fp8_block(w: &Tensor, scale_inv: &Tensor) -> Result<Tensor> {
    dequant_fp8(w, scale_inv, DType::BF16)
}

/// Names of every tensor that belongs to decoder layer `layer` under `prefix` (e.g. `model.`).
pub fn layer_weight_names(src: &dyn WeightSource, prefix: &str, layer: usize) -> Vec<String> {
    let prefix = format!("{prefix}layers.{layer}.");
    let mut v: Vec<String> = src
        .names()
        .into_iter()
        .filter(|n| n.starts_with(&prefix))
        .collect();
    v.sort();
    v
}

#[cfg(test)]
mod tests {
    #[test]
    fn nvfp4_quantization_round_trips() {
        let (rows, cols) = (8usize, 64usize);
        // exactly representable: E2M1 magnitudes times one power-of-two block scale per row
        let mags = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let exact: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let m = mags[i % 8] * (1u32 << (i / cols)) as f32 / 64.0;
                if i % 3 == 0 { -m } else { m }
            })
            .collect();
        let deq = |w: &[f32]| -> Vec<f32> {
            let (p, sc, g) = quantize_nvfp4(w, rows, cols);
            let q = Nvfp4 {
                packed: Tensor::from_vec(p, (rows, cols / 2), &Device::Cpu).unwrap(),
                scale: Tensor::from_vec(sc, (rows, cols / 16), &Device::Cpu).unwrap(),
                global: g,
            };
            dequant_nvfp4(&q, DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };
        let back = deq(&exact);
        let amax = exact.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (a, b) in exact.iter().zip(&back) {
            assert!((a - b).abs() <= amax * 2e-2, "{a} vs {b}");
        }
        // arbitrary values: within half an E2M1 step of each block's scale
        let rand: Vec<f32> = (0..rows * cols).map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * (1 + i % 5) as f32).collect();
        let back = deq(&rand);
        let err: f32 = rand.iter().zip(&back).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        let norm: f32 = rand.iter().map(|a| a * a).sum::<f32>().sqrt();
        assert!(err / norm < 0.12, "relative error {}", err / norm);
    }

    use super::*;

    #[test]
    fn nvfp4_dequant() {
        // row 0: bytes 0x21 -> [code 1, code 2] = [0.5, 1.0]; 0xF7 -> [6.0, -6.0]; scale 2, global 0.5
        let mut bytes = vec![0x21u8, 0xF7];
        bytes.extend(std::iter::repeat_n(0u8, 6));
        bytes.extend((0..8).map(|i| (i as u8) << 4 | 0x8)); // row 1: [-0.0, code i]
        let packed = Tensor::from_vec(bytes, (2, 8), &Device::Cpu).unwrap();
        let scale = Tensor::new(&[[2f32], [1.0]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let q = Nvfp4 {
            packed,
            scale,
            global: 0.5,
        };
        let d = dequant_nvfp4(&q, DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(&d[0][..4], &[0.5, 1.0, 6.0, -6.0]);
        assert!(d[0][4..].iter().all(|x| *x == 0.0));
        assert_eq!(d[1][1], 0.0);
        assert_eq!(d[1][15], E2M1[7] * 0.5);
        assert_eq!(d[1][7], E2M1[3] * 0.5);
    }

    #[test]
    fn fp8_block_dequant() {
        // 3x5 weight with 2x2 blocks -> 2x3 scales
        let w = Tensor::ones((3, 5), DType::F32, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F8E4M3)
            .unwrap();
        let s = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu).unwrap();
        let d = dequant_fp8_block(&w, &s)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(d[0], vec![1., 1., 2., 2., 3.]);
        assert_eq!(d[2], vec![4., 4., 5., 5., 6.]);
    }
}
