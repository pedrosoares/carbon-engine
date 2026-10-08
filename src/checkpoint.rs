// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! carbon weight checkpoints (`carbon checkpoint`): a whole model as one aligned byte region.
//!
//! safetensors files pack tensors back to back with no alignment, so a tensor cannot be read
//! with O_DIRECT at an arbitrary offset and a model's long tail of tiny tensors turns into
//! hundreds of small page-cache reads. Here every tensor starts on a 4096-byte boundary and is
//! padded to one; the region is cut into shard files (default 8 GiB) at aligned offsets. A
//! tensor is then read with several parallel O_DIRECT requests (deep queue on NVMe, no page
//! cache double-buffering) straight into an aligned buffer.
//!
//! ```text
//! <dir>/carbon_weight.json   index: tensors (name, dtype, shape, offset, bytes) + shards
//! <dir>/carbon-00000.cw      the byte region, sliced into <= shard-limit files
//! <dir>/config.json, tokenizer*, chat templates ...   copied: the dir is a full checkpoint
//! ```
//!
//! Tensors are stored exactly as in the source (quantized weights stay quantized) unless a
//! float dtype is requested for floating-point tensors.

use crate::weights::{SafetensorsSource, WeightSource};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const INDEX_NAME: &str = "carbon_weight.json";
pub const FORMAT: &str = "carbon_weight";
pub const ALIGN: u64 = 4096;
pub const DEFAULT_SHARD_LIMIT: u64 = 8 << 30;
/// Bytes per O_DIRECT request when a tensor is read in parallel pieces.
const CHUNK: usize = 8 << 20;

fn shard_name(i: usize) -> String {
    format!("carbon-{i:05}.cw")
}

fn align_up(x: u64) -> u64 {
    x.div_ceil(ALIGN) * ALIGN
}

pub fn is_checkpoint(dir: &Path) -> bool {
    dir.join(INDEX_NAME).is_file()
}

/// Raw little-endian bytes of a CPU tensor.
fn tensor_bytes(t: &Tensor) -> Result<Vec<u8>> {
    let t = t.flatten_all()?.to_device(&Device::Cpu)?;
    fn bytes<T: bytemuck_like::Pod>(v: Vec<T>) -> Vec<u8> {
        let n = std::mem::size_of_val(v.as_slice());
        let mut out = vec![0u8; n];
        // SAFETY: plain-old-data element types, copied byte for byte
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, out.as_mut_ptr(), n) };
        out
    }
    Ok(match t.dtype() {
        DType::U8 => t.to_vec1::<u8>()?,
        DType::U32 => bytes(t.to_vec1::<u32>()?),
        DType::I64 => bytes(t.to_vec1::<i64>()?),
        DType::BF16 => bytes(t.to_vec1::<half::bf16>()?),
        DType::F16 => bytes(t.to_vec1::<half::f16>()?),
        DType::F32 => bytes(t.to_vec1::<f32>()?),
        DType::F64 => bytes(t.to_vec1::<f64>()?),
        DType::F8E4M3 => bytes(t.to_vec1::<float8::F8E4M3>()?),
        d => bail!("cannot serialise dtype {d:?}"),
    })
}

mod bytemuck_like {
    /// Marker for element types whose bytes can be copied out directly.
    pub trait Pod: Copy {}
    impl Pod for u32 {}
    impl Pod for i64 {}
    impl Pod for f32 {}
    impl Pod for f64 {}
    impl Pod for half::bf16 {}
    impl Pod for half::f16 {}
    impl Pod for float8::F8E4M3 {}
}

pub struct ConvertOptions {
    /// Convert floating-point tensors (f32 / f16 / bf16, not FP8 codes) to this dtype.
    pub dtype: Option<DType>,
    pub shard_limit: u64,
    /// Print `CARBONCONVERT <done> <total>` progress lines (for supervisors).
    pub progress: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            dtype: None,
            shard_limit: DEFAULT_SHARD_LIMIT,
            progress: false,
        }
    }
}

/// Files other than weights that make the output a complete checkpoint.
fn copy_side_files(src: &Path, out: &Path) -> Result<()> {
    for e in std::fs::read_dir(src)?.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let keep = (name.ends_with(".json") && name != "model.safetensors.index.json" && name != INDEX_NAME)
            || name.ends_with(".jinja")
            || name.ends_with(".tiktoken")
            || name == "tokenizer.model"
            || (name.ends_with(".txt") && (name.starts_with("merges") || name.starts_with("vocab")));
        if keep && e.path().is_file() {
            std::fs::copy(e.path(), out.join(&name))?;
        }
    }
    Ok(())
}

/// Convert the safetensors checkpoint in `src` into a carbon weight checkpoint in `out`.
pub fn convert(src: &Path, out: &Path, opts: &ConvertOptions) -> Result<Value> {
    if opts.shard_limit < ALIGN || !opts.shard_limit.is_multiple_of(ALIGN) {
        bail!("shard limit must be a positive multiple of {ALIGN}");
    }
    if is_checkpoint(src) {
        bail!("{} is already a carbon weight checkpoint", src.display());
    }
    let source = SafetensorsSource::open(src)?;
    std::fs::create_dir_all(out)?;
    // remove shards of an earlier conversion first
    for e in std::fs::read_dir(out)?.flatten() {
        if e.file_name().to_string_lossy().ends_with(".cw") {
            std::fs::remove_file(e.path())?;
        }
    }
    let mut names = source.names();
    names.sort();
    let total = names.len();
    let mut tensors = vec![];
    let mut offset: u64 = 0;
    let mut shard = 0usize;
    let mut file = std::fs::File::create(out.join(shard_name(0)))?;
    let mut in_shard: u64 = 0;
    let mut shards = vec![];
    let pad = vec![0u8; ALIGN as usize];
    for (i, name) in names.iter().enumerate() {
        let t = source.load_raw(name)?;
        let t = match opts.dtype {
            Some(d) if matches!(t.dtype(), DType::F32 | DType::F16 | DType::BF16) && t.dtype() != d => {
                t.to_dtype(d)?
            }
            _ => t,
        };
        let data = tensor_bytes(&t)?;
        let padded = align_up(data.len() as u64);
        tensors.push(json!({
            "name": name,
            "dtype": t.dtype().as_str(),
            "shape": t.dims(),
            "offset": offset,
            "bytes": data.len(),
        }));
        // write `data` (+ padding), cutting shards at aligned boundaries
        let mut buf: &[u8] = &data;
        let mut remaining_pad = (padded - data.len() as u64) as usize;
        while !buf.is_empty() || remaining_pad > 0 {
            if in_shard == opts.shard_limit {
                file.sync_all()?;
                shards.push(json!({"file": shard_name(shard), "bytes": in_shard}));
                shard += 1;
                file = std::fs::File::create(out.join(shard_name(shard)))?;
                in_shard = 0;
            }
            let room = (opts.shard_limit - in_shard) as usize;
            if !buf.is_empty() {
                let n = buf.len().min(room);
                file.write_all(&buf[..n])?;
                buf = &buf[n..];
                in_shard += n as u64;
            } else {
                let n = remaining_pad.min(room);
                file.write_all(&pad[..n])?;
                remaining_pad -= n;
                in_shard += n as u64;
            }
        }
        offset += padded;
        if opts.progress {
            println!("CARBONCONVERT {} {total}", i + 1);
        } else if (i + 1) % 200 == 0 || i + 1 == total {
            tracing::info!("{}/{total} tensors, {:.2} GiB", i + 1, offset as f64 / (1u64 << 30) as f64);
        }
    }
    file.sync_all()?;
    shards.push(json!({"file": shard_name(shard), "bytes": in_shard}));
    copy_side_files(src, out)?;
    let index = json!({
        "format": FORMAT,
        "version": 1,
        "align": ALIGN,
        "shard_limit": opts.shard_limit,
        "total_bytes": offset,
        "shards": shards,
        "tensors": tensors,
        "meta": {
            "source": src.to_string_lossy(),
            "created": crate::server::metrics::utc_timestamp(),
            "dtype": opts.dtype.map(|d| d.as_str()),
        },
    });
    let tmp = out.join(format!("{INDEX_NAME}.tmp"));
    std::fs::write(&tmp, serde_json::to_string_pretty(&index)?)?;
    std::fs::rename(tmp, out.join(INDEX_NAME))?;
    Ok(index)
}

#[derive(Clone)]
struct Entry {
    dtype: DType,
    shape: Vec<usize>,
    offset: u64,
    bytes: usize,
}

/// A carbon weight checkpoint as a weight source.
pub struct CarbonWeightSource {
    entries: HashMap<String, Entry>,
    shards: Vec<(PathBuf, u64)>,
    shard_limit: u64,
    direct: std::sync::atomic::AtomicBool,
    /// Shard files mapped for row access (PLE tables).
    maps: parking_lot::Mutex<HashMap<usize, std::sync::Arc<memmap2::Mmap>>>,
}

/// A 4096-aligned heap buffer (O_DIRECT destination).
struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    fn new(len: usize) -> Self {
        let len = (len.max(1) as u64).div_ceil(ALIGN) as usize * ALIGN as usize;
        let layout = std::alloc::Layout::from_size_align(len, ALIGN as usize).unwrap();
        // SAFETY: non-zero size, valid alignment; freed in Drop with the same layout
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "out of memory allocating {len} bytes");
        Self { ptr, len }
    }
    fn slice(&self) -> &[u8] {
        // SAFETY: ptr is a live allocation of len bytes
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    /// Disjoint mutable pieces for parallel reads.
    #[allow(clippy::mut_from_ref)]
    fn piece(&self, start: usize, len: usize) -> &mut [u8] {
        assert!(start + len <= self.len);
        // SAFETY: callers hand out non-overlapping ranges, one per reader
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, ALIGN as usize).unwrap();
        // SAFETY: allocated in new() with this layout
        unsafe { std::alloc::dealloc(self.ptr, layout) }
    }
}

fn read_at(f: &std::fs::File, buf: &mut [u8], mut off: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        let n = f.read_at(&mut buf[done..], off)?;
        if n == 0 {
            // end of file inside the padding of the last tensor: the rest stays zero
            break;
        }
        done += n;
        off += n as u64;
    }
    Ok(())
}

impl CarbonWeightSource {
    pub fn open(dir: &Path) -> Result<Self> {
        let index: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join(INDEX_NAME))
                .with_context(|| format!("reading {}", dir.join(INDEX_NAME).display()))?,
        )?;
        if index["format"] != FORMAT {
            bail!("{} is not a carbon weight index", dir.display());
        }
        let shards = index["shards"]
            .as_array()
            .context("index has no shards")?
            .iter()
            .map(|s| {
                (
                    dir.join(s["file"].as_str().unwrap_or("")),
                    s["bytes"].as_u64().unwrap_or(0),
                )
            })
            .collect();
        let mut entries = HashMap::new();
        for t in index["tensors"].as_array().context("index has no tensors")? {
            let dtype: DType = t["dtype"]
                .as_str()
                .unwrap_or("")
                .parse()
                .map_err(|e| anyhow::anyhow!("bad dtype in index: {e:?}"))?;
            entries.insert(
                t["name"].as_str().unwrap_or("").to_string(),
                Entry {
                    dtype,
                    shape: t["shape"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
                        .unwrap_or_default(),
                    offset: t["offset"].as_u64().unwrap_or(0),
                    bytes: t["bytes"].as_u64().unwrap_or(0) as usize,
                },
            );
        }
        Ok(Self {
            entries,
            shards,
            shard_limit: index["shard_limit"].as_u64().unwrap_or(DEFAULT_SHARD_LIMIT),
            direct: std::sync::atomic::AtomicBool::new(
                std::env::var("CARBON_O_DIRECT").map_or(true, |v| v != "0"),
            ),
            maps: Default::default(),
        })
    }

    fn open_shard(&self, i: usize) -> std::io::Result<std::fs::File> {
        use std::os::unix::fs::OpenOptionsExt;
        let path = &self.shards[i].0;
        if self.direct.load(std::sync::atomic::Ordering::Relaxed) {
            match std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(path)
            {
                Ok(f) => return Ok(f),
                // tmpfs and some network filesystems refuse O_DIRECT
                Err(_) => self.direct.store(false, std::sync::atomic::Ordering::Relaxed),
            }
        }
        std::fs::File::open(path)
    }

    /// Read `len` aligned bytes of the region at aligned `offset`, in parallel pieces.
    fn read_region(&self, offset: u64, len: usize) -> Result<AlignedBuf> {
        use rayon::prelude::*;
        let buf = AlignedBuf::new(len);
        let padded = buf.len;
        // pieces never straddle a shard boundary (shard_limit is a multiple of ALIGN)
        let mut pieces = vec![];
        let mut pos = 0usize;
        while pos < padded {
            let abs = offset + pos as u64;
            let shard = (abs / self.shard_limit) as usize;
            let in_shard = abs % self.shard_limit;
            let n = CHUNK
                .min(padded - pos)
                .min((self.shard_limit - in_shard) as usize);
            pieces.push((shard, in_shard, pos, n));
            pos += n;
        }
        let result: std::io::Result<()> = pieces.par_iter().try_for_each(|&(shard, off, at, n)| {
            let f = self.open_shard(shard)?;
            match read_at(&f, buf.piece(at, n), off) {
                Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
                    // O_DIRECT rejected at read time: plain reads from here on
                    self.direct.store(false, std::sync::atomic::Ordering::Relaxed);
                    read_at(&std::fs::File::open(&self.shards[shard].0)?, buf.piece(at, n), off)
                }
                r => r,
            }
        });
        result.context("reading carbon weight shards")?;
        Ok(buf)
    }
}

impl WeightSource for CarbonWeightSource {
    fn row_reader(&self, name: &str) -> Option<Result<crate::weights::RowReader>> {
        let e = self.entries.get(name)?;
        let shard = (e.offset / self.shard_limit) as usize;
        let local = e.offset % self.shard_limit;
        if local + e.bytes as u64 > self.shard_limit || e.shape.len() != 2 {
            return None; // spans two shards: loaded instead
        }
        let map = {
            let mut maps = self.maps.lock();
            match maps.get(&shard) {
                Some(m) => m.clone(),
                None => {
                    // SAFETY: read-only map of a checkpoint file that is not modified while serving
                    let m = match std::fs::File::open(&self.shards.get(shard)?.0)
                        .and_then(|f| unsafe { memmap2::Mmap::map(&f) })
                    {
                        Ok(m) => m,
                        Err(e) => return Some(Err(e.into())),
                    };
                    let preload = std::env::var("CARBON_PLE_PRELOAD").is_ok_and(|v| v == "1");
                    let _ = m.advise(if preload { memmap2::Advice::WillNeed } else { memmap2::Advice::Random });
                    let m = std::sync::Arc::new(m);
                    maps.insert(shard, m.clone());
                    m
                }
            }
        };
        Some(crate::weights::RowReader::new(map, local as usize, e.shape.clone(), e.dtype))
    }

    fn load_raw(&self, name: &str) -> Result<Tensor> {
        let e = self
            .entries
            .get(name)
            .with_context(|| format!("weight '{name}' is not in the checkpoint"))?;
        let buf = self.read_region(e.offset, e.bytes)?;
        Ok(Tensor::from_raw_buffer(
            &buf.slice()[..e.bytes],
            e.dtype,
            &e.shape,
            &Device::Cpu,
        )?)
    }
    fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }
    fn names(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }
}

/// The weight source for a model directory: carbon checkpoint or safetensors.
pub fn open_weights(dir: &Path) -> Result<std::sync::Arc<dyn WeightSource>> {
    Ok(if is_checkpoint(dir) {
        std::sync::Arc::new(CarbonWeightSource::open(dir)?)
    } else {
        std::sync::Arc::new(SafetensorsSource::open(dir)?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{write_tiny_quantized, TinyQuant, TinySpec};

    #[test]
    fn roundtrip_with_small_shards() {
        let d = tempfile::tempdir().unwrap();
        let (src, dense, out) = (d.path().join("q"), d.path().join("d"), d.path().join("c"));
        write_tiny_quantized(&src, &dense, &TinySpec::new("qwen3_moe"), TinyQuant::Fp8Block).unwrap();
        // 64 KiB shards: tensors span shard boundaries
        let index = convert(&src, &out, &ConvertOptions { shard_limit: 16 * ALIGN, ..Default::default() }).unwrap();
        assert!(index["shards"].as_array().unwrap().len() > 2);
        assert!(out.join("config.json").is_file() && out.join("tokenizer.json").is_file());
        let a = SafetensorsSource::open(&src).unwrap();
        let b = CarbonWeightSource::open(&out).unwrap();
        let mut names = a.names();
        names.sort();
        let mut bn = b.names();
        bn.sort();
        assert_eq!(names, bn);
        for n in &names {
            let (x, y) = (a.load_raw(n).unwrap(), b.load_raw(n).unwrap());
            assert_eq!((x.dtype(), x.dims()), (y.dtype(), y.dims()), "{n}");
            assert_eq!(tensor_bytes(&x).unwrap(), tensor_bytes(&y).unwrap(), "{n}");
        }
        // where the filesystem takes O_DIRECT, every read above used it (no silent fallback)
        let direct_ok = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT)
                .open(out.join(shard_name(0))).is_ok()
        };
        if direct_ok && std::env::var("CARBON_O_DIRECT").as_deref() != Ok("0") {
            assert!(b.direct.load(std::sync::atomic::Ordering::Relaxed));
        }
        // row access through the mapped shards matches the loaded tensor
        for n in names.iter().filter(|n| {
            let t = a.load_raw(n).unwrap();
            t.dims().len() == 2 && matches!(t.dtype(), DType::F32 | DType::F8E4M3)
        }) {
            let Some(r) = b.row_reader(n) else { continue };
            let r = r.unwrap();
            let full = a.load_raw(n).unwrap().to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap();
            for (i, want) in full.iter().enumerate() {
                let mut got = vec![];
                r.row_f32(i, &mut got).unwrap();
                assert_eq!(&got, want, "{n} row {i}");
            }
        }
        // FP8 codes stayed FP8
        assert!(index["tensors"].as_array().unwrap().iter().any(|t| t["dtype"] == "f8e4m3"));
        // a float conversion leaves quantized codes alone
        let out16 = d.path().join("c16");
        convert(&dense, &out16, &ConvertOptions { dtype: Some(DType::BF16), ..Default::default() }).unwrap();
        let c = CarbonWeightSource::open(&out16).unwrap();
        assert_eq!(c.load_raw("model.norm.weight").unwrap().dtype(), DType::BF16);
        assert!(convert(&out16, &d.path().join("x"), &ConvertOptions::default()).is_err());
    }
}
