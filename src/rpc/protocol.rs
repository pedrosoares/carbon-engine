// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Wire protocol between the coordinator (`carbon serve`) and RPC workers
//! (`carbon worker`): length-prefixed bincode frames over TCP.

use crate::config::ModelConfig;
use crate::model::moe::MoeOptions;
use crate::model::StageStats;
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

pub const PROTOCOL_VERSION: u32 = 9;
const MAX_FRAME: u64 = 64 << 30;

#[derive(Clone, Serialize, Deserialize)]
pub struct WireTensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

impl std::fmt::Debug for WireTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "WireTensor({}, {:?}, {} bytes)",
            self.dtype,
            self.shape,
            self.data.len()
        )
    }
}

impl WireTensor {
    pub fn from_tensor(t: &Tensor) -> Result<Self> {
        let t = t.to_device(&Device::Cpu)?.contiguous()?;
        let shape = t.dims().to_vec();
        let flat = t.flatten_all()?;
        let (dtype, data): (&str, Vec<u8>) = match t.dtype() {
            DType::F32 => (
                "f32",
                flat.to_vec1::<f32>()?
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect(),
            ),
            DType::BF16 => (
                "bf16",
                flat.to_vec1::<bf16>()?
                    .iter()
                    .flat_map(|x| x.to_bits().to_le_bytes())
                    .collect(),
            ),
            DType::F16 => (
                "f16",
                flat.to_vec1::<f16>()?
                    .iter()
                    .flat_map(|x| x.to_bits().to_le_bytes())
                    .collect(),
            ),
            DType::U32 => (
                "u32",
                flat.to_vec1::<u32>()?
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect(),
            ),
            DType::U8 => ("u8", flat.to_vec1::<u8>()?),
            DType::F8E4M3 => (
                "f8e4m3",
                flat.to_vec1::<float8::F8E4M3>()?
                    .iter()
                    .map(|x| x.to_bits())
                    .collect(),
            ),
            d => bail!("cannot serialise dtype {d:?}"),
        };
        Ok(Self {
            dtype: dtype.into(),
            shape,
            data,
        })
    }

    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        let dtype = match self.dtype.as_str() {
            "f32" => DType::F32,
            "bf16" => DType::BF16,
            "f16" => DType::F16,
            "u32" => DType::U32,
            "u8" => DType::U8,
            "f8e4m3" => DType::F8E4M3,
            d => bail!("unknown wire dtype {d}"),
        };
        let expected = self.shape.iter().product::<usize>() * dtype.size_in_bytes();
        if expected != self.data.len() {
            bail!(
                "wire tensor has {} bytes, expected {expected}",
                self.data.len()
            );
        }
        Ok(
            Tensor::from_raw_buffer(&self.data, dtype, &self.shape, &Device::Cpu)?
                .to_device(device)?,
        )
    }
}

/// A sequence in a forward request. Slot tables are append-only, so only the
/// slots the worker has not seen yet are sent (`slots_from..kv_len`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireSeq {
    pub seq_id: u64,
    pub q_len: u32,
    pub kv_len: u32,
    pub slots_from: u32,
    pub new_slots: Vec<u32>,
    pub state_slot: u32,
    /// Tokens right before the new ones (Qwen3.8 PLE n-gram context).
    pub prev_tokens: Vec<u32>,
    /// `SeqMeta::token_state_slots`.
    pub token_state_slots: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum Request {
    Hello {
        version: u32,
        /// Shared secret; required when the worker was started with `--token`.
        token: Option<String>,
    },
    /// Weights for a worker started without a local model copy.
    PutWeights(Vec<(String, WireTensor)>),
    Load {
        config: ModelConfig,
        layers: (usize, usize),
        dtype: Option<String>,
        moe: MoeOptions,
        max_seq_len: usize,
        use_shipped_weights: bool,
    },
    KvCapacity {
        memory_ratio: f64,
    },
    AllocKv {
        num_slots: usize,
        num_states: usize,
    },
    CopyState {
        src: u32,
        dst: u32,
    },
    ResizeExpertCache {
        experts: usize,
    },
    Forward {
        positions: Vec<u32>,
        out_loc: Vec<u32>,
        seqs: Vec<WireSeq>,
        is_decode: bool,
        /// Token ids of the new positions (empty unless the model's layers need them).
        tokens: Vec<u32>,
        /// `BatchMeta::mrope` (empty without images).
        mrope: Vec<[u32; 3]>,
        /// `BatchMeta::mrope_layout`.
        mrope_layout: Vec<crate::images::MropePositions>,
        hidden: WireTensor,
    },
    Release {
        seq_ids: Vec<u64>,
    },
    Stats,
    Ping,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    Hello {
        version: u32,
        name: String,
        devices: Vec<String>,
        has_model: bool,
    },
    Ok,
    Capacity(Option<usize>),
    Hidden(WireTensor),
    Stats(Vec<StageStats>),
    Pong,
    Error(String),
}

pub fn write_frame<W: Write, T: Serialize>(w: &mut W, msg: &T) -> Result<()> {
    let bytes = bincode::serialize(msg)?;
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    w.write_all(&bytes)?;
    w.flush()?;
    Ok(())
}

pub fn read_frame<R: Read, T: for<'de> Deserialize<'de>>(r: &mut R) -> Result<T> {
    let mut len = [0u8; 8];
    r.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len);
    if len > MAX_FRAME {
        bail!("rpc frame of {len} bytes exceeds the limit");
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    Ok(bincode::deserialize(&buf)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_roundtrip() {
        for dt in [DType::F32, DType::BF16, DType::F16, DType::F8E4M3] {
            let t = Tensor::arange(0f32, 12.0, &Device::Cpu)
                .unwrap()
                .reshape((3, 4))
                .unwrap()
                .to_dtype(dt)
                .unwrap();
            let w = WireTensor::from_tensor(&t).unwrap();
            let back = w.to_tensor(&Device::Cpu).unwrap();
            assert_eq!(back.dtype(), dt);
            assert_eq!(back.dims(), &[3, 4]);
            let a = back
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(a, (0..12).map(|x| x as f32).collect::<Vec<_>>());
        }
    }

    #[test]
    fn frame_roundtrip() {
        let mut buf = vec![];
        write_frame(
            &mut buf,
            &Request::Release {
                seq_ids: vec![1, 2, 3],
            },
        )
        .unwrap();
        let r: Request = read_frame(&mut buf.as_slice()).unwrap();
        assert!(matches!(r, Request::Release { seq_ids } if seq_ids == vec![1, 2, 3]));
    }
}
