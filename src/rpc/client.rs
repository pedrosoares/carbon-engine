// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Coordinator side of the RPC protocol: a pipeline stage executed by a remote worker.

use super::protocol::{
    read_frame, write_frame, Request, Response, WireSeq, WireTensor, PROTOCOL_VERSION,
};
use crate::batch::BatchMeta;
use crate::model::StageStats;
use crate::parallel::{BuildCtx, Stage};
use crate::weights::layer_weight_names;
use anyhow::{anyhow, bail, Context, Result};
use candle_core::{Device, Tensor};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::io::{BufReader, BufWriter};
use std::net::TcpStream;
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

struct Conn {
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
}

pub struct WorkerInfo {
    pub name: String,
    pub devices: Vec<String>,
    pub has_model: bool,
}

pub struct RemoteStage {
    pub addr: String,
    range: Range<usize>,
    conn: Mutex<Conn>,
    /// Per sequence, how many slots the worker already holds.
    synced: Mutex<HashMap<u64, usize>>,
    pub info: WorkerInfo,
}

impl RemoteStage {
    fn connect(addr: &str, token: Option<&str>) -> Result<(Conn, WorkerInfo)> {
        let stream = connect_retry(addr)?;
        stream.set_nodelay(true)?;
        let mut conn = Conn {
            reader: BufReader::new(stream.try_clone()?),
            writer: BufWriter::new(stream),
        };
        write_frame(
            &mut conn.writer,
            &Request::Hello {
                version: PROTOCOL_VERSION,
                token: token.map(String::from),
            },
        )?;
        match read_frame::<_, Response>(&mut conn.reader)? {
            Response::Hello {
                version,
                name,
                devices,
                has_model,
            } => {
                if version != PROTOCOL_VERSION {
                    bail!("worker {addr} speaks protocol v{version}, this build speaks v{PROTOCOL_VERSION}");
                }
                Ok((
                    conn,
                    WorkerInfo {
                        name,
                        devices,
                        has_model,
                    },
                ))
            }
            Response::Error(e) => bail!("worker {addr}: {e}"),
            other => bail!("worker {addr}: unexpected hello reply {other:?}"),
        }
    }

    /// Connect, ship weights if the worker has no local copy, and load `range` on the worker.
    pub fn setup(
        addr: &str,
        token: Option<&str>,
        ctx: &BuildCtx,
        range: Range<usize>,
    ) -> Result<Self> {
        let (conn, info) = Self::connect(addr, token)?;
        let stage = Self {
            addr: addr.to_string(),
            range: range.clone(),
            conn: Mutex::new(conn),
            synced: Mutex::new(HashMap::new()),
            info,
        };
        tracing::info!(
            "rpc worker {addr} ({}, devices {:?}) -> layers {}..{}",
            stage.info.name,
            stage.info.devices,
            range.start,
            range.end
        );
        let ship = !stage.info.has_model;
        if ship {
            let mut bytes = 0usize;
            for l in range.clone() {
                let mut batch = vec![];
                for name in layer_weight_names(&*ctx.src, &ctx.cfg.prefix, l) {
                    let t = ctx.src.load_raw(&name)?;
                    let w = WireTensor::from_tensor(&t)?;
                    bytes += w.data.len();
                    batch.push((name, w));
                }
                stage.call(Request::PutWeights(batch))?;
            }
            tracing::info!(
                "shipped {:.2} GiB of weights to {addr}",
                bytes as f64 / (1u64 << 30) as f64
            );
        }
        stage.call(Request::Load {
            config: (*ctx.cfg).clone(),
            layers: (range.start, range.end),
            dtype: ctx.dtype.map(|d| crate::device::dtype_name(d).to_string()),
            moe: ctx.moe.clone(),
            max_seq_len: ctx.max_seq_len,
            use_shipped_weights: ship,
        })?;
        Ok(stage)
    }

    fn call(&self, req: Request) -> Result<Response> {
        let mut c = self.conn.lock();
        write_frame(&mut c.writer, &req)
            .with_context(|| format!("sending to worker {}", self.addr))?;
        match read_frame::<_, Response>(&mut c.reader)
            .with_context(|| format!("reading from worker {}", self.addr))?
        {
            Response::Error(e) => Err(anyhow!("worker {}: {e}", self.addr)),
            r => Ok(r),
        }
    }

    pub fn ping(&self) -> Result<Duration> {
        let t = std::time::Instant::now();
        match self.call(Request::Ping)? {
            Response::Pong => Ok(t.elapsed()),
            r => bail!("unexpected reply {r:?}"),
        }
    }
}

fn connect_retry(addr: &str) -> Result<TcpStream> {
    let mut last = None;
    for attempt in 0..20 {
        match TcpStream::connect(addr) {
            Ok(s) => return Ok(s),
            Err(e) => {
                last = Some(e);
                std::thread::sleep(Duration::from_millis(100 * (attempt + 1).min(10)));
            }
        }
    }
    Err(anyhow!(
        "could not connect to rpc worker {addr}: {}",
        last.unwrap()
    ))
}

impl Stage for RemoteStage {
    fn layers(&self) -> Range<usize> {
        self.range.clone()
    }

    fn forward(&self, meta: &Arc<BatchMeta>, hidden: Tensor) -> Result<Tensor> {
        let mut synced = self.synced.lock();
        let seqs: Vec<WireSeq> = meta
            .seqs
            .iter()
            .map(|s| {
                let from = synced.get(&s.seq_id).copied().unwrap_or(0).min(s.kv_len);
                WireSeq {
                    seq_id: s.seq_id,
                    q_len: s.q_len as u32,
                    kv_len: s.kv_len as u32,
                    slots_from: from as u32,
                    new_slots: s.slots[from..s.kv_len].to_vec(),
                    state_slot: s.state_slot,
                    prev_tokens: s.prev_tokens.clone(),
                    token_state_slots: s.token_state_slots.clone(),
                }
            })
            .collect();
        let resp = self.call(Request::Forward {
            positions: meta.positions.clone(),
            out_loc: meta.out_loc.clone(),
            seqs,
            is_decode: meta.is_decode,
            tokens: meta.tokens.clone(),
            mrope: meta.mrope.clone(),
            mrope_layout: meta.mrope_layout.clone(),
            hidden: WireTensor::from_tensor(&hidden)?,
        })?;
        // resending a delta is idempotent, so only record progress after success
        for s in &meta.seqs {
            synced.insert(s.seq_id, s.kv_len);
        }
        match resp {
            Response::Hidden(h) => h.to_tensor(&Device::Cpu),
            r => bail!("unexpected forward reply {r:?}"),
        }
    }

    fn release(&self, seq_ids: &[u64]) -> Result<()> {
        {
            let mut s = self.synced.lock();
            for id in seq_ids {
                s.remove(id);
            }
        }
        self.call(Request::Release {
            seq_ids: seq_ids.to_vec(),
        })?;
        Ok(())
    }

    fn kv_capacity(&self, memory_ratio: f64) -> Result<Option<usize>> {
        match self.call(Request::KvCapacity { memory_ratio })? {
            Response::Capacity(c) => Ok(c),
            r => bail!("unexpected reply {r:?}"),
        }
    }

    fn alloc_kv(&self, num_slots: usize, num_states: usize) -> Result<()> {
        self.synced.lock().clear();
        self.call(Request::AllocKv {
            num_slots,
            num_states,
        })?;
        Ok(())
    }

    fn copy_state(&self, src: u32, dst: u32) -> Result<()> {
        self.call(Request::CopyState { src, dst })?;
        Ok(())
    }

    fn resize_expert_cache(&self, experts: usize) -> Result<()> {
        self.call(Request::ResizeExpertCache { experts })?;
        Ok(())
    }

    fn stats(&self) -> Vec<StageStats> {
        match self.call(Request::Stats) {
            Ok(Response::Stats(mut s)) => {
                for x in s.iter_mut() {
                    x.name = format!("rpc:{}/{}", self.addr, x.name);
                }
                s
            }
            _ => vec![StageStats {
                name: format!("rpc:{} (unreachable)", self.addr),
                layers: (self.range.start, self.range.end),
                ..Default::default()
            }],
        }
    }
}
