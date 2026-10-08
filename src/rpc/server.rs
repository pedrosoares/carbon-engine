// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! `carbon worker`: serves a range of decoder layers to a coordinator over TCP.
//!
//! The worker owns one or more local devices. Its assigned layer range is split
//! into pipeline sub-stages over device groups of `tp_size` devices each; a group
//! with more than one device runs tensor/expert parallel.

use super::protocol::{read_frame, write_frame, Request, Response, WireTensor, PROTOCOL_VERSION};
use crate::batch::{BatchMeta, SeqMeta};
use crate::model::moe::MoeOptions;
use crate::parallel::{build_group, split_layers, BuildCtx, PipelineStage, Stage};
use crate::weights::{MapSource, WeightSource};
use anyhow::{bail, Result};
use candle_core::Device;
use std::collections::HashMap;
use std::io::{BufReader, BufWriter};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct WorkerConfig {
    pub listen: String,
    pub devices: Vec<String>,
    pub tp_size: usize,
    pub model_dir: Option<PathBuf>,
    pub name: String,
    /// Default MoE options; the coordinator's options win when it sends any.
    pub moe: MoeOptions,
    /// Shared secret coordinators must present (None = no authentication).
    pub token: Option<String>,
    /// Capture this worker's single-GPU decode steps as CUDA graphs.
    pub cuda_graphs: bool,
}

struct Session {
    shipped: Arc<MapSource>,
    stage: Option<Box<dyn Stage>>,
    tables: HashMap<u64, Vec<u32>>,
}

pub fn serve(cfg: WorkerConfig) -> Result<()> {
    if cfg.token.is_none() {
        tracing::warn!(
            "no --token set: any host that can reach {} can drive this worker",
            cfg.listen
        );
    }
    if cfg.tp_size == 0 || !cfg.devices.len().is_multiple_of(cfg.tp_size) {
        bail!(
            "{} devices cannot be split into tensor-parallel groups of {}",
            cfg.devices.len(),
            cfg.tp_size
        );
    }
    let listener = TcpListener::bind(&cfg.listen)?;
    tracing::info!(
        "carbon worker '{}' listening on {} with devices {:?} (tp {})",
        cfg.name,
        listener.local_addr()?,
        cfg.devices,
        cfg.tp_size
    );
    serve_listener(listener, cfg)
}

pub fn serve_listener(listener: TcpListener, cfg: WorkerConfig) -> Result<()> {
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("accept failed: {e}");
                continue;
            }
        };
        let peer = stream
            .peer_addr()
            .map(|a| a.to_string())
            .unwrap_or_default();
        tracing::info!("coordinator connected from {peer}");
        // one coordinator at a time: a new connection starts a fresh session
        match handle(stream, &cfg) {
            Ok(()) => tracing::info!("coordinator {peer} disconnected"),
            Err(e) => tracing::warn!("session with {peer} ended: {e:#}"),
        }
    }
    Ok(())
}

fn handle(stream: TcpStream, cfg: &WorkerConfig) -> Result<()> {
    stream.set_nodelay(true)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = BufWriter::new(stream);
    let mut s = Session {
        shipped: Arc::new(MapSource::default()),
        stage: None,
        tables: HashMap::new(),
    };
    let mut authed = false;
    loop {
        let req: Request = match read_frame(&mut reader) {
            Ok(r) => r,
            Err(e) => {
                if let Some(io) = e.downcast_ref::<std::io::Error>() {
                    if io.kind() == std::io::ErrorKind::UnexpectedEof {
                        return Ok(());
                    }
                }
                return Err(e);
            }
        };
        if let Request::Hello { token, .. } = &req {
            if !token_matches(cfg.token.as_deref(), token.as_deref()) {
                write_frame(
                    &mut writer,
                    &Response::Error("authentication failed: bad or missing token".into()),
                )?;
                anyhow::bail!("rejected coordinator with a bad token");
            }
            authed = true;
        } else if !authed {
            write_frame(&mut writer, &Response::Error("send Hello first".into()))?;
            anyhow::bail!("coordinator skipped the handshake");
        }
        let resp = match process(&mut s, cfg, req) {
            Ok(r) => r,
            Err(e) => Response::Error(format!("{e:#}")),
        };
        write_frame(&mut writer, &resp)?;
    }
}

fn process(s: &mut Session, cfg: &WorkerConfig, req: Request) -> Result<Response> {
    Ok(match req {
        Request::Hello { .. } => Response::Hello {
            version: PROTOCOL_VERSION,
            name: cfg.name.clone(),
            devices: cfg.devices.clone(),
            has_model: cfg.model_dir.is_some(),
        },
        Request::Ping => Response::Pong,
        Request::PutWeights(list) => {
            for (name, w) in list {
                s.shipped.insert(name, w.to_tensor(&Device::Cpu)?);
            }
            Response::Ok
        }
        Request::Load {
            config,
            layers,
            dtype,
            moe,
            max_seq_len,
            use_shipped_weights,
        } => {
            s.stage = None;
            s.tables.clear();
            let src: Arc<dyn WeightSource> = if use_shipped_weights {
                s.shipped.clone()
            } else {
                match &cfg.model_dir {
                    Some(dir) => crate::checkpoint::open_weights(dir)?,
                    None => bail!("worker has no --model and no weights were shipped"),
                }
            };
            let dtype = match dtype {
                Some(d) => crate::device::parse_dtype(&d)?,
                None => None,
            };
            let moe = if moe.strategy == crate::model::moe::MoeStrategy::Auto
                && moe.cache_size.is_none()
            {
                cfg.moe.clone()
            } else {
                moe
            };
            let ctx = BuildCtx {
                src,
                cfg: Arc::new(config),
                dtype,
                moe,
                max_seq_len,
                cuda_graphs: cfg.cuda_graphs,
            };
            let range = layers.0..layers.1;
            let groups: Vec<Vec<String>> = cfg
                .devices
                .chunks(cfg.tp_size)
                .map(|c| c.to_vec())
                .collect();
            let ranges = split_layers(range.len(), groups.len(), &[])?;
            let mut stages: Vec<Box<dyn Stage>> = vec![];
            for (g, r) in groups.iter().zip(ranges) {
                let r = (range.start + r.start)..(range.start + r.end);
                tracing::info!("loading layers {}..{} on {:?}", r.start, r.end, g);
                stages.push(build_group(&ctx, r, g)?);
            }
            s.stage = Some(if stages.len() == 1 {
                stages.pop().unwrap()
            } else {
                Box::new(PipelineStage { stages })
            });
            // shipped weights are now copied into the stage; free the host copies
            s.shipped = Arc::new(MapSource::default());
            Response::Ok
        }
        Request::KvCapacity { memory_ratio } => {
            Response::Capacity(stage(s)?.kv_capacity(memory_ratio)?)
        }
        Request::AllocKv {
            num_slots,
            num_states,
        } => {
            s.tables.clear();
            stage(s)?.alloc_kv(num_slots, num_states)?;
            tracing::info!("allocated KV cache for {num_slots} tokens");
            Response::Ok
        }
        Request::CopyState { src, dst } => {
            stage(s)?.copy_state(src, dst)?;
            Response::Ok
        }
        Request::ResizeExpertCache { experts } => {
            stage(s)?.resize_expert_cache(experts)?;
            Response::Ok
        }
        Request::Forward {
            positions,
            out_loc,
            seqs,
            is_decode,
            tokens,
            mrope,
            mrope_layout,
            hidden,
        } => {
            let mut metas = Vec::with_capacity(seqs.len());
            for w in seqs {
                let table = s.tables.entry(w.seq_id).or_default();
                if (w.slots_from as usize) > table.len() {
                    bail!(
                        "seq {} slot table out of sync ({} > {})",
                        w.seq_id,
                        w.slots_from,
                        table.len()
                    );
                }
                table.truncate(w.slots_from as usize);
                table.extend_from_slice(&w.new_slots);
                if table.len() != w.kv_len as usize {
                    bail!(
                        "seq {} has {} slots, expected {}",
                        w.seq_id,
                        table.len(),
                        w.kv_len
                    );
                }
                metas.push(SeqMeta {
                    seq_id: w.seq_id,
                    q_len: w.q_len as usize,
                    kv_len: w.kv_len as usize,
                    state_slot: w.state_slot,
                    slots: Arc::new(table.clone()),
                    prev_tokens: w.prev_tokens.clone(),
                    token_state_slots: w.token_state_slots.clone(),
                });
            }
            let meta = Arc::new(BatchMeta {
                positions,
                out_loc,
                seqs: metas,
                is_decode,
                tokens,
                mrope,
                mrope_layout,
            });
            meta.validate()?;
            let h = hidden.to_tensor(&Device::Cpu)?;
            let wire_dtype = h.dtype();
            let out = stage(s)?.forward(&meta, h)?;
            Response::Hidden(WireTensor::from_tensor(&out.to_dtype(wire_dtype)?)?)
        }
        Request::Release { seq_ids } => {
            for id in seq_ids {
                s.tables.remove(&id);
            }
            Response::Ok
        }
        Request::Stats => Response::Stats(stage(s)?.stats()),
    })
}

/// Constant-time comparison of the presented token against the configured one.
fn token_matches(expected: Option<&str>, got: Option<&str>) -> bool {
    match expected {
        None => true,
        Some(e) => {
            let g = got.unwrap_or("");
            e.len() == g.len()
                && e.bytes()
                    .zip(g.bytes())
                    .fold(0u8, |acc, (a, b)| acc | (a ^ b))
                    == 0
        }
    }
}

fn stage(s: &Session) -> Result<&dyn Stage> {
    s.stage
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("no layers loaded on this worker"))
}
