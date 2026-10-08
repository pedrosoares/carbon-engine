// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Continuous-batching scheduler running on its own thread.
//!
//! * chunked prefill under a per-step token budget,
//! * radix-tree prefix reuse (prompt prefixes are shared between requests),
//! * admission control that reserves KV for a request's whole output budget, so
//!   running requests never need to be preempted,
//! * prefill and decode steps interleave so long prompts cannot starve decoding.

use crate::batch::{BatchMeta, SeqMeta, PREV_TOKENS};
use crate::engine::Engine;
use crate::kvcache::radix::{NodeId, ROOT};
use crate::kvcache::{RadixCache, SlotAllocator};
use crate::model::StageStats;
use crate::sampling::{apply_penalties, sample, SamplingParams};
use anyhow::Result;
use candle_core::Tensor;
use rand::rngs::StdRng;
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;
use tokio::sync::{mpsc as tmpsc, oneshot};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// EOS or a stop token.
    Stop,
    /// Output budget or context length exhausted.
    Length,
    /// Cancelled by the client (or a stop string matched in the server).
    Abort,
}

#[derive(Clone, Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
}

#[derive(Debug)]
pub enum Event {
    Token(u32),
    Done { reason: FinishReason, usage: Usage },
    Error(String),
}

pub struct Submit {
    pub id: u64,
    /// Image placeholders are pseudo ids (`images::expand_image_tokens`).
    pub prompt: Vec<u32>,
    pub images: Vec<crate::images::ImageSpan>,
    pub params: SamplingParams,
    pub tx: tmpsc::UnboundedSender<Event>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SchedulerStats {
    pub running: usize,
    pub waiting: usize,
    pub kv_capacity: usize,
    pub kv_free: usize,
    pub prefix_cached_tokens: usize,
    pub prefix_evictable_tokens: usize,
    /// Recurrent-state snapshots held for prefix reuse (hybrid models).
    pub prefix_snapshots: usize,
    pub total_requests: u64,
    pub total_prompt_tokens: u64,
    pub total_cached_tokens: u64,
    pub total_generated_tokens: u64,
    pub steps: u64,
    pub last_decode_tokens_per_s: f64,
    pub last_prefill_tokens_per_s: f64,
    /// Tokens per second over the last 5 s (0 when idle).
    pub decode_tps: f64,
    pub prefill_tps: f64,
    /// Host time between consecutive steps (the end of one step's sampling to the next
    /// forward), averaged: the device idles for this long per step unless work overlaps it.
    pub host_gap_us: f64,
    /// MTP shadow mode: drafts checked against the next sampled token, and how many matched.
    #[serde(default)]
    pub mtp_drafts: u64,
    #[serde(default)]
    pub mtp_accepted: u64,
    /// Time spent drafting, per draft step (ms, averaged).
    #[serde(default)]
    pub mtp_draft_ms: f64,
    /// MTP verify steps: forward + sampling time (ms, averaged).
    #[serde(default)]
    pub mtp_verify_ms: f64,
    pub stages: Vec<StageStats>,
}

/// Token rate over a trailing window, in coarse buckets (bounded memory at any rate).
#[derive(Default)]
pub struct RateWindow {
    buckets: std::collections::VecDeque<(Instant, usize)>,
}

impl RateWindow {
    const WINDOW: std::time::Duration = std::time::Duration::from_secs(5);
    const BUCKET: std::time::Duration = std::time::Duration::from_millis(50);

    pub fn add(&mut self, now: Instant, n: usize) {
        if n == 0 {
            return;
        }
        match self.buckets.back_mut() {
            Some((t, c)) if now.duration_since(*t) < Self::BUCKET => *c += n,
            _ => self.buckets.push_back((now, n)),
        }
        self.trim(now);
    }

    fn trim(&mut self, now: Instant) {
        while self
            .buckets
            .front()
            .is_some_and(|(t, _)| now.duration_since(*t) > Self::WINDOW)
        {
            self.buckets.pop_front();
        }
    }

    pub fn rate(&mut self, now: Instant) -> f64 {
        self.trim(now);
        let Some((first, _)) = self.buckets.front() else {
            return 0.0;
        };
        let total: usize = self.buckets.iter().map(|b| b.1).sum();
        // a lone burst counts over at least one bucket, never as an "infinite" rate
        total as f64 / now.duration_since(*first).max(Self::BUCKET).as_secs_f64()
    }
}

pub struct RebuildRequest {
    pub kv_tokens: Option<usize>,
    pub expert_cache: Option<usize>,
}

enum Cmd {
    Submit(Submit),
    Abort(u64),
    Stats(oneshot::Sender<SchedulerStats>),
    Rebuild(RebuildRequest, oneshot::Sender<Result<(), String>>),
}

/// Prior acceptance of a request's first and second MTP drafts, and the cycle cost of a second
/// draft relative to one (see `Req::wants_two_drafts`).
const MTP_PRIOR_ACC1: f32 = 0.8;
const MTP_PRIOR_ACC2: f32 = 0.75;
const MTP_TWO_DRAFT_COST: f32 = 1.3;

#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    pub max_running: usize,
    pub max_prefill_tokens: usize,
    pub eos_token_ids: Vec<u32>,
    /// `--enable-special-token-ckpt`: on hybrid models, a sampled token with this id (the
    /// tool-call opener) gets a prefix-cache reuse point right after it.
    pub anchor_token: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct EngineInfo {
    pub model_type: String,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub kv_tokens: usize,
    pub topology: Vec<String>,
    /// "mha" | "hybrid_linear"
    pub attn: String,
    pub moe: bool,
    /// Image input: the processor, the placeholder token and the tower's merge size.
    #[serde(skip)]
    pub image_processor: Option<crate::images::ImageProcessor>,
    #[serde(skip)]
    pub image_token_id: Option<u32>,
}

/// Cloneable handle used by the HTTP server to talk to the scheduler thread.
#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<Cmd>,
    next_id: Arc<AtomicU64>,
    pub info: Arc<EngineInfo>,
}

impl EngineHandle {
    pub fn submit(
        &self,
        prompt: Vec<u32>,
        params: SamplingParams,
    ) -> (u64, tmpsc::UnboundedReceiver<Event>) {
        self.submit_with_images(prompt, vec![], params)
    }

    /// `submit` for a prompt with images: `prompt` from `images::expand_image_tokens`, which
    /// also gives the spans.
    pub fn submit_with_images(
        &self,
        prompt: Vec<u32>,
        images: Vec<crate::images::ImageSpan>,
        params: SamplingParams,
    ) -> (u64, tmpsc::UnboundedReceiver<Event>) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = tmpsc::unbounded_channel();
        if self
            .tx
            .send(Cmd::Submit(Submit {
                id,
                prompt,
                images,
                params,
                tx: tx.clone(),
            }))
            .is_err()
        {
            let _ = tx.send(Event::Error("engine is not running".into()));
        }
        (id, rx)
    }

    pub fn abort(&self, id: u64) {
        let _ = self.tx.send(Cmd::Abort(id));
    }

    /// Resize the KV cache and/or expert caches. Admission pauses until running requests drain.
    pub async fn rebuild(&self, req: RebuildRequest) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Cmd::Rebuild(req, tx))
            .map_err(|_| "engine is not running".to_string())?;
        rx.await.map_err(|_| "engine is not running".to_string())?
    }

    pub async fn stats(&self) -> Option<SchedulerStats> {
        let (tx, rx) = oneshot::channel();
        self.tx.send(Cmd::Stats(tx)).ok()?;
        rx.await.ok()
    }
}

/// Penalties + host sampler for one logits row.
fn host_sample(l: &mut [f32], r: &mut Req) -> u32 {
    if r.params.has_penalties() {
        apply_penalties(l, &r.params, &r.tokens[..r.prompt_len], &r.tokens[r.prompt_len..]);
    }
    sample(l, &r.params, &mut r.rng)
}

#[cfg(feature = "cuda")]
static GPU_SAMPLING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// `CARBON_GPU_SAMPLING=0` keeps sampling on the host.
#[cfg(feature = "cuda")]
fn gpu_sampling_enabled() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var("CARBON_GPU_SAMPLING").map_or(true, |v| v != "0"))
        && GPU_SAMPLING.load(Ordering::Relaxed)
}

#[cfg(feature = "cuda")]
fn disable_gpu_sampling(e: &candle_core::Error) {
    if GPU_SAMPLING.swap(false, Ordering::Relaxed) {
        tracing::warn!("GPU sampling disabled, sampling on the host: {e}");
    }
}

struct ReqImage {
    span: crate::images::ImageSpan,
    /// The tower's output while prefill still has to reach the image's last token.
    features: Option<Tensor>,
}

struct Req {
    id: u64,
    /// Image placeholders are pseudo ids (see `images::pseudo_id`).
    tokens: Vec<u32>,
    images: Vec<ReqImage>,
    /// Rotary positions when the prompt has images.
    mrope: crate::images::MropePositions,
    /// The MTP head's guesses for the next tokens to be sampled (verified next step).
    mtp_drafts: Vec<u32>,
    /// The KV slot already assigned to the next position (it holds a further draft step's MTP
    /// KV; the next verify step writes the main KV there).
    mtp_next_slot: Option<u32>,
    /// Running acceptance of the first draft, and of the second given the first was accepted:
    /// they pick one or two drafts per step (see `wants_two_drafts`).
    mtp_acc1: f32,
    mtp_acc2: f32,
    prompt_len: usize,
    /// KV slot of tokens[i]; `slots.len()` tokens have KV.
    slots: Vec<u32>,
    /// Radix node locked by this request and the number of leading slots owned by the tree.
    node: NodeId,
    tree_owned: usize,
    cache_hit: usize,
    max_new: usize,
    params: SamplingParams,
    rng: StdRng,
    tx: tmpsc::UnboundedSender<Event>,
    /// Recurrent-state slot (hybrid models).
    state_slot: u32,
    /// Length of `tokens` just after a sampled anchor token, until that state is cached.
    anchor_len: Option<usize>,
}

impl Req {
    fn generated(&self) -> usize {
        self.tokens.len() - self.prompt_len
    }
    fn in_prefill(&self) -> bool {
        self.slots.len() < self.prompt_len
    }
    /// Tokens that will ever need KV: prompt + all outputs but the last.
    fn total_kv(&self) -> usize {
        self.prompt_len + self.max_new - 1
    }
    /// A second draft pays when its expected extra token outweighs the cost of a 3-token verify
    /// step over a 2-token one (measured ~1.3x per cycle on Qwen3.8-Flash-Next): two drafts
    /// yield `1 + a1 + a1 a2` tokens per cycle against `1 + a1`.
    fn wants_two_drafts(&self) -> bool {
        let (a1, a2) = (self.mtp_acc1, self.mtp_acc2);
        (1.0 + a1 + a1 * a2) > MTP_TWO_DRAFT_COST * (1.0 + a1)
    }

    fn remaining_kv(&self) -> usize {
        // a slot held for the next position (MTP) is already allocated
        let held = self.slots.len() + usize::from(self.mtp_next_slot.is_some());
        self.total_kv().saturating_sub(held)
    }
    fn usage(&self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_len,
            completion_tokens: self.generated(),
            cached_tokens: self.cache_hit,
        }
    }
}

pub struct Scheduler {
    engine: Engine,
    cfg: SchedulerConfig,
    rx: mpsc::Receiver<Cmd>,
    waiting: VecDeque<Submit>,
    running: Vec<Req>,
    alloc: SlotAllocator,
    radix: RadixCache,
    stats: SchedulerStats,
    decode_rate: RateWindow,
    /// When the last step's tokens were sampled (for `host_gap_us`).
    last_step_end: Option<Instant>,
    prefill_rate: RateWindow,
    last_was_prefill: bool,
    pending_rebuild: Option<(RebuildRequest, oneshot::Sender<Result<(), String>>)>,
    /// Free live recurrent-state slots; hybrid models need one per running request.
    free_states: Vec<u32>,
    /// Free snapshot slots: recurrent state saved at radix-node boundaries so hybrid
    /// (linear-attention) models can resume from a cached prefix.
    free_snaps: Vec<u32>,
    /// Free spare state slots for MTP verify steps (a draft's state, kept if it is accepted).
    free_spare: Vec<u32>,
    hybrid: bool,
    /// Prefix reuse: always for attention-only models; for hybrid ones only with snapshots.
    prefix_cache: bool,
}

pub fn spawn(engine: Engine, mut cfg: SchedulerConfig) -> Result<EngineHandle> {
    let hybrid = engine.cfg.is_hybrid();
    if hybrid {
        cfg.max_running = cfg.max_running.min(engine.live_states);
    }
    let (tx, rx) = mpsc::channel();
    let info = Arc::new(EngineInfo {
        model_type: engine.cfg.model_type.clone(),
        vocab_size: engine.cfg.vocab_size,
        num_layers: engine.cfg.num_layers,
        max_seq_len: engine.max_seq_len,
        kv_tokens: engine.num_slots,
        topology: engine.topology.clone(),
        attn: if engine.cfg.is_hybrid() {
            "hybrid_linear".into()
        } else {
            "mha".into()
        },
        moe: engine.cfg.moe.is_some(),
        image_processor: engine.image_processor.clone(),
        image_token_id: engine.cfg.vision.as_ref().map(|v| v.image_token_id),
    });
    let (live, total) = (
        engine.live_states as u32,
        (engine.num_states - engine.spare_states) as u32,
    );
    let free_spare: Vec<u32> = (total..engine.num_states as u32).rev().collect();
    let mut s = Scheduler {
        alloc: SlotAllocator::new(engine.num_slots),
        radix: RadixCache::new(),
        engine,
        cfg,
        rx,
        waiting: VecDeque::new(),
        running: vec![],
        stats: SchedulerStats::default(),
        decode_rate: RateWindow::default(),
        last_step_end: None,
        prefill_rate: RateWindow::default(),
        last_was_prefill: false,
        pending_rebuild: None,
        free_states: (0..live).rev().collect(),
        free_snaps: (live..total).rev().collect(),
        free_spare,
        hybrid,
        prefix_cache: !hybrid || total > live,
    };
    std::thread::Builder::new()
        .name("carbon-scheduler".into())
        .spawn(move || s.run())?;
    Ok(EngineHandle {
        tx,
        next_id: Arc::new(AtomicU64::new(1)),
        info,
    })
}

impl Scheduler {
    fn run(&mut self) {
        loop {
            if self.running.is_empty() && self.pending_rebuild.is_some() {
                self.execute_rebuild();
            }
            let idle = self.waiting.is_empty() && self.running.is_empty();
            if idle {
                match self.rx.recv() {
                    Ok(cmd) => self.on_cmd(cmd),
                    Err(_) => return,
                }
            }
            loop {
                match self.rx.try_recv() {
                    Ok(cmd) => self.on_cmd(cmd),
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        if self.waiting.is_empty() && self.running.is_empty() {
                            return;
                        }
                        break;
                    }
                }
            }
            if !self.running.is_empty()
                || (!self.waiting.is_empty() && self.pending_rebuild.is_none())
            {
                self.step();
            }
        }
    }

    fn execute_rebuild(&mut self) {
        let (req, reply) = self.pending_rebuild.take().unwrap();
        let result = self
            .engine
            .rebuild(req.kv_tokens, req.expert_cache)
            .map_err(|e| format!("{e:#}"));
        if result.is_ok() && req.kv_tokens.is_some() {
            // new pools: every cached prefix points at freed memory
            self.radix = RadixCache::new();
            self.alloc = SlotAllocator::new(self.engine.num_slots);
            let (live, total) = (
                self.engine.live_states as u32,
                self.engine.num_states as u32,
            );
            self.free_snaps = (live..total).rev().collect();
        }
        match &result {
            Ok(()) => tracing::info!(
                "cache rebuilt: kv_tokens={:?} expert_cache={:?}",
                req.kv_tokens,
                req.expert_cache
            ),
            Err(e) => tracing::error!("cache rebuild failed: {e}"),
        }
        let _ = reply.send(result);
    }

    fn on_cmd(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Submit(s) => {
                self.stats.total_requests += 1;
                let max_seq = self.engine.max_seq_len;
                if s.prompt.is_empty() {
                    let _ = s.tx.send(Event::Error("empty prompt".into()));
                } else if s.prompt.len() >= max_seq {
                    let _ = s.tx.send(Event::Error(format!(
                        "prompt has {} tokens, the context limit is {max_seq}",
                        s.prompt.len()
                    )));
                } else if s.prompt.len() > self.alloc.capacity() {
                    let _ = s.tx.send(Event::Error(format!(
                        "prompt has {} tokens but the KV cache holds only {}",
                        s.prompt.len(),
                        self.alloc.capacity()
                    )));
                } else if !s.images.is_empty() && !self.engine.has_vision() {
                    let _ = s.tx.send(Event::Error("this model has no image input".into()));
                } else if let Some(bad) = s.prompt.iter().find(|&&t| {
                    crate::images::model_token(t, self.image_token_id()) as usize >= self.engine.cfg.vocab_size
                }) {
                    let _ = s.tx.send(Event::Error(format!(
                        "token id {bad} is out of the vocabulary"
                    )));
                } else {
                    self.waiting.push_back(s);
                }
            }
            Cmd::Abort(id) => {
                if let Some(pos) = self.waiting.iter().position(|w| w.id == id) {
                    let w = self.waiting.remove(pos).unwrap();
                    let usage = Usage {
                        prompt_tokens: w.prompt.len(),
                        completion_tokens: 0,
                        cached_tokens: 0,
                    };
                    let _ = w.tx.send(Event::Done {
                        reason: FinishReason::Abort,
                        usage,
                    });
                } else if let Some(pos) = self.running.iter().position(|r| r.id == id) {
                    let r = self.running.swap_remove(pos);
                    self.finish(r, FinishReason::Abort);
                }
            }
            Cmd::Rebuild(req, reply) => {
                if self.pending_rebuild.is_some() {
                    let _ = reply.send(Err("another cache rebuild is pending".into()));
                } else {
                    self.pending_rebuild = Some((req, reply));
                }
            }
            Cmd::Stats(tx) => {
                let mut s = self.stats.clone();
                let now = Instant::now();
                s.decode_tps = self.decode_rate.rate(now);
                s.prefill_tps = self.prefill_rate.rate(now);
                s.running = self.running.len();
                s.waiting = self.waiting.len();
                s.kv_capacity = self.alloc.capacity();
                s.kv_free = self.alloc.available();
                s.prefix_cached_tokens = self.radix.cached_tokens();
                s.prefix_evictable_tokens = self.radix.evictable_size();
                s.prefix_snapshots = self.radix.snapshot_count();
                s.stages = self.engine.stats();
                let _ = tx.send(s);
            }
        }
    }

    fn reserved(&self) -> usize {
        self.running.iter().map(|r| r.remaining_kv()).sum()
    }

    fn alloc_slots(&mut self, n: usize) -> Option<Vec<u32>> {
        if self.alloc.available() < n {
            let freed = self.radix.evict(n - self.alloc.available());
            self.alloc.free(&freed);
            let snaps = self.radix.take_freed_snapshots();
            self.free_snaps.extend(snaps);
        }
        self.alloc.alloc(n)
    }

    /// Admit the head of the waiting queue if its whole KV budget can be reserved.
    fn try_admit(&mut self) -> bool {
        if self.pending_rebuild.is_some() {
            return false;
        }
        let Some(sub) = self.waiting.front() else {
            return false;
        };
        let prompt_len = sub.prompt.len();
        let max_new = sub
            .params
            .max_tokens
            .max(1)
            .min(self.engine.max_seq_len - prompt_len);
        if self.hybrid && self.free_states.is_empty() {
            return false;
        }
        // keep at least one prompt token to compute so the first output has logits
        let (mut node, mut matched) = if self.prefix_cache {
            self.radix.match_prefix(&sub.prompt[..prompt_len - 1])
        } else {
            (ROOT, vec![])
        };
        matched.truncate(prompt_len - 1);
        // hybrid: resume only where the recurrent state was snapshotted
        let mut restore = None;
        if self.hybrid && self.prefix_cache {
            match self.radix.snapshot_ancestor(node) {
                Some((a, depth, snap)) => {
                    node = a;
                    matched.truncate(depth);
                    restore = Some(snap);
                }
                None => {
                    node = ROOT;
                    matched.clear();
                }
            }
        }
        let need = prompt_len + max_new - 1 - matched.len();
        if need > self.alloc.capacity() {
            let sub = self.waiting.pop_front().unwrap();
            let _ = sub.tx.send(Event::Error(format!(
                "request needs {need} KV tokens but the cache holds {}; lower max_tokens",
                self.alloc.capacity()
            )));
            return true;
        }
        self.radix.lock(node);
        let free = self.alloc.available() + self.radix.evictable_size();
        if free < self.reserved() + need {
            self.radix.unlock(node);
            return false;
        }
        let state_slot = if self.hybrid {
            self.free_states.pop().unwrap()
        } else {
            0
        };
        if let Some(snap) = restore {
            if let Err(e) = self.engine.copy_state(snap, state_slot) {
                tracing::warn!("restoring a prefix snapshot failed, recomputing: {e:#}");
                self.radix.unlock(node);
                self.radix.lock(ROOT);
                node = ROOT;
                matched.clear();
            }
        }
        let sub = self.waiting.pop_front().unwrap();
        // PLE rows past the first prefill chunk are read from disk while the first one computes
        // (the first chunk's gather requests its own rows)
        let ahead = matched.len() + self.cfg.max_prefill_tokens;
        if ahead < prompt_len {
            let image_id = self.image_token_id();
            let model: Vec<u32> = sub.prompt[ahead - 2..].iter().map(|&t| crate::images::model_token(t, image_id)).collect();
            self.engine.prefetch_prompt(&model[..2], &model[2..]);
        }
        self.stats.total_prompt_tokens += prompt_len as u64;
        self.stats.total_cached_tokens += matched.len() as u64;
        let rng = sub.params.rng();
        self.running.push(Req {
            state_slot,
            anchor_len: None,
            mtp_drafts: vec![],
            mtp_next_slot: None,
            mrope: crate::images::MropePositions::new(&sub.images),
            images: sub.images.iter().map(|span| ReqImage { span: span.clone(), features: None }).collect(),
            mtp_acc1: MTP_PRIOR_ACC1,
            mtp_acc2: MTP_PRIOR_ACC2,
            id: sub.id,
            tokens: sub.prompt,
            prompt_len,
            tree_owned: matched.len(),
            cache_hit: matched.len(),
            slots: matched,
            node,
            max_new,
            params: sub.params,
            rng,
            tx: sub.tx,
        });
        true
    }

    fn step(&mut self) {
        let decode_ready = self.running.iter().any(|r| !r.in_prefill());
        let prefill_pending =
            self.running.iter().any(|r| r.in_prefill()) || !self.waiting.is_empty();
        // interleave: after a prefill step, give waiting decoders a turn
        let do_prefill = prefill_pending && !(decode_ready && self.last_was_prefill);
        if do_prefill && self.prefill_step() {
            self.last_was_prefill = true;
            return;
        }
        self.last_was_prefill = false;
        if decode_ready {
            if !self.spec_decode_step() {
                self.decode_step();
            }
        } else if !self.waiting.is_empty() && self.running.is_empty() {
            // nothing runs and the head request does not fit: it never will
            let sub = self.waiting.pop_front().unwrap();
            let _ = sub
                .tx
                .send(Event::Error("request does not fit in the KV cache".into()));
        }
    }

    /// Returns false when no prefill work could be scheduled.
    fn prefill_step(&mut self) -> bool {
        let mut budget = self.cfg.max_prefill_tokens;
        // admit new requests while there is budget and room
        while budget > 0 && self.running.len() < self.cfg.max_running && !self.waiting.is_empty() {
            if !self.try_admit() {
                break;
            }
        }
        let mut chosen: Vec<(usize, usize)> = vec![]; // (running idx, chunk)
        for (i, r) in self.running.iter().enumerate() {
            if budget == 0 {
                break;
            }
            if r.in_prefill() {
                let mut chunk = budget.min(r.prompt_len - r.slots.len());
                // hybrid: stop one token short of the prompt end and snapshot there, so an identical
                // prompt can later resume and still compute the last token's logits
                let boundary = r.prompt_len - 1;
                if self.hybrid && self.prefix_cache && r.slots.len() < boundary {
                    chunk = chunk.min(boundary - r.slots.len());
                }
                budget -= chunk;
                chosen.push((i, chunk));
            }
        }
        if chosen.is_empty() {
            return false;
        }
        let mut tokens = vec![];
        let mut meta = BatchMeta {
            is_decode: false,
            ..Default::default()
        };
        let mut sample_rows = vec![];
        let mut samplers = vec![];
        let mut inject = vec![];
        for &(i, chunk) in &chosen {
            let new = self.alloc_slots(chunk).expect("KV reservation violated");
            let start = self.running[i].slots.len();
            // image features for the chunk's placeholder tokens (the tower runs once per image)
            for k in 0..self.running[i].images.len() {
                let (s0, len) = (self.running[i].images[k].span.start, self.running[i].images[k].span.len);
                let (a, b) = (s0.max(start), (s0 + len).min(start + chunk));
                if a >= b {
                    continue;
                }
                if self.running[i].images[k].features.is_none() {
                    match self.engine.encode_image(&self.running[i].images[k].span.image) {
                        Ok(f) => self.running[i].images[k].features = Some(f),
                        Err(e) => {
                            tracing::error!("encoding an image failed: {e:#}");
                            continue;
                        }
                    }
                }
                let f = self.running[i].images[k].features.as_ref().unwrap();
                if let Ok(rows) = f.narrow(0, a - s0, b - a) {
                    inject.push((tokens.len() + a - start, rows));
                }
                if b == s0 + len {
                    self.running[i].images[k].features = None; // the image is fully prefilled
                }
            }
            let image_id = self.image_token_id();
            let r = &mut self.running[i];
            tokens.extend(r.tokens[start..start + chunk].iter().map(|&t| crate::images::model_token(t, image_id)));
            meta.positions
                .extend((start..start + chunk).map(|p| p as u32));
            meta.out_loc.extend_from_slice(&new);
            r.slots.extend_from_slice(&new);
            meta.seqs.push(SeqMeta {
                seq_id: r.id,
                q_len: chunk,
                kv_len: r.slots.len(),
                state_slot: r.state_slot,
                slots: Arc::new(r.slots.clone()),
                prev_tokens: r.tokens[start.saturating_sub(PREV_TOKENS)..start]
                    .iter()
                    .map(|&t| crate::images::model_token(t, image_id))
                    .collect(),
                token_state_slots: vec![],
            });
            if !r.in_prefill() {
                sample_rows.push(tokens.len() as u32 - 1);
                samplers.push(i);
            }
        }
        meta.tokens = tokens.clone();
        let batch: Vec<usize> = chosen.iter().map(|c| c.0).collect();
        self.attach_mrope(&mut meta, &batch);
        let n = tokens.len();
        let samplers_n = samplers.len();
        let t0 = Instant::now();
        let ok = self.run_and_sample_with(meta, &tokens, &sample_rows, &samplers, batch, &inject);
        if ok {
            self.stats.last_prefill_tokens_per_s = n as f64 / t0.elapsed().as_secs_f64().max(1e-9);
            self.prefill_rate.add(Instant::now(), n);
            self.decode_rate.add(Instant::now(), samplers_n);
            // share finished prompts with later requests right away
            for &(i, _) in &chosen {
                let r = &self.running[i];
                if self.hybrid && r.slots.len() + 1 == r.prompt_len && r.prompt_len > 1 {
                    self.cache_prompt(i, true);
                } else if samplers.contains(&i) {
                    self.cache_prompt(i, false);
                }
            }
            self.reap();
        }
        true
    }

    /// One MTP verify step: every decoding request feeds its last token and its `n` drafts;
    /// all rows are sampled. Drafts are accepted while each equals the token sampled before it,
    /// and every accepted draft also yields the next sampled token (distribution-exact: each
    /// row was conditioned on exactly the tokens before it). Rejected drafts' KV slots are freed
    /// and their recurrent states (spare slots) dropped. Then the head drafts again: one step
    /// over the verified rows, and with `--mtp-drafts 2` a further step from its own streams.
    /// Returns false (and does nothing) when the step does not qualify; the normal decode step
    /// then runs and keeps drafting.
    fn spec_decode_step(&mut self) -> bool {
        if !self.engine.mtp_speculates() || self.cfg.anchor_token.is_some() {
            return false;
        }
        let idxs: Vec<usize> = (0..self.running.len())
            .filter(|&i| !self.running[i].in_prefill())
            .collect();
        let cnt = idxs.len();
        if cnt == 0 || cnt > crate::engine::MTP_MAX_BATCH {
            return false;
        }
        // the same number of drafts for every sequence (one captured graph per shape)
        let n = idxs
            .iter()
            .map(|&i| {
                let r = &self.running[i];
                let want = if r.wants_two_drafts() { 2 } else { 1 };
                r.mtp_drafts.len().min(want).min((r.max_new - r.generated()).saturating_sub(1))
            })
            .min()
            .unwrap()
            .min(self.engine.mtp_drafts());
        let k = n + 1;
        if n == 0
            || idxs.iter().any(|&i| self.running[i].params.has_penalties())
            || (self.hybrid && self.free_spare.len() < n * cnt)
        {
            return false;
        }
        let need: usize = idxs
            .iter()
            .map(|&i| k - usize::from(self.running[i].mtp_next_slot.is_some()))
            .sum();
        let Some(mut fresh) = self.alloc_slots(need) else {
            return false;
        };
        let mut tokens = Vec::with_capacity(k * cnt);
        let mut meta = BatchMeta {
            is_decode: true,
            ..Default::default()
        };
        let mut new_slots: Vec<Vec<u32>> = Vec::with_capacity(cnt);
        let mut states: Vec<Vec<u32>> = Vec::with_capacity(cnt);
        let mut drafts: Vec<Vec<u32>> = Vec::with_capacity(cnt);
        for &i in &idxs {
            let mut own = Vec::with_capacity(k);
            if let Some(s) = self.running[i].mtp_next_slot.take() {
                own.push(s);
            }
            while own.len() < k {
                own.push(fresh.pop().unwrap());
            }
            let r = &self.running[i];
            let mut st = vec![r.state_slot];
            if self.hybrid {
                for _ in 0..n {
                    st.push(self.free_spare.pop().unwrap());
                }
            }
            let pos = r.slots.len();
            let d: Vec<u32> = r.mtp_drafts[..n].to_vec();
            tokens.push(r.tokens[pos]);
            tokens.extend(&d);
            meta.positions.extend((pos..pos + k).map(|p| p as u32));
            meta.out_loc.extend(&own);
            let mut slots = r.slots.clone();
            slots.extend(&own);
            meta.seqs.push(SeqMeta {
                seq_id: r.id,
                q_len: k,
                kv_len: pos + k,
                state_slot: r.state_slot,
                slots: Arc::new(slots),
                prev_tokens: r.tokens[pos.saturating_sub(PREV_TOKENS)..pos].to_vec(),
                token_state_slots: if self.hybrid { st.clone() } else { vec![] },
            });
            new_slots.push(own);
            states.push(st);
            drafts.push(d);
        }
        meta.tokens = tokens.clone();
        self.attach_mrope(&mut meta, &idxs);
        let meta = Arc::new(meta);
        let rows: Vec<u32> = (0..(k * cnt) as u32).collect();
        let samplers: Vec<usize> = idxs.iter().flat_map(|&i| std::iter::repeat_n(i, k)).collect();
        let t0 = Instant::now();
        self.stats.steps += 1;
        let result = self
            .engine
            .forward_hidden(meta.clone(), &tokens, &rows)
            .and_then(|(logits, hidden)| Ok((self.sample_rows(&logits, &samplers)?, hidden)));
        let (toks, hidden) = match result {
            Ok(x) => x,
            Err(e) => {
                tracing::error!("forward failed: {e:#}");
                for (own, st) in new_slots.iter().zip(&states) {
                    self.alloc.free(own);
                    self.free_spare.extend(&st[1..]);
                }
                let msg = format!("{e:#}");
                let mut batch = idxs.clone();
                batch.sort_unstable_by(|a, b| b.cmp(a));
                for i in batch {
                    let r = self.running.swap_remove(i);
                    let _ = r.tx.send(Event::Error(msg.clone()));
                    self.alloc.free(&r.slots[r.tree_owned..]);
                    self.radix.unlock(r.node);
                    if self.hybrid {
                        self.free_states.push(r.state_slot);
                    }
                    self.engine.release(&[r.id]);
                }
                return true;
            }
        };
        let verify_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut emitted_total = 0;
        let mut draft_rows = Vec::with_capacity(cnt);
        for (q, &i) in idxs.iter().enumerate() {
            let row = |j: usize| toks[k * q + j];
            let mut acc = 0;
            while acc < n && row(acc) == drafts[q][acc] {
                acc += 1;
            }
            self.stats.mtp_drafts += (acc + 1).min(n) as u64;
            self.stats.mtp_accepted += acc as u64;
            {
                let r = &mut self.running[i];
                r.mtp_acc1 = 0.9 * r.mtp_acc1 + 0.1 * f32::from(acc >= 1);
                if n >= 2 && acc >= 1 {
                    r.mtp_acc2 = 0.9 * r.mtp_acc2 + 0.1 * f32::from(acc >= 2);
                } else if n < 2 {
                    // unmeasured while running one draft: drift back so two drafts get retried
                    r.mtp_acc2 += 0.02 * (MTP_PRIOR_ACC2 - r.mtp_acc2);
                }
            }
            // emit the sampled tokens up to the first rejected draft, stopping early at a stop
            // token or the length limit
            let mut e = 0;
            for j in 0..=acc {
                let t = row(j);
                let r = &mut self.running[i];
                r.tokens.push(t);
                let _ = r.tx.send(Event::Token(t));
                e += 1;
                let stop = !r.params.ignore_eos
                    && (self.cfg.eos_token_ids.contains(&t) || r.params.stop_token_ids.contains(&t));
                if stop || r.generated() >= r.max_new {
                    break;
                }
            }
            // KV of the e inputs that were real tokens; the state after them
            let own = &new_slots[q];
            let r = &mut self.running[i];
            r.slots.extend(&own[..e]);
            let live = states[q][e - 1];
            if self.hybrid {
                let old = std::mem::replace(&mut r.state_slot, live);
                self.free_spare.extend(states[q].iter().copied().filter(|&s| s != live));
                debug_assert!(old == states[q][0]);
            }
            // the next position's slot stays assigned (a further draft step writes its MTP KV)
            if e < k {
                r.mtp_next_slot = Some(own[e]);
                self.alloc.free(&own[e + 1..]);
            }
            emitted_total += e;
            self.stats.total_generated_tokens += e as u64;
            draft_rows.push((k * q + e - 1) as u32);
        }
        // the head over every row (its KV for the kept positions), drafting from the last kept one
        let td = Instant::now();
        let drafted = self.mtp_redraft(&meta, &hidden, &toks, &draft_rows, &idxs);
        let ms = td.elapsed().as_secs_f64() * 1e3;
        if let Err(e) = drafted {
            tracing::warn!("mtp draft failed: {e:#}");
            for &i in &idxs {
                self.running[i].mtp_drafts.clear();
            }
        }
        let avg = &mut self.stats.mtp_draft_ms;
        *avg = if *avg == 0.0 { ms } else { 0.9 * *avg + 0.1 * ms };
        let avg = &mut self.stats.mtp_verify_ms;
        *avg = if *avg == 0.0 { verify_ms } else { 0.9 * *avg + 0.1 * verify_ms };
        self.last_step_end = Some(Instant::now());
        self.stats.last_decode_tokens_per_s = emitted_total as f64 / t0.elapsed().as_secs_f64().max(1e-9);
        self.decode_rate.add(Instant::now(), emitted_total);
        self.reap();
        true
    }

    /// New drafts for `idxs` after a step: the head over the step's rows (`next` = each row's
    /// successor), drafting from `rows`; then, with `--mtp-drafts 2`, one more draft step at the
    /// next position from the head's own streams (its KV in the request's `mtp_next_slot`).
    fn mtp_redraft(&mut self, meta: &BatchMeta, hidden: &Tensor, next: &[u32], rows: &[u32], idxs: &[usize]) -> Result<()> {
        let (first, multi) = self.engine.mtp_draft(meta, hidden, next, rows)?;
        for (q, &i) in idxs.iter().enumerate() {
            let r = &mut self.running[i];
            r.mtp_drafts = vec![match &self.engine.mtp_oracle {
                Some(f) => f(&r.tokens),
                None => first[q],
            }];
        }
        if self.engine.mtp_drafts() < 2
            || !self.engine.mtp_speculates()
            || idxs.is_empty()
            || !idxs.iter().all(|&i| self.running[i].wants_two_drafts())
        {
            return Ok(());
        }
        // every request needs its next position's slot for the head's KV there
        let missing = idxs.iter().filter(|&&i| self.running[i].mtp_next_slot.is_none()).count();
        let Some(mut extra) = self.alloc_slots(missing) else {
            return Ok(());
        };
        let mut meta2 = BatchMeta {
            is_decode: true,
            ..Default::default()
        };
        let mut toks2 = Vec::with_capacity(idxs.len());
        for &i in idxs {
            let r = &mut self.running[i];
            let slot = *r.mtp_next_slot.get_or_insert_with(|| extra.pop().unwrap());
            let pos = r.slots.len();
            let mut slots = r.slots.clone();
            slots.push(slot);
            meta2.positions.push(pos as u32);
            meta2.out_loc.push(slot);
            meta2.seqs.push(SeqMeta {
                seq_id: r.id,
                q_len: 1,
                kv_len: pos + 1,
                state_slot: r.state_slot,
                slots: Arc::new(slots),
                prev_tokens: vec![],
                token_state_slots: vec![],
            });
            toks2.push(r.mtp_drafts[0]);
        }
        meta2.tokens = toks2.clone();
        self.attach_mrope(&mut meta2, idxs);
        let rows2: Vec<u32> = (0..idxs.len() as u32).collect();
        let (second, _) = self.engine.mtp_draft(&meta2, &multi, &toks2, &rows2)?;
        for (q, &i) in idxs.iter().enumerate() {
            let r = &mut self.running[i];
            let d = match &self.engine.mtp_oracle {
                Some(f) => {
                    let mut t = r.tokens.clone();
                    t.push(r.mtp_drafts[0]);
                    f(&t)
                }
                None => second[q],
            };
            r.mtp_drafts.push(d);
        }
        Ok(())
    }

    fn decode_step(&mut self) {
        let idxs: Vec<usize> = (0..self.running.len())
            .filter(|&i| !self.running[i].in_prefill())
            .collect();
        let mut tokens = Vec::with_capacity(idxs.len());
        let mut meta = BatchMeta {
            is_decode: true,
            ..Default::default()
        };
        for &i in &idxs {
            // a slot already assigned to this position by the MTP head comes first
            let slot = match self.running[i].mtp_next_slot.take() {
                Some(s) => s,
                None => self.alloc_slots(1).expect("KV reservation violated")[0],
            };
            let r = &mut self.running[i];
            let pos = r.slots.len();
            tokens.push(r.tokens[pos]);
            meta.positions.push(pos as u32);
            meta.out_loc.push(slot);
            r.slots.push(slot);
            meta.seqs.push(SeqMeta {
                seq_id: r.id,
                q_len: 1,
                kv_len: r.slots.len(),
                state_slot: r.state_slot,
                slots: Arc::new(r.slots.clone()),
                prev_tokens: r.tokens[pos.saturating_sub(PREV_TOKENS)..pos].to_vec(),
                token_state_slots: vec![],
            });
        }
        meta.tokens = tokens.clone();
        self.attach_mrope(&mut meta, &idxs);
        let rows: Vec<u32> = (0..idxs.len() as u32).collect();
        let t0 = Instant::now();
        if self.run_and_sample(meta, &tokens, &rows, &idxs, idxs.clone()) {
            self.stats.last_decode_tokens_per_s =
                idxs.len() as f64 / t0.elapsed().as_secs_f64().max(1e-9);
            self.decode_rate.add(Instant::now(), idxs.len());
            // the anchor token's state was just computed: keep a reuse point right after it,
            // so a client that re-renders the tool call only invalidates the call body
            for i in 0..self.running.len() {
                if self.running[i].anchor_len == Some(self.running[i].slots.len()) {
                    self.running[i].anchor_len = Some(usize::MAX);
                    self.cache_prompt(i, true);
                }
            }
            self.reap();
        }
    }

    /// One token per row of `logits` for the requests `samplers`. Greedy batches only bring
    /// back the argmax; on CUDA the sampler runs on the GPU and full logits rows come back only
    /// for requests with penalties (or a top-p nucleus wider than the kernel's candidates).
    /// MTP shadow mode: score the drafts made last step against the tokens just sampled, then
    /// run the head over this step's tokens (it needs every position's KV) and keep its new
    /// drafts. `batch[s]` is the running request of `meta.seqs[s]`.
    #[allow(clippy::too_many_arguments)]
    fn mtp_shadow(
        &mut self,
        meta: &BatchMeta,
        tokens: &[u32],
        rows: &[u32],
        samplers: &[usize],
        batch: &[usize],
        toks: &[u32],
        hidden: &Tensor,
    ) -> Result<()> {
        for (k, &i) in samplers.iter().enumerate() {
            let r = &mut self.running[i];
            if let Some(&d) = r.mtp_drafts.first() {
                self.stats.mtp_drafts += 1;
                self.stats.mtp_accepted += u64::from(d == toks[k]);
            }
            r.mtp_drafts.clear();
            // this step put the next token's KV in a fresh slot
            if let Some(s) = r.mtp_next_slot.take() {
                self.alloc.free(&[s]);
            }
        }
        // the token after each token of the step: the next one in its chunk, the token just
        // sampled at a sequence's end, or the next prompt token after a partial prefill chunk
        let mut next: Vec<u32> = tokens[1..].iter().copied().chain([0]).collect();
        let mut row = 0;
        for (si, seq) in meta.seqs.iter().enumerate() {
            let last = row + seq.q_len - 1;
            let i = batch[si];
            next[last] = match samplers.iter().position(|&x| x == i) {
                Some(k) => toks[k],
                None => {
                    let pos = meta.positions[last] as usize + 1;
                    let t = self.running[i].tokens.get(pos).copied().unwrap_or(0);
                    crate::images::model_token(t, self.image_token_id())
                }
            };
            row += seq.q_len;
        }
        let t0 = Instant::now();
        // more running requests than a verify step takes: nothing will be speculated next, so
        // only the head's KV is kept current (drafting would cost a full head pass, its experts'
        // copies included, every step)
        if self.engine.mtp_speculates() && self.running.len() > crate::engine::MTP_MAX_BATCH {
            for &i in samplers {
                self.running[i].mtp_drafts.clear();
            }
            return self.engine.mtp_fill_kv(meta, hidden, &next);
        }
        self.mtp_redraft(meta, hidden, &next, rows, samplers)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let avg = &mut self.stats.mtp_draft_ms;
        *avg = if *avg == 0.0 { ms } else { 0.9 * *avg + 0.1 * ms };
        Ok(())
    }

    fn sample_rows(&mut self, logits: &Tensor, samplers: &[usize]) -> Result<Vec<u32>> {
        if samplers.is_empty() {
            return Ok(vec![]);
        }
        let host_only = |p: &SamplingParams| p.has_penalties();
        if samplers.iter().all(|&i| {
            let p = &self.running[i].params;
            p.is_greedy() && !host_only(p)
        }) {
            return Ok(logits.argmax(1)?.to_vec1::<u32>()?);
        }
        #[cfg(feature = "cuda")]
        if logits.device().is_cuda() && gpu_sampling_enabled() {
            use crate::kernels::cuda::{sample_on_gpu, RowSampling};
            use rand::Rng;
            let params: Vec<RowSampling> = samplers
                .iter()
                .map(|&i| {
                    let r = &mut self.running[i];
                    let p = &r.params;
                    RowSampling {
                        // penalised rows are re-sampled on the host below
                        temperature: if p.is_greedy() || host_only(p) { 0.0 } else { p.temperature },
                        top_k: p.top_k.max(0) as u32,
                        top_p: p.top_p,
                        min_p: p.min_p,
                        seed: r.rng.gen(),
                    }
                })
                .collect();
            match sample_on_gpu(logits, &params) {
                Ok(out) => {
                    let mut toks = Vec::with_capacity(out.len());
                    for (row, (&i, (tok, fallback))) in samplers.iter().zip(out).enumerate() {
                        let r = &mut self.running[i];
                        toks.push(if fallback || host_only(&r.params) {
                            let mut l = logits.get(row)?.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
                            host_sample(&mut l, r)
                        } else {
                            tok
                        });
                    }
                    return Ok(toks);
                }
                Err(e) => disable_gpu_sampling(&e),
            }
        }
        let full = logits.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;
        Ok(samplers
            .iter()
            .zip(full)
            .map(|(&i, mut l)| host_sample(&mut l, &mut self.running[i]))
            .collect())
    }

    /// Forward + sample. On failure every request in the batch is failed and freed.
    fn run_and_sample(
        &mut self,
        meta: BatchMeta,
        tokens: &[u32],
        rows: &[u32],
        samplers: &[usize],
        batch: Vec<usize>,
    ) -> bool {
        self.run_and_sample_with(meta, tokens, rows, samplers, batch, &[])
    }

    /// The image token id (placeholders map back to it), or a value no token has.
    fn image_token_id(&self) -> u32 {
        self.engine.cfg.vision.as_ref().map_or(u32::MAX, |v| v.image_token_id)
    }

    /// Give `meta` the mRoPE positions of its tokens when a sequence in it has images (`batch[s]`
    /// is the request of `meta.seqs[s]`).
    fn attach_mrope(&self, meta: &mut BatchMeta, batch: &[usize]) {
        if batch.iter().all(|&i| self.running[i].mrope.is_empty()) {
            return;
        }
        let mut out = Vec::with_capacity(meta.positions.len());
        let mut row = 0;
        for (si, seq) in meta.seqs.iter().enumerate() {
            let m = &self.running[batch[si]].mrope;
            for &p in &meta.positions[row..row + seq.q_len] {
                out.push(m.at(p as usize));
            }
            row += seq.q_len;
        }
        meta.mrope = out;
        if self.engine.cfg.qwen38.is_some() {
            meta.mrope_layout = batch.iter().map(|&i| self.running[i].mrope.clone()).collect();
        }
    }

    fn run_and_sample_with(
        &mut self,
        meta: BatchMeta,
        tokens: &[u32],
        rows: &[u32],
        samplers: &[usize],
        batch: Vec<usize>,
        inject: &[(usize, Tensor)],
    ) -> bool {
        self.stats.steps += 1;
        if let Some(t) = self.last_step_end.take() {
            let gap = t.elapsed();
            // back-to-back steps only: an idle scheduler is not overhead
            if gap < std::time::Duration::from_millis(50) {
                let us = gap.as_secs_f64() * 1e6;
                let h = &mut self.stats.host_gap_us;
                *h = if *h == 0.0 { us } else { 0.9 * *h + 0.1 * us };
            }
        }
        let meta = Arc::new(meta);
        let result = self
            .engine
            .forward_hidden_with(meta.clone(), tokens, rows, inject)
            .and_then(|(logits, hidden)| {
                let hidden = self.engine.has_mtp().then_some(hidden);
                Ok((self.sample_rows(&logits, samplers)?, hidden))
            });
        let result = result.and_then(|(toks, hidden)| {
            if let Some(hidden) = hidden {
                self.mtp_shadow(&meta, tokens, rows, samplers, &batch, &toks, &hidden)?;
            }
            Ok(toks)
        });
        self.last_step_end = Some(Instant::now());
        match result {
            Ok(toks) => {
                for (row, &i) in samplers.iter().enumerate() {
                    let r = &mut self.running[i];
                    let tok = toks[row];
                    r.tokens.push(tok);
                    if self.hybrid && Some(tok) == self.cfg.anchor_token && r.anchor_len.is_none() {
                        r.anchor_len = Some(r.tokens.len());
                    }
                    self.stats.total_generated_tokens += 1;
                    let _ = r.tx.send(Event::Token(tok));
                }
                true
            }
            Err(e) => {
                tracing::error!("forward failed: {e:#}");
                if format!("{e:#}").contains("OUT_OF_MEMORY") {
                    tracing::error!(
                        "the GPU ran out of memory: lower --moe-cache-size / --moe-cache-rate, \
                         --max-prefill-tokens or --kv-tokens, or raise --kv-reserve-tokens"
                    );
                }
                let msg = format!("{e:#}");
                let mut batch = batch;
                batch.sort_unstable_by(|a, b| b.cmp(a));
                for i in batch {
                    let r = self.running.swap_remove(i);
                    let _ = r.tx.send(Event::Error(msg.clone()));
                    // KV contents of a failed forward are not trusted: free without caching
                    self.alloc.free(&r.slots[r.tree_owned..]);
                    if let Some(s) = r.mtp_next_slot {
                        self.alloc.free(&[s]);
                    }
                    self.radix.unlock(r.node);
                    if self.hybrid {
                        self.free_states.push(r.state_slot);
                    }
                    self.engine.release(&[r.id]);
                }
                false
            }
        }
    }

    /// Insert a request's prompt into the radix tree once it is fully prefilled. Skipped when
    /// another request cached the same tokens meanwhile (the slot table must stay append-only).
    fn cache_prompt(&mut self, i: usize, snapshot: bool) {
        if !self.prefix_cache {
            return;
        }
        let r = &self.running[i];
        let key = &r.tokens[..r.slots.len()];
        let (_, existing) = self.radix.match_prefix(key);
        if existing.len() != r.tree_owned {
            return;
        }
        self.radix.insert(key, &r.slots);
        let (node, _) = self.radix.match_prefix(key);
        self.radix.lock(node);
        let r = &mut self.running[i];
        let old = std::mem::replace(&mut r.node, node);
        r.tree_owned = r.slots.len();
        let live = r.state_slot;
        self.radix.unlock(old);
        if snapshot {
            self.snapshot_at(node, live);
        }
    }

    /// Save the live recurrent state of a request at the end of `node` (if it has none yet).
    fn snapshot_at(&mut self, node: NodeId, live: u32) {
        if node == ROOT || self.radix.snapshot(node).is_some() {
            return;
        }
        let Some(slot) = self
            .free_snaps
            .pop()
            .or_else(|| self.radix.evict_snapshot())
        else {
            return;
        };
        if let Err(e) = self.engine.copy_state(live, slot) {
            tracing::warn!("saving a prefix snapshot failed: {e:#}");
            self.free_snaps.push(slot);
            return;
        }
        if let Some(old) = self.radix.set_snapshot(node, slot) {
            self.free_snaps.push(old);
        }
    }

    /// Finish requests that hit EOS, a stop token or their length budget.
    fn reap(&mut self) {
        let mut i = 0;
        while i < self.running.len() {
            let r = &self.running[i];
            let reason = if r.generated() == 0 {
                None
            } else {
                let last = *r.tokens.last().unwrap();
                if !r.params.ignore_eos
                    && (self.cfg.eos_token_ids.contains(&last)
                        || r.params.stop_token_ids.contains(&last))
                {
                    Some(FinishReason::Stop)
                } else if r.generated() >= r.max_new {
                    Some(FinishReason::Length)
                } else {
                    None
                }
            };
            match reason {
                Some(reason) => {
                    let r = self.running.swap_remove(i);
                    self.finish(r, reason);
                }
                None => i += 1,
            }
        }
    }

    fn finish(&mut self, r: Req, reason: FinishReason) {
        if let Some(s) = r.mtp_next_slot {
            self.alloc.free(&[s]);
        }
        if self.prefix_cache {
            let n = r.slots.len();
            let cached = self.radix.insert(&r.tokens[..n], &r.slots);
            // slots[tree_owned..cached] duplicate KV the tree already had
            if cached > r.tree_owned {
                self.alloc.free(&r.slots[r.tree_owned..cached]);
            }
            if self.hybrid && n > 0 {
                // the live state is exactly "after tokens[..n]": keep it for continuations
                let (node, _) = self.radix.match_prefix(&r.tokens[..n]);
                self.snapshot_at(node, r.state_slot);
            }
        } else {
            self.alloc.free(&r.slots);
        }
        if self.hybrid {
            self.free_states.push(r.state_slot);
        }
        self.radix.unlock(r.node);
        self.engine.release(&[r.id]);
        let _ = r.tx.send(Event::Done {
            reason,
            usage: r.usage(),
        });
        debug_assert!(self.radix.check().is_ok(), "{:?}", self.radix.check());
    }
}
