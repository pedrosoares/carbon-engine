// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! The engine: model head on the coordinator plus a pipeline of stages that may be
//! local single devices, local tensor-parallel groups, or remote RPC workers.

use crate::batch::BatchMeta;
use crate::config::ModelConfig;
use crate::model::layers::Placement;
use crate::model::moe::MoeOptions;
use crate::model::{ModelHead, StageStats};
use crate::parallel::{build_group, split_layers, BuildCtx, Stage};
use crate::rpc::RemoteStage;
use crate::weights::WeightSource;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Tensor};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub model_dir: PathBuf,
    /// Local devices of the coordinator. Grouped into pipeline stages of `tp_size` devices.
    pub devices: Vec<String>,
    pub tp_size: usize,
    /// `host:port` of RPC workers, each becomes one pipeline stage after the local ones.
    pub rpc_workers: Vec<String>,
    /// Per-stage layer counts or weights (empty = even split).
    pub layer_split: Vec<f64>,
    pub dtype: Option<DType>,
    /// KV cache size in tokens (None = sized from free memory).
    pub kv_tokens: Option<usize>,
    pub memory_ratio: f64,
    /// Max context length (0 = from the checkpoint, capped at 32768).
    pub max_seq_len: usize,
    pub moe: MoeOptions,
    /// Shared secret presented to RPC workers.
    pub rpc_token: Option<String>,
    /// Max concurrent sequences; sizes the recurrent-state pools of hybrid models.
    pub max_running: usize,
    /// Recurrent-state snapshots kept for prefix reuse on hybrid models (None = max_running).
    pub state_snapshots: Option<usize>,
    /// Replay single-GPU decode steps as CUDA graphs (local stages).
    pub cuda_graphs: bool,
    /// `--hf-overrides`: merged into config.json before parsing.
    pub hf_overrides: Option<serde_json::Value>,
    /// `--dummy-weight`: random weights shaped from the config, no checkpoint read.
    pub dummy_weights: bool,
    /// Load the checkpoint's MTP head (Qwen3.8) and draft with it.
    pub mtp: bool,
    /// Speculative decoding with the MTP head's drafts (implies `mtp`).
    pub mtp_speculate: bool,
    /// Drafts per verify step (1..=MTP_MAX_DRAFTS).
    pub mtp_drafts: usize,
    /// Skip the checkpoint's vision tower (no image input, its VRAM left to the caches).
    pub text_only: bool,
}

impl EngineConfig {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            devices: vec!["cpu".into()],
            tp_size: 1,
            rpc_workers: vec![],
            layer_split: vec![],
            dtype: None,
            kv_tokens: None,
            memory_ratio: 0.9,
            max_seq_len: 0,
            moe: MoeOptions::default(),
            rpc_token: None,
            max_running: 16,
            state_snapshots: None,
            cuda_graphs: false,
            hf_overrides: None,
            dummy_weights: false,
            mtp: false,
            mtp_speculate: false,
            mtp_drafts: 1,
            text_only: false,
        }
    }
}

/// Most drafts per verify step (`--mtp-drafts`).
pub const MTP_MAX_DRAFTS: usize = 2;

/// Tests only: a draft from the request's tokens so far (see `Engine::mtp_oracle`).
pub type MtpOracle = std::sync::Arc<dyn Fn(&[u32]) -> u32 + Send + Sync>;

/// Most decoding sequences a speculative (MTP verify) step takes; larger batches decode one
/// token per step. Measured on Qwen3.8-Flash-Next with offloaded experts (RTX 5090): +37% at one
/// sequence, +13% at two, -21% at four (each verified token adds its own expert copies).
pub const MTP_MAX_BATCH: usize = 2;

pub struct Engine {
    pub cfg: Arc<ModelConfig>,
    head: ModelHead,
    stages: Vec<Box<dyn Stage>>,
    pub num_slots: usize,
    /// Recurrent-state slots per linear-attention layer (= max concurrent sequences).
    pub num_states: usize,
    /// Slots `0..live_states` hold running requests; the rest hold prefix snapshots.
    pub live_states: usize,
    pub max_seq_len: usize,
    pub topology: Vec<String>,
    pub dtype: DType,
    /// The MTP head (`--mtp`), on the head device.
    mtp: Option<crate::model::mtp::MtpHead>,
    /// Speculative decoding with the head's drafts; the last `spare_states` recurrent-state
    /// slots hold drafts' states during verify steps.
    mtp_speculate: bool,
    mtp_max_drafts: usize,
    pub spare_states: usize,
    /// Tests only: replaces the MTP head's draft with `oracle(request tokens so far)`, so the
    /// accept path of speculative decoding can be exercised with a random head.
    #[doc(hidden)]
    pub mtp_oracle: Option<MtpOracle>,
    /// The vision tower (image input), on the head device.
    vision: Option<crate::model::vision::VisionTower>,
    /// Preprocessing for the tower's input (`preprocessor_config.json`).
    pub image_processor: Option<crate::images::ImageProcessor>,
    /// Qwen3.8 PLE: a thread that reads prompts' n-gram table rows from disk ahead of prefill.
    ple_prefetch: Option<std::sync::mpsc::Sender<(Vec<u32>, Vec<u32>)>>,
}

impl Engine {
    pub fn load(ec: &EngineConfig) -> Result<Self> {
        let cfg = Arc::new(ModelConfig::from_dir_with(
            &ec.model_dir,
            ec.hf_overrides.as_ref(),
        )?);
        let src: Arc<dyn WeightSource> = if ec.dummy_weights {
            tracing::warn!("--dummy-weight: serving random weights, outputs are meaningless");
            Arc::new(crate::dummy::DummySource::new(&cfg, 0))
        } else {
            crate::checkpoint::open_weights(&ec.model_dir)?
        };
        Self::load_with(ec, cfg, src)
    }

    pub fn load_with(
        ec: &EngineConfig,
        cfg: Arc<ModelConfig>,
        src: Arc<dyn WeightSource>,
    ) -> Result<Self> {
        // multimodal checkpoints nest the decoder under model.language_model.
        let cfg = if src.contains("model.language_model.embed_tokens.weight") {
            let mut c = (*cfg).clone();
            c.prefix = "model.language_model.".into();
            Arc::new(c)
        } else {
            cfg
        };
        if ec.tp_size == 0 || !ec.devices.len().is_multiple_of(ec.tp_size) {
            bail!(
                "{} local devices cannot be split into tensor-parallel groups of {}",
                ec.devices.len(),
                ec.tp_size
            );
        }
        if ec.devices.is_empty() {
            bail!("at least one local device is required (it hosts the embedding and LM head)");
        }
        let max_seq_len = if ec.max_seq_len > 0 {
            ec.max_seq_len
        } else {
            cfg.max_position_embeddings.min(32768)
        };
        let live_states = ec.max_running.max(1);
        let snapshots = if cfg.is_hybrid() {
            ec.state_snapshots.unwrap_or(live_states)
        } else {
            0
        };
        // a rejected draft is rolled back through the fused multi-token decode path (per-token
        // state slots), which runs on one local CUDA device
        let speculation_supported = ec.devices.len() == 1
            && ec.tp_size == 1
            && ec.rpc_workers.is_empty()
            && ec.devices[0].starts_with("cuda");
        if ec.mtp_speculate && !speculation_supported {
            tracing::warn!("--mtp needs a single local CUDA device: the MTP head only drafts");
        }
        // verify steps put each draft's recurrent state in a spare slot
        let spare_states = if ec.mtp_speculate && speculation_supported && cfg.is_hybrid() {
            MTP_MAX_BATCH * ec.mtp_drafts.clamp(1, MTP_MAX_DRAFTS)
        } else {
            0
        };
        let num_states = live_states + snapshots + spare_states;
        let mut moe = ec.moe.clone();
        moe.kv_reserve_tokens.get_or_insert(max_seq_len);
        moe.state_slots = num_states;
        let ctx = BuildCtx {
            src: src.clone(),
            cfg: cfg.clone(),
            dtype: ec.dtype,
            moe,
            max_seq_len,
            cuda_graphs: ec.cuda_graphs,
        };

        let head_dev = crate::device::parse_device(&ec.devices[0])?;
        let head_dtype = ctx.dtype_for(&head_dev);
        let head = ModelHead::load(
            &*src,
            &cfg,
            &Placement::single(head_dev.clone(), head_dtype),
        )?;
        // before the stages, so their memory sizing sees them
        let (vision, image_processor) = match (&cfg.vision, ec.text_only) {
            (Some(vc), false) if src.contains("model.visual.patch_embed.proj.weight") => {
                let tower = crate::model::vision::VisionTower::load(
                    &*src,
                    vc,
                    &Placement::single(head_dev.clone(), head_dtype),
                )?;
                (Some(tower), Some(crate::images::ImageProcessor::load(&ec.model_dir, vc)))
            }
            _ => (None, None),
        };
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut mtp = if ec.mtp || ec.mtp_speculate {
            let m = crate::model::mtp::MtpHead::load(
                &*src,
                &cfg,
                max_seq_len,
                &Placement::single(head_dev.clone(), head_dtype),
            )?;
            if m.is_none() {
                tracing::warn!("--mtp: the checkpoint has no MTP head");
            }
            m
        } else {
            None
        };

        let groups: Vec<Vec<String>> = ec.devices.chunks(ec.tp_size).map(|c| c.to_vec()).collect();
        let n_stages = groups.len() + ec.rpc_workers.len();
        let ranges = split_layers(cfg.num_layers, n_stages, &ec.layer_split)?;
        let mut stages: Vec<Box<dyn Stage>> = vec![];
        let mut topology = vec![format!(
            "head (embed + lm_head) on {} [{}]",
            ec.devices[0],
            crate::device::dtype_name(head_dtype)
        )];
        for (i, range) in ranges.into_iter().enumerate() {
            if range.is_empty() {
                continue;
            }
            if i < groups.len() {
                let g = &groups[i];
                tracing::info!(
                    "stage {i}: layers {}..{} on {:?}",
                    range.start,
                    range.end,
                    g
                );
                topology.push(format!(
                    "stage {i}: layers {}..{} on {}{}",
                    range.start,
                    range.end,
                    g.join(","),
                    if g.len() > 1 {
                        format!(" (tensor parallel x{})", g.len())
                    } else {
                        String::new()
                    }
                ));
                stages.push(build_group(&ctx, range, g)?);
            } else {
                let addr = &ec.rpc_workers[i - groups.len()];
                topology.push(format!(
                    "stage {i}: layers {}..{} on rpc worker {addr}",
                    range.start, range.end
                ));
                stages.push(Box::new(RemoteStage::setup(
                    addr,
                    ec.rpc_token.as_deref(),
                    &ctx,
                    range,
                )?));
            }
        }

        let num_slots = match ec.kv_tokens {
            Some(n) => n,
            None => {
                let mut cap: Option<usize> = None;
                for s in &stages {
                    if let Some(c) = s.kv_capacity(ec.memory_ratio)? {
                        cap = Some(cap.map_or(c, |x: usize| x.min(c)));
                    }
                }
                // without an explicit size keep host-memory pools modest: a few full contexts
                cap.unwrap_or(16 * 1024).min(8 * max_seq_len).max(1024)
            }
        };
        for s in &stages {
            s.alloc_kv(num_slots, num_states)?;
        }
        #[cfg(feature = "cuda")]
        if let (Some(m), Some(bank), Some(moe), false) = (
            mtp.as_mut(),
            stages.first().and_then(|s| s.expert_bank()),
            cfg.moe.as_ref(),
            // CARBON_MTP_BANK=0: keep the head's experts in the mapped checkpoint (bf16)
            std::env::var("CARBON_MTP_BANK").is_ok_and(|v| v == "0"),
        ) {
            // the head's experts join the stage's slot cache (NVFP4, like the main layers)
            m.attach_bank(bank, moe, &cfg.hidden_act)?;
            tracing::info!("mtp experts in the expert slot cache");
        }
        if let Some(m) = &mtp {
            m.alloc_kv(num_slots)?;
            topology.push(format!("mtp head on {}", ec.devices[0]));
        }
        topology.push(format!(
            "kv cache: {num_slots} tokens per stage, max context {max_seq_len}"
        ));
        for line in &topology {
            tracing::info!("{line}");
        }
        let prefetchers: Vec<_> = stages.iter().flat_map(|s| s.ple_prefetchers()).collect();
        let ple_prefetch = (!prefetchers.is_empty()).then(|| {
            let (tx, rx) = std::sync::mpsc::channel::<(Vec<u32>, Vec<u32>)>();
            std::thread::Builder::new()
                .name("ple-prefetch".into())
                .spawn(move || {
                    for (prev, tokens) in rx {
                        // in prefill order, so the next chunk's rows are requested first
                        for (i, part) in tokens.chunks(1024).enumerate() {
                            let before = if i == 0 { &prev[..] } else { &tokens[i * 1024 - 2..i * 1024] };
                            prefetchers.iter().for_each(|p| p.run(before, part));
                        }
                    }
                })
                .expect("spawning the PLE prefetch thread");
            tx
        });
        Ok(Self {
            ple_prefetch,
            cfg,
            head,
            stages,
            num_slots,
            num_states,
            live_states,
            max_seq_len,
            topology,
            dtype: head_dtype,
            mtp_speculate: ec.mtp_speculate && mtp.is_some() && speculation_supported,
            mtp_max_drafts: ec.mtp_drafts.clamp(1, MTP_MAX_DRAFTS),
            mtp,
            spare_states,
            mtp_oracle: None,
            vision,
            image_processor,
        })
    }

    /// Run the model; returns f32 logits `[sample_rows.len(), vocab]`.
    pub fn forward(
        &self,
        meta: Arc<BatchMeta>,
        tokens: &[u32],
        sample_rows: &[u32],
    ) -> Result<Tensor> {
        Ok(self.forward_hidden(meta, tokens, sample_rows)?.0)
    }

    /// Like `forward`, also returning the last stage's output for every token (Qwen3.8: the
    /// hyper-connection streams before the final mixer, `[T, hc * H]`).
    pub fn forward_hidden(
        &self,
        meta: Arc<BatchMeta>,
        tokens: &[u32],
        sample_rows: &[u32],
    ) -> Result<(Tensor, Tensor)> {
        self.forward_hidden_with(meta, tokens, sample_rows, &[])
    }

    /// `forward_hidden` with some token embeddings replaced: `(row, rows [n, hidden])` puts the
    /// rows at batch rows `row..row + n` (image features at their placeholder tokens).
    pub fn forward_hidden_with(
        &self,
        meta: Arc<BatchMeta>,
        tokens: &[u32],
        sample_rows: &[u32],
        inject: &[(usize, Tensor)],
    ) -> Result<(Tensor, Tensor)> {
        debug_assert!(meta.validate().is_ok());
        let mut hidden = self.head.embed_with(tokens, inject)?;
        for s in &self.stages {
            hidden = s.forward(&meta, hidden)?;
        }
        if sample_rows.is_empty() {
            let z = Tensor::zeros((0, self.cfg.vocab_size), DType::F32, &self.head.device)?;
            return Ok((z, hidden));
        }
        Ok((self.head.logits(&hidden, sample_rows)?, hidden))
    }

    /// Run the vision tower on one image: its language-model tokens `[n, hidden]` on the head
    /// device.
    pub fn encode_image(&self, img: &crate::images::PreparedImage) -> Result<Tensor> {
        let tower = self.vision.as_ref().context("this server has no vision tower")?;
        let n = img.grid.0 * img.grid.1 * img.grid.2;
        let pixels = Tensor::from_slice(&img.pixels, (n, img.pixels.len() / n.max(1)), &tower.device)?;
        Ok(tower.forward(&pixels, img.grid)?.to_device(&self.head.device)?.to_dtype(self.dtype)?)
    }

    /// Start reading the PLE table rows `tokens` (following `prev`, model token ids) will need, in
    /// the background; a no-op for models without PLE.
    pub fn prefetch_prompt(&self, prev: &[u32], tokens: &[u32]) {
        if let (Some(tx), false) = (&self.ple_prefetch, tokens.is_empty()) {
            let _ = tx.send((prev.to_vec(), tokens.to_vec()));
        }
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Decode with the MTP head's drafts (verify steps) rather than only scoring them.
    pub fn mtp_speculates(&self) -> bool {
        self.mtp_speculate
    }

    /// Greedy MTP drafts for `rows`: runs the head over every token of the step (`hidden` from
    /// `forward_hidden`, or a previous draft step's streams; `next[i]` the token after token `i`)
    /// and returns, per row, its guess for the token after `next[row]`, and the head's streams
    /// for those rows (the input of a further draft step).
    pub fn mtp_draft(&self, meta: &BatchMeta, hidden: &Tensor, next: &[u32], rows: &[u32]) -> Result<(Vec<u32>, Tensor)> {
        let m = self.mtp.as_ref().context("no MTP head")?;
        let emb = self.head.embed_single(next)?;
        let (mixed, multi) = m.forward(meta, hidden, &emb, rows)?;
        if rows.is_empty() {
            return Ok((vec![], multi));
        }
        Ok((self.head.project(&mixed)?.argmax(1)?.to_vec1::<u32>()?, multi))
    }

    /// Only the MTP head's KV for the step's tokens (`mtp_draft`'s inputs, no draft).
    pub fn mtp_fill_kv(&self, meta: &BatchMeta, hidden: &Tensor, next: &[u32]) -> Result<()> {
        let m = self.mtp.as_ref().context("no MTP head")?;
        m.fill_kv(meta, hidden, &self.head.embed_single(next)?)
    }

    /// Drafts per verify step (`--mtp-drafts`).
    pub fn mtp_drafts(&self) -> usize {
        self.mtp_max_drafts
    }

    pub fn release(&self, seq_ids: &[u64]) {
        for s in &self.stages {
            if let Err(e) = s.release(seq_ids) {
                tracing::warn!("release failed: {e:#}");
            }
        }
    }

    /// Idle-only runtime resize of the KV pools and/or the expert caches on every stage.
    pub fn rebuild(&mut self, kv_tokens: Option<usize>, expert_cache: Option<usize>) -> Result<()> {
        if let Some(n) = expert_cache {
            for s in &self.stages {
                s.resize_expert_cache(n)?;
            }
        }
        if let Some(n) = kv_tokens {
            if n == 0 {
                bail!("kv_tokens must be > 0");
            }
            for s in &self.stages {
                s.alloc_kv(n, self.num_states)?;
            }
            if let Some(m) = &self.mtp {
                m.alloc_kv(n)?;
            }
            self.num_slots = n;
        }
        Ok(())
    }

    /// Copy recurrent-state slot `src` to `dst` on every stage.
    pub fn copy_state(&self, src: u32, dst: u32) -> Result<()> {
        self.stages.iter().try_for_each(|s| s.copy_state(src, dst))
    }

    pub fn stats(&self) -> Vec<StageStats> {
        self.stages.iter().flat_map(|s| s.stats()).collect()
    }
}
