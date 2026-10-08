// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

use anyhow::{bail, Context, Result};
use carbon_engine::device::{parse_device_list, parse_dtype};
use carbon_engine::engine::{Engine, EngineConfig};
use carbon_engine::model::moe::{MoeOptions, MoeStrategy};
use carbon_engine::rpc::{self, WorkerConfig};
use carbon_engine::sampling::SamplingParams;
use carbon_engine::scheduler::{self, Event, SchedulerConfig};
use carbon_engine::server::inline_system::InlineSystem;
use carbon_engine::server::{self, AppState, SamplingDefaults, ServerOptions};
use carbon_engine::tokenizer::{DecodeStream, TokenizerBundle};
use clap::{Args, Parser, Subcommand};
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "carbon",
    version,
    about = "Multi-GPU, multi-machine LLM / MoE inference engine"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Start the API server (OpenAI /v1/*, Anthropic /v1/messages).
    Serve(ServeArgs),
    /// Serve a range of layers to a coordinator over TCP (multi-machine inference).
    Worker(WorkerArgs),
    /// Generate text offline, without the HTTP server.
    Generate(GenerateArgs),
    /// Query a running server.
    Ctl(CtlArgs),
    /// Configure a coding agent (claude, codex, opencode, openclaw, hermes, dsh) for a running
    /// server and start it. Arguments after `--` go to the agent.
    Launch(LaunchArgs),
    /// Supervisor for one `carbon serve`: without a verb, run the daemon; with one, talk to it.
    Daemon(DaemonArgs),
    /// Chat with a running server in the terminal; with `--model ...` (any serve flags) an
    /// engine is started in this process first.
    Shell(ShellArgs),
    /// List visible compute devices.
    Devices {
        /// Run every kernel the engine needs on each device and report failures.
        #[arg(long)]
        selftest: bool,
        /// Devices to test (default: all visible).
        #[arg(long, default_value = "")]
        devices: String,
        /// Activation dtype to test (default: what the engine would pick).
        #[arg(long, default_value = "auto")]
        dtype: String,
    },
    /// Benchmarks: `bench bw` measures host vs PCIe bandwidth for the MoE offload/hybrid choice.
    Bench {
        #[command(subcommand)]
        cmd: BenchCmd,
    },
    /// Convert a safetensors checkpoint into carbon's aligned weight format (parallel O_DIRECT
    /// loading); point --model at the output directory.
    Checkpoint {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Convert floating-point tensors to this dtype (bf16 | f16 | f32); quantized codes stay.
        #[arg(long)]
        dtype: Option<String>,
        /// Shard file size limit in GiB.
        #[arg(long, default_value_t = 8.0)]
        shard_gib: f64,
        /// Print machine-readable `CARBONCONVERT <done> <total>` lines.
        #[arg(long, env = "CARBON_CONVERT_PROGRESS")]
        progress: bool,
    },
    /// Write a tiny random checkpoint (for tests and demos).
    TinyModel(TinyArgs),
    /// Render a conversation with the model's chat template (debugging templates).
    ChatTemplate(TemplateArgs),
}

#[derive(Args)]
struct TemplateArgs {
    /// Directory with tokenizer.json / tokenizer_config.json / chat_template.jinja.
    #[arg(long)]
    model: PathBuf,
    /// JSON file: {"messages": [...], "tools": [...], "chat_template_kwargs": {...}} (OpenAI format).
    #[arg(long)]
    request: PathBuf,
    /// Also print the token ids.
    #[arg(long)]
    ids: bool,
}

#[derive(Args, Clone)]
struct EngineArgs {
    /// Local model directory (HF safetensors layout) or a hub repo id (`org/name[@revision]`),
    /// downloaded into the hub cache on first use.
    #[arg(long, alias = "model-path")]
    model: PathBuf,
    /// Where repo ids are downloaded from: huggingface | modelscope.
    #[arg(long, default_value = "huggingface")]
    model_source: String,
    /// JSON merged into config.json before loading (vLLM semantics), e.g.
    /// '{"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}}'.
    #[arg(long)]
    hf_overrides: Option<String>,
    /// Random weights shaped from config.json (benchmarking / smoke tests without a checkpoint).
    #[arg(long)]
    dummy_weight: bool,
    /// Comma separated local devices, e.g. `cuda:0,cuda:1`, `cuda:all`, `cpu`. Default: cuda:0 if present, else cpu.
    #[arg(long, default_value = "")]
    devices: String,
    /// Tensor-parallel size: local devices are grouped into pipeline stages of this many GPUs.
    #[arg(long, default_value_t = 1)]
    tp_size: usize,
    /// RPC worker `host:port`; repeat for several machines. Each becomes a pipeline stage after the local ones.
    #[arg(long = "rpc-worker")]
    rpc_workers: Vec<String>,
    /// Shared secret for RPC workers started with `--token`.
    #[arg(long, env = "CARBON_RPC_TOKEN", hide_env_values = true)]
    rpc_token: Option<String>,
    /// Layers per stage (`20,12`) or relative weights (`2:1`). Default: even split.
    #[arg(long, default_value = "")]
    layer_split: String,
    /// auto | f32 | f16 | bf16
    #[arg(long, default_value = "auto")]
    dtype: String,
    /// KV cache capacity in tokens (default: sized from free memory).
    #[arg(long, alias = "num-tokens")]
    kv_tokens: Option<usize>,
    /// Fraction of free device memory the KV cache may use.
    #[arg(long, default_value_t = 0.9)]
    memory_ratio: f64,
    /// Max context length (0 = from the checkpoint, capped at 32768).
    #[arg(long, alias = "max-seq-len-override", default_value_t = 0)]
    max_seq_len: usize,
    /// MoE expert placement: auto | fused | offload | cpu | hybrid.
    #[arg(long, alias = "moe-backend", default_value = "auto")]
    moe_strategy: String,
    /// GPU expert-cache size in experts (offload/hybrid). Default: the free memory left after
    /// the KV reserve.
    #[arg(long, conflicts_with = "moe_cache_rate")]
    moe_cache_size: Option<usize>,
    /// GPU expert-cache size as a fraction of the stage's experts (offload/hybrid).
    #[arg(long)]
    moe_cache_rate: Option<f64>,
    /// KV tokens kept free when sizing the expert cache and deciding `auto` placement
    /// (default: one full context, `--max-seq-len`).
    #[arg(long)]
    kv_reserve_tokens: Option<usize>,
    /// Offload/hybrid: MoE layers computed on the CPU instead of through the device cache:
    /// ids (`3,7,11`), a count (`8`, evenly strided) or a fraction (`0.5`).
    #[arg(long, default_value = "")]
    moe_cpu_layers: String,
    /// CPU threads for CPU compute (CPU experts, CPU stages); 0 = physical cores.
    #[arg(long, default_value_t = 0)]
    moe_cpu_threads: usize,
    /// Hybrid: max experts fetched over PCIe per layer step, the rest run on the CPU.
    #[arg(long)]
    moe_hybrid_max_fetch: Option<usize>,
    /// Dequantize block-FP8 experts to bf16 at load (default: keep them FP8, half the memory).
    #[arg(long)]
    dequant_fp8_experts: bool,
    /// Max concurrently running requests.
    #[arg(long, default_value_t = 16)]
    max_running_requests: usize,
    /// Hybrid (Qwen3.5/3.6) models: recurrent-state snapshots kept for prefix reuse
    /// (default: max running requests; 0 disables prefix caching for them).
    #[arg(long)]
    state_snapshots: Option<usize>,
    /// Chunked-prefill token budget per step.
    #[arg(long, alias = "max-prefill-length", default_value_t = 4096)]
    max_prefill_tokens: usize,
    /// Replay single-GPU decode steps (up to 16 sequences) as CUDA graphs.
    #[arg(long, env = "CARBON_CUDA_GRAPHS")]
    cuda_graphs: bool,
    /// Load the checkpoint's MTP head (Qwen3.8) and measure how often its draft matches the
    /// next sampled token, without using it (`ctl stats` / the generate summary report it).
    #[arg(long)]
    mtp_shadow: bool,
    /// Speculative decoding with the checkpoint's MTP head (Qwen3.8): one draft per step,
    /// verified in the next step; output is unchanged (exact for greedy and for sampling).
    #[arg(long)]
    mtp: bool,
    /// Drafts per verify step with `--mtp` (1-2).
    #[arg(long, default_value_t = 2)]
    mtp_drafts: usize,
    /// Do not load the checkpoint's vision tower: no image input, its VRAM (~0.9 GB for
    /// Qwen3.6 / 3.8) goes to the KV and expert caches.
    #[arg(long)]
    text_only: bool,
}

impl EngineArgs {
    /// Replace a hub repo id in `--model` by its local (downloaded) directory.
    fn resolve_model(&mut self) -> Result<()> {
        let source = self.model_source.parse()?;
        self.model = carbon_engine::hub::resolve_with(&self.model, source, !self.dummy_weight)?;
        Ok(())
    }

    fn engine_config(&self) -> Result<EngineConfig> {
        let layer_split = if self.layer_split.trim().is_empty() {
            vec![]
        } else {
            self.layer_split
                .split([',', ':'])
                .map(|x| {
                    x.trim()
                        .parse::<f64>()
                        .with_context(|| format!("bad --layer-split entry '{x}'"))
                })
                .collect::<Result<_>>()?
        };
        Ok(EngineConfig {
            model_dir: self.model.clone(),
            devices: parse_device_list(&self.devices)?,
            tp_size: self.tp_size,
            rpc_workers: self.rpc_workers.clone(),
            layer_split,
            dtype: parse_dtype(&self.dtype)?,
            kv_tokens: self.kv_tokens,
            memory_ratio: self.memory_ratio,
            max_seq_len: self.max_seq_len,
            moe: self.moe_options()?,
            rpc_token: self.rpc_token.clone(),
            max_running: self.max_running_requests.max(1),
            state_snapshots: self.state_snapshots,
            cuda_graphs: self.cuda_graphs,
            hf_overrides: self
                .hf_overrides
                .as_deref()
                .map(serde_json::from_str)
                .transpose()
                .context("--hf-overrides is not valid JSON")?,
            dummy_weights: self.dummy_weight,
            mtp: self.mtp_shadow,
            mtp_speculate: self.mtp,
            mtp_drafts: self.mtp_drafts,
            text_only: self.text_only,
        })
    }

    fn moe_options(&self) -> Result<MoeOptions> {
        Ok(MoeOptions {
            strategy: self.moe_strategy.parse::<MoeStrategy>()?,
            cache_size: self.moe_cache_size,
            cache_rate: self.moe_cache_rate,
            hybrid_max_fetch: self.moe_hybrid_max_fetch,
            dequant_fp8: self.dequant_fp8_experts,
            kv_reserve_tokens: self.kv_reserve_tokens,
            state_slots: 0,
            cpu_layers: self.moe_cpu_layers.parse()?,
            offload_on_cpu: false,
        })
    }

    fn start(&self, anchor_token: Option<u32>) -> Result<(scheduler::EngineHandle, Vec<u32>)> {
        init_cpu_threads(self.moe_cpu_threads);
        let ec = self.engine_config()?;
        let engine = Engine::load(&ec)?;
        let eos = engine.cfg.eos_token_ids.clone();
        let handle = scheduler::spawn(
            engine,
            SchedulerConfig {
                max_running: self.max_running_requests.max(1),
                max_prefill_tokens: self.max_prefill_tokens.max(16),
                eos_token_ids: eos.clone(),
                anchor_token,
            },
        )?;
        Ok((handle, eos))
    }
}

#[derive(Args)]
struct ServeArgs {
    #[command(flatten)]
    engine: EngineArgs,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 1919)]
    port: u16,
    /// Model id reported by /v1/models (default: basename of --model).
    #[arg(long)]
    served_model_name: Option<String>,
    /// Default output budget for requests that omit one.
    #[arg(long, default_value_t = 4096)]
    max_output_tokens: usize,
    /// `model`: fill unspecified sampling params from generation_config.json; `none`: framework defaults.
    #[arg(long, default_value = "model")]
    sampling_defaults: String,
    /// auto | on | off: split `<think>` blocks into reasoning content.
    #[arg(long, default_value = "auto")]
    reasoning_parser: String,
    /// auto | hermes | off: parse `<tool_call>` JSON into tool calls.
    #[arg(long, default_value = "auto")]
    tool_call_parser: String,
    /// Jinja chat template file overriding the checkpoint's.
    #[arg(long)]
    chat_template: Option<PathBuf>,
    /// Hybrid (Gated-DeltaNet) models: when a request samples the tool-call opener, keep a
    /// prefix-cache reuse point right after it, so a client that re-renders the tool call
    /// only invalidates the call body, not the whole turn.
    #[arg(long)]
    enable_special_token_ckpt: bool,
    /// Report prefix-cache hits in usage (OpenAI cached_tokens, Anthropic
    /// cache_read_input_tokens, Responses input_tokens_details). On by default.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    enable_cache_report: bool,
    /// Late system messages in Anthropic requests: auto (probe the template) | preserve | fold.
    #[arg(long, default_value = "auto")]
    anthropic_inline_system: String,
    /// Comma separated browser origins allowed by CORS (`*` = any, empty = none).
    #[arg(long, default_value = "tauri://localhost,http://tauri.localhost,http://localhost:1420")]
    cors_origins: String,
}

#[derive(Args)]
struct WorkerArgs {
    /// Address to listen on.
    #[arg(long, default_value = "0.0.0.0:7070")]
    listen: String,
    /// Local devices this worker uses (see `serve --devices`).
    #[arg(long, default_value = "")]
    devices: String,
    /// Tensor-parallel size inside the worker.
    #[arg(long, default_value_t = 1)]
    tp_size: usize,
    /// Local model copy; without it the coordinator ships the needed layer weights over the network.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Worker name shown in stats.
    #[arg(long)]
    name: Option<String>,
    #[arg(long, default_value = "auto")]
    moe_strategy: String,
    #[arg(long)]
    moe_cache_size: Option<usize>,
    /// Replay single-GPU decode steps as CUDA graphs (see `serve --cuda-graphs`).
    #[arg(long, env = "CARBON_CUDA_GRAPHS")]
    cuda_graphs: bool,
    /// Shared secret coordinators must present (strongly recommended off localhost).
    #[arg(long, env = "CARBON_RPC_TOKEN", hide_env_values = true)]
    token: Option<String>,
}

#[derive(Args)]
struct GenerateArgs {
    #[command(flatten)]
    engine: EngineArgs,
    #[arg(long, default_value = "")]
    prompt: String,
    /// Comma separated prompt token ids (instead of --prompt).
    #[arg(long)]
    prompt_ids: Option<String>,
    /// Print the generated token ids after the text.
    #[arg(long)]
    print_ids: bool,
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = -1)]
    top_k: i32,
    #[arg(long)]
    seed: Option<u64>,
    /// Wrap the prompt in the chat template as a user message.
    #[arg(long)]
    chat: bool,
    /// Image (path, http(s) or data: URL) placed before the prompt; repeatable. Implies --chat.
    #[arg(long)]
    image: Vec<String>,
    /// Image as base64 data (raw or a data: URL), after the --image ones; repeatable.
    #[arg(long)]
    image_base64: Vec<String>,
    #[arg(long)]
    ignore_eos: bool,
}

#[derive(Subcommand)]
enum BenchCmd {
    /// CPU expert kernels vs expert uploads per format; writes the profile `--moe-strategy auto`
    /// and hybrid's fetch split read.
    Bw {
        /// Expert formats: all | bf16,fp8,nvfp4.
        #[arg(long, alias = "formats", default_value = "all")]
        dtype: String,
        /// GPU to bench (`cuda:N`); none = host-only measurements.
        #[arg(long, default_value = "")]
        device: String,
        /// Take the expert geometry from this model's config.json.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Recommend hybrid when the CPU kernel is this many times faster than the upload.
        #[arg(long, default_value_t = 2.0)]
        threshold: f64,
        /// Seconds per measurement.
        #[arg(long, default_value_t = 2.0)]
        seconds: f64,
        /// Host memory per format's synthetic expert bank (MiB).
        #[arg(long, default_value_t = 1024)]
        bank_mib: usize,
        /// CPU threads (0 = physical cores).
        #[arg(long, default_value_t = 0)]
        cpu_threads: usize,
        /// Profile path (default: $XDG_CACHE_HOME/carbon/benchbw/<gpu-uuid>.json).
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
}

#[derive(Args)]
struct DaemonArgs {
    #[command(subcommand)]
    verb: Option<DaemonVerb>,
    /// Address the daemon listens on.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = carbon_engine::daemon::DEFAULT_PORT)]
    port: u16,
    /// State directory (serve state, logs, accounting receipts); default ~/.carbon/daemon.
    #[arg(long, env = "CARBON_DAEMON_DIR")]
    state_dir: Option<PathBuf>,
    /// Shared secret required in `X-Carbon-Token` (daemon) / sent by the client verbs.
    #[arg(long, env = "CARBON_DAEMON_TOKEN", hide_env_values = true)]
    token: Option<String>,
    /// Port serves listen on unless a start request names one.
    #[arg(long, default_value_t = 1919)]
    default_serve_port: u16,
    /// Seconds between SIGTERM and SIGKILL when stopping a serve.
    #[arg(long, default_value_t = 10.0)]
    grace: f64,
    #[arg(long, default_value_t = 1.0)]
    poll_interval: f64,
    /// oom_score_adj for the serve's process tree.
    #[arg(long, default_value_t = 500)]
    oom_child_score: i32,
    /// Leave OOM scores alone.
    #[arg(long)]
    no_oom: bool,
    /// Restart a serve that crashed.
    #[arg(long)]
    auto_restart: bool,
    /// Stop the serve when the daemon exits (default: it outlives the daemon and is re-adopted).
    #[arg(long)]
    stop_serve_on_exit: bool,
    #[arg(long, default_value_t = 4000)]
    log_capacity: usize,
    /// Daemon URL for the client verbs.
    #[arg(long, env = "CARBON_DAEMON_URL", default_value = "http://127.0.0.1:1900")]
    url: String,
}

#[derive(Subcommand)]
enum DaemonVerb {
    /// The daemon's own health.
    #[command(name = "self")]
    SelfHealth,
    /// Serve status (pid, model, port, uptime, last exit).
    Status,
    /// The serve's /health, proxied.
    Health,
    /// The serve's process-tree RSS / PSS / VRAM.
    Metrics,
    /// The serve's /v1/stats, proxied.
    Stats,
    /// Start a serve: MODEL [--port N] [-- serve args].
    Start {
        model: String,
        #[arg(long)]
        port: Option<u16>,
        #[arg(last = true)]
        serve_args: Vec<String>,
    },
    /// Replace the running serve.
    Switch {
        model: String,
        #[arg(long)]
        port: Option<u16>,
        /// Stop even when the old serve cannot seal its accounting.
        #[arg(long)]
        force: bool,
        #[arg(last = true)]
        serve_args: Vec<String>,
    },
    Stop {
        #[arg(long)]
        force: bool,
    },
    /// Stop the serve and the daemon.
    Shutdown {
        #[arg(long)]
        force: bool,
    },
    /// Unacknowledged accounting receipts.
    Pending,
    /// Stream the serve's output.
    Logs {
        #[arg(long, default_value_t = 0)]
        since: u64,
    },
}

#[derive(Args)]
struct ShellArgs {
    /// Server URL (default: $CARBON_HOST, then http://127.0.0.1:1919).
    #[arg(long, alias = "base-url")]
    server: Option<String>,
    /// `--model PATH` and other `serve` flags: start an engine here and chat with it.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    serve_args: Vec<String>,
}

#[derive(Args)]
struct LaunchArgs {
    /// claude | codex | dsh | hermes | openclaw | opencode
    agent: String,
    /// Server origin or /v1 URL (default: $CARBON_HOST, then http://127.0.0.1:1919).
    #[arg(long)]
    server: Option<String>,
    /// Print the planned configuration and command without changing anything.
    #[arg(long)]
    dry_run: bool,
    /// Approve install and configuration prompts.
    #[arg(short = 'y', long)]
    yes: bool,
    /// Run the agent's installer even when it is already installed.
    #[arg(long)]
    force_reinstall: bool,
    /// Install and configure the agent without starting it.
    #[arg(long = "config")]
    config_only: bool,
    /// Install the agent CLI if missing, then exit (needs no server).
    #[arg(long)]
    install_only: bool,
    #[arg(last = true)]
    agent_args: Vec<String>,
}

#[derive(Args)]
struct CtlArgs {
    #[arg(long, default_value = "http://127.0.0.1:1919")]
    base_url: String,
    #[command(subcommand)]
    cmd: CtlCmd,
}

#[derive(Subcommand)]
enum CtlCmd {
    Health,
    Stats,
    /// Recent requests (status, latency, TTFT, tokens), paged with --since.
    Requests {
        #[arg(long, default_value_t = 0)]
        since: u64,
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },
    /// Show the cache pools, or resize them live with --kv / --moe.
    Cache {
        /// New KV cache size in tokens.
        #[arg(long)]
        kv: Option<usize>,
        /// New expert-cache size in experts.
        #[arg(long)]
        moe: Option<usize>,
        /// Seconds to wait for running requests to drain.
        #[arg(long, default_value_t = 300)]
        wait: u64,
    },
    Generate {
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        #[arg(long)]
        ignore_eos: bool,
    },
    /// Throughput benchmark: concurrent requests with random prompts (no prefix-cache hits).
    Bench {
        /// Requests in flight at once.
        #[arg(long, default_value_t = 8)]
        concurrency: usize,
        /// Total requests.
        #[arg(long, default_value_t = 32)]
        requests: usize,
        /// Prompt length in tokens.
        #[arg(long, default_value_t = 512)]
        prompt_tokens: usize,
        /// Generated tokens per request (always generated in full: EOS is ignored).
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Token ids are drawn from 100..vocab_limit.
        #[arg(long, default_value_t = 30000)]
        vocab_limit: u32,
    },
}

#[derive(Args)]
struct TinyArgs {
    #[arg(long)]
    out: PathBuf,
    /// llama | qwen2 | qwen3 | qwen3_moe | mixtral | qwen3_5_moe | qwen4_exp
    #[arg(long, default_value = "qwen3_moe")]
    arch: String,
    #[arg(long, default_value_t = 4)]
    layers: usize,
    #[arg(long, default_value_t = 64)]
    hidden: usize,
    #[arg(long, default_value_t = 8)]
    experts: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value = "f32")]
    dtype: String,
    /// Head count / head dim / expert width (defaults 4 / 16 / 32).
    #[arg(long)]
    heads: Option<usize>,
    #[arg(long)]
    head_dim: Option<usize>,
    #[arg(long)]
    moe_intermediate: Option<usize>,
    /// Quantize the projections: fp8 | nvfp4 | nvfp4-ct. The exact dequantized twin goes to `<out>-dense`.
    #[arg(long)]
    quant: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // the shell owns the terminal: only warnings unless RUST_LOG asks for more
    let default_level = if matches!(cli.cmd, Cmd::Shell(_)) {
        "warn"
    } else {
        "info"
    };
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| default_level.into()),
        )
        .init();
    match cli.cmd {
        Cmd::Serve(a) => serve(a),
        Cmd::Worker(a) => {
            init_cpu_threads(0);
            let devices = parse_device_list(&a.devices)?;
            let name = a
                .name
                .unwrap_or_else(|| std::env::var("HOSTNAME").unwrap_or_else(|_| "worker".into()));
            rpc::serve(WorkerConfig {
                listen: a.listen,
                devices,
                tp_size: a.tp_size,
                model_dir: a.model,
                name,
                moe: MoeOptions {
                    strategy: a.moe_strategy.parse()?,
                    cache_size: a.moe_cache_size,
                    ..Default::default()
                },
                token: a.token,
                cuda_graphs: a.cuda_graphs,
            })
        }
        Cmd::Generate(a) => generate(a),
        Cmd::Ctl(a) => ctl(a),
        Cmd::Launch(a) => launch(a),
        Cmd::Shell(a) => shell(a),
        Cmd::Bench { cmd } => bench_cmd(cmd),
        Cmd::Daemon(a) => daemon_cmd(a),
        Cmd::Checkpoint {
            model,
            out,
            dtype,
            shard_gib,
            progress,
        } => {
            use carbon_engine::checkpoint as c;
            let model = carbon_engine::hub::resolve(&model, carbon_engine::hub::Source::HuggingFace)?;
            let limit = ((shard_gib * (1u64 << 30) as f64) as u64 / c::ALIGN).max(1) * c::ALIGN;
            let index = c::convert(
                &model,
                &out,
                &c::ConvertOptions {
                    dtype: dtype.as_deref().map(parse_dtype).transpose()?.flatten(),
                    shard_limit: limit,
                    progress,
                },
            )?;
            println!(
                "wrote {} tensors ({:.2} GiB) in {} shards to {}",
                index["tensors"].as_array().map(|a| a.len()).unwrap_or(0),
                index["total_bytes"].as_f64().unwrap_or(0.0) / (1u64 << 30) as f64,
                index["shards"].as_array().map(|a| a.len()).unwrap_or(0),
                out.display()
            );
            Ok(())
        }
        Cmd::ChatTemplate(a) => {
            let tok = TokenizerBundle::load(&a.model, None)?;
            let req: Value = serde_json::from_str(&std::fs::read_to_string(&a.request)?)?;
            let msgs = req
                .get("messages")
                .and_then(|m| m.as_array())
                .context("request needs messages")?;
            let msgs = carbon_engine::server::openai::template_messages(msgs)
                .map_err(|e| anyhow::anyhow!(e.message))?;
            let kwargs = req
                .get("chat_template_kwargs")
                .and_then(|k| k.as_object())
                .cloned()
                .unwrap_or_default();
            let text = tok.apply_chat_template(&msgs, req.get("tools"), true, &kwargs)?;
            print!("{text}");
            if a.ids {
                eprintln!("{:?}", tok.encode_prompt(&text)?);
            }
            Ok(())
        }
        Cmd::Devices {
            selftest,
            devices,
            dtype,
        } => {
            for d in carbon_engine::device::describe_devices() {
                println!("{d}");
            }
            if selftest {
                let list = if devices.is_empty() {
                    let mut l = vec!["cpu".to_string()];
                    l.extend(
                        (0..carbon_engine::device::cuda_device_count())
                            .map(|i| format!("cuda:{i}")),
                    );
                    l
                } else {
                    parse_device_list(&devices)?
                };
                let mut failed = 0;
                for name in list {
                    let dev = carbon_engine::device::parse_device(&name)?;
                    let dtypes = match parse_dtype(&dtype)? {
                        Some(d) => vec![d],
                        None if dev.is_cuda() => vec![
                            candle_core::DType::BF16,
                            candle_core::DType::F16,
                            candle_core::DType::F32,
                        ],
                        None => vec![carbon_engine::device::default_dtype(&dev)],
                    };
                    for dt in dtypes {
                        // carbon never runs a CPU stage in another dtype than its default
                        // (f32): a missing CPU kernel there is informational, not a failure
                        let unused = !dev.is_cuda() && dt != carbon_engine::device::default_dtype(&dev);
                        println!(
                            "\n{name} [{}]{}",
                            carbon_engine::device::dtype_name(dt),
                            if unused { "  (not used by carbon on this device; failures are informational)" } else { "" }
                        );
                        for (op, r) in carbon_engine::device::kernel_selftest(&dev, dt) {
                            match r {
                                Ok(()) => println!("  ok    {op}"),
                                Err(e) if unused => println!("  n/a   {op}: {e}"),
                                Err(e) => {
                                    failed += 1;
                                    println!("  FAIL  {op}: {e}");
                                }
                            }
                        }
                    }
                }
                if failed > 0 {
                    bail!("{failed} kernel checks failed");
                }
            }
            Ok(())
        }
        Cmd::TinyModel(a) => {
            let mut spec = carbon_engine::testing::TinySpec::new(&a.arch);
            spec.layers = a.layers;
            spec.hidden = a.hidden;
            spec.experts = a.experts;
            spec.seed = a.seed;
            spec.dtype = parse_dtype(&a.dtype)?.unwrap_or(candle_core::DType::F32);
            if let Some(h) = a.heads {
                spec.heads = h;
                spec.kv_heads = (h / 2).max(1);
            }
            spec.head_dim = a.head_dim.unwrap_or(spec.head_dim);
            spec.moe_intermediate = a.moe_intermediate.unwrap_or(spec.moe_intermediate);
            spec.intermediate = spec.moe_intermediate * 4;
            match a.quant.as_deref() {
                None => carbon_engine::testing::write_tiny_model(&a.out, &spec)?,
                Some(q) => {
                    use carbon_engine::testing::TinyQuant;
                    let quant = match q {
                        "fp8" => TinyQuant::Fp8Block,
                        "nvfp4" => TinyQuant::Nvfp4ModelOpt,
                        "nvfp4-ct" => TinyQuant::Nvfp4CompressedTensors,
                        o => bail!("unknown --quant '{o}' (fp8 | nvfp4 | nvfp4-ct)"),
                    };
                    let dense = PathBuf::from(format!("{}-dense", a.out.display()));
                    carbon_engine::testing::write_tiny_quantized(&a.out, &dense, &spec, quant)?;
                    println!("wrote dense twin to {}", dense.display());
                }
            }
            println!("wrote {} model to {}", a.arch, a.out.display());
            Ok(())
        }
    }
}

fn parser_flag(v: &str, available: bool, name: &str) -> Result<bool> {
    Ok(match v {
        "auto" => available,
        "on" | "hermes" | "think" | "qwen3" => true,
        "off" | "none" => false,
        o => bail!("unknown {name} '{o}'"),
    })
}

fn serve(a: ServeArgs) -> Result<()> {
    let addr = format!("{}:{}", a.host, a.port);
    let state = build_state(a)?;
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(server::serve(state, &addr))
}

#[derive(Parser)]
struct ServeOnly {
    #[command(flatten)]
    serve: ServeArgs,
}

fn shell(a: ShellArgs) -> Result<()> {
    if a.serve_args.is_empty() {
        let url = carbon_engine::launch::resolve_server_url(a.server.as_deref())?;
        return carbon_engine::shell::run(&url.origin);
    }
    // start an engine here, serving one conversation at a time on a loopback port
    let mut argv = vec!["carbon-shell".to_string()];
    argv.extend(a.serve_args);
    let mut sa = ServeOnly::try_parse_from(argv)?.serve;
    sa.engine.max_running_requests = 1;
    let state = build_state(sa)?;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        rt.block_on(async move {
            let l = match tokio::net::TcpListener::bind("127.0.0.1:0").await {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(Err(e.to_string()));
                    return;
                }
            };
            let _ = tx.send(Ok(l.local_addr().unwrap()));
            if let Err(e) = server::serve_listener(state, l).await {
                tracing::error!("in-process server stopped: {e:#}");
            }
        })
    });
    let addr = rx.recv()?.map_err(anyhow::Error::msg)?;
    carbon_engine::shell::run(&format!("http://{addr}"))
}

fn build_state(mut a: ServeArgs) -> Result<Arc<AppState>> {
    let model_spec = a.engine.model.clone();
    a.engine.resolve_model()?;
    let tok = Arc::new(TokenizerBundle::load(
        &a.engine.model,
        a.chat_template.as_deref(),
    )?);
    let reasoning = parser_flag(
        &a.reasoning_parser,
        tok.token_id("<think>").is_some(),
        "reasoning parser",
    )?;
    let tools = parser_flag(
        &a.tool_call_parser,
        tok.token_id("<tool_call>").is_some(),
        "tool-call parser",
    )?;
    let anchor = a
        .enable_special_token_ckpt
        .then(|| tok.token_id("<tool_call>"))
        .flatten();
    let (handle, _) = a.engine.start(anchor)?;
    let model_name = a.served_model_name.clone().unwrap_or_else(|| {
        let spec = model_spec.to_string_lossy();
        let spec = spec.split('@').next().unwrap_or(&spec);
        std::path::Path::new(spec.trim_end_matches('/'))
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".into())
    });
    let defaults = if a.sampling_defaults == "none" {
        SamplingDefaults::default()
    } else {
        SamplingDefaults::from_model_dir(&a.engine.model)
    };
    let inline_system = match a.anthropic_inline_system.as_str() {
        "auto" => None,
        "preserve" => Some(InlineSystem::Preserve),
        "fold" => Some(InlineSystem::Fold),
        o => bail!("unknown --anthropic-inline-system '{o}' (auto | preserve | fold)"),
    };
    let cors_origins = a
        .cors_origins
        .split(',')
        .map(|o| o.trim().to_string())
        .filter(|o| !o.is_empty())
        .collect();
    let state = Arc::new(AppState::new(
        handle,
        tok,
        model_name,
        defaults,
        ServerOptions {
            max_output_tokens: a.max_output_tokens,
            reasoning,
            tools,
            cache_report: a.enable_cache_report,
            inline_system,
            cors_origins,
        },
    ));
    Ok(state)
}

fn generate(mut a: GenerateArgs) -> Result<()> {
    a.engine.resolve_model()?;
    let tok = TokenizerBundle::load(&a.engine.model, None)?;
    // base64 images become data: URLs, after the --image ones
    for b in std::mem::take(&mut a.image_base64) {
        a.image.push(if b.starts_with("data:") { b } else { format!("data:image;base64,{}", b.trim()) });
    }
    let ids = if let Some(list) = &a.prompt_ids {
        list.split(',')
            .map(|x| x.trim().parse::<u32>().context("bad --prompt-ids"))
            .collect::<Result<Vec<_>>>()?
    } else if a.chat || !a.image.is_empty() {
        let mut content: Vec<Value> = a.image.iter().map(|i| json!({"type": "image", "image": i})).collect();
        content.push(json!({"type": "text", "text": a.prompt}));
        let prompt = tok.apply_chat_template(
            &[json!({"role": "user", "content": content})],
            None,
            true,
            &Default::default(),
        )?;
        tok.encode_prompt(&prompt)?
    } else {
        tok.encode(&a.prompt, true)?
    };
    let (handle, _) = a.engine.start(None)?;
    let (ids, images) = if a.image.is_empty() {
        (ids, vec![])
    } else {
        let info = &handle.info;
        let (Some(proc), Some(image_id)) = (info.image_processor.as_ref(), info.image_token_id) else {
            bail!("this model does not take image input");
        };
        let images = a
            .image
            .iter()
            .map(|i| proc.prepare(&carbon_engine::images::fetch(i)?).with_context(|| format!("image {i}")))
            .collect::<Result<Vec<_>>>()?;
        carbon_engine::images::expand_image_tokens(&ids, image_id, images, proc.merge_size)?
    };
    let params = SamplingParams {
        temperature: a.temperature,
        top_p: a.top_p,
        top_k: a.top_k,
        max_tokens: a.max_tokens,
        seed: a.seed,
        ignore_eos: a.ignore_eos,
        ..Default::default()
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let t0 = Instant::now();
        let (_, mut rx) = handle.submit_with_images(ids.clone(), images, params);
        let mut detok = DecodeStream::new(true);
        let mut first = None;
        let mut out = std::io::stdout();
        while let Some(ev) = rx.recv().await {
            match ev {
                Event::Token(t) => {
                    first.get_or_insert_with(|| t0.elapsed());
                    if let Some(d) = detok.step(&tok, t) {
                        write!(out, "{d}")?;
                        out.flush()?;
                    }
                }
                Event::Done { reason, usage } => {
                    if a.print_ids {
                        eprintln!("\noutput_ids: {:?}", detok.ids());
                    }
                    let total = t0.elapsed().as_secs_f64();
                    let ttft = first.map(|d| d.as_secs_f64()).unwrap_or(total);
                    let decode_tps = (usage.completion_tokens.saturating_sub(1)) as f64 / (total - ttft).max(1e-9);
                    eprintln!(
                        "\n[{reason:?}] prompt {} tok, output {} tok, ttft {:.3}s, decode {:.1} tok/s",
                        usage.prompt_tokens, usage.completion_tokens, ttft, decode_tps
                    );
                    if let Some(st) = handle.stats().await {
                        if st.mtp_drafts > 0 {
                            eprintln!(
                                "[mtp] {} of {} drafts accepted ({:.1}%), {:.2} ms per draft step, {:.2} ms per verify step",
                                st.mtp_accepted,
                                st.mtp_drafts,
                                100.0 * st.mtp_accepted as f64 / st.mtp_drafts as f64,
                                st.mtp_draft_ms,
                                st.mtp_verify_ms
                            );
                        }
                    }
                    return Ok(());
                }
                Event::Error(e) => bail!("generation failed: {e}"),
            }
        }
        bail!("engine stopped")
    })
}

/// Size rayon's global pool (all CPU compute): `n`, else physical cores, unless
/// RAYON_NUM_THREADS is set.
fn init_cpu_threads(n: usize) {
    if n == 0 && std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return;
    }
    let n = if n == 0 {
        carbon_engine::device::physical_cores()
    } else {
        n
    };
    let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
}

fn daemon_cmd(a: DaemonArgs) -> Result<()> {
    use carbon_engine::daemon as d;
    let Some(verb) = a.verb else {
        return d::run(d::DaemonConfig {
            host: a.host,
            port: a.port,
            state_dir: a.state_dir.unwrap_or_else(d::DaemonConfig::default_state_dir),
            token: a.token,
            default_serve_port: a.default_serve_port,
            grace: std::time::Duration::from_secs_f64(a.grace.max(0.0)),
            poll_interval: std::time::Duration::from_secs_f64(a.poll_interval.max(0.05)),
            oom_child_score: (!a.no_oom).then_some(a.oom_child_score),
            auto_restart: a.auto_restart,
            stop_serve_on_exit: a.stop_serve_on_exit,
            log_capacity: a.log_capacity,
            setsid: true,
            serve_exe: std::env::current_exe()?,
        });
    };
    let (name, body, since) = match verb {
        DaemonVerb::SelfHealth => ("self", None, 0),
        DaemonVerb::Status => ("status", None, 0),
        DaemonVerb::Health => ("health", None, 0),
        DaemonVerb::Metrics => ("metrics", None, 0),
        DaemonVerb::Stats => ("stats", None, 0),
        DaemonVerb::Pending => ("pending", None, 0),
        DaemonVerb::Start {
            model,
            port,
            serve_args,
        } => (
            "start",
            Some(json!({"model": model, "port": port, "args": serve_args})),
            0,
        ),
        DaemonVerb::Switch {
            model,
            port,
            force,
            serve_args,
        } => (
            "switch",
            Some(json!({"model": model, "port": port, "args": serve_args, "force": force})),
            0,
        ),
        DaemonVerb::Stop { force } => ("stop", Some(json!({"force": force})), 0),
        DaemonVerb::Shutdown { force } => ("shutdown", Some(json!({"force": force})), 0),
        DaemonVerb::Logs { since } => ("logs", None, since),
    };
    d::client(&a.url, a.token.as_deref(), name, body, since)
}

fn bench_cmd(cmd: BenchCmd) -> Result<()> {
    use carbon_engine::benchbw as b;
    let BenchCmd::Bw {
        dtype,
        device,
        model,
        threshold,
        seconds,
        bank_mib,
        cpu_threads,
        out,
    } = cmd;
    init_cpu_threads(cpu_threads);
    let mut opts = b::BenchOptions {
        formats: b::ExpertFormat::parse(&dtype)?,
        device: (!device.is_empty()).then(|| device.clone()),
        threshold,
        seconds,
        bank_bytes: bank_mib << 20,
        ..Default::default()
    };
    if let Some(m) = model {
        let cfg = carbon_engine::config::ModelConfig::from_dir(&m)?;
        opts.hidden = cfg.hidden_size;
        opts.intermediate = cfg
            .moe
            .as_ref()
            .map(|m| m.intermediate_size)
            .context("--model is not a MoE model")?;
    }
    let prof = b::run(&opts)?;
    print!("{}", b::summary(&prof));
    if opts.device.is_some() {
        let path = out.unwrap_or_else(|| b::default_profile_path(opts.device.as_deref()));
        b::write_profile(&path, &prof)?;
        println!("profile written to {}", path.display());
    } else {
        println!("(host only: pass --device cuda:N to measure uploads and write a profile)");
    }
    Ok(())
}

fn launch(a: LaunchArgs) -> Result<()> {
    use carbon_engine::launch as l;
    if !l::AGENTS.contains(&a.agent.as_str()) {
        bail!("unknown agent '{}' ({})", a.agent, l::AGENTS.join(" | "));
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("HOME is not set")?;
    if a.install_only {
        l::ensure_installed(&a.agent, &home, a.yes, a.force_reinstall)?;
        return Ok(());
    }
    let server = l::resolve_server_url(a.server.as_deref())?;
    let model = l::discover_model(&server)?;
    let mut ctx = l::LaunchContext {
        server,
        model,
        extra_args: a.agent_args,
        dry_run: true,
        assume_yes: a.yes,
        home: home.clone(),
    };
    let spec = l::prepare(&a.agent, &ctx)?;
    if a.dry_run {
        l::print_plan(&ctx, &spec);
        return Ok(());
    }
    let binary = l::ensure_installed(&a.agent, &home, a.yes, a.force_reinstall)?;
    ctx.dry_run = false;
    let mut spec = l::prepare(&a.agent, &ctx)?;
    spec.argv[0] = binary.to_string_lossy().into();
    if a.config_only {
        return Ok(());
    }
    std::process::exit(l::run(&spec)?);
}

fn ctl(a: CtlArgs) -> Result<()> {
    let base = a.base_url.trim_end_matches('/');
    if let CtlCmd::Bench {
        concurrency,
        requests,
        prompt_tokens,
        max_tokens,
        vocab_limit,
    } = a.cmd
    {
        return bench(
            base,
            concurrency.max(1),
            requests,
            prompt_tokens,
            max_tokens,
            vocab_limit.max(101),
        );
    }
    let v: Value = match a.cmd {
        CtlCmd::Health => ureq::get(&format!("{base}/health")).call()?.into_json()?,
        CtlCmd::Stats => ureq::get(&format!("{base}/v1/stats")).call()?.into_json()?,
        CtlCmd::Requests { since, limit } => {
            ureq::get(&format!("{base}/v1/requests?since={since}&limit={limit}"))
                .call()?
                .into_json()?
        }
        CtlCmd::Cache { kv: None, moe: None, .. } => ureq::get(&format!("{base}/v1/cache/status")).call()?.into_json()?,
        CtlCmd::Cache { kv, moe, wait } => ureq::post(&format!("{base}/v1/cache/rebuild"))
            .send_json(json!({"kv_tokens": kv, "moe_cache_size": moe, "wait": wait}))?
            .into_json()?,
        CtlCmd::Generate { prompt, max_tokens, ignore_eos } => ureq::post(&format!("{base}/generate"))
            .send_json(json!({"text": prompt, "sampling_params": {"max_new_tokens": max_tokens, "ignore_eos": ignore_eos}}))?
            .into_json()?,
        CtlCmd::Bench { .. } => unreachable!(),
    };
    println!("{}", serde_json::to_string_pretty(&v)?);
    Ok(())
}

/// `ctl bench`: `concurrency` client threads share `requests` requests of random prompts.
fn bench(
    base: &str,
    concurrency: usize,
    requests: usize,
    prompt_tokens: usize,
    max_tokens: usize,
    vocab_limit: u32,
) -> Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, usize, f64)>> = Mutex::new(vec![]);
    let errors = AtomicUsize::new(0);
    let t0 = std::time::Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..concurrency {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= requests {
                    break;
                }
                // deterministic per request, distinct across requests: no shared prefixes
                let mut x =
                    0x9e37_79b9_7f4a_7c15u64 ^ (i as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                let ids: Vec<u32> = (0..prompt_tokens)
                    .map(|_| {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        100 + (x % (vocab_limit as u64 - 100)) as u32
                    })
                    .collect();
                let body = json!({"input_ids": ids, "sampling_params":
                    {"max_new_tokens": max_tokens, "ignore_eos": true, "temperature": 0.0}});
                let started = std::time::Instant::now();
                match ureq::post(&format!("{base}/generate"))
                    .send_json(body)
                    .map_err(anyhow::Error::from)
                    .and_then(|r| Ok(r.into_json::<Value>()?))
                {
                    Ok(v) => {
                        let m = &v["meta_info"];
                        let p = m["prompt_tokens"].as_u64().unwrap_or(0) as usize;
                        let c = m["completion_tokens"].as_u64().unwrap_or(0) as usize;
                        results
                            .lock()
                            .unwrap()
                            .push((p, c, started.elapsed().as_secs_f64()));
                    }
                    Err(e) => {
                        errors.fetch_add(1, Ordering::Relaxed);
                        eprintln!("request {i} failed: {e:#}");
                    }
                }
            });
        }
    });
    let wall = t0.elapsed().as_secs_f64();
    let r = results.into_inner().unwrap();
    let (p, c): (usize, usize) = r.iter().fold((0, 0), |a, x| (a.0 + x.0, a.1 + x.1));
    let mut lat: Vec<f64> = r.iter().map(|x| x.2).collect();
    lat.sort_by(f64::total_cmp);
    let pct = |q: f64| {
        lat.get(((lat.len() as f64 - 1.0) * q).round() as usize)
            .copied()
            .unwrap_or(0.0)
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "requests_ok": r.len(),
            "requests_failed": errors.into_inner(),
            "concurrency": concurrency,
            "wall_s": wall,
            "prompt_tokens": p,
            "completion_tokens": c,
            "output_tok_per_s": c as f64 / wall,
            "total_tok_per_s": (p + c) as f64 / wall,
            "latency_s": {"p50": pct(0.5), "p90": pct(0.9), "max": pct(1.0)},
        }))?
    );
    Ok(())
}
