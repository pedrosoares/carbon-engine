# carbon-engine

A Mixture-of-Experts serving engine in Rust, inspired by [FreeToken](https://github.com/FlashML-org/FreeToken).
Like FreeToken it runs large open-weight MoE models on hardware you own, keeping experts that do
not fit in VRAM in host memory, and serves OpenAI and Anthropic compatible APIs so coding agents
(Claude Code, Codex, OpenCode, ...) can point at it unchanged. It adds two things FreeToken does
not do:

- **Multi-GPU inference**: pipeline parallelism across GPUs, tensor parallelism (attention heads +
  dense MLPs sharded, NCCL or host all-reduce) and expert parallelism for MoE layers, freely
  combined.
- **Multi-machine inference over RPC**: any number of `carbon worker` processes on other hosts each
  run a slice of the layers as a remote pipeline stage. Workers can load weights from their own
  disk or receive their shard over the network.

## Stack

| Concern | Choice | Why |
|---|---|---|
| Language | **Rust** | memory safety around concurrent schedulers/RPC, no GIL, single static binary |
| Tensors / kernels | [candle](https://github.com/huggingface/candle) 0.11 | one code path for CPU, CUDA (cuBLAS) and Metal; NCCL via cudarc |
| HTTP | axum + tokio, SSE streaming | |
| Tokenizer | HF `tokenizers`, chat templates rendered with `minijinja` (+ Python string-method compat) | loads `tokenizer.json` / `tokenizer_config.json` unchanged |
| RPC | length-prefixed bincode frames over TCP | no codegen, raw tensor bytes, tiny dependency surface |

## Compared with FreeToken

Both engines target the same use: frontier MoE models on consumer and workstation hardware, served
to agents. They overlap in most serving features and differ in scale-out, model coverage and a
few runtime choices.

**Similar**

| Feature | FreeToken | carbon-engine |
|---|---|---|
| APIs | OpenAI (chat, completions, Responses) and Anthropic compatible, streaming, tool calls, reasoning | the same |
| Batching | continuous batching, chunked prefill | continuous batching, chunked prefill under a token budget, reservation-based admission |
| Prefix cache | radix tree; recurrent-state checkpoints for hybrid models; anchor checkpoints for agent edits | radix tree; recurrent-state snapshots for hybrid models (`--state-snapshots`); anchor checkpoints (`--enable-special-token-ckpt`) |
| MoE placement | resident, offload (global LRU expert cache), CPU, hybrid CPU-GPU split, picked from a bandwidth benchmark | the same four strategies, picked from `carbon bench bw` |
| Offloaded experts | expert cache on the GPU fed from pinned host memory, CUDA-graph compatible | a device slot cache fed from pinned host memory by a GPU kernel at PCIe speed, routing and caching on the GPU, CUDA-graph compatible |
| Elastic memory | VRAM moved between KV and experts at runtime | `POST /v1/cache/rebuild`, `carbon ctl cache --kv N --moe N` |
| Quantized weights | FP8, NVFP4, MXFP4, GGUF Q4_0, BF16 | FP8 (block and per-tensor), NVFP4 (ModelOpt and compressed-tensors), BF16/F16/F32 |
| Hybrid attention models | Gated DeltaNet (Qwen3.5/3.6/3.8) | Gated DeltaNet (Qwen3.5/3.6/3.8), fused decode and prefill kernels |
| CUDA graphs | decode steps | decode steps, including offloaded experts and multi-token (MTP verify) steps |
| CLI | `ft serve / ctl / bench` | `carbon serve / generate / ctl / bench bw / devices / worker / launch / shell` |

**Different**

| | FreeToken | carbon-engine |
|---|---|---|
| GPUs per server | one (`--gpu`) | any number, in pipeline and/or tensor-parallel groups |
| Machines | one | several, over RPC (`carbon worker`) |
| Speculative decoding | none | MTP drafts for Qwen3.8-Flash-Next (`--mtp`), verified in one multi-token step |
| Models | Llama, Qwen2/3, Qwen3-MoE, Qwen3.5/3.6/3.8, DeepSeek-V4, GLM, gpt-oss, Gemma-4, MiniMax, ... | Llama/Mistral, Qwen2/2.5, Qwen3, Qwen3-MoE, Mixtral, Qwen3.5/3.6 (dense + MoE), Qwen3.8-Flash-Next |
| Image input | Qwen3.6/3.8, Qwen3-VL, Gemma-4, GLM, ... | Qwen3.5/3.6/3.8 (their vision tower); no video |
| Fast-load weight format | FTW | the checkpoint format of `carbon checkpoint` |
| Prefill with offloaded experts | whole layers streamed, double-buffered behind compute | through the slot cache: a layer's missing experts are copied in batches on a side stream while its resident ones compute, then grouped tensor-core NVFP4 expert GEMMs |
| Hardware | NVIDIA RTX 30/40/50; AMD ROCm (experimental) | NVIDIA (its CUDA kernels compile at start-up for the installed driver) and CPU; a Metal build through candle |
| Distribution | Python package, desktop app | one Rust binary |
| Implementation | Python, Triton and CUDA on PyTorch | Rust on [candle](https://github.com/huggingface/candle), with its own CUDA kernels |

Not supported yet (see [Roadmap](#roadmap)): MXFP4 and GGUF weights, sliding-window and MLA
attention (DeepSeek, GLM, gpt-oss, Gemma), video input.

## Running Qwen3.6-35B-A3B-FP8

Hybrid layout: 40 layers (30 Gated DeltaNet + 10 gated full attention), 256 experts (top-8)
plus a shared expert, block-FP8 weights. Point `--model` at the downloaded HF directory;
the vision tower serves image input (see [Images](#images); `--text-only` skips loading it) and
the MTP head is ignored.

Memory: routed experts stay FP8 (about 32 GB; `--dequant-fp8-experts` makes them bf16,
64 GB), everything else is bf16 (about 6 GB). Per sequence the linear layers keep ~66 MB of
recurrent state; KV costs only ~20 KB per token (10 attention layers × 2 KV heads).

```bash
# 2 x 24 GB GPUs, experts resident (FP8), pipeline parallel
carbon serve --model ~/models/Qwen3.6-35B-A3B-FP8 --devices cuda:0,cuda:1

# one 32 GB GPU (RTX 5090): experts in host RAM (FP8), the hot ones cached on the GPU
carbon serve --model ~/models/Qwen3.6-35B-A3B-FP8 --devices cuda:0 --moe-strategy offload

# 2 x 48 GB GPUs, tensor + expert parallel (heads and experts split across both)
carbon serve --model ~/models/Qwen3.6-35B-A3B-FP8 --devices cuda:0,cuda:1 --tp-size 2   # build with --features nccl

# two machines: half the layers on a worker that receives its weights over the network
carbon worker --listen 0.0.0.0:7070 --devices cuda:0 --token "$SECRET"            # host B
carbon serve --model ~/models/Qwen3.6-35B-A3B-FP8 --devices cuda:0 \
  --rpc-worker hostB:7070 --rpc-token "$SECRET"                                     # host A
```

Thinking is on by default (the template opens `<think>`); output is split into
`reasoning_content` / Anthropic `thinking` blocks. Pass
`"chat_template_kwargs": {"enable_thinking": false}` (OpenAI) or
`"thinking": {"type": "disabled"}` (Anthropic) to turn it off. Tool calls use the model's
XML format and are returned as structured `tool_calls` / `tool_use` blocks.

Prefix reuse works for the hybrid layers through recurrent-state snapshots (as in FreeToken):
the state is saved one token before each prompt's end and when a request finishes, at radix-tree
boundaries. A later request resumes from the deepest snapshot on its path, so repeated system
prompts, regenerations and agent turns (prompt + previous answer + tool result) skip their
cached part. `--state-snapshots N` sets how many are kept (default: `--max-running-requests`;
each costs ~66 MB for Qwen3.6-35B-A3B); `0` disables reuse for hybrid models.

## Running Qwen3.8-Flash-Next

```bash
carbon serve --model RadixArk/Qwen3.8-Flash-Next-NVFP4 --devices cuda:0 --moe-strategy auto
```

Qwen3.8-Flash-Next (`model_type` `qwen4_exp`) is the Qwen3.6 hybrid stack with three additions,
all checked token for token against Hugging Face `transformers` on tiny random checkpoints
(`scripts/qwen38_reference.py` + `scripts/compare_reference.py`):

- **hyper-connections**: the residual stream is `hc_count` (4) streams wide; every mixer and MoE
  block reads a gated mix of the streams and is injected back into each with its own weight.
- **PLE**: one linear layer adds hashed 2- and 3-gram embeddings (16 heads into prime-sized
  tables) through a gated value and a dilated depthwise conv. The FP8 table is 47.7 GiB, so it is
  memory-mapped from the checkpoint and only the looked-up rows are read (in parallel); the page
  cache keeps the hot ones. `CARBON_PLE_PRELOAD=1` asks the kernel to read it ahead instead. Token
  ids travel with each batch (also to RPC workers), so the n-gram context needs no extra state;
  the conv state rides the linear-attention state slots, prefix snapshots included. A prefill
  requests all of a chunk's rows from disk at once (queue depth for the NVMe), and a new prompt's
  rows past its first chunk are prefetched in the background at admission.
- **QSA**: each attention layer's indexer scores blocks of 4 tokens (mean-pooled keys) and keeps
  the best 512 blocks (2048 tokens) plus the partial last block. While a sequence has no more
  blocks than that, attention is dense and uses the usual kernels; beyond it, decode attends over
  the selected slots only and prefill applies the selection as a mask. On CUDA a decode step's
  selection runs in one kernel (`kernels/qsa.cu`: pooled keys, scores, a radix top-k and the slot
  table the fused paged kernel reads), with no host round trip; prefill chunks pick their blocks
  per query with a radix top-k into bitmasks that expand to the attention mask on the device. Indexer keys add 128 values per
  token to the KV cache.

The Gated DeltaNet output gate is `sigmoid` in this model (`output_gate_type`); the fused decode
kernel handles both. `--cuda-graphs` covers it: the PLE rows are gathered before the captured
layers into a static buffer, the PLE conv decodes in one branch-free batched step and the QSA
selection stays on the device.

With `--moe-strategy offload|hybrid` only GPU stages get an expert cache (a CPU stage keeps its
experts resident), and an auto-sized cache is sized after the stage's other weights are loaded,
keeping the KV reserve (`--kv-reserve-tokens`, default one full context), the recurrent-state pools
of hybrid models (`--max-running-requests` live slots plus prefix snapshots) and 1.5 GiB + 5% of
the card free for activations. Cached experts live in a CUDA memory pool of their own, so expert
churn cannot fragment the pool the activations use.

`offload` on a single GPU (no tensor parallelism) keeps the experts in pinned, device-mapped host
memory behind a device **slot cache** (`model/bank.rs`): routing, the LRU slot assignment and the
copy of missing experts (a kernel reading pinned memory at PCIe speed) all run on the GPU, then the
fused expert kernels read the slots through the layer's pointer table. A decode step never waits
on the host, so `--cuda-graphs` captures it. Prefill chunks share the cache: the route, slot map
and copy queue are read back once per layer, the missing experts are copied in batches of 32 on a
side stream, and the expert GEMMs (`kernels/moe_tc.cu`, tensor cores, one launch per batch) start
with the resident experts and take each batch as it lands (`CARBON_MOE_TC=0`: one projection per
expert instead).
It needs host RAM to pin every offloaded expert (64 GB for Qwen3.8-Flash-Next NVFP4) plus 10%
(at least 8 GiB) spare; otherwise, with `CARBON_MOE_BANK=0`, or for experts the fused kernels do
not read, the host-routed LRU cache below serves the stage. The slot cache holds at least one
layer's experts. On an RTX 5090 Qwen3.8-Flash-Next-NVFP4 decodes at ~49 tok/s single-stream
(host-routed: ~18) and `carbon ctl bench` reports ~67 output tok/s (was ~18).

Time to first token on the same card (random-token prompts, no prefix-cache hits; see
`TTFT_PLAN.md` for the profile): 512 tokens 0.7 s, 2048 1.5 s, 4096 2.7 s, 8192 5.4 s,
16384 11.7 s; a 640x480 image with a short question 1.15 s.

On a CUDA head an untied input embedding stays in host memory: each step gathers its rows there.

`--cuda-graphs` (or `CARBON_CUDA_GRAPHS=1`; `worker --cuda-graphs` on RPC workers) replays
single-GPU decode steps of up to 16 sequences as CUDA graphs, removing the CPU cost of the
hundreds of kernel launches per token. It engages only when every layer of the stage decodes on
carbon's fused kernels (whose launch shapes depend on the batch size alone); each batch size is
captured the second time it is seen, a capture during which anything fell back to a generic path
is discarded, and any capture or launch error turns graphs off and decodes eagerly. Each batch
size's capture step and first replay are also checked against an eager run of the same step (the
pool rows the step writes are saved and restored around it); a difference turns graphs off with a
warning and the step's eager result is used (`CARBON_GRAPH_VALIDATE=N` checks N replays, `0`
none). `/v1/stats` reports `cuda_graphs_disabled` with the reason. The stats
endpoint reports how many batch sizes are captured per stage (`cuda_graphs`). It is opt-in until
it has been validated on more hardware: `CARBON_TEST_DEVICE=cuda cargo test --features cuda
--test parallel cuda_graphs` checks graph decoding against eager decoding token for token.

### MTP speculative decoding (`--mtp`)

The checkpoint's multi-token-prediction head (`mtp.*`, after vLLM's `qwen4_exp` MTP) drafts the
next one or two tokens; each step verifies them in one multi-token decode step and keeps the
drafts that match what the model samples, so the output is unchanged (exact for greedy and for
sampling). Per request, two drafts are used while their acceptance stays high (`--mtp-drafts`,
default 2 = adaptive one/two), one otherwise. Rejected drafts roll back without copies (their KV
slots are freed, their recurrent state was written to spare state slots). The head's experts are
quantized to NVFP4 into the expert slot cache. Speculation runs on a single local CUDA device for
up to 2 decoding requests (with offloaded experts it gains +37% at one, +13% at two and would
lose at four: each verified token adds its own expert copies); larger batches decode one token
per step while the head only keeps its KV current.
`--mtp-shadow` only measures how often the drafts would be accepted. Use `--mtp` for one or two
concurrent requests: at higher concurrency it never speculates, and the VRAM its head and spare
state slots take (~0.65 GB, ~230 expert slots on a 32 GB card) costs ~3-8% (`ctl bench`, 8
concurrent: 63.7 / 77.5 tok/s plain vs 62.0 / 71.4 with `--mtp`, cold / warm).

RTX 5090, `--moe-strategy offload --cuda-graphs`, 400-token answers, tok/s (greedy / default
sampling):

| workload | plain | `--mtp` | drafts accepted |
|---|---|---|---|
| story | 52.5 / 49.2 | 56.2 / 53.0 | 64-69% |
| code | 48.4 / 46.8 | 58.9 / 55.5 | 81-83% |
| math | 43.8 / 41.8 | 56.3 / 53.3 | 88-92% |
| JSON | 46.2 / 44.8 | 59.3 / 57.8 | 92-93% |

`ctl bench --prompt-tokens 64 --max-tokens 256` (output tok/s, plain / `--mtp`): one request
47.4 / 55.3, two 59.5 / 74.9, four 81.3 / 74.1.

## Architecture

```
                   HTTP (OpenAI / Anthropic)                 carbon serve (coordinator)
                              |
                      tokenizer + chat template
                              |
                  +-----------v-----------+
                  |  scheduler thread      |  radix prefix cache, slot allocator,
                  |  (continuous batching) |  chunked prefill, sampling
                  +-----------+-----------+
                              | BatchMeta (positions, out_loc, per-seq slot tables)
       +----------------------v---------------------------------------------------+
       | Engine: embed ──► stage 0 ──► stage 1 ──► ... ──► stage N ──► norm + lm_head |
       +-------------|------------|-------------------------|-----------------------+
                     |            |                         |
              LocalStage     TpStage (cuda:1,cuda:2)   RemoteStage ──TCP──► carbon worker
              (cuda:0)       rank threads + NCCL /                         (its own LocalStage /
                             host all-reduce, experts                      TpStage / pipeline,
                             split across ranks                            own KV for its layers)
```

* The **coordinator owns all KV bookkeeping** (slot allocation, radix tree). Every
  stage, local or remote, keeps a KV pool of the same size for its own layers and
  writes/reads it at the slots the scheduler chose, so prefix caching works across the
  whole cluster with no extra protocol.
* Slot tables are append-only per request, so the RPC client only sends the **new
  slots** each step; workers rebuild the full table and drop it on `Release`.
* Every pipeline stage is a `Stage` (`parallel/mod.rs`); stages compose (a worker can
  itself run a tensor-parallel group or a pipeline over several GPUs).
* Attention is paged: new K/V rows are scattered into the pool, each sequence gathers its
  history through its slot table. Prefill is query-tiled (bounded score memory), decode
  batches all sequences into one padded gather.

## GPU kernels

carbon compiles its own CUDA kernels with NVRTC at start-up, for the oldest virtual architecture
the installed toolkit accepts, so they run on every newer GPU. Each falls back to a generic path
by itself if it ever fails.

- **Quantized weights**: FP8 (block and per-tensor) and NVFP4 experts stay quantized and are
  decoded per GEMM: warp-per-row GEMV from the packed weights for decode, one-pass dequant +
  cuBLAS for prefill; fused rayon kernels on the CPU.
- **Attention**: a token-granular paged KV pool with per-sequence slot tables. Decode runs one
  block per sequence and head with an online softmax over the slot table; prefill is
  FlashAttention-2 style on tensor cores (bf16/f16, sm_80+, 64-query x 32-key tiles, causal tile
  skipping), so neither gathers K/V nor materialises scores. GQA, head dims up to 256.
  `CARBON_FUSED_ATTN=0` turns them off, `CARBON_ATTN_TC=0` selects the FP32 prefill kernel.
- **Gated DeltaNet**: a decode step is two kernels that update the state pools in place (causal
  conv + SiLU with the conv-state shift; then the gated delta rule on the f32 state and the gated
  RMSNorm). Prefill runs the exact recurrence in one launch per layer, one warp per state column.
  `CARBON_FUSED_GDN=0` turns both off.
- **MoE**: a decode step's MoE layer is softmax top-k routing on the device, then grouped gate/up
  and down GEMVs reading each routed expert through a per-layer table of weight pointers (dense,
  FP8 or NVFP4). With offloaded experts the same table points into the slot cache. Steps above 64
  tokens use the grouped per-expert path. `CARBON_FUSED_MOE=0` turns it off.
- **Multi-token decode**: steps of up to 8 new tokens per sequence (MTP verify steps, short
  prefill chunks) stay on the decode kernels: one attention row per token, recurrent layers
  advanced one token at a time.

## Build

```bash
cargo build --release                        # CPU only
cargo build --release --features cuda        # NVIDIA GPUs (needs the CUDA toolkit / nvcc)
cargo build --release --features nccl        # + NCCL all-reduce for tensor parallelism
cargo build --release --features metal       # Apple GPUs
```

Set `CUDA_COMPUTE_CAP` (e.g. `80`, `89`, `90`) if `nvidia-smi` is not available at build time.

## Quick start

```bash
# a real checkpoint: any HF directory with config.json + *.safetensors + tokenizer.json
carbon serve --model ~/models/Qwen3-30B-A3B --devices cuda:0 --port 1919

# or a hub repo id: downloaded once into the HF cache (HF_TOKEN for gated repos,
# HF_HUB_OFFLINE=1 to use the cache only, --model-source modelscope for ModelScope)
carbon serve --model Qwen/Qwen3-30B-A3B-FP8 --devices cuda:0

# the model's layout with random weights (benchmarks without the checkpoint)
carbon serve --model Qwen/Qwen3-30B-A3B --dummy-weight

# config overrides, e.g. YaRN for 128k context
carbon serve --model ~/models/Qwen3-8B --max-seq-len 131072 \
  --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'

# no checkpoint at hand? generate a tiny random one (same HF layout)
carbon tiny-model --out /tmp/tiny --arch qwen3_moe
carbon serve --model /tmp/tiny --devices cpu

curl localhost:1919/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":64}'
```

## Multi-GPU

```bash
# pipeline parallel: layers split evenly over 4 GPUs
carbon serve --model M --devices cuda:0,cuda:1,cuda:2,cuda:3

# tensor parallel over 2 GPUs (heads + MLP sharded, experts distributed)
carbon serve --model M --devices cuda:0,cuda:1 --tp-size 2

# 2 pipeline stages x 2-way tensor parallel
carbon serve --model M --devices cuda:all --tp-size 2

# uneven GPUs: give the big card more layers (counts or ratios)
carbon serve --model M --devices cuda:0,cuda:1 --layer-split 3:1
```

Devices are grouped into consecutive tensor-parallel groups of `--tp-size`; each group is
one pipeline stage. With the `nccl` feature, all-CUDA groups all-reduce over NCCL;
otherwise (or for mixed devices) through host memory. The embedding and LM head
live on the first local device.

## Multi-machine (RPC)

```bash
# on each worker host (one process can drive several local GPUs)
carbon worker --listen 0.0.0.0:7070 --devices cuda:0,cuda:1 --tp-size 2 --token "$SECRET"
carbon worker --listen 0.0.0.0:7070 --devices cuda:0 --model /models/M --token "$SECRET"

# on the coordinator: local stages first, then each --rpc-worker in order
carbon serve --model /models/M --devices cuda:0 \
  --rpc-worker 10.0.0.2:7070 --rpc-worker 10.0.0.3:7070 \
  --layer-split 12,24,12 --rpc-token "$SECRET"

# head-only coordinator (embedding + LM head), all layers remote
carbon serve --model /models/M --devices cpu --rpc-worker hostA:7070 --rpc-worker hostB:7070 --layer-split 0,24,24
```

* A worker started **without `--model`** receives exactly the weights of its layers from
  the coordinator at startup (`PutWeights`), so only the coordinator needs the checkpoint.
* KV capacity is negotiated: every stage reports how many tokens fit, the minimum wins
  (override with `--kv-tokens`).
* Per decode step each worker hop carries one hidden-state row per sequence plus newly
  allocated slot ids (a few KB), so the link latency, not bandwidth, bounds decode speed.
* **Security:** the protocol is unencrypted. Use `--token` / `--rpc-token` (or
  `CARBON_RPC_TOKEN`) and keep workers on a trusted network or behind a tunnel
  (WireGuard, SSH). Without a token, any host that can reach the port can drive the worker.

## Agents and the terminal

`carbon launch claude` (or `codex`, `opencode`, `openclaw`, `hermes`, `dsh`) reads the served
model, its context window and modalities from a running server, writes the agent's provider
configuration (Codex: a `carbon-launch` profile over `/v1/responses`; Claude Code: environment
over `/v1/messages`; OpenCode / OpenClaw / Hermes / DeepSeek Harness: their config files, old
versions kept as `*.bak`), removes cloud API keys from the agent's environment and starts it,
offering to install a missing CLI with its official installer.

`carbon shell` is a terminal chat against a server (`--server`) or an engine started in the same
process (`carbon shell --model ... [serve flags]`): reasoning streams dimmed, Ctrl-C cancels a
turn, `/think on|off|<effort>`, `/cache [kv N | moe N]`, `/reset`, `/exit`.

## Supervisor and checkpoints

`carbon daemon` owns one `carbon serve` child behind a loopback control plane (`/engine/start`,
`/engine/switch`, `/engine/stop`, `/engine/status|health|stats|metrics|logs`, `/shutdown`,
`/bench/run`, `/checkpoint/start|cancel|status`; `--token` / `X-Carbon-Token` gates all but
`/health`). The serve's output goes to a log ring tailed over SSE (`?since=` resumes), a serve that
outlives a daemon restart is re-adopted, `--auto-restart` restarts crashes, and the serve tree gets
a positive OOM score so the kernel never picks the daemon. Stopping is fail-closed for token
accounting: the daemon asks the serve to drain and seal its totals (`/v1/admin/prepare-stop`) and
writes a durable receipt before signalling it; a crash leaves a degraded receipt from the last
observed stats. Clients read `/accounting/pending` and acknowledge with `/accounting/ack`.
`scripts/carbon-daemon.service` is a systemd user unit.

`carbon checkpoint` rewrites a safetensors checkpoint as one 4096-aligned byte region in ≤ 8 GiB
shards (tensors exactly as stored, quantized ones included), read back with parallel O_DIRECT
requests; `--model` accepts the output directory like any other (`CARBON_O_DIRECT=0` uses
buffered reads).

`--enable-special-token-ckpt` (hybrid Qwen3.5/3.6): when a request samples `<tool_call>`, the
recurrent state right after it is kept as a prefix-cache reuse point, so an agent that re-renders
the tool call only re-prefills the call itself.

## MoE expert placement

`--moe-strategy` (per stage; workers may set their own default):

| strategy | experts live in | on a miss |
|---|---|---|
| `fused` | device memory | – |
| `offload` | host RAM + device LRU cache (`--moe-cache-size N` / `--moe-cache-rate F`, default: the free VRAM left after a `--kv-reserve-tokens` KV floor); pinned host memory + device slot cache on one GPU | copy expert over PCIe (on the GPU, by a kernel, with the slot cache) |
| `cpu` | host RAM | compute on CPU (`--moe-cpu-threads`, default physical cores) |
| `hybrid` | host RAM + device LRU cache | copy the most-loaded misses (`--moe-hybrid-max-fetch N`, or the bench profile's bandwidth-matched share), compute the rest on CPU |

`auto` (the default) keeps experts resident when they fit next to the KV reserve; otherwise it
takes `carbon bench bw --device cuda:N`'s verdict for the expert format (hybrid when the CPU
expert kernel beats the PCIe upload by `--threshold`), else `offload`. Under offload / hybrid,
`--moe-cpu-layers` (`3,7,11`, a count or a fraction) runs chosen MoE layers entirely on the CPU.

Under tensor parallelism experts are distributed across ranks (expert parallelism), and
each rank's cache holds only its experts. `GET /v1/cache/status` shows per-stage hit rates.

## API

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat, streaming, tools (`tool_choice` none / auto / named), `reasoning_content`, `reasoning_effort`, `thinking`, `stream_options.include_usage`, `chat_template_kwargs` |
| `POST /v1/completions` | text or token-id prompts |
| `POST /v1/responses` | OpenAI Responses (what codex speaks): `instructions`, input items incl. reasoning / function_call / function_call_output, function tools, `reasoning.effort`, typed streaming events. Stateless: `GET /v1/responses/{id}` and `/cancel` answer 404, `previous_response_id` is refused |
| `GET /v1/models` | incl. `context_length`, `supported_reasoning_efforts`, `default_reasoning_effort` |
| `POST /v1/messages` | Anthropic Messages: system, tool_use / tool_result, thinking, `output_config.effort`, streaming events with `ping` keep-alives; late system messages are kept in place or folded into the next user turn (`--anthropic-inline-system`, probed from the template) |
| `POST /v1/messages/count_tokens` | |
| `POST /generate` | raw completion (`text` or `input_ids`, `sampling_params`) |
| `GET /health`, `GET /v1/stats` | model card, KV / recurrent-state use, VRAM and GPUs, 5 s decode / prefill throughput, active / completed requests, p95 latency, mean TTFT, lifetime token totals, per-stage KV / expert-cache stats |
| `GET /v1/requests?since=&limit=` | ring of the last 512 requests (status, latency, TTFT, tokens), paged by cursor |
| `GET /v1/cache/status`, `POST /v1/cache/rebuild` | inspect / live-resize KV and expert caches |
| `POST /v1/admin/prepare-stop` | loopback only: closes admission, drains or aborts running requests, returns sealed token totals (for supervisors) |

Reasoning effort is learned from the checkpoint's own chat template at start-up (which levels it
accepts, which one is its default); requested levels map to the nearest supported one
(`minimal` → `low` on a low/medium/high template), `none` / `off` turn thinking off.

`--enable-cache-report false` hides prefix-cache hits from usage; `--cors-origins` lists browser
origins (`*` for any); `CARBON_API_LOG_DIR=DIR` appends every request body to a JSONL file.
`logit_bias`, `n > 1` and `response_format` other than text are refused (400).

### Images

Qwen3.5/3.6/3.8 checkpoints with a vision tower (`model.visual.*`) take images on every chat
API: OpenAI `image_url` parts (`/v1/chat/completions`), Anthropic `image` blocks with a
`base64` or `url` source (`/v1/messages`, also inside `tool_result`), Responses `input_image`
parts. Images are `data:` (base64) or http(s) URLs or bare base64 data (local paths only from
the CLI: `carbon generate --image FILE --prompt ...`, or `--image-base64 DATA`, both
repeatable), at most 64 per request. They are
resized as the checkpoint's `preprocessor_config.json` says (pixel bounds, bicubic like HF), run
through the vision tower once per request, and their tokens take 3-D mRoPE positions. The prefix
cache keys image tokens by the image's content, so a conversation that resends an image reuses
its KV. `/v1/stats` lists `image` in `input_modalities`; a model without a vision tower answers
400. `--text-only` skips the tower (~0.9 GB in bf16 for Qwen3.8).

On Qwen3.8-Flash-Next (RTX 5090, offloaded experts), a 640x480 image is 300 tokens and the first
token comes after 2.1 s.

Sampling: `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `presence_penalty`,
`frequency_penalty`, `repetition_penalty`, `stop` / `stop_sequences`, `stop_token_ids`,
`ignore_eos`. Unspecified values default to the checkpoint's `generation_config.json`
(`--sampling-defaults none` to disable).

## CLI

```
carbon serve      --model DIR|ORG/NAME [--model-source huggingface|modelscope] [--dummy-weight]
                  [--hf-overrides JSON] [--devices ..] [--tp-size N] [--rpc-worker HOST:PORT]... [--layer-split ..]
                  [--dtype auto|bf16|f16|f32] [--kv-tokens N] [--memory-ratio 0.9] [--max-seq-len N]
                  [--moe-strategy ..] [--moe-cache-size N] [--moe-hybrid-max-fetch N]
                  [--max-running-requests 16] [--max-prefill-tokens 4096] [--host] [--port 1919]
                  [--served-model-name] [--reasoning-parser auto|on|off] [--tool-call-parser auto|hermes|off]
                  [--chat-template FILE] [--rpc-token SECRET] [--enable-cache-report BOOL]
                  [--anthropic-inline-system auto|preserve|fold] [--cors-origins LIST]
carbon worker     [--listen 0.0.0.0:7070] [--devices ..] [--tp-size N] [--model DIR] [--token SECRET]
carbon generate   --model DIR --prompt TEXT [--chat] [--max-tokens N] [engine flags]
carbon ctl        [--base-url URL] health | stats | requests [--since N] | cache [--kv N] [--moe N]
                  | generate PROMPT | bench
carbon shell      [--server URL] | [--model DIR|ORG/NAME serve-flags...]   # terminal chat
carbon launch     claude|codex|opencode|openclaw|hermes|dsh [--server URL] [--dry-run] [-y]
                  [--config] [--install-only] [--force-reinstall] [-- agent args]
carbon bench bw   [--device cuda:0] [--dtype all|bf16,fp8,nvfp4] [--model DIR] [--threshold 2]
carbon checkpoint --model DIR --out DIR [--dtype bf16] [--shard-gib 8]   # aligned fast-load format
carbon daemon     [--port 1900] [--token S] [--auto-restart] [--stop-serve-on-exit]   # supervisor
carbon daemon     self|status|health|metrics|stats|pending|logs|start MODEL [-- serve args]
                  |switch MODEL|stop [--force]|shutdown
carbon devices
carbon tiny-model --out DIR [--arch llama|qwen2|qwen3|qwen3_moe|mixtral|qwen3_5_moe|qwen4_exp]
carbon chat-template --model DIR --request REQ.json [--ids]   # render a conversation (template debugging)
```

## Correctness

```bash
cargo test                      # 51 unit + end-to-end tests, CPU only

# the same equivalence tests on GPUs (logical devices are mapped round-robin)
CARBON_TEST_DEVICE=cuda CARBON_TEST_NUM_DEVICES=2 cargo test --release --features nccl --test parallel
```

* Every architecture is checked token-for-token against Hugging Face `transformers`
  (`scripts/hf_reference.py`) on the same weights. For Qwen3.5/3.6,
  `scripts/qwen35_reference.py` builds tiny random checkpoints in both the HF layout and
  the FP8 release layout; carbon matches HF exactly on both, including under TP x2 with
  small prefill chunks.
* `scripts/check_chat_template.py` checks that chat-template rendering and tokenization
  match HF byte-for-byte (verified on the real Qwen3.6-35B-A3B and Qwen3.8 tokenizers:
  plain, multi-turn, thinking off, tools + tool results, Unicode).
* `tests/parallel.rs` asserts that every topology produces exactly the single-device
  tokens: 3-stage pipeline, TP x2, TP x2 + PP x2, uneven splits, RPC workers with local
  and with shipped weights, a head-only coordinator with two workers (one running TP
  internally), MoE offload / cpu / hybrid, expert parallel + offload, live cache rebuilds
  and token authentication. CPU "devices" stand in for GPUs, so the same code paths run.
* `docs/GPU_TESTING.md` is the step-by-step plan for validating everything on a real GPU
  (selftest, the suite on CUDA, real-model checks with each fused path switched off, CUDA graphs,
  `carbon ctl bench` throughput comparisons, multi-GPU / multi-machine).
* `scripts/cuda_kernels_host_test.cpp`, `scripts/cuda_attention_host_test.cpp`, `scripts/cuda_moe_host_test.cpp` and `scripts/cuda_gdn_host_test.cpp` run the CUDA kernels on the CPU (one thread per lane,
  emulated warp shuffles and block barriers) against an independent reference; on a GPU,
  `carbon devices --selftest` runs every op the engine uses plus carbon's kernels against the
  CPU reference and names anything that fails.
* `tests/vision.rs` checks image input against HF (`scripts/qwen_vl_reference.py`, tiny
  Qwen3.5-MoE and Qwen3.8 checkpoints with a vision tower): preprocessing, the tower's output,
  mRoPE positions and greedy generation through the scheduler (including QSA block selection
  over image tokens), and the three chat APIs over HTTP.
* `tests/server.rs` drives the HTTP APIs (streaming, stop strings, prefix-cache
  accounting, client-disconnect aborts).

The `cuda` and `nccl` builds pass `cargo clippy --features nccl` against CUDA 12.6, but
the test-suite runs on CPU only: GPU execution (and NCCL) has not been exercised on real
hardware yet, so treat it as unvalidated until you run the suite on your GPUs.

## Roadmap

- Image input for Qwen3-VL; video input
- MXFP4 and GGUF Q4 weights
- Prefill: fuse Qwen3.8's hyper-connection elementwise ops (~0.6 s of a 4096-token chunk)
- MTP speculative decoding across several local GPUs
- Micro-batched pipeline execution (overlap stages for different sequences)
- More architectures: DeepSeek-V3/V4 (MLA), GLM, gpt-oss, Gemma, sliding-window attention
- TLS for the RPC transport

## License

Copyright 2026 Carbonis Fundatus

Licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
carbon-engine by you, as defined in the Apache-2.0 license, shall be licensed as above, without any
additional terms or conditions.
