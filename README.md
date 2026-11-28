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
| Quantized weights | FP8, NVFP4, MXFP4, GGUF Q4_0, BF16 | FP8 (block and per-tensor), NVFP4 (ModelOpt and compressed-tensors), GGUF Q4_K / Q6_K (dense Qwen3.5-family), GGUF Q8_0 (LFM2), GGUF NVFP4 + Q8_0 (Inkling 2), BF16/F16/F32 |
| Hybrid attention models | Gated DeltaNet (Qwen3.5/3.6/3.8) | Gated DeltaNet (Qwen3.5/3.6/3.8), fused decode and prefill kernels; LFM2 / LFM2.5 gated short convolutions |
| CUDA graphs | decode steps | decode steps, including offloaded experts and multi-token (MTP verify) steps |
| CLI | `ft serve / ctl / bench` | `carbon serve / generate / ctl / bench bw / devices / worker / launch / shell` |

**Different**

| | FreeToken | carbon-engine |
|---|---|---|
| GPUs per server | one (`--gpu`) | any number, in pipeline and/or tensor-parallel groups |
| Machines | one | several, over RPC (`carbon worker`) |
| Speculative decoding | none | MTP drafts for Qwen3.8-Flash-Next (`--mtp`), verified in one multi-token step |
| Models | Llama, Qwen2/3, Qwen3-MoE, Qwen3.5/3.6/3.8, DeepSeek-V4, GLM, gpt-oss, Gemma-4, MiniMax, ... | Llama/Mistral, Qwen2/2.5, Qwen3, Qwen3-MoE, Mixtral, Qwen3.5/3.6 (dense + MoE), Qwen3.8-Flash-Next, LFM2 / LFM2.5 (dense), Inkling 2 Mini (looped MoE) |
| Image input | Qwen3.6/3.8, Qwen3-VL, Gemma-4, GLM, ... | Qwen3.5/3.6/3.8 (their vision tower); no video |
| Fast-load weight format | FTW | the checkpoint format of `carbon checkpoint` |
| Prefill with offloaded experts | whole layers streamed, double-buffered behind compute | through the slot cache: a layer's missing experts are copied in batches on a side stream while its resident ones compute, then grouped tensor-core NVFP4 expert GEMMs |
| Hardware | NVIDIA RTX 30/40/50; AMD ROCm (experimental) | NVIDIA (its CUDA kernels compile at start-up for the installed driver) and CPU; a Metal build through candle |
| Distribution | Python package, desktop app | one Rust binary |
| Implementation | Python, Triton and CUDA on PyTorch | Rust on [candle](https://github.com/huggingface/candle), with its own CUDA kernels |

Not supported yet (see [Roadmap](#roadmap)): MXFP4 weights, GGUF files of other architectures
or quantizations than the ones below, sliding-window and MLA attention (DeepSeek, GLM, gpt-oss,
Gemma), video input.

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

## Running Qwen3.8-27B from a GGUF file (Q4_K_M)

```bash
# speculative decoding with the file's MTP head
carbon serve --model ~/models/Qwen3.8-27B-Q4_K_M.gguf --devices cuda:0 --cuda-graphs \
  --mtp --mtp-drafts 5 --mtp-min-prob 0.75

# the full 262,144-token window on a 32 GB card: int8 KV, recurrent-state pools for one user
carbon serve --model ~/models/Qwen3.8-27B-Q4_K_M.gguf --devices cuda:0 --cuda-graphs \
  --mtp --mtp-drafts 5 --mtp-min-prob 0.75 --kv-dtype q8 --max-seq-len 262144 \
  --max-running-requests 2 --state-snapshots 4
```

`--model` takes a `.gguf` file (or a directory holding one): the config, the tokenizer and the
chat template come from its metadata (`src/gguf.rs`). Supported: architecture `qwen35` (dense
Qwen3.5 / 3.6 / 3.8: 48 Gated DeltaNet + 16 full-attention layers at 27B), tensors stored F32 /
F16 / BF16 / Q4_K / Q6_K, on one local GPU (no tensor parallelism: the file stores the linear
layers' value heads in a tiled order). Image input: a projector file
(`mmproj-*.gguf`, type `qwen3vl_merger`) next to the model is loaded as the vision tower
(`--mmproj FILE` names another one, `--mmproj none` or `--text-only` loads none); see
[Images](#images). The tower takes about 0.6 GiB, which leaves the int8 KV cache at 246k tokens
on a 32 GiB card: for the full 262,144 window use `--text-only` or `--kv-dtype turbo4`.

- **Weights stay packed** (15.6 GiB). A decode step quantizes each projection's input to 8 bits
  per 32 values and multiplies it with the 4- / 6-bit codes in integer arithmetic: a dp4a kernel
  for single tokens, an integer tensor-core GEMM for the multi-token steps of speculative
  decoding, where 8 tokens cost about 1.2x one. Projections that share an input run in one
  launch. Prefill dequantizes a layer's weights per chunk and uses cuBLAS; the Gated DeltaNet
  layers and attention run in carbon's own prefill kernels.
- **MTP** (`--mtp`): the head (`blk.64` + `nextn.*`) drafts up to `--mtp-drafts` tokens, going
  on while its last draft has probability `--mtp-min-prob`; the next step verifies them all.
  With `--mtp` every K-quant weight uses the tensor-core kernel for every step, so the output is
  the same as without drafts on those kernels (checked token for token on 400-token greedy
  runs). Without `--mtp` the dp4a kernel is used (about 10% faster for single tokens), and its
  rounding differs in the last bits.
- **Decode attention** is chunked (256 positions per block) and reads K and V once per KV head
  for its six query heads; on pools of 131,072 slots or more a verify step's rows also share one
  pass (`CARBON_ATTN_ROWS_MIN`), about 0.38 ms per layer over 129k positions.
- **`--kv-dtype q8`** keeps each key / value vector as 256 int8 values and an f32 scale: half
  the memory and attention traffic of bf16. It is lossy: the text of a greedy run drifts from
  the bf16 run after a few sentences, and the greedy next token agrees with the bf16 cache on
  244 of 250 prefixes of a 44k-token text (`scripts/next_token_agreement.py`); all six changed
  tokens were near-ties, and the KL divergence from the bf16 distribution averages 0.002 nats
  (`turbo4`: 0.017). No benchmark-level
  quality evaluation has been done.
- **`--kv-dtype turbo4`** is TurboQuant (Zandieh et al., arXiv 2504.19874) at 4 bits: keys and
  values are rotated by a fixed random orthogonal matrix and each coordinate is stored as an
  index into the 16-level Lloyd-Max codebook of a Gaussian, with an f32 scale per vector (132
  bytes per 256-dim vector: half of `q8`, a quarter of bf16). Queries are rotated the same way
  and the attention output is rotated back, so the kernels never apply the rotation per key.
  The full window then takes 23.5 GiB instead of 28. It is lossier than `q8`: the greedy next
  token agrees with the bf16 cache on 229 of 250 prefixes (91.6%; `q8` 97.2%), and decode is
  about 4% slower at short context (two small rotations per attention layer and step).
- **`--kv-dtype rotor4`** is RotorQuant (scrya-com/rotorquant): the storage of `turbo4`, the
  rotation made of independent random 3D rotations on consecutive coordinate triples
  (`planar4` and `iso4` use 2D and 4D blocks). Same memory as `turbo4` and no faster here,
  since this engine never applies the rotation per key. It is the lossiest option: next-token
  agreement with the bf16 cache is 80.0% (`planar4` 82.8%, `iso4` 82.0%) against 91.6% for
  `turbo4`. Prefer `turbo4` unless you are comparing the methods.
- **Prefix reuse**: recurrent state is snapshotted at every prefill chunk boundary, so a prompt
  that shares a long beginning with an earlier one resumes from the last boundary inside it.

Measured on an RTX 5090 against LM Studio (llama.cpp runtime 2.71) on the same file, 400-token
greedy answers with thinking off (`scripts/bench_openai.py`; `PLAN.md` has the full tables):

| decode tok/s | story | code | math | json |
|---|---|---|---|---|
| LM Studio, MTP on, short context | 87.5 | 113.6 | 117.8 | 180.6 |
| carbon `--mtp --mtp-drafts 5 --mtp-min-prob 0.75`, short context | 91.4 | 145.5 | 149.4 | 174.2 |
| LM Studio without MTP / carbon without `--mtp` | 73 / 66 | | | |

| decode tok/s at depth (story / json) | 32k | 128k | 250k |
|---|---|---|---|
| LM Studio (MTP on to about 100k; it cannot load more with MTP) | 78 / 154 | 37 / 37 (no MTP) | 25 / 25 (no MTP) |
| carbon, full-window configuration above | 78 / 152 | 66 / 127 | 50 / 95 |

| time to first token, cold prompt (s) | 32k | 128k | 250k |
|---|---|---|---|
| LM Studio | 11.3 | 75 | 217 |
| carbon, full-window configuration above | 12.2 | 75 | 206 |

The 128k and 250k columns are the same prompt on both engines, each on a freshly loaded
server (an earlier 148 s for LM Studio at 250k had reused a cached 128k prefix).

A later prompt that shares the beginning starts in 1.7 s at 32k, 2.2 s at 128k and 4.3 s at
250k. Without `--max-seq-len` the context is capped at 32,768 tokens, or set to the scaled
length when YaRN is on (`--yarn`, or a checkpoint or GGUF that ships YaRN rope scaling).

## Running LFM2.5-350M (Liquid AI)

```bash
# the GGUF file (Q8_0) ...
carbon serve --model ~/models/LFM2.5-350M-Q8_0.gguf --devices cuda:0 --cuda-graphs --max-seq-len 32768
# ... or the HF checkpoint (bf16 safetensors), downloaded into the hub cache
carbon serve --model LiquidAI/LFM2.5-350M --devices cuda:0 --cuda-graphs
# CPU only
carbon generate --model ~/models/LFM2.5-350M-Q8_0.gguf --devices cpu --chat --prompt "Hello!"
```

LFM2 (`model_type: lfm2`, GGUF architecture `lfm2`) is a hybrid of gated short convolutions
and grouped-query attention: at 350M, 16 layers (10 conv + 6 attention with 16 query / 8 KV
heads and per-head q/k RMSNorm), hidden size 1024, SwiGLU width 4608, tied embeddings. A conv
layer splits `in_proj(x)` into gates `b`, `c` and values `x`, runs a causal depthwise conv
(kernel 3) over `b * x` and gates the result with `c` (`src/model/shortconv.rs`). Each request
keeps the last two conv inputs per conv layer (20 KiB in bf16) as its recurrent state, in the
same state slots, prefix snapshots and CUDA graphs as the Gated DeltaNet models; KV costs
12 KiB per token (6 attention layers).

- GGUF: tensors F32 / F16 / BF16 / Q8_0 (Q8_0 is decoded to the compute dtype at load: the
  model is 0.7 GiB in bf16). K-quant LFM2 files have not been tried. The tokenizer is rebuilt
  from the file's metadata and matches Liquid's `tokenizer.json` and llama.cpp token for token.
- HF: configs with `block_auto_adjust_ff_dim` get the SwiGLU width HF derives (6656 -> 4608);
  the `{% generation %}` tags of the LFM2.5 chat templates are accepted.
- Not supported: LFM2-MoE (`lfm2_moe`), sliding-window variants, the vision / audio models.
- Checked against an independent f32 NumPy forward pass of the GGUF
  (`scripts/lfm2_reference.py`): with `--dtype f32` prefill and 32 greedy decode steps agree to
  1e-5 in log-probability on CPU and CUDA (the HF checkpoint against the same pass on its own
  bf16 weights: 6e-5); in bf16 to about 0.1. Against llama.cpp on the same
  file (whose CPU path also quantizes activations to 8 bits) the teacher-forced greedy token
  agrees on 78 of 80 positions, the misses being near-ties.

RTX 5090, one stream, 400-token greedy answer: 731 tok/s, 874 tok/s with `--cuda-graphs`; eight
concurrent streams about 2,800 tok/s (3,400 with graphs) in aggregate.

## Running Inkling 2 Mini (Thinking Machines, GGUF)

```bash
# the Unsloth UD-Q4_K_XL file: 562k-token context (int8 KV), CUDA graphs on by default
carbon serve --model ~/models/Inkling-2-Mini-UD-Q4_K_XL.gguf --devices cuda:0
# bf16 KV instead: about 290k tokens, a little closer to the reference at depth
carbon serve --model ~/models/Inkling-2-Mini-UD-Q4_K_XL.gguf --devices cuda:0 --kv-dtype model
```

Inkling 2 Mini (GGUF architecture `inkling2_loop`, 26.4B total / 3.4B active) is a looped MoE
transformer with its own layer stack (`src/model/inkling.rs`, kernels in
`src/kernels/inkling.cu` and `inkling_tc.cu`), written from llama.cpp's `inkling2` graphs:

- 24 layers; layers 6..18 (the core) run three times with shared weights, so a token makes 48
  layer visits, each with its own caches. From the second pass on, each core layer mixes a
  message from the previous pass into its inputs through tanh gain / forget controllers.
- Block attention residuals: a layer reads the RMS-normalized outputs of earlier 6-layer blocks
  and its own stream through an attention over the depth instead of one residual stream.
- Attention without rope: L2-clipped queries and keys, RMS-normalized values, a learned
  relative-position bias within 1024 tokens, a per-query sink and a sigmoid output gate; keys,
  values and both layer outputs pass a causal conv (input-dependent taps on the full-attention
  layers). 40 of the 48 visits use a 1024-token sliding window (per-request rings), 8 use full
  attention over the paged KV pool (16.5 KiB per token in int8, 32 KiB in bf16).
- MoE: 128 experts, top 8, sigmoid routing with a selection bias, SiTU-gated experts in GGUF
  NVFP4 (Q8_0 in four layers), read in place. Dense projections stay Q8_0.

Decode runs on fused kernels (int8 activations and dp4a GEMVs, one CUDA graph per decode batch
size, `CARBON_INKLING_GRAPHS=0` turns graphs off); prefill uses tensor-core flash attention
(`CARBON_INKLING_FLASH=0` falls back to the CUDA-core kernel). Defaults for this model: int8 KV
(`--kv-dtype auto`), at most 4 running requests (each keeps about 170 MB of sliding-window
and conv state; `--max-running-requests` raises it) and, without `--max-seq-len`, the longest
context the KV cache holds. Chat: the template's `<|message_model|>` / `<|content_*|>` turn
markers become reasoning content and tool calls on the OpenAI, Anthropic and Responses APIs.

Image input: the `mmproj` file next to the model (`inkling2v`, `src/model/inkling_vision.rs`)
is loaded automatically (`--text-only` skips it and gives its 0.6 GiB to the KV cache: 562k
instead of 528k tokens). Images keep their size, padded to a multiple of 40 pixels (scaled down
only past 4,096 image tokens), one token per 40 x 40 pixels: a 640 x 480 image is 192 tokens,
encoded in 44 ms; a 4,096-token image in 3.3 s. The encoder is a 23-layer ViT (CLS token, learned
positions resized to the image, 2D rope, layer scale), a 2x2 merge and a GELU projector; its
rows enter the language model without the token-embedding norm, as in llama.cpp. Against an
independent NumPy pass of the projector (`vision()` in `scripts/inkling_reference.py`) the
image tokens agree to a cosine of 0.9998 (bf16 weights).

RTX 5090, one stream, 400-token greedy answers (`scripts/bench_openai.py`):

| context | carbon (int8 KV) | carbon (bf16 KV) | LM Studio (llama.cpp 2.83) |
|---|---|---|---|
| short | 152 tok/s | 152 tok/s | 98 tok/s |
| 32k | 130 tok/s, prefill 5.5 s | 128 tok/s, prefill 5.5 s | |
| 250k | 91 tok/s, prefill 74 s | 63 tok/s, prefill 73 s | |
| 500k | 64 tok/s, prefill 221 s | does not fit | |
| max context | 528,556 tokens (562,515 with `--text-only`) | 290,047 tokens | |

Checked against an independent f32 NumPy forward pass of the GGUF
(`scripts/inkling_reference.py`): every greedy decode step is the reference's argmax, with
log-probabilities within 0.03 (bf16 KV) / 0.09 (int8 KV), the int8 activations of the GEMVs
being most of it. Where carbon's and LM Studio's greedy text part, at near-ties, carbon's token
is the reference's.

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

# YaRN: stretch a 32k-trained model to 128k (the default --max-seq-len becomes 32768 x 4)
carbon serve --model ~/models/Qwen3-8B --yarn 4 --yarn-original-ctx 32768

# the same through config overrides (vLLM semantics); rope_scaling layers over the
# checkpoint's rope_parameters, so theta / partial rope / mRoPE sections are kept
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
  (override with `--kv-tokens`). On a GPU the room left for activations follows
  `--max-prefill-tokens`: chunks beyond about 16k tokens take pool tokens and do not prefill
  faster, so keep the default. Startup warns when the pool is smaller than `--max-seq-len`;
  clients are then told the pool size as the context (`/v1/models` `max_model_len`, which
  `carbon launch` passes to agents), and a request's whole output budget is reserved up front,
  cut to what fits next to its prompt (the reply ends with finish reason `length`).
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
| `POST /v1/completions` | text or token-id prompts; `logprobs: N` (1-20, non-streaming) adds each output token's log-probability and the N most likely tokens, with their ids in `token_ids` / `top_token_ids` |
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

GGUF models take images at 256 x 256 pixels' worth or more and at most 4096 tokens each
(llama.cpp's cap; there is no preprocessor config to read). On Qwen3.8-27B Q4_K_M (RTX 5090) a
640x480 image is 300 tokens and is encoded in 27 ms; the largest, 4032 tokens, takes 3.6 s. A
snapshot is kept at the end of every image, so another question about the same image starts
in 0.3 s. On three test images (shapes with a caption, eight lines of small text, a colour
grid) the answers and prompt token counts equal llama.cpp's.

On Qwen3.8-Flash-Next (RTX 5090, offloaded experts), a 640x480 image is 300 tokens and the first
token comes after 2.1 s.

Sampling: `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `presence_penalty`,
`frequency_penalty`, `repetition_penalty`, `stop` / `stop_sequences`, `stop_token_ids`,
`ignore_eos`. Unspecified values default to the checkpoint's `generation_config.json`
(`--sampling-defaults none` to disable).

## CLI

```
carbon serve      --model DIR|ORG/NAME [--model-source huggingface|modelscope] [--dummy-weight]
                  [--hf-overrides JSON] [--yarn FACTOR [--yarn-original-ctx N]] [--devices ..] [--tp-size N] [--rpc-worker HOST:PORT]... [--layer-split ..]
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
- MXFP4 weights; GGUF for more architectures and K-quants, other `mmproj` projector types
- LFM2: fused short-conv decode kernel, packed Q8_0 projections, LFM2-MoE
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
