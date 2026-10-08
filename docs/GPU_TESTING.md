# GPU test plan

Everything carbon does on CUDA that could not be run in the development sandbox (no GPU there),
in the order to run it. Each step says what it proves, the exact commands, and what a pass looks
like. Steps 0–3 need no model download; 4–7 use real checkpoints.

Throughout, `MODEL_NVFP4` is a Qwen3-30B-A3B NVFP4 checkpoint (ModelOpt, ~17 GB, fits a 32 GB GPU
entirely, so it exercises every fused kernel and CUDA graphs) and `MODEL_FP8` is
Qwen3.6-35B-A3B-FP8 (does not fit a single 32 GB GPU with its experts: on one RTX 5090 it runs
with `--moe-strategy offload`, which keeps experts in host RAM, so the fused MoE kernels and CUDA
graphs do **not** engage for it; that is expected).

```bash
export MODEL_NVFP4=~/models/Qwen3-30B-A3B-NVFP4
export MODEL_FP8=~/models/Qwen3.6-35B-A3B-FP8
```

## What is new and only verified by emulation so far

| component | file | runtime switch |
|---|---|---|
| quantized GEMV / dequant (FP8, NVFP4) | `src/kernels/quant.cu` | – (falls back per call) |
| paged decode attention | `src/kernels/attention.cu` | `CARBON_FUSED_ATTN=0` |
| tiled prefill attention (FP32 FMA; f32, or GPUs before sm_80) | `src/kernels/attention.cu` | `CARBON_FUSED_ATTN=0` |
| tensor-core prefill attention (bf16/f16, head dim 64/128/256, sm_80+) | `src/kernels/attention_tc.cu` | `CARBON_ATTN_TC=0` (uses the FP32 kernel instead) |
| MoE routing + grouped expert GEMVs | `src/kernels/moe.cu` | `CARBON_FUSED_MOE=0` |
| Gated DeltaNet decode (conv + delta rule + gated norm) | `src/kernels/gdn.cu` | `CARBON_FUSED_GDN=0` |
| Gated DeltaNet prefill recurrence | `src/kernels/gdn.cu` | `CARBON_FUSED_GDN=0` |
| CUDA-graph decode | `src/model/graph.rs` | opt-in `--cuda-graphs` |
| token sampling (temperature / top-k / top-p / min-p) | `src/kernels/sampling.cu` | `CARBON_GPU_SAMPLING=0` |
| QSA decode selection (Qwen3.8) | `src/kernels/qsa.cu` | `CARBON_FUSED_ATTN=0` |

Every fused path turns itself off after its first failure (one `WARN ... disabled, using the
generic path` line in the log) and the engine continues on the generic candle path. **Any such
warning is a bug to report**, even if the output looks fine.

---

## 0. Build

```bash
cd carbon-engine
# single GPU
cargo build --release --features cuda
# several GPUs in one machine (tensor parallel over NCCL)
cargo build --release --features nccl
```

The RTX 5090 (compute capability 12.0) needs CUDA ≥ 12.8. candle's build script picks the
compute capability from `nvidia-smi`; set `CUDA_COMPUTE_CAP=120` if it cannot. carbon's own
kernels are compiled at start-up with NVRTC to PTX for the oldest architecture the installed NVRTC
accepts (compute_70 on CUDA 12, compute_75 on CUDA 13; the tensor-core kernels need compute_80)
and JIT-compiled by the driver, so they do not depend on that setting. `CARBON_NVRTC_ARCH=compute_120`
forces a target if the probe picks a wrong one.

**Pass:** the build finishes.

## 1. (Optional, no GPU) CPU emulation of carbon's kernels

Re-runs the emulation used during development: every kernel executes on the CPU with one thread
per CUDA lane, checked against independent references.

```bash
for t in kernels attention moe gdn sampling qsa; do
  g++ -std=c++20 -O2 -pthread scripts/cuda_${t}_host_test.cpp -o /tmp/ct_$t && /tmp/ct_$t | tail -1
done
```

**Pass:** each prints `all ok`.

The kernels are compiled at runtime with NVRTC, which (unlike nvcc) includes no CUDA headers.
This checks every kernel source the way the engine compiles it, without a GPU:

```bash
python3 scripts/nvrtc_check.py /usr/local/cuda/lib64/libnvrtc.so src/kernels
cargo test --release --features cuda --lib kernels_compile_with_nvrtc   # same, from Rust
```

**Pass:** seven `ok` lines (quant, attention, attention_tc, moe, gdn, sampling, qsa).

## 2. Device selftest

Runs every op the engine needs on the GPU and compares carbon's kernels with CPU references.

```bash
./target/release/carbon devices
./target/release/carbon devices --selftest --dtype bf16
./target/release/carbon devices --selftest --dtype f16
./target/release/carbon devices --selftest --dtype f32
```

The carbon-specific lines (all must be `ok`):

- `carbon nvfp4 kernels (gemv + dequant vs cpu)`
- `carbon fp8 kernels (gemv + dequant vs cpu)`
- `carbon moe kernels, dense experts (vs cpu)` / `fp8 experts` / `nvfp4 experts`
- `carbon gated deltanet decode (vs generic path)`
- `carbon prefill attention (vs cpu)` (tensor-core kernel for bf16/f16 on sm_80+, FP32 kernel
  for f32; run with `CARBON_ATTN_TC=0` too to cover the FP32 kernel in bf16)
- `carbon gated deltanet prefill recurrence (vs cpu)`
- `carbon paged decode attention (vs cpu)`

**Pass:** no `FAIL` for bf16 (the dtype the engine uses on the 5090). If a line fails, keep the
full message: it names the error size or the CUDA error.

## 3. Test suite on the GPU (tiny random models)

The end-to-end tests compare every topology and option token for token. With
`CARBON_TEST_DEVICE=cuda` their "cpu" devices become CUDA devices.

```bash
# whole suite on one GPU
CARBON_TEST_DEVICE=cuda cargo test --release --features cuda -- --test-threads=1

# CUDA graphs vs eager decoding (dense, MoE, hybrid, FP8 and NVFP4 experts).
# Also asserts that graphs were really captured on CUDA.
CARBON_TEST_DEVICE=cuda cargo test --release --features cuda --test parallel cuda_graphs -- --nocapture

# two GPUs (tensor / expert parallel, pipeline) over NCCL
CARBON_TEST_DEVICE=cuda CARBON_TEST_NUM_DEVICES=2 cargo test --release --features nccl -- --test-threads=1
```

`--test-threads=1` keeps tests from competing for GPU memory.

**Pass:** all tests pass. The tests run in f32, so fused and generic paths must agree exactly on
greedy tokens; a mismatch names the case (`{arch} {quant}: {topology}`).

If something fails, rerun the failing test with one switch at a time to find the culprit:

```bash
for sw in CARBON_FUSED_ATTN CARBON_FUSED_MOE CARBON_FUSED_GDN; do
  echo "== $sw=0"; env $sw=0 CARBON_TEST_DEVICE=cuda cargo test --release --features cuda --test parallel <name> 2>&1 | tail -3
done
```

## 4. Real model: correctness

### 4a. Greedy generations with each fused path on and off

Same prompt, greedy, token ids printed. bf16 means generic and fused paths may legitimately
diverge after many tokens (different summation order); the first ~20–30 tokens should match and
the text must stay coherent in every configuration.

```bash
P="Explain in three sentences why the sky is blue."
run() { ./target/release/carbon generate --model "$1" --devices cuda:0 --chat --max-tokens 64 \
          --prompt "$P" --print-ids "${@:2}" 2>&1 | tail -4; }

run $MODEL_NVFP4                                              # everything fused
CARBON_FUSED_ATTN=0 run $MODEL_NVFP4
CARBON_ATTN_TC=0    run $MODEL_NVFP4                          # FP32 prefill kernel
CARBON_FUSED_MOE=0  run $MODEL_NVFP4
CARBON_FUSED_ATTN=0 CARBON_FUSED_MOE=0 CARBON_FUSED_GDN=0 run $MODEL_NVFP4   # all generic
run $MODEL_NVFP4 --cuda-graphs                                # must equal the first run exactly

run $MODEL_FP8 --moe-strategy offload                         # hybrid model: attention + GDN fused
CARBON_FUSED_GDN=0 run $MODEL_FP8 --moe-strategy offload
CARBON_FUSED_ATTN=0 CARBON_FUSED_GDN=0 run $MODEL_FP8 --moe-strategy offload
```

**Pass:** coherent answers everywhere; `--cuda-graphs` output identical to the plain fused run
(graphs replay the same kernels); no `disabled` warnings in the log.

### 4a'. Sampling on the GPU

```bash
./target/release/carbon serve --model $MODEL_NVFP4 --devices cuda:0 --port 1918 &
for i in 1 2; do  # the same seed twice: identical text
  curl -s localhost:1918/v1/chat/completions -H 'content-type: application/json' -d '{"messages":
    [{"role":"user","content":"Write a two-line poem about rain."}],"max_tokens":60,
    "temperature":0.8,"top_p":0.9,"top_k":40,"seed":7}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
done
```

**Pass:** coherent text, identical both times, and no `GPU sampling disabled` warning. Without
`seed` the two poems should differ. (A seed gives different text on the GPU than with
`CARBON_GPU_SAMPLING=0`: the two samplers draw their random numbers differently.)

### 4b. Long prompt (prefill kernels, chunked prefill)

```bash
python3 - <<'EOF' > /tmp/long.txt
import random; random.seed(0)
facts = [f"Item {i}: the code word for room {i} is {random.choice(['apple','river','stone','cloud','ember'])}-{i*7}." for i in range(1500)]
print(" ".join(facts) + "\nQuestion: what is the code word for room 1234? Answer with the code word only.")
EOF
./target/release/carbon generate --model $MODEL_NVFP4 --devices cuda:0 --chat --max-tokens 32 \
  --prompt "$(cat /tmp/long.txt)"
./target/release/carbon generate --model $MODEL_FP8 --devices cuda:0 --moe-strategy offload \
  --chat --max-tokens 32 --prompt "$(cat /tmp/long.txt)"
```

(~30k tokens; add `--max-seq-len 65536` if the checkpoint's default is lower.)

**Pass:** answers `<word>-8638` (the word is fixed by the seed; check `grep "room 1234:" /tmp/long.txt`).
Repeat with `CARBON_FUSED_ATTN=0 CARBON_FUSED_GDN=0` and compare.

### 4c. Server: chat, streaming, tools, reasoning, prefix cache

```bash
./target/release/carbon serve --model $MODEL_FP8 --devices cuda:0 --moe-strategy offload --port 1918 &
curl -s localhost:1918/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"x","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":512}' | jq .
curl -sN localhost:1918/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"x","stream":true,"messages":[{"role":"user","content":"Count to five."}],
  "chat_template_kwargs":{"enable_thinking":false}}'
curl -s localhost:1918/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"x","messages":[{"role":"user","content":"Weather in Paris?"}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Current weather",
  "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}' | jq .choices[0].message
# same long system prompt twice: the second response reports cached_tokens > 0
for i in 1 2; do curl -s localhost:1918/v1/chat/completions -H 'content-type: application/json' -d "{
  \"model\":\"x\",\"max_tokens\":8,\"messages\":[{\"role\":\"system\",\"content\":\"$(head -c 6000 /tmp/long.txt)\"},
  {\"role\":\"user\",\"content\":\"Say hi.\"}]}" | jq .usage; done
./target/release/carbon ctl --base-url http://127.0.0.1:1918 stats
```

**Pass:** `391` with reasoning in `reasoning_content`; streamed deltas; a `get_weather` tool call
with `{"city":"Paris"}`; second usage shows cached tokens; stats list the stages.

## 5. CUDA graphs on a real model

```bash
./target/release/carbon serve --model $MODEL_NVFP4 --devices cuda:0 --port 1918 --cuda-graphs &
./target/release/carbon ctl --base-url http://127.0.0.1:1918 bench --concurrency 4 --requests 16 \
  --prompt-tokens 256 --max-tokens 128
./target/release/carbon ctl --base-url http://127.0.0.1:1918 stats | jq '.scheduler.stages[].cuda_graphs'
```

**Pass:** the log shows `captured a cuda graph for decode batch N` lines and no
`cuda graphs disabled`; `cuda_graphs` in stats is > 0; then repeat step 4a's
`--cuda-graphs` comparison. With `--moe-strategy offload` (MODEL_FP8 on one GPU) graphs must not
engage (`a layer has no fused decode path` in the log, `cuda_graphs: 0`).

## 6. Performance

Compare each fused path against the generic one, and graphs against no graphs. Start a server
per configuration, then run the same bench. Decode-heavy and prefill-heavy runs:

```bash
B="./target/release/carbon ctl --base-url http://127.0.0.1:1918 bench"
$B --concurrency 1  --requests 4  --prompt-tokens 128  --max-tokens 256   # single-stream decode
$B --concurrency 8  --requests 32 --prompt-tokens 512  --max-tokens 256   # batched decode
$B --concurrency 4  --requests 8  --prompt-tokens 8192 --max-tokens 1     # prefill throughput
```

After the batched decode run, `curl -s localhost:1918/v1/stats | jq .throughput.host_gap_us` is
the host time between steps, which is GPU idle time unless the next step's work overlaps it. On
a CPU it is ~30 µs at batch 32. If it is more than ~2% of a decode step on the GPU
(1e6 / (decode_tps / concurrency) µs), overlap scheduling would pay off: report both numbers.

Server configurations (restart between them):

```bash
S="./target/release/carbon serve --model $MODEL_NVFP4 --devices cuda:0 --port 1918"
$S                                              # A: fused
$S --cuda-graphs                                # B: fused + graphs
CARBON_ATTN_TC=0 $S                             # A2: fused, FP32 prefill kernel instead of tensor cores
CARBON_FUSED_ATTN=0 $S                          # C: generic attention
CARBON_FUSED_MOE=0 $S                           # D: generic MoE
CARBON_FUSED_ATTN=0 CARBON_FUSED_MOE=0 CARBON_FUSED_GDN=0 $S   # E: all generic
# hybrid model, offloaded experts
S2="./target/release/carbon serve --model $MODEL_FP8 --devices cuda:0 --moe-strategy offload --port 1918"
$S2 ; CARBON_FUSED_GDN=0 $S2 ; CARBON_FUSED_ATTN=0 $S2
```

`carbon generate ... --max-tokens 256` also prints `ttft` and `decode tok/s` for a single
request. **Expected:** A ≥ C, D, E for decode; B ≥ A for decode, most at concurrency 1–4. The
prefill run (8192 tokens) should rank A (tensor cores) > A2 (FP32 kernel), and A ≥ C; report
the three numbers either way.

## 6b. Qwen3.8-Flash-Next (if you have it)

```bash
M=~/models/Qwen3.8-Flash-Next-NVFP4   # or RadixArk/Qwen3.8-Flash-Next-NVFP4
./target/release/carbon generate --model $M --devices cuda:0 --chat --max-tokens 64 \
  --prompt "Name three prime numbers and explain why they are prime."
./target/release/carbon generate --model $M --devices cuda:0 --moe-strategy offload --chat \
  --max-tokens 32 --prompt "$(cat /tmp/long.txt)"   # step 4b's ~30k-token prompt: QSA selects
```

Repeat both with `--cuda-graphs` (the log must show `captured a cuda graph` and no `cuda graphs
disabled`) and with `CARBON_FUSED_ATTN=0`; the outputs should agree.

**Pass:** coherent answers; the long prompt answers `<word>-8638`. Report the load time, resident
memory (`ps -o rss`), decode tok/s and whether `GPU sampling disabled` / `disabled` appears. The
PLE table is memory-mapped: the first tokens are slower while its pages are read in.

## 7. Multi-GPU and multi-machine (if available)

```bash
# tensor parallel over two GPUs (nccl build)
./target/release/carbon serve --model $MODEL_FP8 --devices cuda:0,cuda:1 --tp-size 2 --port 1918
# pipeline over two GPUs
./target/release/carbon serve --model $MODEL_FP8 --devices cuda:0,cuda:1 --port 1918
# two machines: worker on host B, coordinator on host A
CARBON_RPC_TOKEN=secret ./target/release/carbon worker --devices cuda:0 --listen 0.0.0.0:7070 --model $MODEL_FP8
CARBON_RPC_TOKEN=secret ./target/release/carbon serve --model $MODEL_FP8 --devices cuda:0 \
  --rpc-worker hostB:7070 --port 1918
```

Run step 4a's prompt and step 6's batched bench on each. **Pass:** coherent output; stats list
every stage (`ctl stats`).

## What to send back

1. `nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv` and `nvcc --version`.
2. Output of step 2 (bf16) and the failing test names from step 3, if any.
3. Any log line containing `disabled` or `WARN`.
4. Step 6 bench JSON for configurations A–E (and the FP8 ones), and `host_gap_us`.
5. For any wrong output: the exact command, its env switches, and the printed ids.
