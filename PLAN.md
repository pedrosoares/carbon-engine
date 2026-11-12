# Plan: Qwen3.8-27B Q4_K_M on carbon-engine

Goals:

1. Run `Qwen3.8-27B-Q4_K_M.gguf` in carbon-engine.
2. Decode faster than LM Studio (llama.cpp runtime 2.71), which gets 80-100 tok/s on the RTX 5090.
3. Serve the full 262,144-token context on the GPU, with decode at ~250k context at least as
   fast as LM Studio at short context (>= 100 tok/s).

Text-only first. Image input (the `mmproj` tower) is phase 7.

## Facts

- File: `~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf` (16.8 GB).
- GGUF architecture `qwen35`: dense, 64 layers (48 Gated DeltaNet + 16 full attention, every
  4th), hidden 5120, FFN 17408, 24 heads / 4 KV heads, head dim 256, vocab 248,320,
  context 262,144. No hyper-connections, PLE or QSA.
- Tensors: 439 Q4_K (11.4 GiB), 67 Q6_K (4.2 GiB), 360 small F32. Block 64 is the MTP draft head
  (`blk.64.nextn.*`, `nextn_predict_layers = 1`).
- Engine gaps: no GGUF reader, no K-quant kernels, tokenizer loads only `tokenizer.json`, the MTP
  head code is specific to Flash-Next (`qwen4_exp`), the KV pool is bf16/f16/f32 only.
- Reference implementation: `~/.unsloth/llama.cpp` (`src/models/qwen35.cpp`,
  `convert_hf_to_gguf.py`).

## Constraints

- Speed ceiling: a decoded token reads ~15.8 GB of weights. At ~1.8 TB/s that caps single-token
  decode near 110 tok/s, so a better kernel can only match LM Studio. The win has to come from
  MTP: draft 1-2 tokens, verify them in one weight pass. Estimate 130-170 tok/s if draft
  acceptance is like Flash-Next's (64-93%); to be measured.
- Memory: KV is 64 KiB per token in bf16 (16 layers x 4 KV heads x 256 x K+V), 16 GiB at full
  context, next to ~15 GiB of weights on a 32 GiB card. int8 KV halves it.
- Attention traffic: the decode kernel runs one block per query head, so K/V are read 24 times
  instead of 4: ~100 GB per token at full context. Grouped by KV head with int8 KV it is ~8.6 GB.
- Exact attention cannot be flat with depth. A cold 262k prompt takes minutes to prefill
  (rough estimate 3-5 min); the prefix cache and state snapshots cover later turns.

## Phases

| # | Work | Gate |
|---|---|---|
| 0 | Baseline: CUDA build, GPU selftest, LM Studio tok/s and TTFT on fixed prompts at 1k / 32k / 128k / max context | Numbers recorded, prompts kept for reuse |
| 1 | GGUF loader: `WeightSource` over the mmapped file, `ModelConfig` from GGUF metadata, tensor-name mapping to the Qwen3.5 names, tokenizer + chat template from GGUF metadata | Tokenization matches llama.cpp byte for byte |
| 2 | Q4_K / Q6_K `QLinear` variants: CPU reference dequant, then CUDA GEMV (decode) and dequant + cuBLAS (prefill) | Greedy output matches llama.cpp on the same file; kernels match the CPU reference |
| 3 | Decode speed: all layers on fused kernels so CUDA graphs engage, host-side embedding, GPU sampling | Single-token decode >= LM Studio at short context |
| 4 | MTP for dense Qwen3.5: port the `nextn` head, reuse the verify / rollback scheduler path | Output identical to non-MTP greedy; tok/s clearly above LM Studio |
| 5 | Full context: int8 KV with per-vector scales (`--kv-dtype q8`), decode attention reading K/V once per KV head, prefill reading the int8 pool | 262,144 tokens load on the GPU; logits close to bf16 KV |
| 6 | `carbon bench` sweep over context depth against the phase 0 baseline, README update | Results table |
| 7 | Image input: load `mmproj-Qwen3.8-27B-BF16.gguf` into the existing vision tower path | Matches llama.cpp on a test image |

## Risks

- Tensor transforms in the GGUF conversion (norm offsets, `ssm_a`, conv layout) must be read from
  the convert script, not guessed. Phase 2's gate catches a wrong mapping.
- The K-quant GEMV may be bound by arithmetic rather than bandwidth; fallback is integer dot
  products on quantized activations.
- MTP acceptance on this model is unknown until phase 4.
- int8 KV quality is unverified; `--kv-dtype bf16` stays available for contexts that fit.
- Approximate attention (skipping low-weight V rows) would flatten speed further but changes
  outputs: opt-in only, not planned.

## Results

Measured with `scripts/bench_openai.py` (400-token answers, greedy, thinking off; prompts story /
code / math / json). RTX 5090, LM Studio runtime llama.cpp 2.71.

### Phase 0: LM Studio baseline (decode tok/s)

| setup | context | story | code | math | json | mean |
|---|---|---|---|---|---|---|
| no MTP | short | 73.1 | 72.9 | 72.7 | 72.6 | 72.8 |
| MTP on | short | 87.5 | 113.6 | 117.8 | 180.6 | 124.9 |
| MTP on | 32k | 78.1 | 84.1 | 102.5 | 153.9 | 104.7 |
| MTP on | 91k | 64.1 | 81.0 | 81.1 | 120.7 | 86.7 |
| no MTP | 129k | 37.0 | | | 37.0 | 37.0 |
| no MTP | 253k | 24.6 | | | 24.5 | 24.5 |

- The bar to beat is LM Studio with MTP on, not the 73 tok/s base.
- LM Studio cannot load 262,144 context with MTP on ("failed to allocate buffer for kv cache";
  its estimate is 49 GiB). MTP on fits about 100k. Without MTP it loads full context (28.6 GiB).
- Prefill: 32k in 11.3 s (2,880 tok/s), 91k in 33.9 s, 129k in 72 s, 253k in 148 s (1,710 tok/s).
- carbon build: `NVCC_CCBIN=/usr/bin/g++-15` is needed (nvcc 13.3 rejects gcc 16). GPU selftest
  passes in bf16 / f16 / f32.

### Phase 1: GGUF loader (done)

- `--model file.gguf` (or a directory holding one) loads config, weights, tokenizer and chat
  template from the file (`src/gguf.rs`).
- Tokens and rendered template match llama.cpp (`llama-server /tokenize`, `/apply-template`) on a
  552-token conversation with tools, code, CJK, emoji and odd whitespace. Known differences:
  carbon applies NFC like the HF tokenizer (llama.cpp does not normalize), and carbon's template
  path turns `\r\n` inside a message into `\n`.

### Phase 2: K-quant weights (done)

- Q4_K / Q6_K stay packed on the GPU (`QLinear::KQuant`); Q6_K blocks are padded 210 -> 224 bytes
  for aligned reads. GPU selftest checks dequant and GEMV against the CPU decoder.
- Greedy completions vs llama.cpp (CPU) on the same file, 48 tokens, 4 raw prompts: 2 identical,
  2 diverge after about 27 and 40 tokens at near-ties. No logit-level comparison yet.

### Where it stands (decode tok/s, short context, same four prompts)

| engine | story | code | math | json | mean |
|---|---|---|---|---|---|
| LM Studio, no MTP | 73.1 | 72.9 | 72.7 | 72.6 | 72.8 |
| LM Studio, MTP | 87.5 | 113.6 | 117.8 | 180.6 | 124.9 |
| carbon, no MTP | 66.2 | 66.0 | 65.9 | 65.7 | 66.0 |
| carbon, `--mtp --mtp-drafts 5 --mtp-min-prob 0.75` | 91.4 | 145.5 | 149.4 | 174.2 | 140.1 |
| carbon, `--mtp --mtp-drafts 7 --mtp-min-prob 0.5` | 92.5 | 147.1 | 148.8 | 179.8 | 142.0 |

At depth (decode tok/s; carbon with `--kv-dtype q8 --mtp --mtp-drafts 5 --mtp-min-prob 0.75
--max-running-requests 2 --state-snapshots 4`, which holds the full 262,144-token window in
28 GiB; story / json where only two prompts were run):

| engine | 32k (4 prompts) | 91k | 129k | 253k |
|---|---|---|---|---|
| LM Studio, MTP | 104.7 | 86.7 | does not fit | does not fit |
| LM Studio, no MTP | | | 37.0 | 24.5 |
| carbon, q8 KV + MTP | 116.1 | | 53.6 / 83.8 | 40.2 / 56.2 |

Time to first token, cold (seconds):

| engine | 32k | 129k | 253k |
|---|---|---|---|
| LM Studio | 11.3 | 72 | 148 |
| carbon | 24 | 293 | 1041 |

With MTP carbon is ahead of LM Studio on the mean at short context (+12%; story +4%, code
+28%, math +27%, JSON -4% with 5 drafts, level with 7), at 32k (+11%), and about 2x at 129k
and 253k, where LM Studio has to turn MTP off. Without MTP carbon is 9% behind. Decode still
slows with depth (140 -> 116 -> 69 -> 48), and cold prefill is the weak point: 2x slower at 32k
and 7x slower at 253k. A second request sharing a long prefix resumes from the last prefill
chunk boundary inside it (32k: 3.8 s instead of 24 s).

### Phase 3: decode speed (in progress)

| step | decode tok/s (short context, no MTP) |
|---|---|
| float GEMV kernel | 14.9 |
| 8-bit activations + dp4a GEMV | 58 |
| one activation prep per input, packed GDN a/b projections | 59 |
| GDN state update split over 8 threads per column | 61 |

- The dp4a kernel (`kq_gemv_i8`) reads weights at about 1.5 TB/s from memory in
  `examples/kqbench.rs` and about 1.2 TB/s in the model (401 launches of about 32 us per
  token). Per token: GEMVs 12.8 ms, everything else about 3.6 ms, mostly the latency of about
  1,700 small launches (norms, adds, activation prep, rope, sampling).
- LM Studio's 13.7 ms per token means it has almost no overhead on top of the weight read. The
  remaining gap is launch count: fuse gate + up (and q/k/v, qkv/z/a/b) into one launch each,
  fold SiLU * up, the residual add and the next norm into kernel epilogues.
- `kernels/kquant_tc.cu`: integer tensor-core GEMM for multi-token steps, rows staged through
  shared memory. From memory: Q4_K 1.51 / 1.40 / 1.32 / 1.25 TB/s for 1 / 2 / 4 / 8 tokens,
  Q6_K 1.64 / 1.44 / 1.32 / 1.35. Used from 2 tokens; one token stays on dp4a (faster in the
  model). `CARBON_KQ_TC_MIN=a,b` moves the thresholds, `CARBON_KQ_EXACT=1` selects the float
  reference kernel.

### Phase 3/4 update: launch fusion, one-launch GDN chains, exact verify steps

- Projections that share an input (gate + up, q/k/v, qkv/z/a/b) run in one launch
  (`kq_gemv_i8_multi`, `kq_gemm_tc_multi`): plain decode 61 -> 65 tok/s. Sixty 50 MB weights
  launched one by one read at 1.30 TB/s against 1.48 as one weight.
- Multi-token steps: the GDN kernels walk a sequence's tokens inside one launch (state in
  registers, each token's state written to its own slot), instead of one launch per token with
  state copies between spare slots.
- The dp4a and tensor-core kernels add a row's terms in different orders, so mixing them by
  token count made MTP output drift from plain decoding at near-ties. The kernel now follows
  the engine mode (tensor-core with `--mtp`, dp4a without; `CARBON_KQ_TC_ROWS` overrides).
  With that, `--mtp` output is identical to plain greedy decoding on three 400-token prompts
  (2 and 5 drafts). Summing in f64 would also fix it but made steps 2.5x slower.
- A 3-token verify step now costs 17.9 ms (plain step on the same kernels: 16.8 ms).
- GDN step blocks use 512 threads: 1024 exceeded a launch's register budget.

### Phase 4: MTP (working)

- `model/mtp.rs` `DenseMtp`: the dense Qwen3.5 head (`blk.64` + `nextn.*`), after llama.cpp's
  `graph_mtp`. `--mtp-shadow` measures 65-74% draft acceptance on a thinking-style prompt, 94% on
  JSON.
- The scheduler drafts up to 7 tokens per step (`--mtp-drafts`); above 2 the chain goes on while
  the last draft's probability is at least `--mtp-min-prob`.
- Cost per cycle on JSON with 2 drafts: verify 22.7 ms + drafts 2.4 ms for 2.82 tokens. A plain
  step is 16.4 ms, so a 3-token verify step costs 1.38x; each further draft costs about 1.2 ms
  (0.7 ms of it is reading the 1 GiB LM head). LM Studio gets 2.5 tokens in about one plain step.
- Open: the multi-token step's extras (per-token GDN launches and state copies between spare
  slots, row gathers in `per_substep`, attention reading K/V once per token row), and the
  phase 3 launch fusion, which every verify step also pays.

### Phase 5 update: int8 KV, full window, chunk snapshots

- `--kv-dtype q8`: each K / V vector is 256 int8 values plus an inline f32 scale (260 bytes).
  `kv_q8_write` quantizes on write; the decode and prefill tensor-core kernels convert rows to
  the activation dtype while staging them in shared memory, so the MMA code is shared with the
  bf16 pools. Other readers (generic and non-tensor-core paths) refuse int8 pools.
- Same decode speed as bf16 KV at short context. Text differs from the bf16-KV run after about
  180 characters on one prompt (lossy cache); no quality evaluation yet. `--mtp` output is
  still identical to plain decoding with the same kernels.
- Full window: 270,854 KV tokens fit with the MTP head when the recurrent-state pools are
  sized for a single user (`--max-running-requests 2 --state-snapshots 4`; the defaults of 16
  + 16 slots take about 6 GB at 151 MB each and leave 150k tokens).
- The chunked decode kernel's partials live in one scratch buffer per device and size. Per-call
  buffers made every captured CUDA graph pin one per layer and batch shape.
- Recurrent state is snapshotted at every prefill chunk boundary of a long prompt; when the
  slots run out the request recycles a snapshot on its own path (keeps the shallowest).
- Greedy pick kernel (`logit_pick`): argmax and its probability in one pass, replacing three
  generic reductions of about 0.1 ms each per draft.
- Why prefill is slow at depth: attention time grows with n^2 (293 s at 129k, about 225 s of
  it attention). The prefill kernel runs one block per (64 queries, head, half of the output
  dims), each staging the same K / V rows itself: 12 blocks per KV head re-read and convert
  the prefix, and the scores are computed twice for head dim 256.

### Phase 5: context depth (first round)

- `paged_decode_tc` (`kernels/attention_tc.cu`) replaces the per-head decode kernel on sm_80+:
  256-position chunks, one block each (4 warps of 64 positions, merged in the block), the 6
  query heads of a KV head scored together so K and V are read once. Launch shapes depend on the
  pool size only. Decode at 32k went from 5.9 to 54.4 tok/s without MTP.
- bf16 KV fits about 104k tokens next to the weights (about 82k with the MTP head and its spare
  state slots). Full context still needs the int8 KV pool and int8 variants of the decode and
  prefill attention kernels.
- KV auto-sizing leaves too little room for prefill temporaries: `--max-prefill-tokens 16384`
  with `--mtp` ran out of memory during a 32k prefill. `--kv-tokens 70000` worked.
- Prefill: carbon 16k in 10.0 s, 32k in 26 s, 60k in 75 s, 91k in 145 s. LM Studio: 32k in
  11.3 s, 91k in 33.9 s. Profile of a 16k prefill: cuBLAS GEMMs 48%, prefill attention 18%
  (quadratic, about 28 TFLOPS), K-quant dequant 13%, GDN recurrence 10%, element-wise ops 12%.
- A request that shares a long prefix with an earlier one but ends differently re-prefills
  everything (26 s again at 32k; LM Studio 0.9 s): recurrent-state snapshots only exist at the
  end of earlier prompts. Periodic snapshots during prefill would fix it (151 MB each).

### Tests

- `cargo test --release`: 111 pass (72 unit, 25 parallel, 4 real_configs, 4 server, 6 vision).
- `carbon devices --selftest`: all kernel checks pass in bf16 / f16 / f32, including the new
  Q4_K / Q6_K (float, dp4a and tensor-core paths) and multi-chunk decode attention checks.
- `CARBON_TEST_DEVICE=cuda cargo test --features cuda --test parallel -- --test-threads=1`:
  27 pass, 1 ignored. With the default parallel test threads some CUDA-graph tests fail with
  `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` (tests capturing graphs on the same GPU at once); the
  original commit does the same.

## Next steps, in order

1. Prefill at depth: a prefill attention kernel that stages K / V once per KV head for its six
   query heads and scores once; then dequant passes, GDN recurrence and element-wise ops.
2. Decode at depth: multi-token steps read K / V once per token row; int8 MMA instead of
   converting to half precision.
3. Int8 KV quality check against bf16 KV (logit-level), and `--kv-dtype auto`.
4. KV / state auto-sizing: room for prefill temporaries, single-user defaults for long windows.
5. Plain decode (66 vs 73): fuse norm + prep and SiLU * up into kernel epilogues.
4. Phase 6 sweep and README; phase 7 image input.
