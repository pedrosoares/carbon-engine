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

At depth (decode tok/s, story / json; carbon with `--kv-dtype q8 --mtp --mtp-drafts 5
--mtp-min-prob 0.75 --max-running-requests 2 --state-snapshots 4`, which holds the full
262,144-token window in 28-29.5 GiB):

| engine | short (4 prompts, mean) | 32k (mean) | 128k | 250k |
|---|---|---|---|---|
| LM Studio, MTP | 124.9 | 104.7 | does not fit | does not fit |
| LM Studio, no MTP | 72.8 | | 37.0 / 37.0 | 24.6 / 24.5 |
| carbon, first int8 KV build | 139.0 | 116.1 | 53.6 / 83.8 | 40.2 / 56.2 |
| carbon, row-shared attention + shared slot tables | 132.4 | 115.6 | 58.5 / 109.2 | 44.0 / 80.1 |
| carbon, now (decode attention round) | 140.2 | 122.8 | 65.6 / 126.5 | 49.6 / 95.3 |

Time to first token (seconds):

| engine | 32k cold | 128k cold | 250k cold | same prefix again |
|---|---|---|---|---|
| LM Studio | 11.3 | 72 | 148 | 0.9 - 3.4 |
| carbon, first int8 KV build | 24 | 293 | 1041 | (re-prefilled) |
| carbon, row-shared attention build | 21 | 183 - 201 | not measured cold (481 s on top of a cached 128k prefix) | 3.1 (32k), 7.0 (128k), 14.4 (250k) |
| carbon, now (prefill round) | 12.2 | 73 | 204 | 1.7 (32k), 2.2 (128k), 4.3 (250k) |

With MTP carbon is ahead of LM Studio at short context (+12% on the mean; story +4%, code +28%,
math +27%, JSON -4% with 5 drafts, level with 7), at 32k (+10%), and 2.3x / 2.5x at 128k / 250k,
where LM Studio has to turn MTP off. Without MTP carbon is 9% behind. Decode still slows with
depth (132 -> 116 -> 84 -> 62 on the means): at 128k a verify step is 34 ms against 18 ms at
short context, the difference being attention (the kernel is bound by its own arithmetic, about
1 ms per 13k tokens of context per step) and about 2 ms of host work per step that grows with
the context (slot tables rebuilt and uploaded every step). Cold prefill is level with LM
Studio's to 128k (12.2 s vs 11.3 s at 32k, 73 s vs 72 s at 128k) and 1.4x slower at 250k.

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

### Phase 8: KV cache quantizers behind `--kv-dtype` (TurboQuant and RotorQuant done)

Both methods are the same pipeline: normalise a vector, rotate it so its coordinates look like
independent Gaussians, store a few bits per coordinate from a Lloyd-Max codebook plus the norm.
TurboQuant (Zandieh, Daliri, Hadian, Mirrokni, arXiv 2504.19874) uses a dense random orthogonal
matrix; RotorQuant (github.com/scrya-com/rotorquant) replaces it with block rotations: Cl(3,0)
rotors on 3 coordinates at a time, later 2D Givens ("planar") and 4D quaternion ("iso") blocks.

How it fits this engine: the rotation is orthogonal, so `<q, k> = <Pi q, Pi k>` and
`sum p_i v_i = Pi^T sum p_i (Pi v_i)`. Vectors are rotated once when written, the queries once
per step, the output rotated back once; the attention kernels only see codes. The rotation is a
`[D, D]` matrix applied with a small GEMM, so a different method is a different matrix.

Done, `--kv-dtype turbo4` (`KvFormat::Turbo4`):

- Storage per vector: D / 2 bytes of 4-bit codes (16-level Lloyd-Max of N(0, 1), MSE 0.0095,
  as the paper gives for 4 bits) and an f32 scale `|x| / sqrt(D) / (1 - MSE)`: 132 bytes at
  head dim 256 against 260 (`q8`) and 512 (bf16). The rotation is one seeded Gaussian matrix
  per head dim, orthonormalised on the host (`kvcache::random_rotation`).
- Kernels: `kv_tq4_write` (norm, codes, packing), and a third pool format in
  `flash_prefill_tc` and `paged_decode_tc_rows` (a byte -> two centroids table per block; the
  scales go into scores and probabilities like the int8 ones). TurboQuant pools always take
  the row-shared decode kernel.
- Checks: kernel self-checks for prefill, decode and the writer against a CPU encoder; unit
  tests for the codebook (a Lloyd fixed point) and the rotation.
- Measured (27B, full-window configuration, pool of 270,854 tokens): 23.5 GiB instead of 28;
  left to auto-sizing the pool holds 533,501 tokens. Decode 134.1 tok/s short (q8 140.2),
  67.8 at 128k story (q8 65.6). The attention kernels run at the same speed on both formats
  (0.36 vs 0.38 ms per layer over 129k): they are bound by conversion and arithmetic, not by
  bytes, so 4 bits buys memory, not speed.
- Quality: next token equal to bf16 KV on 229 of 250 prefixes (91.6%; q8 97.2%). A planted
  fact is recalled at 6k, 50k and 120k tokens. One benchmark prompt (JSON after 128k of filler)
  gets a refusal instead of the array. The scale's `1 / (1 - MSE)` factor (unbiased inner
  products) did not change the agreement.

Done, `--kv-dtype rotor4 | planar4 | iso4` (RotorQuant; `KvFormat::{Rotor4, Planar4, Iso4}`):

- The storage and kernels of `turbo4` (`KvFormat::storage`); only the matrix differs:
  `kvcache::block_rotation` puts an independent random rotation (determinant +1, what a rotor
  sandwich or a quaternion pair gives) on each group of 3 / 2 / 4 consecutive coordinates. At
  head dim 256 that is 85 rotors and one coordinate left as is for `rotor4`.
- Measured (27B, same 250 prefixes, next token equal to the bf16 cache): `turbo4` 91.6%
  (re-collected, identical to the earlier run on all 250), `planar4` 82.8%, `iso4` 82.0%,
  `rotor4` 80.0%. `q8` is 97.2%.
- `rotor4` in the full-window configuration: memory as `turbo4`; decode 127.0 tok/s short
  (turbo4 134.1, q8 140.2), 65.9 at 128k story; the planted fact is recalled at 6k and 120k.
- So the expectation held: no speed gain here (the rotation is two `[D, D]` GEMMs per layer
  and step whatever its structure) and clearly lower agreement. Likely reason, not verified:
  one scale per vector assumes coordinates of equal variance, which a dense rotation produces
  and a 2-4 dim block rotation does not. `turbo4` stays the recommended 4-bit format.

Planned, in order:

1. Why the block rotations lose: agreement with no rotation at all (the floor) and with a
   per-block scale, to see whether RotorQuant can be brought to `turbo4`'s level here.
2. The paper's second stage (1-bit QJL of the residual, unbiased inner products for K).
3. 3-bit variants (`turbo3` / `rotor3`), mixed formats per pool (e.g. K at 4 bits, V at 8).
4. The two rotations per attention layer and step as part of existing launches.

### Phase 5 update 4: decode attention at depth

Profile of a verify step at 128k before this round (graphs off, kernels only): attention 40%
(0.55 - 0.77 ms per layer launch), the K-quant GEMMs 49%, GDN 4%. The GPU is busy 96% of the
step: the host work that used to grow with the context is gone since the shared slot tables.

- `paged_decode_tc_rows` restaged: K and V of a tile in one pass (2 block barriers per tile
  instead of 4), int8 codes staged exactly with the scales applied to scores / probabilities,
  V transposed and K rows padded (fragment loads without bank conflicts).
- Blocks have at least 4 warps; the ones without columns only stage. A single-row step had one
  warp doing all the staging.
- Per layer over 129k positions (`attnbench decode`): 1 row 0.59 -> 0.35 ms, 3 rows 0.56 ->
  0.38, 6 rows 0.77 -> 0.53; 250k, 3 rows: 1.09 -> 0.75 ms. About 700 GB/s of int8 K / V; a
  bf16 pool is read at 1.2 TB/s by the same kernel, so the int8 conversion is what is left.
  Of the 0.38 ms: staging 0.10, MMAs and softmax 0.15 - 0.2, launch + merge + barriers 0.07.
- Did not help: skipping the accumulator rescale; two warps per column group on different
  halves of a 32-position tile (0.44 ms: larger blocks, fewer per SM).
- The benchmark at a 32k context flatters (the pool of one layer fits the L2 cache and the
  benchmark re-reads it); 129k and 250k are representative.
- Next-token agreement on a frozen 44k-token text (250 prefixes): bf16 KV vs int8 KV 97.2%,
  bf16 vs int8 on the row-shared kernel 96.8%, the two int8 kernels with each other 96.4%.

### Phase 5 update 3: prefill speed

Profile of a cold 16k prefill before this round (9.6 s): cuBLAS GEMMs 43% (about 250 TFLOPS,
at the hardware's rate), K-quant dequant 15%, attention 13%, GDN recurrence 11%, generic
element-wise ops 17%. After it (5.9 s): GEMMs 66%, recurrence 12%, attention 8%, dequant 3%.

- `kq_dequant`: one warp per 64 weights with lanes on consecutive weights, so every load and
  store covers one contiguous run (a thread per 32-weight group scattered each 2-byte store over
  32 cache lines): 1173 -> 184 ms per 16k tokens.
- Prefill attention (`flash_prefill_tc`), 1060 -> about 330 ms per 16k and 61 -> 109 TFLOPS at a
  127k prefix (`examples/attnbench`):
  - all 256 output dims in one block, so the scores are computed once (the split was a guess
    about registers that did not hold: two heads per block still launch);
  - int8 K / V are staged as their codes, exact in bf16 / f16, and the per-vector scales go
    into the scores and the probabilities (no rounding of scaled values, a third of the work);
  - V is staged transposed, so P V loads its fragments as two words like Q K^T;
  - a thread issues all its staging loads before converting any. The load-convert-store loop
    waited a cache round trip per item and was half the kernel.
  What did not help: skipping the accumulator rescale when the max is unchanged, 64-key tiles,
  and loading the next tile during the MMAs (the extra live registers spill). In this kernel
  the MMAs themselves are about a third of the time; staging is 40% and is bound by the
  registers a block needs (128 (query, head) rows per block, one block per SM).
- Gated DeltaNet prefill on carbon kernels (`gdn_pre`, `gdn_gates`, `gdn_seq2`, `gdn_post`,
  `gdn_conv_tail`): conv + SiLU + l2 norm, gates, recurrence and gated norm, with the state
  read from and left in the sequence's slot. The generic path spent about 1.2 s per 16k tokens
  in casts, copies and broadcast multiplies around the recurrence. `gdn_seq2` holds 32 state
  rows per thread (8 columns per warp, 2 shuffles per reduction instead of 5): 5.3 -> 4.2 ms
  per 4096 tokens (`examples/gdnbench`). 16 or 8 rows per thread, whole-head blocks, vector
  loads and staging the tokens in shared memory all measured the same or worse; the kernel
  runs at about 0.3 instructions per cycle per SM and I have not found what bounds it.
- The 250k cold prefill is 204 s: about half attention (n^2), half the linear part.
- Int8 KV against bf16 KV, first measurement: the greedy next token after 250 growing prefixes
  (600 characters apart, up to 44.5k tokens of this repository's README, PLAN and model code)
  is the same for 245 (98.0%); the 5 differences are whitespace / punctuation / identifier
  choices in code. No logit-level comparison yet (the server has no logprobs output), and a
  planted fact is recalled at 6k and 60k tokens with int8 KV.
- `scripts/cuda_kernels_host_test.cpp` no longer compiles (it lacks shims for what `quant.cu`
  gained on this branch: shared memory, shuffles); the GDN and attention host tests run.

### Phase 5 update 2: row-shared decode attention, shared slot tables, prefix cache

- `paged_decode_tc_rows`: on pools of 131,072 slots or more, a block takes all token rows of a
  sequence and a 256-position chunk, stages each K / V tile once and scores 8 (row, head)
  columns per warp (blocks are sized to the step's rows). Used for single-row steps too, so
  verify steps stay bit-identical to single-token steps.
- Multi-token steps give the rows of a sequence one slot table with an (offset, end) pair per
  row. A copy per row made the tables rows x context long: they stopped fitting captured CUDA
  graphs beyond about 45k tokens (verify steps ran eagerly) and cost the host a pass each step.
  Off for Qwen3.8-Flash-Next, whose QSA selection kernel reads the old layout.
- Prefill attention: two heads that share a KV head go in one block and share its staged tiles
  (32k cold: 24 s -> 20.5 s; three per block exceeds the register budget at head dim 256).
- Prefix cache: a prompt whose first tokens were already in the tree without a snapshot (every
  chat shares the template header) was never cached while running, only when it finished, so
  neither the prompt-end snapshot nor the chunk snapshots existed. `cache_prompt` now lets the
  tree keep its slots for that head and tracks the request's duplicates until it ends.

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

1. Prefill beyond 128k: attention is n^2 and at 107 TFLOPS (more (query, head) rows per staged
   tile needs fewer registers per thread, e.g. Q in shared memory); the GDN recurrence is 12%
   of the linear part and far from the hardware's rate.
2. Decode at depth: 16 layers x 0.75 ms of attention per step at 250k next to about 15 ms of
   GEMMs. Left: int8 MMA for Q K^T (no conversion, but Q becomes 8-bit), and the fixed cost
   per launch (the grid covers the whole pool; the merge kernel walks every chunk).
3. Int8 KV: a logit-level comparison with bf16 KV (needs logprobs output; next-token agreement
   is 98.0% over 250 prefixes), and `--kv-dtype auto`.
4. KV / state auto-sizing: room for prefill temporaries, single-user defaults for long windows.
5. Plain decode (66 vs 73): fuse norm + prep and SiLU * up into kernel epilogues.
4. Phase 6 sweep and README; phase 7 image input.
