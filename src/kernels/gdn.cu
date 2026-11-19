// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// carbon-engine Gated DeltaNet decode kernels, compiled with NVRTC after quant.cu (dtype helpers).
//
// One decode step of a linear-attention layer for B sequences, updating the state pools in place:
//   gdn_conv  causal depthwise conv + SiLU over [conv state ++ x], shifting the conv state
//             (one thread per (sequence, channel), so reads and writes of a channel never race)
//   gdn_step  per (sequence, value head): l2-normalised q/k, gated delta rule on the f32
//             [k_dim, v_dim] state, then the gated RMSNorm of the output
// The numerics follow model/gdn.rs (the HF reference) step by step.

#define GDN_MAX_DK 256
#define GDN_MAX_DV 256
#define GDN_MAX_PARTS 8   // threads per state column in gdn_step
#define GDN_PART_ROWS 32  // most state rows one of them takes

__device__ __forceinline__ float gdn_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the result.
__device__ float gdn_block_sum(float v, float* red) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int w = 0; w < (int)(blockDim.x / 32); ++w) t += red[w];
    __syncthreads();
    return t;
}

// Multi-token steps: `fresh[b]` also links the rows of one sequence (consecutive, in token order).
//   bit 0   the sequence starts here: its state is zero
//   bit 1   this row continues the chain of an earlier row, whose thread / block computes it
//   >> 8    rows that follow this one in its chain
// The head row reads the state in slot idx[b]; token j of the chain leaves its state in slot
// idx[b + j] (rejected drafts roll back by dropping slots), and the state stays in registers
// between the tokens.
#define GDN_MAX_KM1 8

// qkv [B, C]; conv pool [slots, KM1 * C]; taps [KM1 + 1, C]; act [B, C] f32.
template <class D>
__device__ void gdn_conv(const typename D::T* qkv, typename D::T* pool, const unsigned* idx, const unsigned* fresh,
                         const typename D::T* taps, float* act, int B, int C, int KM1) {
    long total = (long)B * C;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
        long b = i / C, c = i % C;
        unsigned link = fresh[b];
        if (link & 2u) continue;
        int len = 1 + (int)(link >> 8);
        bool zero = (link & 1u) != 0;
        float p[GDN_MAX_KM1];
        const typename D::T* src = pool + (long)idx[b] * KM1 * C;
#pragma unroll
        for (int j = 0; j < GDN_MAX_KM1; ++j) p[j] = (j < KM1 && !zero) ? D::load(src, (long)j * C + c) : 0.0f;
        for (int tk = 0; tk < len; ++tk) {
            float x = D::load(qkv, i + (long)tk * C);
            float acc = D::load(taps, (long)KM1 * C + c) * x;
#pragma unroll
            for (int j = 0; j < GDN_MAX_KM1; ++j)
                if (j < KM1) acc += D::load(taps, (long)j * C + c) * p[j];
            // shift: state rows become [prev[1..], x]
#pragma unroll
            for (int j = 0; j + 1 < GDN_MAX_KM1; ++j)
                if (j + 1 < KM1) p[j] = p[j + 1];
            if (KM1 > 0) {
                // the state holds the model dtype: later steps read it back rounded
                typename D::T r;
                D::store(&r, 0, x);
                p[KM1 - 1] = D::load(&r, 0);
            }
            typename D::T* dst = pool + (long)idx[b + tk] * KM1 * C;
#pragma unroll
            for (int j = 0; j < GDN_MAX_KM1; ++j)
                if (j < KM1) D::store(dst, (long)j * C + c, p[j]);
            act[i + (long)tk * C] = acc * gdn_sigmoid(acc);
        }
    }
}

// grid B * HV blocks. act [B, C] (C = 2*HK*DK + HV*DV); z [B, HV*DV]; braw, araw [B, HV];
// ssm pool [slots, HV*DK*DV] f32; y [B, HV*DV].
template <class D>
__device__ void gdn_step(const float* act, const typename D::T* z, const typename D::T* braw,
                         const typename D::T* araw, float* ssm, const unsigned* idx, const unsigned* fresh,
                         const float* dt_bias, const float* a_neg, const typename D::T* norm_w, typename D::T* y,
                         int HK, int HV, int DK, int DV, float eps, int flags) {
    // flags: bit 0 = sigmoid output gate, bit 1 = value heads tiled over the key heads
    int gate_sigmoid = flags & 1;
    __shared__ float sq[GDN_MAX_DK], sk[GDN_MAX_DK], so[GDN_MAX_DV], red[32];
    __shared__ float sm[GDN_MAX_PARTS][GDN_MAX_DV];
    const int b0 = blockIdx.x / HV, h = blockIdx.x % HV, kh = (flags & 2) ? h % HK : h / (HV / HK);
    const unsigned link = fresh[b0];
    if (link & 2u) return;
    const int len = 1 + (int)(link >> 8);
    const long C = 2L * HK * DK + (long)HV * DV;

    // The state update is DK x DV: with a block of P * JW threads (JW = DV rounded up to a warp),
    // thread (part, j) takes DK / P rows of column j and the parts add up through shared memory.
    // One thread per column walking all DK rows twice is latency-bound (24 us for 128 x 128).
    int JW = ((DV + 31) / 32) * 32;
    if (JW > (int)blockDim.x) JW = blockDim.x;
    int P = blockDim.x / JW;
    if (P > GDN_MAX_PARTS) P = GDN_MAX_PARTS;
    if (P > 1 && (DK % P != 0 || DK / P > GDN_PART_ROWS || JW < DV)) P = 1;
    const int part = threadIdx.x / JW, jc = threadIdx.x % JW;
    const bool on = P > 1 && part < P && jc < DV;
    const int nr = DK / P, i0 = part * nr;
    float sreg[GDN_PART_ROWS];

    for (int tk = 0; tk < len; ++tk) {
        const int b = b0 + tk;
        const float* a = act + b * C;
        float pq = 0.0f, pk = 0.0f;
        for (int i = threadIdx.x; i < DK; i += blockDim.x) {
            float qv = a[(long)kh * DK + i], kv = a[(long)HK * DK + (long)kh * DK + i];
            sq[i] = qv;
            sk[i] = kv;
            pq += qv * qv;
            pk += kv * kv;
        }
        float rq = rsqrtf(gdn_block_sum(pq, red) + 1e-6f) * rsqrtf((float)DK);
        float rk = rsqrtf(gdn_block_sum(pk, red) + 1e-6f);
        for (int i = threadIdx.x; i < DK; i += blockDim.x) {
            sq[i] *= rq;
            sk[i] *= rk;
        }
        __syncthreads();

        float beta = gdn_sigmoid(D::load(braw, (long)b * HV + h));
        float ga = D::load(araw, (long)b * HV + h) + dt_bias[h];
        float softplus = fmaxf(ga, 0.0f) + logf(1.0f + expf(-fabsf(ga)));
        float decay = expf(softplus * a_neg[h]);
        const bool zero = tk == 0 && (link & 1u) != 0;
        // read where the previous token (or step) left the state, write this token's slot
        const float* Sr = ssm + (long)idx[tk == 0 ? b : b - 1] * HV * DK * DV + (long)h * DK * DV;
        float* Sw = ssm + (long)idx[b] * HV * DK * DV + (long)h * DK * DV;
        const float* v = a + 2L * HK * DK + (long)h * DV;

        float pss = 0.0f;
        if (P > 1) {
            float mem = 0.0f;
            if (on) {
#pragma unroll
                for (int r = 0; r < GDN_PART_ROWS; ++r) {
                    if (r < nr) {
                        // later tokens of a chain: the state is still in registers
                        float prev = tk == 0 ? (zero ? 0.0f : Sr[(long)(i0 + r) * DV + jc]) : sreg[r];
                        sreg[r] = prev * decay;
                        mem += sk[i0 + r] * sreg[r];
                    }
                }
                sm[part][jc] = mem;
            }
            __syncthreads();
            float tot = 0.0f;
            if (on)
                for (int q = 0; q < P; ++q) tot += sm[q][jc];
            __syncthreads();
            if (on) {
                float delta = (v[jc] - tot) * beta, o = 0.0f;
#pragma unroll
                for (int r = 0; r < GDN_PART_ROWS; ++r) {
                    if (r < nr) {
                        float sv = sreg[r] + sk[i0 + r] * delta;
                        sreg[r] = sv;
                        Sw[(long)(i0 + r) * DV + jc] = sv;
                        o += sq[i0 + r] * sv;
                    }
                }
                sm[part][jc] = o;
            }
            __syncthreads();
            if (on && part == 0) {
                float o = 0.0f;
                for (int q = 0; q < P; ++q) o += sm[q][jc];
                so[jc] = o;
                pss = o * o;
            }
        } else {
            for (int j = threadIdx.x; j < DV; j += blockDim.x) {
                float mem = 0.0f;
                for (int i = 0; i < DK; ++i) mem += sk[i] * (zero ? 0.0f : Sr[(long)i * DV + j] * decay);
                float delta = (v[j] - mem) * beta;
                float o = 0.0f;
                for (int i = 0; i < DK; ++i) {
                    float sv = (zero ? 0.0f : Sr[(long)i * DV + j] * decay) + sk[i] * delta;
                    Sw[(long)i * DV + j] = sv;
                    o += sq[i] * sv;
                }
                so[j] = o;
                pss += o * o;
            }
        }
        float inv = rsqrtf(gdn_block_sum(pss, red) / (float)DV + eps);
        long base = (long)b * HV * DV + (long)h * DV;
        for (int j = threadIdx.x; j < DV; j += blockDim.x) {
            // the reference rounds to the model dtype after the norm and after the weight
            typename D::T r;
            D::store(&r, 0, so[j] * inv);
            D::store(&r, 0, D::load(&r, 0) * D::load(norm_w, j));
            float zg = D::load(z, base + j);
            // output gate: silu(z), or sigmoid(z) (Qwen3.8 output_gate_type)
            D::store(y, base + j, D::load(&r, 0) * (gate_sigmoid ? gdn_sigmoid(zg) : zg * gdn_sigmoid(zg)));
        }
        __syncthreads();
    }
}

#define DEFINE_GDN(SUFFIX, D)                                                                                     \
    extern "C" __global__ void gdn_conv_##SUFFIX(const D::T* qkv, D::T* pool, const unsigned* idx,              \
                                                 const unsigned* fresh, const D::T* taps, float* act, int B,      \
                                                 int C, int KM1) {                                                \
        gdn_conv<D>(qkv, pool, idx, fresh, taps, act, B, C, KM1);                                                 \
    }                                                                                                             \
    extern "C" __global__ void gdn_step_##SUFFIX(const float* act, const D::T* z, const D::T* braw,              \
                                                 const D::T* araw, float* ssm, const unsigned* idx,               \
                                                 const unsigned* fresh, const float* dt_bias, const float* a_neg, \
                                                 const D::T* norm_w, D::T* y, int HK, int HV, int DK, int DV,     \
                                                 float eps, int gate_sigmoid) {                                   \
        gdn_step<D>(act, z, braw, araw, ssm, idx, fresh, dt_bias, a_neg, norm_w, y, HK, HV, DK, DV, eps,         \
                    gate_sigmoid);                                                                                \
    }

DEFINE_GDN(f32, F32)
DEFINE_GDN(bf16, BF16)
DEFINE_GDN(f16, F16)

// ---- prefill: the gated delta rule over a whole sequence ----
//
// Column j of the [DK, DV] state evolves independently of the others, so each warp owns one
// (head, column) with the column in registers (lane holds rows lane, lane+32, ...) and walks
// the tokens: no chunking, no triangular solve, exactly the token-by-token recurrence.
//   q, k [BH, T, DK] (l2-normalised, q pre-scaled), v [BH, T, DV], beta, g [BH, T],
//   s0 [BH, DK, DV] -> o [BH, T, DV], s1 [BH, DK, DV]; all f32.

#define GDN_MAX_R (GDN_MAX_DK / 32)

__device__ __forceinline__ float gdn_warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

extern "C" __global__ void gdn_seq(const float* q, const float* k, const float* v, const float* beta,
                                   const float* g, const float* s0, float* o, float* s1, int BH, int T, int DK,
                                   int DV) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    long col = (long)blockIdx.x * (blockDim.x / 32) + warp;
    if (col >= (long)BH * DV) return;
    int bh = col / DV, j = col % DV, R = DK / 32;
    float S[GDN_MAX_R];
#pragma unroll
    for (int r = 0; r < GDN_MAX_R; ++r) S[r] = r < R ? s0[((long)bh * DK + lane + 32 * r) * DV + j] : 0.0f;
    const float* qb = q + (long)bh * T * DK;
    const float* kb = k + (long)bh * T * DK;
    const float* vb = v + (long)bh * T * DV;
    for (int t = 0; t < T; ++t) {
        float kr[GDN_MAX_R], qr[GDN_MAX_R];
#pragma unroll
        for (int r = 0; r < GDN_MAX_R; ++r) {
            kr[r] = r < R ? kb[(long)t * DK + lane + 32 * r] : 0.0f;
            qr[r] = r < R ? qb[(long)t * DK + lane + 32 * r] : 0.0f;
        }
        float dec = expf(g[(long)bh * T + t]), bt = beta[(long)bh * T + t], vt = vb[(long)t * DV + j];
        float mem = 0.0f;
#pragma unroll
        for (int r = 0; r < GDN_MAX_R; ++r) {
            S[r] *= dec;
            mem += kr[r] * S[r];
        }
        float delta = (vt - gdn_warp_sum(mem)) * bt;
        float ov = 0.0f;
#pragma unroll
        for (int r = 0; r < GDN_MAX_R; ++r) {
            S[r] += kr[r] * delta;
            ov += qr[r] * S[r];
        }
        ov = gdn_warp_sum(ov);
        if (lane == 0) o[((long)bh * T + t) * DV + j] = ov;
    }
#pragma unroll
    for (int r = 0; r < GDN_MAX_R; ++r)
        if (r < R) s1[((long)bh * DK + lane + 32 * r) * DV + j] = S[r];
}

// ---- prefill, fused: conv + SiLU + l2 norm, gates, the recurrence, the gated norm ----
//
// One sequence, T tokens. The numerics follow the decode kernels above (f32 throughout, the
// reference's two rounding steps in the output norm).
//   gdn_pre    qkv [T, C] -> act [T, C] f32: per token, q and k of key head kh interleaved at
//              [kh * 2 * DK + 2 * i + (0: q, 1: k)] (l2-normalised, q scaled by DK^-1/2), then v.
//              One warp per q / k vector or per 32 value channels.
//   gdn_gates  braw, araw [T, HV] -> bd [T, HV, 2]: write strength and decay of every token
//   gdn_seq2   the delta rule; thread (column j, part) holds GDN_SEQ_ROWS rows of state column j,
//              so a warp owns 32 / P columns (P = DK / GDN_SEQ_ROWS) and sums over a column's
//              parts with log2(P) shuffles. A warp per column spends most of its instructions on reductions.
//   gdn_post   gated RMSNorm of each value head: o [T * HV, DV] f32 -> y in the model dtype

template <class D>
__device__ void gdn_pre(const typename D::T* qkv, const typename D::T* pool, const unsigned* idx,
                        const typename D::T* taps, float* act, int T, int C, int KM1, int HK, int DK, int zero) {
    const typename D::T* prev = pool + (long)idx[0] * KM1 * C;
    const int lane = threadIdx.x % 32, QK = 2 * HK, R = DK / 32;
    const int units = QK + (C - QK * DK) / 32;
    const long total = (long)T * units, warps = (long)gridDim.x * (blockDim.x / 32);
    for (long u = (long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; u < total; u += warps) {
        const int t = (int)(u / units), un = (int)(u % units);
        const bool vec = un < QK;
        const int c0 = vec ? un * DK : QK * DK + (un - QK) * 32;
        float x[GDN_MAX_R], ss = 0.0f;
#pragma unroll
        for (int r = 0; r < GDN_MAX_R; ++r) {
            x[r] = 0.0f;
            if (r < (vec ? R : 1)) {
                const int c = c0 + lane + 32 * r;
                float acc = D::load(taps, (long)KM1 * C + c) * D::load(qkv, (long)t * C + c);
#pragma unroll
                for (int j = 0; j < GDN_MAX_KM1; ++j) {
                    if (j < KM1) {
                        const int tau = t - KM1 + j;  // rows before the chunk come from the conv state
                        float p = tau >= 0 ? D::load(qkv, (long)tau * C + c)
                                           : (zero ? 0.0f : D::load(prev, (long)(tau + KM1) * C + c));
                        acc += D::load(taps, (long)j * C + c) * p;
                    }
                }
                acc *= gdn_sigmoid(acc);
                x[r] = acc;
                ss += acc * acc;
            }
        }
        float* dst = act + (long)t * C;
        if (vec) {
            const int which = un / HK, kh = un % HK;  // 0: q, 1: k
            float rs = rsqrtf(gdn_warp_sum(ss) + 1e-6f);
            if (which == 0) rs *= rsqrtf((float)DK);
#pragma unroll
            for (int r = 0; r < GDN_MAX_R; ++r)
                if (r < R) dst[(long)kh * 2 * DK + 2 * (lane + 32 * r) + which] = x[r] * rs;
        } else {
            dst[c0 + lane] = x[0];
        }
    }
}

// The conv state after the chunk: its last KM1 inputs (older rows shift down when T < KM1). One
// thread per channel, rows in order, so a row is read before it is overwritten.
template <class D>
__device__ void gdn_conv_tail(const typename D::T* qkv, typename D::T* pool, const unsigned* idx, int T, int C,
                              int KM1, int zero) {
    typename D::T* st = pool + (long)idx[0] * KM1 * C;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < C; c += gridDim.x * blockDim.x) {
        for (int j = 0; j < KM1; ++j) {
            const int tau = T - KM1 + j;
            if (tau >= 0) st[(long)j * C + c] = qkv[(long)tau * C + c];
            else if (zero) D::store(st, (long)j * C + c, 0.0f);
            else st[(long)j * C + c] = st[(long)(tau + KM1) * C + c];
        }
    }
}

template <class D>
__device__ void gdn_gates(const typename D::T* braw, const typename D::T* araw, const float* dt_bias,
                          const float* a_neg, float* bd, int T, int HV) {
    const long total = (long)T * HV;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
        const int h = (int)(i % HV);
        float ga = D::load(araw, i) + dt_bias[h];
        float softplus = fmaxf(ga, 0.0f) + logf(1.0f + expf(-fabsf(ga)));
        bd[2 * i] = gdn_sigmoid(D::load(braw, i));
        bd[2 * i + 1] = expf(softplus * a_neg[h]);
    }
}

#ifndef GDN_SEQ_ROWS
#define GDN_SEQ_ROWS 32
#endif
#ifdef CARBON_HOST_TEST
struct gdn_f4 { float x, y, z, w; };
#else
typedef float4 gdn_f4;
#endif

#define GDN_SEQ_TOK 16     // tokens staged in shared memory at a time
#define GDN_SEQ_WARPS 4    // warps per block (the launch uses blocks of this size)

// One block per (value head, GDN_SEQ_WARPS * 32 / P state columns). The tokens' q / k / v rows
// are new data at every step, and a warp waits a cache round trip for each: the block copies
// GDN_SEQ_TOK tokens into shared memory in one burst and then walks them from there.
// flags: bit 0 = the state starts at zero, bit 1 = value heads tiled over the key heads
extern "C" __global__ void gdn_seq2(const float* act, const float* bd, float* ssm, const unsigned* idx, float* o,
                                    int T, int C, int HK, int HV, int DK, int DV, int flags) {
    __shared__ gdn_f4 sqk[GDN_SEQ_TOK][GDN_MAX_DK / 2];
    __shared__ float sv[GDN_SEQ_TOK][GDN_SEQ_WARPS * 32];
    __shared__ float sbd[GDN_SEQ_TOK][2];
    // the state is read from and left in the sequence's slot (a thread owns its cells)
    float* st = ssm + (long)idx[0] * HV * DK * DV;
    const int P = DK / GDN_SEQ_ROWS, cpw = 32 / P;  // lanes per column, columns per warp
    const int lane = threadIdx.x % 32, part = lane % P;
    const int bcols = GDN_SEQ_WARPS * cpw;          // columns per block
    const int bph = (DV + bcols - 1) / bcols;       // blocks per head
    const int h = blockIdx.x / bph, j0 = (blockIdx.x % bph) * bcols;
    const int jb = (threadIdx.x / 32) * cpw + lane / P, j = j0 + jb;
    const bool on = j < DV;
    const int kh = (flags & 2) ? h % HK : h / (HV / HK);
    float S[GDN_SEQ_ROWS];
    {
        const float* sp = st + ((long)h * DK + part * GDN_SEQ_ROWS) * DV + (on ? j : DV - 1);
#pragma unroll
        for (int r = 0; r < GDN_SEQ_ROWS; ++r) S[r] = (flags & 1) ? 0.0f : sp[(long)r * DV];
    }
    const float* qk = act + (long)kh * 2 * DK;
    const float* vp = act + 2L * HK * DK + (long)h * DV + j0;
    for (int t0 = 0; t0 < T; t0 += GDN_SEQ_TOK) {
        const int n = min(T - t0, GDN_SEQ_TOK);
        __syncthreads();
        for (int i = threadIdx.x; i < n * (DK / 2); i += blockDim.x) {
            const int tok = i / (DK / 2), w = i % (DK / 2);
            sqk[tok][w] = ((const gdn_f4*)(qk + (long)(t0 + tok) * C))[w];
        }
        for (int i = threadIdx.x; i < n * bcols; i += blockDim.x) {
            const int tok = i / bcols, c = i % bcols;
            sv[tok][c] = j0 + c < DV ? vp[(long)(t0 + tok) * C + c] : 0.0f;
        }
        if ((int)threadIdx.x < 2 * n)
            sbd[threadIdx.x / 2][threadIdx.x & 1] = bd[2 * ((long)(t0 + threadIdx.x / 2) * HV + h) + (threadIdx.x & 1)];
        __syncthreads();
        for (int tk = 0; tk < n; ++tk) {
            const gdn_f4* row = sqk[tk] + part * (GDN_SEQ_ROWS / 2);
            const float beta = sbd[tk][0], dec = sbd[tk][1];
            float kk[GDN_SEQ_ROWS], qq[GDN_SEQ_ROWS];
            float m[4] = {0.0f, 0.0f, 0.0f, 0.0f}, a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int r = 0; r < GDN_SEQ_ROWS; r += 2) {
                const gdn_f4 x = row[r / 2];  // (q, k) of two rows
                qq[r] = x.x;
                kk[r] = x.y;
                qq[r + 1] = x.z;
                kk[r + 1] = x.w;
            }
#pragma unroll
            for (int r = 0; r < GDN_SEQ_ROWS; ++r) {
                S[r] *= dec;
                m[r & 3] += kk[r] * S[r];
            }
            float mem = (m[0] + m[1]) + (m[2] + m[3]);
            for (int off = 1; off < P; off <<= 1) mem += __shfl_xor_sync(0xffffffff, mem, off);
            const float delta = (sv[tk][jb] - mem) * beta;
#pragma unroll
            for (int r = 0; r < GDN_SEQ_ROWS; ++r) {
                S[r] += kk[r] * delta;
                a[r & 3] += qq[r] * S[r];
            }
            float ov = (a[0] + a[1]) + (a[2] + a[3]);
            for (int off = 1; off < P; off <<= 1) ov += __shfl_xor_sync(0xffffffff, ov, off);
            if (on && part == 0) o[((long)(t0 + tk) * HV + h) * DV + j] = ov;
        }
    }
    if (on) {
        float* sp = st + ((long)h * DK + part * GDN_SEQ_ROWS) * DV + j;
#pragma unroll
        for (int r = 0; r < GDN_SEQ_ROWS; ++r) sp[(long)r * DV] = S[r];
    }
}

template <class D>
__device__ void gdn_post(const float* o, const typename D::T* z, const typename D::T* norm_w, typename D::T* y,
                         long rows, int DV, float eps, int gate_sigmoid) {
    const int lane = threadIdx.x % 32;
    const long warps = (long)gridDim.x * (blockDim.x / 32);
    for (long u = (long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; u < rows; u += warps) {
        const long base = u * DV;
        float ss = 0.0f;
        for (int j = lane; j < DV; j += 32) ss += o[base + j] * o[base + j];
        const float inv = rsqrtf(gdn_warp_sum(ss) / (float)DV + eps);
        for (int j = lane; j < DV; j += 32) {
            // the reference rounds to the model dtype after the norm and after the weight
            typename D::T r;
            D::store(&r, 0, o[base + j] * inv);
            D::store(&r, 0, D::load(&r, 0) * D::load(norm_w, j));
            float zg = D::load(z, base + j);
            D::store(y, base + j, D::load(&r, 0) * (gate_sigmoid ? gdn_sigmoid(zg) : zg * gdn_sigmoid(zg)));
        }
    }
}

#define DEFINE_GDN_PREFILL(SUFFIX, D)                                                                             \
    extern "C" __global__ void gdn_pre_##SUFFIX(const D::T* qkv, const D::T* pool, const unsigned* idx,         \
                                                const D::T* taps, float* act, int T, int C, int KM1, int HK,     \
                                                int DK, int zero) {                                              \
        gdn_pre<D>(qkv, pool, idx, taps, act, T, C, KM1, HK, DK, zero);                                           \
    }                                                                                                             \
    extern "C" __global__ void gdn_conv_tail_##SUFFIX(const D::T* qkv, D::T* pool, const unsigned* idx, int T,  \
                                                      int C, int KM1, int zero) {                                \
        gdn_conv_tail<D>(qkv, pool, idx, T, C, KM1, zero);                                                        \
    }                                                                                                             \
    extern "C" __global__ void gdn_gates_##SUFFIX(const D::T* braw, const D::T* araw, const float* dt_bias,      \
                                                  const float* a_neg, float* bd, int T, int HV) {                \
        gdn_gates<D>(braw, araw, dt_bias, a_neg, bd, T, HV);                                                      \
    }                                                                                                             \
    extern "C" __global__ void gdn_post_##SUFFIX(const float* o, const D::T* z, const D::T* norm_w, D::T* y,     \
                                                 long rows, int DV, float eps, int gate_sigmoid) {               \
        gdn_post<D>(o, z, norm_w, y, rows, DV, eps, gate_sigmoid);                                                \
    }

DEFINE_GDN_PREFILL(f32, F32)
DEFINE_GDN_PREFILL(bf16, BF16)
DEFINE_GDN_PREFILL(f16, F16)
