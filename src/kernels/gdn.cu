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
