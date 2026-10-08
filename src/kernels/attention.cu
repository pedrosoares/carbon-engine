// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// carbon-engine paged-attention decode kernel, compiled at runtime with NVRTC (no CUDA headers).
//
// One block per (sequence, query head); 4 warps split the sequence's KV positions, lanes split
// the head dimension (D/32 elements each). Scores use an online softmax, so nothing of size
// kv_len is materialised; the 4 warp partials are merged through shared memory.

typedef unsigned char u8;
typedef unsigned short u16;

#ifndef CARBON_HOST_TEST
__device__ __forceinline__ float f16_to_f32_a(u16 h) {
    float f;
    asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ u16 f32_to_f16_a(float f) {
    u16 h;
    asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}
#else
__device__ __forceinline__ float f16_to_f32_a(u16 h) { _Float16 x; memcpy(&x, &h, 2); return (float)x; }
__device__ __forceinline__ u16 f32_to_f16_a(float f) { _Float16 x = (_Float16)f; u16 h; memcpy(&h, &x, 2); return h; }
#endif

struct AF32 { typedef float T;
    __device__ static float load(const float* p, long i) { return p[i]; }
    __device__ static void store(float* p, long i, float v) { p[i] = v; } };
struct ABF16 { typedef u16 T;
    __device__ static float load(const u16* p, long i) { return __int_as_float(((unsigned)p[i]) << 16); }
    __device__ static void store(u16* p, long i, float v) {
        unsigned u = __float_as_int(v);
        if ((u & 0x7fffffff) > 0x7f800000) { p[i] = (u16)((u >> 16) | 0x40); return; }
        u += 0x7fff + ((u >> 16) & 1);
        p[i] = (u16)(u >> 16);
    } };
struct AF16 { typedef u16 T;
    __device__ static float load(const u16* p, long i) { return f16_to_f32_a(p[i]); }
    __device__ static void store(u16* p, long i, float v) { p[i] = f32_to_f16_a(v); } };

#define ATT_WARPS 4
#define MAX_PER_LANE 8  // head_dim <= 256

// q:      [B, H, D]
// k, v:   [S, KVH, D] (the KV pool, slot-major)
// slots:  concatenated slot tables; starts[b] .. starts[b+1] are sequence b's positions
// out:    [B, H, D]
template <class A>
__device__ void paged_decode(const typename A::T* q, const typename A::T* kc, const typename A::T* vc,
                             const unsigned* slots, const unsigned* starts, typename A::T* out, int H, int KVH,
                             int D, float scale) {
    __shared__ float sm_m[ATT_WARPS];
    __shared__ float sm_l[ATT_WARPS];
    __shared__ float sm_acc[ATT_WARPS][MAX_PER_LANE * 32];

    int b = blockIdx.x / H, h = blockIdx.x % H;
    int kvh = h / (H / KVH);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int per = D / 32;
    unsigned s0 = starts[b], s1 = starts[b + 1];

    float qv[MAX_PER_LANE], acc[MAX_PER_LANE];
    for (int i = 0; i < per; ++i) {
        qv[i] = A::load(q, ((long)b * H + h) * D + lane * per + i) * scale;
        acc[i] = 0.0f;
    }
    float m = -1e30f, l = 0.0f;
    for (unsigned j = s0 + warp; j < s1; j += ATT_WARPS) {
        long base = ((long)slots[j] * KVH + kvh) * D + lane * per;
        float s = 0.0f;
        for (int i = 0; i < per; ++i) s += qv[i] * A::load(kc, base + i);
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
        float m_new = fmaxf(m, s);
        float corr = __expf(m - m_new), p = __expf(s - m_new);
        l = l * corr + p;
        for (int i = 0; i < per; ++i) acc[i] = acc[i] * corr + p * A::load(vc, base + i);
        m = m_new;
    }
    if (lane == 0) { sm_m[warp] = m; sm_l[warp] = l; }
    for (int i = 0; i < per; ++i) sm_acc[warp][lane * per + i] = acc[i];
    __syncthreads();
    if (warp == 0) {
        float mm = sm_m[0];
        for (int w = 1; w < ATT_WARPS; ++w) mm = fmaxf(mm, sm_m[w]);
        float ll = 0.0f;
        float o[MAX_PER_LANE];
        for (int i = 0; i < per; ++i) o[i] = 0.0f;
        for (int w = 0; w < ATT_WARPS; ++w) {
            float c = __expf(sm_m[w] - mm);
            ll += sm_l[w] * c;
            for (int i = 0; i < per; ++i) o[i] += sm_acc[w][lane * per + i] * c;
        }
        float inv = ll > 0.0f ? 1.0f / ll : 0.0f;
        for (int i = 0; i < per; ++i) A::store(out, ((long)b * H + h) * D + lane * per + i, o[i] * inv);
    }
}

#define DEFINE_ATTN(SUFFIX, A)                                                                                \
    extern "C" __global__ void paged_decode_##SUFFIX(const A::T* q, const A::T* k, const A::T* v,            \
                                                     const unsigned* slots, const unsigned* starts, A::T* out, \
                                                     int H, int KVH, int D, float scale) {                    \
        paged_decode<A>(q, k, v, slots, starts, out, H, KVH, D, scale);                                      \
    }

DEFINE_ATTN(f32, AF32)
DEFINE_ATTN(bf16, ABF16)
DEFINE_ATTN(f16, AF16)

// ---- prefill: tiled causal attention over the slot tables (flash-style, nothing O(kv_len)) ----
//
// One block per (query tile, head). The tile's BQ queries, then BK keys/values at a time, are
// staged in shared memory (rows padded by one 32-bit word against bank conflicts). Each of the
// 4 warps owns BQ/4 queries: lane j scores key j, the warp keeps an online softmax per query,
// and lanes split the head dimension (interleaved) for the output accumulation.
//   seqinfo [B, 4]: q_start, q_len, slot_start, kv_len   (the queries are the last q_len positions)
//   tiles   [n, 2]: sequence, first query of a 64-query tile (FP_SUB blocks of FP_BQ queries each)

#define FP_BQ 16
#define FP_TILE 64  // queries per entry of the tile table (shared with the tensor-core kernel)
#define FP_SUB (FP_TILE / FP_BQ)
#define FP_MAXD 256

template <class A, int BK>
__device__ void flash_prefill(const typename A::T* q, const typename A::T* kc, const typename A::T* vc,
                              const unsigned* slots, const unsigned* seqinfo, const unsigned* tiles,
                              typename A::T* out, int H, int KVH, int D, float scale) {
    typedef typename A::T T;
    const int PAD = 4 / sizeof(T);
    const int RS = D + PAD;
    __shared__ T sq[FP_BQ * (FP_MAXD + 2)];
    __shared__ T sk[BK * (FP_MAXD + 2)];
    __shared__ T sv[BK * (FP_MAXD + 2)];
    const int QPW = FP_BQ / ATT_WARPS;  // queries per warp

    int h = blockIdx.x % H, sub = (blockIdx.x / H) % FP_SUB, tile = blockIdx.x / H / FP_SUB;
    int kvh = h / (H / KVH);
    unsigned b = tiles[2 * tile], q0 = tiles[2 * tile + 1] + sub * FP_BQ;
    unsigned q_start = seqinfo[4 * b], q_len = seqinfo[4 * b + 1], s_start = seqinfo[4 * b + 2],
             kv_len = seqinfo[4 * b + 3];
    if (q0 >= q_len) return;  // whole block: past the end of a short last tile
    unsigned prefix = kv_len - q_len;
    int nq = min((int)(q_len - q0), FP_BQ);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int per = D / 32;

    for (int i = threadIdx.x; i < FP_BQ * D; i += blockDim.x) {
        int qi = i / D, d = i % D;
        if (qi < nq) sq[qi * RS + d] = q[((long)(q_start + q0 + qi) * H + h) * D + d];
    }

    float m[QPW], l[QPW], acc[QPW][MAX_PER_LANE];
    for (int a = 0; a < QPW; ++a) {
        m[a] = -1e30f;
        l[a] = 0.0f;
        for (int i = 0; i < MAX_PER_LANE; ++i) acc[a][i] = 0.0f;
    }
    unsigned kv_end = min(kv_len, prefix + q0 + (unsigned)nq);  // causal: last query sees kv_end-1
    for (unsigned k0 = 0; k0 < kv_end; k0 += BK) {
        int nk = min((int)(kv_end - k0), BK);
        __syncthreads();  // previous tile fully consumed (and, first time, the q tile written)
        for (int i = threadIdx.x; i < BK * D; i += blockDim.x) {
            int j = i / D, d = i % D;
            if (j < nk) {
                long row = ((long)slots[s_start + k0 + j] * KVH + kvh) * D + d;
                sk[j * RS + d] = kc[row];
                sv[j * RS + d] = vc[row];
            }
        }
        __syncthreads();
        for (int a = 0; a < QPW; ++a) {
            int qi = warp * QPW + a;
            if (qi >= nq) break;  // uniform per warp
            unsigned qpos = prefix + q0 + qi;
            float s = -1e30f;
            bool valid = lane < nk && k0 + lane <= qpos;
            if (valid) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += A::load(sq, qi * RS + d) * A::load(sk, lane * RS + d);
                s = dot * scale;
            }
            float tmax = s;
            for (int off = 16; off > 0; off >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, off));
            float m_new = fmaxf(m[a], tmax);
            float p = valid ? __expf(s - m_new) : 0.0f;
            float corr = __expf(m[a] - m_new);
            float psum = p;
            for (int off = 16; off > 0; off >>= 1) psum += __shfl_xor_sync(0xffffffff, psum, off);
            l[a] = l[a] * corr + psum;
            m[a] = m_new;
            for (int i = 0; i < per; ++i) acc[a][i] *= corr;
            for (int j = 0; j < nk; ++j) {
                float pj = __shfl_sync(0xffffffff, p, j);
                for (int i = 0; i < per; ++i) acc[a][i] += pj * A::load(sv, j * RS + lane + 32 * i);
            }
        }
    }
    for (int a = 0; a < QPW; ++a) {
        int qi = warp * QPW + a;
        if (qi >= nq) break;
        float inv = l[a] > 0.0f ? 1.0f / l[a] : 0.0f;
        long base = ((long)(q_start + q0 + qi) * H + h) * D;
        for (int i = 0; i < per; ++i) A::store(out, base + lane + 32 * i, acc[a][i] * inv);
    }
}

#define DEFINE_PREFILL(SUFFIX, A, BK)                                                                          \
    extern "C" __global__ void flash_prefill_##SUFFIX(const A::T* q, const A::T* k, const A::T* v,            \
                                                      const unsigned* slots, const unsigned* seqinfo,          \
                                                      const unsigned* tiles, A::T* out, int H, int KVH, int D, \
                                                      float scale) {                                          \
        flash_prefill<A, BK>(q, k, v, slots, seqinfo, tiles, out, H, KVH, D, scale);                           \
    }

DEFINE_PREFILL(f32, AF32, 8)
DEFINE_PREFILL(bf16, ABF16, 32)
DEFINE_PREFILL(f16, AF16, 32)
