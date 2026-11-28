// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// carbon-engine kernels for Inkling 2 (model/inkling.rs), compiled with NVRTC on their own.
//
// Weights as the model keeps them on the device:
//   Q8      codes i8 [rows, K] + f16 scales [rows, K / 32]       (dense projections, Q8_0 experts)
//   NVFP4   codes u8 [rows, K / 2] (byte i: value 2i low nibble, 2i + 1 high) + E4M3 scales
//           [rows, K / 16] + an f32 scale per expert             (the GGUF blocks, repacked)
// Activations are f32 unless a name says otherwise.

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

#define NEG_INF __int_as_float(0xff800000)
#define FULL 0xffffffffu

__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float(((u32)v) << 16); }

__device__ __forceinline__ u16 f2bf(float f) {
    u32 u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (u16)(u >> 16);
}

__device__ __forceinline__ float f16f(u32 h) {
    u32 s = (h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    if (e == 0) {
        float v = (float)m * 5.9604644775390625e-8f;
        return s ? -v : v;
    }
    if (e == 31) return __uint_as_float(s | 0x7f800000u | (m << 13));
    return __uint_as_float(s | ((e + 112) << 23) | (m << 13));
}

__device__ __forceinline__ float e4m3f(u32 b) {
    u32 e = (b >> 3) & 15, m = b & 7;
    float v = e ? __uint_as_float(((e + 120) << 23) | (m << 20)) : (float)m * 0.001953125f;
    return (b & 0x80) ? -v : v;
}

// E2M1 code -> f32 by bit assembly.
__device__ __forceinline__ float e2m1f(u32 c) {
    u32 e = (c >> 1) & 3, m = c & 1;
    u32 bits = e ? (((e + 126u) << 23) | (m << 22)) : (m ? (126u << 23) : 0u);
    return __int_as_float(((c & 8u) << 28) | bits);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets it.
__device__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) t += red[w];
    return t;
}

// Max over the block (as block_sum).
__device__ float block_max(float v, float* red) {
    v = warp_max(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = red[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); ++w) t = fmaxf(t, red[w]);
    return t;
}

// Quantize one value per lane of a warp holding 32 consecutive values (int8, scale amax / 127).
__device__ __forceinline__ void quant32(float v, signed char* q, float* d) {
    const float m = warp_max(fabsf(v));
    *q = (signed char)__float2int_rn(m > 0.0f ? v * (127.0f / m) : 0.0f);
    if ((threadIdx.x & 31) == 0) *d = m / 127.0f;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// ---------------------------------------------------------------------------------------------
// Q8 GEMV for up to 8 tokens: y[t, row] = x_sel(row)[t, :] . W[row, :]. Rows below r1 read x0,
// below r2 x1, the rest x2 (one launch for projections of different inputs). K % 16 == 0, x rows
// 16-byte aligned. grid ceil(N / 8), block 256 (a warp per row).

template <int T>
__device__ void q8_gemv(const signed char* W, const u16* S, const float* x0, const float* x1, const float* x2,
                        int r1, int r2, int ldx, float* y, int ldy, int N, int K) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= N) return;
    const float* x = row < r1 ? x0 : (row < r2 ? x1 : x2);
    const signed char* w = W + (long)row * K;
    const u16* s = S + (long)row * (K / 32);
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    for (int c = lane * 16; c < K; c += 512) {
        int4 q = *(const int4*)(w + c);
        float sc = f16f(s[c >> 5]);
        const signed char* qb = (const signed char*)&q;
        float wv[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) wv[j] = (float)qb[j];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4* xp = (const float4*)(x + (long)t * ldx + c);
            float4 a = xp[0], b = xp[1], cc = xp[2], d = xp[3];
            float part = wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b.x + wv[5] * b.y +
                         wv[6] * b.z + wv[7] * b.w + wv[8] * cc.x + wv[9] * cc.y + wv[10] * cc.z + wv[11] * cc.w +
                         wv[12] * d.x + wv[13] * d.y + wv[14] * d.z + wv[15] * d.w;
            acc[t] += part * sc;
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        float v = warp_sum(acc[t]);
        if (lane == 0) y[(long)t * ldy + row] = v;
    }
}

#define Q8_GEMV(T)                                                                                          \
    extern "C" __global__ void ink_q8_gemv_##T(const signed char* W, const u16* S, const float* x0,         \
                                               const float* x1, const float* x2, int r1, int r2, int ldx,   \
                                               float* y, int ldy, int N, int K) {                           \
        q8_gemv<T>(W, S, x0, x1, x2, r1, r2, ldx, y, ldy, N, K);                                            \
    }
Q8_GEMV(1)
Q8_GEMV(2)
Q8_GEMV(3)
Q8_GEMV(4)
Q8_GEMV(5)
Q8_GEMV(6)
Q8_GEMV(7)
Q8_GEMV(8)

// Q8 weights -> bf16 [rows, K] (row-major, so the scale of element i is S[i / 32]).
extern "C" __global__ void ink_q8_dequant(const signed char* W, const u16* S, u16* out, long n) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        out[i] = f2bf((float)W[i] * f16f(S[i >> 5]));
}

// ---------------------------------------------------------------------------------------------
// Causal short convolution (kernel 4) with a residual, per sequence from its recurrent state:
//   out[t, c] = (x[t, c] + sum_j w[c, j] x[t - 3 + j, c] + sum_lag dyn[t, head(c), lag] x[t - lag, c]) * inv
// The state holds each channel's last 3 inputs (oldest first) at [slot, j, off + c] of a
// [slots, 3, ctot] f32 pool; a fresh sequence starts from zeros. dyn (may be null) is
// [T, heads * 4]. seqs: per sequence q_start, q_len, state slot, fresh. grid (ceil(C / 128), B).
extern "C" __global__ void ink_conv(const float* x, int ldx, float* out, int ldo, const float* w, const float* dyn,
                                    int ldd, int heads, float* state, int ctot, int off, const u32* seqs, int C,
                                    float inv, const float* res, int ldr, int nsum) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const u32* si = seqs + blockIdx.y * 4;
    u32 q0 = si[0], ql = si[1], slot = si[2], fresh = si[3];
    float* st = state + (long)slot * 3 * ctot + off + c;
    float h0 = fresh ? 0.0f : st[0], h1 = fresh ? 0.0f : st[ctot], h2 = fresh ? 0.0f : st[2 * ctot];
    float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    int head = dyn ? c / (C / heads) : 0;
    for (u32 j = 0; j < ql; ++j) {
        long row = q0 + j;
        float xc;
        if (nsum > 0) {  // x holds nsum parts per row [T, nsum, C] (the MoE experts): their sum
            xc = 0.0f;
            for (int k = 0; k < nsum; ++k) xc += x[(row * nsum + k) * C + c];
        } else {
            xc = x[row * ldx + c];
        }
        float o = w0 * h0 + w1 * h1 + w2 * h2 + w3 * xc;
        if (dyn) {
            const float* d = dyn + row * ldd + head * 4;
            o += d[0] * xc + d[1] * h2 + d[2] * h1 + d[3] * h0;
        }
        out[row * ldo + c] = (xc + o) * inv + (res ? res[row * ldr + c] : 0.0f);
        h0 = h1;
        h1 = h2;
        h2 = xc;
    }
    st[0] = h0;
    st[ctot] = h1;
    st[2 * ctot] = h2;
}

// ---------------------------------------------------------------------------------------------
// Per (token, group) row of D values: y = x * scale (optional, [G * D]); then mode 0 clips the
// row to L2 norm `param`, mode 1 RMS-normalizes it (eps `param`); then, with alpha > 0, scales by
// the log position factor 1 + alpha * ln(max(1, (pos + 1) / floor)).
// grid T * G, block 128 (D <= 128).
extern "C" __global__ void ink_rownorm(const float* x, int ldx, float* y, int ldy, int G, int D, const float* scale,
                                       int mode, float param, const u32* pos, float alpha, float floor_) {
    __shared__ float red[4];
    int t = blockIdx.x / G, g = blockIdx.x % G, d = threadIdx.x;
    float v = 0.0f;
    if (d < D) {
        v = x[(long)t * ldx + g * D + d];
        if (scale) v *= scale[g * D + d];
    }
    float ss = block_sum(v * v, red);
    float f;
    if (mode == 0) {
        float nrm = sqrtf(ss);
        f = param / fmaxf(nrm, param);
    } else {
        f = rsqrtf(ss / (float)D + param);
    }
    if (alpha > 0.0f) f *= 1.0f + alpha * logf(fmaxf(1.0f, ((float)pos[t] + 1.0f) / floor_));
    if (d < D) y[(long)t * ldy + g * D + d] = v * f;
}

// RMS norm of [T, H] rows with an optional weight. grid T, block 256.
extern "C" __global__ void ink_rms(const float* x, int ldx, float* y, int ldy, const float* w, int H, float eps) {
    __shared__ float red[8];
    const float* xr = x + (long)blockIdx.x * ldx;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, red);
    float f = rsqrtf(ss / (float)H + eps);
    float* yr = y + (long)blockIdx.x * ldy;
    for (int i = threadIdx.x; i < H; i += blockDim.x) yr[i] = xr[i] * f * (w ? w[i] : 1.0f);
}

// ---------------------------------------------------------------------------------------------
// Attention with a relative-position bias and a per-query sink, no rope.
//
// Row t (a new token at position p of its sequence) attends to positions [lo, p], lo = 0 or
// p - window + 1 (sliding window). Positions before the batch's first new one of the sequence
// (s0 = kv_len - q_len) are read from the cache: the paged pool through the sequence's slot
// table, or the sequence's ring (slot state * ring + pos % ring); later ones from knew / vnew.
//   logit = scale * q . k + bias,  bias = relb[p - pos] . r  (0 past the table)
//   out = sum exp(logit) v / (exp(sink) + sum exp(logit))
// q [T, NH * 128], r [T, NH * nrel], sink [T, NH], knew / vnew [T, NKV * 128] (ldkv).
// FMT 0: bf16 cache rows [*, NKV * 128]; FMT 1: int8 rows of NKV * (128 + 4) bytes (codes, f32 scale).
// seqs: per sequence q_start, q_len, kv_len, slot offset, state slot; rowseq[t]: its sequence.
// grid (T * NKV, nsplit), block 128: the REP query heads of one KV head over a share of the keys;
// with nsplit > 1 the shares go to `pml` ([T, NH, nsplit] (m, l) pairs) and `pacc` ([T, NH,
// nsplit, 128] sums) for ink_attn_combine.

#define HD 128
#define AW 4

template <int REP, int FMT>
__device__ void attn(const float* q, int ldq, const float* r, int ldr, const float* sink, int lds, const float* knew,
                     const float* vnew, int ldkv, const void* kold, const void* vold, const u32* slots, int ring,
                     const u32* seqs, const u32* rowseq, const float* relb, int extent, int nrel, int nkv, int window,
                     float scale, float* out, int ldo, float* pml, float* pacc, int nsplit, int per_split,
                     const float* gate, int ldg, signed char* oq, float* od) {
    __shared__ float qs[REP][HD];
    __shared__ float rs[REP][16];
    __shared__ float wm[AW][REP], wl[AW][REP];
    __shared__ float wacc[AW][REP][HD];
    const int t = blockIdx.x / nkv, g = blockIdx.x % nkv, split = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const u32* si = seqs + rowseq[t] * 5;
    const int q_start = si[0], q_len = si[1], kv_len = si[2], slot_off = si[3], state = si[4];
    const int s0 = kv_len - q_len;
    const int p = s0 + (t - q_start);
    const int lo = window ? max(0, p - window + 1) : 0;
    const int per = per_split > 0 ? per_split : (p - lo + nsplit) / nsplit;
    const int k0 = lo + split * per, k1 = min(p + 1, k0 + per);
    const int nh = nkv * REP;
    if (k0 >= k1) {  // past this row's keys: an empty share
        if (threadIdx.x < REP) pml[(((long)t * nh + g * REP + threadIdx.x) * nsplit + split) * 2] = NEG_INF;
        return;
    }
    for (int i = threadIdx.x; i < REP * HD; i += blockDim.x) {
        int h = i / HD, d = i % HD;
        qs[h][d] = q[(long)t * ldq + (g * REP + h) * HD + d] * scale;
    }
    for (int i = threadIdx.x; i < REP * nrel; i += blockDim.x) {
        int h = i / nrel, j = i % nrel;
        rs[h][j] = r[(long)t * ldr + (g * REP + h) * nrel + j];
    }
    __syncthreads();
    float m[REP], l[REP], acc[REP][4];
#pragma unroll
    for (int h = 0; h < REP; ++h) {
        m[h] = NEG_INF;
        l[h] = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[h][e] = 0.0f;
    }
    for (int kb = k0 + warp * 32; kb < k1; kb += AW * 32) {
        const int pos = kb + lane;
        const bool valid = pos < k1;
        u64 vaddr = 0;
        int kind = 0;
        float logit[REP];
#pragma unroll
        for (int h = 0; h < REP; ++h) logit[h] = NEG_INF;
        if (valid) {
            float dot[REP];
#pragma unroll
            for (int h = 0; h < REP; ++h) dot[h] = 0.0f;
            if (pos >= s0) {
                long row = q_start + (pos - s0);
                const float4* k4 = (const float4*)(knew + row * ldkv + g * HD);
                vaddr = (u64)(vnew + row * ldkv + g * HD);
#pragma unroll 8
                for (int i = 0; i < HD / 4; ++i) {
                    float4 kv = k4[i];
#pragma unroll
                    for (int h = 0; h < REP; ++h)
                        dot[h] += kv.x * qs[h][4 * i] + kv.y * qs[h][4 * i + 1] + kv.z * qs[h][4 * i + 2] +
                                  kv.w * qs[h][4 * i + 3];
                }
            } else {
                long slot = ring ? (long)state * ring + (pos % ring) : (long)slots[slot_off + pos];
                kind = 1;
                if (FMT == 0) {
                    const uint4* k8 = (const uint4*)((const u16*)kold + slot * nkv * HD + g * HD);
                    vaddr = (u64)((const u16*)vold + slot * nkv * HD + g * HD);
#pragma unroll 4
                    for (int i = 0; i < HD / 8; ++i) {
                        uint4 u = k8[i];
                        float kv[8] = {bf2f(u.x & 0xffff), bf2f(u.x >> 16), bf2f(u.y & 0xffff), bf2f(u.y >> 16),
                                       bf2f(u.z & 0xffff), bf2f(u.z >> 16), bf2f(u.w & 0xffff), bf2f(u.w >> 16)};
#pragma unroll
                        for (int h = 0; h < REP; ++h) {
                            float s = 0.0f;
#pragma unroll
                            for (int j = 0; j < 8; ++j) s += kv[j] * qs[h][8 * i + j];
                            dot[h] += s;
                        }
                    }
                } else {
                    const u8* kr = (const u8*)kold + (slot * nkv + g) * (HD + 4);
                    vaddr = (u64)((const u8*)vold + (slot * nkv + g) * (HD + 4));
                    // rows are 132 bytes apart: 4-byte loads only
                    const u32* kq = (const u32*)kr;
                    float ks = *(const float*)(kr + HD);
#pragma unroll 4
                    for (int i = 0; i < HD / 4; ++i) {
                        const u32 u = kq[i];
                        const float b0 = (float)(signed char)(u & 0xff), b1 = (float)(signed char)((u >> 8) & 0xff);
                        const float b2 = (float)(signed char)((u >> 16) & 0xff), b3 = (float)(signed char)(u >> 24);
#pragma unroll
                        for (int h = 0; h < REP; ++h)
                            dot[h] += b0 * qs[h][4 * i] + b1 * qs[h][4 * i + 1] + b2 * qs[h][4 * i + 2] + b3 * qs[h][4 * i + 3];
                    }
#pragma unroll
                    for (int h = 0; h < REP; ++h) dot[h] *= ks;
                }
            }
            const int dist = p - pos;
            float b[REP];
#pragma unroll
            for (int h = 0; h < REP; ++h) b[h] = 0.0f;
            if (dist < extent) {
                if (nrel == 16) {
                    const float4* rb = (const float4*)(relb + (long)dist * 16);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float4 v = rb[j];
#pragma unroll
                        for (int h = 0; h < REP; ++h)
                            b[h] += v.x * rs[h][4 * j] + v.y * rs[h][4 * j + 1] + v.z * rs[h][4 * j + 2] + v.w * rs[h][4 * j + 3];
                    }
                } else {
                    const float* rb = relb + (long)dist * nrel;
                    for (int j = 0; j < nrel; ++j)
#pragma unroll
                        for (int h = 0; h < REP; ++h) b[h] += rb[j] * rs[h][j];
                }
            }
#pragma unroll
            for (int h = 0; h < REP; ++h) logit[h] = dot[h] + b[h];
        }
        float pl[REP];
#pragma unroll
        for (int h = 0; h < REP; ++h) {
            float mx = warp_max(logit[h]);
            float mn = fmaxf(m[h], mx);
            float corr = m[h] == NEG_INF ? 0.0f : __expf(m[h] - mn);
            float pe = valid ? __expf(logit[h] - mn) : 0.0f;
            l[h] = l[h] * corr + warp_sum(pe);
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[h][e] *= corr;
            m[h] = mn;
            pl[h] = pe;
        }
        const int nj = min(32, k1 - kb);
        for (int j = 0; j < nj; ++j) {
            u64 va = __shfl_sync(FULL, vaddr, j);
            int kd = __shfl_sync(FULL, kind, j);
            float v[4];
            if (kd == 0) {
                float4 vv = ((const float4*)va)[lane];
                v[0] = vv.x, v[1] = vv.y, v[2] = vv.z, v[3] = vv.w;
            } else if (FMT == 0) {
                uint2 u = ((const uint2*)va)[lane];
                v[0] = bf2f(u.x & 0xffff), v[1] = bf2f(u.x >> 16), v[2] = bf2f(u.y & 0xffff), v[3] = bf2f(u.y >> 16);
            } else {
                u32 u = ((const u32*)va)[lane];
                float vs = *(const float*)((const u8*)va + HD);
                v[0] = (float)(signed char)(u & 0xff) * vs, v[1] = (float)(signed char)((u >> 8) & 0xff) * vs;
                v[2] = (float)(signed char)((u >> 16) & 0xff) * vs, v[3] = (float)(signed char)(u >> 24) * vs;
            }
#pragma unroll
            for (int h = 0; h < REP; ++h) {
                float pj = __shfl_sync(FULL, pl[h], j);
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[h][e] += pj * v[e];
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int h = 0; h < REP; ++h) wm[warp][h] = m[h], wl[warp][h] = l[h];
    }
#pragma unroll
    for (int h = 0; h < REP; ++h)
#pragma unroll
        for (int e = 0; e < 4; ++e) wacc[warp][h][lane * 4 + e] = acc[h][e];
    __syncthreads();
    for (int i = threadIdx.x; i < REP * HD; i += blockDim.x) {
        const int h = i / HD, d = i % HD, hh = g * REP + h;
        float M = NEG_INF;
        for (int w = 0; w < AW; ++w) M = fmaxf(M, wm[w][h]);
        float L = 0.0f, A = 0.0f;
        for (int w = 0; w < AW; ++w) {
            if (wm[w][h] == NEG_INF) continue;
            float c = __expf(wm[w][h] - M);
            L += wl[w][h] * c;
            A += wacc[w][h][d] * c;
        }
        if (nsplit == 1) {
            float sk = sink[(long)t * lds + hh];
            float mf = fmaxf(M, sk);
            float cm = M == NEG_INF ? 0.0f : __expf(M - mf);
            float o = A * cm / (L * cm + __expf(sk - mf));
            if (gate) o *= sigmoidf_(gate[(long)t * ldg + hh * HD + d]);
            out[(long)t * ldo + hh * HD + d] = o;
            if (oq) quant32(o, oq + (long)t * ldo + hh * HD + d, od + ((long)t * ldo + hh * HD + d) / 32);
        } else {
            long base = ((long)t * nh + hh) * nsplit + split;
            if (d == 0) {
                pml[base * 2] = M;
                pml[base * 2 + 1] = L;
            }
            pacc[base * HD + d] = A;
        }
    }
}

#define ATTN(REP, FMT)                                                                                         \
    extern "C" __global__ void ink_attn_##REP##_##FMT(                                                        \
        const float* q, int ldq, const float* r, int ldr, const float* sink, int lds, const float* knew,       \
        const float* vnew, int ldkv, const void* kold, const void* vold, const u32* slots, int ring,           \
        const u32* seqs, const u32* rowseq, const float* relb, int extent, int nrel, int nkv, int window,       \
        float scale, float* out, int ldo, float* pml, float* pacc, int nsplit, int per_split,                  \
        const float* gate, int ldg, signed char* oq, float* od) {                                              \
        attn<REP, FMT>(q, ldq, r, ldr, sink, lds, knew, vnew, ldkv, kold, vold, slots, ring, seqs, rowseq,     \
                       relb, extent, nrel, nkv, window, scale, out, ldo, pml, pacc, nsplit, per_split, gate, ldg, \
                       oq, od);                                                                                \
    }
ATTN(1, 0)
ATTN(1, 1)
ATTN(2, 0)
ATTN(2, 1)
ATTN(3, 0)
ATTN(3, 1)
ATTN(4, 0)
ATTN(4, 1)

// Join the key shares of ink_attn and add the sink. grid T * NH, block 128.
extern "C" __global__ void ink_attn_combine(const float* pml, const float* pacc, const float* sink, int lds,
                                            float* out, int ldo, int nh, int nsplit, const float* gate, int ldg,
                                            signed char* oq, float* od) {
    const int t = blockIdx.x / nh, h = blockIdx.x % nh, d = threadIdx.x;
    const long base = (long)blockIdx.x * nsplit;
    float M = NEG_INF;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, pml[(base + s) * 2]);
    float L = 0.0f, A = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        float ms = pml[(base + s) * 2];
        if (ms == NEG_INF) continue;
        float c = __expf(ms - M);
        L += pml[(base + s) * 2 + 1] * c;
        A += pacc[(base + s) * HD + d] * c;
    }
    float sk = sink[(long)t * lds + h];
    float mf = fmaxf(M, sk);
    float cm = M == NEG_INF ? 0.0f : __expf(M - mf);
    float o = A * cm / (L * cm + __expf(sk - mf));
    if (gate) o *= sigmoidf_(gate[(long)t * ldg + h * HD + d]);
    out[(long)t * ldo + h * HD + d] = o;
    if (oq) quant32(o, oq + (long)t * ldo + h * HD + d, od + ((long)t * ldo + h * HD + d) / 32);
}

// ---------------------------------------------------------------------------------------------
// Cache writes.

// The last `ring` positions of each sequence's new rows into its ring: x [T, W] (ldx) -> bf16
// rows [state * ring + pos % ring, W]. grid (T, ceil(W / 256)).
extern "C" __global__ void ink_ring_write(const float* x, int ldx, u16* dst, int W, int ring, const u32* seqs,
                                          const u32* rowseq) {
    const int t = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= W) return;
    const u32* si = seqs + rowseq[t] * 5;
    const int q_start = si[0], q_len = si[1], kv_len = si[2], state = si[4];
    const int pos = kv_len - q_len + (t - q_start);
    if (pos < kv_len - ring) return;
    dst[((long)state * ring + pos % ring) * W + c] = f2bf(x[(long)t * ldx + c]);
}

// New rows into a bf16 pool at their slots: x [T, W] -> pool [slot[t], W]. grid (T, ceil(W / 256)).
extern "C" __global__ void ink_pool_write_bf16(const float* x, int ldx, u16* pool, const u32* slot, int W) {
    const int t = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c < W) pool[(long)slot[t] * W + c] = f2bf(x[(long)t * ldx + c]);
}

// New rows into an int8 pool (the `q8` layout of kvcache: per KV head 128 codes then the f32
// scale amax / 127). grid T * NKV, block 128.
extern "C" __global__ void ink_pool_write_q8(const float* x, int ldx, signed char* pool, const u32* slot, int nkv) {
    __shared__ float red[4];
    const int t = blockIdx.x / nkv, g = blockIdx.x % nkv, d = threadIdx.x;
    float v = x[(long)t * ldx + g * HD + d];
    float m = warp_max(fabsf(v));
    if ((d & 31) == 0) red[d >> 5] = m;
    __syncthreads();
    float mx = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    signed char* row = pool + ((long)slot[t] * nkv + g) * (HD + 4);
    row[d] = (signed char)__float2int_rn(mx > 0.0f ? v * (127.0f / mx) : 0.0f);
    if (d == 0) *(float*)(row + HD) = mx / 127.0f;
}

// ---------------------------------------------------------------------------------------------
// MoE.

// Sigmoid routing: top-k of sigmoid(logit) + bias, weights sigmoid(logit) normalized over the
// chosen experts and times wscale. logits [T, E] (E <= 256); ids / wts [T, K]. grid T, block 32.
extern "C" __global__ void ink_route(const float* logits, const float* bias, u32* ids, float* wts, int E, int K,
                                     float wscale) {
    const int t = blockIdx.x, lane = threadIdx.x;
    float prob[8], sel[8];
    const int per = (E + 31) / 32;
    for (int i = 0; i < 8; ++i) {
        int e = lane + 32 * i;
        if (i < per && e < E) {
            prob[i] = sigmoidf_(logits[(long)t * E + e]);
            sel[i] = prob[i] + (bias ? bias[e] : 0.0f);
        } else {
            prob[i] = 0.0f;
            sel[i] = NEG_INF;
        }
    }
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float best = NEG_INF;
        int bi = 0x7fffffff, bslot = -1;
        for (int i = 0; i < 8; ++i)
            if (sel[i] > best) best = sel[i], bi = lane + 32 * i, bslot = i;
        float bv = best;
        int bidx = bi;
        for (int o = 16; o; o >>= 1) {
            float ov = __shfl_xor_sync(FULL, bv, o);
            int oi = __shfl_xor_sync(FULL, bidx, o);
            if (ov > bv || (ov == bv && oi < bidx)) bv = ov, bidx = oi;
        }
        float pr = 0.0f;
        if (bidx == bi && bslot >= 0) {
            pr = prob[bslot];
            sel[bslot] = NEG_INF;
        }
        pr = warp_sum(pr);
        if (lane == 0) {
            ids[(long)t * K + k] = (u32)bidx;
            wts[(long)t * K + k] = pr;
        }
        sum += pr;
    }
    __syncwarp();
    if (lane < K) wts[(long)t * K + lane] = wts[(long)t * K + lane] / (sum + 1e-20f) * wscale;
}

// SiTU: beta tanh(g / beta) sigmoid(g) * lb tanh(u / lb), with g, u already scaled.
__device__ __forceinline__ float situ(float g, float u, float beta, float lb) {
    return beta * tanhf(g / beta) * sigmoidf_(g) * (lb * tanhf(u / lb));
}

// ---- int8 activations and dp4a dot products (decode-sized GEMVs) ----
//
// Activations are quantized per 32 values: int8 codes [T, K] and f32 scales amax / 127
// [T, K / 32]. Q8 weights dot them with dp4a directly; NVFP4 codes become int8 (the E2M1
// values doubled) through a byte-permute table, then dp4a.

// x [T, K] f32 (row stride ldx) -> codes, scales. A warp per 32 values: grid ceil(T K / 256), block 256.
extern "C" __global__ void ink_quant_x(const float* x, int ldx, signed char* q, float* d, int T, int K) {
    const long blk = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const long nblk = (long)T * (K / 32);
    if (blk >= nblk) return;
    const long t = blk / (K / 32), c = (blk % (K / 32)) * 32 + lane;
    const float v = x[t * ldx + c];
    const float m = warp_max(fabsf(v));
    q[t * K + c] = (signed char)__float2int_rn(m > 0.0f ? v * (127.0f / m) : 0.0f);
    if (lane == 0) d[blk] = m / 127.0f;
}

// Doubled E2M1 values of the 8 nibbles of q: .x the low nibbles (bytes 0..3), .y the high.
__device__ __forceinline__ int2 e2m1x2_int8(u32 q) {
    const u32 t0 = 0x03020100u, t1 = 0x0c080604u, t2 = 0xfdfeff00u, t3 = 0xf4f8fafcu;
    const u32 sel = 0x32103210u | ((q & 0x88888888u) >> 1);
    u32 tmp[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const u32 sh = 16 * i;
        const u32 lo = __byte_perm(t0, t1, q >> sh);
        const u32 hi = __byte_perm(t2, t3, q >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2((int)__byte_perm(tmp[0], tmp[1], 0x6420), (int)__byte_perm(tmp[0], tmp[1], 0x7531));
}

// Dot of a weight row (K values) with int8 activations (xq, xd). FMT 0: Q8 (codes [K], f16
// scales [K / 32]); FMT 1: NVFP4 in the GGUF nibble order (per 16 values 8 bytes, byte j holding
// values j and j + 8; an E4M3 scale per 16). Result in every lane.
template <int FMT>
__device__ __forceinline__ float qdot(const u8* codes, const u8* scales, int K, const signed char* xq, const float* xd) {
    const int lane = threadIdx.x & 31;
    float acc = 0.0f;
    if (FMT == 0) {
        const u16* s = (const u16*)scales;
        for (int c = lane * 16; c < K; c += 512) {
            const int4 w = *(const int4*)(codes + c);
            const int4 a = *(const int4*)(xq + c);
            int si = __dp4a(w.x, a.x, 0);
            si = __dp4a(w.y, a.y, si);
            si = __dp4a(w.z, a.z, si);
            si = __dp4a(w.w, a.w, si);
            acc += (float)si * f16f(s[c >> 5]) * xd[c >> 5];
        }
    } else {
        for (int gi = lane; gi < K / 16; gi += 32) {
            const uint2 u = *(const uint2*)(codes + gi * 8);
            const int4 a = *(const int4*)(xq + gi * 16);
            const int2 v0 = e2m1x2_int8(u.x), v1 = e2m1x2_int8(u.y);
            int si = __dp4a(v0.x, a.x, 0);
            si = __dp4a(v1.x, a.y, si);
            si = __dp4a(v0.y, a.z, si);
            si = __dp4a(v1.y, a.w, si);
            acc += (float)si * (0.5f * e4m3f(scales[gi])) * xd[gi >> 1];
        }
    }
    return warp_sum(acc);
}

// Q8 GEMV on int8 activations for up to 8 tokens, rows below r1 from input 0, below r2 from
// input 1, the rest from input 2; then nf f32 rows (Wf [nf, K]) on the f32 inputs f0 / f1 / f2
// (rows below fr1 / fr2 likewise) into yf. Two rows per warp. grid ceil((N + nf) / 16), block 256.
template <int T>
__device__ void q8_gemv_q(const signed char* W, const u16* S, const signed char* q0, const float* d0,
                          const signed char* q1, const float* d1, const signed char* q2, const float* d2, int r1,
                          int r2, float* y, int ldy, int N, int K, const float* Wf, int nf, const float* f0,
                          const float* f1, const float* f2, int fr1, int fr2, float* yf, int ldyf) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * 16 + warp * 2;
    if (row0 >= N) {
        // f32 tail
        for (int rr = 0; rr < 2; ++rr) {
            const int fr = row0 + rr - N;
            if (fr < 0 || fr >= nf) continue;
            const float* x = fr < fr1 ? f0 : (fr < fr2 ? f1 : f2);
            const float* w = Wf + (long)fr * K;
            float acc[T];
#pragma unroll
            for (int t = 0; t < T; ++t) acc[t] = 0.0f;
            for (int c = lane * 4; c < K; c += 128) {
                const float4 wv = *(const float4*)(w + c);
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float4 a = *(const float4*)(x + (long)t * K + c);
                    acc[t] += wv.x * a.x + wv.y * a.y + wv.z * a.z + wv.w * a.w;
                }
            }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const float v = warp_sum(acc[t]);
                if (lane == 0) yf[(long)t * ldyf + fr] = v;
            }
        }
        return;
    }
    const bool two = row0 + 1 < N;
    const signed char* xq0 = row0 < r1 ? q0 : (row0 < r2 ? q1 : q2);
    const float* xd0 = row0 < r1 ? d0 : (row0 < r2 ? d1 : d2);
    const int row1 = two ? row0 + 1 : row0;
    const signed char* xq1 = row1 < r1 ? q0 : (row1 < r2 ? q1 : q2);
    const float* xd1 = row1 < r1 ? d0 : (row1 < r2 ? d1 : d2);
    const signed char* wa = W + (long)row0 * K;
    const signed char* wb = W + (long)row1 * K;
    const u16* sa = S + (long)row0 * (K / 32);
    const u16* sb = S + (long)row1 * (K / 32);
    float acc0[T], acc1[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc0[t] = 0.0f, acc1[t] = 0.0f;
    for (int c = lane * 16; c < K; c += 512) {
        const int4 va = *(const int4*)(wa + c);
        const int4 vb = *(const int4*)(wb + c);
        const float da = f16f(sa[c >> 5]), db = f16f(sb[c >> 5]);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const int4 a = *(const int4*)(xq0 + (long)t * K + c);
            int si = __dp4a(va.x, a.x, 0);
            si = __dp4a(va.y, a.y, si);
            si = __dp4a(va.z, a.z, si);
            si = __dp4a(va.w, a.w, si);
            acc0[t] += (float)si * da * xd0[(long)t * (K / 32) + (c >> 5)];
            const int4 b = xq1 == xq0 ? a : *(const int4*)(xq1 + (long)t * K + c);
            int sj = __dp4a(vb.x, b.x, 0);
            sj = __dp4a(vb.y, b.y, sj);
            sj = __dp4a(vb.z, b.z, sj);
            sj = __dp4a(vb.w, b.w, sj);
            acc1[t] += (float)sj * db * xd1[(long)t * (K / 32) + (c >> 5)];
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v0 = warp_sum(acc0[t]), v1 = warp_sum(acc1[t]);
        if (lane == 0) {
            y[(long)t * ldy + row0] = v0;
            if (two) y[(long)t * ldy + row1] = v1;
        }
    }
}

#define Q8_GEMV_Q(T)                                                                                          \
    extern "C" __global__ void ink_q8_gemv_q_##T(const signed char* W, const u16* S, const signed char* q0,   \
                                                 const float* d0, const signed char* q1, const float* d1,     \
                                                 const signed char* q2, const float* d2, int r1, int r2,      \
                                                 float* y, int ldy, int N, int K, const float* Wf, int nf,    \
                                                 const float* f0, const float* f1, const float* f2, int fr1,  \
                                                 int fr2, float* yf, int ldyf) {                               \
        q8_gemv_q<T>(W, S, q0, d0, q1, d1, q2, d2, r1, r2, y, ldy, N, K, Wf, nf, f0, f1, f2, fr1, fr2, yf, ldyf); \
    }
Q8_GEMV_Q(1)
Q8_GEMV_Q(2)
Q8_GEMV_Q(3)
Q8_GEMV_Q(4)
Q8_GEMV_Q(5)
Q8_GEMV_Q(6)
Q8_GEMV_Q(7)
Q8_GEMV_Q(8)

template <int FMT>
__device__ __forceinline__ long code_bytes(int K) { return FMT == 0 ? K : K / 2; }
template <int FMT>
__device__ __forceinline__ long scale_bytes(int K) { return FMT == 0 ? (K / 32) * 2 : K / 16; }

// Routed pairs p = (token p / K, expert ids[p]): act[p, n] = situ(gate_e[n] . x, up_e[n] . x)
// (gate_up_e [2I, H], gate rows first, times gs[e]), stored as int8 per 32 (aq [P, I], ad
// [P, I / 32]) for the down projection. A block makes 32 consecutive n (4 per warp).
// grid (I / 32, P), block 256. I % 32 == 0.
template <int FMT>
__device__ void moe_gate_up_q(const u8* gc, const u8* gsc, const float* gs, const signed char* xq, const float* xd,
                              const u32* ids, signed char* aq, float* ad, int K, int I, int H, float beta, float lb) {
    __shared__ float vals[32];
    const int p = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const u32 e = ids[p];
    const long rows = 2L * I;
    const u8* cbase = gc + (long)e * rows * code_bytes<FMT>(H);
    const u8* sbase = gsc + (long)e * rows * scale_bytes<FMT>(H);
    const signed char* xt = xq + (long)(p / K) * H;
    const float* dt = xd + (long)(p / K) * (H / 32);
    const float sc = gs[e];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int nl = warp * 4 + j, n = blockIdx.x * 32 + nl;
        const float g = qdot<FMT>(cbase + n * code_bytes<FMT>(H), sbase + n * scale_bytes<FMT>(H), H, xt, dt);
        const float u = qdot<FMT>(cbase + (I + n) * code_bytes<FMT>(H), sbase + (I + n) * scale_bytes<FMT>(H), H, xt, dt);
        if (lane == 0) vals[nl] = situ(g * sc, u * sc, beta, lb);
    }
    __syncthreads();
    if (warp == 0) {
        const float v = vals[lane];
        const float m = warp_max(fabsf(v));
        const long base = (long)p * I + blockIdx.x * 32;
        aq[base + lane] = (signed char)__float2int_rn(m > 0.0f ? v * (127.0f / m) : 0.0f);
        if (lane == 0) ad[base / 32] = m / 127.0f;
    }
}

// part[t, k, h] = wts[t, k] ds[e] down_e[h] . act[t K + k]: one expert of one token per block
// (blockIdx.z = k), 32 rows h (4 per warp); ink_moe_sum adds the K parts in order (no atomics:
// the step is deterministic). grid (H / 32, T, K), block 256.
template <int FMT>
__device__ void moe_down_q(const u8* dc, const u8* dsc, const float* ds, const signed char* aq, const float* ad,
                           const u32* ids, const float* wts, float* y, int ldy, int K, int I, int H) {
    const int t = blockIdx.y, k = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const long pr = (long)t * K + k;
    const u32 e = ids[pr];
    const float wk = wts[pr] * ds[e];
    const signed char* xq = aq + pr * I;
    const float* xd = ad + pr * (I / 32);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int h = blockIdx.x * 32 + warp * 4 + j;
        if (h >= H) break;
        const u8* c = dc + ((long)e * H + h) * code_bytes<FMT>(I);
        const u8* s = dsc + ((long)e * H + h) * scale_bytes<FMT>(I);
        const float d = qdot<FMT>(c, s, I, xq, xd);
        if (lane == 0) y[pr * ldy + h] = wk * d;
    }
}

#define MOE(FMT)                                                                                                 \
    extern "C" __global__ void ink_moe_gate_up_##FMT(const u8* gc, const u8* gsc, const float* gs,               \
                                                      const signed char* xq, const float* xd, const u32* ids,     \
                                                      signed char* aq, float* ad, int K, int I, int H, float beta, \
                                                      float lb) {                                                 \
        moe_gate_up_q<FMT>(gc, gsc, gs, xq, xd, ids, aq, ad, K, I, H, beta, lb);                                  \
    }                                                                                                             \
    extern "C" __global__ void ink_moe_down_##FMT(const u8* dc, const u8* dsc, const float* ds,                  \
                                                   const signed char* aq, const float* ad, const u32* ids,        \
                                                   const float* wts, float* y, int ldy, int K, int I, int H) {    \
        moe_down_q<FMT>(dc, dsc, ds, aq, ad, ids, wts, y, ldy, K, I, H);                                          \
    }
MOE(0)
MOE(1)

// y[t, h] = sum_k part[t, k, h]. grid-stride over T * H.
extern "C" __global__ void ink_moe_sum(const float* part, float* y, int T, int K, int H) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < (long)T * H; i += (long)gridDim.x * blockDim.x) {
        const long t = i / H, h = i % H;
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += part[(t * K + k) * H + h];
        y[i] = s;
    }
}

// Expert e's weights [rows, K] -> bf16 (prefill: then a dense GEMM).
extern "C" __global__ void ink_expert_dequant_0(const u8* codes, const u8* scales, int e, long rows, int K,
                                                float g, u16* out) {
    const signed char* w = (const signed char*)codes + (long)e * rows * K;
    const u16* s = (const u16*)scales + (long)e * rows * (K / 32);
    const long n = rows * K;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        out[i] = f2bf((float)w[i] * f16f(s[i >> 5]) * g);
}

extern "C" __global__ void ink_expert_dequant_1(const u8* codes, const u8* scales, int e, long rows, int K,
                                                float g, u16* out) {
    const u8* w = codes + (long)e * rows * (K / 2);
    const u8* s = scales + (long)e * rows * (K / 16);
    const long n = rows * K;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const long grp = i >> 4;
        const int v = i & 15;
        const u32 c = w[grp * 8 + (v & 7)];
        out[i] = f2bf(e2m1f(v < 8 ? (c & 15) : (c >> 4)) * e4m3f(s[grp]) * g);
    }
}

// gu [n, 2I] (gate | up, already scaled) -> act [n, I]. grid-stride.
extern "C" __global__ void ink_situ_rows(const float* gu, float* act, long n, int I, float beta, float lb) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n * I; i += (long)gridDim.x * blockDim.x) {
        long r = i / I;
        int c = i % I;
        act[i] = situ(gu[r * 2 * I + c], gu[r * 2 * I + I + c], beta, lb);
    }
}

// ---------------------------------------------------------------------------------------------
// Block attention residuals and loop controllers (decode-sized batches; any T works).

// out[q, t, :] = sum_s softmax_s(st_s[t] . query[q] / sqrt(H)) st_s[t] over the npre states
// st_0..st_{npre-1} ([T, H], RMS-normalized) and RMS-normalized cur (also stored to curn when
// not null). With a weight w_q the output row q is RMS-normalized and scaled by it (eps2), and
// with oq / od the rows are also stored as int8 per 32 (for the dp4a GEMVs).
// nq <= 4, npre <= 7, H <= 4096, H % 128 == 0. grid T, block 1024, H floats of shared memory.
struct Mix7 { u64 s[7]; u64 w[4]; u64 oq[4]; u64 od[4]; };

__device__ __forceinline__ const float* mix_state(const Mix7& m, int s) {
    switch (s) {
        case 0: return (const float*)m.s[0];
        case 1: return (const float*)m.s[1];
        case 2: return (const float*)m.s[2];
        case 3: return (const float*)m.s[3];
        case 4: return (const float*)m.s[4];
        case 5: return (const float*)m.s[5];
        default: return (const float*)m.s[6];
    }
}

extern "C" __global__ void ink_mix(Mix7 m, int npre, const float* cur, float* curn, const float* query, int nq,
                                   float* out, int T, int H, float eps, float eps2) {
    extern __shared__ float cn[];
    __shared__ float red[32];
    __shared__ float sc[4][8];
    __shared__ float nrm[4];
    const int t = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // blockIdx.y picks a group of 4 queries (more than 4 queries: no norms or int8 outputs)
    query += (long)blockIdx.y * 4 * H;
    out += (long)blockIdx.y * 4 * T * H;
    nq = min(4, nq - 4 * (int)blockIdx.y);
    if (blockIdx.y > 0) curn = nullptr;
    const float* cr = cur + (long)t * H;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float v = cr[i];
        cn[i] = v;
        ss += v * v;
    }
    ss = block_sum(ss, red);
    const float f = rsqrtf(ss / (float)H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        cn[i] *= f;
        if (curn) curn[(long)t * H + i] = cn[i];
    }
    __syncthreads();
    const int S = npre + 1;
    const float inv = rsqrtf((float)H);
    for (int pr = warp; pr < nq * S; pr += blockDim.x >> 5) {
        const int q = pr / S, s = pr % S;
        const float* a = s < npre ? mix_state(m, s) + (long)t * H : cn;
        const float* b = query + (long)q * H;
        float d = 0.0f;
        for (int i = lane * 4; i < H; i += 128) {
            const float4 x = *(const float4*)(a + i), y = *(const float4*)(b + i);
            d += x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
        }
        d = warp_sum(d);
        if (lane == 0) sc[q][s] = d * inv;
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        const int q = threadIdx.x;
        float mx = NEG_INF;
        for (int s = 0; s < S; ++s) mx = fmaxf(mx, sc[q][s]);
        float z = 0.0f;
        for (int s = 0; s < S; ++s) {
            sc[q][s] = __expf(sc[q][s] - mx);
            z += sc[q][s];
        }
        for (int s = 0; s < S; ++s) sc[q][s] /= z;
    }
    __syncthreads();
    // 4 consecutive values per thread (H <= 4096)
    const int i0 = threadIdx.x * 4;
    const bool on = i0 < H;
    float acc[4][4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[q][e] = 0.0f;
    if (on) {
        for (int s = 0; s < S; ++s) {
            const float4 v = s < npre ? *(const float4*)(mix_state(m, s) + (long)t * H + i0) : *(const float4*)(cn + i0);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                if (q < nq) {
                    const float p = sc[q][s];
                    acc[q][0] += p * v.x, acc[q][1] += p * v.y, acc[q][2] += p * v.z, acc[q][3] += p * v.w;
                }
            }
        }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        if (q >= nq) break;
        if (m.w[q]) {
            float sq = 0.0f;
#pragma unroll
            for (int e = 0; e < 4; ++e) sq += acc[q][e] * acc[q][e];
            sq = block_sum(on ? sq : 0.0f, red);
            const float nf = rsqrtf(sq / (float)H + eps2);
            if (on) {
                const float* w = (const float*)m.w[q];
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[q][e] *= nf * w[i0 + e];
            }
        }
        if (on) *(float4*)(out + ((long)q * T + t) * H + i0) = make_float4(acc[q][0], acc[q][1], acc[q][2], acc[q][3]);
        if (m.oq[q]) {
            // int8 per 32 values: 8 threads (a quarter warp) hold a block
            float a = on ? fmaxf(fmaxf(fabsf(acc[q][0]), fabsf(acc[q][1])), fmaxf(fabsf(acc[q][2]), fabsf(acc[q][3]))) : 0.0f;
#pragma unroll
            for (int o = 4; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(FULL, a, o));
            if (on) {
                const float k = a > 0.0f ? 127.0f / a : 0.0f;
                char4 c = make_char4((signed char)__float2int_rn(acc[q][0] * k), (signed char)__float2int_rn(acc[q][1] * k),
                                     (signed char)__float2int_rn(acc[q][2] * k), (signed char)__float2int_rn(acc[q][3] * k));
                *(char4*)((signed char*)m.oq[q] + (long)t * H + i0) = c;
                if ((threadIdx.x & 7) == 0) ((float*)m.od[q])[(long)t * (H / 32) + i0 / 32] = a / 127.0f;
            }
        }
    }
    (void)nrm;
}

// Loop controller, first half: gate[c, t, r] = down_c[r] . [rms(cur_c[t]); rms(msg_c[t])] for
// up to 4 controllers c (blockIdx.z). down_c [R, 2H]. A block per r: grid (R, T, nc), block 256.
struct Ctrl4 { u64 cur[4], msg[4], down[4], gain[4], forget[4], out[4]; };

extern "C" __global__ void ink_ctrl_down(Ctrl4 c, float* gate, int T, int H, int R, float eps) {
    __shared__ float red[8];
    const int t = blockIdx.y, z = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* cur = (const float*)c.cur[z] + (long)t * H;
    const float* msg = (const float*)c.msg[z] + (long)t * H;
    float a = 0.0f, b = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) a += cur[i] * cur[i], b += msg[i] * msg[i];
    a = block_sum(a, red);
    b = block_sum(b, red);
    const float fa = rsqrtf(a / (float)H + eps), fb = rsqrtf(b / (float)H + eps);
    const int r = blockIdx.x;
    const float* w = (const float*)c.down[z] + (long)r * 2 * H;
    float d = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) d += w[i] * cur[i] * fa + w[H + i] * msg[i] * fb;
    d = block_sum(d, red);
    (void)warp;
    (void)lane;
    if (threadIdx.x == 0) gate[((long)z * T + t) * R + r] = d;
}

// Second half: out_c[t, h] = cur - tanh(forget_c[h] . g) cur + tanh(gain_c[h] . g) msg.
// gain / forget [H, R] (R <= 64). grid (ceil(H / 8), T, nc), block 256.
extern "C" __global__ void ink_ctrl_up(Ctrl4 c, const float* gate, int T, int H, int R) {
    const int t = blockIdx.y, z = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.x * 8 + warp;
    if (h >= H) return;
    const float* g = gate + ((long)z * T + t) * R;
    const float* wg = (const float*)c.gain[z] + (long)h * R;
    const float* wf = (const float*)c.forget[z] + (long)h * R;
    float a = 0.0f, b = 0.0f;
    for (int r = lane; r < R; r += 32) a += wg[r] * g[r], b += wf[r] * g[r];
    a = warp_sum(a);
    b = warp_sum(b);
    if (lane == 0) {
        const float cu = ((const float*)c.cur[z])[(long)t * H + h];
        const float m = ((const float*)c.msg[z])[(long)t * H + h];
        ((float*)c.out[z])[(long)t * H + h] = cu - tanhf(b) * cu + tanhf(a) * m;
    }
}

// f32 GEMV for up to 8 tokens with row-range input selection (as ink_q8_gemv). K % 4 == 0.
template <int T>
__device__ void f32_gemv(const float* W, const float* x0, const float* x1, const float* x2, int r1, int r2, int ldx,
                         float* y, int ldy, int N, int K) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= N) return;
    const float* x = row < r1 ? x0 : (row < r2 ? x1 : x2);
    const float* w = W + (long)row * K;
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    for (int c = lane * 4; c < K; c += 128) {
        float4 wv = *(const float4*)(w + c);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float4 a = *(const float4*)(x + (long)t * ldx + c);
            acc[t] += wv.x * a.x + wv.y * a.y + wv.z * a.z + wv.w * a.w;
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        float v = warp_sum(acc[t]);
        if (lane == 0) y[(long)t * ldy + row] = v;
    }
}

#define F32_GEMV(T)                                                                                          \
    extern "C" __global__ void ink_f32_gemv_##T(const float* W, const float* x0, const float* x1,            \
                                                const float* x2, int r1, int r2, int ldx, float* y, int ldy, \
                                                int N, int K) {                                              \
        f32_gemv<T>(W, x0, x1, x2, r1, r2, ldx, y, ldy, N, K);                                               \
    }
F32_GEMV(1)
F32_GEMV(2)
F32_GEMV(3)
F32_GEMV(4)
F32_GEMV(5)
F32_GEMV(6)
F32_GEMV(7)
F32_GEMV(8)

// ---------------------------------------------------------------------------------------------
// Fused decode steps.

// K and V of the new tokens from the projections y: causal conv (with the per-head taps kd / vd
// when not null, [T, ldd] at the head's 4 columns), then K clipped to L2 norm qk_clip and V
// RMS-normalized, into kn / vn [T, C] and the cache: cache 0 bf16 pool rows at out_loc, 1 the
// int8 pool, 2 bf16 rings of `ring` positions per state slot. grid (2 * nkv, B), block 128 (one
// head per block, blocks >= nkv for V).
extern "C" __global__ void ink_kv_prep(const float* y, int ldy, int kc, int vc, const float* kd, const float* vd,
                                       int ldd, const float* wk, const float* wv, float* state, int ctot,
                                       const u32* conv_seqs, const u32* seqs, const u32* out_loc, float* kn, float* vn,
                                       int nkv, float qk_clip, float eps, float inv_k, float inv_v, int cache,
                                       void* kcache, void* vcache, int ring) {
    __shared__ float red[4];
    const bool isv = blockIdx.x >= (unsigned)nkv;
    const int g = isv ? blockIdx.x - nkv : blockIdx.x;
    const int C = nkv * HD;
    const int c = g * HD + threadIdx.x;
    const u32* si = conv_seqs + blockIdx.y * 4;
    const u32 q0 = si[0], ql = si[1], slot = si[2], fresh = si[3];
    const int kv_len = seqs[blockIdx.y * 5 + 2];
    float* st = state + (long)slot * 3 * ctot + (isv ? C : 0) + c;
    float h0 = fresh ? 0.0f : st[0], h1 = fresh ? 0.0f : st[ctot], h2 = fresh ? 0.0f : st[2 * ctot];
    const float* w = (isv ? wv : wk) + c * 4;
    const float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    const float* dyn = isv ? vd : kd;
    const float inv = isv ? inv_v : inv_k;
    float* outp = isv ? vn : kn;
    void* cp = isv ? vcache : kcache;
    for (u32 j = 0; j < ql; ++j) {
        const long row = q0 + j;
        const float xc = y[row * ldy + (isv ? vc : kc) + c];
        float o = w0 * h0 + w1 * h1 + w2 * h2 + w3 * xc;
        if (dyn) {
            const float* d = dyn + row * ldd + g * 4;
            o += d[0] * xc + d[1] * h2 + d[2] * h1 + d[3] * h0;
        }
        float val = (xc + o) * inv;
        const float ss = block_sum(val * val, red);
        val *= isv ? rsqrtf(ss / (float)HD + eps) : qk_clip / fmaxf(sqrtf(ss), qk_clip);
        outp[row * C + c] = val;
        const int pos = kv_len - (int)ql + (int)j;
        if (cache == 2) {
            if (pos >= kv_len - ring) ((u16*)cp)[((long)slot * ring + pos % ring) * C + c] = f2bf(val);
        } else if (cache == 0) {
            ((u16*)cp)[(long)out_loc[row] * C + c] = f2bf(val);
        } else {
            const float mx = block_max(fabsf(val), red);
            signed char* rp = (signed char*)cp + ((long)out_loc[row] * nkv + g) * (HD + 4);
            rp[threadIdx.x] = (signed char)__float2int_rn(mx > 0.0f ? val * (127.0f / mx) : 0.0f);
            if (threadIdx.x == 0) *(float*)(rp + HD) = mx / 127.0f;
        }
        h0 = h1;
        h1 = h2;
        h2 = xc;
    }
    st[0] = h0;
    st[ctot] = h1;
    st[2 * ctot] = h2;
}

// Queries and relative projections of the new tokens: q head h (128 values at column qc + 128 h
// of y) times qs, clipped to qk_clip; r head h (nrel values at rc + nrel h) times rs, clipped to
// rel_clip; both times the log position factor when alpha > 0. grid (T, nh), block 128.
extern "C" __global__ void ink_qr_prep(const float* y, int ldy, int qc, int rc, const float* qs, const float* rs,
                                       int nh, int nrel, float qk_clip, float rel_clip, const u32* pos, float alpha,
                                       float floor_, float* qo, float* ro) {
    __shared__ float red[4];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float lf = alpha > 0.0f ? 1.0f + alpha * logf(fmaxf(1.0f, ((float)pos[t] + 1.0f) / floor_)) : 1.0f;
    float v = y[(long)t * ldy + qc + h * HD + d] * qs[h * HD + d];
    const float ss = block_sum(v * v, red);
    qo[(long)t * nh * HD + h * HD + d] = v * (qk_clip / fmaxf(sqrtf(ss), qk_clip)) * lf;
    if (d < 32) {
        float r = d < nrel ? y[(long)t * ldy + rc + h * nrel + d] * rs[h * nrel + d] : 0.0f;
        const float rr = warp_sum(r * r);
        if (d < nrel) ro[(long)t * nh * nrel + h * nrel + d] = r * (rel_clip / fmaxf(sqrtf(rr), rel_clip)) * lf;
    }
}

// ---- Unrolled GEMV variants: each lane owns NIT chunks of 8 codes, all loads issued up front ----

// Dot of a Q8 row (K = NIT * 8 * LANES) with int8 activations, over LANES lanes of a warp.
template <int NIT, int LANES>
__device__ __forceinline__ float q8_dot_u(const signed char* w, const u16* s, const signed char* xq, const float* xd,
                                          int l) {
    int2 wv[NIT], xv[NIT];
#pragma unroll
    for (int it = 0; it < NIT; ++it) {
        const int ch = l + it * LANES;
        wv[it] = *(const int2*)(w + ch * 8);
        xv[it] = *(const int2*)(xq + ch * 8);
    }
    float acc = 0.0f;
#pragma unroll
    for (int it = 0; it < NIT; ++it) {
        const int ch = l + it * LANES;
        int si = __dp4a(wv[it].x, xv[it].x, 0);
        si = __dp4a(wv[it].y, xv[it].y, si);
        acc += (float)si * f16f(s[ch >> 2]) * xd[ch >> 2];
    }
    return acc;
}

// Q8 GEMV, one row per warp, K = NIT * 256; f32 tail rows as ink_q8_gemv_q. grid ceil((N + nf) / 8).
template <int T, int NIT>
__device__ void q8_gemv_u(const signed char* W, const u16* S, const signed char* q0, const float* d0,
                          const signed char* q1, const float* d1, const signed char* q2, const float* d2, int r1,
                          int r2, float* y, int ldy, int N, const float* Wf, int nf, const float* f0, const float* f1,
                          const float* f2, int fr1, int fr2, float* yf, int ldyf) {
    const int K = NIT * 256;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= N) {
        const int fr = row - N;
        if (fr >= nf) return;
        const float* x = fr < fr1 ? f0 : (fr < fr2 ? f1 : f2);
        const float* w = Wf + (long)fr * K;
        float acc[T];
#pragma unroll
        for (int t = 0; t < T; ++t) acc[t] = 0.0f;
        for (int c = lane * 4; c < K; c += 128) {
            const float4 wv = *(const float4*)(w + c);
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const float4 a = *(const float4*)(x + (long)t * K + c);
                acc[t] += wv.x * a.x + wv.y * a.y + wv.z * a.z + wv.w * a.w;
            }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float v = warp_sum(acc[t]);
            if (lane == 0) yf[(long)t * ldyf + fr] = v;
        }
        return;
    }
    const signed char* xq = row < r1 ? q0 : (row < r2 ? q1 : q2);
    const float* xd = row < r1 ? d0 : (row < r2 ? d1 : d2);
    const signed char* w = W + (long)row * K;
    const u16* s = S + (long)row * (K / 32);
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = warp_sum(q8_dot_u<NIT, 32>(w, s, xq + (long)t * K, xd + (long)t * (K / 32), lane));
        if (lane == 0) y[(long)t * ldy + row] = v;
    }
}

#define Q8_GEMV_U(T, NIT)                                                                                       \
    extern "C" __global__ void ink_q8_gemv_u_##T##_##NIT(                                                      \
        const signed char* W, const u16* S, const signed char* q0, const float* d0, const signed char* q1,      \
        const float* d1, const signed char* q2, const float* d2, int r1, int r2, float* y, int ldy, int N,      \
        int K, const float* Wf, int nf, const float* f0, const float* f1, const float* f2, int fr1, int fr2,    \
        float* yf, int ldyf) {                                                                                  \
        (void)K;                                                                                                \
        q8_gemv_u<T, NIT>(W, S, q0, d0, q1, d1, q2, d2, r1, r2, y, ldy, N, Wf, nf, f0, f1, f2, fr1, fr2, yf, ldyf); \
    }
#define Q8_GEMV_U_ALL(NIT) Q8_GEMV_U(1, NIT) Q8_GEMV_U(2, NIT) Q8_GEMV_U(3, NIT) Q8_GEMV_U(4, NIT) \
    Q8_GEMV_U(5, NIT) Q8_GEMV_U(6, NIT) Q8_GEMV_U(7, NIT) Q8_GEMV_U(8, NIT)
Q8_GEMV_U_ALL(9)
Q8_GEMV_U_ALL(12)

// ---------------------------------------------------------------------------------------------
// Prefill-sized helpers.

// Input value of channel c at row `row` (the sum of nsum parts when nsum > 0).
__device__ __forceinline__ float conv_in(const float* x, int ldx, long row, int c, int C, int nsum) {
    if (nsum <= 0) return x[row * ldx + c];
    float s = 0.0f;
    for (int k = 0; k < nsum; ++k) s += x[(row * nsum + k) * C + c];
    return s;
}

// ink_conv with every row in parallel (rows before a sequence's first new one come from its
// state, which ink_conv_state then advances). grid (ceil(C / 128), T).
extern "C" __global__ void ink_conv_par(const float* x, int ldx, float* out, int ldo, const float* w, const float* dyn,
                                        int ldd, int heads, const float* state, int ctot, int off, const u32* seqs,
                                        const u32* rowseq, int C, float inv, const float* res, int ldr, int nsum) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const long row = blockIdx.y;
    const u32* si = seqs + rowseq[row] * 4;
    const int q0 = si[0], slot = si[2], fresh = si[3];
    const int j = (int)row - q0;
    const float* st = state + (long)slot * 3 * ctot + off + c;
    float in[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int jj = j - 3 + k;
        in[k] = jj >= 0 ? conv_in(x, ldx, q0 + jj, c, C, nsum) : (fresh ? 0.0f : st[(3 + jj) * ctot]);
    }
    float o = w[c * 4] * in[0] + w[c * 4 + 1] * in[1] + w[c * 4 + 2] * in[2] + w[c * 4 + 3] * in[3];
    if (dyn) {
        const float* d = dyn + row * ldd + (c / (C / heads)) * 4;
        o += d[0] * in[3] + d[1] * in[2] + d[2] * in[1] + d[3] * in[0];
    }
    out[row * ldo + c] = (in[3] + o) * inv + (res ? res[row * ldr + c] : 0.0f);
}

// Advance each sequence's conv state past its new rows. grid (ceil(C / 128), B).
extern "C" __global__ void ink_conv_state(const float* x, int ldx, float* state, int ctot, int off, const u32* seqs,
                                          int C, int nsum) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const u32* si = seqs + blockIdx.y * 4;
    const int q0 = si[0], ql = si[1], slot = si[2], fresh = si[3];
    float* st = state + (long)slot * 3 * ctot + off + c;
    float h[3] = {fresh ? 0.0f : st[0], fresh ? 0.0f : st[ctot], fresh ? 0.0f : st[2 * ctot]};
    float nw[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int i = ql + k;  // index into history (3) + new rows (ql)
        nw[k] = i < 3 ? h[i] : conv_in(x, ldx, q0 + i - 3, c, C, nsum);
    }
    st[0] = nw[0];
    st[ctot] = nw[1];
    st[2 * ctot] = nw[2];
}

// dst[idx[i], :] = src[i, :] * wts[i]. grid (n, ceil(H / 256)).
extern "C" __global__ void ink_scatter_rows(const float* src, const u32* idx, const float* wts, float* dst, int H) {
    const int i = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c < H) dst[(long)idx[i] * H + c] = src[(long)i * H + c] * wts[i];
}
