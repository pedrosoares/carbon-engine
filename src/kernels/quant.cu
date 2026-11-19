// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// carbon-engine quantized-weight kernels, compiled at runtime with NVRTC.
// No CUDA headers: bf16 / f16 are handled as raw 16-bit patterns so the same source builds
// for every architecture (compiled to compute_70 PTX, JIT-compiled by the driver).

typedef unsigned char u8;
typedef unsigned short u16;

__device__ __forceinline__ float e4m3_to_f32(u8 b) {
    unsigned sign = b >> 7, exp = (b >> 3) & 0xF, man = b & 0x7;
    float v;
    if (exp == 0) {
        v = (float)man * 0.001953125f;  // man/8 * 2^-6
    } else if (exp == 0xF && man == 0x7) {
        v = __int_as_float(0x7fc00000);  // NaN
    } else {
        v = (1.0f + (float)man * 0.125f) * __int_as_float((int)(exp + 120) << 23);  // 2^(exp-7)
    }
    return sign ? -v : v;
}

__device__ __constant__ float E2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                          -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float bf16_to_f32(u16 h) { return __int_as_float(((unsigned)h) << 16); }
__device__ __forceinline__ u16 f32_to_bf16(float f) {
    unsigned u = __float_as_int(f);
    if ((u & 0x7fffffff) > 0x7f800000) return (u16)((u >> 16) | 0x40);  // quiet NaN
    u += 0x7fff + ((u >> 16) & 1);  // round to nearest even
    return (u16)(u >> 16);
}
#ifndef CARBON_HOST_TEST
__device__ __forceinline__ float f16_to_f32(u16 h) {
    float f;
    asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ u16 f32_to_f16(float f) {
    u16 h;
    asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}
#else
// host emulation (scripts/test_cuda_kernels_host.sh)
__device__ __forceinline__ float f16_to_f32(u16 h) { _Float16 x; memcpy(&x, &h, 2); return (float)x; }
__device__ __forceinline__ u16 f32_to_f16(float f) { _Float16 x = (_Float16)f; u16 h; memcpy(&h, &x, 2); return h; }
#endif

// element loaders / storers per activation dtype
struct F32 { typedef float T;
    __device__ static float load(const float* p, long i) { return p[i]; }
    __device__ static void store(float* p, long i, float v) { p[i] = v; }
    __device__ static float round(float v) { return v; } };
struct BF16 { typedef u16 T;
    __device__ static float load(const u16* p, long i) { return bf16_to_f32(p[i]); }
    __device__ static void store(u16* p, long i, float v) { p[i] = f32_to_bf16(v); }
    __device__ static float round(float v) { return bf16_to_f32(f32_to_bf16(v)); } };
struct F16 { typedef u16 T;
    __device__ static float load(const u16* p, long i) { return f16_to_f32(p[i]); }
    __device__ static void store(u16* p, long i, float v) { p[i] = f32_to_f16(v); }
    __device__ static float round(float v) { return f16_to_f32(f32_to_f16(v)); } };

// ---- NVFP4: codes [n, k/2] (low nibble first), e4m3 scales [n, k/16], global multiplier ----

template <class D>
__device__ void nvfp4_dequant(const u8* packed, const u8* scales, float global, typename D::T* out,
                              int n, int k) {
    long half = k / 2;
    long total = (long)n * half;
    for (long b = (long)blockIdx.x * blockDim.x + threadIdx.x; b < total; b += (long)gridDim.x * blockDim.x) {
        long row = b / half, c = b % half;
        float s = e4m3_to_f32(scales[row * (k / 16) + c / 8]);
        u8 code = packed[b];
        D::store(out, row * k + 2 * c, E2M1[code & 15] * s * global);
        D::store(out, row * k + 2 * c + 1, E2M1[code >> 4] * s * global);
    }
}

// y[t, r] = sum_k x[t, k] * W[r, k]; one warp per output row, up to 8 rows of x.
template <class D>
__device__ void nvfp4_gemv(const u8* packed, const u8* scales, float global, const typename D::T* x,
                           typename D::T* y, int n, int k, int t) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    long row = (long)blockIdx.x * (blockDim.x / 32) + warp;
    if (row >= n) return;
    int half = k / 2, nb = k / 16;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // each lane takes whole 16-element blocks (8 bytes) so one scale serves the inner loop
    for (int blk = lane; blk < nb; blk += 32) {
        float s = e4m3_to_f32(scales[row * nb + blk]);
        const u8* codes = packed + row * half + blk * 8;
        for (int j = 0; j < 8; ++j) {
            u8 code = codes[j];
            float lo = E2M1[code & 15], hi = E2M1[code >> 4];
            int col = blk * 16 + 2 * j;
            for (int ti = 0; ti < t; ++ti) {
                acc[ti] += s * (lo * D::load(x, (long)ti * k + col) + hi * D::load(x, (long)ti * k + col + 1));
            }
        }
    }
    for (int ti = 0; ti < t; ++ti) {
        float v = acc[ti];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (lane == 0) D::store(y, (long)ti * n + row, v * global);
    }
}

// ---- FP8 e4m3: weights [n, k], f32 scales [sr, sc] over (b0 x b1) blocks (1x1 = per-tensor) ----

template <class D>
__device__ void fp8_dequant(const u8* w, const float* scale, typename D::T* out, int n, int k, int sc, int b0,
                            int b1) {
    long total = (long)n * k;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
        long r = i / k, c = i % k;
        D::store(out, i, e4m3_to_f32(w[i]) * scale[(r / b0) * sc + c / b1]);
    }
}

template <class D>
__device__ void fp8_gemv(const u8* w, const float* scale, const typename D::T* x, typename D::T* y, int n, int k,
                         int t, int sc, int b0, int b1) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    long row = (long)blockIdx.x * (blockDim.x / 32) + warp;
    if (row >= n) return;
    const float* srow = scale + (row / b0) * sc;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int c = lane; c < k; c += 32) {
        float wv = e4m3_to_f32(w[row * k + c]) * srow[c / b1];
        for (int ti = 0; ti < t; ++ti) acc[ti] += wv * D::load(x, (long)ti * k + c);
    }
    for (int ti = 0; ti < t; ++ti) {
        float v = acc[ti];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (lane == 0) D::store(y, (long)ti * n + row, v);
    }
}

// ---- GGUF K-quants: rows of 256-weight super-blocks, kind 0 = Q4_K (144 bytes), 1 = Q6_K (224:
// the file's 210-byte block padded so both kinds are read as aligned 32-bit words) ----
//
// Q4_K block: f16 d, f16 dmin, 12 bytes of 6-bit (scale, min) per 32-weight sub-block, 128 code
// bytes; sub-blocks 2p / 2p+1 are the low / high nibbles of bytes [32p, 32p+32):
//   w = d * scale * code - dmin * min
// Q6_K block: ql[128], qh[64], i8 scales[16] (per 16 weights), f16 d; codes are 6 bits, biased 32:
//   w = d * scale * (code - 32)

#define Q4K_BLOCK 144
#define Q6K_BLOCK 224

__device__ __forceinline__ float kq_half(const u8* p) { return f16_to_f32((u16)(p[0] | (p[1] << 8))); }

__device__ __forceinline__ void q4k_scale_min(int j, const u8* q, float d, float dmin, float* s, float* m) {
    unsigned sc, mn;
    if (j < 4) {
        sc = q[j] & 63;
        mn = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
    *s = d * (float)sc;
    *m = dmin * (float)mn;
}

// weight `l` (0..32) of 32-weight group `j` (0..8) of a Q6_K block
__device__ __forceinline__ float q6k_weight(const u8* b, int j, int l, float d) {
    int half = j >> 2, which = j & 3;
    const u8* ql = b + 64 * half;
    u8 lo = ql[l + 32 * (which & 1)];
    int nib = which >= 2 ? (lo >> 4) : (lo & 15);
    int hb = (b[128 + 32 * half + l] >> (2 * which)) & 3;
    int sc = (signed char)b[192 + 8 * half + (l >> 4) + 2 * which];
    return d * (float)sc * (float)((nib | (hb << 4)) - 32);
}

// out [n, k]; one warp per 64-weight unit (Q4_K: a sub-block pair sharing 32 code bytes; Q6_K:
// two 32-weight groups), lane l taking weights l and 32 + l, so every load and store of the warp
// covers one contiguous run (a thread per group scattered each store over 32 cache lines).
template <class D>
__device__ void kq_dequant(const u8* data, typename D::T* out, int n, int k, int kind) {
    long units = (long)n * (k / 64);
    long row_bytes = (long)(k / 256) * (kind ? Q6K_BLOCK : Q4K_BLOCK);
    long warps = (long)gridDim.x * (blockDim.x / 32);
    int lane = threadIdx.x % 32;
    for (long u = (long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; u < units; u += warps) {
        long row = u / (k / 64);
        int s = (int)(u % (k / 64)), blk = s >> 2, pair = s & 3;
        long o = row * k + (long)s * 64 + lane;
        if (kind == 0) {
            const u8* b = data + row * row_bytes + (long)blk * Q4K_BLOCK;
            float d = kq_half(b), dmin = kq_half(b + 2), s1, m1, s2, m2;
            q4k_scale_min(2 * pair, b + 4, d, dmin, &s1, &m1);
            q4k_scale_min(2 * pair + 1, b + 4, d, dmin, &s2, &m2);
            u8 c = b[16 + 32 * pair + lane];
            D::store(out, o, s1 * (float)(c & 15) - m1);
            D::store(out, o + 32, s2 * (float)(c >> 4) - m2);
        } else {
            const u8* b = data + row * row_bytes + (long)blk * Q6K_BLOCK;
            float d = kq_half(b + 208);
            D::store(out, o, q6k_weight(b, 2 * pair, lane, d));
            D::store(out, o + 32, q6k_weight(b, 2 * pair + 1, lane, d));
        }
    }
}

// y[t, r] = sum_c x[t, c] * W[r, c]; one warp per output row, up to 8 rows of x. Each lane takes
// whole 64-weight units (Q4_K: a sub-block pair, 32 code bytes; Q6_K: two 32-weight groups).
template <class D>
__device__ void kq_gemv(const u8* data, const typename D::T* x, typename D::T* y, int n, int k, int t, int kind) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    long row = (long)blockIdx.x * (blockDim.x / 32) + warp;
    if (row >= n) return;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int units = k / 64;
    if (kind == 0) {
        const u8* rp = data + row * (long)(k / 256) * Q4K_BLOCK;
        for (int u = lane; u < units; u += 32) {
            int blk = u >> 2, pair = u & 3;
            const u8* b = rp + (long)blk * Q4K_BLOCK;
            float d = kq_half(b), dmin = kq_half(b + 2), s1, m1, s2, m2;
            q4k_scale_min(2 * pair, b + 4, d, dmin, &s1, &m1);
            q4k_scale_min(2 * pair + 1, b + 4, d, dmin, &s2, &m2);
            const u8* q = b + 16 + 32 * pair;
            int col = u * 64;
            for (int l = 0; l < 32; ++l) {
                u8 c = q[l];
                float lo = s1 * (float)(c & 15) - m1, hi = s2 * (float)(c >> 4) - m2;
                for (int ti = 0; ti < t; ++ti) {
                    acc[ti] += lo * D::load(x, (long)ti * k + col + l) + hi * D::load(x, (long)ti * k + col + 32 + l);
                }
            }
        }
    } else {
        const u8* rp = data + row * (long)(k / 256) * Q6K_BLOCK;
        for (int u = lane; u < units; u += 32) {
            int blk = u >> 2, j = (u & 3) * 2;
            const u8* b = rp + (long)blk * Q6K_BLOCK;
            float d = kq_half(b + 208);
            int col = u * 64;
            for (int l = 0; l < 32; ++l) {
                float w0 = q6k_weight(b, j, l, d), w1 = q6k_weight(b, j + 1, l, d);
                for (int ti = 0; ti < t; ++ti) {
                    acc[ti] += w0 * D::load(x, (long)ti * k + col + l) + w1 * D::load(x, (long)ti * k + col + 32 + l);
                }
            }
        }
    }
    for (int ti = 0; ti < t; ++ti) {
        float v = acc[ti];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (lane == 0) D::store(y, (long)ti * n + row, v);
    }
}

// ---- K-quant GEMV on 8-bit activations (the fast path) ----
//
// kq_prep quantizes x per 32 inputs: x ~ xs * xq with xq in [-127, 127]. Per row of x the
// prepared buffer holds (in float units) [xq: k bytes][xs: k/32][sums: k/16], sums being
// xs * sum(xq) over each 16 inputs (the K-quants' offset terms).
// kq_gemv_i8 then takes 4 weights per dp4a: the packed codes are split into byte lanes with two
// masks, so a row costs about one instruction per weight and stays memory-bound.

#define KQ_PREP_STRIDE(k) ((k) / 4 + (k) / 32 + (k) / 16)

#ifndef CARBON_HOST_TEST
__device__ __forceinline__ int kq_dp4a(unsigned w, int x, int acc) {
    int r;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(w), "r"(x), "r"(acc));
    return r;
}
#else
__device__ __forceinline__ int kq_dp4a(unsigned w, int x, int acc) {
    for (int i = 0; i < 4; ++i) acc += (int)((w >> (8 * i)) & 0xff) * (int)(signed char)((x >> (8 * i)) & 0xff);
    return acc;
}
#endif

// One scale group's contribution to a row's sum, shared by the dp4a and tensor-core kernels.
// The two kernels add the groups in different orders, so their results differ in the last
// bits: a weight must use the same kernel for every token count (`kquant_use_tc`), or a
// multi-token verify step would not reproduce what single-token steps compute. (Summing in f64
// would make the order irrelevant, but f64 is throttled on consumer GPUs: 2.5x slower steps.)
__device__ __forceinline__ float kq_term_q4k(float s, float m, float xs, float xsum, int isum) {
    return (s * xs) * (float)isum - m * xsum;
}
__device__ __forceinline__ float kq_term_q6k(float f, float xs, float xsum, int isum) {
    return (f * xs) * (float)isum - f * (32.0f * xsum);
}

// One group of 32 inputs by one warp whose lanes hold the values (coalesced loads, the
// reductions through shuffles).
__device__ __forceinline__ void kq_prep_warp(float v, float* base, int k, int s) {
    const int lane = threadIdx.x % 32;
    float mx = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
    float inv = mx > 0.0f ? 127.0f / mx : 0.0f, xs = mx / 127.0f;
    int q = __float2int_rn(v * inv);
    unsigned word = ((unsigned)q & 0xff) << (8 * (lane & 3));
    word |= __shfl_xor_sync(0xffffffff, word, 1);
    word |= __shfl_xor_sync(0xffffffff, word, 2);
    int sum = q;
    for (int off = 1; off < 16; off <<= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if ((lane & 3) == 0) ((unsigned*)base)[s * 8 + (lane >> 2)] = word;
    if (lane == 0) base[k / 4 + s] = xs;
    if ((lane & 15) == 0) base[k / 4 + k / 32 + 2 * s + (lane >> 4)] = xs * (float)sum;
}

template <class D>
__device__ void kq_prep(const typename D::T* x, float* out, int k, int t) {
    const long groups = (long)t * (k / 32), warps = (long)gridDim.x * (blockDim.x / 32);
    const int lane = threadIdx.x % 32;
    for (long g = (long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; g < groups; g += warps) {
        int ti = (int)(g / (k / 32)), s = (int)(g % (k / 32));
        kq_prep_warp(D::load(x, (long)ti * k + (long)s * 32 + lane), out + (long)ti * KQ_PREP_STRIDE(k), k, s);
    }
}

// SiLU(g) * u and its activation prep in one launch (the MLP's input to the down projection;
// the product itself is not kept). Both steps rounded to the activation type.
template <class D>
__device__ void kq_silu_mul_prep(const typename D::T* gt, const typename D::T* up, float* out, int k, int t) {
    const long groups = (long)t * (k / 32), warps = (long)gridDim.x * (blockDim.x / 32);
    const int lane = threadIdx.x % 32;
    for (long g = (long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; g < groups; g += warps) {
        int ti = (int)(g / (k / 32)), s = (int)(g % (k / 32));
        const long i = (long)ti * k + (long)s * 32 + lane;
        float a = D::load(gt, i);
        a = D::round(a / (1.0f + expf(-a)));
        kq_prep_warp(D::round(a * D::load(up, i)), out + (long)ti * KQ_PREP_STRIDE(k), k, s);
    }
}

// one warp: row `row` of an n-row weight
template <class D>
__device__ __forceinline__ void kq_gemv_i8_row(const u8* data, const float* prep, typename D::T* y, int n, int k,
                                               int t, int kind, long row) {
    int lane = threadIdx.x % 32;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int units = k / 64, stride = KQ_PREP_STRIDE(k);
    if (kind == 0) {
        const unsigned* rp = (const unsigned*)(data + row * (long)(k / 256) * Q4K_BLOCK);
        for (int u = lane; u < units; u += 32) {
            int blk = u >> 2, pair = u & 3;
            const unsigned* b = rp + blk * (Q4K_BLOCK / 4);
            unsigned h0 = b[0];
            float d = f16_to_f32((u16)(h0 & 0xffff)), dmin = f16_to_f32((u16)(h0 >> 16)), s1, m1, s2, m2;
            q4k_scale_min(2 * pair, (const u8*)(b + 1), d, dmin, &s1, &m1);
            q4k_scale_min(2 * pair + 1, (const u8*)(b + 1), d, dmin, &s2, &m2);
            const unsigned* q = b + 4 + 8 * pair;
            unsigned lo[8], hi[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                unsigned w = q[i];
                lo[i] = w & 0x0F0F0F0Fu;
                hi[i] = (w >> 4) & 0x0F0F0F0Fu;
            }
            for (int ti = 0; ti < t; ++ti) {
                const float* base = prep + (long)ti * stride;
                const int* xq = (const int*)base + u * 16;
                int ia = 0, ib = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    ia = kq_dp4a(lo[i], xq[i], ia);
                    ib = kq_dp4a(hi[i], xq[8 + i], ib);
                }
                const float* xs = base + k / 4 + 2 * u;
                const float* sm = base + k / 4 + k / 32 + 4 * u;
                acc[ti] += kq_term_q4k(s1, m1, xs[0], sm[0] + sm[1], ia) + kq_term_q4k(s2, m2, xs[1], sm[2] + sm[3], ib);
            }
        }
    } else {
        const unsigned* rp = (const unsigned*)(data + row * (long)(k / 256) * Q6K_BLOCK);
        for (int u = lane; u < units; u += 32) {
            int blk = u >> 2, half = (u >> 1) & 1, sh = (u & 1) * 4;
            const unsigned* b = rp + blk * (Q6K_BLOCK / 4);
            const unsigned* ql = b + 16 * half;
            const unsigned* qh = b + 32 + 8 * half;
            const signed char* sc = (const signed char*)(b + 48) + 8 * half + (u & 1) * 4;
            float d = f16_to_f32((u16)(b[52] & 0xffff));
            unsigned wa[8], wb[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                unsigned h = qh[i] >> sh;
                wa[i] = ((ql[i] >> sh) & 0x0F0F0F0Fu) | ((h & 0x03030303u) << 4);
                wb[i] = ((ql[8 + i] >> sh) & 0x0F0F0F0Fu) | (((h >> 2) & 0x03030303u) << 4);
            }
            // scales per 16 weights: group a (which = 2*hi) and b (which + 1), each in two halves
            float sa0 = d * (float)sc[0], sa1 = d * (float)sc[1], sb0 = d * (float)sc[2], sb1 = d * (float)sc[3];
            for (int ti = 0; ti < t; ++ti) {
                const float* base = prep + (long)ti * stride;
                const int* xq = (const int*)base + u * 16;
                int a0 = 0, a1 = 0, b0 = 0, b1 = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    a0 = kq_dp4a(wa[i], xq[i], a0);
                    a1 = kq_dp4a(wa[4 + i], xq[4 + i], a1);
                    b0 = kq_dp4a(wb[i], xq[8 + i], b0);
                    b1 = kq_dp4a(wb[4 + i], xq[12 + i], b1);
                }
                const float* xs = base + k / 4 + 2 * u;
                const float* sm = base + k / 4 + k / 32 + 4 * u;
                acc[ti] += kq_term_q6k(sa0, xs[0], sm[0], a0) + kq_term_q6k(sa1, xs[0], sm[1], a1) +
                           kq_term_q6k(sb0, xs[1], sm[2], b0) + kq_term_q6k(sb1, xs[1], sm[3], b1);
            }
        }
    }
    for (int ti = 0; ti < t; ++ti) {
        float v = acc[ti];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (lane == 0) D::store(y, (long)ti * n + row, v);
    }
}

template <class D>
__device__ void kq_gemv_i8(const u8* data, const float* prep, typename D::T* y, int n, int k, int t, int kind) {
    long row = (long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= n) return;
    kq_gemv_i8_row<D>(data, prep, y, n, k, t, kind, row);
}

// Up to 4 weights that read the same activations, in one launch (a decode step otherwise pays
// each launch's ramp-up and tail: about 12% of the read rate on 50 MB weights). Rows are
// numbered through the weights; weight s writes y + t * (rows before it).
#define KQ_MULTI 4
struct KqSeg {
    const u8* data;
    int n, kind;
};

template <class D>
__device__ void kq_gemv_i8_multi(const u8* d0, const u8* d1, const u8* d2, const u8* d3, const float* prep,
                                 typename D::T* y, int n0, int n1, int n2, int n3, int kinds, int k, int t) {
    long row = (long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const u8* data[KQ_MULTI] = {d0, d1, d2, d3};
    const int n[KQ_MULTI] = {n0, n1, n2, n3};
    long before = 0;
#pragma unroll
    for (int s = 0; s < KQ_MULTI; ++s) {
        if (row < n[s]) {
            kq_gemv_i8_row<D>(data[s], prep, y + before * t, n[s], k, t, (kinds >> (4 * s)) & 15, row);
            return;
        }
        row -= n[s];
        before += n[s];
    }
}

// ---- greedy pick over logits [R, V] f32: argmax (lowest index on ties) and its probability ----
//
// logit_pick_part: grid R * LP_PARTS blocks; block (r, p) reduces its slice of row r to
//   (max, argmax, sum exp(x - max)) in part [R, LP_PARTS, 3] (f32, the index as its bits).
// logit_pick_merge: one warp per row merges the parts: out [R, 2] = (index bits, probability).
#define LP_PARTS 32

extern "C" __global__ void logit_pick_part(const float* logits, float* part, int R, int V) {
    __shared__ float sm[32];
    __shared__ int si[32];
    int r = blockIdx.x / LP_PARTS, p = blockIdx.x % LP_PARTS;
    long lo = (long)V * p / LP_PARTS, hi = (long)V * (p + 1) / LP_PARTS;
    const float* x = logits + (long)r * V;
    float m = -3.0e38f;
    int mi = 0x7fffffff;
    for (long i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        float v = x[i];
        if (v > m) {
            m = v;
            mi = (int)i;
        }
    }
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float om = __shfl_xor_sync(0xffffffff, m, off);
        int oi = __shfl_xor_sync(0xffffffff, mi, off);
        if (om > m || (om == m && oi < mi)) {
            m = om;
            mi = oi;
        }
    }
    if (lane == 0) {
        sm[warp] = m;
        si[warp] = mi;
    }
    __syncthreads();
    float bm = sm[0];
    int bi = si[0];
    for (int w = 1; w < nw; ++w) {
        if (sm[w] > bm || (sm[w] == bm && si[w] < bi)) {
            bm = sm[w];
            bi = si[w];
        }
    }
    __syncthreads();
    float sum = 0.0f;
    for (long i = lo + threadIdx.x; i < hi; i += blockDim.x) sum += __expf(x[i] - bm);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if (lane == 0) sm[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int w = 0; w < nw; ++w) tot += sm[w];
        float* o = part + ((long)r * LP_PARTS + p) * 3;
        o[0] = bm;
        o[1] = __int_as_float(bi);
        o[2] = tot;
    }
}

extern "C" __global__ void logit_pick_merge(const float* part, float* out, int R) {
    int r = blockIdx.x, lane = threadIdx.x;
    if (r >= R || lane >= LP_PARTS) return;
    const float* p = part + ((long)r * LP_PARTS + lane) * 3;
    float m = p[0], pm = p[0], ps = p[2];
    int mi = __float_as_int(p[1]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float om = __shfl_xor_sync(0xffffffff, m, off);
        int oi = __shfl_xor_sync(0xffffffff, mi, off);
        if (om > m || (om == m && oi < mi)) {
            m = om;
            mi = oi;
        }
    }
    float sum = ps * __expf(pm - m);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if (lane == 0) {
        out[2 * r] = __int_as_float(mi);
        out[2 * r + 1] = 1.0f / sum;
    }
}

#define DEFINE_KERNELS(SUFFIX, D)                                                                            \
    extern "C" __global__ void nvfp4_dequant_##SUFFIX(const u8* p, const u8* s, float g, D::T* o, int n,     \
                                                      int k) { nvfp4_dequant<D>(p, s, g, o, n, k); }          \
    extern "C" __global__ void nvfp4_gemv_##SUFFIX(const u8* p, const u8* s, float g, const D::T* x, D::T* y, \
                                                   int n, int k, int t) { nvfp4_gemv<D>(p, s, g, x, y, n, k, t); } \
    extern "C" __global__ void fp8_dequant_##SUFFIX(const u8* w, const float* s, D::T* o, int n, int k,      \
                                                    int sc, int b0, int b1) {                                \
        fp8_dequant<D>(w, s, o, n, k, sc, b0, b1);                                                            \
    }                                                                                                         \
    extern "C" __global__ void fp8_gemv_##SUFFIX(const u8* w, const float* s, const D::T* x, D::T* y, int n, \
                                                 int k, int t, int sc, int b0, int b1) {                     \
        fp8_gemv<D>(w, s, x, y, n, k, t, sc, b0, b1);                                                         \
    }                                                                                                         \
    extern "C" __global__ void kq_dequant_##SUFFIX(const u8* w, D::T* o, int n, int k, int kind) {           \
        kq_dequant<D>(w, o, n, k, kind);                                                                      \
    }                                                                                                         \
    extern "C" __global__ void kq_gemv_##SUFFIX(const u8* w, const D::T* x, D::T* y, int n, int k, int t,    \
                                                int kind) {                                                   \
        kq_gemv<D>(w, x, y, n, k, t, kind);                                                                   \
    }                                                                                                         \
    extern "C" __global__ void kq_prep_##SUFFIX(const D::T* x, float* o, int k, int t) {                     \
        kq_prep<D>(x, o, k, t);                                                                               \
    }                                                                                                         \
    extern "C" __global__ void kq_silu_mul_prep_##SUFFIX(const D::T* g, const D::T* u, float* o, int k,      \
                                                         int t) {                                             \
        kq_silu_mul_prep<D>(g, u, o, k, t);                                                                   \
    }                                                                                                         \
    extern "C" __global__ void kq_gemv_i8_##SUFFIX(const u8* w, const float* p, D::T* y, int n, int k,       \
                                                   int t, int kind) {                                         \
        kq_gemv_i8<D>(w, p, y, n, k, t, kind);                                                                \
    }                                                                                                         \
    extern "C" __global__ void kq_gemv_i8_multi_##SUFFIX(const u8* d0, const u8* d1, const u8* d2,           \
                                                         const u8* d3, const float* p, D::T* y, int n0,      \
                                                         int n1, int n2, int n3, int kinds, int k, int t) {  \
        kq_gemv_i8_multi<D>(d0, d1, d2, d3, p, y, n0, n1, n2, n3, kinds, k, t);                               \
    }

DEFINE_KERNELS(f32, F32)
DEFINE_KERNELS(bf16, BF16)
DEFINE_KERNELS(f16, F16)
