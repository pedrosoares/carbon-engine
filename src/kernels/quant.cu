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
    __device__ static void store(float* p, long i, float v) { p[i] = v; } };
struct BF16 { typedef u16 T;
    __device__ static float load(const u16* p, long i) { return bf16_to_f32(p[i]); }
    __device__ static void store(u16* p, long i, float v) { p[i] = f32_to_bf16(v); } };
struct F16 { typedef u16 T;
    __device__ static float load(const u16* p, long i) { return f16_to_f32(p[i]); }
    __device__ static void store(u16* p, long i, float v) { p[i] = f32_to_f16(v); } };

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
    }

DEFINE_KERNELS(f32, F32)
DEFINE_KERNELS(bf16, BF16)
DEFINE_KERNELS(f16, F16)
