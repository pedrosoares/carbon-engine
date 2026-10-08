// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// carbon-engine grouped MoE expert GEMMs for prefill on tensor cores (NVFP4 experts, bf16
// activations), compiled with NVRTC for compute_80 after quant.cu (e4m3 / E2M1 / bf16 helpers).
//
// The step's (token, expert) pairs are sorted by expert; `tiles` lists (expert, first pair, pairs)
// for runs of up to MT_M pairs of one expert. One block per (tile, MT_N output columns), 4 warps of
// 16 rows. Per MT_K slice of the reduction dim, the activations and the expert's weight rows are
// staged in shared memory, the weights dequantized to bf16 exactly as nvfp4_dequant does
// (E2M1 * scale * global, rounded once), then mma.m16n8k16 accumulates in f32.
//   gate_up  act[p, n] = silu(x_p . gate_e[n]) * (x_p . up_e[n])         (bf16, [P, I])
//   down     y[p, h]   = w_p * (act_p . down_e[h])                        (bf16, [P, H])
// `tab` / `glob` are the slot table of the layer (moe_slot_ensure points it at the slots).

typedef unsigned int u32;
typedef unsigned long long u64;

#define MT_M 64
#define MT_N 64
#define MT_K 32
#define MT_KP 40  // padded shared row (bf16): conflict-free fragment loads
#define MT_THREADS 128

__device__ __forceinline__ void mt_mma(float* c, const u32* a, const u32* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// rows [r0, r0 + nr) of a [*, K] bf16 matrix, columns [k0, k0 + MT_K), into sa (zero past nr)
__device__ __forceinline__ void mt_load_a(const u16* src, long r0, int nr, int K, int k0, u16* sa) {
    for (int i = threadIdx.x; i < MT_M * (MT_K / 8); i += MT_THREADS) {
        int row = i / (MT_K / 8), c8 = i % (MT_K / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (row < nr) v = *(const uint4*)(src + (r0 + row) * K + k0 + c8 * 8);
        *(uint4*)(sa + row * MT_KP + c8 * 8) = v;
    }
}

// weight rows [n0, n0 + MT_N) of an NVFP4 [N, K] matrix, columns [k0, k0 + MT_K), dequantized.
// `lut` is E2M1 in shared memory (the __constant__ table serialises on divergent codes).
__device__ __forceinline__ void mt_load_w(const u8* packed, const u8* scales, float global, int N, int K, int n0,
                                          int k0, u16* sw, const float* lut) {
    for (int i = threadIdx.x; i < MT_N * (MT_K / 16); i += MT_THREADS) {
        int row = i / (MT_K / 16), h = i % (MT_K / 16);
        long n = n0 + row;
        u32* d = (u32*)(sw + row * MT_KP + h * 16);
        if (n >= N) {
            for (int j = 0; j < 8; ++j) d[j] = 0u;
            continue;
        }
        float s = e4m3_to_f32(scales[n * (K / 16) + k0 / 16 + h]);
        uint2 pv = *(const uint2*)(packed + n * (K / 2) + k0 / 2 + h * 8);
        u32 w[2] = {pv.x, pv.y};
        for (int j = 0; j < 8; ++j) {
            u32 code = (w[j / 4] >> ((j % 4) * 8)) & 255u;
            // same rounding as nvfp4_dequant: (code value * scale) * global
            u32 lo = f32_to_bf16(lut[code & 15] * s * global), hi = f32_to_bf16(lut[code >> 4] * s * global);
            d[j] = lo | (hi << 16);
        }
    }
}

__device__ __forceinline__ void mt_frag_a(const u16* sa, int warp, int kk, int g, int t, u32* a) {
    const u16* A = sa + (warp * 16) * MT_KP + kk;
    a[0] = *(const u32*)(A + g * MT_KP + 2 * t);
    a[1] = *(const u32*)(A + (g + 8) * MT_KP + 2 * t);
    a[2] = *(const u32*)(A + g * MT_KP + 2 * t + 8);
    a[3] = *(const u32*)(A + (g + 8) * MT_KP + 2 * t + 8);
}

__device__ __forceinline__ void mt_frag_b(const u16* sw, int f, int kk, int g, int t, u32* b) {
    const u16* B = sw + (f * 8 + g) * MT_KP + kk;
    b[0] = *(const u32*)(B + 2 * t);
    b[1] = *(const u32*)(B + 2 * t + 8);
}

// grid (tiles, ceil(I / MT_N)), MT_THREADS threads
extern "C" __global__ void moe_grouped_gate_up_bf16(const u64* tab, const float* glob, const u16* xs,
                                                    const unsigned* tiles, u16* act, int I, int H) {
    __shared__ __align__(16) u16 sa[MT_M * MT_KP];
    __shared__ __align__(16) u16 sg[MT_N * MT_KP];
    __shared__ __align__(16) u16 su[MT_N * MT_KP];
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = E2M1[threadIdx.x];
    __syncthreads();
    const unsigned e = tiles[3 * blockIdx.x], r0 = tiles[3 * blockIdx.x + 1];
    const int nr = (int)tiles[3 * blockIdx.x + 2];
    const int n0 = blockIdx.y * MT_N;
    const u64* te = tab + (long)e * 6;
    const u8 *gp = (const u8*)te[0], *gs = (const u8*)te[1], *up = (const u8*)te[2], *us = (const u8*)te[3];
    const float gg = glob[e * 3], gu = glob[e * 3 + 1];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, t = lane & 3;
    float cg[8][4], cu[8][4];
    for (int f = 0; f < 8; ++f)
        for (int j = 0; j < 4; ++j) cg[f][j] = cu[f][j] = 0.0f;
    for (int k0 = 0; k0 < H; k0 += MT_K) {
        mt_load_a(xs, r0, nr, H, k0, sa);
        mt_load_w(gp, gs, gg, I, H, n0, k0, sg, lut);
        mt_load_w(up, us, gu, I, H, n0, k0, su, lut);
        __syncthreads();
        for (int kk = 0; kk < MT_K && warp * 16 < nr; kk += 16) {  // (a partial tile's empty warps idle)
            u32 a[4], b[2];
            mt_frag_a(sa, warp, kk, g, t, a);
            for (int f = 0; f < 8; ++f) {
                mt_frag_b(sg, f, kk, g, t, b);
                mt_mma(cg[f], a, b);
                mt_frag_b(su, f, kk, g, t, b);
                mt_mma(cu[f], a, b);
            }
        }
        __syncthreads();
    }
    for (int f = 0; f < 8; ++f)
        for (int j = 0; j < 4; ++j) {
            int row = warp * 16 + g + (j >= 2 ? 8 : 0);
            int col = n0 + f * 8 + 2 * t + (j & 1);
            if (row < nr && col < I) {
                float x = cg[f][j];
                act[(long)(r0 + row) * I + col] = f32_to_bf16(x / (1.0f + expf(-x)) * cu[f][j]);
            }
        }
}

// grid (tiles, ceil(H / MT_N)), MT_THREADS threads
extern "C" __global__ void moe_grouped_down_bf16(const u64* tab, const float* glob, const u16* act,
                                                 const float* wts, const unsigned* tiles, u16* y, int I, int H) {
    __shared__ __align__(16) u16 sa[MT_M * MT_KP];
    __shared__ __align__(16) u16 sd[MT_N * MT_KP];
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = E2M1[threadIdx.x];
    __syncthreads();
    const unsigned e = tiles[3 * blockIdx.x], r0 = tiles[3 * blockIdx.x + 1];
    const int nr = (int)tiles[3 * blockIdx.x + 2];
    const int n0 = blockIdx.y * MT_N;
    const u64* te = tab + (long)e * 6;
    const u8 *dp = (const u8*)te[4], *ds = (const u8*)te[5];
    const float gd = glob[e * 3 + 2];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, t = lane & 3;
    float c[8][4];
    for (int f = 0; f < 8; ++f)
        for (int j = 0; j < 4; ++j) c[f][j] = 0.0f;
    for (int k0 = 0; k0 < I; k0 += MT_K) {
        mt_load_a(act, r0, nr, I, k0, sa);
        mt_load_w(dp, ds, gd, H, I, n0, k0, sd, lut);
        __syncthreads();
        for (int kk = 0; kk < MT_K && warp * 16 < nr; kk += 16) {
            u32 a[4], b[2];
            mt_frag_a(sa, warp, kk, g, t, a);
            for (int f = 0; f < 8; ++f) {
                mt_frag_b(sd, f, kk, g, t, b);
                mt_mma(c[f], a, b);
            }
        }
        __syncthreads();
    }
    for (int f = 0; f < 8; ++f)
        for (int j = 0; j < 4; ++j) {
            int row = warp * 16 + g + (j >= 2 ? 8 : 0);
            int col = n0 + f * 8 + 2 * t + (j & 1);
            if (row < nr && col < H) y[(long)(r0 + row) * H + col] = f32_to_bf16(c[f][j] * wts[r0 + row]);
        }
}
