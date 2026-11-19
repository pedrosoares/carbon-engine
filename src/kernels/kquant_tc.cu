// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// K-quant GEMM for multi-token decode steps on integer tensor cores (sm_80+), compiled after
// quant.cu. y[t, r] = sum_c x[t, c] * W[r, c] for up to 8 rows of x at about the cost of one:
// a warp takes 16 weight rows and walks the columns; per 32 (Q4_K) or 16 (Q6_K) columns one
// mma.sync multiplies the 16 rows' codes (u8) with the 8 tokens' 8-bit activations (s8, from
// kq_prep), and the integer sums are scaled per row and sub-block into f32 accumulators.
//
// Fragment layouts (PTX ISA, .row.col, lane = 4 * group + tg):
//   m16n8k32  A[16x32] u8: reg0 row g cols 4tg..+3, reg1 row g+8 same cols,
//                          reg2 row g cols 16+4tg..+3, reg3 row g+8 same cols
//             B[32x8]  s8: reg0 rows 4tg..+3, reg1 rows 16+4tg..+3, column g
//   m16n8k16  A[16x16] u8: reg0 row g cols 4tg..+3, reg1 row g+8 same cols
//             B[16x8]  s8: reg0 rows 4tg..+3, column g
//   C[16x8] s32: c0, c1 row g cols 2tg, 2tg+1; c2, c3 row g+8 same cols
// So a lane owns rows (g, g+8) and tokens (2tg, 2tg+1) of the result and feeds token g's
// activations.
//
// Each super-block of the warp's 16 rows is first copied to shared memory with coalesced loads
// (all issued before the first store): fragments pick 4 bytes out of each row, which from global
// memory touches 16 lines per load and halves the throughput once the weights exceed the cache.

#define KQ_TC_WARPS 4
#define KQ_STAGE_WORDS (Q6K_BLOCK / 4)  // the larger block

// Copy block `blk` (W words) of 16 consecutive rows into st; rows past the end repeat row n-1.
__device__ __forceinline__ void kq_stage(unsigned (*st)[KQ_STAGE_WORDS], const unsigned* data, long r0, long n,
                                         long row_words, int blk, int W, int lane) {
    unsigned t0[16], t1[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        long row = r0 + r < n ? r0 + r : n - 1;
        const unsigned* b = data + row * row_words + (long)blk * W;
        t0[r] = b[lane];
        t1[r] = lane + 32 < W ? b[lane + 32] : 0u;
    }
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        st[r][lane] = t0[r];
        if (lane + 32 < W) st[r][lane + 32] = t1[r];
    }
}

__device__ __forceinline__ void kq_mma32(int* c, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void kq_mma16(int* c, const unsigned* a, unsigned b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
}

__device__ __forceinline__ long kq_minl(long a, long b) { return a < b ? a : b; }
__device__ __forceinline__ int kq_mini(int a, int b) { return a < b ? a : b; }

// split = 1: each warp of a block takes its own 16 rows and all the columns.
// split = KQ_TC_WARPS: the block's warps share 16 rows and take a quarter of the columns each
// (wide weights with few rows: a warp walks its columns serially), then add up their sums.
// `tile`: the block's (split > 1) or the warp's 16-row tile of this weight.
template <class D>
__device__ __forceinline__ void kq_gemm_tc_tile(const u8* data, const float* prep, typename D::T* y, int n, int k,
                                                int t, int kind, int split, long tile,
                                                unsigned (*stage)[16][KQ_STAGE_WORDS], float (*red)[32][4]) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const long r0 = tile * 16;
    if (r0 >= n) return;
    // this warp's super-blocks
    const int nblk = k / 256;
    const int blk0 = split > 1 ? (int)((long)nblk * warp / split) : 0;
    const int blk1 = split > 1 ? (int)((long)nblk * (warp + 1) / split) : nblk;
    const int g = lane >> 2, tg = lane & 3;
    unsigned (*st)[KQ_STAGE_WORDS] = stage[warp];
    // tokens past the end read token t-1's scales and are not stored; their activations are zero
    const int ta = kq_mini(2 * tg, t - 1), tb = kq_mini(2 * tg + 1, t - 1);
    const int stride = KQ_PREP_STRIDE(k);
    const unsigned* xq = (const unsigned*)(prep + (long)kq_mini(g, t - 1) * stride);
    const unsigned live = g < t ? 0xffffffffu : 0u;
    const float* pa = prep + (long)ta * stride + k / 4;
    const float* pb = prep + (long)tb * stride + k / 4;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (kind == 0) {
        const long row_words = (long)(k / 256) * (Q4K_BLOCK / 4);
        for (int blk = blk0; blk < blk1; ++blk) {
            kq_stage(st, (const unsigned*)data, r0, n, row_words, blk, Q4K_BLOCK / 4, lane);
            __syncwarp();
            const unsigned* ba = st[g];
            const unsigned* bb = st[g + 8];
            unsigned ha[4] = {ba[0], ba[1], ba[2], ba[3]}, hb[4] = {bb[0], bb[1], bb[2], bb[3]};
            float da = f16_to_f32((u16)(ha[0] & 0xffff)), dma = f16_to_f32((u16)(ha[0] >> 16));
            float db = f16_to_f32((u16)(hb[0] & 0xffff)), dmb = f16_to_f32((u16)(hb[0] >> 16));
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const int u = blk * 4 + pair;
                const unsigned* qa = ba + 4 + 8 * pair;
                const unsigned* qb = bb + 4 + 8 * pair;
                unsigned w[4] = {qa[tg], qb[tg], qa[4 + tg], qb[4 + tg]};
                unsigned lo[4], hi[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    lo[i] = w[i] & 0x0F0F0F0Fu;
                    hi[i] = (w[i] >> 4) & 0x0F0F0F0Fu;
                }
                unsigned xl[2] = {xq[u * 16 + tg] & live, xq[u * 16 + 4 + tg] & live};
                unsigned xh[2] = {xq[u * 16 + 8 + tg] & live, xq[u * 16 + 12 + tg] & live};
                int cl[4] = {0, 0, 0, 0}, ch[4] = {0, 0, 0, 0};
                kq_mma32(cl, lo, xl);
                kq_mma32(ch, hi, xh);
                float s1a, m1a, s2a, m2a, s1b, m1b, s2b, m2b;
                q4k_scale_min(2 * pair, (const u8*)(ha + 1), da, dma, &s1a, &m1a);
                q4k_scale_min(2 * pair + 1, (const u8*)(ha + 1), da, dma, &s2a, &m2a);
                q4k_scale_min(2 * pair, (const u8*)(hb + 1), db, dmb, &s1b, &m1b);
                q4k_scale_min(2 * pair + 1, (const u8*)(hb + 1), db, dmb, &s2b, &m2b);
                // token ta / tb: scale of each sub-block's activations and their sum
                const float* sa = pa + k / 32 + 4 * u;
                const float* sb = pb + k / 32 + 4 * u;
                float xa0 = pa[2 * u], xa1 = pa[2 * u + 1], xb0 = pb[2 * u], xb1 = pb[2 * u + 1];
                float ya0 = sa[0] + sa[1], ya1 = sa[2] + sa[3], yb0 = sb[0] + sb[1], yb1 = sb[2] + sb[3];
                acc[0] += kq_term_q4k(s1a, m1a, xa0, ya0, cl[0]) + kq_term_q4k(s2a, m2a, xa1, ya1, ch[0]);
                acc[1] += kq_term_q4k(s1a, m1a, xb0, yb0, cl[1]) + kq_term_q4k(s2a, m2a, xb1, yb1, ch[1]);
                acc[2] += kq_term_q4k(s1b, m1b, xa0, ya0, cl[2]) + kq_term_q4k(s2b, m2b, xa1, ya1, ch[2]);
                acc[3] += kq_term_q4k(s1b, m1b, xb0, yb0, cl[3]) + kq_term_q4k(s2b, m2b, xb1, yb1, ch[3]);
            }
            __syncwarp();
        }
    } else {
        const long row_words = (long)(k / 256) * (Q6K_BLOCK / 4);
        for (int hb2 = 2 * blk0; hb2 < 2 * blk1; ++hb2) {
            const int blk = hb2 >> 1, half = hb2 & 1;
            if (half == 0) {
                __syncwarp();
                kq_stage(st, (const unsigned*)data, r0, n, row_words, blk, Q6K_BLOCK / 4, lane);
                __syncwarp();
            }
            const unsigned* ba = st[g];
            const unsigned* bb = st[g + 8];
            const unsigned* qla = ba + 16 * half;
            const unsigned* qlb = bb + 16 * half;
            // codes of weights 4tg..4tg+3 of each 16-column group j (0..3 within 64 columns:
            // ql words tg, 4+tg, 8+tg, 12+tg; the high bit pairs of groups j and j+2 share a word)
            unsigned la[4] = {qla[tg], qla[4 + tg], qla[8 + tg], qla[12 + tg]};
            unsigned lb[4] = {qlb[tg], qlb[4 + tg], qlb[8 + tg], qlb[12 + tg]};
            unsigned qa[2] = {ba[32 + 8 * half + tg], ba[32 + 8 * half + 4 + tg]};
            unsigned qb[2] = {bb[32 + 8 * half + tg], bb[32 + 8 * half + 4 + tg]};
            const signed char* sca = (const signed char*)(ba + 48) + 8 * half;
            const signed char* scb = (const signed char*)(bb + 48) + 8 * half;
            float da = f16_to_f32((u16)(ba[52] & 0xffff)), db = f16_to_f32((u16)(bb[52] & 0xffff));
#pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                const int u = hb2 * 2 + hi, sh = 4 * hi;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    // group j: columns u*64 + 16j; j < 2 are the first 32 weights (ql words j,
                    // high bits qh >> sh), j >= 2 the next 32 (ql words j, high bits qh >> (sh+2))
                    const int hs = sh + 2 * (j >> 1);
                    unsigned a[2] = {
                        ((la[j] >> sh) & 0x0F0F0F0Fu) | (((qa[j & 1] >> hs) & 0x03030303u) << 4),
                        ((lb[j] >> sh) & 0x0F0F0F0Fu) | (((qb[j & 1] >> hs) & 0x03030303u) << 4),
                    };
                    int c[4] = {0, 0, 0, 0};
                    kq_mma16(c, a, xq[u * 16 + 4 * j + tg] & live);
                    // scale index: is (= j & 1) + 2 * which, which = 2 * hi + (j >> 1)
                    const int si = (j & 1) + 4 * hi + 2 * (j >> 1);
                    float fa = da * (float)sca[si], fb = db * (float)scb[si];
                    float xa = pa[2 * u + (j >> 1)], xb = pb[2 * u + (j >> 1)];
                    float ya = pa[k / 32 + 4 * u + j], yb = pb[k / 32 + 4 * u + j];
                    acc[0] += kq_term_q6k(fa, xa, ya, c[0]);
                    acc[1] += kq_term_q6k(fa, xb, yb, c[1]);
                    acc[2] += kq_term_q6k(fb, xa, ya, c[2]);
                    acc[3] += kq_term_q6k(fb, xb, yb, c[3]);
                }
            }
        }
    }
    if (split > 1) {
        red[warp][lane][0] = acc[0];
        red[warp][lane][1] = acc[1];
        red[warp][lane][2] = acc[2];
        red[warp][lane][3] = acc[3];
        __syncthreads();
        if (warp != 0) return;
#pragma unroll
        for (int w = 1; w < KQ_TC_WARPS; ++w) {
            acc[0] += red[w][lane][0];
            acc[1] += red[w][lane][1];
            acc[2] += red[w][lane][2];
            acc[3] += red[w][lane][3];
        }
    }
    const int t0 = 2 * tg;
    if (r0 + g < n) {
        if (t0 < t) D::store(y, (long)t0 * n + r0 + g, acc[0]);
        if (t0 + 1 < t) D::store(y, (long)(t0 + 1) * n + r0 + g, acc[1]);
    }
    if (r0 + g + 8 < n) {
        if (t0 < t) D::store(y, (long)t0 * n + r0 + g + 8, acc[2]);
        if (t0 + 1 < t) D::store(y, (long)(t0 + 1) * n + r0 + g + 8, acc[3]);
    }
}


template <class D>
__device__ void kq_gemm_tc(const u8* data, const float* prep, typename D::T* y, int n, int k, int t, int kind,
                           int split) {
    __shared__ unsigned stage[KQ_TC_WARPS][16][KQ_STAGE_WORDS];
    __shared__ float red[KQ_TC_WARPS][32][4];
    const long tile = split > 1 ? (long)blockIdx.x : (long)blockIdx.x * KQ_TC_WARPS + threadIdx.x / 32;
    kq_gemm_tc_tile<D>(data, prep, y, n, k, t, kind, split, tile, stage, red);
}

// Up to 4 weights that read the same activations in one launch (see kq_gemv_i8_multi); each
// weight's rows are padded to whole 16-row tiles in the numbering. split = 1 only.
template <class D>
__device__ void kq_gemm_tc_multi(const u8* d0, const u8* d1, const u8* d2, const u8* d3, const float* prep,
                                 typename D::T* y, int n0, int n1, int n2, int n3, int kinds, int k, int t) {
    __shared__ unsigned stage[KQ_TC_WARPS][16][KQ_STAGE_WORDS];
    __shared__ float red[KQ_TC_WARPS][32][4];
    long tile = (long)blockIdx.x * KQ_TC_WARPS + threadIdx.x / 32;
    const u8* data[4] = {d0, d1, d2, d3};
    const int n[4] = {n0, n1, n2, n3};
    long before = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        long tiles = (n[s] + 15) / 16;
        if (tile < tiles) {
            kq_gemm_tc_tile<D>(data[s], prep, y + before * t, n[s], k, t, (kinds >> (4 * s)) & 15, 1, tile, stage, red);
            return;
        }
        tile -= tiles;
        before += n[s];
    }
}

#define DEFINE_KQ_TC_MULTI(SUFFIX, D)                                                                        \
    extern "C" __global__ void kq_gemm_tc_multi_##SUFFIX(const u8* d0, const u8* d1, const u8* d2,           \
                                                         const u8* d3, const float* p, D::T* y, int n0,      \
                                                         int n1, int n2, int n3, int kinds, int k, int t) {  \
        kq_gemm_tc_multi<D>(d0, d1, d2, d3, p, y, n0, n1, n2, n3, kinds, k, t);                               \
    }

DEFINE_KQ_TC_MULTI(f32, F32)
DEFINE_KQ_TC_MULTI(bf16, BF16)
DEFINE_KQ_TC_MULTI(f16, F16)

#define DEFINE_KQ_TC(SUFFIX, D)                                                                              \
    extern "C" __global__ void kq_gemm_tc_##SUFFIX(const u8* w, const float* p, D::T* y, int n, int k,       \
                                                   int t, int kind, int split) {                              \
        kq_gemm_tc<D>(w, p, y, n, k, t, kind, split);                                                         \
    }

DEFINE_KQ_TC(f32, F32)
DEFINE_KQ_TC(bf16, BF16)
DEFINE_KQ_TC(f16, F16)
