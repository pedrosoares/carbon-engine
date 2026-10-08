// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// carbon-engine tensor-core prefill attention (FlashAttention-2 style), compiled with NVRTC for
// compute_80 after attention.cu (dtype helpers). Used for bf16 / f16 on sm_80+ GPUs.
//
// One block per (64-query tile, head, slice of up to 128 output dims); 4 warps of 16 query rows.
// Per 32-key tile staged in shared memory through the slot table:
//   S = Q K^T     mma.m16n8k16 with the warp's Q fragments held in registers
//   online softmax on the S accumulators (rows g and g+8 of each lane's quad)
//   O += P V      the S accumulators are re-packed in place as the A operand (no shared round trip)
// Fragment layouts follow the PTX ISA for mma.m16n8k16 .row.col with f32 accumulators.

typedef unsigned int u32;

#ifndef CARBON_HOST_TEST
template <class A> struct Mma;
template <> struct Mma<ABF16> {
    __device__ static void run(float* c, const u32* a, const u32* b) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};
template <> struct Mma<AF16> {
    __device__ static void run(float* c, const u32* a, const u32* b) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
};
#else
// host emulation: the warp exchanges its fragments and every lane evaluates its 4 outputs
template <class A> struct Mma {
    static void run(float* c, const u32* a, const u32* b) { mma_emulate<A>(c, a, b); }
};
#endif

template <class A>
__device__ __forceinline__ u32 tc_pack(float lo, float hi) {
    u16 x, y;
    A::store(&x, 0, lo);
    A::store(&y, 0, hi);
    return (u32)x | ((u32)y << 16);
}

#define TC_BQ 64
#define TC_BK 32

template <class A, int D, int DO>
__device__ void flash_prefill_tc(const u16* q, const u16* kc, const u16* vc, const unsigned* slots,
                                 const unsigned* seqinfo, const unsigned* tiles, u16* out, int H, int KVH,
                                 float scale) {
    constexpr int KS = D / 16;       // k-steps of Q K^T
    constexpr int NT = DO / 8;       // output n-tiles
    constexpr int NB = TC_BK / 8;    // S n-tiles per key tile
    constexpr int RSK = D + 8;       // padded shared rows (conflict-free fragment loads)
    constexpr int RSV = DO + 8;
    constexpr int NSPLIT = D / DO;
    __shared__ u16 sk[TC_BK * RSK];
    __shared__ u16 sv[TC_BK * RSV];

    int bid = blockIdx.x;
    int ds = bid % NSPLIT;
    bid /= NSPLIT;
    int h = bid % H, tile = bid / H;
    int kvh = h / (H / KVH);
    unsigned b = tiles[2 * tile], q0 = tiles[2 * tile + 1];
    unsigned q_start = seqinfo[4 * b], q_len = seqinfo[4 * b + 1], s_start = seqinfo[4 * b + 2],
             kv_len = seqinfo[4 * b + 3];
    unsigned prefix = kv_len - q_len;
    unsigned nq = min(q_len - q0, (unsigned)TC_BQ);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, t = lane & 3;
    unsigned r0 = q0 + warp * 16 + g, r1 = r0 + 8;  // query rows (within the sequence) of this lane

    // Q fragments (A operand, row-major 16 x 16 per k-step)
    u32 qa[KS][4];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            unsigned row = (r & 1) ? r1 : r0;
            int col = ks * 16 + 2 * t + ((r & 2) ? 8 : 0);
            qa[ks][r] = row < q0 + nq ? *(const u32*)(q + ((long)(q_start + row) * H + h) * D + col) : 0u;
        }
    }
    float o[NT][4];
#pragma unroll
    for (int i = 0; i < NT; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.0f, l1 = 0.0f;
    unsigned qpos0 = prefix + r0, qpos1 = prefix + r1;
    unsigned warp_last = prefix + q0 + warp * 16 + 15;
    unsigned kv_end = min(kv_len, prefix + q0 + nq);

    for (unsigned k0 = 0; k0 < kv_end; k0 += TC_BK) {
        int nk = min(kv_end - k0, (unsigned)TC_BK);
        __syncthreads();
        for (int i = threadIdx.x; i < TC_BK * (D / 2); i += blockDim.x) {
            int row = i / (D / 2), cw = i % (D / 2);
            u32 w = 0;
            if (row < nk) w = *(const u32*)(kc + ((long)slots[s_start + k0 + row] * KVH + kvh) * D + 2 * cw);
            *(u32*)(sk + row * RSK + 2 * cw) = w;
        }
        for (int i = threadIdx.x; i < TC_BK * (DO / 2); i += blockDim.x) {
            int row = i / (DO / 2), cw = i % (DO / 2);
            u32 w = 0;
            if (row < nk)
                w = *(const u32*)(vc + ((long)slots[s_start + k0 + row] * KVH + kvh) * D + ds * DO + 2 * cw);
            *(u32*)(sv + row * RSV + 2 * cw) = w;
        }
        __syncthreads();
        if (k0 > warp_last) continue;  // causal: every key of this tile is after the warp's queries

        float s[NB][4];
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
            s[nb][0] = s[nb][1] = s[nb][2] = s[nb][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < KS; ++ks) {
                const u16* kr = sk + (nb * 8 + g) * RSK + ks * 16 + 2 * t;
                u32 bf[2] = {*(const u32*)kr, *(const u32*)(kr + 8)};
                Mma<A>::run(s[nb], qa[ks], bf);
            }
        }
        // scale + causal mask, then the online softmax for rows r0 (elements 0,1) and r1 (2,3)
        float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                unsigned key = k0 + nb * 8 + 2 * t + (e & 1);
                bool valid = key <= (e < 2 ? qpos0 : qpos1);
                s[nb][e] = valid ? s[nb][e] * scale : -1e30f;
            }
            mx0 = fmaxf(mx0, fmaxf(s[nb][0], s[nb][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nb][2], s[nb][3]));
        }
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, off));
        }
        float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        float c0 = __expf(m0 - mn0), c1 = __expf(m1 - mn1);
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float mm = e < 2 ? mn0 : mn1;
                float p = s[nb][e] > -1e29f ? __expf(s[nb][e] - mm) : 0.0f;
                s[nb][e] = p;
                if (e < 2) sum0 += p; else sum1 += p;
            }
        }
        for (int off = 1; off < 4; off <<= 1) {
            sum0 += __shfl_xor_sync(0xffffffff, sum0, off);
            sum1 += __shfl_xor_sync(0xffffffff, sum1, off);
        }
        l0 = l0 * c0 + sum0;
        l1 = l1 * c1 + sum1;
        m0 = mn0;
        m1 = mn1;
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            o[i][0] *= c0;
            o[i][1] *= c0;
            o[i][2] *= c1;
            o[i][3] *= c1;
        }
        // O += P V: P's accumulators become A fragments (two S n-tiles per 16-key k-step)
#pragma unroll
        for (int j = 0; j < NB / 2; ++j) {
            u32 pa[4] = {tc_pack<A>(s[2 * j][0], s[2 * j][1]), tc_pack<A>(s[2 * j][2], s[2 * j][3]),
                         tc_pack<A>(s[2 * j + 1][0], s[2 * j + 1][1]), tc_pack<A>(s[2 * j + 1][2], s[2 * j + 1][3])};
            const u16* v0 = sv + (16 * j + 2 * t) * RSV + g;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const u16* vp = v0 + nt * 8;
                u32 bf[2] = {(u32)vp[0] | ((u32)vp[RSV] << 16), (u32)vp[8 * RSV] | ((u32)vp[9 * RSV] << 16)};
                Mma<A>::run(o[nt], pa, bf);
            }
        }
    }
    float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        int d = ds * DO + nt * 8 + 2 * t;
        if (r0 < q0 + nq) {
            long base = ((long)(q_start + r0) * H + h) * D + d;
            A::store(out, base, o[nt][0] * inv0);
            A::store(out, base + 1, o[nt][1] * inv0);
        }
        if (r1 < q0 + nq) {
            long base = ((long)(q_start + r1) * H + h) * D + d;
            A::store(out, base, o[nt][2] * inv1);
            A::store(out, base + 1, o[nt][3] * inv1);
        }
    }
}

#define DEFINE_TC(SUFFIX, A, D)                                                                                \
    extern "C" __global__ void flash_prefill_tc_##SUFFIX##_d##D(                                               \
        const u16* q, const u16* k, const u16* v, const unsigned* slots, const unsigned* seqinfo,              \
        const unsigned* tiles, u16* out, int H, int KVH, float scale) {                                        \
        flash_prefill_tc<A, D, (D > 128 ? 128 : D)>(q, k, v, slots, seqinfo, tiles, out, H, KVH, scale);      \
    }

DEFINE_TC(bf16, ABF16, 64)
DEFINE_TC(bf16, ABF16, 128)
DEFINE_TC(bf16, ABF16, 256)
DEFINE_TC(f16, AF16, 64)
DEFINE_TC(f16, AF16, 128)
DEFINE_TC(f16, AF16, 256)
