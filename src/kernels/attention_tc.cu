// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// carbon-engine tensor-core prefill attention (FlashAttention-2 style), compiled with NVRTC for
// compute_80 after attention.cu (dtype helpers). Used for bf16 / f16 on sm_80+ GPUs.
//
// One block per (64-query tile, heads of one KV head); 4 warps of 16 query rows per head.
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

// int8 KV pools (Q8): a row holds, per KV head, D int8 values and the f32 scale that maps them
// back (D + 4 bytes); the kernels convert rows to the activation dtype as they stage them in
// shared memory, so the tensor-core math is the same for both pool formats.
#define Q8_PAD 4

// A small integer (an int8 code) in A's dtype: exact, so no rounding step is needed for bf16.
template <class A>
__device__ __forceinline__ u32 tc_int(int v) {
    u16 x;
    A::store(&x, 0, (float)v);
    return x;
}
template <>
__device__ __forceinline__ u32 tc_int<ABF16>(int v) { return (u32)__float_as_int((float)v) >> 16; }

// Elements e .. e + 3 of pool vector `vec` (slot * KVH + kv head) in A's dtype; int8 pools give
// the raw codes (the caller applies the vector's scale, `tc_pool_scale`, to whatever the codes
// are multiplied into). Done in two steps, so a thread can issue all its loads before it converts any of them
// (a load-convert-store loop waits a cache round trip per item): the raw words of elements
// e .. e + 3, then their values in A's dtype.
template <bool Q8>
__device__ __forceinline__ void tc_pool_raw(const u16* pool, long vec, int D, int e, u32* w) {
    if (Q8) {
        w[0] = *(const u32*)((const signed char*)pool + vec * (D + Q8_PAD) + e);
    } else {
        w[0] = *(const u32*)(pool + vec * D + e);
        w[1] = *(const u32*)(pool + vec * D + e + 2);
    }
}
template <class A, bool Q8>
__device__ __forceinline__ void tc_pool_cvt(const u32* w, u32* x) {
    if (Q8) {
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = tc_int<A>((int)(signed char)(w[0] >> (8 * i)));
    } else {
        x[0] = w[0] & 0xffffu;
        x[1] = w[0] >> 16;
        x[2] = w[1] & 0xffffu;
        x[3] = w[1] >> 16;
    }
}
template <bool Q8>
__device__ __forceinline__ float tc_pool_scale(const u16* pool, long vec, int D) {
    return Q8 ? *(const float*)((const signed char*)pool + vec * (D + Q8_PAD) + D) : 1.0f;
}

#define TC_SB 8   // staging items a thread loads before converting them
#define TC_RV 40  // u16 per row of the transposed V tile: 20 words, conflict-free fragment loads

// One block per (64-query tile, HB heads of one KV head); 4 warps of 16 query rows per head.
// K is staged [key][dim], V transposed [dim][key] so both MMAs load their B fragments as whole
// words. Int8 pools are staged as their codes (exact in bf16 / f16): a key's K scale multiplies
// its score column and its V scale its probability column.
template <class A, int D, int DO, bool Q8>
__device__ void flash_prefill_tc(const u16* q, const u16* kc, const u16* vc, const unsigned* slots,
                                 const unsigned* seqinfo, const unsigned* tiles, u16* out, int H, int KVH,
                                 float scale) {
    constexpr int KS = D / 16;       // k-steps of Q K^T
    constexpr int NT = DO / 8;       // output n-tiles
    constexpr int NB = TC_BK / 8;    // S n-tiles per key tile
    constexpr int RSK = D + 8;       // padded shared rows (conflict-free fragment loads)
    __shared__ u16 sk[TC_BK * RSK];
    __shared__ u16 sv[DO * TC_RV];
    __shared__ float ksc[TC_BK], vsc[TC_BK];

    // A block of HB * 128 threads takes HB consecutive query heads (HB divides G = H / KVH, so
    // they share a KV head), 4 warps each, and stages every K / V tile once for all of them.
    // The kernel's registers cap a block near 256 threads at head dim 256.
    const int HB = blockDim.x / 128;
    const int wq = (threadIdx.x / 32) % 4;  // which 16 of the tile's 64 queries
    int bid = blockIdx.x;
    const int hgroups = H / HB;
    int h = (bid % hgroups) * HB + threadIdx.x / 128, tile = bid / hgroups;
    int kvh = h / (H / KVH);
    unsigned b = tiles[2 * tile], q0 = tiles[2 * tile + 1];
    unsigned q_start = seqinfo[4 * b], q_len = seqinfo[4 * b + 1], s_start = seqinfo[4 * b + 2],
             kv_len = seqinfo[4 * b + 3];
    unsigned prefix = kv_len - q_len;
    unsigned nq = min(q_len - q0, (unsigned)TC_BQ);
    int lane = threadIdx.x % 32, g = lane >> 2, t = lane & 3;
    unsigned r0 = q0 + wq * 16 + g, r1 = r0 + 8;  // query rows (within the sequence) of this lane

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
    unsigned warp_last = prefix + q0 + wq * 16 + 15;
    unsigned kv_end = min(kv_len, prefix + q0 + nq);

    for (unsigned k0 = 0; k0 < kv_end; k0 += TC_BK) {
        int nk = min(kv_end - k0, (unsigned)TC_BK);
        __syncthreads();
        // K: (key row, 4 dims) per item, lanes on consecutive dims; V: (key pair, 4 dims) per
        // item, lanes on the 16 key pairs: word (dim, pair) of the transposed tile. Batches of
        // TC_SB items: every load of a batch is issued before the first conversion.
        for (int base = threadIdx.x; base < TC_BK * (D / 4); base += TC_SB * blockDim.x) {
            u32 w[TC_SB][Q8 ? 1 : 2], sl[TC_SB];
#pragma unroll
            for (int it = 0; it < TC_SB; ++it) {
                int i = base + it * blockDim.x, row = i / (D / 4);
                sl[it] = (i < TC_BK * (D / 4) && row < nk) ? slots[s_start + k0 + row] : 0xffffffffu;
            }
#pragma unroll
            for (int it = 0; it < TC_SB; ++it) {
                int c4 = (base + it * blockDim.x) % (D / 4);
                w[it][0] = 0u;
                if (!Q8) w[it][1] = 0u;
                if (sl[it] != 0xffffffffu) tc_pool_raw<Q8>(kc, (long)sl[it] * KVH + kvh, D, 4 * c4, w[it]);
            }
#pragma unroll
            for (int it = 0; it < TC_SB; ++it) {
                int i = base + it * blockDim.x, row = i / (D / 4), c4 = i % (D / 4);
                if (i < TC_BK * (D / 4)) {
                    u32 x[4];
                    tc_pool_cvt<A, Q8>(w[it], x);
                    u32* dst = (u32*)(sk + row * RSK + 4 * c4);
                    dst[0] = x[0] | (x[1] << 16);
                    dst[1] = x[2] | (x[3] << 16);
                }
            }
        }
        for (int base = threadIdx.x; base < (TC_BK / 2) * (DO / 4); base += (TC_SB / 2) * blockDim.x) {
            // a thread's items share their key pair: two slots
            const int r2 = 2 * (base % (TC_BK / 2));
            const u32 sl[2] = {r2 < nk ? slots[s_start + k0 + r2] : 0xffffffffu,
                               r2 + 1 < nk ? slots[s_start + k0 + r2 + 1] : 0xffffffffu};
            u32 w[TC_SB][Q8 ? 1 : 2];
#pragma unroll
            for (int it = 0; it < TC_SB; ++it) {
                int i = base + (it / 2) * blockDim.x, c4 = i / (TC_BK / 2);
                w[it][0] = 0u;
                if (!Q8) w[it][1] = 0u;
                if (i < (TC_BK / 2) * (DO / 4) && sl[it & 1] != 0xffffffffu)
                    tc_pool_raw<Q8>(vc, (long)sl[it & 1] * KVH + kvh, D, 4 * c4, w[it]);
            }
#pragma unroll
            for (int it = 0; it < TC_SB; it += 2) {
                int i = base + (it / 2) * blockDim.x, rp = i % (TC_BK / 2), c4 = i / (TC_BK / 2);
                if (i < (TC_BK / 2) * (DO / 4)) {
                    u32 x[4], y[4];
                    tc_pool_cvt<A, Q8>(w[it], x);
                    tc_pool_cvt<A, Q8>(w[it + 1], y);
#pragma unroll
                    for (int e = 0; e < 4; ++e) *(u32*)(sv + (4 * c4 + e) * TC_RV + 2 * rp) = x[e] | (y[e] << 16);
                }
            }
        }
        if (Q8 && threadIdx.x < TC_BK) {
            bool in = (int)threadIdx.x < nk;
            long vec = in ? (long)slots[s_start + k0 + threadIdx.x] * KVH + kvh : 0;
            ksc[threadIdx.x] = in ? tc_pool_scale<Q8>(kc, vec, D) : 0.0f;
            vsc[threadIdx.x] = in ? tc_pool_scale<Q8>(vc, vec, D) : 0.0f;
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
            for (int e = 0; e < 2; ++e) {
                int kc_ = nb * 8 + 2 * t + e;
                unsigned key = k0 + kc_;
                float sc = Q8 ? scale * ksc[kc_] : scale;
                s[nb][e] = key <= qpos0 ? s[nb][e] * sc : -1e30f;
                s[nb][e + 2] = key <= qpos1 ? s[nb][e + 2] * sc : -1e30f;
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
                if (e < 2) sum0 += p; else sum1 += p;
                // int8 V rows are staged as codes: their scale goes into the probability
                s[nb][e] = Q8 ? p * vsc[nb * 8 + 2 * t + (e & 1)] : p;
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
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const u16* vp = sv + (nt * 8 + g) * TC_RV + 16 * j + 2 * t;
                u32 bf[2] = {*(const u32*)vp, *(const u32*)(vp + 8)};
                Mma<A>::run(o[nt], pa, bf);
            }
        }
    }
    float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        int d = nt * 8 + 2 * t;
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
        flash_prefill_tc<A, D, D, false>(q, k, v, slots, seqinfo, tiles, out, H, KVH,      \
                                                           scale);                                             \
    }                                                                                                          \
    extern "C" __global__ void flash_prefill_tc_q8_##SUFFIX##_d##D(                                            \
        const u16* q, const u16* k, const u16* v, const unsigned* slots, const unsigned* seqinfo,              \
        const unsigned* tiles, u16* out, int H, int KVH, float scale) {                                        \
        flash_prefill_tc<A, D, D, true>(q, k, v, slots, seqinfo, tiles, out, H, KVH,       \
                                                          scale);                                              \
    }

DEFINE_TC(bf16, ABF16, 64)
DEFINE_TC(bf16, ABF16, 128)
DEFINE_TC(bf16, ABF16, 256)
DEFINE_TC(f16, AF16, 64)
DEFINE_TC(f16, AF16, 128)
DEFINE_TC(f16, AF16, 256)

// ---- decode over long contexts: chunked, one pass over K and V per KV head ----
//
// paged_decode (attention.cu) runs one block per (row, query head): fine for many short
// sequences, but a long context then has 128 threads walking all of it, once per query head.
// Here the context is cut into chunks of PD_WARPS * PD_WP positions, one block each (a warp
// takes PD_WP of them: a lone warp runs a few hundred instructions per tile at its own pace, so
// short warp chunks keep a step's latency down), and the G = H / KVH
// query heads that share a KV head are scored together, so K and V are read once:
//
//   S[16 pos x G]  = K[16 pos x D] Q^T        D/16 mma.m16n8k16 (Q^T fragments in registers)
//   online softmax per head across the chunk's tiles
//   O^T[D x G]    += V^T[D x 16 pos] P^T      D/16 mma.m16n8k16
//
// A tile's 16 K rows, then its 16 V rows, are staged in shared memory with coalesced loads
// (fragments pick single elements of 16 rows; from global memory each load would touch 16 lines).
// A block merges its warps and writes (m, l, O) per head to `part`; paged_decode_merge combines
// the chunks.
//   part [B, KVH, NC, PD_G, 2 + D] f32

#ifndef CARBON_HOST_TEST
#define PD_WP 64
#define PD_WARPS 4
#define PD_G 8
#define PD_MAXD 256

// Copy 16 pool rows into shared memory: all the loads are issued before the first store, so
// their latencies overlap instead of adding up (a store may alias the next row's pointer).
// Q8 rows are converted to A's dtype on the way (4 int8 per load, two staged words).
template <class A, bool Q8>
__device__ __forceinline__ void pd_stage(u32 (*st)[PD_MAXD / 2], const u16* pool, const unsigned* slots,
                                         unsigned t0, unsigned s1, int KVH, int kvh, int D, int lane) {
    long vec[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        unsigned j = t0 + r < s1 ? t0 + r : s1 - 1;
        vec[r] = (long)slots[j] * KVH + kvh;
    }
    if (Q8) {
        const int per = D / 128;  // int8 words per lane and row
        u32 tmp[16][PD_MAXD / 128];
        float sc[16];
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const u32* row = (const u32*)((const signed char*)pool + vec[r] * (D + Q8_PAD));
            sc[r] = *(const float*)(row + D / 4);
#pragma unroll
            for (int i = 0; i < PD_MAXD / 128; ++i) tmp[r][i] = i < per ? row[lane + 32 * i] : 0u;
        }
#pragma unroll
        for (int r = 0; r < 16; ++r) {
#pragma unroll
            for (int i = 0; i < PD_MAXD / 128; ++i) {
                if (i < per) {
                    u32 w = tmp[r][i];
                    int word = 2 * (lane + 32 * i);
                    st[r][word] = tc_pack<A>((float)(signed char)(w & 0xff) * sc[r],
                                             (float)(signed char)((w >> 8) & 0xff) * sc[r]);
                    st[r][word + 1] = tc_pack<A>((float)(signed char)((w >> 16) & 0xff) * sc[r],
                                                 (float)(signed char)(w >> 24) * sc[r]);
                }
            }
        }
        return;
    }
    const int per = D / 64;  // words per lane and row
    u32 tmp[16][PD_MAXD / 64];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const u32* row = (const u32*)(pool + vec[r] * D);
#pragma unroll
        for (int i = 0; i < PD_MAXD / 64; ++i) tmp[r][i] = i < per ? row[lane + 32 * i] : 0u;
    }
#pragma unroll
    for (int r = 0; r < 16; ++r) {
#pragma unroll
        for (int i = 0; i < PD_MAXD / 64; ++i)
            if (i < per) st[r][lane + 32 * i] = tmp[r][i];
    }
}

// After the block's warps left (m, l) in sp and O^T in their stage rows: combine them into the
// chunk's partial at `pb`. Thread i takes elements i, i + 128, ... of the D * PD_G outputs.
__device__ __forceinline__ void pd_block_merge(u32 (*stage)[16][PD_MAXD / 2], float (*sp)[16][PD_G], float* pb,
                                               int D) {
    for (int e = threadIdx.x; e < D * PD_G; e += PD_WARPS * 32) {
        int hh = e % PD_G, d = e / PD_G;
        float mm = -1e30f;
#pragma unroll
        for (int w = 0; w < PD_WARPS; ++w) mm = fmaxf(mm, sp[w][0][hh]);
        float o = 0.0f, ll = 0.0f;
#pragma unroll
        for (int w = 0; w < PD_WARPS; ++w) {
            float wgt = __expf(sp[w][0][hh] - mm);
            o += wgt * ((const float*)stage[w])[e];
            ll += wgt * sp[w][1][hh];
        }
        float* ph = pb + (long)hh * (D + 2);
        ph[2 + d] = o;
        if (d == 0) {
            ph[0] = mm;
            ph[1] = ll;
        }
    }
}

// Row b's slot table is slots[pd_lo .. pd_hi): `starts` holds B + 1 cumulative offsets, or with
// `pairs` an (offset, end) pair per row, so the rows of a multi-token step can share one table
// (row j of a sequence sees a prefix of the last row's table).
__device__ __forceinline__ unsigned pd_lo(const unsigned* starts, int b, int pairs) {
    return pairs ? starts[2 * b] : starts[b];
}
__device__ __forceinline__ unsigned pd_hi(const unsigned* starts, int b, int pairs) {
    return pairs ? starts[2 * b + 1] : starts[b + 1];
}

template <class A, bool Q8>
__device__ void paged_decode_tc(const u16* q, const u16* kc, const u16* vc, const unsigned* slots,
                                const unsigned* starts, float* part, int H, int KVH, int D, float scale, int NCW,
                                int pairs) {
    __shared__ u32 stage[PD_WARPS][16][PD_MAXD / 2];
    __shared__ float sp[PD_WARPS][16][PD_G];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, tg = lane & 3;
    const int G = H / KVH, KS = D / 16;
    const int c = blockIdx.x % NCW;
    const int kvh = (blockIdx.x / NCW) % KVH, b = blockIdx.x / NCW / KVH;
    const unsigned s0 = pd_lo(starts, b, pairs), s1 = pd_hi(starts, b, pairs);
    if (s0 + (unsigned)c * (PD_WARPS * PD_WP) >= s1) return;  // the whole block is past the end
    const unsigned p0 = s0 + (unsigned)c * (PD_WARPS * PD_WP) + (unsigned)warp * PD_WP;
    const unsigned p1 = p0 + PD_WP < s1 ? p0 + PD_WP : s1;

    u32(*st)[PD_MAXD / 2] = stage[warp];
    float(*pw)[PD_G] = sp[warp];
    if (p0 >= s1) {
        // this warp's part of the chunk is empty: weight exp(-inf) in the merge
        if (lane < PD_G) {
            pw[0][lane] = -1e30f;
            pw[1][lane] = 0.0f;
        }
        for (int e = lane; e < D * PD_G; e += 32) ((float*)st)[e] = 0.0f;
        __syncthreads();
        pd_block_merge(stage, sp, part + ((((long)b * KVH + kvh) * NCW + c) * PD_G) * (D + 2), D);
        return;
    }
    u32 qf[PD_MAXD / 16][2];
    const u32* qr = (const u32*)(q + ((long)b * H + (long)kvh * G + (g < G ? g : 0)) * D);
#pragma unroll
    for (int ks = 0; ks < PD_MAXD / 16; ++ks) {
        bool on = ks < KS && g < G;
        qf[ks][0] = on ? qr[8 * ks + tg] : 0u;
        qf[ks][1] = on ? qr[8 * ks + 4 + tg] : 0u;
    }
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.0f, l1 = 0.0f;
    float acc[PD_MAXD / 16][4];
#pragma unroll
    for (int ds = 0; ds < PD_MAXD / 16; ++ds) acc[ds][0] = acc[ds][1] = acc[ds][2] = acc[ds][3] = 0.0f;

    for (unsigned t0 = p0; t0 < p1; t0 += 16) {
        // positions past the end re-read the last one and get zero weight
        pd_stage<A, Q8>(st, kc, slots, t0, s1, KVH, kvh, D, lane);
        __syncwarp();
        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int ks = 0; ks < PD_MAXD / 16; ++ks) {
            if (ks < KS) {
                u32 a[4] = {st[g][8 * ks + tg], st[g + 8][8 * ks + tg], st[g][8 * ks + 4 + tg],
                            st[g + 8][8 * ks + 4 + tg]};
                Mma<A>::run(s, a, qf[ks]);
            }
        }
        bool va = t0 + g < p1, vb = t0 + g + 8 < p1;
        float s00 = va ? s[0] * scale : -1e30f, s01 = va ? s[1] * scale : -1e30f;
        float s10 = vb ? s[2] * scale : -1e30f, s11 = vb ? s[3] * scale : -1e30f;
        // heads 2tg / 2tg+1 live in the 8 lanes with this tg
        float t0m = fmaxf(s00, s10), t1m = fmaxf(s01, s11);
#pragma unroll
        for (int off = 4; off < 32; off <<= 1) {
            t0m = fmaxf(t0m, __shfl_xor_sync(0xffffffff, t0m, off));
            t1m = fmaxf(t1m, __shfl_xor_sync(0xffffffff, t1m, off));
        }
        float n0 = fmaxf(m0, t0m), n1 = fmaxf(m1, t1m);
        float corr0 = __expf(m0 - n0), corr1 = __expf(m1 - n1);
        float p00 = __expf(s00 - n0), p01 = __expf(s01 - n1), p10 = __expf(s10 - n0), p11 = __expf(s11 - n1);
        float ls0 = p00 + p10, ls1 = p01 + p11;
#pragma unroll
        for (int off = 4; off < 32; off <<= 1) {
            ls0 += __shfl_xor_sync(0xffffffff, ls0, off);
            ls1 += __shfl_xor_sync(0xffffffff, ls1, off);
        }
        l0 = l0 * corr0 + ls0;
        l1 = l1 * corr1 + ls1;
        m0 = n0;
        m1 = n1;
        pw[g][2 * tg] = p00;
        pw[g][2 * tg + 1] = p01;
        pw[g + 8][2 * tg] = p10;
        pw[g + 8][2 * tg + 1] = p11;
        __syncwarp();
        pd_stage<A, Q8>(st, vc, slots, t0, s1, KVH, kvh, D, lane);
        __syncwarp();
        // P^T [16 pos x G]: rows 2tg, 2tg+1 (and +8) of column g
        u32 pf[2] = {tc_pack<A>(pw[2 * tg][g], pw[2 * tg + 1][g]), tc_pack<A>(pw[8 + 2 * tg][g], pw[9 + 2 * tg][g])};
        const u16* v0 = (const u16*)st[2 * tg];
        const u16* v1 = (const u16*)st[2 * tg + 1];
        const u16* v2 = (const u16*)st[8 + 2 * tg];
        const u16* v3 = (const u16*)st[9 + 2 * tg];
#pragma unroll
        for (int ds = 0; ds < PD_MAXD / 16; ++ds) {
            if (ds < KS) {
                acc[ds][0] *= corr0;
                acc[ds][1] *= corr1;
                acc[ds][2] *= corr0;
                acc[ds][3] *= corr1;
                int da = 16 * ds + g, db = da + 8;
                u32 a[4] = {(u32)v0[da] | ((u32)v1[da] << 16), (u32)v0[db] | ((u32)v1[db] << 16),
                            (u32)v2[da] | ((u32)v3[da] << 16), (u32)v2[db] | ((u32)v3[db] << 16)};
                Mma<A>::run(acc[ds], a, pf);
            }
        }
        __syncwarp();
    }
    // merge the block's warps: each leaves (m, l) in sp and O^T [D x PD_G] in its stage rows
    float* so = (float*)st;
    if (g == 0) {
        pw[0][2 * tg] = m0;
        pw[0][2 * tg + 1] = m1;
        pw[1][2 * tg] = l0;
        pw[1][2 * tg + 1] = l1;
    }
#pragma unroll
    for (int ds = 0; ds < PD_MAXD / 16; ++ds) {
        if (ds < KS) {
            int da = 16 * ds + g, db = da + 8;
            so[da * PD_G + 2 * tg] = acc[ds][0];
            so[da * PD_G + 2 * tg + 1] = acc[ds][1];
            so[db * PD_G + 2 * tg] = acc[ds][2];
            so[db * PD_G + 2 * tg + 1] = acc[ds][3];
        }
    }
    __syncthreads();
    pd_block_merge(stage, sp, part + ((((long)b * KVH + kvh) * NCW + c) * PD_G) * (D + 2), D);
}

// grid B * H blocks of D threads: thread d sums its element over the row's chunks
template <class A>
__device__ void paged_decode_merge(const float* part, const unsigned* starts, u16* out, int H, int KVH, int D,
                                   int NCW, int pairs) {
    const int b = blockIdx.x / H, h = blockIdx.x % H, d = threadIdx.x;
    if (d >= D) return;
    const int G = H / KVH, kvh = h / G, hh = h % G;
    const unsigned len = pd_hi(starts, b, pairs) - pd_lo(starts, b, pairs);
    int nc = (int)((len + PD_WARPS * PD_WP - 1) / (PD_WARPS * PD_WP));
    if (nc > NCW) nc = NCW;
    const float* p0 = part + ((((long)b * KVH + kvh) * NCW) * PD_G + hh) * (D + 2);
    const long step = (long)PD_G * (D + 2);
    float mm = -1e30f;
    for (int c = 0; c < nc; ++c) mm = fmaxf(mm, p0[c * step]);
    float ll = 0.0f, o = 0.0f;
    for (int c = 0; c < nc; ++c) {
        float w = __expf(p0[c * step] - mm);
        ll += w * p0[c * step + 1];
        o += w * p0[c * step + 2 + d];
    }
    A::store(out, ((long)b * H + h) * D + d, ll > 0.0f ? o / ll : 0.0f);
}

// ---- decode, several token rows per sequence (speculative verify steps) over long contexts ----
//
// paged_decode_tc reads K and V once per row. Here a block takes the R rows of one sequence
// (consecutive rows, linked by `links` as in gdn.cu: bit 1 = not the first row, >> 8 = rows that
// follow) and a chunk of PDR_POS positions; every tile is staged once and the block's warps
// score all R * G (row, head) columns against it, 8 columns per warp. Row j of the group sees
// the first len_j positions of the last row's slot table (its own table is a prefix of it).
// Each warp writes its columns' partials for paged_decode_merge (same layout).
#define PDR_WARPS 6   // 48 columns: 8 rows of 6 heads
#define PDR_POS 256

#define PDR_KW (PD_MAXD / 2 + 4)  // words per staged K row: padded, conflict-free fragment loads
#define PDR_VS 24                 // u16 per row of the transposed V tile (12 words, likewise)
#define PDR_SB 8                  // staging items a thread loads before converting them

// Per 16-position tile the block stages K [pos][dim] and V transposed [dim][pos] in one pass
// (every thread issues a batch of loads before it converts any), int8 pools as their codes:
// a position's K scale multiplies its scores and its V scale its probabilities.
template <class A, bool Q8>
__device__ void paged_decode_tc_rows(const u16* q, const u16* kc, const u16* vc, const unsigned* slots,
                                     const unsigned* starts, const unsigned* links, float* part, int H, int KVH,
                                     int D, float scale, int NCW, int pairs) {
    __shared__ u32 sk[16][PDR_KW];
    __shared__ u16 sv[PD_MAXD * PDR_VS];
    __shared__ float ksc[16], vsc[16];
    __shared__ float sp[PDR_WARPS][16][PD_G];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, tg = lane & 3;
    const int G = H / KVH, KS = D / 16;
    const int c = blockIdx.x % NCW;
    const int kvh = (blockIdx.x / NCW) % KVH, b = blockIdx.x / NCW / KVH;
    const unsigned link = links[b];
    if (link & 2u) return;  // the group's first row computes this one
    const int R = 1 + (int)(link >> 8);
    // the last row's table covers every row of the group
    const unsigned s0 = pd_lo(starts, b + R - 1, pairs), s1 = pd_hi(starts, b + R - 1, pairs);
    const unsigned p0 = s0 + (unsigned)c * PDR_POS;
    if (p0 >= s1) return;
    const unsigned p1 = p0 + PDR_POS < s1 ? p0 + PDR_POS : s1;
    // this lane feeds column 8 * warp + g (scores) and owns columns 8 * warp + 2tg, + 1 (results)
    const int ncol = R * G;
    const int colq = 8 * warp + g, col0 = 8 * warp + 2 * tg, col1 = col0 + 1;
    const bool live0 = col0 < ncol, live1 = col1 < ncol;
    // a block may have more warps than its columns need: the others only help staging
    const bool work = 8 * warp < ncol;
    const int j0 = live0 ? col0 / G : 0, j1 = live1 ? col1 / G : 0;
    // positions (relative to s0) row j may see
    const unsigned len0 = pd_hi(starts, b + j0, pairs) - pd_lo(starts, b + j0, pairs);
    const unsigned len1 = pd_hi(starts, b + j1, pairs) - pd_lo(starts, b + j1, pairs);

    u32 qf[PD_MAXD / 16][2];
    const bool qon = colq < ncol;
    const u32* qr = (const u32*)(q + ((long)(b + (qon ? colq / G : 0)) * H + (long)kvh * G + (qon ? colq % G : 0)) * D);
#pragma unroll
    for (int ks = 0; ks < PD_MAXD / 16; ++ks) {
        bool on = ks < KS && qon;
        qf[ks][0] = on ? qr[8 * ks + tg] : 0u;
        qf[ks][1] = on ? qr[8 * ks + 4 + tg] : 0u;
    }
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.0f, l1 = 0.0f;
    float acc[PD_MAXD / 16][4];
#pragma unroll
    for (int ds = 0; ds < PD_MAXD / 16; ++ds) acc[ds][0] = acc[ds][1] = acc[ds][2] = acc[ds][3] = 0.0f;
    float(*pw)[PD_G] = sp[warp];
    constexpr int W = Q8 ? 1 : 2;
    const int KI = 16 * (D / 4), VI = 8 * (D / 4);  // staging items: (row, 4 dims), (row pair, 4 dims)

    for (unsigned t0 = p0; t0 < p1; t0 += 16) {
        const int nk = (int)min(s1 - t0, 16u);
        for (int base = threadIdx.x; base < KI; base += PDR_SB * blockDim.x) {
            u32 sl[PDR_SB], w[PDR_SB][W];
#pragma unroll
            for (int it = 0; it < PDR_SB; ++it) {
                int i = base + it * blockDim.x, row = i / (D / 4);
                sl[it] = (i < KI && row < nk) ? slots[t0 + row] : 0xffffffffu;
            }
#pragma unroll
            for (int it = 0; it < PDR_SB; ++it) {
                int c4 = (base + it * blockDim.x) % (D / 4);
#pragma unroll
                for (int x = 0; x < W; ++x) w[it][x] = 0u;
                if (sl[it] != 0xffffffffu) tc_pool_raw<Q8>(kc, (long)sl[it] * KVH + kvh, D, 4 * c4, w[it]);
            }
#pragma unroll
            for (int it = 0; it < PDR_SB; ++it) {
                int i = base + it * blockDim.x, row = i / (D / 4), c4 = i % (D / 4);
                if (i < KI) {
                    u32 x[4];
                    tc_pool_cvt<A, Q8>(w[it], x);
                    sk[row][2 * c4] = x[0] | (x[1] << 16);
                    sk[row][2 * c4 + 1] = x[2] | (x[3] << 16);
                }
            }
        }
        for (int base = threadIdx.x; base < VI; base += (PDR_SB / 2) * blockDim.x) {
            u32 sl[PDR_SB], w[PDR_SB][W];
#pragma unroll
            for (int it = 0; it < PDR_SB; ++it) {
                int i = base + (it / 2) * blockDim.x, row = 2 * (i % 8) + (it & 1);
                sl[it] = (i < VI && row < nk) ? slots[t0 + row] : 0xffffffffu;
            }
#pragma unroll
            for (int it = 0; it < PDR_SB; ++it) {
                int c4 = (base + (it / 2) * blockDim.x) / 8;
#pragma unroll
                for (int x = 0; x < W; ++x) w[it][x] = 0u;
                if (sl[it] != 0xffffffffu) tc_pool_raw<Q8>(vc, (long)sl[it] * KVH + kvh, D, 4 * c4, w[it]);
            }
#pragma unroll
            for (int it = 0; it < PDR_SB; it += 2) {
                int i = base + (it / 2) * blockDim.x, rp = i % 8, c4 = i / 8;
                if (i < VI) {
                    u32 x[4], y[4];
                    tc_pool_cvt<A, Q8>(w[it], x);
                    tc_pool_cvt<A, Q8>(w[it + 1], y);
#pragma unroll
                    for (int e = 0; e < 4; ++e) *(u32*)(sv + (4 * c4 + e) * PDR_VS + 2 * rp) = x[e] | (y[e] << 16);
                }
            }
        }
        if (Q8 && threadIdx.x < 16) {
            const bool in = (int)threadIdx.x < nk;
            const long vec = in ? (long)slots[t0 + threadIdx.x] * KVH + kvh : 0;
            ksc[threadIdx.x] = in ? tc_pool_scale<Q8>(kc, vec, D) : 0.0f;
            vsc[threadIdx.x] = in ? tc_pool_scale<Q8>(vc, vec, D) : 0.0f;
        }
        __syncthreads();
        if (!work) {
            __syncthreads();
            continue;
        }
        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int ks = 0; ks < PD_MAXD / 16; ++ks) {
            if (ks < KS) {
                u32 a[4] = {sk[g][8 * ks + tg], sk[g + 8][8 * ks + tg], sk[g][8 * ks + 4 + tg],
                            sk[g + 8][8 * ks + 4 + tg]};
                Mma<A>::run(s, a, qf[ks]);
            }
        }
        // positions g and g + 8 of the tile, relative to the table's start
        const unsigned ia = t0 - s0 + g, ib = ia + 8;
        const float sca = Q8 ? scale * ksc[g] : scale, scb = Q8 ? scale * ksc[g + 8] : scale;
        float s00 = (live0 && ia < len0) ? s[0] * sca : -1e30f, s01 = (live1 && ia < len1) ? s[1] * sca : -1e30f;
        float s10 = (live0 && ib < len0) ? s[2] * scb : -1e30f, s11 = (live1 && ib < len1) ? s[3] * scb : -1e30f;
        float t0m = fmaxf(s00, s10), t1m = fmaxf(s01, s11);
#pragma unroll
        for (int off = 4; off < 32; off <<= 1) {
            t0m = fmaxf(t0m, __shfl_xor_sync(0xffffffff, t0m, off));
            t1m = fmaxf(t1m, __shfl_xor_sync(0xffffffff, t1m, off));
        }
        float n0 = fmaxf(m0, t0m), n1 = fmaxf(m1, t1m);
        float corr0 = __expf(m0 - n0), corr1 = __expf(m1 - n1);
        // a masked score gets weight 0 even while the running max is still the mask value
        float p00 = s00 > -1e29f ? __expf(s00 - n0) : 0.0f, p01 = s01 > -1e29f ? __expf(s01 - n1) : 0.0f;
        float p10 = s10 > -1e29f ? __expf(s10 - n0) : 0.0f, p11 = s11 > -1e29f ? __expf(s11 - n1) : 0.0f;
        float ls0 = p00 + p10, ls1 = p01 + p11;
#pragma unroll
        for (int off = 4; off < 32; off <<= 1) {
            ls0 += __shfl_xor_sync(0xffffffff, ls0, off);
            ls1 += __shfl_xor_sync(0xffffffff, ls1, off);
        }
        l0 = l0 * corr0 + ls0;
        l1 = l1 * corr1 + ls1;
        m0 = n0;
        m1 = n1;
        // int8 V rows are staged as codes: their scale goes into the probabilities
        const float va = Q8 ? vsc[g] : 1.0f, vb = Q8 ? vsc[g + 8] : 1.0f;
        pw[g][2 * tg] = p00 * va;
        pw[g][2 * tg + 1] = p01 * va;
        pw[g + 8][2 * tg] = p10 * vb;
        pw[g + 8][2 * tg + 1] = p11 * vb;
        __syncwarp();
        // P^T [16 pos x 8 columns]: rows 2tg, 2tg+1 (and +8) of column g
        u32 pf[2] = {tc_pack<A>(pw[2 * tg][g], pw[2 * tg + 1][g]), tc_pack<A>(pw[8 + 2 * tg][g], pw[9 + 2 * tg][g])};
#pragma unroll
        for (int ds = 0; ds < PD_MAXD / 16; ++ds) {
            if (ds < KS) {
                acc[ds][0] *= corr0;
                acc[ds][1] *= corr1;
                acc[ds][2] *= corr0;
                acc[ds][3] *= corr1;
                const u16* va_ = sv + (16 * ds + g) * PDR_VS + 2 * tg;
                const u16* vb_ = va_ + 8 * PDR_VS;
                u32 a[4] = {*(const u32*)va_, *(const u32*)vb_, *(const u32*)(va_ + 8), *(const u32*)(vb_ + 8)};
                Mma<A>::run(acc[ds], a, pf);
            }
        }
        __syncthreads();  // every warp is done with the staged tile
    }
    // partials of this lane's two columns: (row b + j, head) in paged_decode_merge's layout
    float* h0 = part + ((((long)(b + j0) * KVH + kvh) * NCW + c) * PD_G + col0 % G) * (D + 2);
    float* h1 = part + ((((long)(b + j1) * KVH + kvh) * NCW + c) * PD_G + col1 % G) * (D + 2);
    if (g == 0) {
        if (live0) {
            h0[0] = m0;
            h0[1] = l0;
        }
        if (live1) {
            h1[0] = m1;
            h1[1] = l1;
        }
    }
#pragma unroll
    for (int ds = 0; ds < PD_MAXD / 16; ++ds) {
        if (ds < KS) {
            int da = 16 * ds + g, db = da + 8;
            if (live0) {
                h0[2 + da] = acc[ds][0];
                h0[2 + db] = acc[ds][2];
            }
            if (live1) {
                h1[2 + da] = acc[ds][1];
                h1[2 + db] = acc[ds][3];
            }
        }
    }
}

// Quantize new K or V rows into a Q8 pool. grid T * KVH blocks of D threads; x [T, KVH * D],
// idx [T, W]: the slot of row t is idx[t * W] (the batch's scatter table).
template <class A>
__device__ void kv_q8_write(const u16* x, signed char* pool, const unsigned* idx, int KVH, int D, int W) {
    __shared__ float red[8];
    const int t = blockIdx.x / KVH, kvh = blockIdx.x % KVH, d = threadIdx.x;
    const bool on = d < D;
    float v = on ? A::load(x, ((long)t * KVH + kvh) * D + d) : 0.0f;
    float m = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
    __syncthreads();
    float mx = 0.0f;
    for (int w = 0; w < (int)(blockDim.x / 32); ++w) mx = fmaxf(mx, red[w]);
    signed char* row = pool + ((long)idx[(long)t * W] * KVH + kvh) * (D + Q8_PAD);
    if (on) row[d] = (signed char)__float2int_rn(mx > 0.0f ? v * (127.0f / mx) : 0.0f);
    if (d == 0) *(float*)(row + D) = mx / 127.0f;
}

#define DEFINE_PD_TC(SUFFIX, A)                                                                               \
    extern "C" __global__ void paged_decode_tc_##SUFFIX(const u16* q, const u16* k, const u16* v,            \
                                                        const unsigned* slots, const unsigned* starts,        \
                                                        float* part, int H, int KVH, int D, float scale,      \
                                                        int NCW, int pairs) {                                 \
        paged_decode_tc<A, false>(q, k, v, slots, starts, part, H, KVH, D, scale, NCW, pairs);                \
    }                                                                                                         \
    extern "C" __global__ void paged_decode_tc_q8_##SUFFIX(const u16* q, const u16* k, const u16* v,         \
                                                           const unsigned* slots, const unsigned* starts,     \
                                                           float* part, int H, int KVH, int D, float scale,   \
                                                           int NCW, int pairs) {                              \
        paged_decode_tc<A, true>(q, k, v, slots, starts, part, H, KVH, D, scale, NCW, pairs);                 \
    }                                                                                                         \
    extern "C" __global__ void paged_decode_tc_rows_##SUFFIX(const u16* q, const u16* k, const u16* v,       \
                                                             const unsigned* slots, const unsigned* starts,   \
                                                             const unsigned* links, float* part, int H,       \
                                                             int KVH, int D, float scale, int NCW,            \
                                                             int pairs) {                                     \
        paged_decode_tc_rows<A, false>(q, k, v, slots, starts, links, part, H, KVH, D, scale, NCW, pairs);    \
    }                                                                                                         \
    extern "C" __global__ void paged_decode_tc_rows_q8_##SUFFIX(const u16* q, const u16* k, const u16* v,    \
                                                                const unsigned* slots, const unsigned* starts,\
                                                                const unsigned* links, float* part, int H,    \
                                                                int KVH, int D, float scale, int NCW,         \
                                                                int pairs) {                                  \
        paged_decode_tc_rows<A, true>(q, k, v, slots, starts, links, part, H, KVH, D, scale, NCW, pairs);     \
    }                                                                                                         \
    extern "C" __global__ void kv_q8_write_##SUFFIX(const u16* x, signed char* pool, const unsigned* idx,    \
                                                    int KVH, int D, int W) {                                  \
        kv_q8_write<A>(x, pool, idx, KVH, D, W);                                                              \
    }                                                                                                         \
    extern "C" __global__ void paged_decode_merge_##SUFFIX(const float* part, const unsigned* starts,        \
                                                           u16* out, int H, int KVH, int D, int NCW,          \
                                                           int pairs) {                                       \
        paged_decode_merge<A>(part, starts, out, H, KVH, D, NCW, pairs);                                      \
    }

DEFINE_PD_TC(bf16, ABF16)
DEFINE_PD_TC(f16, AF16)
#endif
