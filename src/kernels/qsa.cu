// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// QSA decode selection (Qwen3.8 sparse attention), one block of QSA_THREADS per sequence.
//
// A sequence with more complete blocks of R tokens than K is restricted to its K best blocks
// plus the trailing partial block; otherwise it attends to every token. Per block j the
// indexer key is the mean of the raw keys of its R tokens, RMS-normed (weights already 1+w),
// then roped at position j*R; its score is sum over the NH indexer query heads of
// relu(q_h . key) / sqrt(D). The best K scores win, ties to the lower block index (the host
// path's order). Selected blocks are written in ascending order, then the partial block.
//
// The output is a slot table for carbon's paged decode kernel: sequence b's selection at
// out_starts[b]..out_starts[b+1]. Its length depends only on the sequence length, so the
// tables have a fixed capacity and a decode step stays CUDA-graph capturable.
//
//   q      [B, NH, D]   indexer queries (normed + roped), model dtype
//   pool   [slots, D]   raw indexer keys, model dtype
//   knorm  [D]          k_layernorm weight (1 + w), model dtype
//   cos/sin [P, HALF]   rope tables, model dtype; the first 2*HALF dims rotate (rotate-half)
//   slots/starts        every sequence's slot table (starts [B+1])
//   scores [B, MAXNB]   f32 scratch
//   out_slots [B*(K+1)*R], out_starts [B+1]

#define QSA_THREADS 256
#define QSA_WARPS (QSA_THREADS / 32)
#define QSA_MAX_D 256
#define QSA_MAX_NH 8

__device__ __forceinline__ unsigned int qsa_key(float f) {
    unsigned int u = (unsigned int)__float_as_int(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ unsigned qsa_count(unsigned len, int K, int R) {
    return (len / R > (unsigned)K) ? (unsigned)(K * R) + len % R : len;
}

// exclusive scan of v over the block; returns this thread's prefix, *total gets the sum
__device__ int qsa_scan(int v, int* buf, int* total) {
    int t = threadIdx.x;
    buf[t] = v;
    __syncthreads();
    for (int off = 1; off < QSA_THREADS; off <<= 1) {
        int add = t >= off ? buf[t - off] : 0;
        __syncthreads();
        buf[t] += add;
        __syncthreads();
    }
    int incl = buf[t];
    *total = buf[QSA_THREADS - 1];
    __syncthreads();
    return incl - v;
}

template <class D>
__device__ void qsa_select(const typename D::T* q, const typename D::T* pool, const typename D::T* knorm,
                           const typename D::T* cosv, const typename D::T* sinv, const unsigned* slots,
                           const unsigned* starts, float* scores, unsigned* out_slots, unsigned* out_starts,
                           int NH, int Dm, int HALF, int K, int R, int MAXNB, float eps) {
    __shared__ float sq[QSA_MAX_NH * QSA_MAX_D];
    __shared__ float sk[QSA_WARPS][QSA_MAX_D];
    __shared__ unsigned hist[256];
    __shared__ int scan[QSA_THREADS];
    __shared__ int misc[4];
    const int b = blockIdx.x, t = threadIdx.x, warp = t / 32, lane = t % 32;
    const unsigned s0 = starts[b], len = starts[b + 1] - s0;
    const unsigned* sl = slots + s0;
    const int nb = (int)(len / R);
    unsigned off = 0;
    for (int i = 0; i < b; ++i) off += qsa_count(starts[i + 1] - starts[i], K, R);
    if (t == 0) {
        if (b == 0) out_starts[0] = 0;
        out_starts[b + 1] = off + qsa_count(len, K, R);
    }
    unsigned* out = out_slots + off;
    if (nb <= K) {  // dense: the whole history
        for (unsigned i = t; i < len; i += QSA_THREADS) out[i] = sl[i];
        return;
    }
    for (int i = t; i < NH * Dm; i += QSA_THREADS) sq[i] = D::load(q, (long)b * NH * Dm + i);
    __syncthreads();
    float* sc = scores + (long)b * MAXNB;
    const float inv_sqrt_d = rsqrtf((float)Dm);
    float* k = sk[warp];
    for (int j = warp; j < nb; j += QSA_WARPS) {
        // mean of the block's raw keys, rounded to the model dtype as the host path does
        float ss = 0.0f;
        for (int i = lane; i < Dm; i += 32) {
            float acc = 0.0f;
            for (int r = 0; r < R; ++r) acc += D::load(pool, (long)sl[j * R + r] * Dm + i);
            typename D::T tmp;
            D::store(&tmp, 0, acc / (float)R);
            float v = D::load(&tmp, 0);
            k[i] = v;
            ss += v * v;
        }
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
        float inv = rsqrtf(ss / (float)Dm + eps);
        __syncwarp();
        for (int i = lane; i < Dm; i += 32) {
            typename D::T tmp;
            D::store(&tmp, 0, k[i] * inv * D::load(knorm, i));
            k[i] = D::load(&tmp, 0);
        }
        __syncwarp();
        // rope at the block's first position (rotate-half over the first 2*HALF dims)
        float rot[QSA_MAX_D / 32];
        long pos = (long)j * R;
        for (int i = lane, n = 0; i < Dm; i += 32, ++n) {
            float v = k[i];
            if (i < 2 * HALF) {
                int f = i % HALF;
                float c = D::load(cosv, pos * HALF + f), s = D::load(sinv, pos * HALF + f);
                v = i < HALF ? k[i] * c - k[i + HALF] * s : k[i] * c + k[i - HALF] * s;
                typename D::T tmp;
                D::store(&tmp, 0, v);
                v = D::load(&tmp, 0);
            }
            rot[n] = v;
        }
        float total = 0.0f;
        for (int h = 0; h < NH; ++h) {
            float d = 0.0f;
            for (int i = lane, n = 0; i < Dm; i += 32, ++n) d += sq[h * Dm + i] * rot[n];
            for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffff, d, o);
            total += fmaxf(d, 0.0f);
        }
        if (lane == 0) sc[j] = total * inv_sqrt_d;
        __syncwarp();
    }
    __syncthreads();

    // key of the K-th largest score (8-bit radix select), and how many of its ties to take
    unsigned prefix = 0, mask = 0;
    int remaining = K;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += QSA_THREADS) hist[i] = 0;
        __syncthreads();
        for (int j = t; j < nb; j += QSA_THREADS) {
            unsigned key = qsa_key(sc[j]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (t == 0) {
            int digit = 0;
            for (int d = 255; d >= 0; --d) {
                if (remaining <= (int)hist[d]) {
                    digit = d;
                    break;
                }
                remaining -= (int)hist[d];
            }
            misc[0] = digit;
            misc[1] = remaining;
        }
        __syncthreads();
        prefix |= ((unsigned)misc[0]) << shift;
        mask |= 255u << shift;
        remaining = misc[1];
        __syncthreads();
    }
    // `remaining` blocks equal to the threshold are taken, lowest index first
    int taken = 0, ties = 0;
    for (int base = 0; base < nb; base += QSA_THREADS) {
        int j = base + t;
        unsigned key = j < nb ? qsa_key(sc[j]) : 0u;
        int eq = (j < nb && key == prefix) ? 1 : 0;
        int nt;
        int eq_before = qsa_scan(eq, scan, &nt);
        int take = (j < nb && (key > prefix || (eq && ties + eq_before < remaining))) ? 1 : 0;
        int tot;
        int at = taken + qsa_scan(take, scan, &tot);
        if (take)
            for (int r = 0; r < R; ++r) out[(long)at * R + r] = sl[j * R + r];
        taken += tot;
        ties += nt;
    }
    for (unsigned i = t; i < len - (unsigned)nb * R; i += QSA_THREADS) out[(long)K * R + i] = sl[nb * R + i];
}

#define DEFINE_QSA(SUFFIX, D)                                                                                  \
    extern "C" __global__ void qsa_select_##SUFFIX(                                                            \
        const D::T* q, const D::T* pool, const D::T* knorm, const D::T* cosv, const D::T* sinv,               \
        const unsigned* slots, const unsigned* starts, float* scores, unsigned* out_slots, unsigned* out_starts, \
        int NH, int Dm, int HALF, int K, int R, int MAXNB, float eps) {                                        \
        qsa_select<D>(q, pool, knorm, cosv, sinv, slots, starts, scores, out_slots, out_starts, NH, Dm, HALF, K, \
                      R, MAXNB, eps);                                                                          \
    }

DEFINE_QSA(f32, F32)
DEFINE_QSA(bf16, BF16)
DEFINE_QSA(f16, F16)

// QSA prefill selection: per query row of a chunk, its best K visible blocks as a bitmask.
// Query i sits at position first + i and sees its (first + i + 1) / R complete blocks; with no
// more than K of them every one is set (the whole history). Same order as the decode kernel:
// best K scores, ties to the lower block index.
//
//   scores [N, NB]  f32, block scores (sum over heads of relu(q . key) / sqrt(D))
//   bits   [N, W]   u32, W = ceil(NB / 32), written whole
extern "C" __global__ void qsa_topk_bits(const float* scores, unsigned* bits, int NB, int W, int K, int R,
                                         int first) {
    __shared__ unsigned hist[256];
    __shared__ int scan[QSA_THREADS];
    __shared__ int misc[4];
    const int row = blockIdx.x, t = threadIdx.x;
    const float* sc = scores + (long)row * NB;
    unsigned* out = bits + (long)row * W;
    int vis = (first + row + 1) / R;
    if (vis > NB) vis = NB;
    for (int w = t; w < W; w += QSA_THREADS) {
        int lo = w * 32;
        unsigned v = 0u;
        if (vis <= K && lo < vis) v = (vis - lo >= 32) ? 0xffffffffu : ((1u << (vis - lo)) - 1u);
        out[w] = v;
    }
    if (vis <= K) return;
    __syncthreads();
    unsigned prefix = 0, mask = 0;
    int remaining = K;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += QSA_THREADS) hist[i] = 0;
        __syncthreads();
        for (int j = t; j < vis; j += QSA_THREADS) {
            unsigned key = qsa_key(sc[j]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (t == 0) {
            int digit = 0;
            for (int d = 255; d >= 0; --d) {
                if (remaining <= (int)hist[d]) {
                    digit = d;
                    break;
                }
                remaining -= (int)hist[d];
            }
            misc[0] = digit;
            misc[1] = remaining;
        }
        __syncthreads();
        prefix |= ((unsigned)misc[0]) << shift;
        mask |= 255u << shift;
        remaining = misc[1];
        __syncthreads();
    }
    int ties = 0;
    for (int base = 0; base < vis; base += QSA_THREADS) {
        int j = base + t;
        unsigned key = j < vis ? qsa_key(sc[j]) : 0u;
        int eq = (j < vis && key == prefix) ? 1 : 0;
        int nt;
        int eq_before = qsa_scan(eq, scan, &nt);
        if (j < vis && (key > prefix || (eq && ties + eq_before < remaining))) atomicOr(&out[j >> 5], 1u << (j & 31));
        ties += nt;
    }
}

// Additive attention mask [N, L] (f32) from qsa_topk_bits rows: 0 for the selected blocks' tokens
// and the trailing partial block up to the query, `neg` elsewhere (including the future).
extern "C" __global__ void qsa_bits_mask(const unsigned* bits, float* mask, int W, int L, int R, int first,
                                         float neg) {
    const int row = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= L) return;
    const int pos = first + row;
    float v = neg;
    if (j <= pos) {
        int blk = j / R;
        if (blk >= (pos + 1) / R || ((bits[(long)row * W + (blk >> 5)] >> (blk & 31)) & 1u)) v = 0.0f;
    }
    mask[(long)row * L + j] = v;
}
