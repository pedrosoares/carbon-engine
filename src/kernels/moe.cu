// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// carbon-engine MoE decode kernels, compiled with NVRTC after quant.cu (shares its decoders).
//
// A step's MoE layer runs in three launches with no host round trip:
//   moe_topk      softmax + top-k routing per token (one warp per token)
//   moe_gate_up   act[p, n] = silu(gate_e[n] . x_t) * (up_e[n] . x_t) for every (token, expert) pair p
//   moe_down      y[t, h] = sum_k w[p] * (down_e[h] . act[p]) over the token's k pairs
// Expert weights are not moved: a per-layer table holds each expert's raw device pointers
// [gate_w, gate_s, up_w, up_s, down_w, down_s] (0 = expert owned by another expert-parallel
// rank, which then contributes nothing here) and globals [gate_g, up_g, down_g] for NVFP4.

typedef unsigned long long u64;

#define MOE_MAX_EXPERTS 1024
#define FMT_DENSE 0
#define FMT_FP8 1
#define FMT_NVFP4 2

struct Geo { int sc, b0, b1; };  // FP8 scale grid: columns, block rows, block cols

template <class T>
__device__ __forceinline__ T warp_sum(T v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// E2M1 code -> f32 by bit assembly (a __constant__ table serialises when lanes differ).
__device__ __forceinline__ float e2m1_f32(unsigned c) {
    unsigned e = (c >> 1) & 3, m = c & 1;
    unsigned bits = e ? (((e + 126u) << 23) | (m << 22)) : (m ? (126u << 23) : 0u);
    return __int_as_float(((c & 8u) << 28) | bits);
}

// Row `row` of an [n, k] weight dotted with x[0..k); the result is in every lane.
template <class D, class X, int FMT>
__device__ float wdot(u64 w, u64 s, float g, long row, int k, Geo geo, const typename X::T* x) {
    int lane = threadIdx.x % 32;
    float acc = 0.0f;
    if (FMT == FMT_DENSE) {
        const typename D::T* W = (const typename D::T*)w + row * k;
        for (int c = lane; c < k; c += 32) acc += D::load(W, c) * X::load(x, c);
    } else if (FMT == FMT_FP8) {
        const u8* W = (const u8*)w + row * k;
        const float* srow = (const float*)s + (row / geo.b0) * geo.sc;
        for (int c = lane; c < k; c += 32) acc += e4m3_to_f32(W[c]) * srow[c / geo.b1] * X::load(x, c);
    } else {
        const u8* P = (const u8*)w + row * (k / 2);
        const u8* S = (const u8*)s + row * (k / 16);
        for (int blk = lane; blk < k / 16; blk += 32) {
            float sc = e4m3_to_f32(S[blk]);
            float part = 0.0f;
            for (int j = 0; j < 8; ++j) {
                u8 code = P[blk * 8 + j];
                int col = blk * 16 + 2 * j;
                part += e2m1_f32(code & 15) * X::load(x, col) + e2m1_f32(code >> 4) * X::load(x, col + 1);
            }
            acc += sc * part;
        }
        acc *= g;
    }
    return warp_sum(acc);
}

// logits [T, E] f32 -> ids [T, K], weights [T, K]; ties go to the lower expert index.
extern "C" __global__ void moe_topk(const float* logits, unsigned* ids, float* wts, int E, int K, int norm) {
    __shared__ unsigned char taken[MOE_MAX_EXPERTS];
    int t = blockIdx.x, lane = threadIdx.x % 32;
    const float* row = logits + (long)t * E;
    float mx = -3.4e38f;
    for (int j = lane; j < E; j += 32) {
        taken[j] = 0;
        mx = fmaxf(mx, row[j]);
    }
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
    float sum = 0.0f;
    for (int j = lane; j < E; j += 32) sum += expf(row[j] - mx);
    sum = warp_sum(sum);
    __syncwarp();
    float topsum = 0.0f;
    for (int r = 0; r < K; ++r) {
        float best = -1.0f;
        int bi = 0x7fffffff;
        for (int j = lane; j < E; j += 32) {
            if (taken[j]) continue;
            float v = expf(row[j] - mx);
            if (v > best) { best = v; bi = j; }  // j ascends per lane, so ties keep the lower index
        }
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, best, off);
            int oi = __shfl_xor_sync(0xffffffff, bi, off);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (lane == 0) {
            ids[(long)t * K + r] = (unsigned)bi;
            wts[(long)t * K + r] = best;
            taken[bi] = 1;
        }
        topsum += best;
        __syncwarp();
    }
    if (lane == 0) {
        float inv = 1.0f / (norm ? topsum : sum);
        for (int r = 0; r < K; ++r) wts[(long)t * K + r] *= inv;
    }
}

// grid (ceil(I / warps), P): one warp per (intermediate row, pair).
template <class D, int FMT>
__device__ void moe_gate_up(const u64* tab, const float* glob, const typename D::T* x, const unsigned* ids,
                            float* act, int K, int I, int H, Geo geo) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    long n = (long)blockIdx.x * (blockDim.x / 32) + warp;
    int p = blockIdx.y;
    if (n >= I) return;
    unsigned e = ids[p];
    const u64* te = tab + (long)e * 6;
    float* o = act + (long)p * I + n;
    if (te[0] == 0) {
        if (lane == 0) *o = 0.0f;
        return;
    }
    const typename D::T* xt = x + (long)(p / K) * H;
    float g = wdot<D, D, FMT>(te[0], te[1], glob[e * 3], n, H, geo, xt);
    float u = wdot<D, D, FMT>(te[2], te[3], glob[e * 3 + 1], n, H, geo, xt);
    if (lane == 0) *o = g / (1.0f + expf(-g)) * u;
}

// grid (ceil(H / warps), T): one warp per (hidden row, token), summing the token's K experts.
template <class D, int FMT>
__device__ void moe_down(const u64* tab, const float* glob, const float* act, const unsigned* ids, const float* wts,
                         typename D::T* y, int K, int I, int H, Geo geo) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    long h = (long)blockIdx.x * (blockDim.x / 32) + warp;
    int t = blockIdx.y;
    if (h >= H) return;
    float acc = 0.0f;
    for (int r = 0; r < K; ++r) {
        long p = (long)t * K + r;
        unsigned e = ids[p];
        const u64* te = tab + (long)e * 6;
        if (te[4] == 0) continue;
        acc += wts[p] * wdot<D, F32, FMT>(te[4], te[5], glob[e * 3 + 2], h, I, geo, act + p * I);
    }
    if (lane == 0) D::store(y, (long)t * H + h, acc);
}

// ---- NVFP4 fast path: 8 warps per block, activations staged in shared memory as f32 ----

// Eight packed codes (one u32, low nibble first) against x[0..8).
__device__ __forceinline__ float e2m1_dot8(unsigned v, const float* x) {
    float4 a = *(const float4*)x, b = *(const float4*)(x + 4);
    return e2m1_f32(v & 15) * a.x + e2m1_f32((v >> 4) & 15) * a.y + e2m1_f32((v >> 8) & 15) * a.z +
           e2m1_f32((v >> 12) & 15) * a.w + e2m1_f32((v >> 16) & 15) * b.x + e2m1_f32((v >> 20) & 15) * b.y +
           e2m1_f32((v >> 24) & 15) * b.z + e2m1_f32(v >> 28) * b.w;
}

// Row `row` of an NVFP4 [n, k] weight dotted with xs[0..k) (16-byte aligned f32); result in every lane.
__device__ __forceinline__ float nvfp4_row_dot(u64 w, u64 s, long row, int k, const float* xs) {
    const uint2* P = (const uint2*)((const u8*)w + row * (k / 2));
    const u8* S = (const u8*)s + row * (k / 16);
    float acc = 0.0f;
    for (int blk = threadIdx.x % 32; blk < k / 16; blk += 32) {
        uint2 v = P[blk];
        const float* x = xs + blk * 16;
        acc += e4m3_to_f32(S[blk]) * (e2m1_dot8(v.x, x) + e2m1_dot8(v.y, x + 8));
    }
    return warp_sum(acc);
}

// grid (ceil(I / 8), P), 256 threads, shared H floats: one warp per intermediate row of pair p.
template <class D>
__device__ void moe_gate_up_v(const u64* tab, const float* glob, const typename D::T* x, const unsigned* ids,
                              float* act, int K, int I, int H) {
    extern __shared__ float xs[];
    int p = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const typename D::T* xt = x + (long)(p / K) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x) xs[c] = D::load(xt, c);
    __syncthreads();
    long n = (long)blockIdx.x * (blockDim.x / 32) + warp;
    if (n >= I) return;
    unsigned e = ids[p];
    const u64* te = tab + (long)e * 6;
    float* o = act + (long)p * I + n;
    if (te[0] == 0) {
        if (lane == 0) *o = 0.0f;
        return;
    }
    float g = nvfp4_row_dot(te[0], te[1], n, H, xs) * glob[e * 3];
    float u = nvfp4_row_dot(te[2], te[3], n, H, xs) * glob[e * 3 + 1];
    if (lane == 0) *o = g / (1.0f + expf(-g)) * u;
}

// grid (ceil(H / 8), T), 256 threads, shared K*I floats: one warp per hidden row of token t.
template <class D>
__device__ void moe_down_v(const u64* tab, const float* glob, const float* act, const unsigned* ids,
                           const float* wts, typename D::T* y, int K, int I, int H) {
    extern __shared__ float as[];
    int t = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float* at = act + (long)t * K * I;
    for (int c = threadIdx.x; c < K * I; c += blockDim.x) as[c] = at[c];
    __syncthreads();
    long h = (long)blockIdx.x * (blockDim.x / 32) + warp;
    if (h >= H) return;
    float acc = 0.0f;
    for (int r = 0; r < K; ++r) {
        long p = (long)t * K + r;
        unsigned e = ids[p];
        const u64* te = tab + (long)e * 6;
        if (te[4] == 0) continue;
        acc += wts[p] * glob[e * 3 + 2] * nvfp4_row_dot(te[4], te[5], h, I, as + r * I);
    }
    if (lane == 0) D::store(y, (long)t * H + h, acc);
}

#define DEFINE_MOE_V(SUFFIX, D)                                                                                \
    extern "C" __global__ void moe_gate_up_nvfp4v_##SUFFIX(const u64* tab, const float* glob, const D::T* x,  \
                                                            const unsigned* ids, float* act, int K, int I,     \
                                                            int H) {                                           \
        moe_gate_up_v<D>(tab, glob, x, ids, act, K, I, H);                                                    \
    }                                                                                                          \
    extern "C" __global__ void moe_down_nvfp4v_##SUFFIX(const u64* tab, const float* glob, const float* act,   \
                                                         const unsigned* ids, const float* wts, D::T* y,       \
                                                         int K, int I, int H) {                                \
        moe_down_v<D>(tab, glob, act, ids, wts, y, K, I, H);                                                  \
    }

DEFINE_MOE_V(f32, F32)
DEFINE_MOE_V(bf16, BF16)
DEFINE_MOE_V(f16, F16)

#define DEFINE_MOE(SUFFIX, D, FNAME, FMT)                                                                       \
    extern "C" __global__ void moe_gate_up_##FNAME##_##SUFFIX(const u64* tab, const float* glob, const D::T* x, \
                                                               const unsigned* ids, float* act, int K, int I,    \
                                                               int H, int sc, int b0, int b1) {                  \
        Geo geo = {sc, b0, b1};                                                                                  \
        moe_gate_up<D, FMT>(tab, glob, x, ids, act, K, I, H, geo);                                               \
    }                                                                                                            \
    extern "C" __global__ void moe_down_##FNAME##_##SUFFIX(const u64* tab, const float* glob, const float* act,  \
                                                            const unsigned* ids, const float* wts, D::T* y,      \
                                                            int K, int I, int H, int sc, int b0, int b1) {       \
        Geo geo = {sc, b0, b1};                                                                                  \
        moe_down<D, FMT>(tab, glob, act, ids, wts, y, K, I, H, geo);                                             \
    }

#define DEFINE_MOE_ALL(SUFFIX, D)              \
    DEFINE_MOE(SUFFIX, D, dense, FMT_DENSE)    \
    DEFINE_MOE(SUFFIX, D, fp8, FMT_FP8)        \
    DEFINE_MOE(SUFFIX, D, nvfp4, FMT_NVFP4)

DEFINE_MOE_ALL(f32, F32)
DEFINE_MOE_ALL(bf16, BF16)
DEFINE_MOE_ALL(f16, F16)

// ---- Offloaded experts: a device slot cache fed from pinned host memory ----
//
// Every MoE layer of a stage shares one bank of S slots per expert component (gate/up/down
// weight + scale). A layer's routed experts are made resident with no host round trip:
//   moe_slot_ensure  marks hits, gives each missing expert the least recently used slot that this
//                    step does not need, queues it for copy and points the layer's table entries
//                    (the ones moe_gate_up/moe_down read) at the slots
//   moe_slot_copy    copies the queued experts from pinned host memory (read through its device
//                    mapping) into their slots; its grid is fixed, the queue length is read on
//                    the device, so a decode step stays capturable as a CUDA graph.
// `desc` holds per layer: src[6] (host addresses of the layer's component arrays, expert-major),
// dst[6] (bank base addresses, slot-major) and bytes[6] (one expert's component; 0 = absent).

// One block of MOE_ENSURE_THREADS threads. `route` = ids [P] then weights [P] (moe_topk's layout).
#define MOE_ENSURE_THREADS 1024
extern "C" __global__ void moe_slot_ensure(const unsigned* route, int P, int E, int S, int row, int* slot_of,
                                           int* owner, u64* usage, u64* step, u64* stats, u64* tab,
                                           const u64* desc, unsigned* copy_n, unsigned* copy_slot,
                                           unsigned* copy_exp) {
    __shared__ int seen[MOE_MAX_EXPERTS];
    __shared__ unsigned miss[MOE_MAX_EXPERTS];
    __shared__ int nmiss, nhit, victim;
    __shared__ u64 st;
    __shared__ u64 red_u[MOE_ENSURE_THREADS / 32];
    __shared__ int red_i[MOE_ENSURE_THREADS / 32];
    int tid = threadIdx.x, nt = blockDim.x, lane = tid % 32, warp = tid / 32;
    if (tid == 0) {
        st = *step + 1;
        *step = st;
        nmiss = 0;
        nhit = 0;
    }
    for (int e = tid; e < E; e += nt) seen[e] = 0;
    __syncthreads();
    int* so = slot_of + (long)row * E;
    for (int p = tid; p < P; p += nt) {
        unsigned e = route[p];
        if (e < (unsigned)E && atomicExch(&seen[e], 1) == 0) {
            int s = so[e];
            if (s >= 0) {
                usage[s] = st;
                atomicAdd(&nhit, 1);
            } else {
                miss[atomicAdd(&nmiss, 1)] = e;
            }
        }
    }
    __syncthreads();
    // every thread keeps the least recently used slot among its own (s = tid mod nt) that this
    // step does not use; usage 0 = never filled, so empty slots go first
    const u64 NONE = ~0ull;
    u64 lu = NONE;
    int li = 0x7fffffff;
    for (int s = tid; s < S; s += nt) {
        u64 u = usage[s];
        if (u != st && u < lu) { lu = u; li = s; }
    }
    int n = nmiss;
    int done = 0;
    for (int i = 0; i < n; ++i) {
        u64 bu = lu;
        int bi = li;
        for (int off = 16; off > 0; off >>= 1) {
            u64 ou = __shfl_xor_sync(0xffffffff, bu, off);
            int oi = __shfl_xor_sync(0xffffffff, bi, off);
            if (ou < bu || (ou == bu && oi < bi)) { bu = ou; bi = oi; }
        }
        if (lane == 0) { red_u[warp] = bu; red_i[warp] = bi; }
        __syncthreads();
        if (warp == 0) {
            int nw = nt / 32;
            bu = lane < nw ? red_u[lane] : NONE;
            bi = lane < nw ? red_i[lane] : 0x7fffffff;
            for (int off = 16; off > 0; off >>= 1) {
                u64 ou = __shfl_xor_sync(0xffffffff, bu, off);
                int oi = __shfl_xor_sync(0xffffffff, bi, off);
                if (ou < bu || (ou == bu && oi < bi)) { bu = ou; bi = oi; }
            }
            if (lane == 0) {
                if (bu == NONE) {
                    victim = -1;  // every slot is in use this step: the cache is smaller than E
                } else {
                    int v = bi;
                    int old = owner[v];
                    if (old >= 0) slot_of[old] = -1;
                    unsigned e = miss[i];
                    owner[v] = row * E + (int)e;
                    so[e] = v;
                    usage[v] = st;
                    copy_slot[i] = (unsigned)v;
                    copy_exp[i] = e;
                    victim = v;
                }
            }
        }
        __syncthreads();
        int v = victim;
        if (v < 0) break;
        done = i + 1;
        if (v % nt == tid) {
            lu = NONE;
            li = 0x7fffffff;
            for (int s = tid; s < S; s += nt) {
                u64 u = usage[s];
                if (u != st && u < lu) { lu = u; li = s; }
            }
        }
    }
    if (tid == 0) {
        *copy_n = (unsigned)done;
        stats[0] += (u64)nhit;
        stats[1] += (u64)done;
        stats[2] += (u64)(n - done);  // misses left without a slot (cache too small)
    }
    __syncthreads();
    // point the routed experts' table entries at their slots
    for (int p = tid; p < P; p += nt) {
        unsigned e = route[p];
        if (e >= (unsigned)E) continue;
        int s = so[e];
        for (int c = 0; c < 6; ++c) {
            u64 nb = desc[12 + c];
            tab[(long)e * 6 + c] = (nb && s >= 0) ? desc[6 + c] + (u64)s * nb : 0;
        }
    }
}

// Grid-stride over (queued expert, 64 KiB piece of its components concatenated).
#define MOE_COPY_PIECE 65536
__device__ void slot_copy(unsigned n, const unsigned* copy_slot, const unsigned* copy_exp, const u64* desc) {
    __shared__ u64 off[7];
    if (threadIdx.x == 0) {
        u64 o = 0;
        for (int c = 0; c < 6; ++c) { off[c] = o; o += desc[12 + c]; }
        off[6] = o;
    }
    __syncthreads();
    u64 total = off[6];
    u64 pieces = (total + MOE_COPY_PIECE - 1) / MOE_COPY_PIECE;
    u64 items = (u64)n * pieces;
    for (u64 it = blockIdx.x; it < items; it += gridDim.x) {
        u64 y = it / pieces, a = (it % pieces) * MOE_COPY_PIECE;
        u64 b = a + MOE_COPY_PIECE < total ? a + MOE_COPY_PIECE : total;
        u64 slot = copy_slot[y], e = copy_exp[y];
        for (int c = 0; c < 6; ++c) {
            u64 lo = a > off[c] ? a : off[c], hi = b < off[c + 1] ? b : off[c + 1];
            if (lo >= hi) continue;
            u64 nb = desc[12 + c];
            const char* s = (const char*)desc[c] + e * nb + (lo - off[c]);
            char* d = (char*)desc[6 + c] + slot * nb + (lo - off[c]);
            u64 len = hi - lo;
            if ((((u64)s | (u64)d | len) & 15) == 0) {
                const uint4* s4 = (const uint4*)s;
                uint4* d4 = (uint4*)d;
                for (u64 j = threadIdx.x; j < len / 16; j += blockDim.x) d4[j] = s4[j];
            } else {
                for (u64 j = threadIdx.x; j < len; j += blockDim.x) d[j] = s[j];
            }
        }
    }
}

extern "C" __global__ void moe_slot_copy(const unsigned* copy_n, const unsigned* copy_slot,
                                         const unsigned* copy_exp, const u64* desc) {
    slot_copy(*copy_n, copy_slot, copy_exp, desc);
}

// Queue entries [start, start + n) only (prefill: copies in batches on a side stream, each
// batch's experts computed as soon as it lands).
extern "C" __global__ void moe_slot_copy_range(const unsigned* copy_slot, const unsigned* copy_exp,
                                               const u64* desc, unsigned start, unsigned n) {
    slot_copy(n, copy_slot + start, copy_exp + start, desc);
}
