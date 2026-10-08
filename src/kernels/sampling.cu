// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// carbon's GPU sampler: one block of SAMPLE_THREADS threads per logits row.
//
// Mirrors the host sampler (src/sampling.rs) for temperature, top-k, top-p and min-p:
//   * greedy (temperature <= 0 or top_k == 1): argmax, lowest index on ties;
//   * no top-p: exact Gumbel-max over the allowed set (the top-k tokens, restricted by min-p);
//   * top-p: the candidates (top-k, or the SAMPLE_KMAX most likely tokens when top-k is off or
//     larger) are sorted in shared memory, the nucleus is cut on the cumulative mass and one
//     token is drawn. When the candidates hold less than top_p of the probability mass (a very
//     flat distribution) the row is flagged and the host samples it instead.
// Random numbers come from a counter-based hash of a per-row, per-step seed, so seeded requests
// are reproducible.
//
// Logits are f32, row-major [rows, V].

#define SAMPLE_THREADS 256
#define SAMPLE_KMAX 1024
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

__device__ __forceinline__ unsigned int carbon_key(float f) {
    // order-preserving map of floats to unsigned ints
    unsigned int u = (unsigned int)__float_as_int(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ unsigned long long carbon_mix(unsigned long long x) {
    // splitmix64 finaliser
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// uniform in (0, 1)
__device__ __forceinline__ float carbon_uniform(unsigned long long seed, unsigned long long i) {
    unsigned long long h = carbon_mix(seed ^ carbon_mix(i + 0x632be59bd9b4e019ull));
    return ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
}

// Block-wide (max value, lowest index) of (v, i) across threads.
__device__ void carbon_argmax_reduce(float& v, int& i, float* sv, int* si) {
    int t = threadIdx.x;
    sv[t] = v;
    si[t] = i;
    __syncthreads();
    for (int s = SAMPLE_THREADS / 2; s > 0; s >>= 1) {
        if (t < s) {
            float ov = sv[t + s];
            int oi = si[t + s];
            if (ov > sv[t] || (ov == sv[t] && oi < si[t])) {
                sv[t] = ov;
                si[t] = oi;
            }
        }
        __syncthreads();
    }
    v = sv[0];
    i = si[0];
    __syncthreads();
}

__device__ float carbon_sum_reduce(float v, float* sv) {
    int t = threadIdx.x;
    sv[t] = v;
    __syncthreads();
    for (int s = SAMPLE_THREADS / 2; s > 0; s >>= 1) {
        if (t < s) sv[t] += sv[t + s];
        __syncthreads();
    }
    float r = sv[0];
    __syncthreads();
    return r;
}

// Key of the k-th largest element of l[0..V) (1-based k), by 8-bit radix select.
__device__ unsigned int carbon_kth_key(const float* l, int V, int k, unsigned int* hist, int* s_misc) {
    unsigned int prefix = 0, mask = 0;
    int remaining = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = threadIdx.x; b < 256; b += SAMPLE_THREADS) hist[b] = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < V; i += SAMPLE_THREADS) {
            unsigned int key = carbon_key(l[i]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int digit = 0;
            for (int b = 255; b >= 0; --b) {
                if (remaining <= (int)hist[b]) {
                    digit = b;
                    break;
                }
                remaining -= (int)hist[b];
            }
            s_misc[0] = digit;
            s_misc[1] = remaining;
        }
        __syncthreads();
        prefix |= ((unsigned int)s_misc[0]) << shift;
        mask |= 255u << shift;
        remaining = s_misc[1];
        __syncthreads();
    }
    return prefix;
}

// params: [rows, 6] u32 = temperature bits, top_k (0 = off), top_p bits, min_p bits, seed lo, hi
// out:    [rows, 2] u32 = token, 1 when the host must sample the row instead
extern "C" __global__ void carbon_sample(
    const float* __restrict__ logits, int V,
    const unsigned int* __restrict__ params, unsigned int* __restrict__ out) {
    __shared__ float sv[SAMPLE_THREADS];
    __shared__ int si[SAMPLE_THREADS];
    __shared__ unsigned int hist[256];
    __shared__ int s_misc[4];
    __shared__ float cv[SAMPLE_KMAX];
    __shared__ int ci[SAMPLE_KMAX];

    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const float* l = logits + (size_t)row * V;
    const unsigned int* pr = params + (size_t)row * 6;
    const float temp = __int_as_float((int)pr[0]);
    const int topk = (int)pr[1];
    const float topp = __int_as_float((int)pr[2]);
    const float minp = __int_as_float((int)pr[3]);
    const unsigned long long seed = ((unsigned long long)pr[5] << 32) | pr[4];
    unsigned int* o = out + (size_t)row * 2;

    // max logit (greedy answer, softmax reference)
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = t; i < V; i += SAMPLE_THREADS) {
        float x = l[i];
        if (x > bv || (x == bv && i < bi)) {
            bv = x;
            bi = i;
        }
    }
    carbon_argmax_reduce(bv, bi, sv, si);
    if (temp <= 0.0f || topk == 1) {
        if (t == 0) {
            o[0] = (unsigned int)bi;
            o[1] = 0;
        }
        return;
    }
    const float m = bv;
    const float s = 1.0f / fmaxf(temp, 1e-5f);
    const int K = (topk > 0 && topk < V) ? topk : V;
    // the top-K set: key >= kth (ties at the boundary are all kept)
    const unsigned int kth = K < V ? carbon_kth_key(l, V, K, hist, s_misc) : 0u;
    // min-p in logit space: p_i >= minp * p_max  <=>  (l_i - m) * s >= ln(minp)
    const float minlog = minp > 0.0f ? logf(minp) : -INFINITY;

    if (topp >= 1.0f) {
        // exact sampling from the renormalised allowed set by Gumbel-max
        float gv = -INFINITY;
        int gi = bi;
        for (int i = t; i < V; i += SAMPLE_THREADS) {
            float x = l[i];
            float z = (x - m) * s;
            if (carbon_key(x) >= kth && z >= minlog) {
                float g = z - logf(-logf(carbon_uniform(seed, (unsigned long long)i)));
                if (g > gv || (g == gv && i < gi)) {
                    gv = g;
                    gi = i;
                }
            }
        }
        carbon_argmax_reduce(gv, gi, sv, si);
        if (t == 0) {
            o[0] = (unsigned int)gi;
            o[1] = 0;
        }
        return;
    }

    // top-p: sort up to SAMPLE_KMAX candidates
    const int C = K < SAMPLE_KMAX ? K : SAMPLE_KMAX;
    const unsigned int ckey = C < V ? (C == K ? kth : carbon_kth_key(l, V, C, hist, s_misc)) : 0u;
    // softmax normaliser over the whole top-K set
    float zpart = 0.0f;
    for (int i = t; i < V; i += SAMPLE_THREADS) {
        float x = l[i];
        if (carbon_key(x) >= kth) zpart += __expf((x - m) * s);
    }
    const float Z = carbon_sum_reduce(zpart, sv);
    // collect key >= ckey in index order (deterministic), at most SAMPLE_KMAX
    if (t == 0) s_misc[2] = 0;
    __syncthreads();
    for (int base = 0; base < V; base += SAMPLE_THREADS) {
        int i = base + t;
        int take = (i < V && carbon_key(l[i]) >= ckey) ? 1 : 0;
        si[t] = take;
        __syncthreads();
        // inclusive scan of the flags (Hillis-Steele)
        for (int off = 1; off < SAMPLE_THREADS; off <<= 1) {
            int add = t >= off ? si[t - off] : 0;
            __syncthreads();
            si[t] += add;
            __syncthreads();
        }
        int count = s_misc[2];
        if (take) {
            int pos = count + si[t] - 1;
            if (pos < SAMPLE_KMAX) {
                cv[pos] = l[i];
                ci[pos] = i;
            }
        }
        __syncthreads();
        if (t == 0) s_misc[2] = count + si[SAMPLE_THREADS - 1];
        __syncthreads();
    }
    int n = s_misc[2] < SAMPLE_KMAX ? s_misc[2] : SAMPLE_KMAX;
    for (int i = n + t; i < SAMPLE_KMAX; i += SAMPLE_THREADS) {
        cv[i] = -INFINITY;
        ci[i] = 0x7fffffff;
    }
    __syncthreads();
    // bitonic sort, descending by value, ascending index on ties
    for (int size = 2; size <= SAMPLE_KMAX; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = t; i < SAMPLE_KMAX; i += SAMPLE_THREADS) {
                int j = i ^ stride;
                if (j > i) {
                    bool desc = (i & size) == 0;
                    float a = cv[i], b = cv[j];
                    int ia = ci[i], ib = ci[j];
                    // "a before b" in the final order
                    bool a_first = a > b || (a == b && ia < ib);
                    if (desc != a_first) {
                        cv[i] = b;
                        cv[j] = a;
                        ci[i] = ib;
                        ci[j] = ia;
                    }
                }
            }
            __syncthreads();
        }
    }
    if (t == 0) {
        int nc = n < C ? n : C;
        // nucleus: the first `keep` candidates reaching top_p of the top-K mass
        float acc = 0.0f;
        int keep = nc;
        bool reached = false;
        for (int i = 0; i < nc; ++i) {
            acc += __expf((cv[i] - m) * s);
            if (acc >= topp * Z) {
                keep = i + 1;
                reached = true;
                break;
            }
        }
        if (!reached && nc < K) {
            // the nucleus extends beyond the candidates: let the host sample this row
            o[0] = (unsigned int)bi;
            o[1] = 1;
            return;
        }
        if (minp > 0.0f) {
            int c = 0;
            while (c < keep && (cv[c] - m) * s >= minlog) ++c;
            keep = c > 0 ? (c < keep ? c : keep) : 1;
        }
        float total = 0.0f;
        for (int i = 0; i < keep; ++i) total += __expf((cv[i] - m) * s);
        float r = carbon_uniform(seed, (unsigned long long)V + 1ull) * total;
        int pick = ci[keep - 1];
        for (int i = 0; i < keep; ++i) {
            r -= __expf((cv[i] - m) * s);
            if (r <= 0.0f) {
                pick = ci[i];
                break;
            }
        }
        o[0] = (unsigned int)pick;
        o[1] = 0;
    }
}
