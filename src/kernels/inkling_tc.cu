// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// Inkling 2 prefill attention on tensor cores (mma.sync m16n8k16 bf16, f32 accumulation),
// compiled with NVRTC for compute_80 or newer. Same semantics as ink_attn in inkling.cu:
//   logit = q . k / sqrt(128) + relb[p - pos] . r  (the bias only within `extent`)
//   out = sigmoid(gate) * sum exp(logit) v / (exp(sink) + sum exp(logit))
// over positions [lo, p] (lo = p - window + 1 for sliding-window layers). A block is 64 query
// rows of one sequence and one query head (4 warps of 16 rows); keys stream through shared
// memory 64 at a time from the cache (positions before the batch: the paged pool, bf16 or int8,
// or the sequence's ring) or from the batch's own keys (f32).

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

#define HD 128
#define BQ 64
#define BK 64
#define KPAD 8
#define NEG_INF __int_as_float(0xff800000)
#define FULL 0xffffffffu

__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float(((u32)v) << 16); }

__device__ __forceinline__ u16 f2bf(float f) {
    u32 u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (u16)(u >> 16);
}

__device__ __forceinline__ u32 pack2(float a, float b) { return (u32)f2bf(a) | ((u32)f2bf(b) << 16); }

__device__ __forceinline__ void mma16816(float* c, const u32* a, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// FMT 0: bf16 cache rows [*, nkv * 128]; 1: int8 rows of nkv * (128 + 4) bytes.
template <int FMT>
__device__ void flash(const float* q, int ldq, const float* r, int ldr, const float* sink, int lds, const float* knew,
                      const float* vnew, int ldkv, const void* kold, const void* vold, const u32* slots, int ring,
                      const u32* seqs, const u32* tiles, const float* relb, int extent, int nrel, int nkv, int rep,
                      int window, const float* gate, int ldg, float* out, int ldo) {
    __shared__ __align__(16) u16 Ks[BK][HD + KPAD];
    __shared__ __align__(16) u16 Vt[HD][BK + KPAD];
    __shared__ float Rb[2 * BK][16];
    const int h = blockIdx.y, g = h / rep;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gr = lane >> 2, gc = lane & 3;
    const u32 seq = tiles[2 * blockIdx.x], j0 = tiles[2 * blockIdx.x + 1];
    const u32* si = seqs + seq * 5;
    const int q_start = si[0], q_len = si[1], kv_len = si[2], slot_off = si[3], state = si[4];
    const int s0 = kv_len - q_len;
    const int ra = j0 + warp * 16 + gr, rb = ra + 8;  // rows within the sequence
    const bool va = ra < q_len, vb = rb < q_len;
    const int pa = s0 + ra, pb = s0 + rb;
    const long ta = q_start + ra, tb = q_start + rb;   // batch rows
    const float sc = rsqrtf((float)HD);
    // Q fragments (pre-scaled), 8 k-steps of 16 dims
    u32 qa[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const int d = ks * 16 + 2 * gc;
        const float* xa = q + ta * ldq + h * HD;
        const float* xb = q + tb * ldq + h * HD;
        qa[ks][0] = va ? pack2(xa[d] * sc, xa[d + 1] * sc) : 0u;
        qa[ks][1] = vb ? pack2(xb[d] * sc, xb[d + 1] * sc) : 0u;
        qa[ks][2] = va ? pack2(xa[d + 8] * sc, xa[d + 9] * sc) : 0u;
        qa[ks][3] = vb ? pack2(xb[d + 8] * sc, xb[d + 9] * sc) : 0u;
    }
    float ria[16], rib[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        ria[j] = (va && j < nrel) ? r[ta * ldr + h * nrel + j] : 0.0f;
        rib[j] = (vb && j < nrel) ? r[tb * ldr + h * nrel + j] : 0.0f;
    }
    float m[2] = {NEG_INF, NEG_INF}, l[2] = {0.0f, 0.0f};
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;

    const int last = min((int)j0 + BQ, q_len) - 1;
    const int p_first = s0 + (int)j0, p_last = s0 + last;
    const int lo = window ? max(0, p_first - window + 1) : 0;
    for (int kb = lo; kb <= p_last; kb += BK) {
        __syncthreads();
        // keys kb .. kb + 63 into Ks (row-major) and Vt (transposed): 2 threads per key, 8 values a step
        {
            const int kr = threadIdx.x >> 1, half = threadIdx.x & 1;
            const int pos = kb + kr;
            const int d0 = half * 64;
            const int src = pos > p_last ? 0 : (pos >= s0 ? 1 : 2);
            const float *kp = nullptr, *vp = nullptr;
            const u16 *kb16 = nullptr, *vb16 = nullptr;
            const u8 *k8 = nullptr, *v8 = nullptr;
            float ks8 = 0.0f, vs8 = 0.0f;
            if (src == 1) {
                kp = knew + (long)(q_start + pos - s0) * ldkv + g * HD + d0;
                vp = vnew + (long)(q_start + pos - s0) * ldkv + g * HD + d0;
            } else if (src == 2) {
                const long slot = ring ? (long)state * ring + (pos % ring) : (long)slots[slot_off + pos];
                if (FMT == 0) {
                    kb16 = (const u16*)kold + slot * nkv * HD + g * HD + d0;
                    vb16 = (const u16*)vold + slot * nkv * HD + g * HD + d0;
                } else {
                    k8 = (const u8*)kold + (slot * nkv + g) * (HD + 4);
                    v8 = (const u8*)vold + (slot * nkv + g) * (HD + 4);
                    ks8 = *(const float*)(k8 + HD);
                    vs8 = *(const float*)(v8 + HD);
                    k8 += d0;
                    v8 += d0;
                }
            }
#pragma unroll 2
            for (int e = 0; e < 64; e += 8) {
                float kv[8], vv[8];
                if (src == 0) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) kv[j] = vv[j] = 0.0f;
                } else if (src == 1) {
                    const float4 a0 = *(const float4*)(kp + e), a1 = *(const float4*)(kp + e + 4);
                    const float4 b0 = *(const float4*)(vp + e), b1 = *(const float4*)(vp + e + 4);
                    kv[0] = a0.x, kv[1] = a0.y, kv[2] = a0.z, kv[3] = a0.w, kv[4] = a1.x, kv[5] = a1.y, kv[6] = a1.z, kv[7] = a1.w;
                    vv[0] = b0.x, vv[1] = b0.y, vv[2] = b0.z, vv[3] = b0.w, vv[4] = b1.x, vv[5] = b1.y, vv[6] = b1.z, vv[7] = b1.w;
                } else if (FMT == 0) {
                    const uint4 a = *(const uint4*)(kb16 + e), b = *(const uint4*)(vb16 + e);
                    const u32 aw[4] = {a.x, a.y, a.z, a.w}, bw[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        kv[2 * w] = bf2f(aw[w] & 0xffff), kv[2 * w + 1] = bf2f(aw[w] >> 16);
                        vv[2 * w] = bf2f(bw[w] & 0xffff), vv[2 * w + 1] = bf2f(bw[w] >> 16);
                    }
                } else {
                    // int8 rows are 132 bytes apart: 4-byte loads
                    const uint2 a = make_uint2(*(const u32*)(k8 + e), *(const u32*)(k8 + e + 4));
                    const uint2 b = make_uint2(*(const u32*)(v8 + e), *(const u32*)(v8 + e + 4));
                    const signed char* ac = (const signed char*)&a;
                    const signed char* bc = (const signed char*)&b;
#pragma unroll
                    for (int j = 0; j < 8; ++j) kv[j] = (float)ac[j] * ks8, vv[j] = (float)bc[j] * vs8;
                }
                uint4 kw = make_uint4(pack2(kv[0], kv[1]), pack2(kv[2], kv[3]), pack2(kv[4], kv[5]), pack2(kv[6], kv[7]));
                *(uint4*)&Ks[kr][d0 + e] = kw;
#pragma unroll
                for (int j = 0; j < 8; ++j) Vt[d0 + e + j][kr] = f2bf(vv[j]);
            }
        }
        // the bias table rows this tile's distances use: [dmin, dmin + 2 BK)
        const int dmin = max(0, p_first - (kb + BK - 1));
        const bool biased = dmin < extent;
        if (biased) {
            for (int i = threadIdx.x; i < 2 * BK * 16; i += blockDim.x) {
                const int rr = i >> 4, j = i & 15, d = dmin + rr;
                Rb[rr][j] = (d < extent && j < nrel) ? relb[(long)d * nrel + j] : 0.0f;
            }
        }
        __syncthreads();
        // S = Q K^T, 8 tiles of 8 keys
        float s[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const u32 b0 = *(const u32*)&Ks[nt * 8 + gr][ks * 16 + 2 * gc];
                const u32 b1 = *(const u32*)&Ks[nt * 8 + gr][ks * 16 + 2 * gc + 8];
                mma16816(s[nt], qa[ks], b0, b1);
            }
        }
        // bias and masks
        float mx[2] = {NEG_INF, NEG_INF};
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int pos = kb + nt * 8 + 2 * gc + (e & 1);
                const bool rowb = e >= 2;
                const int p = rowb ? pb : pa;
                const int d = p - pos;
                const bool ok = (rowb ? vb : va) && d >= 0 && (window == 0 || d < window);
                float x = s[nt][e];
                if (ok && biased && d < extent) {
                    const float* rr = Rb[d - dmin];
                    const float* rq = rowb ? rib : ria;
                    float b = 0.0f;
#pragma unroll
                    for (int j = 0; j < 16; ++j) b += rr[j] * rq[j];
                    x += b;
                }
                x = ok ? x : NEG_INF;
                s[nt][e] = x;
                mx[rowb] = fmaxf(mx[rowb], x);
            }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(FULL, mx[i], 1));
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(FULL, mx[i], 2));
        }
        float corr[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float mn = fmaxf(m[i], mx[i]);
            corr[i] = (m[i] == NEG_INF) ? 0.0f : __expf(m[i] - mn);
            m[i] = mn;
            l[i] *= corr[i];
        }
#pragma unroll
        for (int dt = 0; dt < 16; ++dt) {
            o[dt][0] *= corr[0], o[dt][1] *= corr[0];
            o[dt][2] *= corr[1], o[dt][3] *= corr[1];
        }
        u32 pf[4][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            float pe[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float mm = m[e >> 1];
                pe[e] = (s[nt][e] == NEG_INF) ? 0.0f : __expf(s[nt][e] - mm);
                l[e >> 1] += pe[e];
            }
            const int k2 = nt >> 1;
            if ((nt & 1) == 0) {
                pf[k2][0] = pack2(pe[0], pe[1]);
                pf[k2][1] = pack2(pe[2], pe[3]);
            } else {
                pf[k2][2] = pack2(pe[0], pe[1]);
                pf[k2][3] = pack2(pe[2], pe[3]);
            }
        }
        // O += P V
#pragma unroll
        for (int dt = 0; dt < 16; ++dt) {
#pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {
                const u32 b0 = *(const u32*)&Vt[dt * 8 + gr][k2 * 16 + 2 * gc];
                const u32 b1 = *(const u32*)&Vt[dt * 8 + gr][k2 * 16 + 2 * gc + 8];
                mma16816(o[dt], pf[k2], b0, b1);
            }
        }
    }
    // row sums across the 4 lanes of a row, the sink, the gate
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(FULL, l[i], 1);
        l[i] += __shfl_xor_sync(FULL, l[i], 2);
    }
    float inv[2];
    {
        const float ska = va ? sink[ta * lds + h] : 0.0f, skb = vb ? sink[tb * lds + h] : 0.0f;
        const float mfa = fmaxf(m[0], ska), mfb = fmaxf(m[1], skb);
        const float ca = m[0] == NEG_INF ? 0.0f : __expf(m[0] - mfa), cb = m[1] == NEG_INF ? 0.0f : __expf(m[1] - mfb);
        // o and l are relative to m: out = o e^{m - mf} / (l e^{m - mf} + e^{sink - mf})
        inv[0] = ca / (l[0] * ca + __expf(ska - mfa));
        inv[1] = cb / (l[1] * cb + __expf(skb - mfb));
    }
#pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
        const int d = h * HD + dt * 8 + 2 * gc;
        if (va) {
            float x0 = o[dt][0] * inv[0], x1 = o[dt][1] * inv[0];
            if (gate) {
                x0 *= 1.0f / (1.0f + __expf(-gate[ta * ldg + d]));
                x1 *= 1.0f / (1.0f + __expf(-gate[ta * ldg + d + 1]));
            }
            *(float2*)(out + ta * ldo + d) = make_float2(x0, x1);
        }
        if (vb) {
            float x0 = o[dt][2] * inv[1], x1 = o[dt][3] * inv[1];
            if (gate) {
                x0 *= 1.0f / (1.0f + __expf(-gate[tb * ldg + d]));
                x1 *= 1.0f / (1.0f + __expf(-gate[tb * ldg + d + 1]));
            }
            *(float2*)(out + tb * ldo + d) = make_float2(x0, x1);
        }
    }
}

#define FLASH(FMT)                                                                                              \
    extern "C" __global__ void __launch_bounds__(128) ink_flash_##FMT(                                          \
        const float* q, int ldq, const float* r, int ldr, const float* sink, int lds, const float* knew,         \
        const float* vnew, int ldkv, const void* kold, const void* vold, const u32* slots, int ring,             \
        const u32* seqs, const u32* tiles, const float* relb, int extent, int nrel, int nkv, int rep, int window, \
        const float* gate, int ldg, float* out, int ldo) {                                                      \
        flash<FMT>(q, ldq, r, ldr, sink, lds, knew, vnew, ldkv, kold, vold, slots, ring, seqs, tiles, relb,      \
                   extent, nrel, nkv, rep, window, gate, ldg, out, ldo);                                         \
    }
FLASH(0)
FLASH(1)
