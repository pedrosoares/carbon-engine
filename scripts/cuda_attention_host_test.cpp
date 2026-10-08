// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// Runs src/kernels/attention.cu on the CPU: one std::thread per CUDA lane, warp shuffles through a
// per-warp barrier, __syncthreads through a block barrier. Checks against a naive softmax reference.
//   g++ -std=c++20 -O2 -pthread scripts/cuda_attention_host_test.cpp -o /tmp/ak && /tmp/ak
#include <algorithm>
#include <barrier>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <thread>
#include <vector>

struct Dim3 { unsigned x = 0; };
thread_local Dim3 threadIdx, blockIdx;
Dim3 blockDim, gridDim;
thread_local std::barrier<>* warp_bar = nullptr;
thread_local float* warp_buf = nullptr;
std::barrier<>* block_bar = nullptr;

#define CARBON_HOST_TEST
#define __device__
#define __global__
#define __forceinline__ inline
#define __shared__ static
static float __int_as_float(int i) { float f; memcpy(&f, &i, 4); return f; }
static int __float_as_int(float f) { int i; memcpy(&i, &f, 4); return i; }
#define __expf(x) expf(x)
static void __syncthreads() { block_bar->arrive_and_wait(); }
static float __shfl_xor_sync(unsigned, float v, int off) {
    unsigned lane = threadIdx.x % 32;
    warp_buf[lane] = v;
    warp_bar->arrive_and_wait();
    float r = warp_buf[lane ^ off];
    warp_bar->arrive_and_wait();
    return r;
}
static float __shfl_sync(unsigned, float v, int src) {
    unsigned lane = threadIdx.x % 32;
    warp_buf[lane] = v;
    warp_bar->arrive_and_wait();
    float r = warp_buf[src];
    warp_bar->arrive_and_wait();
    return r;
}
using std::min;
#include "../src/kernels/attention.cu"

// mma.m16n8k16 (.row.col, f32 accumulators) as the PTX ISA lays out the fragments:
// lane = 4*g + t; A regs {a0a1, a2a3, a4a5, a6a7} = A[g][2t..], A[g+8][2t..], A[g][2t+8..],
// A[g+8][2t+8..]; B regs {b0b1, b2b3} = B[2t..][g], B[2t+8..][g]; C {c0,c1,c2,c3} = C[g][2t..],
// C[g+8][2t..]. Low half of each 32-bit register is the lower column / row index.
typedef unsigned int u32;
thread_local u32* mma_buf = nullptr;  // 32 lanes x 6 words, per warp
template <class A>
void mma_emulate(float* c, const u32* a, const u32* b) {
    unsigned lane = threadIdx.x % 32;
    for (int i = 0; i < 4; ++i) mma_buf[lane * 6 + i] = a[i];
    mma_buf[lane * 6 + 4] = b[0];
    mma_buf[lane * 6 + 5] = b[1];
    warp_bar->arrive_and_wait();
    float Am[16][16], Bm[16][8];
    auto half = [](u32 w, int hi) { u16 x = hi ? (u16)(w >> 16) : (u16)(w & 0xffff); return A::load(&x, 0); };
    for (int L = 0; L < 32; ++L) {
        int g = L >> 2, t = L & 3;
        const u32* f = &mma_buf[L * 6];
        for (int e = 0; e < 2; ++e) {
            Am[g][2 * t + e] = half(f[0], e);
            Am[g + 8][2 * t + e] = half(f[1], e);
            Am[g][2 * t + 8 + e] = half(f[2], e);
            Am[g + 8][2 * t + 8 + e] = half(f[3], e);
            Bm[2 * t + e][g] = half(f[4], e);
            Bm[2 * t + 8 + e][g] = half(f[5], e);
        }
    }
    warp_bar->arrive_and_wait();
    int g = lane >> 2, t = lane & 3;
    for (int e = 0; e < 4; ++e) {
        int r = g + (e >= 2 ? 8 : 0), n = 2 * t + (e & 1);
        float acc = 0;
        for (int k = 0; k < 16; ++k) acc += Am[r][k] * Bm[k][n];
        c[e] += acc;
    }
}
#include "../src/kernels/attention_tc.cu"

template <class F>
void launch(unsigned grid, unsigned block, F kernel) {
    blockDim.x = block;
    gridDim.x = grid;
    for (unsigned b = 0; b < grid; ++b) {
        unsigned warps = (block + 31) / 32;
        std::vector<std::unique_ptr<std::barrier<>>> bars;
        for (unsigned w = 0; w < warps; ++w) bars.push_back(std::make_unique<std::barrier<>>(32));
        std::barrier<> bb(block);
        block_bar = &bb;
        std::vector<std::vector<float>> bufs(warps, std::vector<float>(32));
        std::vector<std::vector<u32>> mbufs(warps, std::vector<u32>(32 * 6));
        std::vector<std::thread> ts;
        for (unsigned t = 0; t < block; ++t)
            ts.emplace_back([&, t] {
                threadIdx.x = t;
                blockIdx.x = b;
                warp_bar = bars[t / 32].get();
                warp_buf = bufs[t / 32].data();
                mma_buf = mbufs[t / 32].data();
                kernel();
            });
        for (auto& t : ts) t.join();
    }
}

int fails = 0;

template <class A>
void run(const char* name, void (*kern)(const typename A::T*, const typename A::T*, const typename A::T*,
                                        const unsigned*, const unsigned*, typename A::T*, int, int, int, float),
         int H, int KVH, int D, std::vector<int> lens, float tol) {
    typedef typename A::T T;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0, 1);
    int B = lens.size();
    int total = 0;
    for (int l : lens) total += l;
    int S = total + 5;
    auto mk = [&](size_t n) {
        std::vector<T> v(n);
        for (auto& x : v) A::store(&x, 0, nd(rng));
        return v;
    };
    auto q = mk((size_t)B * H * D), k = mk((size_t)S * KVH * D), v = mk((size_t)S * KVH * D);
    std::vector<unsigned> slots(S);
    for (int i = 0; i < S; ++i) slots[i] = i;
    std::shuffle(slots.begin(), slots.end(), rng);
    slots.resize(total);
    std::vector<unsigned> starts{0};
    for (int l : lens) starts.push_back(starts.back() + l);
    std::vector<T> out((size_t)B * H * D);
    float scale = 1.0f / std::sqrt((float)D);
    launch(B * H, 128, [&] { kern(q.data(), k.data(), v.data(), slots.data(), starts.data(), out.data(), H, KVH, D, scale); });
    float worst = 0;
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h) {
            int kvh = h / (H / KVH);
            std::vector<double> sc;
            double mx = -1e300;
            for (unsigned j = starts[b]; j < starts[b + 1]; ++j) {
                double s = 0;
                for (int d = 0; d < D; ++d)
                    s += (double)A::load(q.data(), ((size_t)b * H + h) * D + d) *
                         A::load(k.data(), ((size_t)slots[j] * KVH + kvh) * D + d);
                sc.push_back(s * scale);
                mx = std::max(mx, s * scale);
            }
            double den = 0;
            for (auto& s : sc) den += (s = std::exp(s - mx));
            for (int d = 0; d < D; ++d) {
                double o = 0;
                for (size_t j = 0; j < sc.size(); ++j)
                    o += sc[j] / den * A::load(v.data(), ((size_t)slots[starts[b] + j] * KVH + kvh) * D + d);
                float got = A::load(out.data(), ((size_t)b * H + h) * D + d);
                worst = std::max(worst, (float)std::fabs(got - o));
            }
        }
    bool ok = worst <= tol;
    fails += !ok;
    printf("%-10s H=%d KVH=%d D=%3d B=%d  max abs err %.2e %s\n", name, H, KVH, D, B, worst, ok ? "ok" : "FAIL");
}

// Prefill: sequences with (cached prefix, new queries); causal within the new queries.
template <class A, class K>
void run_prefill(const char* name, K kern, int H, int KVH, int D, std::vector<std::pair<int, int>> seqs, float tol,
                 bool tc = false) {
    typedef typename A::T T;
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0, 1);
    int T_ = 0, total = 0;
    for (auto [pre, n] : seqs) T_ += n, total += pre + n;
    int S = total + 7;
    auto mk = [&](size_t n) {
        std::vector<T> v(n);
        for (auto& x : v) A::store(&x, 0, nd(rng));
        return v;
    };
    auto q = mk((size_t)T_ * H * D), k = mk((size_t)S * KVH * D), v = mk((size_t)S * KVH * D);
    std::vector<unsigned> perm(S);
    for (int i = 0; i < S; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<unsigned> slots, seqinfo, tiles;
    unsigned qs = 0;
    for (size_t b = 0; b < seqs.size(); ++b) {
        auto [pre, n] = seqs[b];
        seqinfo.insert(seqinfo.end(), {qs, (unsigned)n, (unsigned)slots.size(), (unsigned)(pre + n)});
        for (int j = 0; j < pre + n; ++j) slots.push_back(perm[slots.size()]);
        for (int q0 = 0; q0 < n; q0 += 64) tiles.insert(tiles.end(), {(unsigned)b, (unsigned)q0});
        qs += n;
    }
    std::vector<T> out((size_t)T_ * H * D);
    float scale = 1.0f / std::sqrt((float)D);
    unsigned nt = tiles.size() / 2;
    if (tc) {
        unsigned split = D > 128 ? D / 128 : 1;
        launch(nt * H * split, 128, [&] {
            if constexpr (std::is_invocable_v<K, const T*, const T*, const T*, const unsigned*, const unsigned*,
                                              const unsigned*, T*, int, int, float>)
                kern(q.data(), k.data(), v.data(), slots.data(), seqinfo.data(), tiles.data(), out.data(), H, KVH, scale);
        });
    } else {
        launch(nt * H * (64 / FP_BQ), 128, [&] {
            if constexpr (std::is_invocable_v<K, const T*, const T*, const T*, const unsigned*, const unsigned*,
                                              const unsigned*, T*, int, int, int, float>)
                kern(q.data(), k.data(), v.data(), slots.data(), seqinfo.data(), tiles.data(), out.data(), H, KVH, D,
                     scale);
        });
    }
    float worst = 0;
    for (size_t b = 0; b < seqs.size(); ++b) {
        auto [pre, n] = seqs[b];
        for (int qi = 0; qi < n; ++qi)
            for (int h = 0; h < H; ++h) {
                int kvh = h / (H / KVH);
                long qrow = ((long)(seqinfo[4 * b] + qi) * H + h) * D;
                std::vector<double> sc;
                double mx = -1e300;
                for (int j = 0; j <= pre + qi; ++j) {
                    long krow = ((long)slots[seqinfo[4 * b + 2] + j] * KVH + kvh) * D;
                    double s = 0;
                    for (int d = 0; d < D; ++d) s += (double)A::load(q.data(), qrow + d) * A::load(k.data(), krow + d);
                    sc.push_back(s * scale);
                    mx = std::max(mx, s * scale);
                }
                double den = 0;
                for (auto& s : sc) den += (s = std::exp(s - mx));
                for (int d = 0; d < D; ++d) {
                    double o = 0;
                    for (size_t j = 0; j < sc.size(); ++j)
                        o += sc[j] / den * A::load(v.data(), ((long)slots[seqinfo[4 * b + 2] + j] * KVH + kvh) * D + d);
                    float err = std::fabs(A::load(out.data(), qrow + d) - o);
                    worst = std::isnan(err) ? INFINITY : std::max(worst, err);
                }
            }
    }
    bool ok = worst <= tol;
    fails += !ok;
    printf("prefill%s %-5s H=%d KVH=%d D=%3d seqs=%zu  max abs err %.2e %s\n", tc ? "-tc" : "", name, H, KVH, D, seqs.size(), worst,
           ok ? "ok" : "FAIL");
}

int main() {
    run_prefill<AF32>("f32", flash_prefill_f32, 4, 2, 64, {{0, 37}, {20, 5}, {0, 1}}, 1e-5f);
    run_prefill<AF32>("f32", flash_prefill_f32, 2, 1, 256, {{9, 17}}, 1e-5f);
    run_prefill<ABF16>("bf16", flash_prefill_bf16, 4, 2, 128, {{0, 50}, {70, 33}}, 2e-2f);
    run_prefill<AF16>("f16", flash_prefill_f16, 2, 2, 256, {{3, 40}}, 3e-3f);
    // tensor-core kernel: P is rounded to the model dtype before P.V, hence the looser bounds
    run_prefill<ABF16>("bf16", flash_prefill_tc_bf16_d128, 4, 2, 128, {{0, 70}, {45, 33}, {0, 1}}, 3e-2f, true);
    run_prefill<ABF16>("bf16", flash_prefill_tc_bf16_d64, 2, 1, 64, {{100, 129}}, 3e-2f, true);
    run_prefill<AF16>("f16", flash_prefill_tc_f16_d256, 2, 2, 256, {{7, 66}}, 5e-3f, true);
    run_prefill<ABF16>("bf16", flash_prefill_tc_bf16_d256, 4, 4, 256, {{0, 20}, {31, 64}}, 3e-2f, true);
    std::vector<int> lens{1, 3, 37, 130};
    run<AF32>("f32", paged_decode_f32, 4, 2, 64, lens, 1e-5f);
    run<AF32>("f32", paged_decode_f32, 2, 1, 256, lens, 1e-5f);
    run<AF32>("f32", paged_decode_f32, 3, 3, 32, {5, 0 + 2}, 1e-5f);
    run<ABF16>("bf16", paged_decode_bf16, 4, 2, 128, lens, 2e-2f);
    run<AF16>("f16", paged_decode_f16, 4, 1, 128, lens, 3e-3f);
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
