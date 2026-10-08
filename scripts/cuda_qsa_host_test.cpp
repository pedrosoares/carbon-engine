// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// Runs src/kernels/qsa.cu on the CPU (one std::thread per CUDA lane, warp shuffles and
// __syncwarp through per-warp barriers, __syncthreads through a block barrier) and checks the
// selected slot tables against a direct implementation of the host path (src/model/attention.rs).
//   g++ -std=c++20 -O2 -pthread scripts/cuda_qsa_host_test.cpp -o /tmp/qt && /tmp/qt
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
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
#define __constant__
#define __forceinline__ inline
#define __shared__ static
static float __int_as_float(int i) { float f; memcpy(&f, &i, 4); return f; }
static int __float_as_int(float f) { int i; memcpy(&i, &f, 4); return i; }
static void __syncthreads() { block_bar->arrive_and_wait(); }
static void __syncwarp() { warp_bar->arrive_and_wait(); }
static float __shfl_xor_sync(unsigned, float v, int off) {
    unsigned lane = threadIdx.x % 32;
    warp_buf[lane] = v;
    warp_bar->arrive_and_wait();
    float r = warp_buf[lane ^ off];
    warp_bar->arrive_and_wait();
    return r;
}
static float __shfl_down_sync(unsigned, float v, int off) {
    unsigned lane = threadIdx.x % 32;
    warp_buf[lane] = v;
    warp_bar->arrive_and_wait();
    float r = lane + off < 32 ? warp_buf[lane + off] : v;
    warp_bar->arrive_and_wait();
    return r;
}
static unsigned atomicAdd(unsigned* p, unsigned v) { return std::atomic_ref<unsigned>(*p).fetch_add(v); }
static float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
#include "../src/kernels/quant.cu"
#include "../src/kernels/qsa.cu"

// one pool of QSA_THREADS threads runs the blocks in turn (shared memory is static)
template <class F>
void launch(unsigned grid, F kernel) {
    const unsigned block = QSA_THREADS, warps = block / 32;
    blockDim.x = block;
    gridDim.x = grid;
    std::vector<std::unique_ptr<std::barrier<>>> bars;
    for (unsigned w = 0; w < warps; ++w) bars.push_back(std::make_unique<std::barrier<>>(32));
    std::barrier<> bb(block);
    block_bar = &bb;
    std::vector<std::vector<float>> bufs(warps, std::vector<float>(32));
    std::vector<std::thread> ts;
    for (unsigned t = 0; t < block; ++t)
        ts.emplace_back([&, t] {
            threadIdx.x = t;
            warp_bar = bars[t / 32].get();
            warp_buf = bufs[t / 32].data();
            for (unsigned b = 0; b < grid; ++b) {
                blockIdx.x = b;
                kernel();
                bb.arrive_and_wait();
            }
        });
    for (auto& t : ts) t.join();
}

int fails = 0;

template <class D>
float rnd(float v) { typename D::T t; D::store(&t, 0, v); return D::load(&t, 0); }

// the host path: pooled mean (rounded), rms norm (rounded), rope (rounded), relu-summed scores,
// stable top-K (score desc, index asc), ascending blocks then the partial block
template <class D>
std::vector<unsigned> reference(const std::vector<typename D::T>& q, const std::vector<typename D::T>& pool,
                                const std::vector<typename D::T>& kn, const std::vector<typename D::T>& cs,
                                const std::vector<typename D::T>& sn, const std::vector<unsigned>& sl, int b, int NH,
                                int Dm, int HALF, int K, int R, float eps) {
    int len = (int)sl.size(), nb = len / R;
    if (nb <= K) return sl;
    std::vector<float> score(nb);
    for (int j = 0; j < nb; ++j) {
        std::vector<float> k(Dm), r(Dm);
        float ss = 0;
        for (int i = 0; i < Dm; ++i) {
            float acc = 0;
            for (int t = 0; t < R; ++t) acc += D::load(pool.data(), (long)sl[j * R + t] * Dm + i);
            k[i] = rnd<D>(acc / R);
            ss += k[i] * k[i];
        }
        float inv = 1.0f / std::sqrt(ss / Dm + eps);
        for (int i = 0; i < Dm; ++i) k[i] = rnd<D>(k[i] * inv * D::load(kn.data(), i));
        long pos = (long)j * R;
        for (int i = 0; i < Dm; ++i) {
            if (i < 2 * HALF) {
                int f = i % HALF;
                float c = D::load(cs.data(), pos * HALF + f), s = D::load(sn.data(), pos * HALF + f);
                r[i] = rnd<D>(i < HALF ? k[i] * c - k[i + HALF] * s : k[i] * c + k[i - HALF] * s);
            } else {
                r[i] = k[i];
            }
        }
        float tot = 0;
        for (int h = 0; h < NH; ++h) {
            float d = 0;
            for (int i = 0; i < Dm; ++i) d += D::load(q.data(), ((long)b * NH + h) * Dm + i) * r[i];
            tot += std::max(d, 0.0f);
        }
        score[j] = tot / std::sqrt((float)Dm);
    }
    std::vector<int> order(nb);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int c) { return score[a] > score[c]; });
    std::vector<int> pick(order.begin(), order.begin() + K);
    std::sort(pick.begin(), pick.end());
    std::vector<unsigned> out;
    for (int blk : pick)
        for (int t = 0; t < R; ++t) out.push_back(sl[blk * R + t]);
    for (int i = nb * R; i < len; ++i) out.push_back(sl[i]);
    return out;
}

template <class D>
void run(const char* name, void (*kern)(const typename D::T*, const typename D::T*, const typename D::T*,
                                        const typename D::T*, const typename D::T*, const unsigned*, const unsigned*,
                                        float*, unsigned*, unsigned*, int, int, int, int, int, int, float),
         std::vector<int> lens, int NH, int Dm, int HALF, int K, int R, bool ties) {
    typedef typename D::T T;
    std::mt19937 g(lens.size() * 7 + Dm);
    std::normal_distribution<float> nd(0, 1);
    const int B = lens.size(), P = 4096, SLOTS = 8192;
    const float eps = 1e-6f;
    auto fill = [&](std::vector<T>& v, float s) { for (auto& x : v) D::store(&x, 0, s * nd(g)); };
    std::vector<T> q((size_t)B * NH * Dm), pool((size_t)SLOTS * Dm), kn(Dm), cs((size_t)P * HALF), sn((size_t)P * HALF);
    fill(q, 1.0f), fill(pool, 1.0f);
    for (int i = 0; i < Dm; ++i) D::store(kn.data(), i, 1.0f + 0.2f * nd(g));
    for (int p = 0; p < P; ++p)
        for (int f = 0; f < HALF; ++f) {
            double a = p * std::pow(10000.0, -2.0 * f / (2 * HALF));
            D::store(cs.data(), (long)p * HALF + f, (float)std::cos(a));
            D::store(sn.data(), (long)p * HALF + f, (float)std::sin(a));
        }
    if (ties)  // queries that make many scores exactly zero (relu of negative dot products)
        for (auto& x : q) D::store(&x, 0, -std::fabs(D::load(&x, 0)) * 0.01f);
    // every sequence takes its own random slots
    std::vector<unsigned> perm(SLOTS);
    std::iota(perm.begin(), perm.end(), 0u);
    std::shuffle(perm.begin(), perm.end(), g);
    std::vector<unsigned> slots, starts{0};
    std::vector<std::vector<unsigned>> seq;
    size_t at = 0;
    for (int L : lens) {
        seq.emplace_back(perm.begin() + at, perm.begin() + at + L);
        slots.insert(slots.end(), seq.back().begin(), seq.back().end());
        starts.push_back(slots.size());
        at += L;
    }
    int maxnb = SLOTS / R;
    std::vector<float> scores((size_t)B * maxnb);
    std::vector<unsigned> out((size_t)B * (K + 1) * R, 0xffffffffu), out_starts(B + 1, 0xffffffffu);
    launch(B, [&] {
        kern(q.data(), pool.data(), kn.data(), cs.data(), sn.data(), slots.data(), starts.data(), scores.data(),
             out.data(), out_starts.data(), NH, Dm, HALF, K, R, maxnb, eps);
    });
    bool ok = out_starts[0] == 0;
    for (int b = 0; b < B && ok; ++b) {
        auto want = reference<D>(q, pool, kn, cs, sn, seq[b], b, NH, Dm, HALF, K, R, eps);
        std::vector<unsigned> got(out.begin() + out_starts[b], out.begin() + out_starts[b + 1]);
        if (got != want) {
            ok = false;
            printf("   seq %d (len %d): got %zu slots, want %zu\n", b, lens[b], got.size(), want.size());
        }
    }
    printf("%s %-30s B=%d NH=%d D=%d K=%d R=%d\n", ok ? "ok  " : "FAIL", name, B, NH, Dm, K, R);
    fails += !ok;
}

int main() {
    // lengths: dense (<= K blocks), exactly K blocks + partial, just past the budget, long
    std::vector<int> lens{30, 67, 68, 69, 70, 71, 400, 1500};
    run<F32>("f32 (Qwen3.8 geometry)", qsa_select_f32, lens, 4, 128, 32, 16, 4, false);
    run<F32>("f32 small head, partial rope", qsa_select_f32, {5, 40, 90}, 4, 16, 4, 4, 4, false);
    run<F32>("f32 zero-score ties", qsa_select_f32, {200, 333}, 4, 64, 16, 8, 4, true);
    run<BF16>("bf16", qsa_select_bf16, lens, 4, 128, 32, 16, 4, false);
    run<F16>("f16", qsa_select_f16, {120, 501}, 2, 64, 16, 10, 2, false);
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
