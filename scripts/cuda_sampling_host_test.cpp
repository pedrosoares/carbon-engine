// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// Runs src/kernels/sampling.cu on the CPU (one std::thread per CUDA thread, __syncthreads through a
// block barrier) and checks it against the exact sampling distributions.
//   g++ -std=c++20 -O2 -pthread scripts/cuda_sampling_host_test.cpp -o /tmp/sk && /tmp/sk
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

struct Dim3 { unsigned x = 0; };
thread_local Dim3 threadIdx, blockIdx;
std::barrier<>* block_bar = nullptr;

#define __device__
#define __global__
#define __forceinline__ inline
#define __shared__ static
#define __restrict__
static float __int_as_float(int i) { float f; memcpy(&f, &i, 4); return f; }
static int __float_as_int(float f) { int i; memcpy(&i, &f, 4); return i; }
#define __expf(x) expf(x)
static void __syncthreads() { block_bar->arrive_and_wait(); }
static unsigned atomicAdd(unsigned* p, unsigned v) { return std::atomic_ref<unsigned>(*p).fetch_add(v); }
#include "../src/kernels/sampling.cu"

typedef unsigned int u32;

// rows of logits (each V wide) -> (token, fallback) per row
std::vector<std::pair<u32, u32>> run(const std::vector<float>& logits, int V, const std::vector<u32>& params) {
    int rows = (int)params.size() / 6;
    std::vector<u32> out(rows * 2, 7777);
    // one pool of SAMPLE_THREADS threads runs the blocks one after another; the extra barrier
    // keeps a fast thread from starting the next block while others still use shared memory
    std::barrier<> bb(SAMPLE_THREADS);
    block_bar = &bb;
    std::vector<std::thread> ts;
    for (int t = 0; t < SAMPLE_THREADS; ++t)
        ts.emplace_back([&, t] {
            threadIdx.x = t;
            for (int b = 0; b < rows; ++b) {
                blockIdx.x = b;
                carbon_sample(logits.data(), V, params.data(), out.data());
                bb.arrive_and_wait();
            }
        });
    for (auto& t : ts) t.join();
    std::vector<std::pair<u32, u32>> r;
    for (int i = 0; i < rows; ++i) r.push_back({out[2 * i], out[2 * i + 1]});
    return r;
}

u32 fbits(float f) { u32 u; memcpy(&u, &f, 4); return u; }
void push(std::vector<u32>& p, float temp, int topk, float topp, float minp, unsigned long long seed) {
    p.insert(p.end(), {fbits(temp), (u32)topk, fbits(topp), fbits(minp), (u32)seed, (u32)(seed >> 32)});
}

// exact distribution of the host sampler (src/sampling.rs)
std::vector<double> exact(const std::vector<float>& l, float temp, int topk, float topp, float minp) {
    int V = l.size();
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return l[a] > l[b]; });
    int K = (topk > 0 && topk < V) ? topk : V;
    std::vector<double> p(K);
    double m = l[idx[0]], z = 0;
    for (int i = 0; i < K; ++i) z += p[i] = exp((l[idx[i]] - m) / temp);
    for (auto& x : p) x /= z;
    int keep = K;
    if (topp < 1) { double acc = 0; for (int i = 0; i < K; ++i) { acc += p[i]; if (acc >= topp) { keep = i + 1; break; } } }
    if (minp > 0) { int c = 0; while (c < keep && p[c] >= p[0] * minp) ++c; keep = std::max(1, std::min(c, keep)); }
    double t = 0;
    for (int i = 0; i < keep; ++i) t += p[i];
    std::vector<double> out(V, 0);
    for (int i = 0; i < keep; ++i) out[idx[i]] = p[i] / t;
    return out;
}

int fails = 0;
void check(bool ok, const char* what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}

// draw n samples with distinct seeds and compare frequencies to the exact distribution
void dist(const char* name, const std::vector<float>& l, float temp, int topk, float topp, float minp, int n) {
    int V = l.size();
    std::vector<float> all;
    std::vector<u32> p;
    for (int i = 0; i < n; ++i) { all.insert(all.end(), l.begin(), l.end()); push(p, temp, topk, topp, minp, 1000003ull * i + 17); }
    auto r = run(all, V, p);
    auto e = exact(l, temp, topk, topp, minp);
    std::vector<double> f(V, 0);
    bool support = true, flagged = false;
    for (auto [t, fb] : r) { flagged |= fb != 0; if (t >= (u32)V || e[t] == 0) support = false; else f[t] += 1.0 / n; }
    // total variation against the exact distribution, judged against its sampling noise
    // (E|f_i - p_i| ~ sqrt(2/pi) * sqrt(p_i (1 - p_i) / n))
    double tv = 0, noise = 0;
    for (int i = 0; i < V; ++i) {
        tv += fabs(f[i] - e[i]) / 2;
        noise += 0.3989 * sqrt(e[i] * (1 - e[i]) / n);
    }
    char msg[256];
    snprintf(msg, sizeof msg, "%s: support %s, total variation %.3f (noise %.3f)", name,
             support ? "ok" : "BAD", tv, noise);
    // a token the exact sampler can draw (p > 1%) must show up
    bool missing = false;
    for (int i = 0; i < V; ++i)
        if (e[i] > 0.01 && f[i] == 0) { missing = true; printf("     token %d (p=%.3f) never drawn\n", i, e[i]); }
    if (getenv("SAMPLE_DIAG"))
        for (int i = 0; i < V; ++i)
            if (e[i] > 0) printf("     %3d exact %.4f got %.4f\n", i, e[i], f[i]);
    check(support && !flagged && !missing && tv < 1.5 * noise + 0.01, msg);
}

int main() {
    std::mt19937 g(3);
    std::normal_distribution<float> nd(0, 2);
    std::vector<float> l(40);
    for (auto& x : l) x = nd(g);
    // greedy, ties -> lowest index
    {
        std::vector<float> t = {1, 5, 5, 2};
        std::vector<u32> p;
        push(p, 0, 0, 1, 0, 1);
        push(p, 1, 1, 1, 0, 1);
        auto r = run(t, 4, std::vector<u32>(p.begin(), p.begin() + 6));
        auto r2 = run(t, 4, std::vector<u32>(p.begin() + 6, p.end()));
        check(r[0].first == 1 && r2[0].first == 1, "greedy argmax, lowest index on ties");
    }
    dist("temperature 1", l, 1.0, 0, 1, 0, 4000);
    dist("temperature 0.7, top-k 5", l, 0.7, 5, 1, 0, 4000);
    dist("top-p 0.8", l, 1.0, 0, 0.8, 0, 4000);
    dist("top-k 10 + top-p 0.9", l, 1.0, 10, 0.9, 0, 4000);
    dist("min-p 0.2", l, 1.0, 0, 1, 0.2, 4000);
    dist("top-p 0.9 + min-p 0.1", l, 1.2, 0, 0.9, 0.1, 4000);
    // reproducible: same seed, same token
    {
        std::vector<u32> p;
        push(p, 1, 0, 0.95, 0, 42);
        push(p, 1, 0, 0.95, 0, 42);
        std::vector<float> two(l);
        two.insert(two.end(), l.begin(), l.end());
        auto r = run(two, 40, p);
        check(r[0].first == r[1].first, "same seed, same token");
    }
    // a big vocabulary: candidates selected by radix select over 5000 tokens
    {
        std::vector<float> big(5000);
        for (auto& x : big) x = nd(g);
        dist("V=5000, top-k 50", big, 1.0, 50, 1, 0, 3000);
        dist("V=5000, top-p 0.5 (peaked)", [&] { auto b = big; for (int i = 0; i < 30; ++i) b[i * 97] += 9; return b; }(),
             1.0, 0, 0.5, 0, 3000);
        // nearly flat: 1024 candidates cannot hold top_p of the mass -> host fallback
        std::vector<float> flat(5000, 0.0f);
        std::vector<u32> p;
        push(p, 1, 0, 0.9, 0, 5);
        check(run(flat, 5000, p)[0].second == 1, "flat V=5000 top-p 0.9 falls back to the host");
    }
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
