// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// Runs src/kernels/gdn.cu (with quant.cu) on the CPU: one std::thread per CUDA lane, warp
// shuffles through a per-warp barrier, __syncthreads through a block barrier. Runs several decode
// steps against a plain reference of model/gdn.rs (conv, l2norm, gated delta rule, gated RMSNorm),
// checking outputs and both state pools, with fresh and continuing sequences and GQA heads.
//   g++ -std=c++20 -O2 -pthread scripts/cuda_gdn_host_test.cpp -o /tmp/gk && /tmp/gk
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
#define __constant__
#define __shared__ static  // blocks run one after another, so one copy is block-shared
static float __int_as_float(int i) { float f; memcpy(&f, &i, 4); return f; }
static int __float_as_int(float f) { int i; memcpy(&i, &f, 4); return i; }
static float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
static void __syncthreads() { block_bar->arrive_and_wait(); }
static float shfl(float v, unsigned src) {
    unsigned lane = threadIdx.x % 32;
    warp_buf[lane] = v;
    warp_bar->arrive_and_wait();
    float r = warp_buf[src];
    warp_bar->arrive_and_wait();
    return r;
}
static float __shfl_xor_sync(unsigned, float v, int off) { return shfl(v, (threadIdx.x % 32) ^ off); }
static float __shfl_down_sync(unsigned, float v, int off) {
    unsigned lane = threadIdx.x % 32;
    return shfl(v, lane + off < 32 ? lane + off : lane);
}
#include "../src/kernels/quant.cu"
#include "../src/kernels/gdn.cu"

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
        std::vector<std::thread> ts;
        for (unsigned t = 0; t < block; ++t)
            ts.emplace_back([&, t] {
                threadIdx.x = t;
                blockIdx.x = b;
                warp_bar = bars[t / 32].get();
                warp_buf = bufs[t / 32].data();
                kernel();
            });
        for (auto& t : ts) t.join();
    }
}

int fails = 0;
std::mt19937 rng(5);

template <class D>
float rnd(float v) { typename D::T t; D::store(&t, 0, v); return D::load(&t, 0); }

template <class D>
void run(const char* name, void (*conv)(const typename D::T*, typename D::T*, const unsigned*, const unsigned*,
                                        const typename D::T*, float*, int, int, int),
         void (*step)(const float*, const typename D::T*, const typename D::T*, const typename D::T*, float*,
                      const unsigned*, const unsigned*, const float*, const float*, const typename D::T*,
                      typename D::T*, int, int, int, int, float, int),
         int HK, int HV, int DK, int DV, int KM1, float tol, int gate_sigmoid = 0) {
    typedef typename D::T T;
    const int B = 3, SLOTS = 5, STEPS = 3;
    const int C = 2 * HK * DK + HV * DV;
    const float eps = 1e-6f;
    std::normal_distribution<float> nd(0, 1);
    auto fill = [&](std::vector<T>& v, float s) { for (auto& x : v) D::store(&x, 0, s * nd(rng)); };
    std::vector<T> taps((size_t)(KM1 + 1) * C), norm_w(DV), pool((size_t)SLOTS * KM1 * C);
    fill(taps, 0.5f), fill(norm_w, 1.0f), fill(pool, 1.0f);
    std::vector<float> ssm((size_t)SLOTS * HV * DK * DV), dt_bias(HV), a_neg(HV);
    for (auto& x : ssm) x = 0.1f * nd(rng);
    for (auto& x : dt_bias) x = nd(rng);
    for (auto& x : a_neg) x = -std::exp(0.5f * nd(rng));
    std::vector<unsigned> idx{3, 0, 4};
    // reference copies of the pools (as float)
    std::vector<float> rpool(pool.size()), rssm = ssm;
    for (size_t i = 0; i < pool.size(); ++i) rpool[i] = D::load(pool.data(), i);
    float worst = 0;
    for (int it = 0; it < STEPS; ++it) {
        std::vector<unsigned> fresh{it == 0 ? 1u : 0u, 0u, it == 1 ? 1u : 0u};
        std::vector<T> qkv((size_t)B * C), z((size_t)B * HV * DV), braw(B * HV), araw(B * HV), y((size_t)B * HV * DV);
        fill(qkv, 1.0f), fill(z, 1.0f), fill(braw, 1.0f), fill(araw, 1.0f);
        std::vector<float> act((size_t)B * C);
        launch(4, 128, [&] { conv(qkv.data(), pool.data(), idx.data(), fresh.data(), taps.data(), act.data(), B, C, KM1); });
        launch(B * HV, 128, [&] {
            step(act.data(), z.data(), braw.data(), araw.data(), ssm.data(), idx.data(), fresh.data(), dt_bias.data(),
                 a_neg.data(), norm_w.data(), y.data(), HK, HV, DK, DV, eps, gate_sigmoid);
        });
        // reference
        for (int b = 0; b < B; ++b) {
            float* st = &rpool[(size_t)idx[b] * KM1 * C];
            std::vector<double> ra(C);
            for (int c = 0; c < C; ++c) {
                std::vector<double> full;
                for (int j = 0; j < KM1; ++j) full.push_back(fresh[b] ? 0.0 : st[(size_t)j * C + c]);
                full.push_back(D::load(qkv.data(), (size_t)b * C + c));
                double acc = 0;
                for (int j = 0; j <= KM1; ++j) acc += D::load(taps.data(), (size_t)j * C + c) * full[j];
                for (int j = 0; j < KM1; ++j) st[(size_t)j * C + c] = rnd<D>(full[j + 1]);
                ra[c] = acc / (1 + std::exp(-acc));
            }
            for (int h = 0; h < HV; ++h) {
                int kh = h / (HV / HK);
                std::vector<double> q(DK), k(DK);
                double nq = 0, nk = 0;
                for (int i = 0; i < DK; ++i) {
                    q[i] = ra[kh * DK + i], k[i] = ra[HK * DK + kh * DK + i];
                    nq += q[i] * q[i], nk += k[i] * k[i];
                }
                for (int i = 0; i < DK; ++i) q[i] /= std::sqrt(nq + 1e-6) * std::sqrt((double)DK), k[i] /= std::sqrt(nk + 1e-6);
                double beta = 1 / (1 + std::exp(-(double)D::load(braw.data(), b * HV + h)));
                double ga = D::load(araw.data(), b * HV + h) + dt_bias[h];
                double decay = std::exp(std::log1p(std::exp(ga)) * a_neg[h]);
                float* S = &rssm[(size_t)idx[b] * HV * DK * DV + (size_t)h * DK * DV];
                std::vector<double> o(DV, 0.0);
                for (int j = 0; j < DV; ++j) {
                    double mem = 0;
                    for (int i = 0; i < DK; ++i) {
                        double s = fresh[b] ? 0.0 : S[(size_t)i * DV + j] * decay;
                        mem += k[i] * s;
                    }
                    double delta = (ra[2 * HK * DK + h * DV + j] - mem) * beta;
                    for (int i = 0; i < DK; ++i) {
                        double s = (fresh[b] ? 0.0 : S[(size_t)i * DV + j] * decay) + k[i] * delta;
                        S[(size_t)i * DV + j] = s;
                        o[j] += q[i] * s;
                    }
                }
                double var = 0;
                for (double v : o) var += v * v;
                double inv = 1 / std::sqrt(var / DV + eps);
                for (int j = 0; j < DV; ++j) {
                    float n = rnd<D>(rnd<D>(o[j] * inv) * D::load(norm_w.data(), j));
                    double zg = D::load(z.data(), (size_t)b * HV * DV + h * DV + j);
                    double want = gate_sigmoid ? n / (1 + std::exp(-zg)) : n * zg / (1 + std::exp(-zg));
                    double got = D::load(y.data(), (size_t)b * HV * DV + h * DV + j);
                    double err = std::fabs(got - want) / (1 + std::fabs(want));
                    worst = std::isnan(err) ? INFINITY : std::max(worst, (float)err);
                }
            }
        }
    }
    float sworst = 0;
    for (size_t i = 0; i < ssm.size(); ++i) sworst = std::max(sworst, std::fabs(ssm[i] - rssm[i]));
    float pworst = 0;
    for (size_t i = 0; i < pool.size(); ++i) pworst = std::max(pworst, std::fabs(D::load(pool.data(), i) - rpool[i]));
    bool ok = worst <= tol && sworst <= 1e-3f && pworst <= 1e-6f;
    fails += !ok;
    printf("%-5s HK=%d HV=%d DK=%3d DV=%3d K=%d  out %.2e  ssm %.2e  conv %.2e  %s\n", name, HK, HV, DK, DV, KM1 + 1,
           worst, sworst, pworst, ok ? "ok" : "FAIL");
}

// Prefill recurrence: gdn_seq against the token-by-token reference (model/gdn.rs `recurrent`).
void run_seq(int BH, int T, int DK, int DV) {
    std::normal_distribution<float> nd(0, 1);
    auto vec = [&](size_t n, float s) { std::vector<float> v(n); for (auto& x : v) x = s * nd(rng); return v; };
    auto q = vec((size_t)BH * T * DK, 0.1f), k = vec((size_t)BH * T * DK, 0.1f), v = vec((size_t)BH * T * DV, 1.0f);
    auto s0 = vec((size_t)BH * DK * DV, 0.1f);
    std::vector<float> beta((size_t)BH * T), g((size_t)BH * T);
    for (auto& x : beta) x = 1 / (1 + std::exp(-nd(rng)));
    for (auto& x : g) x = -std::exp(0.5f * nd(rng)) * 0.3f;
    std::vector<float> o((size_t)BH * T * DV), s1(s0.size());
    unsigned blocks = (BH * DV + 3) / 4;
    launch(blocks, 128, [&] {
        gdn_seq(q.data(), k.data(), v.data(), beta.data(), g.data(), s0.data(), o.data(), s1.data(), BH, T, DK, DV);
    });
    float worst = 0, sworst = 0;
    for (int bh = 0; bh < BH; ++bh) {
        std::vector<double> S(s0.begin() + (size_t)bh * DK * DV, s0.begin() + (size_t)(bh + 1) * DK * DV);
        for (int t = 0; t < T; ++t) {
            double dec = std::exp((double)g[(size_t)bh * T + t]), bt = beta[(size_t)bh * T + t];
            const float* kt = &k[((size_t)bh * T + t) * DK];
            const float* qt = &q[((size_t)bh * T + t) * DK];
            for (auto& x : S) x *= dec;
            for (int j = 0; j < DV; ++j) {
                double mem = 0;
                for (int i = 0; i < DK; ++i) mem += kt[i] * S[(size_t)i * DV + j];
                double delta = (v[((size_t)bh * T + t) * DV + j] - mem) * bt;
                double ov = 0;
                for (int i = 0; i < DK; ++i) {
                    S[(size_t)i * DV + j] += kt[i] * delta;
                    ov += qt[i] * S[(size_t)i * DV + j];
                }
                float err = std::fabs(o[((size_t)bh * T + t) * DV + j] - ov) / (1 + std::fabs(ov));
                worst = std::isnan(err) ? INFINITY : std::max(worst, err);
            }
        }
        for (size_t i = 0; i < S.size(); ++i)
            sworst = std::max(sworst, (float)std::fabs(s1[(size_t)bh * DK * DV + i] - S[i]));
    }
    bool ok = worst <= 1e-4f && sworst <= 1e-4f;
    fails += !ok;
    printf("seq   BH=%d T=%3d DK=%3d DV=%3d  out %.2e  state %.2e  %s\n", BH, T, DK, DV, worst, sworst, ok ? "ok" : "FAIL");
}

int main() {
    run_seq(3, 37, 64, 40);
    run_seq(2, 70, 256, 32);
    run_seq(1, 5, 32, 128);
    run<F32>("f32", gdn_conv_f32, gdn_step_f32, 1, 2, 64, 64, 3, 1e-4f);
    run<F32>("f32", gdn_conv_f32, gdn_step_f32, 2, 2, 32, 160, 1, 1e-4f);
    run<BF16>("bf16", gdn_conv_bf16, gdn_step_bf16, 1, 2, 128, 128, 3, 3e-2f);
    run<F16>("f16", gdn_conv_f16, gdn_step_f16, 2, 4, 64, 32, 3, 5e-3f);
    run<F32>("f32 sigmoid gate", gdn_conv_f32, gdn_step_f32, 2, 4, 64, 32, 3, 1e-4f, 1);
    run<BF16>("bf16 sigmoid gate", gdn_conv_bf16, gdn_step_bf16, 1, 2, 128, 128, 3, 3e-2f, 1);
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
