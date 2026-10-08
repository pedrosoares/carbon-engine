// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// Runs src/kernels/quant.cu on the CPU: one std::thread per CUDA lane, warp shuffles through a
// per-warp barrier. Checks the kernels against an independent reference.
//   g++ -std=c++20 -O2 -pthread scripts/cuda_kernels_host_test.cpp -o /tmp/qk && /tmp/qk
#include <barrier>
#include <memory>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

struct Dim3 { unsigned x = 0; };
thread_local Dim3 threadIdx, blockIdx;
Dim3 blockDim, gridDim;
thread_local std::barrier<>* warp_bar = nullptr;
thread_local float* warp_buf = nullptr;

#define CARBON_HOST_TEST
#define __device__
#define __global__
#define __forceinline__ inline
#define __constant__
static float __int_as_float(int i) { float f; memcpy(&f, &i, 4); return f; }
static int __float_as_int(float f) { int i; memcpy(&i, &f, 4); return i; }
static float __shfl_down_sync(unsigned, float v, int off) {
    unsigned lane = threadIdx.x % 32;
    warp_buf[lane] = v;
    warp_bar->arrive_and_wait();
    float r = lane + off < 32 ? warp_buf[lane + off] : v;
    warp_bar->arrive_and_wait();
    return r;
}
#include "../src/kernels/quant.cu"

template <class F>
void launch(unsigned grid, unsigned block, F kernel) {
    blockDim.x = block;
    gridDim.x = grid;
    for (unsigned b = 0; b < grid; ++b) {
        unsigned warps = (block + 31) / 32;
        std::vector<std::unique_ptr<std::barrier<>>> bars;
        for (unsigned w = 0; w < warps; ++w) bars.push_back(std::make_unique<std::barrier<>>(32));
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

static float ref_e4m3(u8 b) {
    int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN;
    float v = e == 0 ? std::ldexp(m / 8.0f, -6) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return s ? -v : v;
}
static const float REF_E2M1[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6};

int fails = 0;
void check(const char* what, const std::vector<float>& got, const std::vector<float>& want, float tol) {
    float worst = 0;
    for (size_t i = 0; i < want.size(); ++i) worst = std::max(worst, std::fabs(got[i] - want[i]) / (1 + std::fabs(want[i])));
    bool ok = worst <= tol;
    fails += !ok;
    printf("%-34s max rel err %.2e %s\n", what, worst, ok ? "ok" : "FAIL");
}

template <class D>
std::vector<float> to_f(const std::vector<typename D::T>& v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = D::load(v.data(), i);
    return o;
}

template <class D>
void run_all(const char* name, float tol) {
    std::mt19937 rng(7);
    const int n = 37, k = 96, t = 3;
    std::vector<u8> packed(n * k / 2), scales(n * k / 16), w8(n * k);
    for (auto& b : packed) b = rng() & 0xff;
    for (auto& b : scales) b = 0x30 + rng() % 0x20;  // modest positive e4m3 scales
    for (auto& b : w8) { b = rng() & 0xff; if ((b & 0x7f) == 0x7f) b ^= 1; }
    const float global = 0.37f;
    const int sr = 3, sc = 2, b0 = 16, b1 = 64;  // n=37 -> 3 row blocks, k=96 -> 2 col blocks
    std::vector<float> fs(sr * sc);
    for (auto& s : fs) s = 0.5f + (rng() % 100) / 100.0f;
    std::vector<typename D::T> x(t * k);
    std::vector<float> xf(t * k);
    for (int i = 0; i < t * k; ++i) { D::store(x.data(), i, ((int)(rng() % 2001) - 1000) / 1000.0f); xf[i] = D::load(x.data(), i); }

    std::vector<float> w_nv(n * k), w_f8(n * k);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < k; ++c) {
            u8 code = packed[(r * k + c) / 2];
            code = c % 2 ? code >> 4 : code & 15;
            w_nv[r * k + c] = REF_E2M1[code] * ref_e4m3(scales[r * (k / 16) + c / 16]) * global;
            w_f8[r * k + c] = ref_e4m3(w8[r * k + c]) * fs[(r / b0) * sc + c / b1];
        }
    auto mm = [&](const std::vector<float>& w) {
        std::vector<float> y(t * n, 0);
        for (int ti = 0; ti < t; ++ti)
            for (int r = 0; r < n; ++r)
                for (int c = 0; c < k; ++c) y[ti * n + r] += xf[ti * k + c] * w[r * k + c];
        return y;
    };

    std::vector<typename D::T> out(n * k), y(t * n);
    char label[64];
    launch(4, 64, [&] { nvfp4_dequant<D>(packed.data(), scales.data(), global, out.data(), n, k); });
    snprintf(label, 64, "nvfp4_dequant_%s", name); check(label, to_f<D>(out), w_nv, tol);
    launch((n + 3) / 4, 128, [&] { nvfp4_gemv<D>(packed.data(), scales.data(), global, x.data(), y.data(), n, k, t); });
    snprintf(label, 64, "nvfp4_gemv_%s", name); check(label, to_f<D>(y), mm(w_nv), tol * 4);
    launch(5, 64, [&] { fp8_dequant<D>(w8.data(), fs.data(), out.data(), n, k, sc, b0, b1); });
    snprintf(label, 64, "fp8_dequant_%s", name); check(label, to_f<D>(out), w_f8, tol);
    launch((n + 3) / 4, 128, [&] { fp8_gemv<D>(w8.data(), fs.data(), x.data(), y.data(), n, k, t, sc, b0, b1); });
    snprintf(label, 64, "fp8_gemv_%s", name); check(label, to_f<D>(y), mm(w_f8), tol * 4);
}

int main() {
    run_all<F32>("f32", 1e-5f);
    run_all<BF16>("bf16", 8e-3f);
    run_all<F16>("f16", 1e-3f);
    printf(fails ? "%d FAILED\n" : "all kernels ok\n", fails);
    return fails != 0;
}
