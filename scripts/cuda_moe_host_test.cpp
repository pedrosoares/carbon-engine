// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

// Runs src/kernels/moe.cu (with quant.cu) on the CPU: one std::thread per CUDA lane, warp
// shuffles through a per-warp barrier. Checks routing and the grouped expert GEMVs for dense,
// FP8 and NVFP4 experts against a plain reference.
//   g++ -std=c++20 -O2 -pthread scripts/cuda_moe_host_test.cpp -o /tmp/mk && /tmp/mk
#include <algorithm>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

struct Dim3 { unsigned x = 0, y = 0; };
thread_local Dim3 threadIdx, blockIdx;
Dim3 blockDim, gridDim;
thread_local std::barrier<>* warp_bar = nullptr;
thread_local float* warp_buf = nullptr;

#define CARBON_HOST_TEST
#define __device__
#define __global__
#define __forceinline__ inline
#define __constant__
#define __shared__ static  // blocks run one after another, so one copy is block-shared
static float __int_as_float(int i) { float f; memcpy(&f, &i, 4); return f; }
static int __float_as_int(float f) { int i; memcpy(&i, &f, 4); return i; }
template <class T>
static T shfl(T v, unsigned src) {
    unsigned lane = threadIdx.x % 32;
    memcpy(&warp_buf[lane], &v, 4);
    warp_bar->arrive_and_wait();
    T r;
    memcpy(&r, &warp_buf[src], 4);
    warp_bar->arrive_and_wait();
    return r;
}
template <class T> static T __shfl_xor_sync(unsigned, T v, int off) { return shfl(v, (threadIdx.x % 32) ^ off); }
template <class T> static T __shfl_down_sync(unsigned, T v, int off) {
    unsigned lane = threadIdx.x % 32;
    T r = shfl(v, lane + off < 32 ? lane + off : lane);
    return r;
}
static void __syncwarp() { warp_bar->arrive_and_wait(); }
#include "../src/kernels/quant.cu"
#include "../src/kernels/moe.cu"

template <class F>
void launch(Dim3 grid, unsigned block, F kernel) {
    blockDim.x = block;
    gridDim = grid;
    for (unsigned by = 0; by < std::max(grid.y, 1u); ++by)
        for (unsigned bx = 0; bx < grid.x; ++bx) {
            unsigned warps = (block + 31) / 32;
            std::vector<std::unique_ptr<std::barrier<>>> bars;
            for (unsigned w = 0; w < warps; ++w) bars.push_back(std::make_unique<std::barrier<>>(32));
            std::vector<std::vector<float>> bufs(warps, std::vector<float>(32));
            std::vector<std::thread> ts;
            for (unsigned t = 0; t < block; ++t)
                ts.emplace_back([&, t] {
                    threadIdx.x = t;
                    blockIdx.x = bx;
                    blockIdx.y = by;
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
std::mt19937 rng(11);
int fails = 0;

// One quantized [n, k] matrix plus its dequantized reference.
struct Mat {
    std::vector<u8> codes, scale8;
    std::vector<float> scale32, ref;
    std::vector<u8> dense;  // dense weights in the activation dtype
    float global = 1.0f;
    u64 w = 0, s = 0;
};
static u8 rand_e4m3(float lo, float hi) {
    for (;;) {
        u8 b = rng() & 0x7f;
        float v = ref_e4m3(b);
        if (v >= lo && v <= hi) return b;
    }
}
template <class D>
Mat make(int fmt, int n, int k, int bs) {
    Mat m;
    std::uniform_real_distribution<float> ud(-1, 1);
    m.ref.resize((size_t)n * k);
    if (fmt == FMT_DENSE) {
        m.dense.resize((size_t)n * k * sizeof(typename D::T));
        auto* w = (typename D::T*)m.dense.data();
        for (size_t i = 0; i < (size_t)n * k; ++i) {
            D::store(w, i, ud(rng));
            m.ref[i] = D::load(w, i);
        }
        m.w = (u64)m.dense.data();
    } else if (fmt == FMT_FP8) {
        int sr = (n + bs - 1) / bs, sc = (k + bs - 1) / bs;
        m.codes.resize((size_t)n * k);
        m.scale32.resize((size_t)sr * sc);
        for (auto& c : m.codes) c = rng() & 0xff, c = (c & 0x78) == 0x78 ? c & 0xf7 : c;
        for (auto& s : m.scale32) s = 0.01f + 0.02f * (ud(rng) + 1);
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < k; ++c)
                m.ref[(size_t)r * k + c] = ref_e4m3(m.codes[(size_t)r * k + c]) * m.scale32[(r / bs) * sc + c / bs];
        m.w = (u64)m.codes.data();
        m.s = (u64)m.scale32.data();
    } else {
        m.codes.resize((size_t)n * k / 2);
        m.scale8.resize((size_t)n * k / 16);
        m.global = 0.37f;
        for (auto& c : m.codes) c = rng() & 0xff;
        for (auto& s : m.scale8) s = rand_e4m3(0.02f, 0.3f);
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < k; ++c) {
                u8 code = m.codes[((size_t)r * k + c) / 2];
                float v = REF_E2M1[c % 2 ? code >> 4 : code & 15];
                m.ref[(size_t)r * k + c] = v * ref_e4m3(m.scale8[((size_t)r * k + c) / 16]) * m.global;
            }
        m.w = (u64)m.codes.data();
        m.s = (u64)m.scale8.data();
    }
    return m;
}

template <class D>
void run(const char* name, int fmt, int T, int E, int K, int H, int I, bool norm, bool skip_odd, float tol) {
    typedef typename D::T DT;
    const int bs = 16;
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> logits((size_t)T * E);
    for (auto& v : logits) v = nd(rng);
    std::vector<DT> x((size_t)T * H);
    for (auto& v : x) D::store(&v, 0, nd(rng));
    std::vector<Mat> gate, up, down;
    std::vector<u64> tab((size_t)E * 6, 0);
    std::vector<float> glob((size_t)E * 3, 1.0f);
    for (int e = 0; e < E; ++e) {
        gate.push_back(make<D>(fmt, I, H, bs));
        up.push_back(make<D>(fmt, I, H, bs));
        down.push_back(make<D>(fmt, H, I, bs));
        if (skip_odd && e % 2) continue;  // expert-parallel: odd experts live elsewhere
        u64* te = &tab[(size_t)e * 6];
        te[0] = gate[e].w, te[1] = gate[e].s, te[2] = up[e].w, te[3] = up[e].s, te[4] = down[e].w, te[5] = down[e].s;
        glob[e * 3] = gate[e].global, glob[e * 3 + 1] = up[e].global, glob[e * 3 + 2] = down[e].global;
    }
    // FP8 geometry: gate/up [I, H] and down [H, I] with bs x bs blocks
    int gsc = (H + bs - 1) / bs, dsc = (I + bs - 1) / bs;
    std::vector<unsigned> ids((size_t)T * K);
    std::vector<float> wts((size_t)T * K), act((size_t)T * K * I);
    std::vector<DT> y((size_t)T * H);
    launch({(unsigned)T, 1}, 32, [&] { moe_topk(logits.data(), ids.data(), wts.data(), E, K, norm); });
    // dispatch through the extern "C" entry points the engine launches
    auto gu = [&] {
        if constexpr (std::is_same_v<D, F32>) {
            if (fmt == 0) moe_gate_up_dense_f32(tab.data(), glob.data(), x.data(), ids.data(), act.data(), K, I, H, gsc, bs, bs);
            if (fmt == 1) moe_gate_up_fp8_f32(tab.data(), glob.data(), x.data(), ids.data(), act.data(), K, I, H, gsc, bs, bs);
            if (fmt == 2) moe_gate_up_nvfp4_f32(tab.data(), glob.data(), x.data(), ids.data(), act.data(), K, I, H, gsc, bs, bs);
        } else if constexpr (std::is_same_v<D, BF16>) {
            if (fmt == 0) moe_gate_up_dense_bf16(tab.data(), glob.data(), x.data(), ids.data(), act.data(), K, I, H, gsc, bs, bs);
            if (fmt == 1) moe_gate_up_fp8_bf16(tab.data(), glob.data(), x.data(), ids.data(), act.data(), K, I, H, gsc, bs, bs);
            if (fmt == 2) moe_gate_up_nvfp4_bf16(tab.data(), glob.data(), x.data(), ids.data(), act.data(), K, I, H, gsc, bs, bs);
        } else {
            if (fmt == 0) moe_gate_up_dense_f16(tab.data(), glob.data(), x.data(), ids.data(), act.data(), K, I, H, gsc, bs, bs);
            if (fmt == 1) moe_gate_up_fp8_f16(tab.data(), glob.data(), x.data(), ids.data(), act.data(), K, I, H, gsc, bs, bs);
            if (fmt == 2) moe_gate_up_nvfp4_f16(tab.data(), glob.data(), x.data(), ids.data(), act.data(), K, I, H, gsc, bs, bs);
        }
    };
    auto dn = [&] {
        if constexpr (std::is_same_v<D, F32>) {
            if (fmt == 0) moe_down_dense_f32(tab.data(), glob.data(), act.data(), ids.data(), wts.data(), y.data(), K, I, H, dsc, bs, bs);
            if (fmt == 1) moe_down_fp8_f32(tab.data(), glob.data(), act.data(), ids.data(), wts.data(), y.data(), K, I, H, dsc, bs, bs);
            if (fmt == 2) moe_down_nvfp4_f32(tab.data(), glob.data(), act.data(), ids.data(), wts.data(), y.data(), K, I, H, dsc, bs, bs);
        } else if constexpr (std::is_same_v<D, BF16>) {
            if (fmt == 0) moe_down_dense_bf16(tab.data(), glob.data(), act.data(), ids.data(), wts.data(), y.data(), K, I, H, dsc, bs, bs);
            if (fmt == 1) moe_down_fp8_bf16(tab.data(), glob.data(), act.data(), ids.data(), wts.data(), y.data(), K, I, H, dsc, bs, bs);
            if (fmt == 2) moe_down_nvfp4_bf16(tab.data(), glob.data(), act.data(), ids.data(), wts.data(), y.data(), K, I, H, dsc, bs, bs);
        } else {
            if (fmt == 0) moe_down_dense_f16(tab.data(), glob.data(), act.data(), ids.data(), wts.data(), y.data(), K, I, H, dsc, bs, bs);
            if (fmt == 1) moe_down_fp8_f16(tab.data(), glob.data(), act.data(), ids.data(), wts.data(), y.data(), K, I, H, dsc, bs, bs);
            if (fmt == 2) moe_down_nvfp4_f16(tab.data(), glob.data(), act.data(), ids.data(), wts.data(), y.data(), K, I, H, dsc, bs, bs);
        }
    };
    launch({(unsigned)(I + 3) / 4, (unsigned)(T * K)}, 128, gu);
    launch({(unsigned)(H + 3) / 4, (unsigned)T}, 128, dn);

    // reference: host routing (same rule as MoeBlock::route) and plain f64 expert math
    float worst = 0;
    bool route_ok = true;
    for (int t = 0; t < T; ++t) {
        const float* row = &logits[(size_t)t * E];
        float mx = *std::max_element(row, row + E);
        std::vector<float> ex(E);
        for (int e = 0; e < E; ++e) ex[e] = std::exp(row[e] - mx);
        float sum = std::accumulate(ex.begin(), ex.end(), 0.0f);
        std::vector<int> order(E);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return ex[a] != ex[b] ? ex[a] > ex[b] : a < b; });
        float denom = 0;
        for (int r = 0; r < K; ++r) denom += norm ? ex[order[r]] : 0;
        if (!norm) denom = sum;
        std::vector<double> out(H, 0.0);
        for (int r = 0; r < K; ++r) {
            int e = order[r];
            float w = ex[e] / denom;
            if (ids[(size_t)t * K + r] != (unsigned)e || std::fabs(wts[(size_t)t * K + r] - w) > 1e-5f) route_ok = false;
            if (skip_odd && e % 2) continue;
            std::vector<double> a(I);
            for (int n = 0; n < I; ++n) {
                double g = 0, u = 0;
                for (int c = 0; c < H; ++c) {
                    double xv = D::load(x.data(), (size_t)t * H + c);
                    g += gate[e].ref[(size_t)n * H + c] * xv;
                    u += up[e].ref[(size_t)n * H + c] * xv;
                }
                a[n] = g / (1 + std::exp(-g)) * u;
            }
            for (int h = 0; h < H; ++h) {
                double d = 0;
                for (int n = 0; n < I; ++n) d += down[e].ref[(size_t)h * I + n] * a[n];
                out[h] += w * d;
            }
        }
        double scale = 0;
        for (double v : out) scale = std::max(scale, std::fabs(v));
        for (int h = 0; h < H; ++h)
        {
            float err = std::fabs(D::load(y.data(), (size_t)t * H + h) - out[h]) / (scale + 1e-6);
            worst = std::isnan(err) ? INFINITY : std::max(worst, err);
        }
    }
    bool ok = route_ok && worst <= tol;
    fails += !ok;
    static const char* fmts[] = {"dense", "fp8", "nvfp4"};
    printf("%-5s %-6s T=%d E=%2d K=%d H=%3d I=%3d norm=%d ep=%d  routing %s, max rel err %.2e %s\n", name, fmts[fmt], T,
           E, K, H, I, norm, skip_odd, route_ok ? "ok" : "MISMATCH", worst, ok ? "ok" : "FAIL");
}

int main() {
    for (int fmt = 0; fmt < 3; ++fmt) {
        run<F32>("f32", fmt, 3, 12, 4, 64, 48, true, false, 1e-5f);
        run<BF16>("bf16", fmt, 2, 40, 8, 96, 32, false, false, 2e-2f);
        run<F16>("f16", fmt, 3, 12, 4, 64, 32, true, true, 5e-3f);
    }
    printf(fails ? "%d FAILED\n" : "all ok\n", fails);
    return fails != 0;
}
