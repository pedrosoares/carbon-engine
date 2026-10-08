// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! End-to-end equivalence tests: every parallel topology (pipeline, tensor/expert
//! parallel, RPC workers with local or shipped weights, MoE offload strategies) must
//! produce exactly the tokens of the single-device baseline.

use carbon_engine::engine::{Engine, EngineConfig};
use carbon_engine::model::moe::{MoeOptions, MoeStrategy};
use carbon_engine::rpc::server::{serve_listener, WorkerConfig};
use carbon_engine::sampling::SamplingParams;
use carbon_engine::scheduler::{self, Event, SchedulerConfig, Usage};
use carbon_engine::testing::{write_tiny_model, TinySpec};
use std::net::TcpListener;
use std::path::{Path, PathBuf};

fn tiny(arch: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(arch);
    let mut spec = TinySpec::new(arch);
    spec.layers = 4;
    write_tiny_model(&path, &spec).unwrap();
    (dir, path)
}

fn prompts() -> Vec<Vec<u32>> {
    let base: Vec<u32> = (0..40).map(|i| (i * 7 + 3) % 250).collect();
    let mut shared = base[..30].to_vec();
    shared.extend([1, 2, 3, 4, 5]);
    let long: Vec<u32> = (0..150).map(|i| (i * 13 + 1) % 250).collect();
    vec![base.clone(), shared, base, vec![10, 20, 30, 40, 50], long]
}

fn run(
    ec: &EngineConfig,
    prompts: &[Vec<u32>],
    max_tokens: usize,
    prefill_budget: usize,
) -> Vec<(Vec<u32>, Usage)> {
    run_with_stats(ec, prompts, max_tokens, prefill_budget).0
}

/// [`run`], also returning the stage stats once every request finished.
fn run_with_stats(
    ec: &EngineConfig,
    prompts: &[Vec<u32>],
    max_tokens: usize,
    prefill_budget: usize,
) -> (
    Vec<(Vec<u32>, Usage)>,
    Vec<carbon_engine::model::StageStats>,
) {
    let engine = Engine::load(ec).unwrap();
    let handle = scheduler::spawn(
        engine,
        SchedulerConfig {
            max_running: 8,
            max_prefill_tokens: prefill_budget,
            eos_token_ids: vec![],
            anchor_token: None,
        },
    )
    .unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let mut rxs = vec![];
        for p in prompts {
            let params = SamplingParams {
                max_tokens,
                ignore_eos: true,
                ..Default::default()
            };
            rxs.push(handle.submit(p.clone(), params).1);
        }
        let mut out = vec![];
        for mut rx in rxs {
            let mut toks = vec![];
            loop {
                match rx.recv().await.expect("engine dropped the request") {
                    Event::Token(t) => toks.push(t),
                    Event::Done { usage, .. } => {
                        out.push((toks, usage));
                        break;
                    }
                    Event::Error(e) => panic!("generation failed: {e}"),
                }
            }
        }
        let stages = handle.stats().await.map(|s| s.stages).unwrap_or_default();
        (out, stages)
    })
}

fn tokens(r: Vec<(Vec<u32>, Usage)>) -> Vec<Vec<u32>> {
    r.into_iter().map(|(t, _)| t).collect()
}

/// Map the logical "cpu" devices of a test onto real hardware when
/// `CARBON_TEST_DEVICE=cuda` (or `metal`) is set: the i-th device becomes `cuda:{i % n}`.
fn devices(list: &[&str]) -> Vec<String> {
    match std::env::var("CARBON_TEST_DEVICE")
        .ok()
        .filter(|d| d != "cpu")
    {
        Some(kind) => {
            let n: usize = std::env::var("CARBON_TEST_NUM_DEVICES")
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(1);
            (0..list.len())
                .map(|i| format!("{kind}:{}", i % n))
                .collect()
        }
        None => list.iter().map(|s| s.to_string()).collect(),
    }
}

fn config(model: &Path, devs: &[&str], tp: usize) -> EngineConfig {
    let mut ec = EngineConfig::new(model);
    ec.devices = devices(devs);
    // f32 everywhere: different reduction orders must not flip greedy tokens
    ec.dtype = Some(candle_core::DType::F32);
    ec.tp_size = tp;
    ec.kv_tokens = Some(4096);
    ec
}

fn spawn_worker(devs: &[&str], tp: usize, model: Option<&Path>) -> String {
    spawn_worker_with_token(devs, tp, model, None)
}

fn spawn_worker_with_token(
    devs: &[&str],
    tp: usize,
    model: Option<&Path>,
    token: Option<&str>,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let cfg = WorkerConfig {
        listen: addr.clone(),
        devices: devices(devs),
        tp_size: tp,
        model_dir: model.map(Path::to_path_buf),
        name: "test-worker".into(),
        moe: MoeOptions::default(),
        token: token.map(String::from),
        cuda_graphs: false,
    };
    std::thread::spawn(move || serve_listener(listener, cfg));
    addr
}

fn check_topologies(arch: &str) {
    let (_d, model) = tiny(arch);
    let ps = prompts();
    let baseline = tokens(run(&config(&model, &["cpu"], 1), &ps, 12, 4096));
    assert!(baseline.iter().all(|t| t.len() == 12));
    // identical prompts produce identical outputs, regardless of prefix-cache state
    assert_eq!(baseline[0], baseline[2]);

    let mut cases: Vec<(&str, EngineConfig)> = vec![
        ("chunked prefill", config(&model, &["cpu"], 1)),
        ("pipeline x3", config(&model, &["cpu", "cpu", "cpu"], 1)),
        ("tensor parallel x2", config(&model, &["cpu", "cpu"], 2)),
    ];
    cases.push((
        "tp x2 + pp x2",
        config(&model, &["cpu", "cpu", "cpu", "cpu"], 2),
    ));
    let mut uneven = config(&model, &["cpu", "cpu"], 1);
    uneven.layer_split = vec![3.0, 1.0];
    cases.push(("uneven pipeline", uneven));

    let w = spawn_worker(&["cpu"], 1, Some(&model));
    let mut rpc = config(&model, &["cpu"], 1);
    rpc.rpc_workers = vec![w];
    rpc.layer_split = vec![1.0, 3.0];
    cases.push(("rpc worker with local weights", rpc));

    let w = spawn_worker(&["cpu"], 1, None);
    let mut shipped = config(&model, &["cpu"], 1);
    shipped.rpc_workers = vec![w];
    cases.push(("rpc worker with shipped weights", shipped));

    let w1 = spawn_worker(&["cpu", "cpu"], 2, None);
    let w2 = spawn_worker(&["cpu", "cpu"], 1, Some(&model));
    let mut multi = config(&model, &["cpu"], 1);
    multi.rpc_workers = vec![w1, w2];
    multi.layer_split = vec![0.0, 2.0, 2.0];
    cases.push(("head-only coordinator + 2 workers (tp inside one)", multi));

    for (name, ec) in cases {
        let budget = if name == "chunked prefill" { 16 } else { 64 };
        let got = tokens(run(&ec, &ps, 12, budget));
        assert_eq!(
            got, baseline,
            "{arch}: topology '{name}' diverged from the single-device baseline"
        );
    }
}

#[test]
fn llama_topologies_match_single_device() {
    check_topologies("llama");
}

#[test]
fn qwen3_moe_topologies_match_single_device() {
    check_topologies("qwen3_moe");
}

#[test]
fn qwen2_bias_tensor_parallel() {
    let (_d, model) = tiny("qwen2");
    let ps = prompts();
    let baseline = tokens(run(&config(&model, &["cpu"], 1), &ps, 8, 4096));
    let tp = tokens(run(&config(&model, &["cpu", "cpu"], 2), &ps, 8, 4096));
    assert_eq!(tp, baseline);
}

#[test]
fn moe_strategies_match_fused() {
    for arch in ["qwen3_moe", "mixtral"] {
        let (_d, model) = tiny(arch);
        let ps = prompts();
        let baseline = tokens(run(&config(&model, &["cpu"], 1), &ps, 10, 4096));
        for (strategy, cache, fetch) in [
            (MoeStrategy::Offload, Some(3), None),
            (MoeStrategy::Cpu, None, None),
            (MoeStrategy::Hybrid, Some(4), Some(1)),
        ] {
            let mut ec = config(&model, &["cpu"], 1);
            ec.moe = MoeOptions {
            offload_on_cpu: true,
                strategy,
                cache_size: cache,
                hybrid_max_fetch: fetch,
                ..Default::default()
            };
            let got = tokens(run(&ec, &ps, 10, 4096));
            assert_eq!(got, baseline, "{arch}: moe strategy {strategy:?} diverged");
        }
        // expert parallel across ranks combined with offload
        let mut ec = config(&model, &["cpu", "cpu"], 2);
        ec.moe = MoeOptions {
            offload_on_cpu: true,
            strategy: MoeStrategy::Offload,
            cache_size: Some(2),
            hybrid_max_fetch: None,
            ..Default::default()
        };
        assert_eq!(
            tokens(run(&ec, &ps, 10, 4096)),
            baseline,
            "{arch}: expert parallel + offload diverged"
        );
    }
}

#[test]
fn prefix_cache_reuses_kv() {
    let (_d, model) = tiny("qwen3");
    let ec = config(&model, &["cpu"], 1);
    let engine = Engine::load(&ec).unwrap();
    let handle = scheduler::spawn(
        engine,
        SchedulerConfig {
            max_running: 4,
            max_prefill_tokens: 4096,
            eos_token_ids: vec![],
            anchor_token: None,
        },
    )
    .unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let prompt: Vec<u32> = (0..64).map(|i| (i * 5 + 2) % 250).collect();
    let go = |p: Vec<u32>| {
        let h = handle.clone();
        rt.block_on(async move {
            let (_, mut rx) = h.submit(
                p,
                SamplingParams {
                    max_tokens: 6,
                    ignore_eos: true,
                    ..Default::default()
                },
            );
            let mut toks = vec![];
            loop {
                match rx.recv().await.unwrap() {
                    Event::Token(t) => toks.push(t),
                    Event::Done { usage, .. } => return (toks, usage),
                    Event::Error(e) => panic!("{e}"),
                }
            }
        })
    };
    let (a, ua) = go(prompt.clone());
    assert_eq!(ua.cached_tokens, 0);
    let (b, ub) = go(prompt.clone());
    assert_eq!(a, b);
    assert_eq!(
        ub.cached_tokens, 63,
        "whole prompt but the last token should hit the cache"
    );
    // a continuation of the first conversation reuses prompt + generated tokens
    let mut cont = prompt.clone();
    cont.extend(&a[..5]);
    cont.extend([7, 8, 9]);
    let (_, uc) = go(cont);
    assert_eq!(uc.cached_tokens, 69);
    let stats = rt.block_on(handle.stats()).unwrap();
    assert_eq!(stats.running, 0);
    assert!(stats.prefix_cached_tokens >= 72);
    assert_eq!(
        stats.kv_free + stats.prefix_cached_tokens,
        stats.kv_capacity,
        "KV slots leaked"
    );
}

#[test]
fn small_kv_cache_queues_and_evicts() {
    let (_d, model) = tiny("llama");
    let mut ec = config(&model, &["cpu"], 1);
    ec.kv_tokens = Some(200);
    let ps: Vec<Vec<u32>> = (0..6)
        .map(|k| (0..60).map(|i| (i * 3 + k * 11) % 250).collect())
        .collect();
    let res = run(&ec, &ps, 20, 4096);
    let mut big = config(&model, &["cpu"], 1);
    big.kv_tokens = Some(4096);
    assert_eq!(tokens(res), tokens(run(&big, &ps, 20, 4096)));
}

#[test]
fn rejects_oversized_requests() {
    let (_d, model) = tiny("llama");
    let mut ec = config(&model, &["cpu"], 1);
    ec.kv_tokens = Some(64);
    let engine = Engine::load(&ec).unwrap();
    let handle = scheduler::spawn(
        engine,
        SchedulerConfig {
            max_running: 2,
            max_prefill_tokens: 64,
            eos_token_ids: vec![],
            anchor_token: None,
        },
    )
    .unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let (_, mut rx) = handle.submit(vec![1; 100], SamplingParams::default());
        assert!(matches!(rx.recv().await.unwrap(), Event::Error(_)));
        let (_, mut rx) = handle.submit(
            vec![1; 10],
            SamplingParams {
                max_tokens: 1000,
                ..Default::default()
            },
        );
        assert!(matches!(rx.recv().await.unwrap(), Event::Error(_)));
        let (_, mut rx) = handle.submit(vec![9999], SamplingParams::default());
        assert!(matches!(rx.recv().await.unwrap(), Event::Error(_)));
    });
}

#[test]
fn live_cache_rebuild() {
    let (_d, model) = tiny("qwen3_moe");
    let w = spawn_worker(&["cpu"], 1, Some(&model));
    let mut ec = config(&model, &["cpu"], 1);
    ec.rpc_workers = vec![w];
    ec.moe = MoeOptions {
            offload_on_cpu: true,
        strategy: MoeStrategy::Offload,
        cache_size: Some(6),
        hybrid_max_fetch: None,
        ..Default::default()
    };
    let engine = Engine::load(&ec).unwrap();
    let handle = scheduler::spawn(
        engine,
        SchedulerConfig {
            max_running: 4,
            max_prefill_tokens: 4096,
            eos_token_ids: vec![],
            anchor_token: None,
        },
    )
    .unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let prompt: Vec<u32> = (0..30).map(|i| (i * 9 + 4) % 250).collect();
    let go = |p: Vec<u32>| {
        let h = handle.clone();
        rt.block_on(async move {
            let (_, mut rx) = h.submit(
                p,
                SamplingParams {
                    max_tokens: 8,
                    ignore_eos: true,
                    ..Default::default()
                },
            );
            let mut toks = vec![];
            loop {
                match rx.recv().await.unwrap() {
                    Event::Token(t) => toks.push(t),
                    Event::Done { .. } => return toks,
                    Event::Error(e) => panic!("{e}"),
                }
            }
        })
    };
    let before = go(prompt.clone());
    rt.block_on(handle.rebuild(scheduler::RebuildRequest {
        kv_tokens: Some(1500),
        expert_cache: Some(2),
    }))
    .unwrap();
    let st = rt.block_on(handle.stats()).unwrap();
    assert_eq!(
        (st.kv_capacity, st.kv_free, st.prefix_cached_tokens),
        (1500, 1500, 0)
    );
    assert_eq!(st.stages.len(), 2, "local + remote stage");
    // the CUDA slot cache never holds less than one layer's experts (8 here)
    let cap = if std::env::var("CARBON_TEST_DEVICE").as_deref() == Ok("cuda") { 8 } else { 2 };
    for s in &st.stages {
        assert_eq!(s.kv_slots, 1500);
        assert_eq!(s.experts.as_ref().unwrap().capacity, cap);
        assert!(s.experts.as_ref().unwrap().resident <= cap);
    }
    assert_eq!(
        go(prompt),
        before,
        "outputs must not change after a rebuild"
    );
}

#[test]
fn rpc_token_is_enforced() {
    let (_d, model) = tiny("llama");
    let ps = vec![vec![5, 6, 7, 8]];
    let baseline = tokens(run(&config(&model, &["cpu"], 1), &ps, 4, 4096));

    let w = spawn_worker_with_token(&["cpu"], 1, Some(&model), Some("s3cret"));
    let mut ec = config(&model, &["cpu"], 1);
    ec.rpc_workers = vec![w.clone()];
    let err = Engine::load(&ec)
        .err()
        .expect("missing token must be rejected");
    assert!(
        format!("{err:#}").contains("authentication failed"),
        "{err:#}"
    );
    ec.rpc_token = Some("wrong!".into());
    assert!(Engine::load(&ec).is_err());
    ec.rpc_token = Some("s3cret".into());
    assert_eq!(tokens(run(&ec, &ps, 4, 4096)), baseline);
}

#[test]
fn qwen35_hybrid_topologies_match_single_device() {
    // hybrid Gated-DeltaNet + attention: no prefix reuse, recurrent state slots per request
    check_topologies("qwen3_5_moe");
}

#[test]
fn qwen38_topologies_match_single_device() {
    // hyper-connections, PLE (token ids reach every stage and worker), QSA sparse attention
    check_topologies("qwen4_exp");
}

#[test]
fn qwen35_hybrid_moe_strategies_and_chunking() {
    hybrid_moe_strategies_and_chunking("qwen3_5_moe");
}

#[test]
fn qwen38_moe_strategies_and_chunking() {
    hybrid_moe_strategies_and_chunking("qwen4_exp");
}

fn hybrid_moe_strategies_and_chunking(arch: &str) {
    let (_d, model) = tiny(arch);
    let ps = prompts();
    let baseline = tokens(run(&config(&model, &["cpu"], 1), &ps, 10, 4096));
    // tiny prefill chunks carry conv + delta-rule state across chunk boundaries
    assert_eq!(
        tokens(run(&config(&model, &["cpu"], 1), &ps, 10, 7)),
        baseline,
        "chunked prefill"
    );
    for (strategy, cache) in [
        (MoeStrategy::Offload, Some(3)),
        (MoeStrategy::Hybrid, Some(4)),
        (MoeStrategy::Cpu, None),
    ] {
        let mut ec = config(&model, &["cpu"], 1);
        ec.moe = MoeOptions {
            offload_on_cpu: true,
            strategy,
            cache_size: cache,
            hybrid_max_fetch: Some(1),
            ..Default::default()
        };
        assert_eq!(tokens(run(&ec, &ps, 10, 4096)), baseline, "{strategy:?}");
    }
    // the same prompt twice must not reuse stale recurrent state
    let again = tokens(run(
        &config(&model, &["cpu"], 1),
        &[ps[0].clone(), ps[0].clone()],
        10,
        4096,
    ));
    assert_eq!(again[0], again[1]);
    assert_eq!(again[0], baseline[0]);
}

#[test]
fn qwen35_more_requests_than_state_slots() {
    more_requests_than_state_slots("qwen3_5_moe");
}

#[test]
fn qwen38_more_requests_than_state_slots() {
    more_requests_than_state_slots("qwen4_exp");
}

fn more_requests_than_state_slots(arch: &str) {
    let (_d, model) = tiny(arch);
    let mut ec = config(&model, &["cpu"], 1);
    ec.max_running = 2;
    let ps = prompts();
    let few = tokens(run(&ec, &ps, 8, 4096));
    assert_eq!(few, tokens(run(&config(&model, &["cpu"], 1), &ps, 8, 4096)));
}

fn tiny_quant(
    arch: &str,
    quant: carbon_engine::testing::TinyQuant,
) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let (q, d) = (dir.path().join("quant"), dir.path().join("dense"));
    carbon_engine::testing::write_tiny_quantized(&q, &d, &TinySpec::new(arch), quant).unwrap();
    (dir, q, d)
}

#[test]
fn quantized_checkpoints_match_their_dequantized_twin() {
    use carbon_engine::testing::TinyQuant;
    for (arch, quant) in [
        ("qwen3_moe", TinyQuant::Nvfp4ModelOpt),
        ("qwen3_moe", TinyQuant::Nvfp4CompressedTensors),
        ("qwen3_moe", TinyQuant::Fp8Block),
        ("qwen3_5_moe", TinyQuant::Fp8Block),
        ("qwen3_5_moe", TinyQuant::Nvfp4ModelOpt),
        ("mixtral", TinyQuant::Nvfp4ModelOpt),
        ("llama", TinyQuant::Nvfp4ModelOpt),
    ] {
        let (_d, qdir, ddir) = tiny_quant(arch, quant);
        let ps = prompts();
        let reference = tokens(run(&config(&ddir, &["cpu"], 1), &ps, 8, 4096));
        // experts kept quantized (default), dequantized at load, offloaded, expert-parallel, shipped over RPC
        let mut cases: Vec<(&str, EngineConfig)> =
            vec![("kept quantized", config(&qdir, &["cpu"], 1))];
        let mut deq = config(&qdir, &["cpu"], 1);
        deq.moe.dequant_fp8 = true;
        cases.push(("dequantized at load", deq));
        let mut off = config(&qdir, &["cpu"], 1);
        off.moe = MoeOptions {
            offload_on_cpu: true,
            strategy: MoeStrategy::Hybrid,
            cache_size: Some(3),
            hybrid_max_fetch: Some(1),
            ..Default::default()
        };
        cases.push(("hybrid offload", off));
        // offload with half the MoE layers on the CPU, cache sized by rate
        let mut cpu_layers = config(&qdir, &["cpu"], 1);
        cpu_layers.moe = MoeOptions {
            offload_on_cpu: true,
            strategy: MoeStrategy::Offload,
            cache_rate: Some(0.5),
            cpu_layers: "0.5".parse().unwrap(),
            ..Default::default()
        };
        cases.push(("offload + cpu layers", cpu_layers));
        cases.push(("tp x2", config(&qdir, &["cpu", "cpu"], 2)));
        let w = spawn_worker(&["cpu"], 1, None);
        let mut rpc = config(&qdir, &["cpu"], 1);
        rpc.rpc_workers = vec![w];
        cases.push(("rpc shipped", rpc));
        for (name, ec) in cases {
            assert_eq!(
                tokens(run(&ec, &ps, 8, 4096)),
                reference,
                "{arch} {quant:?}: {name}"
            );
        }
    }
}

/// Run requests one after another on one engine so later ones can hit earlier ones' cache.
fn run_sequential(
    ec: &EngineConfig,
    prompts: &[Vec<u32>],
    max_tokens: usize,
    budget: usize,
) -> Vec<(Vec<u32>, Usage)> {
    run_sequential_anchored(ec, prompts, max_tokens, budget, None)
}

fn run_sequential_anchored(
    ec: &EngineConfig,
    prompts: &[Vec<u32>],
    max_tokens: usize,
    budget: usize,
    anchor_token: Option<u32>,
) -> Vec<(Vec<u32>, Usage)> {
    let engine = Engine::load(ec).unwrap();
    let handle = scheduler::spawn(
        engine,
        SchedulerConfig {
            max_running: 4,
            max_prefill_tokens: budget,
            eos_token_ids: vec![],
            anchor_token,
        },
    )
    .unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    prompts
        .iter()
        .map(|p| {
            rt.block_on(async {
                let params = SamplingParams {
                    max_tokens,
                    ignore_eos: true,
                    ..Default::default()
                };
                let (_, mut rx) = handle.submit(p.clone(), params);
                let mut toks = vec![];
                loop {
                    match rx.recv().await.unwrap() {
                        Event::Token(t) => toks.push(t),
                        Event::Done { usage, .. } => return (toks, usage),
                        Event::Error(e) => panic!("{e}"),
                    }
                }
            })
        })
        .collect()
}

#[test]
fn qwen35_prefix_snapshots_reuse_state_exactly() {
    prefix_snapshots_reuse_state_exactly("qwen3_5_moe");
}

#[test]
fn qwen38_prefix_snapshots_reuse_state_exactly() {
    prefix_snapshots_reuse_state_exactly("qwen4_exp");
}

fn prefix_snapshots_reuse_state_exactly(arch: &str) {
    let (_d, model) = tiny(arch);
    let base: Vec<u32> = (0..50).map(|i| (i * 11 + 5) % 250).collect();
    // first turn, then: same prompt, a continuation of the first conversation, and a prompt
    // that shares only part of `base` (no snapshot at that boundary -> partial or no reuse)
    let first = run_sequential(
        &config(&model, &["cpu"], 1),
        std::slice::from_ref(&base),
        6,
        4096,
    )[0]
    .0
    .clone();
    let mut cont = base.clone();
    cont.extend(&first[..5]);
    cont.extend([9, 8, 7, 6]);
    let mut partial = base[..30].to_vec();
    partial.extend([1, 2, 3]);
    let reqs = vec![
        base.clone(),
        base.clone(),
        cont.clone(),
        partial.clone(),
        cont.clone(),
    ];

    let mut fresh = config(&model, &["cpu"], 1);
    fresh.state_snapshots = Some(0);
    let expected: Vec<Vec<u32>> = reqs
        .iter()
        .map(|p| {
            run_sequential(&fresh, std::slice::from_ref(p), 6, 4096)[0]
                .0
                .clone()
        })
        .collect();

    let w = spawn_worker(&["cpu"], 1, None);
    let mut rpc = config(&model, &["cpu"], 1);
    rpc.rpc_workers = vec![w];
    for (name, ec, budget) in [
        ("single device", config(&model, &["cpu"], 1), 4096),
        ("small chunks", config(&model, &["cpu"], 1), 7),
        ("tp x2", config(&model, &["cpu", "cpu"], 2), 4096),
        ("rpc worker", rpc, 4096),
    ] {
        let got = run_sequential(&ec, &reqs, 6, budget);
        for (i, (toks, _)) in got.iter().enumerate() {
            assert_eq!(
                toks, &expected[i],
                "{name}: request {i} diverged from the cache-free run"
            );
        }
        let cached: Vec<usize> = got.iter().map(|(_, u)| u.cached_tokens).collect();
        assert_eq!(cached[0], 0, "{name}");
        assert_eq!(
            cached[1],
            base.len() - 1,
            "{name}: identical prompt resumes at len-1"
        );
        // first request finished with prompt + 5 generated tokens of KV and state
        assert_eq!(
            cached[2],
            base.len() + 5,
            "{name}: continuation resumes after the first turn"
        );
        assert_eq!(
            cached[3], 0,
            "{name}: no snapshot inside the shared 30-token prefix"
        );
        assert_eq!(cached[4], cont.len() - 1, "{name}: repeated continuation");
    }

    // with a single snapshot slot, older snapshots are evicted but results stay exact
    let mut tight = config(&model, &["cpu"], 1);
    tight.state_snapshots = Some(1);
    let got = run_sequential(&tight, &reqs, 6, 4096);
    for (i, (toks, _)) in got.iter().enumerate() {
        assert_eq!(toks, &expected[i], "1 snapshot: request {i}");
    }
}

/// CUDA-graph decode must reproduce eager decoding exactly (dense, MoE, hybrid, quantized
/// experts). Graphs only engage on a CUDA device (`CARBON_TEST_DEVICE=cuda`); elsewhere the flag
/// must simply have no effect. Batches of 1..=5 sequences decode long enough for every batch
/// size to be captured and then replayed many times, with sequences joining and finishing.
#[test]
fn cuda_graphs_match_eager() {
    use carbon_engine::testing::TinyQuant;
    // head_dim 32: the fused attention kernels (and so graphs) need a multiple of 32
    let spec = |arch: &str| {
        let mut s = TinySpec::new(arch);
        s.head_dim = 32;
        s
    };
    let mut models: Vec<(String, tempfile::TempDir, PathBuf)> = vec![];
    // qwen4_exp: hyper-connections, PLE embeddings gathered before the graph, QSA selection on
    // the device (the 150-token prompt passes the 16-token budget)
    for arch in ["llama", "qwen3_moe", "qwen3_5_moe", "qwen4_exp"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(arch);
        write_tiny_model(&path, &spec(arch)).unwrap();
        models.push((arch.into(), dir, path));
    }
    for quant in [TinyQuant::Fp8Block, TinyQuant::Nvfp4ModelOpt] {
        let dir = tempfile::tempdir().unwrap();
        let (q, d) = (dir.path().join("quant"), dir.path().join("dense"));
        carbon_engine::testing::write_tiny_quantized(&q, &d, &spec("qwen3_5_moe"), quant).unwrap();
        models.push((format!("qwen3_5_moe {quant:?}"), dir, q));
    }
    for (name, _dir, path) in &models {
        let ps = prompts();
        let eager = config(path, &["cpu"], 1);
        let mut graphs = config(path, &["cpu"], 1);
        graphs.cuda_graphs = true;
        let reference = tokens(run(&eager, &ps, 24, 64));
        let (out, stages) = run_with_stats(&graphs, &ps, 24, 64);
        assert_eq!(tokens(out), reference, "{name}");
        let captured: usize = stages.iter().map(|s| s.cuda_graphs).sum();
        if std::env::var("CARBON_TEST_DEVICE").as_deref() == Ok("cuda") {
            let why: Vec<_> = stages.iter().filter_map(|s| s.cuda_graphs_disabled.clone()).collect();
            assert!(captured > 0, "{name}: no decode step was captured; graphs disabled: {why:?}");
        } else {
            assert_eq!(captured, 0, "{name}");
        }
    }
}

/// Prefill in 3-token chunks takes the multi-token decode path on CUDA (one decode row per
/// token, recurrent layers advanced one token at a time); it must produce what whole-prompt
/// prefill does. head_dim 32 so the fused attention kernels apply.
#[test]
fn multi_token_steps_match_prefill() {
    for arch in ["llama", "qwen3_5_moe", "qwen4_exp"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(arch);
        let mut spec = TinySpec::new(arch);
        spec.head_dim = 32;
        write_tiny_model(&path, &spec).unwrap();
        let ec = config(&path, &["cpu"], 1);
        let ps = prompts();
        let whole = tokens(run(&ec, &ps, 12, 4096));
        let chunked = tokens(run(&ec, &ps, 12, 3));
        assert_eq!(chunked, whole, "{arch}");
        // the same steps replayed from CUDA graphs (keyed by sequences x tokens per sequence)
        let mut graphs = config(&path, &["cpu"], 1);
        graphs.cuda_graphs = true;
        let (out, stages) = run_with_stats(&graphs, &ps, 12, 3);
        assert_eq!(tokens(out), whole, "{arch} with cuda graphs");
        if std::env::var("CARBON_TEST_DEVICE").as_deref() == Ok("cuda") {
            let why: Vec<_> = stages.iter().filter_map(|s| s.cuda_graphs_disabled.clone()).collect();
            assert!(stages.iter().any(|s| s.cuda_graphs > 0), "{arch}: nothing captured: {why:?}");
        }
    }
}

/// MTP speculative decoding (tiny Qwen3.8 with a random MTP head, so most drafts are rejected
/// and rolled back): greedy output must equal plain decoding, one request at a time and three
/// at once (batched verify steps). On CUDA only; elsewhere the head only drafts.
#[test]
fn mtp_speculation_matches_plain_decoding() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("qwen4_exp");
    let mut spec = TinySpec::new("qwen4_exp");
    spec.head_dim = 32;
    write_tiny_model(&path, &spec).unwrap();
    let ps = prompts();
    let plain = config(&path, &["cpu"], 1);
    let cuda = std::env::var("CARBON_TEST_DEVICE").as_deref() == Ok("cuda");
    for (drafts, concurrent) in [(1, 1usize), (1, 2), (2, 1), (2, 2), (2, 5)] {
    let mut mtp = config(&path, &["cpu"], 1);
    mtp.mtp_speculate = true;
    mtp.mtp_drafts = drafts;
    mtp.cuda_graphs = true;
        let want: Vec<Vec<u32>> = ps.iter().map(|p| tokens(run_mtp(&plain, std::slice::from_ref(p), 24, concurrent).0)[0].clone()).collect();
        let mut checked = 0;
        let mut got = vec![];
        for group in ps.chunks(concurrent) {
            let (out, st) = run_mtp(&mtp, group, 24, concurrent);
            checked += st.mtp_drafts;
            got.extend(tokens(out));
        }
        let want: Vec<Vec<u32>> = if concurrent == 1 {
            want
        } else {
            ps.chunks(concurrent).flat_map(|g| tokens(run_mtp(&plain, g, 24, concurrent).0)).collect()
        };
        assert_eq!(got, want, "{concurrent} at once");
        // more requests than a verify step takes never speculate (the head only keeps its KV)
        if cuda && concurrent <= carbon_engine::engine::MTP_MAX_BATCH {
            assert!(checked > 0, "no verify step ran ({concurrent} at once)");
        }
    }
}

/// The accept path: drafts from an oracle that knows the plain run's output, wrong at every
/// third position, so verify steps keep all, some or none of their drafts.
#[test]
fn mtp_speculation_accepts_and_rejects() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("qwen4_exp");
    let mut spec = TinySpec::new("qwen4_exp");
    spec.head_dim = 32;
    write_tiny_model(&path, &spec).unwrap();
    let ps = &prompts()[..2];
    let plain = config(&path, &["cpu"], 1);
    let want = tokens(run_mtp(&plain, ps, 24, 2).0);
    for drafts in [1, 2] {
    let mut mtp = config(&path, &["cpu"], 1);
    mtp.mtp_speculate = true;
    mtp.mtp_drafts = drafts;
    mtp.cuda_graphs = true;
    // every full sequence (prompt + output): the oracle finds the request by its prompt
    let full: Vec<Vec<u32>> = ps.iter().zip(&want).map(|(p, o)| p.iter().chain(o).copied().collect()).collect();
    let oracle = std::sync::Arc::new(move |toks: &[u32]| -> u32 {
        let f = full.iter().find(|f| f.len() > toks.len() && f.starts_with(toks));
        match f {
            // right unless the position is a multiple of 3: with two drafts some steps accept both
            Some(f) if !toks.len().is_multiple_of(3) => f[toks.len()],
            Some(f) => f[toks.len()].wrapping_add(1) % 200,
            None => 0,
        }
    });
    let cuda = std::env::var("CARBON_TEST_DEVICE").as_deref() == Ok("cuda");
    let (out, st) = run_mtp_with(&mtp, ps, 24, 2, Some(oracle.clone()));
    assert_eq!(tokens(out), want, "{drafts} drafts");
    if cuda {
        assert!(st.mtp_accepted > 0 && st.mtp_accepted < st.mtp_drafts, "{} of {}", st.mtp_accepted, st.mtp_drafts);
    }
    // offloaded experts: on CUDA the head's experts join the slot cache (NVFP4)
    mtp.moe = MoeOptions {
        strategy: MoeStrategy::Offload,
        ..Default::default()
    };
    let (out, st) = run_mtp_with(&mtp, ps, 24, 2, Some(oracle));
    assert_eq!(tokens(out), want, "offload, {drafts} drafts");
    if cuda {
        assert!(st.mtp_accepted > 0, "offload");
    }
    }
}

/// The MTP head's KV-only pass (used while it cannot speculate) writes the KV its full pass
/// does: after a prompt filled either way, the next draft is the same.
#[test]
fn mtp_kv_only_pass_matches_full_pass() {
    use carbon_engine::batch::{BatchMeta, SeqMeta};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("qwen4_exp");
    let mut spec = TinySpec::new("qwen4_exp");
    spec.head_dim = 32;
    write_tiny_model(&path, &spec).unwrap();
    let mut ec = config(&path, &["cpu"], 1);
    ec.mtp_speculate = true;
    let prompt: Vec<u32> = (0..40).map(|i| (i * 11 + 5) % 250).collect();
    let meta = |toks: &[u32], start: usize| {
        let n = toks.len();
        let slots: Vec<u32> = (0..(start + n) as u32).collect();
        std::sync::Arc::new(BatchMeta {
            positions: (start as u32..(start + n) as u32).collect(),
            out_loc: (start as u32..(start + n) as u32).collect(),
            seqs: vec![SeqMeta {
                seq_id: 1,
                q_len: n,
                kv_len: start + n,
                state_slot: 0,
                slots: std::sync::Arc::new(slots),
                prev_tokens: vec![],
                token_state_slots: vec![],
            }],
            is_decode: start > 0,
            tokens: toks.to_vec(),
            mrope: vec![],
            mrope_layout: vec![],
        })
    };
    let draft_after = |kv_only: bool| -> (u32, Vec<f32>) {
        let engine = Engine::load(&ec).unwrap();
        let n = prompt.len();
        let m = meta(&prompt[..n - 1], 0);
        let (_, hidden) = engine.forward_hidden(m.clone(), &prompt[..n - 1], &[]).unwrap();
        if kv_only {
            engine.mtp_fill_kv(&m, &hidden, &prompt[1..]).unwrap();
        } else {
            engine.mtp_draft(&m, &hidden, &prompt[1..], &[]).unwrap();
        }
        // the last prompt token, then a draft for the token after an arbitrary next token
        let m = meta(&prompt[n - 1..], n - 1);
        let (_, hidden) = engine.forward_hidden(m.clone(), &prompt[n - 1..], &[0]).unwrap();
        let (d, multi) = engine.mtp_draft(&m, &hidden, &[7], &[0]).unwrap();
        (d[0], multi.to_dtype(candle_core::DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap())
    };
    let (full, a) = draft_after(false);
    let (kv_only, b) = draft_after(true);
    assert_eq!(full, kv_only);
    let diff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    assert!(diff < 1e-5, "head streams differ by {diff}");
}

/// Run `prompts` concurrently; returns the outputs and the scheduler stats.
fn run_mtp(ec: &EngineConfig, prompts: &[Vec<u32>], max_tokens: usize, max_running: usize) -> (Vec<(Vec<u32>, Usage)>, scheduler::SchedulerStats) {
    run_mtp_with(ec, prompts, max_tokens, max_running, None)
}

type Oracle = carbon_engine::engine::MtpOracle;

fn run_mtp_with(
    ec: &EngineConfig,
    prompts: &[Vec<u32>],
    max_tokens: usize,
    max_running: usize,
    oracle: Option<Oracle>,
) -> (Vec<(Vec<u32>, Usage)>, scheduler::SchedulerStats) {
    let mut engine = Engine::load(ec).unwrap();
    engine.mtp_oracle = oracle;
    let handle = scheduler::spawn(
        engine,
        SchedulerConfig {
            max_running,
            max_prefill_tokens: 4096,
            eos_token_ids: vec![],
            anchor_token: None,
        },
    )
    .unwrap();
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let rxs: Vec<_> = prompts
            .iter()
            .map(|p| {
                let params = SamplingParams { max_tokens, ignore_eos: true, ..Default::default() };
                handle.submit(p.clone(), params).1
            })
            .collect();
        let mut out = vec![];
        for mut rx in rxs {
            let mut toks = vec![];
            loop {
                match rx.recv().await.expect("engine dropped the request") {
                    Event::Token(t) => toks.push(t),
                    Event::Done { usage, .. } => {
                        out.push((toks, usage));
                        break;
                    }
                    Event::Error(e) => panic!("generation failed: {e}"),
                }
            }
        }
        (out, handle.stats().await.unwrap())
    })
}

/// `--enable-special-token-ckpt`: a sampled anchor token leaves a reuse point right after it, so
/// a follow-up that diverges after the anchor (a re-rendered tool call) reuses the turn up to it.
#[test]
fn qwen35_anchor_token_leaves_a_reuse_point() {
    anchor_token_leaves_a_reuse_point("qwen3_5_moe");
}

#[test]
fn qwen38_anchor_token_leaves_a_reuse_point() {
    anchor_token_leaves_a_reuse_point("qwen4_exp");
}

fn anchor_token_leaves_a_reuse_point(arch: &str) {
    let (_d, model) = tiny(arch);
    let ec = config(&model, &["cpu"], 1);
    let prompt: Vec<u32> = (0..40).map(|i| (i * 7 + 3) % 250).collect();
    let out = run_sequential(&ec, std::slice::from_ref(&prompt), 8, 4096)[0].0.clone();
    // pretend the 3rd sampled token is the tool-call opener
    let anchor = out[2];
    assert!(!out[..2].contains(&anchor), "pick an anchor that occurs once");
    let mut follow = prompt.clone();
    follow.extend(&out[..3]);
    follow.extend([201, 202, 203]);
    let reqs = vec![prompt.clone(), follow.clone()];
    let plain = run_sequential(&ec, &reqs, 8, 4096);
    let anchored = run_sequential_anchored(&ec, &reqs, 8, 4096, Some(anchor));
    // same outputs either way: the snapshot only changes what is reused
    assert_eq!(plain[1].0, anchored[1].0);
    assert_eq!(anchored[1].1.cached_tokens, prompt.len() + 3);
    assert!(plain[1].1.cached_tokens < prompt.len() + 3);
}

/// The CUDA QSA prefill selection (`qsa_topk_bits` + `qsa_bits_mask`) picks what the host path
/// picks: per query the best `topk` of its visible blocks, ties to the lower index (many exact
/// ties: relu zeros and repeated values).
#[cfg(feature = "cuda")]
#[test]
fn qsa_prefill_selection_matches_host() {
    use candle_core::{Device, Tensor};
    use carbon_engine::kernels::cuda::{qsa_bits_mask, qsa_topk_bits};
    if std::env::var("CARBON_TEST_DEVICE").as_deref() != Ok("cuda") {
        return;
    }
    let dev = Device::new_cuda(0).unwrap();
    let (n, nb, topk, r, first) = (300usize, 700usize, 37usize, 4usize, 2500usize);
    let kv_len = first + n;
    let mut seed = 12345u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    let scores: Vec<f32> = (0..n * nb)
        .map(|_| match rnd() % 4 {
            0 => 0.0,
            1 => (rnd() % 8) as f32 * 0.25,
            _ => (rnd() % 100000) as f32 / 977.0,
        })
        .collect();
    let st = Tensor::from_vec(scores.clone(), (n, nb), &dev).unwrap();
    let bits = qsa_topk_bits(&st, topk, r, first).unwrap();
    let mask: Vec<f32> = qsa_bits_mask(&bits, kv_len, r, first, -1e30).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    for i in 0..n {
        let pos = first + i;
        let visible = ((pos + 1) / r).min(nb);
        let row = &scores[i * nb..(i + 1) * nb];
        let mut order: Vec<usize> = (0..visible).collect();
        order.sort_by(|&a, &c| row[c].total_cmp(&row[a]).then(a.cmp(&c)));
        order.truncate(topk);
        let mut want = vec![-1e30f32; kv_len];
        if visible <= topk {
            want[..=pos].fill(0.0);
        } else {
            for &b in &order {
                want[b * r..(b + 1) * r].fill(0.0);
            }
            want[(pos + 1) / r * r..=pos].fill(0.0);
        }
        assert_eq!(&mask[i * kv_len..(i + 1) * kv_len], &want[..], "query {i}");
    }
}

/// The grouped tensor-core prefill MoE (`moe_grouped`) equals dequantize + f32 matmul per
/// expert: runs longer than a tile, a partial tile and a single pair.
#[cfg(feature = "cuda")]
#[test]
fn grouped_nvfp4_moe_matches_reference() {
    use candle_core::{DType, Device, Tensor};
    use carbon_engine::kernels::cuda::{moe_grouped, moe_grouped_supported, nvfp4_dequant, ExpertProj, MoeTable};
    if std::env::var("CARBON_TEST_DEVICE").as_deref() != Ok("cuda") {
        return;
    }
    let dev = Device::new_cuda(0).unwrap();
    let (h, inter) = (128usize, 96usize);
    let runs = [(0usize, 70usize), (2, 1), (1, 130), (2, 9)]; // (expert, pairs), sorted by expert below
    let mut seed = 99u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    let proj = |n: usize, k: usize, rnd: &mut dyn FnMut() -> u32| {
        let packed: Vec<u8> = (0..n * k / 2).map(|_| rnd() as u8).collect();
        // e4m3 scales around 1 (exponent 6..8)
        let scale: Vec<u8> = (0..n * k / 16).map(|_| (((6 + rnd() % 3) << 3) | (rnd() % 8)) as u8).collect();
        ExpertProj::Nvfp4 {
            packed: Tensor::from_vec(packed, (n, k / 2), &dev).unwrap(),
            scale: Tensor::from_vec(scale, (n, k / 16), &dev).unwrap(),
            global: 0.05,
        }
    };
    let experts: Vec<Option<[ExpertProj; 3]>> = (0..3)
        .map(|_| Some([proj(inter, h, &mut rnd), proj(inter, h, &mut rnd), proj(h, inter, &mut rnd)]))
        .collect();
    let table = MoeTable::build(&experts, &dev).unwrap();
    assert!(moe_grouped_supported(&table, DType::BF16, &dev));
    let mut sorted = runs.to_vec();
    sorted.sort_by_key(|r| r.0);
    let pairs: usize = sorted.iter().map(|r| r.1).sum();
    let xs: Vec<f32> = (0..pairs * h).map(|_| (rnd() % 2001) as f32 / 1000.0 - 1.0).collect();
    let xs = Tensor::from_vec(xs, (pairs, h), &dev).unwrap().to_dtype(DType::BF16).unwrap();
    let wts: Vec<f32> = (0..pairs).map(|_| (rnd() % 1000) as f32 / 1000.0).collect();
    let wts_t = Tensor::from_vec(wts.clone(), pairs, &dev).unwrap();
    let (mut tiles, mut a, mut owner) = (vec![], 0usize, vec![]);
    for &(e, n) in &sorted {
        for r0 in (a..a + n).step_by(64) {
            tiles.extend([e as u32, r0 as u32, (64.min(a + n - r0)) as u32]);
        }
        owner.extend(std::iter::repeat_n(e, n));
        a += n;
    }
    let nt = tiles.len() / 3;
    let tiles = Tensor::from_vec(tiles, 3 * nt, &dev).unwrap();
    let act = Tensor::zeros((pairs, inter), DType::BF16, &dev).unwrap();
    let y = Tensor::zeros((pairs, h), DType::BF16, &dev).unwrap();
    moe_grouped(&table, &xs, &wts_t, &tiles, nt, &act, &y).unwrap();
    let got: Vec<Vec<f32>> = y.to_dtype(DType::F32).unwrap().to_vec2().unwrap();
    // reference: weights dequantized to bf16 (as the kernel stages them), f32 math
    let deq = |p: &ExpertProj| match p {
        ExpertProj::Nvfp4 { packed, scale, global } => {
            nvfp4_dequant(packed, scale, *global, DType::BF16).unwrap().to_dtype(DType::F32).unwrap()
        }
        _ => unreachable!(),
    };
    let xf = xs.to_dtype(DType::F32).unwrap();
    let mut worst = 0f32;
    for p in 0..pairs {
        let ex = experts[owner[p]].as_ref().unwrap();
        let x = xf.narrow(0, p, 1).unwrap();
        let g = x.matmul(&deq(&ex[0]).t().unwrap()).unwrap();
        let u = x.matmul(&deq(&ex[1]).t().unwrap()).unwrap();
        let act = (candle_nn::ops::silu(&g).unwrap() * u).unwrap().to_dtype(DType::BF16).unwrap().to_dtype(DType::F32).unwrap();
        let want: Vec<f32> = (act.matmul(&deq(&ex[2]).t().unwrap()).unwrap() * wts[p] as f64)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
        for (a, b) in got[p].iter().zip(&want) {
            worst = worst.max((a - b).abs() / scale);
        }
    }
    assert!(worst < 2e-2, "grouped moe differs from the reference by {worst}");
}

/// `cargo test --release --features cuda --test parallel grouped_moe_bench -- --ignored --nocapture`
/// with CARBON_TEST_DEVICE=cuda: one Qwen3.8 layer's prefill chunk (4096 tokens, top-10 of 512
/// experts, ~300 routed) through the grouped kernels.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn grouped_moe_bench() {
    use candle_core::{DType, Device, Tensor};
    use carbon_engine::kernels::cuda::{moe_grouped, ExpertProj, MoeTable};
    if std::env::var("CARBON_TEST_DEVICE").as_deref() != Ok("cuda") {
        return;
    }
    let dev = Device::new_cuda(0).unwrap();
    let (h, inter, ne) = (2560usize, 640usize, 300usize);
    // random codes and scales (lookups diverge across a warp, as with real weights)
    let proj = |n: usize, k: usize| ExpertProj::Nvfp4 {
        packed: Tensor::rand(0f32, 255f32, (n, k / 2), &dev).unwrap().to_dtype(DType::U8).unwrap(),
        scale: Tensor::rand(48f32, 72f32, (n, k / 16), &dev).unwrap().to_dtype(DType::U8).unwrap(),
        global: 0.05,
    };
    let experts: Vec<Option<[ExpertProj; 3]>> = (0..ne).map(|_| Some([proj(inter, h), proj(inter, h), proj(h, inter)])).collect();
    let table = MoeTable::build(&experts, &dev).unwrap();
    let pairs = 4096 * 10;
    let per = pairs / ne;
    let mut tiles = vec![];
    for e in 0..ne {
        let (a, b) = (e * per, if e + 1 == ne { pairs } else { (e + 1) * per });
        for r0 in (a..b).step_by(64) {
            tiles.extend([e as u32, r0 as u32, (64.min(b - r0)) as u32]);
        }
    }
    let nt = tiles.len() / 3;
    let tiles = Tensor::from_vec(tiles, 3 * nt, &dev).unwrap();
    let xs = Tensor::ones((pairs, h), DType::BF16, &dev).unwrap();
    let wts = Tensor::ones(pairs, DType::F32, &dev).unwrap();
    let act = Tensor::zeros((pairs, inter), DType::BF16, &dev).unwrap();
    let y = Tensor::zeros((pairs, h), DType::BF16, &dev).unwrap();
    for _ in 0..3 {
        moe_grouped(&table, &xs, &wts, &tiles, nt, &act, &y).unwrap();
    }
    dev.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    let n = 20;
    for _ in 0..n {
        moe_grouped(&table, &xs, &wts, &tiles, nt, &act, &y).unwrap();
    }
    dev.synchronize().unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
    let flop = 3.0 * 2.0 * (pairs * h * inter) as f64;
    println!("grouped moe layer: {ms:.3} ms ({:.1} TFLOPS, {nt} tiles)", flop / ms / 1e9);
    // the same tiles in 9 launches (a prefill layer's copy batches)
    let t0 = std::time::Instant::now();
    let per_launch = nt.div_ceil(9);
    for _ in 0..n {
        let mut a = 0;
        while a < nt {
            let m = per_launch.min(nt - a);
            moe_grouped(&table, &xs, &wts, &tiles.narrow(0, 3 * a, 3 * m).unwrap(), m, &act, &y).unwrap();
            a += m;
        }
    }
    dev.synchronize().unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
    println!("grouped moe layer in 9 launches: {ms:.3} ms");
}
