// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! carbon-engine: a Rust re-implementation of the FreeToken serving engine with
//! multi-GPU (pipeline + tensor/expert parallel) and multi-machine (RPC) inference.

pub mod batch;
pub mod benchbw;
pub mod checkpoint;
pub mod config;
pub mod daemon;
pub mod device;
pub mod dummy;
pub mod engine;
pub mod hub;
pub mod images;
pub mod kernels;
pub mod kvcache;
pub mod launch;
pub mod model;
pub mod parallel;
pub mod rpc;
pub mod sampling;
pub mod scheduler;
pub mod server;
pub mod shell;
pub mod testing;
pub mod tokenizer;
pub mod weights;
