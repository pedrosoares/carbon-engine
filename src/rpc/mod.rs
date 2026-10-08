// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Multi-machine inference: remote pipeline stages over TCP.

pub mod client;
pub mod protocol;
pub mod server;

pub use client::RemoteStage;
pub use server::{serve, WorkerConfig};
