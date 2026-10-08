// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Device-independent description of one forward pass, shared by every pipeline
//! stage (local, tensor-parallel or remote).

use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeqMeta {
    /// Unique, never reused request id; remote workers key their slot tables on it.
    pub seq_id: u64,
    /// New tokens of this sequence in the batch.
    pub q_len: usize,
    /// Total tokens with KV after this forward (cached + new).
    pub kv_len: usize,
    /// Recurrent-state slot (hybrid linear-attention models; 0 otherwise).
    #[serde(default)]
    pub state_slot: u32,
    /// KV slot of every token `0..kv_len`.
    #[serde(skip)]
    pub slots: Arc<Vec<u32>>,
    /// The tokens right before this sequence's new ones (at most [`PREV_TOKENS`]), for models
    /// whose layers hash recent token ids (Qwen3.8 PLE).
    #[serde(default)]
    pub prev_tokens: Vec<u32>,
    /// Multi-token steps: the recurrent-state slot each new token's state is written to
    /// (token j continues from token j - 1's). Empty = every token in `state_slot`. MTP verify
    /// steps put a draft's state in a spare slot, so a rejected draft leaves `state_slot` intact.
    #[serde(default)]
    pub token_state_slots: Vec<u32>,
}

/// How many preceding tokens [`SeqMeta::prev_tokens`] carries.
pub const PREV_TOKENS: usize = 8;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BatchMeta {
    pub positions: Vec<u32>,
    /// Slot each new token's KV is written to.
    pub out_loc: Vec<u32>,
    pub seqs: Vec<SeqMeta>,
    pub is_decode: bool,
    /// Token id of every new position (filled for models that need ids inside the layers).
    #[serde(default)]
    pub tokens: Vec<u32>,
    /// mRoPE (t, h, w) rotary position of every new token, when a sequence in the batch has
    /// images (empty: `positions` on all three axes).
    #[serde(default)]
    pub mrope: Vec<[u32; 3]>,
    /// Qwen3.8 batches with images: each sequence's image layout (QSA rotates a pooled block key
    /// at the mRoPE position of the block's first token). Empty otherwise.
    #[serde(default)]
    pub mrope_layout: Vec<crate::images::MropePositions>,
}

impl BatchMeta {
    pub fn num_tokens(&self) -> usize {
        self.positions.len()
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        let total: usize = self.seqs.iter().map(|s| s.q_len).sum();
        anyhow::ensure!(
            total == self.positions.len(),
            "seq q_len sum {total} != tokens {}",
            self.positions.len()
        );
        anyhow::ensure!(
            self.out_loc.len() == self.positions.len(),
            "out_loc length mismatch"
        );
        for s in &self.seqs {
            anyhow::ensure!(
                s.slots.len() == s.kv_len && s.q_len <= s.kv_len && s.q_len > 0,
                "bad seq {}",
                s.seq_id
            );
        }
        Ok(())
    }
}
