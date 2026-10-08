// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Token sampling: greedy, temperature, top-k, top-p, min-p and penalties.

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    /// <= 0 disables top-k.
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub max_tokens: usize,
    pub ignore_eos: bool,
    pub stop_token_ids: Vec<u32>,
    pub seed: Option<u64>,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            min_p: 0.0,
            max_tokens: 1024,
            ignore_eos: false,
            stop_token_ids: vec![],
            seed: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplingParams {
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0 || self.top_k == 1
    }

    pub fn has_penalties(&self) -> bool {
        self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || self.repetition_penalty != 1.0
    }

    pub fn rng(&self) -> StdRng {
        match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        }
    }
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

pub fn apply_penalties(logits: &mut [f32], p: &SamplingParams, prompt: &[u32], generated: &[u32]) {
    if p.repetition_penalty != 1.0 {
        let mut seen: Vec<u32> = prompt.iter().chain(generated).copied().collect();
        seen.sort_unstable();
        seen.dedup();
        for t in seen {
            if let Some(l) = logits.get_mut(t as usize) {
                *l = if *l > 0.0 {
                    *l / p.repetition_penalty
                } else {
                    *l * p.repetition_penalty
                };
            }
        }
    }
    if p.presence_penalty != 0.0 || p.frequency_penalty != 0.0 {
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for &t in generated {
            *counts.entry(t).or_default() += 1;
        }
        for (t, c) in counts {
            if let Some(l) = logits.get_mut(t as usize) {
                *l -= p.presence_penalty + p.frequency_penalty * c as f32;
            }
        }
    }
}

/// Sample one token from `logits` (modified in place).
pub fn sample(logits: &mut [f32], p: &SamplingParams, rng: &mut StdRng) -> u32 {
    if p.is_greedy() {
        return argmax(logits);
    }
    let inv_t = 1.0 / p.temperature.max(1e-5);
    let mut cand: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l * inv_t))
        .collect();
    if p.top_k > 0 && (p.top_k as usize) < cand.len() {
        let k = p.top_k as usize;
        cand.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        cand.truncate(k);
    }
    cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let max = cand[0].1;
    let mut probs: Vec<f32> = cand.iter().map(|(_, l)| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|x| *x /= sum);
    let mut keep = probs.len();
    if p.top_p < 1.0 {
        let mut acc = 0.0;
        for (i, &q) in probs.iter().enumerate() {
            acc += q;
            if acc >= p.top_p {
                keep = i + 1;
                break;
            }
        }
    }
    if p.min_p > 0.0 {
        let thresh = probs[0] * p.min_p;
        keep = keep.min(probs.iter().take_while(|&&q| q >= thresh).count().max(1));
    }
    let total: f32 = probs[..keep].iter().sum();
    let mut r = rng.gen::<f32>() * total;
    for i in 0..keep {
        r -= probs[i];
        if r <= 0.0 {
            return cand[i].0;
        }
    }
    cand[keep - 1].0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_topk() {
        let mut rng = StdRng::seed_from_u64(0);
        let p = SamplingParams::default();
        assert_eq!(sample(&mut [0.1, 3.0, 2.0], &p, &mut rng), 1);
        let p = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            ..Default::default()
        };
        assert_eq!(sample(&mut [0.1, 3.0, 2.0], &p, &mut rng), 1);
        let p = SamplingParams {
            temperature: 1.0,
            top_k: 2,
            ..Default::default()
        };
        for _ in 0..200 {
            assert_ne!(sample(&mut [0.1, 3.0, 2.9, -5.0], &p, &mut rng), 0);
        }
    }

    #[test]
    fn top_p_and_min_p_restrict() {
        let mut rng = StdRng::seed_from_u64(1);
        let p = SamplingParams {
            temperature: 1.0,
            top_p: 0.5,
            ..Default::default()
        };
        for _ in 0..200 {
            assert_eq!(sample(&mut [10.0, 0.0, 0.0, 0.0], &p, &mut rng), 0);
        }
        let p = SamplingParams {
            temperature: 1.0,
            min_p: 0.5,
            ..Default::default()
        };
        for _ in 0..200 {
            assert_eq!(sample(&mut [5.0, 1.0, 1.0], &p, &mut rng), 0);
        }
    }

    #[test]
    fn seeded_is_deterministic() {
        let p = SamplingParams {
            temperature: 1.0,
            seed: Some(7),
            ..Default::default()
        };
        let logits = vec![1.0, 1.1, 0.9, 1.05, 0.95];
        let a: Vec<u32> = {
            let mut r = p.rng();
            (0..20)
                .map(|_| sample(&mut logits.clone(), &p, &mut r))
                .collect()
        };
        let b: Vec<u32> = {
            let mut r = p.rng();
            (0..20)
                .map(|_| sample(&mut logits.clone(), &p, &mut r))
                .collect()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn penalties() {
        let mut l = vec![1.0, 2.0, -1.0];
        let p = SamplingParams {
            repetition_penalty: 2.0,
            frequency_penalty: 0.5,
            ..Default::default()
        };
        apply_penalties(&mut l, &p, &[0], &[1, 1, 2]);
        assert_eq!(l, vec![0.5, 1.0 - 1.0, -2.0 - 0.5]);
    }
}
