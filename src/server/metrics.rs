// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Serving metrics behind `/v1/stats` and `/v1/requests`: sliding-window throughput,
//! lifetime token totals, in-flight request tracking and a ring of recent requests.

use parking_lot::Mutex;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{HashSet, VecDeque};
use std::time::{Duration, Instant};

const RING_CAPACITY: usize = 512;

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct RequestRecord {
    pub ts: String,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub model: Option<String>,
    pub duration_ms: u64,
    pub ttft_ms: Option<u64>,
    pub prompt_tokens: Option<usize>,
    pub completion_tokens: Option<usize>,
    pub cached_tokens: Option<usize>,
    pub stream: Option<bool>,
    pub error: Option<String>,
}

#[derive(Default)]
struct Inner {
    inflight: HashSet<u64>,
    completed: u64,
    prompt_tokens_total: u64,
    completion_tokens_total: u64,
    ring: VecDeque<(u64, RequestRecord)>,
    next_cursor: u64,
}

#[derive(Default)]
pub struct Metrics {
    inner: Mutex<Inner>,
}

impl Metrics {
    pub fn start(&self, id: u64) {
        self.inner.lock().inflight.insert(id);
    }

    /// The request left the engine with `prompt` / `completion` tokens processed;
    /// `completed` is false for aborted / failed ones.
    pub fn finish(&self, id: u64, completed: bool, prompt: usize, completion: usize) {
        let mut g = self.inner.lock();
        if g.inflight.remove(&id) {
            g.prompt_tokens_total += prompt as u64;
            g.completion_tokens_total += completion as u64;
            if completed {
                g.completed += 1;
            }
        }
    }

    pub fn active(&self) -> usize {
        self.inner.lock().inflight.len()
    }

    pub fn inflight(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self.inner.lock().inflight.iter().copied().collect();
        v.sort();
        v
    }

    pub fn totals(&self) -> (u64, u64) {
        let g = self.inner.lock();
        (g.prompt_tokens_total, g.completion_tokens_total)
    }

    pub fn record(&self, rec: RequestRecord) {
        let mut g = self.inner.lock();
        let c = g.next_cursor;
        g.ring.push_back((c, rec));
        g.next_cursor += 1;
        if g.ring.len() > RING_CAPACITY {
            g.ring.pop_front();
        }
    }

    /// Records with cursor >= `since` (at most `limit`) and the cursor to poll with next.
    pub fn requests_since(&self, since: u64, limit: usize) -> (Vec<RequestRecord>, u64) {
        let g = self.inner.lock();
        let matched: Vec<&(u64, RequestRecord)> =
            g.ring.iter().filter(|(c, _)| *c >= since).collect();
        let out: Vec<RequestRecord> = matched.iter().take(limit).map(|x| x.1.clone()).collect();
        let next = if out.len() < matched.len() {
            matched[out.len().saturating_sub(1)].0 + 1
        } else {
            g.next_cursor
        };
        (out, next)
    }

    /// The `requests` section of `/v1/stats`.
    pub fn snapshot(&self) -> Value {
        let g = self.inner.lock();
        let mut durs: Vec<u64> = g.ring.iter().map(|r| r.1.duration_ms).collect();
        durs.sort();
        let p95 = if durs.is_empty() {
            0
        } else {
            durs[((0.95 * durs.len() as f64).ceil() as usize).max(1) - 1]
        };
        let ttfts: Vec<u64> = g.ring.iter().filter_map(|r| r.1.ttft_ms).collect();
        let ttft_mean = if ttfts.is_empty() {
            0
        } else {
            (ttfts.iter().sum::<u64>() as f64 / ttfts.len() as f64).round() as u64
        };
        json!({
            "active": g.inflight.len(),
            "completed": g.completed,
            "p95_ms": p95,
            "ttft_mean_ms": ttft_mean,
            "prompt_tokens_total": g.prompt_tokens_total,
            "completion_tokens_total": g.completion_tokens_total,
            "total": g.next_cursor,
        })
    }
}

/// `YYYY-MM-DDTHH:MM:SSZ` for the current time.
pub fn utc_timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// GPUs as `[{index, name, uuid, total_bytes}]` and this process's VRAM use, from
/// `nvidia-smi` (empty / 0 without NVIDIA GPUs). Cached for a few seconds.
pub fn gpu_info() -> (Vec<Value>, u64) {
    static CACHE: Mutex<Option<(Instant, Vec<Value>, u64)>> = Mutex::new(None);
    let mut c = CACHE.lock();
    if let Some((t, g, v)) = c.as_ref() {
        if t.elapsed() < Duration::from_secs(3) {
            return (g.clone(), *v);
        }
    }
    let run = |args: &[&str]| -> Option<String> {
        let out = std::process::Command::new("nvidia-smi")
            .args(args)
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).to_string())
    };
    let gpus: Vec<Value> = run(&[
        "--query-gpu=index,name,uuid,memory.total",
        "--format=csv,noheader,nounits",
    ])
    .map(|text| {
        text.lines()
            .filter_map(|l| {
                let f: Vec<&str> = l.split(',').map(|x| x.trim()).collect();
                (f.len() == 4).then(|| {
                    json!({
                        "index": f[0].parse::<u64>().unwrap_or(0),
                        "name": f[1],
                        "uuid": f[2],
                        "total_bytes": f[3].parse::<u64>().unwrap_or(0) << 20,
                    })
                })
            })
            .collect()
    })
    .unwrap_or_default();
    let pid = std::process::id().to_string();
    let vram = run(&[
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ])
    .map(|text| {
        text.lines()
            .filter_map(|l| {
                let (p, m) = l.split_once(',')?;
                (p.trim() == pid).then(|| m.trim().parse::<u64>().unwrap_or(0) << 20)
            })
            .sum()
    })
    .unwrap_or(0);
    *c = Some((Instant::now(), gpus.clone(), vram));
    (gpus, vram)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(d: u64, ttft: Option<u64>) -> RequestRecord {
        RequestRecord {
            ts: utc_timestamp(),
            method: "POST".into(),
            path: "/v1/chat/completions".into(),
            status: 200,
            model: None,
            duration_ms: d,
            ttft_ms: ttft,
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
            stream: Some(false),
            error: None,
        }
    }

    #[test]
    fn ring_cursor_and_percentiles() {
        let m = Metrics::default();
        for d in 1..=20 {
            m.record(rec(d * 10, if d % 2 == 0 { Some(d) } else { None }));
        }
        let (all, next) = m.requests_since(0, 100);
        assert_eq!((all.len(), next), (20, 20));
        let (page, next) = m.requests_since(5, 3);
        assert_eq!(page.len(), 3);
        assert_eq!(page[0].duration_ms, 60);
        assert_eq!(next, 8);
        let req = m.snapshot();
        assert_eq!(req["p95_ms"], 190);
        assert_eq!(req["ttft_mean_ms"], 11); // mean of 2,4,..,20
        for _ in 0..RING_CAPACITY {
            m.record(rec(1, None));
        }
        let (all, next) = m.requests_since(0, 10_000);
        assert_eq!(all.len(), RING_CAPACITY);
        assert_eq!(next, 20 + RING_CAPACITY as u64);
    }

    #[test]
    fn inflight_and_totals() {
        let m = Metrics::default();
        m.start(1);
        m.start(2);
        assert_eq!(m.active(), 2);
        assert_eq!(m.inflight(), vec![1, 2]);
        m.finish(1, true, 100, 10);
        m.finish(2, false, 5, 1);
        m.finish(2, true, 5, 1); // already gone: not counted twice
        let req = m.snapshot();
        assert_eq!(req["active"], 0);
        assert_eq!(req["completed"], 1);
        assert_eq!(m.totals(), (105, 11));
        assert!(utc_timestamp().ends_with('Z'));
    }
}
