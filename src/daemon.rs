// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! `carbon daemon`: a small supervisor that owns one `carbon serve` child.
//!
//! It starts, stops and switches the serve over a loopback HTTP control plane (optionally gated
//! by a shared `X-Carbon-Token`), keeps its output in a log ring that clients tail over SSE,
//! re-adopts a serve that outlived a daemon restart, can restart a crashed serve, and keeps the
//! serve (not itself) the kernel's preferred OOM victim.
//!
//! Token accounting is fail-closed: before a serve is signalled the daemon asks it to drain and
//! seal its totals (`/v1/admin/prepare-stop`) and writes a durable receipt; a crash leaves a
//! degraded receipt from the last observed stats. Clients collect receipts from
//! `/accounting/pending` and acknowledge them with `/accounting/ack`.

use anyhow::{bail, Context, Result};
use axum::extract::{Query, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use parking_lot::{Condvar, Mutex};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const DEFAULT_PORT: u16 = 1900;
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Debug)]
pub struct DaemonConfig {
    pub host: String,
    pub port: u16,
    pub state_dir: PathBuf,
    pub token: Option<String>,
    pub default_serve_port: u16,
    /// SIGTERM -> SIGKILL grace when stopping a serve.
    pub grace: Duration,
    pub poll_interval: Duration,
    /// `oom_score_adj` written to the serve's process tree (None = leave it).
    pub oom_child_score: Option<i32>,
    pub auto_restart: bool,
    /// Stop the serve when the daemon exits (default: the serve outlives the daemon).
    pub stop_serve_on_exit: bool,
    pub log_capacity: usize,
    /// Start the serve in its own process group (signals reach its whole tree).
    pub setsid: bool,
    /// The `carbon` binary the serve runs.
    pub serve_exe: PathBuf,
}

impl DaemonConfig {
    pub fn default_state_dir() -> PathBuf {
        std::env::var_os("CARBON_DAEMON_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".carbon/daemon")
            })
    }
}

fn wall_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ------------------------------------------------------------------ log ring

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct LogRec {
    pub seq: u64,
    pub ts: f64,
    /// "stdout" (serve output) or "event" (daemon lifecycle line).
    pub kind: String,
    pub text: String,
}

pub struct LogRing {
    inner: Mutex<(VecDeque<LogRec>, u64)>,
    cap: usize,
    pub notify: tokio::sync::Notify,
    file: Mutex<Option<std::fs::File>>,
}

impl LogRing {
    pub fn new(cap: usize, file: Option<&Path>) -> Self {
        let file = file.and_then(|p| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .ok()
        });
        Self {
            inner: Mutex::new((VecDeque::new(), 0)),
            cap: cap.max(1),
            notify: tokio::sync::Notify::new(),
            file: Mutex::new(file),
        }
    }

    pub fn append(&self, text: &str, kind: &str) {
        let ts = wall_ms() as f64 / 1000.0;
        {
            let mut g = self.inner.lock();
            let seq = g.1;
            g.1 += 1;
            g.0.push_back(LogRec {
                seq,
                ts,
                kind: kind.into(),
                text: text.into(),
            });
            while g.0.len() > self.cap {
                g.0.pop_front();
            }
        }
        if let Some(f) = self.file.lock().as_mut() {
            let _ = writeln!(f, "{text}");
        }
        self.notify.notify_waiters();
    }

    /// Records with seq >= `since`, and the dropped range when `since` fell out of the ring.
    pub fn since(&self, since: u64) -> (Vec<LogRec>, Option<(u64, u64)>) {
        let g = self.inner.lock();
        let oldest = g.0.front().map(|r| r.seq).unwrap_or(g.1);
        let gap = (since < oldest).then_some((since, oldest));
        (g.0.iter().filter(|r| r.seq >= since).cloned().collect(), gap)
    }
}

// ------------------------------------------------------------------ accounting outbox

/// Durable receipts, one JSON file each (atomic write, idempotent by `receiptId`).
pub struct Outbox {
    dir: PathBuf,
    lock: Mutex<()>,
}

pub fn receipt_id(identity: &str) -> String {
    uuid::Uuid::new_v5(
        &uuid::Uuid::NAMESPACE_URL,
        format!("carbon-accounting:{identity}").as_bytes(),
    )
    .to_string()
}

fn valid_receipt_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 200
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

impl Outbox {
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            lock: Mutex::new(()),
        }
    }

    pub fn persist(&self, receipt: Value) -> Result<Value> {
        let id = receipt["receiptId"].as_str().unwrap_or("").to_string();
        if !valid_receipt_id(&id) {
            bail!("invalid receiptId");
        }
        let _g = self.lock.lock();
        let path = self.dir.join(format!("{id}.json"));
        if let Ok(text) = std::fs::read_to_string(&path) {
            // a retried stop reuses the receipt written before the crash
            return Ok(serde_json::from_str(&text)?);
        }
        std::fs::create_dir_all(&self.dir)?;
        let tmp = self.dir.join(format!(".{id}.{}.tmp", uuid::Uuid::new_v4().simple()));
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(serde_json::to_string(&receipt)?.as_bytes())?;
            f.write_all(b"\n")?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;
        if let Ok(d) = std::fs::File::open(&self.dir) {
            let _ = d.sync_all();
        }
        Ok(receipt)
    }

    pub fn pending(&self) -> Result<Vec<Value>> {
        let _g = self.lock.lock();
        let Ok(rd) = std::fs::read_dir(&self.dir) else {
            return Ok(vec![]);
        };
        let mut out = vec![];
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".json") && !name.starts_with('.') {
                let v: Value = serde_json::from_str(&std::fs::read_to_string(e.path())?)
                    .with_context(|| format!("reading receipt {name}"))?;
                out.push(v);
            }
        }
        out.sort_by_key(|v| {
            (
                v["createdAt"].as_u64().unwrap_or(0),
                v["receiptId"].as_str().unwrap_or("").to_string(),
            )
        });
        Ok(out)
    }

    pub fn ack(&self, id: &str) -> Result<Value> {
        if !valid_receipt_id(id) {
            bail!("invalid accounting receiptId");
        }
        let _g = self.lock.lock();
        let already = match std::fs::remove_file(self.dir.join(format!("{id}.json"))) {
            Ok(()) => false,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
            Err(e) => return Err(e.into()),
        };
        Ok(json!({"acked": true, "already": already, "receiptId": id}))
    }
}

/// Token totals out of a prepare-stop or `/v1/stats` document.
fn snapshot_values(doc: &Value, model: Option<&str>) -> Option<(Option<String>, String, u64, u64, u64)> {
    let instance = doc["instance_id"].as_str().map(String::from);
    let model_id = doc["model_id"]
        .as_str()
        .or(doc["model"]["id"].as_str())
        .or(model)?
        .to_string();
    let req = &doc["requests"];
    let p = doc["prompt_tokens_total"]
        .as_u64()
        .or(req["prompt_tokens_total"].as_u64())?;
    let c = doc["completion_tokens_total"]
        .as_u64()
        .or(req["completion_tokens_total"].as_u64())?;
    let up = doc["uptime_s"].as_u64().unwrap_or(0);
    Some((instance, model_id, p, c, up))
}

// ------------------------------------------------------------------ process helpers

/// Field 22 of /proc/<pid>/stat: start time in clock ticks (identifies a pid incarnation).
pub fn proc_starttime(pid: u32) -> Option<u64> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let rest = &s[s.rfind(')')? + 2..];
    rest.split_whitespace().nth(19)?.parse().ok()
}

fn proc_alive(pid: u32, starttime: Option<u64>) -> bool {
    match proc_starttime(pid) {
        Some(st) => starttime.is_none_or(|t| t == st),
        None => false,
    }
}

/// `pid` and all its descendants.
pub fn tree_pids(pid: u32) -> Vec<u32> {
    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
    if let Ok(rd) = std::fs::read_dir("/proc") {
        for e in rd.flatten() {
            let Ok(p) = e.file_name().to_string_lossy().parse::<u32>() else {
                continue;
            };
            if let Ok(s) = std::fs::read_to_string(format!("/proc/{p}/stat")) {
                if let Some(ppid) = s
                    .rfind(')')
                    .and_then(|i| s[i + 2..].split_whitespace().nth(1))
                    .and_then(|x| x.parse::<u32>().ok())
                {
                    children.entry(ppid).or_default().push(p);
                }
            }
        }
    }
    let mut out = vec![pid];
    let mut i = 0;
    while i < out.len() {
        if let Some(c) = children.get(&out[i]) {
            out.extend(c);
        }
        i += 1;
    }
    out
}

fn signal(pid: u32, group: bool, sig: i32) {
    let target = if group { -(pid as i32) } else { pid as i32 };
    // SAFETY: kill(2) with a pid we started (or adopted after checking its start time)
    unsafe {
        libc::kill(target, sig);
    }
}

/// RSS / PSS of a process tree (bytes), from smaps_rollup.
fn tree_memory(pid: u32) -> (u64, u64) {
    let (mut rss, mut pss) = (0, 0);
    for p in tree_pids(pid) {
        if let Ok(t) = std::fs::read_to_string(format!("/proc/{p}/smaps_rollup")) {
            for l in t.lines() {
                let kb = |l: &str| {
                    l.split_whitespace()
                        .nth(1)
                        .and_then(|x| x.parse::<u64>().ok())
                        .unwrap_or(0)
                        * 1024
                };
                if l.starts_with("Rss:") {
                    rss += kb(l);
                } else if l.starts_with("Pss:") {
                    pss += kb(l);
                }
            }
        }
    }
    (rss, pss)
}

fn tree_vram(pid: u32) -> u64 {
    let pids: HashSet<String> = tree_pids(pid).iter().map(|p| p.to_string()).collect();
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter_map(|l| {
                    let (p, m) = l.split_once(',')?;
                    pids.contains(p.trim())
                        .then(|| m.trim().parse::<u64>().unwrap_or(0) << 20)
                })
                .sum()
        })
        .unwrap_or(0)
}

// ------------------------------------------------------------------ the serve manager

struct Reaped {
    done: std::sync::Mutex<bool>,
    cv: std::sync::Condvar,
}

impl Reaped {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            done: std::sync::Mutex::new(false),
            cv: std::sync::Condvar::new(),
        })
    }
    fn set(&self) {
        *self.done.lock().unwrap() = true;
        self.cv.notify_all();
    }
    fn wait(&self, t: Duration) -> bool {
        let g = self.done.lock().unwrap();
        *self.cv.wait_timeout_while(g, t, |d| !*d).unwrap().0
    }
}

#[derive(Clone)]
struct Child {
    pid: u32,
    starttime: Option<u64>,
    adopted: bool,
    reaped: Arc<Reaped>,
}

impl Child {
    fn identity(&self) -> String {
        format!("serve:{}:{}", self.pid, self.starttime.unwrap_or(0))
    }
}

#[derive(Default)]
struct St {
    child: Option<Child>,
    model: Option<String>,
    port: Option<u16>,
    args: Vec<String>,
    started_at: Option<Instant>,
    last_exit: Option<(Option<i32>, String)>,
    starting: bool,
    stopping: bool,
    stop_requested: bool,
    shutdown_requested: bool,
    last_accounting: HashMap<String, Value>,
    durable: HashSet<String>,
}

#[derive(Debug)]
pub enum ManagerError {
    /// Another serve is running (switch replaces it) or the daemon is shutting down.
    Conflict(String),
    /// Accounting could not be sealed; the engine was left running.
    Accounting(String),
    Other(anyhow::Error),
}

impl From<anyhow::Error> for ManagerError {
    fn from(e: anyhow::Error) -> Self {
        Self::Other(e)
    }
}

pub struct Manager {
    cfg: DaemonConfig,
    st: Mutex<St>,
    cv: Condvar,
    lifecycle: Mutex<()>,
    pub ring: Arc<LogRing>,
    pub outbox: Outbox,
}

impl Manager {
    pub fn new(cfg: DaemonConfig) -> Arc<Self> {
        let _ = std::fs::create_dir_all(&cfg.state_dir);
        let ring = Arc::new(LogRing::new(
            cfg.log_capacity,
            Some(&cfg.state_dir.join("serve.log")),
        ));
        Arc::new(Self {
            outbox: Outbox::new(cfg.state_dir.join("accounting")),
            cfg,
            st: Mutex::new(St::default()),
            cv: Condvar::new(),
            lifecycle: Mutex::new(()),
            ring,
        })
    }

    fn emit(&self, text: &str) {
        tracing::info!("{text}");
        self.ring.append(text, "event");
    }

    fn state_file(&self) -> PathBuf {
        self.cfg.state_dir.join("serve.json")
    }

    pub fn status(&self) -> Value {
        let g = self.st.lock();
        let running = g.child.is_some() && !g.stopping;
        json!({
            "running": running,
            "starting": g.starting,
            "stopping": g.stopping,
            "adopted": running && g.child.as_ref().is_some_and(|c| c.adopted),
            "pid": g.child.as_ref().map(|c| c.pid),
            "model": g.model,
            "port": g.port,
            "args": g.args,
            "uptimeS": if running { g.started_at.map(|t| t.elapsed().as_secs()).unwrap_or(0) } else { 0 },
            "lastExitCode": g.last_exit.as_ref().and_then(|e| e.0),
            "lastExitReason": g.last_exit.as_ref().map(|e| e.1.clone()),
        })
    }

    pub fn current(&self) -> Option<(u32, u16)> {
        let g = self.st.lock();
        Some((g.child.as_ref()?.pid, g.port?))
    }

    pub fn start(self: &Arc<Self>, model: &str, port: u16, args: Vec<String>) -> Result<Value, ManagerError> {
        let _l = self.lifecycle.lock();
        self.start_locked(model, port, args, false)
    }

    fn start_locked(self: &Arc<Self>, model: &str, port: u16, args: Vec<String>, auto: bool) -> Result<Value, ManagerError> {
        {
            let mut g = self.st.lock();
            if g.shutdown_requested {
                return Err(ManagerError::Conflict("daemon shutdown is in progress".into()));
            }
            if auto && g.stop_requested {
                return Ok(json!({"pid": null, "aborted": true}));
            }
            g.stop_requested = false;
            if let Some(c) = &g.child {
                if !g.stopping {
                    if g.model.as_deref() == Some(model) && g.port == Some(port) && g.args == args {
                        return Ok(json!({"pid": c.pid, "idempotent": true}));
                    }
                    return Err(ManagerError::Conflict(format!(
                        "serve already running (model={:?} port={:?}); use switch to replace it",
                        g.model, g.port
                    )));
                }
            }
            g.starting = true;
        }
        let spawned = self.spawn(model, port, &args);
        let mut g = self.st.lock();
        g.starting = false;
        self.cv.notify_all();
        let (child, proc_handle) = spawned?;
        g.child = Some(child.clone());
        g.model = Some(model.into());
        g.port = Some(port);
        g.args = args.clone();
        g.started_at = Some(Instant::now());
        g.last_exit = None;
        g.stopping = false;
        drop(g);
        self.persist_state(&child, model, port, &args);
        self.apply_oom();
        self.watch(child.clone(), Some(proc_handle));
        self.emit(&format!("serve started (pid={} model={model} port={port})", child.pid));
        Ok(json!({"pid": child.pid, "idempotent": false}))
    }

    fn spawn(&self, model: &str, port: u16, args: &[String]) -> Result<(Child, std::process::Child)> {
        use std::os::unix::process::CommandExt;
        let mut cmd = std::process::Command::new(&self.cfg.serve_exe);
        cmd.arg("serve")
            .arg("--model")
            .arg(model)
            .arg("--port")
            .arg(port.to_string())
            .args(args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        if self.cfg.setsid {
            cmd.process_group(0);
        }
        let mut proc = cmd
            .spawn()
            .with_context(|| format!("spawning {}", self.cfg.serve_exe.display()))?;
        let pid = proc.id();
        for stream in [
            proc.stdout.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
            proc.stderr.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        ]
        .into_iter()
        .flatten()
        {
            let ring = self.ring.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stream).lines() {
                    match line {
                        Ok(l) => ring.append(&l, "stdout"),
                        Err(_) => break,
                    }
                }
            });
        }
        Ok((
            Child {
                pid,
                starttime: proc_starttime(pid),
                adopted: false,
                reaped: Reaped::new(),
            },
            proc,
        ))
    }

    /// One reaper per child: waits for it to exit, then settles state and accounting.
    fn watch(self: &Arc<Self>, child: Child, proc_handle: Option<std::process::Child>) {
        let me = self.clone();
        let poll = self.cfg.poll_interval;
        std::thread::spawn(move || {
            let (code, source) = match proc_handle {
                Some(mut p) => match p.wait() {
                    Ok(st) => {
                        use std::os::unix::process::ExitStatusExt;
                        match (st.code(), st.signal()) {
                            (Some(c), _) => (Some(c), "exited"),
                            (None, Some(s)) => (Some(-s), "signalled"),
                            _ => (None, "unknown"),
                        }
                    }
                    Err(_) => (None, "unknown"),
                },
                None => {
                    while proc_alive(child.pid, child.starttime) {
                        std::thread::sleep(poll);
                    }
                    (None, "adopted-vanished")
                }
            };
            me.reap(&child, code, source);
        });
    }

    fn reap(self: &Arc<Self>, child: &Child, code: Option<i32>, source: &str) {
        let (was_stopping, is_current, model) = {
            let g = self.st.lock();
            (
                g.stopping || g.stop_requested,
                g.child.as_ref().is_some_and(|c| c.pid == child.pid),
                g.model.clone(),
            )
        };
        if is_current {
            self.crash_receipt(child, model.as_deref(), was_stopping, code, source);
        }
        let mut restart = None;
        {
            let mut g = self.st.lock();
            if g.child.as_ref().is_some_and(|c| c.pid == child.pid) {
                g.child = None;
                g.started_at = None;
                g.stopping = false;
                g.last_exit = Some((code, if was_stopping { "stopped".into() } else { source.into() }));
                if !was_stopping && self.cfg.auto_restart && !g.stop_requested {
                    restart = Some((g.model.clone(), g.port, g.args.clone()));
                }
            }
            self.cv.notify_all();
        }
        if is_current {
            let _ = std::fs::remove_file(self.state_file());
        }
        child.reaped.set();
        if is_current {
            self.emit(&match (was_stopping, code) {
                (true, _) => format!("serve stopped (pid={})", child.pid),
                (_, Some(c)) if c < 0 => format!("serve killed by signal {} (pid={})", -c, child.pid),
                (_, c) if source == "adopted-vanished" => {
                    format!("serve (adopted pid={}) vanished; exit code {c:?}", child.pid)
                }
                (_, c) => format!("serve exited with code {c:?} (pid={})", child.pid),
            });
        }
        if let Some((Some(m), Some(p), a)) = restart {
            let me = self.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(1));
                let _l = me.lifecycle.lock();
                if let Err(e) = me.start_locked(&m, p, a, true) {
                    me.emit(&format!("auto-restart failed: {e:?}"));
                }
            });
        }
    }

    fn crash_receipt(&self, child: &Child, model: Option<&str>, was_stopping: bool, code: Option<i32>, source: &str) {
        let id = child.identity();
        let snap = {
            let mut g = self.st.lock();
            if g.durable.contains(&id) {
                return;
            }
            g.durable.insert(id.clone());
            g.last_accounting.remove(&id)
        };
        let r = json!({
            "receiptId": receipt_id(&format!("crash:{id}")),
            "createdAt": wall_ms(),
            "instanceId": snap.as_ref().and_then(|s| s["instanceId"].as_str()),
            "modelId": snap.as_ref().and_then(|s| s["modelId"].as_str()).or(model),
            "promptTokensTotal": snap.as_ref().map(|s| s["promptTokensTotal"].clone()),
            "completionTokensTotal": snap.as_ref().map(|s| s["completionTokensTotal"].clone()),
            "uptimeS": snap.as_ref().map(|s| s["uptimeS"].clone()),
            "drainComplete": false,
            "degraded": true,
            "bestEffort": true,
            "reason": if was_stopping { "engine-crashed-during-stop" } else { "engine-crashed" },
            "detail": format!("exit source={source} code={code:?}"),
        });
        if let Err(e) = self.outbox.persist(r) {
            self.emit(&format!("failed to persist crash accounting: {e:#}"));
            // keep the gap visible instead of hiding it behind a fresh serve
            self.st.lock().stop_requested = true;
        }
    }

    /// Remember the latest stats totals of the running serve (the crash receipt's source).
    pub fn observe_accounting(&self, stats: &Value) {
        let mut g = self.st.lock();
        let Some(c) = g.child.clone() else {
            return;
        };
        let model = g.model.clone();
        if let Some((inst, m, p, cpl, up)) = snapshot_values(stats, model.as_deref()) {
            g.last_accounting.insert(
                c.identity(),
                json!({"instanceId": inst, "modelId": m, "promptTokensTotal": p,
                       "completionTokensTotal": cpl, "uptimeS": up}),
            );
        }
    }

    /// Sealed totals from the serve, or (when it cannot answer) a best-effort receipt.
    #[allow(clippy::result_large_err)]
    fn prepare_receipt(&self, child: &Child, model: Option<&str>, port: Option<u16>, force: bool) -> Result<Value, ManagerError> {
        let base = port.map(|p| format!("http://127.0.0.1:{p}"));
        let sealed = base.as_ref().map(|b| {
            ureq::post(&format!("{b}/v1/admin/prepare-stop"))
                .timeout(Duration::from_secs(15))
                .send_json(json!({}))
        });
        let fail = |reason: &str, detail: String| -> Result<Value, ManagerError> {
            // best effort: the stats totals, else the last observation
            let snap = base
                .as_ref()
                .and_then(|b| ureq::get(&format!("{b}/v1/stats")).timeout(Duration::from_secs(5)).call().ok())
                .and_then(|r| r.into_json::<Value>().ok())
                .and_then(|v| snapshot_values(&v, model))
                .map(|(i, m, p, c, u)| json!({"instanceId": i, "modelId": m, "promptTokensTotal": p, "completionTokensTotal": c, "uptimeS": u}))
                .or_else(|| self.st.lock().last_accounting.get(&child.identity()).cloned());
            Ok(json!({
                "receiptId": receipt_id(&format!("{reason}:{}", child.identity())),
                "createdAt": wall_ms(),
                "instanceId": snap.as_ref().and_then(|s| s["instanceId"].as_str()),
                "modelId": snap.as_ref().and_then(|s| s["modelId"].as_str()).or(model),
                "promptTokensTotal": snap.as_ref().map(|s| s["promptTokensTotal"].clone()),
                "completionTokensTotal": snap.as_ref().map(|s| s["completionTokensTotal"].clone()),
                "uptimeS": snap.as_ref().map(|s| s["uptimeS"].clone()),
                "drainComplete": false,
                "degraded": true,
                "bestEffort": true,
                "reason": reason,
                "detail": detail,
            }))
        };
        match sealed {
            Some(Ok(r)) => {
                let v: Value = r.into_json().map_err(|e| ManagerError::Accounting(e.to_string()))?;
                match (v["drain_complete"].as_bool(), snapshot_values(&v, model)) {
                    (Some(true), Some((Some(inst), m, p, c, up))) => Ok(json!({
                        "receiptId": receipt_id(&format!("instance:{inst}")),
                        "createdAt": wall_ms(),
                        "instanceId": inst,
                        "modelId": m,
                        "promptTokensTotal": p,
                        "completionTokensTotal": c,
                        "uptimeS": up,
                        "drainComplete": true,
                        "degraded": false,
                        "bestEffort": false,
                        "reason": null,
                    })),
                    _ if force => fail("prepare-stop-failed", format!("unsealed response: {v}")),
                    _ => Err(ManagerError::Accounting(format!("prepare-stop returned an unsealed snapshot: {v}"))),
                }
            }
            // still loading (nothing served yet) or already gone
            Some(Err(ureq::Error::Transport(t))) => fail("engine-not-serving", t.to_string()),
            None => fail("engine-not-serving", "no control port".into()),
            Some(Err(e)) if force => fail("prepare-stop-failed", e.to_string()),
            Some(Err(e)) => Err(ManagerError::Accounting(format!("prepare-stop failed: {e}"))),
        }
    }

    pub fn stop(self: &Arc<Self>, force: bool) -> Result<Value, ManagerError> {
        let _l = self.lifecycle.lock();
        self.stop_locked(force)
    }

    fn stop_locked(self: &Arc<Self>, force: bool) -> Result<Value, ManagerError> {
        let (child, model, port) = {
            let mut g = self.st.lock();
            g.stop_requested = true;
            while g.starting {
                self.cv.wait(&mut g);
            }
            let Some(c) = g.child.clone() else {
                return Ok(json!({"stopped": true, "already": true, "accounting": null}));
            };
            g.stopping = true;
            (c, g.model.clone(), g.port)
        };
        let restore = || {
            let mut g = self.st.lock();
            if g.child.as_ref().is_some_and(|c| c.pid == child.pid) {
                g.stopping = false;
            }
        };
        let receipt = match self
            .prepare_receipt(&child, model.as_deref(), port, force)
            .and_then(|r| self.outbox.persist(r).map_err(|e| ManagerError::Accounting(e.to_string())))
        {
            Ok(r) => {
                self.st.lock().durable.insert(child.identity());
                r
            }
            Err(e) => {
                restore();
                return Err(e);
            }
        };
        signal(child.pid, self.cfg.setsid, libc::SIGTERM);
        if !child.reaped.wait(self.cfg.grace) {
            signal(child.pid, self.cfg.setsid, libc::SIGKILL);
            if !child.reaped.wait(Duration::from_secs(10)) {
                restore();
                return Err(ManagerError::Other(anyhow::anyhow!(
                    "serve pid={} did not exit after SIGKILL",
                    child.pid
                )));
            }
        }
        Ok(json!({"stopped": true, "accounting": receipt}))
    }

    pub fn shutdown(self: &Arc<Self>, force: bool) -> Result<Value, ManagerError> {
        let _l = self.lifecycle.lock();
        self.st.lock().shutdown_requested = true;
        self.stop_locked(force).inspect_err(|_| {
            self.st.lock().shutdown_requested = false;
        })
    }

    pub fn switch(self: &Arc<Self>, model: &str, port: u16, args: Vec<String>, force: bool) -> Result<Value, ManagerError> {
        let _l = self.lifecycle.lock();
        let stopped = self.stop_locked(force)?;
        let mut started = self.start_locked(model, port, args, false)?;
        started["accounting"] = stopped["accounting"].clone();
        Ok(started)
    }

    fn persist_state(&self, c: &Child, model: &str, port: u16, args: &[String]) {
        let v = json!({"pid": c.pid, "starttime": c.starttime, "model": model, "port": port, "args": args});
        let _ = std::fs::write(self.state_file(), v.to_string());
    }

    /// Re-attach a serve recorded by a previous daemon if that exact process is still alive.
    pub fn readopt(self: &Arc<Self>) -> bool {
        let Some(v) = std::fs::read_to_string(self.state_file())
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        else {
            return false;
        };
        let pid = v["pid"].as_u64().unwrap_or(0) as u32;
        let st = v["starttime"].as_u64();
        let cmdline = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
        let is_serve = cmdline.split(|b| *b == 0).any(|a| a == b"serve");
        if pid == 0 || !proc_alive(pid, st) || !is_serve {
            let _ = std::fs::remove_file(self.state_file());
            self.emit(&format!("stale serve state cleared (pid={pid})"));
            return false;
        }
        let child = Child {
            pid,
            starttime: st,
            adopted: true,
            reaped: Reaped::new(),
        };
        {
            let mut g = self.st.lock();
            g.child = Some(child.clone());
            g.model = v["model"].as_str().map(String::from);
            g.port = v["port"].as_u64().map(|p| p as u16);
            g.args = v["args"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default();
            g.started_at = Some(Instant::now());
        }
        self.apply_oom();
        self.watch(child, None);
        self.emit(&format!("re-adopted running serve (pid={pid})"));
        true
    }

    /// Make the serve tree, not the daemon, the kernel's preferred OOM victim.
    pub fn apply_oom(&self) {
        let (Some(score), Some((pid, _))) = (self.cfg.oom_child_score, self.current()) else {
            return;
        };
        for p in tree_pids(pid) {
            let _ = std::fs::write(format!("/proc/{p}/oom_score_adj"), score.to_string());
        }
    }

    pub fn metrics(&self) -> Value {
        match self.current() {
            Some((pid, _)) => {
                let (rss, pss) = tree_memory(pid);
                json!({"pid": pid, "rssBytes": rss, "pssBytes": pss, "vramBytes": tree_vram(pid),
                       "processes": tree_pids(pid).len()})
            }
            None => json!({"pid": null, "rssBytes": 0, "pssBytes": 0, "vramBytes": 0, "processes": 0}),
        }
    }
}

// ------------------------------------------------------------------ checkpoint jobs

/// (job id, live status document, child pid)
type Job = (String, Arc<Mutex<Value>>, Option<u32>);

/// One `carbon checkpoint` conversion at a time, with progress parsed from its output.
#[derive(Default)]
pub struct Checkpoints {
    job: Mutex<Option<Job>>,
}

impl Checkpoints {
    pub fn start(&self, exe: &Path, id: &str, args: Vec<String>) -> Result<Value, ManagerError> {
        let mut g = self.job.lock();
        if let Some((cur, st, _)) = g.as_ref() {
            if st.lock()["state"] == "running" {
                return Err(ManagerError::Conflict(format!("checkpoint '{cur}' is still running")));
            }
        }
        let mut child = std::process::Command::new(exe)
            .arg("checkpoint")
            .args(&args)
            .arg("--progress")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| ManagerError::Other(e.into()))?;
        let status = Arc::new(Mutex::new(
            json!({"id": id, "state": "running", "done": 0, "total": 0, "error": null}),
        ));
        let (out, err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
        let st = status.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                let f: Vec<&str> = line.split_whitespace().collect();
                if f.len() == 3 && f[0] == "CARBONCONVERT" {
                    let mut s = st.lock();
                    s["done"] = json!(f[1].parse::<u64>().unwrap_or(0));
                    s["total"] = json!(f[2].parse::<u64>().unwrap_or(0));
                }
            }
        });
        let tail = Arc::new(Mutex::new(VecDeque::new()));
        let t = tail.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                let mut q = t.lock();
                q.push_back(line);
                if q.len() > 8 {
                    q.pop_front();
                }
            }
        });
        let pid = child.id();
        let st = status.clone();
        std::thread::spawn(move || {
            let ok = child.wait().map(|s| s.success()).unwrap_or(false);
            let mut s = st.lock();
            if s["state"] == "running" {
                s["state"] = json!(if ok { "done" } else { "failed" });
                if !ok {
                    s["error"] = json!(tail.lock().iter().cloned().collect::<Vec<_>>().join("\n"));
                }
            }
        });
        *g = Some((id.to_string(), status.clone(), Some(pid)));
        let v = status.lock().clone();
        Ok(v)
    }

    pub fn cancel(&self, id: &str) -> Value {
        let g = self.job.lock();
        match g.as_ref() {
            Some((cur, st, Some(pid))) if cur == id && st.lock()["state"] == "running" => {
                st.lock()["state"] = json!("cancelled");
                signal(*pid, false, libc::SIGTERM);
                json!({"cancelled": true, "id": id})
            }
            _ => json!({"cancelled": false, "id": id}),
        }
    }

    pub fn status(&self) -> Value {
        self.job
            .lock()
            .as_ref()
            .map(|(_, s, _)| s.lock().clone())
            .unwrap_or(json!({"state": "idle"}))
    }
}

// ------------------------------------------------------------------ HTTP control plane

#[derive(Clone)]
struct App {
    checkpoints: Arc<Checkpoints>,
    m: Arc<Manager>,
    cfg: DaemonConfig,
    started: Instant,
    shutdown: Arc<tokio::sync::Notify>,
}

fn err(status: StatusCode, msg: impl Into<String>, extra: Value) -> Response {
    let mut v = json!({"error": msg.into()});
    if let (Some(o), Some(e)) = (v.as_object_mut(), extra.as_object()) {
        o.extend(e.clone());
    }
    (status, Json(v)).into_response()
}

fn manager_err(e: ManagerError, m: &Manager) -> Response {
    match e {
        ManagerError::Conflict(msg) => {
            let st = m.status();
            err(
                StatusCode::CONFLICT,
                msg,
                json!({"code": "serve_conflict", "currentModel": st["model"], "currentPort": st["port"]}),
            )
        }
        ManagerError::Accounting(msg) => err(
            StatusCode::SERVICE_UNAVAILABLE,
            msg,
            json!({"code": "accounting_prepare_failed", "enginePreserved": true}),
        ),
        ManagerError::Other(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"), json!({})),
    }
}

async fn auth(State(a): State<App>, req: Request, next: Next) -> Response {
    if let Some(t) = &a.cfg.token {
        if req.uri().path() != "/health" {
            let got = req
                .headers()
                .get("x-carbon-token")
                .or_else(|| req.headers().get("x-ft-token"))
                .and_then(|v| v.to_str().ok());
            if got != Some(t.as_str()) {
                return err(StatusCode::UNAUTHORIZED, "invalid or missing X-Carbon-Token", json!({}));
            }
        }
    }
    next.run(req).await
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    tokio::task::spawn_blocking(f).await.expect("blocking task")
}

fn body_args(b: &Value) -> Vec<String> {
    b["args"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default()
}

fn body_port(a: &App, b: &Value) -> u16 {
    b["port"]
        .as_u64()
        .map(|p| p as u16)
        .or(a.m.status()["port"].as_u64().map(|p| p as u16))
        .unwrap_or(a.cfg.default_serve_port)
}

pub fn router(m: Arc<Manager>, cfg: DaemonConfig, shutdown: Arc<tokio::sync::Notify>) -> Router {
    let app = App {
        checkpoints: Arc::new(Checkpoints::default()),
        m,
        cfg,
        started: Instant::now(),
        shutdown,
    };
    Router::new()
        .route(
            "/health",
            get(|State(a): State<App>| async move {
                Json(json!({"status": "ok", "version": VERSION, "uptimeS": a.started.elapsed().as_secs(),
                            "engineRunning": a.m.status()["running"]}))
            }),
        )
        .route(
            "/engine/start",
            post(|State(a): State<App>, Json(b): Json<Value>| async move {
                let Some(model) = b["model"].as_str().map(String::from) else {
                    return err(StatusCode::BAD_REQUEST, "'model' is required", json!({}));
                };
                let port = body_port(&a, &b);
                let m = a.m.clone();
                match blocking(move || m.start(&model, port, body_args(&b))).await {
                    Ok(v) => Json(v).into_response(),
                    Err(e) => manager_err(e, &a.m),
                }
            }),
        )
        .route(
            "/engine/switch",
            post(|State(a): State<App>, Json(b): Json<Value>| async move {
                let Some(model) = b["model"].as_str().map(String::from) else {
                    return err(StatusCode::BAD_REQUEST, "'model' is required", json!({}));
                };
                let port = body_port(&a, &b);
                let force = b["force"].as_bool().unwrap_or(false);
                let m = a.m.clone();
                match blocking(move || m.switch(&model, port, body_args(&b), force)).await {
                    Ok(v) => Json(v).into_response(),
                    Err(e) => manager_err(e, &a.m),
                }
            }),
        )
        .route(
            "/engine/stop",
            post(|State(a): State<App>, body: Option<Json<Value>>| async move {
                let force = body.and_then(|b| b["force"].as_bool()).unwrap_or(false);
                let m = a.m.clone();
                match blocking(move || m.stop(force)).await {
                    Ok(v) => Json(v).into_response(),
                    Err(e) => manager_err(e, &a.m),
                }
            }),
        )
        .route(
            "/shutdown",
            post(|State(a): State<App>, body: Option<Json<Value>>| async move {
                let force = body.and_then(|b| b["force"].as_bool()).unwrap_or(false);
                let m = a.m.clone();
                match blocking(move || m.shutdown(force)).await {
                    Ok(v) => {
                        a.shutdown.notify_one();
                        Json(json!({"stopping": true, "already": v["already"].as_bool().unwrap_or(false),
                                    "accounting": v["accounting"]}))
                        .into_response()
                    }
                    Err(e) => manager_err(e, &a.m),
                }
            }),
        )
        .route("/engine/status", get(|State(a): State<App>| async move { Json(a.m.status()) }))
        .route(
            "/engine/metrics",
            get(|State(a): State<App>| async move {
                let m = a.m.clone();
                Json(blocking(move || m.metrics()).await)
            }),
        )
        .route(
            "/engine/health",
            get(|State(a): State<App>| async move { Json(proxy(&a, "/health").await) }),
        )
        .route(
            "/engine/stats",
            get(|State(a): State<App>| async move {
                let v = proxy(&a, "/v1/stats").await;
                a.m.observe_accounting(&v);
                Json(v)
            }),
        )
        .route("/engine/logs", get(logs))
        .route(
            "/accounting/pending",
            get(|State(a): State<App>| async move {
                match a.m.outbox.pending() {
                    Ok(r) => Json(json!({"receipts": r})).into_response(),
                    Err(e) => err(StatusCode::SERVICE_UNAVAILABLE, format!("{e:#}"), json!({"code": "accounting_outbox_failed"})),
                }
            }),
        )
        .route(
            "/accounting/ack",
            post(|State(a): State<App>, Json(b): Json<Value>| async move {
                match a.m.outbox.ack(b["receiptId"].as_str().unwrap_or("")) {
                    Ok(v) => Json(v).into_response(),
                    Err(e) => err(StatusCode::BAD_REQUEST, format!("{e:#}"), json!({})),
                }
            }),
        )
        .route("/bench/run", post(bench_run))
        .route(
            "/checkpoint/start",
            post(|State(a): State<App>, Json(b): Json<Value>| async move {
                let id = b["id"].as_str().unwrap_or("checkpoint").to_string();
                let m = a.m.clone();
                // a conversion competes with a serve for memory and disk bandwidth
                if let Err(e) = blocking(move || m.stop(false)).await {
                    return manager_err(e, &a.m);
                }
                match a.checkpoints.start(&a.cfg.serve_exe, &id, body_args(&b)) {
                    Ok(v) => Json(v).into_response(),
                    Err(e) => manager_err(e, &a.m),
                }
            }),
        )
        .route(
            "/checkpoint/cancel",
            post(|State(a): State<App>, Json(b): Json<Value>| async move {
                Json(a.checkpoints.cancel(b["id"].as_str().unwrap_or("")))
            }),
        )
        .route(
            "/checkpoint/status",
            get(|State(a): State<App>| async move { Json(a.checkpoints.status()) }),
        )
        .route(
            "/bench/profile",
            get(|| async {
                let latest = std::fs::read_dir(crate::benchbw::profile_dir())
                    .ok()
                    .and_then(|rd| {
                        rd.flatten()
                            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                            .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
                    })
                    .and_then(|e| std::fs::read_to_string(e.path()).ok())
                    .and_then(|t| serde_json::from_str::<Value>(&t).ok());
                Json(latest.unwrap_or(Value::Null))
            }),
        )
        .layer(axum::middleware::from_fn_with_state(app.clone(), auth))
        .with_state(app)
}

/// GET `path` on the running serve (`reachable: false` when it does not answer).
async fn proxy(a: &App, path: &str) -> Value {
    let st = a.m.status();
    if st["running"] != true {
        return json!({"reachable": false, "running": false, "daemon": "up",
                      "model": st["model"], "port": st["port"], "lastExitCode": st["lastExitCode"]});
    }
    let url = format!("http://127.0.0.1:{}{path}", st["port"].as_u64().unwrap_or(0));
    let r = blocking(move || {
        ureq::get(&url)
            .timeout(Duration::from_secs(5))
            .call()
            .ok()
            .and_then(|r| r.into_json::<Value>().ok())
    })
    .await;
    match r {
        Some(mut v) => {
            v["reachable"] = json!(true);
            v["running"] = json!(true);
            v
        }
        None => json!({"reachable": false, "running": true, "daemon": "up", "pid": st["pid"],
                       "port": st["port"], "note": "serve is starting or not answering"}),
    }
}

#[derive(serde::Deserialize)]
struct Since {
    #[serde(default)]
    since: u64,
}

/// SSE: replay from `since` (a `gap` record when it already fell out of the ring), then tail.
async fn logs(State(a): State<App>, Query(q): Query<Since>) -> Response {
    let ring = a.m.ring.clone();
    let stream = async_stream::stream! {
        let mut next = q.since;
        let mut first = true;
        loop {
            let notified = ring.notify.notified();
            let (recs, gap) = ring.since(next);
            if first {
                if let Some((from, to)) = gap {
                    yield Ok::<_, std::convert::Infallible>(Event::default().data(
                        json!({"kind": "gap", "dropped": to - from, "fromSeq": from, "toSeq": to}).to_string()));
                }
                first = false;
            }
            for r in recs {
                next = r.seq + 1;
                yield Ok(Event::default().id(r.seq.to_string()).data(serde_json::to_string(&r).unwrap()));
            }
            notified.await;
        }
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Stop the serve, run `carbon bench bw ARGS`, stream progress and the profile over SSE.
async fn bench_run(State(a): State<App>, body: Option<Json<Value>>) -> Response {
    let args = body.map(|b| body_args(&b)).unwrap_or_default();
    let m = a.m.clone();
    if let Err(e) = blocking(move || m.stop(false)).await {
        return manager_err(e, &a.m);
    }
    let exe = a.cfg.serve_exe.clone();
    let stream = async_stream::stream! {
        let ev = |name: &str, v: Value| Ok::<_, std::convert::Infallible>(Event::default().event(name).data(v.to_string()));
        let child = tokio::process::Command::new(&exe)
            .arg("bench").arg("bw").args(&args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();
        let mut child = match child {
            Ok(c) => c,
            Err(e) => { yield ev("error", json!({"message": format!("failed to spawn bench: {e}")})); return; }
        };
        use tokio::io::AsyncBufReadExt;
        let mut err_lines = tokio::io::BufReader::new(child.stderr.take().unwrap()).lines();
        let mut out_lines = tokio::io::BufReader::new(child.stdout.take().unwrap()).lines();
        let mut profile_path = None;
        let mut tail = VecDeque::new();
        let (mut out_done, mut err_done) = (false, false);
        while !(out_done && err_done) {
            tokio::select! {
                l = err_lines.next_line(), if !err_done => match l {
                    Ok(Some(l)) => { tail.push_back(l.clone()); if tail.len() > 8 { tail.pop_front(); }
                                     yield ev("progress", json!({"label": l})); }
                    _ => err_done = true,
                },
                l = out_lines.next_line(), if !out_done => match l {
                    Ok(Some(l)) => if let Some(p) = l.strip_prefix("profile written to ") { profile_path = Some(p.to_string()); },
                    _ => out_done = true,
                },
            }
        }
        let ok = child.wait().await.map(|s| s.success()).unwrap_or(false);
        if !ok {
            yield ev("error", json!({"message": tail.into_iter().collect::<Vec<_>>().join("\n")}));
            return;
        }
        match profile_path.and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
            Some(p) => yield ev("result", p),
            None => yield ev("error", json!({"message": "bench finished but no profile was written (pass --device cuda:N)"})),
        }
    };
    Sse::new(stream).into_response()
}

/// Run the daemon until /shutdown or a signal.
pub fn run(cfg: DaemonConfig) -> Result<()> {
    std::fs::create_dir_all(&cfg.state_dir)?;
    // one daemon per state dir
    let pidfile = cfg.state_dir.join("daemon.pid");
    if let Some(other) = std::fs::read_to_string(&pidfile)
        .ok()
        .and_then(|t| t.trim().parse::<u32>().ok())
    {
        if other != std::process::id() && proc_alive(other, None) {
            bail!("another carbon daemon (pid {other}) owns {}", cfg.state_dir.display());
        }
    }
    std::fs::write(&pidfile, std::process::id().to_string())?;
    if cfg.oom_child_score.is_some() {
        // the daemon itself should be the last thing the OOM killer picks
        let _ = std::fs::write("/proc/self/oom_score_adj", "-500");
    }
    let m = Manager::new(cfg.clone());
    m.readopt();
    {
        let m = m.clone();
        let every = cfg.poll_interval;
        std::thread::spawn(move || loop {
            std::thread::sleep(every.max(Duration::from_millis(200)));
            m.apply_oom(); // children spawned after start inherit the default score
        });
    }
    let shutdown = Arc::new(tokio::sync::Notify::new());
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let result = rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        tracing::info!("carbon daemon {VERSION} on http://{}", listener.local_addr()?);
        let s = shutdown.clone();
        axum::serve(listener, router(m.clone(), cfg.clone(), shutdown.clone()))
            .with_graceful_shutdown(async move {
                tokio::select! {
                    _ = s.notified() => {}
                    _ = tokio::signal::ctrl_c() => {}
                }
            })
            .await?;
        anyhow::Ok(())
    });
    if cfg.stop_serve_on_exit {
        if let Err(e) = m.stop(true) {
            tracing::warn!("stopping the serve on exit failed: {e:?}");
        }
    }
    let _ = std::fs::remove_file(&pidfile);
    result
}

// ------------------------------------------------------------------ client

/// `carbon daemon <verb>` against a running daemon.
pub fn client(url: &str, token: Option<&str>, verb: &str, body: Option<Value>, since: u64) -> Result<()> {
    let url = url.trim_end_matches('/');
    let (method, path) = match verb {
        "self" => ("GET", "/health"),
        "status" => ("GET", "/engine/status"),
        "health" => ("GET", "/engine/health"),
        "metrics" => ("GET", "/engine/metrics"),
        "stats" => ("GET", "/engine/stats"),
        "start" => ("POST", "/engine/start"),
        "switch" => ("POST", "/engine/switch"),
        "stop" => ("POST", "/engine/stop"),
        "shutdown" => ("POST", "/shutdown"),
        "pending" => ("GET", "/accounting/pending"),
        "logs" => ("GET", "/engine/logs"),
        o => bail!("unknown daemon verb '{o}'"),
    };
    let target = if verb == "logs" {
        format!("{url}{path}?since={since}")
    } else {
        format!("{url}{path}")
    };
    let mut req = ureq::request(method, &target);
    if let Some(t) = token {
        req = req.set("X-Carbon-Token", t);
    }
    let resp = match (method, body) {
        ("POST", Some(b)) => req.send_json(b),
        ("POST", None) => req.send_json(json!({})),
        _ => req.call(),
    };
    let resp = match resp {
        Ok(r) => r,
        Err(ureq::Error::Status(code, r)) => {
            let text = r.into_string().unwrap_or_default();
            bail!("daemon answered {code}: {text}");
        }
        Err(e) => bail!("cannot reach the carbon daemon at {url}: {e}"),
    };
    if verb == "logs" {
        for line in BufReader::new(resp.into_reader()).lines() {
            let line = line?;
            if let Some(d) = line.strip_prefix("data: ") {
                let v: Value = serde_json::from_str(d).unwrap_or(Value::Null);
                match v["kind"].as_str() {
                    Some("gap") => println!("... {} earlier lines dropped ...", v["dropped"]),
                    Some("event") => println!("[daemon] {}", v["text"].as_str().unwrap_or("")),
                    _ => println!("{}", v["text"].as_str().unwrap_or("")),
                }
            }
        }
        return Ok(());
    }
    let v: Value = resp.into_json()?;
    println!("{}", serde_json::to_string_pretty(&v)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_replay_and_gap() {
        let r = LogRing::new(3, None);
        for i in 0..5 {
            r.append(&format!("l{i}"), "stdout");
        }
        let (recs, gap) = r.since(0);
        assert_eq!(recs.iter().map(|x| x.text.as_str()).collect::<Vec<_>>(), ["l2", "l3", "l4"]);
        assert_eq!(gap, Some((0, 2)));
        let (recs, gap) = r.since(4);
        assert_eq!((recs.len(), gap), (1, None));
    }

    #[test]
    fn outbox_is_idempotent() {
        let d = tempfile::tempdir().unwrap();
        let o = Outbox::new(d.path().join("acc"));
        let id = receipt_id("instance:x");
        assert_eq!(id, receipt_id("instance:x"));
        let a = o.persist(json!({"receiptId": id, "createdAt": 2, "n": 1})).unwrap();
        // a retry returns the first document, not the new one
        let b = o.persist(json!({"receiptId": id, "createdAt": 3, "n": 2})).unwrap();
        assert_eq!(a, b);
        o.persist(json!({"receiptId": receipt_id("other"), "createdAt": 1})).unwrap();
        let p = o.pending().unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p[0]["createdAt"], 1);
        assert_eq!(o.ack(&id).unwrap()["already"], false);
        assert_eq!(o.ack(&id).unwrap()["already"], true);
        assert!(o.ack("../etc").is_err());
        assert!(o.persist(json!({"receiptId": "a/b"})).is_err());
    }

    #[test]
    fn proc_helpers() {
        let me = std::process::id();
        let st = proc_starttime(me);
        assert!(st.is_some());
        assert!(proc_alive(me, st));
        assert!(!proc_alive(me, st.map(|s| s + 1)));
        assert!(tree_pids(me).contains(&me));
        let (rss, _) = tree_memory(me);
        assert!(rss > 0);
        let v = json!({"instance_id": "i", "model": {"id": "m"}, "uptime_s": 3,
                       "requests": {"prompt_tokens_total": 5, "completion_tokens_total": 7}});
        assert_eq!(snapshot_values(&v, None), Some((Some("i".into()), "m".into(), 5, 7, 3)));
    }

    /// Full lifecycle against a fake serve: start, idempotent start, conflict, stop with a
    /// sealed receipt, crash receipt, ack.
    #[test]
    fn lifecycle_with_fake_serve() {
        let d = tempfile::tempdir().unwrap();
        // a "serve" that answers prepare-stop on its port, then sleeps
        let fake = d.path().join("fake-carbon");
        std::fs::write(
            &fake,
            r#"#!/usr/bin/env python3
import sys, json, http.server, threading
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, v):
        b = json.dumps(v).encode(); self.send_response(200)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._send({"instance_id": "inst-1", "model_id": "m", "prompt_tokens_total": 10,
                    "completion_tokens_total": 4, "uptime_s": 1, "drain_complete": True})
    def do_GET(self):
        self._send({"status": "ok"})
print("fake serve up", flush=True)
http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
"#,
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
        if std::process::Command::new("python3").arg("-V").output().is_err() {
            eprintln!("skipping: python3 drives the fake serve");
            return;
        }
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let cfg = DaemonConfig {
            host: "127.0.0.1".into(),
            port: 0,
            state_dir: d.path().join("state"),
            token: None,
            default_serve_port: port,
            grace: Duration::from_secs(5),
            poll_interval: Duration::from_millis(50),
            oom_child_score: None,
            auto_restart: false,
            stop_serve_on_exit: false,
            log_capacity: 100,
            setsid: true,
            serve_exe: fake,
        };
        let m = Manager::new(cfg);
        let r = m.start("m", port, vec![]).unwrap();
        assert_eq!(r["idempotent"], false);
        assert_eq!(m.start("m", port, vec![]).unwrap()["idempotent"], true);
        assert!(matches!(m.start("other", port, vec![]), Err(ManagerError::Conflict(_))));
        // wait for the fake serve to listen
        for _ in 0..100 {
            if ureq::get(&format!("http://127.0.0.1:{port}/health")).call().is_ok() {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let stopped = m.stop(false).unwrap();
        assert_eq!(stopped["accounting"]["drainComplete"], true);
        assert_eq!(stopped["accounting"]["promptTokensTotal"], 10);
        assert_eq!(m.status()["running"], false);
        assert_eq!(m.status()["lastExitReason"], "stopped");
        assert_eq!(m.stop(false).unwrap()["already"], true);
        // a crash leaves a degraded receipt
        m.start("m", port, vec![]).unwrap();
        let pid = m.status()["pid"].as_u64().unwrap() as u32;
        signal(pid, true, libc::SIGKILL);
        for _ in 0..100 {
            if m.status()["running"] == false {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let pending = m.outbox.pending().unwrap();
        assert_eq!(pending.len(), 2);
        assert!(pending.iter().any(|r| r["reason"] == "engine-crashed"));
        for r in pending {
            m.outbox.ack(r["receiptId"].as_str().unwrap()).unwrap();
        }
        assert!(m.outbox.pending().unwrap().is_empty());
        let (logs, _) = m.ring.since(0);
        assert!(logs.iter().any(|l| l.text == "fake serve up"));
    }
}
