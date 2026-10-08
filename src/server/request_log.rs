// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Optional inbound-request logging: with `CARBON_API_LOG_DIR` (or `FREETOKEN_API_LOG_DIR`) set,
//! every generation request body is appended as one JSON line to
//! `<dir>/requests-<start>-<pid>.jsonl` (mode 0600: bodies carry prompts). A background thread
//! writes through a bounded queue; when the disk cannot keep up records are dropped, serving is
//! never blocked.

use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;
use std::sync::mpsc;

pub struct RequestLog {
    tx: mpsc::SyncSender<String>,
    pub path: PathBuf,
}

impl RequestLog {
    pub fn from_env() -> Option<Self> {
        let dir = std::env::var_os("CARBON_API_LOG_DIR")
            .or_else(|| std::env::var_os("FREETOKEN_API_LOG_DIR"))?;
        match Self::open(PathBuf::from(dir)) {
            Ok(l) => Some(l),
            Err(e) => {
                tracing::warn!("request logging disabled: {e:#}");
                None
            }
        }
    }

    pub fn open(dir: PathBuf) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&dir)?;
        let stamp = super::metrics::utc_timestamp()
            .replace([':', '-'], "")
            .replace('T', "-")
            .trim_end_matches('Z')
            .to_string();
        let path = dir.join(format!("requests-{stamp}-{}.jsonl", std::process::id()));
        let mut opts = std::fs::OpenOptions::new();
        opts.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(&path)?;
        let (tx, rx) = mpsc::sync_channel::<String>(10_000);
        std::thread::Builder::new()
            .name("request-log".into())
            .spawn(move || {
                let mut warned = false;
                for line in rx {
                    if let Err(e) = f.write_all(line.as_bytes()).and_then(|_| f.flush()) {
                        if !warned {
                            tracing::warn!("failed to write request log: {e}");
                            warned = true;
                        }
                    }
                }
            })?;
        tracing::info!("logging API requests to {}", path.display());
        Ok(Self { tx, path })
    }

    pub fn log(&self, endpoint: &str, client: Option<String>, body: &Value) {
        let rec = json!({
            "timestamp": super::metrics::utc_timestamp(),
            "endpoint": endpoint,
            "client": client,
            "request": body,
        });
        if self.tx.try_send(format!("{rec}\n")).is_err() {
            static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!("request log queue full; dropping records (slow log disk?)");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let log = RequestLog::open(dir.path().to_path_buf()).unwrap();
        log.log("/v1/chat/completions", Some("127.0.0.1".into()), &json!({"a": 1}));
        log.log("/v1/messages", None, &json!({"b": "ü"}));
        let path = log.path.clone();
        drop(log); // closes the queue; the writer drains it
        for _ in 0..100 {
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            if text.lines().count() == 2 {
                let first: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
                assert_eq!(first["endpoint"], "/v1/chat/completions");
                assert_eq!(first["request"]["a"], 1);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mode = std::fs::metadata(&path).unwrap().permissions().mode();
                    assert_eq!(mode & 0o777, 0o600);
                }
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        panic!("log lines never appeared");
    }
}
