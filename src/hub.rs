// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Model download from the Hugging Face Hub or ModelScope.
//!
//! `--model org/name[@revision]` that is not a local directory is fetched into the standard
//! cache layout (`$HF_HUB_CACHE/models--org--name/snapshots/<sha>/` or
//! `$MODELSCOPE_CACHE/models/org/name/`), so other tools share the files. Only what the engine
//! reads is downloaded: configs, tokenizer files, chat templates and the safetensors shards the
//! index references.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    HuggingFace,
    ModelScope,
}

impl std::str::FromStr for Source {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "huggingface" | "hf" => Self::HuggingFace,
            "modelscope" | "ms" => Self::ModelScope,
            o => bail!("unknown model source '{o}' (huggingface | modelscope)"),
        })
    }
}

/// A local directory as is, otherwise a hub repo id downloaded (or found in the cache).
pub fn resolve(model: &Path, source: Source) -> Result<PathBuf> {
    resolve_with(model, source, true)
}

/// `resolve`; `weights = false` fetches only configs and tokenizer files (`--dummy-weight`).
pub fn resolve_with(model: &Path, source: Source, weights: bool) -> Result<PathBuf> {
    if model.is_dir() {
        return Ok(model.to_path_buf());
    }
    let spec = model.to_string_lossy().to_string();
    if model.is_file() || spec.starts_with('.') || spec.starts_with('/') || spec.matches('/').count() != 1 {
        bail!("model '{spec}' is neither a local directory nor a hub repo id (org/name)");
    }
    let (repo, revision) = match spec.split_once('@') {
        Some((r, v)) => (r.to_string(), Some(v.to_string())),
        None => (spec.clone(), None),
    };
    match source {
        Source::HuggingFace => hf_download(&repo, revision.as_deref().unwrap_or("main"), weights),
        Source::ModelScope => ms_download(&repo, revision.as_deref().unwrap_or("master"), weights),
    }
}

/// Files the engine reads; shards come from the safetensors index when there is one.
pub fn wanted(file: &str, shards: Option<&[String]>) -> bool {
    let name = file.rsplit('/').next().unwrap_or(file);
    if file.contains('/') && !file.starts_with("inference/") {
        // sub-folders (original/, onnx/, ...) hold other formats
        return false;
    }
    if name.ends_with(".safetensors") {
        return shards.is_none_or(|s| s.iter().any(|x| x == file));
    }
    name.ends_with(".json")
        || name.ends_with(".jinja")
        || name.ends_with(".tiktoken")
        || name == "tokenizer.model"
        || name.ends_with(".txt") && (name.starts_with("merges") || name.starts_with("vocab"))
}

fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .unwrap_or_else(|_| "https://huggingface.co".into())
        .trim_end_matches('/')
        .to_string()
}

fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn hf_cache_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("HF_HUB_CACHE") {
        return PathBuf::from(d);
    }
    if let Some(d) = std::env::var_os("HF_HOME") {
        return PathBuf::from(d).join("hub");
    }
    home().join(".cache/huggingface/hub")
}

fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .or_else(|| {
            let p = std::env::var_os("HF_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| home().join(".cache/huggingface"))
                .join("token");
            std::fs::read_to_string(p).ok().map(|t| t.trim().to_string())
        })
        .filter(|t| !t.is_empty())
}

fn offline() -> bool {
    ["HF_HUB_OFFLINE", "CARBON_OFFLINE"]
        .iter()
        .any(|k| matches!(std::env::var(k).as_deref(), Ok("1") | Ok("true") | Ok("yes")))
}

/// `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY` for `url`, unless `NO_PROXY` excludes its host.
fn proxy_for(url: &str) -> Option<String> {
    let env = |k: &str| {
        std::env::var(k)
            .or_else(|_| std::env::var(k.to_ascii_lowercase()))
            .ok()
            .filter(|v| !v.is_empty())
    };
    let host = url
        .split("://")
        .nth(1)?
        .split(['/', ':', '?'])
        .next()?
        .to_ascii_lowercase();
    if host == "localhost" || host.starts_with("127.") || host == "[::1]" {
        return None;
    }
    if let Some(no) = env("NO_PROXY") {
        for pat in no.split(',').map(|p| p.trim().trim_start_matches('*').to_ascii_lowercase()) {
            if pat == "*" || (!pat.is_empty() && (host == pat.trim_start_matches('.') || host.ends_with(&pat) && pat.starts_with('.'))) {
                return None;
            }
            if !pat.is_empty() && !pat.starts_with('.') && host.ends_with(&format!(".{pat}")) {
                return None;
            }
        }
    }
    if url.starts_with("https://") {
        env("HTTPS_PROXY").or_else(|| env("ALL_PROXY"))
    } else {
        env("HTTP_PROXY").or_else(|| env("ALL_PROXY"))
    }
}

fn agent(url: &str) -> ureq::Agent {
    let mut b = ureq::AgentBuilder::new().timeout_connect(std::time::Duration::from_secs(30));
    if let Some(p) = proxy_for(url) {
        match ureq::Proxy::new(&p) {
            Ok(p) => b = b.proxy(p),
            Err(e) => tracing::warn!("ignoring proxy '{p}': {e}"),
        }
    }
    b.build()
}

fn get(url: &str, token: Option<&str>) -> Result<ureq::Response> {
    let mut req = agent(url).get(url).set("User-Agent", "carbon-engine");
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    match req.call() {
        Ok(r) => Ok(r),
        Err(ureq::Error::Status(401 | 403, _)) => {
            bail!("{url}: access denied (gated or private repo? set HF_TOKEN)")
        }
        Err(ureq::Error::Status(404, _)) => bail!("{url}: not found"),
        Err(e) => Err(anyhow::Error::from(e).context(url.to_string())),
    }
}

/// Download `url` to `dest` (via a `.part` file), skipping files already complete.
fn fetch(url: &str, dest: &Path, size: Option<u64>, token: Option<&str>) -> Result<()> {
    if let Ok(m) = std::fs::metadata(dest) {
        if size.is_none_or(|s| s == m.len()) {
            return Ok(());
        }
    }
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let part = dest.with_extension(format!(
        "{}part",
        dest.extension()
            .map(|e| format!("{}.", e.to_string_lossy()))
            .unwrap_or_default()
    ));
    let mut last = None;
    for attempt in 0..4 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_secs(1 << attempt));
        }
        match fetch_once(url, &part, token) {
            Ok(n) => {
                if let Some(s) = size.filter(|&s| s != n) {
                    last = Some(anyhow::anyhow!("{url}: got {n} bytes, expected {s}"));
                    continue;
                }
                std::fs::rename(&part, dest)?;
                return Ok(());
            }
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap())
}

fn fetch_once(url: &str, part: &Path, token: Option<&str>) -> Result<u64> {
    let resp = get(url, token)?;
    let mut reader = resp.into_reader();
    let mut f = std::fs::File::create(part)?;
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let k = reader.read(&mut buf)?;
        if k == 0 {
            break;
        }
        f.write_all(&buf[..k])?;
        n += k as u64;
    }
    f.sync_all()?;
    Ok(n)
}

/// Download `files` (`(remote name, size)`) in parallel with `url_of`.
fn fetch_all(
    files: &[(String, Option<u64>)],
    dir: &Path,
    token: Option<&str>,
    url_of: &(dyn Fn(&str) -> String + Sync),
) -> Result<()> {
    let total: u64 = files.iter().filter_map(|f| f.1).sum();
    tracing::info!(
        "downloading {} files ({:.2} GiB) to {}",
        files.len(),
        total as f64 / (1u64 << 30) as f64,
        dir.display()
    );
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let threads = std::env::var("CARBON_DOWNLOAD_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8usize)
        .clamp(1, 64);
    let errors = parking_lot::Mutex::new(vec![]);
    std::thread::scope(|sc| {
        for _ in 0..threads.min(files.len().max(1)) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some((name, size)) = files.get(i) else {
                    break;
                };
                match fetch(&url_of(name), &dir.join(name), *size, token) {
                    Ok(()) => {
                        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                        tracing::info!("[{d}/{}] {name}", files.len());
                    }
                    Err(e) => errors.lock().push(format!("{name}: {e:#}")),
                }
            });
        }
    });
    let errors = errors.into_inner();
    if !errors.is_empty() {
        bail!("download failed:\n  {}", errors.join("\n  "));
    }
    Ok(())
}

/// Shard list from `model.safetensors.index.json` (already downloaded into `dir`).
fn index_shards(dir: &Path) -> Option<Vec<String>> {
    let text = std::fs::read_to_string(dir.join("model.safetensors.index.json")).ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let mut s: Vec<String> = v
        .get("weight_map")?
        .as_object()?
        .values()
        .filter_map(|x| x.as_str().map(String::from))
        .collect();
    s.sort();
    s.dedup();
    Some(s)
}

fn hf_download(repo: &str, revision: &str, with_weights: bool) -> Result<PathBuf> {
    let root = hf_cache_dir().join(format!("models--{}", repo.replace('/', "--")));
    let ref_file = root.join("refs").join(revision);
    if offline() {
        let sha = std::fs::read_to_string(&ref_file)
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| revision.to_string());
        let dir = root.join("snapshots").join(sha);
        if dir.join("config.json").is_file() {
            return Ok(dir);
        }
        bail!("offline mode: '{repo}' is not in the cache ({})", root.display());
    }
    let token = hf_token();
    let ep = hf_endpoint();
    let info: Value = get(
        &format!("{ep}/api/models/{repo}/revision/{revision}?blobs=true"),
        token.as_deref(),
    )?
    .into_json()
    .with_context(|| format!("model info for {repo}"))?;
    let sha = info
        .get("sha")
        .and_then(|s| s.as_str())
        .unwrap_or(revision)
        .to_string();
    let siblings: Vec<(String, Option<u64>)> = info
        .get("siblings")
        .and_then(|s| s.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|f| {
                    let name = f.get("rfilename")?.as_str()?.to_string();
                    Some((name, f.get("size").and_then(|s| s.as_u64())))
                })
                .collect()
        })
        .unwrap_or_default();
    if siblings.is_empty() {
        bail!("{repo}@{revision} lists no files");
    }
    let dir = root.join("snapshots").join(&sha);
    let url_of = |name: &str| format!("{ep}/{repo}/resolve/{sha}/{name}");
    let small: Vec<_> = siblings
        .iter()
        .filter(|(n, _)| wanted(n, Some(&[])) && !n.ends_with(".safetensors"))
        .cloned()
        .collect();
    fetch_all(&small, &dir, token.as_deref(), &url_of)?;
    let shards = index_shards(&dir);
    let weights: Vec<_> = siblings
        .iter()
        .filter(|(n, _)| n.ends_with(".safetensors") && wanted(n, shards.as_deref()))
        .cloned()
        .collect();
    if weights.is_empty() && with_weights {
        bail!("{repo} has no safetensors weights (GGUF / .bin checkpoints are not loaded from the hub)");
    }
    if with_weights {
        fetch_all(&weights, &dir, token.as_deref(), &url_of)?;
    }
    std::fs::create_dir_all(ref_file.parent().unwrap())?;
    std::fs::write(&ref_file, &sha)?;
    Ok(dir)
}

fn ms_endpoint() -> String {
    std::env::var("MODELSCOPE_ENDPOINT")
        .or_else(|_| std::env::var("MODELSCOPE_DOMAIN").map(|d| {
            if d.starts_with("http") {
                d
            } else {
                format!("https://{d}")
            }
        }))
        .unwrap_or_else(|_| "https://www.modelscope.cn".into())
        .trim_end_matches('/')
        .to_string()
}

fn ms_download(repo: &str, revision: &str, with_weights: bool) -> Result<PathBuf> {
    let cache = std::env::var_os("MODELSCOPE_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".cache/modelscope/hub"));
    let dir = cache.join("models").join(repo);
    if offline() {
        if dir.join("config.json").is_file() {
            return Ok(dir);
        }
        bail!("offline mode: '{repo}' is not in the ModelScope cache ({})", dir.display());
    }
    let token = std::env::var("MODELSCOPE_API_TOKEN").ok();
    let ep = ms_endpoint();
    let v: Value = get(
        &format!("{ep}/api/v1/models/{repo}/repo/files?Revision={revision}&Recursive=true"),
        token.as_deref(),
    )?
    .into_json()?;
    let files: Vec<(String, Option<u64>)> = v
        .pointer("/Data/Files")
        .and_then(|f| f.as_array())
        .map(|a| {
            a.iter()
                .filter(|f| f.get("Type").and_then(|t| t.as_str()) != Some("tree"))
                .filter_map(|f| {
                    Some((
                        f.get("Path")?.as_str()?.to_string(),
                        f.get("Size").and_then(|s| s.as_u64()),
                    ))
                })
                .collect()
        })
        .unwrap_or_default();
    if files.is_empty() {
        bail!("ModelScope repo {repo}@{revision} lists no files");
    }
    let url_of =
        |name: &str| format!("{ep}/api/v1/models/{repo}/repo?Revision={revision}&FilePath={name}");
    let small: Vec<_> = files
        .iter()
        .filter(|(n, _)| wanted(n, Some(&[])) && !n.ends_with(".safetensors"))
        .cloned()
        .collect();
    fetch_all(&small, &dir, token.as_deref(), &url_of)?;
    let shards = index_shards(&dir);
    let weights: Vec<_> = files
        .iter()
        .filter(|(n, _)| n.ends_with(".safetensors") && wanted(n, shards.as_deref()))
        .cloned()
        .collect();
    if with_weights {
        fetch_all(&weights, &dir, token.as_deref(), &url_of)?;
    }
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    /// A tiny HTTP/1.1 server answering from a closure (path -> (status, body)).
    fn serve(handler: impl Fn(&str) -> (u16, Vec<u8>) + Send + 'static) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            for s in l.incoming() {
                let mut s = s.unwrap();
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut line = String::new();
                r.read_line(&mut line).unwrap();
                loop {
                    let mut h = String::new();
                    r.read_line(&mut h).unwrap();
                    if h == "\r\n" || h.is_empty() {
                        break;
                    }
                }
                let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
                let (code, body) = handler(&path);
                let _ = write!(
                    s,
                    "HTTP/1.1 {code} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = s.write_all(&body);
            }
        });
        format!("http://{addr}")
    }

    #[test]
    fn proxy_selection() {
        // only this test touches the proxy variables
        std::env::set_var("HTTPS_PROXY", "http://proxy:3128");
        std::env::set_var("NO_PROXY", "internal.example,.corp");
        assert_eq!(proxy_for("https://huggingface.co/api").as_deref(), Some("http://proxy:3128"));
        assert_eq!(proxy_for("http://127.0.0.1:9/x"), None);
        assert_eq!(proxy_for("https://internal.example/x"), None);
        assert_eq!(proxy_for("https://a.internal.example/x"), None);
        assert_eq!(proxy_for("https://hub.corp/x"), None);
        std::env::remove_var("HTTPS_PROXY");
        std::env::remove_var("NO_PROXY");
    }

    #[test]
    fn file_filter() {
        assert!(wanted("config.json", None));
        assert!(wanted("tokenizer.json", None));
        assert!(wanted("chat_template.jinja", None));
        assert!(wanted("model-00001-of-00002.safetensors", None));
        let shards = vec!["model-00001-of-00002.safetensors".to_string()];
        assert!(!wanted("consolidated.safetensors", Some(&shards)));
        assert!(!wanted("original/params.json", None));
        assert!(!wanted("pytorch_model.bin", None));
        assert!(!wanted("README.md", None));
    }

    #[test]
    fn hf_download_into_cache_layout() {
        let base = serve(|path| {
            let body = |s: &str| (200, s.as_bytes().to_vec());
            match path {
                "/api/models/org/tiny/revision/main?blobs=true" => body(
                    r#"{"sha":"abc123","siblings":[
                        {"rfilename":"config.json","size":2},
                        {"rfilename":"model.safetensors.index.json","size":36},
                        {"rfilename":"a.safetensors","size":3},
                        {"rfilename":"unused.safetensors","size":5},
                        {"rfilename":"README.md","size":1}]}"#,
                ),
                "/org/tiny/resolve/abc123/config.json" => body("{}"),
                "/org/tiny/resolve/abc123/model.safetensors.index.json" => {
                    body(r#"{"weight_map":{"x":"a.safetensors"}}"#)
                }
                "/org/tiny/resolve/abc123/a.safetensors" => body("abc"),
                _ => (404, vec![]),
            }
        });
        let cache = tempfile::tempdir().unwrap();
        // env vars are process-global: this is the only test touching them
        std::env::set_var("HF_ENDPOINT", &base);
        std::env::set_var("HF_HUB_CACHE", cache.path());
        let dir = resolve(Path::new("org/tiny"), Source::HuggingFace).unwrap();
        assert_eq!(dir, cache.path().join("models--org--tiny/snapshots/abc123"));
        assert_eq!(std::fs::read_to_string(dir.join("a.safetensors")).unwrap(), "abc");
        assert!(!dir.join("unused.safetensors").exists());
        assert!(!dir.join("README.md").exists());
        assert_eq!(
            std::fs::read_to_string(cache.path().join("models--org--tiny/refs/main")).unwrap(),
            "abc123"
        );
        // offline: served from the cache without the network
        std::env::set_var("HF_HUB_OFFLINE", "1");
        std::env::set_var("HF_ENDPOINT", "http://127.0.0.1:9");
        assert_eq!(resolve(Path::new("org/tiny"), Source::HuggingFace).unwrap(), dir);
        assert!(resolve(Path::new("org/missing"), Source::HuggingFace).is_err());
        std::env::remove_var("HF_HUB_OFFLINE");
        std::env::remove_var("HF_ENDPOINT");
        std::env::remove_var("HF_HUB_CACHE");
    }
}
