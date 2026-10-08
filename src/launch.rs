// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! `carbon launch <agent>`: point a coding agent (Claude Code, Codex, OpenCode, OpenClaw,
//! Hermes, DeepSeek Harness) at a running carbon server and start it.
//!
//! The served model, its context window and its input modalities are read from the server;
//! each agent's provider configuration is written (previous files are kept as `*.bak`), cloud
//! API keys are removed from the child environment so nothing leaks to a hosted provider, and
//! a missing agent CLI can be installed with its official installer.

use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

pub const DEFAULT_SERVER: &str = "http://127.0.0.1:1919";
const CODEX_PROFILE: &str = "carbon-launch";
const CODEX_PROVIDER_NAME: &str = "carbon";
const CODEX_CATALOG_NAME: &str = "carbon-model.json";
const CODEX_API_KEY_ENV: &str = "CARBON_API_KEY";
/// Used when the server reports no context length (too low only costs earlier compaction).
const FALLBACK_CONTEXT_WINDOW: usize = 128_000;
const MAX_OUTPUT_TOKENS_CAP: usize = 32_768;
const PROVIDER: &str = "carbon";
const HERMES_MIN_CONTEXT: usize = 64_000;
const DSH_SETTINGS_NAME: &str = "carbon-launch.settings.yaml";
const DSH_PATCH_NAME: &str = "carbon-launch.cordis.patch.yml";
const CLOUD_KEYS: [&str; 8] = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
    "XAI_API_KEY",
    "OPENROUTER_API_KEY",
];

pub const AGENTS: [&str; 6] = ["claude", "codex", "dsh", "hermes", "openclaw", "opencode"];

#[derive(Clone, Debug, PartialEq)]
pub struct ServerUrl {
    pub origin: String,
    pub openai_base_url: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServedModel {
    pub model_id: String,
    pub models: Vec<String>,
    pub context_length: Option<usize>,
    pub input_modalities: Vec<String>,
}

pub struct LaunchContext {
    pub server: ServerUrl,
    pub model: ServedModel,
    pub extra_args: Vec<String>,
    pub dry_run: bool,
    pub assume_yes: bool,
    /// Configuration files are written under this directory (the user's home).
    pub home: PathBuf,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CommandSpec {
    pub argv: Vec<String>,
    pub env: Vec<(String, String)>,
    pub unset_env: Vec<String>,
}

/// Normalise `--server` (origin or `/v1` URL; `$CARBON_HOST`, then the default).
pub fn resolve_server_url(server: Option<&str>) -> Result<ServerUrl> {
    let raw = server
        .map(String::from)
        .or_else(|| std::env::var("CARBON_HOST").ok())
        .or_else(|| std::env::var("FREETOKEN_HOST").ok())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_SERVER.into());
    let raw = raw.trim();
    let raw = if raw.contains("://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    };
    let (scheme, rest) = raw.split_once("://").unwrap();
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };
    let path = path.trim_end_matches('/');
    if !path.is_empty() && path != "/v1" {
        bail!("--server accepts only an origin or a /v1 base URL");
    }
    if authority.is_empty() {
        bail!("invalid server URL '{raw}'");
    }
    // wildcard binds are not connectable addresses
    let authority = if let Some(port) = authority.strip_prefix("0.0.0.0") {
        format!("127.0.0.1{port}")
    } else if let Some(port) = authority.strip_prefix("[::]") {
        format!("[::1]{port}")
    } else {
        authority.to_string()
    };
    let origin = format!("{scheme}://{authority}");
    Ok(ServerUrl {
        openai_base_url: format!("{origin}/v1"),
        origin,
    })
}

fn get_json(url: &str) -> Result<Value> {
    Ok(ureq::get(url)
        .timeout(std::time::Duration::from_secs(5))
        .set("Accept", "application/json")
        .call()?
        .into_json()?)
}

pub fn discover_model(server: &ServerUrl) -> Result<ServedModel> {
    let payload = get_json(&format!("{}/models", server.openai_base_url))
        .with_context(|| format!("cannot connect to the carbon server at {}", server.origin))?;
    let data = payload["data"]
        .as_array()
        .context("server returned an invalid /v1/models response")?;
    let mut models = vec![];
    let mut context_length = None;
    for item in data {
        let Some(id) = item["id"].as_str() else {
            continue;
        };
        if models.is_empty() {
            context_length = item["max_model_len"]
                .as_u64()
                .or(item["context_length"].as_u64())
                .filter(|&c| c > 0)
                .map(|c| c as usize);
        }
        models.push(id.to_string());
    }
    if models.is_empty() {
        bail!("server at {} reported no models", server.origin);
    }
    let stats = get_json(&format!("{}/v1/stats", server.origin))
        .ok()
        .map(|s| s["model"].clone())
        .unwrap_or(Value::Null);
    if context_length.is_none() {
        context_length = stats["ctx"].as_u64().map(|c| c as usize);
    }
    let mut input_modalities = vec!["text".to_string()];
    if let Some(m) = stats["input_modalities"].as_array() {
        input_modalities.extend(
            m.iter()
                .filter_map(|x| x.as_str())
                .filter(|x| *x != "text")
                .map(String::from),
        );
    }
    Ok(ServedModel {
        model_id: models[0].clone(),
        models,
        context_length,
        input_modalities,
    })
}

impl LaunchContext {
    fn window(&self) -> usize {
        self.model.context_length.unwrap_or(FALLBACK_CONTEXT_WINDOW)
    }

    /// Scales with the window: agents derive their compaction headroom from it.
    fn max_output(&self) -> usize {
        (self.window() / 4).clamp(1, MAX_OUTPUT_TOKENS_CAP)
    }

    fn images(&self) -> bool {
        self.model.input_modalities.iter().any(|m| m == "image")
    }

    fn model_ids(&self) -> Vec<String> {
        let mut out: Vec<String> = vec![];
        for m in std::iter::once(&self.model.model_id).chain(&self.model.models) {
            if !m.is_empty() && !out.contains(m) {
                out.push(m.clone());
            }
        }
        out
    }

    fn modalities(&self) -> Value {
        if self.images() {
            json!(["text", "image"])
        } else {
            json!(["text"])
        }
    }
}

fn env(pairs: &[(&str, String)]) -> Vec<(String, String)> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// Write `text`, keeping a timestamped `.bak` of a different previous version.
pub fn write_with_backup(path: &Path, text: &str) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    if let Ok(old) = std::fs::read_to_string(path) {
        if old != text {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let name = format!(
                "{}.{stamp}.bak",
                path.file_name().unwrap_or_default().to_string_lossy()
            );
            std::fs::copy(path, path.with_file_name(name))?;
        }
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

fn write_json(path: &Path, v: &Value) -> Result<()> {
    write_with_backup(path, &(serde_json::to_string_pretty(v)? + "\n"))
}

fn load_json_object(path: &Path) -> Map<String, Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

fn load_yaml_object(path: &Path) -> serde_yaml::Mapping {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_yaml::from_str::<serde_yaml::Value>(&t).ok())
        .and_then(|v| v.as_mapping().cloned())
        .unwrap_or_default()
}

fn yaml(v: Value) -> serde_yaml::Value {
    serde_yaml::to_value(v).unwrap_or(serde_yaml::Value::Null)
}

fn toml_string(s: &str) -> String {
    Value::String(s.into()).to_string()
}

/// Drop `header` and its keys from a TOML document.
fn remove_toml_section(text: &str, header: &str) -> String {
    let mut kept = vec![];
    let mut skipping = false;
    for line in text.lines() {
        let t = line.trim();
        if t == header {
            skipping = true;
            continue;
        }
        if skipping && t.starts_with('[') && t.ends_with(']') {
            skipping = false;
        }
        if !skipping {
            kept.push(line);
        }
    }
    let r = kept.join("\n");
    let r = r.trim();
    if r.is_empty() {
        String::new()
    } else {
        format!("{r}\n\n")
    }
}

/// Drop a root-level `key = "value"` line.
fn remove_toml_root_value(text: &str, key: &str, value: &str) -> String {
    let expected = toml_string(value);
    let mut root = true;
    let mut kept = vec![];
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with('[') && t.ends_with(']') {
            root = false;
        }
        if root {
            if let Some((lhs, rhs)) = t.split_once('=') {
                let rhs = rhs.split('#').next().unwrap_or("").trim();
                if lhs.trim() == key && rhs == expected {
                    continue;
                }
            }
        }
        kept.push(line);
    }
    let r = kept.join("\n");
    let r = r.trim();
    if r.is_empty() {
        String::new()
    } else {
        format!("{r}\n")
    }
}

fn codex_catalog_entry(id: &str, window: usize, images: bool) -> Value {
    json!({
        "slug": id, "display_name": id, "description": id,
        "context_window": window, "max_context_window": window,
        "effective_context_window_percent": 100, "shell_type": "default", "visibility": "list",
        "supported_in_api": true, "priority": 0, "additional_speed_tiers": [], "service_tiers": [],
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "input_modalities": if images { json!(["text", "image"]) } else { json!(["text"]) },
        "base_instructions": "", "default_reasoning_summary": "none",
        "supported_reasoning_levels": [], "supports_reasoning_summaries": false,
        "experimental_supported_tools": [], "supports_search_tool": false,
        "web_search_tool_type": "text", "supports_image_detail_original": false,
        "support_verbosity": true, "default_verbosity": "low",
        "supports_parallel_tool_calls": false, "use_responses_lite": false,
        "model_messages": null, "upgrade": null,
    })
}

fn prepare_codex(ctx: &LaunchContext) -> Result<CommandSpec> {
    let dir = ctx.home.join(".codex");
    let catalog = dir.join(CODEX_CATALOG_NAME);
    if !ctx.dry_run {
        let models: Vec<Value> = ctx
            .model_ids()
            .iter()
            .map(|id| codex_catalog_entry(id, ctx.window(), ctx.images()))
            .collect();
        write_json(&catalog, &json!({"models": models}))?;
        let base = &ctx.server.openai_base_url;
        let profile = format!(
            "model = {}\nopenai_base_url = {}\nmodel_provider = {}\nmodel_catalog_json = {}\n\n\
             [model_providers.{CODEX_PROFILE}]\nname = {}\nbase_url = {}\nwire_api = \"responses\"\nenv_key = {}\n",
            toml_string(&ctx.model.model_id),
            toml_string(base),
            toml_string(CODEX_PROFILE),
            toml_string(&catalog.to_string_lossy()),
            toml_string(CODEX_PROVIDER_NAME),
            toml_string(base),
            toml_string(CODEX_API_KEY_ENV),
        );
        write_with_backup(&dir.join(format!("{CODEX_PROFILE}.config.toml")), &profile)?;
        // older launches wrote the profile into config.toml itself: migrate it out
        let config = dir.join("config.toml");
        if let Ok(existing) = std::fs::read_to_string(&config) {
            let t = remove_toml_section(&existing, &format!("[profiles.{CODEX_PROFILE}]"));
            let t = remove_toml_section(&t, &format!("[model_providers.{CODEX_PROFILE}]"));
            let t = remove_toml_root_value(&t, "profile", CODEX_PROFILE);
            if t != existing {
                write_with_backup(&config, &t)?;
            }
        }
    }
    let mut argv = strings(&["codex", "--profile", CODEX_PROFILE, "-c"]);
    argv.push(format!(
        "model_catalog_json={}",
        toml_string(&catalog.to_string_lossy())
    ));
    argv.extend(["-m".into(), ctx.model.model_id.clone()]);
    argv.extend(ctx.extra_args.iter().cloned());
    let mut unset = strings(&CLOUD_KEYS);
    unset.push("CODEX_API_KEY".into());
    Ok(CommandSpec {
        argv,
        env: env(&[(CODEX_API_KEY_ENV, "carbon".into())]),
        unset_env: unset,
    })
}

fn prepare_claude(ctx: &LaunchContext) -> Result<CommandSpec> {
    let id = ctx.model.model_id.clone();
    let mut argv = strings(&["claude", "--model"]);
    argv.push(id.clone());
    argv.extend(ctx.extra_args.iter().cloned());
    Ok(CommandSpec {
        argv,
        env: env(&[
            // honoured for model ids not starting with "claude-" (else 200k is assumed)
            ("CLAUDE_CODE_MAX_CONTEXT_TOKENS", ctx.window().to_string()),
            ("CLAUDE_CODE_MAX_OUTPUT_TOKENS", ctx.max_output().to_string()),
            ("ANTHROPIC_BASE_URL", ctx.server.origin.clone()),
            ("ANTHROPIC_API_KEY", String::new()),
            ("ANTHROPIC_AUTH_TOKEN", "carbon".into()),
            ("ANTHROPIC_MODEL", id.clone()),
            ("ANTHROPIC_DEFAULT_OPUS_MODEL", id.clone()),
            ("ANTHROPIC_DEFAULT_SONNET_MODEL", id.clone()),
            ("ANTHROPIC_DEFAULT_HAIKU_MODEL", id.clone()),
            ("CLAUDE_CODE_SUBAGENT_MODEL", id),
            ("CLAUDE_CODE_ATTRIBUTION_HEADER", "0".into()),
        ]),
        unset_env: vec![],
    })
}

fn prepare_opencode(ctx: &LaunchContext) -> Result<CommandSpec> {
    let ids = ctx.model_ids();
    if !ctx.dry_run {
        // put our models at the front of OpenCode's recent list
        let path = ctx.home.join(".local/state/opencode/model.json");
        let mut state = load_json_object(&path);
        let mut recent: Vec<Value> = state
            .get("recent")
            .and_then(|r| r.as_array())
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|e| {
                !(e["providerID"] == PROVIDER
                    && e["modelID"].as_str().is_some_and(|m| ids.iter().any(|i| i == m)))
            })
            .collect();
        for id in ids.iter().rev() {
            recent.insert(0, json!({"providerID": PROVIDER, "modelID": id}));
        }
        recent.truncate(10);
        state.entry("favorite").or_insert(json!([]));
        state.entry("variant").or_insert(json!({}));
        state.insert("recent".into(), Value::Array(recent));
        write_json(&path, &Value::Object(state))?;
    }
    // without limit.context OpenCode reads the window as 0 and never compacts
    let mut models = Map::new();
    for id in &ids {
        let mut e = json!({"name": id, "limit": {"context": ctx.window(), "output": ctx.max_output()}});
        if ctx.images() {
            e["modalities"] = json!({"input": ["text", "image"], "output": ["text"]});
        }
        models.insert(id.clone(), e);
    }
    let config = json!({
        "$schema": "https://opencode.ai/config.json",
        "provider": {PROVIDER: {
            "npm": "@ai-sdk/openai-compatible",
            "name": "carbon",
            "options": {"baseURL": ctx.server.openai_base_url},
            "models": models,
        }},
        "model": format!("{PROVIDER}/{}", ctx.model.model_id),
    });
    let mut argv = strings(&["opencode"]);
    argv.extend(ctx.extra_args.iter().cloned());
    Ok(CommandSpec {
        argv,
        env: env(&[("OPENCODE_CONFIG_CONTENT", config.to_string())]),
        unset_env: vec![],
    })
}

fn confirm(question: &str, assume_yes: bool) -> bool {
    use std::io::{BufRead, IsTerminal, Write};
    if assume_yes {
        return true;
    }
    if !std::io::stdin().is_terminal() {
        eprintln!("{question} cancelled: stdin is not interactive (pass -y)");
        return false;
    }
    loop {
        eprint!("{question} [y/n] ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        if std::io::stdin().lock().read_line(&mut line).is_err() {
            return false;
        }
        match line.trim().to_ascii_lowercase().as_str() {
            "y" => return true,
            "n" => return false,
            _ => eprintln!("Please answer y or n."),
        }
    }
}

fn obj<'a>(parent: &'a mut Map<String, Value>, key: &str) -> &'a mut Map<String, Value> {
    if !parent.get(key).is_some_and(|v| v.is_object()) {
        parent.insert(key.into(), json!({}));
    }
    parent.get_mut(key).unwrap().as_object_mut().unwrap()
}

fn prepare_openclaw(ctx: &LaunchContext) -> Result<CommandSpec> {
    if !ctx.dry_run {
        let path = ctx.home.join(".openclaw/openclaw.json");
        let mut config = load_json_object(&path);
        let has_provider = config
            .get("models")
            .and_then(|m| m.get("providers"))
            .and_then(|p| p.get(PROVIDER))
            .is_some_and(|p| p.is_object());
        if !has_provider {
            eprintln!(
                "carbon launch will modify OpenClaw config for the first time.\n  File: {}\n  Model: {PROVIDER}/{}\n  Base URL: {}",
                path.display(),
                ctx.model.model_id,
                ctx.server.openai_base_url
            );
            if !confirm("Continue and update OpenClaw config?", ctx.assume_yes) {
                bail!("OpenClaw config patch cancelled");
            }
        }
        let providers = obj(obj(&mut config, "models"), "providers");
        let mut provider = providers
            .get(PROVIDER)
            .and_then(|p| p.as_object())
            .cloned()
            .unwrap_or_default();
        let existing: Vec<Value> = provider
            .get("models")
            .and_then(|m| m.as_array())
            .cloned()
            .unwrap_or_default();
        let models: Vec<Value> = ctx
            .model_ids()
            .iter()
            .map(|id| {
                let mut e = json!({
                    "id": id, "name": id, "input": ctx.modalities(),
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": ctx.window(), "maxTokens": ctx.max_output(),
                });
                // keep user additions to an existing entry
                if let Some(old) = existing.iter().find(|o| o["id"] == *id).and_then(|o| o.as_object()) {
                    for (k, v) in old {
                        e.as_object_mut().unwrap().entry(k.clone()).or_insert(v.clone());
                    }
                }
                e
            })
            .collect();
        provider.insert("baseUrl".into(), json!(ctx.server.openai_base_url));
        provider.insert("apiKey".into(), json!("carbon-local"));
        provider.insert("api".into(), json!("openai-completions"));
        provider.insert("models".into(), Value::Array(models));
        providers.insert(PROVIDER.into(), Value::Object(provider));
        let primary = format!("{PROVIDER}/{}", ctx.model.model_id);
        obj(obj(obj(&mut config, "agents"), "defaults"), "model")
            .insert("primary".into(), json!(primary));
        write_json(&path, &Value::Object(config))?;
        // session-level overrides would keep old sessions on another model
        let sessions_path = ctx.home.join(".openclaw/agents/main/sessions/sessions.json");
        if sessions_path.exists() {
            let mut sessions = load_json_object(&sessions_path);
            let mut changed = false;
            for s in sessions.values_mut().filter_map(|s| s.as_object_mut()) {
                changed |= s.remove("modelOverride").is_some();
                changed |= s.remove("providerOverride").is_some();
                if s.get("model").and_then(|m| m.as_str()).is_some_and(|m| !m.is_empty() && m != ctx.model.model_id) {
                    s.insert("model".into(), json!(ctx.model.model_id));
                    changed = true;
                }
            }
            if changed {
                write_json(&sessions_path, &Value::Object(sessions))?;
            }
        }
    }
    let argv = if ctx.extra_args.is_empty() {
        strings(&["openclaw", "chat"])
    } else {
        std::iter::once("openclaw".to_string())
            .chain(ctx.extra_args.iter().cloned())
            .collect()
    };
    Ok(CommandSpec {
        argv,
        env: vec![],
        unset_env: strings(&CLOUD_KEYS),
    })
}

fn prepare_hermes(ctx: &LaunchContext) -> Result<CommandSpec> {
    if !ctx.dry_run {
        let path = ctx.home.join(".hermes/config.yaml");
        let mut config = load_yaml_object(&path);
        let key = serde_yaml::Value::from("model");
        let mut model = config
            .get(&key)
            .and_then(|m| m.as_mapping())
            .cloned()
            .unwrap_or_default();
        let set = |m: &mut serde_yaml::Mapping, k: &str, v: serde_yaml::Value| {
            m.insert(serde_yaml::Value::from(k), v);
        };
        set(&mut model, "default", ctx.model.model_id.clone().into());
        set(&mut model, "provider", "custom".into());
        set(&mut model, "base_url", ctx.server.openai_base_url.clone().into());
        // carbon is unauthenticated, but some clients reject an empty key
        set(&mut model, "api_key", "carbon-local".into());
        set(&mut model, "context_length", (ctx.window() as u64).into());
        config.insert(key, serde_yaml::Value::Mapping(model));
        if ctx.window() < HERMES_MIN_CONTEXT {
            eprintln!(
                "warning: Hermes refuses to start under {HERMES_MIN_CONTEXT} tokens of context and this model reports {}.",
                ctx.window()
            );
        }
        write_with_backup(&path, &serde_yaml::to_string(&config)?)?;
    }
    let argv = if ctx.extra_args.is_empty() {
        strings(&["hermes", "chat"])
    } else {
        std::iter::once("hermes".to_string())
            .chain(ctx.extra_args.iter().cloned())
            .collect()
    };
    Ok(CommandSpec {
        argv,
        env: vec![],
        unset_env: vec![],
    })
}

fn prepare_dsh(ctx: &LaunchContext) -> Result<CommandSpec> {
    let home = std::env::var_os("DSH_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| ctx.home.join(".dsh"));
    let settings = home.join(DSH_SETTINGS_NAME);
    let patch = home.join(DSH_PATCH_NAME);
    if !ctx.dry_run {
        // keep what dsh itself writes back into its settings document
        let mut config = load_yaml_object(&settings);
        let models: Vec<Value> = ctx
            .model_ids()
            .iter()
            .map(|id| {
                json!({"id": id, "name": format!("{id} (carbon)"), "inputModalities": ctx.modalities(),
                       "contextWindow": ctx.window(), "maxTokens": ctx.max_output()})
            })
            .collect();
        config.insert(
            "llm-deepseek".into(),
            yaml(json!({"baseURL": ctx.server.openai_base_url, "models": models})),
        );
        config.insert(
            "agent-default-model".into(),
            yaml(json!({"provider": "deepseek-official", "model": ctx.model.model_id})),
        );
        write_with_backup(&settings, &serde_yaml::to_string(&config)?)?;
        let p = yaml(json!([{"id": "settings", "config": {"path": settings.to_string_lossy()}}]));
        write_with_backup(&patch, &serde_yaml::to_string(&p)?)?;
    }
    let mut profile = "web".to_string();
    let mut app: Vec<String> = ctx.extra_args.clone();
    if app.first().is_some_and(|a| a == "web") {
        app.remove(0);
    } else if app.len() >= 2 && app[0] == "--profile" {
        profile = app[1].clone();
        app.drain(..2);
    }
    let mut argv = strings(&["dsh", "--profile"]);
    argv.push(profile);
    argv.push("--patch".into());
    argv.push(patch.to_string_lossy().into());
    argv.extend(app);
    Ok(CommandSpec {
        argv,
        env: env(&[
            ("DEEPSEEK_BASE_URL", ctx.server.openai_base_url.clone()),
            ("DEEPSEEK_API_KEY", "carbon".into()),
            ("DSH_TELEMETRY_DISABLED", "1".into()),
        ]),
        unset_env: strings(&CLOUD_KEYS),
    })
}

pub fn prepare(agent: &str, ctx: &LaunchContext) -> Result<CommandSpec> {
    match agent {
        "codex" => prepare_codex(ctx),
        "claude" => prepare_claude(ctx),
        "opencode" => prepare_opencode(ctx),
        "openclaw" => prepare_openclaw(ctx),
        "hermes" => prepare_hermes(ctx),
        "dsh" => prepare_dsh(ctx),
        o => bail!("unknown agent '{o}' ({})", AGENTS.join(" | ")),
    }
}

/// The official installer of each agent CLI: (display name, command, required tools).
fn installer(agent: &str) -> Option<(&'static str, Vec<&'static str>, Vec<&'static str>)> {
    Some(match agent {
        "codex" => (
            "Codex CLI",
            vec!["sh", "-c", "curl -fsSL https://chatgpt.com/codex/install.sh | sh"],
            vec!["curl", "sh"],
        ),
        "claude" => (
            "Claude Code",
            vec!["bash", "-c", "curl -fsSL https://claude.ai/install.sh | bash"],
            vec!["curl", "bash"],
        ),
        "opencode" => (
            "OpenCode",
            vec!["bash", "-c", "set -o pipefail; curl -fsSL https://opencode.ai/install | bash"],
            vec!["curl", "bash"],
        ),
        "openclaw" => (
            "OpenClaw",
            vec!["bash", "-c", "curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard"],
            vec!["curl", "bash"],
        ),
        "hermes" => (
            "Hermes",
            vec!["bash", "-lc", "curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --skip-setup --non-interactive"],
            vec!["bash", "curl", "git"],
        ),
        "dsh" => (
            "DeepSeek Harness",
            vec!["npm", "install", "--global", "@deepseek-ai/dsh"],
            vec!["npm", "node"],
        ),
        _ => return None,
    })
}

pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(name))
        .find(|p| is_executable(p))
}

fn is_executable(p: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        p.metadata()
            .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        p.is_file()
    }
}

pub fn resolve_agent_binary(agent: &str, binary: &str, home: &Path) -> Option<PathBuf> {
    let mut names = vec![binary];
    if agent == "openclaw" && binary == "openclaw" {
        names.push("clawdbot");
    }
    if let Some(p) = names.iter().find_map(|n| which(n)) {
        return Some(p);
    }
    let fallbacks: Vec<PathBuf> = match agent {
        "codex" | "hermes" => vec![home.join(".local/bin").join(binary)],
        "claude" => vec![
            home.join(".local/bin").join(binary),
            home.join(".claude/local").join(binary),
        ],
        "opencode" => vec![home.join(".opencode/bin").join(binary)],
        "openclaw" | "dsh" => vec![home.join(".npm-global/bin").join(binary)],
        _ => vec![],
    };
    fallbacks.into_iter().find(|p| is_executable(p))
}

pub fn ensure_installed(agent: &str, home: &Path, assume_yes: bool, force: bool) -> Result<PathBuf> {
    if !force {
        if let Some(p) = resolve_agent_binary(agent, agent, home) {
            return Ok(p);
        }
    }
    let (name, argv, deps) =
        installer(agent).with_context(|| format!("{agent} was not found on PATH"))?;
    let missing: Vec<&str> = deps.iter().filter(|d| which(d).is_none()).copied().collect();
    if !missing.is_empty() {
        bail!(
            "{name} is not installed and required tools are missing:\n  {}\nInstall them, then re-run `carbon launch {agent}`",
            missing.join("\n  ")
        );
    }
    eprintln!(
        "{name} is {}installed.\nInstall command: {}",
        if force { "already " } else { "not " },
        argv.join(" ")
    );
    let verb = if force { "Reinstall" } else { "Install" };
    if !confirm(&format!("{verb} {name} now?"), assume_yes) {
        bail!("{agent} installation cancelled");
    }
    let mut cmd = std::process::Command::new(argv[0]);
    cmd.args(&argv[1..]);
    if agent == "codex" {
        cmd.env("CODEX_NON_INTERACTIVE", "1");
    }
    if !cmd.status()?.success() {
        bail!("failed to install {agent}");
    }
    resolve_agent_binary(agent, agent, home).with_context(|| {
        format!("{agent} was installed but was not found (you may need to restart your shell)")
    })
}

pub fn print_plan(ctx: &LaunchContext, spec: &CommandSpec) {
    println!("carbon server: {}", ctx.server.origin);
    println!("OpenAI base URL: {}", ctx.server.openai_base_url);
    println!("Model: {}", ctx.model.model_id);
    println!("Context window: {}", ctx.window());
    println!("Input modalities: {}", ctx.model.input_modalities.join(", "));
    println!("Command: {}", spec.argv.join(" "));
    if !spec.env.is_empty() {
        println!("Environment:");
        let mut e = spec.env.clone();
        e.sort();
        for (k, v) in e {
            println!("  {k}={v}");
        }
    }
    if !spec.unset_env.is_empty() {
        println!("Unset environment:");
        for k in &spec.unset_env {
            println!("  {k}");
        }
    }
}

pub fn run(spec: &CommandSpec) -> Result<i32> {
    let mut cmd = std::process::Command::new(&spec.argv[0]);
    cmd.args(&spec.argv[1..]);
    for k in &spec.unset_env {
        cmd.env_remove(k);
    }
    for (k, v) in &spec.env {
        cmd.env(k, v);
    }
    let status = cmd
        .status()
        .with_context(|| format!("starting {}", spec.argv[0]))?;
    Ok(status.code().unwrap_or(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(home: &Path, dry: bool, extra: &[&str]) -> LaunchContext {
        LaunchContext {
            server: resolve_server_url(Some("0.0.0.0:1919/v1/")).unwrap(),
            model: ServedModel {
                model_id: "qwen".into(),
                models: vec!["qwen".into(), "other".into()],
                context_length: Some(40_000),
                input_modalities: vec!["text".into()],
            },
            extra_args: strings(extra),
            dry_run: dry,
            assume_yes: true,
            home: home.to_path_buf(),
        }
    }

    #[test]
    fn server_urls() {
        let s = resolve_server_url(Some("0.0.0.0:1919/v1/")).unwrap();
        assert_eq!(s.origin, "http://127.0.0.1:1919");
        assert_eq!(s.openai_base_url, "http://127.0.0.1:1919/v1");
        assert_eq!(resolve_server_url(Some("https://h:8")).unwrap().origin, "https://h:8");
        assert!(resolve_server_url(Some("http://h/api")).is_err());
    }

    #[test]
    fn claude_env() {
        let d = tempfile::tempdir().unwrap();
        let spec = prepare("claude", &ctx(d.path(), true, &["-p", "hi"])).unwrap();
        assert_eq!(spec.argv, strings(&["claude", "--model", "qwen", "-p", "hi"]));
        let get = |k: &str| spec.env.iter().find(|e| e.0 == k).map(|e| e.1.clone());
        assert_eq!(get("ANTHROPIC_BASE_URL").as_deref(), Some("http://127.0.0.1:1919"));
        assert_eq!(get("CLAUDE_CODE_MAX_CONTEXT_TOKENS").as_deref(), Some("40000"));
        assert_eq!(get("CLAUDE_CODE_MAX_OUTPUT_TOKENS").as_deref(), Some("10000"));
        assert_eq!(get("ANTHROPIC_API_KEY").as_deref(), Some(""));
    }

    #[test]
    fn codex_profile_and_migration() {
        let d = tempfile::tempdir().unwrap();
        let cfg = d.path().join(".codex/config.toml");
        std::fs::create_dir_all(cfg.parent().unwrap()).unwrap();
        std::fs::write(
            &cfg,
            "profile = \"carbon-launch\"\nmodel = \"x\"\n\n[profiles.carbon-launch]\nmodel = \"q\"\n\n[other]\na = 1\n",
        )
        .unwrap();
        let spec = prepare("codex", &ctx(d.path(), false, &[])).unwrap();
        assert_eq!(&spec.argv[..3], &strings(&["codex", "--profile", "carbon-launch"]));
        assert!(spec.unset_env.contains(&"OPENAI_API_KEY".to_string()));
        let migrated = std::fs::read_to_string(&cfg).unwrap();
        assert_eq!(migrated, "model = \"x\"\n\n[other]\na = 1\n");
        let catalog: Value = serde_json::from_str(
            &std::fs::read_to_string(d.path().join(".codex/carbon-model.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(catalog["models"].as_array().unwrap().len(), 2);
        assert_eq!(catalog["models"][0]["context_window"], 40000);
        let profile =
            std::fs::read_to_string(d.path().join(".codex/carbon-launch.config.toml")).unwrap();
        assert!(profile.contains("wire_api = \"responses\""));
        // a backup of the old config was kept
        assert!(std::fs::read_dir(d.path().join(".codex"))
            .unwrap()
            .any(|e| e.unwrap().file_name().to_string_lossy().ends_with(".bak")));
    }

    #[test]
    fn json_and_yaml_agents() {
        let d = tempfile::tempdir().unwrap();
        let c = ctx(d.path(), false, &[]);
        let spec = prepare("opencode", &c).unwrap();
        let conf: Value = serde_json::from_str(&spec.env[0].1).unwrap();
        assert_eq!(conf["model"], "carbon/qwen");
        assert_eq!(conf["provider"]["carbon"]["models"]["qwen"]["limit"]["context"], 40000);
        let state: Value = serde_json::from_str(
            &std::fs::read_to_string(d.path().join(".local/state/opencode/model.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(state["recent"][0]["modelID"], "qwen");

        let oc = d.path().join(".openclaw/openclaw.json");
        std::fs::create_dir_all(oc.parent().unwrap()).unwrap();
        std::fs::write(&oc, r#"{"keep": 1, "models": {"providers": {"carbon": {"models": [{"id": "qwen", "extra": true}]}}}}"#).unwrap();
        prepare("openclaw", &c).unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&oc).unwrap()).unwrap();
        assert_eq!(v["keep"], 1);
        assert_eq!(v["models"]["providers"]["carbon"]["models"][0]["extra"], true);
        assert_eq!(v["agents"]["defaults"]["model"]["primary"], "carbon/qwen");

        let hy = d.path().join(".hermes/config.yaml");
        std::fs::create_dir_all(hy.parent().unwrap()).unwrap();
        std::fs::write(&hy, "other: 5\nmodel:\n  temperature: 0.3\n").unwrap();
        assert_eq!(prepare("hermes", &c).unwrap().argv, strings(&["hermes", "chat"]));
        let y: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(&hy).unwrap()).unwrap();
        assert_eq!(y["other"], serde_yaml::Value::from(5));
        assert_eq!(y["model"]["temperature"], serde_yaml::Value::from(0.3));
        assert_eq!(y["model"]["base_url"], serde_yaml::Value::from("http://127.0.0.1:1919/v1"));

        let spec = prepare("dsh", &ctx(d.path(), false, &["--profile", "cli", "x"])).unwrap();
        assert_eq!(spec.argv[..3], strings(&["dsh", "--profile", "cli"]));
        assert_eq!(spec.argv.last().unwrap(), "x");
    }
}
