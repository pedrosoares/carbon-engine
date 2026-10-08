// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! `carbon shell`: an interactive terminal chat over a server's `/v1/chat/completions`
//! (so it gets the same prompt rendering, sampling defaults and reasoning split as every other
//! client). Reasoning streams dimmed, Ctrl-C cancels a turn, `/think`, `/cache`, `/reset`
//! and `/exit` control the session.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

#[derive(Debug, PartialEq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
    Done {
        finish: Option<String>,
        prompt: u64,
        completion: u64,
    },
    Error(String),
}

/// Parse one SSE `data:` payload of a chat completion stream.
pub fn parse_chunk(data: &str) -> Vec<Delta> {
    if data == "[DONE]" {
        return vec![];
    }
    let Ok(v) = serde_json::from_str::<Value>(data) else {
        return vec![];
    };
    if let Some(e) = v.get("error") {
        return vec![Delta::Error(
            e["message"].as_str().unwrap_or("server error").to_string(),
        )];
    }
    let mut out = vec![];
    if let Some(c) = v["choices"].get(0) {
        let d = &c["delta"];
        for k in ["reasoning_content", "reasoning"] {
            if let Some(r) = d[k].as_str().filter(|r| !r.is_empty()) {
                out.push(Delta::Reasoning(r.into()));
            }
        }
        if let Some(t) = d["content"].as_str().filter(|t| !t.is_empty()) {
            out.push(Delta::Content(t.into()));
        }
        if let Some(f) = c["finish_reason"].as_str() {
            out.push(Delta::Done {
                finish: Some(f.into()),
                prompt: 0,
                completion: 0,
            });
        }
    }
    if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
        out.push(Delta::Done {
            finish: None,
            prompt: u["prompt_tokens"].as_u64().unwrap_or(0),
            completion: u["completion_tokens"].as_u64().unwrap_or(0),
        });
    }
    out
}

/// `1k` = 1024, `2m` = 2 MiB worth of units.
pub fn parse_count(s: &str) -> Option<usize> {
    let s = s.trim().to_ascii_lowercase();
    let (num, mul) = match s.chars().last()? {
        'k' => (&s[..s.len() - 1], 1024usize),
        'm' => (&s[..s.len() - 1], 1 << 20),
        _ => (s.as_str(), 1),
    };
    let v: f64 = num.parse().ok()?;
    (v >= 0.0).then_some((v * mul as f64) as usize)
}

/// What the `/think` setting adds to a request.
#[derive(Clone, Debug, PartialEq)]
pub enum Think {
    /// The template default.
    Default,
    On,
    Off,
    Effort(String),
}

impl Think {
    fn apply(&self, req: &mut Value) {
        match self {
            Think::Default => {}
            Think::On => req["chat_template_kwargs"] = json!({"enable_thinking": true}),
            Think::Off => req["chat_template_kwargs"] = json!({"enable_thinking": false}),
            Think::Effort(e) => req["reasoning_effort"] = json!(e),
        }
    }

    fn label(&self) -> String {
        match self {
            Think::Default => "default".into(),
            Think::On => "on".into(),
            Think::Off => "off".into(),
            Think::Effort(e) => e.clone(),
        }
    }
}

/// Resolve a `/think` argument; Err carries the usage line.
pub fn think_command(arg: &str, current: &Think, gears: &[String]) -> Result<Think, String> {
    let usage = || {
        let mut opts = vec!["on", "off", "toggle", "status"];
        opts.extend(gears.iter().map(|g| g.as_str()));
        format!("Usage: /think [{}]", opts.join("|"))
    };
    Ok(match arg.trim() {
        "" | "status" => current.clone(),
        "on" => Think::On,
        "off" | "none" => Think::Off,
        "toggle" => match current {
            Think::Off => Think::On,
            _ => Think::Off,
        },
        g if gears.iter().any(|x| x == g) => Think::Effort(g.into()),
        _ => return Err(usage()),
    })
}

struct Shell {
    base: String,
    model: String,
    gears: Vec<String>,
    history: Vec<Value>,
    think: Think,
    max_tokens: u64,
    sampling: Value,
    color: bool,
}

fn env_num(keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|k| std::env::var(k).ok())
        .and_then(|v| v.trim().parse().ok())
}

impl Shell {
    fn style(&self, code: &'static str) -> &'static str {
        if self.color {
            code
        } else {
            ""
        }
    }

    fn help(&self) -> String {
        let mut think = "/think [on|off|toggle|status".to_string();
        for g in &self.gears {
            think.push('|');
            think.push_str(g);
        }
        think.push(']');
        [
            ("/help", "show this message"),
            (think.as_str(), "thinking on/off or a reasoning effort"),
            ("/cache [status | kv N | moe N]", "show or resize the KV / expert caches (N accepts k / m)"),
            ("/reset", "clear the conversation history"),
            ("/exit", "quit (Ctrl-D also works)"),
        ]
        .iter()
        .map(|(c, d)| format!("  {c:<44} {d}"))
        .collect::<Vec<_>>()
        .join("\n")
    }

    fn cache(&self, args: &[&str]) -> Result<String> {
        if args.is_empty() || args == ["status"] {
            let v: Value = ureq::get(&format!("{}/v1/cache/status", self.base))
                .call()?
                .into_json()?;
            return Ok(serde_json::to_string_pretty(&v)?);
        }
        let mut body = json!({});
        for pair in args.chunks(2) {
            let n = pair
                .get(1)
                .and_then(|n| parse_count(n))
                .context("Usage: /cache [status | kv N | moe N]")?;
            match pair[0] {
                "kv" => body["kv_tokens"] = json!(n),
                "moe" => body["moe_cache_size"] = json!(n),
                o => bail!("unknown cache pool '{o}' (kv | moe)"),
            }
        }
        let v: Value = ureq::post(&format!("{}/v1/cache/rebuild", self.base))
            .send_json(body)?
            .into_json()?;
        Ok(format!("rebuilt:\n{}", serde_json::to_string_pretty(&v)?))
    }

    /// Stream one assistant turn; Ctrl-C (`cancel`) stops it and drops the reply.
    fn turn(&mut self, text: &str, cancel: &Arc<AtomicBool>) -> Result<()> {
        self.history.push(json!({"role": "user", "content": text}));
        let mut req = json!({
            "model": self.model, "messages": self.history, "stream": true,
            "max_tokens": self.max_tokens, "stream_options": {"include_usage": true},
        });
        for (k, v) in self.sampling.as_object().unwrap() {
            req[k] = v.clone();
        }
        self.think.apply(&mut req);
        cancel.store(false, Ordering::SeqCst);
        let t0 = Instant::now();
        let resp = match ureq::post(&format!("{}/v1/chat/completions", self.base))
            .timeout(std::time::Duration::from_secs(900))
            .send_json(req)
        {
            Ok(r) => r,
            Err(ureq::Error::Status(code, r)) => {
                self.history.pop();
                let v: Value = r.into_json().unwrap_or(Value::Null);
                bail!(
                    "server refused the request ({code}): {}",
                    v["error"]["message"].as_str().unwrap_or("")
                );
            }
            Err(e) => {
                self.history.pop();
                return Err(e.into());
            }
        };
        let mut out = std::io::stdout();
        let mut answer = String::new();
        let mut in_reasoning = false;
        let (mut first, mut prompt, mut completion, mut finish) = (None, 0, 0, None);
        for line in BufReader::new(resp.into_reader()).lines() {
            if cancel.load(Ordering::SeqCst) {
                break; // dropping the connection aborts the request on the server
            }
            let line = line?;
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            for d in parse_chunk(data) {
                match d {
                    Delta::Reasoning(r) => {
                        first.get_or_insert_with(Instant::now);
                        if !in_reasoning {
                            write!(out, "{}", self.style(DIM))?;
                            in_reasoning = true;
                        }
                        write!(out, "{r}")?;
                    }
                    Delta::Content(c) => {
                        first.get_or_insert_with(Instant::now);
                        if in_reasoning {
                            write!(out, "{}\n\n", self.style(RESET))?;
                            in_reasoning = false;
                        }
                        write!(out, "{c}")?;
                        answer.push_str(&c);
                    }
                    Delta::Done {
                        finish: f,
                        prompt: p,
                        completion: c,
                    } => {
                        if f.is_some() {
                            finish = f;
                        }
                        if p > 0 || c > 0 {
                            (prompt, completion) = (p, c);
                        }
                    }
                    Delta::Error(e) => {
                        writeln!(out, "{}", self.style(RESET))?;
                        self.history.pop();
                        bail!("{e}");
                    }
                }
                out.flush()?;
            }
        }
        write!(out, "{}", self.style(RESET))?;
        if cancel.load(Ordering::SeqCst) {
            writeln!(out, "\n[cancelled]")?;
            self.history.pop();
            return Ok(());
        }
        let decode = first
            .map(|f| f.elapsed().as_secs_f64())
            .unwrap_or(0.0)
            .max(1e-9);
        let ttft = first
            .map(|f| f.duration_since(t0).as_secs_f64())
            .unwrap_or(0.0);
        writeln!(
            out,
            "\n{}[{}] ↓{prompt} ↑{completion}  ttft {ttft:.2}s  {:.1} tok/s  think {}{}",
            self.style(DIM),
            finish.as_deref().unwrap_or("done"),
            completion.saturating_sub(1) as f64 / decode,
            self.think.label(),
            self.style(RESET),
        )?;
        self.history
            .push(json!({"role": "assistant", "content": answer}));
        Ok(())
    }
}

/// Attach to the server at `base` and run the chat loop until /exit or end of input.
pub fn run(base: &str) -> Result<()> {
    let base = base.trim_end_matches('/').to_string();
    let models: Value = ureq::get(&format!("{base}/v1/models"))
        .call()
        .with_context(|| format!("cannot reach a carbon server at {base}"))?
        .into_json()?;
    let card = &models["data"][0];
    let model = card["id"].as_str().unwrap_or("model").to_string();
    let gears: Vec<String> = card["supported_reasoning_efforts"]
        .as_array()
        .map(|a| a.iter().filter_map(|g| g.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let mut sampling = json!({});
    for (k, keys) in [
        ("temperature", ["CARBON_SHELL_TEMPERATURE", "FREETOKEN_SHELL_TEMPERATURE"]),
        ("top_p", ["CARBON_SHELL_TOP_P", "FREETOKEN_SHELL_TOP_P"]),
        ("top_k", ["CARBON_SHELL_TOP_K", "FREETOKEN_SHELL_TOP_K"]),
    ] {
        if let Some(v) = env_num(&keys) {
            sampling[k] = json!(v);
        }
    }
    let color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let mut sh = Shell {
        base,
        model,
        gears,
        history: vec![],
        think: Think::Default,
        max_tokens: env_num(&["CARBON_SHELL_MAX_TOKENS", "FREETOKEN_SHELL_MAX_TOKENS"])
            .map(|v| v as u64)
            .unwrap_or(2048),
        sampling,
        color,
    };
    let cancel = Arc::new(AtomicBool::new(false));
    {
        // Ctrl-C cancels the running turn instead of killing the shell
        let c = cancel.clone();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        std::thread::spawn(move || {
            rt.block_on(async move {
                while tokio::signal::ctrl_c().await.is_ok() {
                    c.store(true, Ordering::SeqCst);
                }
            })
        });
    }
    println!(
        "{}carbon shell{} — {} at {} (ctx {}). /help for commands.",
        sh.style(BOLD),
        sh.style(RESET),
        sh.model,
        sh.base,
        card["context_length"].as_u64().unwrap_or(0)
    );
    let stdin = std::io::stdin();
    let interactive = stdin.is_terminal();
    let mut lines = stdin.lock().lines();
    loop {
        if interactive {
            print!("{}> {}", sh.style(BOLD), sh.style(RESET));
            std::io::stdout().flush()?;
        }
        // a trailing backslash continues the message on the next line
        let mut text = String::new();
        loop {
            match lines.next() {
                None => {
                    println!();
                    return Ok(());
                }
                Some(l) => {
                    let l = l?;
                    if let Some(s) = l.strip_suffix('\\') {
                        text.push_str(s);
                        text.push('\n');
                        if interactive {
                            print!(". ");
                            std::io::stdout().flush()?;
                        }
                        continue;
                    }
                    text.push_str(&l);
                    break;
                }
            }
        }
        cancel.store(false, Ordering::SeqCst);
        let t = text.trim();
        if t.is_empty() {
            continue;
        }
        if let Some(cmd) = t.strip_prefix('/') {
            let mut parts = cmd.split_whitespace();
            let name = parts.next().unwrap_or("");
            let args: Vec<&str> = parts.collect();
            match name {
                "exit" | "quit" | "q" => return Ok(()),
                "help" | "?" => println!("{}", sh.help()),
                "reset" => {
                    sh.history.clear();
                    println!("conversation cleared");
                }
                "think" | "thinking" => match think_command(&args.join(" "), &sh.think, &sh.gears) {
                    Ok(t) => {
                        sh.think = t;
                        println!("thinking: {}", sh.think.label());
                    }
                    Err(u) => println!("{u}"),
                },
                "cache" => match sh.cache(&args) {
                    Ok(s) => println!("{s}"),
                    Err(e) => println!("{e:#}"),
                },
                o => println!("unknown command /{o} (/help)"),
            }
            continue;
        }
        if let Err(e) = sh.turn(t, &cancel) {
            println!("error: {e:#}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks() {
        assert_eq!(
            parse_chunk(r#"{"choices":[{"delta":{"reasoning_content":"hm"},"finish_reason":null}]}"#),
            vec![Delta::Reasoning("hm".into())]
        );
        assert_eq!(
            parse_chunk(r#"{"choices":[{"delta":{},"finish_reason":"length"}]}"#),
            vec![Delta::Done { finish: Some("length".into()), prompt: 0, completion: 0 }]
        );
        assert_eq!(
            parse_chunk(r#"{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4}}"#),
            vec![Delta::Done { finish: None, prompt: 3, completion: 4 }]
        );
        assert!(parse_chunk("[DONE]").is_empty());
        assert!(matches!(parse_chunk(r#"{"error":{"message":"x"}}"#)[0], Delta::Error(_)));
    }

    #[test]
    fn commands() {
        assert_eq!(parse_count("2k"), Some(2048));
        assert_eq!(parse_count("1.5m"), Some(3 << 19));
        assert_eq!(parse_count("x"), None);
        let gears = vec!["low".to_string(), "high".to_string()];
        assert_eq!(think_command("toggle", &Think::Default, &gears), Ok(Think::Off));
        assert_eq!(think_command("toggle", &Think::Off, &gears), Ok(Think::On));
        assert_eq!(think_command("high", &Think::On, &gears), Ok(Think::Effort("high".into())));
        assert!(think_command("max", &Think::On, &gears).unwrap_err().contains("low|high"));
    }
}
