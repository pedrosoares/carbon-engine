// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Tokenizer, chat-template rendering (Jinja via minijinja) and incremental detokenization.

use anyhow::{anyhow, Context, Result};
use minijinja::{Environment, Error as JinjaError, ErrorKind};
use serde_json::Value;
use std::path::Path;
use tokenizers::Tokenizer;

const CHATML_TEMPLATE: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + (message['content'] or '') + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

pub struct TokenizerBundle {
    pub tokenizer: Tokenizer,
    env: Environment<'static>,
    pub has_template: bool,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

fn token_str(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("content").and_then(|c| c.as_str()).map(String::from),
        _ => None,
    }
}

/// `strftime` subset used by chat templates (`%Y %m %d %b %B %H %M %S`), in UTC.
fn strftime_now(fmt: String) -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    // civil-from-days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let month = MONTHS[(m - 1) as usize];
    let mut out = String::new();
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('b') => out.push_str(&month[..3]),
            Some('B') => out.push_str(month),
            Some('H') => out.push_str(&format!("{:02}", rem / 3600)),
            Some('M') => out.push_str(&format!("{:02}", rem % 3600 / 60)),
            Some('S') => out.push_str(&format!("{:02}", rem % 60)),
            Some(o) => {
                out.push('%');
                out.push(o);
            }
            None => out.push('%'),
        }
    }
    out
}

/// `tojson` with Python `json.dumps` formatting (", " / ": " separators, key order kept,
/// non-ASCII kept, no HTML escaping), which HF chat templates are written against.
fn py_tojson(
    value: minijinja::Value,
    kwargs: minijinja::value::Kwargs,
) -> Result<minijinja::Value, JinjaError> {
    let indent: Option<usize> = kwargs.get("indent")?;
    let _: Option<bool> = kwargs.get("ensure_ascii")?;
    let _: Option<bool> = kwargs.get("sort_keys")?;
    kwargs.assert_all_used()?;
    let v = serde_json::to_value(&value)
        .map_err(|e| JinjaError::new(ErrorKind::InvalidOperation, e.to_string()))?;
    let mut out = String::new();
    py_json(&v, indent, 0, &mut out);
    Ok(minijinja::Value::from_safe_string(out))
}

fn py_json(v: &Value, indent: Option<usize>, depth: usize, out: &mut String) {
    let (open_nl, sep, close_nl) = match indent {
        Some(n) => (
            format!("\n{}", " ".repeat(n * (depth + 1))),
            format!(",\n{}", " ".repeat(n * (depth + 1))),
            format!("\n{}", " ".repeat(n * depth)),
        ),
        None => (String::new(), ", ".to_string(), String::new()),
    };
    match v {
        Value::Array(a) if !a.is_empty() => {
            out.push('[');
            out.push_str(&open_nl);
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(&sep);
                }
                py_json(x, indent, depth + 1, out);
            }
            out.push_str(&close_nl);
            out.push(']');
        }
        Value::Object(o) if !o.is_empty() => {
            out.push('{');
            out.push_str(&open_nl);
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(&sep);
                }
                out.push_str(&Value::String(k.clone()).to_string());
                out.push_str(": ");
                py_json(x, indent, depth + 1, out);
            }
            out.push_str(&close_nl);
            out.push('}');
        }
        // Python prints integral floats as `1.0`; serde_json does too
        other => out.push_str(&other.to_string()),
    }
}

impl TokenizerBundle {
    pub fn load(dir: &Path, template_override: Option<&Path>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading {}: {e}", dir.join("tokenizer.json").display()))?;
        let cfg: Value = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(Value::Null);
        let template = match template_override {
            Some(p) => Some(
                std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?,
            ),
            None => std::fs::read_to_string(dir.join("chat_template.jinja"))
                .ok()
                .or_else(|| match cfg.get("chat_template") {
                    Some(Value::String(s)) => Some(s.clone()),
                    Some(Value::Array(list)) => list
                        .iter()
                        .find(|t| t.get("name").and_then(|n| n.as_str()) == Some("default"))
                        .or_else(|| list.first())
                        .and_then(|t| t.get("template"))
                        .and_then(|t| t.as_str())
                        .map(String::from),
                    _ => None,
                }),
        };
        Self::new(
            tokenizer,
            template,
            token_str(cfg.get("bos_token")),
            token_str(cfg.get("eos_token")),
        )
    }

    pub fn new(
        tokenizer: Tokenizer,
        template: Option<String>,
        bos_token: Option<String>,
        eos_token: Option<String>,
    ) -> Result<Self> {
        let has_template = template.is_some();
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<String, JinjaError> {
                Err(JinjaError::new(ErrorKind::InvalidOperation, msg))
            },
        );
        env.add_function("strftime_now", strftime_now);
        env.add_filter("tojson", py_tojson);
        env.add_template_owned(
            "chat",
            template.unwrap_or_else(|| CHATML_TEMPLATE.to_string()),
        )
        .map_err(|e| anyhow!("invalid chat template: {e}"))?;
        Ok(Self {
            tokenizer,
            env,
            has_template,
            bos_token,
            eos_token,
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        self.tokenizer.decode(ids, skip_special).unwrap_or_default()
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Render the chat template. `messages`/`tools` follow the OpenAI chat format;
    /// `extra` holds `chat_template_kwargs` (e.g. `enable_thinking`).
    pub fn apply_chat_template(
        &self,
        messages: &[Value],
        tools: Option<&Value>,
        add_generation_prompt: bool,
        extra: &serde_json::Map<String, Value>,
    ) -> Result<String> {
        let mut ctx = serde_json::Map::new();
        ctx.insert("messages".into(), Value::Array(messages.to_vec()));
        if let Some(t) = tools.filter(|t| t.as_array().map(|a| !a.is_empty()).unwrap_or(false)) {
            ctx.insert("tools".into(), t.clone());
        }
        ctx.insert(
            "add_generation_prompt".into(),
            Value::Bool(add_generation_prompt),
        );
        ctx.insert(
            "bos_token".into(),
            Value::String(self.bos_token.clone().unwrap_or_default()),
        );
        ctx.insert(
            "eos_token".into(),
            Value::String(self.eos_token.clone().unwrap_or_default()),
        );
        for (k, v) in extra {
            ctx.insert(k.clone(), v.clone());
        }
        let tmpl = self.env.get_template("chat")?;
        tmpl.render(minijinja::Value::from_serialize(Value::Object(ctx)))
            .map_err(|e| {
                let mut msg = format!("chat template error: {e}");
                let mut src = std::error::Error::source(&e);
                while let Some(s) = src {
                    msg.push_str(&format!(": {s}"));
                    src = s.source();
                }
                anyhow!(msg)
            })
    }

    /// Tokenize a rendered chat prompt. The template already emits special tokens such as BOS,
    /// so they are only added by the tokenizer when the template did not produce them.
    pub fn encode_prompt(&self, text: &str) -> Result<Vec<u32>> {
        let starts_with_bos = self
            .bos_token
            .as_deref()
            .map(|b| !b.is_empty() && text.starts_with(b))
            .unwrap_or(false);
        self.encode(text, !starts_with_bos && !self.has_template)
    }
}

/// Incremental detokenizer (prefix/read offsets, as in HF TGI / vLLM): emits text only once
/// the bytes decoded so far form complete characters.
pub struct DecodeStream {
    ids: Vec<u32>,
    prefix_offset: usize,
    read_offset: usize,
    skip_special: bool,
}

impl DecodeStream {
    pub fn new(skip_special: bool) -> Self {
        Self {
            ids: vec![],
            prefix_offset: 0,
            read_offset: 0,
            skip_special,
        }
    }

    pub fn step(&mut self, tok: &TokenizerBundle, id: u32) -> Option<String> {
        self.ids.push(id);
        let prefix = tok.decode(
            &self.ids[self.prefix_offset..self.read_offset],
            self.skip_special,
        );
        let full = tok.decode(&self.ids[self.prefix_offset..], self.skip_special);
        if full.len() > prefix.len() && !full.ends_with('\u{FFFD}') {
            let delta = full
                .get(prefix.len()..)
                .map(String::from)
                .unwrap_or_default();
            self.prefix_offset = self.read_offset;
            self.read_offset = self.ids.len();
            Some(delta)
        } else {
            None
        }
    }

    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
}

/// Stop-string matcher for streamed text. Holds back any suffix that could still grow
/// into a stop string so that the stop string itself is never emitted.
pub struct StopChecker {
    stops: Vec<String>,
    held: String,
}

impl StopChecker {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            held: String::new(),
        }
    }

    /// Returns `(text safe to emit, matched stop string)`.
    pub fn push(&mut self, text: &str) -> (String, Option<String>) {
        if self.stops.is_empty() {
            return (text.to_string(), None);
        }
        self.held.push_str(text);
        let first = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()).map(|i| (i, s)))
            .min_by_key(|(i, _)| *i);
        if let Some((i, s)) = first {
            let out = self.held[..i].to_string();
            let s = s.clone();
            self.held.clear();
            return (out, Some(s));
        }
        let mut keep = 0;
        for s in &self.stops {
            for k in (1..s.len()).rev() {
                if k <= self.held.len()
                    && self.held.is_char_boundary(self.held.len() - k)
                    && s.is_char_boundary(k)
                    && self.held.ends_with(&s[..k])
                {
                    keep = keep.max(k);
                    break;
                }
            }
        }
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_string();
        self.held.drain(..cut);
        (out, None)
    }

    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn stop_checker_holds_partial() {
        let mut s = StopChecker::new(vec!["</end>".into()]);
        assert_eq!(s.push("hello </"), ("hello ".to_string(), None));
        assert_eq!(s.push("en"), (String::new(), None));
        assert_eq!(s.push("d> tail"), (String::new(), Some("</end>".into())));
        let mut s = StopChecker::new(vec!["xyz".into()]);
        assert_eq!(s.push("abc x"), ("abc ".to_string(), None));
        assert_eq!(s.push("a"), ("xa".to_string(), None));
        assert_eq!(s.flush(), "");
    }

    #[test]
    fn tojson_matches_python() {
        let tok =
            tokenizers::Tokenizer::from_str(&crate::testing::tokenizer_json().to_string()).unwrap();
        let t = "{{ x | tojson }}|{{ x | tojson(indent=2) }}";
        let b = TokenizerBundle::new(tok, Some(t.into()), None, None).unwrap();
        let x = serde_json::json!({"z": 1, "a": [1, "é<"], "e": {}});
        let mut extra = serde_json::Map::new();
        extra.insert("x".into(), x);
        let out = b.apply_chat_template(&[], None, false, &extra).unwrap();
        assert_eq!(out, "{\"z\": 1, \"a\": [1, \"é<\"], \"e\": {}}|{\n  \"z\": 1,\n  \"a\": [\n    1,\n    \"é<\"\n  ],\n  \"e\": {}\n}");
    }

    #[test]
    fn strftime_format() {
        let s = strftime_now("%d %b %Y".into());
        assert_eq!(s.len(), 11, "{s}");
    }
}
