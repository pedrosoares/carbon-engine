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
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    // `separators=(item, key)` as in json.dumps
    let seps: Option<minijinja::Value> = kwargs.get("separators")?;
    kwargs.assert_all_used()?;
    let mut v = serde_json::to_value(&value)
        .map_err(|e| JinjaError::new(ErrorKind::InvalidOperation, e.to_string()))?;
    if sort_keys == Some(true) {
        sort_json_keys(&mut v);
    }
    let (item_sep, key_sep) = match &seps {
        Some(s) => {
            let parts: Vec<String> = s
                .try_iter()
                .map_err(|e| JinjaError::new(ErrorKind::InvalidOperation, e.to_string()))?
                .map(|x| x.as_str().unwrap_or_default().to_string())
                .collect();
            match parts.as_slice() {
                [a, b] => (a.clone(), b.clone()),
                _ => return Err(JinjaError::new(ErrorKind::InvalidOperation, "tojson: separators must be a pair")),
            }
        }
        None if indent.is_some() => (",".to_string(), ": ".to_string()),
        None => (", ".to_string(), ": ".to_string()),
    };
    let mut out = String::new();
    py_json(&v, indent, 0, &mut out, &item_sep, &key_sep);
    Ok(minijinja::Value::from_safe_string(out))
}

fn sort_json_keys(v: &mut Value) {
    match v {
        Value::Object(o) => {
            let mut items: Vec<(String, Value)> = std::mem::take(o).into_iter().collect();
            items.sort_by(|a, b| a.0.cmp(&b.0));
            for (k, mut x) in items {
                sort_json_keys(&mut x);
                o.insert(k, x);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(sort_json_keys),
        _ => {}
    }
}

fn py_json(v: &Value, indent: Option<usize>, depth: usize, out: &mut String, item_sep: &str, key_sep: &str) {
    let (open_nl, sep, close_nl) = match indent {
        Some(n) => (
            format!("\n{}", " ".repeat(n * (depth + 1))),
            format!("{item_sep}\n{}", " ".repeat(n * (depth + 1))),
            format!("\n{}", " ".repeat(n * depth)),
        ),
        None => (String::new(), item_sep.to_string(), String::new()),
    };
    match v {
        Value::Array(a) if !a.is_empty() => {
            out.push('[');
            out.push_str(&open_nl);
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(&sep);
                }
                py_json(x, indent, depth + 1, out, item_sep, key_sep);
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
                out.push_str(key_sep);
                py_json(x, indent, depth + 1, out, item_sep, key_sep);
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
        if let Some(file) = crate::gguf::find(dir) {
            let g = crate::gguf::Gguf::open(&file)?;
            let tokenizer: Tokenizer = serde_json::from_value(g.tokenizer_json()?)
                .map_err(|e| anyhow!("building the tokenizer of {}: {e}", file.display()))?;
            let template = match template_override {
                Some(p) => Some(
                    std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?,
                ),
                None => g.chat_template(),
            };
            return Self::new(tokenizer, template, g.special_token("bos"), g.special_token("eos"));
        }
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
            strip_generation_tags(&template.unwrap_or_else(|| CHATML_TEMPLATE.to_string())),
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

/// transformers' `{% generation %}` ... `{% endgeneration %}` only marks assistant tokens for
/// training masks (LFM2.5 templates use it): render it as `{% if true %}` ... `{% endif %}`,
/// keeping the whitespace control of each tag.
fn strip_generation_tags(template: &str) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(open) = rest.find("{%") {
        let Some(len) = rest[open..].find("%}") else { break };
        let tag = &rest[open + 2..open + len];
        let inner = tag.trim_start_matches(['-', '+']).trim_end_matches(['-', '+']).trim();
        let (lead, trail) = (&tag[..tag.len() - tag.trim_start_matches(['-', '+']).len()], &tag[tag.trim_end_matches(['-', '+']).len()..]);
        out.push_str(&rest[..open]);
        match inner {
            "generation" => out.push_str(&format!("{{%{lead} if true {trail}%}}")),
            "endgeneration" => out.push_str(&format!("{{%{lead} endif {trail}%}}")),
            _ => out.push_str(&rest[open..open + len + 2]),
        }
        rest = &rest[open + len + 2..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_tags_render_as_plain_blocks() {
        let t = "a {%- generation -%} b {% endgeneration %}{{ x }}{%+ generation %}c{% endgeneration -%}";
        assert_eq!(
            strip_generation_tags(t),
            "a {%- if true -%} b {% endif %}{{ x }}{%+ if true %}c{% endif -%}"
        );
    }
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

/// Inkling 2 writes its turn structure with special tokens: `<|message_model|>` opens a segment,
/// `<|content_thinking|>` / `<|content_text|>` / `<|content_invoke_tool_json|>` say what it holds
/// (a tool segment's name sits between the two), `<|end_message|>` closes it. The detokenizer
/// skips special tokens, so this maps them to the tags the output parsers read: reasoning
/// becomes `<think>..</think>`, a tool call `<tool_call>{json}</tool_call>`, text stays as is.
#[derive(Clone, Debug)]
pub struct MarkerRewrite {
    open: u32,
    thinking: u32,
    text: u32,
    tool: u32,
    end: u32,
    seg: Segment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Segment {
    Idle,
    /// After `<|message_model|>`: a name may follow before the content marker.
    Header,
    Thinking,
    Text,
    Tool,
}

/// What [`MarkerRewrite::token`] makes of a token.
#[derive(Debug, PartialEq, Eq)]
pub enum Marked {
    /// An ordinary token: detokenize it (or drop it while in a segment header).
    Text { emit: bool },
    /// A marker: emit this text instead.
    Marker(&'static str),
}

impl MarkerRewrite {
    /// The rewriter for a vocabulary that has Inkling's markers.
    pub fn for_vocab(tok: &TokenizerBundle) -> Option<Self> {
        let id = |t: &str| tok.token_id(t);
        Some(Self {
            open: id("<|message_model|>")?,
            thinking: id("<|content_thinking|>")?,
            text: id("<|content_text|>")?,
            tool: id("<|content_invoke_tool_json|>")?,
            end: id("<|end_message|>")?,
            seg: Segment::Idle,
        })
    }

    pub fn token(&mut self, t: u32) -> Marked {
        let (next, out) = if t == self.open {
            (Segment::Header, "")
        } else if t == self.thinking {
            (Segment::Thinking, "<think>")
        } else if t == self.text {
            (Segment::Text, "")
        } else if t == self.tool {
            (Segment::Tool, "<tool_call>")
        } else if t == self.end {
            let close = match self.seg {
                Segment::Thinking => "</think>",
                Segment::Tool => "</tool_call>",
                _ => "",
            };
            (Segment::Idle, close)
        } else {
            // the tool name of a segment header is repeated inside the call's JSON
            return Marked::Text { emit: self.seg != Segment::Header };
        };
        self.seg = next;
        Marked::Marker(out)
    }
}

#[cfg(test)]
mod marker_tests {
    use super::*;

    #[test]
    fn segments_map_to_parser_tags() {
        let mut m = MarkerRewrite { open: 1, thinking: 2, text: 3, tool: 4, end: 5, seg: Segment::Idle };
        let seq = [1, 2, 100, 5, 1, 3, 101, 5, 1, 102, 4, 103, 5];
        let out: Vec<Marked> = seq.iter().map(|&t| m.token(t)).collect();
        use Marked::*;
        assert_eq!(
            out,
            vec![
                Marker(""), Marker("<think>"), Text { emit: true }, Marker("</think>"),
                Marker(""), Marker(""), Text { emit: true }, Marker(""),
                Marker(""), Text { emit: false }, Marker("<tool_call>"), Text { emit: true }, Marker("</tool_call>"),
            ]
        );
    }
}
