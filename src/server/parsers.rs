// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Streaming output parsers: splits generated text into reasoning (`<think>...</think>`),
//! visible content and hermes-style tool calls (`<tool_call>{json}</tool_call>`).

use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    Reasoning(String),
    Content(String),
    ToolCall { name: String, arguments: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Content,
    Reasoning,
    Tool,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";

pub struct OutputParser {
    reasoning: bool,
    tools: bool,
    mode: Mode,
    buf: String,
    tool_buf: String,
    seen_content: bool,
    /// Drop newlines right after `</think>` (template formatting), even across chunks.
    trim_newlines: bool,
}

/// Length of the longest suffix of `s` that is a proper prefix of `tag`.
fn partial_suffix(s: &str, tag: &str) -> usize {
    (1..tag.len().min(s.len() + 1))
        .rev()
        .find(|&k| s.is_char_boundary(s.len() - k) && s.ends_with(&tag[..k]))
        .unwrap_or(0)
}

/// Parse the body of a `<tool_call>` block: hermes JSON (`{"name":..,"arguments":..}`) or the
/// Qwen3-Coder / Qwen3.5+ XML form (`<function=name><parameter=k>v</parameter></function>`).
pub fn parse_tool_call(raw: &str) -> Option<(String, String)> {
    if let Ok(v) = serde_json::from_str::<Value>(raw.trim()) {
        let name = v.get("name")?.as_str()?.to_string();
        let args = match v.get("arguments").or_else(|| v.get("parameters")) {
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => "{}".into(),
        };
        return Some((name, args));
    }
    parse_xml_tool_call(raw)
}

fn parse_xml_tool_call(raw: &str) -> Option<(String, String)> {
    let start = raw.find("<function=")? + "<function=".len();
    let name_end = start + raw[start..].find('>')?;
    let name = raw[start..name_end].trim().to_string();
    let body = &raw[name_end + 1..];
    let body = &body[..body.find("</function>").unwrap_or(body.len())];
    let mut args = serde_json::Map::new();
    let mut rest = body;
    while let Some(p) = rest.find("<parameter=") {
        let k0 = p + "<parameter=".len();
        let k1 = k0 + rest[k0..].find('>')?;
        let key = rest[k0..k1].trim().to_string();
        let after = &rest[k1 + 1..];
        let end = after.find("</parameter>").unwrap_or(after.len());
        let mut val = &after[..end];
        val = val.strip_prefix('\n').unwrap_or(val);
        val = val.strip_suffix('\n').unwrap_or(val);
        // numbers, booleans, objects and arrays keep their JSON type; anything else is a string
        let parsed = match serde_json::from_str::<Value>(val) {
            Ok(v) if !v.is_string() => v,
            _ => Value::String(val.to_string()),
        };
        args.insert(key, parsed);
        rest = &after[(end + "</parameter>".len()).min(after.len())..];
    }
    if name.is_empty() {
        return None;
    }
    Some((name, Value::Object(args).to_string()))
}

impl OutputParser {
    /// `start_in_reasoning`: the prompt already opened a `<think>` block.
    pub fn new(reasoning: bool, tools: bool, start_in_reasoning: bool) -> Self {
        Self {
            reasoning,
            tools,
            mode: if reasoning && start_in_reasoning {
                Mode::Reasoning
            } else {
                Mode::Content
            },
            buf: String::new(),
            tool_buf: String::new(),
            seen_content: false,
            trim_newlines: false,
        }
    }

    pub fn push(&mut self, text: &str) -> Vec<Piece> {
        self.buf.push_str(text);
        let mut out = vec![];
        loop {
            match self.mode {
                Mode::Reasoning => {
                    if let Some(i) = self.buf.find(THINK_CLOSE) {
                        let r: String = self.buf.drain(..i).collect();
                        self.buf.drain(..THINK_CLOSE.len());
                        push_piece(&mut out, Piece::Reasoning(r));
                        self.mode = Mode::Content;
                        self.trim_newlines = true;
                        self.seen_content = true;
                        continue;
                    }
                    let keep = partial_suffix(&self.buf, THINK_CLOSE);
                    let r: String = self.buf.drain(..self.buf.len() - keep).collect();
                    push_piece(&mut out, Piece::Reasoning(r));
                    break;
                }
                Mode::Tool => {
                    if let Some(i) = self.buf.find(TOOL_CLOSE) {
                        self.tool_buf.extend(self.buf.drain(..i));
                        self.buf.drain(..TOOL_CLOSE.len());
                        let raw = std::mem::take(&mut self.tool_buf);
                        match parse_tool_call(&raw) {
                            Some((name, arguments)) => {
                                out.push(Piece::ToolCall { name, arguments })
                            }
                            None => push_piece(
                                &mut out,
                                Piece::Content(format!("{TOOL_OPEN}{raw}{TOOL_CLOSE}")),
                            ),
                        }
                        self.mode = Mode::Content;
                        continue;
                    }
                    let keep = partial_suffix(&self.buf, TOOL_CLOSE);
                    let cut = self.buf.len() - keep;
                    self.tool_buf.extend(self.buf.drain(..cut));
                    break;
                }
                Mode::Content => {
                    if self.trim_newlines {
                        let trimmed = self.buf.trim_start_matches('\n').len();
                        self.buf.drain(..self.buf.len() - trimmed);
                        if self.buf.is_empty() {
                            break;
                        }
                        self.trim_newlines = false;
                    }
                    if self.reasoning && !self.seen_content {
                        let t = self.buf.trim_start();
                        if t.starts_with(THINK_OPEN) {
                            let skip = self.buf.len() - t.len() + THINK_OPEN.len();
                            self.buf.drain(..skip);
                            self.mode = Mode::Reasoning;
                            continue;
                        }
                        if THINK_OPEN.starts_with(t) {
                            break; // could still become <think>
                        }
                    }
                    let tool_at = if self.tools {
                        self.buf.find(TOOL_OPEN)
                    } else {
                        None
                    };
                    if let Some(i) = tool_at {
                        let c: String = self.buf.drain(..i).collect();
                        self.emit_content(&mut out, c);
                        self.buf.drain(..TOOL_OPEN.len());
                        self.mode = Mode::Tool;
                        continue;
                    }
                    let keep = if self.tools {
                        partial_suffix(&self.buf, TOOL_OPEN)
                    } else {
                        0
                    };
                    let c: String = self.buf.drain(..self.buf.len() - keep).collect();
                    self.emit_content(&mut out, c);
                    break;
                }
            }
        }
        out
    }

    fn emit_content(&mut self, out: &mut Vec<Piece>, c: String) {
        if !c.trim().is_empty() {
            self.seen_content = true;
        }
        push_piece(out, Piece::Content(c));
    }

    pub fn finish(&mut self) -> Vec<Piece> {
        let mut out = vec![];
        let rest = std::mem::take(&mut self.buf);
        match self.mode {
            Mode::Reasoning => push_piece(&mut out, Piece::Reasoning(rest)),
            Mode::Content => push_piece(&mut out, Piece::Content(rest)),
            Mode::Tool => {
                let raw = std::mem::take(&mut self.tool_buf) + &rest;
                match parse_tool_call(&raw) {
                    Some((name, arguments)) => out.push(Piece::ToolCall { name, arguments }),
                    None => push_piece(&mut out, Piece::Content(format!("{TOOL_OPEN}{raw}"))),
                }
            }
        }
        self.mode = Mode::Content;
        out
    }
}

fn push_piece(out: &mut Vec<Piece>, p: Piece) {
    match &p {
        Piece::Reasoning(s) | Piece::Content(s) if s.is_empty() => {}
        _ => out.push(p),
    }
}

/// Whole-text convenience wrapper.
pub fn parse_all(
    text: &str,
    reasoning: bool,
    tools: bool,
    start_in_reasoning: bool,
) -> (String, String, Vec<(String, String)>) {
    let mut p = OutputParser::new(reasoning, tools, start_in_reasoning);
    let mut pieces = p.push(text);
    pieces.extend(p.finish());
    let (mut r, mut c, mut t) = (String::new(), String::new(), vec![]);
    for piece in pieces {
        match piece {
            Piece::Reasoning(s) => r.push_str(&s),
            Piece::Content(s) => c.push_str(&s),
            Piece::ToolCall { name, arguments } => t.push((name, arguments)),
        }
    }
    (r, c, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(
        text: &str,
        chunk: usize,
        reasoning: bool,
        tools: bool,
        start: bool,
    ) -> (String, String, Vec<(String, String)>) {
        let mut p = OutputParser::new(reasoning, tools, start);
        let mut pieces = vec![];
        let chars: Vec<char> = text.chars().collect();
        for c in chars.chunks(chunk) {
            pieces.extend(p.push(&c.iter().collect::<String>()));
        }
        pieces.extend(p.finish());
        let (mut r, mut c, mut t) = (String::new(), String::new(), vec![]);
        for piece in pieces {
            match piece {
                Piece::Reasoning(s) => r.push_str(&s),
                Piece::Content(s) => c.push_str(&s),
                Piece::ToolCall { name, arguments } => t.push((name, arguments)),
            }
        }
        (r, c, t)
    }

    #[test]
    fn reasoning_and_tools_any_chunking() {
        let text = "<think>\nlet me think</think>\n\nSure.<tool_call>\n{\"name\": \"get\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        for chunk in [1, 2, 3, 7, 100] {
            let (r, c, t) = stream(text, chunk, true, true, false);
            assert_eq!(r, "\nlet me think", "chunk {chunk}");
            assert_eq!(c, "Sure.", "chunk {chunk}");
            assert_eq!(
                t,
                vec![("get".to_string(), "{\"city\":\"Paris\"}".to_string())],
                "chunk {chunk}"
            );
        }
    }

    #[test]
    fn prompt_opened_think() {
        let (r, c, _) = parse_all("plan</think>answer", true, false, true);
        assert_eq!((r.as_str(), c.as_str()), ("plan", "answer"));
        // disabled parsers keep text verbatim
        let (r, c, t) = parse_all("<think>x</think>y<tool_call>z", false, false, false);
        assert_eq!(
            (r.as_str(), c.as_str(), t.len()),
            ("", "<think>x</think>y<tool_call>z", 0)
        );
    }

    #[test]
    fn xml_tool_calls() {
        let text = "Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nSão Paulo\n</parameter>\n<parameter=days>\n2\n</parameter>\n<parameter=note>\nline one\nline two\n</parameter>\n</function>\n</tool_call>";
        for chunk in [1, 3, 200] {
            let (_, c, t) = stream(text, chunk, true, true, false);
            assert_eq!(c, "Checking.\n\n", "chunk {chunk}");
            assert_eq!(t.len(), 1);
            assert_eq!(t[0].0, "get_weather");
            let args: Value = serde_json::from_str(&t[0].1).unwrap();
            assert_eq!(
                args,
                serde_json::json!({"city": "São Paulo", "days": 2, "note": "line one\nline two"})
            );
        }
    }

    #[test]
    fn bad_tool_json_becomes_content() {
        let (_, c, t) = parse_all("a<tool_call>not json</tool_call>b", false, true, false);
        assert!(t.is_empty());
        assert_eq!(c, "a<tool_call>not json</tool_call>b");
    }
}
