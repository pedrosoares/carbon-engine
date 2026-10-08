// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Late ("inline") system messages: clients such as Claude Code inject system instructions in
//! the middle of a conversation. Templates that render a distinct, prefix-stable system turn
//! keep them in place (`preserve`); otherwise they are folded into the neighbouring user turn
//! (`fold`), so earlier prompt tokens (and the prefix cache) are never disturbed. `auto` probes
//! the checkpoint's template once at start-up.

use crate::tokenizer::TokenizerBundle;
use serde_json::{json, Value};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InlineSystem {
    Preserve,
    Fold,
}

pub fn has_inline_system(messages: &[Value]) -> bool {
    let mut started = false;
    for m in messages {
        if m["role"] == "system" {
            if started {
                return true;
            }
        } else {
            started = true;
        }
    }
    false
}

fn join(a: &str, b: &str) -> String {
    [a, b]
        .iter()
        .filter(|s| !s.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn add_text(message: &Value, text: &str, prepend: bool) -> Result<Value, String> {
    let mut m = message.clone();
    match m.get("content").cloned() {
        None | Some(Value::Null) => m["content"] = json!(text),
        Some(Value::String(c)) => {
            m["content"] = json!(if prepend { join(text, &c) } else { join(&c, text) })
        }
        Some(Value::Array(mut parts)) => {
            let idx = if prepend { 0 } else { parts.len().wrapping_sub(1) };
            match parts.get_mut(idx) {
                Some(p) if p["type"] == "text" => {
                    let prev = p["text"].as_str().unwrap_or("").to_string();
                    p["text"] = json!(if prepend { join(text, &prev) } else { join(&prev, text) });
                }
                _ => {
                    let part = json!({"type": "text", "text": text});
                    if prepend {
                        parts.insert(0, part);
                    } else {
                        parts.push(part);
                    }
                }
            }
            m["content"] = Value::Array(parts);
        }
        Some(_) => return Err("inline system folding requires text or content blocks".into()),
    }
    Ok(m)
}

/// Hold system messages that arrive between an assistant tool call and its results until the
/// results are complete (templates require tool results right after the call).
fn after_tool_results(messages: Vec<Value>) -> Result<Vec<Value>, String> {
    let mut out = vec![];
    let mut held = vec![];
    let mut pending: HashSet<String> = HashSet::new();
    for m in messages {
        let role = m["role"].as_str().unwrap_or("").to_string();
        if role == "system" && !pending.is_empty() {
            held.push(m);
            continue;
        }
        if role == "tool" {
            pending.remove(m["tool_call_id"].as_str().unwrap_or(""));
            out.push(m);
            if pending.is_empty() {
                out.append(&mut held);
            }
            continue;
        }
        if !pending.is_empty() && !held.is_empty() {
            return Err("inline system instruction interrupts incomplete tool results".into());
        }
        pending = if role == "assistant" {
            m["tool_calls"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|c| c["id"].as_str().unwrap_or("").to_string())
                        .collect()
                })
                .unwrap_or_default()
        } else {
            HashSet::new()
        };
        out.push(m);
    }
    if !held.is_empty() {
        return Err("inline system instruction is waiting for missing tool results".into());
    }
    Ok(out)
}

pub fn normalize(messages: Vec<Value>, mode: InlineSystem) -> Result<Vec<Value>, String> {
    if !has_inline_system(&messages) {
        return Ok(messages);
    }
    let ordered = after_tool_results(messages)?;
    if mode == InlineSystem::Preserve && ordered.first().is_some_and(|m| m["role"] == "system") {
        return Ok(ordered);
    }
    let mut out: Vec<Value> = vec![];
    let mut pending: Vec<String> = vec![];
    let mut started = false;
    for m in ordered {
        let role = m["role"].as_str().unwrap_or("").to_string();
        if role == "system" && started {
            let text = match &m["content"] {
                Value::String(s) => s.clone(),
                Value::Null => String::new(),
                _ => return Err("inline system instructions must be text".into()),
            };
            if !text.is_empty() {
                let last_role = out.last().map(|l| l["role"].clone());
                if pending.is_empty() && matches!(last_role.as_ref().and_then(|r| r.as_str()), Some("user" | "tool")) {
                    let last = out.pop().unwrap();
                    out.push(add_text(&last, &text, false)?);
                } else {
                    pending.push(text);
                }
            }
            continue;
        }
        let mut m = m;
        if !pending.is_empty() {
            let text = pending.join("\n\n");
            pending.clear();
            if role == "user" {
                m = add_text(&m, &text, true)?;
            } else {
                out.push(json!({"role": "user", "content": text}));
            }
        }
        out.push(m);
        started = started || role != "system";
    }
    if !pending.is_empty() {
        out.push(json!({"role": "user", "content": pending.join("\n\n")}));
    }
    Ok(out)
}

/// `(start, inserted tokens)` when `new` is `base` with one contiguous insertion.
fn insertion(base: &[u32], new: &[u32]) -> Option<(usize, Vec<u32>)> {
    let n = base.len().min(new.len());
    let mut start = 0;
    while start < n && base[start] == new[start] {
        start += 1;
    }
    let mut suffix = 0;
    while suffix < n - start && base[base.len() - 1 - suffix] == new[new.len() - 1 - suffix] {
        suffix += 1;
    }
    (start + suffix == base.len()).then(|| (start, new[start..new.len() - suffix].to_vec()))
}

/// Conservatively require a distinct, prefix-stable, role-marked system segment.
pub fn probe(tok: &TokenizerBundle) -> (InlineSystem, String) {
    const SLACK: usize = 8;
    let note = "Echo inline instruction";
    let reasoning = "Foxtrot previous reasoning";
    let render = |msgs: &[Value]| -> Result<Vec<u32>, String> {
        let text = tok
            .apply_chat_template(msgs, None, true, &Default::default())
            .map_err(|e| e.to_string())?;
        tok.encode(&text, false).map_err(|e| e.to_string())
    };
    let special: HashSet<u32> = tok.tokenizer.get_added_tokens_decoder().keys().copied().collect();
    if special.is_empty() {
        return (InlineSystem::Fold, "renderer has no identifiable special tokens".into());
    }
    let system = json!({"role": "system", "content": "Alpha stable instructions"});
    let user = json!({"role": "user", "content": "Bravo first question"});
    let cases: Vec<(&str, Vec<Value>, usize)> = vec![
        (
            "middle",
            vec![
                system.clone(),
                user.clone(),
                json!({"role": "assistant", "content": "Charlie first answer"}),
                json!({"role": "user", "content": "Delta second question"}),
            ],
            3,
        ),
        ("trailing", vec![system.clone(), user.clone()], 2),
        (
            "tool result",
            vec![
                system.clone(),
                user.clone(),
                json!({"role": "assistant", "content": "", "reasoning_content": reasoning,
                       "tool_calls": [{"id": "probe_call", "type": "function",
                                       "function": {"name": "read", "arguments": {}}}]}),
                json!({"role": "tool", "tool_call_id": "probe_call", "content": "Golf tool output"}),
            ],
            4,
        ),
    ];
    let mut max_system_tokens: Option<usize> = None;
    for (name, msgs, index) in cases {
        let with_note = |role: &str| {
            let mut v = msgs.clone();
            v.insert(index, json!({"role": role, "content": note}));
            v
        };
        let (base, new) = match (render(&msgs), render(&with_note("system"))) {
            (Ok(b), Ok(n)) => (b, n),
            _ => return (InlineSystem::Fold, format!("{name}: probe could not render")),
        };
        let Some((pos, segment)) = insertion(&base, &new) else {
            return (InlineSystem::Fold, format!("{name}: other prompt tokens change"));
        };
        if index == msgs.len() && pos >= base.len() {
            return (InlineSystem::Fold, format!("{name}: instruction follows the generation opening"));
        }
        if tok.decode(&segment, false).matches(note).count() != 1
            || !segment.iter().any(|t| special.contains(t))
        {
            return (InlineSystem::Fold, format!("{name}: no distinct role-marked instruction"));
        }
        let base_text = tok.decode(&base, false);
        let new_text = tok.decode(&new, false);
        if base_text.contains(reasoning) != new_text.contains(reasoning) {
            return (InlineSystem::Fold, format!("{name}: reasoning retention changes"));
        }
        if name == "middle" {
            let Ok(as_user) = render(&with_note("user")) else {
                return (InlineSystem::Fold, "middle: cannot render a user instruction".into());
            };
            let Some((_, user_seg)) = insertion(&base, &as_user) else {
                return (InlineSystem::Fold, "middle: cannot isolate a user instruction".into());
            };
            if user_seg == segment {
                return (InlineSystem::Fold, "system instructions render as ordinary user turns".into());
            }
            max_system_tokens = Some(user_seg.len() + SLACK);
        }
        if let Some(max) = max_system_tokens {
            if segment.len() > max {
                return (
                    InlineSystem::Fold,
                    format!("{name}: system insertion overhead ({} > {max} tokens)", segment.len()),
                );
            }
        }
    }
    (
        InlineSystem::Preserve,
        "renderer preserves distinct system turns and existing prompt tokens".into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> Value {
        json!({"role": role, "content": content})
    }

    #[test]
    fn fold_moves_late_system_into_user_turns() {
        let msgs = vec![
            msg("system", "S"),
            msg("user", "u1"),
            msg("assistant", "a1"),
            msg("system", "late"),
            msg("user", "u2"),
        ];
        let out = normalize(msgs.clone(), InlineSystem::Fold).unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!(out[3], msg("user", "late\n\nu2"));
        // after a user turn: appended to it
        let out = normalize(
            vec![msg("user", "u1"), msg("system", "late")],
            InlineSystem::Fold,
        )
        .unwrap();
        assert_eq!(out, vec![msg("user", "u1\n\nlate")]);
        // preserve keeps it in place when there is a leading system message
        assert_eq!(normalize(msgs.clone(), InlineSystem::Preserve).unwrap(), msgs);
        // no inline system: untouched
        let plain = vec![msg("system", "S"), msg("user", "u")];
        assert_eq!(normalize(plain.clone(), InlineSystem::Fold).unwrap(), plain);
    }

    #[test]
    fn waits_for_tool_results() {
        let call = json!({"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}, {"id": "c2"}]});
        let msgs = vec![
            msg("user", "u"),
            call,
            json!({"role": "tool", "tool_call_id": "c1", "content": "r1"}),
            msg("system", "late"),
            json!({"role": "tool", "tool_call_id": "c2", "content": "r2"}),
        ];
        let out = normalize(msgs, InlineSystem::Fold).unwrap();
        assert_eq!(out.last().unwrap()["role"], "tool");
        assert_eq!(out.last().unwrap()["content"], "r2\n\nlate");
        let broken = vec![
            msg("user", "u"),
            json!({"role": "assistant", "tool_calls": [{"id": "c1"}]}),
            msg("system", "late"),
        ];
        assert!(normalize(broken, InlineSystem::Fold).is_err());
    }

    #[test]
    fn insertion_detection() {
        assert_eq!(insertion(&[1, 2, 5], &[1, 2, 3, 4, 5]), Some((2, vec![3, 4])));
        assert_eq!(insertion(&[1, 2, 5], &[1, 9, 3, 4, 5]), None);
    }

    #[test]
    fn probe_follows_the_template() {
        let dir = tempfile::tempdir().unwrap();
        crate::testing::write_tiny_model(dir.path(), &crate::testing::TinySpec::new("llama")).unwrap();
        // the tiny template drops system messages after the first: they must be folded
        let tok = TokenizerBundle::load(dir.path(), None).unwrap();
        assert_eq!(probe(&tok).0, InlineSystem::Fold);
        // plain ChatML renders every system turn in place
        let tpl = dir.path().join("chatml.jinja");
        std::fs::write(&tpl, "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}").unwrap();
        let tok = TokenizerBundle::load(dir.path(), Some(&tpl)).unwrap();
        let (mode, why) = probe(&tok);
        assert_eq!(mode, InlineSystem::Preserve, "{why}");
    }
}
