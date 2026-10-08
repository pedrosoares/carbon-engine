// Copyright 2026 Carbonis Fundatus
// SPDX-License-Identifier: Apache-2.0

//! Reasoning effort and thinking toggles.
//!
//! Each checkpoint's chat template accepts only its own effort vocabulary (and may hard-fail on
//! the rest), while clients speak whatever dialect their provider taught them. The vocabulary is
//! learned by rendering probes through the checkpoint's template; requested levels then map onto
//! the nearest supported gear (vLLM / SGLang scale), or are dropped when none is close.

use serde_json::{json, Map, Value};

/// Named levels on the shared 0..1 scale.
pub const EFFORT_SCALE: [(&str, f64); 7] = [
    ("none", 0.0),
    ("minimal", 0.1),
    ("low", 0.2),
    ("medium", 0.7),
    ("high", 0.9),
    ("xhigh", 0.99),
    ("max", 0.99),
];

const OPENAI_TRIPLE: [&str; 3] = ["low", "medium", "high"];
/// Ladder of templates that grade effort without validating it (`reasoning_strength`).
const GRADED_LADDER: [&str; 4] = ["low", "medium", "high", "xhigh"];
/// A gear farther than this from the request misrepresents it: send nothing instead.
const MAX_QUANTIZE_DISTANCE: f64 = 0.15;
/// Kwarg keys that mean "the client already set thinking explicitly".
const THINKING_KEYS: [&str; 4] = [
    "enable_thinking",
    "thinking_mode",
    "thinking",
    "reasoning_effort",
];
const DISABLE_EFFORTS: [&str; 3] = ["none", "off", "disabled"];

fn scale(name: &str) -> Option<f64> {
    EFFORT_SCALE
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, v)| *v)
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EffortProfile {
    /// Names the template accepts (when `validates`) or interpolates (otherwise).
    pub supported: Vec<String>,
    /// The supported name whose rendering equals sending no effort.
    pub default: Option<String>,
    /// False: the template ignores effort, requests should not carry it.
    pub consumes_effort: bool,
    /// A probe was rejected, so `supported` is a real vocabulary.
    pub validates: bool,
    /// The template reads `reasoning_strength` (graded ladder up to xhigh).
    pub strength_dialect: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ThinkingProfile {
    pub efforts: EffortProfile,
    /// The off / on toggles render differently.
    pub toggleable: bool,
    /// `thinking_mode: adaptive` is a third distinct state.
    pub has_adaptive: bool,
    /// "on" | "off" | "adaptive": what a bare render matches.
    pub default_state: String,
}

pub fn thinking_off() -> Map<String, Value> {
    json!({"enable_thinking": false, "thinking_mode": "disabled"})
        .as_object()
        .unwrap()
        .clone()
}

pub fn thinking_on() -> Map<String, Value> {
    json!({"enable_thinking": true, "thinking_mode": "enabled"})
        .as_object()
        .unwrap()
        .clone()
}

/// Render function: `(chat_template_kwargs, with_tools) -> rendering`, Err on rejection.
pub type Render<'a> = dyn Fn(&Map<String, Value>, bool) -> Result<String, String> + 'a;

impl EffortProfile {
    /// Learn the vocabulary by rendering effort probes (plain, with a tool, with thinking on).
    pub fn probe(render: &Render) -> Self {
        let rounds: [(Map<String, Value>, bool); 3] = [
            (Map::new(), false),
            (Map::new(), true),
            (
                json!({"enable_thinking": true}).as_object().unwrap().clone(),
                false,
            ),
        ];
        let names: Vec<&str> = EFFORT_SCALE.iter().map(|x| x.0).collect();
        let mut rejected = std::collections::HashSet::new();
        let mut diverged = std::collections::HashSet::new();
        let mut matches_baseline: std::collections::HashMap<&str, bool> =
            names.iter().map(|n| (*n, true)).collect();
        let mut ran = 0;
        for (base, tools) in &rounds {
            let Ok(baseline) = render(base, *tools) else {
                continue; // the template rejected the probe conversation, not the effort
            };
            ran += 1;
            for &name in &names {
                let mut kw = base.clone();
                kw.insert("reasoning_effort".into(), json!(name));
                match render(&kw, *tools) {
                    Err(_) => {
                        rejected.insert(name);
                        matches_baseline.insert(name, false);
                    }
                    Ok(r) if r != baseline => {
                        diverged.insert(name);
                        matches_baseline.insert(name, false);
                    }
                    Ok(_) => {}
                }
            }
        }
        if ran == 0 {
            return Self::default();
        }
        let strength_dialect = match (
            render(&Map::new(), false),
            render(
                json!({"reasoning_strength": "low"}).as_object().unwrap(),
                false,
            ),
        ) {
            (Ok(a), Ok(b)) => a != b,
            _ => false,
        };
        let supported: Vec<String> = names
            .iter()
            .filter(|n| !rejected.contains(*n))
            .map(|n| n.to_string())
            .collect();
        let consumes = !rejected.is_empty() || !diverged.is_empty();
        let default = if consumes {
            supported
                .iter()
                .filter(|n| matches_baseline[n.as_str()])
                .max_by(|a, b| scale(a).partial_cmp(&scale(b)).unwrap())
                .cloned()
        } else {
            None
        };
        Self {
            supported,
            default,
            consumes_effort: consumes,
            validates: !rejected.is_empty(),
            strength_dialect,
        }
    }

    /// The vocabulary the checkpoint is served (and advertised) with.
    pub fn effective(&self) -> Vec<String> {
        if !self.consumes_effort {
            return vec![];
        }
        if self.validates || self.supported.len() != EFFORT_SCALE.len() {
            return self.supported.clone();
        }
        // accepted everything: cap to the dialect's documented ladder
        let ladder: &[&str] = if self.strength_dialect {
            &GRADED_LADDER
        } else {
            &OPENAI_TRIPLE
        };
        let mut v: Vec<String> = ladder
            .iter()
            .filter(|n| self.supported.iter().any(|s| s == *n))
            .map(|n| n.to_string())
            .collect();
        if let Some(d) = &self.default {
            if !v.contains(d) {
                v.push(d.clone());
            }
        }
        v.sort_by(|a, b| scale(a).partial_cmp(&scale(b)).unwrap());
        v
    }

    /// Map a requested effort onto this template; None means "send nothing".
    pub fn quantize(&self, value: &str) -> Option<String> {
        let supported = self.effective();
        if supported.is_empty() {
            return None;
        }
        if supported.iter().any(|s| s == value) {
            return Some(value.to_string());
        }
        let pos = scale(value)?;
        // "max" is reachable only by its own name; ties prefer the higher gear
        let best = supported
            .iter()
            .filter(|n| *n != "max")
            .min_by(|a, b| {
                let (da, db) = (
                    (scale(a).unwrap() - pos).abs(),
                    (scale(b).unwrap() - pos).abs(),
                );
                da.partial_cmp(&db)
                    .unwrap()
                    .then(scale(b).partial_cmp(&scale(a)).unwrap())
            })?;
        ((scale(best).unwrap() - pos).abs() <= MAX_QUANTIZE_DISTANCE).then(|| best.clone())
    }
}

impl ThinkingProfile {
    pub fn probe(render: &Render) -> Self {
        let efforts = EffortProfile::probe(render);
        let none = Self {
            efforts: efforts.clone(),
            toggleable: false,
            has_adaptive: false,
            default_state: "on".into(),
        };
        let (Ok(base), Ok(off), Ok(on)) = (
            render(&Map::new(), false),
            render(&thinking_off(), false),
            render(&thinking_on(), false),
        ) else {
            return none;
        };
        let toggleable = off != on;
        let adaptive = if toggleable {
            render(
                json!({"thinking_mode": "adaptive"}).as_object().unwrap(),
                false,
            )
            .ok()
        } else {
            None
        };
        let has_adaptive = adaptive.as_ref().is_some_and(|a| *a != off && *a != on);
        let default_state = if !toggleable {
            "on"
        } else if base == off {
            "off"
        } else if has_adaptive && adaptive.as_deref() == Some(base.as_str()) {
            "adaptive"
        } else {
            "on"
        };
        Self {
            efforts,
            toggleable,
            has_adaptive,
            default_state: default_state.into(),
        }
    }

    /// Fold a protocol-level effort / thinking request into the template kwargs. Explicit
    /// thinking kwargs from the client win; effort "none"/"off" disables thinking, any other
    /// effort enables it and is quantized to the template's vocabulary.
    pub fn apply(
        &self,
        effort: Option<&str>,
        thinking_type: Option<&str>,
        mut kwargs: Map<String, Value>,
    ) -> Map<String, Value> {
        if THINKING_KEYS.iter().any(|k| kwargs.contains_key(*k)) {
            // still keep an explicit reasoning_effort inside the template's vocabulary
            self.sanitize(&mut kwargs);
            return kwargs;
        }
        let effort = effort.map(|e| e.trim().to_ascii_lowercase());
        if effort.is_none() && thinking_type.is_none() {
            return kwargs;
        }
        let mut disabled = effort
            .as_deref()
            .is_some_and(|e| DISABLE_EFFORTS.contains(&e));
        match thinking_type {
            Some("disabled") => disabled = true,
            Some("enabled") | Some("adaptive") => disabled = false,
            _ => {}
        }
        let mut out = if disabled {
            thinking_off()
        } else if thinking_type == Some("adaptive") && self.has_adaptive {
            json!({"thinking_mode": "adaptive"})
                .as_object()
                .unwrap()
                .clone()
        } else {
            thinking_on()
        };
        if let Some(e) = effort.filter(|_| !disabled) {
            if let Some(q) = self.efforts.quantize(&e) {
                out.insert("reasoning_effort".into(), json!(q.clone()));
                if self.efforts.strength_dialect {
                    out.insert("reasoning_strength".into(), json!(q));
                }
            }
        }
        out.extend(kwargs);
        out
    }

    /// Replace an out-of-vocabulary `reasoning_effort` kwarg (or drop it).
    pub fn sanitize(&self, kwargs: &mut Map<String, Value>) {
        if let Some(e) = kwargs
            .get("reasoning_effort")
            .and_then(|v| v.as_str())
            .map(|s| s.to_ascii_lowercase())
        {
            match self.efforts.quantize(&e) {
                Some(q) => {
                    kwargs.insert("reasoning_effort".into(), json!(q));
                }
                None => {
                    kwargs.remove("reasoning_effort");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A template that validates effort against `vocab` and grades its output with it.
    fn grader(vocab: &'static [&'static str]) -> impl Fn(&Map<String, Value>, bool) -> Result<String, String> {
        move |kw, _| {
            let think = kw.get("enable_thinking").and_then(|v| v.as_bool()).unwrap_or(true);
            match kw.get("reasoning_effort").and_then(|v| v.as_str()) {
                None => Ok(format!("think={think} effort=high")),
                Some(e) if vocab.contains(&e) => Ok(format!("think={think} effort={e}")),
                Some(e) => Err(format!("bad effort {e}")),
            }
        }
    }

    #[test]
    fn probes_validating_template() {
        let p = ThinkingProfile::probe(&grader(&["low", "medium", "high", "xhigh"]));
        assert!(p.toggleable);
        assert_eq!(p.default_state, "on");
        let e = &p.efforts;
        assert!(e.validates && e.consumes_effort);
        assert_eq!(e.default.as_deref(), Some("high"));
        assert_eq!(e.effective(), vec!["low", "medium", "high", "xhigh"]);
        assert_eq!(e.quantize("medium").as_deref(), Some("medium"));
        assert_eq!(e.quantize("minimal").as_deref(), Some("low"));
        // "max" maps onto xhigh by distance (0.0), "none" is too far from low
        assert_eq!(e.quantize("max").as_deref(), Some("xhigh"));
        assert_eq!(e.quantize("none"), None);
        assert_eq!(e.quantize("bogus"), None);
    }

    #[test]
    fn ignoring_template_gets_no_effort() {
        let p = ThinkingProfile::probe(&|kw: &Map<String, Value>, _| {
            Ok(format!("{:?}", kw.get("enable_thinking")))
        });
        assert!(!p.efforts.consumes_effort);
        let kw = p.apply(Some("high"), None, Map::new());
        assert_eq!(kw.get("enable_thinking"), Some(&json!(true)));
        assert!(!kw.contains_key("reasoning_effort"));
        let kw = p.apply(Some("none"), None, Map::new());
        assert_eq!(kw.get("enable_thinking"), Some(&json!(false)));
    }

    #[test]
    fn apply_rules() {
        let p = ThinkingProfile::probe(&grader(&["low", "medium", "high"]));
        let kw = p.apply(Some("minimal"), None, Map::new());
        assert_eq!(kw["reasoning_effort"], "low");
        assert_eq!(kw["enable_thinking"], true);
        // the DeepSeek-style toggle decides direction outright
        let kw = p.apply(Some("high"), Some("disabled"), Map::new());
        assert_eq!(kw["enable_thinking"], false);
        assert!(!kw.contains_key("reasoning_effort"));
        // explicit client kwargs win, but stay in vocabulary
        let mut explicit = Map::new();
        explicit.insert("enable_thinking".into(), json!(false));
        explicit.insert("reasoning_effort".into(), json!("xhigh"));
        let kw = p.apply(Some("low"), None, explicit);
        assert_eq!(kw["enable_thinking"], false);
        assert_eq!(kw["reasoning_effort"], "high");
        // nothing requested: kwargs untouched
        assert!(p.apply(None, None, Map::new()).is_empty());
    }
}
