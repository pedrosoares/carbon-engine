#!/usr/bin/env python3
# Copyright 2026 Carbonis Fundatus
# SPDX-License-Identifier: Apache-2.0

"""Compare carbon's chat-template rendering + tokenization with Hugging Face's.

    python scripts/check_chat_template.py --model ../Qwen3.6-35B-A3B-FP8-tokenizer --carbon target/debug/carbon
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

from transformers import AutoTokenizer

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]

CASES = {
    "simple": {"messages": [{"role": "user", "content": "Hello! Who are you?"}]},
    "system+multi-turn": {"messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "and 3+3?"}]},
    "no-thinking": {"messages": [{"role": "user", "content": "hi"}],
                    "chat_template_kwargs": {"enable_thinking": False}},
    "tools": {"tools": TOOLS, "messages": [
        {"role": "user", "content": "Weather in Paris for 2 days?"},
        {"role": "assistant", "content": "", "reasoning_content": "I should call the tool.",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 2}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny, 21C"},
        {"role": "user", "content": "thanks"}]},
    "unicode": {"messages": [{"role": "user", "content": "Olá! 你好 🤖 — ça va?"}]},
}


def hf_messages(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        for tc in m.get("tool_calls", []) or []:
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                tc["function"]["arguments"] = json.loads(args)
        out.append(m)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--carbon", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    failed = 0
    for name, req in CASES.items():
        expected = tok.apply_chat_template(hf_messages(req["messages"]), tools=req.get("tools"),
                                           add_generation_prompt=True, tokenize=False,
                                           **req.get("chat_template_kwargs", {}))
        expected_ids = tok(expected, add_special_tokens=False)["input_ids"]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(req, f)
        p = subprocess.run([args.carbon, "chat-template", "--model", args.model, "--request", f.name, "--ids"],
                           capture_output=True, text=True)
        os.unlink(f.name)
        got = p.stdout
        got_ids = json.loads(p.stderr.strip().splitlines()[-1]) if p.returncode == 0 else None
        ok = got == expected and got_ids == expected_ids
        failed += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {name}: {len(expected_ids)} tokens")
        if not ok:
            print("--- expected\n" + expected + "\n--- got\n" + got + "\n" + p.stderr[-2000:])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
