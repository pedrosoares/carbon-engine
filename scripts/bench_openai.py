#!/usr/bin/env python3
"""Decode-speed benchmark against any OpenAI-compatible server (carbon, LM Studio, llama-server).

Sends fixed prompts with streaming on, greedy sampling, and reports per run: prompt tokens,
completion tokens, time to first token and decode tok/s (tokens after the first over the time
after the first). `--context N` prepends a deterministic filler document of about N tokens so
the same request can be replayed at depth.

    scripts/bench_openai.py --base-url http://localhost:1234/v1 --model qwen/qwen3.8-27b
    scripts/bench_openai.py --base-url http://localhost:1919/v1 --context 32000 --out run.json
"""
import argparse
import json
import random
import time
import urllib.request

PROMPTS = {
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "code": "Write a Python implementation of an LRU cache class with get and put, with type hints and docstrings.",
    "math": "Explain step by step how to compute the determinant of a 4x4 matrix by cofactor expansion, with a worked numeric example.",
    "json": "Output a JSON array of 20 fictional employees, each with id, name, department, salary and start_date fields. Output only JSON.",
}

WORDS = (
    "the of and to in is that for it as was with be by on not he this are or his from at which but have an "
    "they you were her all she there would their we him been has when who will more no if out so said what up "
    "its about into than them can only other new some could time these two may then do first any my now such "
    "like our over man me even most made after also did many before must through years where much your way "
    "well down should because each just those people how too little state good very make world still own see "
    "men work long get here between both life being under never day same another know while last might us "
    "great old year off come since against go came right used take three system engine memory kernel token "
    "cache layer model tensor weight vector matrix thread buffer signal river mountain garden window letter"
).split()


def filler(tokens: int, seed: int = 1234) -> str:
    """About `tokens` tokens of seeded pseudo-prose (roughly one token per word)."""
    rng = random.Random(seed)
    out, n, para = [], 0, 0
    while n < tokens:
        k = rng.randint(8, 20)
        s = " ".join(rng.choice(WORDS) for _ in range(k))
        out.append(s.capitalize() + ".")
        n += k + 1
        para += 1
        if para % 6 == 0:
            out.append("\n\n")
    return " ".join(out)


def run(base_url, model, prompt, max_tokens, no_think, timeout):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if model:
        body["model"] = model
    if no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    t0 = time.perf_counter()
    first = last = None
    chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    chunks += 1
    out_tokens = usage.get("completion_tokens", chunks)
    span = (last - first) if first and last and last > first else 0.0
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": out_tokens,
        "ttft_s": round(first - t0, 3) if first else None,
        "decode_tok_s": round((out_tokens - 1) / span, 2) if span > 0 and out_tokens > 1 else None,
        "total_s": round(time.perf_counter() - t0, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:1919/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--context", type=int, default=0, help="filler tokens before each prompt")
    ap.add_argument("--prompts", default="story,code,math,json")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--think", action="store_true", help="leave thinking on")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    doc = filler(a.context) + "\n\nIgnore the text above. " if a.context else ""
    rows = []
    for name in a.prompts.split(","):
        for i in range(a.repeat):
            r = run(a.base_url, a.model, doc + PROMPTS[name], a.max_tokens, not a.think, a.timeout)
            r.update(prompt=name, rep=i, context=a.context, label=a.label)
            rows.append(r)
            print(json.dumps(r), flush=True)
    speeds = [r["decode_tok_s"] for r in rows if r["decode_tok_s"]]
    if speeds:
        print(f"# {a.label or a.base_url}: mean decode {sum(speeds) / len(speeds):.1f} tok/s over {len(speeds)} runs")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
