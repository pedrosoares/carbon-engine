#!/usr/bin/env python3
# Copyright 2026 Carbonis Fundatus
# SPDX-License-Identifier: Apache-2.0

"""Compare `carbon generate` with the greedy outputs a *_reference.py script recorded.

    python scripts/qwen38_reference.py --out /tmp/q38 --tokenizer-from /tmp/tiny-qwen3
    python scripts/compare_reference.py /tmp/q38 ./target/release/carbon [extra generate flags]

OUT holds reference.json (prompts, and per variant the tokens and float64 top-1/top-2 margins of
each step) plus one checkpoint dir per variant. A divergence at a margin below 1e-3 is a tie: the
run continues from the reference token. Any other divergence is a mismatch (exit status 1).
"""
import json
import re
import subprocess
import sys

TIE = 1e-3


def run(exe, model, prompt, n, extra):
    out = subprocess.run([exe, "generate", "--model", model, "--devices", "cpu",
                          "--prompt-ids", ",".join(map(str, prompt)), "--max-tokens", str(n),
                          "--ignore-eos", "--print-ids", *extra], capture_output=True, text=True)
    m = re.search(r"output_ids: \[([^\]]*)\]", out.stdout + out.stderr)
    if not m:
        raise RuntimeError(out.stderr[-1500:])
    return [int(x) for x in m.group(1).split(",") if x.strip()]


def main():
    root, exe, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    ref = json.load(open(f"{root}/reference.json"))
    bad = 0
    for name in [k for k in ref if k != "prompts"]:
        for i, p in enumerate(ref["prompts"]):
            want, margins = ref[name][i]["tokens"], ref[name][i]["margins"]
            k, ties, status = 0, 0, "ok"
            while k < len(want):
                got = run(exe, f"{root}/{name}", p + want[:k], len(want) - k, extra)
                n = next((j for j in range(len(got)) if got[j] != want[k + j]), len(got))
                k += n
                if k >= len(want):
                    break
                if margins[k] < TIE:  # a genuine tie: continue from the reference token
                    ties += 1
                    k += 1
                    continue
                status = f"MISMATCH at {k} (margin {margins[k]}): got {got[n]} want {want[k]}"
                bad += 1
                break
            print(f"{name:8} prompt {i}: {status}" + (f" ({ties} ties)" if ties else ""))
    print("ALL MATCH" if not bad else f"{bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
