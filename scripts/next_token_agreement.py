#!/usr/bin/env python3
"""Next-token agreement between two server configurations (for lossy options such as
`--kv-dtype q8`): greedy next token after growing prefixes of a text, through /v1/completions.

Run `collect` once per configuration, then `compare`:

    scripts/next_token_agreement.py collect a.json README.md PLAN.md src/model/*.rs
    (restart the server with the other configuration)
    scripts/next_token_agreement.py collect b.json README.md PLAN.md src/model/*.rs
    scripts/next_token_agreement.py compare a.json b.json

`collect` also records the top 20 log-probabilities per prefix (`--logprobs`, carbon's
`/v1/completions` `logprobs` field), and `compare` then reports how far the distributions are.
"""
import argparse
import json
import urllib.request


def collect(a):
    text = "".join(open(f).read() for f in a.files)
    res, last = {}, None
    for i in range(1, a.points + 1):
        cut = i * a.step
        if cut > len(text):
            break
        body = {"prompt": text[:cut], "max_tokens": 1, "temperature": 0}
        if a.logprobs:
            body["logprobs"] = a.logprobs
        req = urllib.request.Request(
            a.base_url.rstrip("/") + "/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        d = json.load(urllib.request.urlopen(req, timeout=a.timeout))
        choice = d["choices"][0]
        lp = choice.get("logprobs")
        # with log-probabilities: the token's text and the top tokens as [id, logprob]
        res[cut] = {"text": choice["text"], "top": lp["top_token_ids"][0]} if lp else choice["text"]
        last = d.get("usage", {}).get("prompt_tokens")
    json.dump(res, open(a.out, "w"))
    print(f"{len(res)} prefixes, the longest {last} tokens")


def text_of(v):
    return v["text"] if isinstance(v, dict) else v


def compare(a):
    x, y = json.load(open(a.a)), json.load(open(a.b))
    keys = [k for k in x if k in y]
    same = [k for k in keys if text_of(x[k]) == text_of(y[k])]
    print(f"next-token agreement: {len(same)}/{len(keys)} = {100 * len(same) / max(len(keys), 1):.1f}%")
    scored = [k for k in keys if isinstance(x[k], dict) and isinstance(y[k], dict)]
    if scored:
        distributions(x, y, scored)
    if a.list or not scored:
        for k in keys:
            if text_of(x[k]) != text_of(y[k]):
                print(f"  after {k} chars: {text_of(x[k])!r} vs {text_of(y[k])!r}")


def distributions(x, y, keys):
    """How far b's distribution is from a's, over a's top tokens (collected with --logprobs)."""
    import math
    import statistics

    kls, shifts, flips, covered = [], [], [], []
    for k in keys:
        ta, tb = x[k]["top"], dict(map(tuple, y[k]["top"]))
        floor = min(tb.values())  # b's tokens outside its list are at most this likely
        kl = mass = 0.0
        for tok, lp in ta:
            pa = math.exp(lp)
            kl += pa * (lp - tb.get(tok, floor))
            mass += pa
        kls.append(kl)
        covered.append(mass)
        shifts.append(abs(ta[0][1] - tb.get(ta[0][0], floor)))
        if ta[0][0] != y[k]["top"][0][0]:
            # a's margin between its first and second choice, and its top probability
            flips.append((ta[0][1] - ta[1][1], math.exp(ta[0][1])))
    q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))]
    print(f"over a's top {len(x[keys[0]]['top'])} tokens ({100 * statistics.mean(covered):.1f}% of its probability):")
    print(f"  KL(a || b): mean {statistics.mean(kls):.4f} nats, median {q(kls, 0.5):.4f}, 90th percentile {q(kls, 0.9):.4f}")
    print(f"  |logprob change| of a's top token: mean {statistics.mean(shifts):.3f}, 90th percentile {q(shifts, 0.9):.3f}")
    if flips:
        margins, tops = zip(*flips)
        near = sum(m < 0.5 for m in margins)
        print(
            f"  {len(flips)} different top tokens: a's margin over its second choice median {q(margins, 0.5):.2f} nats "
            f"({near} under 0.5), a's top probability median {q(tops, 0.5):.2f}"
        )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("out")
    c.add_argument("files", nargs="+")
    c.add_argument("--base-url", default="http://localhost:1919/v1")
    c.add_argument("--points", type=int, default=250)
    c.add_argument("--step", type=int, default=600, help="characters between prefixes")
    c.add_argument("--timeout", type=float, default=3600)
    c.add_argument("--logprobs", type=int, default=20, help="top tokens to record per prefix (0: text only)")
    c.set_defaults(fn=collect)
    m = sub.add_parser("compare")
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--list", action="store_true", help="print every disagreement")
    m.set_defaults(fn=compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
