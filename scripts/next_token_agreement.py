#!/usr/bin/env python3
"""Next-token agreement between two server configurations (for lossy options such as
`--kv-dtype q8`): greedy next token after growing prefixes of a text, through /v1/completions.

Run `collect` once per configuration, then `compare`:

    scripts/next_token_agreement.py collect a.json README.md PLAN.md src/model/*.rs
    (restart the server with the other configuration)
    scripts/next_token_agreement.py collect b.json README.md PLAN.md src/model/*.rs
    scripts/next_token_agreement.py compare a.json b.json
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
        req = urllib.request.Request(
            a.base_url.rstrip("/") + "/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        d = json.load(urllib.request.urlopen(req, timeout=a.timeout))
        res[cut] = d["choices"][0]["text"]
        last = d.get("usage", {}).get("prompt_tokens")
    json.dump(res, open(a.out, "w"))
    print(f"{len(res)} prefixes, the longest {last} tokens")


def compare(a):
    x, y = json.load(open(a.a)), json.load(open(a.b))
    keys = [k for k in x if k in y]
    same = sum(x[k] == y[k] for k in keys)
    print(f"next-token agreement: {same}/{len(keys)} = {100 * same / max(len(keys), 1):.1f}%")
    for k in keys:
        if x[k] != y[k]:
            print(f"  after {k} chars: {x[k]!r} vs {y[k]!r}")


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
    c.set_defaults(fn=collect)
    m = sub.add_parser("compare")
    m.add_argument("a")
    m.add_argument("b")
    m.set_defaults(fn=compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
