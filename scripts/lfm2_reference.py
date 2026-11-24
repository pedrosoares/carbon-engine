#!/usr/bin/env python3
# Copyright 2026 Carbonis Fundatus
# SPDX-License-Identifier: Apache-2.0

"""Check a running carbon server against an independent f32 forward pass of an LFM2 GGUF.

The reference is plain NumPy, written from HF `Lfm2ForCausalLM` / llama.cpp `lfm2` (short-conv
and GQA attention layers, per-head q/k RMSNorm, neox rope, SwiGLU, tied head) and fed the
file's weights decoded to f32. It checks (1) the next-token log-probabilities after a few
prompts (prefill) and (2) a greedy generation token by token (decode steps through the conv
state): every generated token must be the reference's argmax.

    pip install numpy gguf
    carbon serve --model LFM2.5-350M-Q8_0.gguf --dtype f32 --port 8000 &
    python scripts/lfm2_reference.py --gguf LFM2.5-350M-Q8_0.gguf --server http://localhost:8000

With `--dtype f32` the log-probabilities agree to about 1e-5; in bf16 expect about 0.1.
"""
import argparse
import json
import urllib.request

import gguf
import numpy as np


def byte_decoder() -> dict:
    """GPT-2 byte-level BPE: the printable stand-in of each byte -> the byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class Lfm2:
    def __init__(self, path: str):
        r = gguf.GGUFReader(path)
        f = lambda k: r.fields[k].contents()
        if f("general.architecture") != "lfm2":
            raise SystemExit(f"{path} is not an lfm2 GGUF")
        self.w = {t.name: gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32) for t in r.tensors}
        self.h, self.heads, self.layers = f("lfm2.embedding_length"), f("lfm2.attention.head_count"), f("lfm2.block_count")
        kv = f("lfm2.attention.head_count_kv")
        self.kv = kv if isinstance(kv, list) else [kv] * self.layers
        self.eps, self.theta = f("lfm2.attention.layer_norm_rms_epsilon"), f("lfm2.rope.freq_base")
        self.vocab = f("tokenizer.ggml.tokens")
        self.special = {i for i, t in enumerate(f("tokenizer.ggml.token_type")) if t != 1}
        self.bd = byte_decoder()
        self.head = self.w.get("output.weight", self.w["token_embd.weight"])

    def piece(self, t: int) -> str:
        s = self.vocab[t]
        if t in self.special:
            return s
        return bytes(self.bd[c] for c in s).decode("utf-8", errors="replace")

    def rms(self, x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + self.eps) * w

    def logprobs(self, ids: list) -> np.ndarray:
        """Next-token log-probabilities after `ids`."""
        w, H, nh = self.w, self.h, self.heads
        hd, T = H // nh, len(ids)
        x = w["token_embd.weight"][ids].copy()
        ang = np.arange(T)[:, None] * self.theta ** (-np.arange(0, hd, 2) / hd)[None]
        cos, sin = np.cos(np.concatenate([ang, ang], -1)), np.sin(np.concatenate([ang, ang], -1))

        def rope(q):  # [T, heads, hd], rotate_half (neox) layout
            q1, q2 = q[..., : hd // 2], q[..., hd // 2 :]
            return q * cos[:, None] + np.concatenate([-q2, q1], -1) * sin[:, None]

        for l in range(self.layers):
            p = f"blk.{l}."
            h = self.rms(x, w[p + "attn_norm.weight"])
            if self.kv[l] == 0:
                bcx = h @ w[p + "shortconv.in_proj.weight"].T
                bx = bcx[:, :H] * bcx[:, 2 * H :]
                taps = w[p + "shortconv.conv.weight"]  # [H, K]
                k = taps.shape[1]
                pad = np.concatenate([np.zeros((k - 1, H), np.float32), bx])
                conv = sum(pad[j : j + T] * taps[:, j] for j in range(k))
                o = (bcx[:, H : 2 * H] * conv) @ w[p + "shortconv.out_proj.weight"].T
            else:
                kv = self.kv[l]
                q = (h @ w[p + "attn_q.weight"].T).reshape(T, nh, hd)
                kk = (h @ w[p + "attn_k.weight"].T).reshape(T, kv, hd)
                v = (h @ w[p + "attn_v.weight"].T).reshape(T, kv, hd)
                q, kk = rope(self.rms(q, w[p + "attn_q_norm.weight"])), rope(self.rms(kk, w[p + "attn_k_norm.weight"]))
                kk, v = np.repeat(kk, nh // kv, 1), np.repeat(v, nh // kv, 1)
                s = np.einsum("thd,shd->hts", q, kk) / np.sqrt(hd) + np.triu(np.full((T, T), -1e30, np.float32), 1)
                s = np.exp(s - s.max(-1, keepdims=True))
                s /= s.sum(-1, keepdims=True)
                o = np.einsum("hts,shd->thd", s, v).reshape(T, H) @ w[p + "attn_output.weight"].T
            x = x + o
            h = self.rms(x, w[p + "ffn_norm.weight"])
            g, u = h @ w[p + "ffn_gate.weight"].T, h @ w[p + "ffn_up.weight"].T
            x = x + (g / (1 + np.exp(-g)) * u) @ w[p + "ffn_down.weight"].T
        z = self.rms(x[-1], w["token_embd_norm.weight"]) @ self.head.T
        m = z.max()
        return z - (np.log(np.exp(z - m).sum()) + m)


def complete(server: str, ids: list, max_tokens: int) -> dict:
    body = {"prompt": ids, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True, "logprobs": 20}
    req = urllib.request.Request(server + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["choices"][0]["logprobs"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--decode-steps", type=int, default=32)
    args = ap.parse_args()
    m = Lfm2(args.gguf)
    # BOS + ChatML turns and plain text, as token ids (special tokens by id, text by vocabulary)
    tok = {t: i for i, t in enumerate(m.vocab)}
    prompts = [
        [1, tok["<|im_start|>"], tok["user"], tok["Ċ"], tok["Hello"], tok["!"], tok["<|im_end|>"], tok["Ċ"],
         tok["<|im_start|>"], tok["assistant"], tok["Ċ"]],
        [1, tok["The"], tok["Ġhistory"], tok["Ġof"], tok["Ġthe"], tok["ĠRoman"], tok["ĠEmpire"]],
        [1, tok["The"]],
    ]
    worst = 0.0
    for ids in prompts:
        ref = m.logprobs(ids)
        got = complete(args.server, ids, 1)["top_logprobs"][0]
        top = np.argsort(-ref)[:5]
        d = max(abs(got[m.piece(int(t))] - ref[t]) for t in top if m.piece(int(t)) in got)
        worst = max(worst, d)
        print(f"prefill T={len(ids)}: reference top-1 {m.piece(int(top[0]))!r}, max |dlogprob| over its top-5 {d:.2e}")
    lp = complete(args.server, prompts[0], args.decode_steps)
    seq, off, flips = list(prompts[0]), 0, 0
    by_piece = {}
    for i, t in enumerate(m.vocab):
        by_piece.setdefault(m.piece(i), i)
    for piece, clp in zip(lp["tokens"], lp["token_logprobs"]):
        ref = m.logprobs(seq)
        t = by_piece[piece]
        flips += ref.max() - ref[t] > 1e-3
        worst = max(worst, abs(ref[t] - clp))
        seq.append(t)
    print(f"decode: {len(lp['tokens'])} steps, tokens that are not the reference argmax: {flips}")
    print(f"max |dlogprob| overall: {worst:.2e}")


if __name__ == "__main__":
    main()
