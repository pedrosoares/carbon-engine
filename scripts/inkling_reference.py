#!/usr/bin/env python3
# Copyright 2026 Carbonis Fundatus
# SPDX-License-Identifier: Apache-2.0

"""Check a running carbon server against an independent f32 forward pass of an Inkling 2 GGUF.

The reference is plain NumPy, written from llama.cpp's `inkling2` / `inkling2_loop` graphs
(src/models/inkling2*.cpp): a prelude, a core of layers run `loop.count` times with cross-pass
messages, a coda; block attention residuals over the depth; relative-position-biased attention
with per-query sinks and no rope; causal short convolutions on K, V and both layer outputs;
sigmoid-routed SiTU experts. Weights are decoded to f32 per use (experts per routed expert).

It checks (1) the next-token log-probabilities after a few prompts (prefill) and (2) a greedy
generation token by token (decode steps through the conv and sliding-window state): every
generated token must be the reference's argmax.

    pip install numpy gguf   # gguf >= the version with NVFP4 (or PYTHONPATH=llama.cpp/gguf-py)
    carbon serve --model Inkling-2-Mini-UD-Q4_K_XL.gguf --port 8000 &
    python scripts/inkling_reference.py --gguf Inkling-2-Mini-UD-Q4_K_XL.gguf --server http://localhost:8000
"""
import argparse
import json
import math
import urllib.request

import gguf
import numpy as np


def byte_decoder() -> dict:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def rms(x, w=None, eps=1e-6):
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return y if w is None else y * w


def clip(x, threshold):
    """x * min(1, t / |x|) over the last axis."""
    n = np.sqrt((x * x).sum(-1, keepdims=True))
    return x * threshold / np.maximum(n, threshold)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Inkling:
    def __init__(self, path: str):
        r = gguf.GGUFReader(path)
        self.fields = r.fields
        self.arch = self.f("general.architecture")
        if self.arch not in ("inkling2", "inkling2_loop"):
            raise SystemExit(f"{path} is not an inkling2 GGUF")
        self.t = {t.name: t for t in r.tensors}
        self.cache = {}
        a = lambda k: self.f(f"{self.arch}.{k}")
        self.H, self.layers = a("embedding_length"), a("block_count")
        hc = a("attention.head_count")
        self.nh = hc[0] if isinstance(hc, list) else hc
        hk = a("attention.head_count_kv")
        self.nkv = hk[0] if isinstance(hk, list) else hk
        self.hd = a("attention.key_length")
        self.eps = a("attention.layer_norm_rms_epsilon")
        self.swa = a("attention.sliding_window_pattern")
        self.window = a("attention.sliding_window")
        self.qk_clip, self.rel_clip = a("attention.qk_clip"), a("attention.relative_clip")
        self.n_rel, self.extent = a("attention.relative_length"), a("attention.relative_extent")
        self.log_floor, self.log_alpha = a("attention.log_scale_floor"), a("attention.log_scale_alpha")
        self.block = a("attn_res.block_size")
        self.top_k, self.w_scale = a("expert_used_count"), a("expert_weights_scale")
        self.beta, self.lbeta = a("activation.situ_beta"), a("activation.situ_linear_beta")
        self.softcap = a("final_logit_softcapping")
        self.vocab_n = a("vocab_size_unpadded")
        if self.arch == "inkling2_loop":
            self.loop = (a("loop.start"), a("loop.end"), a("loop.count"))
        else:
            self.loop = (0, 0, 1)
        self.vocab = self.f("tokenizer.ggml.tokens")
        self.special = {i for i, t in enumerate(self.f("tokenizer.ggml.token_type")) if t != 1}
        self.bd = byte_decoder()

    def f(self, k):
        return self.fields[k].contents()

    def w(self, name):
        if name not in self.cache:
            t = self.t[name]
            self.cache[name] = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        return self.cache[name]

    def has(self, name):
        return name in self.t

    def expert(self, name, e):
        t = self.t[name]
        return gguf.quants.dequantize(t.data[e], t.tensor_type).astype(np.float32)

    def piece(self, t: int) -> str:
        s = self.vocab[t]
        if t in self.special:
            return s
        return bytes(self.bd[c] for c in s).decode("utf-8", errors="replace")

    # depth attention: mix the normalized earlier blocks and cur with learned queries [nq, H]
    def mix(self, blocks, cur, query):
        states = np.stack([rms(b) for b in blocks] + [rms(cur)], 1)  # [T, S, H]
        q = query.reshape(-1, self.H)
        s = np.einsum("tsh,qh->tqs", states, q) / math.sqrt(self.H)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        return np.einsum("tqs,tsh->qth", s, states)  # [nq, T, H]

    def control(self, cur, msg, p, ctrl):
        name = ["q", "k", "v", "ffn"][ctrl]
        gate = np.concatenate([rms(cur), rms(msg)], -1)
        gate = gate @ self.w(f"{p}loop_{name}_ctrl_down.weight").T
        gain = np.tanh(gate @ self.w(f"{p}loop_{name}_ctrl_gain.weight").T)
        forget = np.tanh(gate @ self.w(f"{p}loop_{name}_ctrl_forget.weight").T)
        return cur - forget * cur + gain * msg

    def conv(self, x, w, dyn=None):
        """Causal conv over time from a zero history, plus the input; dyn: per-head taps [T, heads, K]."""
        T, C = x.shape
        K = w.shape[1]
        pad = np.concatenate([np.zeros((K - 1, C), np.float32), x])
        out = sum(pad[j : j + T] * w[:, j] for j in range(K))
        if dyn is not None:
            heads = dyn.shape[1]
            for lag in range(K):
                win = pad[K - 1 - lag : K - 1 - lag + T].reshape(T, heads, C // heads)
                out = out + (win * dyn[:, :, lag : lag + 1]).reshape(T, C)
        return (x + out) / math.sqrt((2 if dyn is not None else 1) * K + 1)

    def layer(self, ip, cur, blocks, pos, messages=None):
        H, nh, nkv, hd, T = self.H, self.nh, self.nkv, self.hd, cur.shape[0]
        p = f"blk.{ip}."
        swa = bool(self.swa[ip])
        inp = cur
        ins = [cur, cur, cur]
        if self.has(p + "attn_res_query.weight"):
            mixed = self.mix(blocks, cur, self.w(p + "attn_res_query.weight"))
            ins = [mixed[0 if swa else j] for j in range(3)]
        if messages is not None:
            ins = [self.control(ins[j], messages[j], p, j) for j in range(3)]
        if swa:
            ins = [rms(x, self.w(p + "attn_norm.weight"), self.eps) for x in ins]
        else:
            ins = [rms(ins[j], self.w(p + f"attn_{n}_inp_norm.weight"), self.eps) for j, n in enumerate("qkv")]
        q = ins[0] @ self.w(p + "attn_q.weight").T
        kdyn = vdyn = None
        if not swa:
            kdyn = (ins[1] @ self.w(p + "attn_k_conv_proj.weight").T).reshape(T, nkv, -1)
            vdyn = (ins[2] @ self.w(p + "attn_v_conv_proj.weight").T).reshape(T, nkv, -1)
        k = self.conv(ins[1] @ self.w(p + "attn_k.weight").T, self.w(p + "attn_k_conv.weight"), kdyn)
        v = self.conv(ins[2] @ self.w(p + "attn_v.weight").T, self.w(p + "attn_v_conv.weight"), vdyn)
        r = ins[0] @ self.w(p + "attn_rel.weight").T
        q = clip(q.reshape(T, nh, hd) * self.w(p + "attn_q_clip.weight"), self.qk_clip)
        k = clip(k.reshape(T, nkv, hd), self.qk_clip)
        v = rms(v.reshape(T, nkv, hd), None, self.eps)
        r = clip(r.reshape(T, nh, self.n_rel) * self.w(p + "attn_rel_clip.weight"), self.rel_clip)
        if not swa:
            tau = 1 + self.log_alpha * np.log(np.maximum(1.0, (pos + 1.0) / self.log_floor))
            q, r = q * tau[:, None, None], r * tau[:, None, None]
        sinks = ins[0] @ self.w(p + "attn_sink_proj.weight").T  # [T, nh]
        bias = self.w(p + "attn_rel_bias.weight")  # [extent, n_rel]
        table = np.einsum("er,thr->the", bias, r)
        table = np.concatenate([table, np.zeros((T, nh, 1), np.float32)], -1)
        # this call's tokens attend to each other only (one prefill from position 0)
        dist = (pos[:, None] - pos[None, :]).astype(np.int64)
        visible = (dist >= 0) & ((dist < self.window) if swa else True)
        idx = np.minimum(np.maximum(dist, 0), table.shape[-1] - 1)
        rep = nh // nkv
        out = np.zeros((T, nh, hd), np.float32)
        for h in range(nh):
            s = q[:, h] @ k[:, h // rep].T / math.sqrt(hd)
            s = s + np.take_along_axis(table[:, h], idx, 1) - sinks[:, h : h + 1]
            s = np.where(visible, s, -np.inf)
            m = np.maximum(s.max(-1, keepdims=True), 0.0)
            e = np.exp(s - m)
            out[:, h] = e @ v[:, h // rep] / (e.sum(-1, keepdims=True) + np.exp(-m))
        a = out.reshape(T, nh * hd) * sigmoid(ins[0] @ self.w(p + "attn_gate.weight").T)
        a = self.conv(a @ self.w(p + "attn_output.weight").T, self.w(p + "attn_post_conv.weight"))
        if ip % self.block == 0:
            blocks.append(inp)
            cur = a
        else:
            cur = inp + a
        ffn_inp = cur
        h = self.mix(blocks, cur, self.w(p + "ffn_res_query.weight"))[0]
        if messages is not None:
            h = self.control(h, messages[3], p, 3)
        h = rms(h, self.w(p + "ffn_norm.weight"), self.eps)
        logits = h @ self.w(p + "ffn_gate_inp.weight").T
        probs = sigmoid(logits)
        sel = probs + (self.w(p + "exp_probs_b.bias") if self.has(p + "exp_probs_b.bias") else 0)
        y = np.zeros_like(h)
        gs = self.w(p + "ffn_gate_up_exps.scale") if self.has(p + "ffn_gate_up_exps.scale") else None
        ds = self.w(p + "ffn_down_exps.scale") if self.has(p + "ffn_down_exps.scale") else None
        for t in range(T):
            top = np.argsort(-sel[t], kind="stable")[: self.top_k]
            wts = probs[t, top]
            wts = wts / (wts.sum() + 1e-20) * self.w_scale
            for e, wt in zip(top, wts):
                gu = self.expert(p + "ffn_gate_up_exps.weight", e) @ h[t]
                if gs is not None:
                    gu = gu * gs[e]
                g, u = gu[: gu.shape[0] // 2], gu[gu.shape[0] // 2 :]
                act = self.beta * np.tanh(g / self.beta) * sigmoid(g) * self.lbeta * np.tanh(u / self.lbeta)
                d = self.expert(p + "ffn_down_exps.weight", e) @ act
                if ds is not None:
                    d = d * ds[e]
                y[t] += wt * d
        y = self.conv(y, self.w(p + "ffn_post_conv.weight"))
        return ffn_inp + y

    def logprobs(self, ids: list) -> np.ndarray:
        """Next-token log-probabilities after `ids`."""
        T = len(ids)
        pos = np.arange(T, dtype=np.float32)
        cur = self.w("token_embd.weight")[ids]
        cur = rms(cur, self.w("token_embd_norm.weight"), self.eps)
        blocks = []
        start, end, passes = self.loop
        if self.arch == "inkling2":
            start = end = self.layers
        for ip in range(start):
            cur = self.layer(ip, cur, blocks, pos)
        prelude, prelude_blocks = cur, list(blocks)
        for pss in range(passes):
            msg_bank = None
            if pss > 0:
                msg_bank = (blocks[len(prelude_blocks) + 1 :], cur)
                cur, blocks = prelude, list(prelude_blocks)
            for ip in range(start, end):
                messages = None
                if msg_bank is not None:
                    messages = self.mix(msg_bank[0], msg_bank[1], self.w(f"blk.{ip}.loop_msg_query.weight"))
                cur = self.layer(ip, cur, blocks, pos, messages)
        for ip in range(end, self.layers):
            cur = self.layer(ip, cur, blocks, pos)
        last = self.mix([b[-1:] for b in blocks], cur[-1:], self.w("output_res_query.weight"))[0][0]
        z = rms(last, self.w("output_norm.weight"), self.eps) @ self.w("output.weight").T
        z = np.tanh(z / self.softcap) * self.softcap
        z = z[: self.vocab_n]
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
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--ids", help="only print the reference top-5 after these comma-separated token ids")
    args = ap.parse_args()
    m = Inkling(args.gguf)
    tok = {t: i for i, t in enumerate(m.vocab)}
    if args.ids:
        ids = [int(x) for x in args.ids.split(",")]
        ref = m.logprobs(ids)
        for t in np.argsort(-ref)[:5]:
            print(f"{int(t):>7} {m.piece(int(t))!r:>16} {ref[t]:.5f}")
        return
    prompts = [
        [tok["The"], tok["Ġcapital"], tok["Ġof"], tok["ĠFrance"], tok["Ġis"]],
        [tok["Once"], tok["Ġupon"], tok["Ġa"], tok["Ġtime"], tok[","]],
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
    seq, flips = list(prompts[0]), 0
    by_piece = {}
    for i in range(len(m.vocab)):
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


def vision(mmproj: str, rgb: np.ndarray) -> np.ndarray:
    """Image tokens [n, out] of an RGB uint8 image [h, w, 3] through the inkling2v projector
    (llama.cpp tools/mtmd/models/inkling2v.cpp and mtmd_image_preprocessor_inkling2v), in f32."""
    r = gguf.GGUFReader(mmproj)
    t = {x.name: x for x in r.tensors}
    w = lambda n: gguf.quants.dequantize(t[n].data, t[n].tensor_type).astype(np.float32)
    f = lambda k: r.fields[k].contents()
    p, m = f("clip.vision.patch_size"), f("clip.vision.spatial_merge_size")
    eps, heads = f("clip.vision.attention.layer_norm_epsilon"), f("clip.vision.attention.head_count")
    mean, std = np.array(f("clip.vision.image_mean")), np.array(f("clip.vision.image_std"))
    h0, w0, _ = rgb.shape
    a = p * m
    hh, ww = -(-h0 // a) * a, -(-w0 // a) * a
    img = np.full((hh, ww, 3), -1.0, np.float32)
    img[:h0, :w0] = rgb
    img = (img - mean * 255.0) / (std * 255.0)
    gh, gw = hh // p, ww // p
    patches = img.reshape(gh, p, gw, p, 3).transpose(0, 2, 4, 1, 3).reshape(gh * gw, 3 * p * p)
    pw = w("v.patch_embd.weight")  # [D, 3, P, P]
    d = pw.shape[0]
    x = patches @ pw.reshape(d, -1).T
    pos = w("v.position_embd.weight")
    side = int(round(np.sqrt(pos.shape[0])))
    grid = pos.reshape(side, side, d)

    def axis(n_out):
        c = (np.arange(n_out) + 0.5) * side / n_out - 0.5
        lo = np.clip(np.floor(c).astype(int), 0, side - 1)
        hi = np.clip(np.floor(c).astype(int) + 1, 0, side - 1)
        return lo, hi, np.clip(c - lo, 0.0, 1.0)

    if (gh, gw) != (side, side):
        y0, y1, dy = axis(gh)
        x0, x1, dx = axis(gw)
        top = grid[y0][:, x0] * (1 - dx)[None, :, None] + grid[y0][:, x1] * dx[None, :, None]
        bot = grid[y1][:, x0] * (1 - dx)[None, :, None] + grid[y1][:, x1] * dx[None, :, None]
        grid = top * (1 - dy)[:, None, None] + bot * dy[:, None, None]
    x = x + grid.reshape(gh * gw, d)
    cls = w("v.class_embd") + w("v.class_position_embd.weight")
    x = np.concatenate([cls[None], x])

    def ln(x, n):
        mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * w(n + ".weight") + w(n + ".bias")

    def lin(x, n, bias=True):
        y = x @ w(n + ".weight").T
        return y + w(n + ".bias") if bias and n + ".bias" in t else y

    def gelu(x):
        from math import erf
        return 0.5 * x * (1 + np.vectorize(erf)(x / np.sqrt(2)).astype(np.float32))

    x = ln(x, "v.pre_ln")
    hd = d // heads
    T = x.shape[0]
    px = np.array([0] + [i % gw + 1 for i in range(gh * gw)], np.float64)
    py = np.array([0] + [i // gw + 1 for i in range(gh * gw)], np.float64)
    pairs, half = hd // 2, hd // 4
    ang = np.zeros((T, pairs))
    for j in range(pairs):
        pos_, k = (px, j) if j < half else (py, j - half)
        ang[:, j] = pos_ * 10000.0 ** (-2.0 * k / (hd / 2))
    cos, sin = np.cos(ang)[:, None, :].astype(np.float32), np.sin(ang)[:, None, :].astype(np.float32)

    def rope(q):  # [T, H, hd], adjacent pairs
        q0, q1 = q[..., 0::2], q[..., 1::2]
        out = np.empty_like(q)
        out[..., 0::2] = q0 * cos - q1 * sin
        out[..., 1::2] = q1 * cos + q0 * sin
        return out

    for l in range(f("clip.vision.block_count")):
        b = f"v.blk.{l}."
        h = ln(x, b + "ln1")
        q = rope(lin(h, b + "attn_q").reshape(T, heads, hd))
        k = rope(lin(h, b + "attn_k").reshape(T, heads, hd))
        v = lin(h, b + "attn_v").reshape(T, heads, hd)
        s = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", s, v).reshape(T, d)
        x = x + lin(o, b + "attn_out") * w(b + "ls1.weight")
        h = ln(x, b + "ln2")
        x = x + lin(gelu(lin(h, b + "ffn_up")), b + "ffn_down") * w(b + "ls2.weight")
    x = x[1:].reshape(gh // m, m, gw // m, m, d).transpose(0, 2, 1, 3, 4).reshape(-1, m * m * d)
    return lin(gelu(lin(x, "mm.1")), "mm.2")
