#!/usr/bin/env python3
# Copyright 2026 Carbonis Fundatus
# SPDX-License-Identifier: Apache-2.0

"""Build tiny random Qwen3.8-Flash-Next (qwen4_exp) checkpoints with HF transformers and record
HF's greedy output for each.

    python scripts/qwen38_reference.py --out /tmp/q38 --tokenizer-from /tmp/tiny-qwen3

Writes under OUT:
  dense/    - hyper-connections, Gated DeltaNet, gated attention, shared-expert MoE; no PLE and an
              indexer budget larger than any prompt, so attention is dense
  full/     - plus PLE on two linear layers and a small indexer budget (QSA really selects)
  release/  - `full` in the layout of the real release: multimodal config wrapper,
              `model.language_model.*` names, per-expert tensors, the PLE tables as FP8 row
              blocks `ngram_embedding.shard_<i>.weight` with a scalar `weight_scale`
and reference.json: per checkpoint, greedy tokens for each prompt. Prompts include the eos id
(n-gram hash boundary) and run past the indexer budget and the 64-token delta-rule chunk.
"""
import argparse
import json
import math
import os
import shutil

import torch
from safetensors.torch import save_file
from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

VOCAB, EOS = 263, 258


def tiny_config(**kw) -> Qwen4ExpTextConfig:
    base = dict(
        vocab_size=VOCAB, hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        moe_intermediate_size=16, shared_expert_intermediate_size=24, num_experts=8,
        num_experts_per_tok=2, max_position_embeddings=4096, full_attention_interval=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        eos_token_id=EOS, tie_word_embeddings=False,
        hc_count=3, hc_lowrank=12,
        indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=16,
        indexer_budget=4096, indexer_compress_ratio=4,
        ple_layer_ids=[], ple_embed_dim=16, ple_conv_kernel_size=3, ngram_size=3,
        heads_per_ngram=2, ngram_vocab_size_base=97, make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4,
    )
    base.update(kw)
    return Qwen4ExpTextConfig(**base)


def randomize(model: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p).uniform_(1, 8, generator=g)))
            elif name.endswith("dt_bias"):
                p.copy_(torch.empty_like(p).uniform_(-1, 1, generator=g))
            elif "norm" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)
            elif "embed" in name:
                p.copy_(torch.randn(p.shape, generator=g))
            else:
                fan_in = p.shape[-1]
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(fan_in))


def prompts():
    a = [(i * 7 + 3) % 250 for i in range(80)]  # > 64-token chunk, > indexer budget
    b = [(i * 13 + 5) % 250 for i in range(30)]
    b[9] = EOS  # n-gram windows restart after eos
    b[20] = EOS
    c = [17, 42]  # shorter than the n-gram context
    return [a, b, c]


def greedy(model, prompt, n):
    """Greedy tokens and, per step, the top-1 / top-2 logit margin (float64, so HF's cached and
    uncached paths agree; a comparison treats a divergence at a tiny margin as a tie)."""
    m64 = model.to(torch.float64)
    m64.config._experts_implementation = "eager"  # grouped_mm has no float64
    with torch.no_grad():
        # an explicit mask: without it generate treats every token equal to pad_token_id as padding
        out = m64.generate(torch.tensor([prompt]), attention_mask=torch.ones(1, len(prompt), dtype=torch.long),
                           max_new_tokens=n, min_new_tokens=n, do_sample=False, eos_token_id=None,
                           pad_token_id=0)
        toks = out[0, len(prompt):].tolist()
        logits = m64(torch.tensor([prompt + toks])).logits[0, len(prompt) - 1:-1]
    top = logits.topk(2, dim=-1).values
    model.to(torch.float32)
    return {"tokens": toks, "margins": [round(float(x), 5) for x in top[:, 0] - top[:, 1]]}


def fp8_rows(w: torch.Tensor):
    """Per-tensor FP8 e4m3 with a scalar scale, as the release stores the PLE tables."""
    scale = w.abs().max().clamp(min=1e-12) / 448.0
    q = (w / scale).to(torch.float8_e4m3fn)
    return q, scale.reshape(()).float(), q.float() * scale


def release_layout(model, cfg, out_dir, text_cfg_json, tok_from):
    """Write `model` in the release layout; returns the HF state dict with the FP8 PLE values."""
    state = model.state_dict()
    tensors, deq = {}, {}
    inter = cfg.moe_intermediate_size
    for name, t in state.items():
        if name == "lm_head.weight":
            tensors[name] = t
            continue
        new = name.replace("model.", "model.language_model.", 1)
        if name.endswith("mlp.experts.gate_up_proj"):
            base = new[: -len("gate_up_proj")]
            for e in range(cfg.num_experts):
                tensors[f"{base}{e}.gate_proj.weight"] = t[e, :inter]
                tensors[f"{base}{e}.up_proj.weight"] = t[e, inter:]
        elif name.endswith("mlp.experts.down_proj"):
            base = new[: -len("down_proj")]
            for e in range(cfg.num_experts):
                tensors[f"{base}{e}.down_proj.weight"] = t[e]
        elif name.endswith("ngram_embedding.weight"):
            q, scale, d = fp8_rows(t)
            parts = cfg.split_ngram_parts
            assert t.shape[0] % parts == 0
            rows = t.shape[0] // parts
            stem = new[: -len("weight")]
            for i in range(parts):
                tensors[f"{stem}shard_{i}.weight"] = q[i * rows:(i + 1) * rows]
            tensors[f"{stem}weight_scale"] = scale
            deq[name] = d
        elif name.endswith(("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")):
            continue  # buffers recomputed from the config
        else:
            tensors[new] = t
    os.makedirs(out_dir, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(out_dir, "model.safetensors"))
    wrapper = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
               "tie_word_embeddings": False, "text_config": text_cfg_json}
    json.dump(wrapper, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    copy_tokenizer(tok_from, out_dir)
    return deq


def copy_tokenizer(src, dst):
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(src, f), dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-from", required=True, help="dir with tokenizer.json (vocab 263)")
    ap.add_argument("--max-tokens", type=int, default=20)
    args = ap.parse_args()
    torch.manual_seed(0)
    ref = {"prompts": prompts()}

    variants = {
        "dense": tiny_config(),
        # the release's settings: sigmoid Gated-DeltaNet output gate, PLE, a small QSA budget
        "full": tiny_config(ple_layer_ids=[1, 3], indexer_budget=16, output_gate_type="sigmoid"),
    }
    models = {}
    for name, cfg in variants.items():
        model = Qwen4ExpForCausalLM(cfg).eval()
        randomize(model, 1 if name == "dense" else 2)
        d = os.path.join(args.out, name)
        model.save_pretrained(d, safe_serialization=True)
        copy_tokenizer(args.tokenizer_from, d)
        ref[name] = [greedy(model, p, args.max_tokens) for p in ref["prompts"]]
        models[name] = (model, cfg, d)

    model, cfg, full_dir = models["full"]
    text = json.load(open(os.path.join(full_dir, "config.json")))
    deq = release_layout(model, cfg, os.path.join(args.out, "release"), text, args.tokenizer_from)
    with torch.no_grad():
        sd = model.state_dict()
        for name, d in deq.items():
            sd[name].copy_(d)
    ref["release"] = [greedy(model, p, args.max_tokens) for p in ref["prompts"]]

    json.dump(ref, open(os.path.join(args.out, "reference.json"), "w"))
    print(json.dumps({k: v for k, v in ref.items() if k != "prompts"}))


if __name__ == "__main__":
    main()
