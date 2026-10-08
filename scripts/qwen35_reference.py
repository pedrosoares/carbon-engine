#!/usr/bin/env python3
# Copyright 2026 Carbonis Fundatus
# SPDX-License-Identifier: Apache-2.0

"""Build tiny random Qwen3.5-MoE checkpoints with HF transformers and record HF's greedy output.

Writes two checkpoints under OUT:
  OUT/hf   - saved by transformers.save_pretrained (text-only Qwen3_5MoeForCausalLM)
  OUT/fp8  - the layout of the real Qwen3.6-35B-A3B-FP8 release: multimodal config wrapper,
             `model.language_model.*` names, per-expert tensors and 16x16 block-FP8 linear weights
             with `weight_scale_inv`
and OUT/reference.json with the greedy tokens HF produces for each (fp8 is run on the
dequantized weights, i.e. what an exact FP8 loader must reproduce).

    python scripts/qwen35_reference.py --out /tmp/q35 --tokenizer-from /tmp/tiny-qwen3
"""
import argparse
import json
import math
import os
import shutil

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

BLOCK = 16


def tiny_config(vocab: int) -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        vocab_size=vocab, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        moe_intermediate_size=32, shared_expert_intermediate_size=48, num_experts=8,
        num_experts_per_tok=2, max_position_embeddings=4096, full_attention_interval=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
        eos_token_id=258, tie_word_embeddings=False,
    )


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
            elif "embed_tokens" in name:
                p.copy_(torch.randn(p.shape, generator=g))
            else:
                fan_in = p.shape[-1]
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(fan_in))


def greedy(model, prompt, n):
    with torch.no_grad():
        out = model.generate(torch.tensor([prompt]), attention_mask=torch.ones(1, len(prompt), dtype=torch.long),
                             max_new_tokens=n, min_new_tokens=n,
                             do_sample=False, eos_token_id=None, pad_token_id=0)
    return out[0, len(prompt):].tolist()


def quantize_block_fp8(w: torch.Tensor):
    rows, cols = w.shape
    sr, sc = math.ceil(rows / BLOCK), math.ceil(cols / BLOCK)
    padded = torch.zeros(sr * BLOCK, sc * BLOCK)
    padded[:rows, :cols] = w.float()
    blocks = padded.reshape(sr, BLOCK, sc, BLOCK)
    scale = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12) / 448.0
    q = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    deq = (q.float() * scale[:, None, :, None]).reshape(sr * BLOCK, sc * BLOCK)[:rows, :cols]
    return q.reshape(sr * BLOCK, sc * BLOCK)[:rows, :cols].contiguous(), scale, deq


QUANTIZED = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
             "in_proj_qkv", "in_proj_z", "out_proj")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-from", required=True, help="dir with tokenizer.json (vocab 263)")
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args()

    cfg = tiny_config(263)
    model = Qwen3_5MoeForCausalLM(cfg).eval()
    randomize(model, 0)
    prompt = [(i * 7 + 3) % 250 for i in range(70)]  # > one 64-token delta-rule chunk
    ref = {"prompt": prompt}

    hf_dir = os.path.join(args.out, "hf")
    model.save_pretrained(hf_dir, safe_serialization=True)
    ref["hf"] = greedy(model, prompt, args.max_tokens)

    # FP8 release layout
    state = model.state_dict()
    tensors = {}
    num_experts, inter = cfg.num_experts, cfg.moe_intermediate_size
    for name, t in state.items():
        if name == "lm_head.weight":
            tensors[name] = t
            continue
        new = name.replace("model.", "model.language_model.", 1)
        if name.endswith("mlp.experts.gate_up_proj"):
            base = new[: -len("gate_up_proj")]
            for e in range(num_experts):
                tensors[f"{base}{e}.gate_proj.weight"] = t[e, :inter]
                tensors[f"{base}{e}.up_proj.weight"] = t[e, inter:]
        elif name.endswith("mlp.experts.down_proj"):
            base = new[: -len("down_proj")]
            for e in range(num_experts):
                tensors[f"{base}{e}.down_proj.weight"] = t[e]
        else:
            tensors[new] = t
    deq_state = {}
    for name in list(tensors):
        if name.endswith(".weight") and any(f".{q}." in name for q in QUANTIZED) and tensors[name].dim() == 2:
            q, scale, deq = quantize_block_fp8(tensors[name])
            tensors[name] = q
            tensors[name + "_scale_inv"] = scale
            deq_state[name] = deq
    fp8_dir = os.path.join(args.out, "fp8")
    os.makedirs(fp8_dir, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(fp8_dir, "model.safetensors"))
    text = json.load(open(os.path.join(hf_dir, "config.json")))
    wrapper = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
        "tie_word_embeddings": False, "text_config": text,
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [BLOCK, BLOCK]},
    }
    json.dump(wrapper, open(os.path.join(fp8_dir, "config.json"), "w"), indent=1)

    # HF output on the dequantized weights
    with torch.no_grad():
        sd = model.state_dict()
        for name, deq in deq_state.items():
            hf_name = name.replace("model.language_model.", "model.", 1)
            if ".mlp.experts." in hf_name and ".shared_expert" not in hf_name:
                prefix, rest = hf_name.split(".mlp.experts.")
                e, proj, _ = rest.split(".")
                e = int(e)
                if proj == "down_proj":
                    sd[f"{prefix}.mlp.experts.down_proj"][e].copy_(deq)
                else:
                    off = 0 if proj == "gate_proj" else inter
                    sd[f"{prefix}.mlp.experts.gate_up_proj"][e, off:off + inter].copy_(deq)
            else:
                sd[hf_name].copy_(deq)
    ref["fp8"] = greedy(model, prompt, args.max_tokens)

    for d in (hf_dir, fp8_dir):
        for f in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copy(os.path.join(args.tokenizer_from, f), d)
    json.dump(ref, open(os.path.join(args.out, "reference.json"), "w"))
    print(json.dumps(ref))


if __name__ == "__main__":
    main()
