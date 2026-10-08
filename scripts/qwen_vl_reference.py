#!/usr/bin/env python3
# Copyright 2026 Carbonis Fundatus
# SPDX-License-Identifier: Apache-2.0

"""Build tiny random image-input checkpoints (Qwen3.5-MoE and Qwen3.8-Flash-Next layouts, each with
a 2-block vision tower) with HF transformers and record what HF computes for one test image.

    carbon tiny-model --arch qwen3 --out /tmp/tiny-tok     # any dir with tokenizer.json (vocab 263)
    python scripts/qwen_vl_reference.py --out /tmp/qvl --tokenizer-from /tmp/tiny-tok

Writes under OUT, per variant (`q35`, `q38`, `q38s` = q38 with an 8-token QSA budget), a checkpoint dir (multimodal config wrapper,
`model.visual.*` + `model.language_model.*` tensors, preprocessor_config.json) and
OUT/<variant>.safetensors with
  pixel_values  [N, C*T*P*P]  what HF's image processor makes of OUT/image.png
  image_embeds  [N/4, H]      the vision tower's output for them (float64 run)
  position_ids  [3, L]        mRoPE positions of the prompt (t, h, w)
and OUT/reference.json: per variant the prompt ids (image placeholders expanded), the image grid
and HF's greedy tokens with float64 top-1/top-2 margins.
"""
import argparse
import json
import math
import os
import shutil

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import (
    Qwen2VLImageProcessor,
    Qwen3_5MoeConfig,
    Qwen3_5MoeForConditionalGeneration,
    Qwen4ExpConfig,
    Qwen4ExpForConditionalGeneration,
)

VOCAB, EOS = 263, 258
IMAGE, VIDEO, VSTART, VEND = 259, 260, 261, 262
PATCH, MERGE, TEMPORAL = 4, 2, 2


def vision_config(out_hidden: int) -> dict:
    return dict(depth=2, hidden_size=32, num_heads=2, intermediate_size=48, patch_size=PATCH,
                temporal_patch_size=TEMPORAL, spatial_merge_size=MERGE, num_position_embeddings=16,
                out_hidden_size=out_hidden, hidden_act="gelu_pytorch_tanh")


def ids(**kw):
    return dict(image_token_id=IMAGE, video_token_id=VIDEO, vision_start_token_id=VSTART,
                vision_end_token_id=VEND, **kw)


def q35_config():
    text = dict(
        vocab_size=VOCAB, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        moe_intermediate_size=32, shared_expert_intermediate_size=48, num_experts=8,
        num_experts_per_tok=2, max_position_embeddings=4096, full_attention_interval=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True},
        eos_token_id=EOS, tie_word_embeddings=False,
    )
    return Qwen3_5MoeConfig(text_config=text, vision_config=vision_config(64), **ids())


def q38_config(budget=4096):
    text = dict(
        vocab_size=VOCAB, hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        moe_intermediate_size=16, shared_expert_intermediate_size=24, num_experts=8,
        num_experts_per_tok=2, max_position_embeddings=4096, full_attention_interval=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        eos_token_id=EOS, tie_word_embeddings=False, hc_count=3, hc_lowrank=12,
        indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=16,
        indexer_budget=budget, indexer_compress_ratio=4, ple_layer_ids=[],
    )
    return Qwen4ExpConfig(text_config=text, vision_config=vision_config(32), **ids())


def randomize(model: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p).uniform_(1, 8, generator=g)))
            elif name.endswith("dt_bias"):
                p.copy_(torch.empty_like(p).uniform_(-1, 1, generator=g))
            elif "norm" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.2 + (1.0 if "visual" in name and name.endswith("weight") else 0.0))
            elif "embed" in name:
                p.copy_(torch.randn(p.shape, generator=g))
            elif name.endswith("bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
            else:
                fan_in = int(np.prod(p.shape[1:]))
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(fan_in))


def processor():
    # factor = patch * merge = 8; tiny pixel bounds so the test image is not rescaled much
    return Qwen2VLImageProcessor(size={"shortest_edge": 64, "longest_edge": 64 * 64}, patch_size=PATCH,
                                 temporal_patch_size=TEMPORAL, merge_size=MERGE,
                                 image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])


def test_image(path):
    rng = np.random.default_rng(7)
    # smooth gradients plus noise: resampling differences stay small
    y, x = np.mgrid[0:28, 0:40]
    img = np.stack([x * 6, y * 9, (x + y) * 3], -1) + rng.integers(0, 40, (28, 40, 3))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
    return Image.open(path).convert("RGB")


def greedy(model, inputs, n):
    m64 = model.to(torch.float64)
    m64.config._experts_implementation = "eager"
    kw = {k: (v.to(torch.float64) if v.is_floating_point() else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = m64.generate(**kw, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                           eos_token_id=None, pad_token_id=0)
        prompt_len = kw["input_ids"].shape[1]
        toks = out[0, prompt_len:].tolist()
        full = dict(kw)
        full["input_ids"] = out
        full["attention_mask"] = torch.ones_like(out)
        full["mm_token_type_ids"] = (out == IMAGE).long()
        logits = m64(**full).logits[0, prompt_len - 1:-1]
    top = logits.topk(2, dim=-1).values
    return {"tokens": toks, "margins": [round(float(x), 5) for x in top[:, 0] - top[:, 1]]}


def save_checkpoint(model, cfg, out_dir, tok_from, proc):
    os.makedirs(out_dir, exist_ok=True)
    state = model.state_dict()
    tensors = {}
    for name, t in state.items():
        if name.endswith(("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")):
            continue
        tensors[name] = t.contiguous()
    save_file(tensors, os.path.join(out_dir, "model.safetensors"))
    cfg.save_pretrained(out_dir)
    proc.save_pretrained(out_dir)
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(tok_from, f), out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-from", required=True)
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    image = test_image(os.path.join(args.out, "image.png"))
    proc = processor()
    pix = proc(images=[image], return_tensors="pt")
    pixel_values, grid = pix["pixel_values"], pix["image_grid_thw"]
    n_img = int(grid.prod()) // (MERGE * MERGE)
    before = [(i * 7 + 3) % 250 for i in range(12)]
    after = [(i * 11 + 5) % 250 for i in range(9)]
    prompt = before + [VSTART] + [IMAGE] * n_img + [VEND] + after
    ref = {"prompt": prompt, "grid": grid[0].tolist(), "image_tokens": n_img}

    for name, (cfg, cls, seed) in {
        "q35": (q35_config(), Qwen3_5MoeForConditionalGeneration, 1),
        "q38": (q38_config(), Qwen4ExpForConditionalGeneration, 2),
        # QSA selects 2 of the prompt's 11 blocks: pooled keys rotate at 3-D positions
        "q38s": (q38_config(8), Qwen4ExpForConditionalGeneration, 3),
    }.items():
        model = cls(cfg).eval()
        randomize(model, seed)
        model.to(torch.float32)
        save_checkpoint(model, cfg, os.path.join(args.out, name), args.tokenizer_from, proc)
        input_ids = torch.tensor([prompt])
        inputs = dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                      pixel_values=pixel_values, image_grid_thw=grid,
                      mm_token_type_ids=(input_ids == IMAGE).long())
        with torch.no_grad():
            m64 = model.to(torch.float64)
            emb = m64.model.visual(pixel_values.to(torch.float64), grid_thw=grid).pooler_output
            pos, delta = m64.model.get_rope_index(input_ids, inputs["mm_token_type_ids"], image_grid_thw=grid)
        save_file({"pixel_values": pixel_values.float().contiguous(),
                   "image_embeds": emb.float().contiguous(),
                   "position_ids": pos[:, 0].to(torch.int64).contiguous()},
                  os.path.join(args.out, f"{name}.safetensors"))
        ref[name] = greedy(model, inputs, args.max_tokens)
        ref[name]["rope_delta"] = int(delta[0, 0])
    json.dump(ref, open(os.path.join(args.out, "reference.json"), "w"))
    print(json.dumps({k: v for k, v in ref.items() if k != "prompt"}))


if __name__ == "__main__":
    main()
