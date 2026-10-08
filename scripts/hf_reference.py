#!/usr/bin/env python3
# Copyright 2026 Carbonis Fundatus
# SPDX-License-Identifier: Apache-2.0

"""Greedy reference generation with Hugging Face transformers.

Prints the generated token ids (one JSON line) so they can be compared with
`carbon generate` / the `/generate` endpoint on the same checkpoint:

    python scripts/hf_reference.py --model DIR --prompt-ids 1,2,3 --max-tokens 16
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-ids", required=True, help="comma separated token ids")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.eval()
    ids = torch.tensor([[int(x) for x in args.prompt_ids.split(",")]])
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()
        out = model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_tokens,
            do_sample=False, eos_token_id=None, pad_token_id=0, min_new_tokens=args.max_tokens,
        )
    print(json.dumps({
        "output_ids": out[0, ids.shape[1]:].tolist(),
        "last_logits_head": [round(x, 5) for x in logits[:8].tolist()],
    }))


if __name__ == "__main__":
    main()
