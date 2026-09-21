"""Activation patching: replace target's hidden state at layer ``L`` with a donor's.

For each (target, donor) pair, forward the donor's bare sentence to capture
hidden states at the positions implicated by ``--mode``. Then forward the
target's QA / Paraphrase prompt with a hook that REPLACES the target's
positions with the donor's cached vectors. Score QA / Paraphrase continuations.

Default donor policy ``same_lang_type`` restricts pairs to those sharing both
language and control type - the cleanest within-cell test.

The "causal-rescue lift" reported in the paper is the correct-donor rescue
rate minus the wrong-donor rescue rate. This script writes raw per-pair
outcomes; aggregate post-hoc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import (
    FEATURE_MODES,
    MODEL_HF_MAP,
    continuation_score,
    find_paraphrase_positions,
    find_qa_positions,
    find_target_token_index,
    get_transformer_layer_module,
    iter_items,
    load_model_and_tokenizer,
    resolve_device,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activation patching for control dependencies.")
    parser.add_argument("model", choices=sorted(MODEL_HF_MAP))
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--mode", choices=list(FEATURE_MODES), required=True)
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument(
        "--donor-policy",
        choices=["same_lang_type", "same_type", "all"],
        default="same_lang_type",
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=["qa", "paraphrase"], default=["qa", "paraphrase"],
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument(
        "--language", default="all",
    )
    return parser.parse_args()


def precompute_donor_hiddens(model, tokenizer, items, layer_idx, mode, device):
    donor = {}
    for item in items:
        ids = tokenizer(item["sentence"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        positions = {
            "a": find_target_token_index(tokenizer, item["sentence"], item["candidate_a"]),
            "b": find_target_token_index(tokenizer, item["sentence"], item["candidate_b"]),
            "v": find_target_token_index(tokenizer, item["sentence"], item["lower_predicate"]),
        }
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
            hs = out.hidden_states[layer_idx][0]
        donor[item["id"]] = {
            k: (hs[positions[k]].detach().cpu().float() if positions[k] is not None else None)
            for k in mode
        }
    return donor


def make_patching_hook(target_positions, donor_h, mode):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None
        h = h.clone()
        for k in mode:
            tpos = target_positions.get(k)
            dh = donor_h.get(k)
            if tpos is None or dh is None:
                continue
            h[0, tpos, :] = dh.to(h.device, h.dtype)
        if rest is None:
            return h
        return (h,) + rest
    return hook


def pair_passes_filter(target, donor, policy):
    if target["id"] == donor["id"]:
        return False
    if policy == "all":
        return True
    if target["control_type"] != donor["control_type"]:
        return False
    if policy == "same_type":
        return True
    return target["language"] == donor["language"]  # same_lang_type


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    source, tokenizer, model = load_model_and_tokenizer(
        args.model, args.model_root, device=device, dtype=dtype,
    )
    layer_module = get_transformer_layer_module(model, args.layer)

    items = [it for it in iter_items(args.benchmark_file) if "lower_predicate" in it]
    if args.language != "all":
        items = [it for it in items if it["language"] == args.language]

    donor_hiddens = precompute_donor_hiddens(
        model, tokenizer, items, args.layer, args.mode, device,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for target in items:
            target_positions = {
                "qa": find_qa_positions(tokenizer, target),
                "paraphrase": find_paraphrase_positions(tokenizer, target),
            }
            for donor in items:
                if not pair_passes_filter(target, donor, args.donor_policy):
                    continue
                if any(donor_hiddens[donor["id"]].get(k) is None for k in args.mode):
                    continue
                for task in args.tasks:
                    positions = target_positions[task]
                    if any(positions.get(k) is None for k in args.mode):
                        continue
                    if task == "qa":
                        prompt = target["qa_prompt"]
                        cont_a = target["qa_answer_a"]
                        cont_b = target["qa_answer_b"]
                    else:
                        prompt = target["paraphrase_prompt"]
                        cont_a = target["paraphrase_a"]
                        cont_b = target["paraphrase_b"]
                    hook = make_patching_hook(
                        positions, donor_hiddens[donor["id"]], args.mode,
                    )
                    handle = layer_module.register_forward_hook(hook)
                    try:
                        a = continuation_score(model, tokenizer, prompt, cont_a, device)
                        b = continuation_score(model, tokenizer, prompt, cont_b, device)
                    finally:
                        handle.remove()
                    pred_label = "a" if a > b else "b"
                    fh.write(json.dumps({
                        "model": args.model,
                        "model_source": source,
                        "target_id": target["id"],
                        "donor_id": donor["id"],
                        "target_lang": target["language"],
                        "donor_lang": donor["language"],
                        "target_ctrl": target["control_type"],
                        "donor_ctrl": donor["control_type"],
                        "task": task,
                        "layer": args.layer,
                        "feature_mode": args.mode,
                        "patched_a_score": a,
                        "patched_b_score": b,
                        "patched_margin_a_minus_b": a - b,
                        "patched_pred": pred_label,
                        "target_gold_label": target["gold_label"],
                        "patched_correct": pred_label == target["gold_label"],
                        "donor_gold_label": donor["gold_label"],
                    }, ensure_ascii=False) + "\n")
                    written += 1
    print(f"Wrote {written} rows to {args.out}")


if __name__ == "__main__":
    main()
