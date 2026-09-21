"""Directional steering: add ``alpha * sigma * unit(direction)`` to the residual stream.

For each (item, alpha, task) trial, register a forward hook on the target
transformer layer that adds a scaled probe-direction vector at the positions
implicated by the probe's feature mode, then re-score QA / Paraphrase
continuations. Output JSONL has one row per trial.

The probe direction is read from a JSON produced by :mod:`src.dump_probe_weights`.
The scalar ``sigma`` is computed once per position over all items at the
target layer so ``alpha`` is in units of "natural variation at that position".

Paper finding: linear additive steering is null at all tested magnitudes
(no flips on subject-control). Output is intended to support that null result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import (
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


DEFAULT_ALPHAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Directional steering experiment.")
    parser.add_argument("model", choices=sorted(MODEL_HF_MAP))
    parser.add_argument(
        "--probe-weights", type=Path, required=True,
        help="JSON written by src.dump_probe_weights.",
    )
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
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


def split_direction(weight, mode, hidden):
    """Split a probe weight vector into per-position blocks, unit-normalize each."""
    if weight.numel() != len(mode) * hidden:
        raise ValueError(
            f"weight length {weight.numel()} != len(mode)={len(mode)} * hidden={hidden}"
        )
    blocks = weight.view(len(mode), hidden)
    return {k: blocks[i] for i, k in enumerate(mode)}


def compute_sigmas(model, tokenizer, items, layer_idx, mode, device):
    """Per-position sigma at ``layer_idx``: mean over dimensions of std across items."""
    stacks = {key: [] for key in mode}
    for item in items:
        ids = tokenizer(item["sentence"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        positions = {
            "a": find_target_token_index(tokenizer, item["sentence"], item["candidate_a"]),
            "b": find_target_token_index(tokenizer, item["sentence"], item["candidate_b"]),
            "v": find_target_token_index(tokenizer, item["sentence"], item["lower_predicate"]),
        }
        if any(positions[k] is None for k in mode):
            continue
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
            hs = out.hidden_states[layer_idx][0]
        for k in mode:
            stacks[k].append(hs[positions[k]].detach().cpu().float())
    sigmas = {}
    for key, vecs in stacks.items():
        sigmas[key] = float(torch.stack(vecs).std(dim=0).mean().item()) if vecs else 0.0
    return sigmas


def make_steering_hook(positions, directions, sigmas, alpha, mode, gold_sign):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None
        h = h.clone()
        for k in mode:
            pos = positions[k]
            if pos is None:
                continue
            d = directions[k].to(h.device, h.dtype)
            sigma = sigmas[k]
            h[0, pos, :] = h[0, pos, :] + alpha * gold_sign * sigma * d
        if rest is None:
            return h
        return (h,) + rest
    return hook


def main() -> None:
    args = parse_args()

    payload = json.loads(args.probe_weights.read_text(encoding="utf-8"))
    layer = payload["layer"]
    mode = payload["feature_mode"]
    hidden = payload["hidden_size"]
    weight = torch.tensor(payload["weight"], dtype=torch.float32)
    if payload["model"] != args.model:
        raise ValueError(
            f"probe weights are for {payload['model']}, but --model={args.model}"
        )

    direction_split = split_direction(weight, mode, hidden)
    direction_norm = {
        k: (v / v.norm()) if v.norm() > 0 else v
        for k, v in direction_split.items()
    }

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    source, tokenizer, model = load_model_and_tokenizer(
        args.model, args.model_root, device=device, dtype=dtype,
    )
    layer_module = get_transformer_layer_module(model, layer)

    items = [it for it in iter_items(args.benchmark_file) if "lower_predicate" in it]
    if args.language != "all":
        items = [it for it in items if it["language"] == args.language]

    sigmas = compute_sigmas(model, tokenizer, items, layer, mode, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for item in items:
            qa_pos = find_qa_positions(tokenizer, item)
            para_pos = find_paraphrase_positions(tokenizer, item)
            gold_sign = 1.0 if item["gold_label"] == "a" else -1.0
            for task in args.tasks:
                positions = qa_pos if task == "qa" else para_pos
                if any(positions[k] is None for k in mode):
                    continue
                if task == "qa":
                    prompt = item["qa_prompt"]
                    cont_a = item["qa_answer_a"]
                    cont_b = item["qa_answer_b"]
                else:
                    prompt = item["paraphrase_prompt"]
                    cont_a = item["paraphrase_a"]
                    cont_b = item["paraphrase_b"]
                for alpha in args.alphas:
                    hook = make_steering_hook(
                        positions, direction_norm, sigmas, alpha, mode, gold_sign,
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
                        "id": item["id"],
                        "language": item["language"],
                        "control_type": item["control_type"],
                        "task": task,
                        "alpha": alpha,
                        "gold_sign": gold_sign,
                        "layer": layer,
                        "feature_mode": mode,
                        "candidate_a_score": a,
                        "candidate_b_score": b,
                        "margin_a_minus_b": a - b,
                        "predicted_label": pred_label,
                        "gold_label": item["gold_label"],
                        "correct": pred_label == item["gold_label"],
                    }, ensure_ascii=False) + "\n")
                    written += 1
    print(f"Wrote {written} rows to {args.out}")


if __name__ == "__main__":
    main()
