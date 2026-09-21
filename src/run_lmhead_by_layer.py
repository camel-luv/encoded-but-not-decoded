"""Score control benchmark candidates via layerwise LM-head readout.

For each layer index ``L``, take the model's hidden state at ``L``, optionally
apply the final layer-norm, and project through the LM head. Score candidate
continuations under that projection. The output JSONL contains one row per
(item, layer).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import (
    MODEL_HF_MAP,
    iter_items,
    load_model_and_tokenizer,
    resolve_device,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layerwise LM-head readout.")
    parser.add_argument("model", choices=sorted(MODEL_HF_MAP))
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument("--task", choices=["qa", "paraphrase", "bare"], default="qa")
    parser.add_argument(
        "--labelled", action="store_true",
        help="With --task bare, use the bare_prompt_labelled variant (Continuation: label).",
    )
    parser.add_argument(
        "--language", default="all",
    )
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument(
        "--layers", nargs="*", type=int, default=None,
        help="Specific layer indices. If unset, scans all layers (0 = embeddings).",
    )
    return parser.parse_args()


def get_final_norm_module(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "base_model") and hasattr(model.base_model, "norm"):
        return model.base_model.norm
    return None


def get_output_embeddings(model):
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("Model does not expose output embeddings / lm_head.")
    return lm_head


def resolve_layers(model, requested_layers):
    if requested_layers:
        return list(requested_layers)
    n_hidden = getattr(model.config, "num_hidden_layers", None)
    if n_hidden is None:
        text_config = getattr(model.config, "text_config", None)
        n_hidden = getattr(text_config, "num_hidden_layers", None)
    if n_hidden is None:
        raise ValueError("Cannot infer num_hidden_layers from model config.")
    return list(range(n_hidden + 1))


def candidate_texts(item, task):
    if task == "qa":
        return item["qa_answer_a"], item["qa_answer_b"]
    if task == "paraphrase":
        return item["paraphrase_a"], item["paraphrase_b"]
    return item["bare_a"], item["bare_b"]


def prompt_text(item, task, labelled=False):
    if task == "qa":
        return item["qa_prompt"]
    if task == "paraphrase":
        return item["paraphrase_prompt"]
    return item["bare_prompt_labelled"] if labelled else item["bare_prompt"]


def layerwise_continuation_score(
    model, tokenizer, prompt, continuation,
    *, layer_idx, device, final_norm, lm_head,
):
    prefix_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    full_ids = tokenizer(prompt + continuation, return_tensors="pt", add_special_tokens=False)
    prefix_len = prefix_ids["input_ids"].shape[1]
    full_input = full_ids["input_ids"].to(device)

    with torch.no_grad():
        out = model(input_ids=full_input, output_hidden_states=True)
        hs = out.hidden_states[layer_idx][0]

    if final_norm is not None:
        hs = final_norm(hs)
    logits = lm_head(hs)
    log_probs = torch.log_softmax(logits, dim=-1)
    target_ids = full_input[0, prefix_len:]
    score = 0.0
    for pos, tok_id in enumerate(target_ids, start=prefix_len):
        score += float(log_probs[pos - 1, tok_id].item())
    return score


def main() -> None:
    args = parse_args()
    items = list(iter_items(args.benchmark_file))
    if args.language != "all":
        items = [it for it in items if it["language"] == args.language]

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    source, tokenizer, model = load_model_and_tokenizer(
        args.model, args.model_root, device=device, dtype=dtype,
    )

    final_norm = get_final_norm_module(model)
    lm_head = get_output_embeddings(model)
    layers = resolve_layers(model, args.layers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for layer_idx in layers:
            for item in items:
                prompt = prompt_text(item, args.task, labelled=args.labelled)
                cand_a, cand_b = candidate_texts(item, args.task)
                a = layerwise_continuation_score(
                    model, tokenizer, prompt, cand_a,
                    layer_idx=layer_idx, device=device,
                    final_norm=final_norm, lm_head=lm_head,
                )
                b = layerwise_continuation_score(
                    model, tokenizer, prompt, cand_b,
                    layer_idx=layer_idx, device=device,
                    final_norm=final_norm, lm_head=lm_head,
                )
                pred_label = "a" if a > b else "b"
                pred_host = item["candidate_a"] if pred_label == "a" else item["candidate_b"]
                ok = pred_label == item["gold_label"]
                fh.write(json.dumps({
                    **item,
                    "task": args.task,
                    "prompt_variant": "labelled" if args.labelled else "unlabelled",
                    "model": args.model,
                    "model_source": source,
                    "layer": layer_idx,
                    "candidate_a_score": a,
                    "candidate_b_score": b,
                    "predicted_label": pred_label,
                    "predicted_host": pred_host,
                    "margin_a_minus_b": a - b,
                    "correct": ok,
                }, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items) * len(layers)} rows to {args.out}")


if __name__ == "__main__":
    main()
