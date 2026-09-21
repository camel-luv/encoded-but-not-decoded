"""Item-out leave-one-out probe for controller recoverability.

For each layer, extract hidden states at three token positions per item
(candidate-A, candidate-B, lower-predicate verb), train a linear probe under
five feature modes (``ab``, ``av``, ``bv``, ``abv``, ``v``), and use
leave-one-item-out cross-validation. Output JSONL has one row per
(item, layer, feature mode).

The paper reports best-(layer, mode) probe accuracy. Run this script for the
desired layer set, then take the maximum accuracy across (layer, mode) groups
in post-processing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import (
    FEATURE_MODES,
    MODEL_HF_MAP,
    build_examples,
    feature_vector,
    iter_items,
    load_model_and_tokenizer,
    resolve_device,
    resolve_dtype,
    resolve_layers_from_model,
    train_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise controller probe (item-out LOOCV).")
    parser.add_argument("model", choices=sorted(MODEL_HF_MAP))
    parser.add_argument("--benchmark-file", type=Path, required=True)
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--layers", nargs="*", type=int, default=None,
        help="Layer indices. If unset, scans all layers (0 = embeddings).",
    )
    parser.add_argument(
        "--seed", type=int, default=1729,
        help="Probe-training seed (paper uses 1729, 2718, 3141 for robustness).",
    )
    return parser.parse_args()


def loocv_predictions(examples, mode, epochs, lr, seed, layer_idx):
    """Run leave-one-item-out CV; return list of per-item prediction rows."""
    if not examples:
        return []
    rows = []
    for fold_idx in range(len(examples)):
        train = examples[:fold_idx] + examples[fold_idx + 1:]
        test = examples[fold_idx]
        probe = train_probe(train, mode, epochs, lr, seed=seed + fold_idx)
        with torch.no_grad():
            score = torch.sigmoid(probe(feature_vector(test, mode).unsqueeze(0))).item()
        pred = 1 if score >= 0.5 else 0
        gold = int(test["controller_label"].item())
        rows.append({
            "id": test["id"],
            "language": test["language"],
            "control_type": test["control_type"],
            "feature_mode": mode,
            "layer": layer_idx,
            "score": score,
            "pred": pred,
            "gold": gold,
            "correct": pred == gold,
        })
    return rows


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
    layers = resolve_layers_from_model(model, args.layers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for layer_idx in layers:
            examples = build_examples(model, tokenizer, items, layer_idx, device)
            for mode in FEATURE_MODES:
                rows = loocv_predictions(
                    examples, mode, args.epochs, args.lr, args.seed, layer_idx,
                )
                for row in rows:
                    row.update({
                        "model": args.model,
                        "model_source": source,
                        "seed": args.seed,
                    })
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_rows += 1
    print(f"Wrote {total_rows} rows to {args.out}")


if __name__ == "__main__":
    main()
