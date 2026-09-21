"""Verb-out cross-validation control (paper Appendix H, Plan B).

Tests whether the probe ceiling on subject-control reflects encoding of the
control relation or memorization of matrix-verb identity. For each unique
matrix predicate ``V``:

* train fold: items where ``matrix_predicate != V``
* test fold:  items where ``matrix_predicate == V``

Training pools mix subject- and object-control items so the probe still has a
non-trivial decision boundary. Predictions are aggregated by control type post
hoc. A small gap between item-out LOOCV and verb-out CV on subject-control
indicates the probe is not relying on verb identity.

The same script also runs item-out LOOCV at the same (layer, mode, seed) so
the two protocols can be compared directly. Output JSONL has one row per
(model, layer, mode, control_type, seed).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


SEEDS = (1729, 2718, 3141)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verb-out CV vs item-out LOOCV.")
    parser.add_argument("model", choices=sorted(MODEL_HF_MAP))
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--layers", nargs="*", type=int, default=None,
        help="Layers to evaluate. If unset, scans all layers.",
    )
    parser.add_argument(
        "--modes", nargs="*", default=list(FEATURE_MODES), choices=list(FEATURE_MODES),
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    return parser.parse_args()


def item_out_loocv(examples, mode, epochs, lr, seed):
    """Leave-one-item-out CV on the mixed-control-type pool."""
    if not examples:
        return []
    preds = []
    for fold_idx in range(len(examples)):
        train = examples[:fold_idx] + examples[fold_idx + 1:]
        test = examples[fold_idx]
        probe = train_probe(train, mode, epochs, lr, seed=seed + fold_idx)
        with torch.no_grad():
            score = torch.sigmoid(probe(feature_vector(test, mode).unsqueeze(0))).item()
        pred = 1 if score >= 0.5 else 0
        gold = int(test["controller_label"].item())
        preds.append({
            "control_type": test["control_type"],
            "score": score,
            "pred": pred,
            "gold": gold,
            "correct": pred == gold,
        })
    return preds


def verb_out_cv(examples, mode, epochs, lr, seed):
    """Leave-one-matrix-predicate-out CV."""
    if not examples:
        return [], {}
    by_verb: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        verb = ex.get("matrix_predicate") or "UNKNOWN"
        by_verb[verb].append(ex)
    preds = []
    for fold_idx, (held_verb, held_items) in enumerate(sorted(by_verb.items())):
        train = [ex for ex in examples if (ex.get("matrix_predicate") or "UNKNOWN") != held_verb]
        if not train:
            continue
        probe = train_probe(train, mode, epochs, lr, seed=seed + fold_idx * 13)
        with torch.no_grad():
            for test_ex in held_items:
                score = torch.sigmoid(probe(feature_vector(test_ex, mode).unsqueeze(0))).item()
                pred = 1 if score >= 0.5 else 0
                gold = int(test_ex["controller_label"].item())
                preds.append({
                    "control_type": test_ex["control_type"],
                    "score": score,
                    "pred": pred,
                    "gold": gold,
                    "correct": pred == gold,
                    "held_verb": held_verb,
                })
    return preds, {v: len(items) for v, items in by_verb.items()}


def aggregate(preds, control_type):
    subset = [p for p in preds if p["control_type"] == control_type]
    if not subset:
        return float("nan"), 0
    correct = sum(1 for p in subset if p["correct"])
    return correct / len(subset), len(subset)


def main() -> None:
    args = parse_args()
    items = list(iter_items(args.benchmark_file))

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    source, tokenizer, model = load_model_and_tokenizer(
        args.model, args.model_root, device=device, dtype=dtype,
    )
    layers = resolve_layers_from_model(model, args.layers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for layer_idx in layers:
            examples = build_examples(model, tokenizer, items, layer_idx, device)
            for mode in args.modes:
                for seed in args.seeds:
                    ioo_preds = item_out_loocv(examples, mode, args.epochs, args.lr, seed)
                    voo_preds, verb_groups = verb_out_cv(
                        examples, mode, args.epochs, args.lr, seed,
                    )
                    for ctype in ("subject_control", "object_control"):
                        ioo_acc, ioo_n = aggregate(ioo_preds, ctype)
                        voo_acc, voo_n = aggregate(voo_preds, ctype)
                        fh.write(json.dumps({
                            "model": args.model,
                            "model_source": source,
                            "layer": layer_idx,
                            "mode": mode,
                            "control_type": ctype,
                            "seed": seed,
                            "item_out_loocv_acc": ioo_acc,
                            "item_out_loocv_n": ioo_n,
                            "verb_out_cv_acc": voo_acc,
                            "verb_out_cv_n": voo_n,
                            "verb_groups": verb_groups,
                        }, ensure_ascii=False) + "\n")
                        written += 1
    print(f"Wrote {written} records to {args.out}")


if __name__ == "__main__":
    main()
