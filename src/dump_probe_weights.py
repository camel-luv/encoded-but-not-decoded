"""Dump trained linear probe weights for use as a steering direction.

Trains one probe on all benchmark items at a given (layer, feature mode),
then writes the weight vector + metadata to JSON for downstream use by
:mod:`src.run_steering`.

Two methods are supported:

* ``logreg`` - full-data fit of the same Adam/BCE probe used in :mod:`src.run_probe_loocv`
* ``diff_of_means`` - per-class mean difference (no optimizer, less overfit)

Convention: probe direction points toward "candidate_a is the controller"
because ``controller_label=1`` in the benchmark means gold = candidate_a.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from .common import (
    FEATURE_MODES,
    MODEL_HF_MAP,
    build_examples,
    feature_vector,
    iter_items,
    load_model_and_tokenizer,
    make_linear_probe,
    resolve_device,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump probe weights as a steering direction.")
    parser.add_argument("model", choices=sorted(MODEL_HF_MAP))
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--mode", choices=list(FEATURE_MODES), required=True)
    parser.add_argument(
        "--direction-method", choices=["logreg", "diff_of_means"], default="logreg",
    )
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--language", default="all",
    )
    return parser.parse_args()


def train_full_data(examples, mode, *, epochs, lr, seed):
    """Train one probe on all examples (no leave-one-out)."""
    if not examples:
        raise ValueError("No examples to train on.")
    input_dim = feature_vector(examples[0], mode).shape[0]
    torch.manual_seed(seed)
    probe = make_linear_probe(input_dim)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    feats = torch.stack([feature_vector(ex, mode) for ex in examples])
    labels = torch.stack([ex["controller_label"] for ex in examples])
    for _ in range(epochs):
        opt.zero_grad()
        logits = probe(feats)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(probe(feats)) > 0.5).float()
        train_acc = (preds == labels).float().mean().item()
    return probe, train_acc


def diff_of_means_direction(examples, mode):
    """Per-class mean-difference direction in feature space."""
    feats = torch.stack([feature_vector(ex, mode) for ex in examples])
    labels = torch.stack([ex["controller_label"] for ex in examples])
    direction = feats[labels == 1].mean(dim=0) - feats[labels == 0].mean(dim=0)
    centered = feats - feats.mean(dim=0)
    scores = centered @ direction
    sanity_acc = ((scores > 0).float() == labels).float().mean().item()
    return direction, sanity_acc


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    source, tokenizer, model = load_model_and_tokenizer(
        args.model, args.model_root, device=device, dtype=dtype,
    )
    items = [it for it in iter_items(args.benchmark_file) if "lower_predicate" in it]
    if args.language != "all":
        items = [it for it in items if it["language"] == args.language]

    examples = build_examples(model, tokenizer, items, args.layer, device)
    if not examples:
        raise RuntimeError("No examples produced - check benchmark file and tokenizer.")
    hidden_size = examples[0]["a"].shape[0]
    input_dim = feature_vector(examples[0], args.mode).shape[0]

    if args.direction_method == "logreg":
        probe, train_acc = train_full_data(
            examples, args.mode, epochs=args.epochs, lr=args.lr, seed=args.seed,
        )
        weight_tensor = probe.linear.weight.detach().cpu().squeeze(0)
        bias_val = probe.linear.bias.detach().cpu().item()
        method_label = "train_accuracy"
    else:
        weight_tensor, train_acc = diff_of_means_direction(examples, args.mode)
        bias_val = 0.0
        method_label = "diff_of_means_threshold_0_accuracy"

    payload = {
        "model": args.model,
        "model_source": source,
        "layer": args.layer,
        "feature_mode": args.mode,
        "direction_method": args.direction_method,
        "input_dim": input_dim,
        "hidden_size": hidden_size,
        "weight": weight_tensor.tolist(),
        "bias": bias_val,
        "weight_norm": float(weight_tensor.norm().item()),
        "n_train_items": len(examples),
        "train_accuracy": train_acc,
        "train_accuracy_label": method_label,
        "epochs": args.epochs if args.direction_method == "logreg" else None,
        "lr": args.lr if args.direction_method == "logreg" else None,
        "seed": args.seed,
        "language_filter": args.language,
        "items_used": [ex["id"] for ex in examples],
        "device": device,
        "dtype": str(dtype),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: {args.out} ({method_label}={train_acc:.4f}, input_dim={input_dim})")


if __name__ == "__main__":
    main()
