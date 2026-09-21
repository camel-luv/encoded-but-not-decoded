"""Parallel verb-out cross-validation (multiprocessing version).

Identical semantics to ``run_verb_out_cv.py`` — same output rows, same seed
convention (``seed + fold_idx`` for item-out, ``seed + fold_idx * 13`` for
verb-out), same reproducibility. Per-(layer, mode, seed) double CV is
dispatched to a process pool for efficient use of many-core machines.

Architecture mirrors ``run_probe_loocv_parallel.py``:
    1. Main process loads model on GPU.
    2. For each layer: ``build_examples`` runs on GPU once.
    3. Each (layer, mode, seed) is dispatched to ProcessPoolExecutor (spawn).
    4. Workers run item_out_loocv + verb_out_cv on CPU, return (layer, mode,
       seed, rows).
    5. Main process writes JSONL in deterministic (layer, mode, seed, ctype)
       order.

Worker env sets OMP_NUM_THREADS=1 to avoid thread oversubscription. Examples
are encoded as numpy so pickle bypasses torch reduction (fd exhaustion fix).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _to_numpy_examples(examples):
    """Convert CPU-tensor examples to plain numpy (pickle-friendly)."""
    import numpy as np
    out = []
    for ex in examples:
        out.append({
            **{k: v for k, v in ex.items() if k not in ("a", "b", "v", "controller_label")},
            "a": ex["a"].numpy(),
            "b": ex["b"].numpy(),
            "v": ex["v"].numpy(),
            "controller_label": ex["controller_label"].numpy(),
        })
    return out


def _from_numpy_examples(examples):
    """Inverse: rebuild tensor examples inside the worker."""
    out = []
    for ex in examples:
        out.append({
            **{k: v for k, v in ex.items() if k not in ("a", "b", "v", "controller_label")},
            "a": torch.from_numpy(ex["a"]),
            "b": torch.from_numpy(ex["b"]),
            "v": torch.from_numpy(ex["v"]),
            "controller_label": torch.from_numpy(ex["controller_label"]),
        })
    return out


def _item_out_loocv(examples, mode, epochs, lr, seed):
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


def _verb_out_cv(examples, mode, epochs, lr, seed):
    if not examples:
        return [], {}
    by_verb = defaultdict(list)
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


def _aggregate(preds, control_type):
    subset = [p for p in preds if p["control_type"] == control_type]
    if not subset:
        return float("nan"), 0
    correct = sum(1 for p in subset if p["correct"])
    return correct / len(subset), len(subset)


def _run_one_unit(task):
    """Run item_out + verb_out for one (layer, mode, seed) on CPU only."""
    np_examples, mode, seed, epochs, lr, layer_idx = task
    examples = _from_numpy_examples(np_examples)
    ioo_preds = _item_out_loocv(examples, mode, epochs, lr, seed)
    voo_preds, verb_groups = _verb_out_cv(examples, mode, epochs, lr, seed)
    rows = []
    for ctype in ("subject_control", "object_control"):
        ioo_acc, ioo_n = _aggregate(ioo_preds, ctype)
        voo_acc, voo_n = _aggregate(voo_preds, ctype)
        rows.append({
            "layer": layer_idx,
            "mode": mode,
            "control_type": ctype,
            "seed": seed,
            "item_out_loocv_acc": ioo_acc,
            "item_out_loocv_n": ioo_n,
            "verb_out_cv_acc": voo_acc,
            "verb_out_cv_n": voo_n,
            "verb_groups": verb_groups,
        })
    return layer_idx, mode, seed, rows


def _worker_init():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    torch.set_num_threads(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel verb-out CV vs item-out LOOCV.")
    parser.add_argument("model", choices=sorted(MODEL_HF_MAP))
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", nargs="*", type=int, default=None)
    parser.add_argument("--modes", nargs="*", default=list(FEATURE_MODES),
                        choices=list(FEATURE_MODES))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype",
                        choices=["auto", "float32", "float16", "bfloat16"],
                        default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--workers", type=int, default=None,
                        help="Process pool size. Default: min(nproc//2, 32).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = list(iter_items(args.benchmark_file))

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    source, tokenizer, model = load_model_and_tokenizer(
        args.model, args.model_root, device=device, dtype=dtype,
    )
    layers = resolve_layers_from_model(model, args.layers)

    n_workers = args.workers or min(max(1, mp.cpu_count() // 2), 32)
    n_tasks = len(layers) * len(args.modes) * len(args.seeds)
    print(f"[parallel] model={args.model} layers={len(layers)} modes={len(args.modes)} "
          f"seeds={len(args.seeds)} total_units={n_tasks} workers={n_workers}")

    # Pre-extract examples per layer on GPU, encode as numpy for pickling.
    examples_per_layer = {}
    for layer_idx in layers:
        examples = build_examples(model, tokenizer, items, layer_idx, device)
        examples_per_layer[layer_idx] = _to_numpy_examples(examples)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    tasks = [
        (examples_per_layer[L], mode, seed, args.epochs, args.lr, L)
        for L in layers
        for mode in args.modes
        for seed in args.seeds
    ]

    ctx = mp.get_context("spawn")
    results: dict[tuple[int, str, int], list] = {}
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                             initializer=_worker_init) as ex:
        futures = {ex.submit(_run_one_unit, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            layer_idx, mode, seed, rows = fut.result()
            results[(layer_idx, mode, seed)] = rows
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] layer {layer_idx} mode {mode} seed {seed}")

    # Write in deterministic order; field order matches run_verb_out_cv.py
    # (model + model_source first) so the two scripts produce byte-identical
    # output for the same arguments.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for L in layers:
            for mode in args.modes:
                for seed in args.seeds:
                    for row in results[(L, mode, seed)]:
                        ordered = {
                            "model": args.model,
                            "model_source": source,
                            "layer": row["layer"],
                            "mode": row["mode"],
                            "control_type": row["control_type"],
                            "seed": row["seed"],
                            "item_out_loocv_acc": row["item_out_loocv_acc"],
                            "item_out_loocv_n": row["item_out_loocv_n"],
                            "verb_out_cv_acc": row["verb_out_cv_acc"],
                            "verb_out_cv_n": row["verb_out_cv_n"],
                            "verb_groups": row["verb_groups"],
                        }
                        fh.write(json.dumps(ordered, ensure_ascii=False) + "\n")
                        written += 1
    print(f"Wrote {written} records to {args.out}")


if __name__ == "__main__":
    main()
