"""Parallel item-out leave-one-out probe (multiprocessing version).

Identical semantics to ``run_probe_loocv.py`` — same output rows, same seed
convention (``seed + fold_idx`` per fold), same reproducibility. The only
difference: per-(layer, mode) LOOCV is dispatched to a process pool so
many-core machines are used efficiently instead of training 6,720 probes
serially on a single thread.

Architecture:
    1. Main process loads model on GPU.
    2. For each layer: ``build_examples`` runs on GPU once, producing one
       CPU-resident example dict per item (tensors already ``.cpu().float()``).
    3. Each (layer, mode) pair is submitted to a ProcessPoolExecutor with
       spawn context (fork + CUDA parent = trouble).
    4. Workers run leave-one-item-out LOOCV on CPU only, return rows.
    5. Main process writes JSONL in deterministic (layer, mode, item) order.

Worker env sets OMP_NUM_THREADS=MKL_NUM_THREADS=1 to avoid thread oversub.

CLI mirrors ``run_probe_loocv.py`` plus ``--workers`` (default = nproc // 4,
capped to 32). Output JSONL is byte-identical to the serial version when
written in the same iteration order.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
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


def _to_numpy_examples(examples):
    """Convert CPU-tensor examples to plain numpy so multiprocessing pickle
    bypasses torch's special reduction (which exhausts fds across workers)."""
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


def _loocv_for_layer_mode(task):
    """Run leave-one-item-out LOOCV for a single (layer, mode).

    Receives numpy-encoded examples (pickled across processes without torch
    reduction). Returns ``(layer_idx, mode, rows)`` so the main process can
    re-order.
    """
    np_examples, mode, epochs, lr, seed, layer_idx = task
    examples = _from_numpy_examples(np_examples)
    if not examples:
        return layer_idx, mode, []
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
    return layer_idx, mode, rows


def _worker_init():
    """Cap BLAS threads to 1 inside each worker (avoids oversubscription)."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    torch.set_num_threads(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel item-out LOOCV probe.")
    parser.add_argument("model", choices=sorted(MODEL_HF_MAP))
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument("--language", default="all")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype",
                        choices=["auto", "float32", "float16", "bfloat16"],
                        default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--layers", nargs="*", type=int, default=None,
                        help="Layer indices. If unset, scans all layers.")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--workers", type=int, default=None,
                        help="Process pool size. Default: min(nproc//2, 32).")
    return parser.parse_args()


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

    n_workers = args.workers or min(max(1, mp.cpu_count() // 2), 32)
    print(f"[parallel] model={args.model} layers={len(layers)} modes={len(FEATURE_MODES)} "
          f"seeds=1 workers={n_workers}")

    # Pre-extract examples per layer on GPU (main process), keep on CPU.
    examples_per_layer: dict[int, list] = {}
    for layer_idx in layers:
        examples = build_examples(model, tokenizer, items, layer_idx, device)
        # Encode tensors as numpy so worker pickle avoids torch reduction
        # (which leaks file descriptors across spawn workers).
        examples_per_layer[layer_idx] = _to_numpy_examples(examples)
    # Free GPU memory before spawning workers (workers shouldn't touch GPU).
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    tasks = [
        (examples_per_layer[L], mode, args.epochs, args.lr, args.seed, L)
        for L in layers for mode in FEATURE_MODES
    ]

    # Spawn (not fork): safe with PyTorch + multiprocessing.
    ctx = mp.get_context("spawn")
    results: dict[tuple[int, str], list] = {}
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                             initializer=_worker_init) as ex:
        futures = {ex.submit(_loocv_for_layer_mode, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            layer_idx, mode, rows = fut.result()
            results[(layer_idx, mode)] = rows
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] layer {layer_idx} mode {mode} "
                      f"({len(rows)} rows)")

    # Write in deterministic order (layer, mode) so output is byte-stable.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for L in layers:
            for mode in FEATURE_MODES:
                for row in results[(L, mode)]:
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
