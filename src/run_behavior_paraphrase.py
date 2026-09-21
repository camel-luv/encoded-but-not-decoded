"""Run the paraphrase-style behavioral evaluation on the trilingual control benchmark.

Same interface as :mod:`src.run_behavior_qa` but uses ``paraphrase_prompt`` /
``paraphrase_a`` / ``paraphrase_b`` fields from each item.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import (
    MODEL_HF_MAP,
    continuation_score,
    iter_items,
    load_model_and_tokenizer,
    resolve_device,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paraphrase-style behavioral evaluation.")
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for item in items:
            a_score = continuation_score(
                model, tokenizer, item["paraphrase_prompt"], item["paraphrase_a"], device,
            )
            b_score = continuation_score(
                model, tokenizer, item["paraphrase_prompt"], item["paraphrase_b"], device,
            )
            pred_label = "a" if a_score > b_score else "b"
            pred_host = item["candidate_a"] if pred_label == "a" else item["candidate_b"]
            ok = pred_label == item["gold_label"]
            fh.write(json.dumps({
                **item,
                "task": "paraphrase",
                "model": args.model,
                "model_source": source,
                "candidate_a_score": a_score,
                "candidate_b_score": b_score,
                "predicted_label": pred_label,
                "predicted_host": pred_host,
                "margin_a_minus_b": a_score - b_score,
                "correct": ok,
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items)} rows to {args.out}")


if __name__ == "__main__":
    main()
