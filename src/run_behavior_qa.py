"""Run the QA-style behavioral evaluation on the trilingual control benchmark.

For each item, compute the log-probability of each candidate continuation under
``qa_prompt`` and predict the higher-scoring one. Output a JSONL with one row
per item; the ``correct`` field is the binary judgment.

Example:

    python -m src.run_behavior_qa qwen3_0.6b_instruct \\
        --benchmark-file data/control_behavior_benchmark_trilingual.jsonl \\
        --out results/qa_qwen06b_inst.jsonl

To use locally cached checkpoints instead of the HuggingFace Hub:

    python -m src.run_behavior_qa qwen3_0.6b_instruct \\
        --model-root /path/to/local/models \\
        --benchmark-file data/control_behavior_benchmark_trilingual.jsonl \\
        --out results/qa_qwen06b_inst.jsonl
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
    parser = argparse.ArgumentParser(description="QA-style behavioral evaluation.")
    parser.add_argument("model", choices=sorted(MODEL_HF_MAP))
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        required=True,
        help="Path to control_behavior_benchmark_trilingual.jsonl",
    )
    parser.add_argument(
        "--language", default="all",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="Optional local directory containing model checkpoints "
             "(see common.MODEL_DIR_MAP for expected layout). "
             "If unset, models are loaded from the HuggingFace Hub.",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output JSONL path",
    )
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
                model, tokenizer, item["qa_prompt"], item["qa_answer_a"], device,
            )
            b_score = continuation_score(
                model, tokenizer, item["qa_prompt"], item["qa_answer_b"], device,
            )
            pred_label = "a" if a_score > b_score else "b"
            pred_host = item["candidate_a"] if pred_label == "a" else item["candidate_b"]
            ok = pred_label == item["gold_label"]
            fh.write(json.dumps({
                **item,
                "task": "qa",
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
