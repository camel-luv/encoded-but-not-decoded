"""Counterbalanced-ordering ablation (paper §4.3 / Table 6).

Three variants:

* ``debiased_forced_choice_ab`` - explicit "A vs B" answer prompt under both
  orderings (ab, ba). Order-sensitivity is reported by averaging the two.
* ``debiased_label_choice`` - paraphrase "Option A: ... / Option B: ..."
  prompt under both orderings.
* ``contrastive_scoring`` - single ungated prompt; included for comparison.

Output JSONL contains one row per (variant, order, item).
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


COUNTERBALANCED_VARIANTS = ("debiased_forced_choice_ab", "debiased_label_choice")
ALL_VARIANTS = COUNTERBALANCED_VARIANTS + ("contrastive_scoring",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterbalanced ordering ablation.")
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
    parser.add_argument(
        "--variants", nargs="*", choices=ALL_VARIANTS, default=list(ALL_VARIANTS),
    )
    return parser.parse_args()


def build_forced_choice_prompt(item, *, order):
    if order == "ab":
        option_a, option_b = item["candidate_a"], item["candidate_b"]
    else:
        option_a, option_b = item["candidate_b"], item["candidate_a"]
    if item["language"] == "zh":
        prompt = (
            f"句子：{item['sentence']}\n"
            f"问题：{item['question']}\n"
            f"选项：A. {option_a}  B. {option_b}\n"
            "只回答 A 或 B："
        )
    else:
        prompt = (
            f"Sentence: {item['sentence']}\n"
            f"Question: {item['question']}\n"
            f"Options: A. {option_a}  B. {option_b}\n"
            "Answer with A or B only:"
        )
    gold_label = "a" if item["gold_host"] == option_a else "b"
    return prompt, " A", " B", gold_label, option_a, option_b


def build_label_choice_prompt(item, *, order):
    if order == "ab":
        text_a = item["paraphrase_a"].strip()
        text_b = item["paraphrase_b"].strip()
        gold_label = item["gold_label"]
    else:
        text_a = item["paraphrase_b"].strip()
        text_b = item["paraphrase_a"].strip()
        gold_label = "a" if item["gold_label"] == "b" else "b"
    if item["language"] == "zh":
        prompt = (
            f"句子：{item['sentence']}\n候选解释：\n"
            f"Option A: {text_a}\nOption B: {text_b}\n更合适的是："
        )
    else:
        prompt = (
            f"Sentence: {item['sentence']}\nCandidate interpretations:\n"
            f"Option A: {text_a}\nOption B: {text_b}\nMore faithful option:"
        )
    option_a_host = item["candidate_a"] if order == "ab" else item["candidate_b"]
    option_b_host = item["candidate_b"] if order == "ab" else item["candidate_a"]
    return prompt, " Option A", " Option B", gold_label, option_a_host, option_b_host


def build_contrastive_prompt(item):
    if item["language"] == "zh":
        prompt = (
            f"句子：{item['sentence']}\n动作"
            f"“{item['action_phrase']}”的实际执行者是"
        )
        return prompt, f"{item['candidate_a']}。", f"{item['candidate_b']}。", item["gold_label"]
    prompt = (
        f"Sentence: {item['sentence']}\n"
        f"The understood subject of \"{item['action_phrase']}\" is"
    )
    return prompt, f" {item['candidate_a']}.", f" {item['candidate_b']}.", item["gold_label"]


def score_counterbalanced(model, tokenizer, items, variant, device, builder):
    rows = []
    for order in ("ab", "ba"):
        for item in items:
            prompt, cont_a, cont_b, gold_label, opt_a_host, opt_b_host = builder(item, order=order)
            a = continuation_score(model, tokenizer, prompt, cont_a, device)
            b = continuation_score(model, tokenizer, prompt, cont_b, device)
            pred_label = "a" if a > b else "b"
            pred_host = opt_a_host if pred_label == "a" else opt_b_host
            rows.append({
                **item,
                "variant": variant,
                "order": order,
                "candidate_a_score": a,
                "candidate_b_score": b,
                "predicted_label": pred_label,
                "predicted_host": pred_host,
                "gold_label_for_variant": gold_label,
                "correct": pred_label == gold_label,
            })
    return rows


def score_contrastive(model, tokenizer, items, device):
    rows = []
    for item in items:
        prompt, cont_a, cont_b, gold_label = build_contrastive_prompt(item)
        a = continuation_score(model, tokenizer, prompt, cont_a, device)
        b = continuation_score(model, tokenizer, prompt, cont_b, device)
        pred_label = "a" if a > b else "b"
        pred_host = item["candidate_a"] if pred_label == "a" else item["candidate_b"]
        rows.append({
            **item,
            "variant": "contrastive_scoring",
            "order": None,
            "candidate_a_score": a,
            "candidate_b_score": b,
            "predicted_label": pred_label,
            "predicted_host": pred_host,
            "gold_label_for_variant": gold_label,
            "correct": pred_label == gold_label,
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for variant in args.variants:
            if variant == "debiased_forced_choice_ab":
                rows = score_counterbalanced(model, tokenizer, items, variant, device, build_forced_choice_prompt)
            elif variant == "debiased_label_choice":
                rows = score_counterbalanced(model, tokenizer, items, variant, device, build_label_choice_prompt)
            else:
                rows = score_contrastive(model, tokenizer, items, device)
            for r in rows:
                r.update({"model": args.model, "model_source": source})
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += 1
    print(f"Wrote {total} rows to {args.out}")


if __name__ == "__main__":
    main()
