# Encoded but Not Decoded — Code & Data

Code and benchmark data for the paper:

> **Encoded but Not Decoded: Layer-Localized Evidence for a Three-Level Gap in LLM Syntax**
> Lu et al. — Accepted to AACL-IJCNLP 2026 (Main Conference).

This repository ships the **minimum required to reproduce every numerical claim in the paper**: the trilingual control benchmark (48 items × 3 languages), and one canonical script per experiment. Plotting and internal aggregation utilities are intentionally not included.

---

## Contents

```
├── data/
│   ├── control_behavior_benchmark_trilingual.jsonl    Behavioral benchmark (QA / paraphrase / bare prompts, 48 items)
│   └── control_probe_benchmark_trilingual.jsonl       Probe benchmark (token-position annotations, 48 items)
├── extension/                                         Item templates and a guide for adding new languages
├── src/                                               12 modules; each is the canonical implementation of one paper experiment
├── requirements.txt                                   Dependencies
└── LICENSE                                            Apache-2.0
```

Languages: English, Chinese, German (16 items each; subject-control and object-control items; gold controller annotated).

---

## Setup

```bash
pip install -r requirements.txt
```

Any single CUDA GPU works. Approximate VRAM in fp16: Qwen3-14B needs ~30 GB; all other models fit in 16 GB or less. CPU-only execution works for the smaller models but is slow.

### Choosing a dtype for your hardware

All scripts accept `--dtype {auto, float32, float16, bfloat16}` (default `auto`: fp16 on CUDA, fp32 on CPU). Use `--dtype bfloat16` on Ampere-or-newer GPUs if preferred; `--device` accepts `auto`/`cuda`/`cpu`/an index like `cuda:1`.

### Two model-loading modes

By default, scripts download checkpoints from the HuggingFace Hub on first use. Identifiers are in `src/common.py::MODEL_HF_MAP`. If you have the checkpoints on disk, pass `--model-root /path/to/models` with each model as a subdirectory named by its HuggingFace repo name.

### Model short-names and paper display names

| Short-name (CLI argument) | Paper display name | HuggingFace checkpoint |
|---|---|---|
| `qwen3_0.6b_base` | Qwen3-0.6B Base | Qwen/Qwen3-0.6B-Base |
| `qwen3_0.6b_instruct` | Qwen3-0.6B Instruct | Qwen/Qwen3-0.6B |
| `qwen3_14b` | Qwen3-14B Instruct | Qwen/Qwen3-14B |
| `llama3_2_1b` | Llama-3.2-1B Base | meta-llama/Llama-3.2-1B |
| `llama3_2_1b_instruct` | Llama-3.2-1B Instruct | meta-llama/Llama-3.2-1B-Instruct |
| `gemma4_e4b` | Gemma-4 E4B | google/gemma-4-E4B |
| `gemma4_e4b_it` | Gemma-4 E4B Instruct | google/gemma-4-E4B-it |
| `qwen3_1.7b_base` | Qwen3-1.7B Base (scaling extension, §4.5) | Qwen/Qwen3-1.7B-Base |
| `qwen3_1.7b_instruct` | Qwen3-1.7B Instruct (scaling extension, §4.5) | Qwen/Qwen3-1.7B |
| `llama3_2_3b` | Llama-3.2-3B Base (scaling extension, §4.5) | meta-llama/Llama-3.2-3B |
| `llama3_2_3b_instruct` | Llama-3.2-3B Instruct (scaling extension, §4.5) | meta-llama/Llama-3.2-3B-Instruct |

> Note: the paper evaluates Qwen3-14B in its instruction-tuned form; the HuggingFace checkpoint `Qwen/Qwen3-14B` is that instruction-tuned release. The base variant was not evaluated.

All scripts run as Python modules from the repository root:

```bash
python -m src.<module_name> [arguments]
```

A quick smoke test (single small model, one language):

```bash
python -m src.run_behavior_qa qwen3_0.6b_instruct \
    --benchmark-file data/control_behavior_benchmark_trilingual.jsonl \
    --language en \
    --out results/smoke_qa_en.jsonl
```

---

## Reproducing paper numbers

Each script writes a JSONL with one row per observation; aggregate with inline Python as shown. Conventions: `B = data/control_behavior_benchmark_trilingual.jsonl`, `P = data/control_probe_benchmark_trilingual.jsonl`.

### §4 / Table 1 — Behavior (QA and paraphrase)

```bash
python -m src.run_behavior_qa qwen3_0.6b_instruct \
    --benchmark-file $B --out results/behavior_qa_qwen06b_inst.jsonl
python -m src.run_behavior_paraphrase qwen3_0.6b_instruct \
    --benchmark-file $B --out results/behavior_para_qwen06b_inst.jsonl
```

```python
import json
rows = [json.loads(l) for l in open("results/behavior_qa_qwen06b_inst.jsonl")]
print("QA accuracy:", sum(r["correct"] for r in rows) / len(rows))
# Table 1, Qwen3-0.6B Instruct, column "QA": 0.417
```

Mean behavior is the unweighted mean of the QA and paraphrase accuracies, e.g. `(0.417 + 0.604) / 2 = 0.510` for Qwen3-0.6B Instruct.

### §4.1 — Headline behavior–probe gap

The sharpest case: Qwen3-0.6B Instruct QA subject-control — behavior 0.250 vs. probe 0.903, gap 0.653.

```python
rows = [json.loads(l) for l in open("results/behavior_qa_qwen06b_inst.jsonl")]
subj = [r for r in rows if r["control_type"] == "subject_control"]
print("subject-control QA accuracy:", sum(r["correct"] for r in subj) / len(subj))
# Expected: 0.250
```

### §4 / Table 1 — LM-head readout (best layer)

```bash
python -m src.run_lmhead_by_layer qwen3_0.6b_instruct \
    --benchmark-file $B --task qa --out results/lmhead_qa_qwen06b_inst.jsonl
python -m src.run_lmhead_by_layer qwen3_0.6b_instruct \
    --benchmark-file $B --task paraphrase --out results/lmhead_para_qwen06b_inst.jsonl
```

"Best LM-head" is the maximum over (layer × task) of mean accuracy; for Qwen3-0.6B Instruct the paper reports 0.667 (paraphrase peak).

### §4 / Table 1 — Probe recoverability (best layer × mode)

```bash
python -m src.run_probe_loocv qwen3_0.6b_instruct \
    --benchmark-file $P --seed 1729 --out results/probe_qwen06b_inst.jsonl
```

"Best Probe" is the maximum over (layer × feature mode) of LOOCV accuracy; Table 1 uses seed 1729 (seed robustness below). Expected for Qwen3-0.6B Instruct: 0.833. On many-core machines, `src.run_probe_loocv_parallel` produces identical rows faster.

### Appendix C / Table 3 — Feature-mode comparison

Same probe runs as above, aggregated per feature mode (`ab`, `bv`, `av`, `abv`, `v`) instead of taking the max. The paper reports the mean over the three seeds (1729, 2718, 3141); e.g. Qwen3-0.6B Base reaches 0.924 in mode `ab`.

### Appendix D / Table 4 — Probe seed robustness

Run the probe with all three seeds, select the (layer, mode) pair by the **three-seed mean**, then report each seed's accuracy at that fixed pair:

```bash
for SEED in 1729 2718 3141; do
  python -m src.run_probe_loocv qwen3_0.6b_instruct \
      --benchmark-file $P --seed $SEED --out results/probe_qwen06b_inst_seed${SEED}.jsonl
done
```

For Qwen3-0.6B Instruct the fixed pair (L16, `ab`) gives 0.833 / 0.812 / 0.896 across the three seeds — mean 0.847, std 0.035, matching Table 4.

### §4.3 / Table 6 & Figure 6 — Counterbalanced ordering (readout-cleaning)

```bash
python -m src.run_behavior_readout_cleaning qwen3_0.6b_instruct \
    --benchmark-file $B --out results/readout_clean_qwen06b_inst.jsonl
```

Debiased accuracy averages the two option orders (`ab`, `ba`) per item. Paper values (Table 6):

| Model | Debiased forced-choice | Debiased label-choice | Contrastive |
|---|---|---|---|
| Qwen3-0.6B Base | 0.531 | 0.552 | 0.479 |
| Qwen3-0.6B Instruct | 0.500 | 0.562 | 0.500 |
| Qwen3-14B Instruct | 0.917 | 0.938 | 0.708 |

```python
import json, collections
rows = [json.loads(l) for l in open("results/readout_clean_qwen06b_inst.jsonl")]
for variant in ("debiased_forced_choice_ab", "debiased_label_choice"):
    pairs = collections.defaultdict(list)  # id -> [correct under ab, ba]
    for r in rows:
        if r["variant"] == variant and r["order"] in ("ab", "ba"):
            pairs[r["id"]].append(r["correct"])
    debiased = [sum(p) / len(p) for p in pairs.values() if len(p) == 2]
    print(variant, sum(debiased) / len(debiased))
```

### §4.5 — Scaling extensions (Qwen3-1.7B and Llama-3.2-3B pairs)

The paper's scaling analysis adds two base/instruct pairs (deployment, probe ceiling, and surplus only):

```bash
for M in qwen3_1.7b_base qwen3_1.7b_instruct llama3_2_3b llama3_2_3b_instruct; do
  python -m src.run_behavior_qa $M --benchmark-file $B --out results/behavior_qa_${M}.jsonl
  python -m src.run_behavior_paraphrase $M --benchmark-file $B --out results/behavior_para_${M}.jsonl
  python -m src.run_probe_loocv $M --benchmark-file $P --seed 1729 --out results/probe_${M}.jsonl
done
```

Deployment = mean of the two behavior accuracies; probe ceiling = best (layer × mode) at seed 1729; surplus = ceiling − deployment. Expected: Qwen3-1.7B Base 0.708 / 0.986 / 0.278; Qwen3-1.7B Instruct 0.771 / 0.958 / 0.188; Llama-3.2-3B Base 0.708 / 0.958 / 0.250; Llama-3.2-3B Instruct 0.708 / 0.750 / 0.042.

### §4.5 / Appendix G — Causal patching

Patching runs one (layer, mode) at a time; mode is `abv` throughout the paper. The headline lift is at Qwen3-0.6B Instruct L26, QA subject-control:

```bash
python -m src.run_patching qwen3_0.6b_instruct \
    --layer 26 --mode abv \
    --benchmark-file $P \
    --donor-policy same_lang_type \
    --tasks qa \
    --out results/patching_qwen06b_inst_L26.jsonl
```

Causal-rescue lift = correct-donor rescue rate − wrong-donor rescue rate over (target × donor) pairs sharing language and control type. Expected: **+0.37** (Qwen3-14B Instruct replicates at L27 with +0.19).

### Appendix G — Directional steering (null result)

Two steps: extract a probe direction, then steer at scaled magnitudes.

```bash
# Step 1: direction at the canonical probe peak of Qwen3-0.6B Instruct
python -m src.dump_probe_weights qwen3_0.6b_instruct \
    --layer 16 --mode bv --direction-method logreg \
    --benchmark-file $P --out results/probe_weights_qwen06b_inst_L16_bv.json

# Step 2: steering sweep (the paper's main sweep is -2..+2; it extends to ±3, ±5)
python -m src.run_steering qwen3_0.6b_instruct \
    --probe-weights results/probe_weights_qwen06b_inst_L16_bv.json \
    --benchmark-file $B --tasks qa \
    --alphas -2 -1 -0.5 0 0.5 1 2 3 5 \
    --out results/steering_qwen06b_inst.jsonl
```

`--direction-method` accepts `logreg` (probe weights) or `diff_of_means`. The paper reports no flip on subject-control items at any tested magnitude (McNemar vs. α=0: p = 1.000).

### Appendix H / Tables 7–8 — Verb-out cross-validation (leakage control)

Focal cell (Qwen3-14B at its probe peak L17, all five modes):

```bash
python -m src.run_verb_out_cv qwen3_14b \
    --benchmark-file $P \
    --layers 17 \
    --modes ab av bv abv v \
    --seeds 1729 2718 3141 \
    --out results/verb_out_qwen3_14b_L17.jsonl
```

Cross-model trajectory (`bv` mode, full layer sweep), for Table 8:

```bash
for M in qwen3_0.6b_base qwen3_0.6b_instruct qwen3_14b \
         llama3_2_1b llama3_2_1b_instruct \
         gemma4_e4b gemma4_e4b_it; do
  python -m src.run_verb_out_cv $M \
      --benchmark-file $P \
      --modes bv \
      --seeds 1729 2718 3141 \
      --out results/verb_out_${M}_fulltraj.jsonl
done
```

Each output row carries both `item_out_loocv_acc` and `verb_out_cv_acc` for one (model, layer, mode, control type, seed). The paper's verdict criterion follows Appendix H: gap < 0.10 clean, 0.10–0.30 partial, ≥ 0.30 leakage-dominant. On many-core machines, `src.run_verb_out_cv_parallel` produces identical rows faster.

---

## Extending to a new language

The pipeline is language-agnostic: prompt templates live in the data files, not the code. Build items in any language your models cover — any number per language works, not just the 16 the paper used — and every experiment above runs on them unchanged (`--language` accepts any tag used in your data; folds, donor pairs, and aggregations all scale with your item count). See [`extension/README.md`](extension/README.md) for item templates, the field reference with hard constraints, UD/EUD alignment guidance, and usage notes.

---

## Compute cost

A clean rerun of the full suite (all models, three seeds, all analyses) took 19.3 GPU-hours wall-clock on a single V100 32GB, plus about 45 CPU core-hours (measured; CPU-heavy stages are the leave-one-out and verb-out cross-validation training). The 14B block alone is 3.3 GPU-hours. Small-model stages fit comfortably on 16 GB cards.

## Repository conventions

* One model short-name vocabulary across all scripts (`MODEL_HF_MAP` in `src/common.py`).
* `--device auto` and `--dtype auto` defaults work on any CUDA GPU or CPU.
* Random-seed inputs match the paper exactly: 1729, 2718, 3141.
* Output JSONL fields are stable across scripts where possible: `id`, `language`, `control_type`, `task`, `correct`.
* **Numerical note.** Reruns on different GPU architectures may show small numerical differences on near-tie items, mainly due to fp16 accumulation differences across architectures (aggregate accuracies stay within a few hundredths).
* `*_parallel` variants are semantically identical to their serial counterparts, differing only in execution layout.

## Citation

```bibtex
@inproceedings{lu2026encoded,
  title     = {Encoded but Not Decoded: Layer-Localized Evidence for a Three-Level Gap in LLM Syntax},
  author    = {Lu, Zhenyan and Wang, He and Huang, Xiaohui},
  booktitle = {Proceedings of the 5th Asia-Pacific Chapter of the Association for Computational Linguistics and the 15th International Joint Conference on Natural Language Processing (AACL-IJCNLP 2026)},
  year      = {2026}
}
```

## License

Apache-2.0. See `LICENSE`.
