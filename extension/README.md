# Extending the benchmark to a new language

The pipeline in this repository is language-agnostic by design: all prompt
templates live in the **data files**, not in the code. Build items in a new
language that the target models cover — in **any number** — and every
experiment in the paper (behavioral evaluation, LM-head readout, probing,
patching, steering, and verb-out cross-validation) runs on your items
without code changes. Leave-one-item-out folds, patching donor pairs,
steering statistics, and all aggregations scale with your item count;
nothing in the code assumes the paper's 48.

This folder contains:

* `behavior_item_template.jsonl` — two reference items (one subject-control,
  one object-control) showing every field of the behavioral benchmark.
* `probe_item_template.jsonl` — the same two items with the additional
  fields required by the probing pipeline.

---

## 1. Item design

The shipped benchmark uses, per language:

* **8 subject-control items** and **8 object-control items** (16 total),
  built from 4–6 control verbs per type (e.g., *promise* vs. *tell*) — this
  is the paper's configuration, not a requirement of the code.
* Two candidates per item (the matrix subject and the other NP), with
  **gold balance**: half of the items have `gold_label = "a"`, half `"b"`.
* Minimal pairs across control types where possible (same embedded action,
  different matrix verb), as in the templates above.
* ID convention: `{lang}_subj_01..08`, `{lang}_obj_01..08` (extend the
  numbering freely: `_09`, `_10`, …).

Any item count works. Larger sets are straightforwardly better for
statistics — 16/language is statistically thin (see the paper's Limitations),
so if you can build 32, 50, or 100 per control type, do; leave-one-item-out
CV, verb-out folds, donor pairing, and every reported aggregate scale with
n. Keep the same `language` tag on every row of one language.

## 2. Field reference

Fields shared by both files:

| Field | Constraint |
|---|---|
| `id` | unique, `{lang}_{subj\|obj}_{NN}` |
| `language` | any short tag (`en`, `zh`, `de`, `fr`, …) |
| `phenomenon` | `"control"` (constant) |
| `control_type` | `subject_control` or `object_control` |
| `sentence` | the trigger sentence |
| `question` | comprehension question in the target language |
| `candidate_a` / `candidate_b` | the two candidate controllers (in order) |
| `gold_host` / `gold_label` | correct controller / its position (`"a"` or `"b"`) |
| `qa_prompt` | `"Sentence: …\nQuestion: …\nAnswer:"` frame |
| `qa_answer_a` / `qa_answer_b` | continuations scored after `qa_prompt`, leading space |
| `paraphrase_prompt` | `"\nMore faithful interpretation:"` frame |
| `paraphrase_a` / `paraphrase_b` | interpretation continuations, leading space |
| `action_phrase` | bare infinitival predicate + complement (e.g., `water the plants`) |
| `source`, `note`, `split` | provenance metadata (free text; `split` = `"test"`) |

Behavioral file only:

| Field | Constraint |
|---|---|
| `bare_prompt` | exactly `"Sentence: {sentence}"` |
| `bare_prompt_labelled` | `bare_prompt` + `"\nContinuation:"` |
| `bare_a` / `bare_b` | declarative resolutions (past tense in English), leading space |

Probe file only:

| Field | Constraint |
|---|---|
| `matrix_predicate` | the control verb (`promised`, `told`, …) |
| `lower_predicate` | the embedded bare verb (`water`) — **must occur verbatim in `qa_prompt`** |
| `controller_label` | `1` if `gold_label = "a"`, else `0` |

**Hard constraints** (the pipeline localizes token positions by substring
search, so these must hold exactly):

1. `candidate_a`, `candidate_b`, and `lower_predicate` each occur verbatim
   inside `qa_prompt` (probe feature extraction).
2. All continuation fields (`qa_answer_*`, `paraphrase_*`, `bare_*`) carry a
   leading space and are natural continuations of their prompts.
3. `gold_label` ⇔ `gold_host` ⇔ `controller_label` are consistent.

## 3. Aligning with UD / EUD annotation

The shipped English/German items were hand-built but informed by Universal
Dependencies conventions (`source: ud_eud_informed_handbuilt`). To follow the
same practice for a new language:

* Pick control verbs **attested in a UD treebank** of the language, so that
  the construction is corpus-natural rather than translated.
* In **enhanced UD (EUD)**, the diagnostic is the controller of the `xcomp`
  (or language-equivalent open complement) relation: if the matrix clause's
  `nsubj` is the understood subject of the embedded predicate, the verb is
  subject-control (*promise*-type); if the matrix `obj` is, it is
  object-control (*tell*-type). Verify with an enhanced-dependencies query
  on your treebank, not only with an English gloss.
* Keep the `source` field informative, e.g. `ud_eud_informed_handbuilt` or
  `handbuilt_{lang}`, and document verb choices in `note`.
* Some languages realize control with non-infinitival structures (clause
  chaining, serial verbs, causatives, subjunctive complements). Any
  construction works as long as the *understood controller* is ambiguous on
  the surface and resolved by the matrix verb — that is the phenomenon under
  study.

## 4. Running the pipeline on your items

Save your files anywhere and pass them explicitly (do not overwrite the
shipped data unless you intend to):

```bash
python -m src.run_behavior_qa qwen3_0.6b_instruct \
    --benchmark-file extension/my_lang_behavior.jsonl \
    --language fr \
    --out results/behavior_qa_fr.jsonl
```

* `--language` accepts any tag present in your data (`all` = no filtering).
* The probing scripts take the **probe** file; behavior/LM-head scripts take
  the **behavior** file (they need the `bare_*` fields).
* Seeds stay as in the paper: 1729, 2718, 3141.
* One known convention: `src/run_behavior_readout_cleaning.py` has native
  prompt frames for Chinese and an English frame otherwise — this is exactly
  how the German items in the paper were run. For a native frame in your
  language, add a branch to its three `build_*_prompt` functions (≈3 lines
  each).

After running, aggregate exactly as in the repository README: accuracies are
always means over `correct` rows, optionally split by `control_type` or
`language`.
