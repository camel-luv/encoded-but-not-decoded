"""Shared model loader and benchmark utilities.

Models are referenced by short name (e.g. ``qwen3_0.6b_instruct``). Two loading
modes are supported:

1. **HuggingFace Hub (default)** - pass no ``--model-root``. Checkpoints are
   resolved via the public Hub identifier in :data:`MODEL_HF_MAP` and cached
   under ``~/.cache/huggingface``.

2. **Local directory** - pass ``--model-root /path/to/models``. The script
   expects each checkpoint as a subdirectory named according to
   :data:`MODEL_DIR_MAP` (e.g. ``/path/to/models/Qwen3-0.6B-Base/``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional


# Short-name --> local subdirectory name (used when --model-root is set).
MODEL_DIR_MAP = {
    "llama3_2_1b": "Llama-3.2-1B",
    "llama3_2_1b_instruct": "Llama-3.2-1B-Instruct",
    "llama3_2_3b": "Llama-3.2-3B",
    "llama3_2_3b_instruct": "Llama-3.2-3B-Instruct",
    "gemma4_e4b": "gemma-4-E4B",
    "gemma4_e4b_it": "gemma-4-E4B-it",
    "qwen3_0.6b_base": "Qwen3-0.6B-Base",
    "qwen3_0.6b_instruct": "Qwen3-0.6B",
    "qwen3_1.7b_base": "Qwen3-1.7B-Base",
    "qwen3_1.7b_instruct": "Qwen3-1.7B",
    "qwen3_14b": "Qwen3-14B",
}

# Short-name --> public HuggingFace Hub identifier (used when --model-root is unset).
MODEL_HF_MAP = {
    "llama3_2_1b": "meta-llama/Llama-3.2-1B",
    "llama3_2_1b_instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3_2_3b": "meta-llama/Llama-3.2-3B",
    "llama3_2_3b_instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma4_e4b": "google/gemma-4-E4B",
    "gemma4_e4b_it": "google/gemma-4-E4B-it",
    "qwen3_0.6b_base": "Qwen/Qwen3-0.6B-Base",
    "qwen3_0.6b_instruct": "Qwen/Qwen3-0.6B",
    "qwen3_1.7b_base": "Qwen/Qwen3-1.7B-Base",
    "qwen3_1.7b_instruct": "Qwen/Qwen3-1.7B",
    "qwen3_14b": "Qwen/Qwen3-14B",
}


def iter_items(path: Path) -> Iterable[dict]:
    """Yield JSON objects from a JSONL file, one per non-blank line."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def resolve_device(device_arg: str) -> str:
    import torch

    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype_arg: str, device: str):
    import torch

    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    return torch.float16 if device.startswith("cuda") else torch.float32


def resolve_model_source(
    model_name: str,
    model_root: Optional[Path],
) -> tuple[str, bool]:
    """Return ``(source, is_local)`` for ``from_pretrained``.

    If ``model_root`` is provided and contains the expected subdirectory for
    ``model_name``, return its absolute path with ``is_local=True``.
    Otherwise fall back to the public HuggingFace Hub identifier with
    ``is_local=False`` (Transformers will download/cache as needed).
    """
    if model_root is not None:
        candidate = model_root / MODEL_DIR_MAP[model_name]
        if candidate.exists():
            return str(candidate), True
    if model_name not in MODEL_HF_MAP:
        raise KeyError(
            f"Unknown model {model_name!r}; expected one of {list(MODEL_HF_MAP)}"
        )
    return MODEL_HF_MAP[model_name], False


def load_model_and_tokenizer(
    model_name: str,
    model_root: Optional[Path],
    *,
    device: str,
    dtype,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source, is_local = resolve_model_source(model_name, model_root)
    tokenizer_kwargs = {"trust_remote_code": True}
    if is_local:
        tokenizer_kwargs["local_files_only"] = True
    if model_name.startswith("gemma4_"):
        tokenizer_kwargs["extra_special_tokens"] = {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(source, **tokenizer_kwargs)
    except Exception:
        if model_name.startswith("gemma4_"):
            from transformers import AutoProcessor

            processor_kwargs = {"trust_remote_code": True}
            if is_local:
                processor_kwargs["local_files_only"] = True
            processor = AutoProcessor.from_pretrained(source, **processor_kwargs)
            tokenizer = processor.tokenizer
        else:
            raise

    model_kwargs = {"trust_remote_code": True, "dtype": dtype, "low_cpu_mem_usage": True}
    if is_local:
        model_kwargs["local_files_only"] = True

    # Stream weights directly onto the GPU when one is requested: a 14B-class
    # fp16 checkpoint otherwise materializes ~28GB in CPU RAM before .to(device).
    gpu_idx = None
    if device == "cuda":
        gpu_idx = 0
    elif device.startswith("cuda:"):
        try:
            gpu_idx = int(device.split(":")[1])
        except ValueError:
            gpu_idx = None
    if gpu_idx is not None:
        model_kwargs["device_map"] = {"": gpu_idx}

    try:
        model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
        if gpu_idx is None:
            model = model.to(device)
    except Exception:
        if model_name.startswith("gemma4_"):
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(source, **model_kwargs).to(device)
        else:
            raise
    model.eval()
    return source, tokenizer, model


def continuation_score(model, tokenizer, prefix: str, continuation: str, device: str) -> float:
    """Sum log-probability of ``continuation`` tokens given ``prefix`` under ``model``."""
    import torch

    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
    full_ids = tokenizer(prefix + continuation, return_tensors="pt", add_special_tokens=False)
    prefix_len = prefix_ids["input_ids"].shape[1]
    full_input = full_ids["input_ids"].to(device)
    with torch.no_grad():
        logits = model(input_ids=full_input).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
    target_ids = full_input[0, prefix_len:]
    score = 0.0
    for pos, tok_id in enumerate(target_ids, start=prefix_len):
        score += float(log_probs[pos - 1, tok_id].item())
    return score


# ---------------------------------------------------------------------------
# Layer-resolution helper for activation patching / steering
# ---------------------------------------------------------------------------


def get_transformer_layer_module(model, layer_idx: int):
    """Return the transformer block whose forward output is ``hidden_states[layer_idx]``.

    HF causal LMs expose ``model.model.layers`` (Qwen3, Llama, Gemma). Output
    of layer ``layer_idx-1`` corresponds to ``hidden_states[layer_idx]`` for
    ``layer_idx >= 1``. Hooking the embedding layer (idx 0) is not supported.
    """
    if layer_idx < 1:
        raise ValueError("Hooking the embedding layer (idx 0) is not supported.")
    base = getattr(model, "model", None)
    if base is None:
        raise RuntimeError("Model has no .model attribute (unexpected architecture).")
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("model.model has no .layers attribute (unexpected architecture).")
    if layer_idx - 1 >= len(layers):
        raise ValueError(
            f"layer_idx {layer_idx} exceeds available layers ({len(layers)})."
        )
    return layers[layer_idx - 1]


def find_qa_positions(tokenizer, item):
    """Find a/b/v token positions inside ``item['qa_prompt']``."""
    qa_prompt = item["qa_prompt"]
    return {
        "a": find_target_token_index(tokenizer, qa_prompt, item["candidate_a"]),
        "b": find_target_token_index(tokenizer, qa_prompt, item["candidate_b"]),
        "v": find_target_token_index(tokenizer, qa_prompt, item["lower_predicate"]),
    }


def find_paraphrase_positions(tokenizer, item):
    """Find a/b/v token positions inside ``item['paraphrase_prompt']``."""
    para_prompt = item["paraphrase_prompt"]
    return {
        "a": find_target_token_index(tokenizer, para_prompt, item["candidate_a"]),
        "b": find_target_token_index(tokenizer, para_prompt, item["candidate_b"]),
        "v": find_target_token_index(tokenizer, para_prompt, item["lower_predicate"]),
    }


# ---------------------------------------------------------------------------
# Probe primitives - shared by run_probe_loocv and run_verb_out_cv
# ---------------------------------------------------------------------------

FEATURE_MODES = ("ab", "av", "bv", "abv", "v")


def find_target_token_index(tokenizer, sentence, target):
    """Return the last token index covering ``target`` within ``sentence``."""
    char_start = sentence.find(target)
    if char_start >= 0:
        char_end = char_start + len(target)
        encoded = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoded.get("offset_mapping")
        if offsets:
            last_idx = None
            for idx, (start, end) in enumerate(offsets):
                if start < char_end and end > char_start:
                    last_idx = idx
            if last_idx is not None:
                return last_idx
    full = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    candidate_targets = [target]
    if not target.startswith(" "):
        candidate_targets.append(f" {target}")
    for candidate in candidate_targets:
        target_ids = tokenizer(candidate, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        if not full or not target_ids or len(target_ids) > len(full):
            continue
        for start in range(len(full) - len(target_ids) + 1):
            if full[start:start + len(target_ids)] == target_ids:
                return start + len(target_ids) - 1
    return None


def build_examples(model, tokenizer, items, layer_idx, device):
    """Extract per-item hidden states at three positions (a, b, v) at ``layer_idx``."""
    import torch

    examples = []
    for item in items:
        ids = tokenizer(item["sentence"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        ia = find_target_token_index(tokenizer, item["sentence"], item["candidate_a"])
        ib = find_target_token_index(tokenizer, item["sentence"], item["candidate_b"])
        iv = find_target_token_index(tokenizer, item["sentence"], item["lower_predicate"])
        if ia is None or ib is None or iv is None:
            continue
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
            hs = out.hidden_states[layer_idx][0]
        examples.append({
            "id": item["id"],
            "language": item["language"],
            "control_type": item["control_type"],
            "candidate_a": item["candidate_a"],
            "candidate_b": item["candidate_b"],
            "gold_host": item["gold_host"],
            "matrix_predicate": item.get("matrix_predicate"),
            "controller_label": torch.tensor(float(item["controller_label"]), dtype=torch.float32),
            "a": hs[ia].detach().cpu().float(),
            "b": hs[ib].detach().cpu().float(),
            "v": hs[iv].detach().cpu().float(),
        })
    return examples


def feature_vector(example, mode):
    """Concatenate hidden-state features for the requested feature mode."""
    import torch

    features = {"a": example["a"], "b": example["b"], "v": example["v"]}
    return torch.cat([features[k] for k in mode], dim=-1)


def make_linear_probe(input_dim: int):
    """Return a one-output linear probe."""
    import torch
    from torch import nn

    class LinearProbe(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.linear = nn.Linear(dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x).squeeze(-1)

    return LinearProbe(input_dim)


def train_probe(train_examples, mode, epochs, lr, seed):
    """Train a linear probe on ``train_examples`` and return the fitted model."""
    import torch
    from torch import nn

    input_dim = feature_vector(train_examples[0], mode).shape[0]
    torch.manual_seed(seed)
    probe = make_linear_probe(input_dim)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = torch.cat([
            probe(feature_vector(ex, mode).unsqueeze(0)) for ex in train_examples
        ])
        labels = torch.cat([ex["controller_label"].unsqueeze(0) for ex in train_examples])
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
    return probe


def resolve_layers_from_model(model, requested_layers):
    """If ``requested_layers`` is empty, return ``[0, 1, ..., num_hidden_layers]``."""
    if requested_layers:
        return list(requested_layers)
    n_hidden = getattr(model.config, "num_hidden_layers", None)
    if n_hidden is None:
        text_config = getattr(model.config, "text_config", None)
        n_hidden = getattr(text_config, "num_hidden_layers", None)
    if n_hidden is None:
        raise ValueError("Cannot infer num_hidden_layers from model config.")
    return list(range(n_hidden + 1))


def aggregate_correct_rows(rows: list[dict]) -> dict:
    """Aggregate per-item correctness into accuracy summaries."""
    count = len(rows)
    if count == 0:
        return {
            "count": 0,
            "accuracy": None,
            "accuracy_by_language": {},
            "accuracy_by_type": {},
            "accuracy_by_language_type": {},
        }

    correct = 0
    by_language: dict[str, list[bool]] = {}
    by_type: dict[str, list[bool]] = {}
    by_language_type: dict[str, list[bool]] = {}

    for row in rows:
        ok = bool(row["correct"])
        correct += int(ok)
        language = row["language"]
        control_type = row["control_type"]
        by_language.setdefault(language, []).append(ok)
        by_type.setdefault(control_type, []).append(ok)
        by_language_type.setdefault(f"{language}|{control_type}", []).append(ok)

    return {
        "count": count,
        "accuracy": correct / count,
        "accuracy_by_language": {k: sum(v) / len(v) for k, v in by_language.items()},
        "accuracy_by_type": {k: sum(v) / len(v) for k, v in by_type.items()},
        "accuracy_by_language_type": {
            k: sum(v) / len(v) for k, v in by_language_type.items()
        },
    }
