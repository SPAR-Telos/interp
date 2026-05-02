"""Run the variant-transplant intervention.

For each target visit, runs 3 forward passes through the LLM:

  1. baseline      — no intervention (un-modified prompt, deterministic decode)
  2. self          — replace layer-15 prompt-suffix activations with the
                     mean of same-variant visits at the same cell
                     (excluding the target visit itself); this is the
                     "average vs the actual visit" control
  3. cross         — replace with mean of *other-variant* visits at the
                     same cell; the hypothesis test

Writes one JSONL row per (target, condition) trial with the parsed
emitted action.

The hook fires only on the prompt-processing forward pass (seq_len > 1)
and only once. Subsequent generation steps (seq_len == 1) inherit the
intervened K/V cache for layers ≥ hook_layer.

----------

Two execution modes:

  * Real run on RunPod GPU: `--model-id openai/gpt-oss-20b` (default),
    `--device-map auto`, `--torch-dtype bfloat16`.

  * CPU smoke on Mac: `--smoke-model gpt2` plus `--smoke`. The hook
    mechanism, position indexing, output parsing, and JSONL writing
    are exercised end-to-end with a small CPU model and random
    donor tensors (matched to the small model's hidden_dim, NOT the
    real 2880). A few targets are run; the emitted text is gibberish
    but the orchestration is verified.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import traceback
from pathlib import Path

import torch


ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]
ACTION_RE = re.compile(r'"action"\s*:\s*"(LEFT|RIGHT|UP|DOWN)"', re.IGNORECASE)


def parse_action(text: str) -> str | None:
    m = ACTION_RE.search(text)
    if m:
        return m.group(1).upper()
    tail = text[-200:].upper()
    found = []
    for a in ACTIONS:
        idx = tail.rfind(a)
        if idx >= 0:
            found.append((idx, a))
    if found:
        found.sort(reverse=True)
        return found[0][1]
    return None


def make_replace_hook(
    positions: list[int],
    donor_per_token: torch.Tensor | None,
    log_replacement: dict | None = None,
):
    """Forward hook that overwrites ``hidden[:, pos, :]`` with
    ``donor_per_token[i]`` at each of the ``positions`` exactly once,
    on the first prompt-processing pass (seq_len > 1). When
    ``donor_per_token`` is None this is a no-op (used for the baseline
    condition).

    If ``log_replacement`` dict is passed, mutates it with diagnostic
    information after the hook fires (used by smoke tests to verify
    the replacement actually happened).
    """
    state = {"applied": False}

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        # Only on the prompt pass (seq > 1) and only once.
        if state["applied"] or hidden.shape[1] <= 1:
            return output
        if donor_per_token is not None:
            n_pos, hidden_dim = donor_per_token.shape
            assert n_pos == len(positions), f"donor n_pos={n_pos} vs positions={len(positions)}"
            assert hidden_dim == hidden.shape[2], (
                f"donor hidden_dim={hidden_dim} != model hidden_dim={hidden.shape[2]}"
            )
            for i, pos in enumerate(positions):
                if pos >= hidden.shape[1]:
                    continue
                pre_norm = hidden[:, pos, :].detach().float().norm().item()
                v = donor_per_token[i].to(dtype=hidden.dtype, device=hidden.device)
                hidden[:, pos, :] = v
                post_norm = hidden[:, pos, :].detach().float().norm().item()
                if log_replacement is not None:
                    log_replacement.setdefault("replacements", []).append(
                        {"position": int(pos), "pre_norm": pre_norm, "post_norm": post_norm}
                    )
        state["applied"] = True
        if log_replacement is not None:
            log_replacement["fired"] = True
            log_replacement["seq_len"] = int(hidden.shape[1])
        if rest is not None:
            return (hidden,) + rest
        return hidden

    return hook


def get_layer_module(model, layer_idx: int):
    """Locate transformer block `layer_idx` across common HF naming patterns."""
    candidates = [
        ("model.model.layers", lambda m: m.model.layers[layer_idx]),
        ("model.transformer.h", lambda m: m.transformer.h[layer_idx]),
        ("model.layers", lambda m: m.layers[layer_idx]),
        ("transformer.h", lambda m: m.transformer.h[layer_idx]),
    ]
    for name, fn in candidates:
        try:
            mod = fn(model)
            print(f"  using {name}[{layer_idx}] (type={type(mod).__name__})")
            return mod
        except (AttributeError, IndexError):
            continue
    print("ERROR: could not locate transformer layers. Inspect model:", file=sys.stderr)
    print(model, file=sys.stderr)
    raise RuntimeError("layer module not found")


def run_one_trial(
    model, tokenizer, layer_module, target, donor_act, max_new_tokens, device,
    smoke_input_ids: torch.Tensor | None = None,
    log_replacement: dict | None = None,
):
    """Run one forward+generate pass with the optional activation replacement.

    ``smoke_input_ids``: if provided, used INSTEAD of the target's
    real token IDs. The smoke mode passes a synthesised input that
    fits the small model's vocab.
    """
    if smoke_input_ids is not None:
        input_ids = smoke_input_ids.to(device)
    else:
        input_ids = torch.tensor([target["full_token_ids"]], dtype=torch.long, device=device)
    positions = target["suffix_positions"]

    handle = layer_module.register_forward_hook(
        make_replace_hook(positions, donor_act, log_replacement=log_replacement)
    )
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id or 0,
                use_cache=True,
            )
    finally:
        handle.remove()

    new_token_ids = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_token_ids, skip_special_tokens=False)
    return {"emitted_action": parse_action(text), "generated_text_tail": text[-300:]}


def load_donors(path: Path) -> dict:
    """donors.pt holds a dict; keys are tuples like (col, row, variant_str)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def lookup_donor(donors: dict, key: list, hidden_dim_expected: int | None = None) -> torch.Tensor | None:
    """Resolve the donor lookup key from the JSONL row (a list) to the
    tuple key in donors.pt. Returns the donor tensor or None.
    """
    tup = (int(key[0]), int(key[1]), key[2])
    d = donors.get(tup)
    if d is None:
        return None
    if hidden_dim_expected is not None and d.shape[-1] != hidden_dim_expected:
        return None
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path,
                    default=Path("transplant_intervention/targets.jsonl"))
    ap.add_argument("--donors", type=Path,
                    default=Path("transplant_intervention/donors.pt"))
    ap.add_argument("--output", type=Path,
                    default=Path("transplant_intervention/results.jsonl"))
    ap.add_argument("--model-id", type=str, default="openai/gpt-oss-20b")
    ap.add_argument("--smoke-model", type=str, default=None,
                    help="If set, override --model-id with this; intended for CPU smoke.")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU orchestration test: load --smoke-model (default gpt2) on CPU, "
                         "use random donors of the model's hidden_dim, run a few targets to "
                         "verify hooks/positions/parsing.")
    ap.add_argument("--max-targets", type=int, default=0,
                    help="0 = all targets in the file. >0 = first N (for fast testing).")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--torch-dtype", type=str, default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Model loading
    if args.smoke:
        model_id = args.smoke_model or "gpt2"
        device_map = "cpu"
        torch_dtype = "float32"
        max_new_tokens = min(args.max_new_tokens, 64)
        max_targets = args.max_targets if args.max_targets else 4
        print(f"SMOKE: model={model_id}, device=cpu, max_new_tokens={max_new_tokens}, "
              f"max_targets={max_targets}, will use RANDOM donors of the model's hidden_dim "
              f"(scientific result invalid; orchestration only).")
    else:
        model_id = args.smoke_model or args.model_id
        device_map = args.device_map
        torch_dtype = args.torch_dtype
        max_new_tokens = args.max_new_tokens
        max_targets = args.max_targets

    print(f"Loading model {model_id} (dtype={torch_dtype}, device_map={device_map})")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype_t = dtype_map.get(torch_dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype_t, device_map=device_map,
    )
    model.eval()
    layer_module = get_layer_module(model, args.layer)
    hidden_dim = model.config.hidden_size
    device = next(model.parameters()).device
    print(f"  device={device}  hidden_dim={hidden_dim}")

    print(f"Loading targets from {args.targets}")
    with open(args.targets) as f:
        targets = [json.loads(line) for line in f if line.strip()]
    if max_targets and max_targets > 0:
        targets = targets[:max_targets]
    print(f"  {len(targets)} targets to process")

    print(f"Loading donors from {args.donors}")
    donors = load_donors(args.donors)
    print(f"  {len(donors)} donor entries")

    # In smoke mode the real donors won't match the small model's hidden
    # dim. Generate random donors of the right size for each lookup key
    # we encounter.
    if args.smoke:
        rng = torch.Generator().manual_seed(args.seed)
        smoke_donors: dict[tuple, torch.Tensor] = {}

    n_written = 0
    n_smoke_donors_made = 0
    n_hooks_fired = 0
    with open(args.output, "w") as fout:
        for i, target in enumerate(targets):
            # In smoke mode the real prompt's token IDs (gpt-oss-20b vocab)
            # are way out of range for small CPU models; build a safe
            # input filled with a small token id, but keep the real
            # prompt_len so suffix_positions are still inside the prompt.
            if args.smoke:
                pad_tok = tokenizer.eos_token_id or 0
                if pad_tok is None or pad_tok < 0:
                    pad_tok = 0
                vocab = getattr(tokenizer, "vocab_size", None) or model.config.vocab_size
                pad_tok = min(int(pad_tok), int(vocab) - 1)
                smoke_input = torch.full(
                    (1, target["prompt_len"]), pad_tok, dtype=torch.long
                )
            else:
                smoke_input = None

            # 3 conditions: baseline (no donor), self, cross.
            for condition_name, donor_key in (
                ("baseline", None),
                ("self", target["donor_key_self"]),
                ("cross", target["donor_key_cross"]),
            ):
                log: dict = {}
                try:
                    if donor_key is None:
                        donor = None
                    else:
                        donor = lookup_donor(donors, donor_key, hidden_dim_expected=hidden_dim)
                        if donor is None and args.smoke:
                            # Smoke mode: synthesise a random donor of the
                            # right shape for this key (deterministic per-key).
                            tup = (int(donor_key[0]), int(donor_key[1]), donor_key[2])
                            if tup not in smoke_donors:
                                gen_seed = (args.seed + hash(tup)) % (2**32)
                                g = torch.Generator().manual_seed(gen_seed)
                                smoke_donors[tup] = torch.randn(
                                    len(target["suffix_positions"]), hidden_dim, generator=g
                                ).to(torch_dtype_t)
                                n_smoke_donors_made += 1
                            donor = smoke_donors[tup]
                        if donor is None:
                            raise RuntimeError(f"missing donor for key {donor_key}")
                    result = run_one_trial(
                        model, tokenizer, layer_module, target, donor,
                        max_new_tokens=max_new_tokens, device=device,
                        smoke_input_ids=smoke_input,
                        log_replacement=log,
                    )
                    if log.get("fired"):
                        n_hooks_fired += 1
                except Exception as e:
                    print(f"  ERROR target {i} cond={condition_name}: {e}", file=sys.stderr)
                    traceback.print_exc()
                    result = {"emitted_action": None, "error": str(e), "generated_text_tail": ""}

                row = {
                    "target_idx": i,
                    "target_traj": target["target_traj"],
                    "target_step_id": target["target_step_id"],
                    "target_pos": target["target_pos"],
                    "target_variant": target["target_variant"],
                    "donor_variant_cross": target["donor_variant_cross"],
                    "target_agent_action": target["target_agent_action"],
                    "opt_target_variant": target["opt_target_variant"],
                    "opt_donor_variant": target["opt_donor_variant"],
                    "condition": condition_name,
                    "emitted_action": result["emitted_action"],
                    "generated_text_tail": result.get("generated_text_tail", ""),
                    "error": result.get("error"),
                }
                if args.smoke:
                    row["_smoke_log"] = log
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                n_written += 1
            if (i + 1) % 5 == 0 or args.smoke:
                print(f"  done {i+1}/{len(targets)} targets ({n_written} trials)")

    print(f"Wrote {n_written} trials to {args.output}")
    if args.smoke:
        print(f"  (smoke mode generated {n_smoke_donors_made} random donor tensors of "
              f"shape (3, {hidden_dim}))")
        print(f"  hooks fired: {n_hooks_fired}/{n_written}")
        print(f"  expected: {n_written} - n_baseline = {n_written - len(targets)} "
              f"(baseline runs use donor=None and the hook fires but doesn't replace anything)")


if __name__ == "__main__":
    main()
