"""Run the flag-swap intervention.

Six conditions, each sampling N traces at T = 0.7:

  baseline-a       : prompt (a), no hook
  baseline-b       : prompt (b), no hook
  self-a           : prompt (a) + hook replacing layer-15 prompt-suffix
                     positions with act_a (no-op sanity)
  self-b           : prompt (b) + hook replacing with act_b
  swap-a-from-b    : prompt (a) + hook replacing with act_b (the test)
  swap-b-from-a    : prompt (b) + hook replacing with act_a (mirror)

T = 0.7 is the project rule for inference. Distributional comparison
across N samples per condition.

Real run: GPU + openai/gpt-oss-20b. Smoke run: any small CPU model
plus random donors of that model's hidden_dim.
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


def make_replace_hook(positions, donor_per_token, log=None):
    """Forward hook: at the prompt-processing pass (seq_len > 1), once,
    overwrite hidden[:, pos, :] with donor_per_token[i]. No-op if donor
    is None.
    """
    state = {"applied": False}

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden, *rest = output
        else:
            hidden, rest = output, None
        if state["applied"] or hidden.shape[1] <= 1:
            return output
        if donor_per_token is not None:
            n_pos, hd = donor_per_token.shape
            assert n_pos == len(positions), f"positions mismatch: {n_pos} vs {len(positions)}"
            assert hd == hidden.shape[2], f"hidden_dim mismatch: donor={hd} vs model={hidden.shape[2]}"
            for i, pos in enumerate(positions):
                if pos >= hidden.shape[1]:
                    continue
                pre = hidden[0, pos, :].detach().float().norm().item()
                v = donor_per_token[i].to(dtype=hidden.dtype, device=hidden.device)
                hidden[:, pos, :] = v   # broadcasts across batch
                post = hidden[0, pos, :].detach().float().norm().item()
                if log is not None:
                    log.setdefault("replacements", []).append(
                        {"position": int(pos), "pre_norm": pre, "post_norm": post}
                    )
        state["applied"] = True
        if log is not None:
            log["fired"] = True
            log["seq_len"] = int(hidden.shape[1])
            log["batch_size"] = int(hidden.shape[0])
        if rest is not None:
            return (hidden,) + tuple(rest)
        return hidden

    return hook


def get_layer_module(model, layer_idx):
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
    raise RuntimeError("layer module not found")


def run_condition(model, tokenizer, layer_module, prompt_row, donor, n_samples,
                  temperature, max_new_tokens, device, smoke_input_ids=None,
                  batch_size=None):
    """Run one condition: prompt + (optional) hook + N sampled generations.

    Splits N samples into chunks of ``batch_size`` (defaults to N — one
    big batch). Each chunk is one `generate()` call; the hook fires once
    per chunk during prompt processing. Smaller batches reduce peak GPU
    memory at the cost of repeated prompt-processing forward passes.

    Returns: list of {action, text_tail} dicts (length n_samples) and
    the merged hook log from the FIRST chunk (subsequent chunks just
    re-fire the same hook).
    """
    if smoke_input_ids is not None:
        input_ids = smoke_input_ids.to(device)
    else:
        input_ids = torch.tensor([prompt_row["full_token_ids"]], dtype=torch.long, device=device)
    positions = prompt_row["suffix_positions"]

    if batch_size is None or batch_size <= 0:
        batch_size = n_samples

    out: list[dict] = []
    log: dict = {}
    n_remaining = n_samples
    chunk_idx = 0
    while n_remaining > 0:
        this_batch = min(batch_size, n_remaining)
        chunk_log: dict = {}
        handle = layer_module.register_forward_hook(
            make_replace_hook(positions, donor, log=chunk_log)
        )
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=this_batch,
                    pad_token_id=tokenizer.eos_token_id or 0,
                    use_cache=True,
                )
        finally:
            handle.remove()

        for k in range(output_ids.shape[0]):
            new_token_ids = output_ids[k, input_ids.shape[1]:]
            text = tokenizer.decode(new_token_ids, skip_special_tokens=False)
            out.append({"action": parse_action(text), "text_tail": text[-300:]})

        if chunk_idx == 0:
            log = chunk_log
            log["n_chunks"] = (n_samples + batch_size - 1) // batch_size
            log["batch_size_per_chunk"] = batch_size
        del output_ids
        n_remaining -= this_batch
        chunk_idx += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, default=Path("flag_swap/prompts.jsonl"))
    ap.add_argument("--act-a", type=Path, default=Path("flag_swap/act_a.pt"))
    ap.add_argument("--act-b", type=Path, default=Path("flag_swap/act_b.pt"))
    ap.add_argument("--output", type=Path, default=Path("flag_swap/results.jsonl"))
    ap.add_argument("--model-id", type=str, default="openai/gpt-oss-20b")
    ap.add_argument("--smoke-model", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--n-samples", type=int, default=30,
                    help="N samples per condition. Smoke default is 4.")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="Samples per generate() call. 0 = one big batch "
                         "of n_samples. Reduce if OOM (e.g. 4 or 8 if "
                         "the model is dequantised to bf16 and leaves "
                         "little headroom on an 80GB H100).")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature. Project rule: T=0.7, never 0.")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--torch-dtype", type=str, default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.temperature <= 0.0:
        raise SystemExit("Project rule: T must be > 0 (use 0.7). "
                         "Pass --temperature 0.7 (default).")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        model_id = args.smoke_model or "gpt2"
        device_map = "cpu"
        torch_dtype = "float32"
        max_new_tokens = min(args.max_new_tokens, 32)
        n_samples = args.n_samples if args.n_samples != 30 else 4
        print(f"SMOKE: model={model_id}, n_samples={n_samples}, max_new_tokens={max_new_tokens}, "
              f"random donors at the model's hidden_dim (scientific result invalid; orchestration only).")
    else:
        model_id = args.smoke_model or args.model_id
        device_map = args.device_map
        torch_dtype = args.torch_dtype
        max_new_tokens = args.max_new_tokens
        n_samples = args.n_samples

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

    # Load prompts
    rows = [json.loads(line) for line in open(args.prompts) if line.strip()]
    by_label = {r["label"]: r for r in rows}
    if "a" not in by_label or "b" not in by_label:
        raise SystemExit(f"prompts.jsonl must contain rows labelled 'a' and 'b'; got {[r['label'] for r in rows]}")
    prompt_a = by_label["a"]
    prompt_b = by_label["b"]
    print(f"prompt a: variant={prompt_a['variant']} pos={prompt_a['pos']} "
          f"recorded={prompt_a['agent_action_recorded']}")
    print(f"prompt b: variant={prompt_b['variant']} pos={prompt_b['pos']} "
          f"recorded={prompt_b['agent_action_recorded']}")

    # Load activations (will be replaced by random in smoke if shape mismatches)
    act_a = torch.load(args.act_a, map_location="cpu", weights_only=True)
    act_b = torch.load(args.act_b, map_location="cpu", weights_only=True)
    if act_a.shape[-1] != hidden_dim:
        if args.smoke:
            g = torch.Generator().manual_seed(args.seed)
            act_a = torch.randn(act_a.shape[0], hidden_dim, generator=g).to(torch_dtype_t)
            act_b = torch.randn(act_b.shape[0], hidden_dim, generator=g).to(torch_dtype_t)
            print(f"SMOKE: replaced real activations with random donors of shape {tuple(act_a.shape)}")
        else:
            raise SystemExit(f"act_a hidden_dim {act_a.shape[-1]} != model hidden_dim {hidden_dim}")

    # Smoke input: gpt-oss-20b token IDs are out of range for gpt2 etc., so
    # build a safe-vocab synthetic input of the same length.
    smoke_inputs = {}
    if args.smoke:
        pad_tok = tokenizer.eos_token_id or 0
        if pad_tok is None or pad_tok < 0:
            pad_tok = 0
        vocab = getattr(tokenizer, "vocab_size", None) or model.config.vocab_size
        pad_tok = min(int(pad_tok), int(vocab) - 1)
        smoke_inputs["a"] = torch.full((1, prompt_a["prompt_len"]), pad_tok, dtype=torch.long)
        smoke_inputs["b"] = torch.full((1, prompt_b["prompt_len"]), pad_tok, dtype=torch.long)

    torch.manual_seed(args.seed)

    conditions = [
        ("baseline-a",     prompt_a, None),
        ("baseline-b",     prompt_b, None),
        ("self-a",         prompt_a, act_a),
        ("self-b",         prompt_b, act_b),
        ("swap-a-from-b",  prompt_a, act_b),
        ("swap-b-from-a",  prompt_b, act_a),
    ]

    n_written = 0
    with open(args.output, "w") as fout:
        for cond_idx, (cond_name, prompt_row, donor) in enumerate(conditions):
            bs_str = f"chunked (B={args.batch_size})" if args.batch_size > 0 else "single batch"
            print(f"\n=== {cond_name} (N={n_samples}, {bs_str}) ===")
            try:
                samples, log = run_condition(
                    model, tokenizer, layer_module, prompt_row, donor,
                    n_samples=n_samples, temperature=args.temperature,
                    max_new_tokens=max_new_tokens, device=device,
                    smoke_input_ids=smoke_inputs.get(prompt_row["label"]),
                    batch_size=args.batch_size if args.batch_size > 0 else None,
                )
            except torch.cuda.OutOfMemoryError as e:
                print(f"  CUDA OOM: {e}", file=sys.stderr)
                if cond_idx == 0:
                    raise SystemExit(
                        "First condition OOM'd. Aborting before writing 180 "
                        "rows of None. Re-run with --batch-size 4 (or smaller) "
                        "or install MXFP4 deps:\n"
                        "    uv pip install 'triton>=3.4.0' kernels"
                    )
                samples = [{"action": None, "text_tail": "", "error": "OOM"}
                           for _ in range(n_samples)]
                log = {"error": "OOM"}
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                traceback.print_exc()
                samples = [{"action": None, "text_tail": "", "error": str(e)}
                           for _ in range(n_samples)]
                log = {"error": str(e)}

            # Print summary
            from collections import Counter
            ac = Counter(s.get("action") for s in samples)
            print(f"  action distribution: {dict(ac)}")
            print(f"  hook log: fired={log.get('fired')} seq_len={log.get('seq_len')} "
                  f"batch_size={log.get('batch_size')} replacements={len(log.get('replacements', []))}")

            for k, s in enumerate(samples):
                row = {
                    "condition": cond_name,
                    "prompt_label": prompt_row["label"],
                    "prompt_variant": prompt_row["variant"],
                    "sample_idx": k,
                    "emitted_action": s["action"],
                    "text_tail": s.get("text_tail", ""),
                    "error": s.get("error"),
                }
                if k == 0:   # only attach log to first row of each condition
                    row["_hook_log"] = log
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                n_written += 1

    print(f"\nWrote {n_written} rows to {args.output}")


if __name__ == "__main__":
    main()
