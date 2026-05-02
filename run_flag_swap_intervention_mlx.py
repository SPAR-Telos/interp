"""Flag-swap intervention via MLX (Apple Silicon, MXFP4 gpt-oss-20b).

The HF/PyTorch path can't run gpt-oss-20b on a 48-GB Mac because
MXFP4 quantisation requires Triton (CUDA-only) and dequantising to
bf16 needs ~73 GB. MLX runs gpt-oss-20b at MXFP4 natively on Apple
GPU, ~14 GB resident.

To avoid the saved-vs-runtime activation mismatch that broke the
RunPod run, this script **re-extracts `act_a` and `act_b` from
the running MLX model** before the swap conditions, so self-* is a
true no-op by construction.

Six conditions × N samples at T = 0.7 (project rule).

Hook mechanism: monkey-patch `model.model.layers[L].__call__` to
optionally (a) capture the layer's output at the 3 prompt-suffix
positions, or (b) overwrite those positions with a donor tensor.
The patch only fires on the prompt-processing pass (when seq_len > 1);
during generation steps (seq_len == 1), it's a no-op.

Mac-only.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


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


class HookedLayer:
    """Wraps a TransformerBlock. Replaces the parent's
    `model.layers[idx]` entry with this wrapper so the loop in
    GptOssMoeModel.__call__ calls us instead of the real layer.

    Modes:
      'passthrough' — original behaviour
      'capture'     — call original, store last n_pos positions in self.captured
      'replace'     — call original, then overwrite last n_pos positions with self.donor
    """
    def __init__(self, parent_list, idx: int, n_pos: int):
        self.parent_list = parent_list
        self.idx = idx
        self.n_pos = n_pos
        self.original_layer = parent_list[idx]
        self.mode = "passthrough"
        self.donor = None
        self.captured = None
        self.fired = False
        self.last_seq_len = None
        # Splice ourselves in.
        parent_list[idx] = self

    def __call__(self, x, mask, cache, *args, **kwargs):
        out = self.original_layer(x, mask, cache, *args, **kwargs)
        if out.shape[1] <= 1:
            return out
        self.fired = True
        self.last_seq_len = out.shape[1]
        if self.mode == "capture":
            self.captured = out[0, -self.n_pos:, :]
        elif self.mode == "replace":
            assert self.donor is not None, "donor must be set for replace mode"
            n = self.n_pos
            head = out[:, :-n, :]
            tail = mx.broadcast_to(self.donor[None, :, :], (out.shape[0], n, out.shape[2]))
            out = mx.concatenate([head, tail], axis=1)
        return out

    def reset(self):
        self.mode = "passthrough"
        self.donor = None
        self.captured = None
        self.fired = False
        self.last_seq_len = None

    def restore(self):
        """Put the original layer back into the parent list."""
        self.parent_list[self.idx] = self.original_layer


def generate_one(model, tokenizer, prompt_token_ids, sampler, max_tokens: int) -> str:
    """Run a single sampled generation. Returns the generated text."""
    pieces = []
    for chunk in stream_generate(
        model, tokenizer, prompt=prompt_token_ids,
        max_tokens=max_tokens, sampler=sampler,
    ):
        pieces.append(chunk.text)
    return "".join(pieces)


def run_condition(model, tokenizer, hook, prompt_token_ids, donor, n_samples,
                  temperature, max_new_tokens, base_seed):
    """Generate n_samples completions of `prompt_token_ids` with the
    layer-15 hook configured per the (donor or None) argument."""
    if donor is None:
        hook.mode = "passthrough"
        hook.donor = None
    else:
        hook.mode = "replace"
        hook.donor = donor
    hook.fired = False

    out = []
    log = {"fired": False, "seq_len": None, "mode": hook.mode,
           "donor_norm": float(mx.linalg.norm(donor)) if donor is not None else None}

    for k in range(n_samples):
        # Distinct random seed per sample for genuine variability.
        mx.random.seed(base_seed + k)
        sampler = make_sampler(temp=temperature)
        text = generate_one(model, tokenizer, prompt_token_ids, sampler, max_new_tokens)
        out.append({"action": parse_action(text), "text_tail": text[-300:]})
        if k == 0:
            log["fired"] = hook.fired
            log["seq_len"] = hook.last_seq_len
    return out, log


def capture_activations(model, tokenizer, hook, prompt_token_ids):
    """Forward-pass the prompt ONCE with the hook in capture mode.
    Returns the captured (n_pos, hidden_dim) mx.array."""
    hook.reset()
    hook.mode = "capture"
    hook.donor = None
    sampler = make_sampler(temp=0.0)   # need just 1 token; sampling doesn't matter
    # max_tokens=1 forces only the prompt-processing pass + 1 generation step.
    _ = generate_one(model, tokenizer, prompt_token_ids, sampler, max_tokens=1)
    cap = hook.captured
    hook.reset()
    if cap is None:
        raise RuntimeError("capture failed — hook never fired on prompt pass")
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, default=Path("flag_swap/prompts.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("flag_swap/results.jsonl"))
    ap.add_argument("--model-id", type=str, default="mlx-community/gpt-oss-20b-MXFP4-Q4")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.temperature <= 0.0:
        raise SystemExit("Project rule: T must be > 0 (use 0.7).")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} ...", flush=True)
    t0 = time.time()
    model, tokenizer = load(args.model_id)
    print(f"  loaded in {time.time() - t0:.1f}s")

    n_layers = len(model.layers)
    if args.layer >= n_layers:
        raise SystemExit(f"--layer {args.layer} out of range; model has {n_layers} layers")
    print(f"  hooking layer {args.layer}/{n_layers}")

    rows = [json.loads(l) for l in open(args.prompts) if l.strip()]
    by_label = {r["label"]: r for r in rows}
    pa, pb = by_label["a"], by_label["b"]
    print(f"  prompt a: variant={pa['variant']} pos={pa['pos']} "
          f"recorded={pa['agent_action_recorded']}  len={pa['prompt_len']}")
    print(f"  prompt b: variant={pb['variant']} pos={pb['pos']} "
          f"recorded={pb['agent_action_recorded']}  len={pb['prompt_len']}")

    # `model.layers` is a property aliasing model.model.layers (a Python list).
    hook = HookedLayer(model.model.layers, args.layer, n_pos=3)

    try:
        # ── Capture act_a, act_b from the live MLX model ───────────────
        print("\nCapturing act_a from prompt a ...", flush=True)
        act_a = capture_activations(model, tokenizer, hook, pa["full_token_ids"])
        print(f"  act_a shape={tuple(act_a.shape)} dtype={act_a.dtype} "
              f"per-token-norm={[round(float(mx.linalg.norm(act_a[i])), 1) for i in range(act_a.shape[0])]}")
        print("Capturing act_b from prompt b ...", flush=True)
        act_b = capture_activations(model, tokenizer, hook, pb["full_token_ids"])
        print(f"  act_b shape={tuple(act_b.shape)} dtype={act_b.dtype} "
              f"per-token-norm={[round(float(mx.linalg.norm(act_b[i])), 1) for i in range(act_b.shape[0])]}")
        diff = act_a.astype(mx.float32) - act_b.astype(mx.float32)
        diff_norm_per_tok = [round(float(mx.linalg.norm(diff[i])), 2) for i in range(diff.shape[0])]
        print(f"  ||act_a − act_b|| per token: {diff_norm_per_tok}")

        # ── Six conditions ─────────────────────────────────────────────
        conditions = [
            ("baseline-a",     pa, None),
            ("baseline-b",     pb, None),
            ("self-a",         pa, act_a),
            ("self-b",         pb, act_b),
            ("swap-a-from-b",  pa, act_b),
            ("swap-b-from-a",  pb, act_a),
        ]
        n_written = 0
        with open(args.output, "w") as fout:
            for cond_idx, (cond_name, prompt_row, donor) in enumerate(conditions):
                print(f"\n=== {cond_name} (N={args.n_samples}) ===", flush=True)
                t1 = time.time()
                samples, log = run_condition(
                    model, tokenizer, hook,
                    prompt_token_ids=prompt_row["full_token_ids"],
                    donor=donor,
                    n_samples=args.n_samples,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    base_seed=args.seed + cond_idx * 10_000,
                )
                from collections import Counter
                ac = Counter(s["action"] for s in samples)
                print(f"  action distribution: {dict(ac)}", flush=True)
                print(f"  hook log: {log}  elapsed={time.time() - t1:.1f}s", flush=True)
                for k, s in enumerate(samples):
                    row = {
                        "condition": cond_name,
                        "prompt_label": prompt_row["label"],
                        "prompt_variant": prompt_row["variant"],
                        "sample_idx": k,
                        "emitted_action": s["action"],
                        "text_tail": s["text_tail"],
                    }
                    if k == 0:
                        row["_hook_log"] = log
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                    n_written += 1
        print(f"\nWrote {n_written} rows to {args.output}")
    finally:
        hook.restore()


if __name__ == "__main__":
    main()
