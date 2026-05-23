"""RunPod/CUDA runner for clean size15 pre-reasoning action-probe steering.

This is the Transformers version of ``run_size15_pre_reasoning_action_probe_steering_mlx.py``.
It patches the same three prompt-suffix activations during the prompt prefill
pass, then lets the model generate normally until a natural JSON action appears
or ``--max-new-tokens`` is reached. There is no second finalization prompt and
no tail-action heuristic.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from run_size15_pre_reasoning_action_probe_steering_mlx import (
    ACTION_RE,
    DEFAULT_ACTIVATIONS_DIR,
    DEFAULT_ALPHAS,
    DEFAULT_LAYERS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SOURCE_RUN,
    DEFAULT_TRAJECTORIES_DIR,
    build_direction_bundle,
    build_paired_rows,
    classify_generated_action_text,
    collect_size15_samples,
    load_size15_trajectory_records,
    load_split_by_state,
    normalize_args,
    paired_stats_by_layer_alpha,
    pairwise_margin_shift,
    parse_float_selection,
    parse_int_selection,
    prepare_output_dir,
    reconstruct_prompt_row,
    resolve_steering_suffix_positions,
    select_test_nonoptimal_targets,
    summarize_steering_rows,
    write_csv,
    write_json,
    write_plots,
)

DEFAULT_OUTPUT_ROOT = Path("results/size15_pre_reasoning_action_probe_steering_hf")
DEFAULT_MODEL_ID = "openai/gpt-oss-20b"


class PromptAddHook:
    """Forward hook that adds a direction to selected prompt positions once."""

    def __init__(self, selected_positions: list[int], direction: torch.Tensor, scale: float):
        self.selected_positions = list(selected_positions)
        self.direction = direction
        self.scale = float(scale)
        self.applied = False
        self.log: dict[str, Any] = {
            "fired": False,
            "mode": "add",
            "scale": self.scale,
            "selected_prompt_positions": self.selected_positions,
            "patched_prompt_positions": [],
            "pre_norm": None,
            "post_norm": None,
        }

    def __call__(self, _module, _inputs, output):
        if self.applied:
            return output
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        if hidden.ndim != 3 or hidden.shape[1] <= max(self.selected_positions, default=-1):
            return output

        patched = hidden.clone()
        direction = self.direction.to(device=hidden.device, dtype=hidden.dtype)
        touched = []
        before = []
        after = []
        for row_idx, abs_pos in enumerate(self.selected_positions):
            before.append(patched[:, abs_pos : abs_pos + 1, :].detach().float())
            patched[:, abs_pos, :] = patched[:, abs_pos, :] + direction[row_idx] * self.scale
            after.append(patched[:, abs_pos : abs_pos + 1, :].detach().float())
            touched.append(abs_pos)
        self.applied = True
        self.log.update(
            {
                "fired": True,
                "seq_len": int(hidden.shape[1]),
                "batch_size": int(hidden.shape[0]),
                "patched_prompt_positions": touched,
                "pre_norm": float(torch.linalg.norm(torch.cat(before, dim=1)).item()),
                "post_norm": float(torch.linalg.norm(torch.cat(after, dim=1)).item()),
            }
        )
        if rest is not None:
            return (patched, *rest)
        return patched


class JsonActionStoppingCriteria:
    """Stop once generated text contains a JSON action field."""

    def __init__(self, tokenizer: Any, prompt_len: int):
        from transformers import StoppingCriteria

        self._base = StoppingCriteria
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.stopped = False

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor, **_kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0, self.prompt_len :], skip_special_tokens=False)
        self.stopped = ACTION_RE.search(text) is not None
        return self.stopped


def get_layer_module(model: Any, layer_idx: int) -> torch.nn.Module:
    """Return a mutable transformer block by common Hugging Face model layouts."""
    candidates = [
        ("model.model.layers", lambda m: m.model.layers[layer_idx]),
        ("model.layers", lambda m: m.layers[layer_idx]),
        ("model.transformer.h", lambda m: m.transformer.h[layer_idx]),
        ("transformer.h", lambda m: m.transformer.h[layer_idx]),
    ]
    for name, getter in candidates:
        try:
            module = getter(model)
            print(f"  using {name}[{layer_idx}] ({type(module).__name__})", flush=True)
            return module
        except (AttributeError, IndexError):
            continue
    raise RuntimeError(f"Could not find transformer layer {layer_idx}")


def dtype_from_name(name: str) -> torch.dtype:
    """Parse a torch dtype CLI value."""
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported --torch-dtype {name!r}; choose one of {sorted(mapping)}")
    return mapping[name]


def model_input_device(model: Any) -> torch.device:
    """Return the device to use for input_ids."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_one_hf(
    model: Any,
    tokenizer: Any,
    prompt_token_ids: list[int],
    layer_module: torch.nn.Module,
    selected_positions: list[int],
    direction: torch.Tensor,
    alpha: float,
    *,
    generation_seed: int,
    temperature: float,
    max_new_tokens: int,
    stop_on_action: bool,
) -> tuple[str, bool, dict[str, Any]]:
    """Generate one clean autonomous completion with a prompt-prefill patch."""
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(generation_seed)
    torch.manual_seed(generation_seed)
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=model_input_device(model))
    hook = PromptAddHook(selected_positions, direction, alpha)
    handle = layer_module.register_forward_hook(hook)
    stopping = None
    if stop_on_action:
        from transformers import StoppingCriteriaList

        stopping = JsonActionStoppingCriteria(tokenizer, input_ids.shape[1])
        stopping_criteria = StoppingCriteriaList([stopping])
    else:
        stopping_criteria = None
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )
    finally:
        handle.remove()
    new_token_ids = output_ids[0, input_ids.shape[1] :]
    text = tokenizer.decode(new_token_ids, skip_special_tokens=False)
    stopped = bool(stopping.stopped) if stopping is not None else False
    return text, stopped, hook.log


def read_existing_rows(path: Path) -> list[dict[str, Any]]:
    """Read already-written JSONL rows when resuming a RunPod job."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_key(row: dict[str, Any]) -> tuple[str, int, int, float]:
    """Return the resume key for one generation row."""
    return (str(row["trajectory_name"]), int(row["layer"]), int(row["seed_index"]), float(row["alpha"]))


def run_steering_generation_hf(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    """Run the Hugging Face clean autonomous steering generations."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records, skipped = load_size15_trajectory_records(args.activations_dir, args.trajectories_dir, None)
    samples = collect_size15_samples(records, args.grid_size)
    split_by_state = load_split_by_state(args.source_run)
    targets = select_test_nonoptimal_targets(samples, split_by_state, args.max_samples)
    if not targets:
        raise RuntimeError("No held-out non-optimal targets selected")

    print(f"Selected {len(targets)} held-out non-optimal targets; skipped records={skipped}", flush=True)
    print(
        f"Loading HF model {args.model_id} dtype={args.torch_dtype} device_map={args.device_map} ...",
        flush=True,
    )
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype_from_name(args.torch_dtype),
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()
    print(f"Loaded model in {time.time() - t0:.1f}s", flush=True)

    calibration = {
        "skipped": True,
        "reason": "HF runner performs generation-only clean steering; use saved probe metrics for calibration.",
    }
    write_json(output_dir / "hf_probe_calibration.json", calibration)

    rows: list[dict[str, Any]] = []
    jsonl_path = output_dir / "steering_rows.jsonl"
    if args.resume:
        rows = read_existing_rows(jsonl_path)
    done = {row_key(row) for row in rows}
    mode = "a" if args.resume and jsonl_path.exists() else "w"
    with jsonl_path.open(mode) as fout:
        for layer_idx, layer in enumerate(args.layers):
            checkpoint_path = (
                args.source_run / "checkpoints" / f"probe_optimal_action_pre_reasoning_suffix_layer{layer}_linear.pt"
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"missing probe checkpoint: {checkpoint_path}")
            layer_module = get_layer_module(model, layer)
            for sample_idx, target in enumerate(targets):
                prompt_row = reconstruct_prompt_row(
                    target.sample,
                    prefill_analysis_channel=args.prefill_analysis_channel,
                )
                suffix_rel, suffix_positions = resolve_steering_suffix_positions(
                    prompt_row, args.prompt_suffix_indices
                )
                direction_bundle = build_direction_bundle(
                    checkpoint_path, target.optimal_actions, target.original_action
                )
                unit_delta = direction_bundle.margin_unit_delta.detach().cpu().float()
                direction = torch.stack([unit_delta for _ in suffix_positions], dim=0)
                for seed_idx in range(args.n_seeds):
                    generation_seed = args.seed + layer_idx * 1_000_000 + sample_idx * 10_000 + seed_idx
                    for alpha in args.alphas:
                        key = (target.sample.name, layer, seed_idx, float(alpha))
                        if key in done:
                            continue
                        text, stopped_on_action_regex, hook_log = generate_one_hf(
                            model,
                            tokenizer,
                            prompt_row.full_token_ids,
                            layer_module,
                            suffix_positions,
                            direction,
                            float(alpha),
                            generation_seed=generation_seed,
                            temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens,
                            stop_on_action=args.stop_on_action,
                        )
                        generated_action, tail_fallback_action, parse_status = classify_generated_action_text(
                            text,
                            tail_action_fallback=False,
                        )
                        row = {
                            "trajectory_name": target.sample.name,
                            "sample_index": target.sample.sample_index,
                            "state_key": target.sample.state_key,
                            "split": "test",
                            "layer": layer,
                            "alpha": float(alpha),
                            "seed_index": seed_idx,
                            "generation_seed": generation_seed,
                            "original_action": target.original_action,
                            "optimal_actions": list(target.optimal_actions),
                            "generated_action": generated_action,
                            "tail_fallback_action": tail_fallback_action,
                            "generated_action_is_optimal": generated_action in set(target.optimal_actions),
                            "action_changed_from_original": generated_action != target.original_action,
                            "parse_status": parse_status,
                            "generated_text_n_chars": len(text),
                            "stopped_on_action_regex": bool(stopped_on_action_regex),
                            "finalization_fallback_used": False,
                            "tail_action_fallback_enabled": False,
                            "clean_autonomous": True,
                            "generation_protocol": "clean_autonomous_hf",
                            "finalization_text_n_chars": 0,
                            "finalization_stopped_on_action_regex": False,
                            "finalization_text_tail": "",
                            "generated_text_tail": text[-1000:],
                            "prompt_suffix_indices": args.prompt_suffix_indices,
                            "prefill_analysis_channel": bool(args.prefill_analysis_channel),
                            "generation_prefill_token_ids": prompt_row.generation_prefill_token_ids,
                            "suffix_relative_indices": suffix_rel,
                            "selected_prompt_positions": suffix_positions,
                            "direction_raw_norm": direction_bundle.raw_direction_norm,
                            "intended_margin_shift": pairwise_margin_shift(
                                direction_bundle.w_margin,
                                direction_bundle.scaler_std,
                                direction_bundle.margin_unit_delta * float(alpha),
                            ),
                            "hook_log": hook_log,
                        }
                        fout.write(json.dumps(row) + "\n")
                        fout.flush()
                        rows.append(row)
                        done.add(key)
                if (sample_idx + 1) % 10 == 0:
                    print(
                        f"layer={layer} completed {sample_idx + 1}/{len(targets)} targets rows={len(rows)}",
                        flush=True,
                    )
    return rows


def write_outputs_hf(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Write steering summaries using the shared output schema."""
    write_json(output_dir / "steering_rows.json", rows)
    write_csv(output_dir / "steering_rows.csv", rows)
    summaries = summarize_steering_rows(rows)
    paired_rows = build_paired_rows(rows)
    paired_stats = paired_stats_by_layer_alpha(paired_rows)
    write_json(output_dir / "steering_summary_by_layer_alpha.json", summaries)
    write_csv(output_dir / "steering_summary_by_layer_alpha.csv", summaries)
    write_json(output_dir / "paired_steering_rows.json", paired_rows)
    write_csv(output_dir / "paired_steering_rows.csv", paired_rows)
    write_json(output_dir / "paired_steering_stats_by_layer_alpha.json", paired_stats)
    write_csv(output_dir / "paired_steering_stats_by_layer_alpha.csv", paired_stats)
    write_plots(output_dir, summaries, paired_stats, args.layers)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner": "hf",
        "model_id": args.model_id,
        "source_run": str(args.source_run),
        "trajectories_dir": str(args.trajectories_dir),
        "activations_dir": str(args.activations_dir),
        "layers": args.layers,
        "alphas": args.alphas,
        "n_seeds": args.n_seeds,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "finalization_fallback": False,
        "finalization_max_new_tokens": None,
        "tail_action_fallback": False,
        "clean_autonomous": True,
        "generation_protocol": "clean_autonomous_hf",
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "max_samples": args.max_samples,
        "row_count": len(rows),
        "paired_row_count": len(paired_rows),
        "parse_status_counts": dict(Counter(row["parse_status"] for row in rows)),
        "direction_formula": "delta = alpha * ((mean(W_opt)-W_wrong)/std) / ||(mean(W_opt)-W_wrong)/std||^2",
        "prompt_suffix_indices": args.prompt_suffix_indices,
        "prefill_analysis_channel": bool(args.prefill_analysis_channel),
        "outputs": {
            "steering_rows_jsonl": str(output_dir / "steering_rows.jsonl"),
            "steering_rows_csv": str(output_dir / "steering_rows.csv"),
            "paired_steering_rows_json": str(output_dir / "paired_steering_rows.json"),
            "paired_steering_stats_json": str(output_dir / "paired_steering_stats_by_layer_alpha.json"),
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the RunPod/CUDA CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-dir", type=Path, default=DEFAULT_TRAJECTORIES_DIR)
    parser.add_argument("--activations-dir", type=Path, default=DEFAULT_ACTIVATIONS_DIR)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--layers", type=str, default=",".join(str(layer) for layer in DEFAULT_LAYERS))
    parser.add_argument("--alphas", type=str, default=",".join(f"{alpha:g}" for alpha in DEFAULT_ALPHAS))
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--prompt-suffix-indices", type=str, default="-3:-1")
    parser.add_argument("--stop-on-action", action="store_true", default=True)
    parser.add_argument("--no-stop-on-action", action="store_false", dest="stop_on_action")
    parser.add_argument(
        "--no-prefill-analysis-channel",
        action="store_false",
        dest="prefill_analysis_channel",
        help="Do not prefill the standard analysis channel tokens before generation.",
    )
    parser.set_defaults(prefill_analysis_channel=True)
    return parser


def normalize_hf_args(args: argparse.Namespace) -> argparse.Namespace:
    """Normalize shared and HF-specific arguments."""
    args.layers = parse_int_selection(args.layers) if isinstance(args.layers, str) else list(args.layers)
    args.alphas = parse_float_selection(args.alphas) if isinstance(args.alphas, str) else list(args.alphas)
    args.clean_autonomous = True
    args.finalization_fallback = False
    args.tail_action_fallback = False
    args.finalization_max_new_tokens = None
    args.generation_protocol = "clean_autonomous_hf"
    # Reuse the shared validation for alpha/seed/temperature/max-token semantics.
    args = normalize_args(args)
    args.generation_protocol = "clean_autonomous_hf"
    return args


def prepare_hf_output_dir(args: argparse.Namespace) -> Path:
    """Prepare output directory, preserving files when --resume is requested."""
    if args.resume:
        output_dir = args.output_dir
        if output_dir is None:
            raise SystemExit("--resume requires --output-dir")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    return prepare_output_dir(args.output_root, args.output_dir, args.overwrite)


def main() -> None:
    """CLI entrypoint."""
    args = normalize_hf_args(build_arg_parser().parse_args())
    output_dir = prepare_hf_output_dir(args)
    rows = run_steering_generation_hf(args, output_dir)
    write_outputs_hf(output_dir, rows, args)
    print(f"Wrote HF steering outputs to {output_dir}")


if __name__ == "__main__":
    main()
