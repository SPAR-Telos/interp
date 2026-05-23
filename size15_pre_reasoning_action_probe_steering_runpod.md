# Size15 Clean Steering on RunPod

Use this for the clean autonomous version of the size15 pre-reasoning steering experiment on a RunPod NVIDIA GPU.

## Required Inputs

The runner needs:

- `data/trajectories_train_single_step/size15/*.json`
- `results/size15_mistake_aligned_action_probes/20260521T192354Z/split_manifest.json`
- `results/size15_mistake_aligned_action_probes/20260521T192354Z/checkpoints/probe_optimal_action_pre_reasoning_suffix_layer{7,15,23}_linear.pt`

The full saved activation tree is not required for generation if calibration is skipped, but the default paths are still accepted. If the activation directory is absent, the runner discovers samples from trajectory JSON files.

## Setup

```bash
git checkout flag-swap
uv sync
```

If the RunPod base image already has a CUDA PyTorch build, keep that build rather than reinstalling CPU wheels. The runner uses `transformers` with `device_map=auto`.

## Pilot

Run a small clean pilot first:

```bash
uv run python run_size15_pre_reasoning_action_probe_steering_hf.py \
  --model-id openai/gpt-oss-20b \
  --torch-dtype bfloat16 \
  --device-map auto \
  --max-new-tokens 6144 \
  --max-samples 20 \
  --n-seeds 2
```

Check `steering_rows.jsonl` for parse failures and generated lengths before launching the full grid.

## Full Run

```bash
uv run python run_size15_pre_reasoning_action_probe_steering_hf.py \
  --model-id openai/gpt-oss-20b \
  --torch-dtype bfloat16 \
  --device-map auto \
  --max-new-tokens 6144 \
  --layers 7,15,23 \
  --alphas -2,-1,0,1,2 \
  --n-seeds 2
```

Use `--max-new-tokens 8192` for the most faithful cap from the Mac pilot. Use `6144` if wall-clock cost is the priority; it should avoid many cap-burning failures while retaining most late JSON successes observed in the partial clean run.

## Resume

If the pod stops, resume with the same output directory:

```bash
uv run python run_size15_pre_reasoning_action_probe_steering_hf.py \
  --resume \
  --output-dir results/size15_pre_reasoning_action_probe_steering_hf/<RUN_ID> \
  --model-id openai/gpt-oss-20b \
  --torch-dtype bfloat16 \
  --device-map auto \
  --max-new-tokens 6144 \
  --layers 7,15,23 \
  --alphas -2,-1,0,1,2 \
  --n-seeds 2
```

## Outputs

Each run writes:

- `steering_rows.jsonl/.json/.csv`
- `steering_summary_by_layer_alpha.json/.csv`
- `paired_steering_rows.json/.csv`
- `paired_steering_stats_by_layer_alpha.json/.csv`
- `run_manifest.json`
- `plots/steered_optimal_rate_by_alpha_layer.svg`
- `plots/paired_delta_optimal_rate_by_alpha_layer.svg`

Do not commit generated result directories, model caches, trajectory data, activation data, or probe checkpoint artifacts.
