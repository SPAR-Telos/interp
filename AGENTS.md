# Repository Guidelines

## Project Structure & Module Organization

`telos_interp/` contains the Python package for trajectory loading, grid utilities, probe models, shared training code, and the Tyro-based CLI. CLI commands live under `telos_interp/commands/<command>/` with `*_fn.py` implementations, optional `*_utils.py`, and command-specific README files. Tests are in `tests/` and follow `test_*.py`. Experiment configuration templates are in `configs/`, standalone analysis scripts are at the repository root, and generated or published artifacts live in `results/`, `reports/`, `flag_swap/`, and `telos_interp/trace_viewer/`.

## Build, Test, and Development Commands

Use Python 3.10+ and `uv` for dependency management.

```bash
uv sync                 # create/update .venv from pyproject.toml and uv.lock
make install            # install runtime dependencies and editable package
make install-dev        # install all extras, pre-commit hooks, editable package
make test               # run pytest with project config and verbose output
make check-style        # check Ruff formatting and linting without fixes
make fix-style          # apply Ruff formatting and auto-fixes
uv run interp-cli --help # inspect available CLI commands
```

For focused testing, run `uv run pytest tests/test_grid_utils.py -vv`. To skip expensive cases, use `uv run pytest -m "not slow and not require_cuda_gpu"`.

## Coding Style & Naming Conventions

Ruff is the formatter and linter; configuration is in `pyproject.toml`. Keep Python code compatible with Python 3.10+, use a 119-character line length, sorted imports, and Google-style docstrings for public functions. Prefer existing package helpers over duplicating parsing, loading, or training logic. Name tests `test_<behavior>.py`; name command implementation files `<command>_fn.py`; keep CLI subcommand names in snake_case.

## Testing Guidelines

Pytest discovers tests from `tests/` using `test_*.py`. Add or update focused tests when changing shared utilities, activation loading, probe models, training behavior, or command contracts. Mark long-running tests with `slow` and GPU-dependent tests with `require_cuda_gpu`. Run `make test` before opening a PR, or document any skipped tests and why.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects, often with a scope prefix such as `flag-swap:`. Use concise messages like `flag-swap: add MLX intervention runner` or `fix probe normalization`. PRs should describe the motivation, summarize behavioral changes, link related issues or experiments, list validation commands, and include screenshots or report snippets when visualizations or generated outputs change.

## Agent-Specific Instructions

Do not overwrite generated datasets, probe checkpoints, or result files unless the task explicitly requires it. Treat `configs/*.conf` as shell-style experiment templates, not TOML. When changing CLI behavior, update the relevant command README and verify `uv run interp-cli <command> --help`.

## Experiment Ledger

Keep these runs and interpretation boundaries in mind when answering follow-up questions. Result folders are generated
artifacts; do not overwrite or commit them unless explicitly asked.

Seed12 and Seed12_T1:

- Seed12 action-probe reports are under `reports/seed12_action_probes/` and
  `reports/seed12_t1_action_probes/`. The Seed12_T1 action probes used
  `data/activations/Seed12_T1_six_samples` and split by logical state, with separate layer panels and no best-over-layer
  aggregation for final interpretation.
- Seed12_T1 position probes are implemented in `run_seed12_position_probes.py`, with focused tests in
  `tests/test_seed12_position_probes.py`. The latest useful run is
  `results/seed12_t1_position_probes/20260515T124224Z/`. It trains coordinate-conditioned binary probes
  `(activation, row, col) -> agent/not-agent`, evaluates full-grid probability distributions, and plots entropy curves
  separately for layers `7`, `15`, and `23`.
- Seed12 belief-action-gap work is summarized in the next section. Its latest preferred report remains
  `results/seed12_belief_action_gap/20260512T125452Z/report.md`; later folders exist but should not silently replace the
  interpretation notes below.
- Seed12 optimal-action steering reports exist under `reports/seed12_optimal_action_steering/`. Treat these as exploratory
  steering sweeps with many variants; read the specific sub-run report before quoting a result.

Size15 position and action-probe analysis:

- Size15 activation extraction added capped token-group syntax such as `@analysis/10#10` in
  `telos_interp/commands/gather_activations/gather_activations_utils.py`. This was used to keep only common CoT
  checkpoints while extracting size15 activations locally.
- Size15 position probes are implemented in `run_size15_position_probes.py`, with tests in
  `tests/test_size15_position_probes.py`. The full `cot_rank_0..9` run is
  `results/size15_position_probes/20260520T170300Z/`; a reduced `cot_rank_0..3` rerun is
  `results/size15_position_probes_cot0_3/20260520T200259Z/`. Both use exact rendered-state split keys and keep layers
  separate. The `cot0_3` rerun was motivated by the usable sample count drop at later CoT ranks.
- Size15 mistake-aligned action probes are implemented in `run_size15_mistake_aligned_action_probes.py`, with tests in
  `tests/test_size15_mistake_aligned_action_probes.py`. The main run is
  `results/size15_mistake_aligned_action_probes/20260521T192354Z/`. It uses BFS optimal-action sets, marks the first
  high-confidence false claim in non-optimal CoTs when possible, and falls back to final non-optimal action marking when
  no earlier false claim is parseable.
- The same size15 mistake-aligned run also contains the correct-vs-wrong checkpoint comparison outputs:
  `checkpoint_group_*`, `checkpoint_probe_gap_*`, and `wrong_checkpoint_mistake_timing_*`. Use these for comparing
  correct and wrong samples on the shared absolute checkpoint axis; do not invent a fake mistake point for correct
  samples.
- The action probes in the size15 mistake-aligned run are linear probes. The true-action probe label is the final
  `agent_action`; the optimal-action probe label is a multi-hot BFS shortest-path optimal next-action set, evaluated by
  top-1-in-optimal-set accuracy.

Size15 steering:

- The first full size15 pre-reasoning steering run is
  `results/size15_pre_reasoning_action_probe_steering/20260522T223426Z/`, with report
  `steering_analysis_report.md`. It used the MLX runner with a two-stage finalization fallback, because autonomous
  generation often failed to emit parseable final JSON. Treat it as a controlled intervention diagnostic; the result was
  a null effect with all paired CIs crossing zero.
- Partial/failed steering runs before the full diagnostic run are
  `20260522T221544Z`, `20260522T222058Z`, and `20260522T223328Z`. Do not use them as final results.
- The partial clean autonomous Mac run is
  `results/size15_pre_reasoning_action_probe_steering/20260523T062615Z/`. It was intentionally stopped after 70 rows.
  Clean means no finalization fallback and no tail-action heuristic. Many failures burned the whole token cap by looping
  on grid row/column counting, so this run is useful as a pilot/runtime diagnostic only.
- The current RunPod-ready clean steering implementation is commit `29b459a` on branch `flag-swap`. Prefer
  `run_size15_pre_reasoning_action_probe_steering_hf.py` on NVIDIA/RunPod and
  `run_size15_pre_reasoning_action_probe_steering_mlx.py` on Apple Silicon. See
  `size15_pre_reasoning_action_probe_steering_runpod.md` for commands and resume guidance.

## Current Seed12 Belief-Action-Gap Work

The active belief-action-gap experiment is implemented in `run_seed12_belief_action_gap.py`, with focused tests in `tests/test_seed12_belief_action_gap.py`. The latest full run is in `results/seed12_belief_action_gap/20260512T125452Z/`, and its human-readable report is `results/seed12_belief_action_gap/20260512T125452Z/report.md`.

Key interpretation notes for this experiment:

- The primary gap is an eventual-action gap under decoded belief:
  `Delta_t = J_t(a_final) - min_a J_t(a)`, with `J_t(a) = E_{s ~ q_phi(.|h_t)} C(f_env(s,a))`.
- `theta_opt` remains the primary cost model. It is learned from optimal model samples only and is very close to the oracle shortest-path value, so interpret it as task-regret-like rather than as proof of a private model utility.
- `theta_act_pre_action` is a secondary diagnostic added after discussion. It uses averaged pre-reasoning activations as state features, but it is a direct action-cost model `Q_L(s,a) = w_{L,a}^T mean_h_pre(state=s)`, not the strict next-state cost `C(mean_h_pre(f_env(s,a)))`.
- The strict next-state activation-cost version is not available from the current stored activations: 20 successor state-action pairs land in 8 goal/door/key successor states with no pre-reasoning activation. Do not fill those missing states with zeros, means, or oracle values silently.
- `post_reasoning_tail` uses the existing `-16:-14` convention and has nonzero final-answer contamination. Treat clean temporal claims as `pre_reasoning_suffix -> cot_rank_29`; keep post-tail as diagnostic only.
- The shared-position decoder control does not replicate the primary shrinking-gap pattern, so report conclusions as decoder-dependent.

Useful validation commands for this work:

```bash
.venv/bin/ruff format --check run_seed12_belief_action_gap.py tests/test_seed12_belief_action_gap.py
.venv/bin/ruff check run_seed12_belief_action_gap.py tests/test_seed12_belief_action_gap.py
.venv/bin/python -m pytest tests/test_seed12_belief_action_gap.py -vv
```

## Current Size15 Clean Steering Work

The active clean steering experiment is on branch `flag-swap`, commit `29b459a`:
`flag-swap: add size15 clean steering runners`.

Key files:

- Mac/MLX runner: `run_size15_pre_reasoning_action_probe_steering_mlx.py`
- RunPod/CUDA runner: `run_size15_pre_reasoning_action_probe_steering_hf.py`
- Focused tests: `tests/test_size15_pre_reasoning_action_probe_steering_mlx.py`
- RunPod runbook: `size15_pre_reasoning_action_probe_steering_runpod.md`

Interpretation notes:

- The clean protocol means one uninterrupted generation: patch the three pre-reasoning prompt-suffix activations during the prompt pass, then let the model naturally emit final JSON. Do not use the finalization fallback or tail-action heuristic when making clean behavioral claims.
- The older full Mac run in `results/size15_pre_reasoning_action_probe_steering/20260522T223426Z/` used a two-stage finalization fallback. Treat it as an intervention diagnostic, not a clean autonomous CoT completion result. Its main result was a controlled null result.
- A partial clean Mac run in `results/size15_pre_reasoning_action_probe_steering/20260523T062615Z/` was intentionally stopped after 70 rows. It showed that many parse failures burn the whole token cap by looping on grid counting, so use a RunPod pilot before launching the full clean grid.
- For RunPod, prefer the Hugging Face runner. It is self-contained for size15 trajectory parsing, BFS optimal-action sets, split loading, and linear probe checkpoint loading. It still needs the size15 trajectory JSONs and the prior mistake-aligned run's `split_manifest.json` plus pre-reasoning optimal-action probe checkpoints.
- Do not commit RunPod outputs, model caches, trajectory data, activation data, probe checkpoints, or `results/` artifacts.

Useful RunPod commands:

```bash
uv run python run_size15_pre_reasoning_action_probe_steering_hf.py \
  --model-id openai/gpt-oss-20b \
  --torch-dtype bfloat16 \
  --device-map auto \
  --max-new-tokens 6144 \
  --max-samples 20 \
  --n-seeds 2

uv run python run_size15_pre_reasoning_action_probe_steering_hf.py \
  --model-id openai/gpt-oss-20b \
  --torch-dtype bfloat16 \
  --device-map auto \
  --max-new-tokens 6144 \
  --layers 7,15,23 \
  --alphas -2,-1,0,1,2 \
  --n-seeds 2
```

Useful validation commands:

```bash
.venv/bin/ruff format --check run_size15_pre_reasoning_action_probe_steering_mlx.py run_size15_pre_reasoning_action_probe_steering_hf.py tests/test_size15_pre_reasoning_action_probe_steering_mlx.py
.venv/bin/ruff check run_size15_pre_reasoning_action_probe_steering_mlx.py run_size15_pre_reasoning_action_probe_steering_hf.py tests/test_size15_pre_reasoning_action_probe_steering_mlx.py
.venv/bin/python -m pytest tests/test_size15_pre_reasoning_action_probe_steering_mlx.py -q
```
