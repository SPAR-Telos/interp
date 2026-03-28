# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project evaluating **goal-directedness in language model agents** (GPT-OSS-20B navigating 2D grid environments) through two lenses:
1. **Behavioral** — action optimality, robustness to perturbations
2. **Representational** — probing internal model activations for encoded spatial knowledge

Paper: arxiv.org/abs/2602.08964

## Setup

```bash
uv sync                   # create venv + install all deps
uvx pre-commit install    # install pre-commit hooks
```

Requires Python 3.10+ (3.12 recommended). Use `uv` (not pip) for dependency management.

## Commands

```bash
make test           # run pytest -n auto -vv
make check-style    # ruff format + lint check (no fixing)
make fix-style      # ruff format + lint with auto-fix
make clean          # remove __pycache__, build artifacts
```

Run a single test file:
```bash
uv run pytest tests/test_grid_utils.py -vv
```

Skip slow or GPU tests:
```bash
uv run pytest -m "not slow and not require_cuda_gpu"
```

## CLI

The package installs as `interp-cli` (entrypoint: `telos_interp.commands.cli:main`, built with Tyro):

```bash
interp-cli --help
interp-cli gather-activations --help
```

## Code Style

- Line length: 119 characters
- Python 3.10+ syntax
- Google-style docstrings
- Ruff for formatting and linting (config in `pyproject.toml`)

## Architecture

### Processing Pipeline

```
Trajectory JSONs (agent interactions in grid envs)
    ↓ gather_activations        — extract layer activations via nnsight
    ↓ prepare_activations       — reshape/format tensors for probe training
    ↓ train_*_probe             — train classifiers/regressors on activations
    ↓ eval_*_probe              — evaluate probe accuracy
    ↓ apply_cognitive_map_probe — annotate trajectories with predictions
Annotated JSONs + metrics
```

### Key Abstractions

- **Trajectories** — JSON records of agent steps in grid environments
- **Activations** — `.pt` tensor files organized by layer, loaded via `activation_loading.py`
- **Cognitive map probes** — classifiers predicting grid cell identity (A=agent, G=goal, #=wall, _=empty, D=door, K=key, ?=unknown, +=visited) from activations
- **Distance probes** — regressors predicting A* distance-to-goal from activations
- **Probe models** — `LogisticRegressionProbe` and `MLPProbe` in `probe_models.py`

### Package Layout

- `telos_interp/` — main package
  - `commands/` — one subdirectory per CLI command, each with `*_fn.py` (implementation), optional `*_utils.py`, and its own `README.md`
  - `activation_loading.py` — discover and load activation files from folder structures
  - `grid_utils.py` — grid state parsing and cell identity mappings
  - `probe_models.py` — probe model architectures
  - `training.py` — shared training utilities (train_epoch, normalization, device/seed setup)
- `configs/` — TOML config templates for probe training runs
- `evaluation_scripts/` — standalone scripts for computing metrics
- `plotting_scripts/` — visualization of probe evaluation results
- `spaces/` — HuggingFace trace-viewer (git submodule)

### Dependencies

- `nnsight` — activation extraction from transformer internals
- `nnterp` — higher-level interpretability utilities
- `tyro` — CLI framework (dataclass-based argument parsing)
- `transformers` + `torch` — model loading and inference

## IRL & Behavioral Analysis Scripts (root-level)

Standalone scripts for inferring reward functions and behavioral strategies from agent trajectories.

### `run_maxent_irl.py` — One-Step MaxEnt IRL

Treats each (s_t, a_t) transition independently (no value iteration). Works across grids of different sizes/layouts since it doesn't require a fixed MDP.

- Feature vector phi(s) in R^7: `[dist_goal, dist_key, dist_door, has_key, door_open, p_key, p_door]`
- BFS distances normalized by grid_size^2 so values are in [0, 1] regardless of grid size
- Probe features (p_key, p_door) from trained `KeyCollectedProbe` models; falls back to binary proxy when unavailable
- Data: `data/trajectories/trajectories_key_door_100/` (95 different grids, 1 trajectory each) + `data/trajectories/trajectories_train_single_step/` (6000 single-step samples)

```bash
uv run python run_maxent_irl.py --eval
```

### `run_fixed_grid_maxent_irl.py` — n-Step MaxEnt IRL (Fixed Grid)

Full MaxEnt IRL with backward soft value iteration + forward state visitation. Requires all trajectories on the **same fixed grid**.

- State space: `(row, col, has_key, door_open)` — ~117 reachable states for 9x9 grid
- Backward pass: Q(s,a) = R(s) + V(T(s,a)); V(s) = logsumexp_a Q(s,a)
- Forward pass: propagate state visitation from empirical start distribution
- Supports `--token-category {pre,post}` to choose pre-reasoning (prompt suffix) or post-reasoning (output) activations for probe features
- Data: `data/trajectories/fixed_key_door_grid/` (100 trajectories, same grid, varied starting positions)
- Activations: `data/activations/activations_fixed_key_door_grid/` (must be gathered first)

```bash
# Step 1: gather activations (requires GPU + HF login)
uv run huggingface-cli login
uv run interp-cli gather_activations \
    --trajectory-paths "data/trajectories/fixed_key_door_grid/*.json" \
    --output-dir data/activations/activations_fixed_key_door_grid \
    --layers all --steps all --prompt-suffix-indices -1 --output-indices -1

# Step 2: run IRL
uv run python run_fixed_grid_maxent_irl.py --n 50 --epochs 50 --lr 0.1 --token-category post
```

### `run_action_probe.py` — Behavioral Cloning from Activations (Strategy 2)

Trains action-prediction probes (LR + MLP) on agent activations. Compares pre-reasoning vs post-reasoning activations.

```bash
uv run python run_action_probe.py
```

### `run_key_probe.py` / `run_door_probe.py` — Binary State Probes

Train `KeyCollectedProbe` classifiers for `carrying_key` and `door_open` from activations. Outputs used as p_key/p_door features in the IRL scripts.

### Key Design Decisions

- **BFS normalization**: All BFS distances divided by `grid_size^2`, unreachable sentinel = 1.0. This makes distances comparable across grids of different sizes.
- **One-step vs n-step IRL**: One-step works across diverse grids but misses long-horizon credit (e.g., key pickup value). n-step requires a fixed grid but properly propagates credit through value iteration.
- **Pre vs post reasoning**: Pre-reasoning = last token of prompt suffix (before model output). Post-reasoning = last output token (after model commits to action). Post-reasoning activations are generally more informative for action prediction.
- **Probe features for unvisited states**: In n-step IRL, the backward/forward passes need phi(s) for all states. States not visited in any demonstration fall back to binary has_key/door_open as proxy for p_key/p_door.

## CLI Note

The CLI entrypoint requires `uv run` prefix when not installed globally:

```bash
uv run interp-cli gather_activations --help   # underscore in subcommand name
```
