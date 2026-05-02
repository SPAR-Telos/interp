# Variant-transplant intervention (Direction 2a)

Tests whether the layer-15 prompt-suffix activations encoding the
env-variant `(carrying_key, door_open)` are causally read out by
the agent's action selection.

## Files in this directory

- `donors.pt` — dict keyed by `(col, row, variant_str)` (and
  `(col, row, "self::traj::step")` for per-target self-transplant
  donors). Each value is a `(3, 2880)` bfloat16 tensor: the mean
  layer-15 activation across donor visits at the cell, for the 3
  prompt-suffix tokens we intervene on. **Gitignored** (binary,
  regenerable).
- `targets.jsonl` — one line per (target_visit, condition_pair).
  Each row carries the full prompt token IDs, the absolute
  positions of the 3 prompt-suffix tokens, the target's variant /
  agent action, and the donor lookup keys for self / cross
  conditions.
- `results.jsonl` — one line per (target, condition). Output of
  `run_transplant_intervention.py`.

## Pipeline

1. **Mac CPU** — build targets and donors:
   ```bash
   uv run python run_transplant_prepare_targets.py
   ```
   Default: target=K0D0, donor=K1D1. ~46 targets across 14 cells
   on `fourroom_episode_sweep_T0`.

2. **CPU smoke test** — verify the hook mechanism on a small model:
   ```bash
   uv run python run_transplant_intervention.py --smoke --smoke-model gpt2 --layer 5 --max-targets 3
   ```
   Uses synthetic safe-vocab input + random donors of the small
   model's hidden_dim. Verifies hook fires at the right positions
   and replacement actually changes activations (pre/post-norm
   diagnostic in `_smoke_log`).

3. **RunPod GPU** — full run on GPT-OSS-20B:
   ```bash
   huggingface-cli login
   uv run python run_transplant_intervention.py
   ```
   Defaults: `openai/gpt-oss-20b`, layer 15, bfloat16, device_map=auto.
   ~46 targets × 3 conditions = 138 forward passes; budget ~30 min
   on a single A100/H100.

4. **Mac CPU** — analyze and write the report:
   ```bash
   uv run python run_transplant_analyze.py
   ```
   Writes `results/transplant_intervention_results.md`.

## Conditions

For each target visit, three forward passes:

- **baseline**: no intervention. Establishes "what does the
  un-modified K0D0 agent emit at this cell?".
- **self**: replace layer-15 prompt-suffix activations with the
  mean of *same-variant* (K0D0) visits at the same cell, excluding
  the target visit itself. Tests whether the averaging operation
  alone causes drift.
- **cross**: replace with the mean of *other-variant* (K1D1) donor
  visits at the same cell. The hypothesis test: does the agent
  now act like a K1D1 agent?

If cross-transplant flips the action toward the K1D1-optimal set
significantly more than self-transplant does, the variant is
causally read out.

## Target selection

Cells are kept iff:
- visited by ≥ 2 K0D0 trajectories (so self-transplant has a
  donor),
- visited by ≥ 1 K1D1 trajectory (so cross-transplant has a donor),
- K0D0-optimal action set ≠ K1D1-optimal action set (otherwise
  the transplant test is uninformative — both variants want the
  same move).

On `fourroom_episode_sweep_T0` this gives 14 cells in the lower
room. Up to `--max-targets-per-cell` (default 4) K0D0 visits per
cell are included.
