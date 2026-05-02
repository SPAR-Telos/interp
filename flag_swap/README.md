# Flag-swap activation patching (Direction 2c, carrying_key)

Clean activation-patching test of whether the carrying_key flag is
causally encoded at layer-15 prompt-suffix.

- **Cell**: agent at (1, 5) — directly below the locked door at (1, 4).
- **Prompt a** (K0D0): no key, door closed. BFS-optimal: RIGHT/DOWN
  (head back for the key).
- **Prompt b** (K1D0): has key, door closed. BFS-optimal: UP (door
  auto-opens with key).
- Both prompts are real recorded fourroom step records (step 0 of
  K0D0 traj055 and K1D0 traj056). Identical length (864 tokens),
  identical suffix; differ only at one grid token (the K cell at
  (5, 5)) and at whatever positions the "Carrying key" line lives in.
- **Activations**: real layer-15 prompt-suffix activations from the
  recorded forward passes. Shape (3, 2880) bfloat16, ||a − b|| ≈ 55
  per token vs magnitude ~3000 — small but non-zero.

## Six conditions (each at T=0.7 with N samples)

1. baseline-a       prompt a, no hook
2. baseline-b       prompt b, no hook
3. self-a           prompt a, hook replaces with act_a (no-op sanity)
4. self-b           prompt b, hook replaces with act_b (no-op sanity)
5. swap-a-from-b    prompt a, hook replaces with act_b (hypothesis)
6. swap-b-from-a    prompt b, hook replaces with act_a (mirror)

## Pipeline

```bash
# Mac CPU — extract prompts + activations from existing fourroom data:
uv run python run_flag_swap_prepare.py
# Smoke (Mac CPU) — verify hook plumbing on a small model:
uv run python run_flag_swap_intervention.py --smoke --smoke-model gpt2 --layer 5 --n-samples 3
# RunPod GPU — real run:
uv run python run_flag_swap_intervention.py
# Mac CPU — analyze:
uv run python run_flag_swap_analyze.py
```

## Files in this directory

- `prompts.jsonl`  — two rows (a, b) with full prompt token IDs and
  the absolute positions of the last 3 prompt-suffix tokens.
- `act_a.pt` / `act_b.pt` — (3, 2880) bfloat16 layer-15 activations.
  Tiny; committed so RunPod doesn't need access to data/activations.
- `results.jsonl` — written by the intervention script, 6 × N rows.
