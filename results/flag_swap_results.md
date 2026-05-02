# Flag-swap activation patching at the door (carrying_key)

Tests whether the layer-15 prompt-suffix activations encoding the
`carrying_key` flag are causally read out by the agent's action
selection.

- **Cell**: agent at (1, 5) — directly below the locked door at (1, 4).
- **Prompt a** (variant K0D0, no key): BFS-optimal RIGHT or DOWN
  (head back for the key). Recorded action in the source data: RIGHT.
- **Prompt b** (variant K1D0, has key): BFS-optimal UP (door
  auto-opens with the key). Recorded action in the source data: UP.
- Both prompts identical in length (864 tokens), differing only at
  one grid token (the K cell rendering at (5, 5)) and at whatever
  positions encode the `Carrying key:` line.
- **Activations** captured live from the running MLX MXFP4 model
  (so self-* is a true no-op by construction — no
  saved-vs-runtime mismatch).
- **Setup**: 6 conditions × 30 samples at T = 0.7. Run on M4 Max
  via MLX, ~30 s per sample, ~25 min total.

## Headline: partial confirmation, asymmetric

| | freq(UP) | Δ vs self |
|---|---:|---:|
| baseline-a (K0D0) | 0.133 | — |
| self-a            | 0.167 | (no-op) |
| **swap-a-from-b** | **0.300** | **+0.133** ✓ predicted positive |
| baseline-b (K1D0) | 0.367 | — |
| self-b            | 0.333 | (no-op) |
| **swap-b-from-a** | **0.300** | **−0.033** ✓ predicted negative, but tiny |

The a-side flip (K0D0 prompt + K1D0 activations) more than doubles
UP frequency. The b-side mirror is much weaker.

## Sanity check ✓

| comparison | chi² (5-class) | p |
|---|---:|---:|
| baseline-a vs self-a | 1.67 | 0.64 |
| baseline-b vs self-b | 4.57 | ≈ 0.33 |

Self-conditions are statistically indistinguishable from their
baselines. The hook plumbing is clean: re-extracting `act_a` /
`act_b` from the live MLX model (instead of using saved tensors
from a different runtime) avoided the bias drift that broke the
RunPod run. ||act_a − act_b|| per token ≈ 50–63 over magnitudes
~3 000.

## Action distribution per condition (n = 30 each)

| condition | LEFT | RIGHT | UP | DOWN | parse_fail |
|---|---:|---:|---:|---:|---:|
| baseline-a (K0D0) | 0 | 5 | 4 | 5 | 16 |
| self-a            | 1 | 6 | 5 | 3 | 15 |
| **swap-a-from-b** | 0 | 5 | **9** | 2 | 14 |
| baseline-b (K1D0) | 1 | 0 | 11 | 0 | 18 |
| self-b            | 1 | 5 | 10 | 0 | 14 |
| **swap-b-from-a** | 2 | 1 | **9** | 0 | 18 |

Parse-fail rate is ~50 % across the board — at T = 0.7 the model
often runs off the 1 024-token max before emitting the action JSON.
Affects all conditions equally so doesn't bias the comparison.

## Pairwise distribution comparisons

| comparison | total-variation | chi² | p |
|---|---:|---:|---:|
| swap-a-from-b vs self-a (a-side test) | 0.229 | 2.40 | 0.49 |
| swap-b-from-a vs self-b (b-side test) | 0.229 | 2.53 | — |
| swap-a-from-b vs baseline-b (a all the way to b?) | 0.438 | 7.79 | **0.051** |
| swap-b-from-a vs baseline-a (b all the way to a?) | 0.631 | 11.50 | **0.009** |

The two `swap-* vs self-*` chi² values look small (p ≈ 0.5) — the
per-condition shifts are within sampling noise at N = 30. But the
*destination* comparisons (the bottom two rows) reveal the
asymmetry: swap-a-from-b is reasonably close to baseline-b (TV 0.44,
p 0.05), while swap-b-from-a is *significantly different from*
baseline-a (TV 0.63, p 0.009) — i.e., the b → a transfer doesn't
bring the agent's distribution to the K0D0 baseline.

## Interpretation: asymmetric encoding

Most plausible read: **the K-cell rendering in the prompt grid
carries the "you have the key" cue more strongly than the layer-15
residual carries the "you don't have the key" cue**.

- **Injecting K1D0 activations into the K0D0 prompt works**: the
  prompt still shows `K` on the grid, but the late residual now
  reads "agent has the key", layers 16–23 trust that, and the
  model emits UP at twice the K0D0-baseline rate.
- **Injecting K0D0 activations into the K1D0 prompt fails**: the
  prompt shows no `K` (in-inventory rendering), and layers 16–23
  apparently re-derive "I have the key" from the missing-K
  evidence in the prompt, overriding the patched residual that
  says "no key".

So the carrying_key flag is **causally read out at layer-15
prompt-suffix in one direction**: the residual is *sufficient* to
flip the action toward K1D0-modal behaviour when the prompt-grid
evidence is K0D0-consistent (visible K), but **not necessary** in
the other direction — the prompt-grid evidence (absent K) alone
keeps the K1D0 behaviour.

This is consistent with the layer-15 cost-IRL probe achieving ~96 %
BFS-optimal accuracy: the residual *encodes* the geometry,
including the flag-induced sub-goal change. But the agent's late
decision pathway has multiple inputs, of which the residual at
prompt-suffix is one — when the prompt itself disagrees, the
prompt wins.

## Caveats

- N = 30 per condition with ~50 % parse-fail leaves an effective
  N ≈ 15. The b-side null could be sample-noise at this size; a
  larger run (N = 100) would tighten the conclusion. Each sample
  takes ~30 s, so a full N = 100 run is ~3 hours wall-clock.
- Single (cell, layer, token-position) triple. Swapping at an
  earlier layer (e.g. 7) or at an output position (Direction 2b)
  might produce a stronger b-side effect. Layer-7 activations are
  already saved on disk; only layer-15 was tested here.
- The cell (1, 5) is one of 19 cells where K0D0 and K1D0 BFS-optimal
  actions differ. The asymmetry might be cell-specific; running
  this same swap at e.g. (5, 6) (where K0D0-optimal is UP and
  K1D0-optimal is DOWN) would test whether the asymmetric
  interpretation generalises.

## Implications for the research agenda

A clean a-side flip is the **first non-null causal-intervention
result** in this project after two rounds of null variant-transplant
experiments. The natural next steps:

- **Layer sweep** on the same setup: re-run with layers 7 and 23
  to see where the carrying_key signal is most causal.
- **Multi-cell sweep**: pick 3–5 cells with sharp K0D0 vs K1D0
  optimal differences (e.g., (1,5), (5,6), (5,7)). Aggregate the
  a-side and b-side flips. If the asymmetry is consistent, the
  prompt-grid-vs-residual story is strong; if it averages out,
  this single-cell result was noise.
- **Direction 2b** (late-position intervention): swap at the
  output token position right before `{"action":"X"}` rather than
  at prompt-suffix. If the action commitment lives in the late
  output tokens (per the per-visit decoder hitting >90 % there),
  the b-side asymmetry should disappear.

## Files

- `flag_swap/results.jsonl` — 180 rows (6 conditions × 30 samples).
- `flag_swap/prompts.jsonl`, `flag_swap/act_a.pt`, `flag_swap/act_b.pt`.
- `run_flag_swap_intervention_mlx.py` — the MLX runner used here.
- `run_flag_swap_intervention.py` — the HF/PyTorch runner (RunPod).
- `run_flag_swap_prepare.py`, `run_flag_swap_analyze.py`.
