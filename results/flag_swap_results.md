# Flag-swap activation patching at the door (carrying_key)

Tests whether the layer-15 prompt-suffix activations encoding the
`carrying_key` flag are causally read out by the agent's action
selection, using a clean activation-patching design at a single
agent cell.

This is the **symmetric-K version** (latest). Prompt b's grid
tokens are overridden with prompt a's grid so the K cell is always
visible. The only textual difference between the two prompts is
the `Carrying key: False/True` token in the suffix. See
[`flag_swap_results_asymmetric.md`](flag_swap_results_asymmetric.md)
for the previous run where the K cell rendered differently between
prompts (variable confounded with the explicit flag text).

## Setup

- **Cell**: agent at (1, 5), the cell directly below the locked door at (1, 4).
- **Prompt a** (variant K0D0, no key): suffix `Carrying key: False`.
  Grid renders K visible at (5, 5). BFS-optimal action: RIGHT/DOWN
  (head back for the key).
- **Prompt b** (variant K1D0, has key): suffix `Carrying key: True`.
  Grid identical to prompt a (K visible at (5, 5) — overridden from
  the K0D0 trajectory's grid tokens). BFS-optimal action: UP through
  the door (auto-opens).
- **Diff** between prompt a and prompt b: exactly one token at
  suffix offset 9 — `Ġ False` (id 7983) vs `Ġ True` (id 6432).
  Prompt length 858 tokens (prefix 720 + grid 119 + suffix 19).
- **Activations** captured live from the running MLX MXFP4 model at
  the last 3 prompt-suffix positions (last 3 tokens of the prompt).
  ‖act_a − act_b‖ per token = [1138, 467, 364] over magnitudes ~3000.
- **Six conditions × 30 samples** at T = 0.7. Run on M4 Max via
  MLX, ~24 min total wall-clock. Mode `passthrough` for baselines,
  `replace` (last 3 positions) for self/swap.

## Headline: a-side flip strong, b-side null — asymmetry persists

| | freq(UP) | Δ vs self |
|---|---:|---:|
| baseline-a (K0D0) | 0.167 | — |
| self-a            | 0.233 | (no-op sanity) |
| **swap-a-from-b** | **0.500** | **+0.267** ✓ predicted positive |
| baseline-b (K1D0) | 0.200 | — |
| self-b            | 0.267 | (no-op sanity) |
| **swap-b-from-a** | 0.333 | +0.067 ✗ predicted negative |

The a-side flip is **substantially stronger than in the asymmetric
run** (UP 0.13→0.30 → 0.50, going from doubling to tripling the
K1D0-residual-induced shift). The b-side mirror is still null /
slightly wrong direction.

## Sanity check ✓

| comparison | chi² | p |
|---|---:|---:|
| self-a vs baseline-a | 0.50 | 0.78 |
| self-b vs baseline-b | 0.00 | 0.98 |

Self-conditions are statistically indistinguishable from their
baselines. Hook plumbing clean — re-extracting `act_a` / `act_b`
live from the MLX model means self-* is a no-op by construction.

## Action distribution per condition (n = 30 each)

| condition | LEFT | RIGHT | UP | DOWN | parse_fail |
|---|---:|---:|---:|---:|---:|
| baseline-a (K0D0) | 0 | 8 | 5 | 3 | 14 |
| self-a            | 0 | 8 | 7 | 2 | 13 |
| **swap-a-from-b** | 1 | 3 | **15** | 2 | 9 |
| baseline-b (K1D0) | 0 | 3 | 6 | 0 | 21 |
| self-b            | 0 | 6 | 8 | 0 | 16 |
| **swap-b-from-a** | 0 | 5 | 10 | 0 | 15 |

Parse-fail rate is still ~45–70% across conditions — the model
often runs off the 1024-token budget before emitting the action
JSON. Notably parse-fail *dropped* in `swap-a-from-b` (9/30) — the
intervention seems to make the model commit faster.

## Pairwise comparisons (chi² with zero-column drop)

| comparison | TV distance | chi² | p |
|---|---:|---:|---:|
| swap-a-from-b vs self-a (a-side test)        | 0.350 | 5.83 | 0.120 |
| swap-a-from-b vs baseline-a (a-side test)    | 0.358 | 7.94 | **0.047** ✓ |
| swap-b-from-a vs self-b (b-side test)        | 0.095 | 0.02 | 0.885 (null) |
| swap-b-from-a vs baseline-b (b-side test)    | 0.133 | 0.00 | 1.000 (null) |
| swap-a-from-b vs baseline-b (a→b destination) | 0.190 | 2.45 | 0.485 |
| swap-b-from-a vs baseline-a (b→a destination) | 0.354 | 5.33 | 0.070 |

**a-side**: the K1D0 → K0D0 transplant moves the K0D0 action
distribution significantly away from baseline-a (p = 0.047).
The vs-self comparison is at p = 0.12 — close, not over the bar
at N = 30, but consistent direction with the vs-baseline result.

**b-side**: completely null. swap-b-from-a is statistically
indistinguishable from both baseline-b and self-b (p ≈ 0.88-1.0).
The "no key" residual injected into K1D0 prompt does *not* push the
action distribution back toward the K0D0-modal RIGHT/DOWN pattern.

## Comparison to the asymmetric run

| | asymmetric (K visible only in K0D0) | symmetric (K always visible) |
|---|---:|---:|
| Diff between prompts | grid K-cell + suffix flag (2 places) | suffix flag only (1 place) |
| ‖act_a − act_b‖ per pos | ~50–63 | **1138, 467, 364** |
| freq(UP) baseline-a | 0.133 | 0.167 |
| freq(UP) **swap-a-from-b** | 0.300 | **0.500** |
| Δ vs self-a | +0.133 | **+0.267** |
| freq(UP) baseline-b | 0.367 | 0.200 |
| freq(UP) **swap-b-from-a** | 0.300 | 0.333 |
| Δ vs self-b | −0.033 | +0.067 |
| swap-a-from-b vs self-a chi² p | 0.49 | **0.12** |

Removing the grid-rendering confound made the **a-side intervention
much stronger**: UP frequency at swap-a-from-b is now the modal
action (0.50). The activation-difference magnitude tripled
(50→1138 at the first suffix position) — consistent with the
True/False token being only 7-9 positions before the captured
positions, vs the K-cell being ~17 grid-tokens away in the
asymmetric setup (information has to flow through the suffix).

The b-side null **persists**, ruling out my earlier interpretation
that the K cell rendering was carrying the asymmetry. With grids
identical, something else is making the b-side residual not
load-bearing for the K1D0 → K0D0 direction.

## Interpretation

Best read of the symmetric-prompt result:

- **The carrying_key flag IS causally encoded at layer-15 prompt-suffix**
  in the K0D0 → K1D0 direction. Patching K0D0's residual with K1D0's
  activations triples UP frequency (the K1D0-optimal action) and
  produces a distribution significantly different from K0D0 baseline
  (p = 0.047). 30 samples is small but the effect size is large.

- **The b-side null is not explained by grid rendering** (since grid
  is now identical). Two remaining hypotheses:

  1. **OOD prompt b**: prompt b shows K *and* says "Carrying key:
     True" — internally contradictory. The model has likely never
     seen this combination during training; baseline-b activations
     may encode confusion / "I think I have the key but I see one"
     rather than a clean "I have the key". The K0D0 residual injected
     in then isn't fighting a clean "have key" representation.
     Suggests baseline-b is not a reliable causal anchor here.
     Parse-fail rate at baseline-b (21/30, highest of all conditions)
     supports this — model is unsure.

  2. **Action commitment in late layers**: by layer-15, the
     "Carrying key: True" token may already have been processed by
     attention from later positions, and the key-have decision
     re-derived in layers 16-23 from the explicit text token directly,
     bypassing the layer-15 residual we patched. A swap at the
     output-token positions (Direction 2b) or at a later layer would
     test this.

- **The a-side asymmetry isn't an experimental artifact of the
  K-cell rendering** — it's a real asymmetry between the two
  intervention directions, and the most plausible mechanism is some
  combination of (1) and (2).

## Caveats

- N = 30 per condition with 30-70 % parse-fail leaves effective
  N ≈ 9-21. Larger run (N = 100) would tighten the b-side null —
  if it's a true null, p would converge near 1.0; if just
  underpowered, a small effect may emerge.
- Single (cell, layer, token-position) triple. Layer sweep (7, 23)
  and output-position swap (Direction 2b) are the most informative
  follow-ups. Layer-7 activations already exist on disk for the old
  asymmetric prompts.
- Prompt b is OOD. A future run could use a "natural" K1D0 prompt
  (no K visible in grid) AND symmetric prompt b for comparison —
  three conditions instead of two — to isolate the OOD effect.
- The symmetric override is a hand-crafted edit to the grid
  tokens, not a render generated by reveng's environment. The
  one-token swap (`Ġ_` → `ĠK`) is exactly what would happen if
  the agent dropped the key at (5, 5) without the K1D0 carrying
  flag flipping — a minor inconsistency but unlikely to distort
  layer-15 activations beyond the True/False signal.

## Implications

The symmetric run **strengthens the partial-confirmation finding**
of the asymmetric run. The carrying_key flag at layer-15
prompt-suffix is causally read out for action selection in at
least one direction, with the strongest causal signal we've seen
in this project to date (UP triples to 0.50 modal).

The persistent b-side null is now the more interesting question.
The two competing explanations (OOD prompt b vs late-layer
re-derivation) make different predictions:
- A **layer sweep** (especially layer 23) should distinguish them:
  if late-layer residual carries the b-side signal more strongly,
  the swap at layer 23 will produce a clean b-side flip.
- A **non-OOD prompt b** test — natural K1D0 prompt (no K visible)
  with symmetric override at K0D0 — would test OOD as the cause.

## Files

- Trial results: `flag_swap/results_symmetric.jsonl` (180 rows).
- Asymmetric run results: `flag_swap/results.jsonl`.
- Prompts: `flag_swap/prompts.jsonl`.
- Reference activations: `flag_swap/act_a.pt`, `flag_swap/act_b.pt`
  (these are stale relative to the symmetric prompts; the MLX runner
  re-captures live).
- Scripts: `run_flag_swap_prepare.py`, `run_flag_swap_intervention_mlx.py`,
  `run_flag_swap_analyze.py`.
