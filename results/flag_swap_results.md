# Flag-swap activation patching at the door (carrying_key)

**Headline finding (latest, corrected-prompts run):** with internally
consistent prompts, the layer-15 prompt-suffix residual swap
**does not flip action selection**. The "a-side flip" reported in
prior runs (asymmetric grid, symmetric grid) was an artifact of
prompt-OOD confusion, not a clean causal demonstration of the
carrying_key flag being read out by action selection.

> **Bottom line on the research question** ("is the pre-reasoning
> belief causally responsible for the action, or just an
> epiphenomenal correlate of geometry?"): under the cleanest
> intervention setup (single-token prompt difference, internally
> consistent semantics, full-vector swap at the captured
> positions), the answer leans toward **epiphenomenal correlate**.
> The carrying_key flag IS encoded at layer 15 (probe accuracy
> ~96%) but the encoding is not load-bearing for action selection.

## What we ran (three iterations)

| Run | Grid | Suffix | Prefix | Prompt-pair diff |
|---|---|---|---|---|
| **asymmetric** | K visible only in K0D0 | unrendered template `{{carrying_key}}` (BUG: literal placeholder) | original | 1 grid token (K cell only) — flag text not in prompt |
| **symmetric** | always K | step-rendered `True`/`False` | original | 1 suffix token (`True`/`False`) but prompt b is OOD: K visible + "Carrying key: True" contradicts the prefix rules |
| **corrected** *(this run)* | always K | step-rendered `True`/`False` | rewritten so K-stays-visible-while-carrying is consistent with the rules | 1 suffix token (`True`/`False`); prompt b no longer contradicts itself |

Prefix rewrites in the corrected run:
- `Once picked up, the key disappears from the grid and the agent carries it.` → `Once picked up, the key still appears on the grid and the agent carries it.`
- `If you do not see a K on the grid, it means the key has already been picked up.` → `The K symbol stays on the grid regardless of whether you are carrying the key.`

## Setup (corrected run)

- **Cell**: agent at (1, 5), directly below the locked door at (1, 4).
- **Prompt a** (K0D0, no key): suffix `Carrying key: False`. Grid shows K at (5, 5).
- **Prompt b** (K1D0, has key): suffix `Carrying key: True`. Grid shows K at (5, 5).
- **Diff**: exactly one token at suffix offset 9 (`Ġ False` 7983 vs `Ġ True` 6432).
- **Prompt length**: 854 tokens (prefix 716 + grid 119 + suffix 19).
- **Activations**: captured live at the last 3 prompt-suffix positions (positions -3, -2, -1) from `mlx-community/gpt-oss-20b-MXFP4-Q4`. ‖act_a − act_b‖ per position = [1069, 462, 233] over magnitudes ~3000.
- **Six conditions × 30 samples** at T = 0.7. ~24 min on M4 Max.

## Action distribution per condition (corrected run, n = 30 each)

| condition | LEFT | RIGHT | UP | DOWN | parse_fail | freq(UP) |
|---|---:|---:|---:|---:|---:|---:|
| baseline-a (K0D0) | 1 | 7 | 5 | 2 | 15 | 0.167 |
| self-a            | 1 | 3 | 6 | 3 | 17 | 0.200 |
| **swap-a-from-b** | 3 | 9 | **4** | 2 | 12 | **0.133** ↓ |
| baseline-b (K1D0) | 0 | 5 | 10 | 0 | 15 | 0.333 |
| self-b            | 0 | 5 | 12 | 0 | 13 | 0.400 |
| **swap-b-from-a** | 0 | 9 | **10** | 0 | 11 | **0.333** = |

## Pairwise chi² (corrected run)

| comparison | chi² | p |
|---|---:|---:|
| self-a vs baseline-a (sanity) | 1.76 | 0.62 ✓ |
| self-b vs baseline-b (sanity) | 0.00 | 1.00 ✓ |
| swap-a-from-b vs self-a (a-test) | 3.89 | 0.27 (null) |
| swap-a-from-b vs baseline-a (a-test) | 1.10 | 0.78 (null) |
| swap-b-from-a vs self-b (b-test) | 0.58 | 0.45 (null) |
| swap-b-from-a vs baseline-b (b-test) | 0.23 | 0.64 (null) |
| swap-a-from-b vs baseline-b (a→b destination) | 8.51 | 0.0365 (significantly *not* like baseline-b) |
| swap-b-from-a vs baseline-a (b→a destination) | 4.51 | 0.21 |

The two swap conditions are **null** against their own variant's
controls and **far from** the other variant's baseline. Patching
the residual neither flips toward the donor variant nor stays put;
it just adds noise (RIGHT goes up in both conditions: +0.07 a-side,
+0.13 b-side).

## Cross-run comparison (UP frequency)

| | asymmetric | symmetric | **corrected** |
|---|---:|---:|---:|
| baseline-a | 0.133 | 0.167 | 0.167 |
| swap-a-from-b | 0.300 | **0.500** | **0.133** ↓ |
| Δ vs self-a | +0.133 | +0.267 | **−0.067** |
| chi² p (vs self-a) | 0.49 | 0.12 | 0.27 |
| baseline-b | 0.367 | 0.200 | 0.333 |
| swap-b-from-a | 0.300 | 0.333 | 0.333 |
| Δ vs self-b | −0.033 | +0.067 | −0.067 |

The "a-side flip" tracks **prompt-b OOD-ness**, not the
carrying_key flag itself:
- asymmetric: prompt b had K-removed-from-grid (semi-OOD) → mild flip
- symmetric: prompt b had K-visible + Carrying key: True (max OOD, contradicts rules) → strong flip
- corrected: prompt b is internally consistent → null

This is the **strongest evidence to date** that the previous
"causal intervention result" was OOD-driven artifact: as we made
prompt b less contradictory, the apparent causal effect shrank
and then disappeared.

## Interpretation

Three competing explanations remain on the table:

1. **Carrying_key encoding at layer-15 prompt-suffix is epiphenomenal.**
   The probe direction sees the flag (96% accuracy) but the action
   pathway re-derives the flag in late layers (16-23) directly from
   the explicit `True/False` text token via attention, bypassing the
   layer-15 residual representation. Patching the residual stream
   has no effect because the pathway doesn't read from it.

2. **Wrong patching positions.** We swap the *last 3* suffix
   positions (`Ġ False/True`/`<|end|>`/`<|start|>assistant`-ish).
   The True/False token is at suffix offset 9, position -10 from
   the end. Layer-15 attention at the captured positions has had
   to summarize a lot of intervening tokens — by then the
   representation may have shifted away from "carrying_key=X" toward
   "I'm about to commit to an action". A swap at the True/False
   token position itself (or all 19 suffix positions) might still
   show an effect.

3. **Wrong layer.** Causal influence might live at layer 7 or
   layer 23 rather than 15. Layer-15 was chosen because the
   cost-IRL probe peaks there for BFS optimality, but the
   carrying_key flag's *causal* layer might be different.

(2) and (3) are testable; (1) is the null result we'd expect if
neither (2) nor (3) rescues the finding.

## On the experimental design ("is full-vector swap too aggressive?")

The user raised this directly: full-vector swap at 3 positions
overwrites 8640 floats — it conflates many things the residual
encodes (geometry, plan-stage, distance-to-goal, carrying_key, ...).

**With the corrected run showing null even from the full swap**,
this concern shifts from "can we cleanly attribute the effect?" to
"is there any effect to attribute?" The full swap is the *upper
bound* on the causal magnitude across any sub-direction; if it's
null, no narrower probe-direction-only intervention can recover a
causal effect at this (layer, position) pair.

That said, narrower interventions remain valuable for two reasons:
- **Diagnostic of (2)**: a direction-only swap at *all 19 suffix
  positions* tests whether the carrying_key direction has causal
  effect *somewhere* in the suffix block, not just at the last 3
  positions.
- **Diagnostic of (3)**: a direction-only swap across multiple
  layers (7, 11, 15, 19, 23) localizes which layer (if any) carries
  the causal effect.

## Caveats

- N = 30 per condition with 40-60% parse-fail leaves effective
  N ≈ 12-18. We cannot rule out a small causal effect; the
  conclusion is "no large effect detected".
- Prompts a and b are at one specific cell (1, 5). The asymmetry
  between K0D0-optimal and K1D0-optimal actions is sharpest there,
  but a multi-cell sweep would test generalization.
- The corrected prefix is a hand-crafted edit — the model has
  never seen "K stays visible while carrying" during pre-training.
  The model may still treat this as somewhat OOD, just less so
  than the symmetric run.

## Next steps (if pursuing)

In rough order of expected return:

1. **Layer sweep**: re-run the corrected-prompts experiment at
   layers 7, 11, 19, 23 to see if causal effect is at a different
   depth. Layer activations already gathered for layers 7 and 15
   under the asymmetric prompts; would need re-extraction for
   corrected prompts.
2. **Direction-only patch at all 19 suffix positions** using the
   trained `KeyCollectedProbe` weight vector. Tests whether the
   carrying_key direction is causal anywhere in the suffix.
3. **Patch at the True/False token position only** (suffix offset
   9). Single-position, both full-vector and direction-only.
4. **Multi-cell sweep** with the corrected prompts at 3-5 cells
   with sharp K0D0/K1D0 optimal-action differences.

## Files

- `flag_swap/results_corrected.jsonl` — 180 rows, this run.
- `flag_swap/results_symmetric.jsonl` — symmetric (uncorrected) run.
- `flag_swap/results.jsonl` — original asymmetric run.
- `flag_swap/prompts.jsonl` — currently the corrected prompts.
- `run_flag_swap_prepare.py` — `--rewrite-prefix` enables the
  corrected setup.
- `run_flag_swap_intervention_mlx.py` — MLX runner.
- `run_flag_swap_analyze.py` — analyzer.
- Archived reports:
  - `results/flag_swap_results_asymmetric.md`
  - `results/flag_swap_results_symmetric.md` (regenerated by analyzer)
  - `results/flag_swap_results_symmetric_archived.md` (the previous
    write-up that interpreted the symmetric run as a positive result).
