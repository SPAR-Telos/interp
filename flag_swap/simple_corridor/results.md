# Flag-swap activation patching at the door (carrying_key)

- Cell: agent at the cell directly below the locked door.
- Prompt **a** (variant K0D0, recorded action would be the K0D0-optimal one — head back for the key).
- Prompt **b** (variant K1D0, recorded action: UP through the door).
- N samples per condition: 30
- Sampling: T = 0.7 (project rule).

## Hook sanity (first sample of each condition)

| condition | fired | seq_len | mode | donor_norm | replacements | first pre→post norm |
|---|---|---|---|---:|---:|---|
| baseline-a | True | 782 | passthrough |  | 0 |  |
| self-a | True | 782 | replace | 5248.0 | 0 |  |
| swap-a-from-b | True | 782 | replace | 5248.0 | 0 |  |
| baseline-b | True | 782 | passthrough |  | 0 |  |
| self-b | True | 782 | replace | 5248.0 | 0 |  |
| swap-b-from-a | True | 782 | replace | 5248.0 | 0 |  |

## Action distribution per condition

| condition | LEFT | RIGHT | UP | DOWN | parse_fail | n |
|---|---:|---:|---:|---:|---:|---:|
| baseline-a | 0 (0.00) | 0 (0.00) | 1 (0.03) | 29 (0.97) | 0 | 30 |
| self-a | 0 (0.00) | 0 (0.00) | 0 (0.00) | 30 (1.00) | 0 | 30 |
| swap-a-from-b | 0 (0.00) | 0 (0.00) | 1 (0.03) | 29 (0.97) | 0 | 30 |
| baseline-b | 0 (0.00) | 0 (0.00) | 29 (0.97) | 1 (0.03) | 0 | 30 |
| self-b | 1 (0.03) | 0 (0.00) | 29 (0.97) | 0 (0.00) | 0 | 30 |
| swap-b-from-a | 0 (0.00) | 0 (0.00) | 30 (1.00) | 0 (0.00) | 0 | 30 |

## Pairwise distribution comparisons

| comparison | TV distance | chi² | p (4-class, 3 dof) |
|---|---:|---:|---:|
| self-a vs baseline-a (sanity, must NOT differ) | 0.033 | 1.02 | (scipy missing) |
| self-b vs baseline-b (sanity, must NOT differ) | 0.033 | 2.00 | (scipy missing) |
| swap-a-from-b vs self-a (HYPOTHESIS, should differ) | 0.033 | 1.02 | (scipy missing) |
| swap-b-from-a vs self-b (HYPOTHESIS, should differ) | 0.033 | 1.02 | (scipy missing) |
| swap-a-from-b vs baseline-b (does a→b move all the way to b?) | 0.933 | 52.27 | (scipy missing) |
| swap-b-from-a vs baseline-a (does b→a move all the way to a?) | 0.967 | 56.13 | (scipy missing) |

## Direction-of-shift on UP frequency

If the variant flag is causally encoded, swap-a-from-b should INCREASE UP frequency (toward the K1D0-optimal direction), and swap-b-from-a should DECREASE UP frequency.

| condition | freq(UP) |
|---|---:|
| baseline-a | 0.033 |
| self-a | 0.000 |
| swap-a-from-b | 0.033 |
| baseline-b | 0.967 |
| self-b | 0.967 |
| swap-b-from-a | 1.000 |

- Δ freq(UP) on a-side (swap − self): **+0.033** (expected positive under hypothesis).
- Δ freq(UP) on b-side (swap − self): **+0.033** (expected negative under hypothesis).

## Interpretation

Read three things:

1. **Hook sanity**: every condition's `fired=True`; baseline rows should run in passthrough mode and self/swap rows should run in replace mode with a donor tensor. If not, the hook plumbing is broken and the rest is invalid.
2. **Self-controls indistinguishable from baseline**: the two `self-* vs baseline-*` chi² tests should give large p-values. If a self-* differs from its baseline, the hook itself is biasing the model and we can't trust the swap rows.
3. **Swaps move the action distribution toward the donor variant's**: Δ freq(UP) > 0 on the a-side (swap-a-from-b raises UP toward K1D0's bias) and Δ freq(UP) < 0 on the b-side (swap-b-from-a drops UP toward K0D0's bias). Both deltas in the predicted direction with chi² p < 0.05 against the same-variant control is the causal-encoding result.

## Files

- Trial results: `flag_swap/simple_corridor/results.jsonl`
- Prompts: `flag_swap/prompts.jsonl`
- Activations: `flag_swap/act_a.pt`, `flag_swap/act_b.pt`
- Scripts: `run_flag_swap_prepare.py`, `run_flag_swap_intervention.py`, `run_flag_swap_analyze.py`
