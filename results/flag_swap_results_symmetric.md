# Flag-swap activation patching at the door (carrying_key)

- Cell: agent at the cell directly below the locked door.
- Prompt **a** (variant K0D0, recorded action would be the K0D0-optimal one — head back for the key).
- Prompt **b** (variant K1D0, recorded action: UP through the door).
- N samples per condition: 30
- Sampling: T = 0.7 (project rule).

## Hook sanity (first sample of each condition)

| condition | fired | seq_len | batch | replacements | first pre→post norm |
|---|---|---|---|---|---|
| baseline-a | True | 857 | None | 0 |  |
| self-a | True | 857 | None | 0 |  |
| swap-a-from-b | True | 857 | None | 0 |  |
| baseline-b | True | 857 | None | 0 |  |
| self-b | True | 857 | None | 0 |  |
| swap-b-from-a | True | 857 | None | 0 |  |

## Action distribution per condition

| condition | LEFT | RIGHT | UP | DOWN | parse_fail | n |
|---|---:|---:|---:|---:|---:|---:|
| baseline-a | 0 (0.00) | 8 (0.27) | 5 (0.17) | 3 (0.10) | 14 | 30 |
| self-a | 0 (0.00) | 8 (0.27) | 7 (0.23) | 2 (0.07) | 13 | 30 |
| swap-a-from-b | 1 (0.03) | 3 (0.10) | 15 (0.50) | 2 (0.07) | 9 | 30 |
| baseline-b | 0 (0.00) | 3 (0.10) | 6 (0.20) | 0 (0.00) | 21 | 30 |
| self-b | 0 (0.00) | 6 (0.20) | 8 (0.27) | 0 (0.00) | 16 | 30 |
| swap-b-from-a | 0 (0.00) | 5 (0.17) | 10 (0.33) | 0 (0.00) | 15 | 30 |

## Pairwise distribution comparisons

| comparison | TV distance | chi² | p (4-class, 3 dof) |
|---|---:|---:|---:|
| self-a vs baseline-a (sanity, must NOT differ) | 0.099 | 0.50 | (scipy missing) |
| self-b vs baseline-b (sanity, must NOT differ) | 0.095 | 0.21 | (scipy missing) |
| swap-a-from-b vs self-a (HYPOTHESIS, should differ) | 0.350 | 5.83 | 0.120 |
| swap-b-from-a vs self-b (HYPOTHESIS, should differ) | 0.095 | 0.28 | (scipy missing) |
| swap-a-from-b vs baseline-b (does a→b move all the way to b?) | 0.190 | 2.45 | 0.485 |
| swap-b-from-a vs baseline-a (does b→a move all the way to a?) | 0.354 | 5.33 | (scipy missing) |

## Direction-of-shift on UP frequency

If the variant flag is causally encoded, swap-a-from-b should INCREASE UP frequency (toward the K1D0-optimal direction), and swap-b-from-a should DECREASE UP frequency.

| condition | freq(UP) |
|---|---:|
| baseline-a | 0.167 |
| self-a | 0.233 |
| swap-a-from-b | 0.500 |
| baseline-b | 0.200 |
| self-b | 0.267 |
| swap-b-from-a | 0.333 |

- Δ freq(UP) on a-side (swap − self): **+0.267** (expected positive under hypothesis).
- Δ freq(UP) on b-side (swap − self): **+0.067** (expected negative under hypothesis).

## Interpretation

Read three things:

1. **Hook sanity**: every condition's `fired=True`; baseline rows have 0 replacements and self/swap rows have 3. If not, the hook plumbing is broken and the rest is invalid.
2. **Self-controls indistinguishable from baseline**: the two `self-* vs baseline-*` chi² tests should give large p-values. If a self-* differs from its baseline, the hook itself is biasing the model and we can't trust the swap rows.
3. **Swaps move the action distribution toward the donor variant's**: Δ freq(UP) > 0 on the a-side (swap-a-from-b raises UP toward K1D0's bias) and Δ freq(UP) < 0 on the b-side (swap-b-from-a drops UP toward K0D0's bias). Both deltas in the predicted direction with chi² p < 0.05 against the same-variant control is the causal-encoding result.

## Files

- Trial results: `flag_swap/results_symmetric.jsonl`
- Prompts: `flag_swap/prompts.jsonl`
- Activations: `flag_swap/act_a.pt`, `flag_swap/act_b.pt`
- Scripts: `run_flag_swap_prepare.py`, `run_flag_swap_intervention.py`, `run_flag_swap_analyze.py`
