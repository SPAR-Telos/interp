# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_smoke_highalpha.jsonl`
- Layers: 23
- Hooked alphas: -8, 0, 8

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 23 | no_hook |  | 6 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| 23 | hooked | -8 | 6 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| 23 | hooked | 0 | 6 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| 23 | hooked | 8 | 6 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 23 | 6 | 1.000 | 1.000 | 1.000 | 0 |

## Interpretation

The largest optimal-action improvement versus no-hook is layer 23 at alpha -8: optimal-action rate 1.000, delta 0.000.

The largest true-action retention drop versus no-hook is layer 23 at alpha -8: true-action rate 0.000, delta 0.000.

These are distinct behavioral claims. A higher optimal-action rate means the model more often emits an action in `astar_actions`. A lower true-action rate means the model moved away from the recorded `agent_action`. Those can move in opposite directions because many recorded true actions are not in the optimal set.

## Artifacts

- `summary_by_layer_alpha.csv`
- `noop_baseline_agreement.csv`
- `optimal_action_rate_vs_alpha.svg`
- `true_action_rate_vs_alpha.svg`
- `nonoptimal_to_optimal_rate_vs_alpha.svg`
