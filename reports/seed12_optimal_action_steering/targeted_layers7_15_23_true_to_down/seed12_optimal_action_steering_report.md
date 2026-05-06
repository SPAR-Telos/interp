# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_targeted_layers7_15_23_true_to_down.jsonl`
- Layers: 7, 15, 23
- Hooked alphas: -64, -8, 0, 8, 64

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 7 | no_hook |  | 10 | 0.800 | 0.200 | 0.800 | 0.000 | 0.000 |
| 7 | hooked | -64 | 10 | 0.700 | 0.100 | 0.700 | -0.100 | -0.100 |
| 7 | hooked | -8 | 10 | 0.600 | 0.200 | 0.600 | -0.200 | 0.000 |
| 7 | hooked | 0 | 10 | 0.800 | 0.200 | 0.800 | 0.000 | 0.000 |
| 7 | hooked | 8 | 10 | 0.400 | 0.400 | 0.400 | -0.400 | 0.200 |
| 7 | hooked | 64 | 10 | 0.500 | 0.300 | 0.500 | -0.300 | 0.100 |
| 15 | no_hook |  | 10 | 0.600 | 0.400 | 0.600 | 0.000 | 0.000 |
| 15 | hooked | -64 | 10 | 0.200 | 0.500 | 0.200 | -0.400 | 0.100 |
| 15 | hooked | -8 | 10 | 0.700 | 0.200 | 0.700 | 0.100 | -0.200 |
| 15 | hooked | 0 | 10 | 0.600 | 0.400 | 0.600 | 0.000 | 0.000 |
| 15 | hooked | 8 | 10 | 0.500 | 0.300 | 0.500 | -0.100 | -0.100 |
| 15 | hooked | 64 | 10 | 0.400 | 0.400 | 0.400 | -0.200 | 0.000 |
| 23 | no_hook |  | 10 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| 23 | hooked | -64 | 10 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| 23 | hooked | -8 | 10 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| 23 | hooked | 0 | 10 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| 23 | hooked | 8 | 10 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| 23 | hooked | 64 | 10 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 7 | 10 | 1.000 | 1.000 | 1.000 | 0 |
| 15 | 10 | 1.000 | 1.000 | 1.000 | 0 |
| 23 | 10 | 1.000 | 1.000 | 1.000 | 0 |

## Paired Causal Effects

| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | optimal-to-nonoptimal rate | true-to-not-true rate |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | -64 | 10 | 0.500 | 0.200 | 0.300 | 0.200 |
| 7 | -8 | 10 | 0.400 | 0.100 | 0.300 | 0.100 |
| 7 | 0 | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| 7 | 8 | 10 | 0.600 | 0.100 | 0.500 | 0.100 |
| 7 | 64 | 10 | 0.700 | 0.200 | 0.500 | 0.200 |
| 15 | -64 | 10 | 0.400 | 0.000 | 0.400 | 0.000 |
| 15 | -8 | 10 | 0.300 | 0.200 | 0.100 | 0.200 |
| 15 | 0 | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| 15 | 8 | 10 | 0.300 | 0.100 | 0.200 | 0.100 |
| 15 | 64 | 10 | 0.600 | 0.200 | 0.400 | 0.200 |
| 23 | -64 | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| 23 | -8 | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| 23 | 0 | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| 23 | 8 | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| 23 | 64 | 10 | 0.000 | 0.000 | 0.000 | 0.000 |

## Interpretation

The largest optimal-action improvement versus no-hook is layer 15 at alpha -8: optimal-action rate 0.700, delta 0.100.

The largest true-action retention drop versus no-hook is layer 15 at alpha -8: true-action rate 0.200, delta -0.200.

The strongest paired causal effect on emitted action is layer 7 at alpha 64: 0.700 of paired samples changed final action under the same generation seed.

The strongest paired effect away from the recorded true action is layer 7 at alpha -64: 0.200 of pairs changed from true-action emission to a different final action.

The strongest paired nonoptimal-to-optimal conversion is layer 7 at alpha -64: 0.200 of pairs converted from a nonoptimal emitted action to an action in `astar_actions`.

These are distinct behavioral claims. A higher optimal-action rate means the model more often emits an action in `astar_actions`. A lower true-action rate means the model moved away from the recorded `agent_action`. Those can move in opposite directions because many recorded true actions are not in the optimal set.

## Artifacts

- `summary_by_layer_alpha.csv`
- `noop_baseline_agreement.csv`
- `paired_effects_vs_no_hook.csv`
- `optimal_action_rate_vs_alpha.svg`
- `true_action_rate_vs_alpha.svg`
- `nonoptimal_to_optimal_rate_vs_alpha.svg`
