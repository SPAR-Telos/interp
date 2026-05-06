# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_direction_control_layer7_true_to_down.jsonl`
- Layers: 7
- Hooked alphas: -64, -8, 8, 64

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 7 | no_hook |  | 10 | 0.800 | 0.200 | 0.800 | 0.000 | 0.000 |
| 7 | hooked | -64 | 40 | 0.575 | 0.250 | 0.575 | -0.225 | 0.050 |
| 7 | hooked | -8 | 40 | 0.700 | 0.200 | 0.700 | -0.100 | 0.000 |
| 7 | hooked | 8 | 40 | 0.525 | 0.350 | 0.525 | -0.275 | 0.150 |
| 7 | hooked | 64 | 40 | 0.650 | 0.275 | 0.650 | -0.150 | 0.075 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 7 | 0 | 0.000 | 0.000 | 0.000 | 10 |

## Paired Causal Effects

| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | optimal-to-nonoptimal rate | true-to-not-true rate |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | -64 | 40 | 0.450 | 0.100 | 0.325 | 0.125 |
| 7 | -8 | 40 | 0.350 | 0.125 | 0.225 | 0.125 |
| 7 | 8 | 40 | 0.475 | 0.100 | 0.375 | 0.100 |
| 7 | 64 | 40 | 0.450 | 0.150 | 0.300 | 0.150 |

## Direction Controls

| layer | alpha | direction | desired dir | pairs | steered-action delta | desired-action delta | nonoptimal-to-desired rate |
|---:|---:|---|---|---:|---:|---:|---:|
| 7 | -64 | DOWN | True | 10 | -0.100 | -0.100 | 0.200 |
| 7 | -64 | LEFT | False | 10 | 0.500 | -0.400 | 0.100 |
| 7 | -64 | RIGHT | False | 10 | 0.400 | -0.400 | 0.000 |
| 7 | -64 | UP | False | 10 | 0.000 | 0.000 | 0.100 |
| 7 | -8 | DOWN | True | 10 | -0.200 | -0.200 | 0.100 |
| 7 | -8 | LEFT | False | 10 | 0.100 | -0.200 | 0.000 |
| 7 | -8 | RIGHT | False | 10 | 0.100 | -0.100 | 0.200 |
| 7 | -8 | UP | False | 10 | 0.000 | 0.100 | 0.200 |
| 7 | 8 | DOWN | True | 10 | -0.400 | -0.400 | 0.100 |
| 7 | 8 | LEFT | False | 10 | 0.200 | -0.300 | 0.100 |
| 7 | 8 | RIGHT | False | 10 | 0.000 | -0.100 | 0.100 |
| 7 | 8 | UP | False | 10 | 0.000 | -0.300 | 0.100 |
| 7 | 64 | DOWN | True | 10 | -0.300 | -0.300 | 0.200 |
| 7 | 64 | LEFT | False | 10 | 0.000 | -0.200 | 0.000 |
| 7 | 64 | RIGHT | False | 10 | 0.000 | -0.100 | 0.200 |
| 7 | 64 | UP | False | 10 | 0.000 | 0.000 | 0.200 |

## Interpretation

No hooked condition improved optimal-action rate versus no-hook. The least-negative condition is layer 7 at alpha -8: optimal-action rate 0.700, delta -0.100.

No hooked condition reduced true-action retention versus no-hook. The smallest true-action-rate delta is layer 7 at alpha -8: true-action rate 0.200, delta 0.000.

The strongest paired causal effect on emitted action is layer 7 at alpha 8: 0.475 of paired samples changed final action under the same generation seed.

The strongest paired effect away from the recorded true action is layer 7 at alpha 64: 0.150 of pairs changed from true-action emission to a different final action.

The strongest paired nonoptimal-to-optimal conversion is layer 7 at alpha 64: 0.150 of pairs converted from a nonoptimal emitted action to an action in `astar_actions`.

The direction-control check does not support desired-direction steering. The best desired single-action direction is DOWN at alpha -64, with desired-action delta -0.100; the best wrong-direction delta is 0.100.

These are distinct behavioral claims. A higher optimal-action rate means the model more often emits an action in `astar_actions`. A lower true-action rate means the model moved away from the recorded `agent_action`. Those can move in opposite directions because many recorded true actions are not in the optimal set.

## Artifacts

- `summary_by_layer_alpha.csv`
- `noop_baseline_agreement.csv`
- `paired_effects_vs_no_hook.csv`
- `direction_control_effects.csv`
- `optimal_action_rate_vs_alpha.svg`
- `true_action_rate_vs_alpha.svg`
- `nonoptimal_to_optimal_rate_vs_alpha.svg`
