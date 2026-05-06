# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_direction_control_layer15_true_to_down_nonoptimal.jsonl`
- Layers: 15
- Hooked alphas: -256, -128, -64, -32, -16, -8, 8, 16, 32, 64, 128, 256

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | no_hook |  | 4 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| 15 | hooked | -256 | 16 | 0.562 | 0.188 | 0.562 | 0.562 | -0.812 |
| 15 | hooked | -128 | 16 | 0.500 | 0.438 | 0.500 | 0.500 | -0.562 |
| 15 | hooked | -64 | 16 | 0.250 | 0.688 | 0.250 | 0.250 | -0.312 |
| 15 | hooked | -32 | 16 | 0.438 | 0.500 | 0.438 | 0.438 | -0.500 |
| 15 | hooked | -16 | 16 | 0.375 | 0.625 | 0.375 | 0.375 | -0.375 |
| 15 | hooked | -8 | 16 | 0.562 | 0.438 | 0.562 | 0.562 | -0.562 |
| 15 | hooked | 8 | 16 | 0.375 | 0.625 | 0.375 | 0.375 | -0.375 |
| 15 | hooked | 16 | 16 | 0.312 | 0.625 | 0.312 | 0.312 | -0.375 |
| 15 | hooked | 32 | 16 | 0.438 | 0.562 | 0.438 | 0.438 | -0.438 |
| 15 | hooked | 64 | 16 | 0.312 | 0.688 | 0.312 | 0.312 | -0.312 |
| 15 | hooked | 128 | 16 | 0.562 | 0.375 | 0.562 | 0.562 | -0.625 |
| 15 | hooked | 256 | 16 | 0.750 | 0.250 | 0.750 | 0.750 | -0.750 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 15 | 0 | 0.000 | 0.000 | 0.000 | 4 |

## Paired Causal Effects

| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | optimal-to-nonoptimal rate | true-to-not-true rate |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | -256 | 16 | 0.812 | 0.562 | 0.000 | 0.812 |
| 15 | -128 | 16 | 0.562 | 0.500 | 0.000 | 0.562 |
| 15 | -64 | 16 | 0.312 | 0.250 | 0.000 | 0.312 |
| 15 | -32 | 16 | 0.500 | 0.438 | 0.000 | 0.500 |
| 15 | -16 | 16 | 0.375 | 0.375 | 0.000 | 0.375 |
| 15 | -8 | 16 | 0.562 | 0.562 | 0.000 | 0.562 |
| 15 | 8 | 16 | 0.375 | 0.375 | 0.000 | 0.375 |
| 15 | 16 | 16 | 0.375 | 0.312 | 0.000 | 0.375 |
| 15 | 32 | 16 | 0.438 | 0.438 | 0.000 | 0.438 |
| 15 | 64 | 16 | 0.312 | 0.312 | 0.000 | 0.312 |
| 15 | 128 | 16 | 0.625 | 0.562 | 0.000 | 0.625 |
| 15 | 256 | 16 | 0.750 | 0.750 | 0.000 | 0.750 |

## Direction Controls

| layer | alpha | direction | desired dir | pairs | steered-action delta | desired-action delta | nonoptimal-to-desired rate |
|---:|---:|---|---|---:|---:|---:|---:|
| 15 | -256 | DOWN | True | 4 | 0.500 | 0.500 | 0.500 |
| 15 | -256 | LEFT | False | 4 | 0.000 | 0.750 | 0.750 |
| 15 | -256 | RIGHT | False | 4 | -0.750 | 0.500 | 0.500 |
| 15 | -256 | UP | False | 4 | 0.000 | 0.500 | 0.500 |
| 15 | -128 | DOWN | True | 4 | 0.500 | 0.500 | 0.500 |
| 15 | -128 | LEFT | False | 4 | 0.000 | 0.750 | 0.750 |
| 15 | -128 | RIGHT | False | 4 | -0.250 | 0.250 | 0.250 |
| 15 | -128 | UP | False | 4 | 0.000 | 0.500 | 0.500 |
| 15 | -64 | DOWN | True | 4 | 0.000 | 0.000 | 0.000 |
| 15 | -64 | LEFT | False | 4 | 0.250 | 0.250 | 0.250 |
| 15 | -64 | RIGHT | False | 4 | -0.250 | 0.250 | 0.250 |
| 15 | -64 | UP | False | 4 | 0.000 | 0.500 | 0.500 |
| 15 | -32 | DOWN | True | 4 | 0.750 | 0.750 | 0.750 |
| 15 | -32 | LEFT | False | 4 | 0.000 | 0.500 | 0.500 |
| 15 | -32 | RIGHT | False | 4 | 0.000 | 0.000 | 0.000 |
| 15 | -32 | UP | False | 4 | 0.000 | 0.500 | 0.500 |
| 15 | -16 | DOWN | True | 4 | 0.250 | 0.250 | 0.250 |
| 15 | -16 | LEFT | False | 4 | 0.000 | 0.250 | 0.250 |
| 15 | -16 | RIGHT | False | 4 | -0.250 | 0.250 | 0.250 |
| 15 | -16 | UP | False | 4 | 0.000 | 0.750 | 0.750 |
| 15 | -8 | DOWN | True | 4 | 0.500 | 0.500 | 0.500 |
| 15 | -8 | LEFT | False | 4 | 0.000 | 0.750 | 0.750 |
| 15 | -8 | RIGHT | False | 4 | -0.500 | 0.500 | 0.500 |
| 15 | -8 | UP | False | 4 | 0.000 | 0.500 | 0.500 |
| 15 | 8 | DOWN | True | 4 | 0.250 | 0.250 | 0.250 |
| 15 | 8 | LEFT | False | 4 | 0.000 | 0.250 | 0.250 |
| 15 | 8 | RIGHT | False | 4 | -0.750 | 0.750 | 0.750 |
| 15 | 8 | UP | False | 4 | 0.000 | 0.250 | 0.250 |
| 15 | 16 | DOWN | True | 4 | 0.500 | 0.500 | 0.500 |
| 15 | 16 | LEFT | False | 4 | 0.000 | 0.250 | 0.250 |
| 15 | 16 | RIGHT | False | 4 | -0.500 | 0.500 | 0.500 |
| 15 | 16 | UP | False | 4 | 0.000 | 0.000 | 0.000 |
| 15 | 32 | DOWN | True | 4 | 0.500 | 0.500 | 0.500 |
| 15 | 32 | LEFT | False | 4 | 0.000 | 0.500 | 0.500 |
| 15 | 32 | RIGHT | False | 4 | -0.750 | 0.750 | 0.750 |
| 15 | 32 | UP | False | 4 | 0.000 | 0.000 | 0.000 |
| 15 | 64 | DOWN | True | 4 | 0.500 | 0.500 | 0.500 |
| 15 | 64 | LEFT | False | 4 | 0.000 | 0.000 | 0.000 |
| 15 | 64 | RIGHT | False | 4 | -0.500 | 0.500 | 0.500 |
| 15 | 64 | UP | False | 4 | 0.000 | 0.250 | 0.250 |
| 15 | 128 | DOWN | True | 4 | 0.750 | 0.750 | 0.750 |
| 15 | 128 | LEFT | False | 4 | 0.000 | 0.250 | 0.250 |
| 15 | 128 | RIGHT | False | 4 | -1.000 | 1.000 | 1.000 |
| 15 | 128 | UP | False | 4 | 0.000 | 0.250 | 0.250 |
| 15 | 256 | DOWN | True | 4 | 0.750 | 0.750 | 0.750 |
| 15 | 256 | LEFT | False | 4 | 0.000 | 1.000 | 1.000 |
| 15 | 256 | RIGHT | False | 4 | -0.500 | 0.500 | 0.500 |
| 15 | 256 | UP | False | 4 | 0.000 | 0.750 | 0.750 |

## Interpretation

The largest optimal-action improvement versus no-hook is layer 15 at alpha 256: optimal-action rate 0.750, delta 0.750.

The largest true-action retention drop versus no-hook is layer 15 at alpha -256: true-action rate 0.188, delta -0.812.

The strongest paired causal effect on emitted action is layer 15 at alpha -256: 0.812 of paired samples changed final action under the same generation seed.

The strongest paired effect away from the recorded true action is layer 15 at alpha -256: 0.812 of pairs changed from true-action emission to a different final action.

The strongest paired nonoptimal-to-optimal conversion is layer 15 at alpha 256: 0.750 of pairs converted from a nonoptimal emitted action to an action in `astar_actions`.

The direction-control check does not support desired-direction steering. The best desired single-action direction is DOWN at alpha -32, with desired-action delta 0.750; the best wrong-direction delta is 1.000.

These are distinct behavioral claims. A higher optimal-action rate means the model more often emits an action in `astar_actions`. A lower true-action rate means the model moved away from the recorded `agent_action`. Those can move in opposite directions because many recorded true actions are not in the optimal set.

## Artifacts

- `summary_by_layer_alpha.csv`
- `noop_baseline_agreement.csv`
- `paired_effects_vs_no_hook.csv`
- `direction_control_effects.csv`
- `optimal_action_rate_vs_alpha.svg`
- `true_action_rate_vs_alpha.svg`
- `nonoptimal_to_optimal_rate_vs_alpha.svg`
