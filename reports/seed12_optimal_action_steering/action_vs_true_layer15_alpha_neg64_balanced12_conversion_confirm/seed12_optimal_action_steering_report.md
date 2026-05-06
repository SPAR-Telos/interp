# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_action_vs_true_layer15_alpha_neg64_balanced12_conversion_confirm.jsonl`
- Layers: 15
- Hooked alphas: -64

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | no_hook |  | 60 | 0.400 | 0.333 | 0.400 | 0.000 | 0.000 |
| 15 | hooked | -64 | 180 | 0.500 | 0.300 | 0.500 | 0.100 | -0.033 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 15 | 0 | 0.000 | 0.000 | 0.000 | 60 |

## Paired Causal Effects

| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | optimal-to-nonoptimal rate | true-to-not-true rate |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | -64 | 180 | 0.372 | 0.183 | 0.083 | 0.128 |

## Direction Controls

| layer | alpha | direction | desired dir | pairs | steered-action delta | desired-action delta | nonoptimal-to-desired rate |
|---:|---:|---|---|---:|---:|---:|---:|
| 15 | -64 | DOWN | False | 45 | -0.022 | 0.133 | 0.200 |
| 15 | -64 | DOWN | True | 15 | -0.067 | -0.067 | 0.000 |
| 15 | -64 | LEFT | False | 40 | 0.000 | 0.075 | 0.175 |
| 15 | -64 | LEFT | True | 15 | 0.200 | 0.200 | 0.333 |
| 15 | -64 | RIGHT | False | 15 | 0.133 | -0.067 | 0.067 |
| 15 | -64 | RIGHT | True | 15 | 0.200 | 0.200 | 0.200 |
| 15 | -64 | UP | False | 20 | -0.150 | 0.150 | 0.300 |
| 15 | -64 | UP | True | 15 | 0.133 | 0.133 | 0.133 |

## Direction-Control Advantage

| layer | alpha | desired direction | desired delta | best wrong direction | best wrong delta | desired advantage | supports desired direction |
|---:|---:|---|---:|---|---:|---:|---|
| 15 | -64 | LEFT | 0.200 | UP | 0.150 | 0.050 | True |

## Overall Direction Controls

| layer | alpha | direction group | pairs | desired-action delta | nonoptimal-to-desired rate | desired-to-nondesired rate |
|---:|---:|---|---:|---:|---:|---:|
| 15 | -64 | wrong | 120 | 0.092 | 0.192 | 0.100 |
| 15 | -64 | desired | 60 | 0.117 | 0.167 | 0.050 |

## Overall Direction-Control Advantage

| layer | alpha | desired delta | wrong delta | desired advantage | supports desired direction |
|---:|---:|---:|---:|---:|---|
| 15 | -64 | 0.117 | 0.092 | 0.025 | False |

## Interpretation

The largest optimal-action improvement versus no-hook is layer 15 at alpha -64: optimal-action rate 0.500, delta 0.100.

The largest true-action retention drop versus no-hook is layer 15 at alpha -64: true-action rate 0.300, delta -0.033.

The strongest paired causal effect on emitted action is layer 15 at alpha -64: 0.372 of paired samples changed final action under the same generation seed.

The strongest paired effect away from the recorded true action is layer 15 at alpha -64: 0.128 of pairs changed from true-action emission to a different final action.

The strongest paired nonoptimal-to-optimal conversion is layer 15 at alpha -64: 0.183 of pairs converted from a nonoptimal emitted action to an action in `astar_actions`.

The per-action direction-control table contains a local positive slice, but the overall desired-vs-wrong control fails the strict support criterion. The strongest local slice is layer 15 alpha -64, desired direction LEFT delta 0.200, best wrong direction UP delta 0.150, advantage 0.050.

The overall desired-vs-wrong control fails the strict support criterion. The best tested setting is layer 15 alpha -64, desired-direction delta 0.117, wrong-direction delta 0.092, advantage 0.025, nonoptimal-to-desired advantage -0.025.

These are distinct behavioral claims. A higher optimal-action rate means the model more often emits an action in `astar_actions`. A lower true-action rate means the model moved away from the recorded `agent_action`. Those can move in opposite directions because many recorded true actions are not in the optimal set.

## Artifacts

- `summary_by_layer_alpha.csv`
- `noop_baseline_agreement.csv`
- `paired_effects_vs_no_hook.csv`
- `direction_control_effects.csv`
- `direction_control_advantage.csv`
- `direction_control_overall.csv`
- `direction_control_overall_advantage.csv`
- `optimal_action_rate_vs_alpha.svg`
- `true_action_rate_vs_alpha.svg`
- `nonoptimal_to_optimal_rate_vs_alpha.svg`
