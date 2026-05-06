# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_direction_control_layer15_alpha_neg64_balanced12_confirm.jsonl`
- Layers: 15
- Hooked alphas: -64

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | no_hook |  | 60 | 0.517 | 0.267 | 0.517 | 0.000 | 0.000 |
| 15 | hooked | -64 | 240 | 0.467 | 0.283 | 0.467 | -0.050 | 0.017 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 15 | 0 | 0.000 | 0.000 | 0.000 | 60 |

## Paired Causal Effects

| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | optimal-to-nonoptimal rate | true-to-not-true rate |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | -64 | 240 | 0.450 | 0.142 | 0.192 | 0.133 |

## Direction Controls

| layer | alpha | direction | desired dir | pairs | steered-action delta | desired-action delta | nonoptimal-to-desired rate |
|---:|---:|---|---|---:|---:|---:|---:|
| 15 | -64 | DOWN | False | 45 | -0.022 | -0.111 | 0.133 |
| 15 | -64 | DOWN | True | 15 | 0.000 | 0.000 | 0.067 |
| 15 | -64 | LEFT | False | 45 | -0.022 | 0.089 | 0.222 |
| 15 | -64 | LEFT | True | 15 | -0.200 | -0.200 | 0.000 |
| 15 | -64 | RIGHT | False | 45 | 0.000 | -0.111 | 0.111 |
| 15 | -64 | RIGHT | True | 15 | 0.000 | 0.000 | 0.267 |
| 15 | -64 | UP | False | 45 | 0.133 | -0.044 | 0.156 |
| 15 | -64 | UP | True | 15 | -0.067 | -0.067 | 0.067 |

## Direction-Control Advantage

| layer | alpha | desired direction | desired delta | best wrong direction | best wrong delta | desired advantage | supports desired direction |
|---:|---:|---|---:|---|---:|---:|---|
| 15 | -64 | RIGHT | 0.000 | LEFT | 0.089 | -0.089 | False |

## Overall Direction Controls

| layer | alpha | direction group | pairs | desired-action delta | nonoptimal-to-desired rate | desired-to-nondesired rate |
|---:|---:|---|---:|---:|---:|---:|
| 15 | -64 | wrong | 180 | -0.044 | 0.156 | 0.200 |
| 15 | -64 | desired | 60 | -0.067 | 0.100 | 0.167 |

## Overall Direction-Control Advantage

| layer | alpha | desired delta | wrong delta | desired advantage | supports desired direction |
|---:|---:|---:|---:|---:|---|
| 15 | -64 | -0.067 | -0.044 | -0.022 | False |

## Interpretation

No hooked condition improved optimal-action rate versus no-hook. The least-negative condition is layer 15 at alpha -64: optimal-action rate 0.467, delta -0.050.

No hooked condition reduced true-action retention versus no-hook. The smallest true-action-rate delta is layer 15 at alpha -64: true-action rate 0.283, delta 0.017.

The strongest paired causal effect on emitted action is layer 15 at alpha -64: 0.450 of paired samples changed final action under the same generation seed.

The strongest paired effect away from the recorded true action is layer 15 at alpha -64: 0.133 of pairs changed from true-action emission to a different final action.

The strongest paired nonoptimal-to-optimal conversion is layer 15 at alpha -64: 0.142 of pairs converted from a nonoptimal emitted action to an action in `astar_actions`.

The same-alpha direction-control check does not support desired-direction steering. The best tested desired-vs-wrong advantage is layer 15 alpha -64: desired direction RIGHT delta 0.000, best wrong direction LEFT delta 0.089, advantage -0.089.

The overall desired-vs-wrong control is not positive. The best tested setting is layer 15 alpha -64, desired-direction delta -0.067, wrong-direction delta -0.044, advantage -0.022.

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
