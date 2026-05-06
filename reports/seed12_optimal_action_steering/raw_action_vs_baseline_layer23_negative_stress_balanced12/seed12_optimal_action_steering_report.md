# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_raw_action_vs_baseline_layer23_negative_stress_balanced12.jsonl`
- Layers: 23
- Hooked alphas: -1024, -512

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 23 | no_hook |  | 36 | 0.583 | 0.139 | 0.583 | 0.000 | 0.000 |
| 23 | hooked | -1024 | 45 | 0.089 | 0.244 | 0.089 | -0.494 | 0.106 |
| 23 | hooked | -512 | 45 | 0.000 | 0.333 | 0.000 | -0.583 | 0.194 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 23 | 0 | 0.000 | 0.000 | 0.000 | 36 |

## Paired Causal Effects

| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | optimal-to-nonoptimal rate | true-to-not-true rate |
|---:|---:|---:|---:|---:|---:|---:|
| 23 | -1024 | 45 | 0.156 | 0.089 | 0.000 | 0.089 |
| 23 | -512 | 45 | 0.000 | 0.000 | 0.000 | 0.000 |

## Direction Controls

| layer | alpha | direction | desired dir | pairs | steered-action delta | desired-action delta | nonoptimal-to-desired rate |
|---:|---:|---|---|---:|---:|---:|---:|
| 23 | -1024 | DOWN | False | 8 | 0.000 | 0.000 | 0.000 |
| 23 | -1024 | DOWN | True | 6 | 0.000 | 0.000 | 0.000 |
| 23 | -1024 | LEFT | False | 9 | 0.111 | 0.000 | 0.000 |
| 23 | -1024 | LEFT | True | 4 | 0.500 | 0.500 | 0.500 |
| 23 | -1024 | RIGHT | False | 7 | 0.143 | 0.143 | 0.143 |
| 23 | -1024 | RIGHT | True | 5 | 0.200 | 0.200 | 0.200 |
| 23 | -1024 | UP | False | 6 | 0.000 | 0.000 | 0.000 |
| 23 | -512 | DOWN | False | 8 | 0.000 | 0.000 | 0.000 |
| 23 | -512 | DOWN | True | 6 | 0.000 | 0.000 | 0.000 |
| 23 | -512 | LEFT | False | 9 | 0.000 | 0.000 | 0.000 |
| 23 | -512 | LEFT | True | 4 | 0.000 | 0.000 | 0.000 |
| 23 | -512 | RIGHT | False | 7 | 0.000 | 0.000 | 0.000 |
| 23 | -512 | RIGHT | True | 5 | 0.000 | 0.000 | 0.000 |
| 23 | -512 | UP | False | 6 | 0.000 | 0.000 | 0.000 |

## Direction-Control Advantage

| layer | alpha | desired direction | desired delta | best wrong direction | best wrong delta | desired advantage | supports desired direction |
|---:|---:|---|---:|---|---:|---:|---|
| 23 | -1024 | LEFT | 0.500 | RIGHT | 0.143 | 0.357 | True |
| 23 | -512 | DOWN | 0.000 | DOWN | 0.000 | 0.000 | False |

## Overall Direction Controls

| layer | alpha | direction group | pairs | desired-action delta | nonoptimal-to-desired rate | desired-to-nondesired rate |
|---:|---:|---|---:|---:|---:|---:|
| 23 | -1024 | wrong | 30 | 0.033 | 0.033 | 0.000 |
| 23 | -1024 | desired | 15 | 0.200 | 0.200 | 0.000 |
| 23 | -512 | wrong | 30 | 0.000 | 0.000 | 0.000 |
| 23 | -512 | desired | 15 | 0.000 | 0.000 | 0.000 |

## Overall Direction-Control Advantage

| layer | alpha | desired delta | wrong delta | desired advantage | supports desired direction |
|---:|---:|---:|---:|---:|---|
| 23 | -1024 | 0.200 | 0.033 | 0.167 | True |
| 23 | -512 | 0.000 | 0.000 | 0.000 | False |

## Interpretation

No hooked condition improved optimal-action rate versus no-hook. The least-negative condition is layer 23 at alpha -1024: optimal-action rate 0.089, delta -0.494.

No hooked condition reduced true-action retention versus no-hook. The smallest true-action-rate delta is layer 23 at alpha -1024: true-action rate 0.244, delta 0.106.

The strongest paired causal effect on emitted action is layer 23 at alpha -1024: 0.156 of paired samples changed final action under the same generation seed.

The strongest paired effect away from the recorded true action is layer 23 at alpha -1024: 0.089 of pairs changed from true-action emission to a different final action.

The strongest paired nonoptimal-to-optimal conversion is layer 23 at alpha -1024: 0.089 of pairs converted from a nonoptimal emitted action to an action in `astar_actions`.

The per-action same-alpha direction-control check supports desired-direction steering at its strongest tested setting: layer 23 alpha -1024, desired direction LEFT delta 0.500, best wrong direction RIGHT delta 0.143, advantage 0.357.

The overall desired-vs-wrong control is strictly positive at its strongest tested setting: layer 23 alpha -1024, desired-direction delta 0.200, wrong-direction delta 0.033, advantage 0.167, nonoptimal-to-desired advantage 0.167.

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
