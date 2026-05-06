# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_action_contrast_layer15_alpha_neg64_balanced12_smoke.jsonl`
- Layers: 15
- Hooked alphas: -64

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | no_hook |  | 12 | 0.500 | 0.167 | 0.500 | 0.000 | 0.000 |
| 15 | hooked | -64 | 48 | 0.583 | 0.208 | 0.583 | 0.083 | 0.042 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 15 | 0 | 0.000 | 0.000 | 0.000 | 12 |

## Paired Causal Effects

| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | optimal-to-nonoptimal rate | true-to-not-true rate |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | -64 | 48 | 0.417 | 0.208 | 0.125 | 0.083 |

## Direction Controls

| layer | alpha | direction | desired dir | pairs | steered-action delta | desired-action delta | nonoptimal-to-desired rate |
|---:|---:|---|---|---:|---:|---:|---:|
| 15 | -64 | DOWN | False | 9 | 0.000 | 0.222 | 0.333 |
| 15 | -64 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | -64 | LEFT | False | 9 | -0.111 | 0.000 | 0.111 |
| 15 | -64 | LEFT | True | 3 | 0.333 | 0.333 | 0.333 |
| 15 | -64 | RIGHT | False | 9 | 0.000 | 0.111 | 0.222 |
| 15 | -64 | RIGHT | True | 3 | -0.333 | -0.333 | 0.000 |
| 15 | -64 | UP | False | 9 | 0.000 | 0.111 | 0.222 |
| 15 | -64 | UP | True | 3 | 0.000 | 0.000 | 0.333 |

## Direction-Control Advantage

| layer | alpha | desired direction | desired delta | best wrong direction | best wrong delta | desired advantage | supports desired direction |
|---:|---:|---|---:|---|---:|---:|---|
| 15 | -64 | LEFT | 0.333 | DOWN | 0.222 | 0.111 | True |

## Overall Direction Controls

| layer | alpha | direction group | pairs | desired-action delta | nonoptimal-to-desired rate | desired-to-nondesired rate |
|---:|---:|---|---:|---:|---:|---:|
| 15 | -64 | wrong | 36 | 0.111 | 0.222 | 0.111 |
| 15 | -64 | desired | 12 | 0.000 | 0.167 | 0.167 |

## Overall Direction-Control Advantage

| layer | alpha | desired delta | wrong delta | desired advantage | supports desired direction |
|---:|---:|---:|---:|---:|---|
| 15 | -64 | 0.000 | 0.111 | -0.111 | False |

## Interpretation

The largest optimal-action improvement versus no-hook is layer 15 at alpha -64: optimal-action rate 0.583, delta 0.083.

No hooked condition reduced true-action retention versus no-hook. The smallest true-action-rate delta is layer 15 at alpha -64: true-action rate 0.208, delta 0.042.

The strongest paired causal effect on emitted action is layer 15 at alpha -64: 0.417 of paired samples changed final action under the same generation seed.

The strongest paired effect away from the recorded true action is layer 15 at alpha -64: 0.083 of pairs changed from true-action emission to a different final action.

The strongest paired nonoptimal-to-optimal conversion is layer 15 at alpha -64: 0.208 of pairs converted from a nonoptimal emitted action to an action in `astar_actions`.

The same-alpha direction-control check supports desired-direction steering at its strongest tested setting: layer 15 alpha -64, desired direction LEFT delta 0.333, best wrong direction DOWN delta 0.222, advantage 0.111.

The overall desired-vs-wrong control is not positive. The best tested setting is layer 15 alpha -64, desired-direction delta 0.000, wrong-direction delta 0.111, advantage -0.111.

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
