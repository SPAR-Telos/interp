# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_centroid_action_vs_true_layer15_balanced12_alpha_screen.jsonl`
- Layers: 15
- Hooked alphas: -1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | no_hook |  | 12 | 0.417 | 0.250 | 0.417 | 0.000 | 0.000 |
| 15 | hooked | -1 | 36 | 0.528 | 0.278 | 0.528 | 0.111 | 0.028 |
| 15 | hooked | -0.5 | 36 | 0.583 | 0.222 | 0.583 | 0.167 | -0.028 |
| 15 | hooked | -0.25 | 36 | 0.361 | 0.417 | 0.361 | -0.056 | 0.167 |
| 15 | hooked | -0.1 | 36 | 0.472 | 0.389 | 0.472 | 0.056 | 0.139 |
| 15 | hooked | 0.1 | 36 | 0.472 | 0.333 | 0.472 | 0.056 | 0.083 |
| 15 | hooked | 0.25 | 36 | 0.444 | 0.306 | 0.444 | 0.028 | 0.056 |
| 15 | hooked | 0.5 | 36 | 0.611 | 0.222 | 0.611 | 0.194 | -0.028 |
| 15 | hooked | 1 | 36 | 0.583 | 0.167 | 0.583 | 0.167 | -0.083 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 15 | 0 | 0.000 | 0.000 | 0.000 | 12 |

## Paired Causal Effects

| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | optimal-to-nonoptimal rate | true-to-not-true rate |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | -1 | 36 | 0.444 | 0.167 | 0.056 | 0.139 |
| 15 | -0.5 | 36 | 0.472 | 0.250 | 0.083 | 0.167 |
| 15 | -0.25 | 36 | 0.444 | 0.139 | 0.194 | 0.056 |
| 15 | -0.1 | 36 | 0.500 | 0.194 | 0.139 | 0.111 |
| 15 | 0.1 | 36 | 0.389 | 0.167 | 0.111 | 0.083 |
| 15 | 0.25 | 36 | 0.361 | 0.111 | 0.083 | 0.083 |
| 15 | 0.5 | 36 | 0.361 | 0.194 | 0.000 | 0.167 |
| 15 | 1 | 36 | 0.389 | 0.194 | 0.028 | 0.139 |

## Direction Controls

| layer | alpha | direction | desired dir | pairs | steered-action delta | desired-action delta | nonoptimal-to-desired rate |
|---:|---:|---|---|---:|---:|---:|---:|
| 15 | -1 | DOWN | False | 9 | 0.000 | 0.333 | 0.333 |
| 15 | -1 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | -1 | LEFT | False | 8 | -0.125 | 0.000 | 0.000 |
| 15 | -1 | LEFT | True | 3 | 0.333 | 0.333 | 0.333 |
| 15 | -1 | RIGHT | False | 3 | -0.333 | 0.333 | 0.333 |
| 15 | -1 | RIGHT | True | 3 | 0.000 | 0.000 | 0.333 |
| 15 | -1 | UP | False | 4 | 0.000 | 0.000 | 0.000 |
| 15 | -1 | UP | True | 3 | -0.333 | -0.333 | 0.000 |
| 15 | -0.5 | DOWN | False | 9 | 0.000 | 0.333 | 0.333 |
| 15 | -0.5 | DOWN | True | 3 | -0.333 | -0.333 | 0.000 |
| 15 | -0.5 | LEFT | False | 8 | 0.000 | 0.000 | 0.125 |
| 15 | -0.5 | LEFT | True | 3 | 0.667 | 0.667 | 0.667 |
| 15 | -0.5 | RIGHT | False | 3 | 0.000 | 0.333 | 0.333 |
| 15 | -0.5 | RIGHT | True | 3 | 0.333 | 0.333 | 0.333 |
| 15 | -0.5 | UP | False | 4 | 0.000 | 0.000 | 0.000 |
| 15 | -0.5 | UP | True | 3 | 0.000 | 0.000 | 0.333 |
| 15 | -0.25 | DOWN | False | 9 | 0.000 | 0.000 | 0.111 |
| 15 | -0.25 | DOWN | True | 3 | -0.333 | -0.333 | 0.000 |
| 15 | -0.25 | LEFT | False | 8 | 0.000 | -0.375 | 0.000 |
| 15 | -0.25 | LEFT | True | 3 | 0.333 | 0.333 | 0.333 |
| 15 | -0.25 | RIGHT | False | 3 | -0.333 | 0.667 | 0.667 |
| 15 | -0.25 | RIGHT | True | 3 | 0.000 | 0.000 | 0.333 |
| 15 | -0.25 | UP | False | 4 | 0.000 | 0.000 | 0.000 |
| 15 | -0.25 | UP | True | 3 | -0.333 | -0.333 | 0.000 |
| 15 | -0.1 | DOWN | False | 9 | 0.000 | 0.222 | 0.333 |
| 15 | -0.1 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | -0.1 | LEFT | False | 8 | 0.000 | -0.250 | 0.000 |
| 15 | -0.1 | LEFT | True | 3 | 0.333 | 0.333 | 0.333 |
| 15 | -0.1 | RIGHT | False | 3 | -0.333 | 0.667 | 0.667 |
| 15 | -0.1 | RIGHT | True | 3 | 0.333 | 0.333 | 0.333 |
| 15 | -0.1 | UP | False | 4 | 0.000 | -0.250 | 0.000 |
| 15 | -0.1 | UP | True | 3 | -0.333 | -0.333 | 0.000 |
| 15 | 0.1 | DOWN | False | 9 | 0.000 | -0.111 | 0.111 |
| 15 | 0.1 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 0.1 | LEFT | False | 8 | 0.000 | -0.125 | 0.000 |
| 15 | 0.1 | LEFT | True | 3 | 0.667 | 0.667 | 0.667 |
| 15 | 0.1 | RIGHT | False | 3 | -0.333 | 0.667 | 0.667 |
| 15 | 0.1 | RIGHT | True | 3 | 0.000 | 0.000 | 0.333 |
| 15 | 0.1 | UP | False | 4 | 0.000 | 0.000 | 0.000 |
| 15 | 0.1 | UP | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 0.25 | DOWN | False | 9 | 0.000 | 0.111 | 0.222 |
| 15 | 0.25 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 0.25 | LEFT | False | 8 | 0.125 | -0.125 | 0.000 |
| 15 | 0.25 | LEFT | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 0.25 | RIGHT | False | 3 | 0.000 | 0.333 | 0.333 |
| 15 | 0.25 | RIGHT | True | 3 | -0.333 | -0.333 | 0.000 |
| 15 | 0.25 | UP | False | 4 | -0.250 | 0.250 | 0.250 |
| 15 | 0.25 | UP | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 0.5 | DOWN | False | 9 | 0.000 | 0.111 | 0.111 |
| 15 | 0.5 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 0.5 | LEFT | False | 8 | -0.125 | 0.125 | 0.125 |
| 15 | 0.5 | LEFT | True | 3 | 0.333 | 0.333 | 0.333 |
| 15 | 0.5 | RIGHT | False | 3 | -0.333 | 0.333 | 0.333 |
| 15 | 0.5 | RIGHT | True | 3 | 0.333 | 0.333 | 0.333 |
| 15 | 0.5 | UP | False | 4 | -0.250 | 0.250 | 0.250 |
| 15 | 0.5 | UP | True | 3 | 0.333 | 0.333 | 0.333 |
| 15 | 1 | DOWN | False | 9 | 0.000 | 0.222 | 0.222 |
| 15 | 1 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 1 | LEFT | False | 8 | -0.125 | 0.000 | 0.000 |
| 15 | 1 | LEFT | True | 3 | 0.333 | 0.333 | 0.333 |
| 15 | 1 | RIGHT | False | 3 | 0.000 | 0.333 | 0.333 |
| 15 | 1 | RIGHT | True | 3 | -0.333 | -0.333 | 0.000 |
| 15 | 1 | UP | False | 4 | -0.500 | 0.500 | 0.500 |
| 15 | 1 | UP | True | 3 | 0.333 | 0.333 | 0.333 |

## Direction-Control Advantage

| layer | alpha | desired direction | desired delta | best wrong direction | best wrong delta | desired advantage | supports desired direction |
|---:|---:|---|---:|---|---:|---:|---|
| 15 | -1 | LEFT | 0.333 | DOWN | 0.333 | -0.000 | False |
| 15 | -0.5 | LEFT | 0.667 | DOWN | 0.333 | 0.333 | True |
| 15 | -0.25 | LEFT | 0.333 | RIGHT | 0.667 | -0.333 | False |
| 15 | -0.1 | LEFT | 0.333 | RIGHT | 0.667 | -0.333 | False |
| 15 | 0.1 | LEFT | 0.667 | RIGHT | 0.667 | 0.000 | True |
| 15 | 0.25 | DOWN | 0.000 | RIGHT | 0.333 | -0.333 | False |
| 15 | 0.5 | UP | 0.333 | RIGHT | 0.333 | 0.000 | True |
| 15 | 1 | UP | 0.333 | UP | 0.500 | -0.167 | False |

## Overall Direction Controls

| layer | alpha | direction group | pairs | desired-action delta | nonoptimal-to-desired rate | desired-to-nondesired rate |
|---:|---:|---|---:|---:|---:|---:|
| 15 | -1 | wrong | 24 | 0.167 | 0.167 | 0.000 |
| 15 | -1 | desired | 12 | 0.000 | 0.167 | 0.167 |
| 15 | -0.5 | wrong | 24 | 0.167 | 0.208 | 0.042 |
| 15 | -0.5 | desired | 12 | 0.167 | 0.333 | 0.167 |
| 15 | -0.25 | wrong | 24 | -0.042 | 0.125 | 0.167 |
| 15 | -0.25 | desired | 12 | -0.083 | 0.167 | 0.250 |
| 15 | -0.1 | wrong | 24 | 0.042 | 0.208 | 0.167 |
| 15 | -0.1 | desired | 12 | 0.083 | 0.167 | 0.083 |
| 15 | 0.1 | wrong | 24 | 0.000 | 0.125 | 0.125 |
| 15 | 0.1 | desired | 12 | 0.167 | 0.250 | 0.083 |
| 15 | 0.25 | wrong | 24 | 0.083 | 0.167 | 0.083 |
| 15 | 0.25 | desired | 12 | -0.083 | 0.000 | 0.083 |
| 15 | 0.5 | wrong | 24 | 0.167 | 0.167 | 0.000 |
| 15 | 0.5 | desired | 12 | 0.250 | 0.250 | 0.000 |
| 15 | 1 | wrong | 24 | 0.208 | 0.208 | 0.000 |
| 15 | 1 | desired | 12 | 0.083 | 0.167 | 0.083 |

## Overall Direction-Control Advantage

| layer | alpha | desired delta | wrong delta | desired advantage | supports desired direction |
|---:|---:|---:|---:|---:|---|
| 15 | -1 | 0.000 | 0.167 | -0.167 | False |
| 15 | -0.5 | 0.167 | 0.167 | 0.000 | False |
| 15 | -0.25 | -0.083 | -0.042 | -0.042 | False |
| 15 | -0.1 | 0.083 | 0.042 | 0.042 | True |
| 15 | 0.1 | 0.167 | 0.000 | 0.167 | True |
| 15 | 0.25 | -0.083 | 0.083 | -0.167 | False |
| 15 | 0.5 | 0.250 | 0.167 | 0.083 | True |
| 15 | 1 | 0.083 | 0.208 | -0.125 | False |

## Interpretation

The largest optimal-action improvement versus no-hook is layer 15 at alpha 0.5: optimal-action rate 0.611, delta 0.194.

The largest true-action retention drop versus no-hook is layer 15 at alpha 1: true-action rate 0.167, delta -0.083.

The strongest paired causal effect on emitted action is layer 15 at alpha -0.1: 0.500 of paired samples changed final action under the same generation seed.

The strongest paired effect away from the recorded true action is layer 15 at alpha -0.5: 0.167 of pairs changed from true-action emission to a different final action.

The strongest paired nonoptimal-to-optimal conversion is layer 15 at alpha -0.5: 0.250 of pairs converted from a nonoptimal emitted action to an action in `astar_actions`.

The per-action same-alpha direction-control check supports desired-direction steering at its strongest tested setting: layer 15 alpha -0.5, desired direction LEFT delta 0.667, best wrong direction DOWN delta 0.333, advantage 0.333.

The overall desired-vs-wrong control is positive at its strongest tested setting: layer 15 alpha 0.1, desired-direction delta 0.167, wrong-direction delta 0.000, advantage 0.167.

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
