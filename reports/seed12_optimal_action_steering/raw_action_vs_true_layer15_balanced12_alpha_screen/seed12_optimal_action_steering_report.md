# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_raw_action_vs_true_layer15_balanced12_alpha_screen.jsonl`
- Layers: 15
- Hooked alphas: -128, -64, -32, -16, -8, 8, 16, 32, 64, 128

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | no_hook |  | 12 | 0.417 | 0.250 | 0.417 | 0.000 | 0.000 |
| 15 | hooked | -128 | 36 | 0.500 | 0.222 | 0.500 | 0.083 | -0.028 |
| 15 | hooked | -64 | 36 | 0.500 | 0.139 | 0.500 | 0.083 | -0.111 |
| 15 | hooked | -32 | 36 | 0.500 | 0.250 | 0.500 | 0.083 | 0.000 |
| 15 | hooked | -16 | 36 | 0.444 | 0.306 | 0.444 | 0.028 | 0.056 |
| 15 | hooked | -8 | 36 | 0.417 | 0.278 | 0.417 | 0.000 | 0.028 |
| 15 | hooked | 8 | 36 | 0.417 | 0.333 | 0.417 | 0.000 | 0.083 |
| 15 | hooked | 16 | 36 | 0.472 | 0.222 | 0.472 | 0.056 | -0.028 |
| 15 | hooked | 32 | 36 | 0.361 | 0.306 | 0.361 | -0.056 | 0.056 |
| 15 | hooked | 64 | 36 | 0.472 | 0.222 | 0.472 | 0.056 | -0.028 |
| 15 | hooked | 128 | 36 | 0.528 | 0.222 | 0.528 | 0.111 | -0.028 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 15 | 0 | 0.000 | 0.000 | 0.000 | 12 |

## Paired Causal Effects

| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | optimal-to-nonoptimal rate | true-to-not-true rate |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | -128 | 36 | 0.250 | 0.111 | 0.028 | 0.083 |
| 15 | -64 | 36 | 0.472 | 0.167 | 0.083 | 0.167 |
| 15 | -32 | 36 | 0.500 | 0.167 | 0.083 | 0.167 |
| 15 | -16 | 36 | 0.333 | 0.056 | 0.028 | 0.083 |
| 15 | -8 | 36 | 0.250 | 0.056 | 0.056 | 0.056 |
| 15 | 8 | 36 | 0.250 | 0.000 | 0.000 | 0.083 |
| 15 | 16 | 36 | 0.333 | 0.056 | 0.000 | 0.139 |
| 15 | 32 | 36 | 0.194 | 0.000 | 0.056 | 0.028 |
| 15 | 64 | 36 | 0.194 | 0.056 | 0.000 | 0.083 |
| 15 | 128 | 36 | 0.389 | 0.167 | 0.056 | 0.139 |

## Direction Controls

| layer | alpha | direction | desired dir | pairs | steered-action delta | desired-action delta | nonoptimal-to-desired rate |
|---:|---:|---|---|---:|---:|---:|---:|
| 15 | -128 | DOWN | False | 9 | 0.000 | 0.333 | 0.333 |
| 15 | -128 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | -128 | LEFT | False | 8 | -0.125 | 0.000 | 0.000 |
| 15 | -128 | LEFT | True | 3 | -0.333 | -0.333 | 0.000 |
| 15 | -128 | RIGHT | False | 3 | 0.333 | 0.000 | 0.000 |
| 15 | -128 | RIGHT | True | 3 | 0.333 | 0.333 | 0.333 |
| 15 | -128 | UP | False | 4 | 0.000 | 0.000 | 0.000 |
| 15 | -128 | UP | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | -64 | DOWN | False | 9 | 0.000 | 0.111 | 0.222 |
| 15 | -64 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | -64 | LEFT | False | 8 | -0.125 | 0.125 | 0.250 |
| 15 | -64 | LEFT | True | 3 | 0.000 | 0.000 | 0.333 |
| 15 | -64 | RIGHT | False | 3 | 0.667 | 0.000 | 0.000 |
| 15 | -64 | RIGHT | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | -64 | UP | False | 4 | -0.250 | 0.250 | 0.250 |
| 15 | -64 | UP | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | -32 | DOWN | False | 9 | 0.000 | 0.111 | 0.111 |
| 15 | -32 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | -32 | LEFT | False | 8 | -0.250 | 0.250 | 0.250 |
| 15 | -32 | LEFT | True | 3 | 0.333 | 0.333 | 0.333 |
| 15 | -32 | RIGHT | False | 3 | 0.333 | 0.000 | 0.000 |
| 15 | -32 | RIGHT | True | 3 | 0.333 | 0.333 | 0.333 |
| 15 | -32 | UP | False | 4 | 0.000 | 0.000 | 0.250 |
| 15 | -32 | UP | True | 3 | -0.667 | -0.667 | 0.000 |
| 15 | -16 | DOWN | False | 9 | 0.000 | 0.111 | 0.111 |
| 15 | -16 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | -16 | LEFT | False | 8 | -0.125 | 0.000 | 0.000 |
| 15 | -16 | LEFT | True | 3 | 0.333 | 0.333 | 0.333 |
| 15 | -16 | RIGHT | False | 3 | 0.000 | 0.000 | 0.000 |
| 15 | -16 | RIGHT | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | -16 | UP | False | 4 | 0.000 | -0.250 | 0.000 |
| 15 | -16 | UP | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | -8 | DOWN | False | 9 | 0.000 | 0.111 | 0.111 |
| 15 | -8 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | -8 | LEFT | False | 8 | 0.000 | -0.125 | 0.000 |
| 15 | -8 | LEFT | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | -8 | RIGHT | False | 3 | 0.333 | 0.333 | 0.333 |
| 15 | -8 | RIGHT | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | -8 | UP | False | 4 | 0.000 | 0.000 | 0.000 |
| 15 | -8 | UP | True | 3 | -0.333 | -0.333 | 0.000 |
| 15 | 8 | DOWN | False | 9 | 0.000 | 0.000 | 0.000 |
| 15 | 8 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 8 | LEFT | False | 8 | -0.250 | 0.000 | 0.000 |
| 15 | 8 | LEFT | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 8 | RIGHT | False | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 8 | RIGHT | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 8 | UP | False | 4 | 0.000 | 0.000 | 0.000 |
| 15 | 8 | UP | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 16 | DOWN | False | 9 | 0.000 | 0.000 | 0.000 |
| 15 | 16 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 16 | LEFT | False | 8 | 0.000 | 0.000 | 0.000 |
| 15 | 16 | LEFT | True | 3 | 0.333 | 0.333 | 0.333 |
| 15 | 16 | RIGHT | False | 3 | 0.333 | 0.000 | 0.000 |
| 15 | 16 | RIGHT | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 16 | UP | False | 4 | -0.250 | 0.250 | 0.250 |
| 15 | 16 | UP | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 32 | DOWN | False | 9 | 0.000 | -0.111 | 0.000 |
| 15 | 32 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 32 | LEFT | False | 8 | -0.125 | 0.000 | 0.000 |
| 15 | 32 | LEFT | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 32 | RIGHT | False | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 32 | RIGHT | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 32 | UP | False | 4 | 0.250 | -0.250 | 0.000 |
| 15 | 32 | UP | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 64 | DOWN | False | 9 | 0.000 | 0.111 | 0.111 |
| 15 | 64 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 64 | LEFT | False | 8 | 0.000 | 0.000 | 0.000 |
| 15 | 64 | LEFT | True | 3 | 0.333 | 0.333 | 0.333 |
| 15 | 64 | RIGHT | False | 3 | 0.333 | 0.000 | 0.000 |
| 15 | 64 | RIGHT | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 64 | UP | False | 4 | 0.000 | 0.000 | 0.000 |
| 15 | 64 | UP | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 128 | DOWN | False | 9 | 0.000 | 0.111 | 0.111 |
| 15 | 128 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 128 | LEFT | False | 8 | -0.250 | -0.125 | 0.000 |
| 15 | 128 | LEFT | True | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 128 | RIGHT | False | 3 | 0.000 | 0.333 | 0.333 |
| 15 | 128 | RIGHT | True | 3 | 0.667 | 0.667 | 0.667 |
| 15 | 128 | UP | False | 4 | -0.500 | 0.500 | 0.500 |
| 15 | 128 | UP | True | 3 | -0.333 | -0.333 | 0.000 |

## Direction-Control Advantage

| layer | alpha | desired direction | desired delta | best wrong direction | best wrong delta | desired advantage | supports desired direction |
|---:|---:|---|---:|---|---:|---:|---|
| 15 | -128 | RIGHT | 0.333 | DOWN | 0.333 | -0.000 | False |
| 15 | -64 | LEFT | 0.000 | UP | 0.250 | -0.250 | False |
| 15 | -32 | LEFT | 0.333 | LEFT | 0.250 | 0.083 | True |
| 15 | -16 | LEFT | 0.333 | DOWN | 0.111 | 0.222 | True |
| 15 | -8 | DOWN | 0.000 | RIGHT | 0.333 | -0.333 | False |
| 15 | 8 | DOWN | 0.000 | DOWN | 0.000 | 0.000 | False |
| 15 | 16 | LEFT | 0.333 | UP | 0.250 | 0.083 | True |
| 15 | 32 | DOWN | 0.000 | LEFT | 0.000 | 0.000 | False |
| 15 | 64 | LEFT | 0.333 | DOWN | 0.111 | 0.222 | True |
| 15 | 128 | RIGHT | 0.667 | UP | 0.500 | 0.167 | True |

## Overall Direction Controls

| layer | alpha | direction group | pairs | desired-action delta | nonoptimal-to-desired rate | desired-to-nondesired rate |
|---:|---:|---|---:|---:|---:|---:|
| 15 | -128 | wrong | 24 | 0.125 | 0.125 | 0.000 |
| 15 | -128 | desired | 12 | 0.000 | 0.083 | 0.083 |
| 15 | -64 | wrong | 24 | 0.125 | 0.208 | 0.083 |
| 15 | -64 | desired | 12 | 0.000 | 0.083 | 0.083 |
| 15 | -32 | wrong | 24 | 0.125 | 0.167 | 0.042 |
| 15 | -32 | desired | 12 | 0.000 | 0.167 | 0.167 |
| 15 | -16 | wrong | 24 | 0.000 | 0.042 | 0.042 |
| 15 | -16 | desired | 12 | 0.083 | 0.083 | 0.000 |
| 15 | -8 | wrong | 24 | 0.042 | 0.083 | 0.042 |
| 15 | -8 | desired | 12 | -0.083 | 0.000 | 0.083 |
| 15 | 8 | wrong | 24 | 0.000 | 0.000 | 0.000 |
| 15 | 8 | desired | 12 | 0.000 | 0.000 | 0.000 |
| 15 | 16 | wrong | 24 | 0.042 | 0.042 | 0.000 |
| 15 | 16 | desired | 12 | 0.083 | 0.083 | 0.000 |
| 15 | 32 | wrong | 24 | -0.083 | 0.000 | 0.083 |
| 15 | 32 | desired | 12 | 0.000 | 0.000 | 0.000 |
| 15 | 64 | wrong | 24 | 0.042 | 0.042 | 0.000 |
| 15 | 64 | desired | 12 | 0.083 | 0.083 | 0.000 |
| 15 | 128 | wrong | 24 | 0.125 | 0.167 | 0.042 |
| 15 | 128 | desired | 12 | 0.083 | 0.167 | 0.083 |

## Overall Direction-Control Advantage

| layer | alpha | desired delta | wrong delta | desired advantage | supports desired direction |
|---:|---:|---:|---:|---:|---|
| 15 | -128 | 0.000 | 0.125 | -0.125 | False |
| 15 | -64 | 0.000 | 0.125 | -0.125 | False |
| 15 | -32 | 0.000 | 0.125 | -0.125 | False |
| 15 | -16 | 0.083 | 0.000 | 0.083 | True |
| 15 | -8 | -0.083 | 0.042 | -0.125 | False |
| 15 | 8 | 0.000 | 0.000 | 0.000 | False |
| 15 | 16 | 0.083 | 0.042 | 0.042 | True |
| 15 | 32 | 0.000 | -0.083 | 0.083 | False |
| 15 | 64 | 0.083 | 0.042 | 0.042 | True |
| 15 | 128 | 0.083 | 0.125 | -0.042 | False |

## Interpretation

The largest optimal-action improvement versus no-hook is layer 15 at alpha 128: optimal-action rate 0.528, delta 0.111.

The largest true-action retention drop versus no-hook is layer 15 at alpha -64: true-action rate 0.139, delta -0.111.

The strongest paired causal effect on emitted action is layer 15 at alpha -32: 0.500 of paired samples changed final action under the same generation seed.

The strongest paired effect away from the recorded true action is layer 15 at alpha -64: 0.167 of pairs changed from true-action emission to a different final action.

The strongest paired nonoptimal-to-optimal conversion is layer 15 at alpha 128: 0.167 of pairs converted from a nonoptimal emitted action to an action in `astar_actions`.

The per-action same-alpha direction-control check supports desired-direction steering at its strongest tested setting: layer 15 alpha -16, desired direction LEFT delta 0.333, best wrong direction DOWN delta 0.111, advantage 0.222.

The overall desired-vs-wrong control is positive at its strongest tested setting: layer 15 alpha -16, desired-direction delta 0.083, wrong-direction delta 0.000, advantage 0.083.

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
