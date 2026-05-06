# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_raw_action_vs_true_layers7_23_balanced12_alpha_screen.jsonl`
- Layers: 7
- Hooked alphas: -128, -64, -32, -16, -8, 8, 16, 32, 64, 128

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 7 | no_hook |  | 12 | 0.583 | 0.250 | 0.583 | 0.000 | 0.000 |
| 7 | hooked | -128 | 36 | 0.528 | 0.306 | 0.528 | -0.056 | 0.056 |
| 7 | hooked | -64 | 36 | 0.417 | 0.333 | 0.417 | -0.167 | 0.083 |
| 7 | hooked | -32 | 36 | 0.472 | 0.306 | 0.472 | -0.111 | 0.056 |
| 7 | hooked | -16 | 36 | 0.306 | 0.361 | 0.306 | -0.278 | 0.111 |
| 7 | hooked | -8 | 36 | 0.472 | 0.250 | 0.472 | -0.111 | 0.000 |
| 7 | hooked | 8 | 36 | 0.500 | 0.222 | 0.500 | -0.083 | -0.028 |
| 7 | hooked | 16 | 36 | 0.278 | 0.417 | 0.278 | -0.306 | 0.167 |
| 7 | hooked | 32 | 36 | 0.444 | 0.250 | 0.444 | -0.139 | 0.000 |
| 7 | hooked | 64 | 36 | 0.444 | 0.361 | 0.444 | -0.139 | 0.111 |
| 7 | hooked | 128 | 36 | 0.472 | 0.333 | 0.472 | -0.111 | 0.083 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 7 | 0 | 0.000 | 0.000 | 0.000 | 12 |

## Paired Causal Effects

| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | optimal-to-nonoptimal rate | true-to-not-true rate |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | -128 | 36 | 0.556 | 0.167 | 0.222 | 0.167 |
| 7 | -64 | 36 | 0.556 | 0.111 | 0.278 | 0.111 |
| 7 | -32 | 36 | 0.556 | 0.139 | 0.250 | 0.139 |
| 7 | -16 | 36 | 0.556 | 0.083 | 0.361 | 0.083 |
| 7 | -8 | 36 | 0.556 | 0.111 | 0.222 | 0.167 |
| 7 | 8 | 36 | 0.444 | 0.111 | 0.194 | 0.167 |
| 7 | 16 | 36 | 0.500 | 0.028 | 0.333 | 0.056 |
| 7 | 32 | 36 | 0.417 | 0.083 | 0.222 | 0.111 |
| 7 | 64 | 36 | 0.444 | 0.083 | 0.222 | 0.083 |
| 7 | 128 | 36 | 0.444 | 0.111 | 0.222 | 0.111 |

## Direction Controls

| layer | alpha | direction | desired dir | pairs | steered-action delta | desired-action delta | nonoptimal-to-desired rate |
|---:|---:|---|---|---:|---:|---:|---:|
| 7 | -128 | DOWN | False | 9 | 0.000 | -0.111 | 0.111 |
| 7 | -128 | DOWN | True | 3 | 0.333 | 0.333 | 0.333 |
| 7 | -128 | LEFT | False | 8 | -0.250 | 0.250 | 0.250 |
| 7 | -128 | LEFT | True | 3 | -0.667 | -0.667 | 0.000 |
| 7 | -128 | RIGHT | False | 3 | 0.000 | 0.000 | 0.000 |
| 7 | -128 | RIGHT | True | 3 | 0.333 | 0.333 | 0.333 |
| 7 | -128 | UP | False | 4 | 0.000 | 0.000 | 0.250 |
| 7 | -128 | UP | True | 3 | -1.000 | -1.000 | 0.000 |
| 7 | -64 | DOWN | False | 9 | 0.000 | -0.333 | 0.111 |
| 7 | -64 | DOWN | True | 3 | 0.333 | 0.333 | 0.333 |
| 7 | -64 | LEFT | False | 8 | -0.250 | 0.000 | 0.125 |
| 7 | -64 | LEFT | True | 3 | -0.333 | -0.333 | 0.000 |
| 7 | -64 | RIGHT | False | 3 | 0.333 | 0.000 | 0.000 |
| 7 | -64 | RIGHT | True | 3 | -0.333 | -0.333 | 0.000 |
| 7 | -64 | UP | False | 4 | 0.500 | -0.500 | 0.250 |
| 7 | -64 | UP | True | 3 | 0.000 | 0.000 | 0.000 |
| 7 | -32 | DOWN | False | 9 | 0.000 | -0.222 | 0.000 |
| 7 | -32 | DOWN | True | 3 | 0.333 | 0.333 | 0.333 |
| 7 | -32 | LEFT | False | 8 | -0.250 | 0.125 | 0.250 |
| 7 | -32 | LEFT | True | 3 | -0.667 | -0.667 | 0.000 |
| 7 | -32 | RIGHT | False | 3 | 0.000 | 0.000 | 0.000 |
| 7 | -32 | RIGHT | True | 3 | 0.000 | 0.000 | 0.333 |
| 7 | -32 | UP | False | 4 | 0.500 | -0.250 | 0.250 |
| 7 | -32 | UP | True | 3 | -0.333 | -0.333 | 0.000 |
| 7 | -16 | DOWN | False | 9 | 0.000 | -0.556 | 0.000 |
| 7 | -16 | DOWN | True | 3 | 0.333 | 0.333 | 0.333 |
| 7 | -16 | LEFT | False | 8 | -0.250 | 0.000 | 0.125 |
| 7 | -16 | LEFT | True | 3 | -0.667 | -0.667 | 0.000 |
| 7 | -16 | RIGHT | False | 3 | 0.000 | 0.000 | 0.000 |
| 7 | -16 | RIGHT | True | 3 | -0.333 | -0.333 | 0.000 |
| 7 | -16 | UP | False | 4 | 0.750 | -0.500 | 0.250 |
| 7 | -16 | UP | True | 3 | -0.333 | -0.333 | 0.000 |
| 7 | -8 | DOWN | False | 9 | 0.000 | -0.111 | 0.111 |
| 7 | -8 | DOWN | True | 3 | 0.333 | 0.333 | 0.333 |
| 7 | -8 | LEFT | False | 8 | -0.125 | 0.125 | 0.125 |
| 7 | -8 | LEFT | True | 3 | -0.667 | -0.667 | 0.000 |
| 7 | -8 | RIGHT | False | 3 | 0.333 | -0.333 | 0.000 |
| 7 | -8 | RIGHT | True | 3 | -0.333 | -0.333 | 0.000 |
| 7 | -8 | UP | False | 4 | 0.250 | -0.250 | 0.250 |
| 7 | -8 | UP | True | 3 | 0.000 | 0.000 | 0.000 |
| 7 | 8 | DOWN | False | 9 | 0.000 | -0.222 | 0.000 |
| 7 | 8 | DOWN | True | 3 | 0.333 | 0.333 | 0.333 |
| 7 | 8 | LEFT | False | 8 | -0.125 | 0.250 | 0.250 |
| 7 | 8 | LEFT | True | 3 | -0.333 | -0.333 | 0.000 |
| 7 | 8 | RIGHT | False | 3 | 0.000 | 0.000 | 0.000 |
| 7 | 8 | RIGHT | True | 3 | 0.000 | 0.000 | 0.333 |
| 7 | 8 | UP | False | 4 | 0.250 | -0.500 | 0.000 |
| 7 | 8 | UP | True | 3 | -0.333 | -0.333 | 0.000 |
| 7 | 16 | DOWN | False | 9 | 0.000 | -0.556 | 0.000 |
| 7 | 16 | DOWN | True | 3 | 0.000 | 0.000 | 0.000 |
| 7 | 16 | LEFT | False | 8 | -0.250 | -0.125 | 0.000 |
| 7 | 16 | LEFT | True | 3 | -0.667 | -0.667 | 0.000 |
| 7 | 16 | RIGHT | False | 3 | 0.333 | 0.000 | 0.000 |
| 7 | 16 | RIGHT | True | 3 | 0.000 | 0.000 | 0.000 |
| 7 | 16 | UP | False | 4 | 0.750 | -0.500 | 0.250 |
| 7 | 16 | UP | True | 3 | -0.333 | -0.333 | 0.000 |
| 7 | 32 | DOWN | False | 9 | 0.000 | -0.222 | 0.000 |
| 7 | 32 | DOWN | True | 3 | 0.333 | 0.333 | 0.333 |
| 7 | 32 | LEFT | False | 8 | -0.125 | 0.125 | 0.125 |
| 7 | 32 | LEFT | True | 3 | -0.333 | -0.333 | 0.000 |
| 7 | 32 | RIGHT | False | 3 | 0.000 | 0.000 | 0.000 |
| 7 | 32 | RIGHT | True | 3 | 0.000 | 0.000 | 0.333 |
| 7 | 32 | UP | False | 4 | 0.750 | -0.750 | 0.000 |
| 7 | 32 | UP | True | 3 | -0.333 | -0.333 | 0.000 |
| 7 | 64 | DOWN | False | 9 | 0.000 | -0.222 | 0.111 |
| 7 | 64 | DOWN | True | 3 | 0.333 | 0.333 | 0.333 |
| 7 | 64 | LEFT | False | 8 | -0.250 | 0.000 | 0.000 |
| 7 | 64 | LEFT | True | 3 | -0.667 | -0.667 | 0.000 |
| 7 | 64 | RIGHT | False | 3 | 0.000 | 0.000 | 0.000 |
| 7 | 64 | RIGHT | True | 3 | 0.000 | 0.000 | 0.333 |
| 7 | 64 | UP | False | 4 | 0.500 | -0.500 | 0.000 |
| 7 | 64 | UP | True | 3 | 0.000 | 0.000 | 0.000 |
| 7 | 128 | DOWN | False | 9 | 0.000 | -0.333 | 0.000 |
| 7 | 128 | DOWN | True | 3 | 0.333 | 0.333 | 0.333 |
| 7 | 128 | LEFT | False | 8 | -0.125 | 0.125 | 0.125 |
| 7 | 128 | LEFT | True | 3 | -0.333 | -0.333 | 0.000 |
| 7 | 128 | RIGHT | False | 3 | 0.000 | 0.000 | 0.000 |
| 7 | 128 | RIGHT | True | 3 | 0.000 | 0.000 | 0.333 |
| 7 | 128 | UP | False | 4 | 0.250 | 0.000 | 0.250 |
| 7 | 128 | UP | True | 3 | -0.667 | -0.667 | 0.000 |

## Direction-Control Advantage

| layer | alpha | desired direction | desired delta | best wrong direction | best wrong delta | desired advantage | supports desired direction |
|---:|---:|---|---:|---|---:|---:|---|
| 7 | -128 | DOWN | 0.333 | LEFT | 0.250 | 0.083 | True |
| 7 | -64 | DOWN | 0.333 | LEFT | 0.000 | 0.333 | True |
| 7 | -32 | DOWN | 0.333 | LEFT | 0.125 | 0.208 | True |
| 7 | -16 | DOWN | 0.333 | LEFT | 0.000 | 0.333 | True |
| 7 | -8 | DOWN | 0.333 | LEFT | 0.125 | 0.208 | True |
| 7 | 8 | DOWN | 0.333 | LEFT | 0.250 | 0.083 | True |
| 7 | 16 | DOWN | 0.000 | RIGHT | 0.000 | 0.000 | False |
| 7 | 32 | DOWN | 0.333 | LEFT | 0.125 | 0.208 | True |
| 7 | 64 | DOWN | 0.333 | LEFT | 0.000 | 0.333 | True |
| 7 | 128 | DOWN | 0.333 | LEFT | 0.125 | 0.208 | True |

## Overall Direction Controls

| layer | alpha | direction group | pairs | desired-action delta | nonoptimal-to-desired rate | desired-to-nondesired rate |
|---:|---:|---|---:|---:|---:|---:|
| 7 | -128 | wrong | 24 | 0.042 | 0.167 | 0.125 |
| 7 | -128 | desired | 12 | -0.250 | 0.167 | 0.417 |
| 7 | -64 | wrong | 24 | -0.208 | 0.125 | 0.333 |
| 7 | -64 | desired | 12 | -0.083 | 0.083 | 0.167 |
| 7 | -32 | wrong | 24 | -0.083 | 0.125 | 0.208 |
| 7 | -32 | desired | 12 | -0.167 | 0.167 | 0.333 |
| 7 | -16 | wrong | 24 | -0.292 | 0.083 | 0.375 |
| 7 | -16 | desired | 12 | -0.250 | 0.083 | 0.333 |
| 7 | -8 | wrong | 24 | -0.083 | 0.125 | 0.208 |
| 7 | -8 | desired | 12 | -0.167 | 0.083 | 0.250 |
| 7 | 8 | wrong | 24 | -0.083 | 0.083 | 0.167 |
| 7 | 8 | desired | 12 | -0.083 | 0.167 | 0.250 |
| 7 | 16 | wrong | 24 | -0.333 | 0.042 | 0.375 |
| 7 | 16 | desired | 12 | -0.250 | 0.000 | 0.250 |
| 7 | 32 | wrong | 24 | -0.167 | 0.042 | 0.208 |
| 7 | 32 | desired | 12 | -0.083 | 0.167 | 0.250 |
| 7 | 64 | wrong | 24 | -0.167 | 0.042 | 0.208 |
| 7 | 64 | desired | 12 | -0.083 | 0.167 | 0.250 |
| 7 | 128 | wrong | 24 | -0.083 | 0.083 | 0.167 |
| 7 | 128 | desired | 12 | -0.167 | 0.167 | 0.333 |

## Overall Direction-Control Advantage

| layer | alpha | desired delta | wrong delta | desired advantage | supports desired direction |
|---:|---:|---:|---:|---:|---|
| 7 | -128 | -0.250 | 0.042 | -0.292 | False |
| 7 | -64 | -0.083 | -0.208 | 0.125 | False |
| 7 | -32 | -0.167 | -0.083 | -0.083 | False |
| 7 | -16 | -0.250 | -0.292 | 0.042 | False |
| 7 | -8 | -0.167 | -0.083 | -0.083 | False |
| 7 | 8 | -0.083 | -0.083 | 0.000 | False |
| 7 | 16 | -0.250 | -0.333 | 0.083 | False |
| 7 | 32 | -0.083 | -0.167 | 0.083 | False |
| 7 | 64 | -0.083 | -0.167 | 0.083 | False |
| 7 | 128 | -0.167 | -0.083 | -0.083 | False |

## Interpretation

No hooked condition improved optimal-action rate versus no-hook. The least-negative condition is layer 7 at alpha -128: optimal-action rate 0.528, delta -0.056.

The largest true-action retention drop versus no-hook is layer 7 at alpha 8: true-action rate 0.222, delta -0.028.

The strongest paired causal effect on emitted action is layer 7 at alpha -128: 0.556 of paired samples changed final action under the same generation seed.

The strongest paired effect away from the recorded true action is layer 7 at alpha -128: 0.167 of pairs changed from true-action emission to a different final action.

The strongest paired nonoptimal-to-optimal conversion is layer 7 at alpha -128: 0.167 of pairs converted from a nonoptimal emitted action to an action in `astar_actions`.

The per-action direction-control table contains a local positive slice, but the overall desired-vs-wrong control is negative. The strongest local slice is layer 7 alpha 64, desired direction DOWN delta 0.333, best wrong direction LEFT delta 0.000, advantage 0.333.

The overall desired-vs-wrong control is not positive. The best tested setting is layer 7 alpha -64, desired-direction delta -0.083, wrong-direction delta -0.208, advantage 0.125.

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
