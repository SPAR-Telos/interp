# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_raw_action_vs_baseline_layer15_balanced12_alpha_screen.jsonl`
- Layers: 15
- Hooked alphas: 0.25, 0.5, 1, 2, 4, 8, 16

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | no_hook |  | 12 | 0.667 | 0.167 | 0.667 | 0.000 | 0.000 |
| 15 | hooked | 0.25 | 12 | 0.000 | 0.417 | 0.000 | -0.667 | 0.250 |
| 15 | hooked | 0.5 | 12 | 0.167 | 0.250 | 0.167 | -0.500 | 0.083 |
| 15 | hooked | 1 | 12 | 0.083 | 0.417 | 0.083 | -0.583 | 0.250 |
| 15 | hooked | 2 | 12 | 0.083 | 0.417 | 0.083 | -0.583 | 0.250 |
| 15 | hooked | 4 | 12 | 0.250 | 0.250 | 0.250 | -0.417 | 0.083 |
| 15 | hooked | 8 | 12 | 0.000 | 0.750 | 0.000 | -0.667 | 0.583 |
| 15 | hooked | 16 | 12 | 0.000 | 0.417 | 0.000 | -0.667 | 0.250 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 15 | 0 | 0.000 | 0.000 | 0.000 | 12 |

## Paired Causal Effects

| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | optimal-to-nonoptimal rate | true-to-not-true rate |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.25 | 12 | 0.250 | 0.000 | 0.000 | 0.167 |
| 15 | 0.5 | 12 | 0.333 | 0.167 | 0.000 | 0.250 |
| 15 | 1 | 12 | 0.250 | 0.083 | 0.000 | 0.167 |
| 15 | 2 | 12 | 0.417 | 0.083 | 0.000 | 0.250 |
| 15 | 4 | 12 | 0.417 | 0.250 | 0.000 | 0.250 |
| 15 | 8 | 12 | 0.417 | 0.000 | 0.000 | 0.083 |
| 15 | 16 | 12 | 0.250 | 0.000 | 0.000 | 0.167 |

## Direction Controls

| layer | alpha | direction | desired dir | pairs | steered-action delta | desired-action delta | nonoptimal-to-desired rate |
|---:|---:|---|---|---:|---:|---:|---:|
| 15 | 0.25 | DOWN | False | 2 | 0.000 | 0.000 | 0.000 |
| 15 | 0.25 | DOWN | True | 2 | 0.000 | 0.000 | 0.000 |
| 15 | 0.25 | LEFT | False | 2 | 0.000 | 0.000 | 0.000 |
| 15 | 0.25 | LEFT | True | 1 | 0.000 | 0.000 | 0.000 |
| 15 | 0.25 | RIGHT | False | 3 | 0.333 | 0.000 | 0.000 |
| 15 | 0.25 | RIGHT | True | 1 | 0.000 | 0.000 | 0.000 |
| 15 | 0.25 | UP | False | 1 | 0.000 | 0.000 | 0.000 |
| 15 | 0.5 | DOWN | False | 2 | 0.000 | 0.000 | 0.000 |
| 15 | 0.5 | DOWN | True | 2 | 0.500 | 0.500 | 0.500 |
| 15 | 0.5 | LEFT | False | 2 | 0.500 | 0.000 | 0.000 |
| 15 | 0.5 | LEFT | True | 1 | 1.000 | 1.000 | 1.000 |
| 15 | 0.5 | RIGHT | False | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 0.5 | RIGHT | True | 1 | 0.000 | 0.000 | 0.000 |
| 15 | 0.5 | UP | False | 1 | 0.000 | 0.000 | 0.000 |
| 15 | 1 | DOWN | False | 2 | 0.000 | 0.000 | 0.000 |
| 15 | 1 | DOWN | True | 2 | 0.000 | 0.000 | 0.000 |
| 15 | 1 | LEFT | False | 2 | 0.000 | 0.500 | 0.500 |
| 15 | 1 | LEFT | True | 1 | 0.000 | 0.000 | 0.000 |
| 15 | 1 | RIGHT | False | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 1 | RIGHT | True | 1 | 0.000 | 0.000 | 0.000 |
| 15 | 1 | UP | False | 1 | 1.000 | 0.000 | 0.000 |
| 15 | 2 | DOWN | False | 2 | 0.000 | 0.000 | 0.000 |
| 15 | 2 | DOWN | True | 2 | 0.500 | 0.500 | 0.500 |
| 15 | 2 | LEFT | False | 2 | 0.500 | 0.000 | 0.000 |
| 15 | 2 | LEFT | True | 1 | 0.000 | 0.000 | 0.000 |
| 15 | 2 | RIGHT | False | 3 | 0.333 | 0.000 | 0.000 |
| 15 | 2 | RIGHT | True | 1 | 0.000 | 0.000 | 0.000 |
| 15 | 2 | UP | False | 1 | 1.000 | 0.000 | 0.000 |
| 15 | 4 | DOWN | False | 2 | 0.000 | 0.500 | 0.500 |
| 15 | 4 | DOWN | True | 2 | 0.500 | 0.500 | 0.500 |
| 15 | 4 | LEFT | False | 2 | 0.000 | 0.000 | 0.000 |
| 15 | 4 | LEFT | True | 1 | 0.000 | 0.000 | 0.000 |
| 15 | 4 | RIGHT | False | 3 | 0.000 | 0.333 | 0.333 |
| 15 | 4 | RIGHT | True | 1 | 0.000 | 0.000 | 0.000 |
| 15 | 4 | UP | False | 1 | 0.000 | 0.000 | 0.000 |
| 15 | 8 | DOWN | False | 2 | 0.000 | 0.000 | 0.000 |
| 15 | 8 | DOWN | True | 2 | 0.000 | 0.000 | 0.000 |
| 15 | 8 | LEFT | False | 2 | 0.000 | 0.000 | 0.000 |
| 15 | 8 | LEFT | True | 1 | 0.000 | 0.000 | 0.000 |
| 15 | 8 | RIGHT | False | 3 | 0.333 | 0.000 | 0.000 |
| 15 | 8 | RIGHT | True | 1 | 0.000 | 0.000 | 0.000 |
| 15 | 8 | UP | False | 1 | 1.000 | 0.000 | 0.000 |
| 15 | 16 | DOWN | False | 2 | 0.000 | 0.000 | 0.000 |
| 15 | 16 | DOWN | True | 2 | 0.000 | 0.000 | 0.000 |
| 15 | 16 | LEFT | False | 2 | 0.500 | 0.000 | 0.000 |
| 15 | 16 | LEFT | True | 1 | 0.000 | 0.000 | 0.000 |
| 15 | 16 | RIGHT | False | 3 | 0.000 | 0.000 | 0.000 |
| 15 | 16 | RIGHT | True | 1 | 0.000 | 0.000 | 0.000 |
| 15 | 16 | UP | False | 1 | 0.000 | 0.000 | 0.000 |

## Direction-Control Advantage

| layer | alpha | desired direction | desired delta | best wrong direction | best wrong delta | desired advantage | supports desired direction |
|---:|---:|---|---:|---|---:|---:|---|
| 15 | 0.25 | DOWN | 0.000 | DOWN | 0.000 | 0.000 | False |
| 15 | 0.5 | LEFT | 1.000 | DOWN | 0.000 | 1.000 | True |
| 15 | 1 | DOWN | 0.000 | LEFT | 0.500 | -0.500 | False |
| 15 | 2 | DOWN | 0.500 | DOWN | 0.000 | 0.500 | True |
| 15 | 4 | DOWN | 0.500 | DOWN | 0.500 | 0.000 | False |
| 15 | 8 | DOWN | 0.000 | DOWN | 0.000 | 0.000 | False |
| 15 | 16 | DOWN | 0.000 | DOWN | 0.000 | 0.000 | False |

## Overall Direction Controls

| layer | alpha | direction group | pairs | desired-action delta | nonoptimal-to-desired rate | desired-to-nondesired rate |
|---:|---:|---|---:|---:|---:|---:|
| 15 | 0.25 | wrong | 8 | 0.000 | 0.000 | 0.000 |
| 15 | 0.25 | desired | 4 | 0.000 | 0.000 | 0.000 |
| 15 | 0.5 | wrong | 8 | 0.000 | 0.000 | 0.000 |
| 15 | 0.5 | desired | 4 | 0.500 | 0.500 | 0.000 |
| 15 | 1 | wrong | 8 | 0.125 | 0.125 | 0.000 |
| 15 | 1 | desired | 4 | 0.000 | 0.000 | 0.000 |
| 15 | 2 | wrong | 8 | 0.000 | 0.000 | 0.000 |
| 15 | 2 | desired | 4 | 0.250 | 0.250 | 0.000 |
| 15 | 4 | wrong | 8 | 0.250 | 0.250 | 0.000 |
| 15 | 4 | desired | 4 | 0.250 | 0.250 | 0.000 |
| 15 | 8 | wrong | 8 | 0.000 | 0.000 | 0.000 |
| 15 | 8 | desired | 4 | 0.000 | 0.000 | 0.000 |
| 15 | 16 | wrong | 8 | 0.000 | 0.000 | 0.000 |
| 15 | 16 | desired | 4 | 0.000 | 0.000 | 0.000 |

## Overall Direction-Control Advantage

| layer | alpha | desired delta | wrong delta | desired advantage | supports desired direction |
|---:|---:|---:|---:|---:|---|
| 15 | 0.25 | 0.000 | 0.000 | 0.000 | False |
| 15 | 0.5 | 0.500 | 0.000 | 0.500 | True |
| 15 | 1 | 0.000 | 0.125 | -0.125 | False |
| 15 | 2 | 0.250 | 0.000 | 0.250 | True |
| 15 | 4 | 0.250 | 0.250 | 0.000 | False |
| 15 | 8 | 0.000 | 0.000 | 0.000 | False |
| 15 | 16 | 0.000 | 0.000 | 0.000 | False |

## Interpretation

No hooked condition improved optimal-action rate versus no-hook. The least-negative condition is layer 15 at alpha 4: optimal-action rate 0.250, delta -0.417.

No hooked condition reduced true-action retention versus no-hook. The smallest true-action-rate delta is layer 15 at alpha 0.5: true-action rate 0.250, delta 0.083.

The strongest paired causal effect on emitted action is layer 15 at alpha 8: 0.417 of paired samples changed final action under the same generation seed.

The strongest paired effect away from the recorded true action is layer 15 at alpha 4: 0.250 of pairs changed from true-action emission to a different final action.

The strongest paired nonoptimal-to-optimal conversion is layer 15 at alpha 4: 0.250 of pairs converted from a nonoptimal emitted action to an action in `astar_actions`.

The per-action same-alpha direction-control check supports desired-direction steering at its strongest tested setting: layer 15 alpha 0.5, desired direction LEFT delta 1.000, best wrong direction DOWN delta 0.000, advantage 1.000.

The overall desired-vs-wrong control is strictly positive at its strongest tested setting: layer 15 alpha 0.5, desired-direction delta 0.500, wrong-direction delta 0.000, advantage 0.500, nonoptimal-to-desired advantage 0.500.

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
