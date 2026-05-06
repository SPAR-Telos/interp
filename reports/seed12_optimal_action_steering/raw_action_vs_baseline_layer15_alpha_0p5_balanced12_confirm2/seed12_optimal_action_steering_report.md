# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_raw_action_vs_baseline_layer15_alpha_0p5_balanced12_confirm2.jsonl`
- Layers: 15
- Hooked alphas: 0.5

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | no_hook |  | 120 | 0.425 | 0.292 | 0.425 | 0.000 | 0.000 |
| 15 | hooked | 0.5 | 207 | 0.285 | 0.357 | 0.285 | -0.140 | 0.066 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 15 | 0 | 0.000 | 0.000 | 0.000 | 120 |

## Paired Causal Effects

| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | optimal-to-nonoptimal rate | true-to-not-true rate |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.5 | 207 | 0.440 | 0.285 | 0.000 | 0.213 |

## Direction Controls

| layer | alpha | direction | desired dir | pairs | steered-action delta | desired-action delta | nonoptimal-to-desired rate |
|---:|---:|---|---|---:|---:|---:|---:|
| 15 | 0.5 | DOWN | False | 45 | 0.000 | 0.289 | 0.289 |
| 15 | 0.5 | DOWN | True | 24 | 0.208 | 0.208 | 0.208 |
| 15 | 0.5 | LEFT | False | 40 | 0.150 | 0.225 | 0.225 |
| 15 | 0.5 | LEFT | True | 21 | 0.571 | 0.571 | 0.571 |
| 15 | 0.5 | RIGHT | False | 34 | 0.088 | 0.176 | 0.176 |
| 15 | 0.5 | RIGHT | True | 21 | 0.381 | 0.381 | 0.381 |
| 15 | 0.5 | UP | False | 19 | 0.316 | 0.211 | 0.211 |
| 15 | 0.5 | UP | True | 3 | 0.667 | 0.667 | 0.667 |

## Direction-Control Advantage

| layer | alpha | desired direction | desired delta | best wrong direction | best wrong delta | desired advantage | supports desired direction |
|---:|---:|---|---:|---|---:|---:|---|
| 15 | 0.5 | UP | 0.667 | DOWN | 0.289 | 0.378 | True |

## Overall Direction Controls

| layer | alpha | direction group | pairs | desired-action delta | nonoptimal-to-desired rate | desired-to-nondesired rate |
|---:|---:|---|---:|---:|---:|---:|
| 15 | 0.5 | wrong | 138 | 0.232 | 0.232 | 0.000 |
| 15 | 0.5 | desired | 69 | 0.391 | 0.391 | 0.000 |

## Overall Direction-Control Advantage

| layer | alpha | desired delta | wrong delta | desired advantage | supports desired direction |
|---:|---:|---:|---:|---:|---|
| 15 | 0.5 | 0.391 | 0.232 | 0.159 | True |

## Interpretation

No hooked condition improved optimal-action rate versus no-hook. The least-negative condition is layer 15 at alpha 0.5: optimal-action rate 0.285, delta -0.140.

No hooked condition reduced true-action retention versus no-hook. The smallest true-action-rate delta is layer 15 at alpha 0.5: true-action rate 0.357, delta 0.066.

The strongest paired causal effect on emitted action is layer 15 at alpha 0.5: 0.440 of paired samples changed final action under the same generation seed.

The strongest paired effect away from the recorded true action is layer 15 at alpha 0.5: 0.213 of pairs changed from true-action emission to a different final action.

The strongest paired nonoptimal-to-optimal conversion is layer 15 at alpha 0.5: 0.285 of pairs converted from a nonoptimal emitted action to an action in `astar_actions`.

The per-action same-alpha direction-control check supports desired-direction steering at its strongest tested setting: layer 15 alpha 0.5, desired direction UP delta 0.667, best wrong direction DOWN delta 0.289, advantage 0.378.

The overall desired-vs-wrong control is strictly positive at its strongest tested setting: layer 15 alpha 0.5, desired-direction delta 0.391, wrong-direction delta 0.232, advantage 0.159, nonoptimal-to-desired advantage 0.159.

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
