# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_raw_action_vs_baseline_layer15_alpha_0p5_2_balanced12_confirm.jsonl`
- Layers: 15
- Hooked alphas: 0.5, 2

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | no_hook |  | 60 | 0.567 | 0.183 | 0.567 | 0.000 | 0.000 |
| 15 | hooked | 0.5 | 78 | 0.218 | 0.308 | 0.218 | -0.349 | 0.124 |
| 15 | hooked | 2 | 78 | 0.231 | 0.333 | 0.231 | -0.336 | 0.150 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 15 | 0 | 0.000 | 0.000 | 0.000 | 60 |

## Paired Causal Effects

| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | optimal-to-nonoptimal rate | true-to-not-true rate |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.5 | 78 | 0.590 | 0.218 | 0.000 | 0.282 |
| 15 | 2 | 78 | 0.615 | 0.231 | 0.000 | 0.282 |

## Direction Controls

| layer | alpha | direction | desired dir | pairs | steered-action delta | desired-action delta | nonoptimal-to-desired rate |
|---:|---:|---|---|---:|---:|---:|---:|
| 15 | 0.5 | DOWN | False | 15 | 0.000 | 0.333 | 0.333 |
| 15 | 0.5 | DOWN | True | 11 | 0.000 | 0.000 | 0.000 |
| 15 | 0.5 | LEFT | False | 15 | 0.200 | 0.267 | 0.267 |
| 15 | 0.5 | LEFT | True | 6 | 0.333 | 0.333 | 0.333 |
| 15 | 0.5 | RIGHT | False | 11 | 0.182 | 0.091 | 0.091 |
| 15 | 0.5 | RIGHT | True | 8 | 0.375 | 0.375 | 0.375 |
| 15 | 0.5 | UP | False | 11 | 0.364 | 0.091 | 0.091 |
| 15 | 0.5 | UP | True | 1 | 1.000 | 1.000 | 1.000 |
| 15 | 2 | DOWN | False | 15 | 0.067 | 0.400 | 0.400 |
| 15 | 2 | DOWN | True | 11 | 0.091 | 0.091 | 0.091 |
| 15 | 2 | LEFT | False | 15 | 0.133 | 0.133 | 0.133 |
| 15 | 2 | LEFT | True | 6 | 0.500 | 0.500 | 0.500 |
| 15 | 2 | RIGHT | False | 11 | 0.182 | 0.091 | 0.091 |
| 15 | 2 | RIGHT | True | 8 | 0.125 | 0.125 | 0.125 |
| 15 | 2 | UP | False | 11 | 0.545 | 0.273 | 0.273 |
| 15 | 2 | UP | True | 1 | 1.000 | 1.000 | 1.000 |

## Direction-Control Advantage

| layer | alpha | desired direction | desired delta | best wrong direction | best wrong delta | desired advantage | supports desired direction |
|---:|---:|---|---:|---|---:|---:|---|
| 15 | 0.5 | UP | 1.000 | DOWN | 0.333 | 0.667 | True |
| 15 | 2 | UP | 1.000 | DOWN | 0.400 | 0.600 | True |

## Overall Direction Controls

| layer | alpha | direction group | pairs | desired-action delta | nonoptimal-to-desired rate | desired-to-nondesired rate |
|---:|---:|---|---:|---:|---:|---:|
| 15 | 0.5 | wrong | 52 | 0.212 | 0.212 | 0.000 |
| 15 | 0.5 | desired | 26 | 0.231 | 0.231 | 0.000 |
| 15 | 2 | wrong | 52 | 0.231 | 0.231 | 0.000 |
| 15 | 2 | desired | 26 | 0.231 | 0.231 | 0.000 |

## Overall Direction-Control Advantage

| layer | alpha | desired delta | wrong delta | desired advantage | supports desired direction |
|---:|---:|---:|---:|---:|---|
| 15 | 0.5 | 0.231 | 0.212 | 0.019 | True |
| 15 | 2 | 0.231 | 0.231 | 0.000 | False |

## Interpretation

No hooked condition improved optimal-action rate versus no-hook. The least-negative condition is layer 15 at alpha 2: optimal-action rate 0.231, delta -0.336.

No hooked condition reduced true-action retention versus no-hook. The smallest true-action-rate delta is layer 15 at alpha 0.5: true-action rate 0.308, delta 0.124.

The strongest paired causal effect on emitted action is layer 15 at alpha 2: 0.615 of paired samples changed final action under the same generation seed.

The strongest paired effect away from the recorded true action is layer 15 at alpha 2: 0.282 of pairs changed from true-action emission to a different final action.

The strongest paired nonoptimal-to-optimal conversion is layer 15 at alpha 2: 0.231 of pairs converted from a nonoptimal emitted action to an action in `astar_actions`.

The per-action same-alpha direction-control check supports desired-direction steering at its strongest tested setting: layer 15 alpha 0.5, desired direction UP delta 1.000, best wrong direction DOWN delta 0.333, advantage 0.667.

The overall desired-vs-wrong control is strictly positive at its strongest tested setting: layer 15 alpha 0.5, desired-direction delta 0.231, wrong-direction delta 0.212, advantage 0.019, nonoptimal-to-desired advantage 0.019.

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
