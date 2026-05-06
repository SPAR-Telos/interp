# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_raw_action_vs_baseline_layer23_alpha_0p5_balanced12_confirm.jsonl`
- Layers: 23
- Hooked alphas: 0.5

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 23 | no_hook |  | 120 | 0.517 | 0.242 | 0.517 | 0.000 | 0.000 |
| 23 | hooked | 0.5 | 174 | 0.000 | 0.500 | 0.000 | -0.517 | 0.258 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 23 | 0 | 0.000 | 0.000 | 0.000 | 120 |

## Paired Causal Effects

| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | optimal-to-nonoptimal rate | true-to-not-true rate |
|---:|---:|---:|---:|---:|---:|---:|
| 23 | 0.5 | 174 | 0.000 | 0.000 | 0.000 | 0.000 |

## Direction Controls

| layer | alpha | direction | desired dir | pairs | steered-action delta | desired-action delta | nonoptimal-to-desired rate |
|---:|---:|---|---|---:|---:|---:|---:|
| 23 | 0.5 | DOWN | False | 32 | 0.000 | 0.000 | 0.000 |
| 23 | 0.5 | DOWN | True | 24 | 0.000 | 0.000 | 0.000 |
| 23 | 0.5 | LEFT | False | 33 | 0.000 | 0.000 | 0.000 |
| 23 | 0.5 | LEFT | True | 18 | 0.000 | 0.000 | 0.000 |
| 23 | 0.5 | RIGHT | False | 33 | 0.000 | 0.000 | 0.000 |
| 23 | 0.5 | RIGHT | True | 15 | 0.000 | 0.000 | 0.000 |
| 23 | 0.5 | UP | False | 18 | 0.000 | 0.000 | 0.000 |
| 23 | 0.5 | UP | True | 1 | 0.000 | 0.000 | 0.000 |

## Direction-Control Advantage

| layer | alpha | desired direction | desired delta | best wrong direction | best wrong delta | desired advantage | supports desired direction |
|---:|---:|---|---:|---|---:|---:|---|
| 23 | 0.5 | DOWN | 0.000 | DOWN | 0.000 | 0.000 | False |

## Overall Direction Controls

| layer | alpha | direction group | pairs | desired-action delta | nonoptimal-to-desired rate | desired-to-nondesired rate |
|---:|---:|---|---:|---:|---:|---:|
| 23 | 0.5 | wrong | 116 | 0.000 | 0.000 | 0.000 |
| 23 | 0.5 | desired | 58 | 0.000 | 0.000 | 0.000 |

## Overall Direction-Control Advantage

| layer | alpha | desired delta | wrong delta | desired advantage | supports desired direction |
|---:|---:|---:|---:|---:|---|
| 23 | 0.5 | 0.000 | 0.000 | 0.000 | False |

## Interpretation

No hooked condition improved optimal-action rate versus no-hook. The least-negative condition is layer 23 at alpha 0.5: optimal-action rate 0.000, delta -0.517.

No hooked condition reduced true-action retention versus no-hook. The smallest true-action-rate delta is layer 23 at alpha 0.5: true-action rate 0.500, delta 0.258.

The strongest paired causal effect on emitted action is layer 23 at alpha 0.5: 0.000 of paired samples changed final action under the same generation seed.

The strongest paired effect away from the recorded true action is layer 23 at alpha 0.5: 0.000 of pairs changed from true-action emission to a different final action.

The strongest paired nonoptimal-to-optimal conversion is layer 23 at alpha 0.5: 0.000 of pairs converted from a nonoptimal emitted action to an action in `astar_actions`.

The per-action same-alpha direction-control check does not support desired-direction steering. Strict support requires both desired-action-rate and nonoptimal-to-desired conversion advantages. The best tested desired-vs-wrong advantage is layer 23 alpha 0.5: desired direction DOWN delta 0.000, best wrong direction DOWN delta 0.000, advantage 0.000.

The overall desired-vs-wrong control fails the strict support criterion. The best tested setting is layer 23 alpha 0.5, desired-direction delta 0.000, wrong-direction delta 0.000, advantage 0.000, nonoptimal-to-desired advantage 0.000.

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
