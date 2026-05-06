# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_direction_control_layer15_alpha_neg32_balanced12.jsonl`
- Layers: 15
- Hooked alphas: -32

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | no_hook |  | 36 | 0.500 | 0.222 | 0.500 | 0.000 | 0.000 |
| 15 | hooked | -32 | 144 | 0.507 | 0.250 | 0.507 | 0.007 | 0.028 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 15 | 0 | 0.000 | 0.000 | 0.000 | 36 |

## Paired Causal Effects

| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | optimal-to-nonoptimal rate | true-to-not-true rate |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | -32 | 144 | 0.410 | 0.167 | 0.160 | 0.097 |

## Direction Controls

| layer | alpha | direction | desired dir | pairs | steered-action delta | desired-action delta | nonoptimal-to-desired rate |
|---:|---:|---|---|---:|---:|---:|---:|
| 15 | -32 | DOWN | False | 27 | -0.037 | 0.037 | 0.222 |
| 15 | -32 | DOWN | True | 9 | 0.000 | 0.000 | 0.111 |
| 15 | -32 | LEFT | False | 27 | -0.074 | 0.037 | 0.148 |
| 15 | -32 | LEFT | True | 9 | 0.222 | 0.222 | 0.333 |
| 15 | -32 | RIGHT | False | 27 | 0.037 | 0.037 | 0.148 |
| 15 | -32 | RIGHT | True | 9 | 0.000 | 0.000 | 0.111 |
| 15 | -32 | UP | False | 27 | -0.037 | 0.000 | 0.185 |
| 15 | -32 | UP | True | 9 | -0.444 | -0.444 | 0.000 |

## Direction-Control Advantage

| layer | alpha | desired direction | desired delta | best wrong direction | best wrong delta | desired advantage | supports desired direction |
|---:|---:|---|---:|---|---:|---:|---|
| 15 | -32 | LEFT | 0.222 | DOWN | 0.037 | 0.185 | True |

## Overall Direction Controls

| layer | alpha | direction group | pairs | desired-action delta | nonoptimal-to-desired rate | desired-to-nondesired rate |
|---:|---:|---|---:|---:|---:|---:|
| 15 | -32 | wrong | 108 | 0.028 | 0.176 | 0.148 |
| 15 | -32 | desired | 36 | -0.056 | 0.139 | 0.194 |

## Overall Direction-Control Advantage

| layer | alpha | desired delta | wrong delta | desired advantage | supports desired direction |
|---:|---:|---:|---:|---:|---|
| 15 | -32 | -0.056 | 0.028 | -0.083 | False |

## Interpretation

The largest optimal-action improvement versus no-hook is layer 15 at alpha -32: optimal-action rate 0.507, delta 0.007.

No hooked condition reduced true-action retention versus no-hook. The smallest true-action-rate delta is layer 15 at alpha -32: true-action rate 0.250, delta 0.028.

The strongest paired causal effect on emitted action is layer 15 at alpha -32: 0.410 of paired samples changed final action under the same generation seed.

The strongest paired effect away from the recorded true action is layer 15 at alpha -32: 0.097 of pairs changed from true-action emission to a different final action.

The strongest paired nonoptimal-to-optimal conversion is layer 15 at alpha -32: 0.167 of pairs converted from a nonoptimal emitted action to an action in `astar_actions`.

The same-alpha direction-control check supports desired-direction steering at its strongest tested setting: layer 15 alpha -32, desired direction LEFT delta 0.222, best wrong direction DOWN delta 0.037, advantage 0.185.

The overall desired-vs-wrong control is not positive. The best tested setting is layer 15 alpha -32, desired-direction delta -0.056, wrong-direction delta 0.028, advantage -0.083.

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
