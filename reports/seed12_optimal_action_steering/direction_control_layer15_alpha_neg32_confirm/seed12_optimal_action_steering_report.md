# Seed12 Optimal-Action Direction Steering

## Setup

This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds `alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. The no-hook baseline is separate from the alpha 0 hook baseline.

- Results: `data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_direction_control_layer15_alpha_neg32_confirm.jsonl`
- Layers: 15
- Hooked alphas: -32

## Main Rates

| layer | condition | alpha | n | optimal action rate | true action rate | nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | no_hook |  | 40 | 0.350 | 0.575 | 0.350 | 0.000 | 0.000 |
| 15 | hooked | -32 | 160 | 0.431 | 0.500 | 0.431 | 0.081 | -0.075 |

## Alpha 0 No-Op Check

| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |
|---:|---:|---:|---:|---:|---:|
| 15 | 0 | 0.000 | 0.000 | 0.000 | 40 |

## Paired Causal Effects

| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | optimal-to-nonoptimal rate | true-to-not-true rate |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | -32 | 160 | 0.400 | 0.212 | 0.131 | 0.200 |

## Direction Controls

| layer | alpha | direction | desired dir | pairs | steered-action delta | desired-action delta | nonoptimal-to-desired rate |
|---:|---:|---|---|---:|---:|---:|---:|
| 15 | -32 | DOWN | True | 40 | 0.175 | 0.175 | 0.325 |
| 15 | -32 | LEFT | False | 40 | -0.025 | 0.125 | 0.250 |
| 15 | -32 | RIGHT | False | 40 | 0.000 | 0.050 | 0.175 |
| 15 | -32 | UP | False | 40 | 0.000 | -0.025 | 0.100 |

## Direction-Control Advantage

| layer | alpha | desired direction | desired delta | best wrong direction | best wrong delta | desired advantage | supports desired direction |
|---:|---:|---|---:|---|---:|---:|---|
| 15 | -32 | DOWN | 0.175 | LEFT | 0.125 | 0.050 | True |

## Overall Direction Controls

| layer | alpha | direction group | pairs | desired-action delta | nonoptimal-to-desired rate | desired-to-nondesired rate |
|---:|---:|---|---:|---:|---:|---:|
| 15 | -32 | wrong | 120 | 0.050 | 0.175 | 0.125 |
| 15 | -32 | desired | 40 | 0.175 | 0.325 | 0.150 |

## Overall Direction-Control Advantage

| layer | alpha | desired delta | wrong delta | desired advantage | supports desired direction |
|---:|---:|---:|---:|---:|---|
| 15 | -32 | 0.175 | 0.050 | 0.125 | True |

## Interpretation

The largest optimal-action improvement versus no-hook is layer 15 at alpha -32: optimal-action rate 0.431, delta 0.081.

The largest true-action retention drop versus no-hook is layer 15 at alpha -32: true-action rate 0.500, delta -0.075.

The strongest paired causal effect on emitted action is layer 15 at alpha -32: 0.400 of paired samples changed final action under the same generation seed.

The strongest paired effect away from the recorded true action is layer 15 at alpha -32: 0.200 of pairs changed from true-action emission to a different final action.

The strongest paired nonoptimal-to-optimal conversion is layer 15 at alpha -32: 0.212 of pairs converted from a nonoptimal emitted action to an action in `astar_actions`.

The same-alpha direction-control check supports desired-direction steering at its strongest tested setting: layer 15 alpha -32, desired direction DOWN delta 0.175, best wrong direction LEFT delta 0.125, advantage 0.050.

The overall desired-vs-wrong control is positive at its strongest tested setting: layer 15 alpha -32, desired-direction delta 0.175, wrong-direction delta 0.050, advantage 0.125.

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
