# Desired-Direction Steering Audit

## Claim Under Test

The desired proof is: adding a probe-derived optimal-action direction to only the final three prompt-suffix tokens causes the model's emitted final action to move toward the desired optimal action, not merely to change.

The required evidence is therefore stricter than paired action changes:

- The hook must patch only the final three suffix tokens.
- Hooked generations must be paired against no-hook generations with the same prompt and seed.
- The intended desired-action direction must beat wrong action directions on the same states/seeds.
- The intended desired-action direction must also beat wrong directions at converting baseline non-desired outputs to desired outputs.
- A selected alpha or state subset must replicate on held-out sampling, not only in a post-hoc screen.

## Implementation Status

The implementation supports these controls in `run_seed12_optimal_action_steering_mlx.py`.

- `--direction-set all-actions`: tests each single action direction as desired/wrong controls.
- `--direction-set action-contrast`: tests `W_action - mean(W_other_actions)` contrast directions.
- `--direction-set action-vs-true`: tests `W_candidate - W_recorded_true` contrasts, skipping the recorded true action.
- `--direction-set raw-action-contrast`: tests raw probe-logit one-vs-rest contrast directions.
- `--direction-set raw-action-vs-true`: tests raw probe-logit `W_candidate - W_recorded_true` margin directions.
- `--direction-set raw-action-vs-baseline`: tests raw probe-logit `W_candidate - W_no_hook_emitted` margin directions, restricted to paired no-hook samples whose emitted action is parsed and non-optimal.
- `--direction-set centroid-action-contrast`: tests positive-centroid one-vs-rest optimal-action directions.
- `--direction-set centroid-action-vs-true`: tests positive-centroid `target - recorded_true` directions.
- Reports include `direction_control_overall.csv` and `direction_control_overall_advantage.csv`.
- Reports can be filtered by layer with `report --layers`.
- Hook logs record the selected and patched prompt positions.

Focused validation passed:

```bash
uv run ruff check run_seed12_optimal_action_steering_mlx.py tests/test_seed12_optimal_action_steering_mlx.py
uv run ruff format --check run_seed12_optimal_action_steering_mlx.py tests/test_seed12_optimal_action_steering_mlx.py
uv run pytest tests/test_seed12_optimal_action_steering_mlx.py -vv
```

Latest focused test result: 21 passed.

Latest hook/position audit:

- `steering_results_raw_action_vs_baseline_layer15_balanced12_alpha_screen.jsonl`: 84 hooked rows, all selected exactly 3 positions, 0 bad hook logs.
- `steering_results_raw_action_vs_baseline_layer15_alpha_0p5_2_balanced12_confirm.jsonl`: 156 hooked rows, all selected exactly 3 positions, 0 bad hook logs.
- `steering_results_raw_action_vs_baseline_layer15_alpha_0p5_balanced12_confirm2.jsonl`: 207 hooked rows, all selected exactly 3 positions, 0 bad hook logs.
- `steering_results_raw_action_vs_baseline_layer23_alpha_neg1024_seed314159_balanced12_confirm.jsonl`: 21 hooked rows with hook logs, all selected exactly 3 positions, 0 bad hook logs.
- In the larger confirmation, all hooked rows selected suffix relative indices `(16, 17, 18)`, corresponding to distances `(2, 1, 0)` from prompt end.

## Evidence Summary

| run | direction family | layer | alpha | desired delta | wrong delta | desired advantage | supports |
|---|---|---:|---:|---:|---:|---:|---|
| `direction_control_layer15_alpha_neg32_confirm` | raw action | 15 | -32 | 0.175 | 0.050 | 0.125 | true |
| `direction_control_layer15_alpha_neg32_balanced12` | raw action | 15 | -32 | -0.056 | 0.028 | -0.083 | false |
| `direction_control_layer15_balanced12_alpha_screen` | raw action | 15 | -64 | 0.250 | 0.111 | 0.139 | true |
| `direction_control_layer15_alpha_neg64_balanced12_confirm` | raw action | 15 | -64 | -0.067 | -0.044 | -0.022 | false |
| `action_contrast_layer15_alpha_neg64_balanced12_smoke` | action contrast | 15 | -64 | 0.000 | 0.111 | -0.111 | false |
| `action_vs_true_layer15_balanced12_alpha_screen` | action-vs-true contrast | 15 | -64 | -0.083 | -0.208 | 0.125 | false |
| `raw_action_vs_true_layer15_balanced12_alpha_screen` | raw action-vs-true margin | 15 | -16 | 0.083 | 0.000 | 0.083 | true |
| `raw_action_vs_true_layer15_alpha_neg16_balanced12_confirm` | raw action-vs-true margin | 15 | -16 | -0.017 | -0.025 | 0.008 | false |
| `raw_action_vs_true_layer7_balanced12_alpha_screen` | raw action-vs-true margin | 7 | -64 | -0.083 | -0.208 | 0.125 | false |
| `centroid_action_vs_true_layer15_balanced12_alpha_screen` | centroid action-vs-true | 15 | 0.1 | 0.167 | 0.000 | 0.167 | true |
| `centroid_action_vs_true_layer15_alpha_pos0p1_balanced12_confirm` | centroid action-vs-true | 15 | 0.1 | -0.067 | -0.083 | 0.017 | false |
| `action_vs_true_layer15_alpha_neg64_balanced12_conversion_confirm` | action-vs-true conversion check | 15 | -64 | 0.117 | 0.092 | 0.025 | false |
| `raw_action_vs_baseline_layer15_balanced12_alpha_screen` | raw action-vs-baseline margin | 15 | 0.5 | 0.500 | 0.000 | 0.500 | true |
| `raw_action_vs_baseline_layer15_alpha_0p5_2_balanced12_confirm` | raw action-vs-baseline margin | 15 | 0.5 | 0.231 | 0.212 | 0.019 | true |
| `raw_action_vs_baseline_layer15_alpha_0p5_balanced12_confirm2` | raw action-vs-baseline margin | 15 | 0.5 | 0.391 | 0.232 | 0.159 | true |
| `raw_action_vs_baseline_layer23_alpha_0p5_balanced12_confirm` | raw action-vs-baseline margin | 23 | 0.5 | 0.000 | 0.000 | 0.000 | false |
| `raw_action_vs_baseline_layer23_alpha_screen_balanced12` | raw action-vs-baseline margin | 23 | 1--8 | 0.000 | 0.000 | 0.000 | false |
| `raw_action_vs_baseline_layer23_high_alpha_screen_balanced12` | raw action-vs-baseline margin | 23 | 32--512 | 0.000 | 0.000 | 0.000 | false |
| `raw_action_vs_baseline_layer23_negative_stress_balanced12` | raw action-vs-baseline margin | 23 | -1024 | 0.200 | 0.033 | 0.167 | true |
| `raw_action_vs_baseline_layer23_alpha_neg1024_seed314159_balanced12_confirm` | raw action-vs-baseline margin | 23 | -1024 | 0.242 | 0.136 | 0.106 | true |

Before the baseline-source intervention, the only clear positive result was a narrow `DOWN`-target slice at layer 15, alpha -32. That result did not generalize to a balanced 12-state set. The best alpha from the balanced screen, alpha -64, failed a held-out confirmation with a new seed and more samples.

The `action-vs-true` contrast screen found no supported alpha. Its best overall desired-vs-wrong advantage was alpha -64, but the desired direction still reduced desired-action emission versus the paired no-hook baseline; it only looked better because wrong directions were even more negative.

The raw probe-logit margin variant produced one weak layer-15 screen candidate at alpha -16, but the held-out confirmation failed. In confirmation, the desired direction slightly reduced desired-action emission, and wrong directions had a higher nonoptimal-to-desired conversion rate. The completed layer-7 raw margin screen was also negative. A partial layer-23 raw margin screen was stopped after slow rows and visible saturation under wrong directions; it is not counted as a completed proof attempt.

The centroid optimal-action direction variant also produced one layer-15 screen candidate at alpha 0.1, but its held-out confirmation failed. In confirmation, both desired and wrong directions reduced desired-action emission versus no-hook, and the desired direction did not satisfy the positive desired-delta support criterion.

A conversion-focused held-out check for the earlier action-vs-true alpha -64 candidate also failed the stricter support criterion. Although the desired direction slightly beat wrong directions on total desired-action rate, wrong directions had the higher nonoptimal-to-desired conversion rate.

The raw action-vs-baseline variant is the first result that satisfies the stricter desired-direction criterion on a balanced subset and held-out sampling. This intervention uses the paired no-hook emitted action as the source direction and only evaluates samples where that baseline action is parsed and non-optimal. At layer 15, alpha 0.5:

- Screen: desired conversion 0.500 vs wrong-control conversion 0.000 over 4 desired and 8 wrong pairs.
- Held-out confirmation: desired conversion 0.231 vs wrong-control conversion 0.212 over 26 desired and 52 wrong pairs.
- Larger held-out confirmation: desired conversion 0.391 vs wrong-control conversion 0.232 over 69 desired and 138 wrong pairs.

This supports a narrower claim than the original target-only steering setup: the raw optimal-action probe margin from the actual no-hook emitted action toward the desired optimal action causally increases desired-action emission more than wrong target margins, under paired seeds, while patching only the final three prompt-suffix tokens.

The layer-23 follow-up splits into two regimes.
Matched and positive-alpha settings did not move the parsed action at all: alpha 0.5 preserved the source action in 174/174 hooked rows, alpha 1--8 preserved it in 180/180 rows, and alpha 32--512 preserved it in 225/225 rows.
An extreme opposite-sign stress setting at alpha -1024 did move actions and replicated under an independent seed.
The screen gave desired conversion 0.200 vs wrong-control conversion 0.033 over 15 desired and 30 wrong pairs.
The independent-seed confirmation gave desired conversion 0.242 vs wrong-control conversion 0.136 over 33 desired and 66 wrong pairs.
This is valid evidence that layer 23 can steer Seed12 desired actions under this raw baseline-source margin, but it is weaker and less clean than the layer-15 alpha-0.5 result because it needs a very large negative scale and wrong controls also sometimes convert to optimal actions.

## Conclusion

The current evidence supports desired-direction steering for the adaptive raw action-vs-baseline margin intervention at layer 15, alpha 0.5, on the balanced Seed12 subset tested here.
It also supports a weaker layer-23 stress effect at alpha -1024.

The result does not rescue the earlier target-only, action-vs-recorded-true, raw action-vs-recorded-true, or centroid action-vs-recorded-true variants; those remain negative or unreplicated. The supported claim is specifically that, when the source direction is the actual paired no-hook emitted non-optimal action, the raw optimal-action probe margin toward the desired optimal action beats wrong target margins on non-optimal-to-desired conversion.

The remaining limitation is scope: this is not yet a full all-eligible Seed12 sweep across layers. It is a held-out balanced-subset proof-of-principle for desired-direction steering under the stricter causal/control criterion, with a clean moderate-scale layer-15 result and a weaker extreme-scale layer-23 result.
