# Seed12 Desired-Direction Steering Report

Date: May 5, 2026

## Executive Summary

The original steering experiment asked whether adding an optimal-action probe direction to the final three prompt-suffix activations causally changes the model's emitted action. That is too weak: an action change is not enough. The stricter question is whether the intervention moves the action toward the desired optimal action more than comparable wrong-action directions.

The supported result is a proof-of-principle for an adaptive baseline-source margin intervention:

- Layer: `15`
- Alpha: `0.5`
- Source action: the paired no-hook emitted action for the same prompt and sample seed
- Target action: the desired `astar_actions` action
- Direction: raw probe-logit margin `W_target - W_source`
- Positions patched: only the final three prompt-suffix token positions

Largest held-out confirmation:

| group | pairs | conversions to desired | conversion rate |
|---|---:|---:|---:|
| desired target direction | 69 | 27 | 0.391 |
| wrong target controls | 138 | 32 | 0.232 |

The desired-direction advantage is `+0.159`. This passes the strict criterion used in the audit: the desired direction has positive conversion, beats wrong target controls, and replicates in held-out sampling.

I also retried the same raw baseline-source protocol at layer `23`, motivated by the separate final-channel `has_key` result where layer 23 was the effective steering layer. The matched layer-23 retry failed at the layer-15 scale: at alpha `0.5`, desired directions converted `0/58` pairs and wrong controls converted `0/116` pairs. Positive-alpha screens at `1, 2, 4, 8` and then `32, 64, 128, 256, 512` also produced zero desired conversions.

However, a negative-sign stress test found a weaker layer-23 effect at a much larger magnitude. At layer `23`, alpha `-1024`, desired directions converted baseline non-optimal outputs to an A*-optimal desired action more often than wrong directions did. The first screen gave desired `0.200` vs wrong `0.033` over `15` desired and `30` wrong pairs. An independent-seed confirmation gave desired `0.242` vs wrong `0.136` over `33` desired and `66` wrong pairs. This means Seed12 layer-23 steering is possible, but only in the tested extreme opposite-sign regime; it is not the same clean moderate-scale effect as layer 15.

This does not mean every probe direction variant works. Target-only directions, recorded-true-source contrasts, raw recorded-true-source contrasts, and centroid recorded-true-source contrasts were negative or unreplicated. The positive claim is narrower: when the source is the model's actual paired no-hook non-optimal action, the raw optimal-action probe margin steers toward the desired optimal action better than wrong target margins.

## Motivation

The user objection was correct: "changed final action" is not sufficient evidence of causal steering. A perturbation can change output by destabilizing generation, increasing parse failures, or pushing the model toward any action distribution without encoding the intended direction.

The experiment therefore needs controls that distinguish:

- **Causal effect:** the hook changes the output under paired seeds.
- **Desired-direction effect:** the hook changes the output toward the target action more than wrong directions do.
- **Optimality effect:** the target action is in the A* optimal action set, not merely equal to the recorded trajectory action.

This distinction matters because the recorded trajectory action is often not optimal. A steering vector can move the model away from the recorded action while still improving optimality, or it can change actions without improving optimality.

## Data And Model

The experiment uses real Seed12 trajectory-step prompts:

- Trajectories: `data/trajectories/Seed12_T0_state_sweep/`
- Activation/probe source: `data/activations/Seed12_T0_state_sweep/`
- Probe checkpoints: `data/probes/Seed12_T0_state_sweep_action/`
- Direction bundles: `data/steering/Seed12_T0_state_sweep_optimal_action/directions_layer{layer}.pt`
- Main runner: `run_seed12_optimal_action_steering_mlx.py`

The model is run through MLX using:

```text
mlx-community/gpt-oss-20b-MXFP4-Q4
```

The steering probes are step-level linear `optimal_action` probes trained on `pre_reasoning_suffix` activations. That suffix feature is the mean of the final three prompt-suffix token activations. The steering intervention therefore patches those same three positions.

## Direction Construction

For a linear probe, the checkpoint stores weights in normalized feature space. If the probe standardizes activations with `scaler_std`, the activation-space weight for action `a` is:

```text
W_a = normalized_weight_a / scaler_std
```

The original target-only direction uses:

```text
d_a = normalize(W_a)
```

For the successful baseline-source intervention, the direction is instead a raw probe-logit margin:

```text
d_target,source = normalize(W_target - W_source)
```

where:

- `target` is a candidate target action.
- `source` is the action emitted by the no-hook baseline for the same prompt and generation seed.
- The desired candidate is the optimal `astar_actions` action.
- Wrong controls are the other candidate actions, excluding the source action.

The scale is:

```text
scale = alpha * class_gap
```

For the raw margin direction, `class_gap` is computed by projecting the target action's positive-minus-negative activation mean difference onto the unit margin direction:

```text
class_gap = mean_diff_target dot d_target,source
```

This keeps the scale tied to the probe's empirical activation separation while allowing the direction to be target-vs-source rather than one-vs-rest.

## Steering Algorithm

For each eligible Seed12 step and each sampled generation seed:

1. Reconstruct the prompt from trajectory prefix, current grid tokens, and the step-specific suffix tokens.
2. Resolve the final three suffix token positions as absolute prompt positions.
3. Generate the no-hook baseline with sampled decoding.
4. Parse the emitted action.
5. If the baseline action is missing or already optimal, skip baseline-source steering for that sample.
6. For every candidate action other than the baseline source action:
   - Build `normalize(W_candidate - W_source)`.
   - Mark the candidate as desired if it is in `astar_actions`.
   - Add `alpha * class_gap * direction` at the selected layer and only the final three suffix positions.
   - Regenerate with the same prompt and seed.
7. Compare desired target candidates against wrong candidate controls.

Pseudocode:

```text
for step in eligible_steps:
    suffix_positions = final_three_suffix_positions(step.prompt)
    for sample_seed in seeds:
        baseline = generate_no_hook(step.prompt, sample_seed)
        source = parse_action(baseline)

        if source is missing or source in astar_actions:
            continue

        for candidate in ACTIONS - {source}:
            direction = normalize(W_candidate - W_source)
            gap = mean_diff_candidate dot direction
            scale = alpha * gap

            hooked = generate_with_hook(
                prompt=step.prompt,
                seed=sample_seed,
                layer=15,
                positions=suffix_positions,
                additive_delta=scale * direction,
            )

            record candidate, whether candidate is desired, emitted action
```

This design makes the comparison paired at three levels:

- Same prompt
- Same generation seed
- Same baseline source action

The only intended difference is the target direction used by the hook.

## Implementation Details

The root script `run_seed12_optimal_action_steering_mlx.py` now has three primary modes:

- `prepare-directions`: extracts activation-space probe directions and stores direction bundles.
- `steer`: runs MLX generation with optional steering hooks.
- `report`: aggregates JSONL results into CSV, Markdown, and SVG artifacts.

The new direction mode is:

```text
--direction-set raw-action-vs-baseline
```

It differs from `raw-action-vs-true` in one important way:

- `raw-action-vs-true` uses the recorded trajectory action as the source.
- `raw-action-vs-baseline` uses the model's actual no-hook emitted action under the paired seed.

That matters because the recorded action is not necessarily the model's current sampled action. If the goal is to prove "move this output toward the desired action," the source should be the output the model would actually have produced.

The additive hook is implemented by `AdditiveLayerHook`. It wraps one transformer layer and, during the prompt forward pass, adds the intervention vector only at selected absolute prompt positions. The hook records:

- selected prompt positions
- touched positions
- patched positions
- prompt length
- pre/post norm metadata
- whether the hook fired

The latest hook audit for the larger confirmation found:

- 207 hooked rows
- all selected exactly 3 positions
- all selected suffix relative indices `(16, 17, 18)`
- those positions are distances `(2, 1, 0)` from prompt end
- 0 bad hook logs

So the positive result is not from patching generated tokens, hidden non-suffix positions, or a broader prompt region.

The layer-23 retry used the same final-three suffix positions. Hook metadata is recorded on the first sample for each hooked condition; all recorded layer-23 hook logs fired and patched exactly the intended three prompt positions.

## Metrics

The main metric is:

```text
nonoptimal_to_desired_rate =
    count(baseline emitted non-optimal action and hooked emitted desired optimal action)
    / count(paired eligible baseline non-optimal samples)
```

For the baseline-source intervention, every retained baseline sample is non-optimal by construction, so this equals the desired-action emission rate among retained source-action samples.

The strict support criterion is:

```text
desired_delta > 0
desired_delta > wrong_delta
desired_conversion_rate > wrong_conversion_rate
```

This deliberately rejects cases where the desired direction merely damages wrong controls less, or where total desired-action rate improves but wrong controls convert baseline non-desired samples more often.

## Results

Error bars in the SVG figures are Wilson 95% binomial intervals for rate estimates.

### Replication Across Screen And Held-Out Runs

![Conversion replication by stage](figures/conversion_replication_by_stage.svg)

The screen found alpha `0.5` as a candidate. Two held-out confirmations then tested the same intervention with new seeds and more samples.

| stage | desired conversions | desired pairs | wrong conversions | wrong pairs | advantage |
|---|---:|---:|---:|---:|---:|
| screen | 2 | 4 | 0 | 8 | 0.500 |
| held-out 1 | 6 | 26 | 11 | 52 | 0.019 |
| held-out 2 | 27 | 69 | 32 | 138 | 0.159 |

The first confirmation was only marginal, which is why a larger confirmation was necessary. The larger confirmation restored a clearer separation: 39.1% desired conversion versus 23.2% wrong-control conversion.

### Main Confirmation Metrics

![Final confirmation main metrics](figures/final_confirmation_main_metrics.svg)

In the largest confirmation:

| metric | desired direction | wrong controls |
|---|---:|---:|
| changed action rate | 34/69 = 0.493 | 57/138 = 0.413 |
| converted to desired | 27/69 = 0.391 | 32/138 = 0.232 |
| desired among changed | 27/34 = 0.794 | 32/57 = 0.561 |

The third row is especially important. Wrong controls also change actions, but when the desired direction changes the action, it lands on the desired optimal action much more often.

### Action Distribution

![Final confirmation action distribution](figures/final_confirmation_action_distribution.svg)

The retained baseline non-optimal source actions were mostly `UP`:

| group | LEFT | RIGHT | UP | DOWN | PARSE_FAIL |
|---|---:|---:|---:|---:|---:|
| baseline non-optimal source | 8 | 14 | 47 | 0 | 0 |
| after desired direction | 15 | 13 | 36 | 5 | 0 |
| after wrong controls | 29 | 26 | 75 | 8 | 0 |

This plot should be read as diagnostic, not as the core causal metric, because the desired action differs by state. Aggregating labels can obscure state-specific optimality. The causal metric remains whether the emitted action is in the step's `astar_actions`.

### Per-Target Direction Breakdown

![Final confirmation per target conversion](figures/final_confirmation_per_target_conversion.svg)

The effect is not uniform across target labels:

| target direction | desired conversion | wrong-control conversion |
|---|---:|---:|
| LEFT | 12/21 = 0.571 | 9/40 = 0.225 |
| RIGHT | 8/21 = 0.381 | 6/34 = 0.176 |
| UP | 2/3 = 0.667 | 4/19 = 0.211 |
| DOWN | 5/24 = 0.208 | 13/45 = 0.289 |

The overall positive result is driven mainly by `LEFT`, `RIGHT`, and the small-`n` `UP` target slice. `DOWN` does not show a positive target-vs-control advantage in this confirmation. This is a real limitation: the proof-of-principle supports desired-direction steering overall, not equally strong steering for every action label.

### Comparison Against Failed Variants

![Variant conversion advantage comparison](figures/variant_conversion_advantage_comparison.svg)

The stricter conversion metric rejects the earlier variants:

| variant | conversion advantage | strict support |
|---|---:|---|
| target-only raw direction | -0.056 | false |
| action-vs-recorded-true | -0.025 | false |
| raw action-vs-recorded-true | -0.033 | false |
| centroid action-vs-recorded-true | +0.033 | false |
| raw action-vs-baseline | +0.159 | true |

The centroid recorded-true variant has a positive conversion advantage, but it still fails the strict criterion because its desired-action delta is negative. This is exactly why the stricter criterion is useful: it rejects fragile effects that only look good relative to even worse controls.

### Layer-23 Retry And Stress Test

The layer-23 retry answers whether the successful layer-15 Seed12 action-margin intervention transfers to the layer that worked in the final-channel `has_key` experiment.
At the matched layer-15 scale and sign, it does not.
At an extreme opposite sign, it produces a weaker but replicated desired-vs-wrong effect.

| condition | layer | alpha(s) | desired pairs | wrong pairs | desired conversion | wrong conversion | advantage | strict support |
|---|---:|---|---:|---:|---:|---:|---:|---|
| larger confirmation | 15 | 0.5 | 69 | 138 | 0.391 | 0.232 | +0.159 | true |
| matched retry | 23 | 0.5 | 58 | 116 | 0.000 | 0.000 | 0.000 | false |
| stronger-alpha screen | 23 | 1, 2, 4, 8 | 15 per alpha | 30 per alpha | 0.000 | 0.000 | 0.000 | false |
| high positive-alpha screen | 23 | 32, 64, 128, 256, 512 | 15 per alpha | 30 per alpha | 0.000 | 0.000 | 0.000 | false |
| negative stress screen | 23 | -1024 | 15 | 30 | 0.200 | 0.033 | +0.167 | true |
| negative stress confirmation | 23 | -1024 | 33 | 66 | 0.242 | 0.136 | +0.106 | true |

The diagnostic audit is useful for separating regimes.
In the layer-23 alpha-0.5 retry, every hooked completion stayed equal to its paired no-hook source action (`174/174` hooked rows).
In the `1, 2, 4, 8` screen, the same happened for all hooked rows (`180/180`).
In the positive high-alpha screen up to `512`, every hooked completion again stayed equal to the paired source action (`225/225`).
The negative stress test at `-1024` finally moved actions: the first screen changed `7/45` hooked rows, and the independent-seed confirmation changed `23/99` hooked rows.
Hook logs in the stress confirmation showed `21` hooked rows with the hook firing on exactly the final three suffix positions and `0` bad hook logs.

## Interpretation

The positive result supports this causal statement:

> On the tested balanced Seed12 subset, at layer 15 and alpha 0.5, adding the raw optimal-action probe margin from the actual no-hook emitted non-optimal action toward the desired optimal action to the final three suffix-token activations increases desired-action emission more than wrong target margins do.

This is stronger than "the action changed." It shows direction specificity:

- desired target directions convert more often than wrong target directions;
- the comparison is paired on prompt and seed;
- the source action is the baseline action actually produced by the model;
- the hook only patches the final three suffix tokens.

The result also explains why earlier attempts failed. If the source action is the recorded trajectory action, the contrast direction may not correspond to the model's actual sampled behavior. If the source is omitted entirely, the target-only direction can cause broad action instability rather than movement from the current action toward the target. The baseline-source direction aligns the intervention with the behavior we are trying to change.

The layer-23 retry adds another constraint: the useful intervention layer is task/prompt dependent.
Layer 23 is effective for matched final-channel `has_key` prompts.
For Seed12 raw action-margin steering, layer 23 is inert across the matched and positive-alpha regimes, but not completely inert: an extreme negative stress setting produces a smaller desired-vs-wrong effect.
The Seed12 success should therefore be described as two regimes: a cleaner layer-15 effect at alpha `0.5`, and a weaker layer-23 stress effect at alpha `-1024`.

## Limitations

This is a proof-of-principle, not a full universal claim.

- The supported run is on a balanced 12-step subset, not the full eligible Seed12 sweep.
- The clean supported layer/alpha is layer 15 and alpha 0.5.
- Layer 23 only shows support in an extreme opposite-sign stress setting, alpha -1024; the matched retry and positive-alpha screens failed.
- The intervention is adaptive: it needs a no-hook baseline action first. That is suitable for causal testing, but it is not a one-pass deployment method unless the source action is otherwise predicted.
- The `DOWN` target slice did not show a positive desired-vs-control advantage.
- The experiment uses sampled decoding, so confirmation seeds and sample counts matter.
- The report does not establish that the direction is semantically clean in every context; it establishes a controlled behavioral effect under this intervention.

## Reproducibility Pointers

Prepare directions:

```bash
uv run python run_seed12_optimal_action_steering_mlx.py prepare-directions --layers 7 15 23 --overwrite
```

Run the supported larger confirmation:

```bash
uv run python run_seed12_optimal_action_steering_mlx.py steer \
  --layers 15 \
  --direction-set raw-action-vs-baseline \
  --step-ordinals 0 41 80 127 128 168 170 303 345 432 556 1171 \
  --alphas 0.5 \
  --n-samples 10 \
  --seed 20260507 \
  --output data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_raw_action_vs_baseline_layer15_alpha_0p5_balanced12_confirm2.jsonl \
  --overwrite
```

Aggregate the report:

```bash
uv run python run_seed12_optimal_action_steering_mlx.py report \
  --results data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_raw_action_vs_baseline_layer15_alpha_0p5_balanced12_confirm2.jsonl \
  --report-dir reports/seed12_optimal_action_steering/raw_action_vs_baseline_layer15_alpha_0p5_balanced12_confirm2 \
  --layers 15
```

Run the matched layer-23 retry:

```bash
uv run python run_seed12_optimal_action_steering_mlx.py steer \
  --layers 23 \
  --direction-set raw-action-vs-baseline \
  --step-ordinals 0 41 80 127 128 168 170 303 345 432 556 1171 \
  --alphas 0.5 \
  --n-samples 10 \
  --output data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_raw_action_vs_baseline_layer23_alpha_0p5_balanced12_confirm.jsonl \
  --overwrite
```

Run the layer-23 stronger-alpha screen:

```bash
uv run python run_seed12_optimal_action_steering_mlx.py steer \
  --layers 23 \
  --direction-set raw-action-vs-baseline \
  --step-ordinals 0 41 80 127 128 168 170 303 345 432 556 1171 \
  --alphas 1 2 4 8 \
  --n-samples 3 \
  --output data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_raw_action_vs_baseline_layer23_alpha_screen_balanced12.jsonl \
  --overwrite
```

Run the layer-23 positive high-alpha screen:

```bash
uv run python run_seed12_optimal_action_steering_mlx.py steer \
  --layers 23 \
  --direction-set raw-action-vs-baseline \
  --step-ordinals 0 41 80 127 128 168 170 303 345 432 556 1171 \
  --alphas 32 64 128 256 512 \
  --n-samples 3 \
  --output data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_raw_action_vs_baseline_layer23_high_alpha_screen_balanced12.jsonl \
  --overwrite
```

Run the layer-23 negative stress confirmation:

```bash
uv run python run_seed12_optimal_action_steering_mlx.py steer \
  --layers 23 \
  --direction-set raw-action-vs-baseline \
  --step-ordinals 0 41 80 127 128 168 170 303 345 432 556 1171 \
  --alphas -1024 \
  --n-samples 5 \
  --seed 314159 \
  --output data/steering/Seed12_T0_state_sweep_optimal_action/steering_results_raw_action_vs_baseline_layer23_alpha_neg1024_seed314159_balanced12_confirm.jsonl \
  --overwrite
```

Validation commands:

```bash
uv run ruff check run_seed12_optimal_action_steering_mlx.py tests/test_seed12_optimal_action_steering_mlx.py
uv run ruff format --check run_seed12_optimal_action_steering_mlx.py tests/test_seed12_optimal_action_steering_mlx.py
uv run pytest tests/test_seed12_optimal_action_steering_mlx.py -vv
```

Latest validation result: `21 passed`.

## Conclusion

The experiment now answers the user's methodological concern. A changed final action by itself is not treated as causal proof of desired steering. The successful condition demonstrates a desired-direction effect against wrong-direction controls, under paired seeds, with the intervention restricted to the final three suffix tokens.

The strongest claim supported by the evidence is:

> The raw optimal-action probe margin from the model's actual baseline non-optimal emitted action to the desired optimal action can causally steer the emitted action toward that desired action, at layer 15 alpha 0.5, on the balanced Seed12 subset tested. Layer 23 does not work under the matched alpha 0.5 or positive-alpha screens, but an extreme opposite-sign layer-23 stress setting at alpha -1024 gives a smaller replicated desired-vs-wrong effect.
