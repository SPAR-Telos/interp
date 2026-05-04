# 9x9 door-key grid has-key direction steering across layers

## Dataset and probes

- Paired synthetic dataset: 12 prompts (6 template states; probe family door-key-grid).
- Class balance: False=6, True=6.
- Captured features: prompt_suffix_indices='-3:-1', relative indices [0, 1, 2], flattened to 8640 dimensions.

| layer | eval acc | eval balanced acc | all-state train acc | class gap | direction norm |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.500 | 0.500 | 1.000 | 11.163 | 1.000 |
| 15 | 1.000 | 1.000 | 1.000 | 113.274 | 1.000 |
| 23 | 1.000 | 1.000 | 1.000 | 3387.802 | 1.000 |

## Prompt family

The 9x9 target grid is fixed across prompt A and prompt B; only the key-state condition and its policy-implied action differ. Prompt A is the no-key condition, where the baseline next action is RIGHT to start routing toward K. Prompt B is the has-key condition, where the baseline next action is UP through D toward G.

```text
# # # # # # # # #
# _ _ _ # G _ _ #
# _ _ _ # _ _ _ #
# _ _ _ _ _ _ _ #
# D # # # # # # #
# A _ _ # K _ _ #
# _ _ _ # _ _ _ #
# _ _ _ _ _ _ _ #
# # # # # # # # #
```

The paired probe dataset uses six synthetic 9x9 door-key templates with the same rendered grid in each False/True pair. The templates are deliberately final-channel action prompts so the first generated content is the action token being measured. This keeps the intervention local to the pre-action suffix activations, but it also means the finding is scoped to this matched policy-prompt family rather than to arbitrary verbose grid reasoning prompts.

## Detailed finding

This run tests whether the final three prompt-suffix activations can causally control the emitted action when we add the direction learned by a linear `has_key` probe. The intervention is not a donor activation transplant: for each selected token position, the hook adds `alpha * class_gap * direction[layer, token_position]` to the layer output and leaves every other prompt and generation position unchanged.

The selected locations are exactly `prompt_suffix_indices='-3:-1'`. In this target prompt family those are suffix-relative indices [0, 1, 2]; for the target A/B prompts the hook logs resolve them to absolute positions `[214, 215, 216]`, the final-channel suffix tokens `<|channel|>`, `final`, and `<|message|>`. MLX processes this prompt as one multi-token prefill followed by a one-token prompt singleton, so the first two selected suffix tokens are changed in the prefill call and the final selected suffix token is changed in the singleton call. The recorded hook logs below verify this split with `seq_lens=[216, 1]` and `patched_positions=[214, 215, 216]`.

The successful steering evidence is concentrated at layer 23. Positive alpha means moving in the `has_key=True` direction; therefore prompt A, whose baseline action is RIGHT, should move toward UP as alpha increases. Negative alpha means moving away from `has_key=True`; therefore prompt B, whose baseline action is UP, should move toward RIGHT as alpha decreases.

Layer 23 action distributions at the most diagnostic alpha values:

| prompt | alpha | distribution | freq(UP) | freq(RIGHT) |
|---|---:|---|---:|---:|
| a | -8 | RIGHT=30/30 | 0.000 | 1.000 |
| a | -4 | RIGHT=30/30 | 0.000 | 1.000 |
| a | -2 | RIGHT=30/30 | 0.000 | 1.000 |
| a | 0 | RIGHT=30/30 | 0.000 | 1.000 |
| a | 2 | UP=11/30, RIGHT=19/30 | 0.367 | 0.633 |
| a | 4 | UP=30/30 | 1.000 | 0.000 |
| a | 8 | UP=30/30 | 1.000 | 0.000 |
| b | -8 | RIGHT=30/30 | 0.000 | 1.000 |
| b | -4 | RIGHT=30/30 | 0.000 | 1.000 |
| b | -2 | RIGHT=30/30 | 0.000 | 1.000 |
| b | 0 | UP=30/30 | 1.000 | 0.000 |
| b | 2 | UP=30/30 | 1.000 | 0.000 |
| b | 4 | UP=30/30 | 1.000 | 0.000 |
| b | 8 | UP=30/30 | 1.000 | 0.000 |

Layer 23 hook evidence for the endpoint flips:

| prompt | alpha | mode | selected positions | patched positions | seq lens | scale | pre norm | post norm |
|---|---:|---|---|---|---|---:|---:|---:|
| a | 8 | add | [214, 215, 216] | [214, 215, 216] | [216, 1] | 27102.412 | 29026.9 | 34609.9 |
| b | -8 | add | [214, 215, 216] | [214, 215, 216] | [216, 1] | -27102.412 | 27497.5 | 35892.2 |

Endpoint summary: at layer 23, prompt A changes from RIGHT=30/30 at alpha 0 to UP=30/30 at alpha 8. Prompt B changes from UP=30/30 at alpha 0 to RIGHT=30/30 at alpha -8. This is the expected bidirectional pattern for a `has_key` direction: adding it makes a no-key prompt behave like has-key, while subtracting it makes a has-key prompt behave like no-key.

Layer comparison matters for interpretation. The other tested layers do not provide a clean bidirectional action-control result in this sweep. Layer 23 is the only layer that satisfies the monotonic steering criterion across the sampled alphas and produces clean endpoint flips without parse failures at the decisive A alpha 8 and B alpha -8 endpoints.

Scope and caveats: this is evidence for a causal effect in the matched final-channel 9x9 door-key grid family, where the first generated content is the actual action token. It does not claim that the same single probe direction will steer every verbose grid prompt or every reasoning format. The result also depends on relatively large layer-23 scales because the direction is a unit activation-space probe direction multiplied by the probe class gap and alpha. Within this scoped setting, however, the hook logs and action distributions support the causal claim: additive steering of only the final three suffix activations along the learned `has_key` direction is sufficient to change the emitted action.

## Hook sanity

| layer | prompt | alpha | fired | seq lens | selected positions | patched positions | scale | pre norm | post norm |
|---:|---|---:|---|---|---|---|---:|---:|---:|
| 7 | a | -8 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -89.305 | 605.6 | 607.4 |
| 7 | a | -4 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -44.652 | 605.6 | 605.9 |
| 7 | a | -2 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -22.326 | 605.6 | 605.6 |
| 7 | a | -1 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -11.163 | 605.6 | 605.6 |
| 7 | a | 0 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 0.000 | 605.6 | 605.6 |
| 7 | a | 1 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 11.163 | 605.6 | 605.6 |
| 7 | a | 2 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 22.326 | 605.6 | 605.8 |
| 7 | a | 4 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 44.652 | 605.6 | 606.3 |
| 7 | a | 8 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 89.305 | 605.6 | 608.1 |
| 7 | b | -8 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -89.305 | 603.4 | 604.8 |
| 7 | b | -4 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -44.652 | 603.4 | 603.5 |
| 7 | b | -2 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -22.326 | 603.4 | 603.4 |
| 7 | b | -1 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -11.163 | 603.4 | 603.3 |
| 7 | b | 0 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 0.000 | 603.4 | 603.4 |
| 7 | b | 1 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 11.163 | 603.4 | 603.5 |
| 7 | b | 2 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 22.326 | 603.4 | 603.6 |
| 7 | b | 4 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 44.652 | 603.4 | 604.3 |
| 7 | b | 8 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 89.305 | 603.4 | 606.2 |
| 15 | a | -8 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -906.192 | 3106.6 | 3121.8 |
| 15 | a | -4 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -453.096 | 3106.6 | 3106.2 |
| 15 | a | -2 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -226.548 | 3106.6 | 3104.3 |
| 15 | a | -1 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -113.274 | 3106.6 | 3104.9 |
| 15 | a | 0 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 0.000 | 3106.6 | 3106.6 |
| 15 | a | 1 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 113.274 | 3106.6 | 3109.2 |
| 15 | a | 2 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 226.548 | 3106.6 | 3112.8 |
| 15 | a | 4 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 453.096 | 3106.6 | 3122.6 |
| 15 | a | 8 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 906.192 | 3106.6 | 3153.8 |
| 15 | b | -8 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -906.192 | 3156.5 | 3161.7 |
| 15 | b | -4 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -453.096 | 3156.5 | 3151.4 |
| 15 | b | -2 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -226.548 | 3156.5 | 3151.8 |
| 15 | b | -1 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -113.274 | 3156.5 | 3153.7 |
| 15 | b | 0 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 0.000 | 3156.5 | 3156.5 |
| 15 | b | 1 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 113.274 | 3156.5 | 3160.3 |
| 15 | b | 2 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 226.548 | 3156.5 | 3165.0 |
| 15 | b | 4 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 453.096 | 3156.5 | 3177.0 |
| 15 | b | 8 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 906.192 | 3156.5 | 3212.6 |
| 23 | a | -8 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -27102.412 | 29026.9 | 39159.4 |
| 23 | a | -4 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -13551.206 | 29026.9 | 32504.1 |
| 23 | a | -2 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -6775.603 | 29026.9 | 30257.9 |
| 23 | a | -1 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -3387.802 | 29026.9 | 29508.3 |
| 23 | a | 0 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 0.000 | 29026.9 | 29026.9 |
| 23 | a | 1 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 3387.802 | 29026.9 | 28822.6 |
| 23 | a | 2 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 6775.603 | 29026.9 | 28896.2 |
| 23 | a | 4 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 13551.206 | 29026.9 | 29835.0 |
| 23 | a | 8 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 27102.412 | 29026.9 | 34609.9 |
| 23 | b | -8 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -27102.412 | 27497.5 | 35892.2 |
| 23 | b | -4 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -13551.206 | 27497.5 | 29848.0 |
| 23 | b | -2 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -6775.603 | 27497.5 | 28107.3 |
| 23 | b | -1 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | -3387.802 | 27497.5 | 27649.1 |
| 23 | b | 0 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 0.000 | 27497.5 | 27497.5 |
| 23 | b | 1 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 3387.802 | 27497.5 | 27642.4 |
| 23 | b | 2 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 6775.603 | 27497.5 | 28064.0 |
| 23 | b | 4 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 13551.206 | 27497.5 | 29712.0 |
| 23 | b | 8 | True | [216, 1] | [214, 215, 216] | [214, 215, 216] | 27102.412 | 27497.5 | 35639.7 |

## Action counts by layer, prompt, and alpha

| layer | prompt | alpha | LEFT | RIGHT | UP | DOWN | parse_fail | n | freq(UP) | freq(DOWN) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | a | -8 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 7 | a | -4 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 7 | a | -2 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 7 | a | -1 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 7 | a | 0 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 7 | a | 1 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 7 | a | 2 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 7 | a | 4 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 7 | a | 8 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 7 | b | -8 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 7 | b | -4 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 7 | b | -2 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 7 | b | -1 | 0 | 1 | 29 | 0 | 0 | 30 | 0.967 | 0.000 |
| 7 | b | 0 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 7 | b | 1 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 7 | b | 2 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 7 | b | 4 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 7 | b | 8 | 0 | 1 | 29 | 0 | 0 | 30 | 0.967 | 0.000 |
| 15 | a | -8 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 15 | a | -4 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 15 | a | -2 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 15 | a | -1 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 15 | a | 0 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 15 | a | 1 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 15 | a | 2 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 15 | a | 4 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 15 | a | 8 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 15 | b | -8 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 15 | b | -4 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 15 | b | -2 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 15 | b | -1 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 15 | b | 0 | 0 | 2 | 28 | 0 | 0 | 30 | 0.933 | 0.000 |
| 15 | b | 1 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 15 | b | 2 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 15 | b | 4 | 0 | 1 | 29 | 0 | 0 | 30 | 0.967 | 0.000 |
| 15 | b | 8 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 23 | a | -8 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 23 | a | -4 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 23 | a | -2 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 23 | a | -1 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 23 | a | 0 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 23 | a | 1 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 23 | a | 2 | 0 | 19 | 11 | 0 | 0 | 30 | 0.367 | 0.000 |
| 23 | a | 4 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 23 | a | 8 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 23 | b | -8 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 23 | b | -4 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 23 | b | -2 | 0 | 30 | 0 | 0 | 0 | 30 | 0.000 | 0.000 |
| 23 | b | -1 | 0 | 23 | 7 | 0 | 0 | 30 | 0.233 | 0.000 |
| 23 | b | 0 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 23 | b | 1 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 23 | b | 2 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 23 | b | 4 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 23 | b | 8 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |

## UP/DOWN frequency curves

### Layer 7

| alpha | A freq(UP) | A freq(DOWN) | B freq(UP) | B freq(DOWN) |
|---:|---:|---:|---:|---:|
| -8 | 0.000 | 0.000 | 1.000 | 0.000 |
| -4 | 0.000 | 0.000 | 1.000 | 0.000 |
| -2 | 0.000 | 0.000 | 1.000 | 0.000 |
| -1 | 0.000 | 0.000 | 0.967 | 0.000 |
| 0 | 0.000 | 0.000 | 1.000 | 0.000 |
| 1 | 0.000 | 0.000 | 1.000 | 0.000 |
| 2 | 0.000 | 0.000 | 1.000 | 0.000 |
| 4 | 0.000 | 0.000 | 1.000 | 0.000 |
| 8 | 0.000 | 0.000 | 0.967 | 0.000 |

### Layer 15

| alpha | A freq(UP) | A freq(DOWN) | B freq(UP) | B freq(DOWN) |
|---:|---:|---:|---:|---:|
| -8 | 0.000 | 0.000 | 1.000 | 0.000 |
| -4 | 0.000 | 0.000 | 1.000 | 0.000 |
| -2 | 0.000 | 0.000 | 1.000 | 0.000 |
| -1 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0 | 0.000 | 0.000 | 0.933 | 0.000 |
| 1 | 0.000 | 0.000 | 1.000 | 0.000 |
| 2 | 0.000 | 0.000 | 1.000 | 0.000 |
| 4 | 0.000 | 0.000 | 0.967 | 0.000 |
| 8 | 0.000 | 0.000 | 1.000 | 0.000 |

### Layer 23

| alpha | A freq(UP) | A freq(DOWN) | B freq(UP) | B freq(DOWN) |
|---:|---:|---:|---:|---:|
| -8 | 0.000 | 0.000 | 0.000 | 0.000 |
| -4 | 0.000 | 0.000 | 0.000 | 0.000 |
| -2 | 0.000 | 0.000 | 0.000 | 0.000 |
| -1 | 0.000 | 0.000 | 0.233 | 0.000 |
| 0 | 0.000 | 0.000 | 1.000 | 0.000 |
| 1 | 0.000 | 0.000 | 1.000 | 0.000 |
| 2 | 0.367 | 0.000 | 1.000 | 0.000 |
| 4 | 1.000 | 0.000 | 1.000 | 0.000 |
| 8 | 1.000 | 0.000 | 1.000 | 0.000 |

## Layer comparison

Positive alpha means more has_key=True. For prompt A, expected steering is RIGHT to UP as alpha increases. For prompt B, expected steering is UP to RIGHT as alpha decreases, equivalently RIGHT to UP as alpha increases.

| layer | A UP monotone up | A RIGHT monotone down | B UP monotone up | B RIGHT monotone down | A UP slope | B UP slope | verdict |
|---:|---|---|---|---|---:|---:|---|
| 7 | True | True | False | False | +0.000 | -0.001 | not monotonic |
| 15 | True | True | False | False | +0.000 | -0.001 | not monotonic |
| 23 | True | True | True | True | +0.075 | +0.087 | monotonic steering |

## Files

- Paired prompts: `flag_swap/door_key_grid/final_channel_last3_v2/probe_family_matched/paired_probe_prompts.jsonl`
- Captured activations: `flag_swap/door_key_grid/final_channel_last3_v2/probe_family_matched/paired_probe_activations.pt`
- Probe metrics: `flag_swap/door_key_grid/final_channel_last3_v2/probe_family_matched/probe_metrics.json`
- Steering results: `flag_swap/door_key_grid/final_channel_last3_v2/probe_family_matched/steering_results.jsonl`
