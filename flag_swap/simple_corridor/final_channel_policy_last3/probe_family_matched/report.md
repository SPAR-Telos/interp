# Final-channel simple-corridor has-key direction steering across layers

## Dataset and probes

- Paired synthetic dataset: 12 prompts (6 template states; probe family final-channel-policy).
- Class balance: False=6, True=6.
- Captured features: prompt_suffix_indices='-3:-1', relative indices [0, 1, 2], flattened to 8640 dimensions.

| layer | eval acc | eval balanced acc | all-state train acc | class gap | direction norm |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.750 | 0.750 | 1.000 | 15.652 | 1.000 |
| 15 | 0.500 | 0.500 | 1.000 | 125.942 | 1.000 |
| 23 | 0.750 | 0.750 | 1.000 | 1572.378 | 1.000 |

## Prompt family

The simple-corridor run uses a compact final-channel policy abstraction of the corridor. Prompt A is the no-key condition, whose baseline action is DOWN; prompt B is the has-key condition, whose baseline action is UP. The policy is equivalent to the corridor below: without the key, move down toward K; with the key, move up toward D and G.

```text
# # #
# G #
# _ #
# D #
# A #
# _ #
# K #
# # #
```

This is the matched final-channel simple-corridor policy family, not the long verbose reasoning prompt. That distinction matters: earlier replacement/transplant-style runs on longer prompts did not produce a clean causal action switch. The finding reported here is specifically that additive probe-direction steering can control the first action token when the action is emitted immediately after the assistant final-channel suffix.

## Detailed finding

This run tests whether the final three prompt-suffix activations can causally control the emitted action when we add the direction learned by a linear `has_key` probe. The intervention is not a donor activation transplant: for each selected token position, the hook adds `alpha * class_gap * direction[layer, token_position]` to the layer output and leaves every other prompt and generation position unchanged.

The selected locations are exactly `prompt_suffix_indices='-3:-1'`. In this target prompt family those are suffix-relative indices [0, 1, 2]; for the target A/B prompts the hook logs resolve them to absolute positions `[72, 73, 74]`, the final-channel suffix tokens `<|channel|>`, `final`, and `<|message|>`. MLX processes this prompt as one multi-token prefill followed by a one-token prompt singleton, so the first two selected suffix tokens are changed in the prefill call and the final selected suffix token is changed in the singleton call. The recorded hook logs below verify this split with `seq_lens=[74, 1]` and `patched_positions=[72, 73, 74]`.

The successful steering evidence is concentrated at layer 23. Positive alpha means moving in the `has_key=True` direction; therefore prompt A, whose baseline action is DOWN, should move toward UP as alpha increases. Negative alpha means moving away from `has_key=True`; therefore prompt B, whose baseline action is UP, should move toward DOWN as alpha decreases.

Layer 23 action distributions at the most diagnostic alpha values:

| prompt | alpha | distribution | freq(UP) | freq(DOWN) |
|---|---:|---|---:|---:|
| a | -8 | DOWN=30/30 | 0.000 | 1.000 |
| a | -4 | DOWN=30/30 | 0.000 | 1.000 |
| a | -2 | DOWN=30/30 | 0.000 | 1.000 |
| a | 0 | DOWN=30/30 | 0.000 | 1.000 |
| a | 2 | UP=4/30, DOWN=26/30 | 0.133 | 0.867 |
| a | 4 | UP=13/30, DOWN=17/30 | 0.433 | 0.567 |
| a | 8 | UP=30/30 | 1.000 | 0.000 |
| b | -8 | UP=1/30, DOWN=29/30 | 0.033 | 0.967 |
| b | -4 | UP=23/30, DOWN=7/30 | 0.767 | 0.233 |
| b | -2 | UP=29/30, DOWN=1/30 | 0.967 | 0.033 |
| b | 0 | UP=30/30 | 1.000 | 0.000 |
| b | 2 | UP=30/30 | 1.000 | 0.000 |
| b | 4 | UP=30/30 | 1.000 | 0.000 |
| b | 8 | UP=30/30 | 1.000 | 0.000 |

Layer 23 hook evidence for the endpoint flips:

| prompt | alpha | mode | selected positions | patched positions | seq lens | scale | pre norm | post norm |
|---|---:|---|---|---|---|---:|---:|---:|
| a | 8 | add | [72, 73, 74] | [72, 73, 74] | [74, 1] | 12579.020 | 33174.4 | 33800.7 |
| b | -8 | add | [72, 73, 74] | [72, 73, 74] | [74, 1] | -12579.020 | 33143.1 | 34203.2 |

Endpoint summary: at layer 23, prompt A changes from DOWN=30/30 at alpha 0 to UP=30/30 at alpha 8. Prompt B changes from UP=30/30 at alpha 0 to UP=1/30, DOWN=29/30 at alpha -8. This is the expected bidirectional pattern for a `has_key` direction: adding it makes a no-key prompt behave like has-key, while subtracting it makes a has-key prompt behave like no-key.

Layer comparison matters for interpretation. The other tested layers do not provide a clean bidirectional action-control result in this sweep. Layer 23 is the only layer that satisfies the monotonic steering criterion across the sampled alphas and produces clean endpoint flips without parse failures at the decisive A alpha 8 and B alpha -8 endpoints.

Scope and caveats: this is evidence for a causal effect in the matched final-channel simple-corridor policy family, where the first generated content is the actual action token. It does not claim that the same single probe direction will steer every verbose grid prompt or every reasoning format. The result also depends on relatively large layer-23 scales because the direction is a unit activation-space probe direction multiplied by the probe class gap and alpha. Within this scoped setting, however, the hook logs and action distributions support the causal claim: additive steering of only the final three suffix activations along the learned `has_key` direction is sufficient to change the emitted action.

## Hook sanity

| layer | prompt | alpha | fired | seq lens | selected positions | patched positions | scale | pre norm | post norm |
|---:|---|---:|---|---|---|---|---:|---:|---:|
| 7 | a | -8 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -125.219 | 604.9 | 612.3 |
| 7 | a | -4 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -62.610 | 604.9 | 607.2 |
| 7 | a | -2 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -31.305 | 604.9 | 605.7 |
| 7 | a | -1 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -15.652 | 604.9 | 605.3 |
| 7 | a | 0 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 0.000 | 604.9 | 604.9 |
| 7 | a | 1 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 15.652 | 604.9 | 604.8 |
| 7 | a | 2 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 31.305 | 604.9 | 604.9 |
| 7 | a | 4 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 62.610 | 604.9 | 605.6 |
| 7 | a | 8 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 125.219 | 604.9 | 609.2 |
| 7 | b | -8 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -125.219 | 603.5 | 609.6 |
| 7 | b | -4 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -62.610 | 603.5 | 605.1 |
| 7 | b | -2 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -31.305 | 603.5 | 603.9 |
| 7 | b | -1 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -15.652 | 603.5 | 603.7 |
| 7 | b | 0 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 0.000 | 603.5 | 603.5 |
| 7 | b | 1 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 15.652 | 603.5 | 603.5 |
| 7 | b | 2 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 31.305 | 603.5 | 603.8 |
| 7 | b | 4 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 62.610 | 603.5 | 604.8 |
| 7 | b | 8 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 125.219 | 603.5 | 609.0 |
| 15 | a | -8 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -1007.534 | 3016.7 | 3100.3 |
| 15 | a | -4 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -503.767 | 3016.7 | 3044.6 |
| 15 | a | -2 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -251.883 | 3016.7 | 3027.4 |
| 15 | a | -1 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -125.942 | 3016.7 | 3021.0 |
| 15 | a | 0 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 0.000 | 3016.7 | 3016.7 |
| 15 | a | 1 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 125.942 | 3016.7 | 3014.2 |
| 15 | a | 2 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 251.883 | 3016.7 | 3013.0 |
| 15 | a | 4 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 503.767 | 3016.7 | 3017.4 |
| 15 | a | 8 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 1007.534 | 3016.7 | 3046.4 |
| 15 | b | -8 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -1007.534 | 3038.8 | 3109.8 |
| 15 | b | -4 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -503.767 | 3038.8 | 3060.2 |
| 15 | b | -2 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -251.883 | 3038.8 | 3046.5 |
| 15 | b | -1 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -125.942 | 3038.8 | 3041.5 |
| 15 | b | 0 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 0.000 | 3038.8 | 3038.8 |
| 15 | b | 1 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 125.942 | 3038.8 | 3037.8 |
| 15 | b | 2 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 251.883 | 3038.8 | 3038.2 |
| 15 | b | 4 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 503.767 | 3038.8 | 3045.6 |
| 15 | b | 8 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 1007.534 | 3038.8 | 3080.4 |
| 23 | a | -8 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -12579.020 | 33174.4 | 34888.7 |
| 23 | a | -4 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -6289.510 | 33174.4 | 33751.9 |
| 23 | a | -2 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -3144.755 | 33174.4 | 33389.5 |
| 23 | a | -1 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -1572.378 | 33174.4 | 33262.7 |
| 23 | a | 0 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 0.000 | 33174.4 | 33174.4 |
| 23 | a | 1 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 1572.378 | 33174.4 | 33123.0 |
| 23 | a | 2 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 3144.755 | 33174.4 | 33107.8 |
| 23 | a | 4 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 6289.510 | 33174.4 | 33190.0 |
| 23 | a | 8 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 12579.020 | 33174.4 | 33800.7 |
| 23 | b | -8 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -12579.020 | 33143.1 | 34203.2 |
| 23 | b | -4 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -6289.510 | 33143.1 | 33382.9 |
| 23 | b | -2 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -3144.755 | 33143.1 | 33187.9 |
| 23 | b | -1 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | -1572.378 | 33143.1 | 33147.2 |
| 23 | b | 0 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 0.000 | 33143.1 | 33143.1 |
| 23 | b | 1 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 1572.378 | 33143.1 | 33176.4 |
| 23 | b | 2 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 3144.755 | 33143.1 | 33247.9 |
| 23 | b | 4 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 6289.510 | 33143.1 | 33499.7 |
| 23 | b | 8 | True | [74, 1] | [72, 73, 74] | [72, 73, 74] | 12579.020 | 33143.1 | 34436.5 |

## Action counts by layer, prompt, and alpha

| layer | prompt | alpha | LEFT | RIGHT | UP | DOWN | parse_fail | n | freq(UP) | freq(DOWN) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | a | -8 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 7 | a | -4 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 7 | a | -2 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 7 | a | -1 | 0 | 0 | 1 | 29 | 0 | 30 | 0.033 | 0.967 |
| 7 | a | 0 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 7 | a | 1 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 7 | a | 2 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 7 | a | 4 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 7 | a | 8 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 7 | b | -8 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 7 | b | -4 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 7 | b | -2 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 7 | b | -1 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 7 | b | 0 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 7 | b | 1 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 7 | b | 2 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 7 | b | 4 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 7 | b | 8 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 15 | a | -8 | 1 | 0 | 1 | 19 | 9 | 30 | 0.033 | 0.633 |
| 15 | a | -4 | 0 | 0 | 0 | 29 | 1 | 30 | 0.000 | 0.967 |
| 15 | a | -2 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 15 | a | -1 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 15 | a | 0 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 15 | a | 1 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 15 | a | 2 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 15 | a | 4 | 0 | 0 | 1 | 29 | 0 | 30 | 0.033 | 0.967 |
| 15 | a | 8 | 0 | 0 | 2 | 28 | 0 | 30 | 0.067 | 0.933 |
| 15 | b | -8 | 0 | 0 | 13 | 4 | 13 | 30 | 0.433 | 0.133 |
| 15 | b | -4 | 0 | 0 | 14 | 3 | 13 | 30 | 0.467 | 0.100 |
| 15 | b | -2 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 15 | b | -1 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 15 | b | 0 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 15 | b | 1 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 15 | b | 2 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 15 | b | 4 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 15 | b | 8 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 23 | a | -8 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 23 | a | -4 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 23 | a | -2 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 23 | a | -1 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 23 | a | 0 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 23 | a | 1 | 0 | 0 | 0 | 30 | 0 | 30 | 0.000 | 1.000 |
| 23 | a | 2 | 0 | 0 | 4 | 26 | 0 | 30 | 0.133 | 0.867 |
| 23 | a | 4 | 0 | 0 | 13 | 17 | 0 | 30 | 0.433 | 0.567 |
| 23 | a | 8 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 23 | b | -8 | 0 | 0 | 1 | 29 | 0 | 30 | 0.033 | 0.967 |
| 23 | b | -4 | 0 | 0 | 23 | 7 | 0 | 30 | 0.767 | 0.233 |
| 23 | b | -2 | 0 | 0 | 29 | 1 | 0 | 30 | 0.967 | 0.033 |
| 23 | b | -1 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 23 | b | 0 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 23 | b | 1 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 23 | b | 2 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 23 | b | 4 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |
| 23 | b | 8 | 0 | 0 | 30 | 0 | 0 | 30 | 1.000 | 0.000 |

## UP/DOWN frequency curves

### Layer 7

| alpha | A freq(UP) | A freq(DOWN) | B freq(UP) | B freq(DOWN) |
|---:|---:|---:|---:|---:|
| -8 | 0.000 | 1.000 | 1.000 | 0.000 |
| -4 | 0.000 | 1.000 | 1.000 | 0.000 |
| -2 | 0.000 | 1.000 | 1.000 | 0.000 |
| -1 | 0.033 | 0.967 | 1.000 | 0.000 |
| 0 | 0.000 | 1.000 | 1.000 | 0.000 |
| 1 | 0.000 | 1.000 | 1.000 | 0.000 |
| 2 | 0.000 | 1.000 | 1.000 | 0.000 |
| 4 | 0.000 | 1.000 | 1.000 | 0.000 |
| 8 | 0.000 | 1.000 | 1.000 | 0.000 |

### Layer 15

| alpha | A freq(UP) | A freq(DOWN) | B freq(UP) | B freq(DOWN) |
|---:|---:|---:|---:|---:|
| -8 | 0.033 | 0.633 | 0.433 | 0.133 |
| -4 | 0.000 | 0.967 | 0.467 | 0.100 |
| -2 | 0.000 | 1.000 | 1.000 | 0.000 |
| -1 | 0.000 | 1.000 | 1.000 | 0.000 |
| 0 | 0.000 | 1.000 | 1.000 | 0.000 |
| 1 | 0.000 | 1.000 | 1.000 | 0.000 |
| 2 | 0.000 | 1.000 | 1.000 | 0.000 |
| 4 | 0.033 | 0.967 | 1.000 | 0.000 |
| 8 | 0.067 | 0.933 | 1.000 | 0.000 |

### Layer 23

| alpha | A freq(UP) | A freq(DOWN) | B freq(UP) | B freq(DOWN) |
|---:|---:|---:|---:|---:|
| -8 | 0.000 | 1.000 | 0.033 | 0.967 |
| -4 | 0.000 | 1.000 | 0.767 | 0.233 |
| -2 | 0.000 | 1.000 | 0.967 | 0.033 |
| -1 | 0.000 | 1.000 | 1.000 | 0.000 |
| 0 | 0.000 | 1.000 | 1.000 | 0.000 |
| 1 | 0.000 | 1.000 | 1.000 | 0.000 |
| 2 | 0.133 | 0.867 | 1.000 | 0.000 |
| 4 | 0.433 | 0.567 | 1.000 | 0.000 |
| 8 | 1.000 | 0.000 | 1.000 | 0.000 |

## Layer comparison

Positive alpha means more has_key=True. For prompt A, expected steering is DOWN to UP as alpha increases. For prompt B, expected steering is UP to DOWN as alpha decreases, equivalently DOWN to UP as alpha increases.

| layer | A UP monotone up | A DOWN monotone down | B UP monotone up | B DOWN monotone down | A UP slope | B UP slope | verdict |
|---:|---|---|---|---|---:|---:|---|
| 7 | False | False | True | True | -0.000 | +0.000 | not monotonic |
| 15 | False | False | True | True | +0.002 | +0.039 | not monotonic |
| 23 | True | True | True | True | +0.059 | +0.051 | monotonic steering |

## Files

- Paired prompts: `flag_swap/simple_corridor/final_channel_policy_last3/probe_family_matched/paired_probe_prompts.jsonl`
- Captured activations: `flag_swap/simple_corridor/final_channel_policy_last3/probe_family_matched/paired_probe_activations.pt`
- Probe metrics: `flag_swap/simple_corridor/final_channel_policy_last3/probe_family_matched/probe_metrics.json`
- Steering results: `flag_swap/simple_corridor/final_channel_policy_last3/probe_family_matched/steering_results.jsonl`
