# activation_patch_counterfactuals

Run online activation patching between original and counterfactual trajectories.

The command patches semantic boundary tokens rather than arbitrary token slices:

- `prompt_boundary`: the `"<|end|>"`, `"<|start|>"`, `"assistant"` triplet in `prompt_suffix_tokens`
- `pre_final_boundary`: the same triplet immediately before `"<|channel|>", "final", "<|message|>"` in `output_tokens`
- `modified_grid_cells`: only the two moved grid cells from the counterfactual metadata (`from_position`, `to_position`)
- `all_grid_tokens`: every token in `grid_state_tokens`, i.e. the full serialized grid span

`modified_grid_cells` requires exactly one `grid_tile` token per grid cell in `grid_state_tokens`; otherwise the run will fail fast rather than guess a token mapping.

Example:

```bash
interp-cli activation_patch_counterfactuals \
    --layers 7,15,23 \
    --patch-sites prompt_boundary,pre_final_boundary \
    --output-dir data/activation_patching
```

The default `--evaluation-mode answer_forcing` uses teacher forcing up to the action token and reads next-token probabilities over the action vocabulary. Use `--evaluation-mode free_generation` to recover the older full-generation behavior.

Artifacts:

- `manifest.json`
- `runs.jsonl`
- `summary_by_group.csv`
- `parse_failures.jsonl`

By default, runs are skipped early when either the source recorded action or the target recorded action is not already in that trajectory's optimal action set. Disable this with `--require-recorded-actions-in-optimal-set false`.

If `runs.jsonl` already exists in `--output-dir`, the command resumes automatically and skips completed `counterfactual_id × step × direction × patch_site × layer` runs. New records are appended as they finish.
