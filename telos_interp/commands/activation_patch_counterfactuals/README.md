# activation_patch_counterfactuals

Run online activation patching between original and counterfactual trajectories.

The command patches semantic boundary tokens rather than arbitrary token slices:

- `prompt_boundary`: the `"<|end|>"`, `"<|start|>"`, `"assistant"` triplet in `prompt_suffix_tokens`
- `pre_final_boundary`: the same triplet immediately before `"<|channel|>", "final", "<|message|>"` in `output_tokens`

Example:

```bash
interp-cli activation_patch_counterfactuals \
    --layers 7,15,23 \
    --patch-sites prompt_boundary,pre_final_boundary \
    --output-dir data/activation_patching
```

Artifacts:

- `manifest.json`
- `runs.jsonl`
- `summary_by_group.csv`
- `parse_failures.jsonl`

By default, runs are skipped early when either the source recorded action or the target recorded action is not already in that trajectory's optimal action set. Disable this with `--require-recorded-actions-in-optimal-set false`.

If `runs.jsonl` already exists in `--output-dir`, the command resumes automatically and skips completed `counterfactual_id × step × direction × patch_site × layer` runs. New records are appended as they finish.
