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
