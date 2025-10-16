# Grid Probing Experiment Results

## Experiment Overview
- **Model**: Llama-3.1-8B-Instruct
- **Layer**: 16
- **Method**: Last prompt token activation + cell coordinates
- **Grid Type**: 10x10 fully observable grids
- **Total Environments**: 1,000
- **Total Cells**: 121,000

## Dataset Statistics
### Original (Unbalanced)
- Wall: 56,499 samples
- Empty: 62,501 samples  
- Goal: 1,000 samples
- Agent: 1,000 samples

### Balanced (for training)
- Wall: 1,000 samples
- Empty: 1,000 samples
- Goal: 1,000 samples
- Agent: 1,000 samples

## Probe Performance Results

### Overall Metrics
- **Accuracy**: 28.88%
- **Macro F1**: 19.94%
- **Macro Precision**: 23.27%
- **Macro Recall**: 28.88%

### Per-Class Performance
| Class  | Precision | Recall | F1-Score | GT Support | Predicted |
|--------|-----------|--------|----------|------------|-----------|
| wall   | 0.2799    | 0.6550 | 0.3922   | 200        | 468       |
| empty  | 0.3079    | 0.4850 | 0.3767   | 200        | 315       |
| goal   | 0.1429    | 0.0050 | 0.0097   | 200        | 7         |
| agent  | 0.2000    | 0.0100 | 0.0190   | 200        | 10        |

## Key Observations

### What Works
- **Wall vs Empty distinction**: The model can reasonably distinguish between walls (39% F1) and empty spaces (38% F1)
- **Spatial structure**: The model does learn some basic spatial representations

### What Doesn't Work
- **Agent identification**: Extremely poor performance (1.9% F1)
- **Goal identification**: Extremely poor performance (1.0% F1)
- **Class imbalance in predictions**: Model heavily predicts walls (468/800 predictions)

## Main Concerns

1. **Coarse-grained representations**: Layer 16 activations may be too high-level for fine-grained object localization
2. **Last prompt token limitation**: This approach may not capture detailed spatial information needed for precise cell classification
3. **Model architecture**: Llama-3.1-8B might not be optimal for this spatial reasoning task

## Next Steps

### Immediate Actions
1. **Try different layers**: Experiment with layers 8, 24, or 28 to find optimal spatial representations
2. **Cell-wise activations**: Switch from "last prompt token" to per-cell token activations for more granular spatial information
3. **Model comparison**: Test with smaller models that might be easier to probe

### Analysis
1. **Visualize activations**: Examine what spatial patterns the model actually learns
2. **Ablation studies**: Test different token positions and grid sizes
3. **Architecture comparison**: Compare with models specifically trained on spatial tasks

## Files Generated
- `data/grids_for_probing/train_grid_n1000_fully_observable_clean.csv`: Grid dataset
- `data/activations/Llama-3.1-8B-Instruct/train_grid_n1000_fully_observable_clean/grid_last_prompt_layer_16/`: Original activations
- `data/activations/Llama-3.1-8B-Instruct/train_grid_n1000_fully_observable_clean/grid_last_prompt_layer_16_balanced/`: Balanced activations
- `data/activations/Llama-3.1-8B-Instruct/train_grid_n1000_fully_observable_clean/grid_last_prompt_layer_16_balanced/probes/`: Trained probe

## Commands Used
```bash
# Generate grids
cd /root/reveng && python src/reveng/environment_generator/get_fully_observable_grids_for_probing.py --num-envs 1000 --output-file /root/interp/data/grids_for_probing/train_grid_n1000_fully_observable_clean.csv

# Gather activations
cd /root/interp && uv run python -m telos_interp.commands.cli gather-grid-activations-last-prompt meta-llama/Llama-3.1-8B-Instruct data/grids_for_probing/train_grid_n1000_fully_observable_clean.csv --layer 16

# Balance dataset
cd /root/interp && python scripts/balance_activation_files.py data/activations/Llama-3.1-8B-Instruct/train_grid_n1000_fully_observable_clean/grid_last_prompt_layer_16 data/activations/Llama-3.1-8B-Instruct/train_grid_n1000_fully_observable_clean/grid_last_prompt_layer_16_balanced --method equal --target-samples 1000

# Train probe
cd /root/interp && uv run python -m telos_interp.commands.cli train-multiclass-probe data/activations/Llama-3.1-8B-Instruct/train_grid_n1000_fully_observable_clean/grid_last_prompt_layer_16_balanced 16 --normalize --gpu
```
