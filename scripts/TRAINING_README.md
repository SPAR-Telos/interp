# OneShotPrefixProbe Training Guide

## Overview

Train the OneShotPrefixProbe model to predict action sequences from LLM activations using the HuggingFace datasets.

## Quick Start

### Test Run (20 trajectories, 5 epochs)
```bash
cd /root/telos/interp
python scripts/train_decoder_from_hf.py \
  --max-trajectories 20 \
  --num-epochs 5 \
  --batch-size 8 \
  --layer 23 \
  --max-seq-len 20
```

### Full Training (All ~3000 trajectories)
```bash
cd /root/telos/interp
python scripts/train_decoder_from_hf.py \
  --layer 23 \
  --max-seq-len 20 \
  --batch-size 32 \
  --num-epochs 10 \
  --learning-rate 1e-4 \
  --save-path models/decoder_probe_layer23_full.pt
```

### Training Different Layers
```bash
# Layer 7 (early layers)
python scripts/train_decoder_from_hf.py --layer 7 --save-path models/decoder_probe_layer7.pt

# Layer 15 (middle layers)
python scripts/train_decoder_from_hf.py --layer 15 --save-path models/decoder_probe_layer15.pt

# Layer 23 (late layers)
python scripts/train_decoder_from_hf.py --layer 23 --save-path models/decoder_probe_layer23.pt
```

## Key Arguments

### Data Loading
- `--activations-repo`: HuggingFace repo with activations (default: `project-telos/train_and_val_activations`)
- `--trajectories-repo`: HuggingFace repo with trajectory JSONs (default: `project-telos/train_and_val_full_trajectories`)
- `--layer`: Layer number to extract activations from (7, 15, or 23)
- `--max-seq-len`: Maximum action sequence length (default: 20). Longer sequences are skipped.
- `--max-trajectories`: Limit number of trajectories for testing (default: None = all)

### Model Architecture
- `--hidden-dim`: Hidden dimension for the probe (default: 1024)
- `--n-layers`: Number of transformer decoder layers (default: 4)
- `--n-heads`: Number of attention heads (default: 8)
- `--position-loss-decay`: Position-weighted loss decay factor (default: 0.95)
  - Lower values (e.g., 0.90) → focus more on early positions
  - Higher values (e.g., 0.98) → more uniform weighting

### Training
- `--batch-size`: Batch size (default: 32)
- `--num-epochs`: Number of epochs (default: 10)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--device`: Device to train on (`cuda` or `cpu`, default: auto-detect)
- `--save-path`: Path to save trained model (default: `models/decoder_probe_layer23.pt`)

## Output Files

After training, you'll get:
1. **Model file**: `models/decoder_probe_layer23.pt` (or your specified path)
2. **Statistics file**: `models/decoder_probe_layer23_stats.json` with:
   - Training arguments
   - Data loading statistics
   - Training metrics (loss, accuracy)

## Understanding the Metrics

- **Token Accuracy**: Percentage of individual actions predicted correctly
- **Sequence Accuracy**: Percentage of entire sequences predicted perfectly
- **Train/Eval Loss**: Lower is better

For long-tail distributions (median=6, max=20), expect:
- Token accuracy to be higher for early positions
- Lower sequence accuracy initially (improves with more data/epochs)

## Model Architecture

The OneShotPrefixProbe uses:
- **Input**: 3 activation vectors (each 2880-dim) from layer 23
- **Feature Mapper**: Projects activations to 1024-dim bottleneck
- **Learned Queries**: 20 position-specific query vectors
- **Transformer Decoder**: 4 layers, 8 heads, cross-attention
- **Action Head**: Predicts 5 action classes per position

### Long-Tail Optimizations

The model prioritizes early positions (0-15) using:
1. **Position-weighted loss**: Early positions get higher weight (exponential decay)
2. **Priority initialization**: First 15 queries have stronger initialization
3. **Position embeddings**: Each query slot learns position-specific behavior

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python scripts/train_decoder_from_hf.py --batch-size 16  # or 8, 4
```

### Loading Too Slow
The script downloads activation files from HuggingFace. First run will be slow but files are cached locally.

### No Training Data Loaded
Check:
1. Trajectories have matching activation folders
2. Action sequences are ≤ `max-seq-len`
3. HuggingFace authentication if repos are private

## Example: Full Pipeline

```bash
# 1. Train probe on layer 23
python scripts/train_decoder_from_hf.py \
  --layer 23 \
  --max-seq-len 20 \
  --batch-size 32 \
  --num-epochs 20 \
  --save-path models/probe_layer23_full.pt

# 2. Check training statistics
cat models/probe_layer23_full_stats.json

# 3. Use the trained model (in your code)
import torch
from telos_interp.decoder_probe import OneShotPrefixProbe

model = OneShotPrefixProbe(max_path_len=20)
model.load_state_dict(torch.load('models/probe_layer23_full.pt'))
model.eval()

# Make predictions
with torch.no_grad():
    activations = ...  # (batch, 3, 2880)
    predictions = model.generate(activations)  # (batch, 20)
```

## Notes

- The script automatically uses a separate validation set (no need for train/val split)
- Sequences longer than `max-seq-len` are automatically filtered out
- The model uses padding tokens (`-100`) for sequences shorter than `max_path_len`
- Training progress is shown with a progress bar and epoch metrics
