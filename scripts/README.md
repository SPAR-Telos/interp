# Telos Interp Scripts

This directory contains scripts for training, evaluating, and analyzing decoder probes for action sequence prediction.

## evaluate_sequence_length_prediction.py

**NEW**: Evaluates the accuracy of a trained decoder probe on predicting sequence lengths (number of actions) compared to ground truth.

### Quick Start

```bash
# Download model and evaluate on all grid sizes
python scripts/evaluate_sequence_length_prediction.py \
    --hf-model-repo project-telos/interp \
    --hf-model-file decoder_reasoning_probe/decoder_probe_layer15_full.pt \
    --layer 15 \
    --grid-sizes 7 9 11 13 15 \
    --output results/length_prediction_results.json
```

### What It Does

1. Downloads/loads the trained reasoning probe
2. Loads test trajectories from HuggingFace to get ground truth sequence lengths
3. Loads activations and runs the probe to predict action sequences
4. Compares predicted lengths vs ground truth lengths
5. Reports metrics: MAE, RMSE, accuracy, correlation, error distribution

### Key Metrics

- **MAE**: Mean Absolute Error between predicted and true lengths
- **RMSE**: Root Mean Squared Error
- **Exact Match Accuracy**: % where predicted length exactly equals ground truth
- **Within-1/2 Accuracy**: % where prediction is within 1 or 2 steps
- **Correlation**: Pearson correlation between predicted and true lengths

See [SEQUENCE_LENGTH_EVALUATION.md](./SEQUENCE_LENGTH_EVALUATION.md) for detailed documentation.

---

## process_action_sequences.py

Processes trajectory JSON files from HuggingFace to extract action sequences.

### Installation

```bash
pip install datasets torch tqdm
```

### Usage

Process both train and val splits (recommended):
```bash
python scripts/process_action_sequences.py --split both
```

Process only train split:
```bash
python scripts/process_action_sequences.py --split train
```

Process only val split:
```bash
python scripts/process_action_sequences.py --split val
```

### Output

The script generates:

1. **`data/processed/action_sequences_{split}.pt`**: PyTorch file with:
   - `sequences`: Dict mapping trajectory index → action tensor
   - `stats`: Statistics about sequence lengths
   - `metadata`: Grid params and model params for each trajectory

2. **`data/processed/action_sequences_{split}_stats.json`**: Human-readable statistics:
   - Length distribution
   - Min/max length examples
   - Failed trajectories (that hit max steps)
   - Action mapping (string → integer)

### Action Mapping

```python
ACTION_MAP = {
    "LEFT": 0,
    "RIGHT": 1,
    "UP": 2,
    "DOWN": 3,
}
```

### Example Output

```
================================================================================
STATISTICS FOR TRAIN SPLIT
================================================================================
Total trajectories: 3600
Failed trajectories (hit max steps): 450 (12.5%)
Min length: 2 (from trajectories: [15, 42, 89, ...])
Max length: 50 (from trajectories: [1, 23, 67, ...])
Mean length: 12.34
Median length: 10

Length distribution:
  Length  2:   45 trajectories ( 1.25%) ██
  Length  3:  120 trajectories ( 3.33%) ███
  Length  4:  200 trajectories ( 5.56%) █████
  ...
  Length 50:  450 trajectories (12.50%) ████████████ (MAX - likely failed)

✓ Recommended max_path_len for model: 50
```

### Using the Processed Data

```python
import torch

# Load processed sequences
data = torch.load('data/processed/action_sequences_train.pt')

sequences = data['sequences']  # Dict[int, torch.Tensor]
stats = data['stats']
metadata = data['metadata']

# Get action sequence for trajectory 0
actions = sequences[0]  # Tensor of shape (seq_len,)
print(f"Actions: {actions.tolist()}")  # [0, 1, 2, 3, ...]

# Get recommended max_path_len
max_path_len = stats['max_length']
print(f"Set max_path_len={max_path_len} in your model")
```

### Next Steps

After running this script:

1. Check the statistics to determine appropriate `max_path_len`
2. Consider filtering out failed trajectories (length == max_steps)
3. Use the processed sequences in your training script:

```python
from telos_interp.decoder_probe import OneShotPrefixProbe, train_decoder_probe

# Load processed data
train_data = torch.load('data/processed/action_sequences_train.pt')
max_path_len = train_data['stats']['max_length']

# Create model with correct max_path_len
model = OneShotPrefixProbe(
    activation_dim=2880,
    hidden_dim=1024,
    max_path_len=max_path_len,  # e.g., 50
    num_memory_tokens=3,
)

# Train model (you need to load activations separately)
# ...
```
