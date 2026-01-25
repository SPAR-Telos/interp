#!/bin/bash
# Train a NEW decoder probe from scratch on full training dataset
# Architecture: max_path_len=10, with EOS tokens, layer 15 output reasoning activations
# Includes automatic test set evaluation after training

cd /root/telos/interp
source .venv/bin/activate

# Full training on layer 15 with EOS tokens and test evaluation
python scripts/train_decoder_from_hf.py \
  --activations-repo project-telos/train_and_val_activations \
  --trajectories-repo project-telos/train_and_val_full_trajectories \
  --test-activations-repo project-telos/activations_test_full \
  --test-trajectories-repo project-telos/trajectories_test_full \
  --layer 15 \
  --max-seq-len 10 \
  --batch-size 32 \
  --num-epochs 20 \
  --learning-rate 1e-4 \
  --position-loss-decay 0.90 \
  --save-path models/decoder_probe_layer15_max10_with_eos.pt

echo ""
echo "============================================"
echo "Training complete!"
echo "============================================"
echo "Model saved to: models/decoder_probe_layer15_max10_with_eos.pt"
echo "Stats saved to: models/decoder_probe_layer15_max10_with_eos_stats.json"
echo ""
echo "The stats file includes:"
echo "  - Training metrics (loss, accuracy)"
echo "  - Validation metrics"
echo "  - Test set evaluation results"
echo "============================================"
