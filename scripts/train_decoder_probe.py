#!/usr/bin/env python3
"""
Train decoder probe to predict action sequences from LLM activations.

This script:
1. Loads activations from local directory
2. Loads action sequences from CSV
3. Creates training/validation/test datasets
4. Trains the decoder probe
5. Evaluates on test set
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from telos_interp.decoder_probe import (
    EOS_TOKEN_ID,
    ActionDecoderProbe,
    create_dataset,
    load_action_sequences_from_csv,
    load_activations_from_local,
    train_decoder_probe,
)


def main():
    parser = argparse.ArgumentParser(description="Train decoder probe on activations and action sequences")
    parser.add_argument(
        "--activations-dir",
        type=str,
        required=True,
        help="Directory containing activation files (e.g., data/activations/gpt-oss-20b/7by7traingrids/full_prompt_activations)",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="Path to training CSV file with action_sequence column",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default=None,
        help="Path to validation CSV file (optional)",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="Path to test CSV file (optional)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=20,
        help="Layer number to use (default: 20)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=7,
        help="Vocabulary size (default: 7 for 0-3=actions, 4=SOS, 5=EOS, 6=PAD)",
    )
    parser.add_argument(
        "--add-eos-token",
        action="store_true",
        default=True,
        help="Add EOS token to end of action sequences (default: True)",
    )
    parser.add_argument(
        "--no-add-eos-token",
        dest="add_eos_token",
        action="store_false",
        help="Don't add EOS token to end of action sequences",
    )
    parser.add_argument(
        "--eos-weight-threshold",
        type=float,
        default=3.0,
        help="Threshold for position-weighted EOS loss (default: 3.0). Higher values allow earlier EOS.",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="HuggingFace repository ID for loading activations (e.g., 'project-telos/decoder'). If None, uses local filesystem.",
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=4,
        help="Number of decoder layers (default: 4)",
    )
    parser.add_argument(
        "--n-head",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)",
    )
    parser.add_argument(
        "--num-memory-tokens",
        type=int,
        default=8,
        help="Number of memory tokens to split activation into (default: 8). "
             "activation_dim must be divisible by this. E.g., 2880/8=360 dims per token.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=100,
        help="Maximum sequence length (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.2,
        help="Fraction of training data to use for validation if val-csv not provided (default: 0.2)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save trained model (default: models/decoder_probe_layer_{layer}.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (default: auto-detect: mps for macOS, cuda if available, else cpu). Use 'cpu' if you get CUDA errors.",
    )

    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        # Try MPS first (macOS Metal)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = "mps"
        # Try CUDA, but catch errors if it's not actually available
        elif torch.cuda.is_available():
            try:
                # Test if CUDA actually works
                test_tensor = torch.tensor([1.0]).cuda()
                args.device = "cuda"
            except (AssertionError, RuntimeError):
                # CUDA not actually available despite is_available() returning True
                args.device = "cpu"
        else:
            args.device = "cpu"
    
    # Safety check: if device is "cuda" but we're on macOS, force CPU
    import platform
    if args.device == "cuda" and platform.system() == "Darwin":
        print("Warning: CUDA not available on macOS. Using CPU instead.")
        args.device = "cpu"

    print("=" * 60)
    print("Training Decoder Probe")
    print("=" * 60)
    print(f"Activations directory: {args.activations_dir}")
    print(f"Training CSV: {args.train_csv}")
    print(f"Layer: {args.layer}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Step 1: Load activations
    print("\n📥 Step 1: Loading activations...")
    activations = load_activations_from_local(
        activations_dir=args.activations_dir,
        layer=args.layer,
        hf_repo_id=args.hf_repo_id,
    )
    print(f"Loaded {len(activations)} activations")

    # Step 2: Load action sequences from training CSV
    print("\n📥 Step 2: Loading action sequences from training CSV...")
    decoder_inputs, targets = load_action_sequences_from_csv(
        csv_path=args.train_csv,
        max_seq_len=args.max_seq_len,
        add_eos_token=args.add_eos_token,
    )
    print(f"Loaded {len(decoder_inputs)} decoder inputs and {len(targets)} targets")

    # Step 3: Create training dataset
    print("\n🔗 Step 3: Creating training dataset...")
    train_dataset = create_dataset(activations, decoder_inputs, targets, max_seq_len=args.max_seq_len)
    print(f"Created {len(train_dataset)} training pairs")

    # Step 4: Handle validation/test sets
    val_dataset = None
    test_dataset = None
    val_decoder_inputs = None
    val_targets = None
    test_decoder_inputs = None
    test_targets = None

    if args.val_csv:
        print("\n📥 Loading validation activations and sequences...")
        val_activations_dir = args.activations_dir.replace("traingrids", "valgrids") if "traingrids" in args.activations_dir else args.activations_dir
        val_activations = load_activations_from_local(
            activations_dir=val_activations_dir,
            layer=args.layer,
            hf_repo_id=args.hf_repo_id,
        )
        val_decoder_inputs, val_targets = load_action_sequences_from_csv(
            csv_path=args.val_csv,
            max_seq_len=args.max_seq_len,
            add_eos_token=args.add_eos_token,
        )
        val_dataset = create_dataset(val_activations, val_decoder_inputs, val_targets, max_seq_len=args.max_seq_len)
        print(f"Created {len(val_dataset)} validation pairs")

    if args.test_csv:
        print("\n📥 Loading test activations and sequences...")
        test_activations_dir = args.activations_dir.replace("traingrids", "testgrids") if "traingrids" in args.activations_dir else args.activations_dir
        test_activations = load_activations_from_local(
            activations_dir=test_activations_dir,
            layer=args.layer,
            hf_repo_id=args.hf_repo_id,
        )
        test_decoder_inputs, test_targets = load_action_sequences_from_csv(
            csv_path=args.test_csv,
            max_seq_len=args.max_seq_len,
            add_eos_token=args.add_eos_token,
        )
        test_dataset = create_dataset(test_activations, test_decoder_inputs, test_targets, max_seq_len=args.max_seq_len)
        print(f"Created {len(test_dataset)} test pairs")

    # Step 5: Determine activation dimension and calculate max sequence length
    if len(train_dataset) == 0:
        raise ValueError("No training data! Check that activations and action sequences match.")
    
    sample_activation, _, _ = train_dataset[0]
    activation_dim = sample_activation.shape[0]
    print(f"\n📊 Activation dimension: {activation_dim}")
    
    # Calculate max sequence length from all datasets (train, val, test)
    # This ensures consistent padding across all splits
    all_decoder_inputs = list(decoder_inputs.values())
    all_targets = list(targets.values())
    if val_decoder_inputs:
        all_decoder_inputs.extend(list(val_decoder_inputs.values()))
        all_targets.extend(list(val_targets.values()))
    if test_decoder_inputs:
        all_decoder_inputs.extend(list(test_decoder_inputs.values()))
        all_targets.extend(list(test_targets.values()))
    
    max_decoder_input_len = max(len(seq) for seq in all_decoder_inputs) if all_decoder_inputs else 100
    max_target_len = max(len(seq) for seq in all_targets) if all_targets else 100
    global_max_seq_len = max(max_decoder_input_len, max_target_len)
    # Cap at model's max_seq_len
    effective_max_seq_len = min(global_max_seq_len, args.max_seq_len)
    print(f"\n📏 Max sequence lengths:")
    print(f"  Decoder input: {max_decoder_input_len}")
    print(f"  Target: {max_target_len}")
    print(f"  Global max: {global_max_seq_len}")
    print(f"  Effective max (capped at model limit): {effective_max_seq_len}")

    # Step 6: Create model
    print("\n🏗️  Step 4: Creating decoder probe model...")
    print(f"  Using {args.num_memory_tokens} memory tokens ({activation_dim // args.num_memory_tokens} dims each)")
    model = ActionDecoderProbe(
        activation_dim=activation_dim,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        num_memory_tokens=args.num_memory_tokens,
        max_seq_len=args.max_seq_len,
        eos_loss_weight_threshold=args.eos_weight_threshold,
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Step 7: Train model
    print("\n🚀 Step 5: Training decoder probe...")
    if args.save_path is None:
        os.makedirs("models", exist_ok=True)
        args.save_path = f"models/decoder_probe_layer_{args.layer}.pt"

    # If we have separate val dataset, we need to modify train_decoder_probe to accept it
    # Pass validation dataset if provided
    metrics = train_decoder_probe(
        model=model,
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        eval_split=args.eval_split if not args.val_csv else 0.0,  # Use 0 if we have separate val set
        val_dataset=val_dataset,  # Pass separate validation set if available
        save_path=args.save_path,
        max_decoder_input_len=max_decoder_input_len,  # Pass max lengths from all datasets
        max_target_len=max_target_len,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Sequence Accuracy: {metrics['final_sequence_accuracy']:.4f} ⭐ (PRIMARY METRIC)")
    print(f"Final Token Accuracy: {metrics['final_token_accuracy']:.4f} (misleading, 25% random baseline)")
    print(f"Model saved to: {args.save_path}")

    # Step 8: Evaluate on test set if provided
    if test_dataset and len(test_dataset) > 0:
        print("\n🧪 Step 6: Evaluating on test set...")
        from torch.utils.data import DataLoader
        from telos_interp.decoder_probe import PAD_TOKEN_ID, EOS_TOKEN_ID

        model.eval()
        
        # Use the same collate function as training (from train_decoder_probe)
        all_data = test_dataset
        max_decoder_input_len = max(len(decoder_input) for _, decoder_input, _ in all_data) if all_data else 100
        max_target_len = max(len(target) for _, _, target in all_data) if all_data else 100
        effective_max_decoder_len = min(max_decoder_input_len, args.max_seq_len)
        effective_max_target_len = min(max_target_len, args.max_seq_len)
        
        def test_collate_fn(batch):
            activations = []
            decoder_inputs = []
            targets = []
            attention_masks = []

            for activation, decoder_input, target in batch:
                activations.append(activation)
                
                # Truncate if longer than max_seq_len
                if len(decoder_input) > args.max_seq_len:
                    decoder_input = decoder_input[:args.max_seq_len]
                if len(target) > args.max_seq_len:
                    target = target[:args.max_seq_len]
                
                # Pad with PAD tokens
                if len(decoder_input) < effective_max_decoder_len:
                    padding = torch.full(
                        (effective_max_decoder_len - len(decoder_input),), 
                        PAD_TOKEN_ID, 
                        dtype=torch.long
                    )
                    decoder_input = torch.cat([decoder_input, padding])
                
                if len(target) < effective_max_target_len:
                    padding = torch.full(
                        (effective_max_target_len - len(target),), 
                        PAD_TOKEN_ID, 
                        dtype=torch.long
                    )
                    target = torch.cat([target, padding])
                
                attention_mask = (decoder_input != PAD_TOKEN_ID).long()
                
                decoder_inputs.append(decoder_input)
                targets.append(target)
                attention_masks.append(attention_mask)

            activations_tensor = torch.stack(activations)
            decoder_inputs_tensor = torch.stack(decoder_inputs)
            targets_tensor = torch.stack(targets)
            attention_masks_tensor = torch.stack(attention_masks)
            
            return activations_tensor, decoder_inputs_tensor, targets_tensor, attention_masks_tensor
        
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_collate_fn)

        total_tokens = 0
        correct_tokens = 0
        exact_matches = 0
        total_sequences = 0

        # Use the actual device the model is on (may have fallen back to CPU)
        actual_device = next(model.parameters()).device
        print(f"Using device: {actual_device} for test evaluation")
        
        with torch.no_grad():
            for activation_batch, decoder_inputs_batch, targets_batch, attention_masks_batch in test_loader:
                activation_batch = activation_batch.to(actual_device).to(torch.float32)
                decoder_inputs_batch = decoder_inputs_batch.to(actual_device)
                targets_batch = targets_batch.to(actual_device)
                attention_masks_batch = attention_masks_batch.to(actual_device)

                # Generate predictions
                predictions = model.generate(activation_batch, max_length=args.max_seq_len)
                
                # Compare with ground truth
                for i in range(predictions.shape[0]):
                    # Get target sequence (actions + EOS, with PAD)
                    target_seq = targets_batch[i].cpu()
                    
                    # Remove PAD and EOS from target to get true actions
                    true_actions = target_seq[(target_seq != PAD_TOKEN_ID) & (target_seq != EOS_TOKEN_ID)]
                    
                    # Get prediction sequence (EOS already removed by generate())
                    pred_seq = predictions[i].cpu()
                    
                    # Remove PAD from predictions
                    pred_actions = pred_seq[pred_seq != PAD_TOKEN_ID]
                    
                    # Compare up to minimum length
                    min_len = min(len(true_actions), len(pred_actions))
                    if min_len > 0:
                        true_seq_compare = true_actions[:min_len]
                        pred_seq_compare = pred_actions[:min_len]
                        
                        # Token-level accuracy
                        total_tokens += min_len
                        correct_tokens += (pred_seq_compare == true_seq_compare).sum().item()
                        
                        # Sequence-level exact match (only if full sequences match)
                        if len(true_actions) == len(pred_actions):
                            if torch.equal(true_actions, pred_actions):
                                exact_matches += 1
                        total_sequences += 1
                    elif len(true_actions) > 0:
                        # Prediction is empty but true sequence has actions
                        total_sequences += 1

        test_token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        test_seq_acc = exact_matches / total_sequences if total_sequences > 0 else 0.0

        print(f"Test Sequence Accuracy: {test_seq_acc:.4f} ⭐ (PRIMARY METRIC)")
        print(f"Test Token Accuracy: {test_token_acc:.4f} (misleading, 25% random baseline)")


if __name__ == "__main__":
    main()

