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
        default=4,
        help="Action vocabulary size (default: 4 for 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN)",
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
        "--n-embd",
        type=int,
        default=256,
        help="Hidden dimension (default: 256)",
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
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: cuda if available, else cpu)",
    )

    args = parser.parse_args()

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
    )
    print(f"Loaded {len(activations)} activations")

    # Step 2: Load action sequences from training CSV
    print("\n📥 Step 2: Loading action sequences from training CSV...")
    action_sequences = load_action_sequences_from_csv(
        csv_path=args.train_csv,
        max_seq_len=args.max_seq_len,
    )
    print(f"Loaded {len(action_sequences)} action sequences")

    # Step 3: Create training dataset
    print("\n🔗 Step 3: Creating training dataset...")
    train_dataset = create_dataset(activations, action_sequences, max_seq_len=args.max_seq_len, vocab_size=args.vocab_size)
    print(f"Created {len(train_dataset)} training pairs")

    # Step 4: Handle validation/test sets
    val_dataset = None
    test_dataset = None

    if args.val_csv:
        print("\n📥 Loading validation activations and sequences...")
        val_activations = load_activations_from_local(
            activations_dir=args.activations_dir.replace("traingrids", "valgrids")
            if "traingrids" in args.activations_dir
            else args.activations_dir,
            layer=args.layer,
        )
        val_action_sequences = load_action_sequences_from_csv(
            csv_path=args.val_csv,
            max_seq_len=args.max_seq_len,
        )
        val_dataset = create_dataset(val_activations, val_action_sequences, max_seq_len=args.max_seq_len, vocab_size=args.vocab_size)
        print(f"Created {len(val_dataset)} validation pairs")

    if args.test_csv:
        print("\n📥 Loading test activations and sequences...")
        test_activations = load_activations_from_local(
            activations_dir=args.activations_dir.replace("traingrids", "testgrids")
            if "traingrids" in args.activations_dir
            else args.activations_dir,
            layer=args.layer,
        )
        test_action_sequences = load_action_sequences_from_csv(
            csv_path=args.test_csv,
            max_seq_len=args.max_seq_len,
        )
        test_dataset = create_dataset(test_activations, test_action_sequences, max_seq_len=args.max_seq_len, vocab_size=args.vocab_size)
        print(f"Created {len(test_dataset)} test pairs")

    # Step 5: Determine activation dimension
    if len(train_dataset) == 0:
        raise ValueError("No training data! Check that activations and action sequences match.")
    
    sample_activation, _ = train_dataset[0]
    activation_dim = sample_activation.shape[0]
    print(f"\n📊 Activation dimension: {activation_dim}")

    # Step 6: Create model
    print("\n🏗️  Step 4: Creating decoder probe model...")
    model = ActionDecoderProbe(
        activation_dim=activation_dim,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        max_seq_len=args.max_seq_len,
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
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Token Accuracy: {metrics['final_token_accuracy']:.4f}")
    print(f"Final Sequence Accuracy: {metrics['final_sequence_accuracy']:.4f}")
    print(f"Model saved to: {args.save_path}")

    # Step 8: Evaluate on test set if provided
    if test_dataset and len(test_dataset) > 0:
        print("\n🧪 Step 6: Evaluating on test set...")
        from torch.utils.data import DataLoader

        model.eval()
        
        # Use the same collate function as training
        dataset_max_len = max(len(seq) for _, seq in test_dataset) if test_dataset else 100
        effective_max_len = min(dataset_max_len, args.max_seq_len - 1)  # Account for position embedding limit
        
        def test_collate_fn(batch):
            activations = []
            action_seqs = []
            seq_lens = []

            for activation, action_seq in batch:
                activations.append(activation)
                # Truncate if longer than model's max_seq_len - 1
                if len(action_seq) > args.max_seq_len - 1:
                    action_seq = action_seq[:args.max_seq_len - 1]
                seq_lens.append(len(action_seq))
                # Pad sequence to effective_max_len
                if len(action_seq) < effective_max_len:
                    padded_seq = torch.cat(
                        [action_seq, torch.full((effective_max_len - len(action_seq),), -100, dtype=torch.long)]
                    )
                else:
                    padded_seq = action_seq
                # Ensure all action values are in valid range [0, vocab_size-1] (ignore padding -100)
                padded_seq = torch.where(padded_seq == -100, padded_seq, torch.clamp(padded_seq, 0, args.vocab_size - 1))
                action_seqs.append(padded_seq)

            activations_tensor = torch.stack(activations)
            action_seqs_tensor = torch.stack(action_seqs)
            return activations_tensor, action_seqs_tensor, torch.tensor(seq_lens)
        
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_collate_fn)

        total_tokens = 0
        correct_tokens = 0
        exact_matches = 0
        total_sequences = 0

        with torch.no_grad():
            for activation_batch, action_seq_batch, seq_lens in test_loader:
                activation_batch = activation_batch.to(args.device).to(torch.float32)
                action_seq_batch = action_seq_batch.to(args.device)

                # Generate predictions
                predictions = model.generate(activation_batch, max_length=args.max_seq_len - 1)
                
                # Compare with ground truth using actual sequence lengths
                for i in range(predictions.shape[0]):
                    seq_len = seq_lens[i].item()
                    if seq_len > 0:
                        # Use actual sequence length (not padded length)
                        pred_seq = predictions[i, :seq_len].cpu()
                        true_seq = action_seq_batch[i, :seq_len].cpu()
                        
                        # Token-level accuracy
                        mask = true_seq != -100  # Valid tokens (not padding)
                        valid_tokens = mask.sum().item()
                        if valid_tokens > 0:
                            total_tokens += valid_tokens
                            correct_tokens += ((pred_seq == true_seq) & mask).sum().item()
                        
                        # Sequence-level exact match
                        if valid_tokens > 0:
                            if torch.equal(pred_seq[mask], true_seq[mask]):
                                exact_matches += 1
                            total_sequences += 1

        test_token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        test_seq_acc = exact_matches / total_sequences if total_sequences > 0 else 0.0

        print(f"Test Token Accuracy: {test_token_acc:.4f}")
        print(f"Test Sequence Accuracy: {test_seq_acc:.4f}")


if __name__ == "__main__":
    main()

