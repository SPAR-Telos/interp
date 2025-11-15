"""Training script for GridActionDecoder.

Trains a decoder-only transformer to predict action sequences from grid states.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from telos_interp.grid_action_decoder import GridActionDecoder


class GridTokenizer:
    """Simple character-level tokenizer for grid text."""

    def __init__(self):
        # Build vocabulary from grid symbols
        # Characters: #, _, G, A, 0-9, space, newline, and coordinate numbers
        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
        }
        
        # Add common grid characters
        chars = ["#", "_", "G", "A", " ", "\n"] + [str(i) for i in range(10)]
        for char in chars:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        
        self.vocab_size = len(self.vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, max_length: int = 500) -> list[int]:
        """Encode text to token IDs."""
        tokens = []
        for char in text:
            token_id = self.vocab.get(char, self.vocab["<UNK>"])
            tokens.append(token_id)
        
        # Pad or truncate to max_length
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [self.vocab["<PAD>"]] * (max_length - len(tokens))
        
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        return "".join([self.id_to_token.get(tid, "<UNK>") for tid in token_ids if tid != self.vocab["<PAD>"]])


class GridActionDataset(Dataset):
    """Dataset for grid-action pairs."""

    def __init__(self, csv_path: str, tokenizer: GridTokenizer, max_action_len: int = 50):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_action_len = max_action_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Tokenize grid text
        grid_text = row["fo_observation"]
        grid_token_ids = torch.tensor(self.tokenizer.encode(grid_text), dtype=torch.long)
        
        # Parse action sequence
        action_sequence = json.loads(row["action_sequence"])
        action_tensor = torch.tensor(action_sequence, dtype=torch.long)
        
        # Pad or truncate action sequence
        if len(action_tensor) > self.max_action_len:
            action_tensor = action_tensor[:self.max_action_len]
        else:
            padding = torch.full((self.max_action_len - len(action_tensor),), -100, dtype=torch.long)
            action_tensor = torch.cat([action_tensor, padding])
        
        return {
            "grid_token_ids": grid_token_ids,
            "action_sequence": action_tensor,
            "env_idx": row["env_idx"],
            "start_pos": row["start_pos"],
            "goal_pos": row["goal_pos"],
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    grid_token_ids = torch.stack([item["grid_token_ids"] for item in batch])
    action_sequences = torch.stack([item["action_sequence"] for item in batch])
    
    return {
        "grid_token_ids": grid_token_ids,
        "action_sequences": action_sequences,
    }


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        grid_token_ids = batch["grid_token_ids"].to(device)
        action_sequences = batch["action_sequences"].to(device)

        # Forward pass
        loss, logits = model(grid_token_ids, labels=action_sequences)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_tokens = 0
    correct_tokens = 0
    exact_matches = 0
    total_sequences = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            grid_token_ids = batch["grid_token_ids"].to(device)
            action_sequences = batch["action_sequences"].to(device)

            loss, logits = model(grid_token_ids, labels=action_sequences)
            total_loss += loss.item()
            num_batches += 1

            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)

            # Token-level accuracy (only count non-padding tokens)
            mask = action_sequences != -100
            total_tokens += mask.sum().item()
            correct_tokens += ((predictions == action_sequences) & mask).sum().item()

            # Sequence-level exact match
            for i in range(predictions.shape[0]):
                seq_mask = action_sequences[i] != -100
                if seq_mask.sum() > 0:
                    pred_seq = predictions[i][seq_mask].cpu()
                    true_seq = action_sequences[i][seq_mask].cpu()
                    if torch.equal(pred_seq, true_seq):
                        exact_matches += 1
                    total_sequences += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    sequence_accuracy = exact_matches / total_sequences if total_sequences > 0 else 0.0

    return avg_loss, token_accuracy, sequence_accuracy


def main():
    parser = argparse.ArgumentParser(description="Train GridActionDecoder")
    parser.add_argument("--train-csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--val-csv", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--test-csv", type=str, help="Path to test CSV (optional)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n-layer", type=int, default=4, help="Number of decoder layers")
    parser.add_argument("--n-head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-embd", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--max-action-len", type=int, default=50, help="Maximum action sequence length")
    parser.add_argument("--save-dir", type=str, default="outputs/models", help="Directory to save model")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto if None)")
    
    args = parser.parse_args()

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = GridTokenizer()
    print(f"Grid vocabulary size: {tokenizer.vocab_size}")

    # Datasets
    train_dataset = GridActionDataset(args.train_csv, tokenizer, args.max_action_len)
    val_dataset = GridActionDataset(args.val_csv, tokenizer, args.max_action_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Model
    model = GridActionDecoder(
        grid_vocab_size=tokenizer.vocab_size,
        action_vocab_size=4,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        max_action_len=args.max_action_len,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_token_acc, val_seq_acc = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Token Acc: {val_token_acc:.4f}, Seq Acc: {val_seq_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            model_path = os.path.join(args.save_dir, "grid_action_decoder_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer_vocab": tokenizer.vocab,
                "config": {
                    "grid_vocab_size": tokenizer.vocab_size,
                    "action_vocab_size": 4,
                    "n_layer": args.n_layer,
                    "n_head": args.n_head,
                    "n_embd": args.n_embd,
                    "max_action_len": args.max_action_len,
                }
            }, model_path)
            print(f"Saved best model to {model_path}")

    # Final evaluation on test set if provided
    if args.test_csv:
        print("\n" + "="*50)
        print("Final Test Evaluation")
        print("="*50)
        test_dataset = GridActionDataset(args.test_csv, tokenizer, args.max_action_len)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loss, test_token_acc, test_seq_acc = evaluate(model, test_loader, device)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Token Acc: {test_token_acc:.4f}")
        print(f"Test Seq Acc: {test_seq_acc:.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

