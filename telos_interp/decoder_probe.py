"""Decoder probe for predicting action sequences from LLM activations.

This module implements a small transformer decoder that takes LLM activations
as input (similar to prefix/prompt tuning) and predicts sequences of future actions.
"""

import os

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

from telos_interp.probing import load_activations


class ActionDecoderProbe(nn.Module):
    """Small transformer decoder that predicts action sequences from LLM activations.

    This model uses the LLM activation as an input embedding (prefix-tuning style)
    and generates a sequence of action tokens autoregressively.

    Args:
        activation_dim: Dimension of the input LLM activation vector
        vocab_size: Number of action tokens (default: 4 for LEFT, RIGHT, UP, DOWN)
        n_layer: Number of transformer decoder layers (default: 4)
        n_head: Number of attention heads (default: 4)
        n_embd: Hidden dimension of the decoder (default: 256)
        max_seq_len: Maximum sequence length for action predictions (default: 100)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        activation_dim: int,
        vocab_size: int = 4,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 256,
        max_seq_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.activation_dim = activation_dim
        self.n_embd = n_embd

        # Project activation to decoder embedding dimension
        self.activation_projection = nn.Linear(activation_dim, n_embd)

        # Action token embeddings
        self.action_embeddings = nn.Embedding(vocab_size, n_embd)

        # Position embeddings for action sequence
        self.position_embeddings = nn.Embedding(max_seq_len, n_embd)

        # Custom transformer decoder blocks
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder_layers = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)

        # Output projection to vocab
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values."""
        nn.init.normal_(self.activation_projection.weight, std=0.02)
        nn.init.normal_(self.activation_projection.bias, std=0.02)
        nn.init.normal_(self.action_embeddings.weight, std=0.02)

    def forward(
        self,
        activation: torch.Tensor,
        action_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the decoder probe.

        Args:
            activation: LLM activation tensor of shape (batch_size, activation_dim)
            action_ids: Action token IDs of shape (batch_size, seq_len) for teacher forcing.
                If None and labels is provided, uses labels shifted by one position.
            labels: Target action token IDs of shape (batch_size, seq_len) for computing loss.
                If None, only returns logits.

        Returns:
            If labels is not None, returns (loss, logits) tuple.
            Otherwise, returns logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        batch_size = activation.shape[0]
        device = activation.device

        # Convert activation to float32 if needed (HF activations may be bfloat16)
        if activation.dtype != torch.float32:
            activation = activation.to(torch.float32)

        # Project activation to decoder embedding space (prefix-style input)
        # This serves as the "memory" or "context" that conditions action generation
        activation_embed = self.activation_projection(activation)  # (batch_size, n_embd)
        activation_embed = activation_embed.unsqueeze(1)  # (batch_size, 1, n_embd) - this is the "prefix"

        if action_ids is None and labels is not None:
            # For training: shift labels to create input sequence (teacher forcing)
            # Shift: [action_0, action_1, ..., action_n] -> [0, action_0, ..., action_{n-1}]
            seq_len = labels.shape[1]
            if seq_len > 0:
                action_ids = torch.cat(
                    [torch.zeros(batch_size, 1, dtype=torch.long, device=device), labels[:, :-1]], dim=1
                )
            else:
                action_ids = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        elif action_ids is None:
            # For inference: start with just activation
            action_ids = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

        # Build target sequence embeddings
        if action_ids.shape[1] > 0:
            action_embeds = self.action_embeddings(action_ids)  # (batch_size, seq_len, n_embd)
            # Add position embeddings (starting from position 1, since activation is at position 0)
            positions = torch.arange(1, action_ids.shape[1] + 1, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_embeds = self.position_embeddings(positions)
            target_embeds = action_embeds + pos_embeds  # (batch_size, seq_len, n_embd)
        else:
            # No actions yet, just use activation as both memory and target
            target_embeds = activation_embed  # (batch_size, 1, n_embd)

        # Use transformer decoder:
        # - memory (activation_embed): the prefix/context from LLM activation
        # - tgt (target_embeds): the action sequence being generated
        # The decoder attends to both the memory (activation) and previous tokens in the sequence
        decoder_output = self.decoder_layers(
            tgt=target_embeds, memory=activation_embed
        )  # (batch_size, seq_len, n_embd)

        # Project to vocabulary
        logits = self.lm_head(decoder_output)  # (batch_size, seq_len, vocab_size)

        if labels is not None:
            # Compute cross-entropy loss
            # Reshape for loss computation
            logits_flat = logits.view(-1, self.vocab_size)  # (batch_size * seq_len, vocab_size)
            labels_flat = labels.view(-1)  # (batch_size * seq_len)
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
            return loss, logits
        return logits

    def generate(
        self,
        activation: torch.Tensor,
        max_length: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Generate action sequence autoregressively from activation.

        Args:
            activation: LLM activation tensor of shape (batch_size, activation_dim)
            max_length: Maximum length of generated sequence (default: self.max_seq_len)
            temperature: Sampling temperature (default: 1.0)
            top_k: Top-k sampling (default: None, disabled)
            top_p: Nucleus sampling (default: None, disabled)

        Returns:
            Generated action token IDs of shape (batch_size, generated_length)
        """
        if max_length is None:
            max_length = self.max_seq_len

        batch_size = activation.shape[0]
        device = activation.device
        self.eval()

        # Convert activation to float32 if needed (HF activations may be bfloat16)
        if activation.dtype != torch.float32:
            activation = activation.to(torch.float32)

        # Project activation to embedding space (this is the memory/prefix)
        activation_embed = self.activation_projection(activation).unsqueeze(1)  # (batch_size, 1, n_embd)

        generated = []
        with torch.no_grad():
            # Generate tokens autoregressively
            for step in range(max_length):
                if step == 0:
                    # First step: use activation as target (empty sequence)
                    target_embeds = activation_embed  # (batch_size, 1, n_embd)
                else:
                    # Subsequent steps: embed generated actions so far
                    past_action_ids = torch.stack(generated, dim=1)  # (batch_size, step)
                    past_action_embeds = self.action_embeddings(past_action_ids)  # (batch_size, step, n_embd)
                    positions = torch.arange(1, step + 1, device=device).unsqueeze(0).expand(batch_size, -1)
                    pos_embeds = self.position_embeddings(positions)
                    past_action_embeds = past_action_embeds + pos_embeds

                    # Concatenate with activation for decoder input
                    target_embeds = torch.cat([activation_embed, past_action_embeds], dim=1)

                # Run through decoder
                decoder_output = self.decoder_layers(tgt=target_embeds, memory=activation_embed)
                logits = self.lm_head(decoder_output)  # (batch_size, seq_len, vocab_size)

                # Get logits for the last position (next token prediction)
                next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float("-inf")

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
                generated.append(next_token.squeeze(1))  # (batch_size,)

        return torch.stack(generated, dim=1)  # (batch_size, max_length)


def load_activations_from_hf(
    repo_id: str = "project-telos/interp",
    path_in_repo: str = "first_six_hundred_grids",
    grid_indices: list | None = None,
    layer: int = 12,
    cache_dir: str | None = None,
) -> dict[int, torch.Tensor]:
    """Load activations from Hugging Face repository.

    Args:
        repo_id: Hugging Face repository ID
        path_in_repo: Path within the repository to activations
        grid_indices: List of grid indices to load. If None, discovers available grids.
        layer: Layer number to load activations from
        cache_dir: Local cache directory for HF downloads

    Returns:
        Dictionary mapping grid index to activation tensor of shape (hidden_dim,)
    """
    from huggingface_hub import HfApi

    activations = {}

    if grid_indices is None:
        # Discover available grids by listing repository files
        print(f"Discovering available grids with layer_{layer} activations...")
        try:
            api = HfApi()
            repo_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
            # Filter for files matching the pattern: path_in_repo/grid_N/layer_X/activations.pt
            grid_set = set()
            target_pattern = f"/layer_{layer}/activations.pt"
            for file in repo_files:
                if file.startswith(f"{path_in_repo}/grid_") and file.endswith(target_pattern):
                    # Extract grid index from path like "first_six_hundred_grids/grid_123/layer_20/activations.pt"
                    parts = file.split("/")
                    if len(parts) >= 2:
                        grid_part = parts[1]  # "grid_123"
                        if grid_part.startswith("grid_"):
                            try:
                                grid_idx = int(grid_part.replace("grid_", ""))
                                grid_set.add(grid_idx)
                            except ValueError:
                                continue
            grid_indices = sorted(list(grid_set))
            print(
                f"Found {len(grid_indices)} grids with layer_{layer} activations: {grid_indices[:20]}{'...' if len(grid_indices) > 20 else ''}"
            )
        except Exception as e:
            print(f"Warning: Could not list repository files: {e}")
            print("Falling back to trying grids 0-599...")
            grid_indices = list(range(600))

    print(f"Loading activations for {len(grid_indices)} grids...")
    loaded = 0
    for grid_idx in grid_indices:
        try:
            # Construct filename: first_six_hundred_grids/grid_N/layer_X/activations.pt
            filename = f"{path_in_repo}/grid_{grid_idx}/layer_{layer}/activations.pt"
            activation_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                repo_type="model",
            )
            activation = load_activations(activation_path)
            # Ensure it's 1D (hidden_dim,)
            if activation.ndim > 1:
                # If it's (seq_len, hidden_dim), take the last token
                activation = activation[-1]
            activations[grid_idx] = activation
            loaded += 1
        except Exception as e:
            # Only print warnings for non-404 errors (404s are expected for missing grids)
            if "404" not in str(e) and "Not Found" not in str(e):
                print(f"Warning: Could not load activations for grid_{grid_idx}: {e}")
            continue

    print(f"Successfully loaded {loaded}/{len(grid_indices)} grids")
    return activations


def load_action_sequences_from_csv(
    csv_path: str,
    grid_indices: list | None = None,
    max_seq_len: int | None = None,
) -> dict[int, torch.Tensor]:
    """Load action sequences from CSV file.

    Supports two CSV formats:
    1. Format with 'action_sequence' column containing JSON array: [0, 1, 2, 3, ...]
    2. Format with 'last_action' column per trajectory step (legacy)

    Args:
        csv_path: Path to CSV file with trajectory data
        grid_indices: List of grid indices to load. If None, loads all.
        max_seq_len: Maximum sequence length. If None, uses the longest sequence.

    Returns:
        Dictionary mapping grid index to action sequence tensor of shape (seq_len,)
    """
    import json
    import pandas as pd

    df = pd.read_csv(csv_path)

    action_sequences = {}
    
    # Check which format we have
    has_action_sequence_col = "action_sequence" in df.columns
    has_last_action_col = "last_action" in df.columns

    if has_action_sequence_col:
        # Format 1: action_sequence column with JSON array
        for env_idx in df["env_idx"].unique():
            if grid_indices is not None and env_idx not in grid_indices:
                continue

            env_data = df[df["env_idx"] == env_idx]
            if len(env_data) == 0:
                continue
            
            # Get the first row (since action_sequence is the full sequence)
            row = env_data.iloc[0]
            action_sequence_str = row["action_sequence"]
            
            if pd.notna(action_sequence_str) and action_sequence_str != "":
                try:
                    # Parse JSON array
                    actions = json.loads(action_sequence_str)
                    # Ensure all actions are integers in range [0, 3]
                    actions = [int(a) for a in actions if isinstance(a, (int, float)) and 0 <= int(a) <= 3]
                    
                    if len(actions) > 0:
                        action_seq = torch.tensor(actions, dtype=torch.long)
                        if max_seq_len is not None and len(action_seq) > max_seq_len:
                            action_seq = action_seq[:max_seq_len]
                        action_sequences[env_idx] = action_seq
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    print(f"Warning: Failed to parse action_sequence for env_idx {env_idx}: {e}")
                    continue
    
    elif has_last_action_col:
        # Format 2: last_action column per step (legacy format)
        for env_idx in df["env_idx"].unique():
            if grid_indices is not None and env_idx not in grid_indices:
                continue

            env_data = df[df["env_idx"] == env_idx].sort_values("trajectory_step")
            actions = []

            for _, row in env_data.iterrows():
                last_action = row["last_action"]
                if pd.notna(last_action) and last_action != "":
                    # Convert action string to integer
                    try:
                        action_int = int(float(last_action))
                        if 0 <= action_int <= 3:  # Validate range
                            actions.append(action_int)
                    except (ValueError, TypeError):
                        continue

            if len(actions) > 0:
                action_seq = torch.tensor(actions, dtype=torch.long)
                if max_seq_len is not None and len(action_seq) > max_seq_len:
                    action_seq = action_seq[:max_seq_len]
                action_sequences[env_idx] = action_seq
    else:
        raise ValueError("CSV must have either 'action_sequence' or 'last_action' column")

    return action_sequences


def create_dataset(
    activations: dict[int, torch.Tensor],
    action_sequences: dict[int, torch.Tensor],
    max_seq_len: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create dataset pairs of (activation, action_sequence).

    Args:
        activations: Dictionary mapping grid index to activation tensor
        action_sequences: Dictionary mapping grid index to action sequence tensor
        max_seq_len: Maximum sequence length. Sequences longer than this are truncated.

    Returns:
        List of (activation, action_sequence) tuples
    """
    dataset = []

    # Find common grid indices
    common_indices = set(activations.keys()) & set(action_sequences.keys())

    for grid_idx in common_indices:
        activation = activations[grid_idx]
        action_seq = action_sequences[grid_idx]

        if max_seq_len is not None and len(action_seq) > max_seq_len:
            action_seq = action_seq[:max_seq_len]

        dataset.append((activation, action_seq))

    return dataset


def train_decoder_probe(
    model: ActionDecoderProbe,
    dataset: list[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    eval_split: float = 0.2,
    save_path: str | None = None,
) -> dict:
    """Train the decoder probe model.

    Args:
        model: ActionDecoderProbe model to train
        dataset: List of (activation, action_sequence) tuples
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        eval_split: Fraction of data to use for evaluation
        save_path: Path to save trained model. If None, model is not saved.

    Returns:
        Dictionary with training metrics
    """
    import random

    from torch.utils.data import DataLoader

    model = model.to(device)

    # Split dataset
    random.shuffle(dataset)
    split_idx = int(len(dataset) * (1 - eval_split))
    train_data = dataset[:split_idx]
    eval_data = dataset[split_idx:]

    # Prepare data for DataLoader
    # We need to pad sequences to the same length
    max_seq_len = max(len(seq) for _, seq in dataset) if dataset else 100

    def collate_fn(batch):
        activations = []
        action_seqs = []
        seq_lens = []

        for activation, action_seq in batch:
            activations.append(activation)
            seq_lens.append(len(action_seq))
            # Pad sequence to max_seq_len
            if len(action_seq) < max_seq_len:
                padded_seq = torch.cat(
                    [action_seq, torch.full((max_seq_len - len(action_seq),), -100, dtype=torch.long)]
                )
            else:
                padded_seq = action_seq
            action_seqs.append(padded_seq)

        activations_tensor = torch.stack(activations)
        action_seqs_tensor = torch.stack(action_seqs)
        return activations_tensor, action_seqs_tensor, torch.tensor(seq_lens)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    eval_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        for activations_batch, action_seqs_batch, seq_lens in train_loader:
            activations_batch = activations_batch.to(device).to(torch.float32)
            action_seqs_batch = action_seqs_batch.to(device)

            # Forward pass
            loss, _ = model(activations_batch, labels=action_seqs_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0.0
        train_losses.append(avg_train_loss)

        # Evaluation
        model.eval()
        epoch_eval_loss = 0.0
        num_eval_batches = 0
        total_tokens = 0
        correct_tokens = 0
        exact_matches = 0
        total_sequences = 0

        with torch.no_grad():
            for batch_idx, (activations_batch, action_seqs_batch, seq_lens) in enumerate(eval_loader):
                activations_batch = activations_batch.to(device).to(torch.float32)
                action_seqs_batch = action_seqs_batch.to(device)

                loss, logits = model(activations_batch, labels=action_seqs_batch)

                epoch_eval_loss += loss.item()
                num_eval_batches += 1

                # Compute accuracy metrics
                predictions = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)

                # Token-level accuracy (only count non-padding tokens)
                mask = action_seqs_batch != -100
                total_tokens += mask.sum().item()
                correct_tokens += ((predictions == action_seqs_batch) & mask).sum().item()

                # Sequence-level exact match
                for i in range(predictions.shape[0]):
                    seq_len = seq_lens[i].item()
                    if seq_len > 0:
                        pred_seq = predictions[i, :seq_len].cpu()
                        true_seq = action_seqs_batch[i, :seq_len].cpu()
                        if torch.equal(pred_seq, true_seq):
                            exact_matches += 1
                        total_sequences += 1

                # Show sample predictions for first batch of last epoch
                if epoch == num_epochs - 1 and batch_idx == 0:
                    print("\nSample predictions (first batch):")
                    for i in range(min(3, predictions.shape[0])):
                        seq_len = seq_lens[i].item()
                        if seq_len > 0:
                            pred_seq = predictions[i, :seq_len].cpu().tolist()
                            true_seq = action_seqs_batch[i, :seq_len].cpu().tolist()
                            print(f"  Sample {i + 1}:")
                            print(f"    True:  {true_seq}")
                            print(f"    Pred:  {pred_seq}")
                            print(f"    Match: {pred_seq == true_seq}")

        avg_eval_loss = epoch_eval_loss / num_eval_batches if num_eval_batches > 0 else 0.0
        eval_losses.append(avg_eval_loss)

        # Compute accuracy metrics
        token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        sequence_accuracy = exact_matches / total_sequences if total_sequences > 0 else 0.0

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Eval Loss: {avg_eval_loss:.4f}, "
            f"Token Acc: {token_accuracy:.4f}, "
            f"Seq Acc: {sequence_accuracy:.4f}"
        )

    # Save model if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    # Compute final accuracy metrics
    model.eval()
    final_token_accuracy = 0.0
    final_sequence_accuracy = 0.0
    total_tokens = 0
    correct_tokens = 0
    exact_matches = 0
    total_sequences = 0

    with torch.no_grad():
        for activations_batch, action_seqs_batch, seq_lens in eval_loader:
            activations_batch = activations_batch.to(device).to(torch.float32)
            action_seqs_batch = action_seqs_batch.to(device)

            _, logits = model(activations_batch, labels=action_seqs_batch)
            predictions = torch.argmax(logits, dim=-1)

            # Token-level accuracy
            mask = action_seqs_batch != -100
            total_tokens += mask.sum().item()
            correct_tokens += ((predictions == action_seqs_batch) & mask).sum().item()

            # Sequence-level exact match
            for i in range(predictions.shape[0]):
                seq_len = seq_lens[i].item()
                if seq_len > 0:
                    pred_seq = predictions[i, :seq_len].cpu()
                    true_seq = action_seqs_batch[i, :seq_len].cpu()
                    if torch.equal(pred_seq, true_seq):
                        exact_matches += 1
                    total_sequences += 1

    final_token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    final_sequence_accuracy = exact_matches / total_sequences if total_sequences > 0 else 0.0

    return {
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
        "final_token_accuracy": final_token_accuracy,
        "final_sequence_accuracy": final_sequence_accuracy,
    }
