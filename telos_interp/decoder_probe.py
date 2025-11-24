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

# EOS token ID (end-of-sequence marker)
EOS_TOKEN_ID = 4


class ActionDecoderProbe(nn.Module):
    """Small transformer decoder that predicts action sequences from LLM activations.

    This model uses the LLM activation as an input embedding (prefix-tuning style)
    and generates a sequence of action tokens autoregressively.

    Args:
        activation_dim: Dimension of the input LLM activation vector
        vocab_size: Number of action tokens (default: 5 for 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN, 4=EOS)
        n_layer: Number of transformer decoder layers (default: 4)
        n_head: Number of attention heads (default: 4)
        n_embd: Hidden dimension of the decoder (default: 256)
        max_seq_len: Maximum sequence length for action predictions (default: 100)
        dropout: Dropout rate (default: 0.1)
        eos_loss_weight_threshold: Threshold for position-weighted EOS loss (default: 3.0).
            EOS loss weight = min(1.0, (position + 1) / threshold). Higher values allow earlier EOS.
    """

    def __init__(
        self,
        activation_dim: int,
        vocab_size: int = 5,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 256,
        max_seq_len: int = 100,
        dropout: float = 0.1,
        eos_loss_weight_threshold: float = 3.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.activation_dim = activation_dim
        self.n_embd = n_embd
        self.eos_loss_weight_threshold = eos_loss_weight_threshold

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

        # Truncate labels to max_seq_len - 1 if provided
        # This ensures position indices don't exceed embedding range (since positions start at 1)
        if labels is not None and labels.shape[1] > self.max_seq_len - 1:
            labels = labels[:, :self.max_seq_len - 1]
        
        if action_ids is None and labels is not None:
            # For training: shift labels to create input sequence (teacher forcing)
            # Shift: [action_0, action_1, ..., action_n] -> [0, action_0, ..., action_{n-1}]
            seq_len = labels.shape[1]
            if seq_len > 0:
                # Clamp labels to valid range [0, vocab_size-1] before using
                labels_clamped = torch.clamp(labels, 0, self.vocab_size - 1)
                action_ids = torch.cat(
                    [torch.zeros(batch_size, 1, dtype=torch.long, device=device), labels_clamped[:, :-1]], dim=1
                )
            else:
                action_ids = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        elif action_ids is None:
            # For inference: start with just activation
            action_ids = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

        # Build target sequence embeddings
        if action_ids.shape[1] > 0:
            # Clamp action_ids to valid range [0, vocab_size-1]
            action_ids = torch.clamp(action_ids, 0, self.vocab_size - 1)
            # Truncate to max_seq_len - 1 if needed (to ensure position indices are valid)
            if action_ids.shape[1] > self.max_seq_len - 1:
                action_ids = action_ids[:, :self.max_seq_len - 1]
            
            action_embeds = self.action_embeddings(action_ids)  # (batch_size, seq_len, n_embd)
            # Add position embeddings (starting from position 1, since activation is at position 0)
            # Position embeddings have indices [0, max_seq_len-1]
            # Since positions start at 1, we need seq_len <= max_seq_len - 1 to ensure max position <= max_seq_len - 1
            seq_len = action_ids.shape[1]
            if seq_len > self.max_seq_len - 1:
                # Truncate to ensure positions don't exceed embedding range
                seq_len = self.max_seq_len - 1
                action_ids = action_ids[:, :seq_len]
                action_embeds = self.action_embeddings(action_ids)
            
            # Generate positions [1, 2, ..., seq_len] where seq_len <= max_seq_len - 1
            # So max position is seq_len <= max_seq_len - 1, which is valid for embedding
            positions = torch.arange(1, seq_len + 1, device=device).unsqueeze(0).expand(batch_size, -1)
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
            # Truncate labels to match logits length (which is based on action_ids/seq_len)
            seq_len_used = logits.shape[1]
            if labels.shape[1] > seq_len_used:
                labels = labels[:, :seq_len_used]
            # Clamp labels to valid range [0, vocab_size-1] (ignore padding -100)
            labels = torch.where(labels == -100, labels, torch.clamp(labels, 0, self.vocab_size - 1))
            
            # Compute per-token loss for position-weighted EOS loss
            logits_flat = logits.view(-1, self.vocab_size)  # (batch_size * seq_len, vocab_size)
            labels_flat = labels.view(-1)  # (batch_size * seq_len)
            
            # Compute per-token loss (reduction='none')
            per_token_loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100, reduction='none')
            
            # Apply position-weighted loss for EOS token to prevent mode collapse
            # Reshape to (batch_size, seq_len) for position indexing
            batch_size = labels.shape[0]
            per_token_loss_2d = per_token_loss.view(batch_size, seq_len_used)
            labels_2d = labels_flat.view(batch_size, seq_len_used)
            
            # Create position indices: [0, 1, 2, ..., seq_len-1]
            position_indices = torch.arange(seq_len_used, device=labels.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
            
            # Compute EOS loss weights: weight = min(1.0, (position + 1) / threshold)
            # This gives lower weight to early EOS predictions
            eos_weights = torch.clamp((position_indices + 1.0) / self.eos_loss_weight_threshold, max=1.0)
            
            # Apply weights: use eos_weights for EOS tokens, 1.0 for non-EOS tokens
            is_eos = (labels_2d == EOS_TOKEN_ID) & (labels_2d != -100)
            loss_weights = torch.where(is_eos, eos_weights, torch.ones_like(eos_weights))
            
            # Apply weights to per-token loss
            weighted_loss = per_token_loss_2d * loss_weights
            
            # Mask out padding tokens (-100) and compute mean
            valid_mask = (labels_2d != -100)
            loss = weighted_loss[valid_mask].sum() / valid_mask.sum().clamp(min=1)
            
            return loss, logits
        return logits

    def generate(
        self,
        activation: torch.Tensor,
        max_length: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        min_confidence: float = 0.0,
        max_repetition: int = 10,
        entropy_threshold: float = 2.0,
        max_length_override: int | None = None,
        disable_heuristic_stopping: bool = False,
    ) -> torch.Tensor:
        """Generate action sequence autoregressively from activation.

        Args:
            activation: LLM activation tensor of shape (batch_size, activation_dim)
            max_length: Maximum length of generated sequence (default: self.max_seq_len)
            temperature: Sampling temperature (default: 1.0)
            top_k: Top-k sampling (default: None, disabled)
            top_p: Nucleus sampling (default: None, disabled)
            min_confidence: Minimum probability of top action to continue (default: 0.5)
            max_repetition: Maximum consecutive identical actions before stopping (default: 3)
            entropy_threshold: Maximum entropy (uncertainty) before stopping (default: 0.9)
            max_length_override: Hard limit on sequence length, regardless of confidence (default: None, uses max_length)

        Returns:
            Generated action token IDs of shape (batch_size, generated_length)
        """
        if max_length is None:
            max_length = self.max_seq_len
        
        # Use override if provided (e.g., based on training data statistics)
        effective_max_length = max_length_override if max_length_override is not None else max_length

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
            for step in range(effective_max_length):
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
                next_token = next_token.squeeze(1)  # (batch_size,)
                generated.append(next_token)
                
                # Check if EOS token was sampled - EOS takes priority over heuristic stopping
                eos_predicted = (next_token == EOS_TOKEN_ID)
                if eos_predicted.any():
                    # If all sequences predicted EOS, stop completely
                    if eos_predicted.all():
                        break
                    # For mixed batches, continue but EOS will be filtered out later
                    # Continue generation for non-EOS sequences only
                
                # Heuristic stopping (only if not disabled)
                if not disable_heuristic_stopping:
                    # 1. Check confidence (max probability)
                    max_prob, _ = torch.max(probs, dim=-1)  # (batch_size,)
                    
                    # 2. Check entropy (uncertainty)
                    # Entropy = -sum(p * log(p))
                    log_probs = torch.log(probs + 1e-10)  # Add small epsilon to avoid log(0)
                    entropy = -torch.sum(probs * log_probs, dim=-1)  # (batch_size,)
                    
                    # 3. Check for repetitive patterns
                    should_stop = torch.zeros(batch_size, dtype=torch.bool, device=device)
                    
                    for b in range(batch_size):
                        # Skip if this sequence already predicted EOS
                        if eos_predicted[b]:
                            continue
                        
                        # Hard length limit check (safety)
                        if step >= effective_max_length - 1:
                            should_stop[b] = True
                            continue
                        
                        # Low confidence check - if model is uncertain, likely done
                        if min_confidence > 0 and max_prob[b].item() < min_confidence:
                            should_stop[b] = True
                            continue
                        
                        # High entropy (uncertainty) check - uniform distribution = done
                        # For vocab_size=5, max entropy = log(5) ≈ 1.609
                        if entropy[b].item() > entropy_threshold:
                            should_stop[b] = True
                            continue
                        
                        # Repetition check (only if we have generated actions)
                        # If model repeats same action, likely stuck
                        if len(generated) >= max_repetition:
                            recent_actions = [g[b].item() for g in generated[-max_repetition:]]
                            if len(set(recent_actions)) == 1:  # All same action
                                should_stop[b] = True
                                continue
                    
                    # Stop generation for batches that meet stopping criteria
                    if should_stop.all():
                        break

        # Stack all generated tokens
        if len(generated) > 0:
            result = torch.stack(generated, dim=1)  # (batch_size, generated_length)
            
            # Remove EOS tokens from output (keep sequences up to but not including EOS)
            # Process each sequence in the batch
            filtered_results = []
            for b in range(batch_size):
                seq = result[b]  # (generated_length,)
                # Find first EOS token
                eos_indices = (seq == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
                if len(eos_indices) > 0:
                    # Truncate at first EOS
                    seq = seq[:eos_indices[0]]
                filtered_results.append(seq)
            
            # Pad to same length for batching (or return as list if lengths vary)
            # For simplicity, find max length and pad
            max_len = max(len(seq) for seq in filtered_results) if filtered_results else 0
            if max_len > 0:
                padded_results = []
                for seq in filtered_results:
                    if len(seq) < max_len:
                        # Pad with -100 (will be ignored in evaluation)
                        padding = torch.full((max_len - len(seq),), -100, dtype=torch.long, device=device)
                        seq = torch.cat([seq, padding])
                    padded_results.append(seq)
                return torch.stack(padded_results, dim=0)  # (batch_size, max_len)
            else:
                return torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        else:
            # No tokens generated (shouldn't happen, but handle edge case)
            return torch.zeros(batch_size, 0, dtype=torch.long, device=device)


def load_activations_from_hf(
    repo_id: str = "project-telos/decoder",
    path_in_repo: str = "activations/7by7traingrids/full_prompt_activations",
    grid_indices: list | None = None,
    layer: int = 20,
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
            # Filter for files matching the pattern: path_in_repo/env_N_step_0/layer_X/activations.pt
            grid_set = set()
            target_pattern = f"/layer_{layer}/activations.pt"
            for file in repo_files:
                if file.startswith(f"{path_in_repo}/env_") and file.endswith(target_pattern):
                    # Extract env index from path like "activations/7by7traingrids/full_prompt_activations/env_123_step_0/layer_20/activations.pt"
                    parts = file.split("/")
                    # Find the part that starts with "env_"
                    for part in parts:
                        if part.startswith("env_"):
                            try:
                                # Extract number from "env_123_step_0" or "env_123"
                                env_idx_str = part.replace("env_", "").split("_step_")[0]
                                env_idx = int(env_idx_str)
                                grid_set.add(env_idx)
                            except ValueError:
                                continue
                            break
            grid_indices = sorted(list(grid_set))
            print(
                f"Found {len(grid_indices)} environments with layer_{layer} activations: {grid_indices[:20]}{'...' if len(grid_indices) > 20 else ''}"
            )
        except Exception as e:
            print(f"Warning: Could not list repository files: {e}")
            print("Falling back to trying env indices 1-2400...")
            grid_indices = list(range(1, 2401))

    print(f"Loading activations for {len(grid_indices)} environments...")
    loaded = 0
    for env_idx in grid_indices:
        try:
            # Construct filename: activations/7by7traingrids/full_prompt_activations/env_N_step_0/layer_X/activations.pt
            filename = f"{path_in_repo}/env_{env_idx}_step_0/layer_{layer}/activations.pt"
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
            activations[env_idx] = activation
            loaded += 1
        except Exception as e:
            # Only print warnings for non-404 errors (404s are expected for missing grids)
            if "404" not in str(e) and "Not Found" not in str(e):
                print(f"Warning: Could not load activations for env_{env_idx}: {e}")
            continue

    print(f"Successfully loaded {loaded}/{len(grid_indices)} activations")
    return activations


def load_activations_from_local(
    activations_dir: str,
    layer: int,
    grid_indices: list | None = None,
    hf_repo_id: str | None = None,
) -> dict[int, torch.Tensor]:
    """Load activations from local directory or HuggingFace repository.
    
    Supports two modes:
    1. Local directory structure:
        activations_dir/
            env_{env_idx}_step_{step}/layer_{layer}/activations.pt
        OR
            env_{env_idx}/layer_{layer}/activations.pt
    
    2. HuggingFace repository (if hf_repo_id is provided):
        Downloads from: {hf_repo_id}/activations/{activations_dir}/env_{env_idx}_step_0/layer_{layer}/activations.pt
    
    Args:
        activations_dir: Path to directory containing activation files (or subdirectory name for HF)
        layer: Layer number to load activations from
        grid_indices: List of env_idx values to load. If None, discovers all available.
        hf_repo_id: HuggingFace repository ID (e.g., "project-telos/decoder"). If None, uses local filesystem.
    
    Returns:
        Dictionary mapping env_idx to activation tensor of shape (hidden_dim,)
    """
    import glob
    
    activations = {}
    
    # If using HuggingFace, use the existing load_activations_from_hf function
    if hf_repo_id is not None:
        # Extract subdirectory name (e.g., "7by7traingrids/full_prompt_activations")
        hf_path = activations_dir
        activations = load_activations_from_hf(
            repo_id=hf_repo_id,
            path_in_repo=f"activations/{hf_path}",
            grid_indices=grid_indices,
            layer=layer,
        )
        return activations
    
    # Local filesystem loading
    activations_dir = os.path.abspath(activations_dir)
    
    if grid_indices is None:
        # Discover available env_idx values
        pattern = os.path.join(activations_dir, f"env_*/layer_{layer}/activations.pt")
        files = glob.glob(pattern)
        grid_set = set()
        for file in files:
            # Extract env_idx from path like "env_123_step_0/layer_20/activations.pt" or "env_123/layer_20/activations.pt"
            rel_path = os.path.relpath(file, activations_dir)
            parts = rel_path.split(os.sep)
            if len(parts) >= 1:
                env_part = parts[0]  # "env_123" or "env_123_step_0"
                if env_part.startswith("env_"):
                    try:
                        # Extract number after "env_"
                        env_idx_str = env_part.replace("env_", "").split("_step_")[0]
                        env_idx = int(env_idx_str)
                        grid_set.add(env_idx)
                    except ValueError:
                        continue
        grid_indices = sorted(list(grid_set))
        print(f"Found {len(grid_indices)} environments with layer_{layer} activations")
    
    print(f"Loading activations for {len(grid_indices)} environments...")
    loaded = 0
    
    for env_idx in grid_indices:
        # Try both patterns: with and without _step_0
        patterns = [
            os.path.join(activations_dir, f"env_{env_idx}_step_0", f"layer_{layer}", "activations.pt"),
            os.path.join(activations_dir, f"env_{env_idx}", f"layer_{layer}", "activations.pt"),
        ]
        
        activation_path = None
        for pattern in patterns:
            if os.path.exists(pattern):
                activation_path = pattern
                break
        
        if activation_path is None:
            # Try glob pattern as fallback
            glob_pattern = os.path.join(activations_dir, f"env_{env_idx}*", f"layer_{layer}", "activations.pt")
            matches = glob.glob(glob_pattern)
            if matches:
                activation_path = matches[0]
        
        if activation_path and os.path.exists(activation_path):
            try:
                activation = load_activations(activation_path)
                # Ensure it's 1D (hidden_dim,)
                if activation.ndim > 1:
                    # If it's (seq_len, hidden_dim), take the last token
                    activation = activation[-1]
                activations[env_idx] = activation
                loaded += 1
            except Exception as e:
                print(f"Warning: Could not load activations for env_{env_idx}: {e}")
                continue
        else:
            print(f"Warning: Could not find activations for env_{env_idx}")
    
    print(f"Successfully loaded {loaded}/{len(grid_indices)} activations")
    return activations


def load_action_sequences_from_csv(
    csv_path: str,
    grid_indices: list | None = None,
    max_seq_len: int | None = None,
    add_eos_token: bool = True,
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
                        # Append EOS token if requested
                        if add_eos_token:
                            eos_token = torch.tensor([EOS_TOKEN_ID], dtype=torch.long)
                            action_seq = torch.cat([action_seq, eos_token])
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
                # Append EOS token if requested
                if add_eos_token:
                    eos_token = torch.tensor([EOS_TOKEN_ID], dtype=torch.long)
                    action_seq = torch.cat([action_seq, eos_token])
                action_sequences[env_idx] = action_seq
    else:
        raise ValueError("CSV must have either 'action_sequence' or 'last_action' column")

    return action_sequences


def create_dataset(
    activations: dict[int, torch.Tensor],
    action_sequences: dict[int, torch.Tensor],
    max_seq_len: int | None = None,
    vocab_size: int = 4,
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

        # Validate and clamp action values to valid range [0, vocab_size-1]
        action_seq = torch.clamp(action_seq, 0, vocab_size - 1)
        
        # Truncate if needed
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
    val_dataset: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
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

    # Use provided validation dataset or split training dataset
    if val_dataset is not None:
        train_data = dataset
        eval_data = val_dataset
        print(f"Using separate validation set: {len(eval_data)} samples")
    else:
        # Split dataset
        random.shuffle(dataset)
        split_idx = int(len(dataset) * (1 - eval_split))
        train_data = dataset[:split_idx]
        eval_data = dataset[split_idx:]
        print(f"Split training data: {len(train_data)} train, {len(eval_data)} eval")

    # Prepare data for DataLoader
    # We need to pad sequences to the same length, but cap at model's max_seq_len - 1
    # (because positions start at 1, so max position is max_seq_len - 1)
    # Calculate max length across both train and eval datasets
    all_data = train_data + (eval_data if eval_data else [])
    dataset_max_len = max(len(seq) for _, seq in all_data) if all_data else 100
    effective_max_len = min(dataset_max_len, model.max_seq_len - 1)

    def collate_fn(batch):
        activations = []
        action_seqs = []
        seq_lens = []

        for activation, action_seq in batch:
            activations.append(activation)
            # Truncate if longer than model's max_seq_len - 1 (to account for position embedding limit)
            if len(action_seq) > model.max_seq_len - 1:
                action_seq = action_seq[:model.max_seq_len - 1]
            seq_lens.append(len(action_seq))
            # Pad sequence to effective_max_len
            if len(action_seq) < effective_max_len:
                padded_seq = torch.cat(
                    [action_seq, torch.full((effective_max_len - len(action_seq),), -100, dtype=torch.long)]
                )
            else:
                padded_seq = action_seq
            # Ensure all action values are in valid range [0, vocab_size-1] (ignore padding -100)
            padded_seq = torch.where(padded_seq == -100, padded_seq, torch.clamp(padded_seq, 0, model.vocab_size - 1))
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

        if len(eval_data) == 0:
            # No eval data (eval_split was 0 or all data used for training)
            avg_eval_loss = 0.0
            token_accuracy = 0.0
            sequence_accuracy = 0.0
        else:
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
            token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
            sequence_accuracy = exact_matches / total_sequences if total_sequences > 0 else 0.0

        eval_losses.append(avg_eval_loss)

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
