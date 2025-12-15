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

# Special token IDs
# Actions: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
SOS_TOKEN_ID = 4  # Start-of-Sequence token
EOS_TOKEN_ID = 5  # End-of-Sequence token
PAD_TOKEN_ID = 6  # Padding token

# Vocabulary size: 4 actions + 3 special tokens = 7
DEFAULT_VOCAB_SIZE = 7


class ActionDecoderProbe(nn.Module):
    """Small transformer decoder that predicts action sequences from LLM activations.

    This model uses the LLM activation as input, splitting it into multiple memory tokens
    for cross-attention. This allows different decoder positions to attend to different
    aspects of the activation.

    Args:
        activation_dim: Dimension of the input LLM activation vector (e.g., 2880)
        vocab_size: Vocabulary size (default: 7 for 0-3=actions, 4=SOS, 5=EOS, 6=PAD)
        n_layer: Number of transformer decoder layers (default: 4)
        n_head: Number of attention heads (default: 4)
        num_memory_tokens: Number of memory tokens to split activation into (default: 8).
            activation_dim must be divisible by num_memory_tokens.
            Each token will have dimension activation_dim // num_memory_tokens.
        max_seq_len: Maximum sequence length for action predictions (default: 100)
        dropout: Dropout rate (default: 0.1)
        eos_loss_weight_threshold: Threshold for position-weighted EOS loss (default: 3.0).
            EOS loss weight = min(1.0, (position + 1) / threshold). Higher values allow earlier EOS.
    """

    def __init__(
        self,
        activation_dim: int,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        n_layer: int = 4,
        n_head: int = 4,
        num_memory_tokens: int = 8,
        max_seq_len: int = 100,
        dropout: float = 0.1,
        eos_loss_weight_threshold: float = 3.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.activation_dim = activation_dim
        self.eos_loss_weight_threshold = eos_loss_weight_threshold
        
        # Multi-token memory configuration
        # Split the activation into num_memory_tokens tokens, each with memory_token_dim dimensions
        self.num_memory_tokens = num_memory_tokens
        
        if activation_dim % num_memory_tokens != 0:
            raise ValueError(
                f"activation_dim ({activation_dim}) must be divisible by num_memory_tokens ({num_memory_tokens}). "
                f"Try num_memory_tokens in: {[i for i in [4, 5, 6, 8, 10, 12, 16, 20] if activation_dim % i == 0]}"
            )
        
        self.memory_token_dim = activation_dim // num_memory_tokens  # e.g., 2880 // 8 = 360
        
        # Validate that memory_token_dim is divisible by n_head for multi-head attention
        if self.memory_token_dim % n_head != 0:
            raise ValueError(
                f"memory_token_dim ({self.memory_token_dim}) must be divisible by n_head ({n_head}). "
                f"activation_dim={activation_dim}, num_memory_tokens={num_memory_tokens}"
            )
        
        # Store for reference (n_embd is now memory_token_dim)
        self.n_embd = self.memory_token_dim

        # Action token embeddings (dimension matches memory tokens)
        self.action_embeddings = nn.Embedding(vocab_size, self.memory_token_dim)

        # Position embeddings for action sequence
        self.position_embeddings = nn.Embedding(max_seq_len, self.memory_token_dim)

        # Transformer decoder blocks (d_model = memory_token_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.memory_token_dim,
            nhead=n_head,
            dim_feedforward=self.memory_token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder_layers = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)

        # Output projection to vocab
        self.lm_head = nn.Linear(self.memory_token_dim, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stronger forward pass signal."""
        # Using larger std (0.1) instead of GPT-2 default (0.02) 
        # to ensure stronger signal through the network
        nn.init.normal_(self.action_embeddings.weight, std=0.1)

    def forward(
        self,
        activation: torch.Tensor,
        decoder_input: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the decoder probe.

        Args:
            activation: LLM activation tensor of shape (batch_size, activation_dim)
            decoder_input: Decoder input sequence of shape (batch_size, seq_len) for teacher forcing.
                Should start with SOS token: [SOS, action_0, action_1, ...]
                If None and labels is provided, constructs from labels (SOS + labels[:-1]).
            labels: Target sequence of shape (batch_size, seq_len) for computing loss.
                Should be: [action_0, action_1, ..., action_n, EOS, PAD, ...]
                If None, only returns logits.
            attention_mask: Attention mask of shape (batch_size, seq_len) where 1 = valid token, 0 = PAD.
                If None, created from decoder_input (1 where != PAD_TOKEN_ID).

        Returns:
            If labels is not None, returns (loss, logits) tuple.
            Otherwise, returns logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        batch_size = activation.shape[0]
        device = activation.device

        # Convert activation to float32 if needed (HF activations may be bfloat16)
        if activation.dtype != torch.float32:
            activation = activation.to(torch.float32)

        # Reshape activation into multiple memory tokens for richer cross-attention
        # Instead of 1 token of 2880 dims, we get 8 tokens of 360 dims each
        # This allows different decoder positions to attend to different "aspects" of the activation
        activation_embed = activation.view(batch_size, self.num_memory_tokens, self.memory_token_dim)
        # Shape: (batch_size, num_memory_tokens, memory_token_dim) e.g., (32, 8, 360)

        # Construct decoder_input from labels if not provided (teacher forcing)
        if decoder_input is None and labels is not None:
            # For training: decoder_input = [SOS] + labels[:-1] (with EOS replaced by PAD)
            # labels = [action_0, action_1, ..., action_n, EOS, PAD, ...]
            # decoder_input = [SOS, action_0, action_1, ..., action_n, PAD, ...]
            # We want to predict action_i given [SOS, action_0, ..., action_{i-1}]
            # And predict EOS given [SOS, action_0, ..., action_n]
            seq_len = labels.shape[1]
            if seq_len > 0:
                labels_clamped = torch.clamp(labels, 0, self.vocab_size - 1)
                # Replace EOS with PAD in labels for decoder_input (EOS should not appear in input)
                labels_for_input = torch.where(labels_clamped == EOS_TOKEN_ID, PAD_TOKEN_ID, labels_clamped)
                # Shift by one position and prepend SOS
                decoder_input = torch.cat(
                    [torch.full((batch_size, 1), SOS_TOKEN_ID, dtype=torch.long, device=device), 
                     labels_for_input[:, :-1]], dim=1
                )
            else:
                decoder_input = torch.full((batch_size, 1), SOS_TOKEN_ID, dtype=torch.long, device=device)
        elif decoder_input is None:
            # For inference: start with SOS token
            decoder_input = torch.full((batch_size, 1), SOS_TOKEN_ID, dtype=torch.long, device=device)
        
        # Create attention mask if not provided
        if attention_mask is None:
            # 1 where token is not PAD, 0 where token is PAD
            attention_mask = (decoder_input != PAD_TOKEN_ID).long()  # (batch_size, seq_len)
        
        # Truncate to max_seq_len if needed
        if decoder_input.shape[1] > self.max_seq_len:
            decoder_input = decoder_input[:, :self.max_seq_len]
            attention_mask = attention_mask[:, :self.max_seq_len]
        
        if labels is not None and labels.shape[1] > self.max_seq_len:
            labels = labels[:, :self.max_seq_len]

        # Build decoder input sequence embeddings
        seq_len = decoder_input.shape[1]
        
        # Clamp decoder_input to valid range [0, vocab_size-1]
        decoder_input = torch.clamp(decoder_input, 0, self.vocab_size - 1)
        
        # Embed decoder input tokens
        decoder_embeds = self.action_embeddings(decoder_input)  # (batch_size, seq_len, n_embd)
        
        # CRITICAL DEBUG: Check if action_embeddings output requires grad
        if self.training and torch.isnan(decoder_embeds).any():
            print("ERROR: NaN in decoder_embeds immediately after action_embeddings!")
            print(f"  decoder_input sample: {decoder_input[0, :5].cpu().tolist()}")
            print(f"  action_embeddings.weight has NaN: {torch.isnan(self.action_embeddings.weight).any()}")
            decoder_embeds = torch.nan_to_num(decoder_embeds, nan=0.0)
        
        # Add position embeddings (starting from position 0 for SOS, 1 for first action, etc.)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        # Clamp positions to valid range [0, max_seq_len-1]
        positions = torch.clamp(positions, 0, self.max_seq_len - 1)
        pos_embeds = self.position_embeddings(positions)
        decoder_embeds = decoder_embeds + pos_embeds  # (batch_size, seq_len, n_embd)
        
        # Check if decoder_embeds still requires grad after adding position embeddings
        if self.training and not decoder_embeds.requires_grad:
            print("ERROR: decoder_embeds lost requires_grad after position embedding addition!")
        
        # Store decoder_embeds for regularization (to ensure gradients flow to action_embeddings)
        if labels is not None:
            self._decoder_embeds_for_reg = decoder_embeds

        # Create attention masks for transformer decoder
        # A. Causal (Look-Ahead) Mask: Prevents attending to future tokens
        #    Creates a triangular mask where position i can only attend to positions <= i
        #    Shape: (seq_len, seq_len)
        #    PyTorch convention: True = MASKED (cannot attend), False = UNMASKED (can attend)
        #    torch.triu(..., diagonal=1) gives True above diagonal = future positions masked
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        
        # B. Padding Mask: Prevents attending to PAD tokens
        #    Shape: (batch_size, seq_len)
        #    True = PAD token (should be masked), False = valid token
        tgt_key_padding_mask = (attention_mask == 0)  # True where PAD, False where valid
        
        # PyTorch TransformerDecoderLayer automatically combines:
        # 1. Causal mask (tgt_mask) - applied in self-attention
        # 2. Padding mask (tgt_key_padding_mask) - applied to mask out PAD tokens
        # The padding mask is combined with causal mask: tokens cannot attend to:
        #   - Future tokens (causal mask)
        #   - PAD tokens (padding mask)
        
        decoder_output = self.decoder_layers(
            tgt=decoder_embeds, 
            memory=activation_embed,
            tgt_mask=tgt_mask,  # Causal mask: (seq_len, seq_len)
            tgt_key_padding_mask=tgt_key_padding_mask  # Padding mask: (batch_size, seq_len)
        )  # (batch_size, seq_len, n_embd)
        
        # Store decoder_output for regularization loss (to ensure gradient flow to decoder_layers)
        # This is CRITICAL: regularizing decoder_output gives gradients to decoder_layers
        # Regularizing decoder_embeds only gives gradients to action_embeddings
        if labels is not None:
            self._decoder_output_for_reg = decoder_output
        
        # Check for NaN in decoder_output immediately and fix to prevent gradient flow issues
        if torch.isnan(decoder_output).any():
            if self.training:
                print("Warning: NaN in decoder_output! This will break gradient flow to action_embeddings.")
                print(f"  decoder_embeds stats: min={decoder_embeds.min().item():.4f}, max={decoder_embeds.max().item():.4f}")
                print(f"  decoder_embeds has NaN: {torch.isnan(decoder_embeds).any()}")
            # Replace NaN with zeros to maintain computation graph
            decoder_output = torch.nan_to_num(decoder_output, nan=0.0)

        # Project to vocabulary
        logits = self.lm_head(decoder_output)  # (batch_size, seq_len, vocab_size)
        
        # Debug: Check for NaN/Inf in logits (only in training mode)
        if self.training and torch.isnan(logits).any():
            print("Warning: NaN detected in logits!")
            print(f"  Decoder output stats: min={decoder_output.min().item():.4f}, max={decoder_output.max().item():.4f}")
            print(f"  Decoder output has NaN: {torch.isnan(decoder_output).any()}")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        elif self.training and torch.isinf(logits).any():
            print("Warning: Inf detected in logits!")
            logits = torch.clamp(logits, min=-1e6, max=1e6)

        if labels is not None:
            # Truncate labels to match logits length
            seq_len_used = logits.shape[1]
            if labels.shape[1] > seq_len_used:
                labels = labels[:, :seq_len_used]
            
            # Clamp labels to valid range [0, vocab_size-1]
            labels = torch.clamp(labels, 0, self.vocab_size - 1)
            
            # Compute per-token loss for position-weighted EOS loss
            logits_flat = logits.view(-1, self.vocab_size)  # (batch_size * seq_len, vocab_size)
            labels_flat = labels.view(-1)  # (batch_size * seq_len)
            
            # Create mask for PAD tokens (ignore PAD in loss)
            pad_mask = (labels_flat != PAD_TOKEN_ID)  # (batch_size * seq_len,)
            
            # Compute per-token loss (reduction='none')
            # Use ignore_index=PAD_TOKEN_ID to ignore padding tokens
            per_token_loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=PAD_TOKEN_ID, reduction='none')
            
            # Check for NaN in per_token_loss before processing
            if torch.isnan(per_token_loss).any():
                print(f"Warning: NaN in per_token_loss! Replacing with zeros for PAD positions.")
                # Replace NaN with 0 (they should be 0 for PAD tokens anyway due to ignore_index)
                per_token_loss = torch.nan_to_num(per_token_loss, nan=0.0)
            
            # Apply position-weighted loss for EOS token to prevent mode collapse
            # Reshape to (batch_size, seq_len) for position indexing
            per_token_loss_2d = per_token_loss.view(batch_size, seq_len_used)
            labels_2d = labels_flat.view(batch_size, seq_len_used)
            pad_mask_2d = pad_mask.view(batch_size, seq_len_used)
            
            # Create position indices: [0, 1, 2, ..., seq_len-1]
            position_indices = torch.arange(seq_len_used, device=labels.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
            
            # Compute EOS loss weights: weight = min(1.0, (position + 1) / threshold)
            # This gives lower weight to early EOS predictions
            eos_weights = torch.clamp((position_indices + 1.0) / self.eos_loss_weight_threshold, max=1.0)
            
            # Apply weights: use eos_weights for EOS tokens, 1.0 for non-EOS tokens
            is_eos = (labels_2d == EOS_TOKEN_ID) & pad_mask_2d
            loss_weights = torch.where(is_eos, eos_weights, torch.ones_like(eos_weights))
            
            # Apply weights to per-token loss
            weighted_loss = per_token_loss_2d * loss_weights
            
            # Mask out padding tokens and compute mean
            # CRITICAL FIX: Use masked_fill instead of indexing to maintain computation graph
            # Indexing with pad_mask_2d creates a new tensor that breaks gradient flow
            # masked_fill maintains connection to ALL positions while zeroing invalid ones
            num_valid = pad_mask_2d.sum()
            
            # Safety check: ensure we have valid tokens
            if num_valid == 0:
                # If no valid tokens (shouldn't happen), create a small loss from logits to maintain gradient flow
                print("Warning: No valid tokens in loss calculation! This should not happen.")
                # Use a small regularization loss that maintains gradient flow to all parameters
                loss = (logits ** 2).sum() * 1e-6  # Small L2 regularization that flows gradients
            else:
                # Use masked_fill to zero out invalid positions while maintaining computation graph
                # This ensures gradients flow through ALL positions: decoder_output → decoder_layers → action_embeddings
                masked_loss = weighted_loss.masked_fill(~pad_mask_2d, 0.0)
                # Sum over ALL positions (masked positions are zero, so they don't contribute to loss)
                # But gradients still flow through them to maintain the computation graph
                token_level_loss = masked_loss.sum() / num_valid.clamp(min=1)
                
                # SEQUENCE-LEVEL LOSS: Compute negative log probability of correct sequences
                # This directly optimizes for sequence accuracy, not just token accuracy
                # For each sequence, compute P(correct_sequence) and maximize it
                # Vectorized computation to maintain gradients
                log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
                
                # Gather log probabilities of correct tokens: log_probs[b, pos, labels[b, pos]]
                # Use gather to select the correct token's log prob at each position
                labels_expanded = labels_2d.unsqueeze(-1)  # (batch_size, seq_len, 1)
                correct_token_log_probs = log_probs.gather(dim=-1, index=labels_expanded).squeeze(-1)  # (batch_size, seq_len)
                
                # Mask out PAD tokens and sum to get log probability of each correct sequence
                masked_log_probs = correct_token_log_probs * pad_mask_2d.float()  # (batch_size, seq_len)
                seq_log_probs = masked_log_probs.sum(dim=1)  # (batch_size,) - sum of log probs for each sequence
                
                # Normalize by sequence length (number of valid tokens) to make comparable
                seq_lengths = pad_mask_2d.sum(dim=1).float()  # (batch_size,)
                seq_lengths = seq_lengths.clamp(min=1.0)  # Avoid division by zero
                normalized_seq_log_probs = seq_log_probs / seq_lengths  # (batch_size,)
                
                # Negative log-likelihood (we want to minimize this, so maximize likelihood)
                sequence_level_loss = -normalized_seq_log_probs.mean()  # Average across batch
                
                # Combine token-level and sequence-level losses
                # Sequence accuracy is what matters - token accuracy is misleading (25% random baseline)
                # Keep tiny token loss (5%) only for gradient flow, but sequence loss (95%) is primary
                main_loss = 0.05 * token_level_loss + 0.95 * sequence_level_loss
                
                # CRITICAL FIX: Add regularization to ensure gradients flow to all layers
                # The main loss gradient is very small for decoder_layers (1e-8) because:
                # 1. Most positions are PAD (don't contribute to loss)
                # 2. Attention softmax causes vanishing gradients
                # 3. Only ~20% of tokens are valid
                # We add stronger regularization to force meaningful gradient flow
                reg_loss = torch.tensor(0.0, device=main_loss.device)
                
                # Regularize decoder_output to train decoder_layers
                # Weight of 0.001 provides gradient signal without pushing values to zero
                # (0.1 was too aggressive and caused NaN during inference)
                if hasattr(self, '_decoder_output_for_reg') and self._decoder_output_for_reg is not None:
                    reg_loss = reg_loss + (self._decoder_output_for_reg ** 2).mean() * 0.001
                
                # Regularize decoder_embeds to train action_embeddings and position_embeddings
                if hasattr(self, '_decoder_embeds_for_reg') and self._decoder_embeds_for_reg is not None:
                    reg_loss = reg_loss + (self._decoder_embeds_for_reg ** 2).mean() * 0.001
                
                loss = main_loss + reg_loss
            
            # Check for NaN or Inf - if present, use simple unweighted loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Loss is NaN/Inf, using simple unweighted loss. num_valid={num_valid}")
                # Use simple unweighted loss without position weighting
                simple_loss = per_token_loss[pad_mask].sum() / pad_mask.sum().clamp(min=1)
                if torch.isnan(simple_loss) or torch.isinf(simple_loss):
                    print("Warning: Simple loss also NaN/Inf, using regularization loss")
                    # Last resort: use L2 regularization on logits to maintain gradient flow
                    loss = (logits ** 2).sum() * 1e-6
                else:
                    loss = simple_loss
            
            # Clean up temporary storage
            if hasattr(self, '_decoder_output_for_reg'):
                delattr(self, '_decoder_output_for_reg')
            if hasattr(self, '_decoder_embeds_for_reg'):
                delattr(self, '_decoder_embeds_for_reg')
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

        # Reshape activation into multiple memory tokens for cross-attention
        # Shape: (batch_size, num_memory_tokens, memory_token_dim) e.g., (32, 8, 360)
        activation_embed = activation.view(batch_size, self.num_memory_tokens, self.memory_token_dim)

        # Start with SOS token
        decoder_input = torch.full((batch_size, 1), SOS_TOKEN_ID, dtype=torch.long, device=device)
        generated = []
        
        with torch.no_grad():
            # Generate tokens autoregressively
            for step in range(effective_max_length):
                # Early stopping if we've generated enough tokens
                if len(generated) > 0 and len(generated) >= effective_max_length:
                    break
                # Embed decoder input (SOS + generated actions so far)
                decoder_embeds = self.action_embeddings(decoder_input)  # (batch_size, seq_len, n_embd)
                
                # Add position embeddings
                seq_len = decoder_input.shape[1]
                positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                positions = torch.clamp(positions, 0, self.max_seq_len - 1)
                pos_embeds = self.position_embeddings(positions)
                decoder_embeds = decoder_embeds + pos_embeds
                
                # Create attention masks for generation
                # A. Causal mask: Prevent attending to future tokens
                #    PyTorch convention: True = MASKED (cannot attend), False = UNMASKED (can attend)
                tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
                
                # B. Padding mask: No padding during generation (all tokens are valid)
                attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
                tgt_key_padding_mask = (attention_mask == 0)  # False for all (no padding during generation)
                
                # Run through decoder with both causal and padding masks
                decoder_output = self.decoder_layers(
                    tgt=decoder_embeds, 
                    memory=activation_embed,
                    tgt_mask=tgt_mask,  # Causal mask
                    tgt_key_padding_mask=tgt_key_padding_mask  # Padding mask (all False = no padding)
                )
                
                # Check for NaN in decoder output
                if torch.isnan(decoder_output).any():
                    print(f"Warning: NaN detected in decoder_output at step {step}")
                    print(f"  This indicates the model weights may be corrupted or untrained")
                    print(f"  decoder_embeds stats: min={decoder_embeds.min().item():.4f}, max={decoder_embeds.max().item():.4f}")
                    print(f"  decoder_embeds has NaN: {torch.isnan(decoder_embeds).any()}")
                    # Replace NaN to allow generation to continue (though results will be garbage)
                    decoder_output = torch.nan_to_num(decoder_output, nan=0.0)
                
                logits = self.lm_head(decoder_output)  # (batch_size, seq_len, vocab_size)
                
                # Check for NaN in logits
                if torch.isnan(logits).any():
                    print(f"Warning: NaN detected in logits at step {step}, stopping generation")
                    break

                # Get logits for the last position (next token prediction)
                next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
                
                # Check for NaN before processing
                if torch.isnan(next_token_logits).any():
                    print(f"Warning: NaN in next_token_logits at step {step}, stopping generation")
                    break

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
                
                # Debug: Check for NaN/Inf in probabilities
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    # Instead of stopping, use uniform distribution over VALID ACTIONS ONLY (0-3)
                    # This allows generation to continue (though results may be poor)
                    # Don't sample SOS (4), EOS (5), or PAD (6)
                    probs = torch.zeros_like(probs)
                    probs[:, :4] = 0.25  # Uniform over actions 0, 1, 2, 3 only
                
                # Check for negative probabilities (shouldn't happen after softmax, but just in case)
                if (probs < 0).any():
                    print(f"Warning: Negative probabilities at step {step}, clamping to 0")
                    probs = torch.clamp(probs, min=0.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize
                
                next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
                next_token = next_token.squeeze(1)  # (batch_size,)
                generated.append(next_token)
                
                # Append to decoder_input for next iteration (but don't include EOS in input)
                # Only append if not EOS (EOS signals end, so we don't need to continue)
                eos_predicted = (next_token == EOS_TOKEN_ID)
                if eos_predicted.any():
                    # If all sequences predicted EOS, stop completely
                    if eos_predicted.all():
                        break
                    # For mixed batches, only append non-EOS tokens to decoder_input
                    # Replace EOS with PAD for sequences that predicted EOS (they're done)
                    next_token_for_input = torch.where(eos_predicted, torch.full_like(next_token, PAD_TOKEN_ID), next_token)
                else:
                    next_token_for_input = next_token
                
                # Update decoder_input for next iteration
                decoder_input = torch.cat([decoder_input, next_token_for_input.unsqueeze(1)], dim=1)
                
                # Check if EOS token was sampled - EOS takes priority over heuristic stopping
                if eos_predicted.any():
                    # For mixed batches, continue but EOS will be filtered out later
                    # Continue generation for non-EOS sequences only
                    pass
                
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
            error_str = str(e)
            if "404" in error_str or "Not Found" in error_str or "RepositoryNotFoundError" in str(type(e).__name__):
                print(f"⚠️  Warning: Could not access repository '{repo_id}': {error_str}")
                print("   This usually means:")
                print("   1. The repository is private and requires authentication")
                print("   2. Run: huggingface-cli login")
                print("   3. Or the repository doesn't exist at this path")
                print("   Falling back to trying env indices 1-2400 (this will be slower)...")
            else:
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
            error_str = str(e)
            # 404s are expected for missing grids, but if we get many 404s at the start, it might be auth issue
            if "404" in error_str or "Not Found" in error_str:
                # Only print first few 404s to avoid spam
                if loaded == 0 and env_idx <= 5:
                    print(f"  env_{env_idx}: Not found (404) - this is normal if file doesn't exist")
                # If we've tried many and got all 404s, might be auth issue
                if env_idx == 10 and loaded == 0:
                    print(f"  ⚠️  Warning: All first 10 files returned 404. This might indicate:")
                    print(f"     - Authentication needed (run: huggingface-cli login)")
                    print(f"     - Wrong repository path")
                    print(f"     - Files don't exist at these indices")
            else:
                # Non-404 errors are more serious
                if loaded == 0 or env_idx <= 5:
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
        # Construct path_in_repo: if activations_dir already starts with "activations/", use as-is
        # Otherwise, prepend "activations/"
        if activations_dir.startswith("activations/"):
            path_in_repo = activations_dir
        else:
            path_in_repo = f"activations/{activations_dir}"
        activations = load_activations_from_hf(
            repo_id=hf_repo_id,
            path_in_repo=path_in_repo,
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
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Load action sequences from CSV file.

    Supports two CSV formats:
    1. Format with 'action_sequence' column containing JSON array: [0, 1, 2, 3, ...]
    2. Format with 'last_action' column per trajectory step (legacy)

    Args:
        csv_path: Path to CSV file with trajectory data
        grid_indices: List of grid indices to load. If None, loads all.
        max_seq_len: Maximum sequence length. If None, uses the longest sequence.

    Returns:
        Tuple of (decoder_inputs, targets) dictionaries:
        - decoder_inputs: Maps grid index to decoder input tensor [SOS, action_0, action_1, ...]
        - targets: Maps grid index to target tensor [action_0, action_1, ..., EOS]
    """
    import json
    import pandas as pd

    df = pd.read_csv(csv_path)

    decoder_inputs = {}
    targets = {}
    
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
                        
                        # Create target: [action_0, action_1, ..., action_n, EOS]
                        if add_eos_token:
                            target = torch.cat([action_seq, torch.tensor([EOS_TOKEN_ID], dtype=torch.long)])
                        else:
                            target = action_seq
                        
                        # Create decoder_input: [SOS, action_0, action_1, ..., action_n]
                        decoder_input = torch.cat([torch.tensor([SOS_TOKEN_ID], dtype=torch.long), action_seq])
                        
                        decoder_inputs[env_idx] = decoder_input
                        targets[env_idx] = target
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
                
                # Create target: [action_0, action_1, ..., action_n, EOS]
                if add_eos_token:
                    target = torch.cat([action_seq, torch.tensor([EOS_TOKEN_ID], dtype=torch.long)])
                else:
                    target = action_seq
                
                # Create decoder_input: [SOS, action_0, action_1, ..., action_n]
                decoder_input = torch.cat([torch.tensor([SOS_TOKEN_ID], dtype=torch.long), action_seq])
                
                decoder_inputs[env_idx] = decoder_input
                targets[env_idx] = target
    else:
        raise ValueError("CSV must have either 'action_sequence' or 'last_action' column")

    return decoder_inputs, targets


def create_dataset(
    activations: dict[int, torch.Tensor],
    decoder_inputs: dict[int, torch.Tensor],
    targets: dict[int, torch.Tensor],
    max_seq_len: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Create dataset tuples of (activation, decoder_input, target).

    Args:
        activations: Dictionary mapping grid index to activation tensor
        decoder_inputs: Dictionary mapping grid index to decoder input tensor [SOS, action_0, ...]
        targets: Dictionary mapping grid index to target tensor [action_0, ..., EOS]
        max_seq_len: Maximum sequence length. Sequences longer than this are truncated.

    Returns:
        List of (activation, decoder_input, target) tuples
    """
    dataset = []

    # Find common grid indices
    common_indices = set(activations.keys()) & set(decoder_inputs.keys()) & set(targets.keys())

    for grid_idx in common_indices:
        activation = activations[grid_idx]
        decoder_input = decoder_inputs[grid_idx]
        target = targets[grid_idx]

        # Validate and clamp token values to valid range [0, vocab_size-1]
        decoder_input = torch.clamp(decoder_input, 0, DEFAULT_VOCAB_SIZE - 1)
        target = torch.clamp(target, 0, DEFAULT_VOCAB_SIZE - 1)
        
        # Truncate if needed
        if max_seq_len is not None:
            if len(decoder_input) > max_seq_len:
                decoder_input = decoder_input[:max_seq_len]
            if len(target) > max_seq_len:
                target = target[:max_seq_len]

        dataset.append((activation, decoder_input, target))

    return dataset


def train_decoder_probe(
    model: ActionDecoderProbe,
    dataset: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    eval_split: float = 0.2,
    val_dataset: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    save_path: str | None = None,
    max_decoder_input_len: int | None = None,
    max_target_len: int | None = None,
) -> dict:
    """Train the decoder probe model.

    Args:
        model: ActionDecoderProbe model to train
        dataset: List of (activation, decoder_input, target) tuples
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

    # Try to move model to device, with fallback to CPU if CUDA fails
    try:
        model = model.to(device)
        # Test if CUDA actually works by creating a test tensor
        if device == "cuda":
            test_tensor = torch.tensor([1.0]).to(device)
    except RuntimeError as e:
        if "cuda" in str(e).lower() or "CUDA" in str(e):
            print(f"⚠️  Warning: CUDA initialization failed: {e}")
            print("   Falling back to CPU. Training will be slower but will work.")
            device = "cpu"
            model = model.to(device)
        else:
            raise

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
    # Calculate max length across both train and eval datasets
    # If max lengths are provided (from all datasets including test), use those
    # Otherwise calculate from available data
    if max_decoder_input_len is None or max_target_len is None:
        all_data = train_data + (eval_data if eval_data else [])
        if max_decoder_input_len is None:
            max_decoder_input_len = max(len(decoder_input) for _, decoder_input, _ in all_data) if all_data else 100
        if max_target_len is None:
            max_target_len = max(len(target) for _, _, target in all_data) if all_data else 100
    
    effective_max_decoder_len = min(max_decoder_input_len, model.max_seq_len)
    effective_max_target_len = min(max_target_len, model.max_seq_len)

    def collate_fn(batch):
        activations = []
        decoder_inputs = []
        targets = []
        attention_masks = []

        for activation, decoder_input, target in batch:
            activations.append(activation)
            
            # Truncate if longer than max_seq_len
            if len(decoder_input) > model.max_seq_len:
                decoder_input = decoder_input[:model.max_seq_len]
            if len(target) > model.max_seq_len:
                target = target[:model.max_seq_len]
            
            # Pad decoder_input to effective_max_decoder_len with PAD tokens
            if len(decoder_input) < effective_max_decoder_len:
                padding = torch.full(
                    (effective_max_decoder_len - len(decoder_input),), 
                    PAD_TOKEN_ID, 
                    dtype=torch.long
                )
                decoder_input = torch.cat([decoder_input, padding])
            
            # Pad target to effective_max_target_len with PAD tokens
            if len(target) < effective_max_target_len:
                padding = torch.full(
                    (effective_max_target_len - len(target),), 
                    PAD_TOKEN_ID, 
                    dtype=torch.long
                )
                target = torch.cat([target, padding])
            
            # Create attention mask: 1 for valid tokens, 0 for PAD
            attention_mask = (decoder_input != PAD_TOKEN_ID).long()
            
            decoder_inputs.append(decoder_input)
            targets.append(target)
            attention_masks.append(attention_mask)

        activations_tensor = torch.stack(activations)
        decoder_inputs_tensor = torch.stack(decoder_inputs)
        targets_tensor = torch.stack(targets)
        attention_masks_tensor = torch.stack(attention_masks)
        
        return activations_tensor, decoder_inputs_tensor, targets_tensor, attention_masks_tensor

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Optimizer - using same LR for all params, but we scale decoder_layers gradients by 1000x
    # in the backward pass to combat vanishing gradients
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"  Optimizer: LR = {learning_rate:.2e}, decoder_layers gradients scaled by 1000x")

    # Training loop
    train_losses = []
    eval_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        for batch_idx, (activations_batch, decoder_inputs_batch, targets_batch, attention_masks_batch) in enumerate(train_loader):
            activations_batch = activations_batch.to(device).to(torch.float32)
            decoder_inputs_batch = decoder_inputs_batch.to(device)
            targets_batch = targets_batch.to(device)
            attention_masks_batch = attention_masks_batch.to(device)

            # Debug: Print first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                print("\n🔍 Debug: First training batch")
                print(f"  Batch size: {activations_batch.shape[0]}")
                print(f"  Activation shape: {activations_batch.shape}")
                print(f"  Decoder input shape: {decoder_inputs_batch.shape}")
                print(f"  Targets shape: {targets_batch.shape}")
                print(f"  Attention mask shape: {attention_masks_batch.shape}")
                print(f"  Sample decoder_input[0]: {decoder_inputs_batch[0].cpu().tolist()}")
                print(f"  Sample target[0]: {targets_batch[0].cpu().tolist()}")
                print(f"  Sample attention_mask[0]: {attention_masks_batch[0].cpu().tolist()}")
                # Check token distribution in targets
                unique_targets, counts = torch.unique(targets_batch, return_counts=True)
                print(f"  Target token distribution: {dict(zip(unique_targets.cpu().tolist(), counts.cpu().tolist()))}")

            # Forward pass
            loss, logits = model(
                activations_batch,
                decoder_input=decoder_inputs_batch,
                labels=targets_batch,
                attention_mask=attention_masks_batch
            )

            # Debug: Check loss and logits
            if epoch == 0 and batch_idx == 0:
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")
                print(f"  Logits has NaN: {torch.isnan(logits).any()}")
                print(f"  Logits has Inf: {torch.isinf(logits).any()}")
                # Check predictions
                preds = torch.argmax(logits, dim=-1)
                print(f"  Sample predictions[0]: {preds[0].cpu().tolist()}")
                unique_preds, pred_counts = torch.unique(preds, return_counts=True)
                print(f"  Prediction token distribution: {dict(zip(unique_preds.cpu().tolist(), pred_counts.cpu().tolist()))}")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # GRADIENT SCALING: Multiply decoder_layers gradients to combat vanishing gradients
            # This is more direct than just using higher learning rate
            gradient_scale_factor = 1000.0  # Scale decoder_layers gradients by 1000x
            for name, param in model.named_parameters():
                if "decoder_layers" in name and param.grad is not None:
                    param.grad.data *= gradient_scale_factor
            
            # Debug: Check gradients and trace gradient flow
            if epoch == 0 and batch_idx == 0:
                total_grad_norm = 0.0
                num_params = 0
                zero_grad_params = []
                
                # Check decoder_layers parameters to see if they get gradients
                print(f"\n  🔍 Checking decoder_layers gradients (with higher precision):")
                decoder_layer_grads = []
                for name, param in model.named_parameters():
                    if "decoder_layers" in name and "0." in name:
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            decoder_layer_grads.append((name, grad_norm))
                            if len(decoder_layer_grads) <= 3:  # Print first 3
                                # Use scientific notation to see tiny gradients
                                print(f"    {name} grad norm: {grad_norm:.2e}")
                        else:
                            if len(decoder_layer_grads) == 0:  # Only print once
                                print(f"    ⚠️  {name} has NO gradient!")
                if not decoder_layer_grads:
                    print(f"    ⚠️  CRITICAL: No decoder_layers parameters have gradients!")
                elif all(g[1] == 0.0 for g in decoder_layer_grads):
                    print(f"    ⚠️  CRITICAL: All decoder_layers gradients are exactly ZERO!")
                    print(f"    The regularization on decoder_output should fix this.")
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_grad_norm = param.grad.data.norm(2)
                        total_grad_norm += param_grad_norm.item() ** 2
                        num_params += 1
                        if "lm_head" in name or "action_embeddings" in name or "decoder_layers.0" in name:
                            print(f"  {name} grad norm: {param_grad_norm.item():.6f}")
                            if param_grad_norm.item() == 0.0:
                                zero_grad_params.append(name)
                    else:
                        if "lm_head" in name or "action_embeddings" in name:
                            print(f"  ⚠️  {name} has NO gradient!")
                            zero_grad_params.append(name)
                if zero_grad_params:
                    print(f"  ⚠️  Parameters with ZERO gradient: {zero_grad_params}")
                    print(f"  This means these parameters are not being trained!")
                total_grad_norm = total_grad_norm ** (1. / 2)
                print(f"  Total gradient norm: {total_grad_norm:.6f}")
                print(f"  Parameters with gradients: {num_params}/{sum(1 for _ in model.parameters())}")
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
                for batch_idx, (activations_batch, decoder_inputs_batch, targets_batch, attention_masks_batch) in enumerate(eval_loader):
                    activations_batch = activations_batch.to(device).to(torch.float32)
                    decoder_inputs_batch = decoder_inputs_batch.to(device)
                    targets_batch = targets_batch.to(device)
                    attention_masks_batch = attention_masks_batch.to(device)

                    loss, logits = model(
                        activations_batch,
                        decoder_input=decoder_inputs_batch,
                        labels=targets_batch,
                        attention_mask=attention_masks_batch
                    )

                    # Debug: Print first eval batch
                    if batch_idx == 0:
                        print(f"\n🔍 Debug: First eval batch")
                        print(f"  Loss: {loss.item():.4f}")
                        print(f"  Loss is NaN: {torch.isnan(loss)}")
                        print(f"  Loss is Inf: {torch.isinf(loss)}")
                        print(f"  Logits shape: {logits.shape}")
                        print(f"  Logits has NaN: {torch.isnan(logits).any()}")
                        if torch.isnan(logits).any():
                            # Find where NaN is
                            nan_mask = torch.isnan(logits)
                            print(f"  NaN positions: {nan_mask.sum().item()} out of {logits.numel()}")
                            # Note: decoder_output is not in scope here, skip this check
                            pass
                        print(f"  Logits has Inf: {torch.isinf(logits).any()}")
                        # Check valid tokens
                        valid_mask = (targets_batch != PAD_TOKEN_ID) & (targets_batch != EOS_TOKEN_ID)
                        print(f"  Valid tokens in batch: {valid_mask.sum().item()}/{targets_batch.numel()}")
                        print(f"  Sample target[0]: {targets_batch[0].cpu().tolist()}")
                        print(f"  Sample decoder_input[0]: {decoder_inputs_batch[0].cpu().tolist()}")
                        # Check if action_embeddings are being used
                        print(f"  action_embeddings.requires_grad: {next(model.action_embeddings.parameters()).requires_grad}")

                    # Check for NaN/Inf before adding to eval loss
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        epoch_eval_loss += loss.item()
                        num_eval_batches += 1
                    else:
                        print(f"Warning: NaN/Inf loss in eval batch {batch_idx}, skipping...")
                        print(f"  Loss value: {loss.item()}")
                        print(f"  Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
                        print(f"  Targets stats: min={targets_batch.min().item()}, max={targets_batch.max().item()}")

                    # Compute accuracy metrics
                    predictions = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)

                    # Token-level accuracy (only count non-padding tokens)
                    # Remove PAD and EOS from target for comparison (compare only actions)
                    valid_mask = (targets_batch != PAD_TOKEN_ID) & (targets_batch != EOS_TOKEN_ID)
                    total_tokens += valid_mask.sum().item()
                    # For predictions, compare actions only (ignore EOS in predictions too)
                    pred_actions = predictions[valid_mask]
                    target_actions = targets_batch[valid_mask]
                    correct_tokens += (pred_actions == target_actions).sum().item()

                    # Sequence-level exact match (compare action sequences only, ignoring EOS/PAD)
                    for i in range(predictions.shape[0]):
                        # Get valid actions from target (remove PAD and EOS)
                        target_seq = targets_batch[i]
                        valid_target = target_seq[(target_seq != PAD_TOKEN_ID) & (target_seq != EOS_TOKEN_ID)]
                        
                        # Get predictions up to EOS or end
                        pred_seq = predictions[i]
                        # Find EOS in predictions
                        eos_idx = (pred_seq == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
                        if len(eos_idx) > 0:
                            pred_actions_only = pred_seq[:eos_idx[0]]
                        else:
                            # No EOS found, take all non-PAD
                            pred_actions_only = pred_seq[pred_seq != PAD_TOKEN_ID]
                        
                        if len(valid_target) > 0 and len(pred_actions_only) == len(valid_target):
                            if torch.equal(pred_actions_only.cpu(), valid_target.cpu()):
                                exact_matches += 1
                        total_sequences += 1

                    # Show sample predictions for first batch of last epoch
                    if epoch == num_epochs - 1 and batch_idx == 0:
                        print("\nSample predictions (first batch):")
                        for i in range(min(3, predictions.shape[0])):
                            # Get true actions (remove PAD and EOS)
                            target_seq = targets_batch[i].cpu()
                            true_actions = target_seq[(target_seq != PAD_TOKEN_ID) & (target_seq != EOS_TOKEN_ID)].tolist()
                            
                            # Get predicted actions (remove PAD and EOS)
                            pred_seq = predictions[i].cpu()
                            eos_idx = (pred_seq == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
                            if len(eos_idx) > 0:
                                pred_actions = pred_seq[:eos_idx[0]].tolist()
                            else:
                                pred_actions = pred_seq[pred_seq != PAD_TOKEN_ID].tolist()
                            
                            if len(true_actions) > 0 or len(pred_actions) > 0:
                                print(f"  Sample {i + 1}:")
                                print(f"    True:  {true_actions}")
                                print(f"    Pred:  {pred_actions}")
                                print(f"    Match: {true_actions == pred_actions}")

            avg_eval_loss = epoch_eval_loss / num_eval_batches if num_eval_batches > 0 else 0.0
            token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
            sequence_accuracy = exact_matches / total_sequences if total_sequences > 0 else 0.0

        eval_losses.append(avg_eval_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Eval Loss: {avg_eval_loss:.4f}, "
            f"Seq Acc: {sequence_accuracy:.4f} (PRIMARY) | "
            f"Token Acc: {token_accuracy:.4f} (misleading, 25% random baseline)"
        )

    # Save model if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        
        # Check for NaN in model weights before saving
        has_nan = False
        nan_params = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"Warning: NaN detected in {name} before saving!")
                has_nan = True
                nan_params.append(name)
        if has_nan:
            print(f"Warning: Model contains NaN weights in {len(nan_params)} parameters! Model may not be usable.")
            print(f"  Parameters with NaN: {nan_params[:5]}{'...' if len(nan_params) > 5 else ''}")
        else:
            print("Model weights checked: No NaN detected.")
        
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
        # Verify the saved model can be loaded
        try:
            test_load = torch.load(save_path, map_location='cpu')
            print(f"Model verification: Successfully loaded {len(test_load)} parameter groups")
        except Exception as e:
            print(f"Warning: Could not verify saved model: {e}")

    # Compute final accuracy metrics
    model.eval()
    final_token_accuracy = 0.0
    final_sequence_accuracy = 0.0
    total_tokens = 0
    correct_tokens = 0
    exact_matches = 0
    total_sequences = 0

    with torch.no_grad():
        for activations_batch, decoder_inputs_batch, targets_batch, attention_masks_batch in eval_loader:
            activations_batch = activations_batch.to(device).to(torch.float32)
            decoder_inputs_batch = decoder_inputs_batch.to(device)
            targets_batch = targets_batch.to(device)
            attention_masks_batch = attention_masks_batch.to(device)

            _, logits = model(
                activations_batch,
                decoder_input=decoder_inputs_batch,
                labels=targets_batch,
                attention_mask=attention_masks_batch
            )
            predictions = torch.argmax(logits, dim=-1)

            # Token-level accuracy (only count actions, ignore PAD and EOS)
            valid_mask = (targets_batch != PAD_TOKEN_ID) & (targets_batch != EOS_TOKEN_ID)
            total_tokens += valid_mask.sum().item()
            pred_actions = predictions[valid_mask]
            target_actions = targets_batch[valid_mask]
            correct_tokens += (pred_actions == target_actions).sum().item()

            # Sequence-level exact match
            for i in range(predictions.shape[0]):
                target_seq = targets_batch[i]
                valid_target = target_seq[(target_seq != PAD_TOKEN_ID) & (target_seq != EOS_TOKEN_ID)]
                
                pred_seq = predictions[i]
                eos_idx = (pred_seq == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
                if len(eos_idx) > 0:
                    pred_actions_only = pred_seq[:eos_idx[0]]
                else:
                    pred_actions_only = pred_seq[pred_seq != PAD_TOKEN_ID]
                
                if len(valid_target) > 0 and len(pred_actions_only) == len(valid_target):
                    if torch.equal(pred_actions_only.cpu(), valid_target.cpu()):
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
