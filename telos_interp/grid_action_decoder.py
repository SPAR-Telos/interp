"""Decoder-only transformer for predicting action sequences from grid states.

This module implements a small transformer that takes grid text as input
and predicts optimal action sequences (4 actions: LEFT, RIGHT, UP, DOWN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridActionDecoder(nn.Module):
    """Decoder-only transformer that predicts action sequences from grid text.

    Architecture:
    - Grid Encoder: Tokenizes and encodes grid text to embeddings
    - Action Decoder: 4-layer transformer decoder that generates action sequences

    Args:
        grid_vocab_size: Vocabulary size for grid symbols (default: 30)
        action_vocab_size: Vocabulary size for actions (default: 4)
        n_layer: Number of decoder layers (default: 4)
        n_head: Number of attention heads (default: 4)
        n_embd: Hidden dimension (default: 256)
        max_grid_len: Maximum grid text length (default: 500)
        max_action_len: Maximum action sequence length (default: 50)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        grid_vocab_size: int = 30,
        action_vocab_size: int = 4,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 256,
        max_grid_len: int = 500,
        max_action_len: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.grid_vocab_size = grid_vocab_size
        self.action_vocab_size = action_vocab_size
        self.n_embd = n_embd
        self.max_grid_len = max_grid_len
        self.max_action_len = max_action_len

        # Grid encoder: tokenize and encode grid text
        self.grid_embeddings = nn.Embedding(grid_vocab_size, n_embd)
        self.grid_pos_embeddings = nn.Embedding(max_grid_len, n_embd)
        
        # Grid encoder (transformer encoder to process grid)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.grid_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Pool grid sequence to fixed-size representation (mean pooling)
        self.grid_pool = nn.AdaptiveAvgPool1d(1)

        # Action decoder
        self.action_embeddings = nn.Embedding(action_vocab_size, n_embd)
        # Position embeddings: +1 to account for positions starting at 1 (grid is at position 0)
        self.action_pos_embeddings = nn.Embedding(max_action_len + 1, n_embd)
        
        # Transformer decoder (attends to grid memory, generates actions)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder_layers = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)

        # Output projection to action vocabulary
        self.lm_head = nn.Linear(n_embd, action_vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.grid_embeddings.weight, std=0.02)
        nn.init.normal_(self.action_embeddings.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def encode_grid(self, grid_token_ids: torch.Tensor) -> torch.Tensor:
        """Encode grid text to a fixed-size representation.
        
        Args:
            grid_token_ids: Token IDs of shape (batch_size, grid_seq_len)
            
        Returns:
            Grid memory tensor of shape (batch_size, 1, n_embd)
        """
        batch_size, grid_len = grid_token_ids.shape
        device = grid_token_ids.device

        # Embed grid tokens
        grid_embeds = self.grid_embeddings(grid_token_ids)  # (batch_size, grid_len, n_embd)
        
        # Add position embeddings
        grid_positions = torch.arange(grid_len, device=device).unsqueeze(0).expand(batch_size, -1)
        grid_pos_embeds = self.grid_pos_embeddings(grid_positions)
        grid_embeds = grid_embeds + grid_pos_embeds

        # Encode grid
        grid_encoded = self.grid_encoder(grid_embeds)  # (batch_size, grid_len, n_embd)
        
        # Pool to fixed-size representation (mean pooling)
        # Transpose for pooling: (batch_size, n_embd, grid_len)
        grid_encoded_t = grid_encoded.transpose(1, 2)
        grid_memory = self.grid_pool(grid_encoded_t)  # (batch_size, n_embd, 1)
        grid_memory = grid_memory.transpose(1, 2)  # (batch_size, 1, n_embd)
        
        return grid_memory

    def forward(
        self,
        grid_token_ids: torch.Tensor,
        action_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the decoder.

        Args:
            grid_token_ids: Grid token IDs of shape (batch_size, grid_seq_len)
            action_ids: Action token IDs of shape (batch_size, action_seq_len) for teacher forcing.
                If None and labels is provided, uses labels shifted by one position.
            labels: Target action token IDs of shape (batch_size, action_seq_len).
                If None, only returns logits.

        Returns:
            If labels is not None, returns (loss, logits) tuple.
            Otherwise, returns logits tensor of shape (batch_size, action_seq_len, action_vocab_size).
        """
        batch_size = grid_token_ids.shape[0]
        device = grid_token_ids.device

        # Encode grid to memory
        grid_memory = self.encode_grid(grid_token_ids)  # (batch_size, 1, n_embd)

        # Prepare action sequence for teacher forcing
        if action_ids is None and labels is not None:
            # Shift labels for teacher forcing: [0, action_0, action_1, ..., action_{n-1}]
            seq_len = labels.shape[1]
            if seq_len > 0:
                # Shift labels and replace padding (-100) with 0 for embedding
                shifted_labels = labels[:, :-1].clone()
                shifted_labels[shifted_labels == -100] = 0  # Replace padding with 0 for embedding
                action_ids = torch.cat(
                    [torch.zeros(batch_size, 1, dtype=torch.long, device=device), shifted_labels], dim=1
                )
            else:
                action_ids = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        elif action_ids is None:
            action_ids = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

        # Build action sequence embeddings
        if action_ids.shape[1] > 0:
            # Ensure all action_ids are in valid range [0, action_vocab_size-1]
            action_ids = torch.clamp(action_ids, 0, self.action_vocab_size - 1)
            action_embeds = self.action_embeddings(action_ids)  # (batch_size, action_seq_len, n_embd)
            # Add position embeddings (starting from position 1, since grid is at position 0)
            positions = torch.arange(1, action_ids.shape[1] + 1, device=device).unsqueeze(0).expand(batch_size, -1)
            # Clamp positions to valid range [0, max_action_len] for position embeddings
            positions = torch.clamp(positions, 0, self.max_action_len)
            pos_embeds = self.action_pos_embeddings(positions)
            target_embeds = action_embeds + pos_embeds
        else:
            # No actions yet, just use grid memory
            target_embeds = grid_memory  # (batch_size, 1, n_embd)

        # Decoder: attends to grid memory, generates actions
        decoder_output = self.decoder_layers(
            tgt=target_embeds, memory=grid_memory
        )  # (batch_size, action_seq_len, n_embd)

        # Project to action vocabulary
        logits = self.lm_head(decoder_output)  # (batch_size, action_seq_len, action_vocab_size)

        if labels is not None:
            # Compute cross-entropy loss
            logits_flat = logits.view(-1, self.action_vocab_size)
            labels_flat = labels.view(-1)
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
            return loss, logits
        
        return logits

    def generate(
        self,
        grid_token_ids: torch.Tensor,
        max_length: int | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate action sequence autoregressively from grid.

        Args:
            grid_token_ids: Grid token IDs of shape (batch_size, grid_seq_len)
            max_length: Maximum length of generated sequence (default: self.max_action_len)
            temperature: Sampling temperature (default: 1.0)

        Returns:
            Generated action token IDs of shape (batch_size, generated_length)
        """
        if max_length is None:
            max_length = self.max_action_len

        batch_size = grid_token_ids.shape[0]
        device = grid_token_ids.device
        self.eval()

        # Encode grid to memory
        grid_memory = self.encode_grid(grid_token_ids)  # (batch_size, 1, n_embd)

        generated = []
        with torch.no_grad():
            for step in range(max_length):
                if step == 0:
                    # First step: use grid memory as target
                    target_embeds = grid_memory  # (batch_size, 1, n_embd)
                else:
                    # Subsequent steps: embed generated actions so far
                    past_action_ids = torch.stack(generated, dim=1)  # (batch_size, step)
                    past_action_embeds = self.action_embeddings(past_action_ids)  # (batch_size, step, n_embd)
                    positions = torch.arange(1, step + 1, device=device).unsqueeze(0).expand(batch_size, -1)
                    # Clamp positions to valid range [0, max_action_len] for position embeddings
                    positions = torch.clamp(positions, 0, self.max_action_len)
                    pos_embeds = self.action_pos_embeddings(positions)
                    past_action_embeds = past_action_embeds + pos_embeds

                    # Concatenate with grid memory
                    target_embeds = torch.cat([grid_memory, past_action_embeds], dim=1)

                # Run through decoder
                decoder_output = self.decoder_layers(tgt=target_embeds, memory=grid_memory)
                logits = self.lm_head(decoder_output)  # (batch_size, seq_len, action_vocab_size)

                # Get logits for the last position (next token prediction)
                next_token_logits = logits[:, -1, :]  # (batch_size, action_vocab_size)

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
                generated.append(next_token.squeeze(1))  # (batch_size,)

        return torch.stack(generated, dim=1)  # (batch_size, max_length)

