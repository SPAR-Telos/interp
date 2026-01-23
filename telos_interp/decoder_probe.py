"""One-Shot Prefix Probe for predicting action sequences from LLM activations.

This module implements a transformer decoder that takes N reasoning trace activations
and uses learned query slots to predict navigation paths in one shot.

For long-tail distributions (e.g., median=6, max=20), the model uses position-weighted
loss, priority initialization, and position embeddings to focus on early positions.
"""

import os

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

from telos_interp.probing import load_activations

# Special token IDs
# Actions: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN, 4=GOAL/EOS
PAD_TOKEN_ID = -100  # Padding token (using -100 for ignore_index)

# Vocabulary size: 5 actions (0-3=directions, 4=GOAL/EOS)
DEFAULT_VOCAB_SIZE = 5


class OneShotPrefixProbe(nn.Module):
    """One-Shot Prefix Probe for predicting action sequences from LLM activations.
    
    This model takes a sequence of N activations (each activation_dim dimensions) from a reasoning trace.
    It uses a Transformer Decoder with max_path_len learned query slots to predict actions.
    Uses a bottleneck dimension (hidden_dim) to compress the activations.
    
    IMPORTANT: max_path_len should be >= the longest action sequence in your data!
    However, you should filter outliers to avoid excessive query tokens.
    
    Trade-offs:
    - max_path_len=36 means 36 learned query parameters (36 * hidden_dim = 36K params)
    - Model always predicts 36 steps, even for 2-step sequences (uses padding)
    - More query tokens = more computation and potentially harder to train
    
    Handling Long-Tail Distributions (e.g., median=6 but max=20):
    ---------------------------------------------------------------
    When most sequences are short but some are long, the model prioritizes early positions:
    
    1. Position-Weighted Loss (position_loss_decay=0.95):
       - Early positions get higher loss weight (exponential decay)
       - Position 0: weight = 1.0
       - Position 10: weight = 0.60  (0.95^10)
       - Position 15: weight = 0.46  (0.95^15)
       - This makes the model focus on getting early steps right
    
    2. Priority Initialization:
       - First 15 queries: initialized with std=0.15 (strong signal)
       - Remaining queries: initialized with std=0.05 (weak signal)
       - Gives early positions better starting point for learning
    
    3. Position Embeddings:
       - Each query slot has learnable position-specific embedding
       - Allows model to learn that early slots are "primary", later slots "secondary"
    
    This approach lets you use max_path_len=20 to handle the tail, while the model
    naturally focuses most of its capacity on the first 10-15 positions where most
    sequences actually live.
    
    Recommendations:
    - Filter sequences to a reasonable max (e.g., 20-36 steps)
    - Use scripts/filter_sequences_by_length.py to remove outliers
    - For long-tail data, set max_path_len to ~2x your median length
    
    Example usage:
        # Filter your data first to remove outliers
        # python scripts/filter_sequences_by_length.py --max-length 20
        
        # Then load and train
        train_data = torch.load('data/processed/action_sequences_train_filtered.pt')
        max_len = train_data['stats']['max_length']  # e.g., 20
        model = OneShotPrefixProbe(max_path_len=max_len, position_loss_decay=0.95)
    
    Args:
        activation_dim: Dimension of the input LLM activation vector (default: 2880)
        hidden_dim: Compressed bottleneck dimension (default: 1024)
        num_actions: Number of action types (default: 5 for 0-3=directions, 4=GOAL/EOS)
        num_memory_tokens: Number of reasoning trace activations (default: 3)
        max_path_len: Number of learned query slots for trajectory. MUST be >= longest sequence!
        n_layers: Number of transformer decoder layers (default: 4)
        n_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
        position_loss_decay: Exponential decay for position weighting (default: 0.95)
    """
    
    def __init__(
        self,
        activation_dim: int = 2880,
        hidden_dim: int = 1280,
        num_actions: int = DEFAULT_VOCAB_SIZE,
        num_memory_tokens: int = 3,
        max_path_len: int = 20,  # Changed from 8 to accommodate long-tail sequences
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        position_loss_decay: float = 0.95,  # Decay factor for position-based loss weighting (0.95^10 ≈ 0.60)
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_path_len = max_path_len
        self.num_actions = num_actions
        self.num_memory_tokens = num_memory_tokens
        self.activation_dim = activation_dim
        self.position_loss_decay = position_loss_decay
        
        # 1. THE FEATURE MAPPER (The Bottleneck)
        # Processes each of the 3 tokens independently into the 1024 space
        self.feature_map = nn.Sequential(
            nn.Linear(activation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. THE LEARNED POSITION QUERIES
        # These are the "questions" the probe asks the activations
        # Shape: [max_path_len, 1024]
        self.path_queries = nn.Parameter(torch.randn(max_path_len, hidden_dim))
        
        # Position embeddings for queries - helps model learn position-specific behavior
        # Early positions (0-10) should learn to be "primary", later ones "secondary"
        self.query_position_embeddings = nn.Parameter(torch.randn(max_path_len, hidden_dim))
        
        # 3. THE TRANSFORMER DECODER (The Processor)
        # Queries (8) attend to Memory (3)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # 4. THE ACTION READOUT
        self.action_head = nn.Linear(hidden_dim, num_actions)
        
        # Initialize queries with position-based priority
        # Early positions get stronger initialization (higher std)
        self._init_query_priorities(priority_cutoff=min(15, max_path_len))
    
    def _init_query_priorities(self, priority_cutoff: int = 15):
        """Initialize queries with higher magnitude for early positions.
        
        Args:
            priority_cutoff: First N positions get priority initialization
        """
        with torch.no_grad():
            # Early positions (0 to priority_cutoff) get std=0.15 (stronger signal)
            # Later positions get std=0.05 (weaker signal)
            for i in range(self.max_path_len):
                if i < priority_cutoff:
                    # Priority positions: higher variance
                    nn.init.normal_(self.path_queries[i], mean=0.0, std=0.15)
                    nn.init.normal_(self.query_position_embeddings[i], mean=0.0, std=0.15)
                else:
                    # Secondary positions: lower variance
                    nn.init.normal_(self.path_queries[i], mean=0.0, std=0.05)
                    nn.init.normal_(self.query_position_embeddings[i], mean=0.0, std=0.05)
    
    def forward(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the One-Shot Prefix Probe.
        
        Args:
            activations: LLM activations of shape (batch_size, num_memory_tokens, activation_dim)
                         e.g., (batch_size, 3, 2880) for 3 reasoning trace activations
            labels: Target sequence of shape (batch_size, max_path_len) for computing loss.
                    Should be action IDs [0-3] or PAD_TOKEN_ID (-100) for padding.
                    If None, only returns logits.
        
        Returns:
            If labels is not None, returns (loss, logits) tuple.
            Otherwise, returns logits tensor of shape (batch_size, max_path_len, num_actions).
        """
        batch_size = activations.size(0)
        
        # Convert activation to float32 if needed (HF activations may be bfloat16)
        if activations.dtype != torch.float32:
            activations = activations.to(torch.float32)
        
        # 1. Map 3 tokens to hidden space: [Batch, 3, 1024]
        memory = self.feature_map(activations)
        
        # 2. Expand queries for the batch and add position embeddings: [Batch, max_path_len, 1024]
        queries = self.path_queries.unsqueeze(0).expand(batch_size, -1, -1)
        query_pos_emb = self.query_position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + query_pos_emb  # Add position information to queries
        
        # 3. Cross-Attention: 8 queries look at 3 memory tokens
        # Note: No causal mask is needed here because we want all queries
        # to see the entire reasoning trace conclusion.
        path_latent = self.decoder(tgt=queries, memory=memory)
        
        # 4. Final prediction per slot: [Batch, 8, num_actions]
        logits = self.action_head(path_latent)
        
        # 5. Compute loss if labels are provided
        if labels is not None:
            batch_size, seq_len = labels.shape
            
            # Compute per-position loss (without reduction)
            logits_flat = logits.view(-1, logits.size(-1))  # (batch * seq_len, num_actions)
            labels_flat = labels.view(-1)  # (batch * seq_len)
            
            # Compute loss per token
            per_token_loss = F.cross_entropy(
                logits_flat, 
                labels_flat, 
                ignore_index=PAD_TOKEN_ID,
                reduction='none'
            )  # (batch * seq_len)
            
            # Reshape to (batch, seq_len) for position-based weighting
            per_token_loss = per_token_loss.view(batch_size, seq_len)
            
            # Create position weights: early positions get higher weight
            # weight[i] = decay^i, so position 0 gets weight 1.0, position 10 gets weight decay^10
            position_weights = torch.pow(
                self.position_loss_decay, 
                torch.arange(seq_len, device=logits.device, dtype=torch.float32)
            ).unsqueeze(0)  # (1, seq_len)
            
            # Apply position weights (don't weight PAD positions)
            valid_mask = (labels != PAD_TOKEN_ID).float()  # (batch, seq_len)
            weighted_loss = per_token_loss * position_weights * valid_mask
            
            # Compute mean loss (normalize by total weight of valid positions)
            total_weight = (position_weights * valid_mask).sum()
            if total_weight > 0:
                loss = weighted_loss.sum() / total_weight
            else:
                loss = weighted_loss.sum()  # Fallback if no valid tokens
            
            return loss, logits
        
        return logits

    def generate(
        self,
        activations: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate action sequence from activations (one-shot prediction).
        
        Args:
            activations: LLM activations of shape (batch_size, num_memory_tokens, activation_dim)
                         e.g., (batch_size, 3, 2880) for 3 reasoning trace activations
            temperature: Sampling temperature (default: 1.0)
        
        Returns:
            Generated action token IDs of shape (batch_size, max_path_len)
        """
        self.eval()
        
        with torch.no_grad():
            # Get logits for all 8 positions at once
            logits = self.forward(activations)  # (batch_size, max_path_len, num_actions)
            
            # Apply temperature
            logits = logits / temperature
            
            # Sample from the distribution at each position
            probs = torch.softmax(logits, dim=-1)  # (batch_size, max_path_len, num_actions)
            
            # Sample action for each position
            # Reshape to (batch_size * max_path_len, num_actions) for multinomial
            batch_size, path_len, num_actions = probs.shape
            probs_flat = probs.view(-1, num_actions)
            samples_flat = torch.multinomial(probs_flat, num_samples=1)  # (batch_size * max_path_len, 1)
            
            # Reshape back to (batch_size, max_path_len)
            samples = samples_flat.view(batch_size, path_len)
            
            return samples


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
    pad_to_length: bool = True,
) -> tuple[dict[int, torch.Tensor], int]:
    """Load action sequences from CSV file.

    Supports two CSV formats:
    1. Format with 'action_sequence' column containing JSON array: [0, 1, 2, 3, ...]
    2. Format with 'last_action' column per trajectory step (legacy)

    Args:
        csv_path: Path to CSV file with trajectory data
        grid_indices: List of grid indices to load. If None, loads all.
        max_seq_len: Maximum/target sequence length. If None, uses the longest sequence found.
        pad_to_length: If True, pad sequences to max_seq_len with PAD_TOKEN_ID

    Returns:
        Tuple of:
        - Dictionary mapping grid index to action tensor of shape (max_seq_len,)
          Padded with PAD_TOKEN_ID (-100) if pad_to_length=True
        - The actual max_seq_len used (useful when max_seq_len=None to discover it)
    """
    import json
    import pandas as pd

    df = pd.read_csv(csv_path)

    # First pass: collect all raw action sequences to find max length
    raw_sequences: dict[int, list[int]] = {}
    
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
                    # Ensure all actions are integers in range [0, 4] (0-3=directions, 4=GOAL/EOS)
                    actions = [int(a) for a in actions if isinstance(a, (int, float)) and 0 <= int(a) <= 4]
                    
                    if len(actions) > 0:
                        raw_sequences[env_idx] = actions
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
                        if 0 <= action_int <= 4:  # Validate range (0-3=directions, 4=GOAL/EOS)
                            actions.append(action_int)
                    except (ValueError, TypeError):
                        continue

            if len(actions) > 0:
                raw_sequences[env_idx] = actions
    else:
        raise ValueError("CSV must have either 'action_sequence' or 'last_action' column")

    # Determine the actual max sequence length
    if len(raw_sequences) == 0:
        actual_max_len = max_seq_len if max_seq_len is not None else 8
        return {}, actual_max_len
    
    discovered_max_len = max(len(seq) for seq in raw_sequences.values())
    
    if max_seq_len is None:
        # Use discovered max length
        actual_max_len = discovered_max_len
        print(f"Auto-detected max sequence length: {actual_max_len}")
    else:
        actual_max_len = max_seq_len
        if discovered_max_len > max_seq_len:
            print(f"⚠️  Warning: Found sequences of length {discovered_max_len}, but max_seq_len={max_seq_len}. "
                  f"Sequences will be TRUNCATED. Consider increasing max_seq_len or max_path_len in the model.")
    
    # Print sequence length distribution
    lengths = [len(seq) for seq in raw_sequences.values()]
    print(f"Sequence length stats: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.1f}")

    # Second pass: truncate/pad sequences to actual_max_len
    action_sequences = {}
    for env_idx, actions in raw_sequences.items():
        # Truncate if longer than actual_max_len
        if len(actions) > actual_max_len:
            actions = actions[:actual_max_len]
        
        action_seq = torch.tensor(actions, dtype=torch.long)
        
        # Pad to actual_max_len if needed
        if pad_to_length and len(action_seq) < actual_max_len:
            padding = torch.full((actual_max_len - len(action_seq),), PAD_TOKEN_ID, dtype=torch.long)
            action_seq = torch.cat([action_seq, padding])
        
        action_sequences[env_idx] = action_seq

    return action_sequences, actual_max_len


def create_dataset(
    activations_seq: dict[int, list[torch.Tensor]],
    action_sequences: dict[int, torch.Tensor],
    num_memory_tokens: int = 3,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create dataset tuples of (activations, actions).
    
    Args:
        activations_seq: Dictionary mapping grid index to list of activation tensors
                         Each list should contain num_memory_tokens tensors of shape (activation_dim,)
        action_sequences: Dictionary mapping grid index to action sequence tensor of shape (max_path_len,)
        num_memory_tokens: Number of memory tokens expected (default: 3)
    
    Returns:
        List of (activations, actions) tuples where:
        - activations has shape (num_memory_tokens, activation_dim)
        - actions has shape (max_path_len,)
    """
    dataset = []
    
    # Find common grid indices
    common_indices = set(activations_seq.keys()) & set(action_sequences.keys())
    
    for grid_idx in common_indices:
        activations_list = activations_seq[grid_idx]
        actions = action_sequences[grid_idx]
        
        # Validate we have the right number of memory tokens
        if len(activations_list) != num_memory_tokens:
            print(f"Warning: env_idx {grid_idx} has {len(activations_list)} activations, expected {num_memory_tokens}. Skipping.")
            continue
        
        # Stack activations into (num_memory_tokens, activation_dim)
        activations_stacked = torch.stack(activations_list, dim=0)
        
        dataset.append((activations_stacked, actions))
    
    return dataset


def train_decoder_probe(
    model: OneShotPrefixProbe,
    dataset: list[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    eval_split: float = 0.2,
    val_dataset: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    save_path: str | None = None,
) -> dict:
    """Train the One-Shot Prefix Probe model.

    Args:
        model: OneShotPrefixProbe model to train
        dataset: List of (activations, actions) tuples where:
                 - activations: shape (num_memory_tokens, activation_dim)
                 - actions: shape (max_path_len,)
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        eval_split: Fraction of data to use for evaluation
        val_dataset: Optional separate validation dataset
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

    def collate_fn(batch):
        activations = []
        actions = []

        for activation, action in batch:
            activations.append(activation)
            actions.append(action)

        activations_tensor = torch.stack(activations)
        actions_tensor = torch.stack(actions)
        
        return activations_tensor, actions_tensor

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"  Optimizer: LR = {learning_rate:.2e}")

    # Training loop
    train_losses = []
    eval_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        for batch_idx, (activations_batch, actions_batch) in enumerate(train_loader):
            activations_batch = activations_batch.to(device).to(torch.float32)
            actions_batch = actions_batch.to(device)

            # Debug: Print first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                print("\n🔍 Debug: First training batch")
                print(f"  Batch size: {activations_batch.shape[0]}")
                print(f"  Activations shape: {activations_batch.shape}")
                print(f"  Actions shape: {actions_batch.shape}")
                print(f"  Sample actions[0]: {actions_batch[0].cpu().tolist()}")
                # Check token distribution in actions
                valid_actions = actions_batch[actions_batch != PAD_TOKEN_ID]
                unique_actions, counts = torch.unique(valid_actions, return_counts=True)
                print(f"  Action token distribution: {dict(zip(unique_actions.cpu().tolist(), counts.cpu().tolist()))}")

            # Forward pass
            loss, logits = model(activations_batch, labels=actions_batch)

            # Debug: Check loss and logits
            if epoch == 0 and batch_idx == 0:
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")
                # Check predictions
                preds = torch.argmax(logits, dim=-1)
                print(f"  Sample predictions[0]: {preds[0].cpu().tolist()}")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
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
        
        # Prefix accuracy: track how many get first N steps correct
        max_prefix_len = 15  # Track up to 15 steps
        prefix_correct = [0] * max_prefix_len  # Count of sequences with first N steps correct
        prefix_total = [0] * max_prefix_len    # Count of sequences with at least N steps

        if len(eval_data) == 0:
            # No eval data (eval_split was 0 or all data used for training)
            avg_eval_loss = 0.0
            token_accuracy = 0.0
            sequence_accuracy = 0.0
            prefix_accuracies = [0.0] * max_prefix_len
        else:
            with torch.no_grad():
                for batch_idx, (activations_batch, actions_batch) in enumerate(eval_loader):
                    activations_batch = activations_batch.to(device).to(torch.float32)
                    actions_batch = actions_batch.to(device)

                    loss, logits = model(activations_batch, labels=actions_batch)

                    # Check for NaN/Inf before adding to eval loss
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        epoch_eval_loss += loss.item()
                        num_eval_batches += 1

                    # Compute accuracy metrics
                    predictions = torch.argmax(logits, dim=-1)  # (batch_size, max_path_len)

                    # Token-level accuracy (only count non-padding tokens)
                    valid_mask = (actions_batch != PAD_TOKEN_ID)
                    total_tokens += valid_mask.sum().item()
                    pred_actions = predictions[valid_mask]
                    target_actions = actions_batch[valid_mask]
                    correct_tokens += (pred_actions == target_actions).sum().item()

                    # Sequence-level exact match and prefix accuracy
                    for i in range(predictions.shape[0]):
                        # Get valid actions from target (remove PAD)
                        target_seq = actions_batch[i]
                        valid_target = target_seq[target_seq != PAD_TOKEN_ID]
                        
                        # Get predictions (remove PAD)
                        pred_seq = predictions[i]
                        valid_pred = pred_seq[:len(valid_target)]  # Take same length as target
                        
                        seq_len = len(valid_target)
                        if seq_len > 0 and len(valid_pred) == seq_len:
                            # Full sequence match
                            if torch.equal(valid_pred.cpu(), valid_target.cpu()):
                                exact_matches += 1
                            
                            # Prefix accuracy: check first N steps
                            for n in range(1, min(seq_len + 1, max_prefix_len + 1)):
                                prefix_total[n - 1] += 1
                                if torch.equal(valid_pred[:n].cpu(), valid_target[:n].cpu()):
                                    prefix_correct[n - 1] += 1
                        
                        total_sequences += 1

                    # Show sample predictions for first batch of last epoch
                    if epoch == num_epochs - 1 and batch_idx == 0:
                        print("\nSample predictions (first batch):")
                        for i in range(min(3, predictions.shape[0])):
                            # Get true actions (remove PAD)
                            target_seq = actions_batch[i].cpu()
                            true_actions = target_seq[target_seq != PAD_TOKEN_ID].tolist()
                            
                            # Get predicted actions (same length as true)
                            pred_seq = predictions[i].cpu()
                            pred_actions = pred_seq[:len(true_actions)].tolist()
                            
                            print(f"  Sample {i + 1}:")
                            print(f"    True:  {true_actions}")
                            print(f"    Pred:  {pred_actions}")
                            print(f"    Match: {true_actions == pred_actions}")

            avg_eval_loss = epoch_eval_loss / num_eval_batches if num_eval_batches > 0 else 0.0
            token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
            sequence_accuracy = exact_matches / total_sequences if total_sequences > 0 else 0.0
            
            # Compute prefix accuracies
            prefix_accuracies = []
            for n in range(max_prefix_len):
                if prefix_total[n] > 0:
                    prefix_accuracies.append(prefix_correct[n] / prefix_total[n])
                else:
                    prefix_accuracies.append(0.0)

        eval_losses.append(avg_eval_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Eval Loss: {avg_eval_loss:.4f}, "
            f"Token Acc: {token_accuracy:.4f}, "
            f"Seq Acc: {sequence_accuracy:.4f}"
        )
        
        # Print prefix accuracies every few epochs or on last epoch
        if epoch == num_epochs - 1 or (epoch + 1) % 5 == 0:
            print(f"  Prefix Accuracies (first N steps correct):")
            # Show in groups of 5 for readability
            for start in range(0, min(15, max_prefix_len), 5):
                end = min(start + 5, max_prefix_len)
                acc_str = ", ".join([
                    f"N={n+1}: {prefix_accuracies[n]:.1%}" 
                    for n in range(start, end) 
                    if prefix_total[n] > 0
                ])
                if acc_str:
                    print(f"    {acc_str}")
            
            # Show distribution of sequence lengths
            if epoch == num_epochs - 1 and any(prefix_total):
                print(f"  Sequence length distribution:")
                # Count how many sequences of each length
                length_counts = {}
                for i in range(total_sequences):
                    # This is approximate - just show the prefix_total values
                    pass
                # Just show how many samples have at least N steps
                sample_str = ", ".join([
                    f"≥{n+1}: {prefix_total[n]}" 
                    for n in [0, 2, 4, 6, 8, 10] 
                    if n < max_prefix_len and prefix_total[n] > 0
                ])
                if sample_str:
                    print(f"    Samples with at least N steps: {sample_str}")

    # Save model if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    # Compute final accuracy metrics including prefix accuracies
    model.eval()
    final_token_accuracy = 0.0
    final_sequence_accuracy = 0.0
    total_tokens = 0
    correct_tokens = 0
    exact_matches = 0
    total_sequences = 0
    
    # Final prefix accuracy computation
    max_prefix_len = 15
    final_prefix_correct = [0] * max_prefix_len
    final_prefix_total = [0] * max_prefix_len

    with torch.no_grad():
        for activations_batch, actions_batch in eval_loader:
            activations_batch = activations_batch.to(device).to(torch.float32)
            actions_batch = actions_batch.to(device)

            _, logits = model(activations_batch, labels=actions_batch)
            predictions = torch.argmax(logits, dim=-1)

            # Token-level accuracy (only count non-padding tokens)
            valid_mask = (actions_batch != PAD_TOKEN_ID)
            total_tokens += valid_mask.sum().item()
            pred_actions = predictions[valid_mask]
            target_actions = actions_batch[valid_mask]
            correct_tokens += (pred_actions == target_actions).sum().item()

            # Sequence-level exact match and prefix accuracy
            for i in range(predictions.shape[0]):
                target_seq = actions_batch[i]
                valid_target = target_seq[target_seq != PAD_TOKEN_ID]
                
                pred_seq = predictions[i]
                valid_pred = pred_seq[:len(valid_target)]
                
                seq_len = len(valid_target)
                if seq_len > 0 and len(valid_pred) == seq_len:
                    if torch.equal(valid_pred.cpu(), valid_target.cpu()):
                        exact_matches += 1
                    
                    # Prefix accuracy
                    for n in range(1, min(seq_len + 1, max_prefix_len + 1)):
                        final_prefix_total[n - 1] += 1
                        if torch.equal(valid_pred[:n].cpu(), valid_target[:n].cpu()):
                            final_prefix_correct[n - 1] += 1
                
                total_sequences += 1

    final_token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    final_sequence_accuracy = exact_matches / total_sequences if total_sequences > 0 else 0.0
    
    # Compute final prefix accuracies
    final_prefix_accuracies = []
    for n in range(max_prefix_len):
        if final_prefix_total[n] > 0:
            final_prefix_accuracies.append(final_prefix_correct[n] / final_prefix_total[n])
        else:
            final_prefix_accuracies.append(0.0)

    # Print final prefix accuracies
    print(f"\n{'='*80}")
    print("FINAL PREFIX ACCURACIES (First N steps correct)")
    print(f"{'='*80}")
    for start in range(0, min(15, max_prefix_len), 5):
        end = min(start + 5, max_prefix_len)
        for n in range(start, end):
            if final_prefix_total[n] > 0:
                print(f"  First {n+1:2d} steps: {final_prefix_accuracies[n]:6.1%}  ({final_prefix_correct[n]:3d}/{final_prefix_total[n]:3d} sequences)")
    print(f"{'='*80}\n")
    
    return {
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
        "final_token_accuracy": final_token_accuracy,
        "final_sequence_accuracy": final_sequence_accuracy,
        "prefix_accuracies": final_prefix_accuracies,
        "prefix_counts": final_prefix_total,
    }
