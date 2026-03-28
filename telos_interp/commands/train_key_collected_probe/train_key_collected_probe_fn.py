"""Train a binary classification probe to detect key collection from activations."""

from pathlib import Path
from typing import Literal

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from telos_interp.probe_models import ModelType
from telos_interp.probe_models import create_classification_model as _create_model
from telos_interp.training import (
    compute_normalization_params,
    normalize_activations,
    resolve_device,
    set_seed,
    train_epoch as _train_epoch,
)

ClassWeight = Literal["balanced"] | None

# Human-readable names for the two binary classes
_LABEL_NAMES = {0: "not_collected", 1: "collected"}


class KeyCollectedProbe:
    """A trained binary probe for detecting key collection, with normalization.

    Example:
        # Training
        probe = train_key_collected_probe("train.pt", normalize=True)
        probe.save("probe.pt")

        # Inference
        probe = KeyCollectedProbe.load("probe.pt")
        probs = probe.predict_proba(activations)  # (N, 2): [p_not_collected, p_collected]
        labels = probe.predict(activations)        # (N,): 0 or 1
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: ModelType,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float | None = None,
        scaler_mean: torch.Tensor | None = None,
        scaler_std: torch.Tensor | None = None,
        config: dict | None = None,
        results: dict | None = None,
        device: torch.device | str | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.model = model.to(self.device)
        self.model.eval()

        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.config = config or {}
        self.results = results or {}

        if scaler_mean is not None:
            self.scaler_mean = scaler_mean.to(self.device)
            self.scaler_std = scaler_std.to(self.device)
        else:
            self.scaler_mean = None
            self.scaler_std = None

    @property
    def normalized(self) -> bool:
        """Whether this probe applies input normalization."""
        return self.scaler_mean is not None

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.scaler_mean is not None:
            return (x - self.scaler_mean) / self.scaler_std
        return x

    @torch.no_grad()
    def predict_proba(self, activations: torch.Tensor) -> torch.Tensor:
        """Get class probabilities for activations.

        Args:
            activations: Input tensor of shape (N, input_dim)

        Returns:
            Tensor of shape (N, 2): columns are [p_not_collected, p_collected]
        """
        self.model.eval()
        x = activations.float().to(self.device)
        x = self._normalize(x)
        logits = self.model(x)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        """Get binary predictions for activations.

        Args:
            activations: Input tensor of shape (N, input_dim)

        Returns:
            Tensor of shape (N,) with values in {0, 1}
        """
        probs = self.predict_proba(activations)
        return torch.argmax(probs, dim=-1)

    def save(self, path: str | Path) -> None:
        """Save the probe to a file."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_type": self.model_type,
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "scaler_mean": self.scaler_mean.cpu() if self.scaler_mean is not None else None,
                "scaler_std": self.scaler_std.cpu() if self.scaler_std is not None else None,
                "config": self.config,
                "results": self.results,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "KeyCollectedProbe":
        """Load a probe from a file."""
        device_str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        torch_device = torch.device(device_str)

        data = torch.load(path, map_location=torch_device, weights_only=False)

        model = _create_model(
            model_type=data["model_type"],
            input_dim=data["input_dim"],
            num_classes=2,
            hidden_dims=data["hidden_dims"] or [],
            dropout=data["dropout"] or 0.0,
        )
        model.load_state_dict(data["model_state_dict"])

        return cls(
            model=model,
            model_type=data["model_type"],
            input_dim=data["input_dim"],
            hidden_dims=data["hidden_dims"],
            dropout=data["dropout"],
            scaler_mean=data.get("scaler_mean"),
            scaler_std=data.get("scaler_std"),
            config=data.get("config"),
            results=data.get("results"),
            device=torch_device,
        )


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate the binary probe and return metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    class_tp = torch.zeros(2)
    class_gt_support = torch.zeros(2)
    class_pred_count = torch.zeros(2)

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            num_batches += 1

            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            for i in range(2):
                gt_mask = batch_y == i
                pred_mask = predicted == i
                class_gt_support[i] += gt_mask.sum().item()
                class_pred_count[i] += pred_mask.sum().item()
                class_tp[i] += (gt_mask & pred_mask).sum().item()

    per_class_metrics = {}
    for i in range(2):
        tp = class_tp[i].item()
        gt_support = class_gt_support[i].item()
        pred_count = class_pred_count[i].item()

        precision = tp / pred_count if pred_count > 0 else 0.0
        recall = tp / gt_support if gt_support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / gt_support if gt_support > 0 else 0.0

        per_class_metrics[i] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "gt_support": int(gt_support),
            "predicted": int(pred_count),
        }

    per_class_accuracies = [m["accuracy"] for m in per_class_metrics.values() if m["gt_support"] > 0]
    balanced_accuracy = sum(per_class_accuracies) / len(per_class_accuracies) if per_class_accuracies else 0.0

    return {
        "loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "accuracy": correct / total if total > 0 else 0.0,
        "balanced_accuracy": balanced_accuracy,
        "per_class_metrics": per_class_metrics,
        "num_samples": total,
    }


def _compute_class_weights(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute balanced class weights for CrossEntropyLoss."""
    class_weights = torch.zeros(2, dtype=torch.float32)
    unique_classes, class_counts = torch.unique(labels, return_counts=True)
    n_samples = len(labels)
    for class_id, count in zip(unique_classes, class_counts, strict=False):
        class_weights[class_id] = n_samples / (2 * count.float())
    return class_weights.to(device)


def _balance_classes_by_upsampling(
    activations: torch.Tensor,
    labels: torch.Tensor,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Balance binary classes by upsampling the minority class."""
    torch.manual_seed(seed)

    unique_classes = torch.unique(labels)
    class_counts = {c.item(): (labels == c).sum().item() for c in unique_classes}
    target_count = max(class_counts.values())

    balanced_indices = []
    for class_id in unique_classes:
        class_indices = torch.where(labels == class_id)[0]
        current_count = len(class_indices)
        if current_count < target_count:
            num_additional = target_count - current_count
            additional = class_indices[torch.randint(0, current_count, (num_additional,))]
            balanced_indices.append(torch.cat([class_indices, additional]))
        else:
            balanced_indices.append(class_indices)

    all_indices = torch.cat(balanced_indices)
    perm = torch.randperm(len(all_indices))
    all_indices = all_indices[perm]
    return activations[all_indices], labels[all_indices]


def _print_final_results(final_results: dict) -> None:
    """Print final evaluation results."""
    print(f"Accuracy:          {final_results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {final_results['balanced_accuracy']:.4f}")
    print(f"Loss:              {final_results['loss']:.4f}")
    print(f"Number of samples: {final_results['num_samples']}")

    print("\nPer-class metrics:")
    print("-" * 80)
    print(
        f"{'Class':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'GT Support':>10}"
    )
    print("-" * 80)
    for class_idx, metrics in final_results["per_class_metrics"].items():
        name = _LABEL_NAMES.get(class_idx, str(class_idx))
        print(
            f"{name:<20} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['gt_support']:>10}"
        )
    print("-" * 80)


def train_key_collected_probe(
    train_data_path: str,
    model_type: ModelType = "lr",
    eval_data_path: str | None = None,
    output_path: str | None = None,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    hidden_dims: str = "512,256",
    dropout: float = 0.1,
    eval_split: float = 0.2,
    subset: float = 1.0,
    class_weight: ClassWeight = None,
    balance_classes: bool = False,
    normalize: bool = False,
    device: str | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> KeyCollectedProbe:
    """Train a binary probe to detect key collection from activations.

    Takes a .pt file produced by prepare_activations_for_probing with
    probe_type=key_collected and trains a logistic regression or MLP
    classifier to predict whether the agent has collected the key.

    Args:
        train_data_path: Path to .pt file from prepare_activations_for_probing
            with probe_type=key_collected.
        model_type: "lr" for logistic regression or "mlp" for MLP.
        eval_data_path: Optional separate .pt file for evaluation.
            If not provided, splits train_data_path using eval_split.
        output_path: Path to save the trained probe. Defaults to the same
            directory as train_data_path.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for AdamW optimizer.
        batch_size: Batch size for training.
        weight_decay: L2 regularization weight.
        hidden_dims: Comma-separated hidden layer sizes for MLP (e.g., "512,256").
        dropout: Dropout rate for MLP (ignored for lr).
        eval_split: Fraction of data to hold out for evaluation when no
            eval_data_path is given.
        subset: Fraction of samples to use (0.0, 1.0].
        class_weight: "balanced" to upweight the minority class in the loss,
            or None for no weighting.
        balance_classes: If True, upsample the minority class before training.
        normalize: If True, z-score normalize activations using training stats.
            Normalization parameters are saved with the probe.
        device: "cuda", "cpu", or None (auto-detect).
        seed: Random seed for reproducibility.
        verbose: Print training progress.

    Returns:
        Trained KeyCollectedProbe (also saved to output_path).
    """
    set_seed(seed)
    torch_device = resolve_device(device)
    print(f"Using device: {torch_device}")

    train_path = Path(train_data_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    if verbose:
        print(f"Loading training data from {train_path}")

    train_data = torch.load(train_path, map_location="cpu", weights_only=False)

    probe_type = train_data.get("probe_type", train_data.get("config", {}).get("probe_type"))
    if probe_type is not None and probe_type != "key_collected":
        raise ValueError(
            f"Expected probe_type='key_collected', got '{probe_type}'. "
            "Use prepare_activations_for_probing with --probe-type key_collected."
        )

    activations = train_data["activations"]
    labels = train_data["labels"]

    if verbose:
        print(f"Loaded {activations.shape[0]} samples, activation dim: {activations.shape[1]}")
        unique, counts = torch.unique(labels, return_counts=True)
        for u, c in zip(unique.tolist(), counts.tolist(), strict=False):
            print(f"  Class {u} ({_LABEL_NAMES.get(u, '?')}): {c} samples")

    # Filter NaN
    nan_mask = torch.isnan(activations).any(dim=1)
    num_nan = nan_mask.sum().item()
    if num_nan > 0:
        if verbose:
            print(f"WARNING: Filtering {num_nan} samples with NaN values")
        activations = activations[~nan_mask]
        labels = labels[~nan_mask]

    # Validate and apply subset
    if not 0.0 < subset <= 1.0:
        raise ValueError(f"subset must be in (0.0, 1.0], got {subset}")

    if subset < 1.0:
        num_keep = max(1, int(len(activations) * subset))
        perm = torch.randperm(len(activations))
        activations = activations[perm[:num_keep]]
        labels = labels[perm[:num_keep]]
        print(f"Subset: keeping {num_keep}/{len(perm)} samples ({subset * 100:.1f}%)")

    # Train/eval split
    if eval_data_path is None:
        perm = torch.randperm(len(activations))
        split_idx = int(len(activations) * (1 - eval_split))
        train_activations = activations[perm[:split_idx]]
        train_labels = labels[perm[:split_idx]]
        eval_activations = activations[perm[split_idx:]]
        eval_labels = labels[perm[split_idx:]]
        print(f"Split: {len(train_activations)} train, {len(eval_activations)} eval")
    else:
        eval_path = Path(eval_data_path)
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation data not found: {eval_path}")
        eval_data = torch.load(eval_path, map_location="cpu", weights_only=False)
        train_activations = activations
        train_labels = labels
        eval_activations = eval_data["activations"]
        eval_labels = eval_data["labels"]
        if verbose:
            print(f"Loaded {len(eval_activations)} evaluation samples from {eval_path}")

    # Class balancing by upsampling
    if balance_classes:
        print("Balancing classes by upsampling...")
        unique, counts = torch.unique(train_labels, return_counts=True)
        print(f"  Before: {dict(zip(unique.tolist(), counts.tolist(), strict=False))}")
        train_activations, train_labels = _balance_classes_by_upsampling(train_activations, train_labels, seed=seed)
        unique, counts = torch.unique(train_labels, return_counts=True)
        print(f"  After:  {dict(zip(unique.tolist(), counts.tolist(), strict=False))}")

    # Normalization
    scaler_mean = None
    scaler_std = None
    if normalize:
        scaler_mean, scaler_std = compute_normalization_params(train_activations)
        train_activations = normalize_activations(train_activations, scaler_mean, scaler_std)
        eval_activations = normalize_activations(eval_activations, scaler_mean, scaler_std)
        print(f"Normalization enabled: computed mean/std from {train_activations.shape[0]} training samples")

    # Data loaders
    train_loader = DataLoader(
        TensorDataset(train_activations.float(), train_labels.long()),
        batch_size=batch_size,
        shuffle=True,
    )
    eval_loader = DataLoader(
        TensorDataset(eval_activations.float(), eval_labels.long()),
        batch_size=batch_size,
        shuffle=False,
    )

    # Model
    hidden_dims_list = [int(d.strip()) for d in hidden_dims.split(",") if d.strip()]
    input_dim = train_activations.shape[1]
    model = _create_model(
        model_type=model_type,
        input_dim=input_dim,
        num_classes=2,
        hidden_dims=hidden_dims_list,
        dropout=dropout,
    ).to(torch_device)

    if verbose:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {model_type}, input_dim={input_dim}, params={num_params:,}")

    # Loss and optimizer
    if class_weight == "balanced":
        print("Using balanced class weights in loss function")
        weights = _compute_class_weights(train_labels, torch_device)
        print(f"  Class weights: {weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    best_eval_accuracy = 0.0
    best_balanced_accuracy = 0.0
    best_model_state: dict | None = None

    print(f"\nTraining for {num_epochs} epochs...")
    epoch_iterator = tqdm(range(num_epochs), desc="Training", disable=not verbose)

    for _ in epoch_iterator:
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, torch_device)
        eval_results = _evaluate(model, eval_loader, criterion, torch_device)

        if eval_results["balanced_accuracy"] > best_balanced_accuracy:
            best_balanced_accuracy = eval_results["balanced_accuracy"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_eval_accuracy = max(best_eval_accuracy, eval_results["accuracy"])

        epoch_iterator.set_postfix(
            {
                "loss": f"{train_loss:.4f}",
                "eval_acc": f"{eval_results['accuracy']:.4f}",
                "bal_acc": f"{eval_results['balanced_accuracy']:.4f}",
                "best_bal": f"{best_balanced_accuracy:.4f}",
            }
        )

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(torch_device) for k, v in best_model_state.items()})

    # Final evaluation (on best model)
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (best model by balanced accuracy)")
    print("=" * 60)
    final_results = _evaluate(model, eval_loader, criterion, torch_device)

    if verbose:
        _print_final_results(final_results)

    # Build and save probe
    probe = KeyCollectedProbe(
        model=model,
        model_type=model_type,
        input_dim=input_dim,
        hidden_dims=hidden_dims_list if model_type == "mlp" else None,
        dropout=dropout if model_type == "mlp" else None,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        config={
            "train_data_path": str(train_path),
            "eval_data_path": str(eval_data_path) if eval_data_path else None,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "eval_split": eval_split,
            "subset": subset,
            "class_weight": class_weight,
            "balance_classes": balance_classes,
            "normalize": normalize,
            "seed": seed,
        },
        results={
            "best_eval_accuracy": best_eval_accuracy,
            "best_balanced_accuracy": best_balanced_accuracy,
            "final_accuracy": final_results["accuracy"],
            "final_balanced_accuracy": final_results["balanced_accuracy"],
            "final_loss": final_results["loss"],
            "per_class_metrics": {
                _LABEL_NAMES.get(i, str(i)): m for i, m in final_results["per_class_metrics"].items()
            },
        },
        device=torch_device,
    )

    if output_path is None:
        final_output_path = train_path.parent / f"key_collected_probe_{model_type}.pt"
    else:
        final_output_path = Path(output_path)

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    probe.save(final_output_path)

    print(f"\nProbe saved to {final_output_path}")
    print(f"Best eval accuracy:          {best_eval_accuracy:.4f}")
    print(f"Best balanced accuracy:      {best_balanced_accuracy:.4f}")

    return probe
