"""Generate reports and plots for Seed12 action probe sweeps."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/interp_matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

STEP_OUTPUT_DIR = Path("data/probes/Seed12_T0_state_sweep_action")
STATE_AVG_OUTPUT_DIR = Path("data/probes/Seed12_T0_state_sweep_action_state_avg")
REPORT_DIR = Path("reports/seed12_action_probes")
PLOTS_DIR = REPORT_DIR / "plots"

POSITION_PRE = "pre_reasoning_suffix"
POSITION_POST = "post_reasoning_tail"


def position_order(position: str) -> int:
    """Return a stable x-axis order for probe positions."""
    if position == POSITION_PRE:
        return 0
    if position.startswith("cot_rank_"):
        return int(position.removeprefix("cot_rank_")) + 1
    if position == POSITION_POST:
        return 31
    return 10_000


def position_label(position: str) -> str:
    """Return a compact position label."""
    if position == POSITION_PRE:
        return "pre"
    if position == POSITION_POST:
        return "post"
    return position.replace("cot_rank_", "cot ")


def display_position(position: str) -> str:
    """Return a human-readable position name."""
    if position == POSITION_PRE:
        return "pre suffix"
    if position == POSITION_POST:
        return "post tail"
    if position.startswith("cot_rank_"):
        return "CoT rank " + position.removeprefix("cot_rank_")
    return position


def metric_for_label(label_type: str) -> str:
    """Return the primary metric for a label type."""
    if label_type == "true_action":
        return "balanced_accuracy"
    if label_type == "optimal_action":
        return "top1_in_optimal_set_accuracy"
    raise ValueError(f"Unknown label_type: {label_type}")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    with path.open() as f:
        return json.load(f)


def load_probe_rows(output_dir: Path, fallback_variant: str) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Load all metrics JSON files from one probe output directory."""
    manifest_path = output_dir / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else None
    rows: list[dict[str, Any]] = []

    if not output_dir.exists():
        return pd.DataFrame(), manifest

    for metrics_path in sorted(output_dir.glob("metrics_*.json")):
        record = read_json(metrics_path)
        config = record["config"]
        metrics = record["metrics"]
        split = record.get("split", {})
        metadata = record.get("dataset_metadata", {})
        label_type = config["label_type"]
        metric_name = split.get("primary_metric") or metric_for_label(label_type)
        position = config["position"]
        rows.append(
            {
                "dataset_variant": config.get("dataset_variant", fallback_variant),
                "label_type": label_type,
                "position": position,
                "position_kind": config["position_kind"],
                "position_order": position_order(position),
                "position_label": position_label(position),
                "cot_rank": config.get("cot_rank"),
                "layer": config["layer"],
                "model_type": config["model_type"],
                "metric_name": metric_name,
                "metric_value": metrics[metric_name],
                "sample_count": record["sample_count"],
                "raw_step_count": metadata.get("raw_step_count"),
                "eligible_step_count": metadata.get("eligible_step_count"),
                "grouped_state_count": metadata.get("grouped_state_count"),
                "split_group_key": metadata.get("split_group_key") or split.get("split_group_key"),
                "train_sample_count": split.get("train_sample_count"),
                "eval_sample_count": split.get("eval_sample_count"),
                "train_group_count": split.get("train_group_count"),
                "eval_group_count": split.get("eval_group_count"),
                "metrics_file": metrics_path.name,
            }
        )

    return pd.DataFrame(rows), manifest


def write_csvs(step_df: pd.DataFrame, state_df: pd.DataFrame, combined_df: pd.DataFrame) -> None:
    """Write report CSV summaries."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not step_df.empty:
        step_df.to_csv(REPORT_DIR / "all_probe_metrics.csv", index=False)
        best_by_position(step_df).to_csv(REPORT_DIR / "best_by_position.csv", index=False)
        sample_counts_by_position(step_df).to_csv(REPORT_DIR / "sample_counts_by_position.csv", index=False)
    if not state_df.empty:
        state_df.to_csv(REPORT_DIR / "all_probe_metrics_state_avg.csv", index=False)
        best_by_position(state_df).to_csv(REPORT_DIR / "best_by_position_state_avg.csv", index=False)
        sample_counts_by_position(state_df).to_csv(REPORT_DIR / "sample_counts_by_position_state_avg.csv", index=False)
    if not combined_df.empty:
        combined_df.to_csv(REPORT_DIR / "all_probe_metrics_combined.csv", index=False)


def best_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """Return best row per variant, label, and position."""
    if df.empty:
        return df
    sorted_df = df.sort_values(["metric_value", "layer", "model_type"], ascending=[False, True, True])
    return (
        sorted_df.groupby(["dataset_variant", "label_type", "position"], as_index=False)
        .first()
        .sort_values(["dataset_variant", "label_type", "position_order"])
        .reset_index(drop=True)
    )


def sample_counts_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """Return maximum sample count per variant, label, and position."""
    if df.empty:
        return df
    return (
        df.groupby(["dataset_variant", "label_type", "position", "position_order", "position_label"], as_index=False)[
            "sample_count"
        ]
        .max()
        .sort_values(["dataset_variant", "label_type", "position_order"])
    )


def save_plot(fig: plt.Figure, stem: str) -> None:
    """Save a figure as PNG and SVG."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{stem}.png", dpi=180)
    fig.savefig(PLOTS_DIR / f"{stem}.svg")
    plt.close(fig)


def plot_accuracy(df: pd.DataFrame, label_type: str, stem: str, title: str) -> None:
    """Plot primary accuracy by position for one dataset variant and label."""
    subset = df[df["label_type"] == label_type].copy()
    if subset.empty:
        return
    subset = subset.sort_values(["position_order", "layer", "model_type"])
    fig, ax = plt.subplots(figsize=(12, 6))
    for (layer, model_type), group_df in subset.groupby(["layer", "model_type"]):
        sorted_group = group_df.sort_values("position_order")
        ax.plot(
            sorted_group["position_order"],
            sorted_group["metric_value"],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=f"L{layer} {model_type}",
        )
    ticks = subset[["position_order", "position_label"]].drop_duplicates().sort_values("position_order")
    ax.set_xticks(ticks["position_order"])
    ax.set_xticklabels(ticks["position_label"], rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel(metric_for_label(label_type))
    ax.set_xlabel("position")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    save_plot(fig, stem)


def plot_best_by_position(df: pd.DataFrame, stem: str, title: str) -> None:
    """Plot best-over-probes metric by position for each label."""
    best = best_by_position(df)
    if best.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for label_type, group_df in best.groupby("label_type"):
        sorted_group = group_df.sort_values("position_order")
        ax.plot(sorted_group["position_order"], sorted_group["metric_value"], marker="o", label=label_type)
    ticks = best[["position_order", "position_label"]].drop_duplicates().sort_values("position_order")
    ax.set_xticks(ticks["position_order"])
    ax.set_xticklabels(ticks["position_label"], rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("best primary metric")
    ax.set_xlabel("position")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_plot(fig, stem)


def plot_sample_counts(df: pd.DataFrame, stem: str, title: str) -> None:
    """Plot sample counts by position."""
    counts = sample_counts_by_position(df)
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for label_type, group_df in counts.groupby("label_type"):
        sorted_group = group_df.sort_values("position_order")
        ax.plot(sorted_group["position_order"], sorted_group["sample_count"], marker="o", label=label_type)
    ticks = counts[["position_order", "position_label"]].drop_duplicates().sort_values("position_order")
    ax.set_xticks(ticks["position_order"])
    ax.set_xticklabels(ticks["position_label"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("samples")
    ax.set_xlabel("position")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_plot(fig, stem)


def plot_variant_comparison(combined_df: pd.DataFrame) -> None:
    """Plot best-by-position comparison across dataset variants."""
    if combined_df.empty or combined_df["dataset_variant"].nunique() < 2:
        return
    best = best_by_position(combined_df)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for ax, label_type in zip(axes, ["true_action", "optimal_action"], strict=True):
        subset = best[best["label_type"] == label_type]
        for variant, group_df in subset.groupby("dataset_variant"):
            sorted_group = group_df.sort_values("position_order")
            ax.plot(sorted_group["position_order"], sorted_group["metric_value"], marker="o", label=variant)
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel(metric_for_label(label_type))
        ax.set_title(label_type)
        ax.grid(True, alpha=0.25)
        ax.legend()
    ticks = best[["position_order", "position_label"]].drop_duplicates().sort_values("position_order")
    axes[-1].set_xticks(ticks["position_order"])
    axes[-1].set_xticklabels(ticks["position_label"], rotation=60, ha="right", fontsize=8)
    axes[-1].set_xlabel("position")
    save_plot(fig, "accuracy_best_by_position_comparison")


def write_plots(step_df: pd.DataFrame, state_df: pd.DataFrame, combined_df: pd.DataFrame) -> None:
    """Write all report plots."""
    if not step_df.empty:
        plot_accuracy(step_df, "true_action", "accuracy_true_action", "Step-level true-action probes")
        plot_accuracy(step_df, "optimal_action", "accuracy_optimal_action", "Step-level optimal-action probes")
        plot_best_by_position(step_df, "accuracy_best_by_position", "Step-level best accuracy by position")
        plot_sample_counts(step_df, "sample_counts_by_position", "Step-level sample counts by position")
    if not state_df.empty:
        plot_accuracy(state_df, "true_action", "accuracy_true_action_state_avg", "State-averaged true-action probes")
        plot_accuracy(
            state_df,
            "optimal_action",
            "accuracy_optimal_action_state_avg",
            "State-averaged optimal-action probes",
        )
        plot_best_by_position(
            state_df,
            "accuracy_best_by_position_state_avg",
            "State-averaged best accuracy by position",
        )
        plot_sample_counts(
            state_df,
            "sample_counts_by_position_state_avg",
            "State-averaged sample counts by position",
        )
    plot_variant_comparison(combined_df)


def pct(value: float | int | None) -> str:
    """Format a metric as a percentage."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100 * float(value):.2f}%"


def int_text(value: Any) -> str:
    """Format integer-like values for markdown."""
    if value is None or pd.isna(value):
        return "n/a"
    return str(int(value))


def markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    """Render a small markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def best_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return best row per variant, label, model, and layer."""
    sorted_df = df.sort_values(["metric_value", "position_order"], ascending=[False, True])
    return (
        sorted_df.groupby(["dataset_variant", "label_type", "model_type", "layer"], as_index=False)
        .first()
        .sort_values(["dataset_variant", "label_type", "model_type", "layer"])
        .reset_index(drop=True)
    )


def select_best_row(df: pd.DataFrame) -> pd.Series:
    """Select one best row with deterministic tie-breaking."""
    return df.sort_values(
        ["metric_value", "position_order", "layer", "model_type"], ascending=[False, True, True, True]
    ).iloc[0]


def append_best_summary(lines: list[str], df: pd.DataFrame, title: str) -> None:
    """Append best result markdown for one variant."""
    if df.empty:
        return
    lines.extend([f"## {title}", ""])

    overall_rows = []
    for label_type, group in df.groupby("label_type"):
        best = select_best_row(group)
        overall_rows.append(
            [
                label_type,
                display_position(best["position"]),
                best["layer"],
                best["model_type"],
                best["metric_name"],
                pct(best["metric_value"]),
                int_text(best["sample_count"]),
            ]
        )
    lines.append(
        markdown_table(
            overall_rows,
            ["label", "best position", "layer", "model", "metric", "value", "samples"],
        )
    )
    lines.append("")

    rows = []
    for _, row in best_rows(df).iterrows():
        rows.append(
            [
                row["label_type"],
                row["model_type"],
                row["layer"],
                display_position(row["position"]),
                pct(row["metric_value"]),
                int_text(row["sample_count"]),
            ]
        )
    lines.append(markdown_table(rows, ["label", "model", "layer", "best position", "value", "samples"]))
    lines.append("")


def append_sample_coverage(lines: list[str], df: pd.DataFrame, title: str) -> None:
    """Append sample coverage summary."""
    if df.empty:
        return
    counts = sample_counts_by_position(df)
    rows = []
    for label_type, group in counts.groupby("label_type"):
        pre = group[group["position"] == POSITION_PRE]["sample_count"].max()
        post = group[group["position"] == POSITION_POST]["sample_count"].max()
        cot = group[group["position"].str.startswith("cot_rank_")]
        rows.append(
            [
                label_type,
                int_text(pre),
                int_text(cot["sample_count"].max() if not cot.empty else None),
                int_text(cot["sample_count"].min() if not cot.empty else None),
                int_text(post),
            ]
        )
    lines.extend([f"## {title} Sample Coverage", ""])
    lines.append(markdown_table(rows, ["label", "pre", "CoT max", "CoT min", "post"]))
    lines.append("")


def append_variant_comparison(lines: list[str], combined_df: pd.DataFrame) -> None:
    """Append compact comparison between step and state-average variants."""
    if combined_df.empty or combined_df["dataset_variant"].nunique() < 2:
        return
    best = best_by_position(combined_df)
    rows = []
    for label_type in ["true_action", "optimal_action"]:
        for variant in ["step", "state_average"]:
            subset = best[(best["label_type"] == label_type) & (best["dataset_variant"] == variant)]
            if subset.empty:
                continue
            row = select_best_row(subset)
            rows.append(
                [
                    variant,
                    label_type,
                    display_position(row["position"]),
                    row["layer"],
                    row["model_type"],
                    pct(row["metric_value"]),
                    int_text(row["sample_count"]),
                ]
            )
    lines.extend(["## Variant Comparison", ""])
    lines.append(markdown_table(rows, ["variant", "label", "best position", "layer", "model", "value", "samples"]))
    lines.append("")
    lines.append("![Best accuracy comparison](plots/accuracy_best_by_position_comparison.png)")
    lines.append("")


def manifest_count(manifest: dict[str, Any] | None, key: str) -> str:
    """Format a manifest count."""
    if manifest is None:
        return "n/a"
    return str(manifest.get(key, "n/a"))


def write_report(
    step_df: pd.DataFrame,
    state_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    step_manifest: dict[str, Any] | None,
    state_manifest: dict[str, Any] | None,
) -> None:
    """Write the markdown report."""
    lines = [
        "# Seed12 Action Probe Position Sweep Report",
        "",
        "## Run Summary",
        "",
        "Trained action probes from `data/activations/Seed12_T0_state_sweep/` against "
        "`data/trajectories/Seed12_T0_state_sweep/`. The original run uses one sample per trajectory step. "
        "The new state-averaged variant first averages activations for each `(position, has_key, door_open)` state.",
        "",
        markdown_table(
            [
                [
                    "step",
                    "`data/probes/Seed12_T0_state_sweep_action/`",
                    manifest_count(step_manifest, "completed_probe_count"),
                    manifest_count(step_manifest, "expected_probe_count"),
                ],
                [
                    "state_average",
                    "`data/probes/Seed12_T0_state_sweep_action_state_avg/`",
                    manifest_count(state_manifest, "completed_probe_count"),
                    manifest_count(state_manifest, "expected_probe_count"),
                ],
            ],
            ["variant", "output directory", "completed", "expected"],
        ),
        "",
        "CSV summaries are in `reports/seed12_action_probes/`, including "
        "`all_probe_metrics_combined.csv` for direct variant comparisons.",
        "",
        "## What These Probes Are",
        "",
        "An action probe is a small supervised model trained on frozen language-model activations. "
        "For each sample, the probe receives one activation vector from a chosen model layer and token position, "
        "then predicts either the model's recorded action or the set of A* optimal actions. The base GPT-OSS model "
        "is not updated.",
        "",
        '- `true_action`: single-class label from `step["agent_action"]`; invalid non-action steps are skipped.',
        '- `optimal_action`: 4-way multi-hot label from `step["astar_actions"]`; ties are preserved.',
        "- `linear`: `nn.Linear(2880, 4)`.",
        "- `mlp`: hidden dimensions `[512, 256]` with dropout `0.1`.",
        "- Action order: `LEFT=0`, `RIGHT=1`, `UP=2`, `DOWN=3`.",
        "",
        "## Sweep Plan",
        "",
        "Each variant trains the same 384 probe configurations: `(30 + 2) * 3 * 2 * 2`, covering 30 CoT ranks, "
        "pre-reasoning suffix, post-reasoning tail, layers 7/15/23, two label types, and two model types.",
        "",
        "- `pre_reasoning_suffix`: mean-pool saved `prompt_suffix` activations from `-3:-1`.",
        "- `cot_rank_k`: use the kth saved `@analysis/10` checkpoint without pooling across CoT positions.",
        "- `post_reasoning_tail`: mean-pool saved output tail activations from `-16:-14`.",
        "- Missing CoT ranks are skipped per sample; CoT indices overlapping post-tail indices are excluded.",
        "",
        "## Why CoT Sample Counts Decline",
        "",
        "The number of samples drops at later CoT positions because `cot_rank_k` exists only when a step produced "
        "at least `k + 1` saved analysis checkpoints. CoT activations were saved as `@analysis/10`, so `cot_rank_0` "
        "requires one saved analysis checkpoint, while `cot_rank_29` requires 30 saved checkpoints, roughly 290 "
        "analysis tokens. Shorter reasoning traces therefore cannot contribute to late CoT ranks.",
        "",
        "There is one additional intentional skip: if a saved CoT checkpoint overlaps the reserved post-tail token "
        "indices `-16:-14`, that sample is skipped for the CoT probe so the CoT and post-tail probes do not train "
        "on duplicated token positions.",
        "",
        "For the state-averaged variant, this filtering happens before averaging. A state contributes to "
        "`cot_rank_k` only if at least one eligible step for that state has that checkpoint, so grouped state counts "
        "also decline at late CoT ranks.",
        "",
        "## State-Averaged Variant",
        "",
        "The state-averaged variant changes only dataset construction. It parses the agent coordinate from the `A` "
        'marker in `step["grid_state"]`, uses `bool(step["carrying_key"])` as `has_key`, and uses '
        '`bool(step["door_open"])` as `door_open`. All eligible activations with the same state key are averaged '
        "before training, producing one sample per state key.",
        "",
        "Labels are required to agree within a state group. The current data has no label conflicts under this key, "
        "so the implementation fails loudly if a future run introduces one instead of silently majority-voting.",
        "",
        "The split for this variant is by state key rather than trajectory name. This removes the repeated-state "
        "leakage noted in the step-level caveat and makes the reported metrics a held-out-state validation result.",
        "",
        "## Sanity Checks And Caveats",
        "",
        "The recorded model action is optimal only rarely in this dataset: `406 / 3192 = 12.72%` of parseable "
        "`agent_action` labels are in the `astar_actions` set. This does not imply low `true_action` probe accuracy, "
        "because `true_action` predicts what the model did, not whether it was optimal.",
        "",
        "For the step-level variant, the split is disjoint by trajectory name but not by underlying state. In the "
        "pre-reasoning layer-23 true-action split, `554 / 806 = 68.73%` of eval samples share a state key with "
        "training. A state-key majority baseline reaches `89.70%` raw accuracy and `82.98%` balanced accuracy there. "
        "The state-averaged variant is intended to address that specific caveat.",
        "",
    ]

    if not step_df.empty:
        lines.extend(
            [
                "## Step-Level Accuracy Plots",
                "",
                "![Step true-action accuracy](plots/accuracy_true_action.png)",
                "",
                "![Step optimal-action accuracy](plots/accuracy_optimal_action.png)",
                "",
                "![Step best accuracy by position](plots/accuracy_best_by_position.png)",
                "",
                "![Step sample counts by position](plots/sample_counts_by_position.png)",
                "",
            ]
        )
    if not state_df.empty:
        lines.extend(
            [
                "## State-Averaged Accuracy Plots",
                "",
                "![State-averaged true-action accuracy](plots/accuracy_true_action_state_avg.png)",
                "",
                "![State-averaged optimal-action accuracy](plots/accuracy_optimal_action_state_avg.png)",
                "",
                "![State-averaged best accuracy by position](plots/accuracy_best_by_position_state_avg.png)",
                "",
                "![State-averaged sample counts by position](plots/sample_counts_by_position_state_avg.png)",
                "",
            ]
        )

    append_best_summary(lines, step_df, "Step-Level Best Results")
    append_best_summary(lines, state_df, "State-Averaged Best Results")
    append_variant_comparison(lines, combined_df)
    append_sample_coverage(lines, step_df, "Step-Level")
    append_sample_coverage(lines, state_df, "State-Averaged")

    lines.extend(
        [
            "## Interpretation",
            "",
            "The step-level probes answer whether action information is accessible under the original trajectory-name "
            "split. The state-averaged probes answer a stricter question: after collapsing repeated states and "
            "holding out state keys, how much action information remains accessible from the averaged hidden state.",
            "",
            "When comparing positions, read late CoT ranks together with the sample-count plots because some steps do "
            "not have enough saved `@analysis/10` checkpoints to contribute to late ranks.",
            "",
        ]
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "seed12_action_probe_report.md").write_text("\n".join(lines))


def main() -> None:
    """Generate report artifacts."""
    step_df, step_manifest = load_probe_rows(STEP_OUTPUT_DIR, "step")
    state_df, state_manifest = load_probe_rows(STATE_AVG_OUTPUT_DIR, "state_average")
    combined_df = pd.concat([df for df in [step_df, state_df] if not df.empty], ignore_index=True)

    write_csvs(step_df, state_df, combined_df)
    write_plots(step_df, state_df, combined_df)
    write_report(step_df, state_df, combined_df, step_manifest, state_manifest)
    print(f"Wrote report to {REPORT_DIR / 'seed12_action_probe_report.md'}")


if __name__ == "__main__":
    main()
