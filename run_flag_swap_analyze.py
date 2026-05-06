"""Analyze flag-swap intervention results.

Reads `flag_swap/results.jsonl` (180 rows: 6 conditions × N samples)
and writes a markdown report.

Compares action distributions across conditions:

  - Sanity: self-a ≈ baseline-a (chi² p > 0.05)?
  - Sanity: self-b ≈ baseline-b?
  - Hypothesis test 1: swap-a-from-b shifts a's distribution toward b's.
  - Hypothesis test 2: swap-b-from-a shifts b's distribution toward a's.

The expected pattern under the "carrying_key flag is causal" hypothesis:

  - baseline-a / self-a peak on RIGHT/DOWN (K0D0-optimal, no key).
  - baseline-b / self-b peak on UP (K1D0-optimal, has key).
  - swap-a-from-b shifts toward UP (the K1D0-optimal direction).
  - swap-b-from-a shifts toward RIGHT/DOWN.

Mac CPU only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import isnan
from pathlib import Path

CONDITIONS = ("baseline-a", "self-a", "swap-a-from-b", "baseline-b", "self-b", "swap-b-from-a")
ACTIONS = ("LEFT", "RIGHT", "UP", "DOWN")


def chi2_4class(counts_x: dict, counts_y: dict) -> tuple[float, float]:
    """Pearson chi² test for two 4-class action distributions.
    Returns (chi², p_value). Uses scipy if available, else falls back
    to a closed-form approximation that's good enough for our small N.
    """
    obs = [[counts_x.get(a, 0) for a in ACTIONS], [counts_y.get(a, 0) for a in ACTIONS]]
    try:
        from scipy.stats import chi2_contingency  # type: ignore

        chi2, p, _, _ = chi2_contingency(obs)
        return float(chi2), float(p)
    except Exception:
        pass
    # No-scipy fallback: hand-roll Pearson chi² over 2x4 contingency.
    n_x = sum(obs[0])
    n_y = sum(obs[1])
    n = n_x + n_y
    if n == 0 or n_x == 0 or n_y == 0:
        return 0.0, 1.0
    chi2 = 0.0
    for j in range(4):
        col = obs[0][j] + obs[1][j]
        for i in range(2):
            row_n = (n_x, n_y)[i]
            expected = row_n * col / n
            if expected == 0:
                continue
            chi2 += (obs[i][j] - expected) ** 2 / expected
    # p-value via incomplete-gamma approximation (3 dof for 2x4 - 1 = 3).
    # We don't have scipy, so report just the chi2 statistic; user can
    # convert using a chi² table.
    return float(chi2), float("nan")


def total_variation(counts_x: dict, counts_y: dict) -> float:
    n_x = sum(counts_x.get(a, 0) for a in ACTIONS)
    n_y = sum(counts_y.get(a, 0) for a in ACTIONS)
    if n_x == 0 or n_y == 0:
        return 0.0
    return 0.5 * sum(abs(counts_x.get(a, 0) / n_x - counts_y.get(a, 0) / n_y) for a in ACTIONS)


def format_action_count(action_counts: Counter, total: int, action: str) -> str:
    """Format one action count and fraction for report tables."""
    value = action_counts.get(action, 0)
    return f"{value} ({value / total:.2f})" if total else "0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("flag_swap/results.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("results/flag_swap_results.md"))
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in open(args.results) if line.strip()]
    if not rows:
        raise SystemExit("no rows")

    # Group by condition; build action counts.
    by_cond: dict[str, list] = defaultdict(list)
    hook_logs: dict[str, dict] = {}
    for r in rows:
        c = r["condition"]
        by_cond[c].append(r)
        if "_hook_log" in r:
            hook_logs[c] = r["_hook_log"]
    counts: dict[str, Counter] = {
        c: Counter(r["emitted_action"] or "PARSE_FAIL" for r in by_cond[c]) for c in CONDITIONS if c in by_cond
    }
    n_per_cond = {c: len(by_cond[c]) for c in by_cond}

    # ── Build report ──────────────────────────────────────────────────
    out = []
    out.append("# Flag-swap activation patching at the door (carrying_key)\n\n")
    if rows:
        # Pull metadata from the first baseline-a row.
        a0 = next((r for r in rows if r["condition"] == "baseline-a"), rows[0])
        b0 = next((r for r in rows if r["condition"] == "baseline-b"), rows[0])
        out.append("- Cell: agent at the cell directly below the locked door.\n")
        out.append(
            f"- Prompt **a** (variant {a0['prompt_variant']}, recorded action would be "
            f"the K0D0-optimal one — head back for the key).\n"
        )
        out.append(f"- Prompt **b** (variant {b0['prompt_variant']}, recorded action: UP through the door).\n")
        out.append(f"- N samples per condition: {n_per_cond.get('baseline-a', '?')}\n")
        out.append("- Sampling: T = 0.7 (project rule).\n\n")

    out.append("## Hook sanity (first sample of each condition)\n\n")
    out.append("| condition | fired | seq_len | mode | donor_norm | replacements | first pre→post norm |\n")
    out.append("|---|---|---|---|---:|---:|---|\n")
    for c in CONDITIONS:
        log = hook_logs.get(c, {})
        if not log:
            out.append(f"| {c} | (no log) |  |  |  |  |  |\n")
            continue
        reps = log.get("replacements", [])
        first_rep = ""
        if reps:
            r0 = reps[0]
            first_rep = f"{r0['pre_norm']:.1f} → {r0['post_norm']:.1f}"
        donor_norm = log.get("donor_norm")
        donor_norm_str = "" if donor_norm is None else f"{donor_norm:.1f}"
        out.append(
            f"| {c} | {log.get('fired')} | {log.get('seq_len')} | "
            f"{log.get('mode', '')} | {donor_norm_str} | {len(reps)} | {first_rep} |\n"
        )
    out.append("\n")

    out.append("## Action distribution per condition\n\n")
    out.append("| condition | LEFT | RIGHT | UP | DOWN | parse_fail | n |\n")
    out.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for c in CONDITIONS:
        if c not in counts:
            continue
        cn = counts[c]
        n = n_per_cond[c]
        out.append(
            f"| {c} | {format_action_count(cn, n, 'LEFT')} | {format_action_count(cn, n, 'RIGHT')} | "
            f"{format_action_count(cn, n, 'UP')} | {format_action_count(cn, n, 'DOWN')} | "
            f"{cn.get('PARSE_FAIL', 0)} | {n} |\n"
        )
    out.append("\n")

    # ── Pairwise comparisons ───────────────────────────────────────────
    pairs = [
        ("self-a vs baseline-a (sanity, must NOT differ)", "self-a", "baseline-a"),
        ("self-b vs baseline-b (sanity, must NOT differ)", "self-b", "baseline-b"),
        ("swap-a-from-b vs self-a (HYPOTHESIS, should differ)", "swap-a-from-b", "self-a"),
        ("swap-b-from-a vs self-b (HYPOTHESIS, should differ)", "swap-b-from-a", "self-b"),
        ("swap-a-from-b vs baseline-b (does a→b move all the way to b?)", "swap-a-from-b", "baseline-b"),
        ("swap-b-from-a vs baseline-a (does b→a move all the way to a?)", "swap-b-from-a", "baseline-a"),
    ]
    out.append("## Pairwise distribution comparisons\n\n")
    out.append("| comparison | TV distance | chi² | p (4-class, 3 dof) |\n")
    out.append("|---|---:|---:|---:|\n")
    for label, x, y in pairs:
        if x not in counts or y not in counts:
            continue
        tv = total_variation(counts[x], counts[y])
        chi2, p = chi2_4class(counts[x], counts[y])
        p_str = "(scipy missing)" if isnan(p) else f"{p:.3f}"
        out.append(f"| {label} | {tv:.3f} | {chi2:.2f} | {p_str} |\n")
    out.append("\n")

    # ── Direction-of-shift summary ─────────────────────────────────────
    def freq(c, a):
        n = n_per_cond.get(c, 0)
        return counts.get(c, {}).get(a, 0) / n if n else 0

    out.append("## Direction-of-shift on UP frequency\n\n")
    out.append(
        "If the variant flag is causally encoded, swap-a-from-b "
        "should INCREASE UP frequency (toward the K1D0-optimal direction), "
        "and swap-b-from-a should DECREASE UP frequency.\n\n"
    )
    out.append("| condition | freq(UP) |\n|---|---:|\n")
    for c in CONDITIONS:
        out.append(f"| {c} | {freq(c, 'UP'):.3f} |\n")
    out.append("\n")
    delta_a = freq("swap-a-from-b", "UP") - freq("self-a", "UP")
    delta_b = freq("swap-b-from-a", "UP") - freq("self-b", "UP")
    out.append(f"- Δ freq(UP) on a-side (swap − self): **{delta_a:+.3f}** (expected positive under hypothesis).\n")
    out.append(f"- Δ freq(UP) on b-side (swap − self): **{delta_b:+.3f}** (expected negative under hypothesis).\n\n")

    # ── Interpretation ─────────────────────────────────────────────────
    out.append("## Interpretation\n\n")
    out.append("Read three things:\n\n")
    out.append(
        "1. **Hook sanity**: every condition's `fired=True`; baseline rows "
        "should run in passthrough mode and self/swap rows should run in "
        "replace mode with a donor tensor. If not, the hook plumbing is "
        "broken and the rest is invalid.\n"
    )
    out.append(
        "2. **Self-controls indistinguishable from baseline**: the two "
        "`self-* vs baseline-*` chi² tests should give large p-values. "
        "If a self-* differs from its baseline, the hook itself is biasing "
        "the model and we can't trust the swap rows.\n"
    )
    out.append(
        "3. **Swaps move the action distribution toward the donor variant's**: "
        "Δ freq(UP) > 0 on the a-side (swap-a-from-b raises UP toward "
        "K1D0's bias) and Δ freq(UP) < 0 on the b-side (swap-b-from-a "
        "drops UP toward K0D0's bias). Both deltas in the predicted "
        "direction with chi² p < 0.05 against the same-variant control "
        "is the causal-encoding result.\n"
    )

    out.append("\n## Files\n\n")
    out.append(f"- Trial results: `{args.results}`\n")
    out.append("- Prompts: `flag_swap/prompts.jsonl`\n")
    out.append("- Activations: `flag_swap/act_a.pt`, `flag_swap/act_b.pt`\n")
    out.append("- Scripts: `run_flag_swap_prepare.py`, `run_flag_swap_intervention.py`, `run_flag_swap_analyze.py`\n")

    args.output.write_text("".join(out))
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
