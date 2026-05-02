"""Analyze the variant-transplant intervention results.

Reads `transplant_intervention/results.jsonl` (produced by
`run_transplant_intervention.py`) and writes a markdown report.

Per (target, condition) the JSONL row carries:
  - emitted_action: the action the model produced under that condition
  - opt_target_variant: the BFS-optimal action set under the target's
    own variant (e.g. K0D0)
  - opt_donor_variant:  the BFS-optimal action set under the donor's
    variant (e.g. K1D1)

We measure, for each condition (baseline / self / cross):

  - parse rate: P(emitted_action is not None)
  - target-aligned: P(emitted_action ∈ opt_target_variant)
  - donor-aligned:  P(emitted_action ∈ opt_donor_variant)
  - target-only:    P(emitted ∈ target_opt and emitted ∉ donor_opt)
  - donor-only:     P(emitted ∈ donor_opt and emitted ∉ target_opt)

Then a per-target paired-McNemar test: of trials where self and cross
disagree on the emitted action, how many flip toward the donor
variant?

Mac CPU only.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path


CONDITIONS = ("baseline", "self", "cross")


def load_rows(path: Path):
    return [json.loads(line) for line in open(path) if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path("transplant_intervention/results.jsonl"))
    ap.add_argument("--output", type=Path,
                    default=Path("results/transplant_intervention_results.md"))
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.results)
    if not rows:
        raise SystemExit("no rows")

    n_targets = len({r["target_idx"] for r in rows})
    target_variant_counts = Counter(r["target_variant"] for r in rows)
    donor_variant_counts = Counter(r["donor_variant_cross"] for r in rows)
    print(f"Loaded {len(rows)} trials over {n_targets} targets")
    print(f"  target variants: {dict(target_variant_counts)}")
    print(f"  donor (cross) variants: {dict(donor_variant_counts)}")

    by_target_cond = {}
    for r in rows:
        by_target_cond[(r["target_idx"], r["condition"])] = r

    # ── Per-condition aggregates ───────────────────────────────────────
    agg: dict[str, dict] = {c: {"n": 0, "parsed": 0, "target_aligned": 0,
                                "donor_aligned": 0, "target_only": 0,
                                "donor_only": 0,
                                "actions": Counter()} for c in CONDITIONS}
    for r in rows:
        c = r["condition"]
        agg[c]["n"] += 1
        ea = r["emitted_action"]
        if ea is None:
            continue
        agg[c]["parsed"] += 1
        agg[c]["actions"][ea] += 1
        in_t = ea in r["opt_target_variant"]
        in_d = ea in r["opt_donor_variant"]
        if in_t:
            agg[c]["target_aligned"] += 1
        if in_d:
            agg[c]["donor_aligned"] += 1
        if in_t and not in_d:
            agg[c]["target_only"] += 1
        if in_d and not in_t:
            agg[c]["donor_only"] += 1

    # ── Per-target paired comparison (self vs cross) ────────────────────
    # How often, when self and cross disagree on the emitted action, does
    # cross emit a donor-aligned action and self does not?
    paired = {"both_target_only": 0, "both_donor_only": 0,
              "cross_donor_only_self_target_only": 0,
              "cross_target_only_self_donor_only": 0,
              "neither_clean": 0,
              "n_paired": 0,
              "n_disagreements": 0,
              "cross_changes_action_vs_self": 0,
              "cross_changes_action_vs_baseline": 0,
              "n_paired_baseline_cross": 0}

    for tid in range(n_targets):
        rb = by_target_cond.get((tid, "baseline"))
        rs = by_target_cond.get((tid, "self"))
        rc = by_target_cond.get((tid, "cross"))
        if rs is None or rc is None:
            continue
        if rs["emitted_action"] is None or rc["emitted_action"] is None:
            continue
        paired["n_paired"] += 1
        if rs["emitted_action"] != rc["emitted_action"]:
            paired["n_disagreements"] += 1
            paired["cross_changes_action_vs_self"] += 1
        in_t_s = rs["emitted_action"] in rs["opt_target_variant"]
        in_d_s = rs["emitted_action"] in rs["opt_donor_variant"]
        in_t_c = rc["emitted_action"] in rc["opt_target_variant"]
        in_d_c = rc["emitted_action"] in rc["opt_donor_variant"]
        # "Clean" categories
        s_target_only = in_t_s and not in_d_s
        s_donor_only = in_d_s and not in_t_s
        c_target_only = in_t_c and not in_d_c
        c_donor_only = in_d_c and not in_t_c
        if s_target_only and c_target_only:
            paired["both_target_only"] += 1
        elif s_donor_only and c_donor_only:
            paired["both_donor_only"] += 1
        elif s_target_only and c_donor_only:
            paired["cross_donor_only_self_target_only"] += 1
        elif s_donor_only and c_target_only:
            paired["cross_target_only_self_donor_only"] += 1
        else:
            paired["neither_clean"] += 1
        if rb is not None and rb["emitted_action"] is not None:
            paired["n_paired_baseline_cross"] += 1
            if rb["emitted_action"] != rc["emitted_action"]:
                paired["cross_changes_action_vs_baseline"] += 1

    # McNemar exact-binomial p (two-sided) on the paired discordant
    # counts: cross_donor_only_self_target_only vs cross_target_only_self_donor_only
    a = paired["cross_donor_only_self_target_only"]
    b = paired["cross_target_only_self_donor_only"]
    n_disc = a + b
    if n_disc > 0:
        k = min(a, b)
        p_value = sum(comb(n_disc, i) for i in range(k + 1)) * 2 / (2 ** n_disc)
        p_value = min(p_value, 1.0)
    else:
        p_value = 1.0

    # ── Build report ───────────────────────────────────────────────────
    out = []
    out.append("# Variant-transplant intervention (Direction 2a)\n\n")
    out.append("Tests whether the layer-15 prompt-suffix activations encoding "
               "the env-variant `(carrying_key, door_open)` are causally read "
               "out by the agent's action selection. For each target visit "
               "(a real visit at variant T) we run three forward passes:\n\n")
    out.append("- **baseline**: no intervention.\n")
    out.append("- **self**: replace layer-15 prompt-suffix activations with "
               "the mean of *same-variant* visits at the same cell "
               "(excluding the target itself) — control for averaging.\n")
    out.append("- **cross**: replace with the mean of *other-variant* "
               "donor visits at the same cell — the hypothesis test.\n\n")
    out.append("If the variant is causally encoded at this layer/position, "
               "cross-transplant should shift the emitted action toward the "
               "donor-variant's BFS-optimal action set.\n\n")

    # Setup
    target_vs = sorted(target_variant_counts.keys())
    donor_vs = sorted(donor_variant_counts.keys())
    out.append("## Setup\n\n")
    out.append(f"- Targets: {n_targets}  (variant{'s' if len(target_vs)>1 else ''} "
               f"{', '.join(target_vs)})\n")
    out.append(f"- Donor variant(s) for cross condition: {', '.join(donor_vs)}\n")
    out.append(f"- Total trials: {len(rows)}  (3 conditions × {n_targets} targets)\n\n")

    # Per-condition table
    out.append("## Aggregate metrics per condition\n\n")
    out.append("| condition | parsed | target-aligned | donor-aligned | target-only | donor-only |\n")
    out.append("|---|---:|---:|---:|---:|---:|\n")
    for c in CONDITIONS:
        a_ = agg[c]
        n = a_["n"]
        if n == 0:
            continue
        def pct(x):
            return f"{x/n:.3f}"
        out.append(f"| {c} | {a_['parsed']}/{n} ({pct(a_['parsed'])}) | "
                   f"{a_['target_aligned']} ({pct(a_['target_aligned'])}) | "
                   f"{a_['donor_aligned']} ({pct(a_['donor_aligned'])}) | "
                   f"{a_['target_only']} ({pct(a_['target_only'])}) | "
                   f"{a_['donor_only']} ({pct(a_['donor_only'])}) |\n")
    out.append("\n")

    # Action distribution per condition
    out.append("## Emitted-action distribution per condition\n\n")
    out.append("| condition | LEFT | RIGHT | UP | DOWN | (none) |\n")
    out.append("|---|---:|---:|---:|---:|---:|\n")
    for c in CONDITIONS:
        a_ = agg[c]
        ac = a_["actions"]
        none_n = a_["n"] - a_["parsed"]
        out.append(f"| {c} | {ac.get('LEFT',0)} | {ac.get('RIGHT',0)} | "
                   f"{ac.get('UP',0)} | {ac.get('DOWN',0)} | {none_n} |\n")
    out.append("\n")

    # Per-target paired analysis
    out.append("## Paired per-target comparison (self vs cross)\n\n")
    out.append(f"- Targets with both self and cross emitting parseable actions: "
               f"**{paired['n_paired']}**\n")
    out.append(f"- Of those, cross emits a different action than self: "
               f"**{paired['cross_changes_action_vs_self']}**\n")
    if paired['n_paired_baseline_cross'] > 0:
        out.append(f"- Of {paired['n_paired_baseline_cross']} targets where baseline "
                   f"and cross both parse, cross differs from baseline: "
                   f"**{paired['cross_changes_action_vs_baseline']}**\n")
    out.append("\n### Clean-categorisation (action ∈ exactly one variant's optimal set)\n\n")
    out.append("| outcome | count |\n|---|---:|\n")
    out.append(f"| both target-only (both self and cross emit target-aligned) | "
               f"{paired['both_target_only']} |\n")
    out.append(f"| both donor-only (both emit donor-aligned)                  | "
               f"{paired['both_donor_only']} |\n")
    out.append(f"| **cross→donor, self→target** (the hypothesis)              | "
               f"**{paired['cross_donor_only_self_target_only']}** |\n")
    out.append(f"| cross→target, self→donor (opposite direction)              | "
               f"{paired['cross_target_only_self_donor_only']} |\n")
    out.append(f"| neither clean (action in both or in neither set)           | "
               f"{paired['neither_clean']} |\n\n")
    if n_disc > 0:
        out.append(f"**Discordant pairs**: {a} (cross→donor, self→target) vs "
                   f"{b} (opposite). McNemar two-sided exact p ≈ "
                   f"**{p_value:.3f}**.\n\n")
    else:
        out.append("**No discordant pairs** between self and cross — variant "
                   "transplant did not produce different clean classifications.\n\n")

    # Interpretation
    out.append("## Interpretation\n\n")
    if paired["n_paired"] == 0:
        out.append("- No paired data. Re-run with more parsed outputs.\n")
    elif a > b and p_value < 0.05:
        out.append(f"- **Cross-transplant moves the agent toward the donor "
                   f"variant's optimal action significantly more often than "
                   f"the opposite direction** ({a} vs {b}, p={p_value:.3f}). "
                   "The variant identity at layer-15 prompt-suffix is "
                   "*causally* read out by the agent's action selection.\n")
    elif a > b:
        out.append(f"- Cross-transplant favours the donor variant ({a} vs {b}) "
                   f"but the effect isn't significant (p={p_value:.3f}); "
                   "more targets needed.\n")
    elif a < b:
        out.append(f"- Cross-transplant favours the *target* variant "
                   f"({a} vs {b}) — the opposite of the hypothesis. The "
                   "transplant is doing something but not what we expected.\n")
    else:
        out.append("- Cross-transplant doesn't preferentially shift toward "
                   "the donor variant ({a} = {b}). The variant identity at "
                   "layer-15 prompt-suffix is encoded but not load-bearing "
                   "for action selection.\n")

    # Files
    out.append("\n## Files\n\n")
    try:
        rel_results = args.results.relative_to(Path("/Users/wws/interp"))
    except ValueError:
        rel_results = args.results
    out.append(f"- Trial results: `{rel_results}`\n")
    out.append("- Donors: `transplant_intervention/donors.pt`\n")
    out.append("- Targets: `transplant_intervention/targets.jsonl`\n")
    out.append("- Scripts: `run_transplant_prepare_targets.py`, "
               "`run_transplant_intervention.py`, `run_transplant_analyze.py`\n")

    args.output.write_text("".join(out))
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
