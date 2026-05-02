# Variant-transplant intervention (Direction 2a)

Tests whether the layer-15 prompt-suffix activations encoding the
env-variant `(carrying_key, door_open)` are causally read out by
the agent's action selection. For each target visit we run three
forward passes:

- **baseline**: no intervention.
- **self**: replace layer-15 prompt-suffix activations with the mean
  of *same-variant* (K0D0) visits at the same cell, excluding the
  target itself — control for averaging.
- **cross**: replace with the mean of *other-variant* (K1D1) donor
  visits at the same cell — the hypothesis test.

If the variant is causally encoded, cross-transplant should shift
the emitted action toward K1D1-like behaviour.

**TL;DR — null, with a hint of anti-effect.** Cross-transplant does
*not* push the K0D0 agent toward K1D1 behaviour. On the auto-report
metric (action ∈ K1D1's BFS-optimal set), the discordant count is
0 cross→donor vs 4 cross→target (McNemar p=0.125). On the more
direct *behavioural* metric (action == K1D1's modal recorded
emission at this cell), cross-transplant moves alignment from 60 %
(baseline) to 38 % — a 22 pp swing **away** from the donor variant.

---

## Setup

- Targets: 46 K0D0 visits across 14 lower-room cells where K0D0
  and K1D1 BFS-optimal action sets differ.
- Donor variant: K1D1 (mean of K1D1 prompt-suffix activations at
  the same cell).
- Total trials: 138 (3 conditions × 46 targets).
- Model: GPT-OSS-20B, layer 15, last 3 prompt-suffix tokens.
- Decoding: T = 0 (greedy), max 1024 new tokens.

## Aggregate metrics — two views

### View 1: alignment with BFS-optimal action sets

| condition | parsed | ∈ K0D0-optimal | ∈ K1D1-optimal | K0D0-only | K1D1-only |
|---|---:|---:|---:|---:|---:|
| baseline | 30/46 (0.65) | 14 (0.30) | 17 (0.37) | 9 (0.20) | 12 (0.26) |
| self     | 31/46 (0.67) | 10 (0.22) | 15 (0.33) | 6 (0.13) | 11 (0.24) |
| cross    | 34/46 (0.74) | 15 (0.33) | 10 (0.22) | 13 (0.28) | 8 (0.17) |

Cross has *more* K0D0-aligned emissions (15 vs 14 baseline) and
*fewer* K1D1-aligned (10 vs 17). Opposite of hypothesis.

### View 2: alignment with each variant's modal recorded emission

(This is more direct: "does cross-transplant make the agent emit
what real K1D1 agents emit at this cell?", regardless of whether
those K1D1 emissions are BFS-optimal.)

| condition | parsed | matches K0D0 modal | matches K1D1 modal |
|---|---:|---:|---:|
| baseline | 30 | 16 (0.53) | 18 (0.60) |
| self     | 31 | 11 (0.35) | 14 (0.45) |
| **cross**| 34 | **21 (0.62)** | **13 (0.38)** |

Cross-transplant **moves toward K0D0 modal** (53 → 62 %, +9 pp)
and **away from K1D1 modal** (60 → 38 %, −22 pp). Net 31 pp swing
in the wrong direction.

## Action distribution per condition

| condition | LEFT | RIGHT | UP | DOWN | (none) |
|---|---:|---:|---:|---:|---:|
| baseline | 11 | 4 | 15 | 0 | 16 |
| self     | 7  | 4 | 17 | 3 | 15 |
| cross    | 2  | 8 | 24 | 0 | 12 |

Cross over-emits UP (24/34 ≈ 70 % of parsed) and under-emits LEFT
(2/34, vs 11/30 baseline).

## Where do cross-transplant action changes go?

Of 11 paired targets where cross emits a different action than
baseline:

| destination | count |
|---|---:|
| matches K1D1 modal (the hypothesis) | 2 |
| matches K0D0 modal (anti-hypothesis) | **6** |
| matches neither modal | 5 (mostly UP at cells where K0D0/K1D1 modals differ) |

## Why the hypothesis fails — three observations

**1. K1D1 agents themselves are far from BFS-optimal.** The donor
variant doesn't encode "go toward goal correctly"; it encodes "be a
K1D1 agent" — and K1D1 agents have the same UP-bias as K0D0 agents.
Examples from the recorded data:

| cell | K1D1 BFS-optimal | K1D1 emitted (recorded) | % in opt |
|---|---|---|---:|
| (5, 6) | DOWN | UP × 33, RIGHT × 12 | 4 % |
| (5, 7) | LEFT | UP × 3 | 0 % |
| (3, 5) | LEFT | UP × 5, LEFT × 2 | 29 % |
| (6, 6) | LEFT/DOWN | UP × 13, LEFT × 6 | 32 % |

So even a *perfect* "make the agent behave like K1D1" intervention
wouldn't reliably hit K1D1's BFS-optimal set. The auto-report's
View-1 metric is only meaningful where K1D1 agents themselves act
optimally — which is a minority of these cells.

**2. K0D0 baseline is already K1D1-like.** K0D0 baseline matches
K1D1 modal 60 % of the time and K0D0 modal only 53 %. The K0D0
agent's UP-bias makes it emit "K1D1-style" actions at most cells
even before any intervention. There's nowhere for cross-transplant
to push *toward* K1D1.

**3. Self-transplant is also disruptive.** Self drops K0D0-modal
alignment by 18 pp (53 → 35) and K1D1-modal by 15 pp (60 → 45).
The averaging itself perturbs the activation pattern enough to
shift the agent's action ~half the time, but in no consistent
direction. This means the variance in the cross-transplant
condition is partly just "averaging perturbation," not
"variant-specific information being injected."

## Interpretation

Three readings of the null + anti-effect:

(a) **Variant identity at layer-15 prompt-suffix is not load-bearing
for action selection.** The model decodes `has_key` and `door_open`
from the prompt directly during reasoning; the variant
representation at this position is a passive readout that doesn't
influence the action token. This matches the previous null on the
prior dataset and matches the per-visit decoder finding that the
*action* lives at output positions, not at prompt-suffix.

(b) **The transplant creates an out-of-distribution residual
stream.** The mean K1D1 activation, dropped into a K0D0 prompt
context, is a state the network never sees in normal operation.
The model handles it by defaulting to the prompt's content
(K0D0 grid is visible, key is on grid) and possibly by
*over*-correcting away from the perturbation. This would explain
why cross moves *toward* K0D0 modal (the model "fights back" the
perturbation by relying more heavily on the unmodified prompt).

(c) **The signal is there but masked by the model's UP-prior.**
Both K0D0 and K1D1 agents over-emit UP regardless of optimality,
so any intervention that disturbs the residual stream gets
absorbed into "more UP." Look at the action distribution: cross
emits UP 70 % of the time, vs ~50 % for the others. Cross may be
amplifying the UP-prior rather than encoding a variant shift.

## Implications for the research agenda

This is consistent with the *prior* (deleted-branch) causal
intervention being null — the redesigned, more naturalistic
transplant doesn't rescue it. The conclusion holds: **at layer-15
prompt-suffix, the variant/belief representation is not the
causal mediator of the agent's action.**

Next steps from `results/research_direction.md`:

- **Direction 1** (probe through the reasoning chain) — find
  *where* in the trace the geometric belief is overruled. With 2a
  null, this is the natural next test.
- **Direction 2b** (late-position intervention) — intervene at the
  output-token position right before the action JSON. The
  per-visit decoder reads the action off these positions at >90 %
  accuracy, so this is where the action commitment plausibly
  lives.
- **Direction 6** (does φ encode flags vs sub-goal pointer?) — the
  null here is *consistent* with "the model encodes variant
  identity at layer 15 but uses a sub-goal pointer downstream";
  Direction 6 is the cheapest probe-only test of that.

## Files

- Trial results: `transplant_intervention/results.jsonl` (138 rows)
- Donors: `transplant_intervention/donors.pt`
- Targets: `transplant_intervention/targets.jsonl` (46 rows)
- Scripts: `run_transplant_prepare_targets.py`,
  `run_transplant_intervention.py`,
  `run_transplant_analyze.py`
