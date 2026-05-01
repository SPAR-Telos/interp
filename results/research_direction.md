# Research Directions: Belief-Action Gap in GPT-OSS-20B Grid Agents

## Background — what the existing experiments established

On a 9 × 9 fixed-grid task (`Seed12`, door-open variant) the agent
(GPT-OSS-20B with chain-of-thought) reaches the goal in only 48 % of
episodes and emits a BFS-optimal action only 43 % of the time.
However, a **single linear probe** on the layer-15 pre-reasoning
residual stream (last 3 prompt-suffix tokens, 8640-dim φ) picks the
BFS-optimal action with **96.3 %** accuracy on held-out visits
(5-fold CV). That's a **+51.6 pp belief-action gap**: the model
"knows" the right answer at near-ceiling accuracy in its mid-stack
residuals, yet acts on it less than half the time.

A layer sweep at the same prompt-suffix position shows the gap is
flat across depths (+49 / +52 / +42 pp at layers 7 / 15 / 23). A
per-visit probe on **post-reasoning** activations inverts the result:
post-reasoning φ predicts the agent's *emitted* action at 92 %
accuracy and the optimal action at only 52 %. So the geometric belief
exists at every pre-reasoning depth and never commits to the agent's
chosen action — the commitment shows up only in the output-token
positions. **The gap opens during the reasoning chain**, not across
pre-reasoning depth.

The companion easy task (`two_path_no_key_T0`, 7 × 8 two-corridor
grid) shows the same shape with a smaller gap (+22 pp) — consistent
with the agent's higher BFS-optimal rate (75 %) and 98 % success rate.

This document lists the most fruitful follow-ups, ordered by
priority.

---

## Direction 1 — Probe through the reasoning chain

**Question:** at what *point in the chain-of-thought* does the right
belief get lost?

**Motivation.** The two existing probe positions are the extremes:
prompt-suffix (last 3 tokens before reasoning, belief acc 96 %) and
output (last few tokens after reasoning, "belief" reads off the
agent's action with 92 % accuracy). The reasoning trace itself is
hundreds of tokens long. If the same belief decoder is run at
intermediate trace positions, we can plot belief-vs-optimal accuracy
as a function of position-in-trace. Three plausible shapes:

- **Monotone decay** — belief gradually corrupted as reasoning
  progresses → CoT accumulates errors token by token.
- **Cliff** — belief stays high until some specific point, then drops
  → there's a "moment of commitment" we can localise.
- **Plateau then crash at output** — the geometric belief is
  preserved through the whole reasoning trace and only the final
  projection to the action token overrules it → the bug is in the
  unembedding / output-prediction step, not in the reasoning content.

Each shape implies a different intervention target.

**Concrete experiment.**
1. Re-extract activations at a sample of reasoning-trace positions
   (e.g. every 20 tokens) for layers 7 / 15 / 23 on the existing
   trajectories.
2. Run the same 5-fold CV linear probe with BFS-optimal labels per
   position.
3. Plot belief acc, belief-action agreement, and the gap as a
   function of position-in-trace.
4. Bonus: stratify by trajectory outcome (success vs cap-hit) — if
   the cliff appears in failures but not successes, the location is
   the *failure-causing event* in the trace.

**Cost.** Moderate. The expensive part is re-running
`gather_activations` with a wider token window (or all reasoning
tokens) for the existing 79 trajectories. ~1–2 hours of GPU time +
some disk; the analysis itself reuses the existing decoder code.

**What we'd learn.** Where on the temporal axis (within a single
forward pass) the geometric belief is overruled. Localising this is a
prerequisite for any meaningful intervention.

---

## Direction 2 — Causal intervention via the belief direction

**Question:** is the pre-reasoning belief *causally* responsible for
the action, or just an epiphenomenal correlate of geometry?

**Motivation.** All current evidence is correlational — a linear
decoder achieves X % on layer Y. The cost-IRL `θ` matrix is a literal
direction in 8640-dim space. If we *intervene* along that direction
during generation, two outcomes are diagnostic:

- **Steering works** → belief is on the causal path; the gap is a
  bottleneck we can close with a single rank-1 edit.
- **Steering doesn't work** → belief is a passive correlate; the
  decoder accuracy is real but not actionable for behaviour change.

**Concrete experiment.**
1. From the per-visit pre-reasoning probe, extract the per-action
   weight vectors `θ_a ∈ ℝ^{8640}`.
2. At inference time on a held-out trajectory, identify states where
   the agent is about to take a non-optimal action.
3. At layer 15, prompt-suffix positions: add a multiple `+α · θ_optimal`
   and subtract `α · θ_agent` (or just `+α · θ_optimal`) to the residual
   stream, then let the model continue.
4. Sweep `α ∈ {0, 0.5, 1, 2, 4, 8}`. Measure (a) action change rate,
   (b) shift in action-vs-optimal accuracy, (c) any reasoning-chain
   text changes.
5. Negative control: intervene with a random direction of equal norm.

**Cost.** Low. `nnsight` makes this a few-line patch; reuses the
already-trained probe. Wall-clock dominated by inference, ~minutes
per trajectory on a single GPU.

**What we'd learn.** Whether the geometric belief in pre-reasoning φ
is a load-bearing intermediate state in the agent's decision pipeline
or a side-channel readout. Either answer dictates the next step.

---

## Direction 3 — What replaces geometry at layer 23?

**Question:** what does the late residual stream encode that crowds
out the clean linear geometric signal?

**Motivation.** Belief accuracy drops 96.3 → 86.2 % on Seed12 between
layers 15 and 23 at the same token positions. The 10 pp of geometric
information must go somewhere — late layers are doing useful work,
not discarding features. Candidate hypotheses:

- **Action-token bias** — the residual is being rotated to align
  with the unembedding directions of LEFT / RIGHT / UP / DOWN tokens
  (in which case late layers ARE committing, just to the *wrong*
  action — and the post-reasoning result is a continuation of this).
- **Wall / obstacle encoding** — late residual prepares for "is this
  cell a wall?" prediction, sharpening obstacle features at the cost
  of goal-direction features.
- **Plan-state / loop-detection** — the model has noticed it's stuck
  in a loop and is encoding meta-state about its own trajectory.

**Concrete experiment.**
1. Train probes at layer 23 for each candidate target:
   - Action-token bias: project layer-23 residual onto the four
     action-token unembedding rows; see if this projection alone
     predicts the agent's action.
   - Wall map: train a probe to predict "is the cell at position P a
     wall?" for each position P relative to the agent.
   - Loop detection: probe for "have I visited this cell in this
     trajectory before?".
2. Compare each probe's accuracy at layer 15 vs layer 23.
3. Subtract: which feature *gains* the most from layer 15 → 23? That
   is the candidate that's being encoded *instead* of geometry.

**Cost.** Low–moderate. Each probe is a single linear layer; data
already in hand.

**What we'd learn.** A more mechanistic picture of what late
pre-reasoning layers are computing. If the answer is action-token
bias, we have evidence that the action commitment **does** happen at
prompt-suffix positions in the late stack — which would correct the
current finding.

---

## Direction 4 — Failure prediction from live belief-action divergence

**Question:** can we predict episode failure ahead of time from the
running belief-action gap?

**Motivation.** Per-trajectory data already hints at a clean
relationship: Seed12 trajectories with internal disagreement rate
≤ 5 % are 1–9 step successes; trajectories with rate ≥ 80 % all hit
the 30-step cap without reaching the goal. A real-time monitor
watching a single linear projection of layer-15 activations could
flag "this episode is going off the rails" several steps before the
failure manifests in behaviour.

**Concrete experiment.**
1. For each trajectory, compute the per-step belief-action
   disagreement (binary).
2. Train a sequence classifier `P(success | disagreement_history[:t])`
   for varying horizons `t`. Logistic regression on the cumulative
   rate is a fine baseline.
3. Measure: at step `t`, what fraction of about-to-fail trajectories
   does the monitor catch, and at what false-positive rate?
4. Compare against simpler baselines (number of revisits, current
   BFS-distance-to-goal, raw step count).

**Cost.** Trivial. Reuses existing belief predictions; it's a
sequence-modelling problem on ~10 k transitions.

**What we'd learn.** Whether the belief-action gap is a useful
runtime safety signal for grid-LLM agents — a practical payoff
independent of the deeper mechanistic questions.

---

## Direction 5 — Reasoning-induced miscalibration

**Question:** is chain-of-thought making the model *worse* at this
task?

**Motivation.** Strong version of the gap-during-reasoning finding:
CoT *actively corrupts* a correct geometric belief that's already
present pre-reasoning. GPT-OSS-20B exposes a `reasoning_effort`
setting (`low / medium / high`); the existing Seed12 trajectories
use `medium`.

**Concrete experiment.**
1. Re-collect trajectories from the same starting positions with
   `reasoning_effort ∈ {low, medium, high}` (and ideally `none`,
   if the harmony format allows it).
2. Plot:
   - Pre-reasoning belief accuracy (constant across runs — property
     of φ, not of CoT length).
   - Action-vs-optimal accuracy.
   - Belief-action gap.
   - Post-reasoning probe results (does post-reasoning φ commit
     more strongly with longer reasoning?).
3. Compute the *cost* of reasoning per problem: total tokens emitted
   vs accuracy gained.

**Cost.** Moderate. Requires re-running 79 episodes × 3 reasoning
levels = 237 episodes through `together_ai`, plus re-extracting
activations.

**What we'd learn.** Whether CoT is hurting or helping on geometric
tasks for this model. If the gap *grows* with reasoning effort, that
is empirical evidence that CoT can introduce errors absent in
prompt-only inference — a finding directly in tension with the
mainstream "more reasoning = better" narrative for arithmetic and
multi-step problems.

---

## Recommended order

```
2  Causal intervention   ← cheapest qualitative test
└─ if works → 5  Reasoning ablation     ← why is CoT overruling a good belief?
└─ if not   → 1  Probe through trace    ← where is the "real" decision made?
                3  Layer-23 mechanism   ← what crowds out geometry?

4  Failure prediction  ← independent, near-zero cost, run in parallel
```

**Direction 2 first.** A clear causal-intervention result either
hands us a steering tool (one rank-1 edit closes the gap) or rules
out the simplest interpretation of the existing evidence. Either
outcome dictates the rest of the agenda.

**Direction 4 in parallel.** It's almost free — the data is already
on disk and the analysis is a 50-line script. The result is useful
independent of the mechanistic story.

**Direction 1 next** if Direction 2 negative, or alongside Direction
5 if Direction 2 positive.

---

## Files this proposal builds on

- `results/seed12_agent_vs_optimal_results.md` — main Seed12 report,
  including the per-visit and layer-sweep extensions
- `results/two_path_agent_vs_optimal_results.md` — companion easy-task report
- `results/belief_action_gap.md` — the canonical belief-action gap
  analysis (single layer + layer sweep)
- `results/belief_action_gap_layers.md` — layer sweep stand-alone
- `run_belief_action_gap.py`, `run_belief_action_gap_layers.py` — the
  decoder pipeline that any new experiment can extend
- `cost_updated.ipynb` — the original linear `LinearCostIRL` from
  which the per-visit decoder is derived
