# Research Directions: Belief-Action Gap in GPT-OSS-20B Grid Agents

## Background — what the existing experiments establish

On a 9 × 9 fixed-grid doorkey task (`fourroom_episode_sweep_T0`,
111 trajectories spanning three flag variants `K0D0` / `K1D0` /
`K1D1`) the agent (GPT-OSS-20B with chain-of-thought) emits a
BFS-optimal action only **32.5 %** of the time — *worse* than the
54.8 % majority baseline (UP), because the agent over-emits UP
while the optimal under the env's actual dynamics is mostly DOWN
(73 % of optimal labels) or LEFT (toward the key). A **single
linear cost-IRL probe** on the layer-15 pre-reasoning residual
stream (last 3 prompt-suffix tokens, 8640-dim φ) picks the
BFS-optimal action with **96.0 % accuracy** (99.2 % set-membership)
on held-out visits. That's a **+50.6 pp belief-action gap**.

Two algorithmic refinements vs earlier position-only analyses:

1. **State is the 4-tuple** `(col, row, carrying_key, door_open)`,
   not just `(col, row)`. The φ-table averages activations per
   state, and the same position with a different flag combination
   counts as a different state. fourroom has 111 such states across
   39 walkable cells.
2. **BFS for optimal-action labels operates over the full state
   space** using a transition function that models walls/bounds,
   key pickup on contact, and locked-door auto-open with the key.
   A `K0D0` agent in the lower room therefore has a *much* longer
   shortest-path-to-goal (must detour to the key, then to the door)
   than the same position with the key already collected — and its
   optimal action points toward the key, not toward the goal.
   Position-only BFS gave the same distance to both, mislabelling
   `K0D0` records.

Per-visit MLP decoders on the same activations land within ±2 pp
of the linear probe (Model A 0.43, Model B 0.96 on prompt-suffix);
post-reasoning activations are essentially tied with pre-reasoning
once visits are averaged. The visit-averaged φ washes out the
"policy-commitment" signal that distinguishes pre vs post.

Per-action breakdown is striking:

| action | Model A acc | Model B acc | n_test |
|---|---:|---:|---:|
| LEFT  | 0.764 | 0.958 |  89 / 72  |
| RIGHT | 0.810 | 0.853 | 126 / 156 |
| UP    | **0.228** | 0.778 | 527 / 27 |
| DOWN  | 0.717 | 0.992 | 180 / 667 |

Model A is collapsed on UP (23 %) — the activations encode the
optimal direction, which is rarely UP under the state-space BFS,
whereas the agent emits UP 55 % of the time. The activations
"know" what the right move is conditional on env state; the agent
chooses something else.

This document lists the most fruitful follow-ups, ordered by
priority. **Direction 2 has already been run on a prior dataset
with a null result; the others remain open and most should be run
on `fourroom_episode_sweep_T0`.**

---

## Direction 1 — Probe through the reasoning chain

**Question:** at what *point in the chain-of-thought* does the
right belief get lost?

**Motivation.** The two existing probe positions are extremes:
prompt-suffix (last 3 tokens before reasoning, belief acc ~96 %)
and output (last few tokens after reasoning, the per-visit decoder
reads off the agent's action with high accuracy). The reasoning
trace itself is hundreds of tokens long. If the same belief
decoder is run at intermediate trace positions, we can plot
belief-vs-optimal accuracy as a function of position-in-trace.
Three plausible shapes:

- **Monotone decay** — belief gradually corrupted as reasoning
  progresses → CoT accumulates errors token by token.
- **Cliff** — belief stays high until some specific point, then
  drops → there's a "moment of commitment" we can localise.
- **Plateau then crash at output** — geometric belief is preserved
  through the whole reasoning trace and only the final projection
  to the action token overrules it → the bug is in the unembedding
  / output-prediction step, not in the reasoning content.

Each shape implies a different intervention target.

**With the new state-space BFS this is even more interesting**:
distinct flag variants (`K0D0` / `K1D0` / `K1D1`) have very
different optimal actions, so the question becomes "does the model
keep track of `has_key` through the trace, or does CoT drift it
toward a position-only goal-direction belief?"

**Concrete experiment.**
1. Re-extract activations at a sample of reasoning-trace positions
   (e.g. every 20 tokens) for layers 7 / 15 / 23 on the
   `fourroom_episode_sweep_T0` trajectories.
2. Run the same per-visit linear probe with BFS-optimal labels
   (state-space) at each position.
3. Plot belief acc, belief-action agreement, and the gap as a
   function of position-in-trace, **stratified by variant**
   (`K0D0` is harder than `K1D1` and the trace might differ).
4. Bonus: stratify by trajectory outcome (success vs cap-hit) — if
   a cliff appears in failures but not successes, the location is
   the *failure-causing event* in the trace.

**Cost.** Moderate. Re-running `gather_activations` with a wider
token window for the existing 111 trajectories. ~1–2 h GPU + disk.

**What we'd learn.** Where on the temporal axis (within a single
forward pass) the geometric belief is overruled. Localising this
is a prerequisite for any meaningful intervention.

---

## Direction 2 — Causal intervention via the belief direction *(already run on a prior dataset — null)*

**Question:** is the pre-reasoning belief *causally* responsible
for the action, or just an epiphenomenal correlate of geometry?

**Status.** Run with the prior dataset
(`results/causal_intervention_results.md`). **Null result**:
belief-direction perturbation at layer-15 prompt-suffix didn't
flip the action significantly more than a random direction of the
same norm at any α ∈ {0, 0.5, 1, 2, 4, 8}. McNemar p > 0.05 at
every α. The model's output prior dominated: some target actions
were never reached regardless of steering direction (e.g.
`belief=DOWN` flipped 0 / 23 cases).

That run used a problematic dataset (mislabelled flags, only one
variant per agent, position-only BFS labels). The current fourroom
data is cleaner; a re-run is defensible.

**Re-run on fourroom?** Same script with the new dataset and
state-space BFS; budget is the same as the prior run. **Demote
from "first" to "after Direction 1"** — without knowing where the
belief gets lost, intervening at the prompt-suffix is shooting in
the dark.

---

## Direction 3 — What replaces geometry at layer 23?

**Question:** what does the late residual stream encode that
crowds out the clean linear geometric signal?

**Motivation.** Earlier (position-only-BFS) layer sweeps showed
belief accuracy dropping ~10 pp from layer 15 to layer 23 at the
same token positions. Whether this drop reproduces on fourroom
with state-space BFS is **the prerequisite for this direction**.

Pre-fourroom hypotheses for what's encoded at layer 23 instead:

- **Action-token bias** — residual being rotated to align with
  unembedding directions of LEFT / RIGHT / UP / DOWN tokens (i.e.
  late layers ARE committing, but to the *wrong* action — and the
  post-reasoning decoder result is a continuation of this).
- **Wall / obstacle encoding** — late residual prepares for
  "is this cell a wall?" prediction.
- **Plan-state / loop-detection** — the model has noticed it's
  stuck in a loop and is encoding meta-state about its own
  trajectory.
- **Variant / phase tracking** — *new hypothesis*: late layers
  switch from "where is the goal geometrically" to "what task
  phase am I in (find-key / approach-door / approach-goal)". This
  would explain why position-conditional belief drops while the
  action is still being decided correctly.

**Concrete experiment.**
1. **First** re-run the layer sweep on fourroom with the
   state-space BFS labels: does the layer-15 → 23 gap persist?
2. Train probes at layer 23 for each candidate target (action-token
   bias, wall map, loop detection, phase indicator).
3. Compare each probe's accuracy at layer 15 vs layer 23.
4. Subtract: which feature *gains* the most from layer 15 → 23?
   That is the candidate being encoded *instead* of geometry.

**Cost.** Low–moderate. Each probe is a single linear layer; data
in hand once the layer sweep runs.

**What we'd learn.** A more mechanistic picture of what late
pre-reasoning layers are computing.

---

## Direction 4 — Failure prediction from live belief-action divergence

**Question:** can we predict episode failure ahead of time from
the running belief-action gap?

**Motivation.** The belief-action gap is enormous (+51 pp on
fourroom) and the per-trajectory disagreement rate is the natural
real-time monitor. fourroom's 3 variants give a stronger test:
`K0D0` agents face the hardest task (must find key first), so the
disagreement-rate-vs-failure relationship can be characterised per
variant.

**Concrete experiment.**
1. For each trajectory, compute the per-step belief-action
   disagreement (binary).
2. Train a sequence classifier
   `P(success | disagreement_history[:t])` for varying horizons
   `t`. Logistic regression on the cumulative rate is a fine
   baseline.
3. Measure: at step `t`, what fraction of about-to-fail
   trajectories does the monitor catch, at what false-positive
   rate? Stratify by variant.
4. Compare against simpler baselines (number of revisits, current
   BFS-distance-to-goal, raw step count).

**Cost.** Trivial. Reuses existing belief predictions; sequence
modelling on ~4 600 transitions.

**What we'd learn.** Whether the belief-action gap is a useful
runtime safety signal for grid-LLM agents — practical payoff
independent of the deeper mechanistic questions.

---

## Direction 5 — Reasoning-induced miscalibration

**Question:** is chain-of-thought making the model *worse* at this
task?

**Motivation.** Strong version of the gap-during-reasoning
finding: CoT *actively corrupts* a correct geometric belief that's
already present pre-reasoning. GPT-OSS-20B exposes a
`reasoning_effort` setting (`low / medium / high`); the existing
fourroom trajectories use `medium`.

**Concrete experiment.**
1. Re-collect trajectories from the same starting positions and
   variants with `reasoning_effort ∈ {low, medium, high}` (and
   ideally `none` if the harmony format allows).
2. Plot:
   - Pre-reasoning belief accuracy (should be approximately
     constant — property of φ, not of CoT length).
   - Action-vs-optimal accuracy.
   - Belief-action gap.
   - Post-reasoning probe results (does post-reasoning φ commit
     more strongly with longer reasoning?).
3. Compute the *cost* of reasoning per problem: total tokens
   emitted vs accuracy gained.

**Cost.** Moderate. Requires re-running 111 episodes × 3
reasoning levels, plus re-extracting activations.

**What we'd learn.** Whether CoT is hurting on geometric tasks
for this model. If the gap *grows* with reasoning effort, that's
empirical evidence that CoT can introduce errors absent in
prompt-only inference — directly in tension with the mainstream
"more reasoning = better" narrative.

---

## Direction 6 — Does φ encode (has_key, door_open)?

**Question:** the state-space BFS makes the optimal action depend
critically on `(has_key, door_open)`. A `K0D0` agent at (3, 6)
should head LEFT/UP toward the key, while a `K1D1` agent at the
same position should head LEFT/UP toward the door + goal. The
linear probe achieves 96 % at distinguishing these. Is that
because φ encodes the flags directly, or because it encodes
"distance to next sub-goal" which already wraps the flag
information?

**Motivation.** This is the cleanest mechanistic question raised
by the new state-space algorithm. Two separable hypotheses:

- **Flags-then-geometry**: φ has linear directions for `has_key`
  and `door_open`, and geometric directions for goal/key/door.
  Optimal action is decoded by combining them.
- **Direct sub-goal direction**: φ encodes "direction to current
  sub-goal" as a single feature, with the sub-goal already chosen
  internally. Flags don't appear as separable directions.

**Concrete experiment.**
1. Train binary linear probes at layer 15 for `has_key` and
   `door_open`. Hold-out trajectories.
2. Train a 4-class probe for the *current sub-goal* (key / door /
   goal / done).
3. Compare accuracies; if flags are recoverable at near-100 %,
   they're encoded directly. If sub-goal is also recoverable at
   near-100 %, that's the "more compact" representation.
4. Bonus: project the cost-IRL θ onto each of these directions —
   if θ aligns mostly with the sub-goal direction, that's the
   primary signal.

**Cost.** Trivial. Reuses existing activations and trajectory
flags.

**What we'd learn.** Whether the model's grid representation is
factored (flags + geometry) or compressed (sub-goal pointer).
Bears directly on Direction 3 (what changes layer 15 → 23?) — if
the model represents sub-goals, the layer-23 shift might be
"refining the sub-goal pointer" rather than "encoding geometry".

---

## Recommended order

```
1  Probe through reasoning chain     ← localise where belief is lost
└─ then  3  Layer-23 mechanism       ← what crowds out geometry
└─ or    2  Causal intervention      ← only if Direction 1 reveals a localisable target

6  Flags vs sub-goal              ← cheap mechanistic question, run anytime
4  Failure prediction             ← independent, near-zero cost, run in parallel
5  Reasoning-effort sweep         ← moderate cost, run last
```

**Direction 1 first.** With Direction 2 already null on the prior
dataset, locating *where* the belief gets corrupted is the next
informative thing. The intervention experiment makes most sense
once we know the right place to intervene.

**Direction 6 in parallel.** ~50 lines of code on data already on
disk; resolves a cleanly-posed mechanistic question.

**Direction 4 in parallel** for the practical-payoff thread.

**Direction 5 last** — it's the most expensive (re-collecting
trajectories at three reasoning levels) and depends on the
qualitative shape of Direction 1's result.

---

## Files this proposal builds on

- `results/fourroom_episode_sweep_T0_agent_vs_optimal_results.md`
  — main report on the current dataset
- `results/two_path_no_key_T0_agent_vs_optimal_extended.{json,pkl}`
  — companion easy-task data
- `results/causal_intervention_results.md` — Direction 2 (null
  result, prior dataset)
- `run_agent_vs_optimal_extended.py` — the cost-IRL pipeline
  (state-space BFS + auto-open transition)
- `run_belief_action_gap.py`, `run_belief_action_gap_layers.py` —
  the per-visit decoder pipeline that any new experiment can
  extend
- `cost_updated.ipynb` — the original linear `LinearCostIRL`
  reference implementation
