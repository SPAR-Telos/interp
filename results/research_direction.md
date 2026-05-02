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

## Direction 2 — Causal intervention, redesigned

**Question:** is the layer-15 pre-reasoning state representation
*causally* responsible for the agent's action — and if so, which
*part* of it (the geometric cost gradient, the discrete `has_key`
/ `door_open` flags, or the action-token preference at the output
position)?

### What the prior run did, and why it was a null

The first attempt
(`results/causal_intervention_results.md`) constructed a steering
vector from the per-action-heads probe — `v = α · (θ_â − θ_a*)`
where `â` was the agent's wrong action and `a*` was an optimal
action — and added it to the residual stream at layer 15, last 3
prompt-suffix tokens. Across α ∈ {0, 0.5, 1, 2, 4, 8} on the
prior dataset, belief-direction perturbations didn't flip the
action more than a random direction of equal norm (McNemar
p > 0.05 at every α). Some target actions were never reached
regardless of steering direction.

The run had three problems that need addressing in a redesign:

1. **Wrong probe form for the steering vector.** That probe used
   per-action heads `θ_a` on the *current state*'s φ. The
   canonical Cost-IRL formulation we now use is single-θ
   next-state-averaged: `cost(s) = θᵀφ(s)`,
   `P(a|s) = softmax(−β·cost(f(s, a)))`. Under that formulation
   the "belief direction" isn't a per-action vector at all — it's
   a single direction in φ-space whose dot product with φ
   correlates with goal-distance.
2. **Wrong state representation.** The optimal labels then were
   position-only; the steering target was therefore noisy. With
   state-space BFS the optimal action at `(s, F, F)` differs from
   `(s, T, F)` differs from `(s, T, T)`, so the steering target
   needs to be conditioned on the agent's full state.
3. **Wrong intervention site (probably).** Pre-reasoning is
   hundreds of tokens upstream of the action-emission point; CoT
   re-derives the geometry from the unchanged input tokens. A
   small rank-1 edit before the trace begins is easily drowned
   out by the trace itself. The post-reasoning per-visit decoder
   reads the action off the output positions at >90 % accuracy —
   that's where the action lives.

### Redesigned experiment — three sub-tests

Each sub-test isolates a different candidate causal mediator.
They are independent and can be run separately. Run on
`fourroom_episode_sweep_T0` (all three variants).

#### 2a. Variant-transplant intervention *(strongest test, naturalistic)*

For each (col, row) cell visited under multiple variants, we have
real activations under different `(has_key, door_open)`
configurations from real trajectories. These activations differ
*only* in the env's underlying state — same model, same
prompt-template, same position.

**Procedure.** At decision time on a `K0D0` agent at cell `(c, r)`,
replace the layer-15 prompt-suffix activations with the
corresponding `K1D1` activations at `(c, r)` (averaged across all
`K1D1` visits to that cell, or sampled). Let the model continue.
Does the `K0D0` agent now act like a `K1D1` agent — i.e., skip
the key detour and head straight toward the door / goal?

**Mirror.** Reverse direction: transplant `K0D0` activations into
a `K1D1` agent's forward pass. Does the `K1D1` agent now detour
to the key?

**Why this is stronger than synthetic steering:**
- We're injecting *real* activation patterns the network actually
  produces, so we don't need to construct a probe-derived
  direction. No question of "is this the right direction" — we
  use what the network does on the other side.
- Tests an inherently meaningful counterfactual: "if my world
  state were different, would I act differently?".
- The control is automatic: same-variant transplant (e.g.,
  `K0D0` → `K0D0` from a different visit) should be a no-op.
  Cross-variant transplant should differ.

**What we learn.** Whether the layer-15 prompt-suffix activations
*encoding the variant* are causally read out by the action
selection. A positive result is direct evidence that the model's
internal state belief drives action choice. A null result means
the variant identity at this layer/position is a passive readout
and the action is decided elsewhere (downstream layers, output
positions, or in the reasoning trace).

#### 2b. Late-position intervention at the action-emission token

The previous run intervened at prompt-suffix (before the
reasoning trace). The post-reasoning per-visit decoder finds the
agent's emitted action at >90 % accuracy from the *output* token
positions. So the action is decided *late*. Intervene there.

**Procedure.** Identify the output-token position immediately
before the action JSON value (`{"action": "X"}`'s `X`). At layer
N (sweep over 7 / 15 / 23), add a steering vector. Two candidate
vectors:
- The unembedding direction of the optimal-action token minus
  the agent's recorded-action token.
- The cost-IRL θ direction (geometric goal direction in φ-space).

Sweep α and a random control as before. Does the emitted action
flip more than the prompt-suffix experiment showed?

**What we learn.** Whether the residual stream at the late
output positions *is* the action commitment, vs the action having
been committed even earlier (in the reasoning text) and the output
position is just transcribing. A clean positive at late positions
+ null at prompt-suffix narrows the commitment to the reasoning
chain itself — which then makes Direction 1 (probe through the
trace) the natural next step.

#### 2c. Flag-direction steering (mechanistic, follows Direction 6)

If Direction 6 finds that φ encodes `(has_key, door_open)` along
linearly separable directions (call them `θ_key`, `θ_door`),
then we have a clean mechanistic test: steer along `θ_key` to
make a `K0D0` agent's residual stream "look like" `K1D0`. Does
the agent's action change accordingly?

**Procedure.** Train binary linear classifiers for `has_key` and
`door_open` at layer 15. At runtime on a `K0D0` agent, add
`α · θ_key` (the direction along which has_key=True is more
likely under the classifier) to the residual stream at layer 15
prompt-suffix.

This is a *mechanism-decomposed* version of 2a: 2a transplants the
whole activation pattern; 2c isolates the contribution of the
`has_key` feature direction. If 2a flips the action but 2c
doesn't, the flag is encoded but not the only causal mediator —
something else (geometry, sub-goal pointer) is also load-bearing.

### Cost

- 2a (transplant): low. ~5 lines of nnsight to replace activations
  at chosen positions. Re-uses already-extracted activations.
  Runtime ~minutes per target trajectory.
- 2b (late-position steering): low–moderate. Needs to identify
  the output-action-token position per record (offset known from
  the data); then standard hook-and-add.
- 2c (flag-direction steering): trivial *after* Direction 6 runs.
  Reuses the binary classifiers it trains.

### Recommendation

Run **2a first** — the transplant experiment is the most
informative-per-dollar test, and a positive result would settle
the causal question without further work. Direction 1 (probe
through the trace) and 2b (late-position intervention) are
complementary: if 2a is null and 2b is positive, that pins the
commitment to the late stack at output positions and makes
Direction 1 the obvious follow-up to localise *where in the
trace* the geometric belief gets translated into the late-stack
action representation.

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
2a  Variant-transplant intervention   ← the redesigned causal test; cheap, naturalistic
6   Flags vs sub-goal probe           ← parallel; cheap mechanistic question
1   Probe through reasoning chain     ← if 2a is null, localise where belief is lost
2b  Late-position intervention        ← follow-up if 1 shows late commitment
3   Layer-23 mechanism                ← after 1 and 6 give context for what's at layer 23
2c  Flag-direction steering           ← only if 6 gives clean linear flag directions
4   Failure prediction                ← independent, near-zero cost, run in parallel
5   Reasoning-effort sweep            ← moderate cost, run last
```

**Direction 2a first.** The variant-transplant experiment is the
single highest-leverage redesign of the original null result: it
uses real activations (no synthesised steering vector), tests a
cleanly-posed counterfactual ("if my world were different, would
I act differently?"), and a positive result would settle the
causal question. Cost is roughly the same as the prior
intervention run but interpretable end-to-end.

**Direction 6 in parallel.** ~50 lines of code on data already on
disk; resolves a clean mechanistic question and feeds into 2c.

**Direction 1 next** if 2a is null — localising *where* the
belief gets corrupted is the next informative thing.

**Direction 2b** as the natural follow-up to 1: if Direction 1
shows belief surviving most of the trace and only collapsing at
the output, late-position steering is the right place to test
direct causal effect.

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
