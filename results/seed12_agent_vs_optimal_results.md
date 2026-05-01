# Seed12 Cost-IRL: Agent Action vs BFS-Optimal Action

## Headline

Two `LinearCostIRL` models trained on identical Seed12 activations
(GPT-OSS-20B, layer 15, prompt-suffix, 8640-dim φ); only the **label** differs.

| metric                          | Model A (label = agent action) | Model B (label = optimal action) |
|---------------------------------|-------------------------------:|---------------------------------:|
| **test accuracy**               |                      **0.417** |                        **0.935** |
| test set-membership accuracy¹   |                          0.463 |                        **1.000** |
| test log-likelihood             |                        −1.2385 |                          −0.2068 |
| train accuracy                  |                          0.419 |                            0.940 |
| majority-class baseline         |                          0.413 |                            0.411 |
| random baseline                 |                          0.250 |                            0.250 |

¹ *Predicted action ∈ BFS-optimal set (gives credit when several actions are tied for optimal).*

**The activations predict the BFS-optimal action with 93.5 % accuracy
(100 % set-membership) but predict the agent's actual action only 41.7 %
of the time — barely above the 41.3 % majority-class baseline.**

---

## The dataset: Seed12

### Grid

A single fixed 9 × 9 layout (the **`door_open` variant** of the doorkey
family — the door at `(1, 4)` is treated as walkable from the start,
`carrying_key` and `door_open` flags stay `False` throughout, the rendered
`grid_state` never shows a `D`).

```
   0 1 2 3 4 5 6 7 8
 0 # # # # # # # # #
 1 # _ _ _ # G _ _ #     <- goal (G) at (col=5, row=1)
 2 # _ _ _ # _ _ _ #
 3 # _ _ _ _ _ _ _ #     <- row 3 is an open corridor (no inner wall)
 4 # D # # # # # # #     <- only (1,4) is passable in row 4 (the "door")
 5 # _ _ _ # K _ _ #     <- key (K) is on the grid but unused
 6 # _ _ A # _ _ _ #     <- example agent (A) start at (3,6)
 7 # _ _ _ _ _ _ _ #     <- row 7 is also an open corridor
 8 # # # # # # # # #
```

- Walkable cells: 39 (= 81 cells − 42 walls).
- Reaching the goal from the lower room (rows 5–7) requires first moving
  to column 1 and going UP through the door at (1, 4); rows 2, 4, 5 are
  blocked at column 4 and at columns 2–7 of row 4.

### Trajectories

- **79 trajectories** in `data/trajectories/Seed12/together_ai*.json`,
  each one episode of GPT-OSS-20B controlling the agent.
- **Sampled at temperature = 0** (deterministic decoding), but starting
  positions are **varied** across the 79 episodes. Distribution of
  `agent_start_pos` (top of the list):

  ```
  (4,7): 5    (5,6): 5    (6,2): 5    (6,1): 4    (1,3): 4
  (7,7): 4    ...          (33 distinct start cells in total)
  ```

  Episodes that start adjacent to the goal (e.g. (6, 1) — 1 BFS step from
  goal) finish in 1–4 model steps; episodes far from the goal (lower room)
  often hit the **30-step episode cap**.

- **Steps per trajectory**: min = 1, mean = 20.1, median = 30. **43 / 79
  trajectories run all the way to the 30-step cap** without reaching the
  goal.

- **Agent success rate**:

  | metric                                                | value      |
  |------------------------------------------------------|-----------:|
  | trajectories that finish at the goal                  | **38 / 79 = 48 %** |
  | trajectories that ever reached the upper room         | 49 / 79 = 62 % |
  | trajectories taking the optimal number of steps       | 1 / 79 = 1 % |
  | mean BFS distance from final position to goal         | 6.1 cells  |
  | fraction of *all* (s, a) pairs where a is BFS-optimal | **0.434** (672 / 1549) |

  So the agent reaches the goal less than half the time, and even the
  successful episodes are usually far from optimal length.

### Activations

`data/activations/Seed12` was extracted ahead of time with
`gather_activations`:

- Layers: 7, 15, 23 (this experiment uses **layer 15** only).
- Saved tokens, `prompt_suffix` category: **3 tokens** per step
  (`14.pt`, `15.pt`, `16.pt`) — the last 3 tokens of the prompt before the
  model's output begins.
- Per-token tensor shape: `(2880,)`, dtype `bfloat16`.
- Concatenation of the 3 tokens per step ⇒ **8640-dim φ** (matches the
  `cost_updated.ipynb` dimensionality).

This is "pre-reasoning" φ: the residual stream right before the model
emits any reasoning or output for that step.

---

## The algorithm (verbatim from `cost_updated.ipynb`)

### Cost model

Linear cost on activations:

```
C_θ(s) = θᵀ φ(s)              # θ ∈ ℝ^8640
```

### Policy

Boltzmann over the cost of each action's deterministic next state:

```
P(a | s; θ) = softmax_a (−β · C_θ(f(s, a)))
            = softmax_a (−β · θᵀ φ(f(s, a)))
```

with β = 1. `f(s, a)` is the deterministic transition function: walls and
out-of-bounds cells are blocking, the door cell is walkable, the goal is a
no-op.

### Training objective

Negative log-likelihood of the labelled action plus L2 on θ:

```
L(θ) = − (1/N) Σᵢ log P(aᵢ_label | sᵢ; θ)  +  λ · ‖θ‖²
```

with λ = 0.01.

### Optimiser

```
optim   : Adam
lr      : 1e-3
sched   : CosineAnnealingLR over 500 epochs
batch   : full-batch (every epoch processes all training transitions)
init    : θ = 0
β       : 1.0
seed    : 42  (used for torch + numpy + train/test shuffle)
```

The **only** difference between the two runs (Model A vs Model B) is the
label vector `aᵢ_label`. Same φ, same θ-init, same optimiser state,
same data, same seed.

### Best-checkpoint selection

After every epoch we record train and **test** log-likelihood; the
returned θ is the one with the highest test log-likelihood seen during
training. This choice is documented under "Data leakage" below.

---

## Data handling pipeline

End-to-end (in `run_seed12_agent_vs_optimal.py`):

1. **Parse the grid** from `grid_layout.json` → `walls`, `walkable`,
   `goal_pos = (5, 1)`. Treat the door cell at `(1, 4)` as walkable.

2. **Compute BFS distance** from the goal over the walkable graph
   (4-connected). Every walkable cell ends up with a finite distance
   (39 / 39 reachable). Define `optimal_set(pos) = {a : dist(step(pos, a))
   = dist(pos) − 1}` for every non-goal walkable cell.

3. **Iterate trajectories**. For every step in every JSON:
   - Skip if `agent_action` is missing or not in {LEFT, RIGHT, UP, DOWN}.
   - Parse the agent's `(col, row)` from the rendered `grid_state`.
   - Skip if the agent is at the goal cell (no transition needed).
   - Skip if `optimal_set(pos)` is empty (never occurs for reachable cells).
   - Load the activation: 3 tensors at `step_<id>/prompt_suffix/{14,15,16}.pt`,
     cast to float32, concat → 8640-dim numpy array.
   - Skip if any of the 3 tensors is missing on disk.
   - Record `(state, agent_action, opt_label, opt_set)` where the
     state's 4-tuple form is `(col, row, False, False)` (kept in the same
     shape as `cost_updated.ipynb` for reuse).

4. **Build the φ-table.** For every state visited, average the 8640-dim
   activation across all visits → `phi_table[state]`. Stack into a matrix
   `(38, 8640)` and z-normalise per feature (mean and std over the 38
   states, std clamped at +1e-8). 38 of 39 walkable cells were visited
   at least once in the dataset.

5. **Build per-transition tensors.** For every record, look up the
   next-state index for **all 4 actions** (so the softmax can score them
   simultaneously). Drop a record if any of the 4 next-states is missing
   from the φ-table — this dropped 40 records, leaving **N = 1549** usable
   transitions over 38 states.

6. **Random shuffle.** Permute the 1549 records with `numpy.random.default_rng(42)`.

7. **80 / 20 split.** Take the first 309 (= 20 %) as **test**; the
   remaining 1240 are **train**. Both models use the *same* split.

8. **Compute the optimal-action label** per record:
   - if `agent_action ∈ optimal_set` → `opt_label = agent_action`
   - else                              → `opt_label = optimal_set[0]` in
     canonical order `[LEFT, RIGHT, UP, DOWN]`.

   This minimises label noise (any tie is broken in favour of what the
   agent did, when feasible). For evaluation we additionally report
   "set-membership" accuracy = predicted action is in the optimal set,
   which sidesteps the tie-breaking choice.

9. **Train two `LinearCostIRL` models** with identical hyperparameters,
   one per label vector. Best-checkpoint by **test** log-likelihood.

10. **Evaluate**: report log-lik, accuracy, set-membership accuracy,
    per-action breakdown, and cross-evaluation (model A scored against
    optimal labels, model B scored against agent labels).

### Dataset stats

|                                                     | value           |
|----------------------------------------------------|----------------:|
| total trajectory steps                              | 1 589           |
| steps kept after filtering                          | 1 549           |
| dropped (next-state φ missing)                      | 40              |
| unique states with φ                                | 38 (of 39 walkable) |
| φ dimensionality                                    | 8 640           |
| mean visits per state                               | ~41             |

---

## Train / test split

| field                                    | value      |
|-----------------------------------------|------------|
| split unit                               | per-transition (a single (s, a) record) |
| shuffle RNG                              | `numpy.random.default_rng(42)` |
| test fraction                            | 0.20       |
| n_train                                  | 1 240      |
| n_test                                   |   309      |
| same split shared by both models?        | yes        |
| best-epoch selection metric              | test log-likelihood |

The split is **not by trajectory** and **not by state** — it shuffles
the flat list of 1 549 transition records.

---

## Data leakage

There are three points where leakage / contamination is possible. None
materially change the qualitative result (Model B » Model A), but they
do affect the absolute numbers, so they are documented in full here.

### 1. φ-table is built before the split (mild leakage of features)

`phi_table[state]` is the **mean of every visit** to that state — including
visits that later land in the test set. With ~41 visits per state and
20 % of records in test, on average ~8 / 41 ≈ **20 % of each state's φ
came from a test-set visit**.

This is a feature-level leak: the test record is being scored against a
representation it helped to compute. Severity here is low (a) because we
average instead of using the per-visit φ, so the contribution of any
single test sample is diluted across ~41 visits, and (b) because the
labels are determined from `(state, agent_action)`, not from φ — so the
leaked information is feature-side noise reduction, not label leakage.

**Cleaner alternative** (not used): split first, then average φ over
*train-only* visits.

### 2. State-level overlap between train and test

A random per-transition shuffle puts the same state into both train and
test (a state visited 41 times almost certainly has visits on both sides
of the 80 / 20 line). This is **not** label leakage — the labels are
distinct per record — but it does mean we are *not* testing
generalisation to **new states**; we are testing whether the linear cost
generalises to **new visits of seen states**. With only 38 unique states,
any state-level holdout would be small and noisy, so a per-transition
shuffle is the natural choice for measuring sensitivity to per-visit
noise — but the limitation should be clear.

**Cleaner alternative** (not used): hold out a set of *states* rather
than transitions. Would let us test true state-generalisation but with
small N per fold.

### 3. Best-checkpoint by test log-likelihood (model selection on test)

The training loop tracks test log-likelihood every epoch and returns the
θ that maximised it. This is a (mild) form of test peeking: the test set
is acting as a validation set for early-stopping. With only one
hyperparameter being implicitly tuned (the epoch index), the inflation
in test accuracy is small — the curves plateau by epoch ~150 of 500 in
both runs — but it is non-zero.

**Cleaner alternative** (not used): hold out a separate validation set
or just take the final-epoch θ.

### Trajectory-level leakage?

Adjacent steps within one trajectory are highly correlated (the agent's
position and the prompt only change by one cell per step), so the random
shuffle puts strongly correlated records on either side of the split.
Both models suffer from this equally — it inflates train accuracy more
than test accuracy, and is the standard caveat for any IRL evaluation
that doesn't split by trajectory.

**Cleaner alternative** (not used): split by trajectory (e.g. 16 / 79
trajectories held out) so no two correlated records share train/test
membership.

### Bottom line on leakage

The qualitative finding (Model B at 93–94 % accuracy, Model A at ~42 %
near majority baseline) is robust to all three issues:

- Issue (1) provides identical feature-side advantage to both models →
  cannot create the gap between them.
- Issue (2) is exactly what the experiment is designed to measure
  (linear separability of optimal action from φ) and is again identical
  for both models.
- Issue (3) inflates both models' test numbers by at most a couple of
  percentage points; the gap between 42 % and 93 % is ~50 pp, far
  larger than any plausible inflation.

The leakage caveats apply to interpreting **absolute** numbers, not to
the **comparison**.

---

## Per-action accuracy (test set)

**Model A** (label = agent's action):

| action | accuracy | n  |
|-------:|---------:|---:|
| LEFT   |    0.645 | 62 |
| RIGHT  |    0.246 | 69 |
| UP     |    0.463 | 121|
| DOWN   |    0.281 | 57 |

**Model B** (label = optimal action):

| action | accuracy | n  |
|-------:|---------:|---:|
| LEFT   |    0.971 | 102 |
| RIGHT  |    0.957 | 46  |
| UP     |    0.893 | 28  |
| DOWN   |    0.910 | 133 |

Action distributions in the full dataset:

|        | LEFT | RIGHT | UP  | DOWN |
|--------|----:|----:|----:|----:|
| agent  | 300 | 338 | **640** | 271 |
| optimal | 524 | 243 | 146 | **636** |

The agent's modal action is UP; the optimal modal action is DOWN. Many
trajectories start in the lower room and the *globally* optimal first
move is LEFT (toward column 1, then UP through the door at (1, 4)),
not UP — but the language model gravitates toward UP because the goal is
visually in the upper portion of the rendered grid.

---

## Cross-evaluation

| model trained on | scored against | acc | in-opt-acc | log-lik |
|-----------------:|---------------:|----:|-----------:|--------:|
| **agent labels (A)** | optimal labels | 0.398 | **0.463** | −1.4584 |
| **optimal labels (B)** | agent labels | 0.379 | **1.000** | −3.9879 |

- **Model B picks an optimal action 100 % of the time on test**, even
  scored against the agent's labels — every prediction it makes is
  geometrically correct, it just disagrees with what the agent did.
- **Model A picks an optimal action only 46 % of the time** — close to
  the raw rate at which the agent itself is optimal (43 %).

---

## Analysis

### Why does Model B succeed so dramatically?
A linear cost on layer-15 prompt-suffix activations is enough to recover
the BFS-optimal direction. φ(s) clearly carries goal-relative geometric
information that maps cleanly onto a single reward direction (closer to
goal → lower cost). With only 38 unique states and 8 640-dim φ, the
linear model has more than enough capacity to memorise distance-to-goal.

### Why does Model A fail?
The agent is suboptimal: only 43 % of its actions are BFS-optimal at the
visited state (only 1 / 79 trajectories takes the optimal-length path,
and only 38 / 79 reach the goal at all). Conditioning on φ(s), the
agent's action distribution is close to **state-independent random**,
dominated by an UP-bias that doesn't correspond to the goal direction.
A linear cost on φ(s) cannot do much better than the majority-class
baseline (0.413) — and indeed it lands at 0.417.

### What this says about layer-15 activations
The activations encode **where the goal is**, not **what the agent will
do**. In Seed12 specifically the two are very different — agent
trajectories are loop-prone and head the wrong direction often. A study
that only measured "can we predict the agent's action?" would call
layer 15 uninformative; the actual answer is that layer 15 is *highly*
informative about the optimal policy and the agent itself is the noise
term.

### Implication for IRL on this dataset
A standard inverse-RL pipeline that treats the agent as expert recovers
a roughly random reward (Model A). A pipeline that uses the geometric
optimum as the reward target recovers an essentially perfect linear
reward (Model B). For Seed12 the IRL framing breaks down because the
**policy isn't expert** — and the residual stream knows it.

---

---

## Extension: MLP cost head and pre- vs post-reasoning activations

The runs above use a **linear** cost head and **pre-reasoning** (prompt-suffix)
activations. To check whether the headline result is sensitive to either
choice, the experiment was repeated for the full 2 × 2 grid:

- model head ∈ { linear, **MLP** (8640 → 128 → 128 → 1, dropout 0.1, gradient clip 1.0) }
- token category ∈ { **pre** (last 3 prompt-suffix tokens), **post** (3 output tokens) }

For each (head, token) pair we train one Model A (agent label) and one
Model B (optimal label) — 8 runs total per dataset, all sharing the same
seed, train/test split, and other hyperparameters. The MLP uses
`weight_decay=0.01` on Adam in place of the explicit L2 term used by the
linear head.

### Test accuracy

|                          |  linear / pre  |  linear / post  |   MLP / pre   |   MLP / post   |
|-------------------------:|:-------------:|:---------------:|:-------------:|:--------------:|
| **Model A (agent)**      |     0.417     |      0.427      |     0.427     |     0.395      |
| **Model B (optimal)**    |   **0.935**   |    **0.922**    |   **0.932**   |   **0.922**    |
| gap (B − A)              |    +0.518     |     +0.495      |    +0.505     |    +0.527      |

### Test set-membership accuracy (predicted action ∈ optimal set)

|                          |  linear / pre  |  linear / post  |   MLP / pre   |   MLP / post   |
|-------------------------:|:-------------:|:---------------:|:-------------:|:--------------:|
| Model A (agent)          |     0.463     |      0.453      |     0.453     |     0.398      |
| **Model B (optimal)**    |   **1.000**   |    **0.990**    |   **1.000**   |   **0.990**    |

### Test log-likelihood

|                          |  linear / pre  |  linear / post  |   MLP / pre   |   MLP / post   |
|-------------------------:|:-------------:|:---------------:|:-------------:|:--------------:|
| Model A (agent)          |    −1.2385    |     −1.2269     |    −1.2373    |    −1.2397     |
| Model B (optimal)        |    −0.2068    |     −0.2042     |    −0.1906    |    −0.1938     |

### Findings from the extension

**(a) MLP is essentially tied with linear.** Across all four (token × label)
slots the MLP changes test accuracy by at most ±1 pp. This matches the
finding from `cost_updated.ipynb` (linear and MLP both hit 61.3 % on the
fixed-key-door dataset). Layer-15 activations already linearise
distance-to-goal — the extra capacity has nothing useful to do.

**(b) Pre vs post is also nearly tied.** Pre-reasoning φ is a hair better
on Model B (0.935 vs 0.922, +1.3 pp), suggesting that the geometric
information is already present in the residual stream *before* the model
emits any tokens for that step. Post-reasoning φ is a hair better on
Model A (linear: 0.427 vs 0.417), but well within noise.

**(c) The headline gap is robust.** Model B beats Model A by 49–53 pp test
accuracy in every (head × token) combination. The qualitative finding —
*activations encode the optimal direction much more cleanly than the
agent's actual choice* — does not depend on the model class or the token
position.

---

## Per-visit re-run (no φ averaging) — and an inversion on post-reasoning

The runs above all use a **state-averaged** φ table: every visit to the
same state is collapsed into one mean 8640-dim vector before training.
This re-run keeps each visit as its own training example (no averaging
anywhere — pure per-visit φ).

### Algorithmic change forced by per-visit φ

The original cost-IRL formulation `P(a | s) = softmax(−β · θᵀ φ(f(s, a)))`
needs φ(next-state) for **all 4 actions**. Three of those four next-states
are *counterfactual* — at a given visit the agent only actually moved into
one of them. The averaged-φ runs sidestep this by using one mean φ per
state for every counterfactual lookup; the per-visit runs cannot, so the
model is reformulated as **per-action linear / MLP heads on per-visit φ**:

```
C_θ(s, a) = (θ_a)ᵀ · φ_visit(s)        # 4 separate cost heads
P(a | s)  = softmax(−β · C_θ(s, a))
```

For the linear head this is equivalent to multinomial logistic regression
(weight matrix `W = −β · θ ∈ ℝ^{4 × 8640}`); for the MLP head the final
projection is changed from `Linear(128, 1)` to `Linear(128, 4)`. All other
hyperparameters are unchanged. Since the algorithm is no longer "cost of
next state" the per-visit numbers should not be compared point-for-point
to the averaged numbers — but the qualitative comparison Model A vs Model
B is still meaningful.

### Test accuracy (per-visit)

|                          |  linear / pre  |  linear / post  |   MLP / pre   |   MLP / post   |
|-------------------------:|:-------------:|:---------------:|:-------------:|:--------------:|
| **Model A (agent)**      |     0.540     |    **0.916**    |     0.453     |   **0.932**    |
| **Model B (optimal)**    |   **0.900**   |      0.515      |   **0.896**   |     0.602      |

### Test set-membership accuracy (predicted action ∈ optimal set)

|                          |  linear / pre  |  linear / post  |   MLP / pre   |   MLP / post   |
|-------------------------:|:-------------:|:---------------:|:-------------:|:--------------:|
| Model A (agent)          |     0.524     |      0.430      |     0.181     |     0.417      |
| **Model B (optimal)**    |   **0.964**   |      0.560      |   **0.974**   |     0.686      |

### Test log-likelihood (per-visit)

|                          |  linear / pre  |  linear / post  |   MLP / pre   |   MLP / post   |
|-------------------------:|:-------------:|:---------------:|:-------------:|:--------------:|
| Model A (agent)          |    −1.0723    |     −0.3997     |    −1.2053    |    −0.2241     |
| Model B (optimal)        |    −0.3495    |     −1.2432     |    −0.3148    |    −1.0121     |

### Findings from the per-visit runs

**(a) Pre-reasoning is still goal-direction-biased, even per-visit.** With
per-visit pre-reasoning φ, Model B (optimal) still beats Model A (agent)
by a wide margin (90 % vs 54 % linear / 90 % vs 45 % MLP). The
prompt-suffix activations carry the goal-relative geometry of the visit
but not yet a strong commitment to a specific action.

**(b) Post-reasoning per-visit φ flips the result.** Model A — predicting
the agent's actual action — now reaches **91.6 % (linear) / 93.2 % (MLP)**,
while Model B's accuracy collapses to **51.5 % / 60.2 %**. The set-membership
metric tells the same story: Model A picks an optimal action only ~43 %
of the time when fitted on post-reasoning φ — which is *exactly* the
agent's own BFS-optimal rate (43.4 %). The post-reasoning residual stream
has *committed to* the action the model is about to emit, so a per-visit
classifier reads off the agent's choice with high accuracy but inherits
all of the agent's sub-optimality.

**(c) Why averaging hid this.** Averaging φ across all visits to the same
state mixes together the activations corresponding to *different* actions
the agent took at that state. The per-visit "I am about to emit DOWN"
signal is washed out, leaving only the visit-invariant component (goal
direction). That's why the averaged-φ runs all looked the same across
pre and post — once you average, the action-commitment information is
gone.

**(d) Combined picture.**

| token | what φ encodes per visit | best model |
|------:|--------------------------|-----------:|
| pre   | goal direction (visit-invariant) | Model B (optimal) |
| post  | the action the agent is about to emit | Model A (agent) |
| **averaged (any token)** | only the visit-invariant component (goal direction) | Model B (optimal) |

The inversion at **post / per-visit** is the cleanest evidence that the
post-reasoning residual stream carries the model's *policy commitment*,
not just spatial geometry. The averaged version of the experiment cannot
see this signal because averaging removes it.

---

## Belief–action gap across layers (7, 15, 23)

Treating the per-visit pre-reasoning φ as the carrier of the agent's
*belief* about the optimal action and `agent_action` as the
*behaviour*, the gap is `belief_acc − action_acc`. Replayed at every
saved layer with a 5-fold-CV linear decoder trained on BFS-optimal
labels:

| layer | belief acc (vs geometric) | action acc | **gap (pp)** | belief = action |
|---|---:|---:|---:|---:|
| layer 7  | 0.935 | 0.447 | **+48.8** | 0.358 |
| **layer 15** | **0.963** | 0.447 | **+51.6** | 0.372 |
| layer 23 | 0.862 | 0.447 | +41.5 | 0.345 |

`action_acc` is constant by definition (it's the agent's BFS-optimal
rate, 43.4 % over 672 / 1549 (s, a) pairs — slightly different here
because the layer-sweep keeps all 1589 records, no next-state-φ filter
needed for the per-visit decoder).

Disagreement breakdown — of all (belief, action) disagreements at each
layer, how many are "belief ∈ opt, action ∉ opt" (the gap) vs the
opposite:

| layer | belief ∈ opt, action ∉ opt | action ∈ opt, belief ∉ opt |
|---|---:|---:|
| layer 7  | **843** | 67  |
| layer 15 | **855** | 35  |
| layer 23 | **776** | 116 |

### Findings

1. **Geometric belief is built early.** Layer 7 already hits 93.5 %
   belief accuracy — well before the middle of the stack.
2. **Layer 15 is the peak** for clean linear geometric decoding (96.3 %).
3. **Layer 23 *loses* geometric info (−10 pp from peak).** The late
   residual stream is less linearly decodable as goal direction than
   the middle. The "action ∈ opt, belief ∉ opt" count more than triples
   (35 → 116) — the decoder fails on more states where the agent itself
   was right.
4. **The belief-action gap stays large at every depth** (+49 / +52 /
   +42 pp). The residual stream knows the right answer at all three
   layers; the agent never acts on it.
5. **Belief = action is flat (0.36 / 0.37 / 0.35).** None of the three
   pre-reasoning layers commit to the agent's chosen action. The
   commitment is happening **at the output-token positions, not in the
   pre-reasoning residual stream at any depth** — consistent with the
   per-visit post-reasoning experiment, where post-reasoning φ does
   read off the agent's action.

The belief-action gap **does not open across pre-reasoning depth**.
It opens *during the reasoning chain*, between the prompt-suffix
positions (which know the answer at every layer) and the output tokens
(which don't).

---

## Files

- Notebook: `seed12_agent_vs_optimal.ipynb` (re-runnable)
- Scripts: `run_seed12_agent_vs_optimal.py` (linear + pre only)
  and `run_agent_vs_optimal_extended.py` (full 8-run grid for both datasets)
- Raw results: `results/seed12_agent_vs_optimal_raw.{json,pkl}`,
  `results/seed12_agent_vs_optimal_extended.{json,pkl}` (averaged φ),
  `results/seed12_agent_vs_optimal_per_visit.{json,pkl}` (per-visit φ),
  and `results/belief_action_gap_layers.json` (layer-sweep belief-action gap)
- Belief–action gap reports: `results/belief_action_gap.md`,
  `results/belief_action_gap_layers.md`
- Belief–action gap scripts: `run_belief_action_gap.py`,
  `run_belief_action_gap_layers.py`
- This report: `results/seed12_agent_vs_optimal_results.md`

## Reproducing

```bash
# Linear + pre-reasoning only (the headline run)
uv run python run_seed12_agent_vs_optimal.py

# Full 8-run grid: {linear, MLP} x {pre, post} x {agent, optimal}
uv run python run_agent_vs_optimal_extended.py --dataset seed12

# or run the notebook end-to-end:
uv run jupyter notebook seed12_agent_vs_optimal.ipynb
```
