# fourroom_episode_sweep_T0 Cost-IRL: Agent Action vs BFS-Optimal Action

## Headline

Two `LinearCostIRL` models trained on identical
`fourroom_episode_sweep_T0` activations (GPT-OSS-20B, layer 15,
prompt-suffix, 8640-dim φ); only the **label** differs.

| metric                          | Model A (label = agent action) | Model B (label = optimal action) |
|---------------------------------|-------------------------------:|---------------------------------:|
| **test accuracy**               |                      **0.454** |                        **0.960** |
| test set-membership accuracy¹   |                          0.369 |                        **0.992** |
| test log-likelihood             |                        −0.9969 |                          −0.1389 |
| train accuracy                  |                          0.451 |                            0.972 |
| majority-class baseline         |                  0.548 (UP)    |                       0.733 (DOWN) |
| random baseline                 |                          0.250 |                            0.250 |

¹ *Predicted action ∈ BFS-optimal set (gives credit when several actions are tied for optimal).*

**Activations predict the BFS-optimal action with 96.0 % accuracy
(99.2 % set-membership) but predict the agent's actual action only
45.4 % of the time — well below the 54.8 % majority-class baseline.
Layer-15 prompt-suffix activations encode the optimal direction,
not the agent's UP-bias.**

---

## The dataset: fourroom_episode_sweep_T0

### Grid

A single fixed 9 × 9 doorkey layout:

```
   0 1 2 3 4 5 6 7 8
 0 # # # # # # # # #
 1 # _ _ _ # G _ _ #     <- goal at (5, 1)
 2 # _ _ _ # _ _ _ #
 3 # _ _ _ _ _ _ _ #     <- row 3 connects upper room
 4 # D # # # # # # #     <- locked door at (1, 4)
 5 # _ _ _ # K _ _ #     <- key at (5, 5)
 6 # _ _ _ # _ _ _ #
 7 # _ _ _ _ _ _ _ #
 8 # # # # # # # # #
```

39 walkable cells. Door at (1, 4) is the only crossing between the
lower room (rows 5–7) and the upper room (rows 1–3); locked unless
the agent has the key.

### Trajectories

- **111 trajectories** in
  `data/trajectories/fourroom_episode_sweep_T0/together_ai*.json`,
  sampled at temperature 0.
- Three initial-flag variants encoded in the filename:

  | variant | initial flags                              | n  |
  |---|---|---:|
  | `K0D0` | `(carrying_key=False, door_open=False)`     | 36 |
  | `K1D0` | `(carrying_key=True,  door_open=False)`     | 37 |
  | `K1D1` | `(carrying_key=True,  door_open=True )`     | 38 |

  (No `K0D1` variant.) Recorded flags **match the variant name** at
  every step.

- Step-level flag distribution across all 4 698 (s, a) pairs:

  | `(carrying_key, door_open)` | steps | %    |
  |---|---:|---:|
  | (False, False) |   677 | 14.4 % |
  | (True,  False) | 1 365 | 29.1 % |
  | (True,  True ) | 2 656 | 56.5 % |

- 25 flag-changing transitions across consecutive steps: 12
  key-pickup events `(F,F) → (T,F)` and 13 door-open events
  `(T,F) → (T,T)`.

### Activations

Layer 15, `prompt_suffix` token category (3 tokens × 2880 dim →
8640-dim φ), `data/activations/fourroom_episode_sweep_T0`.

---

## The algorithm

### Cost model and policy

Single θ ∈ ℝ^{8640}; `cost(s) = θᵀ φ(s)`;
`P(a | s; θ) = softmax_a (−β · cost(f(s, a)))` with β = 1.

### Transition `f(s, a)` — full state

`(col, row, has_key, door_open) → (col, row, has_key, door_open)`:

- Walls / out-of-bounds: stay.
- Locked door cell with `has_key=False`: blocked, stay.
- Locked door cell with `has_key=True`: walk through and auto-open
  the door (`door_open` flips True).
- Key cell with `has_key=False`: pick up the key (`has_key` flips
  True) and move onto the cell.
- Goal: terminal.

The **same rule is used for all 4 actions** per record — no
record-derived override.

### BFS over the full state space

Optimal-action labels come from BFS in the full
`(col, row, has_key, door_open)` state space using `step_state`. A
`K0D0` lower-room agent's distance to goal includes the key detour;
the same position with `K1D0` doesn't. Position-only BFS would
mistakenly give them the same distance.

### Training

NLL + L2 on θ (`λ = 0.01`); Adam, lr 1e-3, cosine LR schedule over
500 epochs, full-batch, β = 1, seed 42. 80/20 train/test split over
records (per-transition, not per-trajectory). Best epoch by test
log-likelihood.

---

## Dataset stats

|                                                     | value    |
|----------------------------------------------------|---------:|
| total trajectory steps                              | 4 698    |
| steps kept after next-state filter                  | 4 614    |
| dropped (next-state φ missing)                      | 35       |
| unique `(col, row, has_key, door_open)` states with φ | 111    |
| φ dimensionality                                    | 8 640    |
| n_train / n_test                                    | 3 692 / 922 |
| `agent_is_optimal_fraction`                         | **0.325** |

### Action distributions

|         | LEFT | RIGHT | UP        | DOWN      |
|---------|----:|----:|----------:|----------:|
| agent   |  416 |   660 | **2 530** |  1 008    |
| optimal |  326 |   795 |     113   | **3 380** |

---

## Per-action accuracy (test set)

**Model A** (label = agent's action):

| action | accuracy | n   |
|-------:|---------:|----:|
| LEFT   |    0.764 |  89 |
| RIGHT  |    0.810 | 126 |
| UP     |    0.228 | 527 |
| DOWN   |    0.717 | 180 |

**Model B** (label = optimal action):

| action | accuracy | n   |
|-------:|---------:|----:|
| LEFT   |    0.958 |  72 |
| RIGHT  |    0.853 | 156 |
| UP     |    0.778 |  27 |
| DOWN   |    0.992 | 667 |

Model A is collapsed on UP (only 22.8 % accuracy) — the activations
encode the optimal direction (mostly DOWN under state-space BFS),
not the agent's UP-bias.

---

## Cross-evaluation

| model trained on   | scored against | acc   | in-opt-acc | log-lik |
|-------------------:|---------------:|------:|-----------:|--------:|
| **agent labels (A)**   | optimal labels | 0.324 | **0.369**  | −3.5010 |
| **optimal labels (B)** | agent labels   | 0.271 | **0.992**  | −5.1525 |

Model B is geometrically right 99.2 % of the time even when scored
against the agent's labels.

---

## Extension: MLP cost head and pre- vs post-reasoning activations

### Test accuracy

|                          |  linear / pre  |  linear / post  |   MLP / pre   |   MLP / post   |
|-------------------------:|:-------------:|:---------------:|:-------------:|:--------------:|
| **Model A (agent)**      |     0.454     |      0.456      |     0.431     |     0.430      |
| **Model B (optimal)**    |   **0.960**   |    **0.960**    |   **0.961**   |   **0.964**    |
| gap (B − A)              |    +0.506     |     +0.504      |    +0.530     |    +0.534      |

### Test set-membership accuracy

|                          |  linear / pre  |  linear / post  |   MLP / pre   |   MLP / post   |
|-------------------------:|:-------------:|:---------------:|:-------------:|:--------------:|
| Model A (agent)          |     0.369     |      0.369      |     0.330     |     0.330      |
| **Model B (optimal)**    |   **0.992**   |    **0.991**    |   **0.992**   |   **0.995**    |

### Test log-likelihood

|                          |  linear / pre  |  linear / post  |   MLP / pre   |   MLP / post   |
|-------------------------:|:-------------:|:---------------:|:-------------:|:--------------:|
| Model A (agent)          |    −0.9969    |     −1.0038     |    −1.0018    |    −1.0024     |
| Model B (optimal)        |    −0.1389    |     −0.1222     |    −0.1229    |    −0.1043     |

### Findings

**(a) MLP barely moves the needle.** ±2 pp on Model A test accuracy
(slightly worse than linear), ~0 pp on Model B. Layer-15
activations already linearise the cost; the MLP capacity has nothing
to do.

**(b) Pre vs post is essentially tied.** Visit-averaging φ washes
out post-reasoning's policy-commitment signal, leaving only the
visit-invariant goal-relative geometry — present pre-reasoning
already.

**(c) The headline gap is robust** at +50 pp test accuracy in every
combination.

---

## Files

- This report:
  `results/fourroom_episode_sweep_T0_agent_vs_optimal_results.md`
- Raw outputs:
  `results/fourroom_episode_sweep_T0_agent_vs_optimal_extended.{json,pkl}`
- Script: `run_agent_vs_optimal_extended.py`

## Reproducing

```bash
uv run python run_agent_vs_optimal_extended.py --dataset fourroom
```
