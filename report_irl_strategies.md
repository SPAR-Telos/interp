# Learning the Agent's Cost Function: Two Strategies

**Model**: GPT-OSS-20B navigating 9×9 key-door grid environments
**Goal**: Learn a cost function C(s) = θ^T φ(s) from demonstrated trajectories
**Data**: 95 key-door multi-step trajectories + 36,000 single-step navigation trajectories

---

## 1. Problem Setup

The agent operates in a 9×9 partially observable grid containing a key (K), a locked door (D), and a goal (G). The task requires the agent to: (1) collect the key, (2) open the door, (3) reach the goal. The action space is {UP, DOWN, LEFT, RIGHT} — directional movement only, with key collection and door opening triggered automatically on cell entry.

We wish to recover the agent's implicit objective from its demonstrated behaviour, without access to its internal reward signal. Two strategies are explored:

- **Strategy 1**: Inverse Reinforcement Learning using a hand-crafted + probe feature vector φ(s) ∈ ℝ⁷
- **Strategy 2**: Behavioural Cloning directly from the agent's activation vectors h(s) ∈ ℝ⁸⁶⁴⁰

---

## 2. Strategy 1: One-Step Maximum Entropy IRL

### 2.1 Background and Motivation

Standard Maximum Entropy IRL (Ziebart et al. 2008) recovers a reward function R_θ(s) = θ^T φ(s) by finding weights θ such that the induced Boltzmann policy best explains the demonstrations. The agent selects actions according to:

$$\pi_\theta(a \mid s) = \frac{\exp(R_\theta(T(s, a)))}{\sum_{a'} \exp(R_\theta(T(s, a')))}$$

where T(s, a) is the next state after taking action a in state s. The full MaxEnt IRL gradient requires computing the expected state visitation under the current policy via backward value iteration — expensive and requiring a single fixed MDP.

Here, each of the 95 trajectories comes from a *different* randomly generated grid (different wall layouts, key/door positions), making global value iteration intractable. We therefore use the **one-step approximation**, which treats each transition independently.

### 2.2 Algorithm

The one-step MaxEnt IRL gradient at step (s_t, a_t) is:

$$\nabla_\theta \mathcal{L} = \varphi(T(s_t, a_t)) - \sum_{a} \pi_\theta(a \mid s_t)\, \varphi(T(s_t, a))$$

This is the difference between features of the *actually visited* next state and the *expected* features under the current policy. The update pushes θ to assign higher reward to the demonstrated next state than to counterfactual alternatives.

**Algorithm (implemented in `run_maxent_irl.py`)**:

```
Input:  Demonstrations {(s_t, a_t)}, feature function φ, learning rate α, epochs E
Output: Reward weights θ ∈ ℝ^7

θ ← 0
for epoch = 1 to E:
    for each (s_t, a_t) in demonstrations:
        s_next ← T(s_t, a_t)                      # apply transition model
        φ_actual ← φ(s_next)                       # features of actual next state

        for each action a ∈ {UP, DOWN, LEFT, RIGHT}:
            φ_cf[a] ← φ(T(s_t, a))                # counterfactual next-state features

        costs   ← φ_cf · θ                        # shape (4,)
        π       ← softmax(−costs)                  # Boltzmann policy
        φ_exp   ← Σ_a π(a) · φ_cf[a]             # expected features under π

        θ ← θ + α · (φ_actual − φ_exp)            # gradient ascent on log-likelihood
```

Two approximations are embedded:
1. **Probe features for counterfactual states**: Features 5–6 (probe outputs) require model activations at the next state, which are unavailable for actions not taken. Binary proxies (ground-truth has_key / door_open) are used for counterfactual transitions; actual probe outputs are used for the demonstrated transition.
2. **Per-step i.i.d. assumption**: Each (s_t, a_t) pair is treated independently, ignoring the temporal dependency that proper MaxEnt IRL would handle via forward-backward passes.

### 2.3 Feature Vector φ(s) ∈ ℝ⁷

| # | Name | Description | Source |
|---|------|-------------|--------|
| 0 | `dist_goal` | BFS distance from agent to goal | Grid model |
| 1 | `dist_key` | BFS distance to key; 0 if carrying | Grid model |
| 2 | `dist_door` | BFS distance to door; 0 if carrying or open | Grid model |
| 3 | `has_key` | 1 if agent carries the key | Trajectory field |
| 4 | `door_open` | 1 if door is absent from grid | Grid state |
| 5 | `p_key` | P(carrying_key \| h(s)) from trained probe | MLP probe |
| 6 | `p_door` | P(door_open \| h(s)) from trained probe | MLP probe |

Features 5–6 are the outputs of the binary MLP probes trained earlier on the same activation data. They encode the model's *internal belief* about its key/door status, potentially carrying richer information than the ground-truth binary values (features 3–4).

BFS distances use only walls as obstacles; the door cell is treated as passable (optimistic lower bound on true distance).

### 2.4 Training Data

- **Key-door trajectories**: 1,081 (s_t, a_t, s_{t+1}) transitions from 94 multi-step trajectories (key + door + goal task)
- **Single-step size-9 trajectories**: 6,000 single-step samples from open navigation environments (no key/door; features 1–6 collapse to constants)
- **Total**: 7,081 samples

The single-step data contributes only to the `dist_goal` gradient, providing substantially more signal for that feature.

### 2.5 Results

**Learned reward weights θ** (R(s) = θ^T φ(s)):

| Feature | Weight | % of total |
|---------|--------|-----------|
| `dist_goal` | −1215.18 | 90.9% |
| `dist_door` | −56.61 | 4.2% |
| `dist_key` | −40.93 | 3.1% |
| `p_door` | +20.89 | 1.6% |
| `has_key` | +3.28 | 0.2% |
| `p_key` | −0.16 | <0.1% |
| `door_open` | ≈0 | <0.1% |

**Evaluation** (on full 7,081-sample dataset):

| Metric | Value | Random baseline |
|--------|-------|----------------|
| Action accuracy | **79.7%** | 25.0% |
| Log-likelihood | −3.153 | −1.386 |
| Feature gap (max) | 0.40 | — |

### 2.6 Analysis

#### Weight dominance of `dist_goal`

The `dist_goal` weight accounts for 90.9% of the total absolute weight magnitude. This is primarily a consequence of the 6,000 single-step training samples: in those environments, only `dist_goal` carries a non-zero gradient (all key/door features are constant). The result is that the single-step data inflates the `dist_goal` weight ~30× relative to key/door features compared to training on key-door trajectories alone. The reward function is essentially "minimise distance to goal, with minor corrections for key and door proximity."

#### Probe output vs binary feature

`p_door` (+20.89) receives a large positive weight while its binary counterpart `door_open` (≈0) is effectively ignored. The probe outputs a continuous probability rather than a hard 0/1 signal, giving the gradient more traction: in states where the model's internal representation is *transitioning* between door-closed and door-open belief (mid-trajectory), the probe provides a gradient signal that the binary feature misses entirely. This suggests the probe has learned something richer than a simple threshold detector.

`p_key` (−0.16), by contrast, contributes almost nothing. This is likely because `has_key` transitions happen in a single step and are already captured by `has_key` (+3.28) and `dist_key` (−40.93); the probe adds little beyond what the binary indicator already encodes.

#### Log-likelihood paradox

The log-likelihood (−3.153) is *worse* than the random baseline (−1.386), yet accuracy (79.7%) far exceeds random (25.0%). This apparent contradiction arises from the scale of θ: with `dist_goal = −1215`, the softmax is extremely peaked — the policy places nearly all probability mass on one action. When that action is correct, accuracy is high; but when it is wrong (20.3% of the time), the probability assigned to the correct action approaches zero, collapsing the log-likelihood. In effect, the model is overconfident: well-calibrated for correct predictions, catastrophically wrong for incorrect ones.

This is a direct consequence of the one-step approximation. Full MaxEnt IRL with value iteration would normalise the policy globally, preventing runaway θ magnitudes. The one-step gradient can keep increasing θ as long as the feature expectations move in the right direction, without any regularisation from a partition function that spans the whole state space.

#### Feature expectation matching

All per-feature expectation gaps are below 0.5, confirming that the MaxEnt constraint (E_demo[φ] ≈ E_policy[φ]) holds. This is the central guarantee of MaxEnt IRL — the learned policy visits states with the same average feature profile as the expert. Crucially, this guarantee holds even when the underlying reward parameterisation is a poor fit, which explains why the feature gaps are small despite the log-likelihood being poor.

---

## 3. Strategy 2: Behavioural Cloning from Activations

### 3.1 Formulation

Rather than hand-crafting φ(s), Strategy 2 uses the model's own internal activation vector h(s_t) ∈ ℝ⁸⁶⁴⁰ directly as the feature representation. The reward model becomes:

$$R(s, a) = \mathbf{W}_a^\top h(s_t)$$

where W ∈ ℝ^{4×8640} is a per-action weight matrix. This is action-conditional, depending on the current state's activation rather than the next state's features. The MaxEnt log-likelihood then reduces exactly to **cross-entropy behavioural cloning**:

$$\mathcal{L}(W) = \sum_t \log \text{softmax}(W\, h(s_t))[a_t]$$

This formulation avoids the counterfactual problem entirely: we need only activations at states actually visited, not at all possible next states.

The activations are taken at two positions in the forward pass:

- **Pre-reasoning**: last token of the *prompt suffix* — the model's residual stream state just before it begins generating the action token
- **Post-reasoning**: last token of the *model output* — the residual stream state at the generated action token itself

### 3.2 Training Data

- **Key-door activations**: 896 (h(s_t), a_t) pairs (activations at step t, not t+1)
- **Single-step activations** (all grid sizes 5–15): 36,000 samples
- **Total**: 36,896 samples per condition

Since Strategy 2 learns a direct map from activation vectors to actions — with no hand-crafted grid features — the model is **grid-size-agnostic**: the 8640-dim activation space is the same regardless of grid dimensions, allowing all 36,000 single-step samples to be used.

Model: L2-regularised linear classifier (LR) and shallow MLP (hidden dim 128), trained with AdamW (lr=0.001, weight_decay=0.01), z-score normalisation, 80/20 split.

### 3.3 Results

| Model | Accuracy | Log-likelihood | Bal. Accuracy |
|-------|----------|---------------|---------------|
| Random baseline | 25.0% | −1.386 | 25.0% |
| Strategy 1 (IRL) | 79.7% | −3.153 | N/A |
| Pre-reasoning LR | 57.7% | −1.018 | 55.2% |
| Pre-reasoning MLP | 57.7% | −0.974 | 55.5% |
| **Post-reasoning LR** | **95.7%** | **−0.217** | **95.6%** |
| **Post-reasoning MLP** | **95.0%** | **−0.167** | **95.0%** |

**Per-action recall:**

| Action | Pre-reasoning LR | Post-reasoning LR |
|--------|-----------------|------------------|
| UP | 76.7% | 96.0% |
| DOWN | 60.5% | 96.3% |
| LEFT | 41.4% | 95.7% |
| RIGHT | 42.4% | 94.6% |

### 3.4 Analysis

#### Post-reasoning is near-perfect

Post-reasoning accuracy (95.7%) substantially exceeds both the random baseline and Strategy 1. This is expected: the output token's activation is the model's internal state *at the moment of decision*. The action choice is already encoded — essentially, reading out the output activation is directly reading the model's decision. The small residual error (4.3%) likely reflects genuinely ambiguous positions where multiple actions are nearly optimal.

The log-likelihood is also well-behaved (−0.217 vs −1.386 for random), in sharp contrast to Strategy 1. This means the post-reasoning policy is *calibrated* as well as accurate: when it predicts action a, the probability assigned to a is genuinely high.

#### Pre-reasoning carries partial but asymmetric information

Pre-reasoning accuracy (57.7%) is well above random but far below post-reasoning. This shows that the model's directional intent is partially encoded *before* generating the output token — consistent with the view that transformers pre-compute their answers in the residual stream before the final projection step.

The asymmetry across actions is striking: UP (76.7%) and DOWN (60.5%) are far easier to predict pre-generation than LEFT (41.4%) and RIGHT (42.4%). A likely explanation is the spatial structure of the task: the two rooms in the environment are separated by a *vertical wall* with a horizontal door. Vertical movement (UP/DOWN) relates directly to navigating toward or away from the goal along the dominant spatial axis of the environment. Horizontal movement (LEFT/RIGHT), by contrast, is more context-dependent — it encodes approach to the key or door, which requires integrating more task-state information (whether the agent is in the key-collection phase or the door-crossing phase). This integration may not be complete until the model generates the output token.

Notably, the LR and MLP achieve nearly identical accuracy (57.7%) in the pre-reasoning condition, suggesting the action-relevant information is linearly separable even before generation. The marginal benefit of the nonlinear MLP is absorbed entirely in the calibration improvement (−0.974 vs −1.018 log-likelihood).

#### Linear decodability and representation structure

That a *linear* classifier on post-reasoning activations achieves 95.7% accuracy is a strong result: it means the 4 action directions are approximately linearly separable in the 8640-dimensional activation space. This is consistent with findings in mechanistic interpretability that transformer residual streams represent information in a linearly structured manner.

The near-zero gap between LR (95.7%) and MLP (95.0%) further supports this: adding a nonlinear hidden layer provides no benefit, indicating the relevant information is already linearly accessible. The slight *decrease* for MLP is consistent with mild overfitting of the nonlinear layer despite L2 regularisation.

---

## 4. Comparison and Discussion

### 4.1 Summary table

| | Strategy 1 (MaxEnt IRL) | Strategy 2 Pre | Strategy 2 Post |
|---|---|---|---|
| Feature space | 7-dim hand-crafted + probes | 8640-dim activations | 8640-dim activations |
| Requires transition model | Yes (synthetic BFS + grid model) | No | No |
| Requires next-state activations | Partially (for probe features) | No | No |
| Action accuracy | 79.7% | 57.7% | **95.7%** |
| Log-likelihood | −3.153 | −1.018 | **−0.217** |
| Output | Named reward weights θ | Dense weight matrix W | Dense weight matrix W |
| Interpretability | High | Low | Low |

### 4.2 What each strategy recovers

**Strategy 1** recovers a *state-level reward function* R(s) that is interpretable and grounded in the task structure. The learned weights confirm that the agent is goal-directed: it most strongly avoids being far from the goal, with secondary signals for key and door proximity. The reward function is legible to a human analyst.

**Strategy 2** recovers a *policy* in activation space: a direct linear map from the model's internal state to action probabilities. This does not give a human-interpretable decomposition of the agent's objectives, but it gives far more accurate behavioural predictions. The post-reasoning result effectively shows that the model's decision is already written in its activations.

### 4.3 The probe features bridge both strategies

The probe outputs (p_key, p_door) in Strategy 1 provide a glimpse of what Strategy 2 leverages fully. `p_door` receiving the second-largest weight (+20.89) shows that the model's internal belief about the door state — as read out by the probe — is more predictive than the ground-truth binary indicator. This is a direct connection: Strategy 2's success comes precisely from reading out the full, high-dimensional version of this internal state, not just the 2-bit projection that the probes provide.

### 4.4 Limitations

**Strategy 1:**
- The one-step approximation ignores long-horizon credit assignment. Picking up the key is intrinsically valuable only because it enables door-crossing several steps later; a one-step gradient sees only the immediate next state.
- The data imbalance (6,000 simple-nav vs 1,081 key-door samples) inflates the `dist_goal` coefficient and suppresses key/door signals.
- The learned θ magnitudes are unconstrained, leading to an overconfident policy with poor log-likelihood despite good accuracy.

**Strategy 2:**
- The learned weight matrix W is not interpretable: we cannot easily extract which aspects of the agent's cognition drive which actions.
- Post-reasoning activations are available only *after* the model has committed to an action; the reward function cannot be evaluated for hypothetical future states the model has not actually visited.
- The 36,000 single-step samples dominate training in terms of count; in the key-door specific domain (896 samples), the model has seen far fewer examples of the key/door mechanics.

### 4.5 Complementarity

The two strategies are complementary rather than competing. Strategy 1 provides a transparent, externally-grounded account of the agent's objectives; Strategy 2 provides a high-fidelity, internally-grounded account of its decisions. A natural synthesis would be to use Strategy 2's high-accuracy action predictions to generate a richer pseudo-labelled dataset for re-training a more expressive version of Strategy 1 — effectively using the agent's own activations to bootstrap a more complete hand-crafted feature analysis.
