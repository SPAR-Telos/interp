# n-Step Maximum Entropy IRL on a Fixed Key-Door Grid

## 1. Introduction

This report describes an experiment applying **n-step Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL)** to infer a reward function from 100 demonstration trajectories of GPT-OSS-20B navigating a fixed 13x13 key-door grid environment. The goal is to recover the implicit reward structure that explains the agent's observed behavior.

Unlike a simpler one-step approximation (which treats each transition independently), the n-step algorithm models the full multi-step decision process through soft value iteration and state visitation propagation, capturing long-horizon dependencies such as the key-then-door-then-goal task chain.

## 2. Environment

### 2.1 Grid Layout

The environment is a fixed 13x13 grid with a 3x3 room structure, walls separating rooms, a locked door, and a key:

```
   0  1  2  3  4  5  6  7  8  9  10 11 12
0  #  #  #  #  #  #  #  #  #  #  #  #  #
1  #  _  _  _  #  _  _  _  _  _  _  _  #
2  #  G  _  _  #  _  _  _  #  _  _  A  #
3  #  _  _  _  #  _  _  _  #  _  _  _  #
4  #  #  #  _  #  _  #  #  #  #  #  _  #
5  #  _  _  _  D  _  _  _  #  _  _  _  #
6  #  _  _  _  #  _  _  _  #  _  _  _  #
7  #  _  _  _  #  _  _  _  #  _  _  _  #
8  #  #  _  #  #  _  #  #  #  #  #  #  #
9  #  _  _  _  #  _  _  _  #  _  _  _  #
10 #  _  _  _  #  _  _  _  _  _  _  _  #
11 #  _  _  _  #  _  _  K  #  _  _  _  #
12 #  #  #  #  #  #  #  #  #  #  #  #  #

Legend: # = Wall, _ = Open, G = Goal, A = Agent start, D = Door, K = Key
```

| Element | Position | Description |
|---------|----------|-------------|
| Agent start | (2, 11) | Top-right room |
| Goal | (2, 1) | Top-left room |
| Key | (11, 7) | Bottom-center room |
| Door | (5, 4) | Passage between left and center rooms |

### 2.2 Task Structure

The agent must solve a sequential subtask chain:

1. **Navigate to key** at (11, 7) from start at (2, 11)
2. **Collect key** by stepping onto the key cell
3. **Navigate to door** at (5, 4) while carrying key
4. **Open door** automatically when adjacent (Manhattan distance 1) to the locked door while carrying key
5. **Navigate to goal** at (2, 1) through the now-open door passage

### 2.3 Transition Mechanics

- **Movement**: 4 deterministic actions (UP, DOWN, LEFT, RIGHT). Walls and grid boundaries block movement (agent stays in place).
- **Locked door**: Blocks movement like a wall. Cannot be stepped on when locked.
- **Key collection**: Automatic when the agent steps onto the key cell.
- **Door opening**: Automatic when the agent is adjacent (Manhattan distance = 1) to the locked door while carrying the key. The door cell then becomes passable.
- **Goal**: Absorbing state. Once the agent reaches the goal, it remains there.

### 2.4 State Space

Each state is a tuple **(row, col, has_key, door_open)** where:

- `(row, col)` is the agent's grid position (non-wall cells only)
- `has_key` is a boolean indicating whether the agent carries the key
- `door_open` is a boolean indicating whether the door has been opened

The combination `(has_key=False, door_open=True)` is excluded as unreachable (the door can only open while carrying the key). This yields **267 reachable states** for this grid (~89 non-wall cells x 3 valid boolean combinations).

### 2.5 Demonstration Data

- **Source**: 100 trajectories of GPT-OSS-20B on the same fixed grid with varied starting positions
- **Steps per trajectory**: up to 50
- **Total transitions**: 4,134 (state, action) pairs
- **Model**: `openai/gpt-oss-20b` with chain-of-thought reasoning

## 3. Feature Design

The reward function is parameterized as a linear combination of 7 hand-crafted features:

```
R(s) = theta^T phi(s) = sum_{i=0}^{6} theta_i * phi_i(s)
```

### 3.1 Feature Vector phi(s)

| Index | Name | Formula | Range |
|-------|------|---------|-------|
| 0 | `dist_goal` | BFS(agent, goal) / grid_size^2 | [0, 1] |
| 1 | `dist_key` | 0 if has_key, else BFS(agent, key) / grid_size^2 | [0, 1] |
| 2 | `dist_door` | 0 if not has_key or door_open, else BFS(agent, door) / grid_size^2 | [0, 1] |
| 3 | `has_key` | 1 if carrying key, else 0 | {0, 1} |
| 4 | `door_open` | 1 if door is open, else 0 | {0, 1} |
| 5 | `p_key` | P(carrying_key \| activations) from neural probe, or has_key as fallback | [0, 1] |
| 6 | `p_door` | P(door_open \| activations) from neural probe, or door_open as fallback | [0, 1] |

### 3.2 BFS Distance Computation

Distances are computed using breadth-first search on the grid, treating wall cells as impassable:

```
BFS(start, target) = shortest_path_length(start, target) / grid_size^2
```

- **Normalization**: Dividing by `grid_size^2` (= 169 for this 13x13 grid) maps all distances to [0, 1], making features comparable across different grid sizes.
- **Unreachable sentinel**: If no path exists, returns 1.0.
- **Same position**: Returns 0.0.
- **Note**: The BFS uses only the wall set as obstacles. The locked door is NOT included as a BFS obstacle, so `dist_goal` may underestimate the true distance when the door is locked and lies on the shortest path.

### 3.3 Conditional Feature Logic

The distance features encode task-relevant conditioning:

- **dist_key = 0 when has_key**: Once the key is collected, distance to key is irrelevant. Zeroing it acts as a completion signal.
- **dist_door = 0 when not has_key OR door_open**: Distance to door only matters when the agent is carrying the key and the door is still locked. Before collecting the key, the door is not yet actionable. After opening, it is no longer relevant.
- **p_key / p_door fallback**: When neural probes are unavailable (no trained probe models), these features fall back to the binary indicators `has_key` and `door_open`, making them identical to features 3 and 4.

### 3.4 Feature Table Pre-computation

All 267 state feature vectors are pre-computed into a matrix `phi` of shape `(267, 7)` before training. This avoids repeated BFS computations during the IRL optimization loop.

## 4. Algorithm: n-Step MaxEnt IRL

### 4.1 Theoretical Foundation

Maximum Entropy IRL (Ziebart et al., 2008) recovers a reward function from demonstrations by finding the policy that:

1. **Matches feature expectations**: The expected features under the learned policy equal the empirical feature expectations from demonstrations.
2. **Maximizes entropy**: Among all policies matching the feature expectations, choose the one with maximum entropy (most uncertain / least committed).

This leads to the optimization:

```
max_theta  L(theta) = sum_demos sum_t log pi_theta(a_t | s_t)
```

where the optimal policy takes the Boltzmann (softmax) form:

```
pi_theta(a | s) = exp(Q_theta(s, a)) / sum_{a'} exp(Q_theta(s, a'))
```

The gradient of the log-likelihood is:

```
grad_theta L = phi_demo - phi_policy
```

where:
- `phi_demo = (1/N) sum_t phi(s_t)` is the average feature vector over all demonstration states
- `phi_policy = sum_s mu(s) phi(s)` is the expected feature vector under the learned policy, weighted by the state visitation frequency `mu`

### 4.2 Backward Pass: Soft Value Iteration

The backward pass computes the soft Q-values and value function by iterating the soft Bellman equations for `n` steps:

```
Q(s, a) = R(s) + V(T(s, a))
V(s)    = logsumexp_a Q(s, a) = log sum_a exp(Q(s, a))
```

where:
- `R(s) = theta^T phi(s)` is the current reward for state s
- `T(s, a)` is the deterministic next state
- `V` is initialized to zero and updated for `n` iterations
- The logsumexp is the "soft" analogue of the max operator in standard Bellman equations

**Numerical stability**: The logsumexp is computed as:

```
V(s) = max_a Q(s,a) + log sum_a exp(Q(s,a) - max_a Q(s,a))
```

After convergence, the policy is extracted as:

```
pi(a | s) = exp(Q(s,a) - max_a' Q(s,a')) / sum_a' exp(Q(s,a') - max_a' Q(s,a'))
```

### 4.3 Forward Pass: State Visitation Propagation

The forward pass computes the expected state visitation frequency `mu(s)` under the current policy:

```
mu_0 = empirical distribution over first states in demonstrations
mu(s) = sum_{t=0}^{n-1} mu_t(s)     (accumulated visitation)

mu_{t+1}(s') = sum_{s,a} mu_t(s) * pi(a|s) * 1[T(s,a) = s']
```

Starting from the empirical initial state distribution, the visitation is propagated forward through the transition model weighted by the policy for `n` steps. The total visitation is then normalized: `mu = mu / sum(mu)`.

### 4.4 Gradient Update

Each epoch performs:

```
phi_policy = mu^T phi        (expected features under policy)
gradient   = phi_demo - phi_policy
theta      = theta + lr * gradient
```

The gradient pushes the reward weights to increase reward for features that are over-represented in demonstrations relative to the current policy, and decrease reward for under-represented features.

### 4.5 Comparison with One-Step Approximation

| Aspect | One-Step IRL | n-Step IRL |
|--------|-------------|------------|
| Transition model | Treats each (s, a) independently | Full multi-step dynamics via T matrix |
| Value propagation | None (immediate reward only) | n iterations of soft Bellman backup |
| State visitation | Per-transition, no temporal structure | Forward propagation from mu_0 over n steps |
| Credit assignment | Only next-state features matter | Long-horizon credit via V(s) function |
| Key-door dependency | Cannot model "get key THEN open door" chain | Captures sequential subtask structure |
| Compute cost | O(N_transitions x 4 actions) per epoch | O(N_states x 4 actions x n) per epoch |

The n-step approach is essential for this environment because the optimal strategy requires multi-step planning: the agent must first collect the key (which has no immediate reward) in order to later open the door and reach the goal.

## 5. Experimental Setup

### 5.1 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Horizon (n) | 50 | Matches maximum trajectory length; ensures full task coverage |
| Epochs | 50 | Sufficient for gradient convergence (norm < 0.015) |
| Learning rate | 0.1 | Aggressive step size; works well for feature-matching gradient |
| Reward model | R(s) = theta^T phi(s) | Linear in state features; 7 parameters |
| Discount factor | 1.0 (undiscounted) | Finite horizon makes discounting unnecessary |

### 5.2 Probe Configuration

In this experiment, neural probes for `p_key` and `p_door` were **not available** (probe models not yet trained). Features 5 and 6 fall back to binary proxies identical to features 3 and 4 (`has_key` and `door_open`). This means the effective feature space has 5 unique dimensions, with features 3/5 and 4/6 being duplicated.

## 6. Results

### 6.1 Learned Reward Weights

```
R(s) = theta^T phi(s)

Feature        Weight     Share    Interpretation
-----------    -------    -----    ---------------
dist_goal      -0.0272     5.8%   Penalizes distance to goal
dist_key       -0.0404     8.6%   Penalizes distance to key
dist_door      -0.0054     1.2%   Weak penalty for distance to door
has_key        +0.0325     6.9%   Rewards carrying key
door_open      +0.1654    35.3%   Strongly rewards open door
p_key          +0.0325     6.9%   = has_key (binary proxy fallback)
p_door         +0.1654    35.3%   = door_open (binary proxy fallback)
```

### 6.2 Convergence

The gradient norm decreases steadily over 50 epochs:

```
Epoch  10/50: ||grad|| = 0.052046   max_gap = 0.035733
Epoch  20/50: ||grad|| = 0.029001   max_gap = 0.015525
Epoch  30/50: ||grad|| = 0.021611   max_gap = 0.010215
Epoch  40/50: ||grad|| = 0.017287   max_gap = 0.007608
Epoch  50/50: ||grad|| = 0.014552   max_gap = 0.007725
```

The algorithm has largely converged by epoch 50, with all per-feature gaps below 0.015.

### 6.3 Evaluation Metrics

| Metric | Value | Random Baseline |
|--------|-------|-----------------|
| Log-likelihood | -1.2529 | -1.3863 (= log 1/4) |
| Action accuracy | 53.4% | 25.0% |
| Transitions evaluated | 4,131 | - |

### 6.4 Feature Expectation Gap

The feature expectation gap measures how well the learned policy matches the demonstration distribution for each feature:

```
Feature        Gap (demo - policy)     Status
-----------    -------------------     ------
dist_goal        -0.000236              converged
dist_key         +0.000607              converged
dist_door        -0.000164              converged
has_key          -0.013617              near-converged
door_open        -0.014444              near-converged
p_key            -0.013617              near-converged
p_door           -0.014444              near-converged
```

Distance features are tightly matched (gaps < 0.001). Binary/probe features show small remaining gaps (~0.014), indicating the policy slightly over-estimates key collection and door opening frequency compared to demonstrations.

## 7. Interpretation

### 7.1 Reward Structure

The learned reward function reveals a clear task hierarchy:

1. **Door opening is the dominant reward signal** (theta = +0.165, 35.3% of total weight, doubled to 70.6% when counting the p_door duplicate). This reflects the fact that opening the door is the critical bottleneck — the agent cannot reach the goal without it, and it requires the multi-step sub-plan of first collecting the key.

2. **Key collection is modestly rewarded** (theta = +0.033, 6.9%). The key has value primarily as a prerequisite for door opening, not as an end in itself. The n-step algorithm correctly assigns more credit to the door (the actual bottleneck) than to the key (the enabler).

3. **Distance penalties are small but correctly signed**. All three distance features have negative weights, meaning the policy prefers states closer to the goal, key, and door. The key distance penalty (-0.040) is larger than the goal distance penalty (-0.027), reflecting that in most of the trajectory the agent is navigating toward the key before heading to the goal.

4. **Door distance is nearly zero** (-0.005). This makes sense: `dist_door` is only non-zero in the narrow window where the agent has the key but hasn't yet opened the door. The door is typically opened quickly after key collection, so this feature rarely varies.

### 7.2 Why 53.4% Accuracy?

The action accuracy of 53.4% (vs 25% random) is reasonable but not near-perfect. Several factors explain this:

- **Stochastic demonstrations**: GPT-OSS-20B's trajectories include chain-of-thought reasoning that occasionally leads to suboptimal moves, exploratory detours, or corrections. The MaxEnt policy averages over these behaviors rather than mimicking the greedy optimal path.
- **Maximum entropy objective**: The MaxEnt framework explicitly trades off prediction accuracy for entropy. It assigns non-zero probability to all actions, not just the demonstrated one.
- **Linear reward model**: A 7-feature linear model is a limited representation. It cannot capture position-specific strategies (e.g., "go left at this intersection") that would require non-linear features.
- **Binary proxy features**: Without trained neural probes, features 5/6 duplicate features 3/4, reducing the effective feature dimensionality from 7 to 5.

### 7.3 Multi-Step Credit Assignment

The key advantage of n-step IRL over the one-step approximation is visible in the reward weights. The one-step approach would assign reward based only on immediate next-state features, missing the fact that collecting the key (which has no immediate benefit for reaching the goal) is valuable because it enables future door opening. The n-step backward pass propagates this credit backward through the value function: states near the key receive higher V(s) because they lead to states where the agent has the key, which lead to states where the door is open, which lead to the goal.

### 7.4 Limitations

- **No neural probe features**: The p_key and p_door features are identical to has_key and door_open in this experiment. Training and using the key/door probes would provide continuous estimates that could improve feature expressiveness.
- **BFS ignores door state**: The BFS distance computation does not account for the locked door as an obstacle. This means `dist_goal` can underestimate the true distance when the door is locked and on the shortest path.
- **Linear reward model**: Cannot represent non-linear feature interactions (e.g., "distance to goal matters more after the door is open").
- **Fixed grid**: Results are specific to this particular grid layout. Generalization across grids would require the one-step approximation or a different feature representation.

## 8. Conclusion

The n-step MaxEnt IRL successfully recovers an interpretable reward function from GPT-OSS-20B's demonstration trajectories on a 13x13 key-door grid. The learned weights reveal that the agent's behavior is best explained by a strong preference for opening the door (the task bottleneck), moderate preference for collecting the key (the enabler), and weak but consistent preference for minimizing distances to task-relevant objects. The algorithm achieves 53.4% action prediction accuracy with a simple 7-feature linear model, more than doubling the random baseline.
