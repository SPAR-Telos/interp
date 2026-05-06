# `flag-swap` branch — causal-intervention test of the `carrying_key` belief

This branch implements an **activation-patching experiment** asking
a single question:

> Is the agent's pre-reasoning `carrying_key` belief, as encoded in
> the layer-15 prompt-suffix residual stream, **causally
> responsible** for action selection — or just an **epiphenomenal
> correlate** of the geometry/plan that the model has already
> computed by that point?

The experiment was iterated three times. Each iteration tightened
the experimental control and produced a different headline result.
The final, cleanest iteration finds **null causal effect** under a
full-vector swap at the canonical patching positions, supporting
the epiphenomenal-correlate hypothesis. This document explains how
we got there, what each iteration showed, and what's still open.

---

## 1. Research question and design rationale

### Why this experiment exists

Earlier "variant transplant" runs (Direction 2a, on a different
branch) showed that swapping mean K1D1 activations into K0D0 forward
passes shifted action distributions. That result was **not a clean
causal test** for two reasons:

1. **Mean-tensor donor.** Mean activations across many K1D1 visits
   produce a residual stream the model never produces in normal
   operation — out-of-distribution by construction.
2. **Both flags flipped at once.** K0D0 → K1D1 flips
   (carrying_key, door_open) simultaneously, so even a positive
   result couldn't attribute the effect to either flag.

The flag-swap design fixes both problems by using **real
forward-pass activations** from a single donor visit and isolating
**a single flag** (`carrying_key`).

### Canonical setup

- **Cell**: agent at (1, 5), the cell directly below the locked
  door at (1, 4) on the 9×9 doorkey grid. The K0D0 vs K1D0
  BFS-optimal actions diverge sharply here:
  - K0D0 (no key) → BFS-optimal: RIGHT/DOWN (head back to the key
    at (5, 5)).
  - K1D0 (has key) → BFS-optimal: UP (door auto-opens, walk into
    the goal room).
- **Prompts**: one real recorded step from the
  `fourroom_episode_sweep_T0` dataset for each variant
  (K0D0 traj055 step 0, K1D0 traj056 step 0).
- **Donor activations**: layer-15 residual stream at the prompt's
  trailing positions, captured live from the running model.
- **Hook**: `HookedLayer` wrapping `model.model.layers[15]` —
  three modes: `passthrough`, `capture` (record the layer's
  output at the chosen positions), `replace` (overwrite those
  positions with the donor tensor).

### The six conditions

For each iteration we run six conditions, each producing N=30
sampled completions at T=0.7:

| condition | prompt | hook | tests |
|---|---|---|---|
| baseline-a | a (K0D0) | passthrough | reference distribution |
| baseline-b | b (K1D0) | passthrough | reference distribution |
| self-a | a | replace with `act_a` | no-op sanity |
| self-b | b | replace with `act_b` | no-op sanity |
| **swap-a-from-b** | a | replace with `act_b` | **hypothesis (a-side)** |
| **swap-b-from-a** | b | replace with `act_a` | **hypothesis (b-side)** |

**Predictions under "carrying_key is causal":**
- `swap-a-from-b` injects "I have the key" residual into the
  no-key prompt → action distribution should shift toward UP
  (K1D0-modal).
- `swap-b-from-a` mirror → action distribution should shift toward
  RIGHT/DOWN (K0D0-modal).

**Predictions under "carrying_key is epiphenomenal":**
- Both swap conditions roughly track their `self-*` controls.
  Action selection re-derives `carrying_key` from elsewhere
  (e.g., direct attention to the explicit `True/False` text
  token).

### Hard project rule

**T = 0.7 with sampling, never greedy.** Distributional comparison
across N samples per condition. The activation capture itself is a
deterministic forward pass (temperature only affects the final
softmax sampling), so `act_a`/`act_b` are well-defined; the
generation step that follows is sampled.

---

## 2. Three iterations and what each one revealed

The experiment ran three times. Each time we tightened the
experimental control after spotting a confound.

### Iteration 1 — Asymmetric (initial run)

**State of the code:** prepare script took the trajectory-level
`prompt_suffix_tokens` field, which contains the **unrendered
template** with literal `{{carrying_key}}` and `{{door_open}}`
placeholders rather than `True`/`False` strings. The K cell was
rendered as `K` in K0D0 (key on grid) and `_` in K1D0 (key in
inventory, removed from grid).

**Diff between prompts:** exactly one grid token (the K cell). The
suffix was identical in both prompts because both contained
literal `Ġ{{carrying_key}}`.

**Headline:** asymmetric flip. UP frequency baseline-a 0.13 →
swap-a-from-b 0.30 (doubled). Mirror b-side null. Initial
interpretation: the K-cell rendering carries "have key" cue more
strongly than residual carries "no key" cue.

**Bug uncovered later:** the suffix had `Ġ{{carrying_key}}`
literally. The only K0D0/K1D0 differentiator was the K cell. The
"causal" interpretation in the asymmetric report was confounded
with grid rendering. Archived report:
[`results/flag_swap_results_asymmetric.md`](../results/flag_swap_results_asymmetric.md).

### Iteration 2 — Symmetric grid + step-rendered suffix

**Two fixes applied to prepare:**
1. Use `step["prompt_suffix_tokens"]` (step-level, with `True`/`False`
   strings rendered) instead of `tj["prompt"]["prompt_suffix_tokens"]`
   (template-level).
2. Override prompt b's grid tokens with prompt a's grid tokens, so
   the K cell is always visible in both prompts. The only textual
   difference is then a single token at suffix offset 9 (`Ġ False`
   id 7983 vs `Ġ True` id 6432).

**Diff between prompts:** one suffix token only.
**Per-position ‖act_a − act_b‖:** [1138, 467, 364] at the captured
positions (last 3 of the prompt). Much larger than iteration 1's
~50 — consistent with True/False being only 7-9 positions before
the captured spots.

**Headline:** a-side flip strengthens dramatically. UP frequency
baseline-a 0.17 → swap-a-from-b **0.50** (tripled, modal action).
b-side still null.

**New problem spotted:** prompt b is now **internally
contradictory**:
- Grid shows `K` at (5, 5).
- Suffix says `Carrying key: True`.
- Prefix instructions say: *"Once picked up, the key disappears
  from the grid"* and *"If you do not see a K on the grid, it means
  the key has already been picked up."*

Prompt b is therefore an OOD combination the model has never seen
during training. baseline-b parse-fail rate (21/30) was the highest
of any condition, supporting this read.

Hypothesis: the apparent "a-side flip" might be the model
**resolving prompt b's confusion** rather than reading a clean
carrying_key belief. Patching K1D0's confused residual into prompt
a substitutes a "go UP" signal not because UP is K1D0-causal but
because it's whatever prompt b's confused state happened to
produce. Archived report:
[`results/flag_swap_results_symmetric.md`](../results/flag_swap_results_symmetric.md)
and the original "first non-null causal result" writeup:
[`results/flag_swap_results_symmetric_archived.md`](../results/flag_swap_results_symmetric_archived.md).

### Iteration 3 — Corrected prompts (current cleanest test)

**Fix:** rewrite the two prefix sentences that contradict the
symmetric prompt b, so K-visible-while-carrying becomes consistent
with the rules:

- *"Once picked up, the key disappears from the grid and the agent
  carries it."* → *"Once picked up, the key still appears on the
  grid and the agent carries it."*
- *"If you do not see a K on the grid, it means the key has already
  been picked up."* → *"The K symbol stays on the grid regardless
  of whether you are carrying the key."*

Implemented as `--rewrite-prefix` on the prepare script. Both
prompts share the rewritten prefix; the only remaining textual
difference is still the single suffix token (False/True).

**Headline:** the a-side flip **disappears**. With internally
consistent prompts, full-vector swap at the last 3 positions
produces no significant action shift in either direction.

| condition | LEFT | RIGHT | UP | DOWN | parse_fail | freq(UP) |
|---|---:|---:|---:|---:|---:|---:|
| baseline-a (K0D0) | 1 | 7 | 5 | 2 | 15 | 0.167 |
| self-a            | 1 | 3 | 6 | 3 | 17 | 0.200 |
| **swap-a-from-b** | 3 | 9 | 4 | 2 | 12 | **0.133 ↓** |
| baseline-b (K1D0) | 0 | 5 | 10 | 0 | 15 | 0.333 |
| self-b            | 0 | 5 | 12 | 0 | 13 | 0.400 |
| **swap-b-from-a** | 0 | 9 | 10 | 0 | 11 | **0.333 =** |

Pairwise chi² (4-class):

| comparison | chi² | p |
|---|---:|---:|
| self-a vs baseline-a (sanity) | 1.76 | 0.62 ✓ |
| self-b vs baseline-b (sanity) | 0.00 | 1.00 ✓ |
| swap-a-from-b vs self-a (a-test) | 3.89 | 0.27 (null) |
| swap-a-from-b vs baseline-a (a-test) | 1.10 | 0.78 (null) |
| swap-b-from-a vs self-b (b-test) | 0.58 | 0.45 (null) |
| swap-b-from-a vs baseline-b (b-test) | 0.23 | 0.64 (null) |

Both swap conditions are statistically indistinguishable from their
own variant's controls. Mass shifts toward RIGHT in both swaps
(+0.07 a-side, +0.13 b-side) — i.e., the swap perturbs the residual
without producing a directed belief flip. Detailed report:
[`results/flag_swap_results.md`](../results/flag_swap_results.md).

### Cross-iteration comparison

| | iter 1 (asymmetric) | iter 2 (symmetric) | **iter 3 (corrected)** |
|---|---:|---:|---:|
| Prompts differ at | 1 grid token | 1 suffix token | 1 suffix token |
| Prompt b internally consistent? | yes (template) | **no** (OOD) | yes |
| ‖act_a − act_b‖ at last 3 positions | ~50–63 | 1138 / 467 / 364 | 1069 / 462 / 233 |
| swap-a-from-b UP freq | 0.30 | **0.50** | 0.13 |
| Δ vs self-a | +0.13 | +0.27 | **−0.07** |
| swap-b-from-a UP freq | 0.30 | 0.33 | 0.33 |
| Δ vs self-b | −0.03 | +0.07 | −0.07 |
| chi² p (a-test vs self) | 0.49 | 0.12 | 0.27 |

The "a-side flip" magnitude tracks **how OOD prompt b is**, not
the carrying_key flag itself. As we made prompt b less
contradictory, the apparent causal effect shrank and disappeared.

---

## 3. What we currently believe

Reading across the three iterations together, the most
parsimonious account is:

> **The carrying_key flag is encoded at layer-15 prompt-suffix
> (probe accuracy ~96%) but the encoding is not load-bearing for
> action selection in the suffix region we patched.**

i.e., a **correlate** of the model's geometry/plan, not the
**cause** of the action. Three concrete hypotheses remain
testable:

1. **Truly epiphenomenal at this (layer, position) pair.** The
   action pathway re-derives carrying_key in late layers (16–23)
   directly from the explicit `True`/`False` text token via
   attention, bypassing the layer-15 residual representation.
2. **Wrong patching positions.** The `True`/`False` token is at
   suffix offset 9; we patched offsets 16–18 (`<|end|>`,
   `<|start|>`, `assistant`). The residual at those late
   positions has been re-summarized by 15 layers of attention
   into "what comes next" rather than "carrying_key=X". A patch
   at the True/False token itself, or across the entire 19-token
   suffix block, might still produce a flip.
3. **Wrong layer.** Layer-15 was chosen because the cost-IRL
   probe peaks there for BFS optimality, but the *causal* layer
   for the carrying_key flag could be different (e.g., 7 or 23).

A diagnostic captured the per-position ‖act_a − act_b‖ across the
full 19-token suffix to localise the signal:

```
suffix offset  token      ‖act_a − act_b‖
0–8            (identical preface)            0
9              ĠFalse / ĠTrue            1234       ← signal source
10             \n (newline)              1531       ← peak (immediate downstream)
11             - (dash)                   762
12             ĠDoor                      476
13             Ġopen                      563
14             :                          572
15             ĠFalse (door_open=False)  1069
16             <|end|>                    462       ← we patched here
17             <|start|>                  233       ← we patched here
18             assistant                    0       ← we patched here (no diff!)
```

Position 18 is **literally zero** difference — by then the
True/False distinction has fully washed out. The "last 3 positions"
patch was effectively a 2-position patch, at positions that mostly
encode "ready to emit action" rather than the flag value.

The pending wider-position experiment (n_pos=19, full suffix patch)
was kicked off and then cancelled before completion to leave space
for further design discussion. See "Open follow-ups" below.

---

## 4. Methodology details worth knowing

### MLX runtime instead of HF/PyTorch

The HF/PyTorch path can't run gpt-oss-20b on a 48 GB Mac because
MXFP4 quantisation requires Triton (CUDA-only) and dequantising to
bf16 needs ~73 GB. MLX runs gpt-oss-20b at MXFP4 natively on Apple
GPU, ~14 GB resident. The runner uses
`mlx-community/gpt-oss-20b-MXFP4-Q4`.

### `HookedLayer` pattern

Python looks up `__call__` on the **type**, not the instance, so a
naive monkey-patch like `layer.__call__ = self.__call__` doesn't
work. The fix: replace the entry in the parent list directly.

```python
class HookedLayer:
    def __init__(self, parent_list, idx, n_pos):
        self.parent_list = parent_list
        self.idx = idx
        self.n_pos = n_pos
        self.original_layer = parent_list[idx]
        self.mode = "passthrough"
        self.donor = None
        self.captured = None
        parent_list[idx] = self  # splice in

    def __call__(self, x, mask, cache, *args, **kwargs):
        out = self.original_layer(x, mask, cache, *args, **kwargs)
        if out.shape[1] <= 1:
            return out  # generation step, not prompt-processing
        if self.mode == "capture":
            self.captured = out[0, -self.n_pos:, :]
        elif self.mode == "replace":
            head = out[:, :-self.n_pos, :]
            tail = mx.broadcast_to(self.donor[None, :, :],
                                   (out.shape[0], self.n_pos, out.shape[2]))
            out = mx.concatenate([head, tail], axis=1)
        return out
```

The hook fires once per forward pass on the prompt-processing call
(seq_len > 1). During generation steps (seq_len == 1) it's a
no-op, so all N samples per condition share the same patched
prompt cache.

### Live activation capture

In an early RunPod run we extracted activations from the MXFP4
model on disk and passed them to the bf16-dequantised RunPod
process. This produced **drifting self-* conditions** because
the saved activations didn't match the runtime's residual stream.
The MLX runner re-extracts `act_a` and `act_b` live from the loaded
model immediately before the swap conditions, so `self-*` is a
no-op by construction. The disk-saved `act_a.pt` and `act_b.pt`
are reference snapshots only — the live runner ignores them.

### Token structure of the prompt

The prompt has three concatenated regions:

1. **Prefix** (716–720 tokens): static template instructions —
   system message, environment description, action JSON schema.
   Identical between K0D0 and K1D0 in iterations 2 and 3 (in
   iteration 3 the prefix is rewritten once and reused for both
   prompts).
2. **Grid state** (119 tokens): the 9×9 ASCII rendering. Differs
   between K0D0 and K1D0 by exactly one token (the K cell at
   (5, 5)). In iterations 2 and 3 prompt b reuses prompt a's grid
   tokens so the grid is identical between prompts.
3. **Suffix** (19 tokens, step-rendered): `Agent status:` block
   with `Carrying key: {True|False}` and `Door open: False`,
   followed by `<|end|><|start|>assistant`. Differs between K0D0
   and K1D0 by exactly one token (offset 9, `False` vs `True`).

Important historical bug: the prepare script originally pulled
`prompt_suffix_tokens` from the **trajectory level**
(`tj["prompt"]["prompt_suffix_tokens"]`), which is the unrendered
**template** with literal `Ġ{{carrying_key}}` placeholders, not
the rendered values. The fix is to use the **step level**
(`step["prompt_suffix_tokens"]`).

---

## 5. File map

```
flag_swap/
├── README.md                       (this file)
├── prompts.jsonl                   currently the corrected prompts
├── act_a.pt / act_b.pt             reference activations (stale; live runner ignores)
├── results.jsonl                   iteration 1 (asymmetric)
├── results_symmetric.jsonl         iteration 2 (symmetric, OOD prompt b)
├── results_corrected.jsonl         iteration 3 (corrected prompts) — current main result
├── results_corrected_n19.jsonl     wider-position swap (interrupted)
├── run.log                         iteration 1 stdout
├── run_symmetric.log               iteration 2 stdout
├── run_corrected.log               iteration 3 stdout
└── run_corrected_n19.log           wider-position stdout (partial)
```

Top-level scripts (in repo root):

```
run_flag_swap_prepare.py            Mac CPU. Build prompts.jsonl + act_*.pt
                                    --rewrite-prefix to apply iteration-3 prefix fix.
run_flag_swap_intervention_mlx.py   Mac MLX. The 6-condition runner.
                                    --n-pos N to widen the patch beyond the default 3.
run_flag_swap_intervention.py       HF/PyTorch RunPod runner (didn't ultimately work).
run_flag_swap_analyze.py            Mac CPU. Build results markdown from a results.jsonl.
```

Reports (in `results/`):

```
results/flag_swap_results.md                     current main report
results/flag_swap_results_asymmetric.md          iteration 1 archive
results/flag_swap_results_symmetric.md           iteration 2 (re-generated)
results/flag_swap_results_symmetric_archived.md  iteration 2's original "positive result" writeup
```

---

## 6. Reproducing the current main result

```bash
# 1. Prepare iteration-3 prompts (rewritten prefix, symmetric grid,
#    step-rendered suffix). Writes flag_swap/prompts.jsonl.
uv run python run_flag_swap_prepare.py --rewrite-prefix

# 2. Run the six conditions on Mac via MLX (~24 min wall-clock).
uv run python run_flag_swap_intervention_mlx.py \
    --output flag_swap/results_corrected.jsonl

# 3. Analyse and write the markdown report.
uv run python run_flag_swap_analyze.py \
    --results flag_swap/results_corrected.jsonl \
    --output  results/flag_swap_results.md
```

To reproduce iteration 2 (symmetric but OOD):

```bash
uv run python run_flag_swap_prepare.py        # no --rewrite-prefix
uv run python run_flag_swap_intervention_mlx.py \
    --output flag_swap/results_symmetric.jsonl
```

To run a wider-position swap (the diagnostic for the
"wrong-positions" hypothesis), keep the corrected prompts and pass
`--n-pos 19`:

```bash
uv run python run_flag_swap_intervention_mlx.py \
    --n-pos 19 \
    --output flag_swap/results_corrected_n19.jsonl
```

---

## 7. Open follow-ups

In rough order of expected information value:

1. **Wider-position full-vector swap** (`--n-pos 19`). Tests
   whether the carrying_key signal has causal effect anywhere in
   the suffix block, not just at the last 3 positions. If still
   null, this is strong evidence for hypothesis 1 (truly
   epiphenomenal at layer 15).
2. **Single-position swap at the True/False token** (offset 9).
   Most surgical full-vector test: replace the source-position
   activation directly. Forces the late-layer attention to read
   the donor's representation at the source.
3. **Direction-only patch** using the trained
   `KeyCollectedProbe` weight vector. Replaces only the scalar
   projection along the carrying_key direction; everything
   orthogonal stays untouched. Mathematically dominated by
   full-vector swap as an upper bound — only useful if a wider
   full-vector swap shows an effect and we want to attribute it.
4. **Layer sweep**. Re-run the corrected-prompts experiment at
   layers 7, 11, 19, 23 to localise the causal layer (if any).
   ~25 min × 4 layers + activation gathering.
5. **Multi-cell sweep**. Repeat at 3–5 cells with sharp K0D0/K1D0
   optimal-action divergence (e.g., (5, 6), (5, 7) in addition to
   (1, 5)) to test generalisation.

---

## 8. Caveats and limitations

- **N = 30 per condition with 40–60 % parse-fail** leaves
  effective N ≈ 12–18. The current null result rules out a
  *large* causal effect; a small effect could still hide below
  detection. A larger run (N = 100) would tighten this.
- **Single (cell, layer, suffix-position) triple.** Generalisation
  across cells and layers untested.
- **Hand-crafted prefix rewrite.** The corrected prefix
  ("K stays visible while carrying") is a phrase the model has
  never seen during pre-training. Less OOD than iteration 2's
  literal contradiction, but not zero.
- **MXFP4 quantisation.** All runs use the MXFP4 quantised model;
  bf16 dequantised behaviour might differ slightly. Self-*
  sanity passes in every iteration so this isn't biasing the
  causal contrast, but absolute action probabilities are
  quantisation-conditioned.
- **The carrying_key probe achieves ~96 % accuracy at layer 15.**
  Probe accuracy and causal influence are independent properties.
  The current null is consistent with the layer-15 representation
  being a high-fidelity correlate that downstream circuitry does
  not read. This is the central methodological takeaway of the
  branch.
