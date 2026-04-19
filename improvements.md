# BO Improvement Methods

This document records concrete improvement strategies for `bo_tuning.py`
(formerly `rtp_bo_tuning.py` — renamed because Method 1 below removes the
"random tuning profile" semantics from BO),
ranked by expected impact and implementation cost. See [report.md](report.md)
for the underlying findings that motivate these changes.

## Root cause being addressed

From [report.md](report.md) Finding 3:

- BO and RTP both treat each tuning target as a **binary on/off switch only**.
- When a target is enabled, the actual value applied to the heuristic is
  **resampled uniformly per call** from the prior's range.
- Composite search records the *specific* (prior_type, value) pair that
  produced the best per-target binary size reduction, but RTP/BO discards
  that information and only keeps the prior range.
- Result: BO/RTP cannot reproduce composite's single-target peak when the
  response surface around the optimal value is sharp (e.g., `2mm` -7.16%,
  `gemm` -7.67% are unreachable by random sampling within ~450 trials).

---

## Method 1 — Composite best-value injection (✅ implemented, smoke test running)

**Goal**: BO at `enable_i = 1, others = 0` should deterministically reproduce
composite's single-target rel_objective for that function.

### Changes

1. New loader `load_top_k_targets_with_best_value` in
   [bo_tuning.py](augmentum-main/driver/bo_tuning.py):
   - Joins `path_heuristics` ⨝ `probe_log` to find each top-k function's best
     `(prior_type, value, rel_objective)` triple — *not* just an arbitrary
     fitted prior.
   - Includes `Broadcast Prior` results (the previous loader excluded them).
2. `TuningTarget.best_value: Optional[str]` field carries the literal value
   discovered by composite.
3. `generate_bo_extension` codegen emits a **deterministic literal** when
   `best_value` is set:
   ```c++
   *probed = (probed_type)(<best_value>);          // if op == None
   *probed = original_value + (probed_type)(<v>);  // if op == "+"
   *probed = original_value * (probed_type)(<v>);  // if op == "*"
   ```
   Otherwise falls back to `rand() % range` (legacy paper-RTP behavior).
4. New CLI flag `--use_best_value`; `USE_BEST_VALUE=1` env var in
   [sbatch_bo.sh](augmentum-main/driver/scripts/sbatch_bo.sh).

### Why this should work
- Composite-best target (top-1) for the 5-bench config:
  `SCCPSolver::Solve` with `Broadcast Prior value=1` → composite rel=0.9233
  (matches `gemm` -7.67% and `2mm` -7.16% peaks).
- Setting `enable_0 = 1, all_others = 0` should now compile, pass verify, and
  achieve rel ≈ 0.9233 — recovering composite's single-target ceiling.
- BO's RF surrogate then has an **achievable baseline** to stack on top of.

### Test in flight
- Job `47782739` — 200 iterations, top_k=14, `USE_BEST_VALUE=1`,
  `working_dir_rtp_bo_bestval_test/`.
- **Pass criteria**: at least one `enable_i = 1, others = 0` config records
  rel_objective within 1% of the corresponding composite single-target best.

---

## Method 2 — Constraint-aware acquisition (next, not yet implemented)

**Goal**: stop BO from spending ~50% of its iteration budget on `num_targets ≥ 8`
configurations that have 0% verify success rate.

### Plan

1. Build a unified feature/label dataset from existing labeled samples
   (50,422 trials available: 31,996 RTP + 18,426 BO):
   - Features per trial: `[enable_0, enable_1, ..., enable_k, num_targets,
     priors used, value_op flags]`.
   - Label: `verify_result == 'SUCCESS'` (1) vs anything else (0).
2. Train a `RandomForestClassifier` (sklearn) for `P(feasible | config)`.
3. Wrap skopt's acquisition in a feasibility filter:
   ```python
   def constrained_acq(x):
       return ei(x) * P_feasible(x)   # multiplicative, common in cBO
   ```
   Or sample N candidates per BO step, keep only those with
   `P_feasible(x) > τ`, pass survivors to BO's internal acquisition.
4. Recompute classifier periodically (every 200 iters) to incorporate new data.

### Expected effect
- BO compile-fail rate drops from ~50% toward RTP's ~28%.
- Effective iteration budget grows ~2×.
- Combined with Method 1, BO should beat RTP across all 5 benchmarks.

### Notes
- The classifier is trained on **legacy random-sampled** trials. After
  Method 1 ships, the feasibility surface shifts (deterministic values change
  the failure pattern). Plan to retrain after a small Method-1 BO run completes
  and contributes new labels.

---

## Method 3 — Aggressive `top_k` cap (cheapest A/B test, optional)

**Goal**: confirm the over-exploration component of BO's failure mode.

- Lower `top_k` from 14 → 6.
- All else identical.
- If BO with `top_k=6` matches RTP without Method 1, then BO's bias toward
  high-`num_targets` is the dominant problem.
- If BO with `top_k=6` still loses to RTP, then value-selection (Method 1) is
  confirmed to be the bottleneck.

This was queued earlier (jobs 47781592–47781596) and **cancelled** in favor of
the Method 1 smoke test, since the value-sampling argument predicts `top_k=6`
won't help on `2mm`/`gemm` regardless.

If still useful as a control, resubmit any time after Method 1's chain
finishes (cannot run in parallel — both touch shared `llvm10/build/`).

---

## Method 4 — Tiered value search dimensions (extension, future)

**Goal**: let BO actually search around the composite-best value rather than
locking to it.

### Plan

For each target, replace single `Categorical([0, 1])` with two dimensions:

```python
Categorical([0, 1], name=f"enable_{i}")
Categorical(["off", "best", "best_minus", "best_plus", "random"],
            name=f"value_choice_{i}")
```

Where:
- `best` = composite's best literal (Method 1 default)
- `best_minus` = `best * 0.5` (or `best - small_delta` for ints)
- `best_plus` = `best * 2`
- `random` = legacy paper-RTP behavior
- `off` = ignored when `enable_i = 0`

### Trade-off
- Search space grows 5× per dim. With k=14 → 5^14 ≈ 6 billion combinations,
  but BO's RF surrogate handles correlated categoricals fine.
- More iterations needed before BO converges; consider raising
  `n_initial_points` from `2*k` to `3*k`.

### When to do this
- Only after Methods 1 + 2 are validated and you want to push beyond
  composite's per-target ceiling on benchmarks where the response surface has
  multiple basins.

---

## Method 5 — Cross-benchmark target ranking (research extension)

**Goal**: pick top-k targets that work *across* multiple benchmarks rather
than per-benchmark.

Currently `load_top_k_targets[_with_best_value]` ranks by `MIN(rel_objective)`
across **all** probes — this is benchmark-agnostic and may overfit to
whichever benchmark has the cheapest peak.

### Alternatives to try
- Rank by `geometric_mean(rel_objective)` across the 5 benchmarks (rewards
  targets that improve all benchmarks somewhat).
- Rank by `count(benchmarks where rel_objective < 0.99)` then break ties by
  `MIN(rel_objective)`.
- Use composite's `path_heuristics.test_coverage` as a tiebreaker (more
  benchmarks tested → more reliable signal).

This becomes important once we move from 5-bench to 30-bench evaluation, where
`working_dir_composite_30/subset_composite.sqlite` already has the discovery
data needed.

---

## Decision matrix

| Method | Status | Effort | Expected gain | Risk |
|---|---|---|---|---|
| 1. Best-value injection | Implemented, testing | Done | High (recovers composite ceiling) | Low — codegen change is small and reversible via flag |
| 2. Constraint-aware acq | Pending | ~1 day | High (2× effective budget) | Low — skopt supports custom acq |
| 3. Lower `top_k` | Cancelled | None (already wired) | Medium-low (diagnostic only) | None |
| 4. Tiered value dims | Future | ~2 days | Medium (escapes composite ceiling) | Medium — search-space blow-up |
| 5. Cross-benchmark ranking | Future | ~half day | Medium (better target choice for 30-bench) | Low |

---

## Validation order

1. Method 1 smoke test (job `47782739`) — confirms codegen is correct and BO at
   `enable=1, others=0` recovers composite peaks.
2. If pass: full Method-1 chain (5×8h) on `working_dir_bo_bestval/`,
   `top_k=14`, 6400 iter. Compare to RTP and to legacy BO (existing 18k DB
   in `working_dir_rtp_bo/`).
3. Implement Method 2; rerun chain on `working_dir_bo_bestval_cbo/`.
4. Compare all four runs (RTP, legacy BO, Method 1 BO, Method 1+2 BO) on the
   same 5 benchmarks. Expected ranking: M1+2 ≥ M1 > legacy BO ? RTP ≥ composite.
5. If M1+2 BO beats RTP on at least 3/5 benchmarks, scale up to the
   30-benchmark composite DB and rerun.
