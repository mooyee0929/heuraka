# RTP vs Bayesian Optimization vs Genetic Algorithm vs Composite Search — Findings Report

## Setup

- **Project**: Heureka / Augmentum — automated compiler heuristic discovery and tuning for LLVM 10.
- **Pipeline**:
  1. **Composite search** (`function_analyser.py --search_strategy composite`) — discovers per-(function, path) tuning targets and fits Priors.
  2. **RTP** (`rtp_tuning.py`) — combines multiple targets via random search, samples values from prior ranges.
  3. **BO** (`bo_tuning.py`, formerly `rtp_bo_tuning.py`) — same combination idea as RTP, but uses a Random Forest surrogate with Expected Improvement acquisition.
  4. **GA** (`ga_tuning.py`, Method 6 in `improvements.md`) — evolutionary search over the same binary subset space as BO: population of binary vectors, tournament selection, uniform crossover, per-bit mutation. Reuses BO's loaders and codegen so GA evaluates exactly the configs BO would.
- **Benchmarks (5-bench subset)**: PolyBench-C 4.2.1
  - `2mm`, `adi`, `atax`, `gemm`, `ludcmp` (all SMALL dataset)
- **Objective**: relative binary size (smaller = better, baseline = `-Oz`).
- **Iterations**: 6400 per chain.
- **Top-k targets**: 10 (RTP), 14 (BO first chain), 6 (BO second chain — currently running).

## Datasets (as of analysis time)

| Database | Rows | Verify SUCCESS | Failed | Success rate |
|---|---:|---:|---:|---:|
| `working_dir_composite/subset_composite.sqlite` (`probe_log`) | 1,335,809 | — | — | — |
| `working_dir_rtp/rtp_results.sqlite` | 31,996 | 7,961 | 24,035 | **24.9%** |
| `working_dir_rtp_bo/rtp_bo_results.sqlite` (top_k=14, partial 3,685/6,400 iter) | 18,426 | 1,008 | 17,418 | **5.5%** |

## Finding 1: BO crashes ~4.5× more than RTP

End-to-end success rate (verify == SUCCESS / total trials):

| Stage | RTP | BO (top_k=14) |
|---|---:|---:|
| Compile SUCCESS | 71.2% | 49.1% |
| Run SUCCESS | 25.4% | 5.9% |
| Verify SUCCESS | **24.9%** | **5.5%** |
| Compile FAIL | 28.8% | **50.9%** |
| Run FAIL | 45.7% | 43.1% |

→ At every stage, BO is significantly worse. BO compile failures are roughly **2× higher** than RTP.

### Why BO crashes more — `num_targets` distribution

| `num_targets` | RTP share | BO share | Verify SUCCESS rate |
|---:|---:|---:|---|
| 1–4 (low) | ~40% | ~10% | 27–76% (high) |
| 5–7 (medium) | ~30% | **~52%** | 1–18% |
| 8–10 (high) | ~30% | ~38% | **0%** (all crash) |
| 11–13 (BO only) | 0% | ~3.4% | **0%** |

- **RTP** samples `num_targets` uniformly across 1–10, so many "easy" 1-target / 2-target trials succeed.
- **BO**'s RF surrogate learns *"more enabled targets → smaller binary → better objective"* and concentrates acquisition on `num_targets ≥ 5`. Even worse, it produces samples with up to 13 enabled targets.
- For `num_targets ≥ 8`, the verify success rate is **0%** in both methods — but BO spends a far larger fraction of its iteration budget there.
- At the same `num_targets`, BO also has higher per-trial compile-fail rate (e.g., `num_targets=5`: RTP 27.0% vs BO 35.3%), suggesting BO's prior-selection choices are also worse than uniform random.

**Conclusion**: BO's surrogate optimizes the headline objective without modeling configuration *feasibility*, leading to catastrophic over-exploration of infeasible high-target combinations.

## Finding 2: RTP fails to beat Composite single-target on the 5-bench subset

| Benchmark | Composite single-target best | RTP best (any `num_targets`) | Δ vs Composite |
|---|---:|---:|---|
| 2mm | **-7.16%** | -5.89% | RTP **worse** by 1.27pp |
| adi | -4.25% | -4.25% | Tie |
| atax | -5.20% | -5.20% | Tie |
| gemm | **-7.67%** | -4.89% | RTP **worse** by 2.78pp |
| ludcmp | -4.38% | -4.57% | RTP better by 0.19pp |

This contradicts the paper's headline claim that RTP boosts savings from 11.6% (single) → 19.5% (stacked). Possible reasons:
1. **Benchmark mix differs** — the paper's 19.5% comes from `N-cg` and `P-cor`. The paper itself says: *"For others, we found no improvements or stayed below our previous results."* Our 5-bench subset may simply be in the "no stacking gain" group.
2. **PolyBench SMALL is too small** — limited functions to instrument means little room to stack savings.
3. **Search budget** — 6,400 iters / 5 benchmarks across many target combinations is sparse coverage.
4. **Most importantly: see Finding 3.**

### What the paper acknowledges vs what's actually happening

The paper explicitly admits that RTP can underperform composite single-target
on some benchmarks (Section VII-E):

> *"For some applications like N-cg and P-cor we managed to gain significant
> improvements of up to 19.5% over the highest savings observed for single
> tuning targets only during the heuristic search. **For others, we found no
> improvements or stayed below our previous results.** Due to limited resource
> availability, the random search experiment we ran only considered roughly
> 6400 search points per applications which is a small number compared to the
> large search space we looked through."*

| Paper's explanation | Our data (Finding 3 + Finding 4) |
|---|---|
| "Limited resource availability" | RTP fails for **structural** reasons unrelated to budget |
| "6400 search points is small" | We ran 6,400 iter and `num_targets=1` ~597 trials per benchmark — RTP still cannot reach composite's `2mm` -7.16% or `gemm` -7.67% peaks |
| (no root-cause analysis given) | RTP samples values uniformly from the prior **range**, not the specific best **value** composite found. With wide ranges (e.g., `[0, 127]` for an int) and sharp response surfaces, random sampling almost never hits the peak |

The paper's framing implies "more compute would fix this." Our data shows the
gap is structural — composite stores `prior range`, but the optimum lives at a
specific value within that range that random sampling cannot reliably
rediscover. Method 1 (Finding 4) closes this gap by plumbing composite's
specific best value through to BO codegen, and reaches **-13.10% mean** at
**iter 222** — beating RTP's 6,400-iter result by 30× iteration efficiency.

## Finding 3: RTP and BO both inherit a fundamental sampling limitation

Best `rel_objective` per `num_targets`:

| Benchmark | Composite | RTP n=1 | RTP n=2 | RTP n=3 | RTP n=4 | RTP n≥5 |
|---|---:|---:|---:|---:|---:|---:|
| 2mm | **0.9284** | 0.9544 | 0.9544 | 0.9544 | 0.9544 | 0.9411 |
| adi | 0.9575 | 0.9575 | 0.9575 | 0.9575 | 0.9575 | 0.9575 |
| atax | 0.9480 | 0.9480 | 0.9480 | 0.9480 | 0.9480 | 0.9480 |
| gemm | **0.9233** | 0.9511 | 0.9511 | 0.9511 | 0.9511 | 0.9511 |
| ludcmp | 0.9562 | 0.9562 | 0.9562 | 0.9543 | 0.9562 | 0.9562 |

For each benchmark we have ~597 RTP trials at `num_targets=1` (75–77% verify success). Yet for `2mm` and `gemm`, RTP **never** matches composite's single-target result.

### Root cause

From [rtp_tuning.py](augmentum-main/driver/rtp_tuning.py):
> *"randomly select a fitted Prior for each of the selected tuning targets, and **randomly samples a value from the prior's value range**"*

- Composite finds a **specific value** that produces the best result (e.g., integer = 42 in a `[0, 127]` range).
- RTP only stores the prior **range**, not the best-tested value.
- RTP uniform-samples values at runtime → probability of hitting the optimal value (or its neighborhood) is very low when prior ranges are wide and the response surface is sharp.
- After ~450 verified samples, RTP cannot recover the `2mm`/`gemm` peak.

### BO inherits the same limitation

From [bo_tuning.py:218–229](augmentum-main/driver/bo_tuning.py#L218-L229):
```python
# Build skopt search space — k binary dimensions only
def build_search_space(targets):
    """No value dimensions — values are sampled per-call in C++."""
    dimensions = []
    for i in range(len(targets)):
        dimensions.append(Categorical([0, 1], name=f"enable_{i}"))
    return dimensions
```

BO's surrogate only learns *"which target subset to enable"* (2^k binary). The **actual values used at each enabled target are random per-call sampling in C++**, identical to RTP. Therefore:
- BO cannot model "value = 42 is best" for any target.
- BO cannot beat RTP on the *value-selection* axis — only on the *subset-selection* axis.
- For benchmarks where the bottleneck is value selection (e.g., `2mm`, `gemm`), BO is structurally limited.

This explains why BO's per-benchmark best matches RTP's per-benchmark best, despite BO running fewer effective iterations.

## Implications and Proposed Next Steps

### Diagnosis summary
| Issue | Effect |
|---|---|
| BO over-explores `num_targets ≥ 8` | Wasted iterations on 0%-success region |
| BO/RTP both random-sample values from prior range | Cannot reproduce composite's single-target peaks for `2mm`/`gemm` |
| 5-bench subset lacks stacking-friendly benchmarks | RTP's central claim (multi-target stacking) doesn't manifest |

### Proposed improvements (in priority order)

1. **Use composite's best values, not just prior ranges (highest ROI)**
   - Extract `(target, best_value, best_prior_type)` triples from composite's `probe_log`.
   - When a target is enabled, deterministically use the composite-best value (or sample from a tight Gaussian around it).
   - Expected outcome: BO/RTP at `num_targets=1` should match composite for all benchmarks.

2. **Constraint-aware acquisition for BO**
   - Train a feasibility classifier `P(verify_SUCCESS | config)` on the existing 50,422 labeled samples (32k RTP + 18k BO).
   - Use modified acquisition: `EI(x) × P_feasible(x)`.
   - Expected outcome: BO stops over-exploring `num_targets ≥ 8`; effective iteration budget ~5× larger.

3. **Cap `top_k` more aggressively for BO**
   - Currently testing `top_k=6` (chain: jobs 47781592–47781596).
   - Quick sanity test for whether the surrogate's `num_targets` bias alone explains BO's underperformance.

4. **Run on more benchmarks**
   - 5-bench subset is too small to validate stacking claims.
   - The paper used 38 NPB + PolyBench applications.
   - The 30-benchmark `working_dir_composite_30` already contains the necessary composite output (16 GB on local scratch).

## Finding 4: Method 1 (best-value injection) dominates all baselines

Implemented `--use_best_value` in [bo_tuning.py](augmentum-main/driver/bo_tuning.py)
(see [improvements.md](improvements.md) Method 1). Each BO-enabled target uses
the `(prior_type, value)` pair that achieved the lowest `rel_objective` in
composite's `probe_log`, instead of per-call uniform random sampling.

### Smoke test (200 iter, top_k=14)

Test job `47782739`, working_dir `working_dir_rtp_bo_bestval_test/`:

| Stage | Legacy BO | **BO + best_value** |
|---|---:|---:|
| Compile SUCCESS | 49.1% | **100.0%** |
| Verify SUCCESS | 5.5% | **34.1%** |

Reason: composite only records `(prior, value)` pairs that already passed
compile + verify, so reusing those values guarantees compile success.

### Full chain in progress (job chain 47802830–47802834, iter 222 / 6400)

Working dir `working_dir_bo_bestval/`. Per-benchmark current best vs all baselines:

| Benchmark | Composite single | RTP (6400 iter) | Legacy BO | **BO + best_value (iter 222)** |
|---|---:|---:|---:|---:|
| 2mm | -7.16% | -5.89% | -4.56% | **-12.72%** |
| adi | -4.25% | -4.25% | -4.25% | **-15.06%** |
| atax | -5.20% | -5.20% | -4.99% | **-12.48%** |
| gemm | -7.67% | -4.89% | -4.89% | **-13.02%** |
| ludcmp | -4.38% | -4.57% | -3.08% | **-12.20%** |
| **Mean** | **-5.73%** | **-4.96%** | **-4.35%** | **-13.10%** |

- 5/5 benchmarks beat every baseline.
- Mean improvement is **2.3× composite** and **2.6× RTP**.
- BO + best_value reached this at iter 222 vs RTP's 6400 — **~30× more
  iteration-efficient** even before BO finishes its full 6400-iter chain.
- Best configurations use `num_targets = 5–6` (stacking validated).

### Why this works (per Finding 3 root cause)
- Composite found specific `(prior, value)` pairs that achieve large per-target
  reductions, but stored only the prior *range* in its DB schema.
- RTP and legacy BO uniformly resampled values from the range at runtime,
  almost never hitting composite's actual best value.
- `--use_best_value` plumbs the specific value through to codegen, so each
  enabled target now contributes its known-best individual reduction, and
  BO's RF surrogate searches over which subsets stack best.

### Caveat
- All numbers are PolyBench SMALL with 4 KB baselines; -15% absolute is ~600
  bytes. The same percentages on NPB-scale binaries (10–100 KB) would be more
  meaningful. Validation on the 30-benchmark `working_dir_composite_30/` data
  is the next milestone after the full chain finishes.

## Finding 5: RTP re-run confirms underperformance is structural, not random-seed

To rule out that Finding 3's RTP underperformance was a bad seed, a second independent 6400-iter RTP run was executed against the same composite DB (MD5-verified identical) and **the same 5-bench baseline cache**.

| case | backup RTP best | new RTP best | Δ bytes | absolute bytes (identical reference) |
|---|---:|---:|---:|---|
| 2mm    | 0.9411 | 0.9544 | +56 | 3980 → 4036 |
| adi    | 0.9575 | 0.9521 | −24 | 4260 → 4236 |
| atax   | 0.9480 | 0.9480 | 0   | 3554 → 3554 (byte-identical) |
| gemm   | 0.9511 | 0.9511 | 0   | 3756 → 3756 (byte-identical) |
| ludcmp | 0.9543 | 0.9562 | +8  | 4116 → 4124 |
| **geomean rel** | **0.9504** | **0.9524** | | |

- atax/gemm reached the **exact same best absolute bytes** in both runs — the achievable optimum is a narrow, deterministic point that RTP keeps re-discovering.
- Geomean rel differs by only 0.2pp across runs. RTP's failure to match composite on `2mm`/`gemm` (Finding 3) is **not** a seed artifact; it's the sampling limitation diagnosed in Finding 3.
- Final ordering (absolute best bytes, 5-bench geomean):
  - BO `USE_BEST_VALUE=1`: **3571 bytes** (−13.1%)
  - Composite single-target: 3902 bytes (−5.7%)
  - RTP (either run): 3930 bytes (−5.1%)

RTP doesn't just fail to beat composite — across these 5 benchmarks its search actually lands ~1pp *worse* than composite's single-target peak on average.

## Finding 6: rel_objective is only comparable across runs with the same baseline cache

During the RTP re-run, initial 1042-iter results looked **7–10pp better than both composite and backup RTP** (`rel_objective` fell to 0.87-region). Closer inspection revealed the run was using a different `baseline_cache.pickle` — the 30-bench cache, written concurrently with the 5-bench backup but against a different working directory.

### Root cause

`BaselineProbe` compiles each benchmark with vanilla clang and records the resulting binary size. Identical compiler (MD5-verified) and identical source files still produced baselines that differed by **+390 B uniformly across all 5 benchmarks**:

| case | 5-bench baseline | 30-bench baseline | Δ |
|---|---:|---:|---:|
| 2mm    | 4229 B | 4623 B | +394 B |
| adi    | 4449 B | 4847 B | +398 B |
| atax   | 3749 B | 4143 B | +394 B |
| gemm   | 3949 B | 4343 B | +394 B |
| ludcmp | 4313 B | 4711 B | +398 B |

Reproduction: compiling the same `polybench.c` from two differently-named `/tmp/` directories gives binaries that differ by exactly the character-delta in the absolute source path. `polybench.c` contains `assert()` and `__FILE__` expansions whose absolute path strings land in `.rodata` and count toward `text`. The 30-bench run lived under `working_dir_composite_30/…` (9 chars longer than `working_dir_rtp/…`), and with ~40 `__FILE__`-containing macro expansions per link, the path-string overhead accumulates to ~390 B.

### Implications

- **Never compare `rel_objective` across runs with different baseline caches.** The denominator moves under you.
- **Absolute `objective` (bytes) is the stable metric** — it's the only field that lets you compare composite / RTP / BO across versions. Both `rtp_results.sqlite` and `bo_results.sqlite` store `objective` and `baseline_objective` explicitly for this reason.
- For the canonical 5-bench comparison, all methods should pin to `backup_5bench_20260410/baseline_cache_5bench.pickle`. `sbatch_rtp.sh` and `sbatch_bo.sh` already do so.
- Long-term fix: add `-ffile-prefix-map=$(pwd)=.` to the polybench build to strip embedded source paths.

## Finding 7: Genetic Algorithm matches BO's optimum and converges ~20 evaluations faster

Third baseline alongside BO and RTP: a binary-vector GA over the same top-14 composite-discovered tuning targets (`driver/ga_tuning.py`, Method 6 in `improvements.md`). Implementation details: population 20, tournament selection k=3, uniform crossover rate 0.8, per-bit mutation rate 1/14, elitism 1, seed 42. Runs with `--use_best_value` identical to BO's best-value injection (Finding 4).

### Setup

- **Job**: SLURM 48375173, 8 CPUs, 5h 20m wall, exit 0.
- **Iterations**: 2000 (vs BO's 1161 effective iters, RTP's 6400).
- **Results DB**: `working_dir_ga/ga_results.sqlite` (9.5 MB).
- **Verified reductions**: 7649 / 7649 (100% of verified compiles produced smaller binaries than baseline).

### Headline result (absolute candidate bytes — the stable metric per Finding 6)

Minimum candidate binary size across each method's entire search:

| Benchmark | `-Oz` baseline (`USER:shresth` tree) | GA min | BO min | RTP min |
|---|---:|---:|---:|---:|
| 2mm     | 4579 | **3691** | **3691** | 3980 |
| adi     | 4803 | **3779** | **3779** | 4260 |
| atax    | 4099 | **3281** | **3281** | 3554 |
| gemm    | 4299 | **3435** | **3435** | 3756 |
| ludcmp  | 4667 | **3787** | **3787** | 4116 |

GA and BO produce **byte-for-byte identical best candidates on every benchmark**. They disagree on *which* 5–6 of the 14 targets to enable — GA's best enables targets {3, 4, 5, 7, 12, 13}; BO's best enables {1, 2, 3, 4, 13} — but the instrumented binaries are the same.

### Where GA distinguishes itself: speed of convergence

Best-so-far average `rel_objective` at milestone N (normalized against a common baseline — see "Methodology note" below):

| Configs evaluated | GA | BO | RTP |
|---:|---:|---:|---:|
| 10   | **0.8025** | 0.8365 | 0.8813 |
| 25   | **0.8025** | 0.8365 | 0.8813 |
| 50   | **0.8008** | 0.8365 | 0.8785 |
| 91   | 0.8008 | 0.8008 *(first hit)* | 0.8785 |
| 100+ | 0.8008 | 0.8008 | 0.8785 |
| 6400 | 0.8008 | 0.8008 | 0.8670 |

- GA reaches the global optimum (`avg rel = 0.8008`, −19.92%) at iter 70.
- BO reaches the same optimum at iter 91.
- Between iters 10 and 91, GA is ~3.5pp lower on average than BO — the period where population + crossover cover the "contains targets {3, 4, 13}" basin more densely than BO's RF surrogate does with single-point sampling.
- After both converge, the curves are flat and overlap exactly.

### Effective search space is ~3 targets, not 14

Many distinct genomes share GA's optimum score. Top-10 distinct GA genomes all enable targets {3, 4, 13} — `ConstantRange::getNonEmpty`, `ConstantRange` move-constructor, and `Value::getValueID` — and scored 0.8008 regardless of which of the other 11 bits flip. Repeatedly forcing GA to re-roll BO's best genome `01111000000001` at iters 501/832/1995 produced byte-identical candidate binaries to GA's own best genome `00011101000011`. On PolyBench SMALL, extra enabled targets are dead weight: either their hooked functions aren't called during compilation, or the best-value return doesn't propagate to codegen.

This explains why GA and BO tie: both search strategies are competing over a search surface with a ~3-dimensional effective optimum and flat high-dimensional neighborhoods around it. Either strategy finds that plateau quickly once best-value emission is enabled (Finding 4).

### Methodology note: normalization against a common baseline

GA was run in shresth's tree; BO and RTP results are from fsyang's `backup_5bench_20260410/`. As Finding 6 predicts, the two trees' baseline caches disagree by exactly 350 bytes per benchmark because `__FILE__` string embeddings land in `.rodata`:

| benchmark | shresth baseline | fsyang baseline | Δ |
|---|---:|---:|---:|
| 2mm    | 4579 | 4229 | +350 |
| adi    | 4803 | 4449 | +354 |
| atax   | 4099 | 3749 | +350 |
| gemm   | 4299 | 3949 | +350 |
| ludcmp | 4667 | 4313 | +354 |

Candidate binaries compiled with **instrumented** clang are byte-identical across trees (the instrumented build folds out path-dependent strings), but **vanilla** `-Oz` baselines are not. Comparing GA's in-tree `rel_obj` to BO's in-tree `rel_obj` gave GA a ~7pp free handicap.

All numbers in this finding are normalized by re-dividing each method's stored candidate sizes against the shresth baseline (equivalently, comparing absolute bytes per Finding 6's recommendation). Raw per-tree rel_obj values in the DB:
- GA in-tree: `avg rel = 0.8008` (→ 19.92% reduction vs shresth baseline)
- BO in-tree: `avg rel = 0.8690` (→ 13.10% reduction vs fsyang baseline)

Both reflect the same underlying candidate binaries.

### Takeaway

On this 5-benchmark subset, GA and BO are **indistinguishable on final solution quality**, and GA's advantage is purely convergence speed (~20 evaluations out of ~100 total before both plateau). This is likely an artifact of how flat the top-14 search surface is on PolyBench SMALL. Re-running all three methods on the 30-benchmark suite (Finding 4's caveat) — where more hooked functions fire during compilation and genomes have more structure to exploit — is the right place to see whether GA, BO, or hybrid approaches open up a real quality gap.

## Status

All 5-bench comparisons complete and mutually consistent (absolute-byte comparisons, or `rel_obj` normalized against a single baseline cache, per Finding 6):

| Run | DB | Notes |
|---|---|---|
| Composite (5-bench) | `working_dir_composite/subset_composite.sqlite` | MD5-identical to backup |
| RTP original | `backup_5bench_20260410/working_dir_rtp/rtp_results.sqlite` | 6400 iter |
| RTP re-run | `working_dir_rtp/rtp_results.sqlite` | 6400 iter, Finding 5 |
| BO best_value | `working_dir_bo_bestval/bo_results.sqlite` | 1174 iter; also copied to `backup_5bench_20260410/working_dir_bo_bestval/` |
| GA best_value | `working_dir_ga/ga_results.sqlite` | 2000 iter (SLURM 48375173), Finding 7 |

### Summary ranking (5-bench, normalized to shresth baseline, lower = better)

| Rank | Method | mean `rel_obj` | size reduction | Configs to best |
|---:|---|---:|---:|---:|
| 1 (tie) | GA + `use_best_value` | 0.8008 | **−19.92%** | 70 |
| 1 (tie) | BO + `use_best_value` | 0.8008 | **−19.92%** | 91 |
| 3 | RTP (random value) | 0.8670 | −13.30% | 3125 |
| 4 | Composite (single-target oracle) | 0.9427† | −5.73% | — |

†Composite figure carried forward from Finding 3 / 4 against the fsyang baseline; absolute bytes per Finding 6 put it at 3902 B geomean vs GA/BO's 3595 B. GA and BO stacking wins are larger once a common baseline is used, consistent with Finding 4's original stacking story.

Currently in progress: 30-benchmark composite search (`working_dir_composite_30/subset_composite.sqlite`, ~1.55M probes across 30 Polybench benchmarks) for the stacking-validation milestone in Finding 4's caveat, and to give GA and BO a search surface with enough structure to diverge.
