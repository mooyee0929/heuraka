# Heureka — Compiler Heuristic Discovery

Automated discovery and evaluation of LLVM 10 compiler heuristics for binary size reduction. Extends [Augmentum](https://github.com/bfranke1973/Heureka) with Random Tuning Profiles (RTP) and Bayesian Optimization (BO) multi-target search.

---

## Pipeline

```
           benchmarks (PolyBench, NPB, …)
                      │
            ┌─────────▼─────────┐
  Stage 1   │ Composite search  │   function_analyser.py
            │  (per-function)   │ → subset_composite.sqlite
            └─────────┬─────────┘   (best value per target)
                      │
            ┌─────────▼─────────┐
  Stage 2   │ BO / RTP (subset  │   bo_tuning.py / rtp_tuning.py
            │  selection)       │ → bo_results.sqlite
            └─────────┬─────────┘   (best multi-target config)
                      │
                    results
```

Run Stage 1 once per benchmark set. Stage 2 then searches over *which subset of Stage-1 targets to enable*.

---

## Prerequisites

- **LLVM 10** built at `llvm10/` (instrumented build under `llvm10/build/instrumented/`)
- **Conda env** `augmentum` with `scikit-optimize`:
  ```bash
  conda activate augmentum
  pip install scikit-optimize
  ```
- **`evaluation_config.json`** with tool paths (already configured at [augmentum-main/evaluation_config.json](augmentum-main/evaluation_config.json))

---

## Running on SLURM (GreatLakes)

All `sbatch_*.sh` scripts are in `augmentum-main/driver/scripts/`. Each job is capped at 8h — chain multiple jobs with `--dependency=singleton` to continue past iteration budgets that exceed 8h.

### Account selection (important!)

The default account `cse583w26_class` may hit `AssocGrpBillingMinutes` quota. Use **`eecs504s001w26_class`** when submitting:

```bash
sbatch --account=eecs504s001w26_class sbatch_bo.sh ...
```

### Stage 1 — Composite search (find per-target best values)

Only run this if you don't already have `subset_composite.sqlite` for your benchmark set. Backups exist at [backup_5bench_20260410/working_dir_composite/](backup_5bench_20260410/working_dir_composite/).

```bash
cd augmentum-main/driver/scripts
sbatch --account=eecs504s001w26_class sbatch_evaluation.sh
```

**Output**: `working_dir_composite_30/subset_composite.sqlite` (≈500MB; the `probe_log` table holds per-function `(prior, value, rel_objective)` triples).

### Stage 2 — Bayesian Optimization (recommended)

```bash
# Single job (8h time limit)
USE_BEST_VALUE=1 \
WORKING_DIR=/scratch/.../heuraka/working_dir_bo_bestval \
sbatch --account=eecs504s001w26_class sbatch_bo.sh 6400 14
#                                                   ^iterations  ^top_k
```

**Arguments**:

| Arg | Meaning |
|---|---|
| `iterations` | BO loop budget (default 2000; 1000–1500 is usually enough) |
| `top_k` | Number of top Stage-1 targets to consider (default 10; use 14 for broader search) |
| `USE_BEST_VALUE=1` | **Recommended.** Reuses Stage-1's best value per enabled target (deterministic) instead of re-sampling random values at runtime |

**Output**: `${WORKING_DIR}/bo_results.sqlite`
- `bo_results` — per-iteration, per-testcase objective
- `bo_configs` — which targets were enabled each iteration

### Chaining jobs (for >8h runs)

```bash
# Submit 5 dependent jobs — each resumes where the previous left off
for i in {1..5}; do
    USE_BEST_VALUE=1 WORKING_DIR=/scratch/.../heuraka/working_dir_bo_bestval \
    sbatch --account=eecs504s001w26_class --dependency=singleton \
        sbatch_bo.sh 6400 14
done
```

Or use the helper:

```bash
./submit_chain.sh sbatch_bo.sh 5 6400 14
```

**Resume is automatic** — the script loads `bo_results.sqlite` and continues from `max(iteration)+1`.

### Stage 2 alternative — RTP (random search)

```bash
sbatch --account=eecs504s001w26_class sbatch_rtp.sh 6400 10
# Output: working_dir_rtp/rtp_results.sqlite
```

BO (with `USE_BEST_VALUE=1`) dominates RTP by ~2.6× mean improvement on this workload; RTP is mostly useful as a paper-faithful baseline. See [report.md](report.md) for absolute-byte comparisons.

---

## Setup optimization (auto)

The LLVM instrument step takes ~90 min per fresh chain job. The pipeline automatically caches this via `llvm10/build/instrumented/.instrumented_targets.json`:
- First job: full rebuild (~90 min) + marker written
- Subsequent jobs with same target set: **skips rebuild** (~3 min setup)

If you change `top_k` or the target set, the marker is invalidated and a rebuild kicks in.

---

## Monitoring

```bash
# Jobs
squeue -u $USER -o "%.10i %.20j %.8T %.10M %R"

# Tail live log
tail -f augmentum-main/driver/scripts/slurm-<JOB_ID>.out

# Progress
sqlite3 ${WORKING_DIR}/bo_results.sqlite \
    "SELECT MAX(iteration) FROM bo_results;"

# Best full-verified config so far
sqlite3 ${WORKING_DIR}/bo_results.sqlite <<'SQL'
SELECT iteration, AVG(rel_objective) AS avg_ro, COUNT(*) n
FROM bo_results
WHERE verify_result='SUCCESS' AND rel_objective IS NOT NULL
GROUP BY iteration HAVING n = 5
ORDER BY avg_ro ASC LIMIT 5;
SQL
```

`rel_objective < 1.0` = binary is smaller than `-Oz` baseline. Lower is better.

---

## Results locations

| Dir | Contents |
|---|---|
| `working_dir_bo_bestval/` | BO (`USE_BEST_VALUE=1`) final — `bo_results.sqlite` (1174 iter × 5 bench) |
| `working_dir_rtp/` | RTP re-run with 5-bench baseline — `rtp_results.sqlite` (6400 iter × 5 bench) |
| `working_dir_composite_30/` | 30-benchmark composite output (in progress, ~1.55M probes) |
| `backup_5bench_20260410/working_dir_composite/` | Reference 5-benchmark composite DB (identical MD5 to `working_dir_composite/`) |
| `backup_5bench_20260410/working_dir_rtp/` | Original RTP run — reference for cross-validation |
| `backup_5bench_20260410/working_dir_bo_bestval/` | BO results backup |
| `backup_5bench_20260410/baseline_cache_5bench.pickle` | Cached `-Oz` baseline sizes for 5-bench (**use this across all Stage-2 methods**) |

### Baseline cache consistency (gotcha)

`rel_objective` in any `*_results.sqlite` is `candidate_size / baseline_from_cache`. If different runs use different `baseline_cache.pickle` files, rel numbers look dramatically different even when the **absolute** binary sizes are identical.

Root cause: the absolute path of `polybench.c` gets embedded into the binary via `__FILE__` / assert macros. Different `working_dir_*` names (e.g. `working_dir_rtp` vs `working_dir_composite_30`) produce baselines that differ by ~400 bytes uniformly across all 5 benchmarks (~9-char path diff × ~40 `__FILE__` expansions).

**Always pin `--baseline_cache` to `backup_5bench_20260410/baseline_cache_5bench.pickle`** for 5-bench comparisons — `sbatch_rtp.sh` and `sbatch_bo.sh` already point here. To eliminate the embedding entirely, add `-ffile-prefix-map=$(pwd)=.` to `compiler_flags`.

---

## Common issues

- **`AssocGrpBillingMinutes`** pending: switch to `--account=eecs504s001w26_class`.
- **Job crashes resuming at iter N**: the `enabled` column was historically stored as `numpy.int64` bytes; the resume decoder in [bo_tuning.py](augmentum-main/driver/bo_tuning.py) handles both BLOB and INTEGER rows transparently.
- **Setup takes 90 min every job**: marker at `llvm10/build/instrumented/.instrumented_targets.json` was deleted or invalidated. Safe to run — will rebuild once and re-cache.
- **`working_dir_bo/` doesn't exist**: default path in `sbatch_bo.sh`; always set `WORKING_DIR=...` explicitly to the run directory you want to resume (e.g. `working_dir_bo_bestval`).

---

## Analysis

- [report.md](report.md) — findings, method comparison, current best config
- [improvements.md](improvements.md) — implementation notes for the BO enhancements