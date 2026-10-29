# Heureka — Compiler Heuristic Discovery

Automated discovery and evaluation of LLVM 10 compiler heuristics for binary size reduction. Extends [Augmentum](https://github.com/bfranke1973/Heureka), the framework behind *Revealing Compiler Heuristics Through Automated Discovery and Optimization* (Seeker et al., CGO 2024 — [PDF](Compiler%20Heuristics%20Discovery%202024.pdf)), with multi-target search over the discovered tuning targets: Random Tuning Profiles (RTP), Bayesian Optimization (BO), a Genetic Algorithm (GA) and Simulated Annealing (SA), plus composite best-value injection (`--use_best_value`).

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
  Stage 2   │ BO / GA / RTP / SA│   bo_tuning.py / ga_tuning.py /
            │ (subset selection)│   rtp_tuning.py / sa_tuning.py
            └─────────┬─────────┘ → {bo,ga,rtp,sa}_results.sqlite
                      │             (best multi-target config)
                    results
```

Run Stage 1 once per benchmark set. Stage 2 then searches over *which subset of Stage-1 targets to enable*.

| Stage-2 method | Script | Search style |
|---|---|---|
| RTP | `rtp_tuning.py` | Random subset per iteration (paper Section VII-E baseline) |
| BO | `bo_tuning.py` | `skopt` Random Forest surrogate + Expected Improvement over k binary on/off dimensions |
| GA | `ga_tuning.py` | Tournament selection, uniform crossover, per-bit mutation over the same k binary dimensions as BO |
| SA | `sa_tuning.py` | Neighbor moves on the current configuration with a cooling schedule |

All four accept `--use_best_value`: each enabled target applies the exact `(prior, value)` pair that gave Stage 1 its best result, instead of re-sampling a random value from the prior's range on every call.

---

## Results (5-benchmark PolyBench subset)

`2mm`, `adi`, `atax`, `gemm`, `ludcmp` (SMALL dataset), objective = binary size relative to `-Oz`. GA, BO and RTP are normalized against one baseline cache; the composite row is carried over from an earlier baseline (see report.md, Status):

| Rank | Method | mean `rel_obj` | Size reduction | Configs to best |
|---:|---|---:|---:|---:|
| 1 (tie) | GA + `use_best_value` | 0.8008 | −19.92% | 70 |
| 1 (tie) | BO + `use_best_value` | 0.8008 | −19.92% | 91 |
| 3 | RTP (random value) | 0.8670 | −13.30% | 3125 |
| 4 | Composite (single-target oracle) | 0.9427 | −5.73% | — |

Best-value injection is the main win: it lifts BO's compile success from 49.1% to 100% and lets it stack 5–6 targets. GA and BO land on byte-identical best binaries; GA gets there in fewer evaluations. Full analysis in [report.md](report.md).

---

## Prerequisites

- **LLVM 10** built at `llvm10/` (instrumented build under `llvm10/build/instrumented/`). See [augmentum-main/INSTALL.md](augmentum-main/INSTALL.md) for the LLVM build flags (RTTI + EH) and for building the Augmentum libraries:
  ```bash
  cd augmentum-main
  make all LLVM_DIR=/path/to/llvm/install/lib/cmake/llvm/
  ```
- **Conda env** `augmentum` with `scikit-optimize`:
  ```bash
  conda env create --file augmentum-main/environment.yml
  conda activate augmentum
  pip install scikit-optimize
  ```
- **`evaluation_config.json`** with tool paths ([augmentum-main/evaluation_config.json](augmentum-main/evaluation_config.json), the config every job script passes to the driver; a template lives at [augmentum-main/driver/config/evaluation_config.json.template](augmentum-main/driver/config/evaluation_config.json.template)). The committed configs use `/path/to/heuraka/...` placeholders (and `/path/to/augmentum/build/...` for the build outputs in `driver/config/evaluation_config.json`). The driver reads these values literally, with no environment-variable expansion, so replace the placeholders with absolute paths in your checkout before running.
- **`HEURAKA_ROOT`** exported to the absolute path of this checkout (the directory that contains `augmentum-main/`, `llvm10/` and the `working_dir_*` / `backup_*` run directories):
  ```bash
  export HEURAKA_ROOT=/abs/path/to/heuraka
  ```
  Every job script derives its paths from it and exits immediately with an error if it is unset. `sbatch` forwards the submitting shell's environment to the job by default.

---

## Running on SLURM

All `sbatch_*.sh` scripts are in `augmentum-main/driver/scripts/`. Each job is capped at 8h — chain multiple jobs with `--dependency=singleton` to continue past iteration budgets that exceed 8h.

Paths come from `HEURAKA_ROOT` (see Prerequisites):

| Script | Paths derived from `${HEURAKA_ROOT}` |
|---|---|
| `sbatch_evaluation.sh` | `augmentum-main/`, `working_dir_composite_30/` |
| `sbatch_bo.sh` | `augmentum-main/`, `working_dir_bo/` (default `WORKING_DIR`), `backup_5bench_20260410/` composite DB and baseline cache |
| `sbatch_rtp.sh` | `augmentum-main/`, `working_dir_rtp/`, `working_dir_composite/` composite DB, `backup_5bench_20260410/` baseline cache |
| `sbatch_sa.sh` | `augmentum-main/`, `working_dir_sa/`, `backup_5bench_20260410/` composite DB and baseline cache |
| `sbatch_ga.sh` | `working_dir_ga/` (default `WORKING_DIR`), `backup_5bench_20260410/` composite DB (default `COMPOSITE_DB`) |
| `run_evaluation.sh` | `augmentum-main/`, `working_dir/` |
| `augmentum-main/build_augmentum.sh` | `augmentum-main/`, `llvm10/install/lib/cmake/llvm` |

`sbatch_bo.sh` also reads `WORKING_DIR` from the environment. `sbatch_ga.sh` finds `AUGMENTUM_HOME` from the submit directory, accepts `AUGMENTUM_HOME`, `WORKING_DIR`, `COMPOSITE_DB`, `BASELINE_CACHE` and `CONDA_BASE` overrides, needs `HEURAKA_ROOT` only when `WORKING_DIR` or `COMPOSITE_DB` is unset, and defaults `BASELINE_CACHE` to `augmentum-main/driver/config/baseline_cache_${USER}_5bench.pickle`. Submit `build_augmentum.sh` from `augmentum-main/` so its `build_augmentum_<jobid>.out/.err` logs land there.

### Stage 1 — Composite search (find per-target best values)

Only run this if you don't already have `subset_composite.sqlite` for your benchmark set.

```bash
cd augmentum-main/driver/scripts
sbatch sbatch_evaluation.sh
```

**Output**: `working_dir_composite_30/subset_composite.sqlite` (≈500MB; the `probe_log` table holds per-function `(prior, value, rel_objective)` triples).

### Stage 2 — Bayesian Optimization (recommended)

```bash
# Single job (8h time limit)
USE_BEST_VALUE=1 \
WORKING_DIR=${HEURAKA_ROOT}/working_dir_bo_bestval \
sbatch sbatch_bo.sh 6400 14
#                   ^iterations  ^top_k
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
    USE_BEST_VALUE=1 WORKING_DIR=${HEURAKA_ROOT}/working_dir_bo_bestval \
    sbatch --dependency=singleton sbatch_bo.sh 6400 14
done
```

Or use the helper (chains with `--dependency=afterany`):

```bash
./submit_chain.sh sbatch_bo.sh 5 6400 14
```

**Resume is automatic** — the script loads `bo_results.sqlite` and continues from `max(iteration)+1`. RTP, GA and SA resume from their own result databases the same way.

### Stage 2 alternative — Genetic Algorithm

```bash
USE_BEST_VALUE=1 sbatch sbatch_ga.sh 2000 14 20
#                                    ^iterations ^top_k ^pop_size
# Output: ${WORKING_DIR}/ga_results.sqlite (default working_dir_ga/)
```

Same search space, loaders and code generation as BO; only the optimizer differs. Population 20, tournament k=3, crossover rate 0.8, mutation rate 1/k and elitism 1 by default (`--pop_size`, `--tournament_size`, `--crossover_rate`, `--mutation_rate`, `--elitism`, `--seed` on `ga_tuning.py`).

### Stage 2 alternative — RTP (random search)

```bash
sbatch sbatch_rtp.sh 6400 10
#                    ^iterations ^max_targets
# Output: working_dir_rtp/rtp_results.sqlite
```

BO (with `USE_BEST_VALUE=1`) dominates RTP by ~2.6× mean improvement on this workload; RTP is mostly useful as a paper-faithful baseline. See [report.md](report.md) for absolute-byte comparisons.

### Stage 2 alternative — Simulated Annealing

```bash
sbatch sbatch_sa.sh 6400 10
#                   ^iterations ^max_targets
# Output: working_dir_sa/sa_results.sqlite
```

The cooling schedule is set in the script (`TEMP_INIT=1.0`, `TEMP_MIN=0.01`, `COOLING_RATE=0.995`).

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

Working directories are created on the cluster at run time and are git-ignored (`working_dir_*/`, `backup_*/`), so they are not part of this repository.

| Dir | Contents |
|---|---|
| `working_dir_bo_bestval/` | BO (`USE_BEST_VALUE=1`) final — `bo_results.sqlite` (1174 iter × 5 bench) |
| `working_dir_ga/` | GA (`USE_BEST_VALUE=1`) — `ga_results.sqlite` (2000 iter × 5 bench) |
| `working_dir_rtp/` | RTP re-run with 5-bench baseline — `rtp_results.sqlite` (6400 iter × 5 bench) |
| `working_dir_composite_30/` | 30-benchmark composite output (in progress, ~1.55M probes) |
| `backup_5bench_20260410/working_dir_composite/` | Reference 5-benchmark composite DB (identical MD5 to `working_dir_composite/`) |
| `backup_5bench_20260410/working_dir_rtp/` | Original RTP run — reference for cross-validation |
| `backup_5bench_20260410/working_dir_bo_bestval/` | BO results backup |
| `backup_5bench_20260410/baseline_cache_5bench.pickle` | Cached `-Oz` baseline sizes for 5-bench (**use this across all Stage-2 methods**) |

### Baseline cache consistency (gotcha)

`rel_objective` in any `*_results.sqlite` is `candidate_size / baseline_from_cache`. If different runs use different `baseline_cache.pickle` files, rel numbers look dramatically different even when the **absolute** binary sizes are identical.

Root cause: the absolute path of `polybench.c` gets embedded into the binary via `__FILE__` / assert macros. Different `working_dir_*` names (e.g. `working_dir_rtp` vs `working_dir_composite_30`) produce baselines that differ by ~400 bytes uniformly across all 5 benchmarks (~9-char path diff × ~40 `__FILE__` expansions).

**Always pin `--baseline_cache` to `backup_5bench_20260410/baseline_cache_5bench.pickle`** for 5-bench comparisons — `sbatch_rtp.sh`, `sbatch_bo.sh` and `sbatch_sa.sh` already point here. To eliminate the embedding entirely, add `-ffile-prefix-map=$(pwd)=.` to `compiler_flags`.

---

## Tests

The Augmentum unit tests (priors, probes, extensions, utilities) live in `augmentum-main/driver/test/`:

```bash
cd augmentum-main
make test
```

---

## Common issues

- **Job crashes resuming at iter N**: the `enabled` column was historically stored as `numpy.int64` bytes; the resume decoder in [bo_tuning.py](augmentum-main/driver/bo_tuning.py) handles both BLOB and INTEGER rows transparently.
- **Setup takes 90 min every job**: marker at `llvm10/build/instrumented/.instrumented_targets.json` was deleted or invalidated. Safe to run — will rebuild once and re-cache.
- **`working_dir_bo/` doesn't exist**: default path in `sbatch_bo.sh`; always set `WORKING_DIR=...` explicitly to the run directory you want to resume (e.g. `working_dir_bo_bestval`).

---

## Repository layout

| Path | Contents |
|---|---|
| `augmentum-main/extensions/` | C++ extension library and the LLVM instrumentation pass (`augmentum_llvmpass`) |
| `augmentum-main/driver/` | Python driver: `function_analyser.py` (Stage 1), `bo_tuning.py`, `ga_tuning.py`, `rtp_tuning.py`, `sa_tuning.py` (Stage 2), `benchmark_profiler.py` |
| `augmentum-main/driver/scripts/` | `sbatch_*.sh` job scripts, `submit_chain.sh`, `run_evaluation.sh` |
| `augmentum-main/benchmarks/` | PolyBench-C 4.2.1, SNU NPB 3.3 (serial C) and an llvm-test-suite selection |
| `augmentum-main/tools/` | `fpcmp` (floating-point output comparison for verification) and `stlwrapper` (preloadable STL call wrapper) |
| `ALL_tuning_targets.csv` | Tuning target list: application, module, function, path, prior and target range |

---

## Analysis

- [report.md](report.md) — findings, method comparison, current best config
- [improvements.md](improvements.md) — implementation notes for the BO enhancements and the GA baseline
