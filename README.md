# Heureka — Compiler Heuristic Discovery

This project builds on [Augmentum](https://github.com/bfranke1973/Heureka) to automatically discover and evaluate compiler heuristics in LLVM 10 using benchmark-driven analysis. It extends the original framework with **Random Search** and **Bayesian Optimization** strategies.

---

## Overview

The framework instruments LLVM functions, runs benchmarks (e.g., POLYBENCH), and measures code size changes when heuristic parameter values are replaced by alternatives. The goal is to find heuristic configurations that reduce binary size.

**Prior types evaluated per function path:**
- Null → Broadcast → Boolean / All Integers / All Reals → Range → Offset → Scale

**Search strategies (this fork adds):**
- `composite` — original sequential state machine (default)
- `random` — uniform random sampling over the full parameter space
- `bayesian` — Gaussian Process + Expected Improvement acquisition function

---

## Repository Structure

```
augmentum-main/
├── benchmarks/              # POLYBENCH benchmarks
├── driver/
│   ├── function_analyser.py # Main entry point
│   ├── augmentum/
│   │   ├── driver.py        # Evaluation orchestration
│   │   ├── priors.py        # Search strategies (Random, Bayesian added here)
│   │   ├── pathworker.py    # Worker processes per path
│   │   ├── heuristicDB.py   # SQLite result storage
│   │   └── objectives.py    # Code size metric
│   ├── config/
│   │   ├── evaluation_config.json       # Tool paths and benchmark config
│   │   ├── target_functions.csv         # 576 target LLVM functions
│   │   ├── target_functions_subset.csv  # 20-function subset for quick testing
│   │   ├── function_cache_subset.pickle # Serialized subset function cache
│   │   └── baseline_cache.pickle        # Cached baseline benchmark results
│   └── scripts/
│       ├── run_evaluation.sh            # Local run script
│       └── sbatch_evaluation.sh         # SLURM cluster submission script
working_dir/
└── heuristic_data.sqlite    # All evaluation results (append-only)
```

---

## Setup

### Prerequisites

- LLVM 10 built at `llvm10/` (not tracked in git due to size)
- Augmentum LLVM pass built at `/dev/shm/fsyang/augmentum/build/`
- Conda environment `augmentum`

### Install dependencies

```bash
conda activate augmentum
pip install scikit-optimize
```

---

## Running an Evaluation

### 1. Configure `evaluation_config.json`

Located at `augmentum-main/evaluation_config.json`. Key fields:

```json
"tools": {
    "compiler_bin": "/path/to/llvm10/install/bin",
    "augmentum_pass": "/path/to/libaugmentum_llvmpass.so"
},
"sys_prog": {
    "available_functions_and_types": {
        "function_stats": "/path/to/llvm10/build/function_stats.csv",
        "named_struct_stats": "/path/to/llvm10/build/named_struct_stats.csv"
    }
}
```

> Setting `available_functions_and_types` skips the initial LLVM full build.

### 2. Edit `run_evaluation.sh`

Key settings in `augmentum-main/driver/scripts/run_evaluation.sh`:

| Variable | Description |
|---|---|
| `FUN_CACHE` | Path to function cache pickle (use `function_cache_subset.pickle` for quick test) |
| `TARGET_FUNS` | CSV of target functions to evaluate |
| `CHUNK_SIZE` | Max paths per run |
| `SEARCH_STRATEGY` | `composite` / `random` / `bayesian` |
| `SEARCH_BUDGET` | Probes per path for random/bayesian (default: 50) |

### 3. Run

```bash
conda activate augmentum
cd augmentum-main/driver/scripts
bash run_evaluation.sh
```

---

## Search Strategies

### Composite (default)
Original hierarchical state machine. Tests Null → Broadcast → Integer/Real ranges → Offset → Scale. Most thorough but slow (~several days for 576 functions).

### Random Search
Uniformly samples `--search_budget` values from the full parameter space after NullPrior.

```bash
# In run_evaluation.sh, uncomment:
SEARCH_STRATEGY="--search_strategy random"
SEARCH_BUDGET="--search_budget 50"
```

### Bayesian Optimization
Uses Gaussian Process regression with Expected Improvement to guide sampling. Requires `scikit-optimize`. Falls back to random search automatically if unavailable.

```bash
# In run_evaluation.sh, uncomment:
SEARCH_STRATEGY="--search_strategy bayesian"
SEARCH_BUDGET="--search_budget 30"
```

---

## Querying Results

Results are stored in `working_dir/heuristic_data.sqlite`.

```bash
sqlite3 working_dir/heuristic_data.sqlite
```

**Check for size improvements:**
```sql
SELECT module, function, path, rel_objective
FROM probe_log
WHERE rel_objective != 'NA'
  AND CAST(rel_objective AS REAL) < 1.0
ORDER BY CAST(rel_objective AS REAL) ASC
LIMIT 20;
```

**Summary of results:**
```sql
SELECT
  CASE
    WHEN rel_objective = 'NA' THEN 'NA'
    WHEN CAST(rel_objective AS REAL) < 1.0 THEN 'improvement'
    WHEN CAST(rel_objective AS REAL) = 1.0 THEN 'same'
    ELSE 'worse'
  END AS result,
  COUNT(*) AS count
FROM probe_log
GROUP BY result;
```

**Evaluated paths per function:**
```sql
SELECT function, prior_success, COUNT(*) as paths
FROM path_heuristics
WHERE prior_success != 'NA'
GROUP BY function, prior_success
ORDER BY paths DESC
LIMIT 20;
```

---

## Subset Testing

For quick iteration, use the 20-function subset:

```bash
# In run_evaluation.sh:
FUN_CACHE="--function_cache ${CFG_DIR}/function_cache_subset.pickle"
TARGET_FUNS="--target_function ${CFG_DIR}/target_functions_subset.csv"
CHUNK_SIZE="--fn_chunk_size 100000"
```

A subset run with composite strategy completes in ~2 hours.

---

## Notes

- New runs **append** to the SQLite DB — existing results are never overwritten
- Already-evaluated paths are automatically skipped (`prior_success != 'NA'`)
- The Augmentum pass must be rebuilt if you restart after a crash (`/dev/shm/` is cleared on reboot)
- POLYBENCH benchmarks may not show size improvements for simple heuristics — consider using SNU_NPB or LLVM_SUITE for more sensitivity
