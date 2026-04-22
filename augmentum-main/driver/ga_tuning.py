#!/usr/bin/env python3
"""
Multi-target Genetic Algorithm (GA) for compiler heuristic tuning.

Third baseline alongside RTP (random subset search) and BO (RF-surrogate
Bayesian optimisation). Same composite-discovered tuning targets, same
binary on/off search space, same value-sampling modes — only the
exploration strategy differs.

Strategy:
  1. Load top-k tuning targets from composite DB (same loaders as bo_tuning).
  2. Each individual is a length-k binary vector (target on/off).
  3. Initialize a random population (size = --pop_size).
  4. Evaluate each individual: compile + run + verify on every benchmark,
     fitness = mean rel_objective (lower is better, FAILURE_PENALTY on fail).
  5. Produce next generation via:
       - Elitism: keep best 1 individual unchanged
       - Tournament selection (k=3) of two parents
       - Uniform crossover (rate --crossover_rate)
       - Per-bit mutation (rate --mutation_rate, default 1/k)
  6. Repeat until --iterations evaluations consumed.

Two value-sampling modes (identical to bo_tuning):

  --use_best_value : enabled targets use composite's best (prior, value)
      pair. Deterministic; reproduces composite's single-target peak.
  default          : enabled targets sample a value uniformly per call from
      the prior range, in the C++ extension code (paper-RTP behavior).

Usage:
  python ga_tuning.py --config evaluation_config.json \\
      --composite_db sqlite:///path/to/subset_composite.sqlite \\
      --working_dir /path/to/working_dir_ga \\
      --iterations 2000 --top_k 10 --pop_size 20 --cpus 8

Resume:
  Re-running the same command resumes from the largest fully-completed
  generation in ga_results.sqlite. The per-generation RNG seed is
  derived from (--seed, generation) so post-resume evolution is
  reproducible.
"""

import argparse
import json
import logging
import pickle
import random
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augmentum.benchmarks import BenchmarkFactory, ExecutionResult
from augmentum.objectives import CodeSizeObjective
from augmentum.sysProg import ProbeExtension, SysProg
from augmentum.sysUtils import try_create_dir
from augmentum.timer import Timer

# Reuse loaders + codegen from bo_tuning so GA evaluates the *same* configs
# BO would (only the search strategy differs). FAILURE_PENALTY is also
# shared so comparisons across baselines are apples-to-apples.
from bo_tuning import (
    FAILURE_PENALTY,
    TuningTarget,
    generate_bo_extension as generate_extension,
    load_top_k_targets,
    load_top_k_targets_with_best_value,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GA data structures
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    genome: List[int]  # length-k binary vector
    fitness: Optional[float] = None  # mean rel_objective (lower is better)


@dataclass
class GAConfig:
    pop_size: int
    mutation_rate: float
    crossover_rate: float
    tournament_size: int
    seed: int
    elitism: int = 1


# ---------------------------------------------------------------------------
# Binary GA — minimal hand-rolled implementation
# ---------------------------------------------------------------------------

class BinaryGA:
    """
    Per-generation deterministic RNG: self.rng is reseeded as
    Random(seed * 1_000_003 + generation) at the start of each generation.
    This makes post-resume runs reproducible regardless of how many
    individuals were evaluated mid-generation in the killed run.
    """

    def __init__(self, k: int, cfg: GAConfig):
        self.k = k
        self.cfg = cfg
        self.population: List[Individual] = []
        self.generation: int = 0
        self.rng = random.Random()
        self._reseed_for_generation(0)

    def _reseed_for_generation(self, gen: int):
        self.rng.seed(self.cfg.seed * 1_000_003 + gen)

    def initial_population(self) -> List[Individual]:
        """
        Initialise with random binary vectors. We bias the very first
        individual to all-zeros (baseline) and the second to all-ones
        (full stack) so the GA always observes both extremes regardless
        of luck — same intent as BO's ``n_initial_points`` warm-up.
        """
        pop: List[Individual] = []
        pop.append(Individual([0] * self.k))
        if self.cfg.pop_size > 1:
            pop.append(Individual([1] * self.k))
        while len(pop) < self.cfg.pop_size:
            pop.append(Individual(
                [self.rng.randint(0, 1) for _ in range(self.k)]
            ))
        self.population = pop
        return pop

    def tournament_select(self) -> Individual:
        contestants = self.rng.sample(
            self.population, self.cfg.tournament_size
        )
        # Lower fitness wins (we minimise rel_objective). Treat None as +inf.
        return min(
            contestants,
            key=lambda ind: (
                ind.fitness if ind.fitness is not None else float("inf")
            ),
        )

    def crossover(self, p1: Individual, p2: Individual) -> Individual:
        if self.rng.random() < self.cfg.crossover_rate:
            child = [
                p1.genome[i] if self.rng.random() < 0.5 else p2.genome[i]
                for i in range(self.k)
            ]
        else:
            child = list(p1.genome)
        return Individual(child)

    def mutate(self, ind: Individual) -> Individual:
        new_genome = [
            (1 - g) if self.rng.random() < self.cfg.mutation_rate else g
            for g in ind.genome
        ]
        return Individual(new_genome)

    def next_generation(self) -> List[Individual]:
        """Build next population from current self.population (with fitnesses)."""
        self.generation += 1
        self._reseed_for_generation(self.generation)

        # Elitism — copy top-N unchanged
        sorted_pop = sorted(
            self.population,
            key=lambda ind: (
                ind.fitness if ind.fitness is not None else float("inf")
            ),
        )
        new_pop: List[Individual] = [
            Individual(list(ind.genome), ind.fitness)
            for ind in sorted_pop[: self.cfg.elitism]
        ]

        while len(new_pop) < self.cfg.pop_size:
            p1 = self.tournament_select()
            p2 = self.tournament_select()
            child = self.mutate(self.crossover(p1, p2))
            new_pop.append(child)

        self.population = new_pop
        return new_pop


# ---------------------------------------------------------------------------
# Result database
# ---------------------------------------------------------------------------

def init_ga_db(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ga_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER NOT NULL,
            generation INTEGER NOT NULL,
            individual_idx INTEGER NOT NULL,
            test_case TEXT NOT NULL,
            num_targets INTEGER NOT NULL,
            config_desc TEXT NOT NULL,
            compile_result TEXT NOT NULL,
            run_result TEXT NOT NULL,
            verify_result TEXT NOT NULL,
            compile_time REAL,
            run_time REAL,
            objective REAL,
            baseline_objective REAL,
            rel_objective REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ga_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER NOT NULL,
            generation INTEGER NOT NULL,
            individual_idx INTEGER NOT NULL,
            target_idx INTEGER NOT NULL,
            enabled INTEGER NOT NULL,
            module TEXT NOT NULL,
            function TEXT NOT NULL,
            path TEXT NOT NULL,
            prior_type TEXT NOT NULL,
            value_op TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_ga_config(db_path, iteration, generation, ind_idx, targets, genome):
    conn = sqlite3.connect(db_path)
    for i, t in enumerate(targets):
        conn.execute("""
            INSERT INTO ga_configs
            (iteration, generation, individual_idx, target_idx, enabled,
             module, function, path, prior_type, value_op)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (iteration, generation, ind_idx, i, int(genome[i]),
              t.module, t.function, t.path_str, t.prior_type, t.value_op))
    conn.commit()
    conn.close()


def save_ga_result(db_path, iteration, generation, ind_idx, tc_name,
                   num_targets, config_desc, compile_r, run_r, verify_r,
                   compile_t, run_t, objective, baseline, rel_obj):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO ga_results
        (iteration, generation, individual_idx, test_case, num_targets,
         config_desc, compile_result, run_result, verify_result,
         compile_time, run_time, objective, baseline_objective, rel_objective)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (iteration, generation, ind_idx, tc_name, num_targets, config_desc,
          compile_r, run_r, verify_r, compile_t, run_t,
          objective, baseline, rel_obj))
    conn.commit()
    conn.close()


def reload_state(db_path: str, k: int, pop_size: int, n_test_cases: int
                 ) -> Tuple[int, int, List[Individual]]:
    """
    Reconstruct (next_iteration, last_completed_generation, population_at_that_gen).

    A generation is "complete" iff:
      - all pop_size individuals have configs in ga_configs
      - each individual has results for ALL n_test_cases in ga_results
        (a row counts even if compile/verify failed — we just need the slot
        filled so we can compute fitness).

    Mid-generation evaluations (if the previous run was killed mid-gen) are
    left in the DB as historical data but not used to seed the population —
    the GA simply re-runs that generation with the same per-generation RNG
    seed, so the same individuals will be re-proposed and (if their entries
    aren't already there) re-evaluated.
    """
    if not Path(db_path).exists():
        return 0, -1, []

    conn = sqlite3.connect(db_path)

    # Group configs by generation
    cur = conn.execute("""
        SELECT generation, individual_idx, target_idx, enabled
        FROM ga_configs
        ORDER BY generation, individual_idx, target_idx
    """)
    gen_to_individuals: Dict[int, Dict[int, List[int]]] = {}
    for gen, ind_idx, t_idx, enabled in cur:
        gen_to_individuals.setdefault(gen, {}).setdefault(
            ind_idx, [0] * k
        )[t_idx] = int(enabled)

    last_complete_gen = -1
    final_pop: List[Individual] = []
    for gen in sorted(gen_to_individuals.keys()):
        inds = gen_to_individuals[gen]
        if len(inds) != pop_size:
            break
        # All individuals present — verify each has full per-test-case results
        # and compute fitness.
        candidate_pop: List[Individual] = []
        complete = True
        for ind_idx in range(pop_size):
            if ind_idx not in inds:
                complete = False
                break
            cur2 = conn.execute("""
                SELECT rel_objective, verify_result
                FROM ga_results
                WHERE generation = ? AND individual_idx = ?
            """, (gen, ind_idx))
            rows = cur2.fetchall()
            if len(rows) < n_test_cases:
                complete = False
                break
            ok_objs = [
                r[0] for r in rows
                if r[1] == "SUCCESS" and r[0] is not None
            ]
            if ok_objs:
                fitness = float(np.mean(ok_objs))
            else:
                fitness = FAILURE_PENALTY
            candidate_pop.append(Individual(list(inds[ind_idx]), fitness))
        if not complete:
            break
        last_complete_gen = gen
        final_pop = candidate_pop

    # Next iteration counter = max(iteration) + 1 across both tables, so we
    # never collide ids when partial mid-generation rows exist.
    cur3 = conn.execute("SELECT MAX(iteration) FROM ga_results")
    max_iter_results = cur3.fetchone()[0]
    cur4 = conn.execute("SELECT MAX(iteration) FROM ga_configs")
    max_iter_configs = cur4.fetchone()[0]
    next_iter = 1 + max(
        max_iter_results if max_iter_results is not None else -1,
        max_iter_configs if max_iter_configs is not None else -1,
    )

    conn.close()
    return next_iter, last_complete_gen, final_pop


# ---------------------------------------------------------------------------
# Per-individual evaluation
# ---------------------------------------------------------------------------

def evaluate_individual(
    individual: Individual,
    iteration: int,
    generation: int,
    ind_idx: int,
    targets: List[TuningTarget],
    fn_cache,
    sys_prog_src: str,
    test_cases,
    instrumented_bins: Path,
    objective_metric: CodeSizeObjective,
    baseline_cache,
    working_dir: Path,
    tools_cfg,
    probe_mem_limit: int,
    db_path: str,
) -> float:
    """Evaluate one individual on every benchmark, return its fitness."""
    selected = [t for i, t in enumerate(targets) if individual.genome[i] == 1]
    num_enabled = len(selected)
    config_desc = "; ".join(
        f"{t.function}:{t.path_str}" for t in selected
    ) if selected else "none"

    save_ga_config(db_path, iteration, generation, ind_idx, targets,
                   individual.genome)

    logger.info(
        f"  [gen {generation} ind {ind_idx}] enabled={num_enabled}/{len(targets)}"
    )

    # All-off individual: baseline (cost = 1.0) — no compilation needed
    if num_enabled == 0:
        for tc_name in test_cases:
            save_ga_result(db_path, iteration, generation, ind_idx, tc_name,
                           0, "none", "NA", "NA", "NA",
                           None, None, None, None, 1.0)
        return 1.0

    ext_code = generate_extension(selected, fn_cache, sys_prog_src)
    if ext_code is None:
        logger.warning("    extension generation failed")
        for tc_name in test_cases:
            save_ga_result(db_path, iteration, generation, ind_idx, tc_name,
                           num_enabled, config_desc, "EXT_GEN_FAIL", "NA",
                           "NA", None, None, None, None, None)
        return FAILURE_PENALTY

    probe_wd = try_create_dir(working_dir / "ga_probe", use_time=True)
    if not probe_wd:
        logger.error("    probe dir creation failed")
        return FAILURE_PENALTY

    try:
        ext = ProbeExtension(ext_code)
        ext_lib = ext.build_library(probe_wd, tools_cfg)
    except Exception as e:
        logger.warning(f"    extension build failed: {e}")
        for tc_name in test_cases:
            save_ga_result(db_path, iteration, generation, ind_idx, tc_name,
                           num_enabled, config_desc, "EXT_BUILD_FAIL", "NA",
                           "NA", None, None, None, None, None)
        if probe_wd.exists():
            shutil.rmtree(probe_wd, ignore_errors=True)
        return FAILURE_PENALTY

    rel_objs: List[float] = []
    for tc_name, tc in test_cases.items():
        timer = Timer()
        timer.start()
        compile_result = tc.compile(
            instrumented_bins, ext_lib, memory_limit=probe_mem_limit
        )
        compile_time = timer.stop()

        run_result = ExecutionResult.NA
        verify_result = ExecutionResult.NA
        run_time = None
        obj_val = None
        baseline_val = None
        rel_obj = None

        if compile_result == ExecutionResult.SUCCESS:
            timer.start()
            run_result, obj_result = objective_metric.measure(
                tc, memory_limit=probe_mem_limit
            )
            run_time = timer.stop()

            if run_result == ExecutionResult.SUCCESS:
                verify_result = tc.verify()
                if verify_result == ExecutionResult.SUCCESS and obj_result:
                    obj_val = obj_result.score
                    if tc_name in baseline_cache:
                        baseline_val = baseline_cache[tc_name].objective.score
                        if baseline_val and baseline_val > 0:
                            rel_obj = obj_val / baseline_val
                            rel_objs.append(rel_obj)

        save_ga_result(db_path, iteration, generation, ind_idx, tc_name,
                       num_enabled, config_desc,
                       compile_result.name, run_result.name,
                       verify_result.name,
                       compile_time, run_time,
                       obj_val, baseline_val, rel_obj)
        logger.info(
            f"    {tc_name}: compile={compile_result.name} "
            f"verify={verify_result.name} rel_obj={rel_obj}"
        )

    if probe_wd.exists():
        shutil.rmtree(probe_wd, ignore_errors=True)

    return float(np.mean(rel_objs)) if rel_objs else FAILURE_PENALTY


# ---------------------------------------------------------------------------
# Main GA loop
# ---------------------------------------------------------------------------

def run_ga(args):
    with open(args.config) as f:
        config = json.load(f)

    tools_cfg = config["tools"]
    sysprog_cfg = config["sys_prog"]["available"][config["sys_prog"]["active"]]
    sys_prog_src = sysprog_cfg["src_dir"]

    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(working_dir / "ga_results.sqlite")
    init_ga_db(db_path)

    # Load top-k targets — exactly the same loaders BO uses, so the search
    # space is identical. Only the optimiser differs.
    composite_db_file = args.composite_db.replace("sqlite:///", "")
    if args.use_best_value:
        targets = load_top_k_targets_with_best_value(
            composite_db_file, args.top_k
        )
    else:
        targets = load_top_k_targets(composite_db_file, args.top_k)
    if not targets:
        logger.error("No tuning targets found!")
        return

    k = len(targets)
    pop_size = max(2, args.pop_size)
    mutation_rate = args.mutation_rate if args.mutation_rate > 0 else 1.0 / k
    ga_cfg = GAConfig(
        pop_size=pop_size,
        mutation_rate=mutation_rate,
        crossover_rate=args.crossover_rate,
        tournament_size=args.tournament_size,
        seed=args.seed,
        elitism=args.elitism,
    )
    logger.info(
        f"GA: k={k} pop={pop_size} mut={mutation_rate:.4f} "
        f"xover={ga_cfg.crossover_rate} tourn={ga_cfg.tournament_size} "
        f"elitism={ga_cfg.elitism} seed={ga_cfg.seed}"
    )

    ga = BinaryGA(k, ga_cfg)

    # Function cache + benchmark setup (mirrors bo_tuning)
    with open(args.function_cache, "rb") as f:
        fn_list = pickle.load(f)
    fn_cache = {(fn.module, fn.name): fn for fn in fn_list}
    logger.info(f"Loaded {len(fn_cache)} functions from cache.")

    benchmark_cfgs = [
        config["benchmark"]["available"][name]
        for name in config["benchmark"]["active"]
    ]
    benchmark_factory = BenchmarkFactory(
        tools_cfg, benchmark_cfgs, verbose=args.verbose
    )
    bench_dir = working_dir / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    test_cases = benchmark_factory.setup_benchmark_instance(bench_dir)
    n_test_cases = len(test_cases)
    logger.info(f"Created {n_test_cases} test cases.")

    with open(args.baseline_cache, "rb") as f:
        baseline_cache = pickle.load(f)

    objective_metric = CodeSizeObjective(Path(tools_cfg["llvm-size"]))

    # Instrument compiler
    sys_prog_bld_dir = Path(sysprog_cfg["build_dir"])
    instrumented_bins = sys_prog_bld_dir / "instrumented" / "bin"

    target_modules: Dict[str, set] = {}
    for t in targets:
        target_modules.setdefault(t.module, set()).add(t.function)

    logger.info("Setting up system program...")
    opt_program = SysProg(
        tools_cfg, sysprog_cfg,
        Path(sys_prog_src), sys_prog_bld_dir,
        objective_metric,
        cpus=args.cpus, verbose=args.verbose,
    )
    logger.info(
        f"Instrumenting {sum(len(v) for v in target_modules.values())} "
        f"functions in {len(target_modules)} modules..."
    )
    opt_program.instrument(target_modules)

    # ---- Resume support -----------------------------------------------------
    iteration, last_complete_gen, restored_pop = reload_state(
        db_path, k, pop_size, n_test_cases
    )
    if restored_pop:
        logger.info(
            f"Resuming from generation {last_complete_gen + 1} "
            f"(restored pop of {len(restored_pop)} individuals, "
            f"next iteration={iteration})."
        )
        ga.population = restored_pop
        ga.generation = last_complete_gen
        # Build the next generation we will actually evaluate
        ga.next_generation()
    else:
        ga.initial_population()

    # ---- Main loop ----------------------------------------------------------
    best_fitness = float("inf")
    best_genome: Optional[List[int]] = None
    if restored_pop:
        for ind in restored_pop:
            if ind.fitness is not None and ind.fitness < best_fitness:
                best_fitness = ind.fitness
                best_genome = list(ind.genome)

    while iteration < args.iterations:
        gen = ga.generation
        logger.info(
            f"=== GA generation {gen} "
            f"(iter {iteration}/{args.iterations}) ==="
        )

        for ind_idx, ind in enumerate(ga.population):
            if iteration >= args.iterations:
                break
            # If this individual already has a fitness (elitism carry-over),
            # skip re-evaluation — its DB rows from the previous generation
            # are still authoritative.
            if ind.fitness is not None:
                logger.info(
                    f"  [gen {gen} ind {ind_idx}] elite carry-over "
                    f"(fitness={ind.fitness:.6f}), skipping"
                )
                continue

            fitness = evaluate_individual(
                individual=ind,
                iteration=iteration,
                generation=gen,
                ind_idx=ind_idx,
                targets=targets,
                fn_cache=fn_cache,
                sys_prog_src=sys_prog_src,
                test_cases=test_cases,
                instrumented_bins=instrumented_bins,
                objective_metric=objective_metric,
                baseline_cache=baseline_cache,
                working_dir=working_dir,
                tools_cfg=tools_cfg,
                probe_mem_limit=args.probe_mem_limit,
                db_path=db_path,
            )
            ind.fitness = fitness
            iteration += 1

            if fitness < best_fitness:
                best_fitness = fitness
                best_genome = list(ind.genome)
                logger.info(f"  *** new best: fitness={fitness:.6f} ***")

        # Generation done — produce the next one (skip if we're out of budget)
        if iteration < args.iterations:
            ga.next_generation()

    # ---- Summary ------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("GA tuning completed.")
    logger.info(f"Best fitness (avg rel_objective): {best_fitness:.6f}")
    if best_genome is not None:
        enabled = [
            f"{t.function}:{t.path_str}"
            for i, t in enumerate(targets) if best_genome[i] == 1
        ]
        logger.info(f"Best genome enables {len(enabled)} targets:")
        for e in enabled:
            logger.info(f"  - {e}")

    conn = sqlite3.connect(db_path)
    cur = conn.execute("""
        SELECT COUNT(*),
               MIN(rel_objective),
               AVG(rel_objective),
               SUM(CASE WHEN rel_objective < 1.0 THEN 1 ELSE 0 END)
        FROM ga_results
        WHERE verify_result = 'SUCCESS' AND rel_objective IS NOT NULL
    """)
    row = cur.fetchone()
    if row and row[0]:
        logger.info(
            f"Total verified: {row[0]}, min_rel={row[1]:.4f}, "
            f"avg_rel={row[2]:.4f}, reductions={row[3]}"
        )
    conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Genetic Algorithm baseline for compiler heuristic "
                    "tuning (binary subset over composite-discovered targets)."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--composite_db", required=True)
    parser.add_argument("--working_dir", required=True)
    parser.add_argument("--function_cache", required=True)
    parser.add_argument("--baseline_cache", required=True)
    parser.add_argument("--iterations", type=int, default=2000,
                        help="Total per-individual evaluations (default 2000)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-k targets to include (default 10)")
    parser.add_argument("--pop_size", type=int, default=20,
                        help="GA population size (default 20)")
    parser.add_argument("--mutation_rate", type=float, default=0.0,
                        help="Per-bit mutation prob; 0 = use 1/k (default)")
    parser.add_argument("--crossover_rate", type=float, default=0.8,
                        help="Uniform crossover prob (default 0.8)")
    parser.add_argument("--tournament_size", type=int, default=3,
                        help="Tournament selection size (default 3)")
    parser.add_argument("--elitism", type=int, default=1,
                        help="Best-N carried unchanged each generation "
                             "(default 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master RNG seed (default 42)")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--probe_mem_limit", type=int, default=1024)
    parser.add_argument("--use_best_value", action="store_true",
                        help="Use composite-discovered best value per target "
                             "(deterministic) instead of per-call random "
                             "sampling. Same semantics as bo_tuning.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(processName)-12s %(name)-30s "
               "%(levelname)-8s %(message)s",
    )
    run_ga(args)
