#!/usr/bin/env python3
"""
RTP with Bayesian Optimization — replaces the random search in RTP with
a BO approach using Random Forest surrogate (handles mixed discrete/continuous
spaces well and scales better than GP to higher dimensions).

Strategy:
  1. Load top-k tuning targets from composite DB (ranked by individual
     binary size reduction).
  2. Encode search space as 2k dimensions:
       - k Categorical([0, 1]) dimensions (target on/off)
       - k Real/Integer dimensions (value within prior range)
  3. Use skopt.Optimizer with RF surrogate to propose configurations.
  4. Compile benchmark with proposed configuration, measure binary size.
  5. Feed result back to BO and repeat.

Usage:
  python rtp_bo_tuning.py --config evaluation_config.json \\
      --composite_db sqlite:///path/to/subset_composite.sqlite \\
      --working_dir /path/to/working_dir_rtp_bo \\
      --iterations 2000 --top_k 10 --cpus 8
"""

import argparse
import json
import logging
import os
import pickle
import random
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skopt import Optimizer
from skopt.space import Categorical, Integer, Real

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augmentum.benchmarks import BenchmarkFactory, ExecutionResult
from augmentum.function import Function
from augmentum.objectives import CodeSizeObjective
from augmentum.probes import (
    add_null_check,
    build_struct_dependency_graph,
    generate_path_code,
    generate_register_extension_point,
    get_struct_definitions_from_graph,
    type_descs_to_cpp_arg_types,
    type_descs_to_cpp_arg_vals,
    type_descs_to_cpp_args,
)
from augmentum.sysProg import ProbeExtension, SysProg
from augmentum.sysUtils import run_command, try_create_dir
from augmentum.timer import Timer
from augmentum.type_descs import IntTypeDesc, RealTypeDesc, StructTypeDesc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TuningTarget:
    module: str
    function: str
    path_str: str
    prior_type: str
    prior_data: str
    best_rel_obj: float  # best individual rel_objective from composite


# ---------------------------------------------------------------------------
# Load and rank tuning targets
# ---------------------------------------------------------------------------

def load_top_k_targets(composite_db_path: str, top_k: int) -> List[TuningTarget]:
    """
    Load tuning targets ranked by best individual binary size reduction.
    For each (module, function, path), pick the best prior and keep top-k.
    """
    conn = sqlite3.connect(composite_db_path)

    # Get best rel_objective per (module, function, path)
    cursor = conn.execute("""
        SELECT ph.module, ph.function, ph.path,
               MIN(CAST(pl.rel_objective AS FLOAT)) as best_rel
        FROM path_heuristics ph
        JOIN probe_log pl
          ON (ph.module = pl.module
              AND ph.function = pl.function
              AND ph.path = pl.path)
        WHERE ph.obj_improvement = 1
          AND pl.verify = 'SUCCESS'
          AND pl.rel_objective IS NOT NULL
          AND CAST(pl.rel_objective AS FLOAT) < 1.0
        GROUP BY ph.module, ph.function, ph.path
        ORDER BY best_rel ASC
        LIMIT ?
    """, (top_k,))

    top_paths = []
    for row in cursor:
        top_paths.append((row[0], row[1], row[2], row[3]))

    # For each top path, get its best non-null prior
    targets = []
    for module, function, path_str, best_rel in top_paths:
        cur2 = conn.execute("""
            SELECT pr.prior, pr.data
            FROM prior_results pr
            WHERE pr.module = ? AND pr.function = ? AND pr.path = ?
              AND pr.success = 1
              AND pr.prior NOT IN ('Null Prior', 'Broadcast Prior')
            LIMIT 1
        """, (module, function, path_str))
        row2 = cur2.fetchone()
        if row2:
            targets.append(TuningTarget(
                module=module, function=function, path_str=path_str,
                prior_type=row2[0], prior_data=row2[1],
                best_rel_obj=best_rel,
            ))

    conn.close()
    logger.info(f"Loaded top-{len(targets)} tuning targets (requested {top_k}):")
    for t in targets:
        logger.info(f"  {t.function}:{t.path_str} rel={t.best_rel_obj:.4f} "
                     f"prior={t.prior_type}")
    return targets


# ---------------------------------------------------------------------------
# Build skopt search space from targets
# ---------------------------------------------------------------------------

def parse_prior_range(target: TuningTarget) -> Tuple[str, Any, Any, Optional[str]]:
    """
    Parse prior data to get (space_type, lo, hi, value_op).
    Returns:
      space_type: "int", "real", or "cat"
      lo, hi: range bounds
      value_op: None (static), "+" (offset), "*" (scale)
    """
    ptype = target.prior_type
    pdata = target.prior_data

    if ptype == "Boolean Prior":
        return ("cat", 0, 1, None)

    elif ptype in ("All Integers Prior", "All Reals Prior"):
        entries = pdata.split("#")
        values = []
        for e in entries:
            parts = e.split(",")
            if len(parts) >= 2 and parts[1] == "True":
                try:
                    values.append(int(parts[0]) if "Integer" in ptype
                                  else float(parts[0]))
                except ValueError:
                    continue
        if values:
            lo, hi = min(values), max(values)
            if "Integer" in ptype:
                return ("int", lo, hi, None)
            else:
                return ("real", lo, hi, None)
        return ("int", -128, 127, None)

    elif ptype in ("Integer Range Prior", "Real Range Prior"):
        parts = pdata.split(",")
        if "Real" in ptype:
            return ("real", float(parts[0]), float(parts[1]), None)
        else:
            return ("int", int(parts[0]), int(parts[1]), None)

    elif ptype in ("Integer Offset Prior", "Real Offset Prior"):
        parts = pdata.split(",")
        if "Real" in ptype:
            theta, phi = float(parts[0]), float(parts[1])
            return ("real", -theta, phi, "+")
        else:
            theta, phi = int(parts[0]), int(parts[1])
            return ("int", -theta, phi, "+")

    elif ptype in ("Integer Scale Prior", "Real Scale Prior"):
        parts = pdata.split(",")
        if "Real" in ptype:
            alpha, beta = float(parts[0]), float(parts[1])
            return ("real", -alpha, beta, "*")
        else:
            alpha, beta = int(parts[0]), int(parts[1])
            return ("int", -alpha, beta, "*")

    else:
        return ("int", -128, 127, None)


def build_search_space(targets: List[TuningTarget]):
    """
    Build skopt search space: for each target, one on/off switch + one value dim.
    Returns (dimensions, target_meta) where target_meta stores value_op per target.
    """
    dimensions = []
    target_meta = []  # (value_op, space_type) per target

    for i, t in enumerate(targets):
        # On/off switch
        dimensions.append(Categorical([0, 1], name=f"enable_{i}"))

        # Value dimension
        space_type, lo, hi, value_op = parse_prior_range(t)

        if space_type == "cat":
            dimensions.append(Categorical([0, 1], name=f"value_{i}"))
        elif space_type == "int":
            if lo == hi:
                hi = lo + 1  # skopt needs lo < hi
            dimensions.append(Integer(lo, hi, name=f"value_{i}"))
        else:  # real
            if lo == hi:
                hi = lo + 1.0
            dimensions.append(Real(float(lo), float(hi), name=f"value_{i}"))

        target_meta.append((value_op, space_type))

    return dimensions, target_meta


def decode_suggestion(suggestion, targets, target_meta):
    """
    Decode a BO suggestion into a list of (target, value, value_op) for enabled targets.
    """
    selected = []
    for i, t in enumerate(targets):
        enable = suggestion[2 * i]
        value = suggestion[2 * i + 1]
        if enable == 1:
            value_op = target_meta[i][0]
            selected.append((t, value, value_op))
    return selected


# ---------------------------------------------------------------------------
# Extension code generation (reuse from rtp_tuning.py)
# ---------------------------------------------------------------------------

def generate_bo_extension(
    selected,  # List of (TuningTarget, value, value_op)
    functions_by_key: Dict[Tuple[str, str], Function],
    sys_prog_src: str,
) -> Optional[str]:
    """Generate multi-target extension code for the selected configuration."""
    from collections import defaultdict

    # Deduplicate by (module, function)
    fn_targets = defaultdict(list)
    for target, value, value_op in selected:
        key = (target.module, target.function)
        fn_targets[key].append((target, value, value_op))

    unique_targets = []
    for key, items in fn_targets.items():
        item = random.choice(items)
        if key in functions_by_key:
            unique_targets.append((item, functions_by_key[key]))
        else:
            logger.warning(f"Function {key} not found in cache")

    if not unique_targets:
        return None

    # Shared struct graph
    shared_struct_graph: Dict[StructTypeDesc, Any] = {}
    for (target, value, value_op), fn in unique_targets:
        build_struct_dependency_graph(fn.type, shared_struct_graph, deep_traversal=True)
    combined_struct_defs = get_struct_definitions_from_graph(shared_struct_graph)

    modified_functions_code = []
    register_blocks = []
    unregister_blocks = []

    valid_idx = 0
    for (target, value, value_op), fn in unique_targets:
        idx = valid_idx

        # Resolve path
        fn_paths = fn.get_paths()
        path_obj = None
        for p in fn_paths:
            if str(p) == target.path_str:
                path_obj = p
                break
        if path_obj is None:
            continue

        try:
            path_code, probed_type, null_check_id = generate_path_code(
                path_obj, fn, "probed"
            )
        except Exception as e:
            logger.warning(f"Path code gen failed: {e}")
            continue

        orig_type = f"original_t_{idx}"
        orig_id = f"original_fn_{idx}"
        mod_id = f"modified_fn_{idx}"

        return_type = fn.type.return_type.get_cpp_type().get_type_string()
        arg_vals = type_descs_to_cpp_arg_vals(fn.type.arg_types)
        arg_types_str = type_descs_to_cpp_arg_types(fn.type.arg_types)
        args = type_descs_to_cpp_args(fn.type.arg_types)
        probed_type_str = probed_type.get_cpp_type().get_type_string()

        # Value expression
        if value_op is None:
            calc_probed = f"({probed_type_str})({value})"
        elif value_op == "+":
            calc_probed = f"original_value + ({probed_type_str})({value})"
        elif value_op == "*":
            calc_probed = f"original_value * ({probed_type_str})({value})"
        else:
            calc_probed = f"({probed_type_str})({value})"

        if return_type == "void":
            return_stmt, orig_return = "", ""
        else:
            return_stmt = "return r;"
            orig_return = f"{return_type} r = "

        probed_code = f"""
        {path_code}
        {probed_type_str} original_value = *probed;
        *probed = {calc_probed};
"""
        probed_code = add_null_check(null_check_id, probed_code)

        fn_code = f"""
// BO Target {idx}: {target.module} :: {fn.name}
// Path: {target.path_str} | Prior: {target.prior_type} | Value: {value} | Op: {value_op}
typedef {return_type} (*{orig_type})({arg_types_str});
{orig_type} {orig_id};

{return_type} {mod_id}({args}) {{
    {orig_return}{orig_id}({arg_vals});
    {probed_code}
    {return_stmt}
}}
"""
        modified_functions_code.append(fn_code)

        register_blocks.append(f"""
        if (pt.get_module_name() == "{sys_prog_src}/{target.module}" &&
            pt.get_name() == "{fn.name}") {{
            if (!pt.is_replaced()) {{
                pt.replace((Fn) &{mod_id});
                {orig_id} = ({orig_type}) pt.original_direct();
            }}
        }}
""")
        unregister_blocks.append(f"""
        if (pt.get_module_name() == "{sys_prog_src}/{target.module}" &&
            pt.get_name() == "{fn.name}") {{
            if (pt.is_replaced()) {{ pt.reset(); }}
        }}
""")
        valid_idx += 1

    if not modified_functions_code:
        return None

    code = f"""
#include <cstdint>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <mutex>
#include "augmentum.h"

using namespace augmentum;

{combined_struct_defs}

{"".join(modified_functions_code)}

struct BOListener: Listener {{
    void on_extension_point_register(FnExtensionPoint& pt) {{
{"".join(register_blocks)}
    }}

    void on_extension_point_unregister(FnExtensionPoint& pt) {{
{"".join(unregister_blocks)}
    }}
}};
ListenerLifeCycle<BOListener> boListener;
"""
    return code


# ---------------------------------------------------------------------------
# Result database
# ---------------------------------------------------------------------------

def init_bo_db(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bo_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER NOT NULL,
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
        CREATE TABLE IF NOT EXISTS bo_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER NOT NULL,
            target_idx INTEGER NOT NULL,
            enabled INTEGER NOT NULL,
            module TEXT NOT NULL,
            function TEXT NOT NULL,
            path TEXT NOT NULL,
            prior_type TEXT NOT NULL,
            sampled_value REAL NOT NULL,
            value_op TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_bo_result(db_path, iteration, tc_name, num_targets, config_desc,
                   compile_r, run_r, verify_r, compile_t, run_t,
                   objective, baseline, rel_obj):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO bo_results
        (iteration, test_case, num_targets, config_desc,
         compile_result, run_result, verify_result,
         compile_time, run_time, objective, baseline_objective, rel_objective)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (iteration, tc_name, num_targets, config_desc,
          compile_r, run_r, verify_r, compile_t, run_t,
          objective, baseline, rel_obj))
    conn.commit()
    conn.close()


def save_bo_config(db_path, iteration, targets, suggestion, target_meta):
    conn = sqlite3.connect(db_path)
    for i, t in enumerate(targets):
        enable = suggestion[2 * i]
        value = suggestion[2 * i + 1]
        value_op = target_meta[i][0]
        conn.execute("""
            INSERT INTO bo_configs
            (iteration, target_idx, enabled, module, function, path,
             prior_type, sampled_value, value_op)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (iteration, i, enable, t.module, t.function, t.path_str,
              t.prior_type, float(value), value_op))
    conn.commit()
    conn.close()


def get_last_iteration(db_path: str) -> int:
    if not os.path.exists(db_path):
        return -1
    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT MAX(iteration) FROM bo_results")
    row = cur.fetchone()
    conn.close()
    return row[0] if row[0] is not None else -1


# ---------------------------------------------------------------------------
# Main BO loop
# ---------------------------------------------------------------------------

def run_bo_rtp(args):
    with open(args.config) as f:
        config = json.load(f)

    tools_cfg = config["tools"]
    sysprog_cfg = config["sys_prog"]["available"][config["sys_prog"]["active"]]
    sys_prog_src = sysprog_cfg["src_dir"]

    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    bo_db_path = str(working_dir / "rtp_bo_results.sqlite")
    init_bo_db(bo_db_path)

    # Load top-k targets
    composite_db_file = args.composite_db.replace("sqlite:///", "")
    targets = load_top_k_targets(composite_db_file, args.top_k)
    if not targets:
        logger.error("No tuning targets found!")
        return

    # Build search space
    dimensions, target_meta = build_search_space(targets)
    logger.info(f"Search space: {len(dimensions)} dimensions "
                f"({len(targets)} targets × 2)")

    # Initialize BO optimizer with Random Forest surrogate
    optimizer = Optimizer(
        dimensions=dimensions,
        base_estimator="RF",      # Random Forest — handles mixed spaces well
        n_initial_points=50,      # Random exploration before fitting model
        acq_func="EI",            # Expected Improvement
        random_state=42,
    )

    # Load function cache
    with open(args.function_cache, "rb") as f:
        fn_list = pickle.load(f)
    fn_cache = {(fn.module, fn.name): fn for fn in fn_list}
    logger.info(f"Loaded {len(fn_cache)} functions from cache.")

    # Setup benchmarks
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
    logger.info(f"Created {len(test_cases)} test cases.")

    # Load baseline
    with open(args.baseline_cache, "rb") as f:
        baseline_cache = pickle.load(f)

    objective_metric = CodeSizeObjective(Path(tools_cfg["llvm-size"]))

    # Instrument compiler
    sys_prog_bld_dir = Path(sysprog_cfg["build_dir"])
    instrumented_bins = sys_prog_bld_dir / "instrumented" / "bin"

    target_modules = {}
    for t in targets:
        if t.module not in target_modules:
            target_modules[t.module] = set()
        target_modules[t.module].add(t.function)

    logger.info("Setting up system program...")
    opt_program = SysProg(
        tools_cfg, sysprog_cfg,
        Path(sys_prog_src), sys_prog_bld_dir,
        objective_metric,
        cpus=args.cpus, verbose=args.verbose,
    )

    logger.info(f"Instrumenting {sum(len(v) for v in target_modules.values())} "
                f"functions in {len(target_modules)} modules...")
    opt_program.instrument(target_modules)

    # Resume support
    start_iter = get_last_iteration(bo_db_path) + 1
    if start_iter > 0:
        logger.info(f"Resuming from iteration {start_iter}")
        # TODO: re-feed previous results to optimizer for true resume
        # For now, we just skip already-done iterations

    # BO main loop
    best_rel_obj = float("inf")
    best_config = None

    for iteration in range(start_iter, args.iterations):
        logger.info(f"=== BO Iteration {iteration + 1}/{args.iterations} ===")

        # Ask BO for next configuration
        suggestion = optimizer.ask()

        # Decode which targets are enabled
        selected = decode_suggestion(suggestion, targets, target_meta)
        num_enabled = len(selected)

        # Save config
        save_bo_config(bo_db_path, iteration, targets, suggestion, target_meta)

        config_desc = "; ".join(
            f"{t.function}:{t.path_str}={v}({op})"
            for t, v, op in selected
        ) if selected else "none"

        logger.info(f"  Enabled targets: {num_enabled}/{len(targets)}")

        # If no targets enabled, evaluate as baseline (cost = 1.0)
        if num_enabled == 0:
            for tc_name in test_cases:
                save_bo_result(bo_db_path, iteration, tc_name, 0, "none",
                               "NA", "NA", "NA", None, None, None, None, 1.0)
            optimizer.tell(suggestion, 1.0)
            continue

        # Generate extension
        ext_code = generate_bo_extension(selected, fn_cache, sys_prog_src)

        if ext_code is None:
            logger.warning(f"  Extension generation failed")
            for tc_name in test_cases:
                save_bo_result(bo_db_path, iteration, tc_name, num_enabled,
                               config_desc, "EXT_GEN_FAIL", "NA", "NA",
                               None, None, None, None, None)
            optimizer.tell(suggestion, 2.0)  # penalty
            continue

        # Build extension
        probe_wd = try_create_dir(working_dir / "bo_probe", use_time=True)
        if not probe_wd:
            logger.error("  Failed to create probe directory")
            optimizer.tell(suggestion, 2.0)
            continue

        try:
            ext = ProbeExtension(ext_code)
            ext_lib = ext.build_library(probe_wd, tools_cfg)
        except Exception as e:
            logger.warning(f"  Extension build failed: {e}")
            for tc_name in test_cases:
                save_bo_result(bo_db_path, iteration, tc_name, num_enabled,
                               config_desc, "EXT_BUILD_FAIL", "NA", "NA",
                               None, None, None, None, None)
            if probe_wd.exists():
                shutil.rmtree(probe_wd, ignore_errors=True)
            optimizer.tell(suggestion, 2.0)
            continue

        # Evaluate across all test cases
        # BO objective = average rel_objective across test cases
        rel_objs = []

        for tc_name, tc in test_cases.items():
            timer = Timer()

            timer.start()
            compile_result = tc.compile(
                instrumented_bins, ext_lib,
                memory_limit=args.probe_mem_limit
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
                    tc, memory_limit=args.probe_mem_limit
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

            save_bo_result(bo_db_path, iteration, tc_name, num_enabled,
                           config_desc,
                           compile_result.name, run_result.name,
                           verify_result.name,
                           compile_time, run_time,
                           obj_val, baseline_val, rel_obj)

            logger.info(f"  {tc_name}: compile={compile_result.name} "
                        f"verify={verify_result.name} rel_obj={rel_obj}")

        # Cleanup
        if probe_wd.exists():
            shutil.rmtree(probe_wd, ignore_errors=True)

        # Feed result to BO
        # Objective: minimize average rel_objective (lower = smaller binary)
        if rel_objs:
            avg_rel = np.mean(rel_objs)
            optimizer.tell(suggestion, avg_rel)

            if avg_rel < best_rel_obj:
                best_rel_obj = avg_rel
                best_config = config_desc
                logger.info(f"  *** New best: avg_rel={avg_rel:.6f} ***")
        else:
            # All failed — penalize
            optimizer.tell(suggestion, 2.0)

    # Summary
    logger.info("=" * 60)
    logger.info("BO RTP tuning completed.")
    logger.info(f"Best avg rel_objective: {best_rel_obj:.6f}")
    logger.info(f"Best config: {best_config}")

    conn = sqlite3.connect(bo_db_path)
    cur = conn.execute("""
        SELECT COUNT(*),
               MIN(rel_objective),
               AVG(rel_objective),
               SUM(CASE WHEN rel_objective < 1.0 THEN 1 ELSE 0 END)
        FROM bo_results
        WHERE verify_result = 'SUCCESS' AND rel_objective IS NOT NULL
    """)
    row = cur.fetchone()
    if row and row[0]:
        logger.info(f"Total verified: {row[0]}, min_rel={row[1]:.4f}, "
                     f"avg_rel={row[2]:.4f}, reductions={row[3]}")
    conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="RTP with Bayesian Optimization (RF surrogate)"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--composite_db", required=True)
    parser.add_argument("--working_dir", required=True)
    parser.add_argument("--function_cache", required=True)
    parser.add_argument("--baseline_cache", required=True)
    parser.add_argument("--iterations", type=int, default=2000,
                        help="BO iterations (default: 2000)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-k targets to include in search (default: 10)")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--probe_mem_limit", type=int, default=1024)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(processName)-12s %(name)-30s %(levelname)-8s %(message)s",
    )
    run_bo_rtp(args)
