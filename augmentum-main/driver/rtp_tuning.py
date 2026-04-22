#!/usr/bin/env python3
"""
Random Tuning Profile (RTP) — implements the Random Search Tuning from:

  Seeker et al., "Revealing Compiler Heuristics through Automated Discovery
  and Optimization", CGO 2024, Section VII-E.

Given a composite-search heuristic DB that already identified tuning targets,
this script:
  1. Queries successful tuning targets (paths with obj_improvement and a
     fitted prior).
  2. For each search iteration, randomly selects a subset of targets,
     randomly picks a prior for each, randomly samples a value from the
     prior's value range, and generates a single extension library that
     instruments ALL selected targets simultaneously.
  3. Compiles each benchmark test case with the multi-target extension,
     measures binary size, and verifies correctness.
  4. Records results to an RTP-specific SQLite database.

Usage:
  python rtp_tuning.py --config evaluation_config.json \\
      --composite_db sqlite:///path/to/subset_composite.sqlite \\
      --working_dir /path/to/working_dir_rtp \\
      --iterations 6400 --max_targets 10 --cpus 8
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Augmentum imports — reuse as much of the existing infrastructure as we can
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from augmentum.benchmarks import BenchmarkFactory, ExecutionResult
from augmentum.function import Function
from augmentum.objectives import CodeSizeObjective
from augmentum.probes import (
    add_null_check,
    generate_path_code,
    type_descs_to_cpp_arg_types,
    type_descs_to_cpp_arg_vals,
    type_descs_to_cpp_args,
)
from augmentum.sysProg import ProbeExtension, SysProg
from augmentum.sysUtils import try_create_dir
from augmentum.timer import Timer
from augmentum.type_descs import StructTypeDesc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TuningTarget:
    """One discovered tuning target from the composite DB."""
    module: str
    function: str
    path_str: str
    prior_type: str   # e.g. "Integer Scale Prior"
    prior_data: str   # e.g. "127,127" or "-128,0"
    # Composite-discovered best value (set only when --use_best_value is on).
    # When present, codegen emits a deterministic literal instead of per-call
    # uniform random sampling from the prior range.
    best_value: Optional[str] = None
    best_rel_obj: Optional[float] = None


@dataclass
class SelectedTarget:
    """A target selected for an RTP iteration, with its prior range info."""
    target: TuningTarget
    value: float       # sampled value (used for logging/DB only)
    value_op: Optional[str]  # None = static, "+" = offset, "*" = scale
    range_lo: float = 0    # lower bound of the prior value range
    range_hi: float = 0    # upper bound of the prior value range
    is_int: bool = True    # whether to sample integers or reals


@dataclass
class RTPResult:
    """Result of one RTP evaluation."""
    iteration: int
    test_case: str
    num_targets: int
    targets_desc: str
    compile_result: str
    run_result: str
    verify_result: str
    compile_time: Optional[float]
    run_time: Optional[float]
    objective: Optional[float]
    baseline_objective: Optional[float]
    rel_objective: Optional[float]


# ---------------------------------------------------------------------------
# Query composite DB for tuning targets
# ---------------------------------------------------------------------------

def load_tuning_targets(composite_db_path: str) -> List[TuningTarget]:
    """Load all successful tuning targets from the composite heuristic DB."""
    conn = sqlite3.connect(composite_db_path)
    cursor = conn.execute("""
        SELECT DISTINCT ph.module, ph.function, ph.path, pr.prior, pr.data
        FROM path_heuristics ph
        JOIN prior_results pr
          ON (ph.module = pr.module
              AND ph.function = pr.function
              AND ph.path = pr.path)
        WHERE ph.obj_improvement = 1
          AND pr.success = 1
          AND pr.prior NOT IN ('Null Prior', 'Broadcast Prior')
    """)
    targets = []
    for row in cursor:
        targets.append(TuningTarget(
            module=row[0],
            function=row[1],
            path_str=row[2],
            prior_type=row[3],
            prior_data=row[4],
        ))
    conn.close()
    logger.info(f"Loaded {len(targets)} tuning targets from composite DB.")
    return targets


def load_tuning_targets_with_best_value(
    composite_db_path: str,
) -> List[TuningTarget]:
    """
    For each (module, function, path) that showed improvement in composite,
    return ONE TuningTarget whose (prior_type, best_value) comes from the
    specific probe row that achieved the lowest rel_objective. Codegen then
    emits that deterministic value rather than sampling from a range, aligning
    RTP's search space with BO --use_best_value and SA --use_best_value.
    """
    conn = sqlite3.connect(composite_db_path)
    cursor = conn.execute("""
        SELECT pl.module, pl.function, pl.path,
               MIN(CAST(pl.rel_objective AS FLOAT)) AS best_rel
        FROM path_heuristics ph
        JOIN probe_log pl
          ON (ph.module = pl.module
              AND ph.function = pl.function
              AND ph.path = pl.path)
        WHERE ph.obj_improvement = 1
          AND pl.verify = 'SUCCESS'
          AND pl.rel_objective IS NOT NULL
          AND CAST(pl.rel_objective AS FLOAT) < 1.0
        GROUP BY pl.module, pl.function, pl.path
    """)
    top_paths = cursor.fetchall()

    targets: List[TuningTarget] = []
    for module, function, path_str, best_rel in top_paths:
        cur2 = conn.execute("""
            SELECT prior, value FROM probe_log
            WHERE module = ? AND function = ? AND path = ?
              AND verify = 'SUCCESS'
              AND rel_objective IS NOT NULL
              AND ABS(CAST(rel_objective AS FLOAT) - ?) < 1e-9
            LIMIT 1
        """, (module, function, path_str, best_rel))
        row = cur2.fetchone()
        if row is None:
            continue
        best_prior_type, best_value = row[0], row[1]

        cur3 = conn.execute("""
            SELECT data FROM prior_results
            WHERE module = ? AND function = ? AND path = ? AND prior = ?
              AND success = 1
            LIMIT 1
        """, (module, function, path_str, best_prior_type))
        row3 = cur3.fetchone()
        prior_data = row3[0] if row3 else ""

        targets.append(TuningTarget(
            module=module, function=function, path_str=path_str,
            prior_type=best_prior_type, prior_data=prior_data,
            best_value=best_value, best_rel_obj=best_rel,
        ))

    conn.close()
    logger.info(
        f"Loaded {len(targets)} tuning targets with composite best values."
    )
    return targets


# ---------------------------------------------------------------------------
# Value sampling from prior data
# ---------------------------------------------------------------------------

def sample_value_from_prior(target: TuningTarget) -> SelectedTarget:
    """
    Given a tuning target with its prior type and data, sample a random value
    and determine the value operation.

    Prior data formats:
      Boolean Prior:        "0,True#1,True"
      All Integers Prior:   "val,True#val,True#..."
      Integer Range Prior:  "min,max"
      Integer Offset Prior: "theta,phi"   → value in [orig-theta, orig+phi]
      Integer Scale Prior:  "alpha,beta"  → value in [orig*(1-alpha), orig*(1+beta)]
      (same patterns for Real variants)
    """
    ptype = target.prior_type
    pdata = target.prior_data

    if ptype == "Boolean Prior":
        value = random.choice([0, 1])
        return SelectedTarget(target=target, value=value, value_op=None,
                              range_lo=0, range_hi=1, is_int=True)

    elif ptype in ("All Integers Prior", "All Reals Prior"):
        # data format: "val,True#val,True#..."
        is_int = "Integer" in ptype
        entries = pdata.split("#")
        values = []
        for e in entries:
            parts = e.split(",")
            if len(parts) >= 2 and parts[1] == "True":
                try:
                    values.append(int(parts[0]) if is_int else float(parts[0]))
                except ValueError:
                    continue
        if values:
            value = random.choice(values)
            lo, hi = min(values), max(values)
        else:
            value, lo, hi = 0, 0, 0
        return SelectedTarget(target=target, value=value, value_op=None,
                              range_lo=lo, range_hi=hi, is_int=is_int)

    elif ptype in ("Integer Range Prior", "Real Range Prior"):
        # data format: "min,max"
        is_int = "Integer" in ptype
        parts = pdata.split(",")
        if is_int:
            lo, hi = int(parts[0]), int(parts[1])
            value = random.randint(lo, hi)
        else:
            lo, hi = float(parts[0]), float(parts[1])
            value = random.uniform(lo, hi)
        return SelectedTarget(target=target, value=value, value_op=None,
                              range_lo=lo, range_hi=hi, is_int=is_int)

    elif ptype in ("Integer Offset Prior", "Real Offset Prior"):
        # data format: "theta,phi" → offset in [-theta, +phi]
        is_int = "Integer" in ptype
        parts = pdata.split(",")
        if is_int:
            theta, phi = int(parts[0]), int(parts[1])
            value = random.randint(-theta, phi)
        else:
            theta, phi = float(parts[0]), float(parts[1])
            value = random.uniform(-theta, phi)
        return SelectedTarget(target=target, value=value, value_op="+",
                              range_lo=-theta, range_hi=phi, is_int=is_int)

    elif ptype in ("Integer Scale Prior", "Real Scale Prior"):
        # data format: "alpha,beta" where alpha = lower bound, beta = upper bound
        # ScalePrior probes: lower → ScaleProbe(1 - alpha), upper → ScaleProbe(1 + beta)
        # ScaleProbe uses value_op="*": *probed = original_value * factor
        # So the valid factor range is [1 - alpha, 1 + beta]
        is_int = "Integer" in ptype
        parts = pdata.split(",")
        if is_int:
            alpha, beta = int(parts[0]), int(parts[1])
            lo, hi = 1 - alpha, 1 + beta
            value = random.randint(lo, hi) if lo <= hi else lo
        else:
            alpha, beta = float(parts[0]), float(parts[1])
            lo, hi = 1 - alpha, 1 + beta
            value = random.uniform(lo, hi)
        return SelectedTarget(target=target, value=value, value_op="*",
                              range_lo=lo, range_hi=hi, is_int=is_int)

    else:
        logger.warning(f"Unknown prior type: {ptype}, using static 0")
        return SelectedTarget(target=target, value=0, value_op=None,
                              range_lo=0, range_hi=0, is_int=True)


# ---------------------------------------------------------------------------
# Multi-target extension code generation
# ---------------------------------------------------------------------------

def generate_rtp_extension_with_functions(
    selected_targets: List[SelectedTarget],
    functions_by_key: Dict[Tuple[str, str], Function],
    sys_prog_src: str,
    log_file: str,
) -> Optional[str]:
    """
    Generate a multi-target extension using the full Function objects for
    proper struct definitions and path access code generation.

    Each target gets its own modified_function/original_function pair,
    and the listener registers all of them.
    """
    from collections import defaultdict

    # Group by (module, function) — pick one target per function
    fn_targets = defaultdict(list)
    for st in selected_targets:
        key = (st.target.module, st.target.function)
        fn_targets[key].append(st)

    unique_targets = []
    for key, sts in fn_targets.items():
        st = random.choice(sts)
        if key in functions_by_key:
            unique_targets.append((st, functions_by_key[key]))
        else:
            logger.warning(f"Function {key} not found in cache, skipping")

    if not unique_targets:
        return None

    # Collect all struct definitions via a shared dependency graph
    from augmentum.probes import build_struct_dependency_graph, get_struct_definitions_from_graph
    from augmentum.type_descs import StructTypeDesc
    shared_struct_graph: Dict[StructTypeDesc, Any] = {}

    modified_functions_code = []
    register_blocks = []
    unregister_blocks = []

    # First pass: collect struct deps from all functions
    for idx, (st, fn) in enumerate(unique_targets):
        build_struct_dependency_graph(fn.type, shared_struct_graph, deep_traversal=True)

    # Generate combined struct definitions (no duplicates)
    combined_struct_defs = get_struct_definitions_from_graph(shared_struct_graph)

    for idx, (st, fn) in enumerate(unique_targets):
        orig_type = f"original_t_{idx}"
        orig_id = f"original_fn_{idx}"
        mod_id = f"modified_fn_{idx}"

        # Parse path and generate path code
        path_obj = _str_to_path(st.target.path_str, fn)
        if path_obj is None:
            logger.warning(f"Could not parse path {st.target.path_str} for {fn.name}")
            continue

        try:
            path_code, probed_type, null_check_id = generate_path_code(
                path_obj, fn, "probed"
            )
        except Exception as e:
            logger.warning(f"Path code generation failed for {st.target.path_str}: {e}")
            continue

        # Generate function signature components
        return_type = fn.type.return_type.get_cpp_type().get_type_string()
        arg_vals = type_descs_to_cpp_arg_vals(fn.type.arg_types)
        arg_types_str = type_descs_to_cpp_arg_types(fn.type.arg_types)
        args = type_descs_to_cpp_args(fn.type.arg_types)
        probed_type_str = probed_type.get_cpp_type().get_type_string()

        # Value source:
        # - If target.best_value is set (use_best_value mode), emit a
        #   deterministic literal so RTP samples from the composite-peak
        #   landscape (aligned with BO/SA --use_best_value).
        # - Otherwise fall back to per-call random sampling in [lo, hi]
        #   (matches paper RTP).
        lo, hi = st.range_lo, st.range_hi
        if st.target.best_value is not None:
            if st.is_int:
                rand_expr = f"({probed_type_str})({int(st.target.best_value)})"
            else:
                rand_expr = f"({probed_type_str})({float(st.target.best_value)})"
        elif st.is_int:
            if lo == hi:
                rand_expr = f"({probed_type_str})({int(lo)})"
            else:
                rand_expr = f"({probed_type_str})({int(lo)} + (rand() % ({int(hi)} - {int(lo)} + 1)))"
        else:
            rand_expr = (f"({probed_type_str})({lo} + "
                         f"((double)rand() / RAND_MAX) * ({hi} - {lo}))")

        if st.value_op is None:
            calc_probed = rand_expr
        elif st.value_op == "+":
            calc_probed = f"original_value + {rand_expr}"
        elif st.value_op == "*":
            calc_probed = f"original_value * {rand_expr}"
        else:
            calc_probed = rand_expr

        # Build the return logic
        if return_type == "void":
            return_stmt, orig_return = "", ""
        else:
            return_stmt = "return r;"
            orig_return = f"{return_type} r = "

        # Build probe code (modify the value)
        probed_code = f"""
        {path_code}
        {probed_type_str} original_value = *probed;
        *probed = {calc_probed};
"""
        probed_code = add_null_check(null_check_id, probed_code)

        fn_body = f"""
    {orig_return}{orig_id}({arg_vals});
    {probed_code}
    {return_stmt}
"""

        fn_code = f"""
// Target {idx}: {st.target.module} :: {fn.name}
// Path: {st.target.path_str} | Prior: {st.target.prior_type} | Value: {st.value} | Op: {st.value_op}
typedef {return_type} (*{orig_type})({arg_types_str});
{orig_type} {orig_id};

{return_type} {mod_id}({args}) {{
    {fn_body}
}}
"""
        modified_functions_code.append(fn_code)

        # Register block
        register_blocks.append(f"""
        if (pt.get_module_name() == "{sys_prog_src}/{st.target.module}" &&
            pt.get_name() == "{fn.name}") {{
            if (!pt.is_replaced()) {{
                pt.replace((Fn) &{mod_id});
                {orig_id} = ({orig_type}) pt.original_direct();
            }}
        }}
""")
        unregister_blocks.append(f"""
        if (pt.get_module_name() == "{sys_prog_src}/{st.target.module}" &&
            pt.get_name() == "{fn.name}") {{
            if (pt.is_replaced()) {{
                pt.reset();
            }}
        }}
""")

    if not modified_functions_code:
        return None

    code = f"""
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <mutex>
#include "augmentum.h"

// Seed RNG once at load time so per-call sampling is non-deterministic
static struct _RTPRngInit {{ _RTPRngInit() {{ srand((unsigned)time(nullptr)); }} }} _rtp_rng_init;

using namespace augmentum;

{combined_struct_defs}

{"".join(modified_functions_code)}

struct RTPListener: Listener {{
    void on_extension_point_register(FnExtensionPoint& pt) {{
{"".join(register_blocks)}
    }}

    void on_extension_point_unregister(FnExtensionPoint& pt) {{
{"".join(unregister_blocks)}
    }}
}};
ListenerLifeCycle<RTPListener> rtpListener;
"""
    return code


def _str_to_path(path_str: str, function: Function):
    """
    Convert a path string like 'A0.D.S2.S0.S0.S3.L.L.T-i8' back to a Path object
    using the function's type information.
    """
    import augmentum.paths as ap

    fn_paths = function.get_paths()
    for p in fn_paths:
        if str(p) == path_str:
            return p

    # If exact match not found, try to find by string representation
    logger.warning(f"Path {path_str} not found among {len(fn_paths)} paths for {function.name}")
    return None


# ---------------------------------------------------------------------------
# RTP result database
# ---------------------------------------------------------------------------

def init_rtp_db(db_path: str):
    """Create the RTP results database."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rtp_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER NOT NULL,
            test_case TEXT NOT NULL,
            num_targets INTEGER NOT NULL,
            targets_desc TEXT NOT NULL,
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
        CREATE TABLE IF NOT EXISTS rtp_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER NOT NULL,
            target_idx INTEGER NOT NULL,
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


def save_rtp_result(db_path: str, result: RTPResult):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO rtp_results
        (iteration, test_case, num_targets, targets_desc,
         compile_result, run_result, verify_result,
         compile_time, run_time, objective, baseline_objective, rel_objective)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result.iteration, result.test_case, result.num_targets,
        result.targets_desc, result.compile_result, result.run_result,
        result.verify_result, result.compile_time, result.run_time,
        result.objective, result.baseline_objective, result.rel_objective,
    ))
    conn.commit()
    conn.close()


def save_rtp_profile(db_path: str, iteration: int,
                     selected_targets: List[SelectedTarget]):
    conn = sqlite3.connect(db_path)
    for idx, st in enumerate(selected_targets):
        conn.execute("""
            INSERT INTO rtp_profiles
            (iteration, target_idx, module, function, path,
             prior_type, sampled_value, value_op)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            iteration, idx, st.target.module, st.target.function,
            st.target.path_str, st.target.prior_type,
            st.value, st.value_op,
        ))
    conn.commit()
    conn.close()


def get_last_completed_iteration(db_path: str) -> int:
    """Get the last completed iteration to support resuming."""
    if not os.path.exists(db_path):
        return -1
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT MAX(iteration) FROM rtp_results")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row[0] is not None else -1


# ---------------------------------------------------------------------------
# Main RTP loop
# ---------------------------------------------------------------------------

def load_function_cache(cache_path: str) -> Dict[Tuple[str, str], Function]:
    """Load function objects from the pickle cache."""
    with open(cache_path, "rb") as f:
        functions = pickle.load(f)

    fn_dict = {}
    for fn in functions:
        key = (fn.module, fn.name)
        fn_dict[key] = fn

    logger.info(f"Loaded {len(fn_dict)} functions from cache.")
    return fn_dict


def run_rtp(args):
    """Main RTP execution loop."""
    # Load config
    with open(args.config) as f:
        config = json.load(f)

    tools_cfg = config["tools"]
    sysprog_cfg = config["sys_prog"]["available"][config["sys_prog"]["active"]]
    sys_prog_src = sysprog_cfg["src_dir"]

    # Setup directories
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    rtp_db_path = str(working_dir / "rtp_results.sqlite")
    init_rtp_db(rtp_db_path)

    # Load tuning targets
    composite_db_file = args.composite_db.replace("sqlite:///", "")
    if args.use_best_value:
        targets = load_tuning_targets_with_best_value(composite_db_file)
    else:
        targets = load_tuning_targets(composite_db_file)
    if not targets:
        logger.error("No tuning targets found in composite DB!")
        return

    # Load function cache
    fn_cache = load_function_cache(args.function_cache)

    # Setup benchmark factory
    benchmark_cfgs = [
        config["benchmark"]["available"][name]
        for name in config["benchmark"]["active"]
    ]
    benchmark_factory = BenchmarkFactory(
        tools_cfg, benchmark_cfgs, verbose=args.verbose
    )
    # Setup benchmark working directory and test cases
    rtp_bench_dir = working_dir / "benchmarks"
    rtp_bench_dir.mkdir(parents=True, exist_ok=True)
    test_cases = benchmark_factory.setup_benchmark_instance(rtp_bench_dir)
    logger.info(f"Created {len(test_cases)} test cases.")

    # Load baseline cache
    with open(args.baseline_cache, "rb") as f:
        baseline_cache = pickle.load(f)
    logger.info(f"Loaded baseline cache with {len(baseline_cache)} entries.")

    # Setup objective metric
    objective_metric = CodeSizeObjective(Path(tools_cfg["llvm-size"]))

    # Instrumented compiler binaries path
    sys_prog_bld_dir = Path(sysprog_cfg["build_dir"])
    instrumented_bins = sys_prog_bld_dir / "instrumented" / "bin"

    # Determine which functions need to be instrumented
    # We need ALL functions from our tuning targets to be instrumented
    target_modules = {}
    for t in targets:
        if t.module not in target_modules:
            target_modules[t.module] = set()
        target_modules[t.module].add(t.function)

    # Setup SysProg for instrumentation
    logger.info("Setting up system program...")
    opt_program = SysProg(
        tools_cfg, sysprog_cfg,
        Path(sys_prog_src), sys_prog_bld_dir,
        objective_metric,
        cpus=args.cpus, verbose=args.verbose,
    )

    # Instrument all target functions
    logger.info(f"Instrumenting {sum(len(v) for v in target_modules.values())} "
                f"functions in {len(target_modules)} modules...")
    opt_program.instrument(target_modules)

    # Resume support
    start_iter = get_last_completed_iteration(rtp_db_path) + 1
    if start_iter > 0:
        logger.info(f"Resuming from iteration {start_iter}")

    # RTP main loop
    for iteration in range(start_iter, args.iterations):
        logger.info(f"=== RTP Iteration {iteration + 1}/{args.iterations} ===")

        # Randomly select how many targets to combine
        n_targets = random.randint(1, min(args.max_targets, len(targets)))

        # Randomly select targets (without replacement)
        selected = random.sample(targets, n_targets)

        # Sample values for each target
        selected_targets = [sample_value_from_prior(t) for t in selected]

        # Save the profile
        save_rtp_profile(rtp_db_path, iteration, selected_targets)

        # Generate multi-target extension code
        ext_code = generate_rtp_extension_with_functions(
            selected_targets, fn_cache, sys_prog_src, "/dev/null"
        )

        if ext_code is None:
            logger.warning(f"Iteration {iteration}: no valid extension generated, skipping")
            continue

        # Build extension library
        probe_wd = try_create_dir(working_dir / "rtp_probe", use_time=True)
        if not probe_wd:
            logger.error("Failed to create probe working directory")
            continue

        try:
            ext = ProbeExtension(ext_code)
            ext_lib = ext.build_library(probe_wd, tools_cfg)
        except Exception as e:
            logger.warning(f"Iteration {iteration}: extension build failed: {e}")
            # Record failure
            targets_desc = "; ".join(
                f"{st.target.function}:{st.target.path_str}={st.value}"
                for st in selected_targets
            )
            for tc_name, tc in test_cases.items():
                save_rtp_result(rtp_db_path, RTPResult(
                    iteration=iteration, test_case=tc_name,
                    num_targets=n_targets, targets_desc=targets_desc,
                    compile_result="EXT_BUILD_FAIL", run_result="NA",
                    verify_result="NA", compile_time=None, run_time=None,
                    objective=None, baseline_objective=None, rel_objective=None,
                ))
            # Cleanup
            if probe_wd.exists():
                shutil.rmtree(probe_wd, ignore_errors=True)
            continue

        # Test each benchmark with this extension
        targets_desc = "; ".join(
            f"{st.target.function}:{st.target.path_str}={st.value}"
            for st in selected_targets
        )

        for tc_name, tc in test_cases.items():
            timer = Timer()

            # Compile
            timer.start()
            compile_result = tc.compile(
                instrumented_bins, ext_lib,
                memory_limit=args.probe_mem_limit
            )
            compile_time = timer.stop()

            run_result_enum = ExecutionResult.NA
            verify_result_enum = ExecutionResult.NA
            run_time = None
            objective_val = None
            baseline_val = None
            rel_obj = None

            if compile_result == ExecutionResult.SUCCESS:
                # Run and measure
                timer.start()
                run_result_enum, obj_result = objective_metric.measure(
                    tc, memory_limit=args.probe_mem_limit
                )
                run_time = timer.stop()

                if run_result_enum == ExecutionResult.SUCCESS:
                    verify_result_enum = tc.verify()

                    if obj_result is not None:
                        objective_val = obj_result.score
                        if tc_name in baseline_cache:
                            baseline_val = baseline_cache[tc_name].objective.score
                            if baseline_val and baseline_val > 0:
                                rel_obj = objective_val / baseline_val

            save_rtp_result(rtp_db_path, RTPResult(
                iteration=iteration, test_case=tc_name,
                num_targets=n_targets, targets_desc=targets_desc,
                compile_result=str(compile_result.name),
                run_result=str(run_result_enum.name),
                verify_result=str(verify_result_enum.name),
                compile_time=compile_time, run_time=run_time,
                objective=objective_val, baseline_objective=baseline_val,
                rel_objective=rel_obj,
            ))

            logger.info(
                f"  {tc_name}: compile={compile_result.name} "
                f"run={run_result_enum.name} verify={verify_result_enum.name} "
                f"rel_obj={rel_obj}"
            )

        # Cleanup probe directory
        if probe_wd.exists():
            shutil.rmtree(probe_wd, ignore_errors=True)

    logger.info("RTP tuning completed.")

    # Print summary
    conn = sqlite3.connect(rtp_db_path)
    cursor = conn.execute("""
        SELECT COUNT(*), MIN(rel_objective), AVG(rel_objective),
               SUM(CASE WHEN rel_objective < 1.0 THEN 1 ELSE 0 END)
        FROM rtp_results
        WHERE verify_result = 'SUCCESS' AND rel_objective IS NOT NULL
    """)
    row = cursor.fetchone()
    if row and row[0] > 0:
        logger.info(f"Summary: {row[0]} successful probes, "
                    f"min rel_obj={row[1]:.4f}, avg rel_obj={row[2]:.4f}, "
                    f"{row[3]} with size reduction")
    conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Random Tuning Profile (RTP) search — combines multiple "
                    "tuning targets discovered by composite search."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to evaluation_config.json"
    )
    parser.add_argument(
        "--composite_db", required=True,
        help="SQLite URL for composite heuristic DB (e.g. sqlite:///path/to/db)"
    )
    parser.add_argument(
        "--working_dir", required=True,
        help="Working directory for RTP output"
    )
    parser.add_argument(
        "--function_cache", required=True,
        help="Path to function_cache.pickle"
    )
    parser.add_argument(
        "--baseline_cache", required=True,
        help="Path to baseline_cache.pickle"
    )
    parser.add_argument(
        "--iterations", type=int, default=6400,
        help="Number of RTP iterations (default: 6400, as in the paper)"
    )
    parser.add_argument(
        "--max_targets", type=int, default=10,
        help="Maximum number of targets per RTP profile (default: 10)"
    )
    parser.add_argument(
        "--cpus", type=int, default=8,
        help="Number of CPUs for builds"
    )
    parser.add_argument(
        "--probe_mem_limit", type=int, default=1024,
        help="Memory limit in MB for probe compilation/run"
    )
    parser.add_argument(
        "--use_best_value", action="store_true",
        help="For each target, use composite's best (prior, value) "
             "deterministically instead of sampling at runtime. Aligns "
             "RTP's search space with BO/SA --use_best_value."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(processName)-12s %(name)-30s %(levelname)-8s %(message)s",
    )

    run_rtp(args)
