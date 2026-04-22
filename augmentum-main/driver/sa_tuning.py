#!/usr/bin/env python3
"""
Simulated Annealing (SA) Tuning — extends the Random Tuning Profile (RTP)
baseline with a Simulated Annealing search strategy.

Key difference from RTP:
  - Maintains a *current configuration* (set of selected targets + values)
  - Each iteration: perturbs the current config slightly (neighbor move)
  - Accepts better configs always; accepts worse configs with probability
    exp(-delta / T) where T is the current temperature
  - Temperature decreases over time (cooling schedule), making the search
    more greedy as it progresses

This sits between RTP (pure random) and BO (model-guided) in sophistication,
and serves as an additional baseline for comparison.

Usage:
  python sa_tuning.py --config evaluation_config.json \\
      --composite_db sqlite:///path/to/subset_composite.sqlite \\
      --working_dir /path/to/working_dir_sa \\
      --iterations 6400 --max_targets 10 --cpus 8 \\
      --temp_init 1.0 --temp_min 0.01 --cooling_rate 0.995
"""

import argparse
import json
import logging
import math
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
# Data structures (identical to rtp_tuning.py)
# ---------------------------------------------------------------------------

@dataclass
class TuningTarget:
    module: str
    function: str
    path_str: str
    prior_type: str
    prior_data: str
    # Composite-discovered best value (set only when --use_best_value is on).
    # When present, codegen emits a deterministic literal instead of sampling
    # at each C++-side call.
    best_value: Optional[str] = None
    best_rel_obj: Optional[float] = None


@dataclass
class SelectedTarget:
    target: TuningTarget
    value: float
    value_op: Optional[str]
    range_lo: float = 0
    range_hi: float = 0
    is_int: bool = True


@dataclass
class SAResult:
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
    temperature: float          # SA-specific: temperature at this iteration
    accepted: bool              # SA-specific: was this move accepted?
    move_type: str              # SA-specific: 'initial', 'better', 'worse_accepted', 'worse_rejected', 'eval_failed', 'random_restart'


# ---------------------------------------------------------------------------
# Query composite DB (identical to rtp_tuning.py)
# ---------------------------------------------------------------------------

def load_tuning_targets(composite_db_path: str) -> List[TuningTarget]:
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
            module=row[0], function=row[1], path_str=row[2],
            prior_type=row[3], prior_data=row[4],
        ))
    conn.close()
    logger.info(f"Loaded {len(targets)} tuning targets from composite DB.")
    return targets


def load_tuning_targets_with_best_value(
    composite_db_path: str,
) -> List[TuningTarget]:
    """
    For each (module, function, path) that showed improvement in the composite
    search, return ONE TuningTarget whose (prior_type, best_value) is the
    specific probe row that achieved the lowest rel_objective. Codegen then
    emits that deterministic value rather than sampling from a range, giving
    SA subset selection over a space aligned with BO's --use_best_value.
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
# Value sampling (identical to rtp_tuning.py)
# ---------------------------------------------------------------------------

def sample_value_from_prior(target: TuningTarget) -> SelectedTarget:
    # When the target carries a composite-best value (--use_best_value mode),
    # codegen emits a fixed literal and ignores range/value here. Skip all
    # prior_data parsing so a missing/empty prior_data row can't crash us.
    if target.best_value is not None:
        is_int = "Real" not in target.prior_type
        if "Offset" in target.prior_type:
            value_op = "+"
        elif "Scale" in target.prior_type:
            value_op = "*"
        else:
            value_op = None
        return SelectedTarget(
            target=target,
            value=0 if is_int else 0.0,
            value_op=value_op,
            range_lo=0, range_hi=0, is_int=is_int,
        )

    ptype = target.prior_type
    pdata = target.prior_data

    if ptype == "Boolean Prior":
        value = random.choice([0, 1])
        return SelectedTarget(target=target, value=value, value_op=None,
                              range_lo=0, range_hi=1, is_int=True)

    elif ptype in ("All Integers Prior", "All Reals Prior"):
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
# SA-specific: neighbor move
# ---------------------------------------------------------------------------

def perturb_config(
    current: List[SelectedTarget],
    all_targets: List[TuningTarget],
    max_targets: int,
    use_best_value: bool = False,
) -> List[SelectedTarget]:
    """
    Generate a neighbor config from the current one.

    Random-value mode (default) mixes three moves:
      1. Resample a value for one existing target (50%)
      2. Swap one target for a different random target (30%)
      3. Add or remove one target (20%)

    With `use_best_value=True`, each target's value is fixed to its composite
    peak, so move #1 is a no-op. The probability is redistributed:
      - Swap (60%)
      - Add/remove (40%)
    """
    move = random.random()
    neighbor = list(current)  # shallow copy

    if not neighbor:
        # Fallback: generate a fresh random config
        n = random.randint(1, min(max_targets, len(all_targets)))
        selected = random.sample(all_targets, n)
        return [sample_value_from_prior(t) for t in selected]

    if use_best_value:
        swap_threshold, addrem_threshold = 0.6, 1.0
    else:
        swap_threshold, addrem_threshold = 0.8, 1.0
        if move < 0.5:
            # Resample value for one existing target
            idx = random.randrange(len(neighbor))
            neighbor[idx] = sample_value_from_prior(neighbor[idx].target)
            return neighbor

    if move < swap_threshold:
        # Swap one target for a new one
        idx = random.randrange(len(neighbor))
        current_targets = {st.target.function for st in neighbor}
        candidates = [t for t in all_targets if t.function not in current_targets]
        if candidates:
            new_target = random.choice(candidates)
            neighbor[idx] = sample_value_from_prior(new_target)

    elif move < addrem_threshold:
        # Add or remove a target
        if len(neighbor) > 1 and (len(neighbor) >= max_targets or random.random() < 0.5):
            idx = random.randrange(len(neighbor))
            neighbor.pop(idx)
        else:
            current_targets = {st.target.function for st in neighbor}
            candidates = [t for t in all_targets if t.function not in current_targets]
            if candidates:
                new_target = random.choice(candidates)
                neighbor.append(sample_value_from_prior(new_target))

    return neighbor


# ---------------------------------------------------------------------------
# Extension code generation (identical to rtp_tuning.py)
# ---------------------------------------------------------------------------

def generate_rtp_extension_with_functions(
    selected_targets: List[SelectedTarget],
    functions_by_key: Dict[Tuple[str, str], Function],
    sys_prog_src: str,
    log_file: str,
) -> Optional[str]:
    from collections import defaultdict

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

    from augmentum.probes import build_struct_dependency_graph, get_struct_definitions_from_graph
    shared_struct_graph: Dict[StructTypeDesc, Any] = {}

    modified_functions_code = []
    register_blocks = []
    unregister_blocks = []

    for idx, (st, fn) in enumerate(unique_targets):
        build_struct_dependency_graph(fn.type, shared_struct_graph, deep_traversal=True)

    combined_struct_defs = get_struct_definitions_from_graph(shared_struct_graph)

    for idx, (st, fn) in enumerate(unique_targets):
        orig_type = f"original_t_{idx}"
        orig_id = f"original_fn_{idx}"
        mod_id = f"modified_fn_{idx}"

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

        return_type = fn.type.return_type.get_cpp_type().get_type_string()
        arg_vals = type_descs_to_cpp_arg_vals(fn.type.arg_types)
        arg_types_str = type_descs_to_cpp_arg_types(fn.type.arg_types)
        args = type_descs_to_cpp_args(fn.type.arg_types)
        probed_type_str = probed_type.get_cpp_type().get_type_string()

        # Value source:
        # - If target.best_value is set (use_best_value mode), emit a
        #   deterministic literal — SA then searches subset selection over
        #   composite's single-target peak values.
        # - Otherwise fall back to per-call random sampling in [lo, hi].
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
    fn_paths = function.get_paths()
    for p in fn_paths:
        if str(p) == path_str:
            return p
    logger.warning(f"Path {path_str} not found among {len(fn_paths)} paths for {function.name}")
    return None


# ---------------------------------------------------------------------------
# SA result database
# ---------------------------------------------------------------------------

def init_sa_db(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sa_results (
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
            rel_objective REAL,
            temperature REAL NOT NULL,
            accepted INTEGER NOT NULL,
            move_type TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sa_profiles (
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


def save_sa_result(db_path: str, result: SAResult):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO sa_results
        (iteration, test_case, num_targets, targets_desc,
         compile_result, run_result, verify_result,
         compile_time, run_time, objective, baseline_objective, rel_objective,
         temperature, accepted, move_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result.iteration, result.test_case, result.num_targets,
        result.targets_desc, result.compile_result, result.run_result,
        result.verify_result, result.compile_time, result.run_time,
        result.objective, result.baseline_objective, result.rel_objective,
        result.temperature, int(result.accepted), result.move_type,
    ))
    conn.commit()
    conn.close()


def save_sa_profile(db_path: str, iteration: int,
                    selected_targets: List[SelectedTarget]):
    conn = sqlite3.connect(db_path)
    for idx, st in enumerate(selected_targets):
        conn.execute("""
            INSERT INTO sa_profiles
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
    if not os.path.exists(db_path):
        return -1
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT MAX(iteration) FROM sa_results")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row[0] is not None else -1


def get_best_rel_objective(db_path: str) -> Optional[float]:
    """Get the best (lowest) rel_objective seen so far for warm-starting SA state."""
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT MIN(rel_objective) FROM sa_results
        WHERE verify_result = 'SUCCESS' AND rel_objective IS NOT NULL
    """)
    row = cursor.fetchone()
    conn.close()
    return row[0] if row[0] is not None else None


def _load_config_for_iteration(
    conn: sqlite3.Connection,
    iteration: int,
    target_index: Dict[Tuple[str, str, str], TuningTarget],
) -> List[SelectedTarget]:
    """Reconstruct SelectedTargets for a given iteration from sa_profiles."""
    rows = conn.execute(
        "SELECT module, function, path, sampled_value, value_op "
        "FROM sa_profiles WHERE iteration = ? ORDER BY target_idx",
        (iteration,),
    ).fetchall()
    config: List[SelectedTarget] = []
    for module, function, path_str, sampled_value, value_op in rows:
        t = target_index.get((module, function, path_str))
        if t is None:
            logger.warning(
                f"Resume: target ({module}, {function}, {path_str}) not in "
                "composite DB, skipping"
            )
            continue
        st = sample_value_from_prior(t)
        st.value = sampled_value
        st.value_op = value_op
        config.append(st)
    return config


def load_last_accepted_config(
    db_path: str,
    targets: List[TuningTarget],
) -> Tuple[Optional[List[SelectedTarget]], Optional[float]]:
    """
    Restore the SA current state (config + score) from the most recent
    accepted iteration in the DB. Returns (None, None) if none exists.
    """
    if not os.path.exists(db_path):
        return None, None

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT MAX(iteration) FROM sa_results WHERE accepted = 1"
    ).fetchone()
    if not row or row[0] is None:
        conn.close()
        return None, None
    last_iter = row[0]

    row = conn.execute(
        "SELECT AVG(rel_objective) FROM sa_results "
        "WHERE iteration = ? AND rel_objective IS NOT NULL",
        (last_iter,),
    ).fetchone()
    last_score = row[0] if row and row[0] is not None else None

    target_index = {(t.module, t.function, t.path_str): t for t in targets}
    config = _load_config_for_iteration(conn, last_iter, target_index)
    conn.close()

    if not config:
        return None, None
    return config, last_score


def load_best_config(
    db_path: str,
    targets: List[TuningTarget],
) -> Tuple[Optional[List[SelectedTarget]], Optional[float]]:
    """
    Restore the single best config + score seen so far (for warm-starting
    best tracking on resume). Returns (None, None) if none exists.
    """
    if not os.path.exists(db_path):
        return None, None

    conn = sqlite3.connect(db_path)
    # Pick the iteration whose avg rel_objective is the smallest
    row = conn.execute("""
        SELECT iteration, AVG(rel_objective) AS avg_obj
        FROM sa_results
        WHERE verify_result = 'SUCCESS' AND rel_objective IS NOT NULL
        GROUP BY iteration
        ORDER BY avg_obj ASC
        LIMIT 1
    """).fetchone()
    if not row or row[0] is None:
        conn.close()
        return None, None
    best_iter, best_score = row[0], row[1]

    target_index = {(t.module, t.function, t.path_str): t for t in targets}
    config = _load_config_for_iteration(conn, best_iter, target_index)
    conn.close()

    if not config:
        return None, None
    return config, best_score


# ---------------------------------------------------------------------------
# Function cache loader
# ---------------------------------------------------------------------------

def load_function_cache(cache_path: str) -> Dict[Tuple[str, str], Function]:
    with open(cache_path, "rb") as f:
        functions = pickle.load(f)
    fn_dict = {}
    for fn in functions:
        key = (fn.module, fn.name)
        fn_dict[key] = fn
    logger.info(f"Loaded {len(fn_dict)} functions from cache.")
    return fn_dict


# ---------------------------------------------------------------------------
# Main SA loop
# ---------------------------------------------------------------------------

def evaluate_config(
    selected_targets: List[SelectedTarget],
    fn_cache: Dict,
    sys_prog_src: str,
    working_dir: Path,
    test_cases: dict,
    instrumented_bins: Path,
    objective_metric,
    baseline_cache: dict,
    args,
    iteration: int,
    temperature: float,
) -> Tuple[Optional[float], List[SAResult]]:
    """
    Build and evaluate a config. Returns (avg_rel_objective, pending_results).
    `pending_results` has one SAResult per test case with accepted/move_type
    left as placeholders — the caller fills these in after the SA accept
    decision and then writes them to the DB.
    """
    targets_desc = "; ".join(
        f"{st.target.function}:{st.target.path_str}={st.value}"
        for st in selected_targets
    )
    n_targets = len(selected_targets)
    pending: List[SAResult] = []

    ext_code = generate_rtp_extension_with_functions(
        selected_targets, fn_cache, sys_prog_src, "/dev/null"
    )

    if ext_code is None:
        logger.warning(f"Iteration {iteration}: no valid extension generated, skipping")
        return None, pending

    probe_wd = try_create_dir(working_dir / "sa_probe", use_time=True)
    if not probe_wd:
        logger.error("Failed to create probe working directory")
        return None, pending

    try:
        ext = ProbeExtension(ext_code)
        ext_lib = ext.build_library(probe_wd, args._tools_cfg)
    except Exception as e:
        logger.warning(f"Iteration {iteration}: extension build failed: {e}")
        for tc_name in test_cases:
            pending.append(SAResult(
                iteration=iteration, test_case=tc_name,
                num_targets=n_targets, targets_desc=targets_desc,
                compile_result="EXT_BUILD_FAIL", run_result="NA",
                verify_result="NA", compile_time=None, run_time=None,
                objective=None, baseline_objective=None, rel_objective=None,
                temperature=temperature, accepted=False, move_type="",
            ))
        if probe_wd.exists():
            shutil.rmtree(probe_wd, ignore_errors=True)
        return None, pending

    rel_objectives = []
    timer = Timer()

    for tc_name, tc in test_cases.items():
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
                            rel_objectives.append(rel_obj)

        pending.append(SAResult(
            iteration=iteration, test_case=tc_name,
            num_targets=n_targets, targets_desc=targets_desc,
            compile_result=str(compile_result.name),
            run_result=str(run_result_enum.name),
            verify_result=str(verify_result_enum.name),
            compile_time=compile_time, run_time=run_time,
            objective=objective_val, baseline_objective=baseline_val,
            rel_objective=rel_obj,
            temperature=temperature, accepted=False, move_type="",
        ))

        logger.info(
            f"  {tc_name}: compile={compile_result.name} "
            f"run={run_result_enum.name} verify={verify_result_enum.name} "
            f"rel_obj={rel_obj}"
        )

    if probe_wd.exists():
        shutil.rmtree(probe_wd, ignore_errors=True)

    avg = sum(rel_objectives) / len(rel_objectives) if rel_objectives else None
    return avg, pending


def run_sa(args):
    """Main Simulated Annealing execution loop."""
    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"random seeded with {args.seed}")

    with open(args.config) as f:
        config = json.load(f)

    tools_cfg = config["tools"]
    sysprog_cfg = config["sys_prog"]["available"][config["sys_prog"]["active"]]
    sys_prog_src = sysprog_cfg["src_dir"]

    # Stash tools_cfg on args so evaluate_config can access it
    args._tools_cfg = tools_cfg

    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    sa_db_path = str(working_dir / "sa_results.sqlite")
    init_sa_db(sa_db_path)

    composite_db_file = args.composite_db.replace("sqlite:///", "")
    if args.use_best_value:
        targets = load_tuning_targets_with_best_value(composite_db_file)
    else:
        targets = load_tuning_targets(composite_db_file)
    if not targets:
        logger.error("No tuning targets found in composite DB!")
        return

    fn_cache = load_function_cache(args.function_cache)

    benchmark_cfgs = [
        config["benchmark"]["available"][name]
        for name in config["benchmark"]["active"]
    ]
    benchmark_factory = BenchmarkFactory(
        tools_cfg, benchmark_cfgs, verbose=args.verbose
    )
    sa_bench_dir = working_dir / "benchmarks"
    sa_bench_dir.mkdir(parents=True, exist_ok=True)
    test_cases = benchmark_factory.setup_benchmark_instance(sa_bench_dir)
    logger.info(f"Created {len(test_cases)} test cases.")

    with open(args.baseline_cache, "rb") as f:
        baseline_cache = pickle.load(f)
    logger.info(f"Loaded baseline cache with {len(baseline_cache)} entries.")

    objective_metric = CodeSizeObjective(Path(tools_cfg["llvm-size"]))

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
    start_iter = get_last_completed_iteration(sa_db_path) + 1
    if start_iter > 0:
        logger.info(f"Resuming from iteration {start_iter}")

    # -----------------------------------------------------------------------
    # SA state initialization
    # -----------------------------------------------------------------------
    temperature = args.temp_init
    # Cool temperature to match where we would be if resuming
    for _ in range(start_iter):
        temperature = max(args.temp_min, temperature * args.cooling_rate)

    # Prefer resuming the exact last-accepted config + score; otherwise
    # start from a fresh random config and let the SA loop discover a score.
    resumed_config, resumed_score = load_last_accepted_config(sa_db_path, targets)
    if resumed_config is not None:
        current_config = resumed_config
        current_score = resumed_score
        logger.info(
            f"Resumed current config: {len(current_config)} targets, "
            f"score={current_score}"
        )
    else:
        n_init = random.randint(1, min(args.max_targets, len(targets)))
        current_config = [sample_value_from_prior(t)
                          for t in random.sample(targets, n_init)]
        current_score = None

    # Restore (config, score) of the global best so best_config.json stays
    # consistent with best_score across resumes.
    best_config_resumed, best_score = load_best_config(sa_db_path, targets)
    best_config = best_config_resumed if best_config_resumed is not None \
        else list(current_config)
    if best_score is not None:
        logger.info(f"Resumed best config: score={best_score}")

    logger.info(f"SA start: temp={temperature:.4f}, "
                f"cooling={args.cooling_rate}, temp_min={args.temp_min}")

    # -----------------------------------------------------------------------
    # SA main loop
    # -----------------------------------------------------------------------
    for iteration in range(start_iter, args.iterations):
        logger.info(f"=== SA Iteration {iteration + 1}/{args.iterations} | "
                    f"T={temperature:.4f} ===")

        # Generate neighbor config
        if current_score is None:
            # No accepted config yet — resample fresh each iteration so a
            # broken initial config doesn't lock the loop in place.
            n_init = random.randint(1, min(args.max_targets, len(targets)))
            current_config = [sample_value_from_prior(t)
                              for t in random.sample(targets, n_init)]
            candidate_config = current_config
            move_type = "random_restart"
        else:
            candidate_config = perturb_config(
                current_config, targets, args.max_targets,
                use_best_value=args.use_best_value,
            )
            move_type = "perturb"

        save_sa_profile(sa_db_path, iteration, candidate_config)

        # Evaluate candidate — DB writes are deferred until after the SA
        # accept/reject decision below, so each row gets the real move_type.
        candidate_score, pending_results = evaluate_config(
            candidate_config, fn_cache, sys_prog_src, working_dir,
            test_cases, instrumented_bins, objective_metric, baseline_cache,
            args, iteration, temperature,
        )

        # ---- SA acceptance decision ----
        accepted = False
        if candidate_score is None:
            # Evaluation failed — stay with current config
            move_type = "eval_failed"
        elif current_score is None:
            # First successful eval — always accept
            accepted = True
            move_type = "initial"
        elif candidate_score <= current_score:
            # Better or equal — always accept (lower rel_obj = smaller binary)
            accepted = True
            move_type = "better"
        else:
            # Worse — accept with SA probability
            delta = candidate_score - current_score
            prob = math.exp(-delta / temperature) if temperature > 0 else 0.0
            if random.random() < prob:
                accepted = True
                move_type = "worse_accepted"
            else:
                move_type = "worse_rejected"

        # Persist all per-testcase rows with the real decision
        for r in pending_results:
            r.accepted = accepted
            r.move_type = move_type
            save_sa_result(sa_db_path, r)

        if accepted and candidate_score is not None:
            current_config = candidate_config
            current_score = candidate_score
            # Track global best
            if best_score is None or current_score < best_score:
                best_score = current_score
                best_config = list(current_config)
                logger.info(f"  *** New best: rel_obj={best_score:.4f} ***")

        logger.info(
            f"  move={move_type} accepted={accepted} "
            f"candidate={candidate_score} current={current_score} "
            f"best={best_score}"
        )

        # Cool the temperature
        temperature = max(args.temp_min, temperature * args.cooling_rate)

    logger.info("SA tuning completed.")

    # Persist best config to a JSON file for downstream deployment/analysis
    if best_config and best_score is not None:
        best_path = working_dir / "best_config.json"
        with open(best_path, "w") as f:
            json.dump({
                "best_score": best_score,
                "num_targets": len(best_config),
                "targets": [
                    {
                        "module": st.target.module,
                        "function": st.target.function,
                        "path": st.target.path_str,
                        "prior_type": st.target.prior_type,
                        "value": st.value,
                        "value_op": st.value_op,
                        "range_lo": st.range_lo,
                        "range_hi": st.range_hi,
                        "is_int": st.is_int,
                    }
                    for st in best_config
                ],
            }, f, indent=2)
        logger.info(f"Saved best config to {best_path}")

    # Summary
    conn = sqlite3.connect(sa_db_path)
    cursor = conn.execute("""
        SELECT COUNT(*), MIN(rel_objective), AVG(rel_objective),
               SUM(CASE WHEN rel_objective < 1.0 THEN 1 ELSE 0 END)
        FROM sa_results
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
        description="Simulated Annealing (SA) search over compiler heuristic targets."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--composite_db", required=True)
    parser.add_argument("--working_dir", required=True)
    parser.add_argument("--function_cache", required=True)
    parser.add_argument("--baseline_cache", required=True)
    parser.add_argument("--iterations", type=int, default=6400)
    parser.add_argument("--max_targets", type=int, default=10)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--probe_mem_limit", type=int, default=1024)
    # SA-specific
    parser.add_argument("--temp_init", type=float, default=1.0,
                        help="Initial temperature (default: 1.0)")
    parser.add_argument("--temp_min", type=float, default=0.01,
                        help="Minimum temperature (default: 0.01)")
    parser.add_argument("--cooling_rate", type=float, default=0.995,
                        help="Multiplicative cooling rate per iteration (default: 0.995)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for random (default: None, non-deterministic)")
    parser.add_argument("--use_best_value", action="store_true",
                        help="For each target, use composite's best (prior, "
                             "value) deterministically instead of sampling at "
                             "runtime. Aligns SA search space with BO.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(processName)-12s %(name)-30s %(levelname)-8s %(message)s",
    )
    run_sa(args)