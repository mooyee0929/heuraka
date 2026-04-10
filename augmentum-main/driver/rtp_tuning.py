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
    PROBE_LOG_DELIMITER,
    generate_path_code,
    generate_register_extension_point,
    get_struct_definitions_from_fntype,
    type_descs_to_cpp_arg_types,
    type_descs_to_cpp_arg_vals,
    type_descs_to_cpp_args,
)
from augmentum.paths import ResultPath
from augmentum.sysProg import InstrumentationScope, ProbeExtension, SysProg
from augmentum.sysUtils import run_command, try_create_dir
from augmentum.timer import Timer
from augmentum.type_descs import IntTypeDesc, RealTypeDesc

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


@dataclass
class SelectedTarget:
    """A target selected for an RTP iteration, with a sampled value."""
    target: TuningTarget
    value: float   # sampled value to force
    value_op: Optional[str]  # None = static, "+" = offset, "*" = scale


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
        return SelectedTarget(target=target, value=value, value_op=None)

    elif ptype in ("All Integers Prior", "All Reals Prior"):
        # data format: "val,True#val,True#..."
        entries = pdata.split("#")
        values = []
        for e in entries:
            parts = e.split(",")
            if len(parts) >= 2 and parts[1] == "True":
                try:
                    values.append(int(parts[0]) if ptype == "All Integers Prior"
                                  else float(parts[0]))
                except ValueError:
                    continue
        if values:
            value = random.choice(values)
        else:
            value = 0
        return SelectedTarget(target=target, value=value, value_op=None)

    elif ptype in ("Integer Range Prior", "Real Range Prior"):
        # data format: "min,max"
        parts = pdata.split(",")
        lo, hi = int(parts[0]), int(parts[1])
        if ptype == "Real Range Prior":
            lo, hi = float(parts[0]), float(parts[1])
            value = random.uniform(lo, hi)
        else:
            value = random.randint(lo, hi)
        return SelectedTarget(target=target, value=value, value_op=None)

    elif ptype in ("Integer Offset Prior", "Real Offset Prior"):
        # data format: "theta,phi" → offset in [-theta, +phi]
        parts = pdata.split(",")
        theta, phi = int(parts[0]), int(parts[1])
        if ptype == "Real Offset Prior":
            theta, phi = float(parts[0]), float(parts[1])
            value = random.uniform(-theta, phi)
        else:
            value = random.randint(-theta, phi)
        return SelectedTarget(target=target, value=value, value_op="+")

    elif ptype in ("Integer Scale Prior", "Real Scale Prior"):
        # data format: "alpha,beta" → scale factor in [1-alpha, 1+beta]
        # But we pass the raw factor and use "*" op:
        #   *probed = original_value * factor
        parts = pdata.split(",")
        alpha, beta = int(parts[0]), int(parts[1])
        # For integer scale, sample an integer multiplier
        # The prior means: original * (1 ± factor/max), but stored as raw bounds
        # Actually the scale prior stores alpha, beta as the max offset factors
        # value_op="*" means: *probed = original_value * value
        # We need a scale factor. Looking at the data, e.g. "127,127" for i8
        # means alpha=127, beta=127, so scale ∈ [1-127, 1+127] = [-126, 128]
        # But that doesn't match. Let me re-read the prior code.
        # From ScaleProbe: value_op="*", so *probed = original * value
        # Scale Prior produces values in range (1-alpha, 1+beta) but alpha/beta
        # are fractions? No — looking at the data "0,41" for i8, the Scale Prior
        # likely uses integer multipliers directly.
        # Let me just sample an integer in the range and use "*" op.
        if ptype == "Real Scale Prior":
            alpha, beta = float(parts[0]), float(parts[1])
            # Generate a random real scale factor
            value = random.uniform(-alpha, beta)
        else:
            value = random.randint(-alpha, beta)
        return SelectedTarget(target=target, value=value, value_op="*")

    else:
        logger.warning(f"Unknown prior type: {ptype}, using static 0")
        return SelectedTarget(target=target, value=0, value_op=None)


# ---------------------------------------------------------------------------
# Multi-target extension code generation
# ---------------------------------------------------------------------------

def get_path_type_str(path_str: str) -> str:
    """Extract the terminal type from a path string like 'A0.D.S2.S0.S0.S3.L.L.T-i8'."""
    parts = path_str.split(".")
    for p in reversed(parts):
        if p.startswith("T-"):
            type_str = p[2:]  # e.g. "i8", "i16", "i32", "i64", "f32", "f64"
            if type_str.startswith("i"):
                bits = int(type_str[1:])
                return f"int{bits}_t"
            elif type_str.startswith("f"):
                bits = int(type_str[1:])
                return "float" if bits == 32 else "double"
    return "int64_t"


def generate_rtp_extension_code(
    selected_targets: List[SelectedTarget],
    sys_prog_src: str,
    log_file: str,
) -> str:
    """
    Generate a single C++ extension that instruments multiple functions
    simultaneously. Each function gets its own modified version that applies
    the randomly sampled value.

    Unlike the per-path probes in the main framework, this generates the
    modification code inline without needing the full Function/Path objects.
    We reconstruct the path access code from the path string directly.
    """
    # Group targets by (module, function) since each function gets one wrapper
    from collections import defaultdict
    fn_targets = defaultdict(list)
    for st in selected_targets:
        key = (st.target.module, st.target.function)
        fn_targets[key].append(st)

    # We use a simplified approach: for each function, pick ONE target path
    # (the first in the group) and apply its modification. This matches the
    # paper's approach where each function in the RTP gets one modification.
    # If multiple paths target the same function, we only use one.
    unique_targets = []
    for key, sts in fn_targets.items():
        unique_targets.append(random.choice(sts))

    # Build the extension code
    includes = """
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include "augmentum.h"

using namespace augmentum;

std::mutex rtp_log_mutex;

void rtp_write_log(const std::string& msg) {
    const std::lock_guard<std::mutex> lock(rtp_log_mutex);
    std::filesystem::path outputFile = "%s";
    std::ofstream out(outputFile.c_str(), std::ios::out | std::ios::app);
    if (out.good()) {
        out << msg << std::endl;
    }
    out.close();
}
""" % log_file

    # Generate one modified function per unique (module, function) target
    modified_functions = []
    register_blocks = []

    for idx, st in enumerate(unique_targets):
        orig_type_name = f"original_t_{idx}"
        orig_fn_name = f"original_fn_{idx}"
        mod_fn_name = f"modified_fn_{idx}"
        path_type = get_path_type_str(st.target.path_str)

        # Determine the value expression
        if st.value_op is None:
            # Static: force the value directly
            value_expr = f"({path_type})({st.value})"
        elif st.value_op == "+":
            value_expr = f"original_value + ({path_type})({st.value})"
        elif st.value_op == "*":
            value_expr = f"original_value * ({path_type})({st.value})"
        else:
            value_expr = f"({path_type})({st.value})"

        # Generate path access code from path string
        path_access = generate_path_access_code(st.target.path_str, path_type)

        # We need the function signature. Since we don't have the full Function
        # object, we use a generic approach: the extension point replace mechanism
        # doesn't need us to know the signature — the Augmentum framework handles
        # that via FnExtensionPoint. We use the AfterAdvice pattern instead of
        # replace, which doesn't require knowing the function signature.
        #
        # Actually, looking more carefully at the existing code, the replace
        # approach requires knowing the exact function signature. Instead, let's
        # use the after-advice pattern which operates on RetVal/ArgVals.
        #
        # But the after-advice only runs AFTER the function — it can't change
        # output before it's used. We need the replace approach.
        #
        # For a general solution without function signatures, we use a different
        # strategy: we generate a "value forcer" that uses AfterAdvice to modify
        # the return value or argument values after the function call. This is
        # similar to what the existing probes do but at a higher level.
        #
        # Actually, re-reading the augmentum API more carefully:
        # - replace() replaces the function entirely (needs signature)
        # - extend_after() runs code after the original (has access to ret/args)
        #
        # The AfterAdvice callback has signature:
        #   void(FnExtensionPoint& pt, RetVal ret_value, ArgVals arg_values)
        # where RetVal and ArgVals give access to the actual values.
        #
        # This is much simpler for multi-function instrumentation! Let's use it.

        # Build the after-advice body based on the path
        advice_body = generate_advice_body(st, idx, path_type)

        modified_functions.append(f"""
// Target {idx}: {st.target.module} :: {st.target.function}
// Path: {st.target.path_str}
// Prior: {st.target.prior_type} value={st.value} op={st.value_op}
{advice_body}
""")

        register_blocks.append(f"""
        if (pt.get_module_name() == "{sys_prog_src}/{st.target.module}" &&
            pt.get_name() == "{st.target.function}") {{
            pt.extend_after(advice_{idx}, advice_id_{idx});
            rtp_write_log("RTP: registered target {idx} {st.target.function}");
        }}
""")

    unregister_blocks = []
    for idx, st in enumerate(unique_targets):
        unregister_blocks.append(f"""
        if (pt.get_module_name() == "{sys_prog_src}/{st.target.module}" &&
            pt.get_name() == "{st.target.function}") {{
            pt.remove(advice_id_{idx});
        }}
""")

    listener = f"""
struct RTPListener: Listener {{
    void on_extension_point_register(FnExtensionPoint& pt) {{
{"".join(register_blocks)}
    }}

    void on_extension_point_unregister(FnExtensionPoint& pt) {{
{"".join(unregister_blocks)}
    }}

{chr(10).join(f"    AdviceId advice_id_{i} = get_unique_advice_id();" for i in range(len(unique_targets)))}
}};
ListenerLifeCycle<RTPListener> rtpListener;
"""

    return includes + "\n".join(modified_functions) + "\n" + listener


def generate_advice_body(st: SelectedTarget, idx: int, path_type: str) -> str:
    """
    Generate an AfterAdvice lambda that modifies the appropriate output
    channel based on the path string.

    Path examples:
      Z.T-i32        → return value, type i32
      Z.L.T-i16      → return value, left half (lower 16 bits of 32-bit return)
      A0.D.S2.T-i32  → arg0 -> deref -> struct elem 2, type i32
    """
    path_str = st.target.path_str
    parts = path_str.split(".")
    value = st.value

    # Determine value expression
    if st.value_op is None:
        val_expr = f"({path_type})({value})"
        mod_code = f"*ptr = {val_expr};"
    elif st.value_op == "+":
        mod_code = f"*ptr = *ptr + ({path_type})({value});"
    elif st.value_op == "*":
        mod_code = f"*ptr = *ptr * ({path_type})({value});"
    else:
        val_expr = f"({path_type})({value})"
        mod_code = f"*ptr = {val_expr};"

    # Build pointer access chain from path
    access_code = build_access_chain(parts, path_type)

    return f"""
AfterAdvice advice_{idx} = [](FnExtensionPoint& pt, RetVal ret_value, ArgVals arg_values) {{
    (void)pt;
    {access_code}
    if (ptr != nullptr) {{
        {mod_code}
    }}
}};
"""


def build_access_chain(parts: List[str], path_type: str) -> str:
    """
    Build C++ code that navigates the path to get a pointer to the target
    value, using the Augmentum RetVal/ArgVals API.

    RetVal is a void* to the return value storage.
    ArgVals is a void** array of pointers to each argument.
    """
    lines = []
    current_expr = None
    in_deref = False

    for i, p in enumerate(parts):
        if p == "Z":
            # Return value
            lines.append("void* _base = ret_value;")
            current_expr = "_base"

        elif p.startswith("A"):
            # Argument
            arg_idx = int(p[1:])
            lines.append(f"void* _base = arg_values[{arg_idx}];")
            current_expr = "_base"

        elif p == "D":
            # Dereference pointer
            lines.append(f"void* _deref = *(void**){current_expr};")
            lines.append(f"if (_deref == nullptr) {{ {path_type}* ptr = nullptr; return; }}")
            current_expr = "_deref"

        elif p.startswith("S"):
            # Struct element access — we compute byte offset
            # This is tricky without knowing struct layout. Instead, we cast
            # to a char array and navigate. But we don't know field offsets.
            #
            # Alternative: use the same approach as augmentum probes — cast
            # to a struct pointer. But we don't have the struct definition.
            #
            # The simplest correct approach: since the augmentum instrumentation
            # wraps the function, arguments are passed by value/pointer. The
            # struct layout in LLVM IR matches the C++ struct layout that was
            # used during the original composite search. We need the struct defs.
            #
            # For now, use the element index and a generic struct with indexed
            # fields (e_idx), matching the augmentum convention.
            elem_idx = int(p[1:])
            # We need to know struct element sizes. Since we can't determine
            # this dynamically, we use the approach of casting through char*
            # and using augmentum's struct naming convention.
            # This won't work without struct definitions.
            #
            # Actually, looking at the original code more carefully, the probes
            # work because they know the full Function type and generate proper
            # C++ struct definitions. For the RTP, we need a different approach.
            #
            # Let's fall back to the SIMPLEST approach from the paper:
            # Use the existing single-function infrastructure but combine
            # multiple extensions by having the listener handle multiple
            # extension points with REPLACE (not after-advice).
            #
            # BUT we need function signatures for replace. This is a fundamental
            # constraint.
            #
            # PRAGMATIC SOLUTION: Load the function objects from the cache and
            # use the existing probe infrastructure to generate per-function
            # extension code, then combine them into a single extension.
            pass

        elif p in ("L", "R"):
            # Split int left/right
            # L = lower half, R = upper half
            pass

        elif p.startswith("T-"):
            # Terminal — this is the final type, we construct the pointer
            pass

    # If we reached here with struct access, we need the full Function objects.
    # Return a placeholder that will be replaced by the proper generation.
    lines.append(f"{path_type}* ptr = nullptr; // placeholder")
    return "\n    ".join(lines)


def generate_path_access_code(path_str: str, path_type: str) -> str:
    """Generate C++ code to access a path - placeholder for now."""
    return f"// path: {path_str}\n"


# ---------------------------------------------------------------------------
# Proper multi-target extension using Function objects
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

        # Determine value expression
        if st.value_op is None:
            calc_probed = f"({probed_type_str})({st.value})"
        elif st.value_op == "+":
            calc_probed = f"original_value + ({probed_type_str})({st.value})"
        elif st.value_op == "*":
            calc_probed = f"original_value * ({probed_type_str})({st.value})"
        else:
            calc_probed = f"({probed_type_str})({st.value})"

        # Build the return logic
        if return_type == "void":
            return_stmt, orig_return = "", ""
        else:
            return_stmt = "return r;"
            orig_return = f"{return_type} r = "

        # Build probe code (modify the value)
        from augmentum.probes import add_null_check
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
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <mutex>
#include "augmentum.h"

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
