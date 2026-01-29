"""
Evaluator template for Triton GPU kernel benchmarking and validation.
Tests correctness, speed, memory efficiency, numerical stability, and scalability.

The program under test must expose a function with signature:

    def run_triton_kernel(
        input_size: Tuple[int, ...],
        dtype_str: str,
        label: str,
        num_warmup: int,
        num_benchmark: int,
        seed: int,
    ) -> Dict[str, Any]

Returning a dict with the following keys:
    kernel_output       : np.ndarray        – kernel result (CPU, numpy)
    reference_output    : np.ndarray        – reference result (CPU, numpy)
    kernel_time_ms      : float             – median kernel execution time in ms
    reference_time_ms   : float             – median reference execution time in ms
    kernel_times_ms     : List[float]       – all individual kernel timings in ms
    reference_times_ms  : List[float]       – all individual reference timings in ms
    peak_memory_bytes   : int               – peak GPU memory during kernel execution
    num_elements        : int               – total number of elements processed
    bytes_processed     : int               – total bytes read + written (for bandwidth)
    flops               : int               – total floating-point ops (for TFLOPS; 0 if unknown)
    dtype               : str               – dtype string (echoed back)
    label               : str               – config label (echoed back)
"""

import os
import argparse
import math
import json
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

from shinka.core import run_shinka_eval


# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------
# Each config defines one evaluation run. Adjust sizes, dtypes, and labels
# to match the kernel under test.

EVAL_CONFIGS: List[Dict[str, Any]] = [
    # input_size is (M, N, K) for matmul: C[M,N] = A[M,K] @ B[K,N]
    {"input_size": (256, 256, 256), "dtype": "float32", "label": "small_f32"},
    {"input_size": (256, 256, 256), "dtype": "float16", "label": "small_f16"},
    {"input_size": (1024, 1024, 1024), "dtype": "float32", "label": "medium_f32"},
    {"input_size": (1024, 1024, 1024), "dtype": "float16", "label": "medium_f16"},
    {"input_size": (2048, 2048, 2048), "dtype": "float32", "label": "large_f32"},
    {"input_size": (2048, 2048, 2048), "dtype": "float16", "label": "large_f16"},
    {"input_size": (4096, 4096, 4096), "dtype": "float32", "label": "xlarge_f32"},
    {"input_size": (4096, 4096, 4096), "dtype": "float16", "label": "xlarge_f16"},
    # Non-square shapes to test robustness
    {"input_size": (1024, 2048, 512), "dtype": "float32", "label": "nonsquare_f32"},
    {"input_size": (2048, 512, 4096), "dtype": "float16", "label": "nonsquare_f16"},
]

NUM_WARMUP_ITERS = 10
NUM_BENCHMARK_ITERS = 100

# Per-dtype tolerances for correctness.
# Matmul accumulates FP error proportional to K, so tolerances are
# more generous than element-wise operations.
ATOL = {"float32": 1e-3, "float16": 1e-1, "bfloat16": 1e-1}
RTOL = {"float32": 1e-3, "float16": 1e-2, "bfloat16": 1e-2}

# Minimum fraction of elements that must be within tolerance to pass
ELEMENT_WISE_PASS_THRESHOLD = 0.999

# Maximum acceptable coefficient of variation in kernel timings.
# Values above this indicate unstable / noisy benchmarks.
MAX_TIMING_CV = 0.25


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_kernel_output(
    run_output: Dict[str, Any],
    atol_override: Optional[float] = None,
    rtol_override: Optional[float] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validates a single Triton kernel run against its reference output.

    Checks performed:
        1. All required keys are present.
        2. Output shapes match.
        3. No NaN or Inf values in kernel output.
        4. Element-wise numerical accuracy within dtype tolerance.
        5. Timing values are positive.
        6. Peak memory is non-negative.
        7. Timing stability (coefficient of variation).

    Returns:
        (is_valid, message) – message describes the outcome.
    """
    required_keys = [
        "kernel_output",
        "reference_output",
        "dtype",
        "kernel_time_ms",
        "reference_time_ms",
        "peak_memory_bytes",
        "num_elements",
        "bytes_processed",
    ]
    for key in required_keys:
        if key not in run_output:
            return False, f"Missing required key in run output: '{key}'"

    kernel_out = np.asarray(run_output["kernel_output"])
    ref_out = np.asarray(run_output["reference_output"])
    dtype_str = run_output["dtype"]
    label = run_output.get("label", "unknown")

    # --- Shape check ---
    if kernel_out.shape != ref_out.shape:
        return False, (
            f"[{label}] Shape mismatch: kernel {kernel_out.shape} "
            f"vs reference {ref_out.shape}"
        )

    # --- NaN / Inf check ---
    nan_count = int(np.sum(np.isnan(kernel_out)))
    if nan_count > 0:
        return False, f"[{label}] Kernel output contains {nan_count} NaN value(s)"
    inf_count = int(np.sum(np.isinf(kernel_out)))
    if inf_count > 0:
        return False, f"[{label}] Kernel output contains {inf_count} Inf value(s)"

    # --- Element-wise correctness ---
    atol = atol_override if atol_override is not None else ATOL.get(dtype_str, 1e-5)
    rtol = rtol_override if rtol_override is not None else RTOL.get(dtype_str, 1e-5)

    close_mask = np.isclose(kernel_out, ref_out, atol=atol, rtol=rtol)
    fraction_close = float(np.mean(close_mask))

    if fraction_close < ELEMENT_WISE_PASS_THRESHOLD:
        abs_err = np.abs(kernel_out - ref_out)
        max_abs_err = float(np.max(abs_err))
        worst_idx = np.unravel_index(np.argmax(abs_err), kernel_out.shape)
        return False, (
            f"[{label}] Correctness FAIL: {fraction_close * 100:.2f}% within tol "
            f"(need {ELEMENT_WISE_PASS_THRESHOLD * 100:.1f}%). "
            f"Max abs error: {max_abs_err:.6e} at index {worst_idx}. "
            f"Kernel={kernel_out[worst_idx]:.6e}, Ref={ref_out[worst_idx]:.6e}"
        )

    # --- Timing sanity ---
    if run_output["kernel_time_ms"] <= 0:
        return False, f"[{label}] Kernel time must be positive"
    if run_output["reference_time_ms"] <= 0:
        return False, f"[{label}] Reference time must be positive"

    # --- Memory sanity ---
    if run_output["peak_memory_bytes"] < 0:
        return False, f"[{label}] Peak memory bytes cannot be negative"

    # --- Timing stability (optional, warn-only) ---
    stability_warning = ""
    kernel_times = run_output.get("kernel_times_ms")
    if kernel_times and len(kernel_times) > 1:
        arr = np.array(kernel_times)
        cv = float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0.0
        if cv > MAX_TIMING_CV:
            stability_warning = (
                f" WARNING: timing CV={cv:.2f} exceeds {MAX_TIMING_CV} "
                f"(std={np.std(arr):.4f}ms, mean={np.mean(arr):.4f}ms)"
            )

    # --- Summary message ---
    abs_err = np.abs(kernel_out - ref_out)
    max_abs_err = float(np.max(abs_err))
    mean_abs_err = float(np.mean(abs_err))
    speedup = run_output["reference_time_ms"] / run_output["kernel_time_ms"]
    msg = (
        f"[{label}] PASS – {fraction_close * 100:.2f}% within tol, "
        f"max_err={max_abs_err:.2e}, mean_err={mean_abs_err:.2e}, "
        f"speedup={speedup:.2f}x{stability_warning}"
    )
    return True, msg


# ---------------------------------------------------------------------------
# Experiment kwargs provider
# ---------------------------------------------------------------------------

def get_triton_eval_kwargs(run_index: int) -> Dict[str, Any]:
    """Provides keyword arguments for each evaluation run based on EVAL_CONFIGS."""
    if run_index >= len(EVAL_CONFIGS):
        raise IndexError(
            f"Run index {run_index} exceeds config count ({len(EVAL_CONFIGS)})"
        )
    cfg = EVAL_CONFIGS[run_index]
    return {
        "input_size": cfg["input_size"],
        "dtype_str": cfg["dtype"],
        "label": cfg["label"],
        "num_warmup": NUM_WARMUP_ITERS,
        "num_benchmark": NUM_BENCHMARK_ITERS,
        "seed": 42 + run_index,
    }


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def _percentile_from_sorted(sorted_arr: np.ndarray, p: float) -> float:
    """Returns the p-th percentile from a sorted array."""
    idx = (len(sorted_arr) - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(sorted_arr) - 1)
    frac = idx - lo
    return float(sorted_arr[lo] * (1 - frac) + sorted_arr[hi] * frac)


def aggregate_triton_metrics(
    results: List[Dict[str, Any]],
    results_dir: str,
) -> Dict[str, Any]:
    """
    Aggregates metrics across all evaluation configurations.

    Produces:
    - Per-config: speedup, bandwidth (GB/s), TFLOPS, error stats, timing percentiles
    - Overall: geometric-mean speedup, correctness ratio, combined score
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    per_config: List[Dict[str, Any]] = []
    speedups: List[float] = []
    all_correct = True

    for res in results:
        label = res.get("label", "unknown")
        dtype_str = res.get("dtype", "float32")
        kernel_time = res["kernel_time_ms"]
        ref_time = res["reference_time_ms"]
        num_elements = res["num_elements"]
        bytes_processed = res["bytes_processed"]
        peak_mem = res["peak_memory_bytes"]
        flops = res.get("flops", 0)

        kernel_out = np.asarray(res["kernel_output"])
        ref_out = np.asarray(res["reference_output"])

        # ---- Correctness metrics ----
        abs_err = np.abs(kernel_out - ref_out)
        max_abs_err = float(np.max(abs_err))
        mean_abs_err = float(np.mean(abs_err))

        safe_ref = np.where(np.abs(ref_out) > 1e-12, ref_out, 1e-12)
        rel_err = np.abs((kernel_out - ref_out) / safe_ref)
        max_rel_err = float(np.max(rel_err))
        mean_rel_err = float(np.mean(rel_err))

        atol = ATOL.get(dtype_str, 1e-5)
        rtol = RTOL.get(dtype_str, 1e-5)
        frac_close = float(
            np.mean(np.isclose(kernel_out, ref_out, atol=atol, rtol=rtol))
        )
        correct = frac_close >= ELEMENT_WISE_PASS_THRESHOLD
        if not correct:
            all_correct = False

        # ---- Performance metrics ----
        speedup = ref_time / kernel_time if kernel_time > 0 else 0.0
        speedups.append(speedup)

        bandwidth_gbs = (
            (bytes_processed / 1e9) / (kernel_time / 1e3)
            if kernel_time > 0
            else 0.0
        )

        tflops = (
            (flops / 1e12) / (kernel_time / 1e3)
            if kernel_time > 0 and flops > 0
            else None
        )

        peak_mem_mb = peak_mem / (1024 * 1024)

        # ---- Timing distribution (if individual timings provided) ----
        timing_stats: Dict[str, Any] = {}
        kernel_times = res.get("kernel_times_ms")
        if kernel_times and len(kernel_times) > 1:
            arr = np.sort(np.array(kernel_times))
            timing_stats = {
                "p50_ms": round(_percentile_from_sorted(arr, 50), 4),
                "p90_ms": round(_percentile_from_sorted(arr, 90), 4),
                "p99_ms": round(_percentile_from_sorted(arr, 99), 4),
                "min_ms": round(float(arr[0]), 4),
                "max_ms": round(float(arr[-1]), 4),
                "std_ms": round(float(np.std(arr)), 4),
                "cv": round(float(np.std(arr) / np.mean(arr)), 4)
                if np.mean(arr) > 0
                else 0.0,
            }

        config_metrics = {
            "label": label,
            "dtype": dtype_str,
            "num_elements": num_elements,
            "correct": correct,
            "fraction_within_tol": round(frac_close, 6),
            "max_abs_error": max_abs_err,
            "mean_abs_error": mean_abs_err,
            "max_rel_error": max_rel_err,
            "mean_rel_error": mean_rel_err,
            "kernel_time_ms": round(kernel_time, 4),
            "reference_time_ms": round(ref_time, 4),
            "speedup": round(speedup, 4),
            "bandwidth_gbs": round(bandwidth_gbs, 2),
            "tflops": round(tflops, 4) if tflops is not None else None,
            "peak_memory_mb": round(peak_mem_mb, 2),
        }
        if timing_stats:
            config_metrics["timing_distribution"] = timing_stats

        per_config.append(config_metrics)

    # ---- Aggregate scores ----
    if speedups and all(s > 0 for s in speedups):
        geomean_speedup = float(
            math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        )
    else:
        geomean_speedup = 0.0

    correctness_ratio = sum(1 for c in per_config if c["correct"]) / len(per_config)

    # Combined score: geomean speedup if all configs pass correctness, else 0
    combined_score = geomean_speedup if all_correct else 0.0

    # Scalability: ratio of speedup on the largest config to the smallest
    if len(speedups) >= 2:
        scalability_ratio = round(speedups[-1] / speedups[0], 4) if speedups[0] > 0 else 0.0
    else:
        scalability_ratio = 1.0

    sorted_by_speedup = sorted(per_config, key=lambda c: c["speedup"], reverse=True)
    best = sorted_by_speedup[0]
    worst = sorted_by_speedup[-1]

    public_metrics = {
        "num_configs_tested": len(per_config),
        "all_correct": all_correct,
        "correctness_ratio": round(correctness_ratio, 4),
        "geomean_speedup": round(geomean_speedup, 4),
        "best_speedup": best["speedup"],
        "best_speedup_config": best["label"],
        "worst_speedup": worst["speedup"],
        "worst_speedup_config": worst["label"],
        "scalability_ratio": scalability_ratio,
    }

    private_metrics = {
        "per_config": per_config,
    }

    metrics = {
        "combined_score": round(combined_score, 4),
        "public": public_metrics,
        "private": private_metrics,
    }

    # ---- Persist detailed results ----
    extra_file = os.path.join(results_dir, "extra.json")
    try:
        with open(extra_file, "w") as f:
            json.dump(
                {
                    "per_config": per_config,
                    "geomean_speedup": geomean_speedup,
                    "correctness_ratio": correctness_ratio,
                    "scalability_ratio": scalability_ratio,
                    "combined_score": combined_score,
                },
                f,
                indent=2,
            )
        print(f"Detailed benchmark data saved to {extra_file}")
    except Exception as e:
        print(f"Error saving extra.json: {e}")
        metrics["extra_json_save_error"] = str(e)

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(program_path: str, results_dir: str):
    """Runs the Triton kernel evaluation."""
    print(f"Evaluating Triton kernel: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    num_runs = len(EVAL_CONFIGS)

    def _aggregator(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return aggregate_triton_metrics(results, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_triton_kernel",
        num_runs=num_runs,
        get_experiment_kwargs=get_triton_eval_kwargs,
        validate_fn=validate_kernel_output,
        aggregate_metrics_fn=_aggregator,
    )

    if correct:
        print("All correctness checks passed.")
    else:
        print(f"Evaluation failed: {error_msg}")

    # ---- Print summary table ----
    print("\n" + "=" * 80)
    print("TRITON KERNEL EVALUATION SUMMARY")
    print("=" * 80)

    public = metrics.get("public", {})
    print(f"  Combined Score (geomean speedup): {metrics.get('combined_score', 0.0)}")
    print(f"  Configs Tested:      {public.get('num_configs_tested', 0)}")
    print(f"  All Correct:         {public.get('all_correct', False)}")
    print(f"  Correctness Ratio:   {public.get('correctness_ratio', 0.0)}")
    print(f"  Geomean Speedup:     {public.get('geomean_speedup', 0.0)}x")
    print(
        f"  Best Speedup:        {public.get('best_speedup', 0.0)}x "
        f"({public.get('best_speedup_config', 'N/A')})"
    )
    print(
        f"  Worst Speedup:       {public.get('worst_speedup', 0.0)}x "
        f"({public.get('worst_speedup_config', 'N/A')})"
    )
    print(f"  Scalability Ratio:   {public.get('scalability_ratio', 1.0)}")

    per_config = metrics.get("private", {}).get("per_config", [])
    if per_config:
        print(
            f"\n  {'Config':<18} {'OK':<6} {'Speedup':>8} "
            f"{'BW GB/s':>9} {'TFLOPS':>8} {'MaxErr':>10} {'Mem MB':>8} "
            f"{'p50 ms':>8} {'p99 ms':>8}"
        )
        print("  " + "-" * 96)
        for cfg in per_config:
            tflops_str = f"{cfg['tflops']:>8.2f}" if cfg.get("tflops") else f"{'N/A':>8}"
            td = cfg.get("timing_distribution", {})
            p50 = f"{td['p50_ms']:>8.4f}" if td.get("p50_ms") is not None else f"{'N/A':>8}"
            p99 = f"{td['p99_ms']:>8.4f}" if td.get("p99_ms") is not None else f"{'N/A':>8}"
            print(
                f"  {cfg['label']:<18} "
                f"{'PASS' if cfg['correct'] else 'FAIL':<6} "
                f"{cfg['speedup']:>8.2f} "
                f"{cfg['bandwidth_gbs']:>9.2f} "
                f"{tflops_str} "
                f"{cfg['max_abs_error']:>10.2e} "
                f"{cfg['peak_memory_mb']:>8.1f} "
                f"{p50} "
                f"{p99}"
            )

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Triton kernel evaluator using shinka.eval"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'run_triton_kernel')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results (metrics.json, correct.json, extra.json)",
    )
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)

