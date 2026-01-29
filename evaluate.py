"""
Evaluator for Triton matmul kernel optimization.

Evaluates evolved Triton matmul kernels by measuring:
1. Correctness: element-wise closeness to torch.matmul reference
2. Speed: wall-clock speedup over torch.matmul

The evolved program must expose a `run_matmul(M, N, K)` function that returns:
    {
        "C_triton": torch.Tensor,   # (M, K) output from Triton kernel
        "A": torch.Tensor,          # (M, N) input matrix
        "B": torch.Tensor,          # (N, K) input matrix
        "triton_ms": float,         # median Triton kernel time in ms
        "torch_ms": float,          # median torch.matmul time in ms
        "M": int, "N": int, "K": int,
    }
"""

import os
import argparse
import torch
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

from shinka.core import run_shinka_eval

# Matrix sizes (M, N, K) to benchmark
MATRIX_SIZES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]

# Tolerance for correctness check
ATOL = 1e-2
RTOL = 1e-2


def validate_matmul_result(
    run_output: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    Validates a single matmul run by comparing the Triton kernel output
    to a torch.matmul reference.

    Args:
        run_output: Dict returned by run_matmul with keys
                    C_triton, A, B, triton_ms, torch_ms, M, N, K.

    Returns:
        (is_valid, error_or_success_message)
    """
    required_keys = {"C_triton", "A", "B", "triton_ms", "torch_ms", "M", "N", "K"}
    missing = required_keys - set(run_output.keys())
    if missing:
        return False, f"Missing keys in run output: {missing}"

    C_triton = run_output["C_triton"]
    A = run_output["A"]
    B = run_output["B"]
    M, N, K = run_output["M"], run_output["N"], run_output["K"]

    # Shape checks
    if A.shape != (M, N):
        return False, f"A shape mismatch: expected ({M}, {N}), got {tuple(A.shape)}"
    if B.shape != (N, K):
        return False, f"B shape mismatch: expected ({N}, {K}), got {tuple(B.shape)}"
    if C_triton.shape != (M, K):
        return False, (
            f"C_triton shape mismatch: expected ({M}, {K}), "
            f"got {tuple(C_triton.shape)}"
        )

    # NaN / Inf checks
    if torch.isnan(C_triton).any():
        return False, f"C_triton contains NaN values (size {M}x{N}x{K})"
    if torch.isinf(C_triton).any():
        return False, f"C_triton contains Inf values (size {M}x{N}x{K})"

    # Compute reference on the same device
    C_ref = torch.matmul(A, B)

    # Element-wise closeness
    if not torch.allclose(C_triton, C_ref, atol=ATOL, rtol=RTOL):
        max_abs_diff = (C_triton - C_ref).abs().max().item()
        max_rel_diff = (
            (C_triton - C_ref).abs() / (C_ref.abs() + 1e-8)
        ).max().item()
        return False, (
            f"Result mismatch for size {M}x{N}x{K}: "
            f"max_abs_diff={max_abs_diff:.6f}, max_rel_diff={max_rel_diff:.6f} "
            f"(atol={ATOL}, rtol={RTOL})"
        )

    return True, f"Correct for size {M}x{N}x{K}"


def get_matmul_kwargs(run_index: int) -> Dict[str, Any]:
    """Provides matrix dimensions for each benchmark run."""
    M, N, K = MATRIX_SIZES[run_index]
    return {"M": M, "N": N, "K": K}


def aggregate_matmul_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregates matmul benchmark results across all matrix sizes.

    Computes:
    - Per-size speedup (torch_ms / triton_ms; >1 means Triton is faster)
    - Per-size correctness (element-wise closeness to torch.matmul)
    - Combined score: geometric mean of speedups for correct sizes,
      scaled by fraction of correct sizes.
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    speedups = []
    size_details = {}

    for result in results:
        M, N, K = result["M"], result["N"], result["K"]
        triton_ms = result["triton_ms"]
        torch_ms = result["torch_ms"]
        size_label = f"{M}x{N}x{K}"

        C_triton = result["C_triton"]
        A = result["A"]
        B = result["B"]
        C_ref = torch.matmul(A, B)

        max_abs_diff = (C_triton - C_ref).abs().max().item()
        max_rel_diff = (
            (C_triton - C_ref).abs() / (C_ref.abs() + 1e-8)
        ).max().item()
        is_correct = torch.allclose(C_triton, C_ref, atol=ATOL, rtol=RTOL)

        speedup = torch_ms / triton_ms if triton_ms > 0 else 0.0

        if is_correct:
            speedups.append(speedup)

        size_details[size_label] = {
            "triton_ms": round(triton_ms, 4),
            "torch_ms": round(torch_ms, 4),
            "speedup": round(speedup, 4),
            "max_abs_diff": round(max_abs_diff, 6),
            "max_rel_diff": round(max_rel_diff, 6),
            "correct": is_correct,
        }

    num_correct = len(speedups)
    num_total = len(results)

    if num_correct == num_total and speedups:
        # All correct: geometric mean of speedups
        combined_score = float(np.exp(np.mean(np.log(np.array(speedups)))))
    elif speedups:
        # Partial: geometric mean scaled by fraction correct
        geo_mean = float(np.exp(np.mean(np.log(np.array(speedups)))))
        combined_score = geo_mean * (num_correct / num_total)
    else:
        combined_score = 0.0

    public_metrics = {
        "num_sizes_correct": f"{num_correct}/{num_total}",
        "mean_speedup": round(float(np.mean(speedups)), 4) if speedups else 0.0,
    }
    for k, v in size_details.items():
        public_metrics[f"speedup_{k}"] = v["speedup"]

    private_metrics = {
        "size_details": size_details,
        "all_speedups": [round(s, 4) for s in speedups],
    }

    # Build text feedback for the LLM in next evolution round
    text_parts = []
    for size_label, detail in size_details.items():
        status = "PASS" if detail["correct"] else "FAIL"
        text_parts.append(
            f"  {size_label}: {status} | speedup={detail['speedup']:.4f}x | "
            f"triton={detail['triton_ms']:.4f}ms torch={detail['torch_ms']:.4f}ms | "
            f"max_abs_err={detail['max_abs_diff']:.6f}"
        )
    text_feedback = "Matmul benchmark results:\n" + "\n".join(text_parts)

    return {
        "combined_score": round(combined_score, 4),
        "public": public_metrics,
        "private": private_metrics,
        "text_feedback": text_feedback,
    }


def main(program_path: str, results_dir: str):
    """Runs the Triton matmul kernel evaluation."""
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    num_experiment_runs = len(MATRIX_SIZES)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_matmul",
        num_runs=num_experiment_runs,
        get_experiment_kwargs=get_matmul_kwargs,
        validate_fn=validate_matmul_result,
        aggregate_metrics_fn=aggregate_matmul_metrics,
    )

    if correct:
        print("Evaluation completed successfully.")
    else:
        print(f"Evaluation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        elif isinstance(value, str) and len(value) > 200:
            print(f"  {key}: <too_long>")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Triton matmul kernel evaluator using shinka.eval"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'run_matmul')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json)",
    )
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)
