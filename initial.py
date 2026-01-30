# EVOLVE-BLOCK-START
"""Triton based matrix multiplication kernel"""

import torch
import triton
import triton.language as tl
import time
import numpy as np


@triton.jit
def matmul_kernel(
    A, B, C, 
    M, N, K,  
    stride_AM, stride_AK, 
    stride_BK, stride_BN,  
    stride_CM, stride_CN 
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    acc = 0.0

    for kk in range(K):
        a_index = pid_m * stride_AM + kk * stride_AK
        b_index = kk * stride_BK + pid_n * stride_BN

        a_element = tl.load(A + a_index)
        b_element = tl.load(B + b_index)
        acc += a_element * b_element

    c_index = pid_m * stride_CM + pid_n * stride_CN
    tl.store(C + c_index, acc)




def run_matmul(M, N, K):
    '''
    Multiply matrices using triton in most-efficient wat
    function returns:
    {
        "C_triton": torch.Tensor,   # (M, K) output from Triton kernel
        "A": torch.Tensor,          # (M, N) input matrix
        "B": torch.Tensor,          # (N, K) input matrix
        "triton_ms": float,         # median Triton kernel time in ms
        "torch_ms": float,          # median torch.matmul time in ms
        "M": int, "N": int, "K": int,
    }
    '''
    A = torch.randn((M, N), device='cuda')
    B = torch.randn((N, K), device='cuda')
    C_triton = torch.zeros((M, K), device='cuda')

    grid = (M, N)

    triton_times = []
    for _ in range(10):
        torch.cuda.synchronize()
        start_time = time.time()
        matmul_kernel[grid](
            A, B, C_triton,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C_triton.stride(0), C_triton.stride(1)
        )
        # EVOLVE-BLOCK-END

        # This part remains fixed (not evolved)
        torch.cuda.synchronize()
        end_time = time.time()
        triton_times.append((end_time - start_time) * 1000)

    torch_times = []
    for _ in range(10):
        torch.cuda.synchronize()
        start_time = time.time()
        C_torch = torch.matmul(A, B)
        torch.cuda.synchronize()
        end_time = time.time()
        torch_times.append((end_time - start_time) * 1000)

    triton_ms = np.median(triton_times)
    torch_ms = np.median(torch_times)

    return {
        "C_triton": C_triton,
        "A": A,
        "B": B,
        "triton_ms": triton_ms,
        "torch_ms": torch_ms,
        "M": M,
        "N": N,
        "K": K,
    }
