"""
Fused quantization and bit-packing kernel for KIVI KV cache compression.

Replaces the multi-step pipeline in new_pack.py:
    Triton minmax kernel → PyTorch sub/div/clamp/round/cast → Triton pack kernel
with a single Triton kernel, achieving:
    - 7+ kernel launches → 1 kernel launch
    - Elimination of all intermediate tensor allocations (scale broadcast, int32 cast, etc.)
    - Data read from global memory only twice (Phase 1: minmax, Phase 2: quantize+pack,
      where Phase 2 benefits from L2 cache locality)

Drop-in replacement for `triton_quantize_and_pack_along_last_dim` in new_pack.py.
"""

import triton
import triton.language as tl
import torch
import numpy as np


@triton.jit
def _fused_quant_and_pack_along_last_dim(
    bits: tl.constexpr,
    data_ptr,
    code_ptr,
    scale_ptr,
    mn_ptr,
    N,                              # total number of rows = B * nh * D
    num_groups,                     # T // group_size
    group_size: tl.constexpr,       # quantization group size (e.g. 32, 64, 128)
    feat_per_int: tl.constexpr,     # 32 // bits (e.g. 16 for 2-bit, 8 for 4-bit)
    T_packed,                       # T // feat_per_int
    BLOCK_SIZE_N: tl.constexpr,     # number of rows per program
):
    """
    Fused min-max asymmetric quantization + bit-packing kernel.

    Grid: (cdiv(N, BLOCK_SIZE_N), num_groups)
    Each program handles BLOCK_SIZE_N rows for one group.

    Phase 1 — Min/Max:
        Load data in chunks of feat_per_int elements, compute running min/max per row.

    Phase 2 — Quantize + Pack:
        Reload same chunks (L2-cached), quantize to [0, 2^bits - 1], bit-pack
        feat_per_int quantized values into one int32 using shift-and-sum.
        Since each quantized value occupies non-overlapping bit positions,
        integer addition is equivalent to bitwise OR.
    """
    max_int: tl.constexpr = (1 << bits) - 1
    packed_per_group: tl.constexpr = group_size // feat_per_int

    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    # Row indices for this program
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
    mask_n = offs_n < N

    # Base offset into data for this group: data[n, g * group_size]
    T = num_groups * group_size
    base = offs_n * T + pid_g * group_size

    # ======================== Phase 1: Compute min and max ========================
    running_min = tl.full((BLOCK_SIZE_N,), float('inf'), dtype=tl.float32)
    running_max = tl.full((BLOCK_SIZE_N,), float('-inf'), dtype=tl.float32)

    for p in tl.static_range(packed_per_group):
        offs_feat = tl.arange(0, feat_per_int).to(tl.int64)
        data_offs = base[:, None] + (p * feat_per_int + offs_feat[None, :])
        chunk = tl.load(data_ptr + data_offs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        chunk_min = tl.min(chunk, axis=1)
        chunk_max = tl.max(chunk, axis=1)
        running_min = tl.minimum(running_min, chunk_min)
        running_max = tl.maximum(running_max, chunk_max)

    scale_val = (running_max - running_min) / max_int
    inv_scale = 1.0 / (scale_val + 1e-10)

    # Store scale and mn (as fp16, matching original implementation)
    sm_offs = offs_n * num_groups + pid_g
    tl.store(scale_ptr + sm_offs, scale_val.to(tl.float16), mask=mask_n)
    tl.store(mn_ptr + sm_offs, running_min.to(tl.float16), mask=mask_n)

    # ======================== Phase 2: Quantize + Bit-Pack ========================
    for p in tl.static_range(packed_per_group):
        offs_feat = tl.arange(0, feat_per_int).to(tl.int64)
        data_offs = base[:, None] + (p * feat_per_int + offs_feat[None, :])
        # Reload from global memory (should hit L2 cache from Phase 1)
        chunk = tl.load(data_ptr + data_offs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # Quantize: q = round(clamp((x - mn) / scale, 0, max_int))
        q = (chunk - running_min[:, None]) * inv_scale[:, None]
        q = tl.math.rint(q)
        q = tl.maximum(tl.minimum(q, float(max_int)), 0.0)
        q_int = q.to(tl.int32)

        # Bit-pack: shift each quantized value to its bit position and sum.
        # Since bit positions are non-overlapping, integer sum == bitwise OR.
        # Example for 2-bit, feat_per_int=16:
        #   q_int[0] << 0  occupies bits [0, 1]
        #   q_int[1] << 2  occupies bits [2, 3]
        #   ...
        #   q_int[15] << 30 occupies bits [30, 31]
        shift = (tl.arange(0, feat_per_int) * bits).to(tl.int32)
        packed = tl.sum(q_int << shift[None, :], axis=1)

        out_offs = offs_n * T_packed + pid_g * packed_per_group + p
        tl.store(code_ptr + out_offs, packed, mask=mask_n)


def triton_fused_quantize_and_pack_along_last_dim(
    data: torch.Tensor, group_size: int, bit: int
):
    """
    Fused quantize-and-pack along the last dimension of a 4D tensor.
    Drop-in replacement for `triton_quantize_and_pack_along_last_dim` in new_pack.py.

    Args:
        data: input tensor of shape (B, nh, D, T), fp16
        group_size: number of elements per quantization group along the last dim
        bit: number of quantization bits (2 or 4)

    Returns:
        code:  packed int32 tensor of shape (B, nh, D, T // feat_per_int)
        scale: fp16 scale tensor of shape (B, nh, D, num_groups)
        mn:    fp16 min tensor of shape   (B, nh, D, num_groups)
    """
    assert len(data.shape) == 4
    assert bit in (2, 4), "Only 2-bit and 4-bit quantization are supported"
    shape = data.shape
    B, nh, D, T = shape

    assert T % group_size == 0, f"T ({T}) must be divisible by group_size ({group_size})"
    num_groups = T // group_size
    feat_per_int = 32 // bit
    assert group_size % feat_per_int == 0, (
        f"group_size ({group_size}) must be divisible by feat_per_int ({feat_per_int})"
    )

    N = B * nh * D
    data_flat = data.reshape(N, T)
    T_packed = T // feat_per_int

    # Allocate output tensors (no intermediate tensors needed!)
    code = torch.empty((N, T_packed), device=data.device, dtype=torch.int32)
    scale = torch.empty((N, num_groups), device=data.device, dtype=data.dtype)
    mn = torch.empty((N, num_groups), device=data.device, dtype=data.dtype)

    # Choose BLOCK_SIZE_N based on problem size
    if N < 64:
        BLOCK_SIZE_N = 32
    elif N < 256:
        BLOCK_SIZE_N = 64
    else:
        BLOCK_SIZE_N = 128

    grid = (triton.cdiv(N, BLOCK_SIZE_N), num_groups)

    with torch.cuda.device(data.device):
        _fused_quant_and_pack_along_last_dim[grid](
            bit,
            data_flat,
            code,
            scale,
            mn,
            N,
            num_groups,
            group_size,
            feat_per_int,
            T_packed,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_warps=4,
        )

    scale_mn_shape = (B, nh, D, num_groups)
    return (
        code.reshape(B, nh, D, -1),
        scale.reshape(scale_mn_shape),
        mn.reshape(scale_mn_shape),
    )


# ==============================================================================
# Testing & Benchmarking
# ==============================================================================

def _dequantize_packed(code, scale, mn, group_size, bit):
    """Dequantize packed tensor back to fp16 for correctness checking."""
    from new_pack import unpack_and_dequant_kcache
    return unpack_and_dequant_kcache(code, scale.unsqueeze(-1), mn.unsqueeze(-1), group_size, bit)


def test_correctness():
    """Compare fused kernel output against original implementation."""
    from new_pack import triton_quantize_and_pack_along_last_dim as original_fn

    torch.manual_seed(42)
    print("=" * 60)
    print("Correctness Test: Fused vs Original quantize-and-pack")
    print("=" * 60)

    test_configs = [
        # (B, nh, D, T, group_size, bit)
        (1, 32, 128, 128, 32, 2),
        (1, 32, 128, 256, 64, 2),
        (1, 32, 128, 512, 128, 2),
        (2, 32, 128, 256, 32, 4),
        (1, 8, 128, 64, 32, 4),
        (1, 32, 128, 1024, 64, 2),
        (4, 32, 128, 128, 32, 2),
    ]

    all_passed = True
    for B, nh, D, T, gs, bit in test_configs:
        data = torch.randn(B, nh, D, T, device="cuda", dtype=torch.float16)

        # Run original
        code_orig, scale_orig, mn_orig = original_fn(data.clone(), gs, bit)
        # Run fused
        code_fused, scale_fused, mn_fused = triton_fused_quantize_and_pack_along_last_dim(
            data.clone(), gs, bit
        )

        # Compare packed codes (may differ by ±1 quant level due to fp32 vs fp16 rounding)
        match_rate = (code_orig == code_fused).float().mean().item()

        # Compare via dequantization error against original data
        deq_orig = _dequantize_packed(code_orig, scale_orig, mn_orig, gs, bit)
        deq_fused = _dequantize_packed(code_fused, scale_fused, mn_fused, gs, bit)

        err_orig = (data - deq_orig).float().abs().mean().item()
        err_fused = (data - deq_fused).float().abs().mean().item()

        passed = match_rate > 0.95 and abs(err_orig - err_fused) / (err_orig + 1e-8) < 0.1
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(
            f"  [{status}] B={B}, nh={nh}, D={D}, T={T}, gs={gs}, bit={bit} | "
            f"code_match={match_rate:.4f}, err_orig={err_orig:.6f}, err_fused={err_fused:.6f}"
        )

    print("=" * 60)
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)
    return all_passed


def benchmark():
    """Benchmark fused kernel vs original implementation."""
    from new_pack import triton_quantize_and_pack_along_last_dim as original_fn
    import time

    torch.manual_seed(42)
    print("\n" + "=" * 60)
    print("Performance Benchmark: Fused vs Original")
    print("=" * 60)

    configs = [
        # (B, nh, D, T, group_size, bit, label)
        (1, 32, 128, 1024, 32, 2, "Typical decode K-cache"),
        (1, 32, 128, 4096, 64, 2, "Long context K-cache"),
        (1, 32, 128, 128, 32, 2, "Short V-cache"),
        (1, 32, 128, 4096, 128, 2, "Long context, large group"),
        (1, 32, 128, 4096, 64, 4, "4-bit quantization"),
    ]

    warmup_iters = 20
    bench_iters = 100

    for B, nh, D, T, gs, bit, label in configs:
        data = torch.randn(B, nh, D, T, device="cuda", dtype=torch.float16)

        # Warmup
        for _ in range(warmup_iters):
            original_fn(data.clone(), gs, bit)
            triton_fused_quantize_and_pack_along_last_dim(data.clone(), gs, bit)
        torch.cuda.synchronize()

        # Benchmark original
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench_iters):
            original_fn(data.clone(), gs, bit)
        torch.cuda.synchronize()
        t_orig = (time.perf_counter() - t0) / bench_iters * 1000  # ms

        # Benchmark fused
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench_iters):
            triton_fused_quantize_and_pack_along_last_dim(data.clone(), gs, bit)
        torch.cuda.synchronize()
        t_fused = (time.perf_counter() - t0) / bench_iters * 1000  # ms

        speedup = t_orig / t_fused if t_fused > 0 else float("inf")
        print(
            f"  {label:30s} | orig={t_orig:.3f}ms, fused={t_fused:.3f}ms, "
            f"speedup={speedup:.2f}x"
        )

    print("=" * 60)


if __name__ == "__main__":
    test_correctness()
    benchmark()
