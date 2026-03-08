"""
Fused quantization and bit-packing kernel for KIVI KV cache compression.

Replaces the multi-step pipeline in new_pack.py:
    Triton minmax kernel → PyTorch sub/div/clamp/round/cast → Triton pack kernel
with a single Triton kernel.

Key design choices vs. the naive fused approach:
    1. Data reshaped to (N*num_groups, group_size) BEFORE entering the kernel,
       so the stride between consecutive rows is group_size (e.g. 32) instead of T
       (e.g. 672).  This gives ~20× better memory coalescing.
    2. Full group loaded at once in Phase 1 (single vectorised tl.load of
       [BLOCK_SIZE, group_size]) for min/max — no streaming loop needed.
    3. Phase 2 re-reads in feat_per_int chunks; because the working set per
       thread-block is only BLOCK_SIZE × group_size × 2 B (≈ 8 KB for the
       typical 128 × 32 tile), the re-reads hit per-SM L1 cache (128 KB on
       Ampere) with near-100 % hit rate.
    4. 1-D grid — simpler scheduling, fewer concurrent programs competing for
       L2 bandwidth.

Drop-in replacement for `triton_quantize_and_pack_along_last_dim` in new_pack.py.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
import torch
import numpy as np


@triton.jit
def _fused_quant_pack(
    bits: tl.constexpr,
    data_ptr,
    code_ptr,
    scale_ptr,
    mn_ptr,
    num_rows,                       # N * num_groups  (one row = one quant group)
    group_size: tl.constexpr,       # elements per quantization group (32, 64, 128)
    feat_per_int: tl.constexpr,     # 32 // bits  (16 for 2-bit, 8 for 4-bit)
    packed_per_group: tl.constexpr, # group_size // feat_per_int
    BLOCK_SIZE: tl.constexpr,       # rows per program
):
    """
    Fused min-max asymmetric quantization + bit-packing.

    Data layout: (num_rows, group_size), row-major, contiguous.
    Grid:  1-D,  (cdiv(num_rows, BLOCK_SIZE),)

    Phase 1 — Load full group [BLOCK_SIZE, group_size], compute per-row
              min / max entirely from registers.
    Phase 2 — Re-read the same data in [BLOCK_SIZE, feat_per_int] chunks
              (L1-cached), quantise to [0, 2^bits − 1], bit-pack
              feat_per_int values into one int32 via shift-and-sum.
    """
    max_int: tl.constexpr = (1 << bits) - 1

    pid = tl.program_id(0)
    row_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offs < num_rows

    # Base byte-offset for each row (stride = group_size between rows)
    base = row_offs * group_size

    # ======================== Phase 1: full-group min / max ========================
    cols = tl.arange(0, group_size)
    data = tl.load(data_ptr + (base[:, None] + cols[None, :]),
                   mask=row_mask[:, None], other=0.0)

    row_min = tl.min(data, axis=1)          # fp16 native — no precision loss
    row_max = tl.max(data, axis=1)

    mn_f32 = row_min.to(tl.float32)
    scale_val = (row_max.to(tl.float32) - mn_f32) / max_int
    inv_scale = 1.0 / (scale_val + 1e-10)

    tl.store(scale_ptr + row_offs, scale_val.to(tl.float16), mask=row_mask)
    tl.store(mn_ptr   + row_offs, mn_f32.to(tl.float16),     mask=row_mask)

    # ======================== Phase 2: quantise + bit-pack ========================
    for p in tl.static_range(packed_per_group):
        feat_offs = p * feat_per_int + tl.arange(0, feat_per_int)
        chunk = tl.load(data_ptr + (base[:, None] + feat_offs[None, :]),
                        mask=row_mask[:, None], other=0.0).to(tl.float32)

        # Quantise: round(clamp((x − mn) / scale, 0, max_int))
        q = (chunk - mn_f32[:, None]) * inv_scale[:, None]
        q = libdevice.rint(q)
        q = tl.maximum(tl.minimum(q, float(max_int)), 0.0)
        q_int = q.to(tl.int32)

        # Bit-pack: shift each quantised value to its bit lane and sum.
        shift = (tl.arange(0, feat_per_int) * bits).to(tl.int32)
        packed = tl.sum(q_int << shift[None, :], axis=1)

        tl.store(code_ptr + row_offs * packed_per_group + p,
                 packed, mask=row_mask)


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
    B, nh, D, T = data.shape

    assert T % group_size == 0, f"T ({T}) must be divisible by group_size ({group_size})"
    num_groups = T // group_size
    feat_per_int = 32 // bit
    packed_per_group = group_size // feat_per_int
    assert group_size % feat_per_int == 0, (
        f"group_size ({group_size}) must be divisible by feat_per_int ({feat_per_int})"
    )

    N = B * nh * D
    num_rows = N * num_groups

    # ---- Critical layout change ----
    # Reshape to (num_rows, group_size) so that the stride between consecutive
    # rows is group_size (e.g. 32) rather than T (e.g. 672).
    # This dramatically improves memory coalescing for both phases.
    data_2d = data.reshape(num_rows, group_size)

    code  = torch.empty((num_rows, packed_per_group), device=data.device, dtype=torch.int32)
    scale = torch.empty(num_rows, device=data.device, dtype=data.dtype)
    mn    = torch.empty(num_rows, device=data.device, dtype=data.dtype)

    BLOCK_SIZE = 128
    grid = (triton.cdiv(num_rows, BLOCK_SIZE),)

    with torch.cuda.device(data.device):
        _fused_quant_pack[grid](
            bit,
            data_2d, code, scale, mn,
            num_rows,
            group_size,
            feat_per_int,
            packed_per_group,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    T_packed = T // feat_per_int
    scale_mn_shape = (B, nh, D, num_groups)
    return (
        code.reshape(B, nh, D, T_packed),
        scale.reshape(scale_mn_shape),
        mn.reshape(scale_mn_shape),
    )


# ==============================================================================
# Testing & Benchmarking
# ==============================================================================

def _dequantize_packed(code, scale, mn, group_size, bit):
    """Dequantize packed tensor back to fp16 for correctness checking."""
    from new_pack import unpack_and_dequant_vcache
    return unpack_and_dequant_vcache(code, scale.unsqueeze(-1), mn.unsqueeze(-1), group_size, bit)


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
        # Shapes matching the real model (longchat-7b, ~702 tokens, residual=32)
        (1, 32, 128, 640, 32, 2),   # K-cache after transpose: (B, nh, D, T_quant)
        (1, 32, 640, 128, 32, 2),   # V-cache: (B, nh, T_quant, D)
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
        (1, 32, 128, 640, 32, 2,  "K-cache prefill (real)"),
        (1, 32, 640, 128, 32, 2,  "V-cache prefill (real)"),
        (1, 32, 128, 1024, 32, 2, "Typical decode K-cache"),
        (1, 32, 128, 4096, 64, 2, "Long context K-cache"),
        (1, 32, 128, 128, 32, 2,  "Short V-cache"),
        (1, 32, 128, 4096, 128, 2,"Long context, large group"),
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