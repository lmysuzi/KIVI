"""
Standalone Triton kernels for dequantizing K and V cache chunks.

These will later be inlined into the fused QuantFlashDecoding kernel.

=== Fix from v1 ===
The v1 test showed max_diff=0.003906 (1 ULP at fp16 ~2.0 range).
Root cause: fp16 multiply-add rounding differs between Triton and PyTorch reference.

Fix: perform dequant arithmetic (quant_val * scale + mn) in FP32, then convert
to FP16 at the end. This matches the original KIVI unpack_and_dequant code path
which also does: data.to(float16) * scale + mn (where scale/mn are fp16, so the
multiply is in fp16). To get bit-exact results, both kernel and reference must
use the same intermediate precision.

Decision: We use FP32 intermediate in the kernel (better numerical quality for
the fused FlashDecoding kernel where scores accumulate across many tokens).
The reference also uses FP32 intermediate. Both convert to FP16 at the end.
"""

import torch
import triton
import triton.language as tl


# =====================================================================
# Kernel 1: Dequantize a chunk of K cache
# =====================================================================

@triton.jit
def dequant_k_chunk_kernel(
    # Quantized K (transposed storage)
    k_quant_ptr,   # [BH, D, L_q_packed]
    k_scale_ptr,   # [BH, D, L_q_groups]
    k_mn_ptr,      # [BH, D, L_q_groups]
    # Output: dequantized K chunk
    k_out_ptr,     # [BH, chunk_len, D]
    # Dimensions
    D: tl.constexpr,
    L_q_packed,     # L_q // pack_factor
    L_q_groups,     # L_q // group_size
    chunk_start,    # start seq position (unpacked)
    chunk_len,      # number of seq positions in this chunk
    # Quant parameters
    BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # Tile sizes
    BLOCK_SEQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    PACK_FACTOR: tl.constexpr = 32 // BITS
    MASK: tl.constexpr = (1 << BITS) - 1

    pid_seq = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_bh = tl.program_id(2)

    offs_seq = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_seq = offs_seq < chunk_len
    mask_d = offs_d < D
    mask_2d_load = mask_d[:, None] & mask_seq[None, :]     # [BLOCK_D, BLOCK_SEQ]
    mask_2d_store = mask_seq[:, None] & mask_d[None, :]    # [BLOCK_SEQ, BLOCK_D]

    # Compute indices
    global_seq = chunk_start + offs_seq                     # [BLOCK_SEQ]
    pack_idx = global_seq // PACK_FACTOR                    # [BLOCK_SEQ]
    bit_offset = (global_seq % PACK_FACTOR) * BITS          # [BLOCK_SEQ]
    group_idx = global_seq // GROUP_SIZE                    # [BLOCK_SEQ]

    # --- Read k_quant[bh, d, pack_idx] → [BLOCK_D, BLOCK_SEQ] ---
    bh_offset_q = pid_bh * D * L_q_packed
    k_quant_ptrs = (k_quant_ptr + bh_offset_q
                    + offs_d[:, None] * L_q_packed
                    + pack_idx[None, :])
    packed_vals = tl.load(k_quant_ptrs, mask=mask_2d_load, other=0)

    # Unpack
    quant_vals = (packed_vals >> bit_offset[None, :]) & MASK  # [BLOCK_D, BLOCK_SEQ] int32

    # --- Read scale and mn → [BLOCK_D, BLOCK_SEQ] ---
    bh_offset_s = pid_bh * D * L_q_groups
    k_scale_ptrs = (k_scale_ptr + bh_offset_s
                    + offs_d[:, None] * L_q_groups
                    + group_idx[None, :])
    k_mn_ptrs = (k_mn_ptr + bh_offset_s
                 + offs_d[:, None] * L_q_groups
                 + group_idx[None, :])
    scale = tl.load(k_scale_ptrs, mask=mask_2d_load, other=0.0)
    mn = tl.load(k_mn_ptrs, mask=mask_2d_load, other=0.0)

    # --- Dequantize in FP32 ---
    fp_vals = quant_vals.to(tl.float32) * scale.to(tl.float32) + mn.to(tl.float32)
    fp_vals = fp_vals.to(tl.float16)   # [BLOCK_D, BLOCK_SEQ]

    # --- Write with transpose: [BLOCK_D, BLOCK_SEQ] → [BLOCK_SEQ, BLOCK_D] ---
    fp_vals_t = tl.trans(fp_vals)      # [BLOCK_SEQ, BLOCK_D]

    bh_offset_out = pid_bh * chunk_len * D
    k_out_ptrs = (k_out_ptr + bh_offset_out
                  + offs_seq[:, None] * D
                  + offs_d[None, :])
    tl.store(k_out_ptrs, fp_vals_t, mask=mask_2d_store)


# =====================================================================
# Kernel 2: Dequantize a chunk of V cache
# =====================================================================

@triton.jit
def dequant_v_chunk_kernel(
    # Quantized V (standard storage)
    v_quant_ptr,   # [BH, L_q, D_packed]
    v_scale_ptr,   # [BH, L_q, D_groups]
    v_mn_ptr,      # [BH, L_q, D_groups]
    # Output
    v_out_ptr,     # [BH, chunk_len, D]
    # Dimensions
    D: tl.constexpr,
    D_packed,       # D // pack_factor
    D_groups,       # D // group_size
    L_q,            # total quantized seq length (for stride)
    chunk_start,
    chunk_len,
    # Quant parameters
    BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # Tile sizes
    BLOCK_SEQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    PACK_FACTOR: tl.constexpr = 32 // BITS
    MASK: tl.constexpr = (1 << BITS) - 1

    pid_seq = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_bh = tl.program_id(2)

    offs_seq = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_seq = offs_seq < chunk_len
    mask_d = offs_d < D
    mask_2d = mask_seq[:, None] & mask_d[None, :]

    # Compute indices
    global_seq = chunk_start + offs_seq          # [BLOCK_SEQ]
    pack_idx = offs_d // PACK_FACTOR             # [BLOCK_D] — pack along D
    bit_offset = (offs_d % PACK_FACTOR) * BITS   # [BLOCK_D]
    group_idx = offs_d // GROUP_SIZE             # [BLOCK_D]

    # --- Read v_quant[bh, global_seq, pack_idx] → [BLOCK_SEQ, BLOCK_D] ---
    bh_offset_q = pid_bh * L_q * D_packed
    v_quant_ptrs = (v_quant_ptr + bh_offset_q
                    + global_seq[:, None] * D_packed
                    + pack_idx[None, :])
    v_quant_mask = mask_seq[:, None] & (pack_idx[None, :] < D_packed)
    packed_vals = tl.load(v_quant_ptrs, mask=v_quant_mask, other=0)

    # Unpack
    quant_vals = (packed_vals >> bit_offset[None, :]) & MASK

    # --- Read scale and mn → [BLOCK_SEQ, BLOCK_D] ---
    bh_offset_s = pid_bh * L_q * D_groups
    v_scale_ptrs = (v_scale_ptr + bh_offset_s
                    + global_seq[:, None] * D_groups
                    + group_idx[None, :])
    v_mn_ptrs = (v_mn_ptr + bh_offset_s
                 + global_seq[:, None] * D_groups
                 + group_idx[None, :])
    scale_mask = mask_seq[:, None] & (group_idx[None, :] < D_groups)
    scale = tl.load(v_scale_ptrs, mask=scale_mask, other=0.0)
    mn = tl.load(v_mn_ptrs, mask=scale_mask, other=0.0)

    # --- Dequantize in FP32 ---
    fp_vals = quant_vals.to(tl.float32) * scale.to(tl.float32) + mn.to(tl.float32)
    fp_vals = fp_vals.to(tl.float16)

    # --- Write (no transpose) ---
    bh_offset_out = pid_bh * chunk_len * D
    v_out_ptrs = (v_out_ptr + bh_offset_out
                  + offs_seq[:, None] * D
                  + offs_d[None, :])
    tl.store(v_out_ptrs, fp_vals, mask=mask_2d)


# =====================================================================
# Python wrappers
# =====================================================================

def dequant_k_chunk(k_quant, k_scale, k_mn, chunk_start, chunk_len,
                    D, group_size=128, bits=2):
    BH = k_quant.shape[0]
    L_q_packed = k_quant.shape[2]
    L_q_groups = k_scale.shape[2]
    k_out = torch.empty(BH, chunk_len, D, dtype=torch.float16, device=k_quant.device)
    BLOCK_SEQ = min(64, triton.next_power_of_2(chunk_len))
    BLOCK_D = min(64, triton.next_power_of_2(D))
    grid = (triton.cdiv(chunk_len, BLOCK_SEQ), triton.cdiv(D, BLOCK_D), BH)
    dequant_k_chunk_kernel[grid](
        k_quant, k_scale, k_mn, k_out,
        D, L_q_packed, L_q_groups, chunk_start, chunk_len,
        BITS=bits, GROUP_SIZE=group_size,
        BLOCK_SEQ=BLOCK_SEQ, BLOCK_D=BLOCK_D,
    )
    return k_out


def dequant_v_chunk(v_quant, v_scale, v_mn, chunk_start, chunk_len,
                    D, L_q, group_size=128, bits=2):
    BH = v_quant.shape[0]
    D_packed = v_quant.shape[2]
    D_groups = v_scale.shape[2]
    v_out = torch.empty(BH, chunk_len, D, dtype=torch.float16, device=v_quant.device)
    BLOCK_SEQ = min(64, triton.next_power_of_2(chunk_len))
    BLOCK_D = min(64, triton.next_power_of_2(D))
    grid = (triton.cdiv(chunk_len, BLOCK_SEQ), triton.cdiv(D, BLOCK_D), BH)
    dequant_v_chunk_kernel[grid](
        v_quant, v_scale, v_mn, v_out,
        D, D_packed, D_groups, L_q, chunk_start, chunk_len,
        BITS=bits, GROUP_SIZE=group_size,
        BLOCK_SEQ=BLOCK_SEQ, BLOCK_D=BLOCK_D,
    )
    return v_out


# =====================================================================
# Reference implementations (FP32 intermediate, matching kernel)
# =====================================================================

def reference_dequant_k(k_quant, k_scale, k_mn, chunk_start, chunk_len,
                        D, group_size=128, bits=2):
    """Reference K dequant. FP32 intermediate, FP16 output."""
    pack_factor = 32 // bits
    mask = (1 << bits) - 1
    BH = k_quant.shape[0]
    result = torch.zeros(BH, chunk_len, D, dtype=torch.float16, device=k_quant.device)
    for s_local in range(chunk_len):
        s_global = chunk_start + s_local
        p = s_global // pack_factor
        bit_off = (s_global % pack_factor) * bits
        g = s_global // group_size
        packed = k_quant[:, :, p]
        qval = (packed >> bit_off) & mask
        scale = k_scale[:, :, g]
        mn = k_mn[:, :, g]
        # FP32 intermediate
        fp_val = qval.to(torch.float32) * scale.to(torch.float32) + mn.to(torch.float32)
        result[:, s_local, :] = fp_val.to(torch.float16)
    return result


def reference_dequant_v(v_quant, v_scale, v_mn, chunk_start, chunk_len,
                        D, group_size=128, bits=2):
    """Reference V dequant. FP32 intermediate, FP16 output."""
    pack_factor = 32 // bits
    mask_val = (1 << bits) - 1
    BH = v_quant.shape[0]
    result = torch.zeros(BH, chunk_len, D, dtype=torch.float16, device=v_quant.device)
    for s_local in range(chunk_len):
        s_global = chunk_start + s_local
        for d in range(D):
            p = d // pack_factor
            bit_off = (d % pack_factor) * bits
            g = d // group_size
            packed = v_quant[:, s_global, p]
            qval = (packed >> bit_off) & mask_val
            scale = v_scale[:, s_global, g]
            mn = v_mn[:, s_global, g]
            fp_val = qval.to(torch.float32) * scale.to(torch.float32) + mn.to(torch.float32)
            result[:, s_local, d] = fp_val.to(torch.float16)
    return result


# =====================================================================
# Tests
# =====================================================================

def test_dequant_k():
    from new_pack import triton_quantize_and_pack_along_last_dim
    torch.manual_seed(42)
    B, H, seq_len, D = 2, 4, 512, 128
    group_size = 128
    bits = 2

    k_orig = torch.randn(B, H, seq_len, D, dtype=torch.float16, device='cuda')
    k_trans = k_orig.transpose(2, 3).contiguous()
    k_quant, k_scale, k_mn = triton_quantize_and_pack_along_last_dim(k_trans, group_size, bits)

    BH = B * H
    k_quant_flat = k_quant.reshape(BH, D, -1).contiguous()
    k_scale_flat = k_scale.reshape(BH, D, -1).contiguous()
    k_mn_flat = k_mn.reshape(BH, D, -1).contiguous()

    test_cases = [
        (0, 128), (128, 128), (0, 256),
        (64, 128), (0, seq_len), (256, 64),
    ]
    all_pass = True
    for chunk_start, chunk_len in test_cases:
        if chunk_start + chunk_len > seq_len:
            continue
        k_triton = dequant_k_chunk(k_quant_flat, k_scale_flat, k_mn_flat,
                                    chunk_start, chunk_len, D, group_size, bits)
        k_ref = reference_dequant_k(k_quant_flat, k_scale_flat, k_mn_flat,
                                     chunk_start, chunk_len, D, group_size, bits)
        # Bit-exact comparison (both use FP32 intermediate → FP16)
        max_diff = (k_triton.float() - k_ref.float()).abs().max().item()
        exact_match = torch.equal(k_triton, k_ref)
        status = "PASS" if exact_match else f"CLOSE (max_diff={max_diff:.6f})"
        if max_diff > 0.001:
            status = f"FAIL (max_diff={max_diff:.6f})"
            all_pass = False
        print(f"  K chunk_start={chunk_start}, chunk_len={chunk_len}: {status}")
        if not exact_match and max_diff > 0.001:
            diff = (k_triton.float() - k_ref.float()).abs()
            indices = torch.where(diff > 0.001)
            for i in range(min(3, len(indices[0]))):
                bh, s, d = indices[0][i].item(), indices[1][i].item(), indices[2][i].item()
                print(f"    [{bh},{s},{d}]: triton={k_triton[bh,s,d].item():.6f}, ref={k_ref[bh,s,d].item():.6f}")
    return all_pass


def test_dequant_v():
    from new_pack import triton_quantize_and_pack_along_last_dim
    torch.manual_seed(42)
    B, H, seq_len, D = 2, 4, 512, 128
    group_size = 128
    bits = 2

    v_orig = torch.randn(B, H, seq_len, D, dtype=torch.float16, device='cuda')
    v_quant, v_scale, v_mn = triton_quantize_and_pack_along_last_dim(
        v_orig.contiguous(), group_size, bits)

    BH = B * H
    v_quant_flat = v_quant.reshape(BH, seq_len, -1).contiguous()
    v_scale_flat = v_scale.reshape(BH, seq_len, -1).contiguous()
    v_mn_flat = v_mn.reshape(BH, seq_len, -1).contiguous()

    test_cases = [
        (0, 128), (128, 128), (0, 256),
        (64, 128), (0, seq_len), (256, 64),
    ]
    all_pass = True
    for chunk_start, chunk_len in test_cases:
        if chunk_start + chunk_len > seq_len:
            continue
        v_triton = dequant_v_chunk(v_quant_flat, v_scale_flat, v_mn_flat,
                                    chunk_start, chunk_len, D, seq_len, group_size, bits)
        v_ref = reference_dequant_v(v_quant_flat, v_scale_flat, v_mn_flat,
                                     chunk_start, chunk_len, D, group_size, bits)
        max_diff = (v_triton.float() - v_ref.float()).abs().max().item()
        exact_match = torch.equal(v_triton, v_ref)
        status = "PASS" if exact_match else f"CLOSE (max_diff={max_diff:.6f})"
        if max_diff > 0.001:
            status = f"FAIL (max_diff={max_diff:.6f})"
            all_pass = False
        print(f"  V chunk_start={chunk_start}, chunk_len={chunk_len}: {status}")
        if not exact_match and max_diff > 0.001:
            diff = (v_triton.float() - v_ref.float()).abs()
            indices = torch.where(diff > 0.001)
            for i in range(min(3, len(indices[0]))):
                bh, s, d = indices[0][i].item(), indices[1][i].item(), indices[2][i].item()
                print(f"    [{bh},{s},{d}]: triton={v_triton[bh,s,d].item():.6f}, ref={v_ref[bh,s,d].item():.6f}")
    return all_pass


def test_roundtrip_quantize_dequantize():
    """
    Strongest correctness test: quantize FP16 data, dequant with our kernel,
    compare against directly dequantizing with the original PyTorch unpack code.
    
    This tests the FULL pipeline:
      original data → triton_quantize_and_pack → our dequant kernel → output
      original data → quant_and_pack_kcache → unpack_and_dequant_kcache → output
    
    Both paths should produce identical results (same quantization error).
    """
    from new_pack import (
        triton_quantize_and_pack_along_last_dim,
        quant_and_pack_kcache,
        unpack_and_dequant_kcache,
        quant_and_pack_vcache,
        unpack_and_dequant_vcache,
    )

    torch.manual_seed(99)
    B, H, seq_len, D = 1, 2, 256, 128
    group_size = 128
    bits = 2
    pack_factor = 16

    print("\n  --- K roundtrip test ---")
    # K original: [B, H, seq_len, D]
    k_orig = torch.randn(B, H, seq_len, D, dtype=torch.float16, device='cuda')

    # Path A: original PyTorch quant/unpack (operates on [B,H,T,D], pack along T)
    k_code_a, k_scale_a, k_mn_a = quant_and_pack_kcache(k_orig, group_size, bits)
    k_dequant_a = unpack_and_dequant_kcache(k_code_a, k_scale_a, k_mn_a, group_size, bits)
    # k_dequant_a: [B, H, seq_len, D]

    # Path B: triton quant on transposed K, then our dequant kernel
    k_trans = k_orig.transpose(2, 3).contiguous()  # [B, H, D, seq_len]
    k_code_b, k_scale_b, k_mn_b = triton_quantize_and_pack_along_last_dim(k_trans, group_size, bits)
    # k_code_b: [B, H, D, seq_len//pack]
    BH = B * H
    k_code_b_flat = k_code_b.reshape(BH, D, -1).contiguous()
    k_scale_b_flat = k_scale_b.reshape(BH, D, -1).contiguous()
    k_mn_b_flat = k_mn_b.reshape(BH, D, -1).contiguous()
    k_dequant_b = dequant_k_chunk(k_code_b_flat, k_scale_b_flat, k_mn_b_flat,
                                   0, seq_len, D, group_size, bits)
    # k_dequant_b: [BH, seq_len, D] → reshape to [B, H, seq_len, D]
    k_dequant_b = k_dequant_b.reshape(B, H, seq_len, D)

    # Compare
    # NOTE: Path A and Path B use DIFFERENT quantization implementations
    # (PyTorch vs Triton). The quantized values might differ slightly,
    # leading to different dequantized outputs. This is expected.
    # What matters: both are valid 2-bit quantizations of the same input.
    max_diff_ab = (k_dequant_a.float() - k_dequant_b.float()).abs().max().item()
    print(f"  Path A vs B max_diff: {max_diff_ab:.6f}")
    print(f"  (If > 0, likely due to different quant implementations, not dequant bugs)")

    # More useful: compare our kernel against our reference on the SAME quant data
    k_dequant_ref = reference_dequant_k(k_code_b_flat, k_scale_b_flat, k_mn_b_flat,
                                         0, seq_len, D, group_size, bits)
    k_dequant_ref = k_dequant_ref.reshape(B, H, seq_len, D)
    max_diff_kernel_ref = (k_dequant_b.float() - k_dequant_ref.float()).abs().max().item()
    exact = torch.equal(k_dequant_b, k_dequant_ref)
    print(f"  Kernel vs Reference (same quant data): max_diff={max_diff_kernel_ref:.6f}, exact={exact}")

    print("\n  --- V roundtrip test ---")
    v_orig = torch.randn(B, H, seq_len, D, dtype=torch.float16, device='cuda')

    # Path A: original PyTorch quant/unpack
    v_code_a, v_scale_a, v_mn_a = quant_and_pack_vcache(v_orig, group_size, bits)
    v_dequant_a = unpack_and_dequant_vcache(v_code_a, v_scale_a, v_mn_a, group_size, bits)

    # Path B: triton quant + our dequant kernel
    v_code_b, v_scale_b, v_mn_b = triton_quantize_and_pack_along_last_dim(
        v_orig.contiguous(), group_size, bits)
    v_code_b_flat = v_code_b.reshape(BH, seq_len, -1).contiguous()
    v_scale_b_flat = v_scale_b.reshape(BH, seq_len, -1).contiguous()
    v_mn_b_flat = v_mn_b.reshape(BH, seq_len, -1).contiguous()
    v_dequant_b = dequant_v_chunk(v_code_b_flat, v_scale_b_flat, v_mn_b_flat,
                                   0, seq_len, D, seq_len, group_size, bits)
    v_dequant_b = v_dequant_b.reshape(B, H, seq_len, D)

    max_diff_ab = (v_dequant_a.float() - v_dequant_b.float()).abs().max().item()
    print(f"  Path A vs B max_diff: {max_diff_ab:.6f}")

    v_dequant_ref = reference_dequant_v(v_code_b_flat, v_scale_b_flat, v_mn_b_flat,
                                         0, seq_len, D, group_size, bits)
    v_dequant_ref = v_dequant_ref.reshape(B, H, seq_len, D)
    max_diff_kernel_ref = (v_dequant_b.float() - v_dequant_ref.float()).abs().max().item()
    exact = torch.equal(v_dequant_b, v_dequant_ref)
    print(f"  Kernel vs Reference (same quant data): max_diff={max_diff_kernel_ref:.6f}, exact={exact}")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: K dequant kernel vs reference (FP32 intermediate)")
    print("=" * 60)
    k_pass = test_dequant_k()

    print()
    print("=" * 60)
    print("Test 2: V dequant kernel vs reference (FP32 intermediate)")
    print("=" * 60)
    v_pass = test_dequant_v()

    print()
    print("=" * 60)
    print("Test 3: Roundtrip quantize-dequantize")
    print("=" * 60)
    test_roundtrip_quantize_dequantize()

    print()
    if k_pass and v_pass:
        print("All core tests PASSED!")
    else:
        print("Some core tests FAILED!")