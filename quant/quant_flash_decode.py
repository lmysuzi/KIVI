"""
QuantFlashDecoding: Fused attention decoding over quantized KV cache.

Architecture:
  Main kernel (1 block per chunk per query-head):
    - Load Q [1, D]
    - Load K chunk (dequant if quantized, direct load if full-precision)
    - Compute scores = Q @ K^T / sqrt(D)
    - Online softmax: local_max, local_sum, local_weights
    - Load V chunk (dequant if quantized, direct load if full-precision)
    - Compute partial_out = local_weights @ V
    - Write (partial_out, lse) to global memory

  Reduce kernel (1 block per query-head):
    - Read all (partial_out, lse) for this query-head
    - Merge via log-sum-exp rescaling
    - Write final output [1, D]

Data layout (same as QuantKVCache):
  K quant: [B*H_kv, D, L_q_packed] int32, transposed, packed along seq
  K scale: [B*H_kv, D, L_q_groups] fp16
  V quant: [B*H_kv, L_q, D_packed] int32, packed along D
  V scale: [B*H_kv, L_q, D_groups] fp16
  K full:  [B*H_kv, L_f, D] fp16
  V full:  [B*H_kv, L_f, D] fp16

Constraints (for this version):
  - L_q must be divisible by CHUNK_SIZE (pad if needed)
  - CHUNK_SIZE must be divisible by GROUP_SIZE, or GROUP_SIZE divisible by CHUNK_SIZE
  - D must be divisible by BLOCK_D
  - bits = 2 (can be extended to 4, 8)
"""

import math
import torch
import triton
import triton.language as tl
# =====================================================================
# Main Kernel
# =====================================================================

@triton.jit
def _qfd_compute_scores_quant(
    Q_ptr, K_quant_ptr, K_scale_ptr, K_mn_ptr,
    pid_bh_q, pid_bh_kv, chunk_start, mask_chunk, offs_chunk,
    D: tl.constexpr, BLOCK_D: tl.constexpr, CHUNK_SIZE: tl.constexpr,
    L_q_packed, L_q_groups,
    BITS: tl.constexpr, GROUP_SIZE: tl.constexpr,
    PACK_FACTOR: tl.constexpr, MASK: tl.constexpr,
):
    """Compute scores = Q @ K_quant_chunk^T in FP32. Returns [CHUNK_SIZE] scores."""
    scores = tl.zeros((CHUNK_SIZE,), dtype=tl.float32)
    q_base = Q_ptr + pid_bh_q * D
    kq_base = K_quant_ptr + pid_bh_kv * D * L_q_packed
    ks_base = K_scale_ptr + pid_bh_kv * D * L_q_groups
    km_base = K_mn_ptr + pid_bh_kv * D * L_q_groups

    global_seq = chunk_start + offs_chunk
    pack_idx = global_seq // PACK_FACTOR
    bit_offset = (global_seq % PACK_FACTOR) * BITS
    group_idx = global_seq // GROUP_SIZE

    num_d_blocks: tl.constexpr = D // BLOCK_D
    for d_block in range(num_d_blocks):
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        q_tile = tl.load(q_base + offs_d).to(tl.float32)

        kq_ptrs = kq_base + offs_d[:, None] * L_q_packed + pack_idx[None, :]
        packed_vals = tl.load(kq_ptrs, mask=mask_chunk[None, :], other=0)
        quant_vals = (packed_vals >> bit_offset[None, :]) & MASK

        ks_ptrs = ks_base + offs_d[:, None] * L_q_groups + group_idx[None, :]
        km_ptrs = km_base + offs_d[:, None] * L_q_groups + group_idx[None, :]
        scale = tl.load(ks_ptrs, mask=mask_chunk[None, :], other=0.0)
        mn = tl.load(km_ptrs, mask=mask_chunk[None, :], other=0.0)

        k_dequant = quant_vals.to(tl.float32) * scale.to(tl.float32) + mn.to(tl.float32)
        k_tile_f32 = tl.trans(k_dequant)  # [CHUNK_SIZE, BLOCK_D]
        scores += tl.sum(k_tile_f32 * q_tile[None, :], axis=1)

    return scores


@triton.jit
def _qfd_compute_scores_full(
    Q_ptr, K_full_ptr,
    pid_bh_q, pid_bh_kv, chunk_start, mask_chunk, offs_chunk,
    L_q, L_f,
    D: tl.constexpr, BLOCK_D: tl.constexpr, CHUNK_SIZE: tl.constexpr,
):
    """Compute scores = Q @ K_full_chunk^T in FP32. Returns [CHUNK_SIZE] scores."""
    scores = tl.zeros((CHUNK_SIZE,), dtype=tl.float32)
    q_base = Q_ptr + pid_bh_q * D
    full_seq_start = chunk_start - L_q
    k_full_base = K_full_ptr + pid_bh_kv * L_f * D

    num_d_blocks: tl.constexpr = D // BLOCK_D
    for d_block in range(num_d_blocks):
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        q_tile = tl.load(q_base + offs_d).to(tl.float32)

        k_ptrs = k_full_base + (full_seq_start + offs_chunk[:, None]) * D + offs_d[None, :]
        k_tile = tl.load(k_ptrs, mask=mask_chunk[:, None], other=0.0)
        k_tile_f32 = k_tile.to(tl.float32)
        scores += tl.sum(k_tile_f32 * q_tile[None, :], axis=1)

    return scores


@triton.jit
def _qfd_compute_output_quant(
    local_weights, V_quant_ptr, V_scale_ptr, V_mn_ptr,
    Out_ptr,
    pid_bh_q, pid_bh_kv, pid_chunk, chunk_start, mask_chunk, offs_chunk,
    num_chunks,
    D: tl.constexpr, BLOCK_D: tl.constexpr, CHUNK_SIZE: tl.constexpr,
    L_q, D_packed, D_groups,
    BITS: tl.constexpr, GROUP_SIZE: tl.constexpr,
    PACK_FACTOR: tl.constexpr, MASK: tl.constexpr,
):
    """Compute partial_out = local_weights @ V_quant_chunk. Write to Out."""
    out_base = Out_ptr + pid_bh_q * num_chunks * D + pid_chunk * D
    vq_base = V_quant_ptr + pid_bh_kv * L_q * D_packed
    vs_base = V_scale_ptr + pid_bh_kv * L_q * D_groups
    vm_base = V_mn_ptr + pid_bh_kv * L_q * D_groups
    global_seq = chunk_start + offs_chunk

    num_d_blocks: tl.constexpr = D // BLOCK_D
    for d_block in range(num_d_blocks):
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        d_pack_idx = offs_d // PACK_FACTOR
        d_bit_offset = (offs_d % PACK_FACTOR) * BITS
        d_group_idx = offs_d // GROUP_SIZE

        vq_ptrs = vq_base + global_seq[:, None] * D_packed + d_pack_idx[None, :]
        packed_vals = tl.load(vq_ptrs, mask=mask_chunk[:, None], other=0)
        quant_vals = (packed_vals >> d_bit_offset[None, :]) & MASK

        vs_ptrs = vs_base + global_seq[:, None] * D_groups + d_group_idx[None, :]
        vm_ptrs = vm_base + global_seq[:, None] * D_groups + d_group_idx[None, :]
        v_scale_val = tl.load(vs_ptrs, mask=mask_chunk[:, None], other=0.0)
        v_mn_val = tl.load(vm_ptrs, mask=mask_chunk[:, None], other=0.0)

        v_tile_f32 = quant_vals.to(tl.float32) * v_scale_val.to(tl.float32) + v_mn_val.to(tl.float32)
        partial_out_d = tl.sum(local_weights[:, None] * v_tile_f32, axis=0)
        tl.store(out_base + offs_d, partial_out_d)


@triton.jit
def _qfd_compute_output_full(
    local_weights, V_full_ptr,
    Out_ptr,
    pid_bh_q, pid_bh_kv, pid_chunk, chunk_start, mask_chunk, offs_chunk,
    num_chunks, L_q, L_f,
    D: tl.constexpr, BLOCK_D: tl.constexpr, CHUNK_SIZE: tl.constexpr,
):
    """Compute partial_out = local_weights @ V_full_chunk. Write to Out."""
    out_base = Out_ptr + pid_bh_q * num_chunks * D + pid_chunk * D
    full_seq_start = chunk_start - L_q
    v_full_base = V_full_ptr + pid_bh_kv * L_f * D

    num_d_blocks: tl.constexpr = D // BLOCK_D
    for d_block in range(num_d_blocks):
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = v_full_base + (full_seq_start + offs_chunk[:, None]) * D + offs_d[None, :]
        v_tile = tl.load(v_ptrs, mask=mask_chunk[:, None], other=0.0)
        v_tile_f32 = v_tile.to(tl.float32)
        partial_out_d = tl.sum(local_weights[:, None] * v_tile_f32, axis=0)
        tl.store(out_base + offs_d, partial_out_d)


@triton.jit
def quant_flash_decode_kernel(
    # Query
    Q_ptr,          # [B*H_q, 1, D] fp16
    # Quantized K (transposed)
    K_quant_ptr,    # [B*H_kv, D, L_q_packed] int32
    K_scale_ptr,    # [B*H_kv, D, L_q_groups] fp16
    K_mn_ptr,       # [B*H_kv, D, L_q_groups] fp16
    # Quantized V
    V_quant_ptr,    # [B*H_kv, L_q, D_packed] int32
    V_scale_ptr,    # [B*H_kv, L_q, D_groups] fp16
    V_mn_ptr,       # [B*H_kv, L_q, D_groups] fp16
    # Full-precision K, V
    K_full_ptr,     # [B*H_kv, L_f, D] fp16
    V_full_ptr,     # [B*H_kv, L_f, D] fp16
    # Outputs (per-chunk partial results)
    Out_ptr,        # [B*H_q, num_chunks, D] fp32
    Lse_ptr,        # [B*H_q, num_chunks] fp32
    # Dimensions
    D: tl.constexpr,
    L_q,            # quantized seq length (unpacked)
    L_f,            # full-precision seq length
    L_q_packed,     # L_q // pack_factor
    L_q_groups,     # L_q // group_size
    D_packed,       # D // pack_factor
    D_groups,       # D // group_size
    num_chunks,     # total number of chunks
    H_q,            # number of query heads
    H_kv,           # number of KV heads
    # Quant parameters
    BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # Tile sizes
    CHUNK_SIZE: tl.constexpr,   # e.g. 128
    BLOCK_D: tl.constexpr,      # e.g. 64, for D-dimension inner loop
):
    PACK_FACTOR: tl.constexpr = 32 // BITS
    MASK: tl.constexpr = (1 << BITS) - 1
    sm_scale: tl.constexpr = 1.0 / (D ** 0.5)

    # --- Program IDs ---
    pid_chunk = tl.program_id(0)
    pid_bh_q = tl.program_id(1)

    # --- GQA: map query head to KV head ---
    num_kv_groups = H_q // H_kv
    batch_idx = pid_bh_q // H_q
    head_q_idx = pid_bh_q % H_q
    head_kv_idx = head_q_idx // num_kv_groups
    pid_bh_kv = batch_idx * H_kv + head_kv_idx

    # --- Determine chunk position and type ---
    L_total = L_q + L_f
    chunk_start = pid_chunk * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, L_total)
    actual_chunk_len = chunk_end - chunk_start

    is_full_chunk = chunk_start >= L_q

    # --- Offsets ---
    offs_chunk = tl.arange(0, CHUNK_SIZE)
    mask_chunk = offs_chunk < actual_chunk_len

    # ================================================================
    # Phase 1: Compute scores = Q @ K_chunk^T / sqrt(D)
    # ================================================================
    if is_full_chunk:
        scores = _qfd_compute_scores_full(
            Q_ptr, K_full_ptr,
            pid_bh_q, pid_bh_kv, chunk_start, mask_chunk, offs_chunk,
            L_q, L_f, D, BLOCK_D, CHUNK_SIZE,
        )
    else:
        scores = _qfd_compute_scores_quant(
            Q_ptr, K_quant_ptr, K_scale_ptr, K_mn_ptr,
            pid_bh_q, pid_bh_kv, chunk_start, mask_chunk, offs_chunk,
            D, BLOCK_D, CHUNK_SIZE,
            L_q_packed, L_q_groups,
            BITS, GROUP_SIZE, PACK_FACTOR, MASK,
        )

    scores = scores * sm_scale

    # ================================================================
    # Phase 2: Online softmax (local)
    # ================================================================
    scores = tl.where(mask_chunk, scores, float('-inf'))
    local_max = tl.max(scores, axis=0)
    exp_scores = tl.exp(scores - local_max)
    exp_scores = tl.where(mask_chunk, exp_scores, 0.0)
    local_sum = tl.sum(exp_scores, axis=0)
    lse = tl.log(local_sum) + local_max
    local_weights = exp_scores / local_sum

    # ================================================================
    # Phase 3: Compute partial_out = local_weights @ V_chunk
    # ================================================================
    if is_full_chunk:
        _qfd_compute_output_full(
            local_weights, V_full_ptr, Out_ptr,
            pid_bh_q, pid_bh_kv, pid_chunk, chunk_start, mask_chunk, offs_chunk,
            num_chunks, L_q, L_f, D, BLOCK_D, CHUNK_SIZE,
        )
    else:
        _qfd_compute_output_quant(
            local_weights, V_quant_ptr, V_scale_ptr, V_mn_ptr, Out_ptr,
            pid_bh_q, pid_bh_kv, pid_chunk, chunk_start, mask_chunk, offs_chunk,
            num_chunks, D, BLOCK_D, CHUNK_SIZE,
            L_q, D_packed, D_groups,
            BITS, GROUP_SIZE, PACK_FACTOR, MASK,
        )

    # ================================================================
    # Phase 4: Write LSE
    # ================================================================
    lse_base = Lse_ptr + pid_bh_q * num_chunks + pid_chunk
    tl.store(lse_base, lse)


# =====================================================================
# Reduce Kernel
# =====================================================================

@triton.jit
def quant_flash_decode_reduce_kernel(
    # Inputs (from main kernel)
    Out_ptr,        # [B*H_q, num_chunks, D] fp32
    Lse_ptr,        # [B*H_q, num_chunks] fp32
    # Final output
    Final_ptr,      # [B*H_q, 1, D] fp16
    # Dimensions
    D: tl.constexpr,
    num_chunks,
    # Tile sizes
    BLOCK_D: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,   # upper bound on num_chunks, must be power of 2
):
    pid_bh_q = tl.program_id(0)

    # --- Step 1: Load all LSE values and find global max ---
    offs_c = tl.arange(0, MAX_CHUNKS)    # [MAX_CHUNKS]
    mask_c = offs_c < num_chunks

    lse_base = Lse_ptr + pid_bh_q * num_chunks
    lse_vals = tl.load(lse_base + offs_c, mask=mask_c, other=float('-inf'))  # [MAX_CHUNKS]

    global_max_lse = tl.max(lse_vals, axis=0)   # scalar

    # --- Step 2: Compute weights for each chunk ---
    # weight_c = exp(lse_c - global_max_lse)
    #          = local_sum_c * exp(local_max_c) / exp(global_max_lse)
    # This is the total (unnormalized) probability mass of chunk c
    weights = tl.exp(lse_vals - global_max_lse)              # [MAX_CHUNKS]
    weights = tl.where(mask_c, weights, 0.0)
    total_weight = tl.sum(weights, axis=0)                    # scalar

    # --- Step 3: Weighted sum of partial outputs ---
    # Load all partial outputs for this (batch, head) as a 2D tile [MAX_CHUNKS, BLOCK_D]
    # and compute weighted sum via vector ops.
    #
    # We loop over D in blocks of BLOCK_D. For each d-block, we load
    # Out[pid_bh_q, 0:MAX_CHUNKS, d_block] as [MAX_CHUNKS, BLOCK_D],
    # multiply by weights [MAX_CHUNKS, 1], sum over chunks, normalize.

    num_d_blocks: tl.constexpr = D // BLOCK_D
    out_bh_base = Out_ptr + pid_bh_q * num_chunks * D
    final_base = Final_ptr + pid_bh_q * D

    for d_block in range(num_d_blocks):
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)   # [BLOCK_D]

        # Load Out[pid_bh_q, offs_c, offs_d] → [MAX_CHUNKS, BLOCK_D]
        out_ptrs = (out_bh_base
                    + offs_c[:, None] * D      # [MAX_CHUNKS, 1]
                    + offs_d[None, :])          # [1, BLOCK_D]
        out_mask = mask_c[:, None]              # [MAX_CHUNKS, 1] broadcast
        partial_outs = tl.load(out_ptrs, mask=out_mask, other=0.0)  # [MAX_CHUNKS, BLOCK_D]

        # Weighted sum: weights [MAX_CHUNKS] * partial_outs [MAX_CHUNKS, BLOCK_D]
        # → sum over chunk dim → [BLOCK_D]
        acc = tl.sum(weights[:, None] * partial_outs, axis=0)  # [BLOCK_D]

        # Normalize and store
        final_d = (acc / total_weight).to(tl.float16)   # [BLOCK_D]
        tl.store(final_base + offs_d, final_d)


# =====================================================================
# Python wrapper
# =====================================================================

def quant_flash_decode(
    q,              # [B, H_q, 1, D] fp16
    k_quant,        # [B, H_kv, D, L_q_packed] int32  (or None)
    k_scale,        # [B, H_kv, D, L_q_groups] fp16   (or None)
    k_mn,           # [B, H_kv, D, L_q_groups] fp16   (or None)
    v_quant,        # [B, H_kv, L_q, D_packed] int32  (or None)
    v_scale,        # [B, H_kv, L_q, D_groups] fp16   (or None)
    v_mn,           # [B, H_kv, L_q, D_groups] fp16   (or None)
    k_full,         # [B, H_kv, L_f, D] fp16
    v_full,         # [B, H_kv, L_f, D] fp16
    group_size=128,
    bits=2,
    chunk_size=128,
):
    """
    Fused attention decoding over quantized + full-precision KV cache.
    
    Returns: [B, H_q, 1, D] fp16
    """
    import torch
    torch.cuda.nvtx.range_push("quant_flash_decode_total")
    
    torch.cuda.nvtx.range_push("qfd_setup_and_reshape")
    B, H_q, _, D = q.shape
    H_kv = k_full.shape[1]
    L_f = k_full.shape[2]
    
    pack_factor = 32 // bits
    
    if k_quant is not None:
        L_q = k_quant.shape[3] * pack_factor   # unpacked quantized seq length
        L_q_packed = k_quant.shape[3]
        L_q_groups = k_scale.shape[3]
    else:
        L_q = 0
        L_q_packed = 0
        L_q_groups = 0
    
    L_total = L_q + L_f
    D_packed = D // pack_factor
    D_groups = D // group_size
    
    num_chunks = triton.cdiv(L_total, chunk_size)
    
    # Flatten batch and head dims for kernel
    BH_q = B * H_q
    BH_kv = B * H_kv
    
    q_flat = q.reshape(BH_q, 1, D).contiguous()
    if k_quant is not None:
        k_quant_flat = k_quant.reshape(BH_kv, D, L_q_packed).contiguous()
        k_scale_flat = k_scale.reshape(BH_kv, D, L_q_groups).contiguous()
        k_mn_flat = k_mn.reshape(BH_kv, D, L_q_groups).contiguous()
        v_quant_flat = v_quant.reshape(BH_kv, L_q, D_packed).contiguous()
        v_scale_flat = v_scale.reshape(BH_kv, L_q, D_groups).contiguous()
        v_mn_flat = v_mn.reshape(BH_kv, L_q, D_groups).contiguous()
    else:
        # Create dummy tensors (won't be accessed since L_q=0)
        k_quant_flat = torch.empty(0, device=q.device, dtype=torch.int32)
        k_scale_flat = torch.empty(0, device=q.device, dtype=torch.float16)
        k_mn_flat = torch.empty(0, device=q.device, dtype=torch.float16)
        v_quant_flat = torch.empty(0, device=q.device, dtype=torch.int32)
        v_scale_flat = torch.empty(0, device=q.device, dtype=torch.float16)
        v_mn_flat = torch.empty(0, device=q.device, dtype=torch.float16)
    
    k_full_flat = k_full.reshape(BH_kv, L_f, D).contiguous()
    v_full_flat = v_full.reshape(BH_kv, L_f, D).contiguous()

    torch.cuda.nvtx.range_pop() # end qfd_setup_and_reshape
    
    torch.cuda.nvtx.range_push("qfd_allocate_buffers")
    # Allocate intermediate buffers
    out_partial = torch.empty(BH_q, num_chunks, D, dtype=torch.float32, device=q.device)
    lse = torch.empty(BH_q, num_chunks, dtype=torch.float32, device=q.device)
    
    # Choose tile sizes
    BLOCK_D = min(64, D)
    assert D % BLOCK_D == 0, f"D ({D}) must be divisible by BLOCK_D ({BLOCK_D})"
    
    torch.cuda.nvtx.range_pop() # end qfd_allocate_buffers
    
    torch.cuda.nvtx.range_push("qfd_main_kernel")
    # Launch main kernel
    grid_main = (num_chunks, BH_q)
    quant_flash_decode_kernel[grid_main](
        q_flat,
        k_quant_flat, k_scale_flat, k_mn_flat,
        v_quant_flat, v_scale_flat, v_mn_flat,
        k_full_flat, v_full_flat,
        out_partial, lse,
        D=D, L_q=L_q, L_f=L_f,
        L_q_packed=L_q_packed, L_q_groups=L_q_groups,
        D_packed=D_packed, D_groups=D_groups,
        num_chunks=num_chunks, H_q=H_q, H_kv=H_kv,
        BITS=bits, GROUP_SIZE=group_size,
        CHUNK_SIZE=chunk_size, BLOCK_D=BLOCK_D,
    )
    
    torch.cuda.nvtx.range_pop() # end qfd_main_kernel
    
    torch.cuda.nvtx.range_push("qfd_reduce_kernel")
    # Launch reduce kernel
    MAX_CHUNKS = triton.next_power_of_2(num_chunks)
    grid_reduce = (BH_q,)
    final_out = torch.empty(BH_q, 1, D, dtype=torch.float16, device=q.device)
    
    quant_flash_decode_reduce_kernel[grid_reduce](
        out_partial, lse,
        final_out,
        D=D, num_chunks=num_chunks,
        BLOCK_D=BLOCK_D, MAX_CHUNKS=MAX_CHUNKS,
    )
    
    torch.cuda.nvtx.range_pop() # end qfd_reduce_kernel
    
    torch.cuda.nvtx.range_push("qfd_final_reshape")
    ret = final_out.reshape(B, H_q, 1, D)
    torch.cuda.nvtx.range_pop() # end qfd_final_reshape
    
    torch.cuda.nvtx.range_pop() # end quant_flash_decode_total
    
    return ret


# =====================================================================
# Reference implementation (PyTorch, no fusion)
# =====================================================================

def reference_attention(q, k_full_all, v_full_all):
    """
    Standard attention. All inputs in FP16.
    q:           [B, H_q, 1, D]
    k_full_all:  [B, H_kv, L_total, D]
    v_full_all:  [B, H_kv, L_total, D]
    Returns:     [B, H_q, 1, D]
    """
    B, H_q, _, D = q.shape
    H_kv = k_full_all.shape[1]
    num_kv_groups = H_q // H_kv
    
    # Expand KV heads for GQA
    if num_kv_groups > 1:
        k_full_all = k_full_all.repeat_interleave(num_kv_groups, dim=1)
        v_full_all = v_full_all.repeat_interleave(num_kv_groups, dim=1)
    
    # scores = Q @ K^T / sqrt(D)
    scores = torch.matmul(q.float(), k_full_all.float().transpose(2, 3)) / math.sqrt(D)
    
    # softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # output
    output = torch.matmul(attn_weights, v_full_all.float())
    return output.to(torch.float16)


def reference_mixed_attention(q, k_quant, k_scale, k_mn, v_quant, v_scale, v_mn,
                               k_full, v_full, group_size=128, bits=2):
    """
    Reference: dequant all quantized KV, concat with full-precision, then standard attention.
    """
    from dequant import dequant_k_chunk, dequant_v_chunk
    
    B, H_q, _, D = q.shape
    H_kv = k_full.shape[1]
    L_f = k_full.shape[2]
    pack_factor = 32 // bits
    BH_kv = B * H_kv
    
    if k_quant is not None:
        L_q = k_quant.shape[3] * pack_factor
        
        # Dequant all K
        k_quant_flat = k_quant.reshape(BH_kv, D, -1).contiguous()
        k_scale_flat = k_scale.reshape(BH_kv, D, -1).contiguous()
        k_mn_flat = k_mn.reshape(BH_kv, D, -1).contiguous()
        k_dequant = dequant_k_chunk(k_quant_flat, k_scale_flat, k_mn_flat,
                                     0, L_q, D, group_size, bits)
        k_dequant = k_dequant.reshape(B, H_kv, L_q, D)
        
        # Dequant all V
        v_quant_flat = v_quant.reshape(BH_kv, L_q, -1).contiguous()
        v_scale_flat = v_scale.reshape(BH_kv, L_q, -1).contiguous()
        v_mn_flat = v_mn.reshape(BH_kv, L_q, -1).contiguous()
        v_dequant = dequant_v_chunk(v_quant_flat, v_scale_flat, v_mn_flat,
                                     0, L_q, D, L_q, group_size, bits)
        v_dequant = v_dequant.reshape(B, H_kv, L_q, D)
        
        # Concat
        k_all = torch.cat([k_dequant.to(torch.float16), k_full], dim=2)
        v_all = torch.cat([v_dequant.to(torch.float16), v_full], dim=2)
    else:
        k_all = k_full
        v_all = v_full
    
    return reference_attention(q, k_all, v_all)


# =====================================================================
# Tests
# =====================================================================

def test_full_precision_only():
    """Test with no quantized region (L_q=0). Should match standard attention exactly."""
    print("\n--- Test: full-precision only (L_q=0) ---")
    torch.manual_seed(42)
    B, H_q, H_kv, D = 1, 4, 4, 128
    L_f = 256
    
    q = torch.randn(B, H_q, 1, D, dtype=torch.float16, device='cuda')
    k_full = torch.randn(B, H_kv, L_f, D, dtype=torch.float16, device='cuda')
    v_full = torch.randn(B, H_kv, L_f, D, dtype=torch.float16, device='cuda')
    
    out_fused = quant_flash_decode(
        q, None, None, None, None, None, None,
        k_full, v_full,
        chunk_size=128,
    )
    
    out_ref = reference_attention(q, k_full, v_full)
    
    max_diff = (out_fused.float() - out_ref.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_fused.float().reshape(-1), out_ref.float().reshape(-1), dim=0).item()
    print(f"  max_diff={max_diff:.6f}, cos_sim={cos_sim:.8f}")
    print(f"  {'PASS' if max_diff < 0.05 else 'FAIL'}")
    return max_diff < 0.05


def test_quant_and_full():
    """Test with both quantized and full-precision regions."""
    print("\n--- Test: quantized + full-precision ---")
    from new_pack import triton_quantize_and_pack_along_last_dim
    
    torch.manual_seed(42)
    B, H_q, H_kv, D = 1, 4, 4, 128
    L_q_tokens = 512   # tokens to quantize
    L_f = 128          # full-precision tokens
    group_size = 128
    bits = 2
    
    # Generate KV data
    k_orig = torch.randn(B, H_kv, L_q_tokens, D, dtype=torch.float16, device='cuda')
    v_orig = torch.randn(B, H_kv, L_q_tokens, D, dtype=torch.float16, device='cuda')
    k_full = torch.randn(B, H_kv, L_f, D, dtype=torch.float16, device='cuda')
    v_full = torch.randn(B, H_kv, L_f, D, dtype=torch.float16, device='cuda')
    q = torch.randn(B, H_q, 1, D, dtype=torch.float16, device='cuda')
    
    # Quantize K (transposed)
    k_trans = k_orig.transpose(2, 3).contiguous()
    k_quant, k_scale, k_mn = triton_quantize_and_pack_along_last_dim(k_trans, group_size, bits)
    
    # Quantize V (no transpose)
    v_quant, v_scale, v_mn = triton_quantize_and_pack_along_last_dim(v_orig.contiguous(), group_size, bits)
    
    # Fused kernel
    out_fused = quant_flash_decode(
        q, k_quant, k_scale, k_mn, v_quant, v_scale, v_mn,
        k_full, v_full,
        group_size=group_size, bits=bits, chunk_size=128,
    )
    
    # Reference
    out_ref = reference_mixed_attention(
        q, k_quant, k_scale, k_mn, v_quant, v_scale, v_mn,
        k_full, v_full, group_size=group_size, bits=bits,
    )
    
    max_diff = (out_fused.float() - out_ref.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_fused.float().reshape(-1), out_ref.float().reshape(-1), dim=0).item()
    print(f"  max_diff={max_diff:.6f}, cos_sim={cos_sim:.8f}")
    print(f"  {'PASS' if max_diff < 0.05 else 'FAIL'}")
    return max_diff < 0.05


def test_gqa():
    """Test with GQA (H_q > H_kv)."""
    print("\n--- Test: GQA (H_q=8, H_kv=2) ---")
    from new_pack import triton_quantize_and_pack_along_last_dim
    
    torch.manual_seed(42)
    B, H_q, H_kv, D = 1, 8, 2, 128
    L_q_tokens = 256
    L_f = 128
    group_size = 128
    bits = 2
    
    k_orig = torch.randn(B, H_kv, L_q_tokens, D, dtype=torch.float16, device='cuda')
    v_orig = torch.randn(B, H_kv, L_q_tokens, D, dtype=torch.float16, device='cuda')
    k_full = torch.randn(B, H_kv, L_f, D, dtype=torch.float16, device='cuda')
    v_full = torch.randn(B, H_kv, L_f, D, dtype=torch.float16, device='cuda')
    q = torch.randn(B, H_q, 1, D, dtype=torch.float16, device='cuda')
    
    k_trans = k_orig.transpose(2, 3).contiguous()
    k_quant, k_scale, k_mn = triton_quantize_and_pack_along_last_dim(k_trans, group_size, bits)
    v_quant, v_scale, v_mn = triton_quantize_and_pack_along_last_dim(v_orig.contiguous(), group_size, bits)
    
    out_fused = quant_flash_decode(
        q, k_quant, k_scale, k_mn, v_quant, v_scale, v_mn,
        k_full, v_full,
        group_size=group_size, bits=bits, chunk_size=128,
    )
    
    out_ref = reference_mixed_attention(
        q, k_quant, k_scale, k_mn, v_quant, v_scale, v_mn,
        k_full, v_full, group_size=group_size, bits=bits,
    )
    
    max_diff = (out_fused.float() - out_ref.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_fused.float().reshape(-1), out_ref.float().reshape(-1), dim=0).item()
    print(f"  max_diff={max_diff:.6f}, cos_sim={cos_sim:.8f}")
    print(f"  {'PASS' if max_diff < 0.05 else 'FAIL'}")
    return max_diff < 0.05


def test_various_seq_lengths():
    """Test with various sequence lengths."""
    print("\n--- Test: various seq lengths ---")
    from new_pack import triton_quantize_and_pack_along_last_dim
    
    torch.manual_seed(42)
    B, H_q, H_kv, D = 1, 4, 4, 128
    group_size = 128
    bits = 2
    
    configs = [
        (128, 128),    # minimal
        (256, 128),
        (512, 256),
        (1024, 128),
        (2048, 256),
    ]
    
    all_pass = True
    for L_q_tokens, L_f in configs:
        k_orig = torch.randn(B, H_kv, L_q_tokens, D, dtype=torch.float16, device='cuda')
        v_orig = torch.randn(B, H_kv, L_q_tokens, D, dtype=torch.float16, device='cuda')
        k_full = torch.randn(B, H_kv, L_f, D, dtype=torch.float16, device='cuda')
        v_full = torch.randn(B, H_kv, L_f, D, dtype=torch.float16, device='cuda')
        q = torch.randn(B, H_q, 1, D, dtype=torch.float16, device='cuda')
        
        k_trans = k_orig.transpose(2, 3).contiguous()
        k_quant, k_scale, k_mn = triton_quantize_and_pack_along_last_dim(k_trans, group_size, bits)
        v_quant, v_scale, v_mn = triton_quantize_and_pack_along_last_dim(v_orig.contiguous(), group_size, bits)
        
        out_fused = quant_flash_decode(
            q, k_quant, k_scale, k_mn, v_quant, v_scale, v_mn,
            k_full, v_full,
            group_size=group_size, bits=bits, chunk_size=128,
        )
        
        out_ref = reference_mixed_attention(
            q, k_quant, k_scale, k_mn, v_quant, v_scale, v_mn,
            k_full, v_full, group_size=group_size, bits=bits,
        )
        
        max_diff = (out_fused.float() - out_ref.float()).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            out_fused.float().reshape(-1), out_ref.float().reshape(-1), dim=0).item()
        passed = max_diff < 0.05
        print(f"  L_q={L_q_tokens}, L_f={L_f}: max_diff={max_diff:.6f}, cos_sim={cos_sim:.8f} "
              f"[{'PASS' if passed else 'FAIL'}]")
        if not passed:
            all_pass = False
    
    return all_pass


if __name__ == "__main__":
    print("=" * 60)
    print("QuantFlashDecoding Kernel Tests")
    print("=" * 60)
    
    results = []
    results.append(("full_precision_only", test_full_precision_only()))
    results.append(("quant_and_full", test_quant_and_full()))
    results.append(("gqa", test_gqa()))
    results.append(("various_seq_lengths", test_various_seq_lengths()))
    
    print("\n" + "=" * 60)
    print("Summary:")
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    all_pass = all(p for _, p in results)
    print(f"\n{'All tests PASSED!' if all_pass else 'Some tests FAILED!'}")