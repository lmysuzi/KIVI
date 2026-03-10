"""
QuantFlashDecoding v5b: Large chunk_size with 3-way branching.

Strategy:
  - Keep the original 4 helper functions (scores_quant, scores_full,
    output_quant, output_full) UNCHANGED for pure quant/full chunks.
  - Add 2 NEW mixed helpers for the rare boundary chunk.
  - Main kernel: 3-way branch (pure quant / mixed / pure full).

Why not unified helpers (v5 approach):
  v5's unified helpers did both quant and full loads in EVERY d_block iteration,
  even when one side was fully masked out. This caused:
    - 2x register pressure (two full sets of address computation + loads)
    - Triton compiler couldn't optimize away masked loads
    - Result: 21 tok/s (chunk=128) and 16 tok/s (chunk=256) vs 23.88 baseline

v5b approach:
  - Pure quant chunks (majority): exact same code as v4, zero overhead
  - Pure full chunks: exact same code as v4, zero overhead  
  - Mixed chunk (at most 1 per sequence): pays the dual-load cost,
    but only 1 out of N chunks, negligible impact

Mixed chunk handling:
  The mixed helpers split the CHUNK_SIZE positions into quant_part and full_part
  using per-position masks (is_quant / is_full). Within the d_block loop,
  both quant and full loads are issued, merged via tl.where.
  This is expensive per-block, but only 1 block out of num_chunks pays it.

Changes from v4:
  Kernel params: UNCHANGED
  Helpers: 4 original (unchanged) + 2 new mixed helpers
  Main kernel: 2-way if/else → 3-way if/elif/else
  Wrapper: removed L_q % chunk_size constraint
"""

import math
import torch
import triton
import triton.language as tl


# =====================================================================
# Original helpers (UNCHANGED from v4)
# =====================================================================

@triton.jit
def _qfd_compute_scores_quant(
    Q_ptr, K_quant_ptr, K_scale_ptr, K_mn_ptr,
    pid_bh_q, pid_bh_kv, chunk_start, mask_chunk, offs_chunk,
    D: tl.constexpr, BLOCK_D: tl.constexpr, CHUNK_SIZE: tl.constexpr,
    stride_kq_last,
    stride_ks_last,
    BITS: tl.constexpr, GROUP_SIZE: tl.constexpr,
    PACK_FACTOR: tl.constexpr, MASK: tl.constexpr,
):
    """Compute scores = Q @ K_quant_chunk^T. UNCHANGED from v4."""
    scores = tl.zeros((CHUNK_SIZE,), dtype=tl.float32)
    q_base = Q_ptr + pid_bh_q * D
    kq_base = K_quant_ptr + pid_bh_kv * D * stride_kq_last
    ks_base = K_scale_ptr + pid_bh_kv * D * stride_ks_last
    km_base = K_mn_ptr + pid_bh_kv * D * stride_ks_last

    global_seq = chunk_start + offs_chunk
    pack_idx = global_seq // PACK_FACTOR
    bit_offset = (global_seq % PACK_FACTOR) * BITS
    group_idx = global_seq // GROUP_SIZE

    num_d_blocks: tl.constexpr = D // BLOCK_D
    for d_block in range(num_d_blocks):
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        q_tile = tl.load(q_base + offs_d).to(tl.float32)

        kq_ptrs = kq_base + offs_d[:, None] * stride_kq_last + pack_idx[None, :]
        packed_vals = tl.load(kq_ptrs, mask=mask_chunk[None, :], other=0)
        quant_vals = (packed_vals >> bit_offset[None, :]) & MASK

        ks_ptrs = ks_base + offs_d[:, None] * stride_ks_last + group_idx[None, :]
        km_ptrs = km_base + offs_d[:, None] * stride_ks_last + group_idx[None, :]
        scale = tl.load(ks_ptrs, mask=mask_chunk[None, :], other=0.0)
        mn = tl.load(km_ptrs, mask=mask_chunk[None, :], other=0.0)

        k_dequant = quant_vals.to(tl.float32) * scale.to(tl.float32) + mn.to(tl.float32)
        k_tile_f32 = tl.trans(k_dequant)
        scores += tl.sum(k_tile_f32 * q_tile[None, :], axis=1)

    return scores


@triton.jit
def _qfd_compute_scores_full(
    Q_ptr, K_full_ptr,
    pid_bh_q, pid_bh_kv, chunk_start, mask_chunk, offs_chunk,
    L_q,
    stride_kf_seq,
    D: tl.constexpr, BLOCK_D: tl.constexpr, CHUNK_SIZE: tl.constexpr,
):
    """Compute scores = Q @ K_full_chunk^T. UNCHANGED from v4."""
    scores = tl.zeros((CHUNK_SIZE,), dtype=tl.float32)
    q_base = Q_ptr + pid_bh_q * D
    full_seq_start = chunk_start - L_q
    k_full_base = K_full_ptr + pid_bh_kv * stride_kf_seq * D

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
    stride_vq_seq,
    stride_vs_seq,
    D_packed, D_groups,
    BITS: tl.constexpr, GROUP_SIZE: tl.constexpr,
    PACK_FACTOR: tl.constexpr, MASK: tl.constexpr,
):
    """Compute partial_out = local_weights @ V_quant_chunk. UNCHANGED from v4."""
    out_base = Out_ptr + pid_bh_q * num_chunks * D + pid_chunk * D
    vq_base = V_quant_ptr + pid_bh_kv * stride_vq_seq * D_packed
    vs_base = V_scale_ptr + pid_bh_kv * stride_vs_seq * D_groups
    vm_base = V_mn_ptr + pid_bh_kv * stride_vs_seq * D_groups
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
    num_chunks, L_q,
    stride_vf_seq,
    D: tl.constexpr, BLOCK_D: tl.constexpr, CHUNK_SIZE: tl.constexpr,
):
    """Compute partial_out = local_weights @ V_full_chunk. UNCHANGED from v4."""
    out_base = Out_ptr + pid_bh_q * num_chunks * D + pid_chunk * D
    full_seq_start = chunk_start - L_q
    v_full_base = V_full_ptr + pid_bh_kv * stride_vf_seq * D

    num_d_blocks: tl.constexpr = D // BLOCK_D
    for d_block in range(num_d_blocks):
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        v_ptrs = v_full_base + (full_seq_start + offs_chunk[:, None]) * D + offs_d[None, :]
        v_tile = tl.load(v_ptrs, mask=mask_chunk[:, None], other=0.0)
        v_tile_f32 = v_tile.to(tl.float32)
        partial_out_d = tl.sum(local_weights[:, None] * v_tile_f32, axis=0)
        tl.store(out_base + offs_d, partial_out_d)


# =====================================================================
# NEW: Mixed chunk helpers (quant/full boundary within one chunk)
# =====================================================================

@triton.jit
def _qfd_compute_scores_mixed(
    Q_ptr,
    K_quant_ptr, K_scale_ptr, K_mn_ptr,
    K_full_ptr,
    pid_bh_q, pid_bh_kv,
    chunk_start,
    mask_chunk,         # [CHUNK_SIZE] bool: valid positions
    offs_chunk,         # [CHUNK_SIZE] = tl.arange(0, CHUNK_SIZE)
    is_quant,           # [CHUNK_SIZE] bool: True if in quant region
    mask_quant,         # [CHUNK_SIZE] = is_quant & mask_chunk
    mask_full,          # [CHUNK_SIZE] = (~is_quant) & mask_chunk
    # Dimensions
    L_q,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    # Buffer strides
    stride_kq_last,
    stride_ks_last,
    stride_kf_seq,
    # Quant params
    BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PACK_FACTOR: tl.constexpr,
    QK_MASK: tl.constexpr,
):
    """Compute scores for a mixed chunk spanning quant/full boundary."""
    scores = tl.zeros((CHUNK_SIZE,), dtype=tl.float32)
    q_base = Q_ptr + pid_bh_q * D
    global_seq = chunk_start + offs_chunk                       # [CHUNK_SIZE]

    # --- Quant-side indices (clamped for out-of-range positions) ---
    quant_seq_clamped = tl.minimum(global_seq, L_q - 1)
    pack_idx = quant_seq_clamped // PACK_FACTOR
    bit_offset = (quant_seq_clamped % PACK_FACTOR) * BITS
    group_idx = quant_seq_clamped // GROUP_SIZE

    # --- Full-side indices (clamped) ---
    full_seq_clamped = tl.maximum(global_seq - L_q, 0)

    # --- Bases ---
    kq_base = K_quant_ptr + pid_bh_kv * D * stride_kq_last
    ks_base = K_scale_ptr + pid_bh_kv * D * stride_ks_last
    km_base = K_mn_ptr + pid_bh_kv * D * stride_ks_last
    k_full_base = K_full_ptr + pid_bh_kv * stride_kf_seq * D

    num_d_blocks: tl.constexpr = D // BLOCK_D
    for d_block in range(num_d_blocks):
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        q_tile = tl.load(q_base + offs_d).to(tl.float32)       # [BLOCK_D]

        # ---- Quant path → [BLOCK_D, CHUNK_SIZE] → trans → [CHUNK_SIZE, BLOCK_D] ----
        kq_ptrs = kq_base + offs_d[:, None] * stride_kq_last + pack_idx[None, :]
        packed_vals = tl.load(kq_ptrs, mask=mask_quant[None, :], other=0)
        quant_vals = (packed_vals >> bit_offset[None, :]) & QK_MASK

        ks_ptrs = ks_base + offs_d[:, None] * stride_ks_last + group_idx[None, :]
        km_ptrs = km_base + offs_d[:, None] * stride_ks_last + group_idx[None, :]
        scale = tl.load(ks_ptrs, mask=mask_quant[None, :], other=0.0)
        mn = tl.load(km_ptrs, mask=mask_quant[None, :], other=0.0)

        k_quant_f32 = quant_vals.to(tl.float32) * scale.to(tl.float32) + mn.to(tl.float32)
        k_quant_tile = tl.trans(k_quant_f32)                    # [CHUNK_SIZE, BLOCK_D]

        # ---- Full path → [CHUNK_SIZE, BLOCK_D] ----
        k_full_ptrs = k_full_base + full_seq_clamped[:, None] * D + offs_d[None, :]
        k_full_tile = tl.load(k_full_ptrs, mask=mask_full[:, None], other=0.0)
        k_full_f32 = k_full_tile.to(tl.float32)

        # ---- Merge ----
        k_tile = tl.where(is_quant[:, None], k_quant_tile, k_full_f32)

        # ---- Accumulate ----
        scores += tl.sum(k_tile * q_tile[None, :], axis=1)

    return scores


@triton.jit
def _qfd_compute_output_mixed(
    local_weights,
    V_quant_ptr, V_scale_ptr, V_mn_ptr,
    V_full_ptr,
    Out_ptr,
    pid_bh_q, pid_bh_kv, pid_chunk,
    chunk_start,
    mask_chunk,
    offs_chunk,
    is_quant,
    mask_quant,
    mask_full,
    num_chunks,
    L_q,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    # V buffer strides
    stride_vq_seq,
    stride_vs_seq,
    stride_vf_seq,
    D_packed, D_groups,
    # Quant params
    BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PACK_FACTOR: tl.constexpr,
    QV_MASK: tl.constexpr,
):
    """Compute partial_out for a mixed chunk spanning quant/full boundary."""
    out_base = Out_ptr + pid_bh_q * num_chunks * D + pid_chunk * D
    global_seq = chunk_start + offs_chunk

    # --- Quant-side indices (clamped) ---
    quant_seq_clamped = tl.minimum(global_seq, L_q - 1)

    # --- Full-side indices (clamped) ---
    full_seq_clamped = tl.maximum(global_seq - L_q, 0)

    # --- V bases ---
    vq_base = V_quant_ptr + pid_bh_kv * stride_vq_seq * D_packed
    vs_base = V_scale_ptr + pid_bh_kv * stride_vs_seq * D_groups
    vm_base = V_mn_ptr + pid_bh_kv * stride_vs_seq * D_groups
    v_full_base = V_full_ptr + pid_bh_kv * stride_vf_seq * D

    num_d_blocks: tl.constexpr = D // BLOCK_D
    for d_block in range(num_d_blocks):
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)

        # ---- V quant path → [CHUNK_SIZE, BLOCK_D] ----
        d_pack_idx = offs_d // PACK_FACTOR
        d_bit_offset = (offs_d % PACK_FACTOR) * BITS
        d_group_idx = offs_d // GROUP_SIZE

        vq_ptrs = vq_base + quant_seq_clamped[:, None] * D_packed + d_pack_idx[None, :]
        packed_vals = tl.load(vq_ptrs, mask=mask_quant[:, None], other=0)
        quant_vals = (packed_vals >> d_bit_offset[None, :]) & QV_MASK

        vs_ptrs = vs_base + quant_seq_clamped[:, None] * D_groups + d_group_idx[None, :]
        vm_ptrs = vm_base + quant_seq_clamped[:, None] * D_groups + d_group_idx[None, :]
        v_scale_val = tl.load(vs_ptrs, mask=mask_quant[:, None], other=0.0)
        v_mn_val = tl.load(vm_ptrs, mask=mask_quant[:, None], other=0.0)

        v_quant_f32 = quant_vals.to(tl.float32) * v_scale_val.to(tl.float32) + v_mn_val.to(tl.float32)

        # ---- V full path → [CHUNK_SIZE, BLOCK_D] ----
        v_full_ptrs = v_full_base + full_seq_clamped[:, None] * D + offs_d[None, :]
        v_full_tile = tl.load(v_full_ptrs, mask=mask_full[:, None], other=0.0)
        v_full_f32 = v_full_tile.to(tl.float32)

        # ---- Merge ----
        v_tile = tl.where(is_quant[:, None], v_quant_f32, v_full_f32)

        # ---- Weighted sum ----
        partial_out_d = tl.sum(local_weights[:, None] * v_tile, axis=0)
        tl.store(out_base + offs_d, partial_out_d)


# =====================================================================
# Main Kernel
# =====================================================================

@triton.jit
def quant_flash_decode_kernel(
    Q_ptr,
    K_quant_ptr, K_scale_ptr, K_mn_ptr,
    V_quant_ptr, V_scale_ptr, V_mn_ptr,
    K_full_ptr, V_full_ptr,
    Out_ptr, Lse_ptr,
    # Dimensions
    D: tl.constexpr,
    L_q,
    L_f,
    stride_kq_last,
    stride_ks_last,
    D_packed,
    D_groups,
    stride_vq_seq,
    stride_vs_seq,
    stride_kf_seq,
    stride_vf_seq,
    num_chunks,
    H_q, H_kv,
    # Quant parameters
    BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # Tile sizes
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    PACK_FACTOR: tl.constexpr = 32 // BITS
    MASK: tl.constexpr = (1 << BITS) - 1
    sm_scale: tl.constexpr = 1.0 / (D ** 0.5)

    # --- Program IDs (2D grid, unchanged) ---
    pid_chunk = tl.program_id(0)
    pid_bh_q = tl.program_id(1)

    # --- GQA ---
    num_kv_groups = H_q // H_kv
    batch_idx = pid_bh_q // H_q
    head_q_idx = pid_bh_q % H_q
    head_kv_idx = head_q_idx // num_kv_groups
    pid_bh_kv = batch_idx * H_kv + head_kv_idx

    # --- Chunk position ---
    L_total = L_q + L_f
    chunk_start = pid_chunk * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, L_total)
    actual_chunk_len = chunk_end - chunk_start
    offs_chunk = tl.arange(0, CHUNK_SIZE)
    mask_chunk = offs_chunk < actual_chunk_len

    # --- Determine chunk type ---
    # Pure quant:  chunk_end <= L_q  (entire chunk within quant region)
    # Pure full:   chunk_start >= L_q (entire chunk within full region)
    # Mixed:       chunk_start < L_q and chunk_end > L_q
    is_pure_full = chunk_start >= L_q
    is_pure_quant = chunk_end <= L_q  # note: chunk_end = min(chunk_start+CS, L_total)

    # ================================================================
    # Phase 1: Compute scores — 3-way branch
    # ================================================================
    if is_pure_full:
        scores = _qfd_compute_scores_full(
            Q_ptr, K_full_ptr,
            pid_bh_q, pid_bh_kv, chunk_start, mask_chunk, offs_chunk,
            L_q, stride_kf_seq,
            D, BLOCK_D, CHUNK_SIZE,
        )
    elif is_pure_quant:
        scores = _qfd_compute_scores_quant(
            Q_ptr, K_quant_ptr, K_scale_ptr, K_mn_ptr,
            pid_bh_q, pid_bh_kv, chunk_start, mask_chunk, offs_chunk,
            D, BLOCK_D, CHUNK_SIZE,
            stride_kq_last, stride_ks_last,
            BITS, GROUP_SIZE, PACK_FACTOR, MASK,
        )
    else:
        # Mixed chunk: straddles the quant/full boundary
        global_seq = chunk_start + offs_chunk
        is_quant = global_seq < L_q
        mask_quant = is_quant & mask_chunk
        mask_full = (~is_quant) & mask_chunk
        scores = _qfd_compute_scores_mixed(
            Q_ptr,
            K_quant_ptr, K_scale_ptr, K_mn_ptr,
            K_full_ptr,
            pid_bh_q, pid_bh_kv,
            chunk_start, mask_chunk, offs_chunk,
            is_quant, mask_quant, mask_full,
            L_q, D, BLOCK_D, CHUNK_SIZE,
            stride_kq_last, stride_ks_last, stride_kf_seq,
            BITS, GROUP_SIZE, PACK_FACTOR, MASK,
        )

    scores = scores * sm_scale

    # ================================================================
    # Phase 2: Online softmax (unchanged)
    # ================================================================
    scores = tl.where(mask_chunk, scores, float('-inf'))
    local_max = tl.max(scores, axis=0)
    exp_scores = tl.exp(scores - local_max)
    exp_scores = tl.where(mask_chunk, exp_scores, 0.0)
    local_sum = tl.sum(exp_scores, axis=0)
    lse = tl.log(local_sum) + local_max
    local_weights = exp_scores / local_sum

    # ================================================================
    # Phase 3: Compute output — 3-way branch
    # ================================================================
    if is_pure_full:
        _qfd_compute_output_full(
            local_weights, V_full_ptr, Out_ptr,
            pid_bh_q, pid_bh_kv, pid_chunk, chunk_start, mask_chunk, offs_chunk,
            num_chunks, L_q, stride_vf_seq,
            D, BLOCK_D, CHUNK_SIZE,
        )
    elif is_pure_quant:
        _qfd_compute_output_quant(
            local_weights, V_quant_ptr, V_scale_ptr, V_mn_ptr, Out_ptr,
            pid_bh_q, pid_bh_kv, pid_chunk, chunk_start, mask_chunk, offs_chunk,
            num_chunks,
            D, BLOCK_D, CHUNK_SIZE,
            stride_vq_seq, stride_vs_seq,
            D_packed, D_groups,
            BITS, GROUP_SIZE, PACK_FACTOR, MASK,
        )
    else:
        # Mixed chunk
        global_seq2 = chunk_start + offs_chunk
        is_quant2 = global_seq2 < L_q
        mask_quant2 = is_quant2 & mask_chunk
        mask_full2 = (~is_quant2) & mask_chunk
        _qfd_compute_output_mixed(
            local_weights,
            V_quant_ptr, V_scale_ptr, V_mn_ptr,
            V_full_ptr,
            Out_ptr,
            pid_bh_q, pid_bh_kv, pid_chunk,
            chunk_start, mask_chunk, offs_chunk,
            is_quant2, mask_quant2, mask_full2,
            num_chunks, L_q,
            D, BLOCK_D, CHUNK_SIZE,
            stride_vq_seq, stride_vs_seq, stride_vf_seq,
            D_packed, D_groups,
            BITS, GROUP_SIZE, PACK_FACTOR, MASK,
        )

    # ================================================================
    # Phase 4: Write LSE (unchanged)
    # ================================================================
    lse_base = Lse_ptr + pid_bh_q * num_chunks + pid_chunk
    tl.store(lse_base, lse)


# =====================================================================
# Reduce Kernel (unchanged)
# =====================================================================

@triton.jit
def quant_flash_decode_reduce_kernel(
    Out_ptr, Lse_ptr, Final_ptr,
    D: tl.constexpr,
    num_chunks,
    BLOCK_D: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
):
    pid_bh_q = tl.program_id(0)

    offs_c = tl.arange(0, MAX_CHUNKS)
    mask_c = offs_c < num_chunks

    lse_base = Lse_ptr + pid_bh_q * num_chunks
    lse_vals = tl.load(lse_base + offs_c, mask=mask_c, other=float('-inf'))
    global_max_lse = tl.max(lse_vals, axis=0)

    weights = tl.exp(lse_vals - global_max_lse)
    weights = tl.where(mask_c, weights, 0.0)
    total_weight = tl.sum(weights, axis=0)

    num_d_blocks: tl.constexpr = D // BLOCK_D
    out_bh_base = Out_ptr + pid_bh_q * num_chunks * D
    final_base = Final_ptr + pid_bh_q * D

    for d_block in range(num_d_blocks):
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        out_ptrs = out_bh_base + offs_c[:, None] * D + offs_d[None, :]
        partial_outs = tl.load(out_ptrs, mask=mask_c[:, None], other=0.0)
        acc = tl.sum(weights[:, None] * partial_outs, axis=0)
        final_d = (acc / total_weight).to(tl.float16)
        tl.store(final_base + offs_d, final_d)


# =====================================================================
# Python wrapper
# =====================================================================

def quant_flash_decode(
    q,              # [B, H_q, 1, D] fp16
    # --- 3D contiguous buffers from QuantKVCache.get_for_kernel() ---
    k_quant,        # [BH_kv, D, max_packed] int32, or None
    k_scale,        # [BH_kv, D, max_groups] fp16, or None
    k_mn,           # [BH_kv, D, max_groups] fp16, or None
    v_quant,        # [BH_kv, max_quant_len, D_packed] int32, or None
    v_scale,        # [BH_kv, max_quant_len, D_groups] fp16, or None
    v_mn,           # [BH_kv, max_quant_len, D_groups] fp16, or None
    k_full,         # [BH_kv, full_buf_len, D] fp16
    v_full,         # [BH_kv, full_buf_len, D] fp16
    # --- Valid lengths ---
    L_q,            # valid quantized seq length (unpacked)
    L_f,            # valid full-precision seq length
    # --- Config ---
    H_kv,
    group_size=32,
    bits=2,
    chunk_size=128,
):
    """
    Fused attention decoding over quantized + full-precision KV cache.
    
    v5b: Large chunk_size with 3-way branching.
    Pure quant/full chunks use original v4 helpers (zero overhead).
    Mixed boundary chunk uses dedicated mixed helpers.
    No alignment constraint on L_q vs chunk_size.
    
    Returns: [B, H_q, 1, D] fp16
    """
    torch.cuda.nvtx.range_push("quant_flash_decode_total")

    torch.cuda.nvtx.range_push("qfd_setup")
    B, H_q, _, D = q.shape
    pack_factor = 32 // bits

    L_total = L_q + L_f

    # chunk_size must be power of 2 and >= group_size
    chunk_size = triton.next_power_of_2(max(chunk_size, group_size))
    # v5b: No L_q % chunk_size constraint!

    num_chunks = triton.cdiv(L_total, chunk_size)

    # Buffer dim sizes as strides
    D_packed = D // pack_factor
    D_groups = D // group_size

    if k_quant is not None and L_q > 0:
        stride_kq_last = k_quant.shape[2]
        stride_ks_last = k_scale.shape[2]
        stride_vq_seq = v_quant.shape[1]
        stride_vs_seq = v_scale.shape[1]
    else:
        stride_kq_last = 0
        stride_ks_last = 0
        stride_vq_seq = 0
        stride_vs_seq = 0

    stride_kf_seq = k_full.shape[1]
    stride_vf_seq = v_full.shape[1]

    BH_q = B * H_q
    q_flat = q.view(BH_q, 1, D)

    torch.cuda.nvtx.range_pop()  # end qfd_setup

    torch.cuda.nvtx.range_push("qfd_allocate_buffers")
    out_partial = torch.empty(BH_q, num_chunks, D, dtype=torch.float32, device=q.device)
    lse = torch.empty(BH_q, num_chunks, dtype=torch.float32, device=q.device)

    BLOCK_D = min(64, D)
    assert D % BLOCK_D == 0

    _dummy_i32 = torch.empty(0, device=q.device, dtype=torch.int32)
    _dummy_f16 = torch.empty(0, device=q.device, dtype=torch.float16)
    torch.cuda.nvtx.range_pop()  # end qfd_allocate_buffers

    torch.cuda.nvtx.range_push("qfd_main_kernel")
    grid_main = (num_chunks, BH_q)
    quant_flash_decode_kernel[grid_main](
        q_flat,
        k_quant if k_quant is not None else _dummy_i32,
        k_scale if k_scale is not None else _dummy_f16,
        k_mn if k_mn is not None else _dummy_f16,
        v_quant if v_quant is not None else _dummy_i32,
        v_scale if v_scale is not None else _dummy_f16,
        v_mn if v_mn is not None else _dummy_f16,
        k_full, v_full,
        out_partial, lse,
        D=D, L_q=L_q, L_f=L_f,
        stride_kq_last=stride_kq_last,
        stride_ks_last=stride_ks_last,
        D_packed=D_packed,
        D_groups=D_groups,
        stride_vq_seq=stride_vq_seq,
        stride_vs_seq=stride_vs_seq,
        stride_kf_seq=stride_kf_seq,
        stride_vf_seq=stride_vf_seq,
        num_chunks=num_chunks, H_q=H_q, H_kv=H_kv,
        BITS=bits, GROUP_SIZE=group_size,
        CHUNK_SIZE=chunk_size, BLOCK_D=BLOCK_D,
    )
    torch.cuda.nvtx.range_pop()  # end qfd_main_kernel

    torch.cuda.nvtx.range_push("qfd_reduce_kernel")
    MAX_CHUNKS = triton.next_power_of_2(num_chunks)
    grid_reduce = (BH_q,)
    final_out = torch.empty(BH_q, 1, D, dtype=torch.float16, device=q.device)

    quant_flash_decode_reduce_kernel[grid_reduce](
        out_partial, lse, final_out,
        D=D, num_chunks=num_chunks,
        BLOCK_D=BLOCK_D, MAX_CHUNKS=MAX_CHUNKS,
    )
    torch.cuda.nvtx.range_pop()  # end qfd_reduce_kernel

    torch.cuda.nvtx.range_pop()  # end quant_flash_decode_total

    return final_out.reshape(B, H_q, 1, D)