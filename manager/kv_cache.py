"""
QuantKVCache: Pre-allocated KV cache with double-window full-precision region.

Double-window design:
  Full-precision buffer is 2 * residual_length, logically split into:
    - Front window: [0, residual_length)
    - Back  window: [residual_length, 2 * residual_length)

  New tokens always write into the back window.
  When the back window is full:
    1. Quantize the FRONT window (K and V in sync) → write to quant buffers
    2. Copy back window content → front window
    3. Reset back window cursor

  This guarantees:
    - The most recent residual_length ~ 2*residual_length tokens are always FP16
    - K and V quantize at the same time, same tokens
    - No per-token eviction, no torch.cat

Buffer layout:
  k_full / v_full: [B, H, 2 * residual_length, D]
                    |---- front window ----|---- back window ----|

  Quant region (same as before):
    Key:   per-channel, transposed [B, H, D, seq]   packed along seq
    Value: per-token,   standard   [B, H, seq, D]   packed along D

Cursors:
  quant_len:  seq positions in quantized region (unpacked count)
  full_len:   total valid positions in full-precision buffer (0 ~ 2*residual_length)
              front window valid count = min(full_len, residual_length)
              back  window valid count = max(0, full_len - residual_length)
"""

import torch
from typing import Optional, Tuple

from quant.new_pack import triton_quantize_and_pack_along_last_dim


class QuantKVCache:

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        group_size: int = 128,
        k_bits: int = 2,
        v_bits: int = 2,
        residual_length: int = 128,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.group_size = group_size
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.residual_length = residual_length
        self.window_size = residual_length          # one window = residual_length tokens
        self.full_buf_len = 2 * residual_length     # total FP16 buffer = two windows
        self.device = device
        self.dtype = dtype

        self.k_pack = 32 // k_bits   # 2-bit → 16
        self.v_pack = 32 // v_bits

        # Max quantized tokens over the entire generation
        max_quant_len = max_seq_len - residual_length
        assert max_quant_len >= 0, \
            f"max_seq_len ({max_seq_len}) must be >= residual_length ({residual_length})"
        assert residual_length % group_size == 0, \
            f"residual_length ({residual_length}) must be divisible by group_size ({group_size})"
        self.max_quant_len = max_quant_len

        B, H, D = batch_size, n_kv_heads, head_dim

        # ====== Quantized region: Key (per-channel, transposed) ======
        if max_quant_len > 0:
            self.k_quant = torch.zeros(B, H, D, max_quant_len // self.k_pack,
                                       dtype=torch.int32, device=device)
            self.k_scale = torch.zeros(B, H, D, max_quant_len // group_size,
                                       dtype=dtype, device=device)
            self.k_mn = torch.zeros(B, H, D, max_quant_len // group_size,
                                    dtype=dtype, device=device)
        else:
            self.k_quant = self.k_scale = self.k_mn = None

        # ====== Quantized region: Value (per-token, standard) ======
        if max_quant_len > 0:
            self.v_quant = torch.zeros(B, H, max_quant_len, D // self.v_pack,
                                       dtype=torch.int32, device=device)
            self.v_scale = torch.zeros(B, H, max_quant_len, D // group_size,
                                       dtype=dtype, device=device)
            self.v_mn = torch.zeros(B, H, max_quant_len, D // group_size,
                                    dtype=dtype, device=device)
        else:
            self.v_quant = self.v_scale = self.v_mn = None

        # ====== Double-window full-precision buffer ======
        # [B, H, 2 * residual_length, D]
        self.k_full = torch.zeros(B, H, self.full_buf_len, D, dtype=dtype, device=device)
        self.v_full = torch.zeros(B, H, self.full_buf_len, D, dtype=dtype, device=device)

        # ====== Cursors ======
        self.quant_len = 0   # unpacked seq positions in quantized region
        self.full_len = 0    # valid positions in full-precision buffer [0, 2*residual_length]

    @property
    def kv_seq_len(self) -> int:
        """Total KV sequence length = quantized + full-precision."""
        return self.quant_len + self.full_len

    # ------------------------------------------------------------------
    # Phase 1: Prefill — bulk store after flash attention
    # ------------------------------------------------------------------

    def store_prefill(
        self,
        key_states: torch.Tensor,     # [B, H_kv, seq_len, D]
        value_states: torch.Tensor,    # [B, H_kv, seq_len, D]
    ) -> None:
        """Bulk-store KV from prefill.

        Split: tokens that fill complete windows → quantize,
               remaining tail → write into full-precision buffer.

        The tail occupies up to 2*residual_length - 1 tokens.
        If tail > residual_length, front window has residual_length tokens
        and back window has the rest, but no flush yet (back not full).
        """
        seq_len = key_states.shape[2]
        W = self.window_size  # = residual_length

        # How many complete windows can we quantize?
        # We keep at least one window in FP16, so quantize in units of W
        # leaving a tail of seq_len % W (could be 0..W-1) tokens.
        # But we also need to respect: tail should fit in the double buffer.
        #
        # Simple rule: quantize everything except the last (seq_len % W) tokens.
        # If seq_len % W == 0, we still keep the last W tokens in FP16
        # (to always have a non-empty full-precision region for attention).
        if seq_len <= W:
            # Fits entirely in front window, no quantization
            quant_len = 0
            tail_len = seq_len
        elif seq_len % W == 0:
            # Keep last W tokens as FP16 (front window), quantize the rest
            quant_len = seq_len - W
            tail_len = W
        else:
            tail_len = seq_len % W
            quant_len = seq_len - tail_len

        # --- Quantize bulk ---
        if quant_len > 0:
            self._quantize_and_store(
                key_states[:, :, :quant_len, :],
                value_states[:, :, :quant_len, :],
                quant_start=0,
            )
            self.quant_len = quant_len

        # --- Store tail in full-precision buffer ---
        if tail_len > 0:
            self.k_full[:, :, :tail_len, :] = key_states[:, :, quant_len:, :]
            self.v_full[:, :, :tail_len, :] = value_states[:, :, quant_len:, :]
        self.full_len = tail_len

    # ------------------------------------------------------------------
    # Phase 2: Decode — one token at a time
    # ------------------------------------------------------------------

    def update_decode(
        self,
        key_states: torch.Tensor,      # [B, H_kv, 1, D]
        value_states: torch.Tensor,     # [B, H_kv, 1, D]
    ) -> None:
        """Append one token to full-precision buffer.

        Write at position full_len (index assignment, no cat).
        If back window becomes full (full_len == 2 * residual_length):
          1. Quantize front window → write to quant buffers
          2. Copy back window → front window
          3. Reset: full_len = residual_length
        """
        # Write new token
        self.k_full[:, :, self.full_len:self.full_len + 1, :] = key_states
        self.v_full[:, :, self.full_len:self.full_len + 1, :] = value_states
        self.full_len += 1

        # Check if back window is full
        if self.full_len == self.full_buf_len:
            self._flush_front_window()

    def _flush_front_window(self) -> None:
        """Quantize front window, shift back window forward.

        Before:
          k_full: [front: residual_length tokens | back: residual_length tokens]
          full_len = 2 * residual_length

        After:
          front window quantized → appended to quant buffers
          back window → copied to front window position
          full_len = residual_length
        """
        W = self.window_size

        # --- 1. Quantize front window [0:W] ---
        self._quantize_and_store(
            self.k_full[:, :, :W, :],
            self.v_full[:, :, :W, :],
            quant_start=self.quant_len,
        )
        self.quant_len += W

        # --- 2. Copy back window [W:2W] → front window [0:W] ---
        self.k_full[:, :, :W, :] = self.k_full[:, :, W:2 * W, :]
        self.v_full[:, :, :W, :] = self.v_full[:, :, W:2 * W, :]

        # --- 3. Reset cursor ---
        self.full_len = W

    # ------------------------------------------------------------------
    # Shared quantization helper
    # ------------------------------------------------------------------

    def _quantize_and_store(
        self,
        k_data: torch.Tensor,     # [B, H, chunk_len, D]
        v_data: torch.Tensor,     # [B, H, chunk_len, D]
        quant_start: int,          # where to write in quant buffer (unpacked seq index)
    ) -> None:
        """Quantize a chunk of K and V, write into pre-allocated quant buffers.

        Key:   transpose → per-channel quant along seq
        Value: per-token quant along head_dim
        """
        chunk_len = k_data.shape[2]

        # --- Key: transpose → quant ---
        k_trans = k_data.transpose(2, 3).contiguous()  # [B, H, D, chunk_len]
        k_q, k_s, k_m = triton_quantize_and_pack_along_last_dim(
            k_trans, self.group_size, self.k_bits)

        packed_start = quant_start // self.k_pack
        packed_end = packed_start + chunk_len // self.k_pack
        gs_start = quant_start // self.group_size
        gs_end = gs_start + chunk_len // self.group_size

        self.k_quant[:, :, :, packed_start:packed_end] = k_q
        self.k_scale[:, :, :, gs_start:gs_end] = k_s
        self.k_mn[:, :, :, gs_start:gs_end] = k_m

        # --- Value: quant along head_dim ---
        v_cont = v_data.contiguous()
        v_q, v_s, v_m = triton_quantize_and_pack_along_last_dim(
            v_cont, self.group_size, self.v_bits)

        self.v_quant[:, :, quant_start:quant_start + chunk_len, :] = v_q
        self.v_scale[:, :, quant_start:quant_start + chunk_len, :] = v_s
        self.v_mn[:, :, quant_start:quant_start + chunk_len, :] = v_m

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_quant_k(self) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """(k_quant_trans, k_scale_trans, k_mn_trans) sliced to valid range, or all None."""
        if self.quant_len == 0:
            return None, None, None
        pe = self.quant_len // self.k_pack
        ge = self.quant_len // self.group_size
        return (self.k_quant[:, :, :, :pe],
                self.k_scale[:, :, :, :ge],
                self.k_mn[:, :, :, :ge])

    def get_quant_v(self) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """(v_quant, v_scale, v_mn) sliced to valid range, or all None."""
        if self.quant_len == 0:
            return None, None, None
        ql = self.quant_len
        return (self.v_quant[:, :, :ql, :],
                self.v_scale[:, :, :ql, :],
                self.v_mn[:, :, :ql, :])

    def get_full_kv(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """(k_full, v_full) sliced to valid length, or (None, None).

        Returns the entire valid region [0:full_len], which spans
        both front and back windows as a single contiguous slice.
        """
        if self.full_len == 0:
            return None, None
        return (self.k_full[:, :, :self.full_len, :],
                self.v_full[:, :, :self.full_len, :])