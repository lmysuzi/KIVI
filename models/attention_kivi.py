"""
LlamaFlashAttention_KIVI_Opt: KIVI with pre-allocated QuantKVCache.

Key changes vs original llama_kivi.py:
  - All O(total_seq) torch.cat eliminated in decode
  - QuantKVCache auto-created on first forward (use_cache=True)
  - HF-compatible: uses `past_key_values` parameter name throughout
  - Prefill uses Flash Attention + bulk quantize (same as original)
  - Decode uses quantized attention + index-based cache update

Cache lifecycle:
  1. First call (prefill): past_key_values=None → auto-create List[QuantKVCache]
     → flash attn → store_prefill() → return caches as past_key_values
  2. Subsequent calls (decode): past_key_values=List[QuantKVCache]
     → quantized attn → update_decode() → return same caches (mutated in-place)
  3. HF generate() loop passes past_key_values through automatically.
"""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from quant.new_pack import triton_quantize_and_pack_along_last_dim
from quant.matmul import cuda_bmm_fA_qB_outer
from quant.quant_flash_decode import quant_flash_decode
from manager.kv_cache import QuantKVCache

from transformers.models.llama.configuration_llama import *
from transformers.models.llama.modeling_llama import *
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

_CONFIG_FOR_DOC = "LlamaConfig"


# =====================================================================
# Attention layer
# =====================================================================

class LlamaFlashAttention_KIVI_Opt(nn.Module):

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.k_bits = config.k_bits
        self.v_bits = config.v_bits
        self.group_size = config.group_size
        self.residual_length = config.residual_length

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got hidden_size={self.hidden_size}, num_heads={self.num_heads})"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[QuantKVCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[QuantKVCache]]:
        bsz, q_len, _ = hidden_states.size()

        # --- QKV projection ---
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        # --- RoPE ---
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None and past_key_value.kv_seq_len > 0:
            kv_seq_len += past_key_value.kv_seq_len
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Determine which path: prefill (cache empty) vs decode (cache populated)
        is_prefill = (past_key_value is None) or (past_key_value.kv_seq_len == 0)

        if is_prefill:
            # =============================================================
            # PREFILL: Flash Attention on full-precision KV, then quantize
            # =============================================================
            attn_output = self._prefill_forward(
                query_states, key_states, value_states,
                attention_mask, q_len,
            )

            max_seq_len = getattr(self.config, "max_seq_len",
                          getattr(self.config, "max_position_embeddings", 4096))
            past_key_value = QuantKVCache(
                batch_size=bsz,
                max_seq_len=max_seq_len,
                n_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                group_size=self.group_size,
                k_bits=self.k_bits,
                v_bits=self.v_bits,
                residual_length=self.residual_length,
                device=key_states.device,
                dtype=key_states.dtype,
            )
            past_key_value.store_prefill(key_states, value_states)

        else:
            # =============================================================
            # DECODE: quantized attention, then index-based cache update
            # =============================================================
            past_key_value.update_decode(key_states, value_states)
            #attn_output = self._decode_forward(
            #    query_states, key_states, value_states,
            ##)
            attn_output = self._decode_forward_fused(
                query_states, key_states, value_states,
                past_key_value, bsz,
            )
            

        # --- Output projection ---
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    # ------------------------------------------------------------------
    # Prefill: Flash Attention on full-precision KV
    # ------------------------------------------------------------------

    def _prefill_forward(self, query_states, key_states, value_states,
                         attention_mask, q_len):
        from flash_attn import flash_attn_func, flash_attn_varlen_func

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            target_dtype = getattr(self.config, "_pre_quantization_dtype",
                                   self.q_proj.weight.dtype)
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # flash_attn: [B, seq, n_heads, D]
        q = query_states.transpose(1, 2)
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)

        
        attn_output = self._flash_attention_forward(q, k, v, None, q_len, dropout=0.0)

        # [B, q_len, n_heads, D] → [B, q_len, hidden_size]
        return attn_output.reshape(q.shape[0], q.shape[1], -1)

    # ------------------------------------------------------------------
    # Decode: mixed quantized + full-precision attention
    # ------------------------------------------------------------------

    def _decode_forward(self, query_states, new_key, new_value,
                        cache, attention_mask, bsz, q_len, kv_seq_len):
        """
        Attention over quantized + full-precision KV cache.
        New token appended to full-precision part via bounded cat
        (at most residual_length + 1, NOT O(total_seq)).
        """
        k_q, k_s, k_m = cache.get_quant_k()
        v_q, v_s, v_m = cache.get_quant_v()
        k_full, v_full = cache.get_full_kv()

        
        full_ext_len = k_full.shape[2]

        # --- Q @ K ---
        if k_q is not None:
            att_qk_quant = cuda_bmm_fA_qB_outer(
                self.group_size, query_states, k_q, k_s, k_m, self.k_bits)
        else:
            att_qk_quant = None

        att_qk_full = torch.matmul(
            query_states,
            repeat_kv(k_full, self.num_key_value_groups).transpose(2, 3))

        if att_qk_quant is not None:
            attn_weights = torch.cat([att_qk_quant, att_qk_full], dim=-1) / math.sqrt(self.head_dim)
        else:
            attn_weights = att_qk_full / math.sqrt(self.head_dim)

        # --- Mask + Softmax ---
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be {(bsz, self.num_heads, q_len, kv_seq_len)}, "
                f"got {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be {(bsz, 1, q_len, kv_seq_len)}, "
                    f"got {attention_mask.size()}")
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights,
                                     torch.tensor(torch.finfo(attn_weights.dtype).min))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # --- attn @ V ---
        if v_q is not None:
            attn_output = cuda_bmm_fA_qB_outer(
                self.group_size,
                attn_weights[:, :, :, :-full_ext_len],
                v_q, v_s, v_m, self.v_bits)
            attn_output += torch.matmul(
                attn_weights[:, :, :, -full_ext_len:],
                repeat_kv(v_full, self.num_key_value_groups))
        else:
            attn_output = torch.matmul(
                attn_weights,
                repeat_kv(v_full, self.num_key_value_groups))

        # [B, n_heads, q_len, D] → [B, q_len, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output.reshape(bsz, q_len, -1)
    
        # ------------------------------------------------------------------
    # Decode: fused QuantFlashDecoding (NEW)
    # ------------------------------------------------------------------

    def _decode_forward_fused(self, query_states, new_key, new_value,
                              cache, bsz):
        """
        Fused decode attention using QuantFlashDecoding kernel.
        
        Key change: write-before-read ordering.
        
        Old flow (Phase 1):
          1. read cache → 2. cat new token → 3. compute attention → 4. update cache
          Problem: step 2 allocates temporary tensors
          
        New flow:
          1. update cache (write new token, may trigger flush)
          2. read cache (now includes new token)
          3. compute attention via fused kernel
          No cat, no temporary tensors.
        
        Correctness argument:
          - update_decode writes new_key/new_value at position full_len, increments full_len
          - If flush triggers: front window is quantized, back window moves to front,
            full_len resets to window_size. The flushed tokens now appear in quant buffers.
          - get_quant_k/v and get_full_kv now return the complete state including new token.
          - The fused kernel sees exactly the same data as the old flow's cat'd tensors.
        """
        # --- Step 1: Write new token to cache ---
        ## cache.update_decode(new_key, new_value)

        # --- Step 2: Read cache state (now includes new token) ---
        k_q, k_s, k_m = cache.get_quant_k()    # quantized K slices or (None, None, None)
        v_q, v_s, v_m = cache.get_quant_v()    # quantized V slices or (None, None, None)
        k_full, v_full = cache.get_full_kv()    # full-precision K, V slices

        # --- Step 3: Fused attention ---
        attn_output = quant_flash_decode(
            q=query_states,             # [B, H_q, 1, D]
            k_quant=k_q,               # [B, H_kv, D, L_q_packed] or None
            k_scale=k_s,
            k_mn=k_m,
            v_quant=v_q,               # [B, H_kv, L_q, D_packed] or None
            v_scale=v_s,
            v_mn=v_m,
            k_full=k_full,             # [B, H_kv, L_f, D]
            v_full=v_full,
            group_size=self.group_size,
            bits=self.k_bits,          # assume k_bits == v_bits
            chunk_size=max(self.group_size, 32),  # must be >= group_size
        )

        # attn_output: [B, H_q, 1, D] → transpose to [B, 1, H_q, D] for reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        return attn_output
    
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        from flash_attn import flash_attn_func, flash_attn_varlen_func

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=self.is_causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal
            )

        return attn_output


    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )



