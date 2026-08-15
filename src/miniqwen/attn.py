import math

import torch
import triton
import triton.language as tl
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.cuda import nvtx

from .cache import LayerCache
from .norm import RMSNorm
from .rope import RoPE


class GQA(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        layernorm_eps: float,
        rope: RoPE,
        cache: LayerCache,
        dtype: torch.dtype,
    ):
        super().__init__()

        self._head_dim = head_dim
        self._num_attention_heads = num_attention_heads
        self._num_kv_heads = num_kv_heads
        self._num_kv_groups = self._num_attention_heads // self._num_kv_heads
        self._rope = rope

        self._cache = cache

        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * head_dim, bias=False, dtype=dtype
        )
        self.k_proj = nn.Linear(
            hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype
        )
        self.v_proj = nn.Linear(
            hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype
        )
        self.o_proj = nn.Linear(
            num_attention_heads * head_dim, hidden_size, bias=False, dtype=dtype
        )
        self.q_norm = RMSNorm(
            head_dim,
            layernorm_eps,
            dtype=dtype,
        )
        self.k_norm = RMSNorm(
            head_dim,
            layernorm_eps,
            dtype=dtype,
        )

    @nvtx.range("GQA.forward")
    def forward(
        self,
        x: Float[Tensor, "batch seq_len hidden_size"],
        position_ids: Int[Tensor, "batch seq_len"],
    ) -> Float[Tensor, "batch seq_len hidden_size"]:
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self._head_dim)
        # hidden_shape = (batch_size, seq_len, -1, self._head_dim)

        q_states = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        # q_states :: (batch_size, num_attention_heads, seq_len, head_dim)
        k_states = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        # k_states :: (batch_size, num_kv_heads, seq_len, head_dim)
        v_states = self.v_proj(x).view(hidden_shape).transpose(1, 2)
        # v_states :: (batch_size, num_kv_heads, seq_len, head_dim)

        q_states, k_states = self._rope(q_states, k_states, position_ids)

        attn_output = self._attend(q_states, k_states, v_states)
        # attn_output :: (batch_size, seq_len, num_attention_heads, head_dim)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # attn_output :: (batch_size, seq_len, num_attention_heads * head_dim)
        attn_output = self.o_proj(attn_output)
        # attn_output :: (batch_size, seq_len, hidden_size)

        return attn_output

    @nvtx.range("SelfAttention._attend")
    def _attend(
        self,
        q: Float[Tensor, "batch num_attn_heads seq_len head_dim"],
        k: Float[Tensor, "batch num_kv_heads kv_seq_len head_dim"],
        v: Float[Tensor, "batch num_kv_heads kv_seq_len head_dim"],
    ) -> Float[Tensor, "batch seq_len num_attn_heads head_dim"]:
        k, v = self._cache.update(k, v)

        k = self._expand_kv(k)
        v = self._expand_kv(v)
        # k :: (batch_size, num_attention_heads, kv_seq_len, head_dim)
        # v :: (batch_size, num_attention_heads, kv_seq_len, head_dim)

        scale = 1 / math.sqrt(self._head_dim)
        is_causal = q.shape[2] > 1
        attn_output = _flash_gqa(q, k, v, is_causal=is_causal, scale=scale)
        # attn_output :: (batch_size, num_attention_heads, seq_len, head_dim)

        return attn_output.transpose(1, 2).contiguous()

    def _expand_kv(
        self, x: Float[Tensor, "batch num_kv_heads seq_len head_dim"]
    ) -> Float[Tensor, "batch num_attn_heads seq_len head_dim"]:
        batch_size, num_kv_heads, seq_len, head_dim = x.shape

        if self._num_kv_groups <= 1:
            return x

        x = x[:, :, None, :, :].expand(
            batch_size, num_kv_heads, self._num_kv_groups, seq_len, head_dim
        )
        return x.reshape(batch_size, -1, seq_len, head_dim)


def _flash_gqa(
    q: Float[Tensor, "batch num_attn_heads seq_len head_dim"],
    k: Float[Tensor, "batch num_kv_heads kv_seq_len head_dim"],
    v: Float[Tensor, "batch num_kv_heads kv_seq_len head_dim"],
    *,
    is_causal: bool,
    scale: float,
) -> Float[Tensor, "batch num_attn_heads seq_len head_dim"]:
    B = q.shape[0]
    Hq = q.shape[1]
    Sq = q.shape[2]
    D = q.shape[3]
    Hkv = k.shape[1]
    Skv = k.shape[2]

    SQ_TILE_SIZE = 16
    SK_TILE_SIZE = 16

    Sq_tiles = triton.cdiv(Sq, SQ_TILE_SIZE)
    out = torch.empty_like(q)

    # fmt: off
    _flash_gqa_kernel[(B, Hq, Sq_tiles)](
        q, k, v, out,
        B, Hq, Sq, Hkv, Skv, D,  # type: ignore
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale,
        SQ_TILE_SIZE,  # type: ignore
        SK_TILE_SIZE,  # type: ignore
        is_causal,  # type: ignore
    )
    # fmt: on

    return out


# fmt: off
@triton.jit
def _flash_gqa_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    B, Hq, Sq, Hkv, Skv, D: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    scale,
    SQ_TILE_SIZE: tl.constexpr,
    SKV_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # fmt: on
    # launch grid: (B, Hq, Sq_tiles)
    B_idx = tl.program_id(0)
    Hq_idx = tl.program_id(1)
    Sq_tile_idx = tl.program_id(2)

    Hk_idx = Hq_idx // (Hq // Hkv)

    Q_ptr += B_idx * stride_qb + Hq_idx * stride_qh
    K_ptr += B_idx * stride_kb + Hk_idx * stride_kh
    V_ptr += B_idx * stride_vb + Hk_idx * stride_vh
    O_ptr += B_idx * stride_ob + Hq_idx * stride_oh

    offs_Sq = tl.arange(0, SQ_TILE_SIZE) + Sq_tile_idx * SQ_TILE_SIZE
    offs_D = tl.arange(0, D)

    Q_tile_ptrs = Q_ptr + (offs_Sq[:, None] * stride_qs + offs_D[None, :] * stride_qd)
    Q_tile_mask = offs_Sq[:, None] < Sq
    Q_tile = tl.load(Q_tile_ptrs, mask=Q_tile_mask, other=0.0).cast(tl.float32)

    O_tile = tl.zeros((SQ_TILE_SIZE, D), tl.float32)
    rowmax = tl.full((SQ_TILE_SIZE,), -float("inf"), tl.float32)
    expsum = tl.zeros((SQ_TILE_SIZE,), tl.float32)

    for Skv_tile_idx in range(tl.cdiv(Skv, SKV_TILE_SIZE)):
        offs_Skv = tl.arange(0, SKV_TILE_SIZE) + Skv_tile_idx * SKV_TILE_SIZE
        K_tile_ptrs = K_ptr + (offs_Skv[:, None] * stride_ks + offs_D[None, :] * stride_kd)
        K_tile_mask = offs_Skv[:, None] < Skv
        K_tile = tl.load(K_tile_ptrs, mask=K_tile_mask, other=0.0).cast(tl.float32)

        QK_tile = tl.dot(Q_tile, K_tile.T) * scale

        if IS_CAUSAL:
            causal_mask = offs_Sq[:, None] >= offs_Skv[None, :]
            QK_tile = tl.where(causal_mask, QK_tile, -float("inf"))

        V_tile_ptrs = V_ptr + (offs_Skv[:, None] * stride_vs + offs_D[None, :] * stride_vd)
        V_tile = tl.load(V_tile_ptrs, mask=K_tile_mask, other=0.0).cast(tl.float32)

        rowmax_new = tl.maximum(rowmax, QK_tile.max(axis=-1))
        adj = tl.exp(rowmax - rowmax_new)

        p = tl.exp(QK_tile - rowmax_new[:, None])
        O_tile = O_tile * adj[:, None] + tl.dot(p, V_tile)

        rowmax = rowmax_new
        expsum = expsum * adj + tl.exp(QK_tile - rowmax[:, None]).sum(axis=-1)

    O_tile /= expsum[:, None]

    offs_Sq_out = tl.arange(0, SQ_TILE_SIZE) + Sq_tile_idx * SQ_TILE_SIZE
    O_tile_ptrs = O_ptr + (offs_Sq_out[:, None] * stride_os + offs_D[None, :] * stride_od)
    O_tile_mask = offs_Sq_out[:, None] < Sq
    tl.store(O_tile_ptrs, O_tile, mask=O_tile_mask)
