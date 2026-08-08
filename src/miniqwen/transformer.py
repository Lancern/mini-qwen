import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.cuda import nvtx

from .cache import Cache, LayerCache
from .rope import RoPE


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        layernorm_eps: float,
        rope: RoPE,
        cache: LayerCache | None = None,
    ):
        super().__init__()

        self._head_dim = head_dim
        self._num_attention_heads = num_attention_heads
        self._num_kv_heads = num_kv_heads
        self._num_kv_groups = self._num_attention_heads // self._num_kv_heads
        self._rope = rope

        self._cache = cache

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(
            head_dim,
            layernorm_eps,
        )
        self.k_norm = RMSNorm(
            head_dim,
            layernorm_eps,
        )

    @nvtx.range("SelfAttention.forward")
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
        k = self._expand_kv(k)
        v = self._expand_kv(v)
        # k :: (batch_size, num_attention_heads, kv_seq_len, head_dim)
        # v :: (batch_size, num_attention_heads, kv_seq_len, head_dim)

        if self._cache is not None:
            k, v = self._cache.update_and_concat(k, v)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        scaling = self._head_dim**-0.5
        is_causal = q.shape[2] > 1
        attn_output = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=is_causal, scale=scaling
        )
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


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    @nvtx.range("MLP.forward")
    def forward(
        self, x: Float[Tensor, "batch seq_len hidden_size"]
    ) -> Float[Tensor, "batch seq_len hidden_size"]:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()

        self._hidden_size = hidden_size
        self._eps = eps

        self.weight = nn.Parameter(torch.ones(hidden_size))

    @nvtx.range("RMSNorm.forward")
    def forward(
        self, x: Float[Tensor, "... hidden_size"]
    ) -> Float[Tensor, "... hidden_size"]:
        input_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self._eps)
        return self.weight * x.to(input_dtype)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        layernorm_eps: float,
        rope: RoPE,
        cache: Cache | None = None,
    ):
        super().__init__()

        self._layer_idx = layer_idx
        self._hidden_size = hidden_size

        if cache is not None:
            self._cache = cache[layer_idx]
        else:
            self._cache = None

        self.self_attn = SelfAttention(
            hidden_size,
            head_dim,
            num_attention_heads,
            num_kv_heads,
            layernorm_eps,
            rope=rope,
            cache=self._cache,
        )
        self.mlp = MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(
            hidden_size,
            layernorm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            layernorm_eps,
        )

    @nvtx.range("DecoderLayer.forward")
    def forward(
        self,
        x: Float[Tensor, "batch seq_len hidden_size"],
        position_ids: Int[Tensor, "batch seq_len"],
    ) -> Float[Tensor, "batch seq_len hidden_size"]:
        residual = x

        x = self.input_layernorm(x)
        # x :: (batch_size, seq_len, hidden_size)

        x = self.self_attn(x, position_ids)
        x = x + residual
        # x :: (batch_size, seq_len, hidden_size)

        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        x = x + residual
        # x :: (batch_size, seq_len, hidden_size)

        return x
