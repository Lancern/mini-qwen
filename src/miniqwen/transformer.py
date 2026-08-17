import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.cuda import nvtx

from .attn import GQA
from .cache import Cache
from .rope import RoPE


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=dtype, device=device
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=dtype, device=device
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False, dtype=dtype, device=device
        )
        self.act_fn = nn.SiLU()

    @nvtx.range("MLP.forward")
    def forward(
        self, x: Float[Tensor, "1 seq_len hidden_size"]
    ) -> Float[Tensor, "1 seq_len hidden_size"]:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


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
        cache: Cache,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()

        self._layer_idx = layer_idx
        self._hidden_size = hidden_size

        self._cache = cache[layer_idx]

        self.self_attn = GQA(
            hidden_size,
            head_dim,
            num_attention_heads,
            num_kv_heads,
            layernorm_eps,
            rope=rope,
            cache=self._cache,
            dtype=dtype,
            device=device,
        )
        self.mlp = MLP(hidden_size, intermediate_size, dtype=dtype, device=device)
        self.input_layernorm = nn.RMSNorm(
            hidden_size,
            layernorm_eps,
            dtype=dtype,
            device=device,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            hidden_size,
            layernorm_eps,
            dtype=dtype,
            device=device,
        )

    @nvtx.range("DecoderLayer.forward")
    def forward(
        self,
        x: Float[Tensor, "1 seq_len hidden_size"],
        position_ids: Int[Tensor, "1 seq_len"],
    ) -> Float[Tensor, "1 seq_len hidden_size"]:
        residual = x

        with nvtx.range("input RMSNorm"):
            x = self.input_layernorm(x)
            # x :: (1, seq_len, hidden_size)

        with nvtx.range("self attention"):
            x = self.self_attn(x, position_ids)
            x = x + residual
            # x :: (1, seq_len, hidden_size)

        with nvtx.range("MLP"):
            residual = x
            x = self.mlp(self.post_attention_layernorm(x))
            x = x + residual
            # x :: (1, seq_len, hidden_size)

        return x
