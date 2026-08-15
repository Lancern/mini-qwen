from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.cuda import nvtx

from .attn import GQA
from .cache import Cache
from .norm import RMSNorm
from .rope import RoPE


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
