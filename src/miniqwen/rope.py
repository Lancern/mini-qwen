import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.cuda import nvtx


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        assert head_dim % 2 == 0
        assert max_seq_len > 0

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        position_ids = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(position_ids, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.cos = nn.Buffer(emb.cos().to(dtype=dtype, device=device), persistent=False)
        self.sin = nn.Buffer(emb.sin().to(dtype=dtype, device=device), persistent=False)

    @nvtx.range("RoPE.forward")
    def forward(
        self,
        q: Float[Tensor, "1 num_attn_heads seq_len head_dim"],
        k: Float[Tensor, "1 num_kv_heads seq_len head_dim"],
        position_ids: Int[Tensor, "1 seq_len"],
    ) -> tuple[
        Float[Tensor, "1 num_attn_heads seq_len head_dim"],
        Float[Tensor, "1 num_kv_heads seq_len head_dim"],
    ]:
        cos = self.cos[position_ids].unsqueeze(1)
        sin = self.sin[position_ids].unsqueeze(1)
        # cos, sin :: (1, 1, seq_len, head_dim)

        q_embed = _apply_rotary(q, cos.to(q.dtype), sin.to(q.dtype))
        k_embed = _apply_rotary(k, cos.to(k.dtype), sin.to(k.dtype))

        return q_embed, k_embed


def _apply_rotary(
    x: Float[Tensor, "... head_dim"],
    cos: Float[Tensor, "... head_dim"],
    sin: Float[Tensor, "... head_dim"],
) -> Float[Tensor, "... head_dim"]:
    half_head_dim = x.shape[-1] // 2
    x1 = x[..., :half_head_dim]
    x2 = x[..., half_head_dim:]

    # Start with the final output allocation and accumulate the cross-half terms
    # into it. This avoids materializing rotate_half(x) and concatenating its two
    # halves before the elementwise multiply and add.
    output = x * cos
    output[..., :half_head_dim].addcmul_(x2, sin[..., :half_head_dim], value=-1)
    output[..., half_head_dim:].addcmul_(x1, sin[..., half_head_dim:])
    return output
