import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.cuda import nvtx


class RoPE(nn.Module):
    def __init__(self, theta: float, head_dim: int, max_seq_len: int):
        super().__init__()
        assert head_dim % 2 == 0
        assert max_seq_len > 0

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        position_ids = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(position_ids, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.cos = nn.Buffer(emb.cos(), persistent=False)
        self.sin = nn.Buffer(emb.sin(), persistent=False)

    @nvtx.range("RoPE.forward")
    def forward(
        self,
        q: Float[Tensor, "batch num_attn_heads seq_len head_dim"],
        k: Float[Tensor, "batch num_kv_heads seq_len head_dim"],
        position_ids: Int[Tensor, "batch seq_len"],
    ) -> tuple[
        Float[Tensor, "batch num_attn_heads seq_len head_dim"],
        Float[Tensor, "batch num_kv_heads seq_len head_dim"],
    ]:
        cos = self.cos[position_ids].unsqueeze(1)
        sin = self.sin[position_ids].unsqueeze(1)
        # cos, sin :: (batch_size, 1, seq_len, head_dim)

        def rotate_half(
            x: Float[Tensor, "... head_dim"],
        ) -> Float[Tensor, "... head_dim"]:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos.to(q.dtype)) + (rotate_half(q) * sin.to(q.dtype))
        k_embed = (k * cos.to(k.dtype)) + (rotate_half(k) * sin.to(k.dtype))
        return q_embed, k_embed
