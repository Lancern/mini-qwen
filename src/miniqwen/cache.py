from typing import cast

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn


class LayerCache(nn.Module):
    def __init__(
        self,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()

        self._device = device
        self._max_seq_len = max_seq_len

        batch = 1
        self.k_buffer = nn.Buffer(
            torch.empty(
                batch, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device
            ),
            persistent=False,
        )
        self.v_buffer = nn.Buffer(
            torch.empty(
                batch, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device
            ),
            persistent=False,
        )
        self.cached_len = nn.Buffer(
            torch.zeros(batch, dtype=torch.int64, device=device), persistent=False
        )

    @property
    def cached_seq_len(self) -> int:
        return int(self.cached_len.item())

    @property
    def size(self) -> int:
        return self.k_buffer.nbytes + self.v_buffer.nbytes

    def update(
        self,
        k: Float[Tensor, "1 num_kv_heads seq_len head_dim"],
        v: Float[Tensor, "1 num_kv_heads seq_len head_dim"],
    ) -> tuple[
        Float[Tensor, "1 num_kv_heads max_seq_len head_dim"],
        Float[Tensor, "1 num_kv_heads max_seq_len head_dim"],
        Int[Tensor, "1"],
    ]:
        batch, _, seq_len, _ = k.shape
        assert batch == 1

        cache_positions = self.cached_len + torch.arange(
            seq_len, dtype=torch.int64, device=self._device
        )

        self.k_buffer.index_copy_(2, cache_positions, k)
        self.v_buffer.index_copy_(2, cache_positions, v)
        self.cached_len.add_(seq_len)

        return self.k_buffer, self.v_buffer, self.cached_len


class Cache(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()

        self._layer_caches = nn.ModuleList(
            LayerCache(num_kv_heads, max_seq_len, head_dim, dtype, device)
            for _ in range(num_layers)
        )

    def __len__(self) -> int:
        return len(self._layer_caches)

    def __getitem__(self, index: int) -> LayerCache:
        return cast(LayerCache, self._layer_caches[index])

    @property
    def size(self) -> int:
        return sum(cast(LayerCache, lc).size for lc in self._layer_caches)
