import torch
from jaxtyping import Float
from torch import Tensor


class LayerCache:
    def __init__(self):
        self._cached_k: (
            Float[Tensor, "batch num_attn_heads kv_seq_len head_dim"] | None
        ) = None
        self._cached_v: (
            Float[Tensor, "batch num_attn_heads kv_seq_len head_dim"] | None
        ) = None

    @property
    def cached_seq_len(self) -> int:
        if self._cached_k is None:
            return 0
        return self._cached_k.shape[2]

    def update_and_concat(
        self,
        k: Float[Tensor, "batch num_attn_heads seq_len head_dim"],
        v: Float[Tensor, "batch num_attn_heads seq_len head_dim"],
    ) -> tuple[
        Float[Tensor, "batch num_attn_heads kv_seq_len+seq_len head_dim"],
        Float[Tensor, "batch num_attn_heads kv_seq_len+seq_len head_dim"],
    ]:
        self._cached_k = LayerCache._concat_with_cached(self._cached_k, k)
        self._cached_v = LayerCache._concat_with_cached(self._cached_v, v)
        return self._cached_k, self._cached_v

    @staticmethod
    def _concat_with_cached(
        cached: Float[Tensor, "batch num_attn_heads kv_seq_len head_dim"] | None,
        x: Float[Tensor, "batch num_attn_heads seq_len head_dim"],
    ) -> Float[Tensor, "batch num_attn_heads kv_seq_len+seq_len head_dim"]:
        if cached is None:
            return x
        return torch.cat((cached, x), dim=2)


class Cache:
    def __init__(self, num_layers: int):
        self._layer_caches = [LayerCache() for _ in range(num_layers)]

    def __len__(self) -> int:
        return len(self._layer_caches)

    def __getitem__(self, index: int) -> LayerCache:
        return self._layer_caches[index]
