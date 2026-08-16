import torch
from jaxtyping import Float
from torch import Tensor


class LayerCache:
    def __init__(self, max_seq_len: int):
        self._max_seq_len = max_seq_len
        self._cached_seq_len = 0

        self._k_buffer: (
            Float[Tensor, "1 num_attn_heads kv_seq_len head_dim"] | None
        ) = None
        self._v_buffer: (
            Float[Tensor, "1 num_attn_heads kv_seq_len head_dim"] | None
        ) = None

    @property
    def cached_seq_len(self) -> int:
        return self._cached_seq_len

    @property
    def size(self) -> int:
        sz = 0
        if self._k_buffer is not None:
            sz += self._k_buffer.nbytes
        if self._v_buffer is not None:
            sz += self._v_buffer.nbytes
        return sz

    def update(
        self,
        k: Float[Tensor, "1 num_kv_heads seq_len head_dim"],
        v: Float[Tensor, "1 num_kv_heads seq_len head_dim"],
    ) -> tuple[
        Float[Tensor, "1 num_kv_heads kv_seq_len+seq_len head_dim"],
        Float[Tensor, "1 num_kv_heads kv_seq_len+seq_len head_dim"],
    ]:
        batch, num_kv_heads, seq_len, head_dim = k.shape
        assert batch == 1

        if self._k_buffer is None:
            assert seq_len <= self._max_seq_len

            self._k_buffer = torch.empty(
                batch,
                num_kv_heads,
                self._max_seq_len,
                head_dim,
                dtype=k.dtype,
                device=k.device,
            )
            self._v_buffer = torch.empty(
                batch,
                num_kv_heads,
                self._max_seq_len,
                head_dim,
                dtype=v.dtype,
                device=v.device,
            )

        assert self._k_buffer is not None
        assert self._v_buffer is not None

        updated_cached_len = self._cached_seq_len + seq_len
        assert updated_cached_len <= self._max_seq_len

        self._k_buffer[:, :, self._cached_seq_len : updated_cached_len].copy_(k)
        self._v_buffer[:, :, self._cached_seq_len : updated_cached_len].copy_(v)
        self._cached_seq_len = updated_cached_len

        return (
            self._k_buffer[:, :, : self._cached_seq_len, :],
            self._v_buffer[:, :, : self._cached_seq_len, :],
        )


class Cache:
    def __init__(self, num_layers: int, max_seq_len: int):
        self._layer_caches = [LayerCache(max_seq_len) for _ in range(num_layers)]

    def __len__(self) -> int:
        return len(self._layer_caches)

    def __getitem__(self, index: int) -> LayerCache:
        return self._layer_caches[index]

    @property
    def size(self) -> int:
        return sum(lc.size for lc in self._layer_caches)
