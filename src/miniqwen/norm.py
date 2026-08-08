import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.cuda import nvtx


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
