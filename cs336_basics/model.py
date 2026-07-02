import math
import torch
from jaxtyping import Float
from typing import Any
from torch import nn


class LinearLayer(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device=None, dtype=None
    ) -> None:
        super().__init__()
        self.weights = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        # Xavier initialization to avoid vanishing / exploding gradients
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> torch.Tensor:  # noqa: F722
        # Paper usually writes in row vector notation y = x * W^T
        # However, in linear algebra, write in column vector: y = W *x
        # key idea: batching dimension comes last 
        # return torch.matmul(x, self.weights)
        return torch.einsum("...i,oi ->...o", x, self.weights)

    def set_weights(self, weights: Float[torch.Tensor, "d_out d_in"]):  # noqa: F722
        with torch.no_grad():
            self.weights.copy_(weights)


class EmbeddingLayer(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
