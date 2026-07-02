import math
import torch
from jaxtyping import Bool, Float, Int
from torch import nn
from typing import List


class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.empty((in_features, out_features), device=device, dtype=dtype))
        
        # Xavier initialization to avoid vanishing / exploding gradients
        std = math.sqrt(2/(in_features + out_features))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std,  a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weights)
    
    def set_weights(self, weights: Float[Tensor, "d_out d_in"]):
        with torch.no_grad():
            self.weights.copy_(weights.T)

class EmbeddingLayer(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
