import math
import torch
from jaxtyping import Float
from typing import Any
from torch import nn

def _init_2d_weights(in_features, out_features, device, dtype):
    weights = nn.Parameter(
        torch.empty((out_features, in_features), device=device, dtype=dtype)
    )
    # Xavier initialization to avoid vanishing / exploding gradients
    std = math.sqrt(2 / (in_features + out_features))
    nn.init.trunc_normal_(weights, mean=0.0, std=std, a=-3 * std, b=3 * std)
    return weights


class LinearLayer(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device=None, dtype=None
    ) -> None:
        super().__init__()
        self.weights = _init_2d_weights(in_features, out_features, device, dtype)

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
    """
    Map integer tokens into a vector space of dimension d_model
    Intuition. embedding matrix is a place holder of embedding vectors for each vocab
    e.g m (vocab_size, d_model) = (4, 3)
    token_1 = [0.41, 503, 0.3]
    token_2 = [0.23, 503, 0.13]
    token_3 = [0.51, 5151, 0.3]
    token_4 = [0.41, 503, 0.7]

    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) -> None:
        """
        
        num_embeddings: int Dimension of embedding vectors (size of vocab)
        embedding_dim:  int Dimension of the embedding vectors (d_model), aka how long the vector is
        """
        super().__init__()
        self.embedding_matrix = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        #  Xavier initialization trick
        std = math.sqrt(2 / (num_embeddings + embedding_dim))
        nn.init.trunc_normal_(self.embedding_matrix, mean=0.0, std=std, a=-3*std, b=3*std)

    def set_weights(self, weights: Float[torch.Tensor, "vocab_size d_model"]):  # noqa: F722
        with torch.no_grad():
            self.embedding_matrix.copy_(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ return a tensor (or list of learned vectors) that contain mapping of token IDs in X to 
        sequence length
        x token_ids - torch.LongTensor of token IDs with shape (batch_size, sequence_length).

        output: y [batch_size, sequence_length, d_model]
        """

        # matrix transformation
        # einsum matmul is incorrect. require different matrix op
        # return torch.einsum("bs,sd->bd", x, self.embedding_matrix)

        # look up ops in pytorch / numpy
        return self.embedding_matrix[x]

class RMSNormLayer(nn.Module):
    """

    looking at the formula: what should be the dimension of gi so that RMSNorm(x).shape = x.shape
    """
    def __init__(self, d_model: int, eps: float, device, dtype: torch.dtype, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eps = 1e-5 if eps is None else eps
        self.d_model = d_model
        self.learned_vector = nn.Parameter(torch.empty(size=(d_model,), device=device, dtype=dtype))

    def set_weights(self, weights):
        with torch.no_grad():
            self.learned_vector.copy_(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) 
        and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        rms_norm = torch.div(x, rms) * self.learned_vector
        return rms_norm.to(in_dtype)

class SwiGLULayer(nn.Module):
    """ Combination of linear layer + silu q
    """
    def __init__(self, d_model, d_ff, device=None, dtype=None) -> None:
        super().__init__()
        self.w1 = _init_2d_weights(d_model, d_ff, device, dtype)
        self.w3 = _init_2d_weights(d_model, d_ff, device, dtype)
        self.w2 = _init_2d_weights(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tmp = torch.einsum("...i,oi->...o", x, self.w1)
        swish = tmp * torch.sigmoid(tmp)
        point_wise = torch.mul(swish, torch.einsum("...i,oi->...o", x, self.w3))
        swi_glu = torch.einsum("...o,io->...i",point_wise, self.w2)
        return swi_glu
