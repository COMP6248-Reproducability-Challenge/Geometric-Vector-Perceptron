import torch
from torch import nn, einsum
from torch_geometric.nn import MessagePassing
# types
from typing import Optional, List, Union
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor, Tensor

class GVP(nn.Module):
    # https://github.com/lucidrains/geometric-vector-perceptron/blob/main/geometric_vector_perceptron/geometric_vector_perceptron.py
    def __init__(
        self,
        *,
        dim_vectors_in,
        dim_vectors_out,
        dim_feats_in,
        dim_feats_out,
        feats_activation = nn.Sigmoid(),
        vectors_activation = nn.Sigmoid()
    ):
        super().__init__()
        self.dim_vectors_in = dim_vectors_in
        self.dim_feats_in = dim_feats_in

        self.dim_vectors_out = dim_vectors_out
        dim_h = max(dim_vectors_in, dim_vectors_out)

        self.Wh = nn.Parameter(torch.randn(dim_vectors_in, dim_h))
        self.Wu = nn.Parameter(torch.randn(dim_h, dim_vectors_out))

        self.vectors_activation = vectors_activation

        self.to_feats_out = nn.Sequential(
            nn.Linear(dim_h + dim_feats_in, dim_feats_out),
            feats_activation
        )

    def forward(self, feats, vectors):
        print("hi im gvp")
        b, n, _, v, c  = *feats.shape, *vectors.shape

        assert c == 3 and v == self.dim_vectors_in, 'vectors have wrong dimensions'
        assert n == self.dim_feats_in, 'scalar features have wrong dimensions'

        Vh = einsum('b v c, v h -> b h c', vectors, self.Wh)
        Vu = einsum('b h c, h u -> b u c', Vh, self.Wu)

        sh = torch.norm(Vh, p = 2, dim = -1)
        vu = torch.norm(Vu, p = 2, dim = -1, keepdim = True)

        s = torch.cat((feats, sh), dim = 1)

        feats_out = self.to_feats_out(s)
        vectors_out = self.vectors_activation(vu) * Vu

        return feats_out, vectors_out