import pytorch_lightning as pl
import torch
from geometric_vector_perceptron import GVP_MPNN
from torch import nn


class SyntheticGVP(nn.Module):
    def __init__(
        self,
        feats_x_in,
        vectors_x_in,
        feats_edge_in,
        vectors_edge_in,
        feats_h,
        vectors_h,
        dropout=0.0,
        residual=False,
        vector_dim=3,
        verbose=0
    ):
        super().__init__()

        self.feats_x_in = feats_x_in
        self.vectors_x_in = vectors_x_in
        self.feats_edge_in = feats_edge_in
        self.vectors_edge_in = vectors_edge_in
        self.feats_h = feats_h
        self.vectors_h = vectors_h
        self.dropout = dropout
        self.residual = residual
        self.vector_dim = vector_dim
        self.verbose = verbose

        self.layers = nn.ModuleList([
            GVP_MPNN(
                feats_x_in = feats_x_in,
                vectors_x_in = vectors_x_in,
                feats_x_out = feats_h,
                vectors_x_out = vectors_h,
                feats_edge_in = feats_edge_in,
                vectors_edge_in = vectors_edge_in,
                feats_edge_out = feats_h,
                vectors_edge_out = vectors_h,
                dropout = dropout,
                residual = residual,
                vector_dim = vector_dim,
                verbose = verbose 
            ),
            GVP_MPNN(
                feats_x_in = feats_h,
                vectors_x_in = vectors_h,
                feats_x_out = feats_h,
                vectors_x_out = vectors_h,
                feats_edge_in = feats_h,
                vectors_edge_in = vectors_h,
                feats_edge_out = feats_h,
                vectors_edge_out = vectors_h,
                dropout = dropout,
                residual = residual,
                vector_dim = vector_dim,
                verbose = verbose
            ),
            GVP_MPNN(
                feats_x_in = feats_h,
                vectors_x_in = vectors_h,
                feats_x_out = feats_h,
                vectors_x_out = vectors_h,
                feats_edge_in = feats_h,
                vectors_edge_in = vectors_h,
                feats_edge_out = feats_h,
                vectors_edge_out = vectors_h,
                dropout = dropout,
                residual = residual,
                vector_dim = vector_dim,
                verbose = verbose
            )
        ]
        )

        self.dense = nn.Linear(feats_h + (vectors_h * vector_dim), 1)
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        for layer in self.layers:
            out = layer(x, edge_index, edge_attr)

        out = self.dense(out)

        return out



