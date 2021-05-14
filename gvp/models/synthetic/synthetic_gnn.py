from argparse import ArgumentParser
from gvp.gvp import GVP, GVPConvLayer, _split
from pathlib import Path

import gvp
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from gvp import SyntheticDataModule, SyntheticGVP
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn, optim
from torch_geometric import transforms
from torch_geometric.nn import GCNConv, global_max_pool, LayerNorm, MessagePassing
from torchmetrics.functional import mean_squared_error
from torch_scatter import scatter_mean

import wandb


class ExtendedPPF:
    def __init__(self, norm=True, cat=True):
        self.norm = norm
        self.cat = cat

        self.ppf = transforms.PointPairFeatures(cat=False)
        self.distance = transforms.Distance(norm=norm, cat=False)

    def __call__(self, data):
        existing_features = data.edge_attr

        ppf_features = self.ppf(data).edge_attr
        ppf_features = torch.cos(ppf_features)
        dist_features = self.distance(data).edge_attr

        new_features = torch.cat([dist_features, ppf_features[:, 1:]], dim=-1)

        if existing_features is not None and self.cat:
            data.edge_attr = torch.cat([existing_features, new_features], dim=-1)
        else:
            data.edge_attr = new_features

        return data

class GNNConv(MessagePassing):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.
    
    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.
    
    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture
    '''
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean"):
        super(GNNConv, self).__init__(aggr=aggr)
        self.si = in_dims
        self.so = out_dims
        self.se = edge_dims
        
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    nn.Linear(2*self.si + self.se, self.so)
                )
            else:
                module_list.append(
                    nn.Linear((2*self.si + self.se), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(nn.Linear(out_dims, out_dims))
                    module_list.append(nn.Sigmoid())
                module_list.append(nn.Linear(out_dims, out_dims))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        message = self.propagate(edge_index, 
                    x=x,
                    edge_attr=edge_attr)
        return message 

class GNNConvLayer(nn.Module):
    '''
    Full graph convolution / message passing layer with 
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward 
    network to node embeddings, and returns updated node embeddings.
    
    To only compute the aggregated messages, see `GVPConv`.
    
    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    '''
    def __init__(self, node_dims, edge_dims,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False):
        
        super(GNNConvLayer, self).__init__()
        self.conv = GNNConv(node_dims, node_dims, edge_dims, n_message,
                           aggr="add" if autoregressive else "mean")
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([nn.Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(nn.Linear(node_dims, node_dims))
        else:
            hid_dims = 4*node_dims
            ff_func.append(nn.Linear(node_dims, hid_dims))
            ff_func.append(nn.Sigmoid())
            for i in range(n_feedforward-2):
                ff_func.append(nn.Linear(hid_dims, hid_dims))
                ff_func.append(nn.Sigmoid())
            ff_func.append(nn.Linear(hid_dims, node_dims))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr,
                node_mask=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`. 
                If not `None`, will be used as srcqq node embeddings
                for forming messages where src >= dst. The corrent node 
                embeddings `x` will still be the base of the update and the 
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''
        
        dh = self.conv(x, edge_index, edge_attr)
        
        x = self.norm[0](x + self.dropout[0](dh))
        
        dh = self.ff_func(x)
        x = self.norm[1](x + self.dropout[1](dh))
        
        return x

class SyntheticGNN(pl.LightningModule):
    '''
    GVP-GNN for Model Quality Assessment as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.1):
        
        super(SyntheticGNN, self).__init__()

        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        
        if seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            nn.Linear(node_in_dim, node_h_dim)
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            nn.Linear(edge_in_dim, edge_h_dim)
        )
        
        self.layers = nn.ModuleList(
                GNNConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns= node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            nn.Linear(node_h_dim, ns),
            nn.Sigmoid()
        )
            
        self.dense = nn.Sequential(
            nn.Linear(ns, 2*ns), nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2*ns, 1)
        )

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        if seq is not None:
            seq = self.W_s(seq)
            h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)
        
        if batch is None: out = out.mean(dim=0, keepdims=True)
        else: out = scatter_mean(out, batch, dim=0)
        
        return self.dense(out).squeeze(-1)

    def shared_step(self, batch):
        h_V = batch.x
        h_E = batch.edge_attr
            
        y_hat = self(h_V, batch.edge_index, h_E, batch=batch.batch)
        loss = mean_squared_error(y_hat, batch.y)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("test_loss", loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters())

def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--task", default="off_center", type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    data_dir = Path(gvp.__file__).parents[1] / "data/synthetic"

    transform = transforms.Compose([transforms.KNNGraph(k=10), ExtendedPPF()])
    dm = SyntheticDataModule(
        data_dir, args.batch_size, args.task, transform, num_workers=args.num_workers
    )

    # ------------
    # model
    # ------------
    model = SyntheticGNN(4, 32, 4, 32)

    # ------------
    # training
    # ------------
    wandb_logger = WandbLogger(
        name=f"SyntheticGNN-{args.task}", project="GVP", reinit=True
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="model_checkpoints",
        filename=f"SyntheticGNN-{args.task}-" + "{epoch:02d}-{val_loss:.2f}",
        save_weights_only=True,
        save_top_k=3,
        mode="min",
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, dm)

    # ------------
    # testing
    # ------------
    result = trainer.test(datamodule=dm)
    print(result)

    wandb.finish()


if __name__ == "__main__":
    main()
