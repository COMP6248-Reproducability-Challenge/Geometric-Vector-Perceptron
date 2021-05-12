from argparse import ArgumentParser
import gvp
from gvp import SyntheticDataModule


from pathlib import Path
import pytorch_lightning as pl
import torch
from gvp.gvp import GVP, GVPConvLayer, LayerNorm, _split
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric import transforms
from torchmetrics.functional import mean_squared_error
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch_scatter import scatter_mean


class SyntheticGVP(pl.LightningModule):
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
        
        super(SyntheticGVP, self).__init__()

        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        
        if seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))
            
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
        
        return self.dense(out).squeeze(-1) + 0.5

    def shared_step(self, batch):
        h_V = _split(batch.x, self.node_in_dim[1])
        h_E = _split(batch.edge_attr, self.edge_in_dim[1])
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

    transform = transforms.Compose(
        [
            transforms.KNNGraph(k=10),
            transforms.Cartesian(),
            transforms.Distance(),
        ]
    )
    dm = SyntheticDataModule(
        data_dir, args.batch_size, args.task, transform, num_workers=args.num_workers
    )

    # ------------
    # model
    # ------------
    model = SyntheticGVP((1, 1), (20, 4), (1, 1), (20, 4))

    # ------------
    # training
    # ------------
    wandb_logger = WandbLogger(
        name=f"SyntheticGVP-{args.task}", project="GVP", reinit=True
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="model_checkpoints",
        filename=f"SyntheticGVP-{args.task}-" + "{epoch:02d}-{val_loss:.2f}",
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
