from argparse import ArgumentParser
import gvp
from gvp import SyntheticDataModule

from pathlib import Path
import pytorch_lightning as pl
import torch
from geometric_vector_perceptron import GVP_MPNN, GVPLayerNorm
from gvp.gvp import GVP
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric import transforms
from torch_geometric.nn import global_mean_pool
from torchmetrics.functional import mean_squared_error
from gvp.utils import _split, _merge
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


class SyntheticGVP(pl.LightningModule):
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
        verbose=0,
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

        self.W_v = nn.Sequential(
            # GVPLayerNorm(feats_x_in),
            GVP(
                dim_vectors_in=vectors_x_in,
                dim_feats_in=feats_x_in,
                dim_vectors_out=vectors_h,
                dim_feats_out=feats_h,
            ),
        )

        self.W_e = nn.Sequential(
            GVPLayerNorm(feats_edge_in),
            GVP(
                dim_vectors_in=vectors_edge_in,
                dim_feats_in=feats_edge_in,
                dim_vectors_out=vectors_h,
                dim_feats_out=feats_h,
            ),
        )

        self.layers = nn.ModuleList(
            GVP_MPNN(
                feats_x_in=feats_h,
                vectors_x_in=vectors_h,
                feats_x_out=feats_h,
                vectors_x_out=vectors_h,
                feats_edge_in=feats_h,
                vectors_edge_in=vectors_h,
                feats_edge_out=0,
                vectors_edge_out=0,
                dropout=dropout,
                residual=residual,
                vector_dim=vector_dim,
                verbose=verbose,
            )
            for _ in range(3) # 3 message passing layers
        )

        self.W_out = nn.Sequential(
            GVPLayerNorm(feats_h),
            GVP(
                dim_vectors_in=vectors_h,
                dim_vectors_out=0,
                dim_feats_in=feats_h,
                dim_feats_out=feats_h
            )
        )

        self.dense = nn.Sequential(
            nn.Linear(feats_h, 2*feats_h), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(2*feats_h, 1)
        )

        self.dense = nn.Linear(feats_h + (vectors_h * vector_dim), 1)

    def forward(self, data):
        x = data.x
        batch = data.batch
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        print(*_split(x, self.vectors_x_in, self.feats_x_in, self.vector_dim))

        x = _merge(
            self.W_v(_split(x, self.vectors_x_in, self.feats_x_in, self.vector_dim))
        )

        print(x.shape)
        edge_attr = _merge(
            self.W_e(
                *_split(edge_attr, self.vectors_x_in, self.feats_x_in, self.vector_dim)
            )
        )

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        x = _merge(
            self.W_out(
                *_split(x, self.vectors_h, self.feats_h, self.vector_dim)
            )
        )

        x = global_mean_pool(x, batch)

        x = self.dense(x)

        return x

    def shared_step(self, batch):
        data, y = batch, batch.y
        y_hat = self(data).view(-1)
        loss = mean_squared_error(y_hat, y)

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
    model = SyntheticGVP(1, 1, 1, 1, 20, 4)

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
        max_epochs=100,
        gpus=1,
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
