from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

from gvp import CNNDataModule
import gvp

class ShallowCNN(pl.LightningModule):
    """
    Shallow 3-layer CNN
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, (kernel_size,kernel_size,kernel_size))
        self.conv2 = nn.Conv3d(32, 32, (kernel_size,kernel_size,kernel_size)) 
        self.conv3 = nn.Conv3d(32, 32, (kernel_size,kernel_size,kernel_size)) 
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        out = F.relu( self.conv1(x) )
        out = F.relu( self.conv2(out) )
        out = F.relu( self.conv3(out) )
        out = F.adaptive_max_pool3d(out, (1,1,1))
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

def main():
    """
    Training and evaluating the model
    """
    # ------------
    # Args can be set when running script from terminal
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--task', default='off_center', type=str)
    parser.add_argument('--scaling', default=False, type=bool)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # Set directory to get dataset from
    # ------------
    data_dir = Path(gvp.__file__).parents[1] / "data/synthetic"

    dm = CNNDataModule(data_dir, args.batch_size, args.task, num_workers=args.num_workers, scaling=args.scaling)

    # ------------
    # Define model
    # ------------
    model = ShallowCNN(kernel_size=3)

    # ------------
    # Training: use wandb to record and save logs (https://wandb.ai/afiqahk/GVP?workspace=user-afiqahk)
    # ------------
    wandb_logger = WandbLogger(name=f"ShallowCNN-{args.task}", project="GVP", reinit=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="model_checkpoints",
        filename=f"ShallowCNN-{args.task}-"+"{epoch:02d}-{val_loss:.2f}",
        save_weights_only=True,
        save_top_k=3,
        mode="min",
    )
    trainer = pl.Trainer.from_argparse_args(args, max_epochs=100, logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, dm)

    # ------------
    # Testing
    # ------------
    result = trainer.test(datamodule=dm)
    print(result)

    wandb.finish()
    
if __name__ == "__main__":
    main()