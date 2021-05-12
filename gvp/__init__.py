from gvp.models.synthetic.synthetic_data_module import SyntheticDataModule
from gvp.models.synthetic.synthetic_gnn import SyntheticGNN
from gvp.models.synthetic.synthetic_gvp import SyntheticGVP
from gvp.gvp import GVP
import pytorch_lightning as pl

SEED = 42
pl.seed_everything(42)
