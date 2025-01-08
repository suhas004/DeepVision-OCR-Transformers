import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from tqdm import tqdm

from trainer import trocr_LightningModule
from pytorch_lightning.callbacks import RichProgressBar, StochasticWeightAveraging
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme


import pytorch_lightning as pl


import sys

sys.path.append("/mnt/private/suhas/trocr/src")

from configs import constants
from configs import paths

from context import Context
from util import debug_print

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Trainer, seed_everything



trocr_lm = trocr_LightningModule(Context, constants, num_epochs=30)


trocr_lm.load_from_checkpoint('')

print("trocr_lm:", trocr_lm)


# CUDA