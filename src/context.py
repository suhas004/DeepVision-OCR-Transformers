from dataclasses import dataclass

from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import sys
sys.path.append('/mnt/private/suhas/trocr/src')
from dataset import HCRDataset,HCRDataset_test


@dataclass
class Context:
    model: VisionEncoderDecoderModel
    processor: TrOCRProcessor

    train_dataset: HCRDataset
    train_dataloader: DataLoader

    val_dataset: HCRDataset_test
    val_dataloader: DataLoader

    #get paths
    

