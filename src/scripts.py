import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import  TrOCRProcessor, VisionEncoderDecoderModel
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from trainer import trocr_LightningModule
from pytorch_lightning.callbacks import RichProgressBar, LearningRateMonitor
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
from pytorch_lightning.strategies import DDPStrategy,FSDPStrategy
from pytorch_lightning import Trainer, seed_everything


def predict(
    processor: TrOCRProcessor, model: VisionEncoderDecoderModel, dataloader: DataLoader
):
    output: list[tuple[int, str]] = []
    confidence_scores: list[tuple[int, float]] = []
    print("length of dataloader", len(dataloader))
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(tqdm(dataloader)):
            inputs: torch.Tensor = batch["input"].to(constants.device)

            generated_ids = model.generate(
                inputs, return_dict_in_generate=True, output_scores=True
            )
            generated_text = processor.batch_decode(
                generated_ids.sequences, skip_special_tokens=True
            )

            ids = [t.item() for t in batch["idx"]]
            output.extend(zip(ids, generated_text))

            # Compute confidence scores
            batch_confidence_scores = get_confidence_scores(generated_ids)
            confidence_scores.extend(zip(ids, batch_confidence_scores))

    return output, confidence_scores


def get_confidence_scores(generated_ids):
    # Get raw logits, with shape (examples,tokens,token_vals)
    logits = generated_ids.scores
    logits = torch.stack(list(logits), dim=1)

    # Transform logits to softmax and keep only the highest (chosen) p for each token
    logit_probs = F.softmax(logits, dim=2)
    char_probs = logit_probs.max(dim=2)[0]

    # Only tokens of val>2 should influence the confidence. Thus, set probabilities to 1 for tokens 0-2
    mask = generated_ids.sequences[:, :-1] > 2
    char_probs[mask] = 1

    # Confidence of each example is cumulative product of token probs
    batch_confidence_scores = char_probs.cumprod(dim=1)[:, -1]
    return [v.item() for v in batch_confidence_scores]


# will return the accuracy but not print predictions
def validate(context: Context, print_wrong: bool = False):
    predictions, conf = predict(
        context.processor, context.model, context.val_dataloader
    )
    assert len(predictions) > 0

    correct_count = 0
    wrong_count = 0

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    correct_count = 0
    total_high_conf = 0
    correct_high_conf = 0

    CONF_THRESHOLD = 0.75
    for id, prediction in predictions:
        label = context.val_dataset.get_label(id)
        path = context.val_dataset.get_path(id)
        conf_score = conf[id][1]
        # print("prediction: ", prediction, "label: ", label, "conf_score: ", conf_score)

        if prediction == label:
            correct_count += 1

        if conf_score > CONF_THRESHOLD:
            total_high_conf += 1
            if prediction == label:
                tp += 1
                correct_high_conf += 1
            else:
                fp += 1
        else:
            if prediction == label:
                fn += 1
            else:
                tn += 1
    # calculate precision, recall and f1 score by addessinf zero division error
    precesion = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precesion * recall) / (precesion + recall)
        if (precesion + recall) > 0
        else 0
    )
    accuracy = correct_count / len(predictions) if len(predictions) > 0 else 0

    total_high_conf_percent = total_high_conf / len(predictions)
    correct_high_conf_percent = (
        correct_high_conf / total_high_conf if total_high_conf > 0 else 0
    )

    print(
        f"accuracy: {accuracy}, precesion: {precesion}, recall: {recall}, f1: {f1_score}, total_high_conf_percent: {total_high_conf_percent}, correct_high_conf_percent: {correct_high_conf_percent}"
    )

    return accuracy


def train(context: Context, num_epochs: int):
    # Intializing python lightning trainer and training the model.

    wandb_logger = WandbLogger(
        name=paths.model_path.split("/")[-1],
        project="trocr_small_ddp",
        entity="suhas04",
        mode="online",
        log_model=False,
        resume="allow",
    )
    trocr_lm = trocr_LightningModule(context, constants, num_epochs)
    checkpoint_callback_step = [
        pl.callbacks.ModelCheckpoint(
            dirpath=paths.save_path,
            filename="{epoch:02d}-{step:02d}-{val_accuracy:.4f}",  # Add step to the filename
            monitor="val_accuracy",
            mode="max",
            save_top_k=2,
            every_n_epochs=1,
        ),
        # pl.callbacks.ModelCheckpoint(
        #     dirpath=paths.save_path,
        #     filename="{epoch:02d}-{step:02d}-{val_correct_hcp:.4f}",  # Add step to the filename
        #     monitor="val_correct_hcp",
        #     mode="max",
        #     save_top_k=1,
        # ),
        # pl.callbacks.ModelCheckpoint(
        #     dirpath=paths.save_path,
        #     filename="{epoch:02d}-{step:02d}-{val_f1_score:.4f}",  # Add step to the filename
        #     monitor="val_f1_score",
        #     save_top_k=1,
        # ),
        pl.callbacks.ModelCheckpoint(
            dirpath=paths.save_path,
            filename="{epoch:02d}-{step:02d}-{val_cer:.4f}",  # Add step to the filename
            monitor="val_cer",
            mode="min",
            save_top_k=1,
        )
    ]

    custom_theme = RichProgressBarTheme(
        description="pink", 
        progress_bar="red",  
        progress_bar_finished="violet", 
        progress_bar_pulse="#6206E0", 
        batch_progress="green_yellow",  
        time="orange",  # Time color
        processing_speed="violet", 
        metrics="orange",  # Metrics color

    )

    progress_bar = RichProgressBar(
        theme=custom_theme,
        refresh_rate=1,  
        leave=False,  
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    seed_everything(42, workers=True)

    ddp = DDPStrategy(find_unused_parameters=True)
    strategy = FSDPStrategy()
    torch.set_float32_matmul_precision('medium')
    trainer = Trainer(
        max_epochs=constants.train_epochs,
        devices=1,
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        callbacks=checkpoint_callback_step + [progress_bar]+[lr_monitor],
        num_sanity_val_steps=0,
        logger=wandb_logger,
        accumulate_grad_batches=3,
        enable_checkpointing=True,
        # precision="16-mixed",
        # limit_train_batches = 10,
        # limit_val_batches = 20
    )
    ckpt_path='/mnt/private/suhas/trocr/models/exp_epoch_15_0.8926.ckpt'

    trainer.fit(trocr_lm)

    #save checkpoint after training
    trainer.save_checkpoint(paths.save_path + "/final.ckpt")


#  CUDA_VISIBLE_DEVICES=0,1,2 python -m src train --local-model

#  CUDA_VISIBLE_DEVICES=0 python ./src/starter.py 
