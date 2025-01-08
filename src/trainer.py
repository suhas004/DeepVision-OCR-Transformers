# Importing required libraries
import sys
import os
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import TrOCRProcessor, AdamW, get_scheduler, VisionEncoderDecoderModel
from evaluate import load

import pytorch_lightning as pl
from tqdm import tqdm
import gc

# from tabulate import tabulate


from configs import constants
from configs import paths

from context import Context


# Adding path to system path
sys.path.append("/mnt/private/suhas/trocr/src")

# Importing local modules
from configs import paths



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class trocr_LightningModule(pl.LightningModule):
    def __init__(self, context, constants, num_epochs):
        # def __init__(self):
        super().__init__()
        self.context = context
        self.constants = constants
        self.num_epochs = num_epochs
        # self.context = Context
        # self.constants = constants
        # self.num_epochs = 50
        self.processor = context.processor
        self.model = context.model
        # self.model = VisionEncoderDecoderModel.from_pretrained(
        #     "/mnt/suhas/OCR/trocr/trocr_base_model_128", ignore_mismatched_sizes=True
        # )
        self.optimizer = None  # Initialized in configure_optimizers
        self.lr_scheduler = None  # Initialized in configure_optimizers
        self.cer = load("cer")
        self.running_metrics = {
            "running_loss_train": AverageMeter(),
            "accuracy": AverageMeter(),
            "precision": AverageMeter(),
            "recall": AverageMeter(),
            "f1_score": AverageMeter(),
            "total_hcp": AverageMeter(),
            "correct_hcp": AverageMeter(),
            "cer": AverageMeter(),
            "wer": AverageMeter(),
        }
        self.running_epoch = 0
        self.best_accuracy, self.best_cer, self.best_f1, self.correct_hcp = 0, 1, 0, 0
        self.validation_step_outputs = []


    def forward(self, inputs, labels):
        return self.model(pixel_values=inputs, labels=labels)

    # def configure_sharded_model(self):
    #     self.model = FSDP(self.model)

    def training_step(self, batch, batch_idx):
        inputs = batch["input"]
        labels = batch["label"]
        outputs = self(inputs, labels)
        
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,)
        gc.collect()

        del outputs
        return loss
    
    # def configure_sharded_model(self):
    #     print("configure_sharded_model")
    #     self.model = wrap(self.model)

    def on_train_epoch_end(self):
        print("on_train_epoch_end")
        self.running_metrics["running_loss_train"].reset()

    # def configure_optimizers(self):
    #     self.optimizer = AdamW(
    #         self.model.parameters(),
    #         # self.trainer.model.parameters(),  # use only when using fsdp_native strategy
    #         lr=self.constants.learning_rate,
    #     )
    #     num_training_steps = self.num_epochs * len(self.train_dataloader())
    #     self.lr_scheduler = get_scheduler(
    #         "linear",
    #         optimizer=self.optimizer,
    #         num_warmup_steps=200,
    #         num_training_steps=num_training_steps,
    #     )

    #     return [self.optimizer], [self.lr_scheduler]
    def configure_optimizers(self):
        """Load optimizers and schedulers."""
        self.optimizer = AdamW(
            self.model.parameters(),
            # self.trainer.model.parameters(),  # use only when using fsdp_native strategy
            lr=self.constants.learning_rate,
            weight_decay=1e-5
        )
        # optimizer = FusedAdam(self.parameters(), 0.001, weight_decay=1e-5)

        self.lr_scheduler  = {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, [2, 4, 6, 8, 10, 20,30,40,50,60,70,80], 0.4
            ),
            "monitor": "validation-loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return [self.optimizer], [self.lr_scheduler] 

    def train_dataloader(self):
        return self.context.train_dataloader

    def val_dataloader(self):
        return self.context.val_dataloader

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        # labels_text = self.context.processor.batch_decode(
        #     labels, skip_special_tokens=True
        # )

        predictions = self.predict(batch, batch_idx)

        correct_count = 0
        wrong_count = 0

        tp = 0
        fp = 0
        fn = 0
        tn = 0
        correct_count = 0
        total_high_conf = 0
        correct_high_conf = 0

        gts = []
        preds = []

        CONF_THRESHOLD = 0.75
        for id, prediction, conf_score in predictions:
            label = self.context.val_dataset.get_label(id)

            gts.append(label)
            preds.append(prediction)
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

        # calculate precision, recall and f1 score by addessing zero division error
        precesion = float(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recall = float(tp / (tp + fn) if (tp + fn) > 0 else 0)
        f1_score = float(
            2 * (precesion * recall) / (precesion + recall)
            if (precesion + recall) > 0
            else 0
        )
        accuracy = float(
            correct_count / len(predictions) if len(predictions) > 0 else 0
        )

        total_high_conf_percent = float(total_high_conf / len(predictions))
        correct_high_conf_percent = float(
            correct_high_conf / total_high_conf if total_high_conf > 0 else 0
        )
        cer_score = self.cer.compute(predictions=preds, references=gts)

        self.log(
            "val_accuracy",
            accuracy,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        # self.log("val_precesion", precesion, logger=True, sync_dist=True)
        # self.log("val_recall", recall, logger=True, sync_dist=True)
        # self.log("val_f1_score", f1_score, logger=True, sync_dist=True)
        self.log("val_cer", cer_score, on_epoch = True,prog_bar=True,logger=True, sync_dist=True)
        # self.log(
        #     "val_correct_hcp", correct_high_conf_percent, logger=True, sync_dist=True
        # )
        # self.log("val_total_hcp", total_high_conf_percent, logger=True, sync_dist=True)



        self.validation_step_outputs.append(
            [
                accuracy,
                # precesion,
                # recall,
                # f1_score,
                cer_score,
                # correct_high_conf_percent,
            ]
        )
        gc.collect()

        return (
            accuracy,
            # precesion,
            # recall,
            # f1_score,
            cer_score,
            # correct_high_conf_percent,
        )

    def predict(self, batch, batch_idx):
        output: list[tuple[int, str]] = []

        with torch.no_grad():
            inputs: torch.Tensor = batch["input"]

            generated_ids = self.model.generate(
                inputs, return_dict_in_generate=True, output_scores=True
            )
            generated_text = self.processor.batch_decode(
                generated_ids.sequences, skip_special_tokens=True
            )
            ids = [t.item() for t in batch["idx"]]
            batch_confidence_scores = self.get_confidence_scores(generated_ids)
            output.extend(zip(ids, generated_text, batch_confidence_scores))

        return output

    def get_confidence_scores(self, generated_ids):
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

    def on_validation_epoch_end(self):
        avg_accuracy = sum([x[0] for x in self.validation_step_outputs]) / len(
            self.validation_step_outputs
        )
        # avg_precision = sum([x[1] for x in self.validation_step_outputs]) / len(
        #     self.validation_step_outputs
        # )
        # avg_recall = sum([x[2] for x in self.validation_step_outputs]) / len(
        #     self.validation_step_outputs
        # )
        # avg_f1 = sum([x[3] for x in self.validation_step_outputs]) / len(
        #     self.validation_step_outputs
        # )
        avg_cer = sum([x[1] for x in self.validation_step_outputs]) / len(
            self.validation_step_outputs
        )
        # avg_correct_hcp = sum([x[5] for x in self.validation_step_outputs]) / len(
        #     self.validation_step_outputs
        # )

        # log all the above values
        # self.log(
        #     "avg_val_accuracy_epoch",
        #     avg_accuracy,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        #     sync_dist=True,
        # )
        # self.log("avg_val_precision_epoch", avg_precision, logger=True, sync_dist=True)
        # self.log("avg_val_recall_epoch", avg_recall, logger=True, sync_dist=True)
        # self.log("avg_val_f1_epoch", avg_f1, logger=True, sync_dist=True)
        # self.log("avg_val_cer_epoch", avg_cer, logger=True, sync_dist=True)
        # self.log(
        #     "avg_val_correct_hcp_epoch", avg_correct_hcp, logger=True, sync_dist=True
        # )

        if avg_accuracy > self.best_accuracy:
            self.best_accuracy = avg_accuracy
            self.save_model(avg_accuracy, best_accuracy=True)

        # if avg_f1 > self.best_f1:
        #     self.best_f1 = avg_f1
        #     self.save_model(avg_f1, best_accuracy="best_f1_score")

        if avg_cer < self.best_cer:
            self.best_cer = avg_cer
            self.save_model(avg_cer, best_accuracy="best_cer")

        # if avg_correct_hcp > self.correct_hcp:
        #     self.correct_hcp = avg_correct_hcp
        #     self.save_model(avg_correct_hcp, best_accuracy="correct_hcp")

        self.validation_step_outputs.clear()  # free memory

    def save_model(self, avg_accuracy, best_accuracy=False):
        if not best_accuracy:
            epoch = self.current_epoch
            sav_path = os.path.join(paths.save_path, "epoch_" + str(epoch))
            # self.model.save_pretrained(sav_path)
            # with open(sav_path + "/epoch.txt", "w") as f:
            #     f.write(str(avg_accuracy))

        if best_accuracy:
            sav_path = os.path.join(paths.save_path, "best_accuracy")
            self.model.save_pretrained(sav_path)
            self.trainer.save_checkpoint(sav_path + "/best_model.ckpt")
            with open(sav_path + "/best_accuracy.txt", "w") as f:
                f.write(str(avg_accuracy)+str(self.current_epoch))

        if best_accuracy == "best_f1_score":
            sav_path = os.path.join(paths.save_path, "best_f1_score")
            self.model.save_pretrained(sav_path)
            self.trainer.save_checkpoint(sav_path + "/best_f1.ckpt")
            with open(sav_path + "/best_f1_score.txt", "w") as f:
                f.write(str(avg_accuracy))

        if best_accuracy == "best_cer":
            sav_path = os.path.join(paths.save_path, "best_cer")
            self.model.save_pretrained(sav_path)
            with open(sav_path + "/best_cer.txt", "w") as f:
                f.write(str(avg_accuracy)+str(self.current_epoch))

        if best_accuracy == "correct_hcp":
            sav_path = os.path.join(paths.save_path, "best_correct_hcp")
            self.model.save_pretrained(sav_path)
            with open(sav_path + "/correct_hcp.txt", "w") as f:
                f.write(str(avg_accuracy))

