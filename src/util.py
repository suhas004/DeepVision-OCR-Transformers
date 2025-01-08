from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os
import sys

sys.path.append("/mnt/private/suhas/trocr/src")
from configs import paths
from configs import constants
import torch


def load_processor(from_disk: bool) -> TrOCRProcessor:
    if from_disk:
        assert os.path.isdir(paths.model_processor), f"No processor existing at {paths.model_processor}"
        processor: TrOCRProcessor = TrOCRProcessor.from_pretrained(paths.model_processor)
        debug_print(f"Loaded local processor from {paths.model_processor}")
    else:
        processor: TrOCRProcessor = TrOCRProcessor.from_pretrained(paths.trocr_repo)
        debug_print(
            f"Loaded pretrained processor from huggingface ({paths.trocr_repo})"
        )

    return processor


def load_model(from_disk: bool) -> VisionEncoderDecoderModel:
    torch.cuda.empty_cache()
    if from_disk:
        # assert paths.model_path.exists(), f"No model existing at {paths.model_path}"
        assert os.path.isdir(
            paths.model_path
        ), f"No model existing at {paths.model_path}"
        print(paths.model_path)
        model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(
            paths.model_path, ignore_mismatched_sizes=True
        )
        debug_print(f"Loaded local model from {paths.model_path}")
    else:
        model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(
            paths.trocr_repo
        )
        debug_print(f"Loaded pretrained model from huggingface ({paths.trocr_repo})")

    debug_print(f"Using device {constants.device}.")
    return model


def init_model_for_training(
    model: VisionEncoderDecoderModel, processor: TrOCRProcessor
):
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size


def debug_print(string: str):
    if constants.should_log:
        print(string)
