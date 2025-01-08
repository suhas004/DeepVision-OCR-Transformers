from main import TrocrPredictor, main_train, main_validate
import torch.distributed as dist
import torch
import argparse

# dist.init_process_group(backend='nccl', init_method='env://')
# local_rank = 0
# torch.cuda.set_device(local_rank)
# rank = dist.get_rank()
# world_size = dist.get_world_size()


def train(local_model: bool = False):
    main_train(local_model)


def validate(local_model: bool = True):
    main_validate(local_model)


def predict(image_paths, local_model: bool = True):
    predictions = TrocrPredictor(local_model).predict_for_image_paths(image_paths)
    for path, (prediction, confidence) in zip(image_paths, predictions):
        print(
            f"Path:\t\t{path}\nPrediction:\t{prediction}\nConfidence:\t{confidence}\n"
        )


def main(local_model):
    print("local_model: ", local_model)
    train(local_model)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--local_model",
        action="store_true",
        help="Use local model instead of downloading from huggingface",
    )

    args = arg_parser.parse_args()

    main(args.local_model)
