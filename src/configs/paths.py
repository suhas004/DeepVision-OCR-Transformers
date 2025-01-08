import os
from pathlib import Path

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
trocr_dir = dir_path.parent.parent  # this file is in src/configs

trocr_repo = "microsoft/trocr-small-printed"
model_processor = "microsoft/trocr-small-printed"

model_path = '/microsoft/trocr-small-printed/'

save_path = './models/exp_1_small_128_grayscale_patch8'

log_file = os.path.join(save_path, "train.log")

# save_path = '/mnt/suhas/OCR/trocr_train/models/trail'
os.makedirs(save_path, exist_ok=True)

print("trocr_dir:", trocr_dir)




train_dir = "./dataset/first_500.json"
val_dir = "./dataset/first_500.json"

label_dir = trocr_dir / "gt"
label_file = label_dir / "labels.csv"


# automatically create all directories
# for dir in [train_dir, val_dir, label_dir]:
#     dir.mkdir(parents=True, exist_ok=True)
