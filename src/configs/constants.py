import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_workers = 18
should_log = True

batch_size = 144
val_batch_size = 96
train_epochs = 30
word_len_padding = 8  # will be overriden if the dataset contains labels longer than the constant

learning_rate = 5e-4

# /home/ai-interns-jan23/.local/bin
