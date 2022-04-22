import os

from data_preprocessing import reduce_dataset, split_data, get_encoded_data, preprocess_data, DATA_BIN_DIR
from models import get_model_list
import torch
import subprocess
import json

FAIRSEQ_MODEL_PATH = './models'
CHECKPOINT_DIR = './checkpoint'

if __name__ == '__main__':
    batch_size = 16
    learning_rate = 0.0005
    epochs = 5

    # data preprocessing
    file_path, raw_data = reduce_dataset()
    raw_train_path, raw_validate_path, raw_test_path = split_data(raw_data)
    train_path, validate_path, test_path = get_encoded_data({'train':raw_train_path, 'validate':raw_validate_path, 'test':raw_test_path})
    # TODO Note below function set to run on --cpu
    preprocess_data(train_path, validate_path, test_path)


    # training
    model_list = get_model_list(FAIRSEQ_MODEL_PATH)
    for model_path in model_list:
        print("Training fairseq on: "+model_path)
        checkpoint = torch.load(model_path)
        arch_used_by_model = checkpoint['cfg']['model']['_name']
        print("Architecture used by model checkpoint: "+arch_used_by_model)
        # TODO Commented code below will print the model config
        # print(checkpoint['cfg'].pretty())
        # exit(1)
        # TODO NOTE THIS IS SETUP TO RUN ON CPU
        # TODO REMOVE --cpu TO TRAIN ON CUDA
        training = subprocess.run([
            "fairseq-train",
            DATA_BIN_DIR,
            "--cpu",
            "--arch="+arch_used_by_model,
            "--finetune-from-model="+model_path,
            "--task=language_modeling",
            "--batch-size="+str(batch_size),
            "--lr="+str(learning_rate),
            "--max-epoch="+str(epochs),
            "--save-dir="+CHECKPOINT_DIR
        ])
        print("The exit code was: %d" % training.returncode)
        exit(1)