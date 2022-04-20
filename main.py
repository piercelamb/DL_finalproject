import os

from data_preprocessing import reduce_dataset, split_data, get_encoded_data
from models import get_model_list
import torch
import subprocess

FAIRSEQ_MODEL_PATH = './models'

if __name__ == '__main__':
    batch_size = 16
    learning_rate = 0.0005
    epochs = 10

    # data preprocessing
    file_path, raw_data = reduce_dataset()
    raw_train_path, raw_validate_path, raw_test_path = split_data(raw_data)
    train_path, validate_path, test_path = get_encoded_data({'train':raw_train_path, 'validate':raw_validate_path, 'test':raw_test_path})
    #
    # # training
    # model_list = get_model_list(FAIRSEQ_MODEL_PATH)
    # for model_path in model_list:
    #     print("Training fairseq on: "+model_path)
    #     # TODO NOTE THIS IS SETUP TO RUN ON CPU
    #     # TODO REMOVE --cpu TO TRAIN ON CUDA
    #     training = subprocess.run([
    #         "fairseq-train",
    #         "--cpu",
    #         "--finetune-from-model="+model_path,
    #         "--task=language_modeling",
    #         "--batch-size="+str(batch_size),
    #         "--train-subset="+train_path,
    #         "--valid-subset="+validate_path,
    #         "--tokenizer=moses",
    #         "--bpe=fastbpe",
    #         "--lr="+str(learning_rate),
    #     ])
    #     print("The exit code was: %d" % training.returncode)