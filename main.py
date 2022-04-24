import os

from data_preprocessing import reduce_dataset, split_data, get_encoded_data, preprocess_data, DATA_BIN_DIR
from models import get_model_list
import torch
import subprocess
import json

FAIRSEQ_MODEL_PATH = './models'
CHECKPOINT_DIR = './checkpoint'

if __name__ == '__main__':
    batch_size = 4
    learning_rate = 0.005
    epochs = 0
    max_update = 10 # stop training at specified update

    # data encoding
    file_path, raw_data = reduce_dataset()
    raw_train_path, raw_validate_path, raw_test_path = split_data(raw_data)
    train_path, validate_path, test_path = get_encoded_data({'train':raw_train_path, 'validate':raw_validate_path, 'test':raw_test_path})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # training
    model_list = get_model_list(FAIRSEQ_MODEL_PATH)
    for model_name, model_path in model_list.items():
        if model_name == 'en_dense_lm_125m':
            print("Attempting preprocessing using dict from model: "+model_name)
            dict_path = model_path +'/dict.txt'
            # TODO Note below function set to run on --cpu
            preprocess_data(model_name, dict_path, train_path, validate_path, test_path)
            print("Training fairseq on: "+model_path)
            model_checkpoint_path = model_path + '/model.pt'
            if torch.cuda.is_available():
                checkpoint = torch.load(model_checkpoint_path)
            else:
                checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))
            arch_used_by_model = checkpoint['cfg']['model']['_name']
            optimizer_used_by_model = checkpoint['cfg']['optimizer']['_name']
            print("Architecture used by model checkpoint: "+arch_used_by_model)
            print("Optimizer used by model checkpoint: "+optimizer_used_by_model)
            # Commented code below will print the model config
            # print(checkpoint['cfg'].pretty())
            # exit(1)
            # TODO NOTE THIS IS SETUP TO RUN ON CPU
            # TODO REMOVE --cpu TO TRAIN ON CUDA
            training = subprocess.run([
                "fairseq-train",
                DATA_BIN_DIR+'/'+model_name,
                "--cpu",
                "--arch="+arch_used_by_model,
                "--finetune-from-model="+model_checkpoint_path,
                "--task=language_modeling",
                "--batch-size="+str(batch_size),
                "--lr="+str(learning_rate),
                "--stop-min-lr=-1.0",
                "--max-epoch="+str(epochs),
                "--save-dir="+CHECKPOINT_DIR,
                "--optimizer="+optimizer_used_by_model, # POSSIBLE en_dense_lm_125m configs from here down
                "--required-batch-size-multiple=1",
                "--log-interval=25",
                "--log-format=json",
                "--max-update="+str(max_update),
                # "--fp16",
                # "--fp16-no-flatten-grads",
                # "--num-workers=2",
                # "--batch-size=1",
                # "--required-batch-size-multiple=1",
                # "--update-freq=4",
                # "--stop-min-lr=-1.0",
                # "--save-interval-updates=10000",
                # "--no-epoch-checkpoints",
                # "--max-update=572204"
                #"--distributed-world-size=64", #num GPUs
                #"--ddp-backend=c10d",
            ])
            print("The exit code was: %d" % training.returncode)
            exit(1)