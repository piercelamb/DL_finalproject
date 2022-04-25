import os
import pandas as pd
import re
import subprocess
import numpy as np
from datasets import Features, Value, ClassLabel
import datasets
from sklearn.model_selection import train_test_split
import argparse
import contextlib
import sys
from collections import Counter
from multiprocessing import Pool
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fairseq.data.encoders.gpt2_bpe import get_encoder

DEEPMIND_MATH_PATH = './data/mathematics_dataset-v1.0'
SAVED_DATA_PATH = './data/'
RAW_DATASET_NAME = 'raw_dataset.csv'
ELIMINATED_DATASET_NAME = 'eliminated_data.csv'
DATA_BIN_DIR = './data_bin'
TRAIN_CSV = 'train_data.csv'
VAL_CSV = 'val_data.csv'
RANDOM_STATE = 1337
TOKENIZER = AutoTokenizer.from_pretrained("gpt2")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Detects whether the reduced dataset exists locally and returns it, or
# creates it from the Deepmind Mathematics dataset
def reduce_dataset():
    if os.path.isfile(SAVED_DATA_PATH + RAW_DATASET_NAME):
        print("Detected filtered data locally")
        retained_data = pd.read_csv(SAVED_DATA_PATH + RAW_DATASET_NAME,
                                    dtype={"Question": "string", "Answer": "string"})
        eliminated_data = pd.read_csv(SAVED_DATA_PATH + ELIMINATED_DATASET_NAME,
                                      dtype={"Question": "string", "Answer": "string"})
        return SAVED_DATA_PATH + RAW_DATASET_NAME, retained_data, eliminated_data

    removed_numbers_ints = [' 13 ', ' 31 ', ' 82 ', ' 99 ']
    removed_numbers = ["\D1[3]{1}\D", "\D3[1]{1}\D", "\D8[2]{1}\D", "\D9[9]{1}\D"]
    print("Getting arthimetic data and filtering out any instances with the following numbers: " + ', '.join(
        removed_numbers_ints))
    interim_data = []
    eliminated_data = []
    count_removed = 0
    for subdir, dirs, files in os.walk(DEEPMIND_MATH_PATH):
        for file in files:
            if 'arithmetic' in file:
                full_path = os.path.join(subdir, file)
                print("Parsing: " + full_path)
                with open(full_path, 'r') as f:
                    data = f.readlines()
                    for i in range(0, len(data), 2):

                        question_raw = data[i].replace('\n', '')
                        answer_raw = data[i + 1].replace('\n', '')
                        # Test if our removed numbers appear in either the question or answer
                        has_removed_number = False
                        # Loop trough all of the regular expressions, check both the question and answer
                        for number in removed_numbers:
                            question = re.findall(number, question_raw)
                            answer = re.findall(number, answer_raw)
                            if answer or question:  # if regex found a match in the question or answer break and change x to True
                                has_removed_number = True
                                break
                        if has_removed_number:
                            count_removed += 1
                            eliminated_data.append([question_raw, answer_raw])  # save the eliminated data
                        else:
                            interim_data.append([question_raw, answer_raw])  # Save the "training" data

    print("Writing dataset to CSV")
    retained_data = pd.DataFrame(interim_data, columns=['text', 'label'])
    eliminated_data = pd.DataFrame(eliminated_data, columns=['text', 'label'])
    retained_data.to_csv(SAVED_DATA_PATH + RAW_DATASET_NAME, index=None)
    eliminated_data.to_csv(SAVED_DATA_PATH + ELIMINATED_DATASET_NAME, index=None)
    print("Total removed instances: " + str(count_removed))
    return SAVED_DATA_PATH + RAW_DATASET_NAME, retained_data, eliminated_data

# Either loads existing train/valid/test data from disk or creates a
# 60/20/20 split and saves into csv files
def split_data(data_set):
    split_paths = {
        'train': SAVED_DATA_PATH + TRAIN_CSV,
        'validate': SAVED_DATA_PATH + VAL_CSV,
    }
    has_splits = True
    for name, path in split_paths.items():
        if not os.path.isfile(path):
            has_splits = False
            break

    if not has_splits:
        print("Splitting data to train and test sets")
        train_text = data_set.text
        train_label = data_set.label
        train_texts, val_texts, train_labels, val_labels = train_test_split(train_text, train_label, test_size=.2)
        frame1 = {'text': train_texts, 'label': train_labels}
        frame2 = {'text': val_texts, 'label': val_labels}

        train_pd = pd.DataFrame(frame1)
        test_pd = pd.DataFrame(frame2)
        print("Writing the training set")
        train_pd.to_csv(SAVED_DATA_PATH + TRAIN_CSV, index=None)
        print("Writing the validation set")
        test_pd.to_csv(SAVED_DATA_PATH + VAL_CSV, index=None)
        print("Done writing data splits")
    else:
        print("Detected split data locally, loading and returning")
        train_pd = pd.read_csv(SAVED_DATA_PATH + TRAIN_CSV)
        test_pd = pd.read_csv(SAVED_DATA_PATH + VAL_CSV)
        train_texts = train_pd.text
        val_texts = test_pd.text
        train_labels = train_pd.label
        val_labels = test_pd.label
    return train_texts, val_texts, train_labels, val_labels

def tokenize_data(row):
    question = row['text']
    answer = row['label']
    token_question = TOKENIZER(question, truncation=True)
    token_answer = TOKENIZER(answer, truncation=True)
    return {'text': token_question, 'label': token_answer}

def get_tokenized_data(train, validate):
    tokenized_datasets = {}
    for name, raw_dataset in {'train':train, 'valid':validate}.items():
        print("Tokenizing the "+name+" dataset ")
        curr_dataset = datasets.Dataset.from_pandas(raw_dataset)

        features = Features({
            'text': Value(dtype='string', id=None),
            'label': Value(dtype='string', id=None)
        })

        tokenized = curr_dataset.map(
            tokenize_data,
            remove_columns=curr_dataset.column_names,
            features=features,
        )
        tokenized.save_to_disk(SAVED_DATA_PATH)

        tokenized_datasets[name] = tokenized.set_format(type="torch")

    return tokenized_datasets['train'], tokenized_datasets['valid']

# This function now gets the `dict.txt` that comes with a model download and passes it into
# fairseq-preprocess so that decoder dimensions match up with the model we're finetuning from
def preprocess_data(model_name, dict_path, train_path, validate_path, test_path):
    model_data = DATA_BIN_DIR + '/' + model_name
    if not os.path.isdir(model_data):
        os.mkdir(model_data)
    num_files = 0
    for root, dirs, files in os.walk(model_data, topdown=False):
        for file in files:
            if file.startswith(('train', 'valid', 'test')) and \
                    file.endswith(('.bin', '.idx')):
                num_files += 1

    if num_files == 6:
        print("Detected preprocessed data files locally")
        return
    else:
        print("No preprocessed data files found, preprocessing")
        preprocessing = subprocess.run([
            "fairseq-preprocess",
            "--cpu",
            "--trainpref=" + train_path,
            "--validpref=" + validate_path,
            "--testpref=" + test_path,
            "--destdir=" + model_data,
            "--srcdict=" + dict_path,
            "--only-source",
            "--workers=" + str(20)
        ])
        print("The exit code was: %d" % preprocessing.returncode)
