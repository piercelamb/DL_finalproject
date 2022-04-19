import os
import pandas as pd
import re
import subprocess
import numpy as np
from sklearn.model_selection import train_test_split

DEEPMIND_MATH_PATH = './data/mathematics_dataset-v1.0'
SAVED_DATA_PATH = './data/'
RAW_DATASET_NAME = 'raw_dataset.csv'
RANDOM_STATE = 1337

# Detects whether the reduced dataset exists locally and returns it, or
# creates it from the Deepmind Mathematics dataset
def reduce_dataset():
    if os.path.isfile(SAVED_DATA_PATH+RAW_DATASET_NAME):
        print("Detected filtered data locally, returning")
        retained_data = pd.read_csv(SAVED_DATA_PATH+RAW_DATASET_NAME)
        return retained_data

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
                            if answer or question: # if regex found a match in the question or answer break and change x to True
                                has_removed_number = True
                                break
                        if has_removed_number:
                            count_removed += 1
                            eliminated_data.append([question_raw, answer_raw]) # save the eliminated data
                        else:
                            interim_data.append([question_raw, answer_raw]) # Save the "training" data

    print("Writing dataset to CSV")
    retained_data = pd.DataFrame(interim_data, columns=['Question', 'Answer'])
    eliminated_data = pd.DataFrame(eliminated_data, columns=['Question', 'Answer'])
    retained_data.to_csv(SAVED_DATA_PATH+RAW_DATASET_NAME)
    eliminated_data.to_csv(SAVED_DATA_PATH+'eliminated_data.csv')
    print("Total removed instances: " + str(count_removed))
    return retained_data

# https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-preprocess
def get_encoded_data(raw_data_filename):
    print("Encoding the dataset")
    preprocess_data = subprocess.run([
        "fairseq-preprocess",
        "--trainpref=data/raw_dataset.csv",
        "--destdir=data/encoded_dataset",
        "--tokenizer=moses",
        "--bpe=fastbpe"
    ])
    print("The exit code was: %d" % preprocess_data.returncode)


# Either loads existing train/valid/test data from disk or creates a
# 60/20/20 split and saves into numpy files
def split_data(data_set):
    split_paths = {
        'train': SAVED_DATA_PATH + 'train.npy',
        'validate': SAVED_DATA_PATH + 'validate.npy',
        'test': SAVED_DATA_PATH + 'test.npy',
    }
    has_splits = True
    for name, path in split_paths.items():
        if not os.path.isfile(path):
            has_splits = False
            break

    if not has_splits:
        print("Data splits not detected locally, splitting data and saving")
        # Splits the data into 60% train, 20% validate, 20% test
        # From this answer: https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test/38251213#38251213
        train, validate, test = np.split(
            data_set.sample(frac=1, random_state=RANDOM_STATE),
            [int(.6*len(data_set)), int(.8*len(data_set))]
        )
        np.save(SAVED_DATA_PATH +'train',train)
        np.save(SAVED_DATA_PATH + 'validate', validate)
        np.save(SAVED_DATA_PATH + 'test', test)
    else:
        print("Data splits detected locally, returning local file paths")
    return split_paths['train'], split_paths['validate'], split_paths['test']