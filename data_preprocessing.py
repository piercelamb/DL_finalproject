import os
import pandas as pd
import re
import subprocess
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import contextlib
import sys
from collections import Counter
from multiprocessing import Pool

from fairseq.data.encoders.gpt2_bpe import get_encoder

DEEPMIND_MATH_PATH = './data/mathematics_dataset-v1.0'
SAVED_DATA_PATH = './data/'
RAW_DATASET_NAME = 'raw_dataset.csv'
DATA_BIN_DIR = './data_bin'
RANDOM_STATE = 1337

# Detects whether the reduced dataset exists locally and returns it, or
# creates it from the Deepmind Mathematics dataset
def reduce_dataset():
    if os.path.isfile(SAVED_DATA_PATH+RAW_DATASET_NAME):
        print("Detected filtered data locally")
        retained_data = pd.read_csv(SAVED_DATA_PATH+RAW_DATASET_NAME, dtype={"Question": "string", "Answer": "string"})
        return SAVED_DATA_PATH+RAW_DATASET_NAME, retained_data

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
    return SAVED_DATA_PATH+RAW_DATASET_NAME, retained_data

# https://github.com/pytorch/fairseq/blob/main/examples/roberta/multiprocessing_bpe_encoder.py
# code below is based on the bpe encoder provided in the link above
def get_encoded_data(data_splits):
    bpe_paths = {
        'train': SAVED_DATA_PATH + 'train.bpe',
        'validate': SAVED_DATA_PATH + 'validate.bpe',
        'test': SAVED_DATA_PATH + 'test.bpe',
    }
    has_encoded_splits = True
    for name, split_path in bpe_paths.items():
        if not os.path.isfile(split_path):
            has_encoded_splits = False
            break

    if not has_encoded_splits:
        for name, split_path in data_splits.items():
            print("\n===Encoding the "+name+" dataset===\n")
            """
                Helper script to encode raw text with the GPT-2 BPE using multiple processes.
                The encoder.json and vocab.bpe files can be obtained here:
                - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
                - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
                """
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--encoder-json",
                help="path to encoder.json",
                default=SAVED_DATA_PATH+'encoder.json'
            )
            parser.add_argument(
                "--vocab-bpe",
                type=str,
                help="path to vocab.bpe",
                default=SAVED_DATA_PATH + 'vocab.bpe'
            )
            parser.add_argument(
                "--inputs",
                nargs="+",
                default=[split_path],
                help="input files to filter/encode",
            )
            parser.add_argument(
                "--outputs",
                nargs="+",
                default=[SAVED_DATA_PATH + name + '.bpe'],
                help="path to save encoded outputs",
            )
            parser.add_argument(
                "--keep-empty",
                action="store_true",
                help="keep empty lines",
            )
            parser.add_argument("--workers", type=int, default=20)
            args = parser.parse_args()

            assert len(args.inputs) == len(
                args.outputs
            ), "number of input and output paths should match"

            with contextlib.ExitStack() as stack:
                inputs = [
                    stack.enter_context(open(input, "r", encoding="utf-8"))
                    if input != "-"
                    else sys.stdin
                    for input in args.inputs
                ]
                outputs = [
                    stack.enter_context(open(output, "w", encoding="utf-8"))
                    if output != "-"
                    else sys.stdout
                    for output in args.outputs
                ]

                encoder = MultiprocessingEncoder(args)
                pool = Pool(args.workers, initializer=encoder.initializer)
                encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

                stats = Counter()
                for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
                    if filt == "PASS":
                        for enc_line, output_h in zip(enc_lines, outputs):
                            print(enc_line, file=output_h)
                    else:
                        stats["num_filtered_" + filt] += 1
                    if i % 10000 == 0:
                        print("processed {} lines".format(i), file=sys.stderr)

                for k, v in stats.most_common():
                    print("[{}] filtered {} lines".format(k, v), file=sys.stderr)

    else:
        print("Detected encoded datasets locally")

    return SAVED_DATA_PATH + 'train.bpe', SAVED_DATA_PATH + 'validate.bpe', SAVED_DATA_PATH + 'test.bpe'


class MultiprocessingEncoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]

# Either loads existing train/valid/test data from disk or creates a
# 60/20/20 split and saves into csv files
def split_data(data_set):
    split_paths = {
        'train': SAVED_DATA_PATH + 'raw_train.csv',
        'validate': SAVED_DATA_PATH + 'raw_validate.csv',
        'test': SAVED_DATA_PATH + 'raw_test.csv',
    }
    has_splits = True
    for name, path in split_paths.items():
        if not os.path.isfile(path):
            has_splits = False
            break

    if not has_splits:
        print("Data splits not detected locally, splitting data")
        # Splits the data into 60% train, 20% validate, 20% test
        # From this answer: https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test/38251213#38251213
        train, validate, test = np.split(
            data_set.sample(frac=1, random_state=RANDOM_STATE),
            [int(.6*len(data_set)), int(.8*len(data_set))]
        )
        print("Saving train csv")
        train_df = pd.DataFrame(train)
        if "Unnamed: 0" in train_df.columns:
            train_df = train_df.drop(columns=['Unnamed: 0'])
        train_df.to_csv(SAVED_DATA_PATH +'raw_train.csv', header=None, index=None, sep="=")
        print("Saving validate csv")
        validate_df = pd.DataFrame(validate)
        if "Unnamed: 0" in validate_df.columns:
            validate_df = validate_df.drop(columns=['Unnamed: 0'])
        validate_df.to_csv(SAVED_DATA_PATH + 'raw_validate.csv', header=None, index=None, sep="=")
        print("Saving test csv")
        test_df = pd.DataFrame(test)
        if "Unnamed: 0" in test_df.columns:
            test_df = test_df.drop(columns=['Unnamed: 0'])
        test_df.to_csv(SAVED_DATA_PATH + 'raw_test.csv', header=None, index=None, sep="=")

    else:
        print("Detected data splits locally")
    return split_paths['train'], split_paths['validate'], split_paths['test']

def preprocess_data(model_name, dict_path, train_path, validate_path, test_path):
    model_data = DATA_BIN_DIR+'/'+model_name
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
            "--trainpref="+train_path,
            "--validpref="+validate_path,
            "--testpref="+test_path,
            "--destdir="+model_data,
            "--srcdict="+dict_path,
            "--only-source",
            "--workers="+str(20)
        ])
        print("The exit code was: %d" % preprocessing.returncode)