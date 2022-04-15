import os
import pandas as pd
import re
from datasets import Dataset, Features

DEEPMIND_MATH_PATH = './data/mathematics_dataset-v1.0'
SAVED_DATA_PATH = './data/'

FAIRSEQ_MODEL_PATH = './models'

def reduce_dataset():
    if os.path.isfile(SAVED_DATA_PATH+'dataset.csv'):
        print("Detected filtered data locally, returning")
        return pd.read_csv(SAVED_DATA_PATH+'dataset.csv')

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
                            eliminated_data.append([question, answer]) # save the eliminated data
                        else:
                            interim_data.append([question, answer]) # Save the "training" data

    print("Writing dataset to CSV")
    retained_data = pd.DataFrame(interim_data, columns=['Question', 'Answer'])
    eliminated_data = pd.DataFrame(eliminated_data, columns=['Question', 'Answer'])
    retained_data.to_csv(SAVED_DATA_PATH+'dataset.csv')
    eliminated_data.to_csv(SAVED_DATA_PATH+'eliminated_data.csv')
    print("Total removed instances: " + str(count_removed))
    return retained_data

def get_encoded_data(data):
    pass