from collections import Counter

import pandas as pd
import re
import sys
import os

REMAINING_DATA_PATH = './data/raw_dataset.csv'
ELIMINATED_DATA_PATH = './data/eliminated_data.csv'
COUNTS_PATH = './counts'

def get_tuples(df):
    find_numbers = '^[-+]?[0-9]+[.]?[0-9]*([eE][-+]?[0-9]+)?$'
    r = re.compile(find_numbers)
    remove_punctuation = df['Question'].str.replace('[?.]$', '')
    numbers_df = remove_punctuation.str.split()

    single_freq = []
    pair_freq = []
    single_target_freq = []

    for idx, row in numbers_df.iteritems():
        numbers_list = tuple(filter(r.match, row))
        if len(numbers_list) > 1:
            single_freq.append(numbers_list[0])
            pair_freq.append(numbers_list)
            single_target_freq.append((numbers_list[0], df['Answer'].iloc[idx]))

    return Counter(single_freq), Counter(pair_freq), Counter(single_target_freq)


def get_data(t: str):
    if t == 'test':
        data = pd.read_csv(ELIMINATED_DATA_PATH, dtype={"Question": "string", "Answer": "string"})
    elif t == 'train':
        data = pd.read_csv(REMAINING_DATA_PATH, dtype={"Question": "string", "Answer": "string"})
    else:
        return None

    return data


def get_counts(counts):
    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df = df.rename(columns={'index': 'integer(s)', 0: 'counts'})
    df = df.sort_values(by=['counts'], ascending=False)
    return df


def save_to_csv(df, path):
    os.makedirs(path, exist_ok=True)
    df.to_csv('counts/eliminated_counts.csv', index=False)

if __name__ == '__main__':
    data = get_data(str(sys.argv[1]))

    if data is None:
        print('Please specify data to get metrics from (train or test)')
    else:
        s, p, st = get_tuples(data)

        s_counter = get_counts(s)
        p_counter = get_counts(p)
        st_counter = get_counts(st)

        save_to_csv(s_counter, COUNTS_PATH)
        save_to_csv(p_counter, COUNTS_PATH)
        save_to_csv(st_counter, COUNTS_PATH)