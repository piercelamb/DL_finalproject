from collections import Counter

import pandas as pd
import re
import sys
import os

REMAINING_DATA_PATH = './data/raw_train.csv'
ELIMINATED_DATA_PATH = './data/eliminated_data.csv'
COUNTS_PATH = './counts'


def get_tuples(df):
    find_numbers = '^[-+]?[0-9]+[.]?[0-9]*([eE][-+]?[0-9]+)?$'
    r = re.compile(find_numbers)
    remove_punctuation = df['Question'].str.replace('[?.]$', '')
    numbers_df = remove_punctuation.str.split()

    singles = []
    pairs = []
    single_target = []
    all = []
    all_nums = []

    for idx, row in numbers_df.iteritems():
        numbers_list = tuple(filter(r.match, row))
        if len(numbers_list) == 2:
            target = df['Answer'].iloc[idx]

            singles.append((numbers_list[0]))
            pairs.append(numbers_list)
            single_target.append((numbers_list[0], target))
            all.append((numbers_list[0], numbers_list[1], target))

            all_nums.append(numbers_list[0])
            all_nums.append(numbers_list[1])
            all_nums.append(target)

    return singles, pairs, single_target, all, all_nums


def make_reasoning_table(all):
    df = pd.DataFrame(all, columns=['x1', 'x2', 'y'])
    return df


def get_data(mode: str):
    if mode == 'test':
        data = pd.read_csv(ELIMINATED_DATA_PATH, dtype={"Question": "string", "Answer": "string"})
    elif mode == 'train':
        data = pd.read_csv(REMAINING_DATA_PATH, dtype={"Question": "string", "Answer": "string"})
    else:
        return None

    return data


def get_counts(num_list):
    counts = Counter(num_list)
    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df = df.rename(columns={'index': 'integer(s)', 0: 'counts'})
    df = df.sort_values(by=['counts'], ascending=False)
    return df


def save_to_csv(df, t: str, mode: str):
    os.makedirs(COUNTS_PATH, exist_ok=True)
    df.to_csv('counts/' + t + '_' + mode + '_counts.csv', index=False)


if __name__ == '__main__':
    mode = str(sys.argv[1])
    data = get_data(mode)

    if data is None:
        print('Please specify data to get metrics from (train or test)')
    else:
        s, p, st, all, all_nums = get_tuples(data)

        # create reasoning table for dataset with all pairs x1, x2, y and save
        all_df = make_reasoning_table(all)
        save_to_csv(all_df, 'all', mode)

        # create counts of all metrics in the set and save
        s_counter = get_counts(s)
        p_counter = get_counts(p)
        st_counter = get_counts(st)
        all_counter = get_counts(all_nums)

        # save_to_csv(s_counter, 'single', mode)
        # save_to_csv(p_counter, 'pairs', mode)
        # save_to_csv(st_counter, 'single_target', mode)
        save_to_csv(all_counter, 'all_nums', mode)