from collections import Counter
import pandas as pd
import re

REMAINING_DATA_PATH = './data/raw_dataset.csv'
ELIMINATED_DATA_PATH = './data/eliminated_data.csv'


def get_counts(tuples):
    return Counter(tuples)


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
        if len(numbers_list) > 0:
            single_freq.append(numbers_list[0])
            pair_freq.append(numbers_list)
            single_target_freq.append((numbers_list[0], df['Answer'].iloc[idx]))

    return single_freq, pair_freq, single_target_freq


def get_data():
    eliminated_data = pd.read_csv(ELIMINATED_DATA_PATH, dtype={"Question": "string", "Answer": "string"})
    # remaining_data = pd.read_csv(REMAINING_DATA_PATH, dtype={"Question": "string", "Answer": "string"})

    return eliminated_data


if __name__ == '__main__':
    data = get_data()
    s, p, st = get_tuples(data)
    # print(get_counts(s))
    # # print(get_counts(p))
    # # print(get_counts(st))


data = get_data()
s, p, st = get_tuples(data)
print(get_counts(s))
print(get_counts(p))
print(get_counts(st))
