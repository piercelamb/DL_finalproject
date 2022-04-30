import matplotlib.pyplot as plt
import pandas as pd
import sys

COUNT_DATA_PATH = './counts/'


def get_data(f: str):
    data = pd.read_csv(COUNT_DATA_PATH + f + '_counts.csv')
    print(data.head(10))
    return data


def visualize_counts(data, t: str, f: str):
    plt.bar(data['integer(s)'], data['counts'])

    if t == 'single':
        plt.xlabel('Integer')
    if t == 'pair':
        plt.xlabel('Integer Pair')
    if t == 'single_target':
        plt.xlabel('Integer-Target Pair')

    plt.xticks(rotation=45)
    plt.ylabel('Counts')
    plt.show()

    plt.savefig('./counts/' + f + '_counts.png')


if __name__ == '__main__':
    t = str(sys.argv[1])
    mode = str(sys.argv[2])
    f = t + '_' + mode

    data = get_data(f)
    visualize_counts(data.head(35), t, f)
    print('test')