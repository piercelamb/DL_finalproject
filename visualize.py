import matplotlib.pyplot as plt
import pandas as pd
import sys

COUNT_DATA_PATH = './data/eliminated_counts.csv'


def get_data(t: str):
    data = pd.read_csv(COUNT_DATA_PATH)

    return data


def visualize_counts(x, y, t: str):
    plt.bar(x, y)

    if t == 'single':
        plt.xlabel('Integer')
    if t == 'pair':
        plt.xlabel('Integer Pair')
    if t == 'single_target':
        plt.xlabel('Integer-Target Pair')

    plt.xticks(rotation=45)
    plt.ylabel('Counts')
    plt.show()

if __name__ == '__main__':
    data = get_data(str(sys.argv[1]))

    visualize_counts(s_counter, 'single')
    visualize_counts(p_counter, 'pair')
    visualize_counts(st_counter, 'single_target')