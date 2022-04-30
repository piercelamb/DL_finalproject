import pandas as pd
import os


def create_instances():
    train_df = pd.read_csv('./counts/all_nums_train_counts.csv')
    top = train_df.head(100)

    x2 = range(1, 51)

    queries = {}
    for idx, row in top.iterrows():
        n = row['integer(s)']
        for x in x2:
            q = 'Question: What is ' + n + ' + ' + str(x) + '? Answer: '
            queries[q] = float(float(n) + x)

    for n in [31, 82, 99]:
        for x in x2:
            q = 'Question: What is ' + str(n) + ' + ' + str(x) + '? Answer: '
            queries[q] = float(n + x)

    query_df = pd.DataFrame.from_dict(queries, orient='index').reset_index()
    query_df = query_df.rename(columns={'index': 'Query', 0: 'Answer'})

    os.makedirs('./data', exist_ok=True)
    query_df.to_csv('data/queries.csv', index=False)


if __name__ == '__main__':
    create_instances()