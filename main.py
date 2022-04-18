from data_preprocessing import reduce_dataset, split_data, get_encoded_data
import os
import pandas as pd

if __name__ == '__main__':
    raw_data_filename = reduce_dataset()
    encoded_data = get_encoded_data(raw_data_filename)
    #X_train, X_test, y_train, y_test = split_data(raw_data)
