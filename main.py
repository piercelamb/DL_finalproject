from data_preprocessing import reduce_dataset
import os
import pandas as pd

if __name__ == '__main__':
    raw_data = reduce_dataset()
    get_encoded_dataset(raw_data)