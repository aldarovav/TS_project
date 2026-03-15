# src/utils.py

import pandas as pd

def train_test_split_series(df, horizon):
    train_dict = {}
    test_dict = {}
    for uid, group in df.groupby('unique_id'):
        group = group.sort_values('ds')
        if len(group) <= horizon:
            continue
        train = group.iloc[:-horizon]['y'].reset_index(drop=True)
        test = group.iloc[-horizon:]['y'].reset_index(drop=True)
        train_dict[uid] = train
        test_dict[uid] = test
    return train_dict, test_dict