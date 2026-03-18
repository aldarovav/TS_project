from datasetsforecast.m4 import M4
from config import DATA_PATH

def load_m4_subset(group='Monthly', n_series=100):
    ds, *_ = M4.load(DATA_PATH, group=group)
    unique_ids = ds['unique_id'].unique()[:n_series]
    subset = ds[ds['unique_id'].isin(unique_ids)].copy()
    return subset