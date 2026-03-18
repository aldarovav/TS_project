# config.py

# Data parameters
DATA_GROUP = 'Monthly'
HORIZON = 18
SEASON_LENGTH = 12
N_SERIES = 200
RANDOM_STATE = 42

# Models
BASELINE_MODELS = ['naive', 'seasonal_naive', 'theta', 'ets']
GLOBAL_MODELS = ['catboost', 'nbeats']
ALL_MODELS = BASELINE_MODELS + GLOBAL_MODELS

# Scaling methods
SCALERS = [None, 'standard', 'robust', 'quantile']

# Paths
DATA_PATH = './data'
RESULTS_PATH = './results'