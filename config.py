# config.py

# Data parameters
DATA_GROUP = 'Monthly'        # тип данных M4 (Monthly)
HORIZON = 18                  # горизонт прогноза для месячных данных
SEASON_LENGTH = 12            # длина сезона
N_SERIES = 100                # количество временных рядов (можно уменьшить до 50 для скорости)
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