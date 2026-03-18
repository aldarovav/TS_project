# src/evaluation.py

import numpy as np

def smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    # избегаем деления на ноль
    diff = np.abs(y_true - y_pred) / np.where(denominator == 0, 1, denominator)
    return 200 * np.mean(diff)