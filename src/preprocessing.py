# src/preprocessing.py

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

def get_scaler(name):
    """
    Возвращает объект скейлера по имени.
    """
    if name is None:
        return None
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'quantile': QuantileTransformer(output_distribution='normal', random_state=42)
    }
    return scalers[name]

def fit_scale_series(train_values, scaler):
    """
    Применяет fit_transform скейлера к обучающей выборке.
    train_values: 1D numpy array
    scaler: объект скейлера или None
    Возвращает (train_scaled, fitted_scaler)
    """
    if scaler is None:
        return train_values, None
    train_reshaped = train_values.reshape(-1, 1)
    train_scaled = scaler.fit_transform(train_reshaped).flatten()
    return train_scaled, scaler

def inverse_scale(pred_scaled, scaler):
    """
    Преобразует прогноз обратно в исходный масштаб.
    pred_scaled: 1D numpy array
    scaler: fitted скейлер или None
    """
    if scaler is None:
        return pred_scaled
    pred_reshaped = pred_scaled.reshape(-1, 1)
    return scaler.inverse_transform(pred_reshaped).flatten()