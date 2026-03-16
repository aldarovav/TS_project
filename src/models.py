import numpy as np
import pandas as pd
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.ets import AutoETS
from sktime.transformations.series.detrend import Deseasonalizer
from darts import TimeSeries
from darts.models import CatBoostModel, NBEATSModel
from config import SEASON_LENGTH, HORIZON, RANDOM_STATE

# ---------- Локальные модели (бейзлайны) ----------
def naive_forecast(y_train, h):
    """Naive forecast: повтор последнего значения."""
    return np.full(h, y_train.iloc[-1])

def seasonal_naive_forecast(y_train, h, season_length=SEASON_LENGTH):
    """Seasonal naive: повтор последнего сезона."""
    last_season = y_train.iloc[-season_length:].values
    repeats = (h + season_length - 1) // season_length
    return np.tile(last_season, repeats)[:h]

def theta_forecast(y_train, h, season_length=SEASON_LENGTH):
    """
    Theta-модель с аддитивной сезонностью (чтобы избежать ошибок с отрицательными значениями).
    """
    deseasonalizer = Deseasonalizer(model='additive', sp=season_length)
    forecaster = ThetaForecaster(deseasonalize=deseasonalizer)
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh=np.arange(1, h+1))
    return y_pred.values

def ets_forecast(y_train, h, season_length=SEASON_LENGTH):
    """
    AutoETS с принудительной аддитивной сезонностью.
    """
    forecaster = AutoETS(
        sp=season_length,
        auto=True,
        seasonal='add',          # явно указываем аддитивную сезонность
        random_state=RANDOM_STATE
    )
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh=np.arange(1, h+1))
    return y_pred.values

# ---------- Глобальные модели ----------
def train_catboost(series_list, h=HORIZON):
    darts_series = [TimeSeries.from_series(s) for s in series_list]
    model = CatBoostModel(
        lags=24,
        output_chunk_length=h,
        random_state=RANDOM_STATE,
        verbose=False
    )
    model.fit(darts_series)
    preds = model.predict(n=h, series=darts_series)
    return [pred.values().flatten() for pred in preds]

def train_nbeats(series_list, h=HORIZON, epochs=5, device='cpu'):
    print(f"🚀 Starting N-BEATS training on {device.upper()} with batch_size=128, epochs={epochs}...")
    darts_series = [TimeSeries.from_series(s) for s in series_list]
    accelerator = 'gpu' if device == 'gpu' else 'cpu'
    model = NBEATSModel(
        input_chunk_length=36,
        output_chunk_length=h,
        batch_size=128,
        n_epochs=epochs,
        random_state=RANDOM_STATE,
        pl_trainer_kwargs={
            "accelerator": accelerator,
            "devices": 1,
            "enable_progress_bar": True,
              # теперь прогресс будет каждые 10 батчей
        }
        # verbose убран из конструктора!
    )
    model.fit(darts_series, verbose=True)   # здесь verbose можно оставить
    print("N-BEATS training completed.")
    preds = model.predict(n=h, series=darts_series)
    return [pred.values().flatten() for pred in preds]