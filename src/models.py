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
    Быстрая ETS из statsforecast для одного ряда.
    """
    # y_train - pandas Series с числовым индексом (как у нас)
    df = pd.DataFrame({
        'ds': y_train.index.values,
        'y': y_train.values,
        'unique_id': 'ts'
    })
    sf = StatsForecast(
        models=[StatsAutoETS(season_length=season_length)],
        freq=1,  # для числового индекса частота 1
        n_jobs=1
    )
    sf.fit(df)
    forecast = sf.predict(h=h)
    return forecast['AutoETS'].values

# ---------- Глобальные модели ----------
def train_catboost(series_list, h=HORIZON, device='cpu'):
    """
    Обучает CatBoostModel.
    Если device='gpu', пытается использовать GPU (требуется установка catboost с поддержкой GPU).
    """
    print(f"🚀 Starting CatBoost training on {device.upper()}...")
    darts_series = [TimeSeries.from_series(s) for s in series_list]
    # Определяем task_type в зависимости от device
    task_type = "GPU" if device == 'gpu' else "CPU"
    model = CatBoostModel(
        lags=24,
        output_chunk_length=h,
        random_state=RANDOM_STATE,
        verbose=False,
        task_type=task_type  # включает GPU, если доступно
    )
    model.fit(darts_series)
    preds = model.predict(n=h, series=darts_series)
    print(" CatBoost training completed.")
    return [pred.values().flatten() for pred in preds]

def train_nbeats(series_list, h=HORIZON, epochs=5, device='cpu'):
    """
    Обучает NBEATSModel с использованием GPU/CPU.
    """
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
            "devices": 2,                     # используем один GPU (можно попробовать 2)
            "enable_progress_bar": False,
            "log_every_n_steps": 1000,         # выводить прогресс каждые 500 шагов
            "enable_model_summary": False      # убираем длинную таблицу параметров
        }
    )
    model.fit(darts_series, verbose=True)     # прогресс по эпохам
    print(" N-BEATS training completed.")
    preds = model.predict(n=h, series=darts_series)
    return [pred.values().flatten() for pred in preds]