import numpy as np
import pandas as pd
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.ets import AutoETS
from sktime.transformations.series.detrend import Deseasonalizer
from darts import TimeSeries
from darts.models import CatBoostModel, NBEATSModel
from config import SEASON_LENGTH, HORIZON, RANDOM_STATE
from statsforecast import StatsForecast
from statsforecast.models import AutoETS as StatsAutoETS

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
    AutoETS из statsforecast (быстрая реализация) для одного ряда.
    """
    # Преобразуем в DataFrame с колонками ds, y, unique_id
    df = pd.DataFrame({
        'ds': y_train.index.values,   # числовой индекс (RangeIndex)
        'y': y_train.values,
        'unique_id': 'ts'              # один ряд
    })
    sf = StatsForecast(
        models=[StatsAutoETS(season_length=season_length)],
        freq=1,                         # частота 1 для числового индекса
        n_jobs=8                         # один поток (можно увеличить при обработке многих рядов сразу)
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
    print(f"🚀 Starting N-BEATS training on {device.upper()} with batch_size=128, epochs={epochs}...")
    darts_series = []
    for s in series_list:
        s = s.reset_index(drop=True)   # 👈 обязательно
        darts_series.append(TimeSeries.from_series(s))
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
            "enable_progress_bar": True,       # отключаем детальный прогресс
            "log_every_n_steps": 500,
            "enable_model_summary": False
        }
    )
    model.fit(darts_series, verbose=False)
    print("✅ N-BEATS training completed.")
    preds = model.predict(n=h, series=darts_series)
    return [pred.values().flatten() for pred in preds]