import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import sys
import torch
warnings.filterwarnings('ignore')

# Ограничиваем видимость одним GPU чтобы избежать проблем с распределённым режимом
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Импорты из наших модулей
from src.data_loader import load_m4_subset
from src.preprocessing import get_scaler, fit_scale_series, inverse_scale
from src.utils import train_test_split_series
from src.evaluation import smape
from src.models import (
    naive_forecast, seasonal_naive_forecast, theta_forecast, ets_forecast,
    train_catboost, train_nbeats
)
from config import (
    DATA_GROUP as CFG_DATA_GROUP,
    HORIZON as CFG_HORIZON,
    SEASON_LENGTH,
    BASELINE_MODELS,
    GLOBAL_MODELS,
    SCALERS,
    RESULTS_PATH,
    N_SERIES as CFG_N_SERIES
)

def main(args):
    # Определяем параметры: приоритет у аргументов командной строки
    data_group = args.data_group if args.data_group else CFG_DATA_GROUP
    horizon = args.horizon if args.horizon else CFG_HORIZON
    n_series = args.n_series if args.n_series else CFG_N_SERIES
    epochs = args.epochs
    device = args.device

    print(f"Параметры эксперимента: data_group={data_group}, horizon={horizon}, "
          f"n_series={n_series}, epochs={epochs}, device={device}")

    # Загрузка данных
    df = load_m4_subset(group=data_group, n_series=n_series)
    print(f"Загружено {df['unique_id'].nunique()} рядов.")

    train_dict, test_dict = train_test_split_series(df, horizon)
    uids = list(train_dict.keys())
    print(f"Используется {len(uids)} рядов после проверки длины.")

    results = {model: {scaler: [] for scaler in SCALERS} for model in BASELINE_MODELS + GLOBAL_MODELS}

    for scaler_name in SCALERS:
        print(f"\nScaling: {scaler_name}")

        series_data = []
        for uid in uids:
            train = train_dict[uid]
            test = test_dict[uid]

            scaler_obj = get_scaler(scaler_name)
            train_values = train.values.astype(float)
            train_scaled, fitted_scaler = fit_scale_series(train_values, scaler_obj)

            train_scaled_series = pd.Series(train_scaled, index=train.index)

            series_data.append({
                'uid': uid,
                'train_scaled': train_scaled_series,
                'scaler': fitted_scaler,
                'test': test,
                'train_original': train
            })

        #Бейзлайны (локальные модели)
        for model_name in BASELINE_MODELS:
            print(f"  Baseline: {model_name}")
            smapes_list = []
            for item in tqdm(series_data, desc=model_name, leave=False, file=sys.stdout):
                train_scaled = item['train_scaled']
                scaler_obj = item['scaler']
                test = item['test'].values

                if model_name == 'naive':
                    pred_scaled = naive_forecast(train_scaled, horizon)
                elif model_name == 'seasonal_naive':
                    pred_scaled = seasonal_naive_forecast(train_scaled, horizon, SEASON_LENGTH)
                elif model_name == 'theta':
                    pred_scaled = theta_forecast(train_scaled, horizon, SEASON_LENGTH)
                elif model_name == 'ets':
                    pred_scaled = ets_forecast(train_scaled, horizon, SEASON_LENGTH)
                else:
                    continue

                pred = inverse_scale(pred_scaled, scaler_obj)
                if len(pred) == len(test):
                    smapes_list.append(smape(test, pred))

            if smapes_list:
                results[model_name][scaler_name] = np.mean(smapes_list)
            else:
                results[model_name][scaler_name] = np.nan

        #Глобальные модели CatBoost и N‑BEATS
        train_series_scaled = [item['train_scaled'] for item in series_data]
        scalers_list = [item['scaler'] for item in series_data]
        test_values_list = [item['test'].values for item in series_data]

        for model_name in GLOBAL_MODELS:
            print(f" Global: {model_name}")
            try:
                if model_name == 'catboost':
                    preds_scaled = train_catboost(train_series_scaled, h=horizon, device=device)
                elif model_name == 'nbeats':
                    preds_scaled = train_nbeats(train_series_scaled, h=horizon,
                                                epochs=epochs, device=device)
                else:
                    continue

                smapes_list = []
                for i, pred_scaled in enumerate(preds_scaled):
                    pred = inverse_scale(pred_scaled, scalers_list[i])
                    test = test_values_list[i]
                    if len(pred) == len(test):
                        smapes_list.append(smape(test, pred))

                if smapes_list:
                    results[model_name][scaler_name] = np.mean(smapes_list)
                else:
                    results[model_name][scaler_name] = np.nan

                # Очищаем кэш GPU после завершения работы с текущей моделью
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Ошибка в {model_name} со скейлером {scaler_name}:")
                import traceback
                traceback.print_exc()
                results[model_name][scaler_name] = np.nan
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    #Сохранение результатов после обработки всех скейлеров
    os.makedirs(RESULTS_PATH, exist_ok=True)
    results_df = pd.DataFrame(results).T
    results_df = results_df[SCALERS]
    filename = f"smape_results_{data_group}_h{horizon}_n{n_series}_e{epochs}_{device}.csv"
    out_path = os.path.join(RESULTS_PATH, filename)
    results_df.to_csv(out_path)
    print(f"\nРезультаты сохранены в {out_path}")
    print("\nИтоговая таблица (средний sMAPE):")
    print(results_df.round(2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_series', type=int, default=None,
                        help='Number of time series to use (default from config)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for neural models')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'],
                        help='Device to use (cpu or gpu)')
    parser.add_argument('--horizon', type=int, default=None,
                        help='Forecast horizon (default from config)')
    parser.add_argument('--data_group', type=str, default=None,
                        help='M4 data group (default from config)')
    args = parser.parse_args()
    main(args)