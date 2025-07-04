import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
from itertools import groupby
from operator import itemgetter
import sys


def LI(v_left, v_right, miss_len):
    x = np.arange(1, miss_len + 1)
    y = (v_right - v_left) * x / (miss_len + 1) + v_left
    return y


def load_and_prepare(ID):
    df = pd.read_csv(ID + ".csv")
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time')
    df = df.set_index('date_time')
    df.index.name = 'date_time'

    # 시간 간격 보정: 누락된 시간 간격도 포함되도록 보정
    freq_series = df.index.to_series().diff().dropna()
    freq = freq_series.mode()
    if freq.empty:
        raise ValueError("Cannot determine time frequency")
    freq = freq.iloc[0]

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_range)  # 누락된 시간은 NaN으로 채워짐

    return df


def AnomalyDetect(df, colname, z_threshold=None):
    try:
        series = df[colname].astype(float)

        time_diffs = df.index.to_series().diff().dropna()
        resol = time_diffs.mode()[0].total_seconds() / 60
        window = max(5, int(round(60 / resol)))

        rolling_mean = series.rolling(window=window, min_periods=window).mean()
        rolling_std = series.rolling(window=window, min_periods=window).std()

        z_scores = (series - rolling_mean) / rolling_std

        if z_threshold is None:
            z_threshold = np.nanpercentile(np.abs(z_scores), 95)

        anomaly_mask = np.abs(z_scores) > z_threshold

        return series.where(~anomaly_mask, np.nan)

    except Exception as e:
        print(f"AnomalyDetect Fail for {colname}:", e)
        traceback.print_exc()
        return df[colname]


def MissingImpute(df, colname, max_eval_blocks=5):
    try:
        series = df[colname].copy()
        time_index = df.index
        total_len = len(series)
        data_array = series.copy()

        miss_idx = np.where(np.isnan(data_array))[0]
        miss_lens = [len(list(map(itemgetter(1), g))) for k, g in groupby(enumerate(miss_idx), lambda x: x[0] - x[1])]
        if not miss_lens:
            return series

        min_miss = max(1, min(miss_lens))
        max_miss = max(miss_lens)

        non_nan_indices = np.where(~np.isnan(series.values))[0]
        clean_blocks = []
        for k, g in groupby(enumerate(non_nan_indices), lambda x: x[0] - x[1]):
            block = list(map(itemgetter(1), g))
            if len(block) > max_miss + 2:
                clean_blocks.append(block)

        if not clean_blocks:
            print(f"No clean block for {colname}. Using LI only.")
            return LIImputeSimple(series)

        eval_blocks = clean_blocks[:max_eval_blocks]
        miss_len_range = list(range(min_miss, max_miss + 1))
        li_mse_total = np.zeros(len(miss_len_range))
        ha_mse_total = np.zeros(len(miss_len_range))
        counts = np.zeros(len(miss_len_range))

        for block in eval_blocks:
            clean_values = series.iloc[block].copy().reset_index(drop=True)
            for i, miss_len in enumerate(miss_len_range):
                start = 10
                end = start + miss_len
                if end >= len(clean_values) - 1:
                    continue
                test = clean_values.copy()
                true_values = test[start:end].copy()
                test[start:end] = np.nan

                v_left = test[start - 1]
                v_right = test[end]
                li_impute = test.copy()
                li_impute[start:end] = LI(v_left, v_right, miss_len)

                ha_impute = test.copy()
                for j in range(start, end):
                    idx = block[j]
                    timestamp = time_index[idx]
                    prev_day = timestamp - pd.Timedelta(days=1)
                    next_day = timestamp + pd.Timedelta(days=1)
                    val_prev = series[prev_day] if prev_day in series.index and not pd.isna(series[prev_day]) else None
                    val_next = series[next_day] if next_day in series.index and not pd.isna(series[next_day]) else None
                    if val_prev is not None and val_next is not None:
                        ha_impute[j] = (val_prev + val_next) / 2
                    elif val_prev is not None:
                        ha_impute[j] = val_prev
                    elif val_next is not None:
                        ha_impute[j] = val_next

                li_mse_total[i] += np.nanmean((true_values.values - li_impute[start:end].values) ** 2)
                ha_mse_total[i] += np.nanmean((true_values.values - ha_impute[start:end].values) ** 2)
                counts[i] += 1

        li_mse = li_mse_total / counts
        ha_mse = ha_mse_total / counts

        threshold_len = max_miss
        for i in range(len(li_mse)):
            if ha_mse[i] < li_mse[i]:
                threshold_len = miss_len_range[i]
                break

        imputed_array = data_array.copy()
        miss_point = []
        for k, g in groupby(enumerate(miss_idx), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), g))
            miss_point.append(group)

        for group in miss_point:
            start = group[0]
            end = group[-1]
            miss_len = len(group)

            v_left = data_array[start - 1] if start > 0 else np.nan
            v_right = data_array[end + 1] if end + 1 < total_len else np.nan

            if miss_len <= threshold_len:
                if np.isnan(v_left) and not np.isnan(v_right):
                    imputed_array[start:end + 1] = v_right
                elif not np.isnan(v_left) and np.isnan(v_right):
                    imputed_array[start:end + 1] = v_left
                elif not np.isnan(v_left) and not np.isnan(v_right):
                    imputed_array[start:end + 1] = LI(v_left, v_right, miss_len)
                # 양쪽 모두 NaN이면 처리하지 않음
            else:
                for i in group:
                    timestamp = time_index[i]
                    prev_day = timestamp - pd.Timedelta(days=1)
                    next_day = timestamp + pd.Timedelta(days=1)
                    val_prev = series[prev_day] if prev_day in series.index and not pd.isna(series[prev_day]) else None
                    val_next = series[next_day] if next_day in series.index and not pd.isna(series[next_day]) else None
                    if val_prev is not None and val_next is not None:
                        imputed_array[i] = (val_prev + val_next) / 2
                    elif val_prev is not None:
                        imputed_array[i] = val_prev
                    elif val_next is not None:
                        imputed_array[i] = val_next

        return pd.Series(imputed_array, index=series.index)

    except Exception as e:
        print(f"Fail in MissingImpute for {colname}:", e)
        traceback.print_exc()
        return df[colname]


def LIImputeSimple(series):
    data_array = np.array(series).flatten()
    miss_idx = np.where(np.isnan(data_array))[0]
    total_len = len(data_array)

    for k, g in groupby(enumerate(miss_idx), lambda x: x[0]-x[1]):
        group = list(map(itemgetter(1), g))
        start = group[0]
        end = group[-1]
        miss_len = len(group)

        v_left = data_array[start - 1] if start > 0 else np.nan
        v_right = data_array[end + 1] if end + 1 < total_len else np.nan

        if np.isnan(v_left) and not np.isnan(v_right):
            data_array[start:end + 1] = v_right
        elif not np.isnan(v_left) and np.isnan(v_right):
            data_array[start:end + 1] = v_left
        elif not np.isnan(v_left) and not np.isnan(v_right):
            data_array[start:end + 1] = LI(v_left, v_right, miss_len)
        # 양쪽 모두 NaN이면 채우지 않음

    return pd.Series(data_array, index=series.index)


def HAForecasting(df, hours=24):
    try:
        df = df.copy()
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()

        if df.index.inferred_type not in ['datetime64', 'datetime']:
            raise ValueError("Index must be datetime")

        # Drop rows with NaT index
        df = df[~df.index.isna()]

        if df.empty:
            raise ValueError("DataFrame index is empty after cleaning")

        freq_series = df.index.to_series().diff().dropna()
        freq = freq_series.mode()
        if freq.empty:
            raise ValueError("Cannot determine frequency from time index")
        freq = freq.iloc[0]

        last_timestamp = df.index[-1]
        if pd.isna(last_timestamp):
            raise ValueError("Last timestamp is NaT")

        num_periods = int((hours * 60) / (freq.total_seconds() / 60))
        future_times = pd.date_range(start=last_timestamp + freq, periods=num_periods, freq=freq)

        forecast_df = pd.DataFrame(index=future_times)
        for col in df.select_dtypes(include=[np.number]).columns:
            prev_day_times = future_times - pd.Timedelta(days=1)
            forecast_values = df[col].reindex(prev_day_times).values
            forecast_df[col] = forecast_values

        combined = pd.concat([df, forecast_df])
        return combined

    except Exception as e:
        print("Forecasting failed:", e)
        traceback.print_exc()
        return df


def process_all_columns(ID, mode="missing"):
    df = load_and_prepare(ID)
    output = df.copy()
    target_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in target_cols:
        print(f"Processing column: {col}")
        if mode == "anomaly":
            cleaned = AnomalyDetect(df, col)
            output[col] = cleaned
        elif mode == "missing":
            imputed = MissingImpute(df, col)
            output[col] = imputed
        elif mode == "forecast":
            output = HAForecasting(df)
            break

    if mode == "anomaly":
        suffix = "_AD.csv"
    elif mode == "forecast":
        suffix = "_F.csv"
    else:
        suffix = "_MI.csv"

    output.reset_index(names='date_time').to_csv(ID + suffix, index=False)
    print(f"All columns processed in mode: {mode}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python your_script_name.py <ID> [mode]")
        print("mode 옵션: anomaly | missing | forecast (기본값 missing)")
        print("예시: python your_script_name.py building_energy forecast")
    else:
        ID = sys.argv[1]
        mode = sys.argv[2] if len(sys.argv) > 2 else "missing"
        process_all_columns(ID, mode)
