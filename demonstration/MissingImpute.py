import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
from itertools import groupby
from operator import itemgetter


def LI(v_left, v_right, miss_len):
    x = np.arange(1, miss_len+1)
    y = (v_right - v_left) * x / (miss_len + 1) + v_left
    return y


def LIImpute(ID):
    try:
        data_input = pd.read_csv(ID + ".csv", index_col=0)
        data_array = np.array(data_input.iloc[:, 0]).flatten()
        data_impute_array = data_array.copy()

        miss_idx = np.where(np.isnan(data_array))[0]

        from itertools import groupby
        from operator import itemgetter
        miss_point = []
        for k, g in groupby(enumerate(miss_idx), lambda x: x[0]-x[1]):
            group = list(map(itemgetter(1), g))
            miss_point.append(group)

        for group in miss_point:
            start = group[0]
            end = group[-1]
            if start == 0 or end == len(data_array) - 1:
                continue

            v_left = data_array[start - 1]
            v_right = data_array[end + 1]
            miss_len = len(group)
            data_impute_array[start:end+1] = LI(v_left, v_right, miss_len)

        data_output = data_input.copy()
        data_output['value_impute'] = data_impute_array
        data_output.to_csv(ID + "_li_impute.csv")
        print("Success")

    except Exception as e:
        print("Fail:", e)
        traceback.print_exc()


def HAImpute(ID, resol):
    try:
        data_input = pd.read_csv(ID + ".csv", index_col=0)
        data_input.index = pd.to_datetime(data_input.index)
        data_input = data_input.sort_index()

        data_array = data_input.iloc[:, 0].copy()
        data_impute = data_array.copy()

        time_delta = pd.to_timedelta(f"{resol}min")
        day_delta = pd.to_timedelta("1D")

        for timestamp in data_array.index:
            if pd.isna(data_array[timestamp]):
                prev_day = timestamp - day_delta
                next_day = timestamp + day_delta

                val_prev = None
                val_next = None

                if prev_day in data_array.index and not pd.isna(data_array[prev_day]):
                    val_prev = data_array[prev_day]
                elif (prev_day - day_delta) in data_array.index and not pd.isna(data_array[prev_day - day_delta]):
                    val_prev = data_array[prev_day - day_delta]

                if next_day in data_array.index and not pd.isna(data_array[next_day]):
                    val_next = data_array[next_day]
                elif (next_day + day_delta) in data_array.index and not pd.isna(data_array[next_day + day_delta]):
                    val_next = data_array[next_day + day_delta]

                if val_prev is not None and val_next is not None:
                    data_impute[timestamp] = (val_prev + val_next) / 2
                elif val_prev is not None:
                    data_impute[timestamp] = val_prev
                elif val_next is not None:
                    data_impute[timestamp] = val_next

        data_output = data_input.copy()
        data_output['value_impute'] = data_impute
        data_output.to_csv(ID + "_ha_impute.csv")
        print("Success")

    except Exception as e:
        print("Fail:", e)
        traceback.print_exc()
        
        
def RecommendedImpute(ID, resol, max_eval_blocks=5):
    try:
        # 1. 데이터 불러오기
        data_input = pd.read_csv(ID + ".csv", index_col=0)
        data_input.index = pd.to_datetime(data_input.index)
        data_input = data_input.sort_index()
        series = data_input.iloc[:, 0].copy()
        time_index = series.index
        total_len = len(series)

        # 2. 실제 missing 구간 길이 분석
        data_array = series.copy()
        miss_idx = np.where(np.isnan(data_array))[0]

        miss_lens = []
        for k, g in groupby(enumerate(miss_idx), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), g))
            miss_lens.append(len(group))

        if not miss_lens:
            raise ValueError("No missing data found in original series.")

        min_miss = max(1, min(miss_lens))
        max_miss = max(miss_lens)

        # 3. 결측 없는 구간 찾기 (synthetic test용)
        non_nan_indices = np.where(~np.isnan(series.values))[0]
        clean_blocks = []
        for k, g in groupby(enumerate(non_nan_indices), lambda x: x[0] - x[1]):
            block = list(map(itemgetter(1), g))
            if len(block) > max_miss + 2:
                clean_blocks.append(block)

        if not clean_blocks:
            raise ValueError("No clean block available for performance evaluation.")

        # 최대 max_eval_blocks개의 블록 사용
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

                # LI
                v_left = test[start - 1]
                v_right = test[end]
                li_impute = test.copy()
                li_impute[start:end] = LI(v_left, v_right, miss_len)

                # HA
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

                # MSE 누적
                li_mse_total[i] += np.nanmean((true_values.values - li_impute[start:end].values) ** 2)
                ha_mse_total[i] += np.nanmean((true_values.values - ha_impute[start:end].values) ** 2)
                counts[i] += 1

        # 평균 MSE 계산
        li_mse = li_mse_total / counts
        ha_mse = ha_mse_total / counts

        # 4. MSE 그래프 저장
        plt.figure(figsize=(8, 5))
        plt.plot(miss_len_range, li_mse, label='LI MSE', marker='o')
        plt.plot(miss_len_range, ha_mse, label='HA MSE', marker='s')
        threshold_est = miss_len_range[np.argmin(np.abs(np.array(li_mse) - np.array(ha_mse)))]
        plt.axvline(x=threshold_est, color='gray', linestyle='--', label=f'Threshold ≈ {threshold_est}')
        plt.xlabel("Missing Length")
        plt.ylabel("MSE")
        plt.title("LI vs HA MSE by Missing Length (Average over blocks)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(ID + "_mse_plot.png")
        plt.close()

        # 5. 임계 missing 길이 n* 계산
        threshold_len = max_miss
        for i in range(len(li_mse)):
            if ha_mse[i] < li_mse[i]:
                threshold_len = miss_len_range[i]
                break

        # 6. 실제 missing 구간에 대해 선택 적용
        imputed_array = data_array.copy()
        miss_point = []
        for k, g in groupby(enumerate(miss_idx), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), g))
            miss_point.append(group)

        for group in miss_point:
            start = group[0]
            end = group[-1]
            miss_len = len(group)

            if start == 0 or end == total_len - 1:
                continue

            if miss_len <= threshold_len:
                v_left = data_array[start - 1]
                v_right = data_array[end + 1]
                imputed_array[start:end + 1] = LI(v_left, v_right, miss_len)
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

        # 7. 저장
        data_output = data_input.copy()
        data_output['value_impute'] = imputed_array
        data_output.to_csv(ID + "_recommended_impute.csv")
        print("Success")

    except Exception as e:
        print("Fail:", e)
        traceback.print_exc()