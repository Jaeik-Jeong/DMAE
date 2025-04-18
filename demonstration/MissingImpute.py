import pandas as pd
import numpy as np
import traceback


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