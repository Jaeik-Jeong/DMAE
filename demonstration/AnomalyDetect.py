import pandas as pd
import numpy as np
import traceback

def Zscore(ID, window=12, threshold=3):
    try:
        data_input = pd.read_csv(ID + ".csv", index_col=0)

        if "value_impute" not in data_input.columns:
            raise ValueError('"value_impute" column not found in input file.')

        series = data_input["value_impute"]

        anomaly = []
        for i in range(len(series)):
            if i < window:
                anomaly.append('X')
                continue

            window_data = series[i - window:i]

            if window_data.isna().any() or pd.isna(series[i]):
                anomaly.append('X')
                continue

            mean = window_data.mean()
            std = window_data.std()

            if std == 0:
                anomaly.append('X')
                continue

            z = (series[i] - mean) / std
            if abs(z) > threshold:
                anomaly.append('O')
            else:
                anomaly.append('X')

        data_output = data_input.copy()
        data_output['Anomaly'] = anomaly
        data_output.to_csv(ID + "_detect.csv")
        print("Success")

    except Exception as e:
        print("Fail:", e)
        traceback.print_exc()