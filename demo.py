import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
from itertools import groupby
from operator import itemgetter
import sys


def load_and_prepare(ID, freq_str='5min', extend_str='0min'):
    # 1. CSV 불러오기
    df = pd.read_csv(ID + ".csv")

    if 'date_time' not in df.columns:
        raise ValueError("'date_time' column is missing in the input file.")

    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time')
    df = df.set_index('date_time')
    df.index.name = 'date_time'

    # 2. 숫자형 열만 다운샘플링
    try:
        delta = pd.Timedelta(freq_str)
        freq_str = pd.tseries.frequencies.to_offset(delta).freqstr
    except Exception:
        raise ValueError(f"Invalid frequency format: {freq_str}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    df_downsampled = df_numeric.resample(freq_str).mean()

    # 3. 업샘플링 범위 생성
    start_time = df.index.min()
    end_time = df.index.max()
    
    try:
        extend_delta = pd.Timedelta(extend_str)
    except ValueError:
        raise ValueError(f"Invalid extend_str value: '{extend_str}'")

    total_end_time = end_time + extend_delta

    full_time_range = pd.date_range(start=start_time, end=total_end_time, freq=freq_str)
    df_aligned = df_downsampled.reindex(full_time_range)

    return df_aligned


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
        anomalies = series[anomaly_mask]

        # 새로운 플래그 열 생성
        flag_col = pd.Series(index=series.index, dtype=object)
        flag_col[anomaly_mask] = "anomaly"

        return series, anomalies, flag_col

    except Exception as e:
        print(f"AnomalyDetect Fail for {colname}:", e)
        traceback.print_exc()
        return df[colname], pd.Series(dtype='float64'), pd.Series(dtype=object)


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

                if not np.isnan(v_left) and not np.isnan(v_right):
                    data_array[start:end + 1] = (v_right-v_left)*np.arange(1,miss_len+1)/(miss_len+1) + v_left
                else:
                    for i in group:
                        timestamp = time_index[i]
                        values = []
                        for offset_days in [-7, -2, -1, 1, 2, 7]:
                            target_time = timestamp + pd.Timedelta(days=offset_days)
                            if target_time in series.index:
                                val = series[target_time]
                                if not pd.isna(val):
                                    values.append(val)
                        data_array[i] = np.mean(values) if values else 0
            
            impute_flag = pd.Series(index=series.index, dtype=object)
            impute_flag.iloc[miss_idx] = "imputed"

            return pd.Series(data_array, index=series.index), impute_flag

        eval_blocks = clean_blocks[:max_eval_blocks]
        miss_len_range = list(range(min_miss, max_miss + 1))
        li_mse_total = np.zeros(len(miss_len_range))
        ha_mse_total = np.zeros(len(miss_len_range))
        counts = np.zeros(len(miss_len_range))
        
        # 결측 구간 길이에 따른 HA vs LI 평가 단계
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
                li_impute[start:end] = (v_right-v_left)*np.arange(1,miss_len+1)/(miss_len+1) + v_left

                ha_impute = test.copy()
                for j in range(start, end):
                    idx = block[j]
                    timestamp = time_index[idx]
                    values = []
                    for offset_days in [-7, -2, -1, 1, 2, 7]:
                        target_time = timestamp + pd.Timedelta(days=offset_days)
                        if target_time in series.index:
                            val = series[target_time]
                            if not pd.isna(val):
                                values.append(val)
                    
                    if values:
                        ha_impute[j] = np.mean(values)
                    else:
                        ha_impute[j] = 0

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
        
        # 실제 결측 보간 단계
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

            if miss_len <= threshold_len and not (np.isnan(v_left) or np.isnan(v_right)):
                imputed_array[start:end + 1] = (v_right-v_left)*np.arange(1,miss_len+1)/(miss_len+1) + v_left
            else:
                # HA 방식으로 전환
                for i in group:
                    timestamp = time_index[i]
                    values = []
                    for offset_days in [-7, -2, -1, 1, 2, 7]:
                        target_time = timestamp + pd.Timedelta(days=offset_days)
                        if target_time in series.index:
                            val = series[target_time]
                            if not pd.isna(val):
                                values.append(val)
                    imputed_array[i] = np.mean(values) if values else 0

        # 기존 결측 위치 기록
        impute_flag = pd.Series(index=series.index, dtype=object)
        impute_flag.iloc[miss_idx] = "imputed"

        return pd.Series(imputed_array, index=series.index), impute_flag

    except Exception as e:
        print(f"Fail in MissingImpute for {colname}:", e)
        traceback.print_exc()
        return df[colname]


def plot_recent_day(df, columns, mode, ID, anomalies_dict=None, extend_str='0min'):
    """
    최근 1일간의 데이터를 시각화하여 JPG로 저장합니다.
    - df: 처리된 데이터프레임 (datetime 인덱스 필요)
    - columns: 시각화할 열 이름 리스트
    - mode: 'anomaly', 'missing' 중 하나
    - ID: 파일 이름 접두사
    - anomalies_dict: mode가 'anomaly'일 때 이상치 시점과 값을 담은 딕셔너리 {colname: pd.Series}
    - extend_str이 0이 아니면 연장 구간만 시각화
    """
    try:
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'])
            df = df.set_index('date_time')

        df = df.sort_index()

        # 시간 간격 추정
        time_diffs = df.index.to_series().diff().dropna()
        resol = time_diffs.mode()[0].total_seconds() / 60
        one_day_points = int(round(24 * 60 / resol))

        # 1. 연장 구간만 추출
        if extend_str != '0min':
            try:
                extend_delta = pd.Timedelta(extend_str)
                last_actual_time = df.dropna().index.max() - extend_delta
                recent_df = df[df.index > last_actual_time]
            except Exception:
                print(f"Invalid extend_str format: {extend_str}")
                return
        else:
            # 2. 전체 샘플 수가 하루보다 적으면 전체 사용
            if len(df) <= one_day_points:
                recent_df = df.copy()
            else:
                recent_df = df.iloc[-one_day_points:].copy()

        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, 1, figsize=(10, 3 * n_cols), sharex=True)
        if n_cols == 1:
            axes = [axes]

        for ax, col in zip(axes, columns):
            ax.plot(recent_df.index, recent_df[col], label=col)

            if mode == "anomaly" and anomalies_dict is not None:
                anomalies = anomalies_dict.get(col)
                if anomalies is not None and not anomalies.empty:
                    recent_anomalies = anomalies.loc[anomalies.index.isin(recent_df.index)]
                    recent_anomalies = recent_anomalies.dropna()
                    if not recent_anomalies.empty:
                        ax.scatter(recent_anomalies.index, recent_anomalies.values,
                                   color='red', label='Anomaly', zorder=5)

            ax.set_ylabel(col)
            ax.legend()

        fig.suptitle(f"{mode.capitalize()} Visualization")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_path = f"{ID}_{mode}_plot.jpg"
        plt.savefig(plot_path)
        plt.close()
        print(f"{mode.capitalize()} plot saved to: {plot_path}")

    except Exception as e:
        print(f"Plotting failed for {mode}:", e)


def process_all_columns(ID, mode="missing", freq_str="5min", extend_str="0min"):
    df = load_and_prepare(ID, freq_str=freq_str, extend_str=extend_str)
    output = df.copy()
    target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    anomalies_dict = {} if mode == "anomaly" else None

    imputed_flag = None

    for col in target_cols:
        print(f"Processing column: {col}")
        if mode == "anomaly":
            cleaned, anomalies, flag = AnomalyDetect(df, col)
            output[col] = cleaned
            output[col + "_flag"] = flag
            anomalies_dict[col] = anomalies
        elif mode == "missing":
            imputed, _ = MissingImpute(df, col)
            output[col] = imputed

    # imputed_flag는 마지막 열 기준으로 하나만
    if mode == "missing":
        last_col = target_cols[-1]
        _, imputed_flag = MissingImpute(df, last_col)
        output["imputed_flag"] = imputed_flag

    # 열 순서 재정렬
    reordered_cols = []
    if mode == "missing":
        reordered_cols = target_cols.copy()
        reordered_cols.append("imputed_flag")
    elif mode == "anomaly":
        for col in target_cols:
            reordered_cols.append(col)
            flag_col = col + "_flag"
            if flag_col in output.columns:
                reordered_cols.append(flag_col)

    # 숫자형 외의 나머지 열도 보존
    remaining_cols = [col for col in output.columns if col not in reordered_cols]
    output = output[reordered_cols + remaining_cols]

    # 통계 요약 및 결과 CSV 저장
    summary = {}
    summary["start_time"] = [df.index.min()]
    summary["end_time"] = [df.index.max()]

    if mode == "missing":
        actual_range = df.index[df.index <= df.index.max() - pd.Timedelta(extend_str)]
        actual_len = len(actual_range)
        flag_count = output.loc[actual_range, "imputed_flag"].eq("imputed").sum()
        ratio = round(flag_count / actual_len, 4) if actual_len else 0
        summary["missing_ratio"] = [ratio]
        summary["imputed_ratio"] = [ratio]
    
    elif mode == "anomaly":
        for col in target_cols:
            flag_col = col + "_flag"
            if flag_col in output.columns:
                ratio = output[flag_col].eq("anomaly").mean()
                summary[f"{col}_anomaly_ratio"] = [round(ratio, 4)]
    
    # 평균 및 표준편차
    for col in target_cols:
        summary[f"{col}_mean"] = [output[col].mean()]
        summary[f"{col}_std"] = [output[col].std()]
    
    # summary_df 생성
    summary_df = pd.DataFrame(summary)
    
    # 각 내용을 문자열로 변환 (lineterminator 사용)
    summary_csv = summary_df.to_csv(index=False, lineterminator='\n').strip()
    output_csv = output.to_csv(index_label='date_time', lineterminator='\n').strip()
    
    # 둘 사이 줄바꿈 하나만 넣어서 연결
    merged_csv = summary_csv + '\n\n' + output_csv
    
    # 파일로 저장
    with open(f"{ID}_{mode}.csv", 'w', encoding='utf-8') as f:
        f.write(merged_csv)
    
    print(f"Summary and data merged into file: {ID}_{mode}_merged.csv")

    # 시각화
    df_forplot = df.copy() if mode == "anomaly" else output.copy()
    try:
        plot_recent_day(
            df_forplot,
            target_cols,
            mode,
            ID,
            anomalies_dict if mode == "anomaly" else None,
            extend_str=extend_str,
        )
    except Exception as e:
        print("Plot generation failed:", e)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python your_script_name.py <ID> [mode] [freq_str] [extend_str]")
        print("예시: python your_script_name.py building_energy missing 5min 1hour")
    else:
        ID = sys.argv[1]
        mode = sys.argv[2] if len(sys.argv) > 2 else "missing"
        freq_str = sys.argv[3] if len(sys.argv) > 3 else "5min"
        extend_str = sys.argv[4] if len(sys.argv) > 4 else "0min"
        process_all_columns(ID, mode, freq_str, extend_str)
