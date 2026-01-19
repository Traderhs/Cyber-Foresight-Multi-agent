from typing import List
import numpy as np
import pandas as pd

ALPHA = 0.1
BETA = 0.3

# Input files
data_file_all = "Cyber_Trend_Forecasting_All.csv"  # All PTs columns
data_file = "Cyber_Trend_Forecasting.csv"  # Filtered PTs columns

# Output files with smoothed-only data
smoothed_file_all = "Smoothed_CyberTrend_Forecasting_All.csv"
smoothed_file = "Smoothed_CyberTrend_Forecasting.csv"

# Output files with smoothed and normalized data
norm_file_all = "Norm_CyberTrend_Forecasting_All.csv"
norm_file = "Norm_CyberTrend_Forecasting.csv"


def double_exp_smoothing(
    series: pd.Series,
    alpha: float,
    beta: float
) -> List[float]:
    """
    Double exponential smoothing function.

    :param series: Input time series data.
    :param alpha: How fast smoothed values react to new data.
    :param beta: How fast the trend adapts.

    :return: List with double exponential smoothing output.
    """
    result = [series[0]]
    level, trend = None, None

    for n in range(1, len(series)):
        if n == 1:
            level, trend = series[0], series[1] - series[0]

        value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result


def smooth_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Utility function to smooth all the columns of the produced dataset.

    :param df: The original time series dataset.
    :return: Dataset with double exponential smoothed applied to all
    columns (except dates).
    """
    for col in df.columns:
        if col != "Date":
            df[col] = double_exp_smoothing(df[col], ALPHA, BETA)
    return df


def normalize(series: pd.Series) -> np.ndarray:
    """
    Function for normalizind data by dividing with the max column value. To be
    used with pandas.DataFrame.apply method.

    :param series: Input time series.
    :return: Array with normalized data.
    """
    series = series.to_numpy()
    max_value = np.max(np.abs(series))
    if max_value > 0:
        return series / max_value
    return series


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Utility function to normalize all the columns of the smoothed dataset using
    the max value of each column.

    :param df: The smoothed time series dataset.
    :return: Dataset with normalized columns (except dates).
    """
    for col in df.columns:
        if col != "Date":
            df[col] = normalize(df[col])
    return df


if __name__ == "__main__":
    df_all = pd.read_csv(data_file_all)
    filtered_df = pd.read_csv(data_file)

    # Smooth both datasets
    df_all = smooth_df(df_all)
    filtered_df = smooth_df(filtered_df)

    # Export smoothed-only datasets
    df_all.to_csv(smoothed_file_all, index=False)
    filtered_df.to_csv(smoothed_file, index=False)

    # Normalize both smoothed datasets
    df_all = normalize_df(df_all)
    filtered_df = normalize_df(filtered_df)

    # Export smoothed and normalized datasets
    df_all.to_csv(norm_file_all, index=False)
    filtered_df.to_csv(norm_file, index=False)
