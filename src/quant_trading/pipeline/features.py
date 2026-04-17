from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    result["is_valid"] = result["close"].notna().astype(int)

    grouped = result.groupby("ts_code", sort=False)
    result["ret_1d"] = grouped["close"].pct_change(1, fill_method=None)
    result["ret_5d"] = grouped["close"].pct_change(5, fill_method=None)
    result["mom_20d"] = grouped["close"].transform(lambda s: s / s.shift(20) - 1.0)
    result["vol_20d"] = grouped["ret_1d"].transform(lambda s: s.rolling(20, min_periods=1).std())
    amount_20d_mean = grouped["amount"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    result["amt_ratio_20d"] = result["amount"] / amount_20d_mean

    result["close"] = result["close"].fillna(0.0)
    feature_cols = ["ret_1d", "ret_5d", "mom_20d", "vol_20d", "amt_ratio_20d"]
    result[feature_cols] = (
        result[feature_cols].replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
    )
    return result


def build_label_and_split(df: pd.DataFrame, train_end: str, valid_end: str) -> pd.DataFrame:
    result = df.copy()
    result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    train_end_ts = pd.to_datetime(train_end)
    valid_end_ts = pd.to_datetime(valid_end)
    if train_end_ts > valid_end_ts:
        raise ValueError("train_end must be less than or equal to valid_end")

    grouped = result.groupby("ts_code", sort=False)
    result["label_ret_t1"] = grouped["close"].shift(-1) / result["close"] - 1.0
    result["label_ret_t1"] = result["label_ret_t1"].replace([float("inf"), float("-inf")], float("nan"))

    result["split_set"] = "test"
    next_trade_date = pd.to_datetime(grouped["trade_date"].shift(-1))

    result.loc[next_trade_date <= train_end_ts, "split_set"] = "train"
    result.loc[(next_trade_date > train_end_ts) & (next_trade_date <= valid_end_ts), "split_set"] = "valid"
    return result
