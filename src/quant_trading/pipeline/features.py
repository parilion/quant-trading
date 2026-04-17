from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    result["is_valid"] = result["close"].notna().astype(int)
    result["close"] = result["close"].fillna(0.0)

    grouped = result.groupby("ts_code", sort=False)
    result["ret_1d"] = grouped["close"].pct_change(1)
    result["ret_5d"] = grouped["close"].pct_change(5)
    result["mom_20d"] = grouped["close"].transform(lambda s: s / s.shift(20) - 1.0)
    result["vol_20d"] = grouped["vol"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    amount_20d_mean = grouped["amount"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    result["amt_ratio_20d"] = result["amount"] / amount_20d_mean

    feature_cols = ["ret_1d", "ret_5d", "mom_20d", "vol_20d", "amt_ratio_20d"]
    result[feature_cols] = result[feature_cols].fillna(0.0)
    return result
