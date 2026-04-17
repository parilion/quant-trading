from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor


def fit_and_predict(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    result = df.copy()

    required_cols = {"trade_date", "ts_code", "split_set", *feature_cols}
    missing_cols = sorted(col for col in required_cols if col not in result.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if "label_ret_t1" in result.columns and result["label_ret_t1"].notna().any():
        label_col = "label_ret_t1"
    elif "label" in result.columns and result["label"].notna().any():
        label_col = "label"
    else:
        raise ValueError("No usable label column: expected non-null values in 'label_ret_t1' or 'label'")

    train_mask = (result["split_set"] == "train") & result[label_col].notna()
    test_mask = result["split_set"] == "test"

    train_df = result.loc[train_mask, feature_cols + [label_col]].copy()
    if train_df.empty:
        raise ValueError("training set is empty: require split_set=='train' with non-null label")

    test_df = result.loc[test_mask, ["trade_date", "ts_code"] + feature_cols].copy()
    if test_df.empty:
        raise ValueError("test set is empty: require split_set=='test'")

    train_df[feature_cols] = train_df[feature_cols].fillna(0.0)
    test_df[feature_cols] = test_df[feature_cols].fillna(0.0)

    model = LGBMRegressor(random_state=42)
    model.fit(train_df[feature_cols], train_df[label_col])

    test_df["y_pred"] = model.predict(test_df[feature_cols])
    return test_df[["trade_date", "ts_code", "y_pred"]]
