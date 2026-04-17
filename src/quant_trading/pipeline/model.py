from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor


def fit_and_predict(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    result = df.copy()

    label_col = "label_ret_t1" if "label_ret_t1" in result.columns else "label"
    if label_col not in result.columns:
        raise ValueError("label column not found, expected 'label_ret_t1' or 'label'")

    train_mask = (result["split_set"] == "train") & result[label_col].notna()
    test_mask = result["split_set"] == "test"

    train_df = result.loc[train_mask, feature_cols + [label_col]].copy()
    test_df = result.loc[test_mask, ["trade_date", "ts_code"] + feature_cols].copy()

    train_df[feature_cols] = train_df[feature_cols].fillna(0.0)
    test_df[feature_cols] = test_df[feature_cols].fillna(0.0)

    model = LGBMRegressor(random_state=42)
    model.fit(train_df[feature_cols], train_df[label_col])

    test_df["y_pred"] = model.predict(test_df[feature_cols])
    return test_df[["trade_date", "ts_code", "y_pred"]]
