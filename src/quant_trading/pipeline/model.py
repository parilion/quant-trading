from __future__ import annotations

import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


def fit_and_predict(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    result = df.copy()

    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")

    required_cols = {"trade_date", "ts_code", "split_set", *feature_cols}
    missing_cols = sorted(col for col in required_cols if col not in result.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    train_rows = result["split_set"] == "train"
    if not train_rows.any():
        raise ValueError("training set is empty: require split_set=='train'")

    if "label_ret_t1" in result.columns and result.loc[train_rows, "label_ret_t1"].notna().any():
        label_col = "label_ret_t1"
    elif "label" in result.columns and result.loc[train_rows, "label"].notna().any():
        label_col = "label"
    else:
        raise ValueError(
            "No usable label column in training set: expected non-null values in 'label_ret_t1' or 'label' where split_set=='train'"
        )

    train_mask = (result["split_set"] == "train") & result[label_col].notna()
    valid_mask = (result["split_set"] == "valid") & result[label_col].notna()
    test_mask = result["split_set"] == "test"

    train_df = result.loc[train_mask, feature_cols + [label_col]].copy()
    if train_df.empty:
        raise ValueError("training set is empty: require split_set=='train' with non-null label")

    test_df = result.loc[test_mask, ["trade_date", "ts_code"] + feature_cols].copy()
    if test_df.empty:
        raise ValueError("test set is empty: require split_set=='test'")

    train_df[feature_cols] = train_df[feature_cols].fillna(0.0)
    test_df[feature_cols] = test_df[feature_cols].fillna(0.0)

    # Quick parameter search on validation split (if available).
    # Keep it lightweight for daily iteration and fallback to baseline safely.
    enable_search = os.environ.get("MODEL_PARAM_SEARCH", "1") == "1"
    valid_df = result.loc[valid_mask, feature_cols + [label_col]].copy()
    if not valid_df.empty:
        valid_df[feature_cols] = valid_df[feature_cols].fillna(0.0)

    best_params: dict[str, object] = {}
    if enable_search and not valid_df.empty:
        candidates = [
            {},
            {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 20},
            {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 30},
            {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 31, "min_child_samples": 20},
            {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 63, "min_child_samples": 30},
            {"n_estimators": 600, "learning_rate": 0.015, "num_leaves": 63, "min_child_samples": 40},
            {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 20},
            {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 40},
            {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 127, "min_child_samples": 50},
            {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 20},
        ]
        best_mse = float("inf")
        for params in candidates:
            trial = LGBMRegressor(random_state=42, **params)
            trial.fit(train_df[feature_cols], train_df[label_col])
            pred_valid = trial.predict(valid_df[feature_cols])
            mse = float(np.mean((pred_valid - valid_df[label_col].to_numpy()) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_params = params
        print(f"[model] validation search selected params={best_params}, mse={best_mse:.8f}", flush=True)

    fit_mask = train_mask | valid_mask
    fit_df = result.loc[fit_mask, feature_cols + [label_col]].copy()
    fit_df[feature_cols] = fit_df[feature_cols].fillna(0.0)

    model = LGBMRegressor(random_state=42, **best_params)
    model.fit(fit_df[feature_cols], fit_df[label_col])

    test_df["y_pred"] = model.predict(test_df[feature_cols])
    return test_df[["trade_date", "ts_code", "y_pred"]]
