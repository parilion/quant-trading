import pandas as pd
import pytest

from quant_trading.pipeline.features import build_features


def test_build_features_contains_required_columns():
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "trade_date": ["2026-01-02", "2026-01-03", "2026-01-06"],
            "close": [10.0, 10.5, None],
            "vol": [100.0, 120.0, 90.0],
            "amount": [1000.0, 1300.0, 800.0],
        }
    )

    result = build_features(df)

    required_columns = {
        "ret_1d",
        "ret_5d",
        "mom_20d",
        "vol_20d",
        "amt_ratio_20d",
        "is_valid",
    }
    assert required_columns.issubset(set(result.columns))


def test_build_features_handles_missing_close_and_prevents_inf_leakage():
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ"],
            "trade_date": ["2026-01-02", "2026-01-03", "2026-01-06", "2026-01-07"],
            "close": [10.0, None, 11.0, 12.0],
            "vol": [100.0, 120.0, 130.0, 140.0],
            "amount": [1000.0, 0.0, 1500.0, 1600.0],
        }
    )

    result = build_features(df)

    feature_cols = ["ret_1d", "ret_5d", "mom_20d", "vol_20d", "amt_ratio_20d"]
    assert not result[feature_cols].isin([float("inf"), float("-inf")]).to_numpy().any()

    missing_close_row = result.loc[result["trade_date"] == "2026-01-03"].iloc[0]
    assert missing_close_row["is_valid"] == 0


def test_build_features_uses_return_volatility_for_vol_20d():
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "trade_date": ["2026-01-02", "2026-01-03", "2026-01-06"],
            "close": [10.0, 11.0, 12.0],
            "vol": [100.0, 120.0, 130.0],
            "amount": [1000.0, 1200.0, 1300.0],
        }
    )

    result = build_features(df)

    raw_ret_1d = df["close"].pct_change(1, fill_method=None)
    expected_vol_20d = raw_ret_1d.rolling(20, min_periods=1).std().fillna(0.0)
    assert result["vol_20d"].tolist() == pytest.approx(expected_vol_20d.tolist())
