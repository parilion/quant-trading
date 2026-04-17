import pandas as pd

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

    required_columns = {"ret_1d", "ret_5d", "mom_20d", "amt_ratio_20d", "is_valid"}
    assert required_columns.issubset(set(result.columns))
