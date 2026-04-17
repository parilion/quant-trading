import pandas as pd

from quant_trading.pipeline.model import fit_and_predict


def test_fit_and_predict_returns_two_test_rows_with_required_columns():
    df = pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-03", "2026-01-06", "2026-01-07"],
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ"],
            "split_set": ["train", "train", "test", "test"],
            "label_ret_t1": [0.10, -0.05, None, None],
            "ret_1d": [0.01, -0.02, 0.03, 0.00],
            "mom_20d": [0.10, 0.12, 0.11, 0.09],
        }
    )

    result = fit_and_predict(df, feature_cols=["ret_1d", "mom_20d"])

    assert len(result) == 2
    assert {"trade_date", "ts_code", "y_pred"}.issubset(result.columns)
