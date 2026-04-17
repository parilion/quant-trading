import pandas as pd
import pytest

from quant_trading.pipeline.model import fit_and_predict


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-03", "2026-01-06", "2026-01-07"],
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ"],
            "split_set": ["train", "train", "test", "test"],
            "label_ret_t1": [0.10, -0.05, None, None],
            "label": [0.11, -0.06, None, None],
            "ret_1d": [0.01, -0.02, 0.03, 0.00],
            "mom_20d": [0.10, 0.12, 0.11, 0.09],
        }
    )


def test_fit_and_predict_returns_two_test_rows_with_required_columns_and_numeric_predictions():
    df = _sample_df()

    result = fit_and_predict(df, feature_cols=["ret_1d", "mom_20d"])

    assert len(result) == 2
    assert {"trade_date", "ts_code", "y_pred"}.issubset(result.columns)
    assert result["y_pred"].notna().all()
    assert pd.api.types.is_numeric_dtype(result["y_pred"])


def test_fit_and_predict_falls_back_to_label_when_label_ret_t1_is_all_null():
    df = _sample_df()
    df["label_ret_t1"] = None

    result = fit_and_predict(df, feature_cols=["ret_1d", "mom_20d"])

    assert len(result) == 2
    assert result["y_pred"].notna().all()


def test_fit_and_predict_uses_label_when_label_ret_t1_only_available_in_test():
    df = _sample_df()
    df.loc[df["split_set"] == "train", "label_ret_t1"] = None
    df.loc[df["split_set"] == "test", "label_ret_t1"] = [0.01, -0.01]

    result = fit_and_predict(df, feature_cols=["ret_1d", "mom_20d"])

    assert len(result) == 2
    assert result["y_pred"].notna().all()


def test_fit_and_predict_raises_when_train_set_is_empty():
    df = _sample_df()
    df["split_set"] = ["valid", "valid", "test", "test"]

    with pytest.raises(ValueError, match="training set is empty"):
        fit_and_predict(df, feature_cols=["ret_1d", "mom_20d"])


def test_fit_and_predict_raises_when_test_set_is_empty():
    df = _sample_df()
    df["split_set"] = ["train", "train", "valid", "valid"]

    with pytest.raises(ValueError, match="test set is empty"):
        fit_and_predict(df, feature_cols=["ret_1d", "mom_20d"])


def test_fit_and_predict_raises_when_feature_cols_is_empty():
    df = _sample_df()

    with pytest.raises(ValueError, match="feature_cols"):
        fit_and_predict(df, feature_cols=[])


@pytest.mark.parametrize(
    "missing_col",
    ["trade_date", "ts_code", "split_set", "ret_1d", "mom_20d"],
)
def test_fit_and_predict_raises_value_error_when_required_columns_missing(missing_col: str):
    df = _sample_df().drop(columns=[missing_col])

    with pytest.raises(ValueError, match="Missing required columns"):
        fit_and_predict(df, feature_cols=["ret_1d", "mom_20d"])


def test_fit_and_predict_raises_when_no_usable_label_column():
    df = pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-03", "2026-01-06", "2026-01-07"],
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ"],
            "split_set": ["train", "train", "test", "test"],
            "ret_1d": [0.01, -0.02, 0.03, 0.00],
            "mom_20d": [0.10, 0.12, 0.11, 0.09],
        }
    )

    with pytest.raises(ValueError, match="No usable label column"):
        fit_and_predict(df, feature_cols=["ret_1d", "mom_20d"])
