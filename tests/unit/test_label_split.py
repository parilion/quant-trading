import pandas as pd
import pytest

from quant_trading.pipeline.features import build_label_and_split


def test_build_label_and_split_computes_t_plus_1_label_from_sorted_rows():
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "trade_date": ["2026-01-03", "2026-01-02", "2026-01-06"],
            "close": [11.0, 10.0, 12.0],
        }
    )

    result = build_label_and_split(df, train_end="2026-01-02", valid_end="2026-01-03")

    assert result.iloc[0]["label_ret_t1"] == pytest.approx(0.1)


def test_build_label_and_split_assigns_train_valid_test_by_date():
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ"],
            "trade_date": ["2026-01-03", "2026-01-02", "2026-01-06", "2026-01-02", "2026-01-03"],
            "close": [11.0, 10.0, 12.0, 20.0, 22.0],
        }
    )

    result = build_label_and_split(df, train_end="2026-01-02", valid_end="2026-01-03")

    assert result["split_set"].tolist() == ["train", "valid", "test", "train", "valid"]
