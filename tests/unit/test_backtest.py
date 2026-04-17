import pandas as pd
import pytest

from quant_trading.pipeline.backtest import topk_backtest


def test_topk_backtest_returns_nav_and_core_metrics():
    pred = pd.DataFrame(
        {
            "trade_date": [
                "2026-01-02",
                "2026-01-02",
                "2026-01-02",
                "2026-01-03",
                "2026-01-03",
                "2026-01-03",
            ],
            "ts_code": [
                "000001.SZ",
                "000002.SZ",
                "000003.SZ",
                "000001.SZ",
                "000002.SZ",
                "000003.SZ",
            ],
            "y_pred": [0.9, 0.8, 0.1, 0.2, 0.7, 0.6],
        }
    )
    realized = pd.DataFrame(
        {
            "trade_date": [
                "2026-01-02",
                "2026-01-02",
                "2026-01-02",
                "2026-01-03",
                "2026-01-03",
                "2026-01-03",
            ],
            "ts_code": [
                "000001.SZ",
                "000002.SZ",
                "000003.SZ",
                "000001.SZ",
                "000002.SZ",
                "000003.SZ",
            ],
            "label_ret_t1": [0.02, 0.01, -0.03, 0.05, -0.02, 0.01],
        }
    )

    nav, metrics = topk_backtest(pred, realized, top_k=2, trade_cost_bps=10)

    assert len(nav) == 2
    assert list(nav.columns) == ["trade_date", "nav"]
    assert metrics["cum_return"] == pytest.approx(0.007916, abs=1e-8)
    assert metrics["avg_daily_ret"] == pytest.approx(0.004, abs=1e-10)


def test_topk_backtest_metrics_values_are_correct():
    pred = pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02"],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "y_pred": [0.9, 0.8],
        }
    )
    realized = pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02"],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "label_ret_t1": [0.01, 0.03],
        }
    )

    _, metrics = topk_backtest(pred, realized, top_k=1, trade_cost_bps=0)

    assert metrics["avg_daily_ret"] == pytest.approx(0.01)
    assert metrics["cum_return"] == pytest.approx(0.01)


def test_topk_backtest_returns_empty_nav_and_zero_metrics_on_empty_merge():
    pred = pd.DataFrame(
        {
            "trade_date": ["2026-01-02"],
            "ts_code": ["000001.SZ"],
            "y_pred": [0.9],
        }
    )
    realized = pd.DataFrame(
        {
            "trade_date": ["2026-01-03"],
            "ts_code": ["000001.SZ"],
            "label_ret_t1": [0.01],
        }
    )

    nav, metrics = topk_backtest(pred, realized, top_k=1, trade_cost_bps=10)

    assert list(nav.columns) == ["trade_date", "nav"]
    assert len(nav) == 0
    assert "cum_return" in metrics
    assert "avg_daily_ret" in metrics
    assert metrics["cum_return"] == pytest.approx(0.0)
    assert metrics["avg_daily_ret"] == pytest.approx(0.0)


@pytest.mark.parametrize("top_k", [0, -1])
def test_topk_backtest_raises_value_error_when_top_k_is_not_positive(top_k: int):
    pred = pd.DataFrame(
        {
            "trade_date": ["2026-01-02"],
            "ts_code": ["000001.SZ"],
            "y_pred": [0.9],
        }
    )
    realized = pd.DataFrame(
        {
            "trade_date": ["2026-01-02"],
            "ts_code": ["000001.SZ"],
            "label_ret_t1": [0.01],
        }
    )

    with pytest.raises(ValueError, match="top_k"):
        topk_backtest(pred, realized, top_k=top_k, trade_cost_bps=0)


def test_topk_backtest_handles_duplicate_keys_without_row_amplification():
    pred = pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02", "2026-01-03", "2026-01-03"],
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ"],
            "y_pred": [0.10, 0.20, 0.30, 0.40],
        }
    )
    realized = pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02", "2026-01-03", "2026-01-03"],
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ"],
            "label_ret_t1": [0.01, 0.02, 0.03, 0.04],
        }
    )

    nav, metrics = topk_backtest(pred, realized, top_k=1, trade_cost_bps=0)

    assert len(nav) == 2
    assert metrics["avg_daily_ret"] == pytest.approx(0.03)
    assert metrics["cum_return"] == pytest.approx(0.0608)


def test_topk_backtest_breaks_ties_by_ts_code_ascending():
    pred = pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02"],
            "ts_code": ["000002.SZ", "000001.SZ"],
            "y_pred": [0.5, 0.5],
        }
    )
    realized = pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02"],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "label_ret_t1": [0.01, 0.09],
        }
    )

    nav, metrics = topk_backtest(pred, realized, top_k=1, trade_cost_bps=0)

    assert len(nav) == 1
    assert nav.iloc[0]["nav"] == pytest.approx(1.01)
    assert metrics["avg_daily_ret"] == pytest.approx(0.01)
    assert metrics["cum_return"] == pytest.approx(0.01)
