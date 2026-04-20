from types import SimpleNamespace

import pandas as pd
import pytest

from quant_trading.pipeline import orchestrator


class _DummyBegin:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyEngine:
    def begin(self):
        return _DummyBegin()


def _make_trade_cal(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"cal_date": dates, "is_open": [1] * len(dates)})


def test_stage_universe_daily_expand_fails_when_run_start_before_first_snapshot(monkeypatch):
    snapshots = pd.DataFrame(
        {
            "trade_date": ["2020-01-31"] * 500,
            "index_code": ["000905.SH"] * 500,
            "ts_code": [f"000{i:03d}.SZ" for i in range(1, 501)],
        }
    )

    def fake_read_sql(sql, _conn, params=None):
        return snapshots.copy()

    def fake_call_with_retry(fn):
        return _make_trade_cal(["20200102", "20200103"])

    def fake_expand_snapshot_membership(**_kwargs):
        return pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2020-01-31"] * 500),
                "index_code": ["000905.SH"] * 500,
                "ts_code": [f"000{i:03d}.SZ" for i in range(1, 501)],
                "source": ["meta_universe_expand"] * 500,
            }
        )

    monkeypatch.setattr(orchestrator.pd, "read_sql", fake_read_sql)
    monkeypatch.setattr(orchestrator, "_get_thread_tushare_client", lambda _settings: object())
    monkeypatch.setattr(orchestrator, "_call_with_retry", fake_call_with_retry)
    monkeypatch.setattr(orchestrator, "expand_snapshot_membership", fake_expand_snapshot_membership)
    monkeypatch.setattr(orchestrator, "_upsert_dataframe", lambda *_args, **_kwargs: 500)

    settings = SimpleNamespace(
        run_start_date="2020-01-01",
        run_end_date="2020-02-07",
        universe_index="000905.SH",
    )

    with pytest.raises(RuntimeError, match="earlier than first snapshot"):
        orchestrator._stage_universe_daily_expand(settings, _DummyEngine())


def test_stage_universe_daily_expand_fails_when_daily_count_not_500(monkeypatch):
    snapshots = pd.DataFrame(
        {
            "trade_date": ["2020-01-31"] * 500,
            "index_code": ["000905.SH"] * 500,
            "ts_code": [f"000{i:03d}.SZ" for i in range(1, 501)],
        }
    )

    def fake_read_sql(sql, _conn, params=None):
        return snapshots.copy()

    def fake_call_with_retry(fn):
        return _make_trade_cal(["20200131"])

    def fake_expand_snapshot_membership(**_kwargs):
        return pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2020-01-31"] * 499),
                "index_code": ["000905.SH"] * 499,
                "ts_code": [f"000{i:03d}.SZ" for i in range(1, 500)],
                "source": ["meta_universe_expand"] * 499,
            }
        )

    monkeypatch.setattr(orchestrator.pd, "read_sql", fake_read_sql)
    monkeypatch.setattr(orchestrator, "_get_thread_tushare_client", lambda _settings: object())
    monkeypatch.setattr(orchestrator, "_call_with_retry", fake_call_with_retry)
    monkeypatch.setattr(orchestrator, "expand_snapshot_membership", fake_expand_snapshot_membership)
    monkeypatch.setattr(orchestrator, "_upsert_dataframe", lambda *_args, **_kwargs: 499)

    settings = SimpleNamespace(
        run_start_date="2020-01-31",
        run_end_date="2020-01-31",
        universe_index="000905.SH",
    )

    with pytest.raises(RuntimeError, match="exactly 500"):
        orchestrator._stage_universe_daily_expand(settings, _DummyEngine())
