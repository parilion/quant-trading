from types import SimpleNamespace

import pandas as pd

from quant_trading.pipeline import orchestrator


class _DummyBegin:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyEngine:
    def begin(self):
        return _DummyBegin()


def test_stage_clean_align_filters_by_daily_universe(monkeypatch):
    captured: dict[str, object] = {}

    def fake_read_sql(sql, _conn, params=None):
        captured['sql'] = str(sql)
        captured['params'] = params
        return pd.DataFrame(
            {
                'trade_date': ['2024-01-02'],
                'ts_code': ['000001.SZ'],
                'close': [10.0],
                'amount': [1000.0],
                'pe_ttm': [12.0],
                'pb': [1.2],
                'ps_ttm': [2.3],
                'dv_ttm': [0.5],
            }
        )

    def fake_build_features(df):
        out = df.copy()
        out['ret_1d'] = 0.0
        out['ret_5d'] = 0.0
        out['vol_20d'] = 0.0
        out['mom_20d'] = 0.0
        out['amt_ratio_20d'] = 1.0
        out['is_valid'] = 1
        return out

    def fake_upsert(_engine, _table, frame, _key_cols, **_kwargs):
        captured['rows_written'] = len(frame)
        return len(frame)

    monkeypatch.setattr(orchestrator.pd, 'read_sql', fake_read_sql)
    monkeypatch.setattr(orchestrator, 'build_features', fake_build_features)
    monkeypatch.setattr(orchestrator, '_upsert_dataframe', fake_upsert)

    settings = SimpleNamespace(
        run_start_date='2024-01-01',
        run_end_date='2024-01-31',
        universe_index='000905.SH',
    )

    result = orchestrator._stage_clean_align(settings, _DummyEngine())

    sql_text = str(captured['sql']).lower()
    assert 'join meta_universe u' in sql_text
    assert 'u.index_code = :index_code' in sql_text
    assert captured['params']['index_code'] == '000905.SH'
    assert result == {'status': 'ok', 'rows': 1}


def test_stage_label_build_filters_by_daily_universe(monkeypatch):
    captured: dict[str, object] = {}

    def fake_read_sql(sql, _conn, params=None):
        captured['sql'] = str(sql)
        captured['params'] = params
        return pd.DataFrame(
            {
                'trade_date': ['2024-01-02'],
                'ts_code': ['000001.SZ'],
                'close': [10.0],
            }
        )

    def fake_build_label_and_split(df, _train_end, _valid_end):
        out = df.copy()
        out['label_ret_t1'] = 0.01
        out['split_set'] = 'train'
        return out

    def fake_upsert(_engine, _table, frame, _key_cols, **_kwargs):
        captured['rows_written'] = len(frame)
        return len(frame)

    monkeypatch.setattr(orchestrator.pd, 'read_sql', fake_read_sql)
    monkeypatch.setattr(orchestrator, 'build_label_and_split', fake_build_label_and_split)
    monkeypatch.setattr(orchestrator, '_upsert_dataframe', fake_upsert)

    settings = SimpleNamespace(
        run_start_date='2024-01-01',
        run_end_date='2024-01-31',
        train_end_date='2024-01-20',
        valid_end_date='2024-01-25',
        universe_index='000905.SH',
    )

    result = orchestrator._stage_label_build(settings, _DummyEngine())

    sql_text = str(captured['sql']).lower()
    assert 'join meta_universe u' in sql_text
    assert 'u.index_code = :index_code' in sql_text
    assert captured['params']['index_code'] == '000905.SH'
    assert result == {'status': 'ok', 'rows': 1}
