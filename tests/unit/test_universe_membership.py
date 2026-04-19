import pandas as pd

from quant_trading.pipeline.universe_membership import expand_snapshot_membership


def test_expand_snapshot_membership_builds_daily_rows_with_required_schema():
    snapshots = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-05", "2024-01-02"],
            "index_code": ["000905.SH", "000905.SH", "000905.SH"],
            "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
        }
    )
    trade_days = pd.to_datetime([
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-10",
    ])

    out = expand_snapshot_membership(snapshots, trade_days, "2024-01-01", "2024-01-08")

    assert list(out.columns) == ["trade_date", "index_code", "ts_code", "source"]
    assert (out["source"] == "meta_universe_expand").all()
    assert ((out["trade_date"] >= pd.Timestamp("2024-01-01")) & (out["trade_date"] <= pd.Timestamp("2024-01-08"))).all()
    assert set(out["trade_date"]) <= set(pd.to_datetime(trade_days))
    assert not out.duplicated(subset=["trade_date", "index_code", "ts_code"]).any()

    expected = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-08",
                ]
            ),
            "index_code": ["000905.SH"] * 10,
            "ts_code": [
                "000001.SZ",
                "000001.SZ",
                "000001.SZ",
                "000001.SZ",
                "000001.SZ",
                "000002.SZ",
                "000002.SZ",
                "000002.SZ",
                "000002.SZ",
                "000002.SZ",
            ],
            "source": ["meta_universe_expand"] * 10,
        }
    ).sort_values(["trade_date", "index_code", "ts_code"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        out.sort_values(["trade_date", "index_code", "ts_code"]).reset_index(drop=True),
        expected,
    )

    boundary = pd.Timestamp("2024-01-05")
    first_interval_days = out.loc[
        (out["ts_code"] == "000001.SZ") & (out["trade_date"] < boundary),
        "trade_date",
    ].tolist()
    assert first_interval_days == pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]).tolist()
    assert not (
        ((out["ts_code"] == "000001.SZ") & (out["trade_date"] > boundary))
        & (out["trade_date"] < pd.Timestamp("2024-01-08"))
    ).any()


def test_expand_snapshot_membership_does_not_backfill_before_first_snapshot():
    snapshots = pd.DataFrame(
        {
            "trade_date": ["2024-01-05"],
            "index_code": ["000905.SH"],
            "ts_code": ["000001.SZ"],
        }
    )
    trade_days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])

    out = expand_snapshot_membership(snapshots, trade_days, "2024-01-01", "2024-01-05")

    assert out["trade_date"].min() == pd.Timestamp("2024-01-05")


def test_expand_snapshot_membership_caps_last_interval_at_run_end():
    snapshots = pd.DataFrame(
        {
            "trade_date": ["2024-01-02"],
            "index_code": ["000905.SH"],
            "ts_code": ["000001.SZ"],
        }
    )
    trade_days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])

    out = expand_snapshot_membership(snapshots, trade_days, "2024-01-01", "2024-01-04")

    assert out["trade_date"].max() == pd.Timestamp("2024-01-04")


def test_expand_snapshot_membership_empty_snapshots_returns_empty_with_schema():
    snapshots = pd.DataFrame(columns=["trade_date", "index_code", "ts_code"])
    trade_days = pd.to_datetime(["2024-01-02", "2024-01-03"])

    out = expand_snapshot_membership(snapshots, trade_days, "2024-01-01", "2024-01-03")

    assert out.empty
    assert list(out.columns) == ["trade_date", "index_code", "ts_code", "source"]
