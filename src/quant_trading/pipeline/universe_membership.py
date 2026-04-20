from __future__ import annotations

import pandas as pd


def expand_snapshot_membership(
    snapshots: pd.DataFrame,
    trade_days: pd.Series | pd.DatetimeIndex,
    run_start_date: str,
    run_end_date: str,
) -> pd.DataFrame:
    columns = ["trade_date", "index_code", "ts_code", "source"]
    if snapshots.empty:
        return pd.DataFrame(columns=columns)

    members = snapshots[["trade_date", "index_code", "ts_code"]].copy()
    members["trade_date"] = pd.to_datetime(members["trade_date"])
    members = members.drop_duplicates(["trade_date", "index_code", "ts_code"])
    members = members.sort_values(["index_code", "trade_date", "ts_code"])

    run_start = pd.to_datetime(run_start_date)
    run_end = pd.to_datetime(run_end_date)

    days = pd.to_datetime(pd.Series(trade_days)).drop_duplicates().sort_values()
    days = days[(days >= run_start) & (days <= run_end)]
    if days.empty:
        return pd.DataFrame(columns=columns)

    expanded_rows: list[tuple[pd.Timestamp, str, str, str]] = []

    for index_code, group in members.groupby("index_code", sort=False):
        snapshot_days = sorted(group["trade_date"].drop_duplicates().tolist())
        snapshot_members: dict[pd.Timestamp, list[str]] = {}

        for snapshot_date in snapshot_days:
            day_members = sorted(
                group.loc[group["trade_date"] == snapshot_date, "ts_code"].drop_duplicates().tolist()
            )
            if len(day_members) != 500:
                raise ValueError(
                    f"Snapshot {index_code} {snapshot_date.date()} must contain exactly 500 members, "
                    f"got {len(day_members)}."
                )
            snapshot_members[snapshot_date] = day_members

        for idx, snapshot_date in enumerate(snapshot_days):
            next_date = snapshot_days[idx + 1] if idx + 1 < len(snapshot_days) else None
            interval_end = (next_date - pd.Timedelta(days=1)) if next_date is not None else run_end

            effective_start = max(snapshot_date, run_start)
            effective_end = min(interval_end, run_end)
            if effective_start > effective_end:
                continue

            effective_days = days[(days >= effective_start) & (days <= effective_end)]
            expanded_rows.extend(
                (day, index_code, ts_code, "meta_universe_expand")
                for day in effective_days
                for ts_code in snapshot_members[snapshot_date]
            )

    out = pd.DataFrame(expanded_rows, columns=columns)
    if out.empty:
        return out

    out = out.drop_duplicates(["trade_date", "index_code", "ts_code"])
    return out.sort_values(["trade_date", "index_code", "ts_code"]).reset_index(drop=True)
