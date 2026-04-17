from __future__ import annotations

import pandas as pd


def topk_backtest(
    pred: pd.DataFrame,
    realized: pd.DataFrame,
    top_k: int,
    trade_cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    pred_dup_mask = pred.duplicated(subset=["trade_date", "ts_code"], keep=False)
    if pred_dup_mask.any():
        raise ValueError("Duplicate keys found in pred for [trade_date, ts_code]")

    realized_base = realized[["trade_date", "ts_code", "label_ret_t1"]]
    realized_dup_mask = realized_base.duplicated(subset=["trade_date", "ts_code"], keep=False)
    if realized_dup_mask.any():
        raise ValueError("Duplicate keys found in realized for [trade_date, ts_code]")

    merged = pred.merge(
        realized_base,
        on=["trade_date", "ts_code"],
        how="inner",
    )
    if merged.empty:
        nav = pd.DataFrame(columns=["trade_date", "nav"])
        metrics = {"cum_return": 0.0, "avg_daily_ret": 0.0}
        return nav, metrics

    ranked = merged.sort_values(
        ["trade_date", "y_pred", "ts_code"],
        ascending=[True, False, True],
    )
    selected = ranked.groupby("trade_date", sort=True, as_index=False).head(top_k)

    daily = (
        selected.groupby("trade_date", as_index=False)["label_ret_t1"]
        .mean()
        .rename(columns={"label_ret_t1": "gross_ret"})
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    daily["net_ret"] = daily["gross_ret"] - (trade_cost_bps / 10000.0)
    daily["nav"] = (1.0 + daily["net_ret"]).cumprod()

    # Defensive fallback: normally unreachable because merged.empty is handled above.
    if daily.empty:
        nav = pd.DataFrame(columns=["trade_date", "nav"])
        metrics = {"cum_return": 0.0, "avg_daily_ret": 0.0}
        return nav, metrics

    nav = daily[["trade_date", "nav"]].copy()
    metrics = {
        "cum_return": float(nav["nav"].iloc[-1] - 1.0),
        "avg_daily_ret": float(daily["net_ret"].mean()),
    }
    return nav, metrics
